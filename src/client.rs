use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{
    arith::*, client::*, discrete_gaussian::*, gadget::*, number_theory::*, params::*, poly::*,
};
//use crate::server::Precomp;

//use super::util::*;
use super::convolution::negacyclic_matrix_u32;
use super::{lwe::*, noise_analysis::measure_noise_width_squared, scheme::*, util::*};

pub fn rlwe_to_lwe<'a>(params: &'a Params, ct: &PolyMatrixRaw<'a>) -> Vec<u64> {
    let a = ct.get_poly(0, 0);
    let mut negacylic_a = negacyclic_matrix(&a, params.modulus);
    negacylic_a.extend(ct.get_poly(1, 0));

    negacylic_a
}

pub fn pack_query(params: &Params, query: &[u64]) -> AlignedMemory64 {
    let query_packed = query
        .iter()
        .enumerate()
        .map(|(_i, x)| {
            let crt0 = (*x) % params.moduli[0];
            let crt1 = (*x) % params.moduli[1];
            crt0 | (crt1 << 32)
        })
        .collect::<Vec<_>>();
    let mut aligned_query_packed = AlignedMemory64::new(query_packed.len());
    aligned_query_packed
        .as_mut_slice()
        .copy_from_slice(&query_packed);
    aligned_query_packed
}

pub fn get_reg_sample<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let a = PolyMatrixRaw::random_rng(params, 1, 1, rng_pub);
    let e = PolyMatrixRaw::noise(
        params,
        1,
        1,
        &DiscreteGaussian::init(params.noise_width),
        rng,
    );
    let b_p = &sk_reg.ntt() * &a.ntt();
    let b = &e.ntt() + &b_p;
    let mut p = PolyMatrixNTT::zero(params, 2, 1);
    p.copy_into(&(-&a).ntt(), 0, 0);
    p.copy_into(&b, 1, 0);
    p
}

pub fn get_fresh_reg_public_key<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    m: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let mut p = PolyMatrixNTT::zero(params, 2, m);

    for i in 0..m {
        p.copy_into(&get_reg_sample(params, sk_reg, rng, rng_pub), 0, i);
    }
    p
}

pub fn raw_generate_expansion_params<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    num_exp: usize,
    m_exp: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> Vec<PolyMatrixNTT<'a>> {
    let g_exp = build_gadget(params, 1, m_exp);
    //debug!("using gadget base {}", g_exp.get_poly(0, 1)[0]);
    println!("gadget: {}, {}, {}", g_exp.get_poly(0, 0)[0], g_exp.get_poly(0, 1)[0], g_exp.get_poly(0, 2)[0]);
    let g_exp_ntt = g_exp.ntt();
    let mut res = Vec::new();

    for i in 0..num_exp {
        let t = (params.poly_len / (1 << i)) + 1;
        let tau_sk_reg = automorph_alloc(&sk_reg, t);
        let prod = &tau_sk_reg.ntt() * &g_exp_ntt;

        // let w_exp_i = client.encrypt_matrix_reg(&prod, rng, rng_pub);
        let sample = get_fresh_reg_public_key(params, &sk_reg, m_exp, rng, rng_pub);
        let w_exp_i = &sample + &prod.pad_top(1);
        res.push(w_exp_i);
    }

    res
}

pub fn decrypt_ct_reg_measured<'a>(
    client: &Client<'a>,
    params: &'a Params,
    ct: &PolyMatrixNTT<'a>,
    coeffs_to_measure: usize,
) -> PolyMatrixRaw<'a> {
    let dec_result = client.decrypt_matrix_reg(ct).raw();

    let mut dec_rescaled = PolyMatrixRaw::zero(&params, dec_result.rows, dec_result.cols);
    for z in 0..dec_rescaled.data.len() {
        dec_rescaled.data[z] = rescale(dec_result.data[z], params.modulus, params.pt_modulus);
    }

    // measure noise width
    let s_2 = measure_noise_width_squared(params, client, ct, &dec_rescaled, coeffs_to_measure);
    debug!("log2(measured noise): {}", s_2.log2());

    dec_rescaled
}

pub struct YClient<'a> {
    inner: &'a mut Client<'a>,
    params: &'a Params,
    lwe_client: LWEClient,
}

pub fn get_seed(public_seed_idx: u8) -> [u8; 32] {
    let mut seed = STATIC_PUBLIC_SEED;
    seed[0] = public_seed_idx;
    seed
}

pub fn generate_matrix_ring(
    rng_pub: &mut ChaCha20Rng,
    n: usize,
    rows: usize,
    cols: usize,
) -> Vec<u32> {
    assert_eq!(rows % n, 0);
    assert_eq!(cols % n, 0);
    let rows_outer = rows / n;
    let cols_outer = cols / n;

    let mut out = vec![0u32; rows * cols];
    for i in 0..rows_outer {
        for j in 0..cols_outer {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }

            let mat = negacyclic_matrix_u32(&a);
            for k in 0..n {
                for l in 0..n {
                    let idx = (i * n + k) * cols + (j * n + l);
                    out[idx] = mat[k * n + l];
                }
            }
        }
    }

    out
}

impl<'a> YClient<'a> {
    pub fn new(inner: &'a mut Client<'a>, params: &'a Params) -> Self {
        Self {
            inner,
            params,
            lwe_client: LWEClient::new(LWEParams::default()),
        }
    }

    pub fn lwe_client(&self) -> &LWEClient {
        &self.lwe_client
    }

    fn rlwes_to_lwes(&self, ct: &[PolyMatrixRaw<'a>]) -> Vec<u64> {
        let v = ct
            .iter()
            .map(|ct| rlwe_to_lwe(self.params, ct))
            .collect::<Vec<_>>();
        concat_horizontal(&v, self.params.poly_len + 1, self.params.poly_len)
    }

    pub fn generate_query_impl(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index: usize,
    ) -> Vec<PolyMatrixRaw<'a>> {
        // let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);
        // let idx_dim1 = index / db_cols;

        let multiply_ct = true;

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx));

        // Generate dim1_bits LWE samples under public randomness
        let mut out = Vec::new();

        let scale_k = self.params.modulus / self.params.pt_modulus;
	println!("encrypt modulus: {}, polylen: {}, scale_k: {}", self.params.modulus, self.params.poly_len, scale_k);

        for i in 0..(1 << dim_log2) {
            let mut scalar = PolyMatrixRaw::zero(self.params, 1, 1);
            let is_nonzero = i == (index / self.params.poly_len);

            if is_nonzero {
                scalar.data[index % self.params.poly_len] = scale_k;
            }

            if packing {
		println!("packing: {}", true);
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
                scalar = scalar_multiply_alloc(
                    &PolyMatrixRaw::single_value(self.params, factor).ntt(),
                    &scalar.ntt(),
                )
                .raw();
            }

            // if public_seed_idx == SEED_0 {
            //     out.push(scalar.pad_top(1));
            //     continue;
            // }

            let ct = if multiply_ct {
		println!("multiply_ct: {}", true);
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();

                self.inner.encrypt_matrix_scaled_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                    factor,
                )
            } else {
                self.inner.encrypt_matrix_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                )
            };

            // let mut ct = self.inner.encrypt_matrix_reg(
            //     &scalar.ntt(),
            //     &mut ChaCha20Rng::from_entropy(),
            //     &mut rng_pub,
            // );

            // if multiply_ct && packing {
            //     let factor =
            //         invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
            //     ct = scalar_multiply_alloc(
            //         &PolyMatrixRaw::single_value(self.params, factor).ntt(),
            //         &ct,
            //     );
            // };

            // if multiply_error && is_nonzero && packing {
            //     let factor =
            //         invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
            //     ct = scalar_multiply_alloc(
            //         &PolyMatrixRaw::single_value(self.params, factor).ntt(),
            //         &ct,
            //     );
            // }

            let ct_raw = ct.raw();
            // let ct_0_nega = negacyclic_perm(ct_raw.get_poly(0, 0), 0, self.params.modulus);
            // let ct_1_nega = negacyclic_perm(ct_raw.get_poly(1, 0), 0, self.params.modulus);
            // let mut ct_nega = PolyMatrixRaw::zero(self.params, 2, 1);
            // ct_nega.get_poly_mut(0, 0).copy_from_slice(&ct_0_nega);
            // ct_nega.get_poly_mut(1, 0).copy_from_slice(&ct_1_nega);

            // self-test
            // {
            //     let test_ct = self.inner.encrypt_matrix_reg(
            //         &PolyMatrixRaw::single_value(self.params, scale_k * 7).ntt(),
            //         &mut ChaCha20Rng::from_entropy(),
            //         &mut ChaCha20Rng::from_entropy(),
            //     );
            //     let lwe = rlwe_to_lwe(self.params, &test_ct.raw());
            //     let result = self.decode_response(&lwe);
            //     assert_eq!(result[0], 7);
            // }

            out.push(ct_raw);
        }

        out
    }

    pub fn generate_query(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index_row: usize,
    ) -> Vec<u64> {
        if public_seed_idx == SEED_0 && !packing {
	    println!("packing");
            let lwe_params = LWEParams::default();
            let dim = 1 << (dim_log2 + self.params.poly_len_log2);

            // lwes must be (n + 1) x (dim) matrix
            let mut lwes = vec![0u64; (lwe_params.n + 1) * dim];

            let scale_k = lwe_params.scale_k() as u32;
            let mut vals_to_encrypt = vec![0u32; dim];
            vals_to_encrypt[index_row] = scale_k;

            let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx));

            for i in (0..dim).step_by(lwe_params.n) {
                let out = self
                    .lwe_client
                    .encrypt_many(&mut rng_pub, &vals_to_encrypt[i..i + lwe_params.n])
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<Vec<_>>();
                assert_eq!(out.len(), (lwe_params.n + 1) * lwe_params.n);
                for r in 0..lwe_params.n + 1 {
                    for c in 0..lwe_params.n {
                        lwes[r * dim + i + c] = out[r * lwe_params.n + c];
                    }
                }
            }

            lwes
        } else {
	    println!("else");
            let out = self.generate_query_impl(public_seed_idx, dim_log2, packing, index_row);
            let lwes = self.rlwes_to_lwes(&out);
            lwes
        }
    }

    pub fn decode_response(&self, response: &[u64]) -> Vec<u64> {
        debug!("Decoding response: {:?}", &response[..16]);
        let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);

        let sk = self.inner.get_sk_reg().as_slice().to_vec();

        let mut out = Vec::new();
        for col in 0..db_cols {
            let mut sum = 0u128;
            for i in 0..self.params.poly_len {
                let v1 = response[i * db_cols + col];
                let v2 = sk[i];
                sum += v1 as u128 * v2 as u128;
            }

            sum += response[self.params.poly_len * db_cols + col] as u128;

            let result = (sum % self.params.modulus as u128) as u64;
            let result_rescaled = rescale(result, self.params.modulus, self.params.pt_modulus);
            out.push(result_rescaled);
        }

        out
    }

    pub fn client(&self) -> &Client<'a> {
        self.inner
    }
}



pub struct NewClient<'a> {
    inner: &'a mut Client<'a>,
    params: &'a Params,
    lwe_client: LWEClient,
    pt_byte : usize,
}


impl<'a> NewClient<'a> {
    pub fn new(inner: &'a mut Client<'a>, params: &'a Params, pt_byte : usize) -> Self {
        Self {
            inner,
            params,
            lwe_client: LWEClient::new(LWEParams::default()),
	    pt_byte,
        }
    }

    pub fn lwe_client(&self) -> &LWEClient {
        &self.lwe_client
    }

    fn rlwes_to_lwes(&self, ct: &[PolyMatrixRaw<'a>]) -> Vec<u64> {
        let v = ct
            .iter()
            .map(|ct| rlwe_to_lwe(self.params, ct))
            .collect::<Vec<_>>();
        concat_horizontal(&v, self.params.poly_len + 1, self.params.poly_len)
    }

    pub fn generate_query_impl(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index: usize,
	//pt_byte : usize
    ) -> Vec<PolyMatrixRaw<'a>> {
        // let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);
        // let idx_dim1 = index / db_cols;

        let multiply_ct = true;

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx));

        // Generate dim1_bits LWE samples under public randomness
        let mut out = Vec::new();

        let scale_k = self.params.modulus / self.params.pt_modulus;
	//println!("encrypt modulus: {}, polylen: {}, scale_k: {}", self.params.modulus, self.params.poly_len, scale_k);

        for i in 0..(1 << dim_log2) {
            let mut scalar = PolyMatrixRaw::zero(self.params, 1, 1);
            let is_nonzero = i == (index / self.params.poly_len);

            if is_nonzero {
                scalar.data[index % self.params.poly_len] = scale_k;
            }

            if packing {
		//println!("packing: {}", true);
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
                scalar = scalar_multiply_alloc(
                    &PolyMatrixRaw::single_value(self.params, factor).ntt(),
                    &scalar.ntt(),
                )
                .raw();
            }

            // if public_seed_idx == SEED_0 {
            //     out.push(scalar.pad_top(1));
            //     continue;
            // }

            let ct = if multiply_ct {
		println!("multiply_ct: {}", true);
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();

                self.inner.encrypt_matrix_scaled_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                    factor,
                )
            } else {
                self.inner.encrypt_matrix_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                )
            };

            // let mut ct = self.inner.encrypt_matrix_reg(
            //     &scalar.ntt(),
            //     &mut ChaCha20Rng::from_entropy(),
            //     &mut rng_pub,
            // );

            // if multiply_ct && packing {
            //     let factor =
            //         invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
            //     ct = scalar_multiply_alloc(
            //         &PolyMatrixRaw::single_value(self.params, factor).ntt(),
            //         &ct,
            //     );
            // };

            // if multiply_error && is_nonzero && packing {
            //     let factor =
            //         invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
            //     ct = scalar_multiply_alloc(
            //         &PolyMatrixRaw::single_value(self.params, factor).ntt(),
            //         &ct,
            //     );
            // }

            let ct_raw = ct.raw();
            // let ct_0_nega = negacyclic_perm(ct_raw.get_poly(0, 0), 0, self.params.modulus);
            // let ct_1_nega = negacyclic_perm(ct_raw.get_poly(1, 0), 0, self.params.modulus);
            // let mut ct_nega = PolyMatrixRaw::zero(self.params, 2, 1);
            // ct_nega.get_poly_mut(0, 0).copy_from_slice(&ct_0_nega);
            // ct_nega.get_poly_mut(1, 0).copy_from_slice(&ct_1_nega);

            // self-test
            // {
            //     let test_ct = self.inner.encrypt_matrix_reg(
            //         &PolyMatrixRaw::single_value(self.params, scale_k * 7).ntt(),
            //         &mut ChaCha20Rng::from_entropy(),
            //         &mut ChaCha20Rng::from_entropy(),
            //     );
            //     let lwe = rlwe_to_lwe(self.params, &test_ct.raw());
            //     let result = self.decode_response(&lwe);
            //     assert_eq!(result[0], 7);
            // }

            out.push(ct_raw);
        }

        out
    }

    pub fn generate_query(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index_row: usize,
    ) -> Vec<PolyMatrixRaw<'a>> {
        let out = self.generate_query_impl(public_seed_idx, dim_log2, packing, index_row);
	out        
	//let lwes = self.rlwes_to_lwes(&out);
        //lwes
    }

    pub fn decode_response(&self, response: &[u64]) -> Vec<u64> {
        debug!("Decoding response: {:?}", &response[..16]);
        let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);

        let sk = self.inner.get_sk_reg().as_slice().to_vec();

        let mut out = Vec::new();
        for col in 0..db_cols {
            let mut sum = 0u128;
            for i in 0..self.params.poly_len {
                let v1 = response[i * db_cols + col];
                let v2 = sk[i];
                sum += v1 as u128 * v2 as u128;
            }

            sum += response[self.params.poly_len * db_cols + col] as u128;

            let result = (sum % self.params.modulus as u128) as u64;
            let result_rescaled = rescale(result, self.params.modulus, self.params.pt_modulus);
            out.push(result_rescaled);
        }

        out
    }

    pub fn client(&self) -> &Client<'a> {
        self.inner
    }
}
    #[cfg(not(target_feature = "avx2"))]
    pub fn mul() {
	println!("using non-avx2 multiplication");
    }

    #[cfg(target_feature = "avx2")]
    pub fn mul() {
	println!("using avx2 multiplication");
    }

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
	params::params_for_scenario, server::*
	//packing::*
    };
    use std::time::Instant;

    #[test]
    fn test_mlwe_encryption() {
	let mut params = params_for_scenario(1 << 30, 1);
	let mut mlwe_params = params.clone();
	let mlwe_bit = 7;
        let mut client = Client::init(&params);
        client.generate_secret_keys();
	//params.poly_len_log2 = 3;
	//params.poly_len = 1<<params.poly_len_log2;
	mlwe_params.poly_len_log2 = mlwe_bit;
	mlwe_params.poly_len = 1<<mlwe_bit;

	//println!("{}, {}, {}", params.poly_len_log2, mlwe_params.poly_len_log2, 1<<(params.poly_len_log2 - mlwe_bit));

	let mlwe_dim = 1<<(params.poly_len_log2 - mlwe_bit);
	//println!("mlwe_dim : {}", mlwe_dim);

	let mut secret = client.get_sk_reg();

	let mut mlwe_secret = PolyMatrixRaw::zero(&mlwe_params, 1, mlwe_dim);
	let mlwe_sec = rlwe_to_mlwe_b(&params, &secret.as_slice().to_vec(), mlwe_bit);
	//println!("{}", mlwe_secret.as_slice().len());
	mlwe_secret.as_mut_slice().copy_from_slice(&mlwe_sec);

	for i in 0..2048 {
	    assert_eq!(mlwe_secret.data[i], mlwe_sec[i+1]);
	}

	

    }

    #[test]
    fn test_lwe() {
        let lwe_params = LWEParams::default();
        let client = LWEClient::new(lwe_params.clone());
        let pt = fastrand::u32(0..lwe_params.pt_modulus as u32);
        let scaled_pt = pt.wrapping_mul(lwe_params.scale_k() as u32);
        let ct = client.encrypt(&mut ChaCha20Rng::from_entropy(), scaled_pt);
        let pt_dec = client.decrypt(&ct);
        let result = rescale(pt_dec as u64, lwe_params.modulus, lwe_params.pt_modulus) as u32;
	println!("1");        
	assert_eq!(result, pt);
    }
    #[test]
    fn test_rlwe() { // encryption, decryption
	let params = params_for_scenario(1 << 30, 1);
	let mut client = Client::init(&params);
	client.generate_secret_keys();
	//let y_client = YClient::new(&mut client, &params);
	let mut rng_pub = ChaCha20Rng::from_seed(get_seed(1));
	let scale_k = params.modulus / params.pt_modulus;
	
	let mut plaintext_1 = PolyMatrixRaw::zero(&params, 1, 1); // how to assign
	let mut plaintext_2 = PolyMatrixRaw::zero(&params, 1, 1);
	
	plaintext_1.data[0] = 63 * scale_k; // scale up plaintext
	plaintext_1.data[1] = 84 * scale_k;
	
	plaintext_2.data[0] = 41;
	plaintext_2.data[1] = 52;
	
	let nega = negacyclic_perm(plaintext_2.get_poly(0, 0), 0, params.modulus);
	for i in 0..params.poly_len {
	    plaintext_2.data[i] = nega[i];
	}	
	let mut plaintext_2_ntt = plaintext_2.ntt();
	let mut p = 3;
	println!("1: {}", params.modulus);
	//println!("printpoly: {:?}", nega);
	println!("printlen: {}", plaintext_1.as_slice().len());
	println!("{:?}", plaintext_2.as_slice()); // how to print
	//assert_eq!(plaintext_1.as_slice(), plaintext_2.as_slice());
	
	//let mut ct = y_client.inner.encrypt_matrix_reg(&plaintext_1.ntt(), &mut ChaCha20Rng::from_entropy(), &mut rng_pub).raw(); //->polymatNTT(2, 1) ->encryption and how to "NTT"
	// if i want to encrypt scaled, then "encrypt_matrix_scaled_reg"
	//to_ntt(polyNTT, polyRaw) -> ntt all the elements of matPolyRaw
	let mut ct = client.encrypt_matrix_reg(&plaintext_1.ntt(), &mut ChaCha20Rng::from_entropy(), &mut rng_pub).raw();
	let nega_1 = negacyclic_perm(ct.get_poly(0, 0), 0, params.modulus);
	let nega_2 = negacyclic_perm(ct.get_poly(1, 0), 0, params.modulus);

	//for i in 0..params.poly_len{
	//    ct.data[i] = nega_1[i];
	//}
	//for i in 0..params.poly_len{
	//    ct.data[i + params.poly_len] = nega_2[i];
	//}
	//let mut dec = y_client.inner.decrypt_matrix_reg(&ct.ntt()).raw();

	//let mut dec = decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len);	
	let dec_result = client.decrypt_matrix_reg(&ct.ntt()).raw();
	let mut dec_rescaled = PolyMatrixRaw::zero(&params, dec_result.rows, dec_result.cols);
	for z in 0..dec_rescaled.data.len() {
	    dec_rescaled.data[z] = rescale(dec_result.data[z], params.modulus, params.pt_modulus);
	}	
	println!("dec_nega: {:?}", dec_rescaled.as_slice());

	let mut prod = PolyMatrixNTT::zero(&params, 2, 1);

	//multiply(&mut prod, &ct.ntt(), &plaintext_2_ntt);
	//scalar_multiply_alloc(&mut prod, &plaintext_2_ntt , &ct.ntt());
	scalar_multiply_avx(&mut prod, &plaintext_2_ntt, &ct.ntt());

	let dec_result = client.decrypt_matrix_reg(&prod).raw();
	let mut dec_rescaled = PolyMatrixRaw::zero(&params, dec_result.rows, dec_result.cols);
	for z in 0..dec_rescaled.data.len() {
	    dec_rescaled.data[z] = rescale(dec_result.data[z], params.modulus, params.pt_modulus);
	}	
	println!("dec: {:?}", dec_rescaled.as_slice());

	//let result = decrypt_ct_reg_measured(y_client.client(), &params, &prod, params.poly_len);
	//println!("result: {:?}", result.as_slice());
	
	
	
	
    }

    #[test]
    fn test_multiply_poly_time() {
	let mut params = params_for_scenario(1<<30, 1);
	params.poly_len = 128;
	params.poly_len_log2 = 7;
	let dimension = 2048;
	let mut exp = PolyMatrixNTT::zero(&params, 1, 1);
	println!("length: {}", exp.as_slice().len());
	let mut db = PolyMatrixNTT::zero(&params, dimension, dimension);
	let mut query1 = PolyMatrixRaw::zero(&params, dimension, 1); // query 1
	let mut query2 = PolyMatrixNTT::zero(&params, dimension, 1);
	let mut res_poly = PolyMatrixNTT::zero(&params, dimension, 1); // result 1
	let mut poly_3 = PolyMatrixRaw::zero(&params, 4, dimension); // decomped
	let mut res_poly_2 = PolyMatrixNTT::zero(&params, 4, 1); // result 2
	let mut query1_ntt = query1.ntt();
	let mut poly_3_ntt = poly_3.ntt();

	let start = Instant::now();
	multiply(&mut res_poly, &db, &query1_ntt); // db * query 1
	let mut res_poly_raw = res_poly.raw(); // db*query 1 = dimension by 1
	multiply(&mut res_poly_2, &poly_3.ntt(), &query2); 
	let end = Instant::now();
	println!("time: {:?}", end - start);
	mul();

	//let mut po1ynomial_1 = PolyMatrixNTT::zero(&params, 1, 1);
	//let mut polynomial_2 = PolyMatrixNTT::zero(&params, 1, 1);
	//let mut pol = PolyMatrixNTT::zero(&params, 1, 1);

	//start = Instant::now();
	//multiply(&mut pol, &polynomial_1, &polynomial_2);
	//end = Instant::now();
	//println!("time: {:?}", end - start);

	params.poly_len = 128;
	params.poly_len_log2 = 7;
	let mut polyn_1 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut polyn_2 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut polyn_1_ntt = polyn_1.ntt();
	let mut polyn_2_ntt = polyn_2.ntt();
	let start = Instant::now();
	for i in 0..16 {
	    //let mut polyn_1_ntt = polyn_1.ntt();
	    //let mut polyn_2_ntt = polyn_2.ntt();
	    let mut polyn_result_ntt = &polyn_1_ntt * &polyn_2_ntt;
	    //let mut polyn_result = polyn_result_ntt.raw();
	}	
	let end = Instant::now();

	let time_ = end - start;

	println!("time: {:?}", (end - start));

	params.poly_len = 2048;
	params.poly_len_log2 = 11;
	let mut polyn_1 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut polyn_2 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut polyn_1_ntt = polyn_1.ntt();
//	let mut polyn_2_ntt = polyn_2.ntt();	
	
	let start = Instant::now();

	let mut polyn_2_ntt = polyn_2.ntt();
	let mut polyn_result_ntt = &polyn_1_ntt * &polyn_2_ntt;
	let mut polyn_result = polyn_result_ntt.raw();
		
	let end = Instant::now();
	//let mut polyn = PolyMatrixNTT::zero(&params, 1, 1);
	
	//start = Instant::now();
	//multiply(&mut polyn, &polyn_1, &polyn_2);
	//end = Instant::now();
	println!("time: {:?}", (end - start));
    }

    #[test]
    fn test_poly() { // multiply : matrix multiplication
	let params = params_for_scenario(1<<30, 1);
	let mut client = Client::init(&params);
	let mut poly_1 = PolyMatrixRaw::zero(&params, 1, 1);
	poly_1.data[0] = 6;
	poly_1.data[1] = 1;
	//poly_1.data[2048] = 1;
	//poly_1.data[2049] = 2;
	let mut poly_2 = PolyMatrixRaw::zero(&params, 1, 1);
	poly_2.data[0] = 3;
	poly_2.data[1] = 4;
	//poly_2.data[2048] = 4;
	//poly_2.data[2049] = 3;
	//poly_2.data[4096] = 5;
	//poly_2.data[4097] = 6;
	//poly_2.data[6144] = 6;
	//poly_2.data[6145] = 5;
	let mut poly_1_ntt = poly_1.ntt();
	//println!("{:?}", poly_1_ntt.as_slice());
	let mut poly_2_ntt = poly_2.ntt();
	let mut res_poly_ntt = PolyMatrixNTT::zero(&params, 2, 1);

	//add_into(&mut poly_2_ntt, &mut poly_1_ntt);
	//println!("poly2: {:?}", poly_2_ntt.raw().as_slice());
	//println!("{:?}", res	
	//multiply(&mut res_poly_ntt, &poly_1_ntt, &poly_2_ntt);	 // 2 by 1  *  1 by 1
	//scalar_multiply_avx(&mut res_poly_ntt, &poly_1_ntt, &poly_2_ntt);	
	multiply_poly(&params, res_poly_ntt.get_poly_mut(0, 0), poly_1_ntt.get_poly(0, 0), poly_2_ntt.get_poly(0, 0));

	println!("res: {:?}", res_poly_ntt.raw().as_slice());
	
	
	//println!("{} {} {}", res_poly_ntt.raw().get_poly(0, 0)[0], res_poly_ntt.raw().get_poly(0, 0)[1], res_poly_ntt.raw().get_poly(0, 0)[2]); // how to print
	//println!("{} {} {}", res_poly_ntt.raw().get_poly(0, 1)[0], res_poly_ntt.raw().get_poly(0, 1)[1], res_poly_ntt.raw().get_poly(0, 1)[2]); // how to print

	
	//assert_eq!(poly_1_ntt.get_poly(1, 0), poly_2_ntt.get_poly(0, 1));
	
    }
    
    #[test]
    fn test_array_to_matrix(){
	let params = params_for_scenario(1<<30, 1);
	let mut poly_1 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut poly_2 = PolyMatrixRaw::zero(&params, 1, 1);
	poly_1.data[0] = 3;
	poly_1.data[1] = 4;
	poly_2.data[0] = 5;
	poly_2.data[1] = 6;

	let mut copied_poly = PolyMatrixRaw::zero(&params, 2, 1);
	
	let mut combined = vec![0u64; params.poly_len * 2];
	combined[0..params.poly_len].copy_from_slice(poly_1.get_poly(0, 0));
	combined[params.poly_len..2*params.poly_len].copy_from_slice(poly_2.get_poly(0, 0));
	println!("{:?}", combined);
	
	copied_poly.as_mut_slice().copy_from_slice(&combined);
	println!("{:?}", copied_poly.as_slice());
	println!("{:?}", copied_poly.get_poly(1, 0));
	
	
    }
	
    #[test]
    fn test_submatrix(){
	let params = params_for_scenario(1<<30, 1);
	let mut poly_1 = PolyMatrixRaw::zero(&params, 2, 1);
	poly_1.data[0] = 3;
	poly_1.data[1] = 4;
	poly_1.data[2048] = 5;
	poly_1.data[2049] = 6;

	let poly_0 = poly_1.submatrix(0, 0, 1, 1);
	let poly_00 = poly_1.submatrix(1, 0, 1, 1);
	
	println!("0th: {:?}", poly_0.as_slice());
	println!("1st: {:?}", poly_00.as_slice());
    }
}
