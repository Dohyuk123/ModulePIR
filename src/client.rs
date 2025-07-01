use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{
    arith::*, client::*, discrete_gaussian::*, gadget::*, number_theory::*, params::*, poly::*
};
use crate::server::Precomp;

use super::util::*;
use super::convolution::negacyclic_matrix_u32;
use super::{lwe::*, noise_analysis::measure_noise_width_squared, scheme::*, util::*, server::*};

pub fn mlwe_to_mlwe_sk(params: &Params, a: &Vec<u8>, mlwe_dim: usize, n: usize) -> Vec<u8>{
    let mut combined_vec: Vec<u8> = vec![0; n]; // Allocate memory once
    let half_mlwe_dim = mlwe_dim / 2;

    for j in 0..n / mlwe_dim {
       	// Process even_front_odd_back
        let mut even_front_odd_back: Vec<u8> = (0..half_mlwe_dim)
	    .map(|k| a[j * mlwe_dim + 2 * k])
	    .chain ((0..half_mlwe_dim).map(|k| a[j * mlwe_dim + 2 * k + 1]))
	    .collect();
	//println!("len: {}", even_front_odd_back.len());

	combined_vec[j * mlwe_dim .. j * mlwe_dim + mlwe_dim].copy_from_slice(&even_front_odd_back);    
	//println!("{:?}", combined_vec);
    }
    combined_vec
}

pub fn rlwe_to_mlwe_sk(params: &Params, a: &Vec<u8>, log_pt_byte: usize) -> Vec<u8> {
    let n = 1 << params.poly_len_log2;
    let mut mlwe_vector = mlwe_to_mlwe_sk(params, &a, n, n);

    for i in 1..(params.poly_len_log2 - log_pt_byte) {
	mlwe_vector = mlwe_to_mlwe_sk(params, &mlwe_vector, n >> i, n);
    }
    mlwe_vector
}

pub fn mlwe_to_mlwe_b_combined<'a>(params: &'a Params, a: &mut Vec<u64>, pt_byte:usize) -> Vec<u64> {
    let mut k = 0;
    let mut new_vec: Vec<u64> = vec![0;params.poly_len];

    while k < params.poly_len {
	for i in 0..pt_byte {
	    let idx = k+i;
	    new_vec[2*i + k] = a[idx];
	    new_vec[2*i+1 + k] = a[pt_byte + idx];
	}
	k = k + 2*pt_byte;
    }
    new_vec
}

pub fn mlwe_to_rlwe_b_combined<'a>(params: &'a Params, mut vector: Vec<u64>, pt_byte_log2 : usize) -> Vec<u64>{
    let mut i = 1<<pt_byte_log2;
    while i<=params.poly_len / 2 {
	vector = mlwe_to_mlwe_b_combined(&params, &mut vector, i);
	i = i*2;
    }
    vector
}

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
    //sk_reg_orig: &PolyMatrixRaw<'a>,
    sk_reg: &PolyMatrixRaw<'a>,
    num_exp: usize,
    m_exp: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> Vec<PolyMatrixNTT<'a>> {
    let g_exp = build_gadget(params, 1, m_exp);
    //debug!("using gadget base {}", g_exp.get_poly(0, 1)[0]);
    //println!("gadget: {}, {}, {}", g_exp.get_poly(0, 0)[0], g_exp.get_poly(0, 1)[0], g_exp.get_poly(0, 2)[0]);
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

pub fn raw_generate_expansion_params_mlwe<'a>(
    params: &'a Params,
    //sk_reg_orig: &PolyMatrixRaw<'a>,
    sk_reg: &PolyMatrixRaw<'a>,
    num_exp: usize,
    m_exp: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> Vec<PolyMatrixNTT<'a>> {
    let g_exp = build_gadget(params, 1, m_exp);
    //debug!("using gadget base {}", g_exp.get_poly(0, 1)[0]);
    //println!("gadget: {}, {}, {}", g_exp.get_poly(0, 0)[0], g_exp.get_poly(0, 1)[0], g_exp.get_poly(0, 2)[0]);
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

pub fn generate_query_expansion_key<'a>(
    params: &'a Params,
    mlwe_params: &'a Params,
    m_exp: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
    client: &mut Client<'a>,
) -> (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>) {

    let pt_byte_log2 = mlwe_params.poly_len_log2;
    let mut mlwe_secret_tmp = rlwe_to_mlwe_b(&params, &client.get_sk_reg().as_slice().to_vec(), mlwe_params.poly_len_log2);
    let mut sk_reg = PolyMatrixRaw::zero(&mlwe_params, params.poly_len / mlwe_params.poly_len, 1);
    sk_reg.as_mut_slice().copy_from_slice(&mlwe_secret_tmp);

    let g_exp = build_gadget(params, 1, m_exp);

    //println!("gadget: {}, {}, {}", g_exp.get_poly(0, 0)[0], g_exp.get_poly(0, 1)[0], g_exp.get_poly(0, 2)[0]);

    let g_exp_ntt = g_exp.ntt();
    let mut res_a = Vec::new(); // result a part
    let mut res_b = Vec::new(); // result b part
    let pt_byte = 1<<pt_byte_log2;
    let dimension = params.poly_len / pt_byte;

    let tmp_sk = mlwe_to_rlwe_b_combined(&params, sk_reg.as_slice().to_vec(), pt_byte_log2);
    for i in 0..params.poly_len{
	assert_eq!(tmp_sk[i], client.get_sk_reg().data[i]);
    }

/////////////////////////////////////////////////

    for i in 0..pt_byte_log2 {
	let t = (mlwe_params.poly_len / (1<<i)) + 1; 
	let tau_sk_reg = automorph_alloc(&sk_reg, t); // automorph mlwe secret key 16 by 1 // raw

	let auto_sk_combined = mlwe_to_rlwe_b_combined(&params, tau_sk_reg.as_slice().to_vec(), pt_byte_log2); // mlwe -> rlwe

	let mut auto_combined_poly = PolyMatrixRaw::zero(&params, 1, 1);
	auto_combined_poly.as_mut_slice().copy_from_slice(&auto_sk_combined);
	let prod = &auto_combined_poly.ntt() * &g_exp_ntt; // automorph rlwe secret key * gadget // 1 by 3 //ntt

	let sample = get_fresh_reg_public_key(params, &client.get_sk_reg(), m_exp, rng, rng_pub);
	let w_exp_i = &sample + &prod.pad_top(1); // ntt
	
	let mut w_exp_i_a = w_exp_i.submatrix(0, 0, 1, m_exp).raw(); //rlwe_a
	let mut w_exp_i_b = w_exp_i.submatrix(1, 0, 1, m_exp).raw(); //rlwe_b

	let mut mlwe_a_poly = PolyMatrixRaw::zero(&mlwe_params, m_exp * dimension, dimension);
	let mut mlwe_b_poly = PolyMatrixRaw::zero(&mlwe_params, m_exp*dimension, 1);
	
	for j in 0..m_exp {
	    let mlwe_a = rlwe_to_mlwe_a(&params, &w_exp_i_a.submatrix(0, j, 1, 1).as_slice().to_vec(), pt_byte_log2);
	    let mlwe_b = rlwe_to_mlwe_b(&params, &w_exp_i_b.submatrix(0, j, 1, 1).as_slice().to_vec(), pt_byte_log2);

	    mlwe_a_poly.as_mut_slice()[j*mlwe_a.len()..(j+1)*mlwe_a.len()].copy_from_slice(&mlwe_a);
	    mlwe_b_poly.as_mut_slice()[j*mlwe_b.len()..(j+1)*mlwe_b.len()].copy_from_slice(&mlwe_b);
	}

	res_a.push(mlwe_a_poly.ntt());
	res_b.push(mlwe_b_poly.ntt());
    }
    (res_a, res_b)
}




pub fn decrypt_mlwe_auto<'a> ( 
    params: &'a Params,
    mlwe_params: &'a Params,
    ct_a: &PolyMatrixNTT<'a>,
    ct_b: &PolyMatrixNTT<'a>,
    client: &Client<'a>,
    t: usize,
) ->PolyMatrixRaw<'a> {
    //let pt_byte = 1<<pt_byte_log2;
    //let scale_k = params.modulus / params.pt_modulus;

    let mut mlwe_secret_tmp = rlwe_to_mlwe_b(&params, &client.get_sk_reg().as_slice().to_vec(), mlwe_params.poly_len_log2);
    let mut secret_tmp = PolyMatrixRaw::zero(&mlwe_params, params.poly_len / mlwe_params.poly_len, 1);

    secret_tmp.as_mut_slice().copy_from_slice(&mlwe_secret_tmp);
    let mut secret = automorph_alloc(&secret_tmp, t);
    let ct_as = ct_a * &secret.ntt();
    let b = ct_b + &ct_as;
    let mut dec_mlwe = PolyMatrixRaw::zero(&mlwe_params, b.rows, b.cols);
    let b_raw = b.raw();

    //println!("result ct : {:?}", b_raw.as_slice());
    
    for z in 0..dec_mlwe.data.len() {
	dec_mlwe.data[z] = rescale(b_raw.data[z], params.modulus, params.pt_modulus);
    }

    dec_mlwe
}

pub fn decrypt_mlwe<'a> ( 
    params: &'a Params,
    mlwe_params: &'a Params,
    ct_a: &PolyMatrixNTT<'a>,
    ct_b: &PolyMatrixNTT<'a>,
    client: &Client<'a>,
) ->PolyMatrixRaw<'a> {
    //let pt_byte = 1<<pt_byte_log2;
    //let scale_k = params.modulus / params.pt_modulus;

    let mut mlwe_secret = rlwe_to_mlwe_b(&params, &client.get_sk_reg().as_slice().to_vec(), mlwe_params.poly_len_log2);
    let mut secret = PolyMatrixRaw::zero(&mlwe_params, params.poly_len / mlwe_params.poly_len, 1);

    secret.as_mut_slice().copy_from_slice(&mlwe_secret);
    let ct_as = ct_a * &secret.ntt();
    let b = ct_b + &ct_as;
    let mut dec_mlwe = PolyMatrixRaw::zero(&mlwe_params, b.rows, b.cols);
    let b_raw = b.raw();

    let mask: u64 = (1u64 << 41) - 1;

    //println!("braw: {}", ((b_raw.data[0] & mask) as f64).log2());
    
    for z in 0..dec_mlwe.data.len() {
	dec_mlwe.data[z] = rescale(b_raw.data[z], params.modulus, params.pt_modulus);
    }

    dec_mlwe
}

pub fn decrypt_mlwe_batch<'a> (
    params: &'a Params,
    mlwe_params: &'a Params,
    dimension : usize,
    ct_a: &PolyMatrixNTT<'a>,
    ct_b: &PolyMatrixNTT<'a>,
    client: &Client<'a>,
) -> Vec<PolyMatrixRaw<'a>> {

    let mut vector = Vec::new();

    for i in 0..ct_b.get_rows(){
	let mut dec = decrypt_mlwe(params, mlwe_params, &ct_a.submatrix(i, 0, 1, dimension), &ct_b.submatrix(i, 0, 1, 1), client);
        vector.push(dec);
    }
    vector
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
	let mut new_params = params.clone();
	let dimension = 1024;
	let mut db_tmp = PolyMatrixNTT::zero(&new_params, dimension, 1);

	params.poly_len = 128;
	params.poly_len_log2 = 7;
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
	//let mut query1_ntt = query1.ntt();
	db_tmp.raw();
	multiply(&mut res_poly, &db, &query1_ntt); // db * query 1
	let mut res_poly_raw = res_poly.raw(); // db*query 1 = dimension by 1
	//let mut query2_ntt = query2.raw();
	//query2 = query2_ntt.ntt();
	multiply(&mut res_poly_2, &poly_3.ntt(), &query2); 
	let end = Instant::now();
	println!("time: {:?}", end - start);
	mul();

	

	params.poly_len = 256;
	params.poly_len_log2 = 8;
	let mut polyn_1 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut polyn_2 = PolyMatrixRaw::zero(&params, 1, 1);
	let mut polyn_1_ntt = polyn_1.ntt();
	let mut polyn_2_ntt = polyn_2.ntt();

	let mut poly_tmp1 = PolyMatrixNTT::zero(&params, 512, 1);
	let mut poly_tmp2 = PolyMatrixRaw::zero(&params, 512*7, 1);
	let mut poly_tmp3 = PolyMatrixNTT::zero(&params, 63, 1);	

	let start = Instant::now();
	let mut raw_1 = poly_tmp1.raw();
	let mut ntt_1 = poly_tmp2.ntt();
	let mut raw_2 = poly_tmp3.raw();
	let mut ntt_2 = raw_2.ntt();	
	let end = Instant::now();

	let time_ = end - start;

	println!("ntt time: {:?}", (end - start));

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
    #[test]
    fn test_mlwe_to_rlwe_b(){
	let pt_byte = 128;
	let pt_byte_log2 = 7;
	let params = params_for_scenario(1<<30, 1);
	let mut poly = PolyMatrixRaw::zero(&params, 1, 1);

	for i in 0..params.poly_len {
	    poly.data[i] = (i) as u64;
	}
	
	let mut mlwe = rlwe_to_mlwe_b(&params, &poly.as_slice().to_vec(), pt_byte_log2);
	println!("mlwe: {:?}", mlwe);

	mlwe = mlwe_to_rlwe_b_combined(&params, mlwe, pt_byte_log2);
	println!("mlwe: {:?}", mlwe);
    }

    #[test]
    fn test_expansion_key_new(){
	let pt_byte_log2 = 7;
	let pt_byte = 128;
	let params = params_for_scenario(1<<30, 1);
	let mut mlwe_params = params.clone();
	mlwe_params.poly_len = pt_byte;
	mlwe_params.poly_len_log2 = pt_byte_log2;
	let dimension = params.poly_len / pt_byte;

	let mut client = Client::init(&params);
	client.generate_secret_keys();

	let scale_k = params.modulus / params.pt_modulus; 
	
	//let mut mlwe_secret = rlwe_to_mlwe_b(&params, &client.get_sk_reg().as_slice().to_vec(), pt_byte_log2);

	//let mut mlwe_secret_poly = PolyMatrixRaw::zero(&mlwe_params, dimension, 1); // mlwe secret key

	//mlwe_secret_poly.as_mut_slice().copy_from_slice(&mlwe_secret);

	let t_exp = params.t_exp_left;
	let pack_seed = [1u8; 32];
	let mut rng_pub = ChaCha20Rng::from_seed(get_seed(1)); 
	let mut poly = PolyMatrixRaw::zero(&params, 1, 1);	

	for i in 0..pt_byte{
	    poly.data[i*dimension] = (i as u64) * scale_k; // plaintext
	}
	
	let ct = client.encrypt_matrix_reg(&poly.ntt(), &mut ChaCha20Rng::from_entropy(), &mut rng_pub); // encrypt plaintext

	let (expansion_key_a, expansion_key_b) = generate_query_expansion_key(&params, &mlwe_params, t_exp, &mut ChaCha20Rng::from_entropy(), &mut ChaCha20Rng::from_seed(pack_seed), &mut client);

	let mut ct_a = ct.submatrix(0, 0, 1, 1); // a part
	let mut ct_b = ct.submatrix(1, 0, 1, 1); // b part

	let mlwe_a = rlwe_to_mlwe_a(&params, &ct_a.raw().as_slice().to_vec(), pt_byte_log2); //mlwe a raw
	let mlwe_b = rlwe_to_mlwe_b(&params, &ct_b.raw().as_slice().to_vec(), pt_byte_log2); //mlwe b raw

	let mut mlwe_a_poly = PolyMatrixRaw::zero(&mlwe_params, dimension, dimension);
	let mut mlwe_b_poly = PolyMatrixRaw::zero(&mlwe_params, dimension, 1); //raw

	mlwe_a_poly.as_mut_slice().copy_from_slice(&mlwe_a); // put it into polynomials // 16 by 16 raw
	mlwe_b_poly.as_mut_slice().copy_from_slice(&mlwe_b); // raw

	let mlwe_a_first = mlwe_a_poly.submatrix(0, 0, 1, dimension); // original mlwe a // 1 by 16 raw
	let mut mlwe_a_first_transpose = PolyMatrixRaw::zero(&mlwe_params, dimension, 1); // raw
	mlwe_a_first_transpose.as_mut_slice().copy_from_slice(&mlwe_a_first.as_slice()); // transpose a : 16 by 1 raw

	let mut ct_auto_a_tmp = automorph_alloc(&mlwe_a_first, 129);

	let mut ct_auto_a = automorph_alloc(&mlwe_a_first_transpose, 129); // automorph a 16 by 1 raw

	let mut g_inv_a = PolyMatrixRaw::zero(&mlwe_params, t_exp*dimension, 1); // decompose a : 48 by 1 raw
	gadget_invert_rdim(&mut g_inv_a, &ct_auto_a, dimension); // raw

	let mut g_inv_a_final = PolyMatrixRaw::zero(&mlwe_params, 1, t_exp*dimension);
	g_inv_a_final.as_mut_slice().copy_from_slice(&g_inv_a.as_slice());

	//////////////////a///////////////////

	let mut mlwe_b_first = mlwe_b_poly.submatrix(0, 0, 1, 1);
	let mut ct_auto_b_tmp = automorph_alloc(&mlwe_b_first, 129);


	let result_a = &g_inv_a_final.ntt() * &expansion_key_a[0];

	let key_switch_b = &g_inv_a_final.ntt() * &expansion_key_b[0];
	let result_b = &key_switch_b + &ct_auto_b_tmp.ntt();//&(-&mlwe_b_first.ntt());

	let mut dec = decrypt_mlwe(&params, &mlwe_params, &result_a, &result_b, &client);

	println!("dec : {:?}", dec.as_slice());

    }

    #[test]
    fn test_gadget() {
	let params = params_for_scenario(1 << 30, 1);
	let mut mat = PolyMatrixRaw::zero(&params, 4, 1);
	let g_exp = build_gadget(&params, 1, 3);
	println!("gadget : {}, {}, {}", g_exp.get_poly(0, 0)[0], g_exp.get_poly(0, 1)[0], g_exp.get_poly(0, 2)[0]);

	//824634769409 1, 2, 3 low -> high

	mat.get_poly_mut(0, 0)[37] = 824634769409;
	mat.get_poly_mut(1, 0)[36] = 824634769409;
	mat.get_poly_mut(2, 0)[35] = 824634769409;
	mat.get_poly_mut(3, 0)[34] = 824634769409;

	let mut g_inv_a = PolyMatrixRaw::zero(&params, 12, 1);
	gadget_invert_rdim(&mut g_inv_a, &mat, 4);

	println!("{}", g_inv_a.get_poly(0, 0)[37]);
	println!("{}", g_inv_a.get_poly(1, 0)[36]);
	println!("{}", g_inv_a.get_poly(2, 0)[35]);
	println!("{}", g_inv_a.get_poly(3, 0)[34]);
	println!("{}", g_inv_a.get_poly(4, 0)[37]);
	println!("{}", g_inv_a.get_poly(5, 0)[36]);	
	println!("{}", g_inv_a.get_poly(6, 0)[35]);
	println!("{}", g_inv_a.get_poly(7, 0)[34]);
	println!("{}", g_inv_a.get_poly(8, 0)[37]);
	println!("{}", g_inv_a.get_poly(9, 0)[36]);
	println!("{}", g_inv_a.get_poly(10, 0)[35]);
	println!("{}", g_inv_a.get_poly(11, 0)[34]);
	
	//let a = 
    } 
}
