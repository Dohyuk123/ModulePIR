#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;
use std::{marker::PhantomData, ops::Range, time::Instant};

use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
//use rayon::prelude::*;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{arith::*, client::*, params::*, poly::*};

use crate::convolution::naive_multiply_matrices;
use crate::measurement::Measurement;

use super::{
    bits::*,
    client::*,
    convolution::{negacyclic_perm_u32, Convolution},
    kernel::*,
    lwe::*,
    matmul::matmul_vec_packed,
    modulus_switch::ModulusSwitch,
    packing::*,
    params::*,
    scheme::*,
    transpose::*,
    util::*,
};

//pub fn query_mlwe_pack<'a>(params: &Params, client: &mut Client, query_num: usize)->AlignedMemory64 {
//    let y_client = YClient::new(&mut client, &params);
//    let query_col = y_client.generate_query(SEED_1, params.db_dim_1, true, query_num);
//
//    let query_col_last_row = &query_col[params.poly_len * (1<<(params.db_dim_1 + params.poly_len_log2))..];
//    let packed_query_col = pack_query(&params, query_col_last_row);

//    packed_query_col
//}

pub fn transpose_poly<'a>(params: &Params, mat: &'a PolyMatrixRaw<'a>)-> Vec<u64>{
    //assert_eq!(transpose.get_rows(), mat.get_cols());
    //assert_eq!(transpose.get_cols(), mat.get_rows());

    let mut array = vec![0u64; mat.as_slice().len()];

    let poly_len = params.poly_len;

    let mat_slice = mat.as_slice();
    let row = mat.get_rows();
    let col = mat.get_cols();

    //let mut transposed = PolyMatrixRaw::zero(&params, col, row);

    for c in 0..col{
	for r in 0..row{
	    array[(row*poly_len*c + r*poly_len)..(row*poly_len*c + (r+1)*poly_len)].copy_from_slice(&mat_slice[(col*poly_len*r + c*poly_len)..(col*poly_len*r + (c+1)*poly_len)]);
	}
    }
    array
    //for r in 0..row{
//	for c in 0..col{
//    	    for i in 0..poly_len{
//		assert_eq!(mat.get_poly(r, c)[i], transpose.get_poly_mut(c, r)[i]);
//	    }
//	} 
//    }
}

pub fn mlwe_to_mlwe_a(params: &Params, a: &Vec<u64>, mlwe_dim: usize, n: usize, len: usize) -> Vec<u64>{
    let mut combined_vec: Vec<u64> = vec![0; 2 * len]; // Allocate memory once
    let half_mlwe_dim = mlwe_dim / 2;

    for i in 0..len / n {
       	for j in 0..n / mlwe_dim {
       	// Process even_front_odd_back
       	    let mut even_front_odd_back: Vec<u64> = (0..half_mlwe_dim)
		.map(|k| a[i * n + j * mlwe_dim + 2 * k])
		.chain ((0..half_mlwe_dim).map(|k| a[i * n + j * mlwe_dim + 2 * k + 1]))
		.collect();

       	// Rotate and negate
       	    //let mut even_back = even_front_odd_back.clone();
       	    even_front_odd_back[half_mlwe_dim..mlwe_dim].rotate_right(1);
       	    even_front_odd_back[half_mlwe_dim] = params.modulus - even_front_odd_back[half_mlwe_dim];

       	// Process odd_front_even_back
	    let odd_front_even_back: Vec<u64> = (0..half_mlwe_dim)
		.map(|k| a[i * n + j * mlwe_dim + 2 * k + 1])
		.chain((0..half_mlwe_dim).map(|k| a[i * n + j * mlwe_dim + 2 * k]))
		.collect();

	    // Copy the processed data into the combined vector
            combined_vec[2 * i * n + j * mlwe_dim..2 * i * n + j * mlwe_dim + mlwe_dim]
                .copy_from_slice(&even_front_odd_back);
            combined_vec[(2 * i + 1) * n + j * mlwe_dim..(2 * i + 1) * n + j * mlwe_dim + mlwe_dim]	
                .copy_from_slice(&odd_front_even_back);
        }
    }
    combined_vec
}
 // n : poly len //mlwe_dim : input mlwe dimension
pub fn mlwe_to_mlwe_b(params: &Params, a: &Vec<u64>, mlwe_dim: usize, n: usize) -> Vec<u64>{
    let mut combined_vec: Vec<u64> = vec![0; n]; // Allocate memory once
    let half_mlwe_dim = mlwe_dim / 2;

    for j in 0..n / mlwe_dim {
       	// Process even_front_odd_back
        let mut even_front_odd_back: Vec<u64> = (0..half_mlwe_dim)
	    .map(|k| a[j * mlwe_dim + 2 * k])
	    .chain ((0..half_mlwe_dim).map(|k| a[j * mlwe_dim + 2 * k + 1]))
	    .collect();
	//println!("len: {}", even_front_odd_back.len());

	combined_vec[j * mlwe_dim .. j * mlwe_dim + mlwe_dim].copy_from_slice(&even_front_odd_back);    
	//println!("{:?}", combined_vec);
    }
    combined_vec
}

pub fn mlwe_to_mlwe_b_inverse(params: &Params, a: &mut Vec<u64>, mlwe_bit: usize){
    let poly_len = params.poly_len;
    let mut output: Vec<u64> = vec![0; poly_len];
    let mlwe_byte = 1<<mlwe_bit;

    for i in 0..poly_len / (2*mlwe_byte) {
	for j in 0..mlwe_byte {
	    output[2*j + 2*i*mlwe_byte] = a[j + i*2*mlwe_byte];
	    output[2*j + 1 + 2*i*mlwe_byte] = a[mlwe_byte + j + 2*i*mlwe_byte];
	}
    }
    //println!("a: {:?}", a);
    //println!("output: {:?}", output);
    a.as_mut_slice().copy_from_slice(&output.as_slice());
    
}

pub fn mlwe_to_rlwe_b_packed(params: &Params, a:&mut Vec<u64>, mlwe_bit: usize){
    //let mut tmp_output = mlwe_to_mlwe_b_inverse(params, &a, mlwe_bit);
    let mut mlwe_bit_tmp = mlwe_bit;
    for i in 0..(params.poly_len_log2 - mlwe_bit) {
	mlwe_to_mlwe_b_inverse(params, a , mlwe_bit_tmp);
	mlwe_bit_tmp += 1; 
    }

}

// vector to vector
pub fn rlwe_to_mlwe_a(params: &Params, a: &Vec<u64>, log_pt_byte: usize) -> Vec<u64> {
    let n = 1 << params.poly_len_log2;
    let mut mlwe_vector = mlwe_to_mlwe_a(params, &a, n, n, n);

    for i in 1..(params.poly_len_log2 - log_pt_byte) {
	mlwe_vector = mlwe_to_mlwe_a(params, &mlwe_vector, n >> i, n, n << i);
    }
    mlwe_vector
}


//params: original params // vector to vector
pub fn rlwe_to_mlwe_b(params: &Params, a: &Vec<u64>, log_pt_byte: usize) -> Vec<u64> {
    let n = 1 << params.poly_len_log2;
    let mut mlwe_vector = mlwe_to_mlwe_b(params, &a, n, n);

    for i in 1..(params.poly_len_log2 - log_pt_byte) {
	mlwe_vector = mlwe_to_mlwe_b(params, &mlwe_vector, n >> i, n);
    }
    mlwe_vector
}

pub fn mlwe_to_rlwe_b<'a>(params: &'a Params, output_poly: &mut PolyMatrixRaw<'a>, mlwe_poly: &[u64], mlwe_bit: usize, index: usize) {
    for i in 0..(1<<(mlwe_bit)) {
	output_poly.data[index * (params.poly_len) + i * (1<<(params.poly_len_log2 - mlwe_bit))] = mlwe_poly[i];
    }
}

//a: input mlwe, log_pt_byte: mlwe byte     ->      mlwe_dim -> 2*mlwe_dim
pub fn inverse_mlwe_to_mlwe_a(params: &Params, mut a: Vec<u64>, log_pt_byte: usize) ->Vec<u64> {
    let poly_len = 1<<log_pt_byte;
    let mlwe_len = a.len() / (poly_len << 1);
    //println!("a len, poly len : {}, {}", a.len(), poly_len);
    //println!("mlwe_len : {}", mlwe_len);
    let mut tmp_vec = vec![0u64; poly_len << 1];
    for i in 0..mlwe_len{
	a[i * (poly_len << 1) + poly_len .. (i+1) * (poly_len << 1)].rotate_left(1);
	a[(i+1) * (poly_len <<1) - 1] = params.modulus - a[(i+1) * (poly_len <<1) - 1];

	//println!("{}, {:?}", poly_len, a);

	for j in 0..(poly_len){
	    tmp_vec[2*j] = a[i * (poly_len << 1) + j];
	}
	for j in 0..(poly_len){
	    tmp_vec[2*j+1] = a[i * (poly_len << 1) + poly_len + j];
	}
	a[i * (poly_len << 1) .. (i+1) * (poly_len << 1)].copy_from_slice(&tmp_vec);
    }
    //println!("a: {:?}", a);
    a
}

//a: input mlwe, log_pt_byte : mlwe byte -> params.poly_len // we have to use original parameter
pub fn mlwe_to_rlwe_a(params: &Params, output_poly: &mut PolyMatrixRaw, mut a: Vec<u64>, log_pt_byte: usize){
    let depth = params.poly_len_log2 - log_pt_byte;
    //println!("{}", log_pt_byte);
    //println!("a length: {}", a.len());
    let mut result = inverse_mlwe_to_mlwe_a(params, a, log_pt_byte);
    for i in 1..(depth){
	//println!("{}", log_pt_byte<<i);
	result = inverse_mlwe_to_mlwe_a(params, result, log_pt_byte+i);
        //println!("result leng: {}", result.len());
    }
    
    for i in 0..params.poly_len {
	output_poly.data[i] = result[i];
    }
}

pub fn generate_y_constants<'a>(
    params: &'a Params,
) -> (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>) {
    let mut y_constants = Vec::new();
    let mut neg_y_constants = Vec::new();
    for num_cts_log2 in 1..params.poly_len_log2 + 1 {
        let num_cts = 1 << num_cts_log2;
	//println!("num_cts: {}", num_cts_log2);

        // Y = X^(poly_len / num_cts)
        let mut y_raw = PolyMatrixRaw::zero(params, 1, 1);
        y_raw.data[params.poly_len / num_cts] = 1;
	//println!("polylen: {}, num_cts: {}, poly/num_cts: {}", params.poly_len, num_cts, params.poly_len / num_cts);
	//println!("{:?}", y_raw.as_slice());
        let y = y_raw.ntt();
	//println!("y: {:?}", y);

        let mut neg_y_raw = PolyMatrixRaw::zero(params, 1, 1);
        neg_y_raw.data[params.poly_len / num_cts] = params.modulus - 1;
	//println!("{}", params.modulus - 1);
        let neg_y = neg_y_raw.ntt();

        y_constants.push(y);
        neg_y_constants.push(neg_y);
    }

    (y_constants, neg_y_constants)
}

/// Takes a matrix of u64s and returns a matrix of T's.
///
/// Input is row x cols u64's.
/// Output is out_rows x cols T's.
pub fn split_alloc_mlwe(
    buf: &[u64],
    special_bit_offs: usize,
    rows: usize,
    cols: usize,
    out_rows: usize,
    inp_mod_bits: usize,
    pt_bits: usize,
    pt_byte: usize
) -> Vec<u16> {
    let mut col_byte = cols*pt_byte;
    let mut out = vec![0u16; out_rows * col_byte];
    println!("pt_bits: {}", pt_bits);
    println!("specialbits: {}", special_bit_offs);
    println!("inpmodbits: {}", inp_mod_bits);

    assert!(out_rows >= rows);
    assert!(inp_mod_bits >= pt_bits);
    
    for j in 0..col_byte {
        let mut bytes_tmp = vec![0u8; out_rows * inp_mod_bits / 8];

        // read this column
        let mut bit_offs = 0;
        for i in 0..rows {
            let inp = buf[i * cols + j];
            // if j < 10 {
            //     debug!("({},{}) inp: {}", i, j, inp);
            // }

            if i == rows - 1 {
                bit_offs = special_bit_offs;
            }

            // if j == 4095 {
            //     debug!("write: {}/{} {}/{}", j, cols, i, rows);
            // }
            write_bits(&mut bytes_tmp, inp, bit_offs, inp_mod_bits);
            bit_offs += inp_mod_bits;
        }

        // debug!("stretch: {}", j);

        // now, 'stretch' the column vertically
        let mut bit_offs = 0;
        for i in 0..out_rows {
            // if j == 4095 {
            //     debug!("stretch: {}/{}", i, out_rows);
            //     debug!("reading at offs: {}, {} bits", bit_offs, pt_bits);
            //     debug!("into byte buffer of len: {}", bytes_tmp.len());
            //     debug!("writing at {} in out of len {}", i * cols + j, out.len());
            // }
            let out_val = read_bits(&bytes_tmp, bit_offs, pt_bits);
            out[i * cols + j] = out_val as u16;
            // if j == 4095 {
            //     debug!("wrote at {} in out of len {}", i * cols + j, out.len());
            // }
            bit_offs += pt_bits;
            if bit_offs >= out_rows * inp_mod_bits {
                break;
            }
        }

        // debug!("here {}", j);
        // debug!(
        //     "out {}",
        //     out[(special_bit_offs / pt_bits) * cols + j].to_u64()
        // );
        // debug!("buf {}", buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1));

        assert_eq!(
            out[(special_bit_offs / pt_bits) * cols + j] as u64,
            buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1)
        );
    }

    out
}

pub fn split_alloc(
    buf: &[u64],
    special_bit_offs: usize,
    rows: usize,
    cols: usize,
    out_rows: usize,
    inp_mod_bits: usize,
    pt_bits: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; out_rows * cols];
    println!("pt_bits: {}", pt_bits);
    println!("specialbits: {}", special_bit_offs);
    println!("inpmodbits: {}", inp_mod_bits);

    assert!(out_rows >= rows);
    assert!(inp_mod_bits >= pt_bits);

    for j in 0..cols {
        let mut bytes_tmp = vec![0u8; out_rows * inp_mod_bits / 8];

        // read this column
        let mut bit_offs = 0;
        for i in 0..rows {
            let inp = buf[i * cols + j];
            // if j < 10 {
            //     debug!("({},{}) inp: {}", i, j, inp);
            // }

            if i == rows - 1 {
                bit_offs = special_bit_offs;
            }

            // if j == 4095 {
            //     debug!("write: {}/{} {}/{}", j, cols, i, rows);
            // }
            write_bits(&mut bytes_tmp, inp, bit_offs, inp_mod_bits);
            bit_offs += inp_mod_bits;
        }

        // debug!("stretch: {}", j);

        // now, 'stretch' the column vertically
        let mut bit_offs = 0;
        for i in 0..out_rows {
            // if j == 4095 {
            //     debug!("stretch: {}/{}", i, out_rows);
            //     debug!("reading at offs: {}, {} bits", bit_offs, pt_bits);
            //     debug!("into byte buffer of len: {}", bytes_tmp.len());
            //     debug!("writing at {} in out of len {}", i * cols + j, out.len());
            // }
            let out_val = read_bits(&bytes_tmp, bit_offs, pt_bits);
            out[i * cols + j] = out_val as u16;
            // if j == 4095 {
            //     debug!("wrote at {} in out of len {}", i * cols + j, out.len());
            // }
            bit_offs += pt_bits;
            if bit_offs >= out_rows * inp_mod_bits {
                break;
            }
        }

        // debug!("here {}", j);
        // debug!(
        //     "out {}",
        //     out[(special_bit_offs / pt_bits) * cols + j].to_u64()
        // );
        // debug!("buf {}", buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1));

        assert_eq!(
            out[(special_bit_offs / pt_bits) * cols + j] as u64,
            buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1)
        );
    }

    out
}

pub fn generate_fake_pack_pub_params<'a>(params: &'a Params) -> Vec<PolyMatrixNTT<'a>> {
    let pack_pub_params = raw_generate_expansion_params(
        &params,
        &PolyMatrixRaw::zero(&params, 1, 1),
        params.poly_len_log2,
        params.t_exp_left,
        &mut ChaCha20Rng::from_entropy(),
        &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
    );
    pack_pub_params
}

pub type Precomp<'a> = Vec<(PolyMatrixNTT<'a>, Vec<PolyMatrixNTT<'a>>, Vec<Vec<usize>>)>;

#[derive(Clone)]
pub struct OfflinePrecomputedValues<'a> {
    pub hint_0: Vec<u64>,
    pub hint_1: Vec<u64>,
    pub pseudorandom_query_1: Vec<PolyMatrixNTT<'a>>,
    pub y_constants: (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
    pub smaller_server: Option<YServer<'a, u16>>,
    pub prepacked_lwe: Vec<Vec<PolyMatrixNTT<'a>>>,
    pub fake_pack_pub_params: Vec<PolyMatrixNTT<'a>>,
    pub precomp: Precomp<'a>,
}

pub struct NewServer<'a, T> {
    params: &'a Params,
    mlwe_params: &'a Params,
    smaller_params: Params,
    db_buf_aligned: AlignedMemory64, // db_buf: Vec<u8>, // stored transposed
    phantom: PhantomData<T>,
    pad_rows: bool,
    ypir_params: YPIRParams,
    //log_pt_byte: usize,
}



#[derive(Clone)]
pub struct YServer<'a, T> {
    params: &'a Params,
    smaller_params: Params,
    db_buf_aligned: AlignedMemory64, // db_buf: Vec<u8>, // stored transposed
    phantom: PhantomData<T>,
    pad_rows: bool,
    ypir_params: YPIRParams,
}

pub trait DbRowsPadded {
    fn db_rows_padded(&self) -> usize;
}

impl DbRowsPadded for Params {
    fn db_rows_padded(&self) -> usize {
        let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
        db_rows
        // let db_rows_padded = db_rows + db_rows / (16 * 8);
        // db_rows_padded
    }
}

impl<'a, T> YServer<'a, T>
where
    T: Sized + Copy + ToU64 + Default,
    *const T: ToM512,
{
    pub fn new<'b, I>(
        params: &'a Params,
        mut db: I,
        is_simplepir: bool,
        inp_transposed: bool,
        pad_rows: bool,
    ) -> Self
    where
        I: Iterator<Item = T>,
    {
        // TODO: hack
        // let lwe_params = LWEParams::default();
        let mut ypir_params = YPIRParams::default();
        ypir_params.is_simplepir = is_simplepir;
        let bytes_per_pt_el = std::mem::size_of::<T>(); //1; //((lwe_params.pt_modulus as f64).log2() / 8.).ceil() as usize;

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
	//println!("db_rows: {}", db_rows);
        let db_rows_padded = if pad_rows {
	    //println!("rows_padded : {}", params.db_rows_padded());            
	    params.db_rows_padded()
	    
        } else {
	    db_rows
	    
        };
        let db_cols = if is_simplepir {
            params.instances * params.poly_len
        } else {
            1 << (params.db_dim_2 + params.poly_len_log2)
        };
	println!("poly len: {}", params.poly_len_log2);
	
	//println!("padrows: {}", pad_rows);

        let sz_bytes = db_rows_padded * db_cols * bytes_per_pt_el;

        let mut db_buf_aligned = AlignedMemory64::new(sz_bytes / 8);
        let db_buf_mut = as_bytes_mut(&mut db_buf_aligned);
        let db_buf_ptr = db_buf_mut.as_mut_ptr() as *mut T;

        for i in 0..db_rows {
            for j in 0..db_cols {
                let idx = if inp_transposed {
                    i * db_cols + j
                } else {
			
                    j * db_rows_padded + i
                };
		//println!("idx: {}", idx);

                unsafe {
                    *db_buf_ptr.add(idx) = db.next().unwrap();
		    //println!("db: {}", db_buf_ptr[idx]);
                    // *db_buf_ptr.add(idx) = if i < db_rows {
                    //     db.next().unwrap()
                    // } else {
                    //     T::default()
                    // };
                }
            }
        }

        // Parameters for the second round (the "DoublePIR" round)
        let smaller_params = if is_simplepir {
            params.clone()
        } else {
            let lwe_params = LWEParams::default();
            let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
	    println!("pt_mod: {}", params.pt_modulus);
            let blowup_factor = lwe_params.q2_bits as f64 / pt_bits as f64;
            let mut smaller_params = params.clone();
            smaller_params.db_dim_1 = params.db_dim_2;
            smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
                / params.poly_len as f64)
                .log2()
                .ceil() as usize;

	    println!("lwe_n: {}, polylen: {}, db_dim_2: {}", lwe_params.n, params.poly_len, smaller_params.db_dim_2);

            let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
	    println!("out_rows: {}, params.poly_len_log2: {}", out_rows, params.poly_len_log2);
            assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
            assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));
            smaller_params
        };

        Self {
            params,
            smaller_params,
            db_buf_aligned,
            phantom: PhantomData,
            pad_rows,
            ypir_params,
        }
    }

    pub fn db_rows_padded(&self) -> usize {
        if self.pad_rows {
            self.params.db_rows_padded()
        } else {
            1 << (self.params.db_dim_1 + self.params.poly_len_log2)
        }
    }

    pub fn db_cols(&self) -> usize {
        if self.ypir_params.is_simplepir {
            self.params.instances * self.params.poly_len
        } else {
            1 << (self.params.db_dim_2 + self.params.poly_len_log2)
        }
    }

    pub fn multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u64],
        query_rows: usize,
    ) -> AlignedMemory64 {
        // let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = self.db_cols();
        assert_eq!(aligned_query_packed.len(), K * query_rows * db_rows_padded);
        assert_eq!(query_rows, 1);

        let now = Instant::now();
        let mut result = AlignedMemory64::new(K * db_cols);
        fast_batched_dot_product_avx512::<K, _>(
            self.params,
            result.as_mut_slice(),
            aligned_query_packed,
            db_rows_padded,
            &self.db(),
            db_rows_padded,
            db_cols,
        );
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn lwe_multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u32],
    ) -> Vec<u32> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let db_rows_padded = self.db_rows_padded();
        assert_eq!(aligned_query_packed.len(), K * db_rows_padded);
        // assert_eq!(aligned_query_packed[db_rows + 1], 0);

        let mut result = vec![0u32; (db_cols + 8) * K];
        let now = Instant::now();
        // let mut result = AlignedMemory64::new(K * db_cols + 8);
        // lwe_fast_batched_dot_product_avx512::<K, _>(
        //     self.params,
        //     result.as_mut_slice(),
        //     aligned_query_packed,
        //     db_rows,
        //     &self.db(),
        //     db_rows,
        //     db_cols,
        // );
        let a_rows = db_cols;
        let a_true_cols = db_rows_padded;
        let a_cols = a_true_cols / 4; // order is inverted on purpose, because db is transposed
        let b_rows = a_true_cols;
        let b_cols = K;
        matmul_vec_packed(
            result.as_mut_slice(),
            self.db_u32(),
            aligned_query_packed,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
        );
        let t = Instant::now();
        let result = transpose_generic(&result, db_cols, K);
        debug!("Transpose in {} us", t.elapsed().as_micros());
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn multiply_with_db_ring(
        &self,
        preprocessed_query: &[PolyMatrixNTT], // 8
        col_range: Range<usize>,
        seed_idx: u8,
    ) -> Vec<u64> {
        let db_rows_poly = 1 << (self.params.db_dim_1);
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        assert_eq!(preprocessed_query.len(), db_rows_poly);
	

        let mut result = Vec::new();
        let db = self.db();

        let mut prod = PolyMatrixNTT::zero(self.params, 1, 1);
        let mut db_elem_poly = PolyMatrixRaw::zero(self.params, 1, 1);
        let mut db_elem_ntt = PolyMatrixNTT::zero(self.params, 1, 1);

        for col in col_range.clone() {
            let mut sum = PolyMatrixNTT::zero(self.params, 1, 1);

            for row in 0..db_rows_poly {
                for z in 0..self.params.poly_len {
                    db_elem_poly.data[z] =
                        db[col * db_rows + row * self.params.poly_len + z].to_u64();
                }
                to_ntt(&mut db_elem_ntt, &db_elem_poly);

                multiply(&mut prod, &preprocessed_query[row], &db_elem_ntt);

                if row == db_rows_poly - 1 {
                    add_into(&mut sum, &prod);
                } else {
                    add_into_no_reduce(&mut sum, &prod);
                }
            }

            let sum_raw = sum.raw();
	    //println!("sum_raw length: {}", sum_raw.as_slice().len());

            // do negacyclic permutation (for first mul only)
            if seed_idx == SEED_0 && !self.ypir_params.is_simplepir {
		//println!("if");
                let sum_raw_transformed =
                    negacyclic_perm(sum_raw.get_poly(0, 0), 0, self.params.modulus);
                result.extend(&sum_raw_transformed);
            } else {
		//println!("if else");
                result.extend(sum_raw.as_slice());
            }
        }
	println!("result length: {}", result.len());

        // result
        let now = Instant::now();
        let res = transpose_generic(&result, col_range.len(), self.params.poly_len);
        debug!("transpose in {} us", now.elapsed().as_micros());
        res
    }

    pub fn generate_pseudorandom_query(&self, public_seed_idx: u8) -> Vec<PolyMatrixNTT<'a>> {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys();
        let y_client = YClient::new(&mut client, &self.params);
        let query = y_client.generate_query_impl(public_seed_idx, self.params.db_dim_1, true, 0);
        let query_mapped = query
            .iter()
            .map(|x| x.submatrix(0, 0, 1, 1))
            .collect::<Vec<_>>();

        let mut preprocessed_query = Vec::new();
        for query_raw in query_mapped {
            // let query_raw_transformed =
            //     negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
            // let query_raw_transformed = query_raw.get_poly(0, 0);
            let query_raw_transformed = if public_seed_idx == SEED_0 {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
                // query_raw.get_poly(0, 0).to_owned()
            } else {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
            };
            let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
            query_transformed_pol
                .as_mut_slice()
                .copy_from_slice(&query_raw_transformed);
            preprocessed_query.push(query_transformed_pol.ntt());
        }

        preprocessed_query
    }

    pub fn answer_hint_ring(&self, public_seed_idx: u8, cols: usize) -> Vec<u64> {
        let preprocessed_query = self.generate_pseudorandom_query(public_seed_idx);
	println!("{}", preprocessed_query[0].raw().data[1000]);

        let res = self.multiply_with_db_ring(&preprocessed_query, 0..cols, public_seed_idx);

        res
    }

    pub fn generate_double_hint_mlwe(
	&self,
	mlwe_params: &'a Params,
	public_seed_idx: u8,
	dimension : usize,
	dim_log1: usize,
	pt_byte_log2: usize,
	index: usize,
	client: &'a Client,
    ) -> PolyMatrixNTT<'a> {
	let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx)); // very important!!

	let mut rlwe_ct_vec = Vec::new();

	for i in 0..(1<<dim_log1) {
	    let plaintext = PolyMatrixNTT::zero(self.params, 1, 1);
	    let ct = client.encrypt_matrix_reg(&plaintext, &mut ChaCha20Rng::from_entropy(), &mut rng_pub);
	    //let tmp_array = ct.submatrix(0, 0, 1, 1).raw().as_slice();
	    rlwe_ct_vec.push(rlwe_to_mlwe_a(self.params, &ct.submatrix(0, 0, 1, 1).raw().as_slice().to_vec(), pt_byte_log2));
	}
	
	let mut poly_hint = PolyMatrixRaw::zero(mlwe_params, 1<<(dim_log1 + self.params.poly_len_log2 - pt_byte_log2), dimension);
	let length = rlwe_ct_vec[0].as_slice().len();	
	for i in 0..rlwe_ct_vec.len() {
	    poly_hint.as_mut_slice()[i*length..(i+1)*length].copy_from_slice(&rlwe_ct_vec[i].as_slice());
	}

	let poly_hint_ntt = poly_hint.ntt();
	poly_hint_ntt
    }

    pub fn generate_hint_0(&self) -> Vec<u64> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let lwe_params = LWEParams::default();

        // pseudorandom LWE query is n x db_rows
        let psuedorandom_query =
            generate_matrix_ring(&mut rng_pub, lwe_params.n, lwe_params.n, db_cols);

        // db is db_cols x db_rows (!!!)
        // hint_0 is n x db_cols
        let hint_0 = naive_multiply_matrices(
            &psuedorandom_query,
            lwe_params.n,
            db_cols,
            &self.db(),
            self.db_rows_padded(), // TODO: doesn't quite work
            db_cols,
            true,
        );
        hint_0.iter().map(|&x| x as u64).collect::<Vec<_>>()
    }

    pub fn generate_hint_0_ring(&self) -> Vec<u64> {
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        let lwe_params = LWEParams::default();
        let n = lwe_params.n;
	println!("n hint 0: {}", n);
        let conv = Convolution::new(n);
	println!("conv modulus: {}", conv.params().modulus);

        let mut hint_0 = vec![0u64; n * db_cols];

        let convd_len = conv.params().crt_count * conv.params().poly_len;
	println!("convd_len: {}, conv_polylen: {}", convd_len, conv.params().poly_len);

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));

        let mut v_nega_perm_a = Vec::new();
        for _ in 0..db_rows / n {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }
            let nega_perm_a = negacyclic_perm_u32(&a);
            let nega_perm_a_ntt = conv.ntt(&nega_perm_a);
            v_nega_perm_a.push(nega_perm_a_ntt);
        }

        // limit on the number of times we can add results modulo M before we wrap
        let log2_conv_output =
            log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
        let log2_modulus = log2(conv.params().modulus);
        let log2_max_adds = log2_modulus - log2_conv_output - 1;
        assert!(log2_max_adds > 0);
        let max_adds = 1 << log2_max_adds;
	println!("lwe_modulus: {}", lwe_params.modulus);

        for col in 0..db_cols {
            let mut tmp_col = vec![0u64; convd_len];
            for outer_row in 0..db_rows / n {
                let start_idx = col * self.db_rows_padded() + outer_row * n;
                let pt_col = &self.db()[start_idx..start_idx + n];
                let pt_col_u32 = pt_col
                    .iter()
                    .map(|&x| x.to_u64() as u32)
                    .collect::<Vec<_>>();
                assert_eq!(pt_col_u32.len(), n);
		//println!("length: {}", pt_col_u32.len());                
		let pt_ntt = conv.ntt(&pt_col_u32);
		//println!("length: {}", pt_ntt.len());
		//println!("ptcol: {}", pt_col_u32[0]);

                let convolved_ntt = conv.pointwise_mul(&v_nega_perm_a[outer_row], &pt_ntt);
		//println!("length: {}", convolved_ntt.len());
                for r in 0..convd_len {
                    tmp_col[r] += convolved_ntt[r] as u64;
                }

                if outer_row % max_adds == max_adds - 1 || outer_row == db_rows / n - 1 {
                    let mut col_poly_u32 = vec![0u32; convd_len];
                    for i in 0..conv.params().crt_count {
                        for j in 0..conv.params().poly_len {
                            let val = barrett_coeff_u64(
                                conv.params(),
                                tmp_col[i * conv.params().poly_len + j],
                                i,
                            );
			    //println!("moduli: {}", conv.params().moduli[i]);
                            col_poly_u32[i * conv.params().poly_len + j] = val as u32;
                        }
                    }
                    let col_poly_raw = conv.raw(&col_poly_u32);
                    for i in 0..n {
                        hint_0[i * db_cols + col] += col_poly_raw[i] as u64;
                        hint_0[i * db_cols + col] %= 1u64 << 32;
                    }
                    tmp_col.fill(0);
                }
            }
        }

        hint_0
    }

    pub fn answer_query(&self, aligned_query_packed: &[u64]) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<1>(aligned_query_packed, 1)
    }

    pub fn answer_batched_queries<const K: usize>(
        &self,
        aligned_queries_packed: &[u64],
    ) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<K>(aligned_queries_packed, 1)
    }

    pub fn perform_offline_precomputation_simplepir(
        &self,
        measurement: Option<&mut Measurement>,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters

        let params = self.params;
        assert!(self.ypir_params.is_simplepir);

        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;

        // Begin offline precomputation

        let now = Instant::now();
        let hint_0: Vec<u64> = self.answer_hint_ring(SEED_0, db_cols);
        // hint_0 is poly_len x db_cols
        let simplepir_prep_time_ms = now.elapsed().as_millis();
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }

        let now = Instant::now();
        let y_constants = generate_y_constants(&params);

        let combined = [&hint_0[..], &vec![0u64; db_cols]].concat();
        assert_eq!(combined.len(), db_cols * (params.poly_len + 1));
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, num_rlwe_outputs);

        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            let tup = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        OfflinePrecomputedValues {
            hint_0,
            hint_1: vec![],
            pseudorandom_query_1: vec![],
            y_constants,
            smaller_server: None,
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        }
    }

    pub fn perform_offline_precomputation(
        &self,
        measurement: Option<&mut Measurement>,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters

        let params = self.params;
        let lwe_params = LWEParams::default();
        assert!(!self.ypir_params.is_simplepir);

        let db_cols = 1 << (params.db_dim_2 + params.poly_len_log2);

        // LWE reduced moduli
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let lwe_q_prime = lwe_params.get_q_prime_2();
	println!("lwe_q_prime: {}", lwe_q_prime);
	

        // The number of bits represented by a plaintext RLWE coefficient
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
	println!("pt bits : {}, lwe_q_prim: {}", pt_bits, lwe_q_prime_bits);
        // assert_eq!(pt_bits, 16);

        // The factor by which ciphertext values are bigger than plaintext values
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);
        // assert!(blowup_factor.ceil() - blowup_factor >= 0.05);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        // This is rounded to start on a pt_bits boundary
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;
        let special_bit_offs = special_offs * pt_bits;
	println!("n: {}, q_prime_bits: {}, pt_bits: {}, special_off: {}, special_bit_offs: {}", lwe_params.n, lwe_q_prime_bits, pt_bits , special_offs, special_bit_offs);

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;
	println!("blowup: {}, lwe_params.n: {}, poly_len: {}", blowup_factor, lwe_params.n, params.poly_len);

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
	println!("outrows: {}", out_rows);
        let rho = 1 << smaller_params.db_dim_2;
	println!("rho: {}", rho);
        assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
        assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

        debug!(
            "the first {} LWE output ciphertexts of the DoublePIR round (out of {} total) are query-indepednent",
            special_offs, out_rows
        );
        debug!(
            "the next {} LWE output ciphertexts are query-dependent",
            blowup_factor.ceil() as usize
        );
        debug!("the rest are zero");

        // Begin offline precomputation

        let now = Instant::now();
        let hint_0: Vec<u64> = self.generate_hint_0_ring();
        // hint_0 is n x db_cols
        let simplepir_prep_time_ms = now.elapsed().as_millis();
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }
        debug!("Answered hint (ring) in {} us", now.elapsed().as_micros());

        // compute (most of) the secondary hint
        let intermediate_cts = [&hint_0[..], &vec![0u64; db_cols]].concat();
        let intermediate_cts_rescaled = intermediate_cts
            .iter()
            .map(|x| rescale(*x, lwe_params.modulus, lwe_q_prime))
            .collect::<Vec<_>>();
	
	println!("lwe_params.modulus: {}, lwe_q_prime: {}", lwe_params.modulus, lwe_q_prime);
	//println!("{:?}", intermediate_cts_rescaled);

        // split and do a second PIR over intermediate_cts
        // split into blowup_factor=q/p instances (so that all values are now mod p)
        // the second PIR is over a database of db_cols x (blowup_factor * (lwe_params.n + 1)) values mod p

        // inp: (lwe_params.n + 1, db_cols)
        // out: (out_rows >= (lwe_params.n + 1) * blowup_factor, db_cols)
        //      we are 'stretching' the columns (and padding)

        debug!("Splitting intermediate cts...");

        let smaller_db = split_alloc(
            &intermediate_cts_rescaled,
            special_bit_offs,
            lwe_params.n + 1,
            db_cols,
            out_rows,
            lwe_q_prime_bits,
            pt_bits,
        );
	//println!("{:?}", smaller_db);
        assert_eq!(smaller_db.len(), db_cols * out_rows);

        debug!("Done splitting intermediate cts.");

        // This is the 'intermediate' db after the first pass of PIR and expansion
        let smaller_server: YServer<u16> = YServer::<u16>::new(
            &self.smaller_params,
            smaller_db.into_iter(),
            false,
            true,
            false,
        );
        debug!("gen'd smaller server.");

        let hint_1 = smaller_server.answer_hint_ring(
            SEED_1,
            1 << (smaller_server.params.db_dim_2 + smaller_server.params.poly_len_log2),
        );
	println!("seed: {}, {}", SEED_0, SEED_1);
        assert_eq!(hint_1.len(), params.poly_len * out_rows);
        assert_eq!(hint_1[special_offs], 0);
        assert_eq!(hint_1[special_offs + 1], 0);

        let pseudorandom_query_1 = smaller_server.generate_pseudorandom_query(SEED_1);
        let y_constants = generate_y_constants(&params);
	//println!("yconst: {:?}", y_constants.data);

        let combined = [&hint_1[..], &vec![0u64; out_rows]].concat();
        assert_eq!(combined.len(), out_rows * (params.poly_len + 1));
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, rho);


        let now = Instant::now();
        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            let tup = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        OfflinePrecomputedValues {
            hint_0,
            hint_1,
            pseudorandom_query_1,
            y_constants,
            smaller_server: Some(smaller_server),
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        }
    }

    /// Perform SimplePIR-style YPIR
    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[u64],
        offline_vals: &OfflinePrecomputedValues<'a>,
        pack_pub_params_row_1s: &[&[PolyMatrixNTT<'a>]],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<u8>> {
        assert!(self.ypir_params.is_simplepir);

        // Set up some parameters

        let params = self.params;

        let y_constants = &offline_vals.y_constants;
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let precomp = &offline_vals.precomp;

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        assert_eq!(first_dim_queries_packed.len(), params.db_rows_padded());

        // Begin online computation

        let first_pass = Instant::now();
        debug!("Performing mul...");
        let mut intermediate = AlignedMemory64::new(db_cols);
        fast_batched_dot_product_avx512::<1, T>(
            &params,
            intermediate.as_mut_slice(),
            first_dim_queries_packed,
            db_rows,
            self.db(),
            db_rows,
            db_cols,
        );
        debug!("Done w mul...");
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
        }

        let ring_packing = Instant::now();
        let num_rlwe_outputs = db_cols / params.poly_len;
        let packed = pack_many_lwes(
            &params,
            &prepacked_lwe,
            &precomp,
            intermediate.as_slice(),
            num_rlwe_outputs,
            &pack_pub_params_row_1s[0],
            &y_constants,
        );
        debug!("Packed...");
        if let Some(m) = measurement {
            m.online.ring_packing_time_ms = ring_packing.elapsed().as_millis() as usize;
        }

        let mut packed_mod_switched = Vec::with_capacity(packed.len());
        for ct in packed.iter() {
            let res = ct.raw();
            let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
            packed_mod_switched.push(res_switched);
        }

        packed_mod_switched
    }

    pub fn perform_online_computation<const K: usize>(
        &self,
        offline_vals: &mut OfflinePrecomputedValues<'a>,
        first_dim_queries_packed: &[u32],
        second_dim_queries: &[(&[u64], &[PolyMatrixNTT<'a>])],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        // Set up some parameters

        let params = self.params;
	println!("params.ptbyte: {}", params.pt_modulus);
        let lwe_params = LWEParams::default();

        let db_cols = self.db_cols();

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        // LWE reduced moduli
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let lwe_q_prime = lwe_params.get_q_prime_2();

	println!("rlwe prime 1: {}, rlwe prime 2: {}, lwe prime2: {}", rlwe_q_prime_1, rlwe_q_prime_2, lwe_q_prime);

        // The number of bits represented by a plaintext RLWE coefficient
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
        // assert_eq!(pt_bits, 16);

        // The factor by which ciphertext values are bigger than plaintext values
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);
        // assert!(blowup_factor.ceil() - blowup_factor >= 0.05);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        // This is rounded to start on a pt_bits boundary
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
        let rho = 1 << smaller_params.db_dim_2;
        assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
        assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

        // Load offline precomputed values
        let hint_1_combined = &mut offline_vals.hint_1;
        let pseudorandom_query_1 = &offline_vals.pseudorandom_query_1;
        let y_constants = &offline_vals.y_constants;
        let smaller_server = offline_vals.smaller_server.as_mut().unwrap();
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let fake_pack_pub_params = &offline_vals.fake_pack_pub_params;
        let precomp = &offline_vals.precomp;

        // Begin online computation

        let online_phase = Instant::now();
        let first_pass = Instant::now();
        let intermediate = self.lwe_multiply_batched_with_db_packed::<K>(first_dim_queries_packed);
        let simplepir_resp_bytes = intermediate.len() / K * (lwe_q_prime_bits as usize) / 8;
        debug!("simplepir_resp_bytes {} bytes", simplepir_resp_bytes);
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        debug!("First pass took {} us", first_pass.elapsed().as_micros());

        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
            m.online.simplepir_resp_bytes = simplepir_resp_bytes;
        }

        debug!("intermediate.len(): {}", intermediate.len());
        let mut second_pass_time_ms = 0;
        let mut ring_packing_time_ms = 0;
        let mut responses = Vec::new();
        for (intermediate_chunk, (packed_query_col, pack_pub_params_row_1s)) in intermediate
            .as_slice()
            .chunks(db_cols)
            .zip(second_dim_queries.iter())
        {
            let second_pass = Instant::now();
            let intermediate_cts_rescaled = intermediate_chunk
                .iter()
                .map(|x| rescale(*x as u64, lwe_params.modulus, lwe_q_prime))
                .collect::<Vec<_>>();
            assert_eq!(intermediate_cts_rescaled.len(), db_cols);
            debug!(
                "intermediate_cts_rescaled[0] = {}",
                intermediate_cts_rescaled[0]
            );

            let now = Instant::now();
            // modify the smaller_server db to include the intermediate values
            // let mut smaller_server_clone = smaller_server.clone();
            {
                // remember, this is stored in 'transposed' form
                // so it is out_cols x db_cols
                let smaller_db_mut: &mut [u16] = smaller_server.db_mut();
                for j in 0..db_cols {
                    // new value to write into the db
                    let val = intermediate_cts_rescaled[j];

                    for m in 0..blowup_factor.ceil() as usize {
                        // index in the transposed db
                        let out_idx = (special_offs + m) * db_cols + j;

                        // part of the value to write into the db
                        let val_part = ((val >> (m * pt_bits)) & ((1 << pt_bits) - 1)) as u16;

                        // assert_eq!(smaller_db_mut[out_idx], DoubleType::default());
                        smaller_db_mut[out_idx] = val_part;
                    }
                }
            }
            debug!("load secondary hint {} us", now.elapsed().as_micros());

            let now = Instant::now();
            {
                let blowup_factor_ceil = blowup_factor.ceil() as usize;

                let phase = Instant::now();
                let secondary_hint = smaller_server.multiply_with_db_ring(
                    &pseudorandom_query_1,
                    special_offs..special_offs + blowup_factor_ceil,
                    SEED_1,
                );
                debug!(
                    "multiply_with_db_ring took: {} us",
                    phase.elapsed().as_micros()
                );
                // let phase = Instant::now();
                // let secondary_hint =
                //     smaller_server_clone.answer_hint(SEED_1, special_offs..special_offs + blowup_factor_ceil);
                // debug!(
                //     "traditional answer_hint took: {} us",
                //     phase.elapsed().as_micros()
                // );

                assert_eq!(secondary_hint.len(), params.poly_len * blowup_factor_ceil);

                for i in 0..params.poly_len {
                    for j in 0..blowup_factor_ceil {
                        let inp_idx = i * blowup_factor_ceil + j;
                        let out_idx = i * out_rows + special_offs + j;

                        // assert_eq!(hint_1_combined[out_idx], 0); // we no longer clone for each query, just overwrite
                        hint_1_combined[out_idx] = secondary_hint[inp_idx];
                    }
                }
            }
            debug!("compute secondary hint in {} us", now.elapsed().as_micros());

            assert_eq!(hint_1_combined.len(), params.poly_len * out_rows);

            let response: AlignedMemory64 = smaller_server.answer_query(packed_query_col);

            second_pass_time_ms += second_pass.elapsed().as_millis();
            let ring_packing = Instant::now();
            let now = Instant::now();
            assert_eq!(response.len(), 1 * out_rows);

            // combined is now (poly_len + 1) * (out_rows)
            // let combined = [&hint_1_combined[..], response.as_slice()].concat();
            let mut excess_cts = Vec::with_capacity(blowup_factor.ceil() as usize);
            for j in special_offs..special_offs + blowup_factor.ceil() as usize {
                let mut rlwe_ct = PolyMatrixRaw::zero(&params, 2, 1);

                // 'a' vector
                // put this in negacyclic order
                let mut poly = Vec::new();
                for k in 0..params.poly_len {
                    poly.push(hint_1_combined[k * out_rows + j]);
                }
                let nega = negacyclic_perm(&poly, 0, params.modulus);

                rlwe_ct.get_poly_mut(0, 0).copy_from_slice(&nega);

                // for k in 0..params.poly_len {
                //     rlwe_ct.get_poly_mut(0, 0)[k] = nega[k];
                // }

                // let j_within_last = j % params.poly_len;
                // prepacked_lwe_mut.last_mut().unwrap()[j_within_last] = rlwe_ct.ntt();
                excess_cts.push(rlwe_ct.ntt());
            }
            debug!("in between: {} us", now.elapsed().as_micros());

            // assert_eq!(pack_pub_params_row_1s[0].rows, 1);

            let mut packed = pack_many_lwes(
                &params,
                &prepacked_lwe,
                &precomp,
                response.as_slice(),
                rho,
                &pack_pub_params_row_1s,
                &y_constants,
            );

            let now = Instant::now();
            let mut pack_pub_params = fake_pack_pub_params.clone();
            for i in 0..pack_pub_params.len() {
                let uncondensed = uncondense_matrix(params, &pack_pub_params_row_1s[i]);
                pack_pub_params[i].copy_into(&uncondensed, 1, 0);
            }
            debug!("uncondense pub params: {} us", now.elapsed().as_micros());
            let now = Instant::now();
            let other_packed =
                pack_using_single_with_offset(&params, &pack_pub_params, &excess_cts, special_offs);
            add_into(&mut packed[0], &other_packed);
            debug!(
                "pack_using_single_with_offset: {} us",
                now.elapsed().as_micros()
            );

            let now = Instant::now();
            let mut packed_mod_switched = Vec::with_capacity(packed.len());
            for ct in packed.iter() {
                let res = ct.raw();
                let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
                packed_mod_switched.push(res_switched);
            }
            debug!("switching: {} us", now.elapsed().as_micros());
            // debug!("Preprocessing pack in {} us", now.elapsed().as_micros());
            // debug!("");
            ring_packing_time_ms += ring_packing.elapsed().as_millis();

            // packed is blowup_factor ring ct's
            // these encode, contiguously [poly_len + 1, blowup_factor]
            // (and some padding)
            assert_eq!(packed.len(), rho);

            responses.push(packed_mod_switched);
        }
        debug!(
            "Total online time: {} us",
            online_phase.elapsed().as_micros()
        );
        debug!("");

        if let Some(ref mut m) = measurement {
            m.online.second_pass_time_ms = second_pass_time_ms as usize;
            m.online.ring_packing_time_ms = ring_packing_time_ms as usize;
        }

        responses
    }

    // generic function that returns a u8 or u16:
    pub fn db(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.db_buf_aligned.as_ptr() as *mut T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_u16(&self) -> &[u16] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u16,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u16>(),
            )
        }
    }

    pub fn db_u32(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u32,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u32>(),
            )
        }
    }

    pub fn get_elem(&self, row: usize, col: usize) -> T {
        self.db()[col * self.db_rows_padded() + row] // stored transposed
    }

    pub fn get_row(&self, row: usize) -> Vec<T> {
        let db_cols = self.db_cols();
        let mut res = Vec::with_capacity(db_cols);
        for col in 0..db_cols {
            res.push(self.get_elem(row, col));
        }
        res
        // // convert to u8 contiguously
        // let mut res_u8 = Vec::with_capacity(db_cols * std::mem::size_of::<T>());
        // for &x in res.iter() {
        //     res_u8.extend_from_slice(&x.to_u64().to_le_bytes()[..std::mem::size_of::<T>()]);
        // }
        // res_u8
    }
}

#[cfg(not(target_feature = "avx2"))]
#[allow(non_camel_case_types)]
type __m512i = u64;

pub trait ToM512 {
    fn to_m512(self) -> __m512i;
}

#[cfg(target_feature = "avx512f")]
mod m512_impl {
    use super::*;

    impl ToM512 for *const u8 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu8_epi64(_mm_loadl_epi64(self as *const _)) }
        }
    }

    impl ToM512 for *const u16 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu16_epi64(_mm_load_si128(self as *const _)) }
        }
    }

    impl ToM512 for *const u32 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu32_epi64(_mm256_load_si256(self as *const _)) }
        }
    }
}

#[cfg(not(target_feature = "avx512f"))]
mod m512_impl {
    use super::*;

    impl ToM512 for *const u8 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            self as __m512i
        }
    }

    impl ToM512 for *const u16 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            self as __m512i
        }
    }

    impl ToM512 for *const u32 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            self as __m512i
        }
    }
}

pub trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}



///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////new protocol///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////



impl<'a, T> NewServer<'a, T>
where
    T: Sized + Copy + ToU64 + Default,
    *const T: ToM512,
{
    pub fn new<'b, I>(
        params: &'a Params,
	mlwe_params: &'a Params, // plaintext_byte
        mut db: I,
        is_simplepir: bool,
        inp_transposed: bool,
        pad_rows: bool,
	 // new param : plaintext byte
    ) -> Self
    where
        I: Iterator<Item = T>,
    {
        // TODO: hack
        // let lwe_params = LWEParams::default();
        let mut ypir_params = YPIRParams::default();
        ypir_params.is_simplepir = is_simplepir;
	let msg_byte_log2 = mlwe_params.poly_len_log2; // mlwe
	let msg_byte = 1<<msg_byte_log2;

        let bytes_per_pt_el = std::mem::size_of::<T>(); //1; //((lwe_params.pt_modulus as f64).log2() / 8.).ceil() as usize;
	//let pt_byte = 1 << mlwe_params.poly_len_log2;
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2); // changed
	//println!("db_rows: {}", db_rows);
        let db_rows_padded = if pad_rows {
	    //println!("rows_padded : {}", params.db_rows_padded());            
	    params.db_rows_padded()
	    
        } else {
	    db_rows
	    
        };
        let db_cols = if is_simplepir {
            params.instances * params.poly_len
        } else {
            1 << (params.db_dim_2 + params.poly_len_log2)
        };
	
	//println!("padrows: {}", pad_rows);

        let sz_bytes = db_rows_padded * db_cols * bytes_per_pt_el * msg_byte;

        let mut db_buf_aligned = AlignedMemory64::new(sz_bytes / 8);
        let db_buf_mut = as_bytes_mut(&mut db_buf_aligned);
        let db_buf_ptr = db_buf_mut.as_mut_ptr() as *mut T;

        for i in 0..(db_rows) {
            for j in 0..db_cols {
		for k in 0..msg_byte{
                    let idx = if inp_transposed {
                        i * db_cols + j
                    } else {
		    
                        (j * db_rows_padded << msg_byte) + (i << msg_byte) + k // ???
                    };
		//println!("idx: {}", idx);

                    unsafe {
                        *db_buf_ptr.add(idx) = db.next().unwrap();
                        // *db_buf_ptr.add(idx) = if i < db_rows {
                        //     db.next().unwrap()
                        // } else {
                        //     T::default()
                        // };
                    }
		}
            }
        }

////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////next time from here//////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

        // Parameters for the second round (the "DoublePIR" round)
        let smaller_params = if is_simplepir {
            params.clone()
        } else {
            let lwe_params = LWEParams::default();
            let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
            let blowup_factor = lwe_params.q2_bits as f64 / pt_bits as f64;
            let mut smaller_params = params.clone();
            smaller_params.db_dim_1 = params.db_dim_2;
            smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
                / params.poly_len as f64)
                .log2()
                .ceil() as usize;

            let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
            assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
            assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));
            smaller_params
        };

        Self {
            params,
	    mlwe_params,
            smaller_params,
            db_buf_aligned,
            phantom: PhantomData,
            pad_rows,
            ypir_params,
	    //log_pt_byte,
        }
    }

    pub fn db_rows_padded(&self) -> usize {
        if self.pad_rows {
            self.params.db_rows_padded()
        } else {
            1 << (self.params.db_dim_1 + self.params.poly_len_log2 + self.mlwe_params.poly_len_log2)
        }
    }

    pub fn db_cols(&self) -> usize {
        if self.ypir_params.is_simplepir {
            self.params.instances * self.params.poly_len
        } else {
            1 << (self.params.db_dim_2 + self.params.poly_len_log2)
        }
    }

    pub fn multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u64],
        query_rows: usize,
    ) -> AlignedMemory64 {
        // let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = self.db_cols();
        assert_eq!(aligned_query_packed.len(), K * query_rows * db_rows_padded);
        assert_eq!(query_rows, 1);

        let now = Instant::now();
        let mut result = AlignedMemory64::new(K * db_cols);
        fast_batched_dot_product_avx512::<K, _>(
            self.params,
            result.as_mut_slice(),
            aligned_query_packed,
            db_rows_padded,
            &self.db(),
            db_rows_padded,
            db_cols,
        );
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn lwe_multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u32],
    ) -> Vec<u32> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let db_rows_padded = self.db_rows_padded();
        assert_eq!(aligned_query_packed.len(), K * db_rows_padded);
        // assert_eq!(aligned_query_packed[db_rows + 1], 0);

        let mut result = vec![0u32; (db_cols + 8) * K];
        let now = Instant::now();
        // let mut result = AlignedMemory64::new(K * db_cols + 8);
        // lwe_fast_batched_dot_product_avx512::<K, _>(
        //     self.params,
        //     result.as_mut_slice(),
        //     aligned_query_packed,
        //     db_rows,
        //     &self.db(),
        //     db_rows,
        //     db_cols,
        // );
        let a_rows = db_cols;
        let a_true_cols = db_rows_padded;
        let a_cols = a_true_cols / 4; // order is inverted on purpose, because db is transposed
        let b_rows = a_true_cols;
        let b_cols = K;
        matmul_vec_packed(
            result.as_mut_slice(),
            self.db_u32(),
            aligned_query_packed,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
        );
        let t = Instant::now();
        let result = transpose_generic(&result, db_cols, K);
        debug!("Transpose in {} us", t.elapsed().as_micros());
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn multiply_with_db_ring(
        &self,
        preprocessed_query: &[PolyMatrixNTT],
        col_range: Range<usize>,
        seed_idx: u8,
    ) -> Vec<u64> {
        let db_rows_poly = 1 << (self.params.db_dim_1);
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        assert_eq!(preprocessed_query.len(), db_rows_poly);

        // assert_eq!(db_rows_poly, 1); // temporary restriction

        // let mut preprocessed_query = Vec::new();
        // for query_el in query {
        //     let query_raw = query_el.raw();
        //     let query_raw_transformed =
        //         negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
        //     let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
        //     query_transformed_pol
        //         .as_mut_slice()
        //         .copy_from_slice(&query_raw_transformed);
        //     preprocessed_query.push(query_transformed_pol.ntt());
        // }

        let mut result = Vec::new();
        let db = self.db();

        let mut prod = PolyMatrixNTT::zero(self.params, 1, 1);
        let mut db_elem_poly = PolyMatrixRaw::zero(self.params, 1, 1);
        let mut db_elem_ntt = PolyMatrixNTT::zero(self.params, 1, 1);

        for col in col_range.clone() {
            let mut sum = PolyMatrixNTT::zero(self.params, 1, 1);

            for row in 0..db_rows_poly {
                for z in 0..self.params.poly_len {
                    db_elem_poly.data[z] =
                        db[col * db_rows + row * self.params.poly_len + z].to_u64();
                }
                to_ntt(&mut db_elem_ntt, &db_elem_poly);

                multiply(&mut prod, &preprocessed_query[row], &db_elem_ntt);

                if row == db_rows_poly - 1 {
                    add_into(&mut sum, &prod);
                } else {
                    add_into_no_reduce(&mut sum, &prod);
                }
            }

            let sum_raw = sum.raw();

            // do negacyclic permutation (for first mul only)
            if seed_idx == SEED_0 && !self.ypir_params.is_simplepir {
                let sum_raw_transformed =
                    negacyclic_perm(sum_raw.get_poly(0, 0), 0, self.params.modulus);
                result.extend(&sum_raw_transformed);
            } else {
                result.extend(sum_raw.as_slice());
            }
        }

        // result
        let now = Instant::now();
        let res = transpose_generic(&result, col_range.len(), self.params.poly_len);
        debug!("transpose in {} us", now.elapsed().as_micros());
        res
    }

    pub fn generate_pseudorandom_query(&self, public_seed_idx: u8) -> Vec<PolyMatrixNTT<'a>> {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys();
        let y_client = YClient::new(&mut client, &self.params);
        let query = y_client.generate_query_impl(public_seed_idx, self.params.db_dim_1, true, 0);
        let query_mapped = query
            .iter()
            .map(|x| x.submatrix(0, 0, 1, 1))
            .collect::<Vec<_>>();

        let mut preprocessed_query = Vec::new();
        for query_raw in query_mapped {
            // let query_raw_transformed =
            //     negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
            // let query_raw_transformed = query_raw.get_poly(0, 0);
            let query_raw_transformed = if public_seed_idx == SEED_0 {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
                // query_raw.get_poly(0, 0).to_owned()
            } else {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
            };
            let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
            query_transformed_pol
                .as_mut_slice()
                .copy_from_slice(&query_raw_transformed);
            preprocessed_query.push(query_transformed_pol.ntt());
        }

        preprocessed_query
    }

    pub fn answer_hint_ring(&self, public_seed_idx: u8, cols: usize) -> Vec<u64> {
        let preprocessed_query = self.generate_pseudorandom_query(public_seed_idx);

        let res = self.multiply_with_db_ring(&preprocessed_query, 0..cols, public_seed_idx);

        res
    }

    pub fn generate_hint_0(&self) -> Vec<u64> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let lwe_params = LWEParams::default();

        // pseudorandom LWE query is n x db_rows
        let psuedorandom_query =
            generate_matrix_ring(&mut rng_pub, lwe_params.n, lwe_params.n, db_cols);

        // db is db_cols x db_rows (!!!)
        // hint_0 is n x db_cols
        let hint_0 = naive_multiply_matrices(
            &psuedorandom_query,
            lwe_params.n,
            db_cols,
            &self.db(),
            self.db_rows_padded(), // TODO: doesn't quite work
            db_cols,
            true,
        );
        hint_0.iter().map(|&x| x as u64).collect::<Vec<_>>()
    }

////////////////////////////from here///////////////////////

    //log_pt_byte //mlwe_dim : dimension of mlwe // n: dim of rlwe // len: length of vector a
    pub fn mlwe_to_mlwe(a: &Vec<u32>, mlwe_dim: usize, n: usize, len: usize) -> Vec<u32>{
    	let mut combined_vec: Vec<u32> = vec![0; 2 * len]; // Allocate memory once
    	let half_mlwe_dim = mlwe_dim / 2;

    	for i in 0..len / n {
        	for j in 0..n / mlwe_dim {
            	// Process even_front_odd_back
            	let even_front_odd_back: Vec<u32> = (0..half_mlwe_dim)
                    .map(|k| a[i * n + j * mlwe_dim + 2 * k])
                    .chain((0..half_mlwe_dim).map(|k| a[i * n + j * mlwe_dim + 2 * k + 1]))
                    .collect();

            	// Rotate and negate
            	let mut even_back = even_front_odd_back.clone();
            	even_back[half_mlwe_dim..mlwe_dim].rotate_right(1);
            	even_back[half_mlwe_dim] = even_back[half_mlwe_dim].wrapping_neg();

            	// Process odd_front_even_back
            	let odd_front_even_back: Vec<u32> = (0..half_mlwe_dim)
            	    .map(|k| a[i * n + j * mlwe_dim + 2 * k + 1])
            	    .chain((0..half_mlwe_dim).map(|k| a[i * n + j * mlwe_dim + 2 * k]))
            	    .collect();

            	// Copy the processed data into the combined vector
            	combined_vec[2 * i * n + j * mlwe_dim..2 * i * n + j * mlwe_dim + mlwe_dim]
            	    .copy_from_slice(&even_back);
            	combined_vec[(2 * i + 1) * n + j * mlwe_dim..(2 * i + 1) * n + j * mlwe_dim + mlwe_dim]	
         	    .copy_from_slice(&odd_front_even_back);
            }
    	}
    	combined_vec
    }

    pub fn rlwe_to_mlwe(a: &Vec<u32>, log_pt_byte: usize, poly_len_log2: usize) -> Vec<u32> {
	let n = 1 << poly_len_log2;
    	let mut mlwe_vector = Self::mlwe_to_mlwe(&a, n, n, n); // mlwedim, n, len

    	for i in 1..(poly_len_log2 - log_pt_byte) {
    	    mlwe_vector = Self::mlwe_to_mlwe(&mlwe_vector, n >> i, n, n << i);
    	}
    	mlwe_vector
    }

    fn mlwe_transpose( // db_row, db_col: 저장 매트릭스 기준
    	vec: &Vec<u32>, row: usize, col: usize, pt_byte: usize
    ) -> Vec<u32> {
        // 결과를 저장할 벡터를 row*col*pt_byte 크기로 생성
        let mut result = vec![0u32; row * col * pt_byte];

        // (i, j)의 위치에서, 기존 열 단위 저장을 행 단위로 변경
        for i in 0..col {
            for j in 0..row {
                let src_index = (j * col + i) * pt_byte;
                let dst_index = (i * row + j) * pt_byte;

                // 각 pt_byte 크기만큼 복사
                result[dst_index..dst_index + pt_byte]
                    .copy_from_slice(&vec[src_index..src_index + pt_byte]);
            }
        }

        result
    }

    /*fn mlwe_matrix_mul(
        a: &Vec<u32>, row_a: usize, col_a: usize,
        b: &Vec<u32>, row_b: usize, col_b: usize,
        pt_byte: usize
    ) -> Vec<u32> {
	//let log2_conv_output = log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.
        // 결과 행렬의 크기는 row_a x col_b
        assert_eq!(col_a, row_b, "행렬 곱셈을 위해서는 col_a와 row_b가 같아야 합니다.");
        let mut result = vec![0u32; row_a * col_b * pt_byte];

        // 순차적으로 각 행렬 곱셈을 수행
        for i in 0..row_a {
            for j in 0..col_b {
                let mut temp = vec![0u32; pt_byte]; // pt_byte 크기의 임시 버퍼
                for k in 0..col_a {
                    let a_index = (k * row_a + i) * pt_byte; // a는 열 단위로 저장됨
                    let b_index = (j * row_b + k) * pt_byte; // b도 열 단위로 저장됨
                
                    let a_element = &a[a_index..a_index + pt_byte];
                    let b_element = &b[b_index..b_index + pt_byte];

                    // NTT 변환 후 pointwise 곱셈
                    let ntt_a = conv.ntt(a_element);
                    let ntt_b = conv.ntt(b_element);

                    // pointwise_mul의 결과를 temp에 더해줌
                    let mul_result = conv.pointwise_mul(&ntt_a, &ntt_b);
                    for byte_idx in 0..pt_byte {
                        temp[byte_idx] = temp[byte_idx].wrapping_add(mul_result[byte_idx]);
                    }
                }

                // 계산된 temp를 결과 배열에 저장
                let result_index = (i * col_b + j) * pt_byte;
                result[result_index..result_index + pt_byte].copy_from_slice(&temp);
            }
        }

        result
    }
    */
    //pub fn mlwe_transpose(a: &Vec<u32>, log_pt_byte: usize, row: usize, col: usize)->Vec<u32> {

    //}

    pub fn generate_hint_0_ring(&self) -> Vec<u64> {
        let db_rows = 1 << (self.params.db_dim_1);
        let db_cols = self.db_cols();
	let msg_byte = 1 << self.mlwe_params.poly_len_log2;

        let lwe_params = LWEParams::default();
        let n = 1<<self.params.poly_len_log2;
	let mlwe_dimension = n/msg_byte;
	//println!("n: {}", n);
        let conv = Convolution::new(msg_byte);

        let mut hint_0 = vec![0u64; n * db_cols];

        let convd_len = conv.params().crt_count * msg_byte;//conv.params().poly_len;
	//println!("convd_len: {}", convd_len);

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));

        let mut mlwes = Vec::new();
        for _ in 0..db_rows * msg_byte / n {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }
	    let mut mlwe_vector = Self::rlwe_to_mlwe(&a, self.mlwe_params.poly_len_log2, self.params.poly_len_log2);
	    for poly_idx in 0..n / msg_byte {
		mlwes.extend(&mlwe_vector[poly_idx * msg_byte .. (poly_idx + 1) * msg_byte]); // a to mlwe a
	    }
            //let nega_perm_a = negacyclic_perm_u32(&a);
            //let nega_perm_a_ntt = conv.ntt(&nega_perm_a);
            //v_nega_perm_a.push(nega_perm_a_ntt);
	    //mlwes.push(mlwe_vector);
        }
	let mut mlwe_transposed = Self::mlwe_transpose(&mlwes, mlwe_dimension, db_rows, msg_byte); // db_rows: column length

////////////////////////ignore////////////////
        // limit on the number of times we can add results modulo M before we wrap
        let log2_conv_output =
            log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
        let log2_modulus = log2(conv.params().modulus);
        let log2_max_adds = log2_modulus - log2_conv_output - 1;
        assert!(log2_max_adds > 0);
        let max_adds = 1 << log2_max_adds;
//////////////////////ignore////////////////////
// db_col -> number of columns = elements per rows.
	for i in 0..mlwe_dimension{   
	         
	   for j in 0..db_cols { // each rows
		                
		let mut tmp_col = vec![0u64; convd_len];
                for k in 0..db_rows { // each columns elements
		    let mlwe_idx = (i * db_rows + k)*msg_byte;
		    let mut tmp_mlwe = conv.ntt(&mlwe_transposed[mlwe_idx..mlwe_idx + msg_byte]);             
		    let start_idx = (j * self.db_rows_padded() + k) * msg_byte; // check
                    let pt_col = &self.db()[start_idx..start_idx + msg_byte];
                    let pt_col_u32 = pt_col
                        .iter()
                        .map(|&x| x.to_u64() as u32)
                        .collect::<Vec<_>>();
                    //assert_eq!(pt_col_u32.len(), n);
		    //println!("length: {}", pt_col_u32.len());                
		    let pt_ntt = conv.ntt(&pt_col_u32);
		    //println!("length: {}", pt_ntt.len());
		    //println!("ptcol: {}", pt_col_u32[0]);
///////////////////////////////////from here/////////////////////////////
                    let convolved_ntt = conv.pointwise_mul(&tmp_mlwe, &pt_ntt); // mlwe_num : mlwe number per row
		    //println!("length: {}", convolved_ntt.len());
                    for r in 0..convd_len {
                        tmp_col[r] += convolved_ntt[r] as u64;
                    }
/////////////////
                    if k % max_adds == max_adds - 1 || k == db_rows / n - 1 {
                        let mut col_poly_u32 = vec![0u32; convd_len];
                        for l in 0..conv.params().crt_count {
                            for m in 0..msg_byte {
                                let val = barrett_coeff_u64(
                                    conv.params(),
                                    tmp_col[l * msg_byte + m],
                                    l,
                                );
                                col_poly_u32[l * msg_byte + m] = val as u32;
                            }
                        }
                        let col_poly_raw = conv.raw(&col_poly_u32);
                        for l in 0..msg_byte {
                            hint_0[(i * db_cols + j)*msg_byte + l] += col_poly_raw[i] as u64; // have to check again
                            hint_0[(i * db_cols + j)*msg_byte + l] %= 1u64 << 32;
                        }
                        tmp_col.fill(0);
                    }
                }
            }
	}
        hint_0
    }

    pub fn answer_query(&self, aligned_query_packed: &[u64]) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<1>(aligned_query_packed, 1)
    }

    pub fn answer_batched_queries<const K: usize>(
        &self,
        aligned_queries_packed: &[u64],
    ) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<K>(aligned_queries_packed, 1)
    }

    pub fn perform_offline_precomputation_simplepir(
        &self,
        measurement: Option<&mut Measurement>,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters

        let params = self.params;
        assert!(self.ypir_params.is_simplepir);

        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;

        // Begin offline precomputation

        let now = Instant::now();
        let hint_0: Vec<u64> = self.answer_hint_ring(SEED_0, db_cols);
        // hint_0 is poly_len x db_cols
        let simplepir_prep_time_ms = now.elapsed().as_millis();
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }

        let now = Instant::now();
        let y_constants = generate_y_constants(&params);

        let combined = [&hint_0[..], &vec![0u64; db_cols]].concat();
        assert_eq!(combined.len(), db_cols * (params.poly_len + 1));
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, num_rlwe_outputs);

        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            let tup = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        OfflinePrecomputedValues {
            hint_0,
            hint_1: vec![],
            pseudorandom_query_1: vec![],
            y_constants,
            smaller_server: None,
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        }
    }

    pub fn perform_offline_precomputation(
        &self,
        measurement: Option<&mut Measurement>,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters

        let params = self.params;
        let lwe_params = LWEParams::default();
        assert!(!self.ypir_params.is_simplepir);

        let db_cols = 1 << (params.db_dim_2 + params.poly_len_log2);

        // LWE reduced moduli
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let lwe_q_prime = lwe_params.get_q_prime_2();

        // The number of bits represented by a plaintext RLWE coefficient
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
	println!("pt bits : {}, lwe_q_prim: {}", pt_bits, lwe_q_prime_bits);
        // assert_eq!(pt_bits, 16);

        // The factor by which ciphertext values are bigger than plaintext values
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);
        // assert!(blowup_factor.ceil() - blowup_factor >= 0.05);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        // This is rounded to start on a pt_bits boundary
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;
        let special_bit_offs = special_offs * pt_bits;

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
        let rho = 1 << smaller_params.db_dim_2;
        assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
        assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

        debug!(
            "the first {} LWE output ciphertexts of the DoublePIR round (out of {} total) are query-indepednent",
            special_offs, out_rows
        );
        debug!(
            "the next {} LWE output ciphertexts are query-dependent",
            blowup_factor.ceil() as usize
        );
        debug!("the rest are zero");

        // Begin offline precomputation

        let now = Instant::now();
        let hint_0: Vec<u64> = self.generate_hint_0_ring();
        // hint_0 is n x db_cols
        let simplepir_prep_time_ms = now.elapsed().as_millis();
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }
        debug!("Answered hint (ring) in {} us", now.elapsed().as_micros());

        // compute (most of) the secondary hint
        let intermediate_cts = [&hint_0[..], &vec![0u64; db_cols]].concat();
        let intermediate_cts_rescaled = intermediate_cts
            .iter()
            .map(|x| rescale(*x, lwe_params.modulus, lwe_q_prime))
            .collect::<Vec<_>>();

        // split and do a second PIR over intermediate_cts
        // split into blowup_factor=q/p instances (so that all values are now mod p)
        // the second PIR is over a database of db_cols x (blowup_factor * (lwe_params.n + 1)) values mod p

        // inp: (lwe_params.n + 1, db_cols)
        // out: (out_rows >= (lwe_params.n + 1) * blowup_factor, db_cols)
        //      we are 'stretching' the columns (and padding)

        debug!("Splitting intermediate cts...");

        let smaller_db = split_alloc(
            &intermediate_cts_rescaled,
            special_bit_offs,
            lwe_params.n + 1,
            db_cols,
            out_rows,
            lwe_q_prime_bits,
            pt_bits,
        );
        assert_eq!(smaller_db.len(), db_cols * out_rows);

        debug!("Done splitting intermediate cts.");

        // This is the 'intermediate' db after the first pass of PIR and expansion
        let smaller_server: YServer<u16> = YServer::<u16>::new(
            &self.smaller_params,
            smaller_db.into_iter(),
            false,
            true,
            false,
        );
        debug!("gen'd smaller server.");

        let hint_1 = smaller_server.answer_hint_ring(
            SEED_1,
            1 << (smaller_server.params.db_dim_2 + smaller_server.params.poly_len_log2),
        );
        assert_eq!(hint_1.len(), params.poly_len * out_rows);
        assert_eq!(hint_1[special_offs], 0);
        assert_eq!(hint_1[special_offs + 1], 0);

        let pseudorandom_query_1 = smaller_server.generate_pseudorandom_query(SEED_1);
        let y_constants = generate_y_constants(&params);

        let combined = [&hint_1[..], &vec![0u64; out_rows]].concat();
        assert_eq!(combined.len(), out_rows * (params.poly_len + 1));
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, rho);

        let now = Instant::now();
        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            let tup = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        OfflinePrecomputedValues {
            hint_0,
            hint_1,
            pseudorandom_query_1,
            y_constants,
            smaller_server: Some(smaller_server),
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        }
    }

    /// Perform SimplePIR-style YPIR
    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[u64],
        offline_vals: &OfflinePrecomputedValues<'a>,
        pack_pub_params_row_1s: &[&[PolyMatrixNTT<'a>]],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<u8>> {
        assert!(self.ypir_params.is_simplepir);

        // Set up some parameters

        let params = self.params;

        let y_constants = &offline_vals.y_constants;
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let precomp = &offline_vals.precomp;

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        assert_eq!(first_dim_queries_packed.len(), params.db_rows_padded());

        // Begin online computation

        let first_pass = Instant::now();
        debug!("Performing mul...");
        let mut intermediate = AlignedMemory64::new(db_cols);
        fast_batched_dot_product_avx512::<1, T>(
            &params,
            intermediate.as_mut_slice(),
            first_dim_queries_packed,
            db_rows,
            self.db(),
            db_rows,
            db_cols,
        );
        debug!("Done w mul...");
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
        }

        let ring_packing = Instant::now();
        let num_rlwe_outputs = db_cols / params.poly_len;
        let packed = pack_many_lwes(
            &params,
            &prepacked_lwe,
            &precomp,
            intermediate.as_slice(),
            num_rlwe_outputs,
            &pack_pub_params_row_1s[0],
            &y_constants,
        );
        debug!("Packed...");
        if let Some(m) = measurement {
            m.online.ring_packing_time_ms = ring_packing.elapsed().as_millis() as usize;
        }

        let mut packed_mod_switched = Vec::with_capacity(packed.len());
        for ct in packed.iter() {
            let res = ct.raw();
            let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
            packed_mod_switched.push(res_switched);
        }

        packed_mod_switched
    }

    pub fn perform_online_computation<const K: usize>(
        &self,
        offline_vals: &mut OfflinePrecomputedValues<'a>,
        first_dim_queries_packed: &[u32],
        second_dim_queries: &[(&[u64], &[PolyMatrixNTT<'a>])],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        // Set up some parameters

        let params = self.params;
        let lwe_params = LWEParams::default();

        let db_cols = self.db_cols();

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        // LWE reduced moduli
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let lwe_q_prime = lwe_params.get_q_prime_2();

        // The number of bits represented by a plaintext RLWE coefficient
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
        // assert_eq!(pt_bits, 16);

        // The factor by which ciphertext values are bigger than plaintext values
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);
        // assert!(blowup_factor.ceil() - blowup_factor >= 0.05);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        // This is rounded to start on a pt_bits boundary
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
        let rho = 1 << smaller_params.db_dim_2;
        assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
        assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

        // Load offline precomputed values
        let hint_1_combined = &mut offline_vals.hint_1;
        let pseudorandom_query_1 = &offline_vals.pseudorandom_query_1;
        let y_constants = &offline_vals.y_constants;
        let smaller_server = offline_vals.smaller_server.as_mut().unwrap();
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let fake_pack_pub_params = &offline_vals.fake_pack_pub_params;
        let precomp = &offline_vals.precomp;

        // Begin online computation

        let online_phase = Instant::now();
        let first_pass = Instant::now();
        let intermediate = self.lwe_multiply_batched_with_db_packed::<K>(first_dim_queries_packed);
        let simplepir_resp_bytes = intermediate.len() / K * (lwe_q_prime_bits as usize) / 8;
        debug!("simplepir_resp_bytes {} bytes", simplepir_resp_bytes);
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        debug!("First pass took {} us", first_pass.elapsed().as_micros());

        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
            m.online.simplepir_resp_bytes = simplepir_resp_bytes;
        }

        debug!("intermediate.len(): {}", intermediate.len());
        let mut second_pass_time_ms = 0;
        let mut ring_packing_time_ms = 0;
        let mut responses = Vec::new();
        for (intermediate_chunk, (packed_query_col, pack_pub_params_row_1s)) in intermediate
            .as_slice()
            .chunks(db_cols)
            .zip(second_dim_queries.iter())
        {
            let second_pass = Instant::now();
            let intermediate_cts_rescaled = intermediate_chunk
                .iter()
                .map(|x| rescale(*x as u64, lwe_params.modulus, lwe_q_prime))
                .collect::<Vec<_>>();
            assert_eq!(intermediate_cts_rescaled.len(), db_cols);
            debug!(
                "intermediate_cts_rescaled[0] = {}",
                intermediate_cts_rescaled[0]
            );

            let now = Instant::now();
            // modify the smaller_server db to include the intermediate values
            // let mut smaller_server_clone = smaller_server.clone();
            {
                // remember, this is stored in 'transposed' form
                // so it is out_cols x db_cols
                let smaller_db_mut: &mut [u16] = smaller_server.db_mut();
                for j in 0..db_cols {
                    // new value to write into the db
                    let val = intermediate_cts_rescaled[j];

                    for m in 0..blowup_factor.ceil() as usize {
                        // index in the transposed db
                        let out_idx = (special_offs + m) * db_cols + j;

                        // part of the value to write into the db
                        let val_part = ((val >> (m * pt_bits)) & ((1 << pt_bits) - 1)) as u16;

                        // assert_eq!(smaller_db_mut[out_idx], DoubleType::default());
                        smaller_db_mut[out_idx] = val_part;
                    }
                }
            }
            debug!("load secondary hint {} us", now.elapsed().as_micros());

            let now = Instant::now();
            {
                let blowup_factor_ceil = blowup_factor.ceil() as usize;

                let phase = Instant::now();
                let secondary_hint = smaller_server.multiply_with_db_ring(
                    &pseudorandom_query_1,
                    special_offs..special_offs + blowup_factor_ceil,
                    SEED_1,
                );
                debug!(
                    "multiply_with_db_ring took: {} us",
                    phase.elapsed().as_micros()
                );
                // let phase = Instant::now();
                // let secondary_hint =
                //     smaller_server_clone.answer_hint(SEED_1, special_offs..special_offs + blowup_factor_ceil);
                // debug!(
                //     "traditional answer_hint took: {} us",
                //     phase.elapsed().as_micros()
                // );

                assert_eq!(secondary_hint.len(), params.poly_len * blowup_factor_ceil);

                for i in 0..params.poly_len {
                    for j in 0..blowup_factor_ceil {
                        let inp_idx = i * blowup_factor_ceil + j;
                        let out_idx = i * out_rows + special_offs + j;

                        // assert_eq!(hint_1_combined[out_idx], 0); // we no longer clone for each query, just overwrite
                        hint_1_combined[out_idx] = secondary_hint[inp_idx];
                    }
                }
            }
            debug!("compute secondary hint in {} us", now.elapsed().as_micros());

            assert_eq!(hint_1_combined.len(), params.poly_len * out_rows);

            let response: AlignedMemory64 = smaller_server.answer_query(packed_query_col);

            second_pass_time_ms += second_pass.elapsed().as_millis();
            let ring_packing = Instant::now();
            let now = Instant::now();
            assert_eq!(response.len(), 1 * out_rows);

            // combined is now (poly_len + 1) * (out_rows)
            // let combined = [&hint_1_combined[..], response.as_slice()].concat();
            let mut excess_cts = Vec::with_capacity(blowup_factor.ceil() as usize);
            for j in special_offs..special_offs + blowup_factor.ceil() as usize {
                let mut rlwe_ct = PolyMatrixRaw::zero(&params, 2, 1);

                // 'a' vector
                // put this in negacyclic order
                let mut poly = Vec::new();
                for k in 0..params.poly_len {
                    poly.push(hint_1_combined[k * out_rows + j]);
                }
                let nega = negacyclic_perm(&poly, 0, params.modulus);

                rlwe_ct.get_poly_mut(0, 0).copy_from_slice(&nega);

                // for k in 0..params.poly_len {
                //     rlwe_ct.get_poly_mut(0, 0)[k] = nega[k];
                // }

                // let j_within_last = j % params.poly_len;
                // prepacked_lwe_mut.last_mut().unwrap()[j_within_last] = rlwe_ct.ntt();
                excess_cts.push(rlwe_ct.ntt());
            }
            debug!("in between: {} us", now.elapsed().as_micros());

            // assert_eq!(pack_pub_params_row_1s[0].rows, 1);

            let mut packed = pack_many_lwes(
                &params,
                &prepacked_lwe,
                &precomp,
                response.as_slice(),
                rho,
                &pack_pub_params_row_1s,
                &y_constants,
            );

            let now = Instant::now();
            let mut pack_pub_params = fake_pack_pub_params.clone();
            for i in 0..pack_pub_params.len() {
                let uncondensed = uncondense_matrix(params, &pack_pub_params_row_1s[i]);
                pack_pub_params[i].copy_into(&uncondensed, 1, 0);
            }
            debug!("uncondense pub params: {} us", now.elapsed().as_micros());
            let now = Instant::now();
            let other_packed =
                pack_using_single_with_offset(&params, &pack_pub_params, &excess_cts, special_offs);
            add_into(&mut packed[0], &other_packed);
            debug!(
                "pack_using_single_with_offset: {} us",
                now.elapsed().as_micros()
            );

            let now = Instant::now();
            let mut packed_mod_switched = Vec::with_capacity(packed.len());
            for ct in packed.iter() {
                let res = ct.raw();
                let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
                packed_mod_switched.push(res_switched);
            }
            debug!("switching: {} us", now.elapsed().as_micros());
            // debug!("Preprocessing pack in {} us", now.elapsed().as_micros());
            // debug!("");
            ring_packing_time_ms += ring_packing.elapsed().as_millis();

            // packed is blowup_factor ring ct's
            // these encode, contiguously [poly_len + 1, blowup_factor]
            // (and some padding)
            assert_eq!(packed.len(), rho);

            responses.push(packed_mod_switched);
        }
        debug!(
            "Total online time: {} us",
            online_phase.elapsed().as_micros()
        );
        debug!("");

        if let Some(ref mut m) = measurement {
            m.online.second_pass_time_ms = second_pass_time_ms as usize;
            m.online.ring_packing_time_ms = ring_packing_time_ms as usize;
        }

        responses
    }

    // generic function that returns a u8 or u16:
    pub fn db(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.db_buf_aligned.as_ptr() as *mut T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_u16(&self) -> &[u16] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u16,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u16>(),
            )
        }
    }

    pub fn db_u32(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u32,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u32>(),
            )
        }
    }

    pub fn get_elem(&self, row: usize, col: usize) -> T {
        self.db()[col * self.db_rows_padded() + row] // stored transposed
    }

    pub fn get_row(&self, row: usize) -> Vec<T> {
        let db_cols = self.db_cols();
        let mut res = Vec::with_capacity(db_cols);
        for col in 0..db_cols {
            res.push(self.get_elem(row, col));
        }
        res
        // // convert to u8 contiguously
        // let mut res_u8 = Vec::with_capacity(db_cols * std::mem::size_of::<T>());
        // for &x in res.iter() {
        //     res_u8.extend_from_slice(&x.to_u64().to_le_bytes()[..std::mem::size_of::<T>()]);
        // }
        // res_u8
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use spiral_rs::{client::Client, number_theory::invert_uint_mod, util::get_test_params, arith::*, gadget::*, ntt::*, params::*, poly::*, aligned_memory::AlignedMemory64};
    //use super::server::ToM512;
    use crate::{
        client::*, params::params_for_scenario,
        server::*,
    };

    #[test]
    fn test_mlwe_a() {
	let mut params = get_test_params();
	let mut mlwe_params = params.clone();
	let mlwe_bit = 5;
        let mut client = Client::init(&params);
        client.generate_secret_keys();
	params.poly_len_log2 = 11;
	params.poly_len = 1<<params.poly_len_log2;
	mlwe_params.poly_len_log2 = mlwe_bit;
	mlwe_params.poly_len = 1<<mlwe_bit;

	let mut poly = PolyMatrixRaw::zero(&params, 1, 1);
	for i in 0..params.poly_len{
	    poly.data[i] = i as u64;
	}

	let mut mlwe = rlwe_to_mlwe_a(&params, &poly.as_slice().to_vec(), mlwe_bit);

	let mut mlwe_result = PolyMatrixRaw::zero(&mlwe_params, params.poly_len / mlwe_params.poly_len, 1);
	mlwe_result.as_mut_slice().copy_from_slice(&mlwe[0..params.poly_len]);
	println!("{:?}", mlwe_result.as_slice());

	let mut output_poly = PolyMatrixRaw::zero(&params, 1, 1);

	mlwe_to_rlwe_a(&params, &mut output_poly, mlwe_result.as_slice().to_vec(), mlwe_bit);

	println!("new rlwe: {:?}", output_poly.as_slice());
	println!("modulus : {}", params.modulus);
    }

    #[test]
    fn test_mlwe_b() {
	let mut params = get_test_params();
	let mut mlwe_params = params.clone();
	let mlwe_bit = 3;
        let mut client = Client::init(&params);
        client.generate_secret_keys();
	params.poly_len_log2 = 6;
	params.poly_len = 1<<params.poly_len_log2;
	mlwe_params.poly_len_log2 = mlwe_bit;
	mlwe_params.poly_len = 1<<mlwe_bit;
	let dimension = params.poly_len / mlwe_params.poly_len;

	let mut poly = PolyMatrixRaw::zero(&params, 1, 1);
	for i in 0..params.poly_len{
	    poly.data[i] = (i) as u64;
	}

	//poly.data[4] = 1 as u64;

	println!("orig : {:?}", poly.as_slice());

	let mut mlwe = rlwe_to_mlwe_b(&params, &poly.as_slice().to_vec(), mlwe_bit);
	let mut mlwe_result = PolyMatrixRaw::zero(&mlwe_params, params.poly_len / mlwe_params.poly_len, 1);
	mlwe_result.as_mut_slice().copy_from_slice(&mlwe);

	println!("mlwe : {:?}", mlwe_result.as_slice());
	
	let mut output_poly = PolyMatrixRaw::zero(&params, 1, 1);

	let mut slice = mlwe_result.as_mut_slice().to_vec();
	println!("{:?}", slice);

	let start = Instant::now();
	mlwe_to_rlwe_b_packed(&params, &mut slice, mlwe_bit);
	mlwe_to_rlwe_b(&params, &mut output_poly, mlwe_result.get_poly(0, 0), mlwe_bit, 0);
	let end = Instant::now();
	println!("time: {:?}", end - start);

	println!("{:?}", slice);

	println!("{:?}", output_poly.as_slice());
	

    }

    #[test]
    fn test_matrix_mult(){
	let mut params = params_for_scenario(1<<30, 1);
	params.db_dim_1 = 0;
	params.db_dim_2 = 0;
	
	println!("{}, {}", params.db_dim_1, params.db_dim_2);
	let db_rows: usize = 1<<(params.db_dim_1 + params.poly_len_log2);
	let db_cols: usize = 1<<(params.db_dim_2 + params.poly_len_log2);
	let mut rng = rand::thread_rng();
	
	let mut matrix = Vec::with_capacity(db_rows * db_cols);

	for i in 0..db_rows{
	    for j in 0..db_cols{
		matrix.push(1 as u16);
	    }
	}	
	//for i in 0..(db_rows * db_cols){
	//    matrix.push(rng.gen::<u16>());
	//}
	
	let server: YServer<u16> = YServer::<u16>::new(
	    &params,
	    matrix.into_iter(),
	    false,
	    false,
	    false
	);
	//println!("{:?}", server.db());


	let mut rng = rand::thread_rng();
	let mut buf = AlignedMemory64::new(db_rows);

	//assert_eq!(buf.len(), server.db_rows_padded());

	let mut index = 0;
	for val in buf.as_mut_slice(){
	    if (index == 2) {
		*val = 1;
	    }
	    else{
	        *val = 0;
	    }
	    index = index + 1;

	    //*val = rng.gen::<u64>() & 0x00FF_FFFF_FFFF_FFFF;
	}

	//println!("{:?}", buf.as_slice());

	

	let response: AlignedMemory64 = server.answer_query(buf.as_slice());
	
    }

    #[test]
    fn test_encryption(){
	
	let params = params_for_scenario(1 << 30, 1);
	let mut client = Client::init(&params);
	client.generate_secret_keys(); //generate secret key
	

	let t_exp = params.t_exp_left;
	let pack_seed = [1u8; 32];
	
	let mut rng_pub = ChaCha20Rng::from_seed(get_seed(1));

	let scale_k = params.modulus / params.pt_modulus; //scaling factor

	let t = 3;
	let mut query = PolyMatrixRaw::zero(&params, 1, 1);
	query.data[3] = scale_k; // query index

	let factor = invert_uint_mod(params.poly_len as u64, params.modulus).unwrap();

	let ct_0 = client.encrypt_matrix_scaled_reg(&query.ntt(), &mut ChaCha20Rng::from_entropy(), &mut rng_pub, factor); //

	let dec_result = client.decrypt_matrix_reg(&ct_0);
	let dec_result_raw = dec_result.raw();
	let mut dec_rescaled = PolyMatrixRaw::zero(&params, 1, 1);
	for z in 0..dec_rescaled.data.len() {
	    dec_rescaled.data[z] = rescale(dec_result_raw.data[z], params.modulus, params.pt_modulus);
	}
	println!("auto result c1 : {:?}", dec_rescaled.as_slice());
    }

    #[test]
    fn test_query_mult(){

	let mut rng_pub = ChaCha20Rng::from_seed(get_seed(1));
	let mut params = params_for_scenario(1<<30, 1);
	params.pt_modulus = 1<<16;

	let mut client = Client::init(&params);
	//let y_client = YClient::new(&mut client, &params);
	client.generate_secret_keys();
	
	let scale_k = params.modulus / params.pt_modulus;

	println!("instance: {}", params.instances);
	println!("dimension: {}, {}", params.db_dim_1, params.db_dim_2);
	let db_rows: usize = 1<<(params.db_dim_1 + params.poly_len_log2);
	let db_cols: usize = 1<<(params.db_dim_2 + params.poly_len_log2);
	println!("rows: {}, cols: {}", db_rows, db_cols);
	let mut rng = rand::thread_rng();
	
	let mut matrix = Vec::with_capacity(db_rows * db_cols);

	for i in 0..db_rows{
	    for j in 0..db_cols{
		matrix.push((10*i+j) as u16);
	    }
	}	

	
	let server: YServer<u16> = YServer::<u16>::new( // database
	    &params,
	    matrix.into_iter(),
	    false,
	    true,
	    false
	);

	let mut y_client = YClient::new(&mut client, &params);

	let preprocessed_query = server.generate_pseudorandom_query(SEED_1);
	let pseudo = y_client.generate_query_impl(SEED_1, params.db_dim_1, true, 3);
	let pseudo_2 = y_client.generate_query_impl(SEED_1, params.db_dim_1, true, 3);


	let hint = server.answer_hint_ring(SEED_1, db_cols);//db_cols); // hint -> a part
	
	let packed_query_col = {
	    
	    let query_col = y_client.generate_query(SEED_1, params.db_dim_1, true, 15); // query b -> b part

	    let query_col_last_row = &query_col[params.poly_len * db_rows..];
	    let packed_query_col_tmp = pack_query(&params, query_col_last_row); // query b
	    
	    packed_query_col_tmp
	
	};

	let response: AlignedMemory64 = server.answer_query(packed_query_col.as_slice());//buf.as_slice());

	let mut ct_vec = Vec::new();
	for i in 0..response.len(){
	    let mut ct_result = PolyMatrixRaw::zero(&params, 2, 1);
	    let mut ct_a_result = PolyMatrixRaw::zero(&params, 1, 1);

	    let mut poly = Vec::new();
	    for j in 0..params.poly_len{
		poly.push(hint.as_slice()[j*response.len() + i]);//j*response.len() + i]);
	    }
	    let nega = negacyclic_perm(&poly, 0, params.modulus);

	    for j in 0..params.poly_len {
		ct_result.get_poly_mut(0, 0)[j] = nega[j];
	    }

	    ct_result.get_poly_mut(1, 0)[0] = response[i] as u64;

	    ct_vec.push(ct_result);
	}
	
	for i in 0..db_cols{
	    let decrypted = decrypt_ct_reg_measured(&mut y_client.inner, &params, &ct_vec[i].ntt(), 0);
	    //assert_eq!(decrypted.data[0], server.db()[i*db_rows + 15] as u64); // this is correct one
	    assert_eq!(decrypted.data[0], server.db()[i*db_rows + 15] as u64);
	}
	println!("{}", ct_vec.len());

	
    }

    #[test]
    fn test_packing_lwe_to_mlwe(){

	let mut rng_pub = ChaCha20Rng::from_seed(get_seed(1));
	let mut params = params_for_scenario(1<<30, 1);
	params.pt_modulus = 1<<16;

	let mut simple_params = params.clone();
	simple_params.modulus = 1<<30;
	simple_params.modulus_log2 = 30;

	let mut mlwe_params = params.clone(); // create mlwe parameter
	mlwe_params.poly_len_log2 = 7;
	mlwe_params.poly_len = 1<<mlwe_params.poly_len_log2;
	let dimension = params.poly_len / mlwe_params.poly_len; // dimension

	let mut client = Client::init(&params);
	
	client.generate_secret_keys();

	let db_rows: usize = 1<<(params.db_dim_1 + params.poly_len_log2); // db_row 
	let db_cols: usize = 1<<(params.db_dim_2 + params.poly_len_log2); // db_col
	println!("rows: {}, cols: {}", db_rows, db_cols);
	let mut rng = rand::thread_rng();
	
	let mut matrix = Vec::with_capacity(db_rows * db_cols); // database matrix

	for i in 0..db_rows{
	    for j in 0..db_cols{
		matrix.push((10*i+j) as u16);
	    }
	}	

	
	let server: YServer<u16> = YServer::<u16>::new( // database
	    &params,
	    matrix.into_iter(),
	    false,
	    true,
	    false
	);

	let mut y_client = YClient::new(&mut client, &params);

	let hint = server.answer_hint_ring(SEED_1, db_cols); // hint -> a part
	
	let packed_query_col = { // create query

	    let query_col = y_client.generate_query_mlwe(SEED_1, params.db_dim_1, mlwe_params.poly_len_log2, true, 15); // query b -> b part
	    let query_col_last_row = &query_col[params.poly_len * db_rows..];
	    let packed_query_col_tmp = pack_query(&params, query_col_last_row); // query b
	    
	    packed_query_col_tmp
	
	};

	let response: AlignedMemory64 = server.answer_query(packed_query_col.as_slice());//buf.as_slice()); // simple pir response

	let mut ct_vec_a = Vec::new();
	let mut ct_vec_b = Vec::new();
	for i in 0..response.len(){
	    let mut ct_result = PolyMatrixRaw::zero(&params, 2, 1);
	    let mut ct_a_result = PolyMatrixRaw::zero(&mlwe_params, 1, dimension);
	    //println!("ct_a length: {}", ct_a_result.as_slice().len());
	    let mut ct_b_result = PolyMatrixRaw::zero(&mlwe_params, 1, 1);

	    let mut poly = Vec::new();
	    for j in 0..params.poly_len{
		poly.push(hint.as_slice()[j*response.len() + i]);
	    }
	    let nega = negacyclic_perm(&poly, 0, params.modulus);

	    let mlwe_a = rlwe_to_mlwe_a(&params, &nega, mlwe_params.poly_len_log2);

	    for j in 0..params.poly_len {
		ct_a_result.data[j] = mlwe_a[j];
	    }

	    ct_b_result.data[0] = response[i] as u64;

	    ct_vec_a.push(ct_a_result.ntt());
	    ct_vec_b.push(ct_b_result);
	}

	//mlwe expansion

	let t_exp = 3;
	let auto_table = generate_automorph_tables_brute_force(&mlwe_params);
	let pack_seed = [1u8; 32];

	let (expansion_key_a, expansion_key_b) = generate_query_expansion_key(&params, &mlwe_params, t_exp, &mut ChaCha20Rng::from_entropy(), &mut ChaCha20Rng::from_seed(pack_seed), &mut y_client.inner);

	let ct_a_128 = ct_vec_a[0..mlwe_params.poly_len].to_vec();
	let mut ct_b_128 = Vec::new();
	for i in 0..mlwe_params.poly_len{
	    ct_b_128.push(ct_vec_b[i].data[0] as u64);
	}
	
	let expansion_table_pos = create_packing_table_mlwe_pos(&mlwe_params);
	let expansion_table_neg = create_packing_table_mlwe_neg(&mlwe_params);

	let (result_a, decomp_a_vec) = prep_pack_lwe_to_mlwe(&params, &mlwe_params, dimension, t_exp, &expansion_key_a, &ct_a_128, &auto_table, &expansion_table_neg, &expansion_table_pos);

	let result_b = pack_lwes_to_mlwe(&params, &mlwe_params, dimension, t_exp, &ct_b_128, &expansion_key_b, &auto_table, &expansion_table_neg, &expansion_table_pos, &decomp_a_vec);

	let mut dec = decrypt_mlwe(&params, &mlwe_params, &result_a, &result_b.ntt(), &y_client.inner);
	println!("result: {:?}", dec.as_slice());

/////////////////////multiple////////////////////////	

	let (result_a_tmps, decomps) = prep_pack_lwes_to_mlwe_db(&params, &mlwe_params, &hint, dimension, t_exp, &expansion_key_a,  &auto_table, &expansion_table_neg, &expansion_table_pos);

	let result_b_tmps = pack_lwes_to_mlwe_db(&params, &mlwe_params, &response, dimension, t_exp, &expansion_key_b, &auto_table, &expansion_table_neg, &expansion_table_pos, decomps);

	let decrypted = decrypt_mlwe_batch(&params, &mlwe_params, dimension, &result_a_tmps, &result_b_tmps.ntt(), &y_client.inner);

	for i in 0..decrypted.len() {
	    println!("{:?}", decrypted[i].as_slice());
	}
	
    }

    #[test]
    fn test_whole_protocol(){ // start : row: 16384 / second : col:8192

	////////first settings
	let mut params = params_for_scenario(1<<30, 1);
	params.pt_modulus = 1<<16;

	println!("{}, {}", params.get_q_prime_1(), params.get_q_prime_2());	
	let rlwe_q_prime_1 = params.get_q_prime_1();
	let rlwe_q_prime_2 = params.get_q_prime_2();

	//256MB: 3, 2 //1GB: 4, 3 //8GB: 5, 5
	params.db_dim_1 = 5;
	params.db_dim_2 = 5;

	let mut mlwe_params = params.clone();
	mlwe_params.poly_len_log2 = 7;
	mlwe_params.poly_len = 1<<mlwe_params.poly_len_log2;
	
	let mut simple_params = mlwe_params.clone();
	simple_params.modulus = 1<<30;
	simple_params.modulus_log2 = 30;
	
	let dimension = params.poly_len / mlwe_params.poly_len;

	let mut client = Client::init(&params);
	client.generate_secret_keys();

	let db_rows: usize = 1<<(params.db_dim_1 + params.poly_len_log2); // db_row : first query length
	let db_cols: usize = 1<<(params.db_dim_2 + params.poly_len_log2); // db_col : second query length
	println!("rows: {}, cols: {}", db_rows, db_cols);
	let mut rng = rand::thread_rng();
	
	let mut matrix = Vec::with_capacity(db_rows * db_cols); // database matrix

	for i in 0..db_rows{
	    for j in 0..db_cols{
		let value: u16 = rng.gen(); // 0 ~ 65535 중 임의의 값
        	matrix.push(value);
	    }
	}	

	let server: YServer<u16> = YServer::<u16>::new( // database
	    &params,
	    matrix.into_iter(),
	    false,
	    true,
	    false
	);

	let mut y_client = YClient::new(&mut client, &params);

	let g_exp = build_gadget(&simple_params, 1, 2); // for gadget decomposition
	
	let mut g_exp_56 = PolyMatrixRaw::zero(&mlwe_params, 1, 2);
	g_exp_56.as_mut_slice().copy_from_slice(&g_exp.as_slice());
	let g_exp_56_ntt = g_exp_56.ntt();

	let mut g_inv_a = PolyMatrixRaw::zero(&simple_params, dimension * 2, db_cols / mlwe_params.poly_len);
	let mut g_inv_a_56 = PolyMatrixRaw::zero(&mlwe_params, dimension * 2, db_cols / mlwe_params.poly_len);
	let mut g_inv_b = PolyMatrixRaw::zero(&simple_params, 2, db_cols / mlwe_params.poly_len);
	let mut g_inv_b_56 = PolyMatrixRaw::zero(&mlwe_params, 2, db_cols / mlwe_params.poly_len);

	////////////////////////////for keyswitching///////////////////
	let t_exp = params.t_exp_left;
	let auto_table = generate_automorph_tables_brute_force(&mlwe_params); //automorphism table for mlwe
	let pack_seed = [1u8; 32];

	let (expansion_key_a, expansion_key_b) = generate_query_expansion_key(&params, &mlwe_params, t_exp, &mut ChaCha20Rng::from_entropy(), &mut ChaCha20Rng::from_seed(pack_seed), &mut y_client.inner); // key switching keys for lwe to mlwe

	let expansion_table_pos = create_packing_table_mlwe_pos(&mlwe_params); // expansion table for lwe to mlwe
	let expansion_table_neg = create_packing_table_mlwe_neg(&mlwe_params);

	let y_constants = generate_y_constants(&params);
	
	let pack_pub_params = raw_generate_expansion_params( // mlwe to rlwe compression key
            &params,
            y_client.inner.get_sk_reg(),
            params.poly_len_log2,
            params.t_exp_left,
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_seed(pack_seed),
        );

	let mut fake_pack_pub_params = pack_pub_params.clone();
        // zero out all of the second rows
        for i in 0..pack_pub_params.len() {
            for col in 0..pack_pub_params[i].cols {
                fake_pack_pub_params[i].get_poly_mut(1, col).fill(0);
            }
        }

        let mut pack_pub_params_row_1s = pack_pub_params.clone();
        for i in 0..pack_pub_params.len() {
            pack_pub_params_row_1s[i] =
                pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
            pack_pub_params_row_1s[i] = condense_matrix(&params, &pack_pub_params_row_1s[i]);
        }


	///////////////////////////////////////////////////////////////
	///////////////////////hint preprocessing//////////////////////
	///////////////////////////////////////////////////////////////


	let hint = server.answer_hint_ring(SEED_1, db_cols);

	let (response_a, decomp_a) = prep_pack_lwes_to_mlwe_db(&params, &mlwe_params, &hint, dimension, t_exp, &expansion_key_a, &auto_table, &expansion_table_neg, &expansion_table_pos);

	let response_a_simple_raw = response_a.raw();
	let mut response_a_simple_raw_transposed = PolyMatrixRaw::zero(&mlwe_params, response_a_simple_raw.get_cols(), response_a_simple_raw.get_rows());

	let transposed_array = transpose_poly(&mlwe_params, &response_a_simple_raw);
	response_a_simple_raw_transposed.as_mut_slice().copy_from_slice(transposed_array.as_slice());

	let mut response_a_simple_raw_rescaled = PolyMatrixRaw::zero(&simple_params, response_a_simple_raw.get_cols(), response_a_simple_raw.get_rows());
	
	for i in 0..response_a_simple_raw_transposed.as_slice().len() {
	    response_a_simple_raw_rescaled.data[i] = rescale(response_a_simple_raw_transposed.data[i], params.modulus, rlwe_q_prime_2);
	}

	gadget_invert_rdim(&mut g_inv_a, &response_a_simple_raw_rescaled, dimension);

	g_inv_a_56.as_mut_slice().copy_from_slice(&g_inv_a.as_slice());

	let g_inv_a_ntt = g_inv_a_56.ntt(); // hint 0

	let mut hint_0s = Vec::new();
	for i in 0..2{
	    hint_0s.push(g_inv_a_ntt.submatrix(i*dimension, 0, dimension, db_cols / mlwe_params.poly_len)); //hint0 decomposed
	}

	let double_query_a = server.generate_double_hint_mlwe(&mlwe_params, SEED_0, dimension, params.db_dim_2, mlwe_params.poly_len_log2, 2, &y_client.inner); // double query a

	let mut hint_final_0 = Vec::new(); // hint only made of a parts
	for i in 0..2{
	    let hint_final_element = &hint_0s[i] * &double_query_a;
	    hint_final_0.push(hint_final_element);
	}

	////////////////packing mlwes to rlwe A//////////////////////////

	let mut packed_a_vec = Vec::new();
	for i in 0..2{
	    let mut packed_a = Vec::new();
	    for j in 0..dimension{
		let mut input_poly = PolyMatrixRaw::zero(&params, 2, 1);
		let mut input_poly_tmp = PolyMatrixRaw::zero(&params, 1, 1);

		mlwe_to_rlwe_a(&params, &mut input_poly_tmp, hint_final_0[i].submatrix(j, 0, 1, dimension).raw().as_slice().to_vec(), mlwe_params.poly_len_log2);
		input_poly.as_mut_slice()[0..params.poly_len].copy_from_slice(&input_poly_tmp.as_slice());
		packed_a.push(input_poly.ntt());
	    }

	    let (precomp_res, precomp_vals, precomp_tables) = precompute_pack_mlwe_to_rlwe(
	        &params,
	        params.poly_len_log2,
	        mlwe_params.poly_len_log2,
	        &packed_a,
	        &fake_pack_pub_params,
	        &y_constants,
	    );
	 
	    packed_a_vec.push((precomp_res, precomp_vals, precomp_tables))
	}

	/////////////////////////preprocessing finished////////////////////////////
	

	///////////////////////////////////////////////////////////////
	//////////////////////query////////////////////////////////////

////////////////////////////////client creates query////////////////////////////
	let val_row: u32 = rng.gen_range(0..(db_rows as u32));
	let val_col: u16 = rng.gen_range(0..((db_cols / mlwe_params.poly_len) as u16));
	let target_row = val_row as usize;
	let target_col = val_col as usize;	

	// simple query
	let query_col = y_client.generate_query_mlwe(SEED_1, params.db_dim_1, mlwe_params.poly_len_log2, true, target_row); // query b -> b part
	let query_col_last_row = &query_col[params.poly_len * db_rows..];
	let packed_query_col = pack_query(&params, query_col_last_row); // query b // row: 16384

	//double query
	let double_query_b = y_client.generate_query_double_mlwe(&mlwe_params, SEED_0, dimension, params.db_dim_2,  mlwe_params.poly_len_log2, target_col); // col: 8192

////////////////////////////////things to create before response////////////////////////////////
	let mut response_b_simple_transposed = PolyMatrixRaw::zero(&mlwe_params, 1, db_cols / mlwe_params.poly_len);
	let mut response_b_simple_rescaled = PolyMatrixRaw::zero(&simple_params, 1, db_cols / mlwe_params.poly_len);

	let mut response_0s = Vec::new(); // decomped response lwe
	let mut response_0_times_hint_double_vec = Vec::new(); // b1 times A2
	let mut response_mult_vec = Vec::new(); // response times response // b1 time b2
	let mut hint_0_times_response_1_vec = Vec::new(); // hint only made of a parts // A1 time b2
	let mut packed_b = Vec::new();
	let mut last_pack_a_vec = Vec::new(); // final pack input a 
	let mut last_b_values = Vec::new();
	let mut tmp_last_rlwe_poly = PolyMatrixRaw::zero(&params, 1, 1);

///////////////////////////////response starts/////////////////////////////

	let start = Instant::now();

	//simple response
	let response: AlignedMemory64 = server.answer_query(packed_query_col.as_slice()); 

	let start_0 = Instant::now();

	//pack simple response into mlwe
	let response_b_simple = pack_lwes_to_mlwe_db(&params, &mlwe_params, &response, dimension, t_exp, &expansion_key_b, &auto_table, &expansion_table_neg, &expansion_table_pos, decomp_a);

	let transposed_array_b = transpose_poly(&mlwe_params, &response_b_simple);
	response_b_simple_transposed.as_mut_slice().copy_from_slice(transposed_array_b.as_slice());

	let res_start = Instant::now();
	for i in 0..response_b_simple_rescaled.as_slice().len() {
	    response_b_simple_rescaled.data[i] = rescale(response_b_simple_transposed.data[i], params.modulus, rlwe_q_prime_1);
	}
	let res_end = Instant::now();
	println!("{:?}", res_end - res_start);

	gadget_invert_rdim(&mut g_inv_b, &response_b_simple_rescaled, 1); // db * b1 ->gadget : g_invb

	g_inv_b_56.as_mut_slice().copy_from_slice(&g_inv_b.as_slice());

	let g_inv_b_ntt = g_inv_b_56.ntt(); // response 0

	//let mut response_0s = Vec::new();
	
	for i in 0..2{
	    response_0s.push(g_inv_b_ntt.submatrix(i, 0, 1, db_cols / mlwe_params.poly_len));  // db * b1 gadget
	}
	
	let start_1 = Instant::now();

	//let mut response_0_times_hint_double_vec = Vec::new(); // b1 times A2
	for i in 0..2{
	    let response_0_times_hint_double = &response_0s[i] * &double_query_a; // db* b1 gadget / times / A2
	    response_0_times_hint_double_vec.push(response_0_times_hint_double);
	}

	//let mut response_mult_vec = Vec::new(); // response times response // b1 time b2
	for i in 0..2{
	    let response_mult = &response_0s[i] * &double_query_b; // db* b1 gadget / times / b2
	    response_mult_vec.push(response_mult);
	}

	//let mut hint_0_times_response_1_vec = Vec::new(); // hint only made of a parts // A1 time b2
	for i in 0..2{
	    let hint_0_times_response_1 = &hint_0s[i] * &double_query_b;// db*A1 gadget / times / b2
	    hint_0_times_response_1_vec.push(hint_0_times_response_1);
	}

	////////////////////mlwe to rlwe packing//////////////////////////
	let packing_start = Instant::now();
	//let mut packed_b = Vec::new();//mlwe to rlwe
	for num in 0..2{
	    let mut b_values = Vec::new();
	    let mut tmp_rlwe_poly = PolyMatrixRaw::zero(&params, 1, 1);
	    for i in 0..dimension {
	        let mlwe_tmp = hint_0_times_response_1_vec[num].submatrix(i, 0, 1, 1);
		let mlwe_tmp_raw = mlwe_tmp.raw();
		
		for j in 0..mlwe_params.poly_len{
		    tmp_rlwe_poly.as_mut_slice()[j*dimension + i] = mlwe_tmp_raw.as_slice()[j];
	        }
	    }
	    for i in 0..params.poly_len {
		b_values.push(tmp_rlwe_poly.as_slice()[i]);
	    }

	    let (tmp_precomp_res, tmp_precomp_vals, tmp_precomp_tables) = &packed_a_vec[num];

	    let packed = pack_using_precomp_vals_mlwe_to_rlwe(
	        &params,
	        params.poly_len_log2,
	        mlwe_params.poly_len_log2,
	        &pack_pub_params_row_1s,
	        &b_values,
	        &tmp_precomp_res,
	        &tmp_precomp_vals,
	        &tmp_precomp_tables,
	        &y_constants,
	    );
	    
	    packed_b.push(packed);
	}

///////////////////////////////////////////////////

	//let mut last_pack_a_vec = Vec::new(); // final pack input a 
	for i in 0..2{
	    let mut input_poly = PolyMatrixRaw::zero(&params, 2, 1);
	    let mut input_poly_tmp = PolyMatrixRaw::zero(&params, 1, 1);
	    
	    mlwe_to_rlwe_a(&params, &mut input_poly_tmp, response_0_times_hint_double_vec[i].raw().as_slice().to_vec(), mlwe_params.poly_len_log2);
	    input_poly.as_mut_slice()[0..params.poly_len].copy_from_slice(&input_poly_tmp.as_slice());
	    last_pack_a_vec.push(input_poly.ntt());
	}

	for j in 2..dimension{
	    let mut input_poly = PolyMatrixRaw::zero(&params, 2, 1);
	    last_pack_a_vec.push(input_poly.ntt());
	}

        let (last_precomp_res, last_precomp_vals, last_precomp_tables) = precompute_pack_mlwe_to_rlwe(
	    &params,
	    params.poly_len_log2,
	    mlwe_params.poly_len_log2,
	    &last_pack_a_vec,
	    &fake_pack_pub_params,
	    &y_constants,
	);

///////////////////////////////////////////////////////////////////

	//let mut last_b_values = Vec::new();
	//let mut tmp_last_rlwe_poly = PolyMatrixRaw::zero(&params, 1, 1);
	for num in 0..2{
	    let mut last_mlwe_tmp = response_mult_vec[num].submatrix(0, 0, 1, 1);
	    let last_mlwe_tmp_raw = last_mlwe_tmp.raw();
	    for j in 0..mlwe_params.poly_len{
		tmp_last_rlwe_poly.as_mut_slice()[j*dimension + num] = last_mlwe_tmp_raw.as_slice()[j];
	    }
	}

	for i in 0..params.poly_len{
	    last_b_values.push(tmp_last_rlwe_poly.as_slice()[i]);
	}

	let last_packed = pack_using_precomp_vals_mlwe_to_rlwe(
	    &params,
	    params.poly_len_log2,
	    mlwe_params.poly_len_log2,
	    &pack_pub_params_row_1s,
	    &last_b_values,
	    &last_precomp_res,
	    &last_precomp_vals,
	    &last_precomp_tables,
	    &y_constants,
	);

	let end = Instant::now();

	//modulus switch

	let mut packed_mod_switched_a = Vec::with_capacity(packed_b.len());
	for ct in packed_b.iter() {
	    let res = ct.raw();
	    let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
	    packed_mod_switched_a.push(res_switched);
	}
	
	let res_b = last_packed.raw();
	let res_switched_b = res_b.switch(rlwe_q_prime_1, rlwe_q_prime_2);
	
	//println!("hellooo");

	//////////////////final debug//////////////////
	//packed_b : a parts

	let mut double_response_a = Vec::new();
	for ct_bytes in packed_mod_switched_a.iter() {
	    let ct = PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
	    double_response_a.push(ct);
	}

	let mut decomposed_a = PolyMatrixRaw::zero(&mlwe_params, 2, dimension);

	for i in 0..2{
	    let decrypt_a_parts = decrypt_ct_reg_measured(&y_client.inner, &params, &double_response_a[i].ntt(), 0);
	    let mut poly_temp = PolyMatrixRaw::zero(&mlwe_params, 1, dimension);
	    for j in 0..dimension{
		for k in 0..mlwe_params.poly_len{
		    poly_temp.data[mlwe_params.poly_len*j+k] = decrypt_a_parts.as_slice()[k*dimension+j];
		}
	    }
	    for j in 0..params.poly_len{
		//println!("{}", j);
	        assert_eq!(poly_temp.data[j], hint_0s[i].raw().submatrix(0, target_col, dimension, 1).data[j]);
	    }
	    decomposed_a.as_mut_slice()[i*params.poly_len..(i+1)*params.poly_len].copy_from_slice(&poly_temp.as_slice());
	}

	let composed_a = &g_exp_56_ntt * &decomposed_a.ntt(); // decrypted a

	let composed_a_raw = composed_a.raw();

	let mut composed_a_recovered = PolyMatrixRaw::zero(&mlwe_params, 1, dimension);

	for i in 0..composed_a_recovered.as_slice().len() {
	    let val = composed_a_raw.data[i];
	    composed_a_recovered.data[i] = rescale(val, rlwe_q_prime_2, params.modulus);
	}

	for i in 0..params.poly_len{
	    assert_eq!(response_a_simple_raw_rescaled.submatrix(0, target_col, dimension, 1).data[i], composed_a.raw().data[i]);
	}

	//last_packed

	
	let double_response_b = PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, &res_switched_b);

	let decrypt_b_part = decrypt_ct_reg_measured(&y_client.inner, &params, &double_response_b.ntt(), 0);
	let mut poly_temp_b = PolyMatrixRaw::zero(&mlwe_params, 2, 1);
	for j in 0..2{
	    for k in 0..mlwe_params.poly_len{
		poly_temp_b.data[mlwe_params.poly_len*j+k] = decrypt_b_part.as_slice()[k*dimension+j];
	    }
	    for k in 0..mlwe_params.poly_len {
		//assert_eq!(poly_temp_b.as_slice()[j*mlwe_params.poly_len..(j+1)*mlwe_params.poly_len][k], response_0s[j].submatrix(0, target_col, 1, 1).raw().data[k]);
	    }
	}

	let composed_b = &g_exp_56_ntt * &poly_temp_b.ntt(); //decrypted b

	let composed_b_raw = composed_b.raw();

	let mut composed_b_recovered = PolyMatrixRaw::zero(&mlwe_params, 1, 1);

	for i in 0..composed_b_recovered.as_slice().len() {
	    let val = composed_b_raw.data[i];
	    composed_b_recovered.data[i] = rescale(val, rlwe_q_prime_1, params.modulus);
	}

	for i in 0..mlwe_params.poly_len{
	    assert_eq!(composed_b.raw().data[i], response_b_simple_rescaled.submatrix(0, target_col, 1, 1).data[i]);
	}

	let decrypted_answer = decrypt_mlwe_batch(&params, &mlwe_params, dimension, &composed_a_recovered.ntt(), &composed_b_recovered.ntt(), &y_client.inner);

	for i in 0..mlwe_params.poly_len{
	    //println!("{}, {}", server.db()[(target_col*mlwe_params.poly_len+i)*db_rows + target_row], decrypted_answer[0].data[i]);
	    assert_eq!(server.db()[(target_col*mlwe_params.poly_len+i)*db_rows + target_row] as u64, decrypted_answer[0].data[i]);// +i = +10
	}

	println!("simple time: {:?}", start_0 - start);

	println!("first packing time: {:?}", start_1 - start_0);
	
	println!("double time : {:?}", packing_start - start_1); 

	println!("final packing time: {:?}", end - packing_start);
	
	println!("total time: {:?}", end - start);


	//let decrypted = decrypt_mlwe_batch(&params, &mlwe_params, dimension, &double_query_a, &double_query_b, &y_client.inner);

	//println!("decrypted_len: {}", decrypted.len());

	//for i in 0..decrypted.len() {
	//    println!("{:?}", decrypted[i].as_slice());
	//}

	/////////////////////////debug/////////////////////////////////
//	let mut decrypted_vec = Vec::new();
//	for i in 0..4{	
//	    let decrypted = decrypt_mlwe_batch(&params, &mlwe_params, dimension, &hint_final_0[i], &hint_0_times_response_1_vec[i], &y_client.inner);
//	    decrypted_vec.push(decrypted);
//	}

//	for i in 0..decrypted_vec[0].len() {
//	    for j in 0..mlwe_params.poly_len {
		//assert_eq!(decrypted_vec[1][i].as_slice()[j], hint_0s[1].raw().submatrix(i, target_col, 1, 1).data[j]); 
//	    }
//	}

//	let mut poly_decomposed = PolyMatrixRaw::zero(&mlwe_params, 4, dimension);

//	for i in 0..4{
//	    for j in 0..dimension{
//	    	poly_decomposed.as_mut_slice()[i*params.poly_len + j*mlwe_params.poly_len..i*params.poly_len + (j+1)*mlwe_params.poly_len].copy_from_slice(&decrypted_vec[i][j].as_slice());
//	    }	
//	}

//	let poly_composed = &g_exp.ntt() * &poly_decomposed.ntt(); // first query a part decrypted!!

	//response_a_simple_raw_transposed

//	for i in 0..params.poly_len{
	    //assert_eq!(response_a_simple_raw_transposed.submatrix(0, target_col, dimension, 1).data[i], poly_composed.raw().data[i]);
//	}

	//response_0_times_hint_double_vec = Vec::new(); // b1 times A2
	//response_mult_vec = Vec::new(); // response times response // b1 time b2

//	let mut decrypted_b_vec = Vec::new();
//	for i in 0..4{	
//	    let decrypted_b = decrypt_mlwe_batch(&params, &mlwe_params, dimension, &response_0_times_hint_double_vec[i], &response_mult_vec[i], &y_client.inner);
//	    for j in 0..mlwe_params.poly_len {
		//assert_eq!(decrypted_b[0].data[j], response_0s[i].submatrix(0, target_col, 1, 1).raw().data[j]);
//	    }
//	    decrypted_b_vec.push(decrypted_b);
	    
//	}
//	println!("done");

	//response_0s

//	let mut poly_b_decomposed = PolyMatrixRaw::zero(&mlwe_params, 4, 1);
	
//	for i in 0..4{
//	    poly_b_decomposed.as_mut_slice()[i*mlwe_params.poly_len..(i+1)*mlwe_params.poly_len].copy_from_slice(&decrypted_b_vec[i][0].as_slice());
//	}

//	let poly_b_composed = &g_exp.ntt() * &poly_b_decomposed.ntt();

//	for i in 0..mlwe_params.poly_len{
	    //assert_eq!(poly_b_composed.raw().data[i], response_b_simple_transposed.submatrix(0, target_col, 1, 1).data[i]);
//	}

//	let decrypted_ans = decrypt_mlwe_batch(&params, &mlwe_params, dimension, &poly_composed, &poly_b_composed, &y_client.inner);


//	for i in 0..mlwe_params.poly_len{
	    //println!("{}, {}", server.db()[target_row*db_rows + target_col+i], decrypted_ans[0].data[i]);
	    //assert_eq!(server.db()[(target_col*mlwe_params.poly_len+i)*db_rows + target_row] as u64, decrypted_ans[0].data[i]);// +i = +10
//	}

    }
}
