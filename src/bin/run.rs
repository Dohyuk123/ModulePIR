use ypir::scheme::{run_module_pir_on_params, run_module_pir_on_params_no_fn};
use ypir::params::params_for_scenario;

use clap::Parser;

/// Run the YPIR scheme with the given parameters
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    item_size_bytes: Option<usize>
}

fn main() {
    let args = Args::parse();
    let Args {
        item_size_bytes,
    } = args;

    let item_size_bytes = item_size_bytes.unwrap_or(256) as u32;
    //item_size_bytes -= 1;

    if item_size_bytes > 2048 {
	panic!("Items must be smaller than 2048 bytes!");
    }

    if item_size_bytes <= 8 {
	panic!("Items must be larger than 8 bytes!");
    }

    let mlwe_bit = 32 - item_size_bytes.leading_zeros() - 1;

    /*
    let mut params = params_for_scenario(1<<30, 1);
    params.pt_modulus = 1<<16;
    
    //256MB: 3, 2 //1GB: 4, 3 //8GB: 5, 5
    params.db_dim_1 = 6;
    params.db_dim_2 = 5;

    let mut mlwe_params = params.clone();
    mlwe_params.poly_len_log2 = mlwe_bit as usize;
    mlwe_params.poly_len = 1<<mlwe_params.poly_len_log2;

    println!("mlwe_dimension : {}", mlwe_params.poly_len);
	
    let mut simple_params = mlwe_params.clone();
    simple_params.modulus = 1<<30;
    simple_params.modulus_log2 = 30;
*/
    //run_module_pir_on_params_no_fn(&params, &mlwe_params, &simple_params);

    run_module_pir_on_params(mlwe_bit);

}
