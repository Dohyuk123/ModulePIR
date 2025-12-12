use ypir::scheme::{run_module_pir_on_params, run_module_pir_on_params_no_fn, test_module_pir_on_params_no_fn};
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

    let item_size_bytes = item_size_bytes.unwrap_or(128) as u32;
    //item_size_bytes -= 1;

    if item_size_bytes > 2048 {
	panic!("Items must be smaller than 2048 bytes!");
    }

    //if item_size_bytes <= 8 {
//	panic!("Items must be larger than 8 bytes!");
//    }

    let mlwe_bit = 32 - item_size_bytes.leading_zeros() - 1;

    //test_module_pir_on_params_no_fn(mlwe_bit);
    run_module_pir_on_params(mlwe_bit);

}
