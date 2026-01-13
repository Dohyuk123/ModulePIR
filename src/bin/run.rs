use ypir::scheme::{run_module_pir_on_params, run_module_pir_on_params_no_fn, test_module_pir_on_params_no_fn};
use ypir::params::params_for_scenario;
use clap::Parser;

/// Run the YPIR scheme with the given parameters
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Polynomial degree (default: 128)
    #[arg(short = 'p', long, default_value_t = 128)]
    polynomial_degree: usize,
    
    /// Database dimension 1 (must be <= 7)
    #[arg(short = 'd', long, default_value_t = 4)]
    db_dim_1: usize,
    
    /// Database dimension 2 (must be <= 7)
    #[arg(short = 'e', long, default_value_t = 3)]
    db_dim_2: usize,
}

fn main() {
    let args = Args::parse();
    let Args {
        polynomial_degree,
        db_dim_1,
        db_dim_2,
    } = args;
    
    // Validate db_dim_1 and db_dim_2
    if db_dim_1 > 7 {
        panic!("db_dim_1 must be <= 7!");
    }
    if db_dim_2 > 7 {
        panic!("db_dim_2 must be <= 7!");
    }
    
    // Validate polynomial_degree
    if polynomial_degree < 8 || polynomial_degree > 1024 {
        panic!("polynomial_degree must be between 8 and 1024!");
    }
    if !polynomial_degree.is_power_of_two() {
        panic!("polynomial_degree must be a power of 2!");
    }
    
    let item_size_bytes = polynomial_degree as u32;
    
    if item_size_bytes > 2048 {
        panic!("Items must be smaller than 2048 bytes!");
    }
    
    let mlwe_bit = 32 - item_size_bytes.leading_zeros() - 1;
    
    println!("mlwe bit : {}", mlwe_bit);
    println!("polynomial_degree: {}", polynomial_degree);
    println!("db_dim_1: {}, db_dim_2: {}", db_dim_1, db_dim_2);
    
    println!("db size is {}B", 15u64 * 2048 * 2048 * (1u64 << db_dim_1) * (1u64 << db_dim_2) / 8);
    println!("record size is {}B", 15 * (1<<mlwe_bit) / 8);
    
    
    run_module_pir_on_params(mlwe_bit, db_dim_1, db_dim_2);
}
