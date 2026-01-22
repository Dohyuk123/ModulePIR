# ModulePIR

This is an implementation of the ModulePIR scheme for single-server private information retrieval, introduced in **"ModulePIR: High-Throughput Single-Server PIR with Constant Communication and Flexible Record Sizes"**.

## Running

To build and run this code:

1. Ensure you are running on Ubuntu, and that AVX-512 is available on the CPU (you can run `lscpu` and look for the `avx512f` flag). Our benchmarks were collected using the AWS `r6i.16xlarge` instance type, which has all necessary CPU features.
2. Run `sudo apt-get update && sudo apt-get install -y build-essential`.
3. **Install Rust using rustup** using `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
   - Select `1) Proceed with installation (default)` when prompted
   - After installation, configure the current shell as instructed by running `source "$HOME/.cargo/env"`
4. Run `git clone https://github.com/[REPO-URL]/modulepir.git` and `cd modulepir`.
5. Run `cargo run --release` to run ModulePIR with default parameters. The first time you run this command, Cargo will download and install the necessary libraries to build the code (~2 minutes); later calls will not take as long. Stability warnings can be safely ignored.

We have tested the above steps on a fresh AWS `r6i.16xlarge` Ubuntu 22.04 instance and confirmed they work.

## Options

To pass arguments, make sure to run `cargo run --release -- <ARGS>` (the `--` is important).
```
Usage: cargo run --release -- [OPTIONS]

Options:
  -p, --polynomial-degree <POLYNOMIAL_DEGREE>
          Polynomial degree (default: 128)
          Must be a power of 2
          Range: 8 to 1024

  -d, --db-dim-1 <DB_DIM_1>
          Database dimension 1 (default: 4)
          Must be <= 7

  -e, --db-dim-2 <DB_DIM_2>
          Database dimension 2 (default: 3)
          Must be <= 7

  -h, --help     Print help
  -V, --version  Print version
```

## Database Configuration

The database is configured as a matrix of size `(2^db_dim_1 * 2048) x (2^db_dim_2 * 2048)` with entries in modulus `2^pt_mod`.

### Database Size Calculation

The database size is calculated as:
```
db_size (bytes) = pt_mod * 2048 * 2048 * (2^db_dim_1) * (2^db_dim_2) / 8
```

| db_dim_1 | db_dim_2 | Database Size |
|----------|----------|---------------|
| 3        | 2        | 256 MB        |
| 4        | 3        | 1 GB          |
| 5        | 5        | 8 GB          |
| 6        | 6        | 32 GB         |

### Record Size

The record size is determined by the polynomial degree and plaintext modulus:
```
record_size (bytes) = pt_mod * polynomial_degree / 8
```

| polynomial_degree | pt_mod | Record Size  |
|-------------------|--------|--------------|
| 16                | 16     | 32 bytes     |
| 32                | 16     | 64 bytes     |
| 64                | 16     | 128 bytes    |
| 128               | 16     | 256 bytes    |
| 256               | 16     | 512 bytes    |
| 512               | 16     | 1024 bytes   |
| 1024              | 15     | 1920 bytes   |

### Parameter Constraints

| pt_mod | polynomial_degree range |
|--------|-------------------------|
| 15     | 1024                    |
| 16     | 8 to 512                |

### Benchmark Configurations

The benchmarks in the paper use `polynomial_degree = 128` with the following database dimensions:

| Database Size | db_dim_1 | db_dim_2 | Command                            |
|---------------|----------|----------|------------------------------------|
| 256 MB        | 3        | 2        | `cargo run --release -- -d 3 -e 2` |
| 1 GB          | 4        | 3        | `cargo run --release -- -d 4 -e 3` |
| 8 GB          | 5        | 5        | `cargo run --release -- -d 5 -e 5` |
| 32 GB         | 6        | 6        | `cargo run --release -- -d 6 -e 6` |

### Examples
```bash
# Default (1 GB database, 256-byte records)
cargo run --release

# 8 GB database
cargo run --release -- -d 5 -e 5

# 32 GB database
cargo run --release -- -d 6 -e 6

# 512-byte records
cargo run --release -- -p 256

# 1024-byte records
cargo run --release -- -p 512

# 1920-byte records (uses pt_mod=15)
cargo run --release -- -p 1024

# 8 GB database with 512-byte records
cargo run --release -- -p 256 -d 5 -e 5
```
