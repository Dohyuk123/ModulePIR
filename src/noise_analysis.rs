use std::f64::consts::PI;

use log::debug;

use spiral_rs::{arith::rescale, client::Client, params::Params, poly::*};

use super::{
    lwe::LWEParams,
    params::{params_for_scenario, GetQPrime},
};

/*
    \taudouble &= \frac{\qtilde_{2, 2}}{2 p} - (\qtilde_{2, 2} \bmod p) - \frac{1}{2}
        \left( 2 + (\qtilde_{2, 2} \bmod p) + (\qtilde_{2, 2} / q_2)(q_2 \bmod p) \right) \\
    \sigmadouble^2 &\le (\qtilde_{2, 2} / \qtilde_{2, 1})^2 d_2 \sigma_2^2 / 4 +
        (\qtilde_{2, 2} / q_2)^2 (\sigma_2^2 / 4) (\ell_2 p^2 + (d_2^2 - 1)(t d_2 z^2) / 3) \\
    \tausimple &= \frac{\qtilde_1}{2N} - (\qtilde_1 \bmod N) -
      \frac{1}{2} \left( 2 + \qtilde_1 \bmod N + (\qtilde_1 / q_1) (q_1 \bmod N) \right) / 2 \\
    \sigmasimple &\le d_1 \sigma_1^2 / 4 + (\qtilde_1 / q_1)^2 \ell_1 N^2 \sigma_1^2 / 4\iftoggle{fullversion}{.}{}

    t = \lfloor \log_z q_2 \rfloor + 1
*/

fn tau_double(q_2: f64, qtilde_2_2: f64, p: f64) -> f64 {
    let term1 = qtilde_2_2 / (2.0 * p);
    let term2 = -(qtilde_2_2 % p);
    let term3 = -(1.0 / 2.0) * (2.0 + (qtilde_2_2 % p) + (qtilde_2_2 / q_2) * (q_2 % p)) / 2.0;

    term1 + term2 + term3
}

fn sigma_2_double(
    d_2: f64,
    q_2: f64,
    qtilde_2_2: f64,
    qtilde_2_1: f64,
    sigma_2: f64,
    p: f64,
    z: f64,
    l_2: f64,
) -> f64 {
    let t = (q_2.log2() / z.log2()).floor() + 1.0;
    let term1 = (qtilde_2_2 / qtilde_2_1).powi(2) * d_2 * sigma_2.powi(2) / 4.0;
    let term2 = (qtilde_2_2 / q_2).powi(2)
        * (sigma_2.powi(2) / 4.0)
        * (l_2 * p.powi(2) + (d_2.powi(2) - 1.0) * (t * d_2 * z.powi(2)) / 3.0);

    

    println!("l2, p, term1, term2 : {}, {}, {}, {}", l_2, p, term1, term2);
    println!("term1 + term2 = {}", term1 + term2);

    term1 + term2
}

fn delta_double(
    d_2: f64,
    q_2: f64,
    qtilde_2_2: f64,
    qtilde_2_1: f64,
    sigma_2: f64,
    p: f64,
    z: f64,
    l_2: f64,
) -> (f64, f64) {
    let tau = tau_double(q_2, qtilde_2_2, p);
    let sigma_2 = sigma_2_double(d_2, q_2, qtilde_2_2, qtilde_2_1, sigma_2, p, z, l_2);
    let delta = 2.0 * (-PI * tau.powi(2) / sigma_2).exp();
    (delta, sigma_2)
}

fn tau_simple(q_1: f64, qtilde_1: f64, n: f64) -> f64 {
    let term1 = qtilde_1 / (2.0 * n);
    let term2 = -(qtilde_1 % n);
    let term3 = -(1.0 / 2.0) * (2.0 + qtilde_1 % n + (qtilde_1 / q_1) * (q_1 % n)) / 2.0;

    term1 + term2 + term3
}

fn sigma_2_simple(d_1: f64, q_1: f64, qtilde_1: f64, sigma_1: f64, n: f64, l_1: f64) -> f64 {
    let term1 = d_1 * sigma_1.powi(2) / 4.0;
    let term2 = (qtilde_1 / q_1).powi(2) * l_1 * n.powi(2) * sigma_1.powi(2) / 4.0;
    term1 + term2
}

fn delta_simple(d_1: f64, q_1: f64, qtilde_1: f64, sigma_1: f64, n: f64, l_1: f64) -> (f64, f64) {
    let tau = tau_simple(q_1, qtilde_1, n);
    let sigma_2 = sigma_2_simple(d_1, q_1, qtilde_1, sigma_1, n, l_1);
    let delta = 2.0 * (-PI * tau.powi(2) / sigma_2).exp();
    (delta, sigma_2)
}

pub fn simplepir_correctness(_lwe_n: f64, lwe_q: f64, lwe_s: f64, lwe_p: f64, upper_n: f64) -> f64 {
    let log2_sqrt_upper_n = (upper_n as f64).sqrt().log2();
    let pt_term = lwe_p.log2() * 2.0 - 2.0; // (p/2)^2
    let noise_term = lwe_s.log2() * 2.0;
    let log2_s_2 = log2_sqrt_upper_n + pt_term + noise_term; // log2(s'^2)
    debug!("log2_s_2: {}", log2_s_2);

    // get probability that noise is less than q/2p
    // subgaussian formula: Pr[noise > T] = 2 \exp(-\pi T^2 / s'^2)
    let noise_threshold = lwe_q / (2.0 * lwe_p);
    let noise_threshold_term = -PI * noise_threshold.powi(2) / 2f64.powf(log2_s_2);
    let err_prob = 2.0 * noise_threshold_term.exp();
    let log2_err_prob = err_prob.log2();
    debug!("log2_err_prob: {}", log2_err_prob);
    log2_err_prob
}

pub struct ModulePIRParams {
    /// Ciphertext modulus (≈ 2^56)
    pub q: f64,
    /// Gaussian error parameter (6.4 * sqrt(2π))
    pub sigma: f64,
    /// Modulus switching targets: q̃₁ = 2^28, q̃₂ = 2^20
    pub q_tilde_1: f64,
    pub q_tilde_2: f64,
    /// RLWE ring degree d = 2^11
    pub d: f64,
    /// MLWE ring degree n (d = kn)
    pub n: f64,
    /// MLWE rank k
    pub k: f64,
    /// Gadget base for key switching (z = 2^19)
    pub z: f64,
    /// Plaintext modulus
    pub p: f64,
    /// Database dimensions: ℓ₁ = m₁·d rows, ℓ₂ = m₂·n columns
    pub l1: f64,
    pub l2: f64,
}

impl Default for ModulePIRParams {
    fn default() -> Self {
        Self {
            q: 66974689739603969.0,
            sigma: 6.4 * (2.0 * PI).sqrt(),
            q_tilde_1: 2.0f64.powi(28),
            q_tilde_2: 2.0f64.powi(20),
            d: 2.0f64.powi(11),  // 2048
            n: 2.0,            // example: n = 256
            k: 1024.0,              // d = kn = 2048
            z: 2.0f64.powi(19),
            p: 2.0f64.powi(16),
            l1: 2.0f64.powi(18), // max 128GB
            l2: 2.0f64.powi(18),
            
        }
    }
}

impl ModulePIRParams {
    /// t = ⌈log_z(q)⌉
    pub fn t(&self) -> f64 {
        (self.q.log2() / self.z.log2()).ceil()
    }

    /// ρ = ⌈kappa(k+1)/k⌉
    pub fn rho(&self) -> f64 {
        let kappa = 2.0;
        (kappa * (self.k + 1.0) / self.k).ceil()
    }

    /// τ_LM: Error threshold for LWE-to-MLWE packing
    /// τ_LM = q̃₂/(2p) - (q̃₂ mod p) - ½(2 + (q̃₂ mod p) + (q̃₂/q)(q mod p))
    pub fn tau_lm(&self) -> f64 {
        tau_common(self.q, self.q_tilde_2, self.p)
    }

    /// τ_MR: Error threshold for MLWE-to-RLWE packing (same formula)
    pub fn tau_mr(&self) -> f64 {
        tau_common(self.q, self.q_tilde_2, self.p)
    }

    /// σ²_scan for SimplePIR: √ℓ₁ · (p/2) · σ
    fn sigma_scan_1(&self) -> f64 {
        self.l1.sqrt() * (self.p / 2.0) * self.sigma
    }

    /// σ²_pack,LM: LWE-to-MLWE packing error variance
    /// σ²_pack,LM ≤ ((n² - 1)/3) · (ktn·z²·σ²/4)
    fn sigma_pack_lm_sq(&self) -> f64 {
        let t = self.t();
        ((self.n.powi(2) - 1.0) / 3.0) * (self.k * t * self.n * self.z.powi(2) * self.sigma.powi(2) / 4.0)
    }

    /// σ²_LM: Total variance for LWE-to-MLWE stage
    /// σ²_LM ≤ (q̃₂/q̃₁)² · (nk·σ²/4) + (q̃₂/q)² · (σ²/4) · (ℓ₁·p² + ((n²-1)·ktn·z²)/3)
    pub fn sigma_lm_sq(&self) -> f64 {
        let t = self.t();
        
        // Term 1: modulus switching rounding error
        let term1 = (self.q_tilde_2 / self.q_tilde_1).powi(2) 
            * (self.n * self.k * self.sigma.powi(2) / 4.0);
        
        // Term 2: scan error + packing error
        let scan_sq = self.l1 * self.p.powi(2);
        let pack_term = (self.n.powi(2) - 1.0) * self.k * t * self.n * self.z.powi(2) / 3.0;
        let term2 = (self.q_tilde_2 / self.q).powi(2) 
            * (self.sigma.powi(2) / 4.0) 
            * (scan_sq + pack_term);
        
        term1 + term2
    }

    /// σ²_scan,2 for DoublePIR: √m₂ · (p/2) · σ where m₂ = ℓ₂/n
    fn sigma_scan_2(&self) -> f64 {
        let m2 = self.l2 / self.n;
        m2.sqrt() * (self.p / 2.0) * self.sigma
    }

    /// σ²_pack,MR: MLWE-to-RLWE packing error variance
    /// σ²_pack,MR ≤ ((k² - 1)/3) · (td·z²·σ²/4)
    fn sigma_pack_mr_sq(&self) -> f64 {
        let t = self.t();
        ((self.k.powi(2) - 1.0) / 3.0) * (t * self.d * self.z.powi(2) * self.sigma.powi(2) / 4.0)
    }

    /// σ²_MR: Total variance for MLWE-to-RLWE stage
    /// σ²_MR ≤ (q̃₂/q̃₁)² · (d·σ²/4) + (q̃₂/q)² · (σ²/4) · (ℓ₂·p² + ((k²-1)·td·z²)/3)
    pub fn sigma_mr_sq(&self) -> f64 {
        let t = self.t();
        
        // Term 1: modulus switching rounding error (d = kn terms)
        let term1 = (self.q_tilde_2 / self.q_tilde_1).powi(2) 
            * (self.d * self.sigma.powi(2) / 4.0);
        
        // Term 2: scan error + packing error
        let scan_sq = self.l2 * self.p.powi(2);
        let pack_term = (self.k.powi(2) - 1.0) * t * self.d * self.z.powi(2) / 3.0;
        let term2 = (self.q_tilde_2 / self.q).powi(2) 
            * (self.sigma.powi(2) / 4.0) 
            * (scan_sq + pack_term);
        
        term1 + term2
    }

    /// δ_LM: Correctness error for LWE-to-MLWE packing
    /// δ_LM = 2n · exp(-π·τ²_LM / σ²_LM)
    pub fn delta_lm(&self) -> f64 {
        let tau = self.tau_lm();
        let sigma_sq = self.sigma_lm_sq();
        2.0 * self.n * (-PI * tau.powi(2) / sigma_sq).exp()
    }

    /// δ_MR: Correctness error for MLWE-to-RLWE packing
    /// δ_MR = 2dρ · exp(-π·τ²_MR / σ²_MR)
    pub fn delta_mr(&self) -> f64 {
        let tau = self.tau_mr();
        let sigma_sq = self.sigma_mr_sq();
        let rho = self.rho();
        2.0 * self.d * rho * (-PI * tau.powi(2) / sigma_sq).exp()
    }

    /// Total correctness error: δ = δ_LM + δ_MR
    pub fn delta(&self) -> f64 {
        self.delta_lm() + self.delta_mr()
    }

    /// Log2 of individual error components (for debugging)
    pub fn log2_deltas(&self) -> (f64, f64, f64) {
        (self.delta_lm().log2(), self.delta_mr().log2(), self.delta().log2())
    }
}

/// Common tau formula used in both stages:
/// τ = q̃₂/(2p) - (q̃₂ mod p) - ½(2 + (q̃₂ mod p) + (q̃₂/q)(q mod p))
fn tau_common(q: f64, q_tilde: f64, p: f64) -> f64 {
    let term1 = q_tilde / (2.0 * p);
    let term2 = -(q_tilde % p);
    let term3 = -0.5 * (2.0 + (q_tilde % p) + (q_tilde / q) * (q % p));
    term1 + term2 + term3
}

pub struct YPIRSchemeParams {
    pub l1: f64,
    pub d1: f64,
    pub p1: f64,
    pub s1: f64,
    pub q1: f64,
    pub q1_prime: f64,

    pub d2: f64,
    pub l2: f64,
    pub p2: f64,
    pub s2: f64,
    pub q2: f64,
    pub q2_1_prime: f64,
    pub q2_2_prime: f64,
    pub t: f64,
}

impl Default for YPIRSchemeParams {
    fn default() -> Self {
        let max_db_bits = 64 * (1 << 33); // 64 GB
        let lwe_params = LWEParams::default();
        let params = params_for_scenario(max_db_bits, 1);
        Self::from_params(&params, &lwe_params)
    }
}

impl YPIRSchemeParams {
    pub fn from_params(params: &Params, lwe_params: &LWEParams) -> Self {
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = 1 << (params.db_dim_2 + params.poly_len_log2);

        // Warning: the paper uses reversed reference to q_prime_1 and q_prime_2
        let q2_1_prime = params.get_q_prime_2() as f64;
        let q2_2_prime: f64 = params.get_q_prime_1() as f64;

        assert!(q2_1_prime >= q2_2_prime);

        Self {
            l1: db_rows as f64, // max size for 64 GB
            d1: lwe_params.n as f64,
            p1: lwe_params.pt_modulus as f64,
            s1: lwe_params.noise_width,
            q1: lwe_params.modulus as f64,
            q1_prime: lwe_params.get_q_prime_1() as f64,

            d2: params.poly_len as f64,
            l2: db_cols as f64,
            p2: params.pt_modulus as f64,
            s2: params.noise_width,
            q2: params.modulus as f64,
            q2_1_prime,
            q2_2_prime,
            t: params.t_exp_left as f64,
        }
    }

    /// Returns the modulus for plaintext database elements, N in the paper, and p_1 in the implementation.
    pub fn n(&self) -> f64 {
        self.p1
    }

    /// Returns the gadget decomposition base, z in the paper. In the implementation, we just set t = 3.
    pub fn z(&self) -> f64 {
        2.0f64.powf(self.q2.log2().ceil() / self.t).ceil()
    }

    pub fn delta_simple(&self) -> (f64, f64) {
        delta_simple(self.d1, self.q1, self.q1_prime, self.s1, self.n(), self.l1)
    }

    pub fn delta_double(&self) -> (f64, f64) {
	println!("z : {}", self.z());        
	delta_double(
            self.d2,
            self.q2,
            self.q2_2_prime,
            self.q2_1_prime,
            self.s2,
            self.p2,
            self.z(),
            self.l2,
        )
    }

    pub fn delta(&self) -> f64 {
        let (delta_simple, _) = self.delta_simple();
        let (delta_double, _) = self.delta_double();
	println!("delta simple, delta double: {} {}", delta_simple, delta_double);
	delta_double + delta_double

    }

    /// The square of the subgaussian parameter for the outer ciphertext noise.
    pub fn expected_outer_noise(&self) -> f64 {
        let (_, sigma_2) = self.delta_double();
        (self.q2 / self.q2_2_prime).powi(2) * sigma_2
    }
}

pub fn measure_noise_width_squared<'a>(
    params: &Params,
    client: &Client<'a>,
    ct: &PolyMatrixNTT<'a>,
    pt: &PolyMatrixRaw<'a>,
    coeffs_to_measure: usize,
) -> f64 {
    let m_i64 = params.modulus as i64;
    let dec_result = client.decrypt_matrix_reg(ct).raw();
    let mut total = 0f64;
    for i in 0..coeffs_to_measure {
        // let decrypted_val = wrapped(dec_result.data[i], params.modulus);
        // let true_val = wrapped(
        //     rescale(pt.data[i], params.pt_modulus, params.modulus),
        //     params.modulus,
        // );
        let decrypted_val = dec_result.data[i] as i64;
        let true_val = rescale(pt.data[i], params.pt_modulus, params.modulus) as i64;
        let diff = decrypted_val - true_val;
        let diff_mod = diff.min(m_i64 - diff);
        let noise_2 = (diff_mod as f64).powi(2);
        assert!(noise_2 >= 0.0);
        // if noise_2.log2() >= 77.0 {
        //     debug!(
        //         "i: {}, noise_2: {}, diff: {}, diff_mod: {}, decrypted_val: {}, true_val: {}",
        //         i, noise_2, diff, diff_mod, decrypted_val, true_val
        //     );
        // }
        total += noise_2;
    }
    let variance = total / params.poly_len as f64;
    assert!(variance >= 0.0);

    // noise_standard_deviation * sqrt(2*pi) == subg_width
    // variance * 2*pi == subg_width^2
    let subg_width_2 = variance * 2.0 * PI;
    subg_width_2
}

#[cfg(test)]
mod tests {
    use spiral_rs::arith::barrett_reduction_u128;

    use super::super::{
        client::{decrypt_ct_reg_measured, YClient},
        scheme::*,
    };
    use super::*;

    #[test]
    fn test_new_calc() {
        let ysp = YPIRSchemeParams::default();
        let (delta_simple, _) = ysp.delta_simple();
        debug!("log2(delta_simple): {}", delta_simple.log2());

        let (delta_double, _) = ysp.delta_double();
        debug!("log2(delta_double): {}", delta_double.log2());

        let expected_outer_noise = ysp.expected_outer_noise();
        debug!(
            "log2(expected_outer_noise): {}",
            expected_outer_noise.log2()
        );
    }

    #[test]
    fn test_ypir_scheme_params() {
        let mut ysp = YPIRSchemeParams::default();

	let lwe_params = LWEParams::default();
	let mut params = params_for_scenario(1<<30, 8);
	params.db_dim_1 = 1<<18;
	params.db_dim_2 = 1<<18;
	params.pt_modulus = 1<<15;
	//params.modulus = 1<<46;

	let mut ysp_tmp = YPIRSchemeParams::from_params(&params, &lwe_params);

        let total_log2_delta = ysp.delta().log2();
        println!("total_log2_delta: {}", total_log2_delta);

        assert!(total_log2_delta < -40.);

	println!("q2 : {}", ysp.q2);

	//ysp.p2 = (1<<8) as f64;
	//ysp.q2 = 66974689739603970 as f64;
	//ysp.q2 = 281474976710656 as f64;
	//ysp.l2 = (1<<) as f64;
	//println!("{}, {}", ysp.q2_1_prime, ysp.q2_2_prime); // 1 is larger
	//ysp.q2_2_prime = (1<<30) as f64;
	//println!("{}", ysp.l2);

	println!("\ntmp!!!!!!!!!!!!!!\n");


        let total_log2_delta = ysp_tmp.delta().log2();
        println!("total_log2_delta: {}", total_log2_delta);

        assert!(total_log2_delta < -40.);
    }
    
        #[test]
    fn test_modulepir_correctness() {
        let params = ModulePIRParams::default();
        
        println!("=== ModulePIR Parameters ===");
        println!("q = 2^{:.1}", params.q.log2());
        println!("σ = {:.2}", params.sigma);
        println!("q̃₁ = 2^{:.0}, q̃₂ = 2^{:.0}", params.q_tilde_1.log2(), params.q_tilde_2.log2());
        println!("d = {}, n = {}, k = {}", params.d, params.n, params.k);
        println!("t = {}, ρ = {}", params.t(), params.rho());
        println!("ℓ₁ = 2^{:.0}, ℓ₂ = 2^{:.0}", params.l1.log2(), params.l2.log2());
        
        println!("\n=== Error Analysis ===");
        println!("τ_LM = {:.2e}", params.tau_lm());
        println!("τ_MR = {:.2e}", params.tau_mr());
        println!("σ²_LM = 2^{:.1}", params.sigma_lm_sq().log2());
        println!("σ²_MR = 2^{:.1}", params.sigma_mr_sq().log2());
        
        let (log2_delta_lm, log2_delta_mr, log2_delta_total) = params.log2_deltas();
        println!("\n=== Correctness Error ===");
        println!("log₂(δ_LM) = {:.1}", log2_delta_lm);
        println!("log₂(δ_MR) = {:.1}", log2_delta_mr);
        println!("log₂(δ_total) = {:.1}", log2_delta_total);
        
        // Verify correctness error ≤ 2^{-40}
        assert!(log2_delta_total < -40.0, 
            "Correctness error too high: 2^{:.1} > 2^-40", log2_delta_total);
    }

    #[test]
    fn test_parameter_variations() {
        // Test with different record sizes (varying n and k)
        let test_cases = vec![
            (1.0, 2048.0, "15bit records"),
            (2.0, 1024.0, "3.75B records"),
            (4.0, 512.0, "7.5B records"),
            (8.0, 256.0, "15B records"),
            (16.0, 128.0, "30B records"),
            (32.0, 64.0, "60B records"),
            (64.0, 32.0, "120B records"),
            (128.0, 16.0, "240B records"),
            (256.0, 8.0, "480B records"),   // n=256, k=8, d=2048
            (512.0, 4.0, "960B records"),   // n=512, k=4, d=2048
            (1024.0, 2.0, "1920B records"), 
            //(1.0, 2048.0, "4KB records"),// n=1024, k=2, d=2048
        ];
        
        for (n, k, desc) in test_cases {
            let params = ModulePIRParams {
                n,
                k,
                ..Default::default()
            };
            
            let (_, _, log2_delta) = params.log2_deltas();
            println!("{}: log₂(δ) = {:.1}", desc, log2_delta);
            //assert!(log2_delta < -40.0);
        }
    }

    #[test]
    #[ignore]
    fn test_linear_accumulation_noise() {
        let params = params_for_scenario(1 << 43, 1);
        let upper_n = 1 << (11 + 6);

        let mut client = Client::init(&params);
        client.generate_secret_keys();
        let y_client = YClient::new(&mut client, &params);
        let target_idx = 0;
        let query = y_client.generate_query(SEED_0, params.db_dim_1, false, target_idx);

        let db = (0..upper_n)
            .map(|_| fastrand::u64(0..params.pt_modulus))
            .collect::<Vec<_>>();

        let mut acc = vec![0u128; params.poly_len + 1];
        for idx in 0..upper_n {
            for dim in 0..params.poly_len + 1 {
                let query_val = query[dim * upper_n + idx];
                let db_val = db[idx];
                let product = query_val as u128 * db_val as u128;
                acc[dim] += product;
            }
        }

        let mut ct = PolyMatrixRaw::zero(&params, 2, 1);
        for dim in 0..params.poly_len + 1 {
            ct.data[dim] = barrett_reduction_u128(&params, acc[dim]);
        }

        let _plaintext =
            decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len);
        todo!("problem w test: negacyclic");
        // assert_eq!(plaintext.data[0], db[target_idx]);
    }
}
