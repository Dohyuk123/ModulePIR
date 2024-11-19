use spiral_rs::{arith::barrett_coeff_u64, params::Params, poly::*}; // 필요한 외부 모듈 임포트
use ypir::server::ToU64;  // `ToU64` trait 임포트

static DEFAULT_MODULI: [u64; 2] = [268369921u64, 249561089u64];

pub struct Convolution {
    n: usize,
    params: Params,
}

impl Convolution {
    pub fn params_for(n: usize) -> Params {
        Params::init(
            n,
            &DEFAULT_MODULI,
            6.4,
            1,
            2,
            28,
            0,
            0,
            0,
            0,
            true,
            1,
            1,
            1,
            1,
            0,
        )
    }

    pub fn new(n: usize) -> Self {
        let params = Self::params_for(n);
        Self { n, params }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn ntt(&self, a: &[u32]) -> Vec<u32> {
        assert_eq!(a.len(), self.n);
        let mut inp = PolyMatrixRaw::zero(&self.params, 1, 1);
        for i in 0..self.n {
            inp.data[i] = a[i] as u64;
        }
        let ntt = inp.ntt();
        let mut out = vec![0u32; self.params.crt_count * self.n];
        for i in 0..self.params.crt_count * self.n {
            out[i] = ntt.data[i] as u32;
        }
        out
    }

    pub fn raw(&self, a: &[u32]) -> Vec<u32> {
        assert_eq!(a.len(), self.params.crt_count * self.n);
        let mut inp = PolyMatrixNTT::zero(&self.params, 1, 1);
        for i in 0..self.params.crt_count * self.n {
            inp.data[i] = a[i] as u64;
        }
        let raw = inp.raw();
        let mut out = vec![0u32; self.n];
        for i in 0..self.n {
            let mut val = raw.data[i] as i64;
            assert!(val < self.params.modulus as i64);
            if val > self.params.modulus as i64 / 2 {
                val -= self.params.modulus as i64;
            }
            if val < 0 {
                val += ((self.params.modulus as i64) / (1i64 << 32)) * (1i64 << 32);
                val += 1i64 << 32;
            }
            assert!(val >= 0);
            out[i] = (val % (1i64 << 32)) as u32;
        }
        out
    }
}

// test 함수 직접 정의
fn run_ntt_raw_test() {
    let n = 1 << 10;
    let conv = Convolution::new(n);
    let a = (0..n).map(|_| fastrand::u32(..)).collect::<Vec<_>>();
    let ntt = conv.ntt(&a);
    let raw = conv.raw(&ntt);
    assert_eq!(a, raw);
    println!("test_ntt_raw passed!");
}

fn main() {
    // 메인 함수에서 테스트 함수 호출
    run_ntt_raw_test();
}
