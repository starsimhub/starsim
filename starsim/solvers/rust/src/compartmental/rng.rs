//! A NumPy Generator, drawn from through its bit generator's C interface as Numba draws from it, so the Generator's state advances
//! exactly as with the Numba kernels: `random()`, `integers(0, 2**62)`, and `exponential()`, with Numba's algorithms for them; and
//! the exact draws the kernels take on their views (`dr`)

use std::ffi::c_void;
use super::ziggurat::{EXP_R, FE, KE, WE};

/// NumPy's `bitgen_t` (numpy/random/bitgen.h)
#[repr(C)]
pub struct BitGen {
    state: *mut c_void,
    next_uint64: unsafe extern "C" fn(*mut c_void) -> u64,
    #[allow(dead_code)]
    next_uint32: unsafe extern "C" fn(*mut c_void) -> u32,
    next_double: unsafe extern "C" fn(*mut c_void) -> f64,
    #[allow(dead_code)]
    next_raw: unsafe extern "C" fn(*mut c_void) -> u64,
}

/// A Generator's bit generator, held by the caller for the call
#[derive(Clone, Copy)]
pub struct Rng { g: *const BitGen }

unsafe impl Send for Rng {} // Each thread of advance_threads() draws from its own runs' generators only
unsafe impl Sync for Rng {}

impl Rng {
    pub fn new(g: *const BitGen) -> Self { Rng { g } }

    #[inline(always)]
    pub fn next_u64(&self) -> u64 { unsafe { ((*self.g).next_uint64)((*self.g).state) } }

    /// `rng.random()`
    #[inline(always)]
    pub fn random(&self) -> f64 { unsafe { ((*self.g).next_double)((*self.g).state) } }

    /// `rng.integers(0, 2**62)`, by Lemire's rejection (Numba's `bounded_lemire_uint64()`)
    pub fn integers62(&self) -> i64 {
        let rng: u64 = (1u64 << 62) - 1;
        let excl = rng + 1;
        let mut x = self.next_u64();
        let mut leftover = x.wrapping_mul(excl);
        if leftover < excl {
            let threshold = (u64::MAX - rng) % excl;
            while leftover < threshold {
                x = self.next_u64();
                leftover = x.wrapping_mul(excl);
            }
        }
        ((x as u128*excl as u128) >> 64) as i64
    }

    /// `rng.exponential(scale)`: the ziggurat of NumPy (Numba's `random_standard_exponential()`)
    pub fn exponential(&self, scale: f64) -> f64 {
        scale*loop {
            let mut ri = self.next_u64() >> 3;
            let idx = (ri & 0xFF) as usize;
            ri >>= 8;
            let x = ri as f64*WE[idx];
            if ri < KE[idx] {
                break x;
            } else if idx == 0 {
                break EXP_R - (-self.random()).ln_1p();
            } else if (FE[idx - 1] - FE[idx])*self.random() + FE[idx] < (-x).exp() {
                break x;
            }
        }
    }
}

/// The exact draws the kernels take (`starsim/solvers/numba/draws.py`, as `crate::draws`), on the kernels' views
pub mod dr {
    use crate::compartmental::views::{A1, A2};
    use crate::draws;

    pub use draws::{binomial, mix, poisson};

    /// Uniforms 0, 1, ... of the stream with key k (an int64 with the bits of the key, as Numba passes it), into out (`draws.uniforms()`)
    #[inline(always)]
    pub fn uniforms(k: i64, out: A1<f64>) -> A1<f64> {
        draws::uniforms(k as u64, 0, out.sl_mut());
        out
    }

    /// A binomial for each lane of a row, into out; w is scratch of 9 rows (`draws.binomials()`)
    #[inline(always)]
    pub fn binomials(n: A1<f64>, p: A1<f64>, u: A1<f64>, out: A1<f64>, w: A2<f64>) {
        draws::binomials(n.sl(), p.sl(), u.sl(), out.sl_mut(), w.flat(), w.s0());
    }

    /// A Poisson draw for each lane of a row (`draws.poissons()`)
    #[inline(always)]
    pub fn poissons(lam: A1<f64>, u: A1<f64>, out: A1<f64>, w: A2<f64>) {
        draws::poissons(lam.sl(), u.sl(), out.sl_mut(), w.flat(), w.s0());
    }
}
