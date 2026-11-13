//! The exact draws the kernels take (`starsim/solvers/backends/draws.py`, as `crate::draws`), on the kernels' views

use super::views::{A1, A2};
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
