//! Exact random draws, from one uniform each where possible: `starsim/perflib/draws.py`, bit for bit (see its docstring).
//! Each function has the name and arguments of the Numba one. Where Numba turns a float into an integer, `to_i64()` and
//! `to_u64()` do it as Numba does on this CPU, so even out-of-range values (NaN, inf, n above 2^63) give the same draws.

pub const GOLDEN: u64 = 0x9e3779b97f4a7c15;
pub const MIX1: u64 = 0xbf58476d1ce4e5b9;
pub const MIX2: u64 = 0x94d049bb133111eb;
pub const U11: u64 = 11;
pub const U40: u64 = 40; // 64 - 24: the top 24 bits, for a uniform that is a float32 (see crn_uniform())
pub const INV53: f64 = 1.0/9007199254740992.0; // 2^-53
pub const INV24: f64 = 1.0/16777216.0; // 2^-24
pub const TWO53: f64 = 9007199254740992.0;
pub const SALT: u64 = 0x5851f42d4c957f2d; // Seeds a rejection sampler's stream from its uniform: see stream()
pub const INV_MAX: f64 = 30.0; // The mean up to which binomial and Poisson draws are by inversion (as NumPy's)
pub const LN2_HI: f64 = 0.693147180369123816490; // ln 2 in two parts, for an exact range reduction (Cody and Waite)
pub const LN2_LO: f64 = 1.90821492927058770002e-10;
pub const LOG2E: f64 = 1.4426950408889634;
pub const SQRT2: f64 = 1.4142135623730951;
pub const EXPO: i64 = 0x7ff0000000000000;
pub const MANT: i64 = 0x000fffffffffffff;
pub const ONE_BITS: i64 = 0x3ff0000000000000;

/// 1/k, for the walks of the inversions (which take at most about 100 steps), so they multiply instead of divide
pub const RECIP: [f64; 128] = {
    let mut r = [0.0; 128];
    let mut k = 0;
    while k < 128 {
        r[k] = 1.0/(k + 1) as f64;
        k += 1;
    }
    r
};

unsafe extern "C" {
    fn lgamma(x: f64) -> f64; // The C library's, as Numba's math.lgamma
}


//%% Numba's arithmetic, where Rust's differs

/// Python's max(a, b) as Numba compiles it: b if b > a, else a (so a NaN a stays)
#[inline(always)]
pub fn max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's min(a, b) as Numba compiles it: b if b < a, else a
#[inline(always)]
pub fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// A float to an int64 as Numba gives it (int(), math.floor()): out of range or NaN, x86 gives -2^63 where Rust would saturate
#[inline(always)]
pub fn to_i64(x: f64) -> i64 {
    #[cfg(target_arch = "x86_64")]
    unsafe { std::arch::x86_64::_mm_cvttsd_si64(std::arch::x86_64::_mm_set_sd(x)) }
    #[cfg(not(target_arch = "x86_64"))]
    { x as i64 }
}

/// A float to a uint64 as Numba gives it (np.uint64()): on x86, from two conversions to int64, as LLVM lowers it
#[inline(always)]
pub fn to_u64(x: f64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let a = to_i64(x);
        let b = to_i64(x - 9223372036854775808.0);
        (a | (b & (a >> 63))) as u64
    }
    #[cfg(not(target_arch = "x86_64"))]
    { x as u64 }
}

/// The float64 with these bits
#[inline(always)]
fn as_float(i: i64) -> f64 {
    f64::from_bits(i as u64)
}

/// The bits of a float64
#[inline(always)]
fn as_int(x: f64) -> i64 {
    x.to_bits() as i64
}


//%% Uniforms

/// The mixing function of SplitMix64, which turns a counter into random bits
#[inline(always)]
pub fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(MIX1);
    let z = (z ^ (z >> 27)).wrapping_mul(MIX2);
    z ^ (z >> 31)
}

/// The key of a stream of uniforms, from a seed and the index of a call (as `ss.distributions.hash_uniforms()`): the bits of the Numba one's int64
#[inline(always)]
pub fn key(seed: i64, ind: i64) -> u64 {
    mix((seed as u64 ^ (ind as u64).wrapping_mul(GOLDEN)).wrapping_add(GOLDEN))
}

/// Uniform i in (0, 1) of the stream with key k (as `ss.distributions.hash_uniforms()` gives for slot i)
#[inline(always)]
pub fn uniform(k: u64, i: u64) -> f64 {
    ((mix(i.wrapping_add(k).wrapping_mul(GOLDEN)) >> U11) as f64).max(0.5)*INV53
}

/// Uniform `slot` in (0, 1) of the stream with key k, from the hash's top bits (shift and scale as `crn_bits()`)
#[inline(always)]
pub fn crn_uniform(k: u64, slot: u64, shift: u64, scale: f64) -> f64 {
    ((mix(slot.wrapping_add(k).wrapping_mul(GOLDEN)) >> shift) as f64).max(0.5)*scale
}

/// How many bits of each hash a uniform keeps, as the shift and scale that `crn_uniform()` takes, from the size of its dtype in bytes
pub fn crn_bits(itemsize: usize) -> (u64, f64) {
    if itemsize <= 4 { (U40, INV24) } else { (U11, INV53) }
}

avx2! {
    /// Uniforms start, start+1, ... of the stream with key k
    pub fn uniforms(k: u64, start: u64, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = uniform(k, start.wrapping_add(i as u64));
        }
    }
}

/// The state of a stream of uniforms for a rejection sampler, seeded by the bits of the uniform it stands for
#[inline(always)]
pub fn stream(u: f64) -> u64 {
    mix(to_u64(u*TWO53) ^ SALT)
}

/// The next uniform of a stream, and its new state
#[inline(always)]
pub fn next_u(st: u64) -> (f64, u64) {
    let st = st.wrapping_add(GOLDEN);
    (((mix(st) >> U11) as f64).max(0.5)*INV53, st)
}


//%% float64 exp and log that LLVM can vectorize (to within 1-2 ulp of libm's)

/// exp(x), for x in [-708, 709] (clipped to it): x = k*ln2 + r with |r| <= ln2/2, a degree-13 Taylor polynomial for exp(r) (by Estrin's scheme, for a short chain of dependent operations), times 2^k by adding k to its exponent
#[inline(always)]
pub fn exp(x: f64) -> f64 {
    let x = min(max(x, -708.0), 709.0);
    let k = (x*LOG2E + 0.5).floor(); // Numba's is an int64; the same here, since k is whole and small, or NaN (0 as an int here, and a NaN result either way)
    let r = (x - k*LN2_HI) - k*LN2_LO;
    let r2 = r*r;
    let r4 = r2*r2;
    let a = (1.0 + r) + (1.0/2.0 + r*(1.0/6.0))*r2;
    let b = (1.0/24.0 + r*(1.0/120.0)) + (1.0/720.0 + r*(1.0/5040.0))*r2;
    let c = (1.0/40320.0 + r*(1.0/362880.0)) + (1.0/3628800.0 + r*(1.0/39916800.0))*r2;
    let d = 1.0/479001600.0 + r*(1.0/6227020800.0);
    let p = (a + b*r4) + (c + d*r4)*(r4*r4);
    as_float(as_int(p).wrapping_add((k as i32 as i64) << 52)) // Through an int32, which AVX2 converts four at a time (it has no int64 conversion)
}

/// log(x), for x > 0 and normal: x = m*2^e with m in [sqrt(1/2), sqrt(2)), and log(m) = 2*atanh(s), s = (m-1)/(m+1), as a series to s^21 (by Estrin's scheme)
#[inline(always)]
pub fn log(x: f64) -> f64 {
    let bits = as_int(x);
    let e = ((bits & EXPO) >> 52) - 1023;
    let m = as_float((bits & MANT) | ONE_BITS);
    let big = m > SQRT2;
    let m = if big { m*0.5 } else { m };
    let e = (if big { e + 1 } else { e }) as i32 as f64; // Through an int32 (e is within +-1100), as exp()
    let s = (m - 1.0)/(m + 1.0);
    let z = s*s;
    let z2 = z*z;
    let z4 = z2*z2;
    let a = (1.0 + z*(1.0/3.0)) + (1.0/5.0 + z*(1.0/7.0))*z2;
    let b = (1.0/9.0 + z*(1.0/11.0)) + (1.0/13.0 + z*(1.0/15.0))*z2;
    let c = (1.0/17.0 + z*(1.0/19.0)) + z2*(1.0/21.0);
    let t = (a + b*z4) + c*(z4*z4); // By Estrin's scheme
    (e*LN2_HI + 2.0*s*t) + e*LN2_LO
}

/// log(1 + x), for x > -1, accurate for small x too (Goldberg's correction)
#[inline(always)]
pub fn log1p(x: f64) -> f64 {
    let w = 1.0 + x;
    let d = w - 1.0;
    if d == 0.0 { x } else { log(w)*(x/d) }
}

/// exp(x) - 1, for x <= 709, accurate for small x too: a Taylor series near 0
#[inline(always)]
pub fn expm1(x: f64) -> f64 {
    let x2 = x*x;
    let x4 = x2*x2;
    let a = (1.0 + x*(1.0/2.0)) + (1.0/6.0 + x*(1.0/24.0))*x2;
    let b = (1.0/120.0 + x*(1.0/720.0)) + (1.0/5040.0 + x*(1.0/40320.0))*x2;
    let c = (1.0/362880.0 + x*(1.0/3628800.0)) + (1.0/39916800.0 + x*(1.0/479001600.0))*x2;
    let small = x*((a + b*x4) + (c + x4*(1.0/6227020800.0))*(x4*x4));
    if x.abs() < 0.35 { small } else { exp(x) - 1.0 }
}


//%% Binomial and Poisson

/// A term of the Stirling series in BTPE's bound
#[inline(always)]
fn stirling(x2: f64) -> f64 {
    13860. - (462. - (132. - (99. - 140./x2)/x2)/x2)/x2
}

/// Binomial(n, p) for p <= 1/2 and n*p > 30: BTPE (Kachitvichyanukul and Schmeiser 1988), as NumPy's, from the stream st
pub fn binomial_btpe(n: f64, p: f64, mut st: u64) -> f64 {
    let r = p;
    let q = 1.0 - r;
    let fm = n*r + r;
    let m = to_i64(fm.floor());
    let mf = m as f64;
    let p1 = to_i64((2.195*(n*r*q).sqrt() - 4.6*q).floor()) as f64 + 0.5;
    let xm = mf + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5/(15.3 + mf);
    let a = (fm - xl)/(fm - xl*r);
    let laml = a*(1.0 + a/2.0);
    let a = (xr - fm)/(xr*q);
    let lamr = a*(1.0 + a/2.0);
    let p2 = p1*(1.0 + 2.0*c);
    let p3 = p2 + c/laml;
    let p4 = p3 + c/lamr;
    let nrq = n*r*q;
    loop {
        let u;
        let mut v;
        (u, st) = next_u(st);
        let u = u*p4;
        (v, st) = next_u(st);
        if u <= p1 {
            return to_i64((xm - p1*v + u).floor()) as f64;
        }
        let y;
        if u <= p2 {
            let x = xl + (u - p1)/c;
            v = v*c + 1.0 - (mf - x + 0.5).abs()/p1;
            if v > 1.0 {
                continue;
            }
            y = to_i64(x.floor());
        } else if u <= p3 {
            y = to_i64((xl + v.ln()/laml).floor());
            if y < 0 {
                continue;
            }
            v = v*(u - p2)*laml;
        } else {
            y = to_i64((xr - v.ln()/lamr).floor());
            if y as f64 > n {
                continue;
            }
            v = v*(u - p3)*lamr;
        }
        let k = y.wrapping_sub(m).wrapping_abs();
        if k <= 20 || k as f64 >= nrq/2.0 - 1.0 { // Evaluate the pmf ratio directly
            let s = r/q;
            let a = s*(n + 1.0);
            let mut f = 1.0;
            if m < y {
                for i in m + 1..y + 1 {
                    f *= a/i as f64 - s;
                }
            } else if m > y {
                for i in y + 1..m + 1 {
                    f /= a/i as f64 - s;
                }
            }
            if v <= f {
                return y as f64;
            }
            continue;
        }
        let kf = k as f64;
        let rho = (kf/nrq)*((kf*(kf/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5); // Squeeze, then Stirling's bound
        let t = k.wrapping_neg().wrapping_mul(k) as f64/(2.0*nrq);
        let aa = v.ln();
        if aa < t - rho {
            return y as f64;
        }
        if aa > t + rho {
            continue;
        }
        let x1 = y as f64 + 1.0;
        let f1 = mf + 1.0;
        let z = n + 1.0 - mf;
        let w = n - y as f64 + 1.0;
        let (x2, f2, z2, w2) = (x1*x1, f1*f1, z*z, w*w);
        let bound = xm*(f1/x1).ln() + (n - mf + 0.5)*(z/w).ln() + y.wrapping_sub(m) as f64*(w*r/(x1*q)).ln()
            + stirling(f2)/f1/166320.
            + stirling(z2)/z/166320.
            - stirling(x2)/x1/166320.
            - stirling(w2)/w/166320.;
        if aa <= bound {
            return y as f64;
        }
    }
}

/// Poisson(lam) for lam > 30: PTRS (Hörmann 1993), as NumPy's, from the stream st
pub fn poisson_ptrs(lam: f64, mut st: u64) -> f64 {
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53*slam;
    let a = -0.059 + 0.02483*b;
    let invalpha = 1.1239 + 1.1328/(b - 3.4);
    let vr = 0.9277 - 3.6224/(b - 2.0);
    loop {
        let u;
        let v;
        (u, st) = next_u(st);
        let u = u - 0.5;
        (v, st) = next_u(st);
        let us = 0.5 - u.abs();
        let k = to_i64(((2.0*a/us + b)*u + lam + 0.43).floor());
        if us >= 0.07 && v <= vr {
            return k as f64;
        }
        if k < 0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + invalpha.ln() - (a/(us*us) + b).ln() <= -lam + k as f64*loglam - unsafe { lgamma(k.wrapping_add(1) as f64) } {
            return k as f64;
        }
    }
}

/// How far the inversion of Binomial(n, q) walks at most (as NumPy's): beyond it the mass is below 1e-20
#[inline(always)]
pub fn binomial_bound(n: f64, q: f64) -> f64 {
    let m = n*q;
    min(n, m + 10.0*(m*(1.0 - q) + 1.0).sqrt())
}

/// Binomial(n, p) from the uniform u (n is rounded down to a whole number): by inversion for a mean up to INV_MAX
/// (so monotone in u), with u <= 1 - mean a 0 at once; else by BTPE, from a stream seeded by u. NaN for a mean of
/// 2^53 or more (which no model has: it comes from a bug upstream, and BTPE's integers would overflow) or NaN
pub fn binomial(n: f64, p: f64, u: f64) -> f64 {
    let n = n.floor();
    if n <= 0.0 || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return n;
    }
    let flip = p > 0.5; // Draw on the side of 1/2 with the smaller mean
    let q = if flip { 1.0 - p } else { p };
    let m = n*q;
    if !(m <= INV_MAX) {
        if !(m < TWO53) { // Including NaN
            return f64::NAN;
        }
        let y = binomial_btpe(n, q, stream(u));
        return if flip { n - y } else { y };
    }
    let u = if flip { 1.0 - u } else { u };
    if u <= 1.0 - m { // A 0, with no exp or log
        return if flip { n } else { 0.0 };
    }
    let mut px = exp(n*log1p(-q));
    let r = q/(1.0 - q);
    let bound = binomial_bound(n, q);
    let mut cdf = px;
    let mut x = 0.0;
    let mut k = 0; // x as an index: the walk takes at most 86 steps (see binomial_bound()), so the mask only spares a bounds check
    while u > cdf && x < bound { // As binomials() does it, so they give the same counts
        x += 1.0;
        px *= (n - x + 1.0)*r*RECIP[k & 127];
        k += 1;
        cdf += px;
    }
    if flip { n - x } else { x }
}

/// Poisson(lam) from the uniform u: by inversion for a mean up to INV_MAX (monotone in u), with u <= 1 - lam a 0 at once; else by PTRS, from a stream seeded by u; NaN for a mean of 2^53 or more, or NaN (as binomial())
pub fn poisson(lam: f64, u: f64) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if !(lam <= INV_MAX) {
        return if lam < TWO53 { poisson_ptrs(lam, stream(u)) } else { f64::NAN };
    }
    if u <= 1.0 - lam {
        return 0.0;
    }
    let mut px = exp(-lam);
    let bound = lam + 10.0*lam.sqrt() + 10.0;
    let mut cdf = px;
    let mut x = 0.0;
    let mut k = 0; // x as an index: at most 95 steps, so the mask only spares a bounds check
    while u > cdf && x < bound {
        x += 1.0;
        px *= lam*RECIP[k & 127];
        k += 1;
        cdf += px;
    }
    x
}


//%% Rows of binomial and Poisson draws, vectorized

/// For every lane, branch-free: n rounded down, the side of 1/2 to draw on, and its uniform and chance. Lanes the walk
/// shouldn't take (settled at once, or means above INV_MAX or NaN) get a uniform of 0, so they stay at 0. Returns the
/// number of lanes left to walk, and of those with a mean above INV_MAX or NaN.
#[inline(always)]
pub fn prep_binomial(n: &[f64], p: &[f64], u: &[f64], uu: &mut [f64], qq: &mut [f64], flip: &mut [f64], nn: &mut [f64]) -> (usize, usize) {
    let len = n.len();
    let (p, u, uu, qq, flip, nn) = (&p[..len], &u[..len], &mut uu[..len], &mut qq[..len], &mut flip[..len], &mut nn[..len]);
    let mut big = 0;
    let mut need = 0;
    for i in 0..len {
        let ni = n[i].floor();
        let f = p[i] > 0.5;
        let qi = if f { 1.0 - p[i] } else { p[i] };
        let ui = if f { 1.0 - u[i] } else { u[i] };
        let m = ni*qi;
        let large = !(m <= INV_MAX); // Including NaN
        let walk = (ni > 0.0) & (qi > 0.0) & !large & (ui > 1.0 - m);
        nn[i] = ni;
        qq[i] = if walk { qi } else { 0.0 };
        uu[i] = if walk { ui } else { 0.0 };
        flip[i] = f as u8 as f64;
        big += (large & !(ni <= 0.0) & !(qi <= 0.0)) as usize;
        need += walk as usize;
    }
    (need, big)
}

/// For the lanes that walk: the ratio q/(1-q), the bound of the walk, and the chance of 0, (1-q)^n, as binomial() has them
#[inline(always)]
pub fn prep_walk(nn: &[f64], qq: &[f64], rr: &mut [f64], bb: &mut [f64], px: &mut [f64]) {
    let len = nn.len();
    let (qq, rr, bb, px) = (&qq[..len], &mut rr[..len], &mut bb[..len], &mut px[..len]);
    for i in 0..len {
        let q = qq[i];
        let n = if q > 0.0 { nn[i] } else { 0.0 }; // The lanes that don't walk, as if n were 0 (as draws.py)
        let m = n*q;
        rr[i] = q/(1.0 - q);
        bb[i] = min(n, m + 10.0*(m*(1.0 - q) + 1.0).sqrt());
        px[i] = exp(n*log1p(-q));
    }
}

/// exp() of each value of x, in place (as `_exp_row(x, x)`, its one use)
#[inline(always)]
pub fn exp_row(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = exp(*v);
    }
}

/// The start of the walks: a count of 0 with the chance of 0 as the CDF, and how many lanes have further to go
#[inline(always)]
pub fn start(px: &[f64], cdf: &mut [f64], x: &mut [f64], uu: &[f64], bb: &[f64]) -> usize {
    let len = px.len();
    let (cdf, x, uu, bb) = (&mut cdf[..len], &mut x[..len], &uu[..len], &bb[..len]);
    let mut left = 0;
    for j in 0..len {
        cdf[j] = px[j];
        x[j] = 0.0;
        left += ((uu[j] > cdf[j]) & (bb[j] > 0.0)) as usize;
    }
    left
}

/// One step of the walk for all lanes, simple enough for LLVM to vectorize: as binomial() takes it
#[inline(always)]
pub fn step_binomial(px: &mut [f64], cdf: &mut [f64], x: &mut [f64], uu: &[f64], nn: &[f64], rr: &[f64], bb: &[f64], k: f64, inv: f64) -> usize {
    let len = px.len();
    let (cdf, x, uu, nn, rr, bb) = (&mut cdf[..len], &mut x[..len], &uu[..len], &nn[..len], &rr[..len], &bb[..len]);
    let mut left = 0;
    for j in 0..len {
        let a = (uu[j] > cdf[j]) & (k - 1.0 < bb[j]);
        x[j] += a as u8 as f64;
        px[j] *= (nn[j] - k + 1.0)*rr[j]*inv;
        cdf[j] += px[j];
        left += ((uu[j] > cdf[j]) & (k < bb[j])) as usize;
    }
    left
}

/// One step of the Poisson walk for all lanes: as poisson() takes it
#[inline(always)]
pub fn step_poisson(px: &mut [f64], cdf: &mut [f64], x: &mut [f64], uu: &[f64], ll: &[f64], bb: &[f64], k: f64, inv: f64) -> usize {
    let len = px.len();
    let (cdf, x, uu, ll, bb) = (&mut cdf[..len], &mut x[..len], &uu[..len], &ll[..len], &bb[..len]);
    let mut left = 0;
    for j in 0..len {
        let a = (uu[j] > cdf[j]) & (k - 1.0 < bb[j]);
        x[j] += a as u8 as f64;
        px[j] *= ll[j]*inv;
        cdf[j] += px[j];
        left += ((uu[j] > cdf[j]) & (k < bb[j])) as usize;
    }
    left
}

/// The first R rows of the scratch w (a C-contiguous 2D array with rows of `stride` values), each cut to len (as `w[0, :N], w[1, :N], ...`)
#[inline(always)]
fn rows<const R: usize>(w: &mut [f64], stride: usize, len: usize) -> [&mut [f64]; R] {
    assert!(len <= stride && w.len() >= R*stride, "the scratch of binomials() or poissons() is too small");
    let mut it = w.chunks_exact_mut(stride).map(|r| &mut r[..len]);
    std::array::from_fn(|_| it.next().unwrap())
}

avx2! {
    /// Binomial(n, p) for each lane of a row, from the uniforms u, into out: as binomial() gives, but with every lane
    /// that isn't settled at once walking up its pmf in lockstep. w is scratch, at least 9 rows of `stride` >= len(n) values
    pub fn binomials(n: &[f64], p: &[f64], u: &[f64], out: &mut [f64], w: &mut [f64], stride: usize) {
        let len = n.len();
        let (p, u, out) = (&p[..len], &u[..len], &mut out[..len]);
        let [uu, qq, rr, bb, flip, px, cdf, x, nn] = rows::<9>(w, stride, len);
        let (need, big) = prep_binomial(n, p, u, uu, qq, flip, nn);
        if need > 0 {
            prep_walk(nn, qq, rr, bb, px);
            let mut left = start(px, cdf, x, uu, bb);
            let mut k = 0.0;
            while left > 0 {
                k += 1.0;
                left = step_binomial(px, cdf, x, uu, nn, rr, bb, k, 1.0/k);
            }
        } else {
            x.fill(0.0);
        }
        for i in 0..len {
            out[i] = if flip[i] > 0.0 { nn[i] - x[i] } else { x[i] };
            if nn[i] <= 0.0 || p[i] <= 0.0 {
                out[i] = 0.0;
            } else if p[i] >= 1.0 {
                out[i] = nn[i];
            }
        }
        if big > 0 { // Means above INV_MAX (BTPE), or NaN
            for i in 0..len {
                if !(nn[i] <= 0.0 || p[i] <= 0.0 || p[i] >= 1.0 || nn[i]*min(p[i], 1.0 - p[i]) <= INV_MAX) {
                    out[i] = binomial(n[i], p[i], u[i]);
                }
            }
        }
    }
}

avx2! {
    /// Poisson(lam) for each lane of a row, from the uniforms u, into out: as poisson() gives, with the lanes walking in lockstep. w is scratch, at least 6 rows of `stride` >= len(lam) values
    pub fn poissons(lam: &[f64], u: &[f64], out: &mut [f64], w: &mut [f64], stride: usize) {
        let len = lam.len();
        let (u, out) = (&u[..len], &mut out[..len]);
        let [uu, ll, bb, px, cdf, x] = rows::<6>(w, stride, len);
        let mut big = 0;
        let mut need = 0;
        for i in 0..len {
            let l = lam[i];
            let large = !(l <= INV_MAX); // Including NaN
            let walk = (l > 0.0) & !large & (u[i] > 1.0 - l);
            ll[i] = if walk { l } else { 0.0 };
            uu[i] = if walk { u[i] } else { 0.0 };
            bb[i] = l + 10.0*max(l, 0.0).sqrt() + 10.0;
            px[i] = -ll[i];
            big += large as usize;
            need += walk as usize;
        }
        if need > 0 {
            exp_row(px);
            let mut left = start(px, cdf, x, uu, bb);
            let mut k = 0.0;
            while left > 0 {
                k += 1.0;
                left = step_poisson(px, cdf, x, uu, ll, bb, k, 1.0/k);
            }
        } else {
            x.fill(0.0);
        }
        out.copy_from_slice(x);
        if big > 0 { // Means above INV_MAX (PTRS), or NaN
            for i in 0..len {
                if !(lam[i] <= INV_MAX) {
                    out[i] = poisson(lam[i], u[i]);
                }
            }
        }
    }
}


//%% The other distributions: one uniform each

/// The inverse of the standard normal CDF at u in (0, 1): Wichura's AS241 (PPND16), to about 1e-16 relative
#[inline(always)]
pub fn ndtri(u: f64) -> f64 {
    let q = u - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q*q;
        return q*(((((((2509.0809287301226727*r + 33430.575583588128105)*r + 67265.770927008700853)*r + 45921.953931549871457)*r
                     + 13731.693765509461125)*r + 1971.5909503065514427)*r + 133.14166789178437745)*r + 3.387132872796366608)
               /(((((((5226.495278852545925*r + 28729.085735721942674)*r + 39307.89580009271061)*r + 21213.794301586595867)*r
                   + 5394.1960214247511077)*r + 687.1870074920579083)*r + 42.313330701600911252)*r + 1.0);
    }
    let mut r = (-log(min(u, 1.0 - u))).sqrt();
    let x;
    if r <= 5.0 {
        r -= 1.6;
        x = (((((((7.7454501427834140764e-4*r + 0.0227238449892691845833)*r + 0.24178072517745061177)*r + 1.27045825245236838258)*r
                + 3.64784832476320460504)*r + 5.7694972214606914055)*r + 4.6303378461565452959)*r + 1.42343711074968357734)
            /(((((((1.05075007164441684324e-9*r + 5.475938084995344946e-4)*r + 0.0151986665636164571966)*r + 0.14810397642748007459)*r
                + 0.68976733498510000455)*r + 1.6763848301838038494)*r + 2.05319162663775882187)*r + 1.0);
    } else {
        r -= 5.0;
        x = (((((((2.01033439929228813265e-7*r + 2.71155556874348757815e-5)*r + 0.0012426609473880784386)*r + 0.026532189526576123093)*r
                + 0.29656057182850489123)*r + 1.7848265399172913358)*r + 5.4637849111641143699)*r + 6.6579046435011037772)
            /(((((((2.04426310338993978564e-15*r + 1.4215117583164458887e-7)*r + 1.8463183175100546818e-5)*r + 7.868691311456132591e-4)*r
                + 0.0148753612908506148525)*r + 0.13692988092273580531)*r + 0.59983220655588793769)*r + 1.0);
    }
    if q < 0.0 { -x } else { x }
}

/// Exponential with a scale
#[inline(always)]
pub fn expon(scale: f64, u: f64) -> f64 {
    -scale*log1p(-u)
}

/// Weibull with shape c and a scale (as SciPy's weibull_min)
#[inline(always)]
pub fn weibull(c: f64, scale: f64, u: f64) -> f64 {
    scale*exp(log(-log1p(-u))/c)
}

/// Gamma(a) with unit scale: Marsaglia and Tsang (2000), with a normal by inversion; for a < 1, Gamma(a+1)*U^(1/a), in logs. Returns the draw and the stream's state (`_gamma()`)
pub fn gamma(a: f64, mut st: u64) -> (f64, u64) {
    let small = a < 1.0;
    let d = (if small { a + 1.0 } else { a }) - 1.0/3.0;
    let c = 1.0/(9.0*d).sqrt();
    let mut v;
    loop {
        let mut u;
        (u, st) = next_u(st);
        let x = ndtri(u);
        v = 1.0 + c*x;
        if v <= 0.0 {
            continue;
        }
        v = v*v*v;
        (u, st) = next_u(st);
        let x2 = x*x;
        if u < 1.0 - 0.0331*x2*x2 || log(u) < 0.5*x2 + d*(1.0 - v + log(v)) {
            break;
        }
    }
    if small {
        let u;
        (u, st) = next_u(st);
        return (exp(log(d*v) + log(u)/a), st);
    }
    (d*v, st)
}

/// Negative binomial: failures before the n-th success, each with chance p (as NumPy's and SciPy's). By inversion
/// for a mean up to INV_MAX (monotone in u), walking until the mass left is below about 1e-20; else as a Poisson
/// with a gamma mean, from a stream seeded by u
pub fn nbinom(n: f64, p: f64, u: f64) -> f64 {
    if p >= 1.0 || n <= 0.0 {
        return 0.0;
    }
    let mean = n*(1.0 - p)/p;
    if mean > INV_MAX {
        let (g, st) = gamma(n, stream(u));
        let lam = g*(1.0 - p)/p;
        let (v, _) = next_u(st);
        return poisson(lam, v);
    }
    let mut px = exp(n*log(p));
    if u <= px {
        return 0.0;
    }
    let mut cdf = px;
    let mut x = 0.0;
    while u > cdf && !(x > mean && px < 1e-20*p) {
        x += 1.0;
        px *= (x + n - 1.0)*(1.0 - p)*(1.0/x);
        cdf += px;
    }
    x
}


//%% Many draws at once, with parameters that are numbers or arrays (one per draw)

/// A parameter of many draws: one number for all of them, or one value per draw (`at()`)
pub trait At: Copy {
    /// The parameter for n draws (an array cut to n, so its reads need no bounds checks)
    fn cut(self, n: usize) -> Self;
    /// Its value for draw i
    fn at(self, i: usize) -> f64;
}

impl At for f64 {
    #[inline(always)]
    fn cut(self, _: usize) -> Self { self }
    #[inline(always)]
    fn at(self, _: usize) -> f64 { self }
}

impl At for &[f64] {
    #[inline(always)]
    fn cut(self, n: usize) -> Self { &self[..n] }
    #[inline(always)]
    fn at(self, i: usize) -> f64 { self[i] }
}

/// The loop of the many_*() draws: out[i] = f(each parameter's value for draw i, u[i]) (as `arrays()`); the draw is called
/// by name, not through a closure, so that it is inlined and the loop vectorized
macro_rules! draw_each {
    ($f:path, $u:ident, $out:ident, $($p:ident),+) => {{
        let len = $out.len();
        $(let $p = $p.cut(len);)+
        let $u = &$u[..len];
        for i in 0..len {
            $out[i] = $f($($p.at(i)),+, $u[i]);
        }
    }};
}

avx2! {
    /// Exponentials with a scale, from the uniforms u, into out (`many.expon()`)
    pub fn many_expon<A: At>(scale: A, u: &[f64], out: &mut [f64]) { draw_each!(expon, u, out, scale) }
}

avx2! {
    /// Weibulls with shape c and a scale (`many.weibull()`)
    pub fn many_weibull<A: At, B: At>(c: A, scale: B, u: &[f64], out: &mut [f64]) { draw_each!(weibull, u, out, c, scale) }
}

/// Binomials (`many.binomial()`)
pub fn many_binomial<A: At, B: At>(n: A, p: B, u: &[f64], out: &mut [f64]) { draw_each!(binomial, u, out, n, p) }

/// Poissons (`many.poisson()`)
pub fn many_poisson<A: At>(lam: A, u: &[f64], out: &mut [f64]) { draw_each!(poisson, u, out, lam) }

/// Negative binomials (`many.nbinom()`)
pub fn many_nbinom<A: At, B: At>(n: A, p: B, u: &[f64], out: &mut [f64]) { draw_each!(nbinom, u, out, n, p) }


/// The bindings: `many` (the draws of `ss.Dist`), and, for checking the draws against Numba's, each scalar draw over arrays of its parameters and uniforms (out[i] = f(a[i], ..., u[i])) and the rows as they are
pub mod bind {
    use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use crate::draws as d;
    use crate::py::{flat_mut, sl, sl_mut};

    /// A binding that sets out[i] = f(a[i], b[i], ...), with each array of the type given
    macro_rules! over {
        ($name:ident, $f:expr, $O:ty, $($a:ident: $T:ty),+) => {
            #[pyfunction]
            fn $name($($a: &Bound<PyAny>,)+ out: &Bound<PyAny>) -> PyResult<()> {
                $(let $a = sl::<$T>($a)?;)+
                let out = sl_mut::<$O>(out)?;
                for i in 0..out.len() {
                    out[i] = $f($($a[i]),+);
                }
                Ok(())
            }
        };
    }

    over!(draws_mix, d::mix, u64, z: u64);
    over!(draws_stream, d::stream, u64, u: f64);
    over!(draws_exp, d::exp, f64, x: f64);
    over!(draws_log, d::log, f64, x: f64);
    over!(draws_log1p, d::log1p, f64, x: f64);
    over!(draws_expm1, d::expm1, f64, x: f64);
    over!(draws_ndtri, d::ndtri, f64, u: f64);
    over!(draws_expon, d::expon, f64, scale: f64, u: f64);
    over!(draws_weibull, d::weibull, f64, c: f64, scale: f64, u: f64);
    over!(draws_binomial_btpe, d::binomial_btpe, f64, n: f64, p: f64, st: u64);
    over!(draws_poisson_ptrs, d::poisson_ptrs, f64, lam: f64, st: u64);
    over!(draws_binomial_bound, d::binomial_bound, f64, n: f64, q: f64);
    over!(draws_binomial, d::binomial, f64, n: f64, p: f64, u: f64);
    over!(draws_poisson, d::poisson, f64, lam: f64, u: f64);
    over!(draws_nbinom, d::nbinom, f64, n: f64, p: f64, u: f64);

    #[pyfunction]
    fn draws_key(seed: i64, ind: i64) -> i64 {
        d::key(seed, ind) as i64
    }

    #[pyfunction]
    fn draws_crn_bits(itemsize: usize) -> (u64, f64) {
        d::crn_bits(itemsize)
    }

    #[pyfunction]
    fn draws_uniform(k: i64, i: &Bound<PyAny>, out: &Bound<PyAny>) -> PyResult<()> {
        let (i, out) = (sl::<i64>(i)?, sl_mut::<f64>(out)?);
        for j in 0..out.len() {
            out[j] = d::uniform(k as u64, i[j] as u64);
        }
        Ok(())
    }

    #[pyfunction]
    fn draws_uniforms(k: i64, start: i64, out: &Bound<PyAny>) -> PyResult<()> {
        d::uniforms(k as u64, start as u64, sl_mut(out)?);
        Ok(())
    }

    /// The uniform and new state of next_u() for each state
    #[pyfunction]
    fn draws_next_u(st: &Bound<PyAny>, out: &Bound<PyAny>, st_out: &Bound<PyAny>) -> PyResult<()> {
        let (st, out, st_out) = (sl::<u64>(st)?, sl_mut::<f64>(out)?, sl_mut::<u64>(st_out)?);
        for i in 0..out.len() {
            (out[i], st_out[i]) = d::next_u(st[i]);
        }
        Ok(())
    }

    #[pyfunction]
    fn draws_gamma(a: &Bound<PyAny>, st: &Bound<PyAny>, out: &Bound<PyAny>, st_out: &Bound<PyAny>) -> PyResult<()> {
        let (a, st, out, st_out) = (sl::<f64>(a)?, sl::<u64>(st)?, sl_mut::<f64>(out)?, sl_mut::<u64>(st_out)?);
        for i in 0..out.len() {
            (out[i], st_out[i]) = d::gamma(a[i], st[i]);
        }
        Ok(())
    }

    /// The row stride of a 2D scratch array
    fn stride(w: &Bound<PyAny>) -> PyResult<usize> {
        let a = w.cast::<PyArrayDyn<f64>>()?;
        Ok(if a.ndim() == 2 { a.shape()[1] } else { a.len() })
    }

    #[pyfunction]
    fn draws_binomials(n: &Bound<PyAny>, p: &Bound<PyAny>, u: &Bound<PyAny>, out: &Bound<PyAny>, w: &Bound<PyAny>) -> PyResult<()> {
        d::binomials(sl(n)?, sl(p)?, sl(u)?, sl_mut(out)?, flat_mut(w)?, stride(w)?);
        Ok(())
    }

    #[pyfunction]
    fn draws_poissons(lam: &Bound<PyAny>, u: &Bound<PyAny>, out: &Bound<PyAny>, w: &Bound<PyAny>) -> PyResult<()> {
        d::poissons(sl(lam)?, sl(u)?, sl_mut(out)?, flat_mut(w)?, stride(w)?);
        Ok(())
    }

    /// A parameter of `many`: a number, or an array of float64 (one value per draw; an array broadcast from one value, with a stride of 0, is that value)
    enum Par<'a> {
        Num(f64),
        Arr(&'a [f64]),
    }

    fn par<'a>(o: &'a Bound<PyAny>) -> PyResult<Par<'a>> {
        if let Ok(a) = o.cast::<PyArrayDyn<f64>>() {
            if a.ndim() == 0 || (a.ndim() == 1 && a.len() > 0 && a.strides()[0] == 0) {
                return Ok(Par::Num(unsafe { *a.data() }));
            }
            return Ok(Par::Arr(sl::<f64>(o)?));
        }
        if o.cast::<PyUntypedArray>().is_ok() {
            return Err(PyTypeError::new_err("the parameters of the Rust draws must be numbers or float64 arrays"));
        }
        Ok(Par::Num(o.extract::<f64>()?))
    }

    /// many[name](a, u, out) and many[name](a, b, u, out), each parameter a number or an array; returns out
    macro_rules! many {
        ($name:ident, $f:expr, $a:ident) => {
            #[pyfunction]
            fn $name<'py>($a: &Bound<'py, PyAny>, u: &Bound<'py, PyAny>, out: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                let (uu, o) = (sl::<f64>(u)?, sl_mut::<f64>(out)?);
                match par($a)? {
                    Par::Num(x) => $f(x, uu, o),
                    Par::Arr(x) => $f(x, uu, o),
                }
                Ok(out.clone())
            }
        };
        ($name:ident, $f:expr, $a:ident, $b:ident) => {
            #[pyfunction]
            fn $name<'py>($a: &Bound<'py, PyAny>, $b: &Bound<'py, PyAny>, u: &Bound<'py, PyAny>, out: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                let (uu, o) = (sl::<f64>(u)?, sl_mut::<f64>(out)?);
                match (par($a)?, par($b)?) {
                    (Par::Num(x), Par::Num(y)) => $f(x, y, uu, o),
                    (Par::Num(x), Par::Arr(y)) => $f(x, y, uu, o),
                    (Par::Arr(x), Par::Num(y)) => $f(x, y, uu, o),
                    (Par::Arr(x), Par::Arr(y)) => $f(x, y, uu, o),
                }
                Ok(out.clone())
            }
        };
    }

    many!(many_expon, d::many_expon, scale);
    many!(many_weibull, d::many_weibull, c, scale);
    many!(many_binomial, d::many_binomial, n, p);
    many!(many_poisson, d::many_poisson, lam);
    many!(many_nbinom, d::many_nbinom, n, p);

    /// Add these bindings to the module
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(draws_mix, m)?)?;
        m.add_function(wrap_pyfunction!(draws_key, m)?)?;
        m.add_function(wrap_pyfunction!(draws_crn_bits, m)?)?;
        m.add_function(wrap_pyfunction!(draws_uniform, m)?)?;
        m.add_function(wrap_pyfunction!(draws_uniforms, m)?)?;
        m.add_function(wrap_pyfunction!(draws_stream, m)?)?;
        m.add_function(wrap_pyfunction!(draws_next_u, m)?)?;
        m.add_function(wrap_pyfunction!(draws_exp, m)?)?;
        m.add_function(wrap_pyfunction!(draws_log, m)?)?;
        m.add_function(wrap_pyfunction!(draws_log1p, m)?)?;
        m.add_function(wrap_pyfunction!(draws_expm1, m)?)?;
        m.add_function(wrap_pyfunction!(draws_binomial_btpe, m)?)?;
        m.add_function(wrap_pyfunction!(draws_poisson_ptrs, m)?)?;
        m.add_function(wrap_pyfunction!(draws_binomial_bound, m)?)?;
        m.add_function(wrap_pyfunction!(draws_binomial, m)?)?;
        m.add_function(wrap_pyfunction!(draws_poisson, m)?)?;
        m.add_function(wrap_pyfunction!(draws_binomials, m)?)?;
        m.add_function(wrap_pyfunction!(draws_poissons, m)?)?;
        m.add_function(wrap_pyfunction!(draws_ndtri, m)?)?;
        m.add_function(wrap_pyfunction!(draws_expon, m)?)?;
        m.add_function(wrap_pyfunction!(draws_weibull, m)?)?;
        m.add_function(wrap_pyfunction!(draws_gamma, m)?)?;
        m.add_function(wrap_pyfunction!(draws_nbinom, m)?)?;
        m.add_function(wrap_pyfunction!(many_expon, m)?)?;
        m.add_function(wrap_pyfunction!(many_weibull, m)?)?;
        m.add_function(wrap_pyfunction!(many_binomial, m)?)?;
        m.add_function(wrap_pyfunction!(many_poisson, m)?)?;
        m.add_function(wrap_pyfunction!(many_nbinom, m)?)?;
        Ok(())
    }
}
