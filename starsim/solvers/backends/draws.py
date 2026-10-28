"""
Exact random draws in Numba, from one uniform each where possible

Every draw here is a function of a uniform u in (0, 1): by inversion of the CDF where
that is cheap (binomial and Poisson up to a mean of 30, negative binomial, exponential,
Weibull), so the draw is monotone in u, which suits common random numbers; otherwise
(binomial and Poisson above a mean of 30, and the gamma mean of a large negative binomial)
by a rejection sampler (BTPE, PTRS, Marsaglia and Tsang) that takes its own uniforms from a
stream seeded by the bits of u, so the same u still gives the same draw. The uniforms
themselves are counter-based: `uniform(key, i)` is a SplitMix64 hash of a key and an index,
the hash of `ss.distributions.hash_uniforms()` (which calls `key()` and `crn_uniform()`, as
the passes over agents and the kernels do), so a stream does not depend on how the work is
split. Other distributions are drawn by SciPy (see v4_design/oct25/backlog/perf_jax.md).

The tricks that make them cheap (see v4_design/oct21/jax_plan.md):

- Draws already settled cost nothing: n = 0, p = 0 or 1; and u <= 1 - mean is a 0, since
  (1-p)^n >= 1 - np and exp(-lam) >= 1 - lam, so most small draws need no exp or log.
- Rows (`binomials()`, `poissons()`): the draws that aren't settled walk up their pmfs in
  lockstep (a multiply, an add and a compare a step, with the division replaced by a
  reciprocal shared by all lanes), so LLVM vectorizes them. The exp and log they start
  from are written out here (`exp()`, `log()`, `log1p()`, `expm1()`) so they vectorize too,
  with no SVML. A row gives exactly the counts the scalar draws give for the same uniforms.
- Numba vectorizes only with `error_model='numpy'` (its division-by-zero checks block
  LLVM) and with no `fastmath` (which would also let the scalar and row draws differ).

The functions run as plain Python too (in the copy `plain_copy()` makes, used by
`ss.kernels.uncompiled()`), with the same results.
"""

import math
import importlib
import numpy as np
import numba as nb
import sciris as sc
from numba import types
from numba.extending import intrinsic, overload

plain = __name__.endswith('plain_draws') # Whether this is the copy that runs as plain Python: see plain_copy()

def jit(**kwargs):
    """ Compile with Numba (cached on disk), or leave as plain Python in the copy that plain_copy() gives """
    return (lambda func: func) if plain else nb.jit(nopython=True, cache=True, error_model='numpy', **kwargs)

U64 = np.uint64
I64 = np.int64
GOLDEN = U64(0x9e3779b97f4a7c15)
MIX1 = U64(0xbf58476d1ce4e5b9)
MIX2 = U64(0x94d049bb133111eb)
U30, U27, U31, U11, U52 = U64(30), U64(27), U64(31), U64(11), U64(52)
U40 = U64(40) # 64 - 24: the top 24 bits, for a uniform that is a float32 (see crn_uniform())
INV53 = 1.0/9007199254740992.0 # 2^-53
INV24 = 1.0/16777216.0 # 2^-24
TWO53 = 9007199254740992.0
SALT = U64(0x5851f42d4c957f2d) # Seeds a rejection sampler's stream from its uniform: see stream()
INV_MAX = 30.0 # The mean up to which binomial and Poisson draws are by inversion (as NumPy's)
RECIP = 1.0/np.arange(1.0, 129.0) # 1/k, for the walks of the inversions (which take at most about 100 steps), so they multiply instead of divide
LN2_HI = 0.693147180369123816490 # ln 2 in two parts, for an exact range reduction (Cody and Waite)
LN2_LO = 1.90821492927058770002e-10
LOG2E = 1.4426950408889634
SQRT2 = 1.4142135623730951
EXPO = I64(0x7ff0000000000000)
MANT = I64(0x000fffffffffffff)
ONE_BITS = I64(0x3ff0000000000000)

plain_draws = None # The copy that runs as plain Python, once made: see plain_copy()


def plain_copy():
    """ This module as plain Python: the same code run without compiling it, which gives the same draws (for ss.kernels.uncompiled()) """
    global plain_draws
    if plain_draws is None:
        spec = importlib.util.spec_from_file_location('starsim.solvers.backends.plain_draws', __file__)
        plain_draws = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plain_draws)
    return plain_draws


#%% Bits of a float, and parameters that are a number or an array

if plain:
    def as_float(i):
        """ The float64 with these bits """
        return float(np.int64(i).view(np.float64))

    def as_int(x):
        """ The bits of a float64 """
        return int(np.float64(x).view(np.int64))

    def at(x, i):
        """ A parameter's value for draw i: x[i] for an array, x for a number """
        return x[i] if np.ndim(x) else x

else:
    @intrinsic
    def as_float(typingctx, i):
        def codegen(context, builder, signature, args):
            return builder.bitcast(args[0], context.get_value_type(types.float64))
        return types.float64(types.int64), codegen

    @intrinsic
    def as_int(typingctx, x):
        def codegen(context, builder, signature, args):
            return builder.bitcast(args[0], context.get_value_type(types.int64))
        return types.int64(types.float64), codegen

    def at(x, i):
        return x[i] if np.ndim(x) else x

    @overload(at, inline='always')
    def _at(x, i):
        if isinstance(x, types.Array):
            return lambda x, i: x[i]
        return lambda x, i: x


#%% Uniforms

@jit(inline='always')
def mix(z):
    """ The mixing function of SplitMix64, which turns a counter into random bits """
    z = (z ^ (z >> U30))*MIX1
    z = (z ^ (z >> U27))*MIX2
    return z ^ (z >> U31)

@jit()
def key(seed, ind):
    """ The key of a stream of uniforms, from a seed and the index of a call (as `ss.distributions.hash_uniforms()`), as an int64 with the same bits (so Python can pass it back) """
    return I64(mix((U64(seed) ^ (U64(ind)*GOLDEN)) + GOLDEN))

@jit(inline='always')
def uniform(k, i):
    """ Uniform i in (0, 1) of the stream with key k (as `ss.distributions.hash_uniforms()` gives for slot i) """
    return max(mix((U64(i) + U64(k))*GOLDEN) >> U11, 0.5)*INV53

@jit(inline='always')
def crn_uniform(k, slot, shift, scale):
    """ Uniform `slot` in (0, 1) of the stream with key k, from the hash's top bits: (shift, scale) are (11, 2^-53) for a float64, as `uniform()`, or (40, 2^-24) for a float32, so every value is exact at that width (as `ss.distributions.hash_uniforms()` gives for an agent's slot) """
    return max(mix((U64(slot) + U64(k))*GOLDEN) >> shift, 0.5)*scale

def crn_bits(dtype):
    """ How many bits of each hash a uniform of a dtype keeps, as the shift and scale that `crn_uniform()` takes """
    return (U40, INV24) if np.dtype(dtype).itemsize <= 4 else (U11, INV53)

@jit()
def uniforms(k, start, out):
    """ Uniforms start, start+1, ... of the stream with key k """
    for i in range(out.size):
        out[i] = uniform(k, start + i)
    return out

@jit(inline='always')
def stream(u):
    """ The state of a stream of uniforms for a rejection sampler, seeded by the bits of the uniform it stands for """
    return mix(U64(u*TWO53) ^ SALT)

@jit(inline='always')
def next_u(st):
    """ The next uniform of a stream, and its new state """
    st += GOLDEN
    return max(mix(st) >> U11, 0.5)*INV53, st


#%% float64 exp and log that LLVM can vectorize (to within 1-2 ulp of libm's)

@jit(inline='always')
def exp(x):
    """ exp(x), for x in [-708, 709] (clipped to it): x = k*ln2 + r with |r| <= ln2/2, a degree-13 Taylor polynomial for exp(r) (by Estrin's scheme, for a short chain of dependent operations), times 2^k by adding k to its exponent """
    x = min(max(x, -708.0), 709.0)
    k = math.floor(x*LOG2E + 0.5)
    r = (x - k*LN2_HI) - k*LN2_LO
    r2 = r*r
    r4 = r2*r2
    a = (1.0 + r) + (1/2 + r*(1/6))*r2
    b = (1/24 + r*(1/120)) + (1/720 + r*(1/5040))*r2
    c = (1/40320 + r*(1/362880)) + (1/3628800 + r*(1/39916800))*r2
    d = 1/479001600 + r*(1/6227020800)
    p = (a + b*r4) + (c + d*r4)*(r4*r4)
    return as_float(as_int(p) + (I64(k) << I64(52)))

@jit(inline='always')
def log(x):
    """ log(x), for x > 0 and normal: x = m*2^e with m in [sqrt(1/2), sqrt(2)), and log(m) = 2*atanh(s), s = (m-1)/(m+1), as a series to s^21 (by Estrin's scheme) """
    bits = as_int(x)
    e = ((bits & EXPO) >> I64(52)) - I64(1023)
    m = as_float((bits & MANT) | ONE_BITS)
    big = m > SQRT2
    m = m*0.5 if big else m
    e = e + I64(1) if big else e
    s = (m - 1.0)/(m + 1.0)
    z = s*s
    z2 = z*z
    z4 = z2*z2
    a = (1.0 + z*(1/3)) + (1/5 + z*(1/7))*z2
    b = (1/9 + z*(1/11)) + (1/13 + z*(1/15))*z2
    c = (1/17 + z*(1/19)) + z2*(1/21)
    t = (a + b*z4) + c*(z4*z4) # By Estrin's scheme
    return (e*LN2_HI + 2.0*s*t) + e*LN2_LO

@jit(inline='always')
def log1p(x):
    """ log(1 + x), for x > -1, accurate for small x too (Goldberg's correction) """
    w = 1.0 + x
    d = w - 1.0
    return x if d == 0.0 else log(w)*(x/d)

@jit(inline='always')
def expm1(x):
    """ exp(x) - 1, for x <= 709, accurate for small x too: a Taylor series near 0 """
    x2 = x*x
    x4 = x2*x2
    a = (1.0 + x*(1/2)) + (1/6 + x*(1/24))*x2
    b = (1/120 + x*(1/720)) + (1/5040 + x*(1/40320))*x2
    c = (1/362880 + x*(1/3628800)) + (1/39916800 + x*(1/479001600))*x2
    small = x*((a + b*x4) + (c + x4*(1/6227020800))*(x4*x4))
    return small if abs(x) < 0.35 else exp(x) - 1.0


#%% Binomial and Poisson

@jit()
def binomial_btpe(n, p, st):
    """ Binomial(n, p) for p <= 1/2 and n*p > 30: BTPE (Kachitvichyanukul and Schmeiser 1988), as NumPy's, from the stream st """
    r = p
    q = 1.0 - r
    fm = n*r + r
    m = int(math.floor(fm))
    p1 = math.floor(2.195*math.sqrt(n*r*q) - 4.6*q) + 0.5
    xm = m + 0.5
    xl = xm - p1
    xr = xm + p1
    c = 0.134 + 20.5/(15.3 + m)
    a = (fm - xl)/(fm - xl*r)
    laml = a*(1.0 + a/2.0)
    a = (xr - fm)/(xr*q)
    lamr = a*(1.0 + a/2.0)
    p2 = p1*(1.0 + 2.0*c)
    p3 = p2 + c/laml
    p4 = p3 + c/lamr
    nrq = n*r*q
    while True:
        u, st = next_u(st)
        u *= p4
        v, st = next_u(st)
        if u <= p1:
            return float(math.floor(xm - p1*v + u))
        if u <= p2:
            x = xl + (u - p1)/c
            v = v*c + 1.0 - abs(m - x + 0.5)/p1
            if v > 1.0:
                continue
            y = int(math.floor(x))
        elif u <= p3:
            y = int(math.floor(xl + math.log(v)/laml))
            if y < 0:
                continue
            v = v*(u - p2)*laml
        else:
            y = int(math.floor(xr - math.log(v)/lamr))
            if y > n:
                continue
            v = v*(u - p3)*lamr
        k = abs(y - m)
        if k <= 20 or k >= nrq/2.0 - 1: # Evaluate the pmf ratio directly
            s = r/q
            a = s*(n + 1)
            F = 1.0
            if m < y:
                for i in range(m + 1, y + 1):
                    F *= (a/i - s)
            elif m > y:
                for i in range(y + 1, m + 1):
                    F /= (a/i - s)
            if v <= F:
                return float(y)
            continue
        rho = (k/nrq)*((k*(k/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5) # Squeeze, then Stirling's bound
        t = -k*k/(2*nrq)
        A = math.log(v)
        if A < t - rho:
            return float(y)
        if A > t + rho:
            continue
        x1 = y + 1.0
        f1 = m + 1.0
        z = n + 1.0 - m
        w = n - y + 1.0
        x2, f2, z2, w2 = x1*x1, f1*f1, z*z, w*w
        bound = (xm*math.log(f1/x1) + (n - m + 0.5)*math.log(z/w) + (y - m)*math.log(w*r/(x1*q))
                 + (13860. - (462. - (132. - (99. - 140./f2)/f2)/f2)/f2)/f1/166320.
                 + (13860. - (462. - (132. - (99. - 140./z2)/z2)/z2)/z2)/z/166320.
                 - (13860. - (462. - (132. - (99. - 140./x2)/x2)/x2)/x2)/x1/166320.
                 - (13860. - (462. - (132. - (99. - 140./w2)/w2)/w2)/w2)/w/166320.)
        if A <= bound:
            return float(y)

@jit()
def poisson_ptrs(lam, st):
    """ Poisson(lam) for lam > 30: PTRS (Hörmann 1993), as NumPy's, from the stream st """
    slam = math.sqrt(lam)
    loglam = math.log(lam)
    b = 0.931 + 2.53*slam
    a = -0.059 + 0.02483*b
    invalpha = 1.1239 + 1.1328/(b - 3.4)
    vr = 0.9277 - 3.6224/(b - 2)
    while True:
        U, st = next_u(st)
        U -= 0.5
        V, st = next_u(st)
        us = 0.5 - abs(U)
        k = math.floor((2*a/us + b)*U + lam + 0.43)
        if us >= 0.07 and V <= vr:
            return float(k)
        if k < 0 or (us < 0.013 and V > us):
            continue
        if math.log(V) + math.log(invalpha) - math.log(a/(us*us) + b) <= -lam + k*loglam - math.lgamma(k + 1):
            return float(k)

@jit()
def binomial_bound(n, q):
    """ How far the inversion of Binomial(n, q) walks at most (as NumPy's): beyond it the mass is below 1e-20 """
    m = n*q
    return min(n, m + 10.0*math.sqrt(m*(1.0 - q) + 1.0))

@jit()
def binomial(n, p, u):
    """
    Binomial(n, p) from the uniform u (n is rounded down to a whole number): by inversion for a mean up to INV_MAX
    (so monotone in u), with u <= 1 - mean a 0 at once; else by BTPE, from a stream seeded by u. NaN for a mean (of the side drawn,
    n*min(p, 1 - p)) of 2^53 or more (which no model has: it comes from a bug upstream, and BTPE's integers would overflow) or NaN
    """
    n = np.floor(n)
    if n <= 0.0 or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return n
    flip = p > 0.5 # Draw on the side of 1/2 with the smaller mean
    q = 1.0 - p if flip else p
    m = n*q
    if not m <= INV_MAX:
        if not m < TWO53: # Including NaN
            return np.nan
        y = binomial_btpe(n, q, stream(u))
        return n - y if flip else y
    if flip:
        u = 1.0 - u
    if u <= 1.0 - m: # A 0, with no exp or log
        return n if flip else 0.0
    px = exp(n*log1p(-q))
    r = q/(1.0 - q)
    bound = binomial_bound(n, q)
    cdf = px
    X = 0.0
    while u > cdf and X < bound: # As binomials() does it, so they give the same counts
        X += 1.0
        px *= (n - X + 1.0)*r*RECIP[int(X) - 1]
        cdf += px
    return n - X if flip else X

@jit()
def poisson(lam, u):
    """ Poisson(lam) from the uniform u: by inversion for a mean up to INV_MAX (monotone in u), with u <= 1 - lam a 0 at once; else by PTRS, from a stream seeded by u; NaN for a mean of 2^53 or more, or NaN (as binomial()) """
    if lam <= 0.0:
        return 0.0
    if not lam <= INV_MAX:
        return poisson_ptrs(lam, stream(u)) if lam < TWO53 else np.nan
    if u <= 1.0 - lam:
        return 0.0
    px = exp(-lam)
    bound = lam + 10.0*math.sqrt(lam) + 10.0
    cdf = px
    X = 0.0
    while u > cdf and X < bound:
        X += 1.0
        px *= lam*RECIP[int(X) - 1]
        cdf += px
    return X


#%% Rows of binomial and Poisson draws, vectorized

@jit()
def _prep_binomial(n, p, u, uu, qq, flip, nn):
    """
    For every lane, branch-free: n rounded down, the side of 1/2 to draw on, and its uniform and chance. Lanes the walk
    shouldn't take (settled at once, or means above INV_MAX or NaN) get a uniform of 0, so they stay at 0. Returns the
    number of lanes left to walk, and of those with a mean above INV_MAX or NaN.
    """
    big = 0
    need = 0
    for i in range(n.size):
        ni = np.floor(n[i])
        f = p[i] > 0.5
        qi = 1.0 - p[i] if f else p[i]
        ui = 1.0 - u[i] if f else u[i]
        m = ni*qi
        large = not m <= INV_MAX # Including NaN
        walk = (ni > 0.0) & (qi > 0.0) & (not large) & (ui > 1.0 - m)
        nn[i] = ni
        qq[i] = qi if walk else 0.0
        uu[i] = ui if walk else 0.0
        flip[i] = f
        big += large & (not ni <= 0.0) & (not qi <= 0.0)
        need += walk
    return need, big

@jit()
def _prep_walk(nn, qq, rr, bb, px):
    """ For the lanes that walk: the ratio q/(1-q), the bound of the walk, and the chance of 0, (1-q)^n, as binomial() has them (the others, with q = 0, as if n were 0, so a NaN n there is never exp()'s, which plain Python can't take) """
    for i in range(nn.size):
        q = qq[i]
        n = nn[i] if q > 0.0 else 0.0
        m = n*q
        rr[i] = q/(1.0 - q)
        bb[i] = min(n, m + 10.0*math.sqrt(m*(1.0 - q) + 1.0))
        px[i] = exp(n*log1p(-q))

@jit()
def _exp_row(x, out):
    for i in range(x.size):
        out[i] = exp(x[i])

@jit()
def _start(px, cdf, X, uu, bb):
    left = 0
    for j in range(px.size):
        cdf[j] = px[j]
        X[j] = 0.0
        left += (uu[j] > cdf[j]) & (bb[j] > 0.0)
    return left

@jit()
def _step_binomial(px, cdf, X, uu, nn, rr, bb, k, inv):
    """ One step of the walk for all lanes, simple enough for LLVM to vectorize: as binomial() takes it """
    left = 0
    for j in range(px.size):
        a = (uu[j] > cdf[j]) & (k - 1.0 < bb[j])
        X[j] += a
        px[j] *= (nn[j] - k + 1.0)*rr[j]*inv
        cdf[j] += px[j]
        left += (uu[j] > cdf[j]) & (k < bb[j])
    return left

@jit()
def _step_poisson(px, cdf, X, uu, ll, bb, k, inv):
    left = 0
    for j in range(px.size):
        a = (uu[j] > cdf[j]) & (k - 1.0 < bb[j])
        X[j] += a
        px[j] *= ll[j]*inv
        cdf[j] += px[j]
        left += (uu[j] > cdf[j]) & (k < bb[j])
    return left

@jit()
def binomials(n, p, u, out, w):
    """
    Binomial(n, p) for each lane of a row, from the uniforms u, into out: as binomial() gives, but with every lane
    that isn't settled at once walking up its pmf in lockstep. w is scratch, at least (9, len(n)).
    """
    N = n.size
    uu, qq, rr, bb, flip, px, cdf, X, nn = w[0, :N], w[1, :N], w[2, :N], w[3, :N], w[4, :N], w[5, :N], w[6, :N], w[7, :N], w[8, :N]
    need, big = _prep_binomial(n, p, u, uu, qq, flip, nn)
    if need > 0:
        _prep_walk(nn, qq, rr, bb, px)
        left = _start(px, cdf, X, uu, bb)
        k = 0.0
        while left > 0:
            k += 1.0
            left = _step_binomial(px, cdf, X, uu, nn, rr, bb, k, 1.0/k)
    else:
        X[:] = 0.0
    for i in range(N):
        out[i] = nn[i] - X[i] if flip[i] > 0.0 else X[i]
        if nn[i] <= 0.0 or p[i] <= 0.0:
            out[i] = 0.0
        elif p[i] >= 1.0:
            out[i] = nn[i]
    if big > 0: # Means above INV_MAX (BTPE), or NaN
        for i in range(N):
            if not (nn[i] <= 0.0 or p[i] <= 0.0 or p[i] >= 1.0 or nn[i]*min(p[i], 1.0 - p[i]) <= INV_MAX):
                out[i] = binomial(n[i], p[i], u[i])
    return out

@jit()
def poissons(lam, u, out, w):
    """ Poisson(lam) for each lane of a row, from the uniforms u, into out: as poisson() gives, with the lanes walking in lockstep. w is scratch, at least (6, len(lam)) """
    N = lam.size
    uu, ll, bb, px, cdf, X = w[0, :N], w[1, :N], w[2, :N], w[3, :N], w[4, :N], w[5, :N]
    big = 0
    need = 0
    for i in range(N):
        L = lam[i]
        large = not L <= INV_MAX # Including NaN
        walk = (L > 0.0) & (not large) & (u[i] > 1.0 - L)
        ll[i] = L if walk else 0.0
        uu[i] = u[i] if walk else 0.0
        bb[i] = L + 10.0*math.sqrt(max(L, 0.0)) + 10.0
        px[i] = -ll[i]
        big += large
        need += walk
    if need > 0:
        _exp_row(px, px)
        left = _start(px, cdf, X, uu, bb)
        k = 0.0
        while left > 0:
            k += 1.0
            left = _step_poisson(px, cdf, X, uu, ll, bb, k, 1.0/k)
    else:
        X[:] = 0.0
    for i in range(N):
        out[i] = X[i]
    if big > 0: # Means above INV_MAX (PTRS), or NaN
        for i in range(N):
            if not lam[i] <= INV_MAX:
                out[i] = poisson(lam[i], u[i])
    return out


#%% The other distributions: one uniform each

@jit(inline='always')
def ndtri(u):
    """ The inverse of the standard normal CDF at u in (0, 1): Wichura's AS241 (PPND16), to about 1e-16 relative """
    q = u - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q*q
        return q*(((((((2509.0809287301226727*r + 33430.575583588128105)*r + 67265.770927008700853)*r + 45921.953931549871457)*r
                     + 13731.693765509461125)*r + 1971.5909503065514427)*r + 133.14166789178437745)*r + 3.387132872796366608) \
               /(((((((5226.495278852545925*r + 28729.085735721942674)*r + 39307.89580009271061)*r + 21213.794301586595867)*r
                   + 5394.1960214247511077)*r + 687.1870074920579083)*r + 42.313330701600911252)*r + 1.0)
    r = math.sqrt(-log(min(u, 1.0 - u)))
    if r <= 5.0:
        r -= 1.6
        x = (((((((7.7454501427834140764e-4*r + 0.0227238449892691845833)*r + 0.24178072517745061177)*r + 1.27045825245236838258)*r
                + 3.64784832476320460504)*r + 5.7694972214606914055)*r + 4.6303378461565452959)*r + 1.42343711074968357734) \
            /(((((((1.05075007164441684324e-9*r + 5.475938084995344946e-4)*r + 0.0151986665636164571966)*r + 0.14810397642748007459)*r
                + 0.68976733498510000455)*r + 1.6763848301838038494)*r + 2.05319162663775882187)*r + 1.0)
    else:
        r -= 5.0
        x = (((((((2.01033439929228813265e-7*r + 2.71155556874348757815e-5)*r + 0.0012426609473880784386)*r + 0.026532189526576123093)*r
                + 0.29656057182850489123)*r + 1.7848265399172913358)*r + 5.4637849111641143699)*r + 6.6579046435011037772) \
            /(((((((2.04426310338993978564e-15*r + 1.4215117583164458887e-7)*r + 1.8463183175100546818e-5)*r + 7.868691311456132591e-4)*r
                + 0.0148753612908506148525)*r + 0.13692988092273580531)*r + 0.59983220655588793769)*r + 1.0)
    return -x if q < 0.0 else x

@jit(inline='always')
def expon(scale, u):
    return -scale*log1p(-u)

@jit(inline='always')
def weibull(c, scale, u):
    """ Weibull with shape c and a scale (as SciPy's weibull_min) """
    return scale*exp(log(-log1p(-u))/c)

@jit()
def _gamma(a, st):
    """ Gamma(a) with unit scale: Marsaglia and Tsang (2000), with a normal by inversion; for a < 1, Gamma(a+1)*U^(1/a), in logs. Returns the draw and the stream's state """
    small = a < 1.0
    d = (a + 1.0 if small else a) - 1/3
    c = 1.0/math.sqrt(9.0*d)
    while True:
        u, st = next_u(st)
        x = ndtri(u)
        v = 1.0 + c*x
        if v <= 0.0:
            continue
        v = v*v*v
        u, st = next_u(st)
        x2 = x*x
        if u < 1.0 - 0.0331*x2*x2 or log(u) < 0.5*x2 + d*(1.0 - v + log(v)):
            break
    if small:
        u, st = next_u(st)
        return exp(log(d*v) + log(u)/a), st
    return d*v, st

@jit()
def nbinom(n, p, u):
    """
    Negative binomial: failures before the n-th success, each with chance p (as NumPy's and SciPy's). By inversion
    for a mean up to INV_MAX (monotone in u), walking until the mass left is below about 1e-20; else as a Poisson
    with a gamma mean, from a stream seeded by u
    """
    if p >= 1.0 or n <= 0.0:
        return 0.0
    mean = n*(1.0 - p)/p
    if mean > INV_MAX:
        g, st = _gamma(n, stream(u))
        lam = g*(1.0 - p)/p
        v, st = next_u(st)
        return poisson(lam, v)
    px = exp(n*log(p))
    if u <= px:
        return 0.0
    cdf = px
    X = 0.0
    while u > cdf and not (X > mean and px < 1e-20*p):
        X += 1.0
        px *= (X + n - 1.0)*(1.0 - p)*(1.0/X)
        cdf += px
    return X

#%% Many draws at once, with parameters that are numbers or arrays (one per draw)

def arrays(f, npars):
    """ For a draw f(*pars, u) with one or two parameters: from_u(*pars, u, out), which sets out[i] = f(*pars, u[i]); each parameter is a number or an array (one value per draw) """
    if npars == 1:
        @jit()
        def from_u(a, u, out):
            for i in range(out.size):
                out[i] = f(at(a, i), u[i])
            return out
    else:
        @jit()
        def from_u(a, b, u, out):
            for i in range(out.size):
                out[i] = f(at(a, i), at(b, i), u[i])
            return out
    return from_u

many = sc.objdict({name: arrays(f, n) for name, f, n in [ # E.g. many.weibull(c, scale, u, out)
    ('expon', expon, 1), ('weibull', weibull, 2), ('binomial', binomial, 2), ('poisson', poisson, 1), ('nbinom', nbinom, 2)]})
