# `distributions.py` bug audit

Audit of `starsim/distributions.py` (all 1948 lines: hashing, `Dists`, `Dist` and every concrete distribution, `multi_random`, and the exceptions) for real, unambiguous bugs, on branch `rc3.6.2` at HEAD `3d8dc9d5` with Cliff's uncommitted working-tree changes as they were (Starsim 3.6.1, numpy 2.4.6, `ss.dtypes.float = float32`, `ss.options.crn = True` by default). Method: line-by-line reading, then a repro script for each candidate, run against the editable install. Scripts are in the session scratchpad under `distributions/`. Every "actual" value below comes from those runs. Style, docs, performance, and contrived edge cases are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Dist.rvs()` → `Dist.ppf()` (normal, lognorm_ex, expon, gamma, weibull, ...) | Drawing by UID with a zero scale (`std=0`, `scale=0`) returns NaN; the same dist drawn by count returns the constant | 1084, 1033 |
| 2 | High | `randint.ppf()` | Drawing by UID has only 24 bits of resolution, so wide-range integers come out on a coarse lattice. As a result `ErdosRenyiNet` edges are strongly correlated (transitivity 0.50 instead of 0.02) | 1546 |
| 3 | Medium | `randint.ppf()` | Drawing by UID with negative `low` truncates toward zero: `low` is never drawn and `0` comes up twice as often | 1547 |
| 4 | Medium | `Dist.convert_callable()` | The function signature is cached once per Dist, so a second callable parameter with a different signature is called with the wrong arguments | 942-956 |
| 5 | Medium | `Dist.convert_timepars()` | The automatic predraw conversion for `ss.years(ss.poisson(3))` is never applied to an actual draw | 808-810 |
| 6 | Medium | `hash_uniforms()` / `_hash_uniforms_fill()` | The CRN uniform can be exactly `0.0` (about 1 in 16.7M draws), so `ss.normal` returns `-inf` | 61, 88 |
| 7 | Medium | `beta_mean.__init__()` | `force=True` never produces a valid distribution: it either raises or gives `a=b=inf` and NaN draws | 1463-1485 |
| 8 | Medium | `rand_raw` | `rvs(uids)` crashes under the default CRN, because the hash path calls a `ppf` that doesn't exist | 1083-1084, 1033 |
| 9 | Low | `Dist.rvs()` | `debug=True` crashes on any UID draw under CRN (`if self._slots` on an array) | 1131 |
| 10 | Low | `Dist.rvs()` / `Dist.reset()` | Under `ss.options.crn=False`, `reset(-1)` rewinds to the initial state rather than the previous call. The docstring example also raises | 1072, 544 |
| 11 | Low | `Dist.process_seed()` | Re-initialising adds the offset again, so the seed changes (e.g. with `rand_seed=0` or a named standalone dist) | 697 |
| 12 | Low | `Dists.init()` | `Dists(base_seed=...)` from the constructor is stored but never used | 178 |
| 13 | Low | `multi_random.rvs()` | `crn=True` is silently ignored when `ss.options.crn=False` | 1901 |

## High severity

### 1. Drawing by UID with a zero scale returns NaN — `distributions.py:1084`

When you draw by UID (the path every module uses), `Dist.rvs()` routes all distributions through `self.ppf(rands)`. For SciPy-backed distributions that means `self.dist.ppf`. SciPy defines `scale=0` (or lognormal `s=0`) as an invalid parameter and silently returns NaN. The native NumPy path used for `rvs(n)` accepts a zero scale and returns the degenerate constant. So the same distribution gives the right answer by count and NaN by UID. `std=0` / `scale=0` is a common way to switch off variability, e.g. for a fixed duration or a sensitivity run. It can also arise per agent when a parameter is array-valued or callable.

```python
u = ss.uids(np.arange(4))
for d in [ss.normal(loc=3, scale=0, mock=10), ss.lognorm_ex(mean=5, std=0, mock=10), ss.expon(scale=0, mock=10), ss.gamma(a=2, scale=0, mock=10)]:
    print(d, d.rvs(4), d.rvs(u))

sir = ss.SIR(dur_inf=ss.lognorm_ex(mean=ss.years(2), std=std), beta=0.1)   # std = years(0.01) vs years(0)
sim = ss.Sim(n_agents=2000, diseases=sir, networks='random', start=2000, stop=2020).run()
```

Actual:
```
ss.normal(loc=3, scale=0)                rvs(4) -> [3. 3. 3. 3.] | rvs(uids) -> [nan nan nan nan]
ss.lognorm_ex(mean=5, std=0)             rvs(4) -> [5. 5. 5. 5.] | rvs(uids) -> [nan nan nan nan]
ss.expon(scale=0)                        rvs(4) -> [0. 0. 0. 0.] | rvs(uids) -> [nan nan nan nan]
ss.gamma(a=2, loc=0.0, scale=0)          rvs(4) -> [0. 0. 0. 0.] | rvs(uids) -> [nan nan nan nan]
std=0.01: n_infected at end=11.0, n_recovered at end=1132.0
std=0: n_infected at end=2000.0, n_recovered at end=0.0
```

Expected: the by-UID draw should equal the degenerate value (3, 5, 0, 0), and the SIR with `std=0` should behave like `std=0.01`. Instead the NaN duration means nobody ever recovers, and the whole population stays infected. There is no warning.

Blast radius: any model that sets a zero spread on a duration or other loc/scale distribution drawn by UID, which covers essentially all disease durations. This is silent and produces grossly wrong epidemics.

**Fix**: in `Dist.ppf()` (or in overrides for the loc/scale families), handle zero scale explicitly. For example, compute `loc + scale*sps.norm.ppf(rands)` for `normal` and `scale*sps.expon.ppf(rands)` for `expon`, which gives `loc` when `scale=0`. Alternatively, post-process with `np.where(scale == 0, degenerate_value, rvs)`. For `lognorm_ex`/`lognorm_im`, `s=0` should map to `scale` (i.e. `exp(mean_im)`).

### 2. `randint` drawn by UID has only 24 bits of resolution, which breaks `ErdosRenyiNet` — `distributions.py:1546`

`randint.ppf()` computes `rands * (high - low) + low`, where `rands` comes from `hash_uniforms()` at `ss.dtypes.float` = float32. That gives only 2^24 distinct values, so for any range wider than 2^24 the output is a lattice with the low bits zeroed. The default `ss.randint()` (`high = 2^31-1`) only returns multiples of 128. The uint64 randint used by `ErdosRenyiNet` (working-tree `theoretical.py:46`) returns values whose low 40 bits are all zero. `combine_rands(a, b) = xor(a*b, a-b)` then degenerates: `a*b ≡ 0 (mod 2^64)`, so the edge value is just `(a-b) mod 2^64`. That is additive (`r_ij + r_jk ≡ r_ik`), so the edges are far from independent.

```python
d = ss.randint(mock=n);  v = d.rvs(ss.uids(np.arange(n)))           # n = 100_000
d = ss.randint(low=0, high=np.iinfo('uint64').max, dtype=np.uint64, mock=n); v = d.rvs(uids)

# ErdosRenyiNet (n=1000, p=0.02), triangles/transitivity via networkx
sim = ss.Sim(n_agents=1000, networks=ssl.ErdosRenyiNet(p=0.02), dur=1); sim.init()
```

Actual:
```
default randint: dtype int32 high 2147483647 frac multiple of 128: 1.0 n unique 99708
uint64 randint: frac with low 32 bits zero: 1.0
uint64 randint native: frac with low 32 bits zero: 0.0
crn=True: edges=9915 (expected 9990), triangles=32434 (expected 1329), clustering=0.4975 (expected 0.02)
crn=False: edges=10067 (expected 9990), triangles=1361 (expected 1329), clustering=0.0202 (expected 0.02)
```

Expected: uniform integers over the full range, as the native `rng.integers` path gives. For an Erdős–Rényi graph, transitivity should be about p = 0.02. Under default settings it is 0.50, with 24x the expected number of triangles. That is a completely different network structure. The same thing happens with HEAD's int64 min..max randint.

Blast radius: every `ErdosRenyiNet` run with CRN on (the default). It also affects anyone drawing wide-range integers by UID, e.g. to derive per-agent seeds or keys.

**Fix**: for randint's by-UID path, build the integer from enough random bits for the range rather than a float32 uniform. For example, have `hash_uniforms` (or a sibling `hash_uint64`) return the raw 64-bit hash `z` and use `low + z % (high - low)` (or Lemire's multiply-shift). At minimum, use float64 uniforms (53 bits) when `high - low > 2**24`. For `ErdosRenyiNet` in particular, `ss.rand_raw` would be the natural source, but see finding 8.

## Medium severity

### 3. `randint` drawn by UID with negative `low` is biased (truncation toward zero) — `distributions.py:1547`

`.astype(p.dtype)` truncates toward zero instead of flooring. For `low < 0`, values in `[low, low+1)` become `low+1`, and values in `(-1, 0)` become `0`. So `low` is (almost) never drawn, and `0` has double probability.

```python
d = ss.randint(-5, 5, mock=100_000)
d.rvs(ss.uids(np.arange(100_000)))   # vs d.rvs(100_000)
```

Actual:
```
CRN randint(-5,5) counts: {-4: 9981, -3: 10108, -2: 9925, -1: 9954, 0: 20102, 1: 9933, 2: 10143, 3: 9935, 4: 9919}
int-n randint(-5,5) counts: {-5: 10168, -4: 10039, -3: 9976, -2: 9879, -1: 10016, 0: 9765, 1: 9994, 2: 10087, 3: 10009, 4: 10067}
```

Expected: about 10,000 each of -5..4, as the native path gives.

Blast radius: any `ss.randint` with a negative lower bound drawn by UID, e.g. symmetric jitter or offsets. Results are silently wrong.

**Fix**: use `np.floor(rands * (high - low)) + low` (or integer arithmetic as in finding 2) before the clamp and cast.

### 4. Cached callable signature is reused for every callable parameter — `distributions.py:942-956`

`convert_callable()` inspects `func`'s signature only when `self._callable_keys` is unset, and stores it per Dist, not per parameter. If a Dist has two callable parameters with different signatures, the second is called with the first one's argument list. This either crashes or, if the arity happens to match, silently passes the wrong objects.

```python
def loc_func(self, sim, uids): return np.full(len(uids), 100.0)
def scale_func(uids): return np.full(len(uids), 1.0)
ss.normal(loc=loc_func, scale=scale_func).mock().rvs(ss.uids(np.arange(5)))

def loc2(uids): ...
def scale2(module, sim, uids): ...
ss.normal(loc=loc2, scale=scale2).mock().rvs(...)
```

Actual:
```
ERROR: TypeError scale_func() takes 1 positional argument but 3 were given
ERROR: TypeError scale2() missing 2 required positional arguments: 'sim' and 'uids'
```

Expected: each function is called with its own declared arguments.

Blast radius: users with more than one dynamic parameter in one distribution, e.g. age-dependent mean and a simple `uids`-only std. When the signatures have the same arity but different order (e.g. `(sim, uids)` and `(uids, sim)`), the arguments are silently swapped.

**Fix**: cache the keys per parameter, e.g. `self._callable_keys = {parkey: keys}`. The shared `mapping` can stay as is.

### 5. The auto-converted predraw unit for `ss.years(ss.poisson(...))` is never used — `distributions.py:808-810`

When a distribution-level unit is given to a predraw-only distribution with one parameter, `convert_timepars()` converts `self._pars[0]` (the per-call scratch copy) to a timepar and then sets `self.unit = None`. It also warns that the input "has been automatically converted to predrawn scaling". This runs once at `init()`. But every `rvs()` call rebuilds `_pars` from the untouched `self.pars`, and `unit` is now `None`, so the conversion is never applied to a real draw.

```python
for dt in [ss.years(1), ss.days(1)]:
    ss.years(ss.poisson(3)).mock(dt=dt).rvs(ss.uids(np.arange(50))).mean()
    ss.poisson(ss.years(3)).mock(dt=dt).rvs(ss.uids(np.arange(50))).mean()   # what the warning says it is equivalent to
```

Actual:
```
ss.peryear? years(poisson(3)) dt=1 mean: 2.98
  poisson(ss.years(3)) dt=1 mean: 2.98
ss.peryear? years(poisson(3)) dt=1 mean: 2.98
  poisson(ss.years(3)) dt=1 mean: 1094.9
```

Expected: with `dt=days(1)`, `ss.years(ss.poisson(3))` should match `ss.poisson(ss.years(3))` (about 1095), as the warning claims. Instead it stays at 3 per timestep, whatever `dt` is.

Blast radius: anyone who writes `ss.years(ss.poisson(x))` (or `unit=` on any single-parameter predraw dist) with a `dt` other than one unit. The results are dt-dependent and wrong, even though a warning says they were fixed.

**Fix**: persist the conversion to `self.pars` (e.g. `k = self.pars.keys()[0]; self.pars[k] = self.unit(self.pars[k])`) before setting `self.unit = None`, then re-copy `_pars`.

### 6. The CRN uniform can be exactly 0, giving `-inf` from `ss.normal` — `distributions.py:61, 88`

For float32, `_hash_uniforms_fill` returns `(z >> 40) * 2^-24`, which is exactly `0.0` whenever the top 24 hash bits are zero, i.e. with probability 2^-24 per draw. `ppf(0)` is `-inf` for distributions that are unbounded below (normal), and `0` for lognormal and beta. A sim draws tens of millions of values, so this happens in ordinary runs.

```python
d = ss.normal(loc=10, scale=2, mock=10_000); uids = ss.uids(np.arange(10_000))
for i in range(3000): v = d.rvs(uids)   # count infinities
```

Actual:
```
30,000,000 draws of ss.normal(10, 2) by UID: 3 infinite values; first at call 1756, array([-inf])
ss.normal(loc=10, scale=2) .rvs(uids) -> [       -inf 11.00471711]
native rng.normal never gives inf: True
```

Expected: finite values. For example, a 10,000-agent sim drawing one normal per agent per day for 10 years gets about two `-inf` values.

Blast radius: models using `ss.normal` (or other unbounded-below dists) by UID. A single `-inf` duration or age can crash later integer conversion or corrupt results. The float32 native `self.rand()` path used when `_use_ppf=True` has the same property.

**Fix**: map to the open interval, e.g. `((z >> shift) + 0.5) * scale`, which stays strictly inside (0, 1) for both widths. Apply the same idea to `Dist.rand()` when its output feeds a ppf.

### 7. `beta_mean(force=True)` never yields a valid distribution — `distributions.py:1463-1485`

The documented `force` option ("scale the parameters to the valid range") clips to the closed bounds. With `var` clipped to `max_var`, `a = ((1-mean)/var - 1/mean)*mean**2` is exactly 0, and SciPy rejects it. With `var` clipped to 0, it divides by zero and gives `a=b=inf`. `max_var` is also computed from the unclipped mean, so clipping the mean gives a negative `max_var`.

```python
ss.beta_mean(mean=0.5, var=0.3, force=True, strict=False)
ss.beta_mean(mean=0.5, var=-0.01, force=True, strict=False).rvs(5)
ss.beta_mean(mean=1.2, var=0.01, force=True, strict=False)
```

Actual:
```
{'mean': 0.5, 'var': 0.3, 'force': True} ERROR: ValueError a <= 0
{'mean': 0.5, 'var': -0.01, 'force': True} pars: {'a': inf, 'b': inf} rvs: [nan nan nan nan nan]
{'mean': 1.2, 'var': 0.01, 'force': True} ERROR: ValueError a <= 0
Clipping the variance from 0.01 to 0 < var < -0.24.
```

Expected: a usable beta distribution with the parameters clipped into the open valid range.

Blast radius: anyone relying on `force=True`, e.g. during calibration sweeps where mean/var pairs fall outside the valid region. The option either crashes or silently produces NaN.

**Fix**: clip the mean first to `[eps, 1-eps]`, then compute `max_var` from the clipped mean, then clip `var` to `[eps, max_var*(1-eps)]` (strictly inside the bounds).

### 8. `rand_raw().rvs(uids)` crashes under default CRN — `distributions.py:1083-1084`

The CRN hash path sends every by-UID draw to `self.ppf()` unless `_use_ppf is False`. `rand_raw` has no `dist` and no `ppf` override, so `Dist.ppf` dereferences `None`. The native path only works for `rvs(n)`. The `combine_rands()` docstring points users to `ss.rand_raw()` as the input source, and a per-agent draw (by UID) is the natural way to use it.

```python
ss.rand_raw().mock().rvs(ss.uids(np.arange(10)))
ss.rand_raw().mock().rvs(3)
```

Actual:
```
rand_raw(uids): ERROR: AttributeError 'NoneType' object has no attribute 'ppf'
rand_raw(n): [11339330433777850381  9324381008891167034  9411679931974332980]
```

Expected: one raw uint64 per UID.

Blast radius: any by-UID use of `ss.rand_raw` with CRN on. This is presumably why `ErdosRenyiNet` uses `randint` instead, which leads to finding 2.

**Fix**: give `rand_raw` a hashed path that returns the raw 64-bit hash per slot (the same splitmix64 output before the shift/scale). Alternatively, set `_use_ppf = False` and gather from `slots.max()+1` native draws.

## Low severity

### 9. `debug=True` crashes on UID draws under CRN — `distributions.py:1131`

`slotstr = ... if self._slots else ...` evaluates the truth value of an ndarray.

```python
self.myrng = ss.random(debug=True)   # inside a module; step() calls self.myrng.rvs(self.sim.people.auids)
```

Actual:
```
Debug: ss.random(dtype=<class 'numpy.float32'>) called on ti=0 with size=3, <no slots>, Σ(rvs)=1.97, |rvs|=0.6558, state 19612→04813
ERROR: ValueError The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

Expected: a debug line. Blast radius: only people debugging randomness, but the documented flag is unusable for the main code path.

**Fix**: `if self._slots is not None`.

### 10. `reset(-1)` rewinds too far under `ss.options.crn=False`; the docstring example raises — `distributions.py:1072, 544`

Under `crn=False`, `rvs()` skips `make_history()`, so `history` holds only the init state, and `reset(-1)` returns to the start rather than to the previous call. `plot_hist()`'s "as if nothing ever happened" relies on this. Separately, the `reset()` docstring example `ss.random(seed=5).init()` raises, because `strict=True` requires a sim.

```python
dist = ss.random(seed=5, strict=False)
r1 = dist(5); r2 = dist(5); dist.reset(-1); r3 = dist(5)
```

Actual:
```
crn=True: r2==r3 True r4==r1 True r3==r1 False
crn=False: r2==r3 False r4==r1 True r3==r1 True
docstring example ss.random(seed=5).init(): ERROR: RuntimeError Distribution ss.random(dtype=<class 'numpy.float32'>) is not fully initialized, ...
```

Expected: `r2 == r3` in both modes. Blast radius: users of `reset(-1)` / `plot_hist()` with the opt-in `crn=False` fast path.

**Fix**: `reset(state=-1)` could raise, or store history lazily, when `crn=False`. At minimum, document the limitation. Change the example to `.init(force=True)` or `strict=False`.

### 11. Re-initialising a Dist changes its seed — `distributions.py:697`

`self.seed = self.offset + (seed or self.seed or 0)`: on a second `init()` where the base seed is `None` or `0`, `self.seed` already includes the offset, so the offset is added twice.

```python
d = ss.random(name='x', seed=5, strict=False); d.init(force=True)
sim = ss.Sim(..., rand_seed=0); sim.init(); sim.dists.init(obj=sim, base_seed=0, force=True)
```

Actual:
```
B. seed after creation 470594483 after init(force=True) 941188961
D. rand_seed=0 re-init: pars_networks_randomnet_n_contacts 472120882 -> 944241764
```

Expected: an unchanged seed. Blast radius: small. A normal `Sim.run()` initialises only once (MultiSim with `rand_seed=0` was verified unaffected), but it breaks reproducibility of manually re-initialised dists.

**Fix**: keep the user seed separately (e.g. `self.user_seed`) and compute `self.offset + (seed if seed is not None else (self.user_seed or 0))`.

### 12. `Dists(base_seed=...)` is ignored — `distributions.py:178`

`Dists.init()` sets `self.base_seed` but then passes the local `base_seed` argument (often `None`) to each `dist.init()`.

```python
ss.Dists(o, base_seed=5).init()   # vs ss.Dists(o).init(base_seed=5)
```

Actual:
```
A. Dists(base_seed=5).init(): seed = 735047973 offset = 735047973
A. Dists().init(base_seed=5): seed = 735047978
```

Expected: both give `offset + 5`. Blast radius: direct users of `ss.Dists`. `Sim` always passes `base_seed` to `init()`, so it is unaffected.

**Fix**: pass `seed=self.base_seed`.

### 13. `multi_random(crn=True)` is ignored when `ss.options.crn=False` — `distributions.py:1901`

The per-instance override only works in one direction. With `crn=True`, `rvs()` calls `dist.rvs(uids)`, but `Dist.process_size()` checks the global `ss.options.crn` and takes the non-CRN path, so draws depend on which UIDs are present.

Actual:
```
options.crn=False, multi_random(crn=True): uid5 same across different uid sets: False
options.crn=True: uid5 same: True
```

Expected: `True` in both cases, since `crn=True` is documented as "whether to use common random numbers". Blast radius: only users who opt a single `multi_random` into CRN while the global fast path is on.

**Fix**: propagate the override to the underlying dists (e.g. a per-Dist `crn` attribute consulted by `process_size()`), or document that only `crn=False` is honoured.

## Verified clean

The following were tested and found correct (or intended):

- The splitmix64 hash is in [0, 1) for both float widths, and the CRN property holds: the same slot gives the same value regardless of the other UIDs.
- `multi_random` XOR-combine gives a uniform output (KS p=0.52) with no detectable 4-cycle correlation for pairwise draws.
- The `lognorm_ex` ex→im conversion is correct, including callable means referencing module states (mean≈2.05 for target 2).
- The poisson fast-ppf CDF cache is correct.
- `bernoulli` rate conversion (`peryear(0.5)` → 0.393 at dt=1y) and `probperyear` handling are correct.
- `choice` ppf with and without `p`, and `replace=False` avoiding the slot path, both work.
- The `histogram` default-bin and appended-edge logic is correct.
- `Dists.init` re-initialises `strict=False` dists inside a sim, so `rand_seed` is still honoured.
- MultiSim runs with `rand_seed=0` are reproducible.
- `uniform` one-argument swapping is intended.
- A user-supplied `seed=` on a dist inside a sim is overridden by `rand_seed`. This appears intended, since the sim seed governs.
- `bool(ss.bernoulli(p=ss.peryear(...)))` raises TypeError, but no code path in Starsim calls it.
- Unnormalised `choice(p=...)` differs between paths, but it is out of contract (the docstring requires `p` to sum to 1).
- Float32 quantisation of very small `bernoulli` probabilities (<1e-6) was noted as a design limitation, not recorded as a bug.
