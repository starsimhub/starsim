# `demographics.py` bug audit

Audit of `starsim/demographics.py` (all 1162 lines: `Births`, `Deaths`, `PregnancyPars`, `Pregnancy`) for real, unambiguous defects only: wrong results on in-contract input, documented arguments that crash or do nothing, silent data corruption, and logic/unit errors. Style, docstrings, error-message wording, performance, and contrived corner cases are out of scope. Audited against the working tree on branch `rc3.6.2` at commit `3d8dc9d5` (Starsim 3.6.1 editable install, numpy 2.4.6). Every "actual" value below was produced by running the repro script against that install; scripts are in the session scratchpad under `demographics/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Deaths.make_p_death()` | Death rates given per month/week/day are treated as per year (`drd.unit == 1` is true for `months(1)`, `days(1)`, ...), so deaths are 12x/52x/365x too low | 259 |
| 2 | Medium | `Pregnancy.process_maternal_deaths()` / `process_prenatal_deaths()` | When a pregnant woman dies, her fetus's death is always counted as a stillbirth, whatever the gestational age | 1054, 1067-1072 |
| 3 | Medium | `Pregnancy.init_post()` | Burn-in starts one step too late (`np.ceil` of a negative number), so births at `ti=0` are about 1/3 of steady state | 643 |
| 4 | Medium | `Pregnancy.step()` / `update_results()` | All burn-in pregnancies are recorded as new pregnancies at `ti=0` (about 10x the steady-state value) | 1004, 1117-1118 |
| 5 | Medium | `Births.update_results()`, `Deaths.finalize()`, `Pregnancy.finalize()` | CBR/CMR are divided by the sim's `dt` instead of the module's, so they are wrong whenever the module has its own `dt` | 171, 322, 1149 |
| 6 | Medium | `Pregnancy.make_embryos()` | Burn-in fetus ages use the sim's `dt` instead of the module's, so fetuses have the wrong age when the module `dt` differs | 946 |
| 7 | Medium | `Pregnancy.finish_step()` | Compares `people.ti_dead` (sim timestep) with the module's `self.ti`, so with a module-specific `dt` most prenatal/maternal deaths are never processed | 1035 |
| 8 | Medium | `Births.get_births()` | `ss.options.crn=False` together with a dataframe `birth_rate` crashes (`float()` of a 1-element 1-D array) | 126 |
| 9 | Medium | `Pregnancy` (`dur_pregnancy`) | The documented `float/dur` form of `dur_pregnancy` crashes, and any non-`ss.choice` distribution crashes when fertility data is a dataframe | 347, 566 |
| 10 | Low | `Pregnancy.update_maternal_deaths()` | With the default burn-in, a sim with fewer timesteps than the burn-in length crashes with `IndexError` | 864 |

## High severity

### 1. Death rates with a non-year unit are silently treated as per-year — `demographics.py:259`

`make_p_death()` checks `if drd.unit == 1:` to mean "the rate is already per year". But `ss.dur.__eq__` against a number compares the raw value, so `ss.months(1) == 1`, `ss.weeks(1) == 1`, and `ss.days(1) == 1` are all `True`. Any `ss.permonth()`, `ss.perweek()`, or `ss.perday()` death rate therefore takes the "already in years" branch, and its value is then wrapped in `ss.peryear()` at line 290. The conversion branch at line 263 (which is correct) is reachable only for units like `months(12)`. `Births` and `Pregnancy` handle the same input correctly, because they always convert via `.to_prob(ss.years(1))`.

```python
import starsim as ss
for rate in [ss.peryear(120), ss.permonth(10), ss.perday(120/365)]:
    sim = ss.Sim(n_agents=5000, demographics=ss.Deaths(death_rate=rate), start=2000, stop=2005, dt=ss.month, verbose=0, rand_seed=1)
    sim.run()
    print(f'{type(rate).__name__}({rate.value:.4g}): total deaths = {sim.results.deaths.new.sum():.0f}')
```

Actual (all three are the same rate, 12% per year):

```
peryear(120): total deaths = 2270
permonth(10): total deaths = 223
perday(0.3288): total deaths = 10
```

Expected: about 2270 in all three cases. Per step, `make_p_death()` returns 0.000833 for `permonth(10)` instead of about 0.00995.

Blast radius: anyone who writes a background death rate in any unit other than per year. This fails silently: the sim runs normally and just has 12-365x too few deaths.

**Fix**: test the unit itself, not its value. For example, `if drd.unit == ss.years(1)` (or compare `drd.unit.years == 1`). Simpler still, always take the conversion path at line 263, which gives the right answer for per-year input too.

## Medium severity

### 2. Fetal deaths caused by maternal death are always classified as stillbirths — `demographics.py:1054`, `1067-1072`

When a pregnant woman dies (from any cause), `finish_step()` → `process_maternal_deaths()` requests the fetus's death and then calls `self.step_die(mother_uids)`, which sets the mother's `gestation` to NaN. People have already run `step_die()` for this step, so the fetus actually dies on the *next* step. That is when `process_prenatal_deaths()` classifies it, using `ga = self.gestation[mother_uids]`. That value is now NaN, `NaN < threshold` is `False`, and every such loss lands in `stillbirths` (and `n_stillbirths` is incremented on the dead mother).

```python
import numpy as np, starsim as ss

class KillEarlyPregnant(ss.Intervention):
    """ At ti=24, kill every woman who is < 10 weeks pregnant """
    def step(self):
        if self.ti == 24:
            preg = self.sim.demographics.pregnancy
            uids = (preg.pregnant & (preg.gestation < 10)).uids
            self.n_killed = len(uids)
            self.sim.people.request_death(uids)

sim = ss.Sim(n_agents=5000, demographics=ss.Pregnancy(fertility_rate=ss.peryear(200)), interventions=KillEarlyPregnant(),
             start=2000, stop=2004, dt=ss.month, verbose=0, rand_seed=1)
sim.run()
r = sim.results.pregnancy
print('mothers killed at <10 weeks GA:', sim.interventions[0].n_killed)
print('miscarriages:', r.miscarriages.sum(), ' stillbirths:', r.stillbirths.sum())
```

Actual: `mothers killed at <10 weeks GA: 62` / `miscarriages: 0.0  stillbirths: 62.0`. Expected: 62 miscarriages and 0 stillbirths. With ordinary `ss.Deaths` plus `ss.Pregnancy`, 126 of 269 fetal deaths over 10 years had a NaN gestational age at classification time.

Blast radius: any sim that has pregnancy plus any source of maternal mortality (background `Deaths`, disease deaths, or `p_maternal_death`). Stillbirth counts are inflated and miscarriage counts deflated. The existing test (`test_demographics.py:295`) covers only `p_loss`, where the mother survives.

**Fix**: record the fetus's gestational age before wiping the mother's state. For example, in `process_maternal_deaths()`, classify the unborn deaths immediately using the mother's current `gestation` (and increment the miscarriage/stillbirth results there). Alternatively, store the gestational age on the fetus, and don't let `process_prenatal_deaths()` re-classify fetuses whose mother is already dead.

### 3. Burn-in window is one step too short — `demographics.py:643`

`dtis = np.arange(np.ceil(-max_time), 0, 1)`. With a monthly `dt`, the default `dur_pregnancy` gives `max_time ≈ 9.67` steps. `np.ceil(-9.67)` is `-9` (ceil rounds toward zero for negatives), so burn-in starts at `ti=-9` instead of `-10`. There are therefore no conceptions at `ti=-10`, and those are exactly the women who should deliver at `ti=0`.

```python
import numpy as np, starsim as ss
B = []
for seed in range(8):
    sim = ss.Sim(n_agents=20000, demographics=ss.Pregnancy(fertility_rate=ss.peryear(100)),
                 start=2000, stop=2002, dt=ss.month, verbose=0, rand_seed=seed)
    sim.run()
    B.append(sim.results.pregnancy.births[:12])
print('mean births by step', np.round(np.mean(B,0),1))
```

Actual: `[16.6 48.2 51.2 46.1 45.6 45.8 47.9 45.2 43.6 45.2 43.8 48.4]`. Expected: `births[0]` in line with later steps (about 46). The same script, with a subclass that uses `np.floor(-max_time)`, gives `[48.2 51.2 46.1 45.6 45.8 47.9]`.

Blast radius: every `Pregnancy` sim with the default `burnin=True` and a sub-annual `dt`. There is a visible dip in births and CBR at the first timestep, and the number of women pregnant at `t=0` is slightly low.

**Fix**: use `np.floor(-max_time)` (or `-np.ceil(max_time)`).

### 4. Burn-in pregnancies are recorded as new pregnancies at `ti=0` — `demographics.py:1004`, `1117-1118`

During burn-in, `step()` runs about 9 times with negative `ti`, and each call does `self._counts.pregnancies += len(conceivers)`. `_counts` is only written out and reset in `update_results()`, which first runs at `ti=0`. So `results.pregnancies[0]` contains all of the roughly 9 months of pre-simulation pregnancies, and the burn-in `n_preterm` and `births` are dumped into `ti=0` the same way. The inline comment says the `+=` is "to handle burn-in", but the result is that pre-simulation events are counted as simulation events.

```python
import numpy as np, starsim as ss
for burnin in [True, False]:
    sim = ss.Sim(n_agents=5000, demographics=ss.Pregnancy(fertility_rate=ss.peryear(100), burnin=burnin),
                 start=2000, stop=2003, dt=ss.month, verbose=0, rand_seed=1)
    sim.run()
    print(burnin, sim.results.pregnancy.pregnancies[:4])
```

Actual: `True [91. 11. 15.  9.]`, `False [ 8. 11. 16.  9.]`. Averaged over 8 seeds at 20k agents, `pregnancies[0]` was 465 against a steady state of about 46. Expected: `pregnancies[0]` of about 10 (or the same as the `burnin=False` value).

Blast radius: every `Pregnancy` sim with the default `burnin=True`. Plots show a 10x spike at t=0, and the summarized total (`summarize_by='sum'`) over-counts pregnancies by about 9 months' worth.

**Fix**: reset `self._counts` (to zeros) at the end of the burn-in loop in `init_post()`. Alternatively, skip incrementing `_counts` while `self.ti < 0`, as `process_prenatal_deaths()` and `process_neonatal_deaths()` already do for their results.

### 5. Crude birth/mortality rates use the sim's `dt`, not the module's — `demographics.py:171`, `322`, `1149`

`Births.update_results()` computes `births_per_year = n_births_this_step/self.sim.t.dt_year`. `Deaths.finalize()` and `Pregnancy.finalize()` compute `units = rate_units*self.sim.t.dt_year`. But `n_births_this_step`, `results.new`, and `results.births` are counts per *module* step. Whenever a demographic module is given its own `dt` (a supported feature), the rate is off by the ratio of the two `dt`s.

```python
import starsim as ss
for mod_dt in [ss.month, ss.year]:
    sim = ss.Sim(n_agents=20000, demographics=[ss.Births(birth_rate=ss.peryear(20), dt=mod_dt), ss.Deaths(death_rate=ss.peryear(10), dt=mod_dt)],
                 start=2000, stop=2010, dt=ss.month, verbose=0, rand_seed=1)
    sim.run()
    print(sim.results.births.new.sum(), sim.results.births.cbr.mean(), sim.results.deaths.new.sum(), sim.results.deaths.cmr.mean())
```

Actual:

```
module dt=month: total births=4324.0, mean cbr=20.3 (input 20); total deaths=2193.0, mean cmr=10.3 (input 10)
module dt=year:  total births=4590.0, mean cbr=235.9 (input 20); total deaths=2291.0, mean cmr=117.8 (input 10)
```

`Pregnancy(dt=ss.month)` in a weekly sim gives a mean CBR of 108.2 instead of 24.9 (off by 52/12). Expected: CBR and CMR independent of the module `dt`. The event counts themselves are correct.

Blast radius: users running demographics at a coarser `dt` than the sim (a common way to save time). CBR/CMR are the headline calibration targets for demographics.

**Fix**: use `self.t.dt_year` in all three places.

### 6. Burn-in fetus ages use the sim's `dt` — `demographics.py:946`

`make_embryos()` ages burn-in embryos forward to `ti=0` with `people.age[new_uids] += -self.ti * self.sim.t.dt_year`. Here `self.ti` is the module's timestep, so it must be multiplied by the module's `dt` (as `_set_embryo_states()` correctly does at line 924 with `self.t.dt_year`).

```python
import numpy as np, starsim as ss
for sim_dt in [ss.month, ss.week]:
    sim = ss.Sim(n_agents=20000, demographics=ss.Pregnancy(fertility_rate=ss.peryear(100), dt=ss.month),
                 start=2000, stop=2002, dt=sim_dt, verbose=0, rand_seed=1)
    sim.init()
    p = sim.demographics.pregnancy; ppl = sim.people
    fetus = ppl.parent.notnan.uids
    exp_age = -p.ti_delivery[ss.uids(ppl.parent[fetus])] / 12  # months until delivery
    print(np.mean(ppl.age[fetus]-exp_age), np.max(np.abs(ppl.age[fetus]-exp_age)))
```

Actual: sim `dt=month` gives mean error `-0.000`, max `0.000`. Sim `dt=week` gives mean error `-0.340`, max `0.577` years. Expected: zero error in both cases.

Blast radius: `Pregnancy` with a module `dt` that differs from the sim's `dt`, with the default burn-in. Babies from burn-in pregnancies are "born" with negative ages (up to about 7 months), which feeds into anything age-based (e.g. `process_prenatal_deaths()` treats `age < 0` as unborn, and neonatal-death detection requires `age >= 0`).

**Fix**: use `self.t.dt_year`.

### 7. `finish_step()` compares a sim timestep with a module timestep — `demographics.py:1035`

`death_uids = ss.uids(self.sim.people.ti_dead <= self.ti)`. `people.ti_dead` is set by `request_death()` in sim timesteps, but `self.ti` is the module's timestep. With a finer sim `dt` (e.g. weekly sim, monthly `Pregnancy`), `self.ti` is several times smaller than the sim `ti`, so almost no deaths match. As a result, miscarriages/stillbirths are not recorded, pregnancy states are not reset for mothers whose fetus died, and unborn children of dead mothers are not killed.

```python
import starsim as ss
for sim_dt in [ss.month, ss.week]:
    sim = ss.Sim(n_agents=10000, demographics=ss.Pregnancy(fertility_rate=ss.peryear(100), p_loss=ss.bernoulli(0.02), dt=ss.month),
                 start=2000, stop=2005, dt=sim_dt, verbose=0, rand_seed=1)
    sim.run()
    r = sim.results.pregnancy
    print(r.pregnancies.sum(), r.births.sum(), r.miscarriages.sum()+r.stillbirths.sum())
```

Actual: monthly sim gives `1580.0 1108.0 256.0`, weekly sim gives `1574.0 1103.0 16.0`. Expected: about 256 losses in both. A subclass using `self.sim.ti` in this comparison gives 255 losses for the weekly sim.

Blast radius: `Pregnancy` with its own `dt`. Losses caused by `p_loss` are almost entirely unrecorded, and the affected mothers stay `pregnant=True` with no fetus.

**Fix**: compare against `self.sim.ti`, i.e. `self.sim.people.ti_dead <= self.sim.ti`. (Deaths from other modules in sim steps where `Pregnancy` doesn't run would still be missed. Fixing that fully needs those deaths to be looked up over the whole module step, but the one-line fix covers the module's own `p_loss` and delivery deaths.)

### 8. Non-CRN births crash with dataframe birth rates — `demographics.py:126`

In the dataframe branch of `get_births()`, `scaled_birth_prob` is the 1-element 1-D array returned by `ss.prob.array_to_prob()` (the scalar branch extracts `[0]` at line 112; the dataframe branch doesn't). `np.clip` keeps it 1-D, and on the `ss.options.crn=False` path `float(scaled_birth_prob)` raises under numpy 2.x.

```python
import pandas as pd, starsim as ss
ss.options.crn = False
df = pd.DataFrame(dict(Year=[2000, 2010], CBR=[40, 40]))
sim = ss.Sim(n_agents=20000, demographics=ss.Births(birth_rate=df), start=2000, stop=2005, dt=ss.month, verbose=0)
sim.run()
```

Actual: `TypeError: only 0-dimensional arrays can be converted to Python scalars` (at `demographics.py:126`). Expected: runs, as it does with `crn=True` or with a scalar/`Rate` birth rate.

Blast radius: anyone using the documented `crn=False` fast path together with data-driven birth rates (e.g. the UN CBR data format used in `tests/test_demographics.py`).

**Fix**: take the scalar in the dataframe branch too (`...array_to_prob(...)[0]`), or use `float(np.asarray(scaled_birth_prob).item())` / `.squeeze()` at line 126.

### 9. Documented `dur_pregnancy` forms crash — `demographics.py:347`, `566`

The docstring documents `dur_pregnancy (float/dur)`, but the default is `ss.choice(a=ss.weeks(...), p=...)`. Passing a duration replaces the choice's `a` with a scalar, so `numpy.choice` fails when burn-in draws from it. Separately, when fertility data is a dataframe, `make_p_conceive()` reads `self.pars.dur_pregnancy.pars.a.years`, which only exists for `ss.choice`. Any other distribution (e.g. `ss.normal`) crashes.

```python
import pandas as pd, starsim as ss
asfr = pd.read_csv(ss.root/'tests/test_data/nigeria_asfr.csv')
for fr in [ss.peryear(100), asfr]:
    for dp in [ss.weeks(39), ss.normal(ss.weeks(39), ss.weeks(1))]:
        ss.Sim(n_agents=2000, demographics=ss.Pregnancy(fertility_rate=fr, dur_pregnancy=dp), start=2000, stop=2003, dt=ss.month, verbose=0).run()
```

Actual:

```
peryear weeks(39) -> ValueError: a must be a sequence or an integer, not <class 'float'>
peryear ss.normal(loc=39, scale=1) -> OK, births 165.0
DataFrame weeks(39) -> ValueError: a must be a sequence or an integer, not <class 'float'>
DataFrame ss.normal(loc=39, scale=1) -> AttributeError: 'objdict' object has no attribute 'a'
```

Expected: all four run.

Blast radius: anyone following the docstring and setting a fixed pregnancy duration, or anyone using an alternative gestational-age distribution together with ASFR data.

**Fix**: at init, convert a scalar/`dur` `dur_pregnancy` into a distribution (e.g. `ss.constant`). In `make_p_conceive()`, estimate the time to birth without relying on `.pars.a` (e.g. use the mean of a sample from the distribution, as `init_post()` already does for the maximum).

## Low severity

### 10. Short sims crash during burn-in — `demographics.py:864`

`update_maternal_deaths()` writes `self.results['maternal_deaths'][self.ti]` unconditionally. During burn-in `self.ti` is negative, which (because of Python negative indexing) harmlessly writes into the end of the array, where the value is overwritten later. But if the sim has fewer timesteps than the burn-in length, the index is out of bounds. (`process_prenatal_deaths()` and `process_neonatal_deaths()` already guard with `0 <= ti < npts`.)

```python
import starsim as ss
sim = ss.Sim(n_agents=1000, demographics=ss.Pregnancy(), start=2000, stop=2000.5, dt=ss.month, verbose=0)
sim.run()
```

Actual: `IndexError: index -9 is out of bounds for axis 0 with size 7`. Expected: runs (a sim with `stop=2001` runs fine).

Blast radius: quick or short sims (less than about 9-10 steps) with default `Pregnancy` and a monthly or finer `dt`.

**Fix**: guard the write with `if 0 <= self.ti < self.t.npts:` as in the other two methods.

## Verified clean

The following hypotheses were tested and found correct:

- **Rates and data inputs**: `Births` with scalar, `Rate`, and dataframe birth rates (CBR about 40 for an input of 40); CRN and non-CRN birth counts agree (4647±66 vs 4610±87 over 6 seeds); `Births`/`Pregnancy` `Rate` unit conversion for per-month/per-week rates.
- **`Deaths` with data**: age- and sex-stratified data gives per-agent `p_death` matching the digitized bins to within 6.5e-8, including after agents are removed.
- **Pregnancy outcomes**: preterm fraction (4.3%, versus 4.6% implied by the default distribution); `gestation_at_birth` (mean 40 weeks); trimester UIDs partition the pregnant women; `dur_gestation` range.
- **Multiples and networks**: twin handling (a births/pregnancies ratio of 1.44 with a 50% twin probability); every pregnant woman has an unborn child; prenatal-network edges (the excess edges at the end are fetuses due next step).
- **Maternal deaths with burn-in**: the negative-index writes are overwritten, so the results are correct.
- **Other calculations**: the ASFR rolling-annual sum in `Pregnancy.finalize()` (including `tdim == npts`); `set_ptb()` age binning; the `rel_ptb`-sorted assignment of pregnancy durations; the fertility-rate denominator adjustment for already-pregnant women.

The per-agent `Deaths` rate also applies to unborn agents (negative ages fall into the first bin by design, per the inline comment); this was noted but not recorded as a bug.
