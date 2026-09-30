# `diseases.py` bug audit

Audit of `starsim/diseases.py` (all 927 lines: `Disease`, `Infection` including transmission, congenital outcomes and results, `InfectionLog`, `NCD`, `SIR`, `SEIR`, `SIS`) for real, unambiguous bugs only. Method: line-by-line reading, then a repro script for every candidate, run against the editable install on branch `rc3.6.2` at commit `3d8dc9d5` (working tree as-is; Starsim 3.6.1, numpy 2.4.6). Repro scripts are in the session scratchpad under `diseases/`. Style, docstrings, performance, and error-message quality are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `SIR.step_state()` (and `SEIR` via `super()`) | Compares `ti_recovered`/`ti_dead` (module timesteps) against `sim.ti` (sim timesteps), so recovery and death happen at the wrong time when the disease has its own `dt` | 689, 694 |
| 2 | Low | `NCD.step()` / `NCD.step_state()` | A prognosis that rounds to 0 timesteps sets `ti_dead` to the current step after deaths were already checked, so the agent never dies but is still counted in `new_deaths` | 608, 619-620, 641 |

## Medium severity

### 1. `SIR`/`SEIR` recovery and death use `sim.ti` instead of the module's `ti` — `diseases.py:689`, `diseases.py:694`

`set_infection()`/`set_progression()` schedule `ti_recovered` and `ti_dead` in the disease's own timestep units (`self.ti + dur_inf.rvs(...)`, where the duration is sampled against the module's `dt`). `SIR.step_state()` then triggers them with `self.ti_recovered <= sim.ti` and `self.ti_dead <= sim.ti`, i.e. against the sim's timestep counter. These are equal only when the disease and sim share a `dt`. `SIS.step_state()`, `SEIR.step_state()` (for exposed→infectious), and `NCD` all correctly use `self.ti`, and mixed module/sim timesteps are a tested feature (`tests/test_timeline.py::test_mixed_timesteps`, with SIS).

```python
sir = ss.SIR(dur_inf=ss.constant(ss.years(2)), p_death=0, beta=ss.peryear(0), init_prev=0.1, dt=ss.years(1))
sim = ss.Sim(n_agents=2000, start=2000, stop=2006, dt=sim_dt, diseases=sir, networks='random', verbose=0)
sim.run()
```

Actual:

```
sim dt=1 year,   SIR dt=1 year, p_death=0: sir.n_infected by year = [216 216   0   0   0   0   0]
sim dt=1 month,  SIR dt=1 year, p_death=0: sir.n_infected by year = [216   0   0   0   0   0   0]
sim dt=1 year , SIR dt=1 year, p_death=1: deaths occur in year(s) [2002.]
sim dt=1 month, SIR dt=1 year, p_death=1: deaths occur in year(s) [2001.]
```

Expected: identical disease dynamics in both cases (a yearly SIR with a 2-year infection recovers/dies in 2002), since only the sim's timestep changed. With a monthly sim, `sim.ti` runs 12x ahead of the disease's `ti`, so every infection ends at the disease's next step regardless of `dur_inf`. In the reverse case (disease `dt` finer than the sim's) `sim.ti` lags, so infections last far longer than `dur_inf`: with a yearly sim and a monthly SIR (`dt=ss.months(1)`, same pars, `stop=2008`), `sir.n_infected` stays at 216 for all 97 monthly points, i.e. no one recovers from a 2-year infection within 8 years (`ti_recovered = 24` is never reached by `sim.ti`, which ends at 8).

Blast radius: anyone giving `ss.SIR` or `ss.SEIR` (or subclasses that call `super().step_state()`) a `dt` different from the sim's, e.g. a daily-resolution disease in a yearly demographic sim. Infection durations, prevalence and deaths are silently wrong; nothing errors.

**Fix**: use `self.ti` in both comparisons in `SIR.step_state()` (lines 689 and 694), as `SIS` and `SEIR` already do.

## Low severity

### 2. `NCD` agents whose prognosis rounds to zero never die, but are counted as deaths — `diseases.py:608`, `diseases.py:619-620`, `diseases.py:641`

In each step, `step_state()` (which runs first) requests deaths for `ti_dead == ti`, then `step()` sets `ti_dead = ti + prognosis.rvs(new_cases, round=True)` for newly affected agents. When the sampled prognosis rounds to 0 timesteps, `ti_dead` equals the current `ti`, which has already been checked; since the test is `==` rather than `<=`, it never matches again, and the agent lives forever in the `affected` state. `update_results()` then recomputes `new_deaths` as `count_nonzero(ti_dead == ti)`, which does include these agents, overwriting the (correct) count from `step_state()`. With the default `weibull(c=2, scale=ss.years(5))` and `dt=1` year, about 1% of cases round to 0.

```python
ncd = ss.NCD()
sim = ss.Sim(n_agents=5000, start=2000, stop=2050, dt=ss.years(1), diseases=ncd, verbose=0, copy_inputs=False, rand_seed=1)
sim.run()
```

Actual:

```
alive & affected: 22  of which ti_dead is already in the past (will never die): 18
years since scheduled death for these agents: [16. 21. 25. 34. 35. 37. 40. 43. 44. 45. 47. 49. 50.]
NCD new_deaths total: 1519  sim deaths total: 1501
```

Expected: no affected agent alive past its `ti_dead`, and `ncd.results.new_deaths` summing to the deaths that actually occurred (1501 here, the NCD being the only cause of death).

Blast radius: users of the example `ss.NCD` (and models built by copying it) on annual or coarser timesteps; ~1% of cases become immortal and the NCD death result disagrees with the sim's death count.

**Fix**: use `self.ti_dead <= ti` (restricted to affected, not-yet-dead agents) in `step_state()`, or clamp the prognosis to at least one timestep in `step()`; and drop the recount in `update_results()` in favour of the value recorded in `step_state()`.

## Verified clean

Hypothesised and tested, found correct: `SIS.update_immunity()` calling `waning.to_prob()` with no argument (it uses the `default_dur` that `Module.link_timepars()` sets to the module's `dt`); the mixing-pool branch of `Infection.infect()` passes `(rel_sus, rel_trans, ...)` in the order `MixingPool.compute_transmission()` expects; the numba transmission kernel and the `unique(return_index=True)` source/network alignment in `infect()`; `SIR`/`SEIR` durations measured in module timesteps on a shared `dt` (e.g. `ti_recovered = 24` for a 2-year infection on a monthly sim); `set_congenital()` scheduling from the mother's `ti_delivery` (indexed by source UID) and the one-step-early firing of prenatal death keys; `Infection.set_prognoses()` not counting `init_prev` seeds as incident; `InfectionLog` entries matching seeds plus `cum_infections` (7516 = seeds + 7500), `to_df()` sort/nullable-int conversion, and `add_data()` selecting the latest in-edge; `update_results()` prevalence denominators. Congenital infections (via `set_congenital()`) not appearing in `new_infections` looked like a design choice rather than a clear bug and was not recorded.
