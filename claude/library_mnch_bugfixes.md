# `starsim/library/mnch/` bug audit

Audit of `maternal_infections.py` (102 lines), `neonatal_sepsis.py` (107), and `fetal_health.py` (528) under `starsim/library/mnch/`, on branch `rc3.6.2` (HEAD `3d8dc9d5`), Starsim 3.6.1 (editable install), numpy 2.4.6, sciris 3.3.0. Method: line-by-line reading together with `ss.SIR`, `Infection.set_congenital()`/`step_congenital()` in `starsim/diseases.py`, `Module.update_pars()`, and the integration-loop order in `starsim/loop.py`, then a repro script for every candidate (scripts in the session scratchpad under `library/`). Only defects the repro demonstrated are recorded.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `fetal_infection.step()` | Checks `ti_infected == self.ti`, but connectors run before `disease.step()` sets `ti_infected`, so infections during pregnancy never cause fetal damage | `fetal_health.py:524` |
| 2 | High | `NeonatalSepsis.set_prognoses()` | `super()` (`SIR`) schedules death with `p_death=0.5` for every infected agent, so about half of the 30% seeded initial population is killed | `neonatal_sepsis.py:98` |
| 3 | Medium | `NeonatalSepsis.set_prognoses()` | Neonates get two independent `p_death` draws, so case fatality is about 75%, not the documented 50% | `neonatal_sepsis.py:98, 104` |
| 4 | Medium | `treat_pregnant.step()` | "Curing" sets `infected=False` but leaves the SIR-scheduled `ti_dead`, so cured women still die of the disease | `fetal_health.py:420-421` |
| 5 | Medium | `treat_pregnant.step()` → `reverse_growth_restriction()` | `tx_growth_reversal` is documented as a fraction (0-1) but subtracted as an absolute amount, reversing 100% instead of 70% | `fetal_health.py:425, 321` |
| 6 | Medium | `treat_pregnant.__init__()`, `fetal_infection.__init__()` | Documented parameters (`p_treat`, `growth_penalty`, ..., or `pars=`) raise `TypeError`, including the docstring's own example | `fetal_health.py:376, 471` |
| 7 | Low | `FetalHealth` | `n_exposures` is never incremented, so the `mean_exposures` result is always 0 | `fetal_health.py:125, 226` |

## High severity

### 1. `fetal_infection` never sees infections that occur during pregnancy — `fetal_health.py:524`

`newly_infected = sim.diseases.sir.ti_infected == self.ti`. In the integration loop (`loop.py:48-50`), `connector.step()` runs after `disease.step_state()` but before `disease.step()`, which is where `SIR.set_infection()` sets `ti_infected = self.ti`. On the timestep an infection happens, the connector has already run. On the next timestep, `ti_infected == ti - 1`. The equality therefore never holds for transmission-driven infections, and damage point 2 in the class docstring ("During pregnancy, when a new infection occurs") never fires. Only the conception-time callback applies damage.

```python
class probe(ssl.mnch.fetal_infection):
    def step(self):
        preg = self.sim.people.pregnancy; sir = self.sim.diseases.sir; pu = preg.pregnant.uids
        counts['pregnant_infected_during_pregnancy'] += int(np.sum(sir.ti_infected[pu] == self.ti - 1))
        newly = sir.ti_infected == self.ti
        counts['apply_from_step'] += int(len(pu[newly[pu]]))
        return super().step()

sim = ss.Sim(n_agents=3000, start=2020, stop=2030, dt=ss.month, rand_seed=1,
    diseases=ss.SIR(beta=ss.peryear(0.5), init_prev=0.1),
    demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30)), ss.Deaths()],
    connectors=probe(name='fetal_infection'), custom=ssl.mnch.FetalHealth(),
    networks=[ss.PrenatalNet(), ss.RandomNet()])
sim.run(verbose=0)
```

Actual:

```
{'apply_from_step': 0, 'apply_from_conception': 0, 'pregnant_infected_during_pregnancy': 22}
```

Expected: the 22 women infected while pregnant get fetal damage (`apply_from_step` = 22).

Blast radius: every user of `fetal_infection`, which is the template the module docstring tells people to subclass. The disease effect on birth weight and preterm birth is limited to women already infected at conception. `tests/test_diseases.py::test_fetal_health` still passes because the conception path produces some damage.

**Fix**: compare against the previous timestep (`ti_infected == self.ti - 1`, or `>=` the last processed step, which also catches the final step), or run the check from a post-transmission hook.

### 2. `NeonatalSepsis` kills about half of the infected initial population — `neonatal_sepsis.py:98`

`super().set_prognoses(uids, sources)` is `SIR.set_prognoses()`, which calls `SIR.set_progression()`. That draws `p_death` (0.5) for every infected agent and sets `ti_dead = ti_infected + dur_inf` (about 7 days). `init_prev=0.3` also seeds the initial population (`Infection.init_post()`), so about 15% of all agents, of every age, die within the first weeks. The method's own docstring says "Non-neonates who get infected (via init_prev at sim start) just recover normally", and the class is documented as producing *neonatal* deaths.

```python
n0 = 20000
sim = ss.Sim(n_agents=n0, start=2000, stop=2005, dt=ss.days(7), rand_seed=1,
    diseases=ssl.mnch.NeonatalSepsis(),
    demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30)), ss.Deaths(death_rate=0)],
    networks=[ss.PrenatalNet(), ss.RandomNet()])
sim.run(verbose=0)
```

Actual:

```
initial-population agents infected: 5966; killed by sepsis: 3061 (documented: 0, "just recover normally")
```

Expected: 0 non-neonatal deaths.

Blast radius: every sim using `NeonatalSepsis` (e.g. `tests/test_demographics.py`, `tests/test_library.py`). It causes an immediate mass-mortality event that distorts the population and all downstream demographic results.

**Fix**: bypass `SIR.set_progression()`. Call `ss.Infection.set_prognoses()` plus `set_infection()`, then schedule recovery for everyone and death only for neonates. Alternatively, override `set_progression()` so non-neonates get recovery only.

## Medium severity

### 3. Neonatal case fatality is about 75% rather than `p_death = 0.5` — `neonatal_sepsis.py:98, 104`

For neonates, the `SIR` path (finding 2) already schedules death for a `p_death` fraction. Line 104 then draws `p_death.filter(neonates)` again, and those agents are also scheduled to die. The union gives 1 - 0.5² = 0.75.

Same run as finding 2:

```
newborns infected: 220; died: 164; case fatality: 0.745 (documented p_death = 0.5)
```

Expected: about 0.5.

Blast radius: all neonatal-death counts from this model are inflated by 50%.

**Fix**: fixing finding 2 so that `p_death` is drawn once, for neonates only, resolves this too.

### 4. Treatment "cures" infection but not the scheduled death — `fetal_health.py:420-421`

`treat_pregnant.step()` sets `disease.infected[treated] = False` and `recovered = True`, but leaves `ti_dead` (set by `SIR.set_progression()`) in place. `SIR.step_state()` then requests death when `ti_dead <= ti` whether or not the agent is still infected.

```python
sir = ss.SIR(beta=ss.peryear(2), init_prev=0.1, p_death=0.3, dur_inf=ss.lognorm_ex(mean=ss.years(1)))
sim = ss.Sim(n_agents=5000, start=2020, stop=2025, dt=ss.month, rand_seed=2, diseases=sir,
    demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30)), ss.Deaths()],
    connectors=ssl.mnch.fetal_infection(), interventions=probe(name='treat_pregnant', disease='sir'),
    custom=ssl.mnch.FetalHealth(), networks=[ss.PrenatalNet(), ss.RandomNet()])
```

(`probe` subclasses `treat_pregnant` to record who was treated.) Actual:

```
treated: 75  died from SIR after being cured: 24
people.ti_dead vs sir.ti_dead (first 5): [25.  3.  5. 12.  7.] [25.  3.  5. 12.  7.]  ti_treated: [0. 1. 2. 2. 2.]
still infected flag at death: [False False False False False]
```

Expected: none of the treated, cured women die from SIR.

Blast radius: any use of `treat_pregnant` with a lethal disease. Treatment has no effect on maternal mortality. The same applies to any subclass copying this cure logic. With default SIR `p_death=0.01` the effect is small but systematic.

**Fix**: also clear the scheduled outcomes on cure, e.g. `disease.ti_dead[treated] = np.nan` and `disease.ti_recovered[treated] = self.ti`.

### 5. `tx_growth_reversal` is applied as an absolute amount, not a fraction — `fetal_health.py:425, 321`

The `treat_pregnant` docstring says `tx_growth_reversal (float): fraction of growth restriction to reverse (0-1)`, default 0.7. It is passed to `FetalHealth.reverse_growth_restriction(uids, amount)`, which computes `max(0, current - amount)`. Growth restriction per infection is 0.15, so any value ≥ 0.15 wipes it out completely. (`tx_timing_reversal` goes to `reverse_timing_shift(uids, fraction)`, which does treat it as a fraction, so the two parameters are inconsistent.)

Same run as finding 4:

```
treated with nonzero restriction: 53
growth_restriction before (first 5): [0.15 0.15 0.15 0.15 0.15]
growth_restriction after  (first 5): [0. 0. 0. 0. 0.]
fraction reversed (mean): 1.0  expected: 0.7
```

Expected: 0.15 → 0.045.

Blast radius: every `treat_pregnant` user. Treatment benefit on birth weight is overstated, and the parameter has almost no effect over its documented range.

**Fix**: call `fh.reverse_growth_restriction(treated, fh.growth_restriction[treated] * self.pars.tx_growth_reversal)`. Alternatively, add a fractional reversal method matching `reverse_timing_shift()`.

### 6. `treat_pregnant` and `fetal_infection` reject their own parameters — `fetal_health.py:376, 471`

Both `__init__` methods pass `**kwargs` straight to `super().__init__()` (which ends up in `ss.Timeline`), call `define_pars()` afterwards, and never call `self.update_pars(...)`. So every documented par, including the docstring example `treat_pregnant(disease='sir', start_year=2025, p_treat=ss.bernoulli(p=0.5))`, raises. The `pars=` form fails too.

```python
ssl.mnch.treat_pregnant(disease='sir', start_year=2025, p_treat=ss.bernoulli(p=0.5))
ssl.mnch.fetal_infection(growth_penalty=0.3)
ssl.mnch.treat_pregnant(pars=dict(tx_growth_reversal=0.5))
ssl.mnch.fetal_infection(pars=dict(growth_penalty=0.3))
```

Actual:

```
treat_pregnant ERROR: TypeError("Timeline.__init__() got an unexpected keyword argument 'p_treat'")
fetal_infection ERROR: TypeError("Timeline.__init__() got an unexpected keyword argument 'growth_penalty'")
ERROR TypeError("Timeline.__init__() got an unexpected keyword argument 'pars'")
ERROR TypeError("Timeline.__init__() got an unexpected keyword argument 'pars'")
```

Expected: the parameters are set.

Blast radius: anyone configuring these modules. The only workaround is mutating `.pars` after construction.

**Fix**: use the standard pattern, `def __init__(self, ..., pars=None, **kwargs): super().__init__(); self.define_pars(...); self.update_pars(pars, **kwargs)`.

## Low severity

### 7. `mean_exposures` is always zero — `fetal_health.py:125, 226`

`n_exposures` is documented as "Disease exposures during pregnancy" and is tracked on mothers. It is reset to 0 at conception (line 178), but nothing in `starsim` ever increments it: neither `_apply_damage()` nor `apply_growth_restriction()`/`apply_timing_shift()` touches it (confirmed by grep). The `mean_exposures` result is therefore identically 0 even when damage is applied.

```python
# SIR + fetal_infection + treat_pregnant(start_year=2025) + FetalHealth, 3000 agents, 2020-2030, monthly
sim.custom.fetal_health.results.mean_exposures.sum()
```

Actual: `sum mean_exposures: 0.0`. Expected: > 0 whenever fetal damage is applied.

Blast radius: users reporting exposures per pregnancy from this result.

**Fix**: increment `self.n_exposures[uids] += 1` in `apply_growth_restriction()` or `apply_timing_shift()` (or in `fetal_infection._apply_damage()`).

## Verified clean

`CongenitalDisease` (docstring example, 5000 agents, 10 years): assigned outcome fractions [0.302, 0.419, 0.279] match `p=[0.3, 0.4, 0.3]`; stillbirths register in `pregnancy.stillbirths`; `congenital` flags fire; no `ti_stillborn` is left pending. `NeonatalSepsis` screens each newborn exactly once, and the initial population is exempted from newborn screening. FetalHealth baseline birth weight and GA are in range (the existing test covers this). The `apply_timing_shift()` one-way ratchet and `min_ga` floor, the week/timestep conversions via `self.dt.weeks` (with FetalHealth and Pregnancy on the same dt), twins' parent lookup in `on_delivery()`, and `treat_pregnant`'s default `start_year`/`end_year` comparisons with `ss.date` bounds (runs without error) all work.
