# `starsim/library/diseases/` bug audit

Audit of `measles.py` (56 lines), `ebola.py` (143), `cholera.py` (203), and `hiv.py` (221) under `starsim/library/diseases/`, working-tree version on branch `rc3.6.2` (HEAD `3d8dc9d5`), Starsim 3.6.1 (editable install), numpy 2.4.6, sciris 3.3.0. Method: line-by-line reading together with the base classes in `starsim/diseases.py` (`Infection`, `SIR`, `SEIR`) and the loop order in `starsim/loop.py`, then a repro script for every candidate (scripts in the session scratchpad under `library/`). Only defects the repro demonstrated are recorded. Out of scope: style, docs, tests, performance, contrived inputs.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Cholera.calc_environmental_prev()` | Shedding is converted with `.to_prob()`, capping new environmental bacteria at 1 per timestep; environmental transmission is effectively switched off (decay is also mis-converted) | 145-146 |
| 2 | High | `Ebola.step_state()` | Dead bodies transmit for exactly one timestep, whatever `dur_dead2buried` is; `buried` is never set for dead agents | 162-169 |
| 3 | Medium | `Cholera.update_results()` | `new_deaths`/`cum_deaths` are always 0 because `ti_dead` is fractional and compared with `== ti` | 201-202 |
| 4 | Medium | `Cholera.infect()` | Environmental cases are concatenated with direct cases without deduplication, so the same agent is infected twice and `new_infections` is overcounted | 184-187 |
| 5 | Low | `HIV.step_state()` | Hard-coded `people.hiv` crashes when the module is given any other `name` | 299 |

## High severity

### 1. Environmental shedding is converted to a probability, so the reservoir never fills — `cholera.py:145`

`new_bacteria = (p.shedding_rate * (n_symptomatic + p.asymp_trans * n_asymptomatic)).to_prob()`. `shedding_rate` is `ss.freqperday(10)` (a count of bacteria per day), and `freq * N` returns another `freq`, so `.to_prob()` computes `1 - exp(-rate*dt)`, which is at most 1. However many agents are shedding, the reservoir gains at most one unit per timestep, against a `half_sat_rate` of 1,000,000, so `env_conc` stays around 1e-5 and the environmental route contributes essentially nothing. The same migration commit (`df2a662f`, which replaced the original `shedding_rate * (...) * self.t.dt` and `np.exp(-(decay_rate * self.t.dt))`) also changed line 146 to `old_prev * np.exp(-p.decay_rate.to_prob())`. That applies `exp` to a probability rather than to `rate*dt`. It is close at dt = 1 day, but wrong at coarser timesteps (dt = 1 week: survival 0.814 instead of 0.794).

```python
class Fixed(ssl.Cholera):
    def calc_environmental_prev(self):
        p = self.pars; r = self.results; ti = self.ti
        n_s = self.symptomatic.sum(); n_a = self.asymptomatic.sum()
        new_bacteria = (p.shedding_rate * (n_s + p.asymp_trans * n_a)).to_events(self.t.dt)
        old_bacteria = r.env_prev[ti-1] * (1 - p.decay_rate.to_prob(self.t.dt))
        r.env_prev[ti] = new_bacteria + old_bacteria
        r.env_conc[ti] = r.env_prev[ti] / (r.env_prev[ti] + p.half_sat_rate)

for cls in [ssl.Cholera, Fixed]:
    sim = ss.Sim(n_agents=5000, rand_seed=1, dt=ss.days(1), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'),
                 diseases=cls(name='cholera', beta={'random':0.0}, init_prev=0.05), networks='random')
    sim.run(verbose=0)
    ...
c = ssl.Cholera()
(c.pars.shedding_rate*100).to_prob(), (c.pars.shedding_rate*100).to_events(ss.days(1))
```

Actual:

```
Cholera  (direct beta=0): env_prev max=13.8, env_conc max=1.38e-05, new infections (all environmental)=1
Fixed    (direct beta=0): env_prev max=5.28e+03, env_conc max=0.00525, new infections (all environmental)=325
freqperday(10) x 100 shedders, dt=1 day, .to_prob(): 1.0  .to_events(1 day): 1000.0
```

Expected: 100 shedders at 10 per day add 1000 units per day, and environmental transmission is a material route (325 infections instead of 1 in this setup).

Blast radius: every `ssl.Cholera` user. Environmental transmission, which is the distinguishing feature of the model, is silently off, and `env_prev`/`env_conc` results are meaningless. `tests/test_library.py:68` only checks that `env_prev` is nonzero.

**Fix**: use `.to_events(self.t.dt)` (or multiply by `self.t.dt` as the original did) for `new_bacteria`. For the decay factor, use `np.exp(-p.decay_rate * self.t.dt)` in rate terms, or equivalently `1 - p.decay_rate.to_prob(self.t.dt)`.

### 2. Post-mortem transmission lasts one timestep, regardless of `dur_dead2buried` — `ebola.py:162-169`

`buried = (self.ti_buried <= ti).uids` and `unburied = ((self.ti_dead <= ti) & (self.ti_buried > ti)).uids` operate on active agents only. Dead agents are removed from the population and from every network at the end of the timestep they die. So an unsafe-burial body is "unburied and infectious" only on the step its death is requested (it is still alive until `people.step_die`), and never after. `dur_dead2buried` therefore has no effect at all, and dead agents never get `buried=True` (the only `buried` flags come from safe burials on the death step itself).

```python
for d in [2, 20, 200]:
    sim = ss.Sim(n_agents=5000, rand_seed=seed, dt=ss.days(1), start=ss.date('2000-01-01'), stop=ss.date('2000-12-31'),
                 diseases=ssl.Ebola(beta=ss.prob(0.003, ss.days(1)), unburied_factor=50, p_safe_bury=ss.bernoulli(0.0),
                                    dur_dead2buried=ss.constant(ss.days(d))),
                 networks=ss.RandomNet(n_contacts=10))   # seeds 0-3
```

Actual:

```
dur_dead2buried=  2 days: infections per seed = [416, 192, 473, 612]
dur_dead2buried= 20 days: infections per seed = [416, 192, 473, 612]
dur_dead2buried=200 days: infections per seed = [416, 192, 473, 612]
dead agents: 236 ; of which buried=True: 0 ; with ti_buried set: 236
```

Expected: longer times to burial give more transmission from bodies, and dead agents become `buried` once `ti_buried` passes.

Blast radius: every `ssl.Ebola` user. The documented unsafe-burial mechanism (`dur_dead2buried`, and in practice most of `unburied_factor`'s intended effect) does nothing, and `n_buried` badly undercounts. `tests/test_library.py:74` passes only because of safe burials.

**Fix**: the body must stay in the transmission system until burial. For example, request death at `ti_buried` rather than `ti_dead`, and treat the interval `[ti_dead, ti_buried)` as an "unburied" infectious state on a still-active agent (counted as a death for results at `ti_dead`). Alternatively, keep dead agents in the network until burial. Either way, `buried` and `rel_trans` must be computed over agents that still exist.

## Medium severity

### 3. Cholera `new_deaths` and `cum_deaths` are always zero — `cholera.py:201`

`res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)`. `ti_dead` is set to `ti_symptomatic + dur_symp2dead.rvs(...)`, where `ti_symptomatic = ti_infectious = ti + dur_exp`, so it is a fractional timestep and is essentially never exactly equal to an integer `ti`. Deaths themselves are triggered correctly by `SIR.step_state()` (`ti_dead <= ti`). (`HIV.update_results()` uses the same idiom correctly because it sets `ti_dead = self.ti`.)

```python
sim = ss.Sim(n_agents=5000, rand_seed=1, dt=ss.days(1), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'),
             diseases=ssl.Cholera(beta=ss.prob(0.1, ss.days(1)), p_death=ss.bernoulli(0.2)), networks='random')
sim.run(verbose=0)
```

Actual:

```
agents killed by cholera: 518
cholera.results.new_deaths.sum(): 0.0  cum_deaths[-1]: 0.0
sample ti_dead values: [14.706134 10.153085 10.885918 16.73643  20.203314]
people deaths total: 518.0
```

Expected: `new_deaths.sum() == 518`.

Blast radius: anyone reading or plotting `sim.results.cholera.new_deaths`/`cum_deaths`, which report zero mortality.

**Fix**: count deaths that became due this step, e.g. `np.count_nonzero((self.ti_dead > ti - 1) & (self.ti_dead <= ti))`, matching the `<= ti` trigger in `SIR.step_state()`, or record `len(deaths)` when they are requested.

### 4. Agents infected both directly and environmentally are infected twice — `cholera.py:184-187`

`Infection.infect()` deduplicates network cases (`new_cases.unique(...)`), but `Cholera.infect()` then draws `p_env_transmit.filter(self.susceptible)`. Agents just infected via the network are still `susceptible` at that point, and the two sets are concatenated without deduplication. Duplicate UIDs go to `set_prognoses()`, and `Infection.set_prognoses()` adds `len(uids)` to `new_infections`.

```python
class Probe(ssl.Cholera):
    ndup = 0
    def infect(self):
        new_cases, sources, networks = super().infect()
        Probe.ndup += len(new_cases) - len(np.unique(new_cases))
        return new_cases, sources, networks

sim = ss.Sim(n_agents=5000, rand_seed=1, dt=ss.days(1), start=ss.date('2000-01-01'), stop=ss.date('2000-06-01'),
             diseases=Probe(name='cholera', beta=ss.prob(0.05, ss.days(1)), half_sat_rate=10, init_prev=0.02), networks='random')
sim.run(verbose=0)
```

Actual:

```
duplicate UIDs returned by infect(): 220
new_infections.sum(): 5122.0  agents infected after t=0: 4902
```

Expected: no duplicates; `new_infections.sum() == 4902`.

Blast radius: currently masked by finding 1, since environmental transmission barely happens. Once finding 1 is fixed, every cholera sim overcounts incidence. Duplicate UIDs also mean duplicated distribution draws and a duplicated infection log.

**Fix**: exclude agents already in `new_cases` from the environmental draw (or run `.unique()` on the concatenation, keeping the network source for duplicates).

## Low severity

### 5. `HIV` crashes when renamed — `hiv.py:299`

`can_die = people.hiv.infected.uids` hard-codes the module name, when `self.infected.uids` was intended. Any `name` other than `'hiv'` raises.

```python
ss.Sim(n_agents=500, diseases=ssl.HIV(name='hiv2', beta=0.02), networks='random').run()
```

Actual: `AttributeError("'People' object has no attribute 'hiv'")`. Expected: runs.

Blast radius: users running two HIV instances or naming the module; loud crash, no silent corruption. (`ART` and `CD4_analyzer` also look up `sim.diseases.hiv`, but they declare `requires = HIV`, so that coupling is documented.)

**Fix**: use `self.infected.uids`.

## Verified clean

Measles: `ss.normal(loc=ss.days(8))` with the default unitless `scale=1.0` gives a mean of 8 days and an SD of 1 day consistently at dt = 1 day, 1 week, and 1/52 year, and `dur_inf` behaves the same. Ebola: the severe/death/recovery scheduling in `set_progression()` is consistent, `severe` is cleared on recovery and death, and `unburied_factor` does act on the death step (it is only the duration that is broken). Cholera: `env_prev[ti-1]` at `ti=0` reads a zero slot (harmless), and the symptomatic/asymptomatic bookkeeping and recovery scheduling are correct. `asymp_trans` affecting only shedding and not direct transmission matches the class docstring ("shed far less bacteria"), so it is not reported. HIV: `results.new_deaths` matches actual deaths (2746 vs 2746); the ART delay equals 12 timesteps for `art_delay=years(1)` at monthly dt; the CD4 update and `death_prob_func` scaling are correct; and `CD4_analyzer` truncation on population growth works.
