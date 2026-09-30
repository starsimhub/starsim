# `interventions.py` bug audit

Audit of `starsim/interventions.py` (all 519 lines: `Intervention`, `RoutineDelivery`, `CampaignDelivery`, `BaseTest`/`BaseScreening`/`BaseTriage` and their routine/campaign variants, `BaseTreatment`/`treat_num`, `BaseVaccination`/`routine_vx`/`campaign_vx`) for real, unambiguous bugs only. Method: line-by-line reading, then a repro script for every candidate, run against the editable install on branch `rc3.6.2` at commit `3d8dc9d5` (working tree as-is; Starsim 3.6.1, numpy 2.4.6). Repro scripts are in the session scratchpad under `interventions/`. Style, docstrings, performance, and error-message quality are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `BaseTriage.step()` | Tests `self.sim.t` (the `Timeline` object) instead of `sim.ti` against `timepoints`, so `routine_triage`/`campaign_triage` never deliver anything | 291 |
| 2 | High | `treat_num.add_to_queue()` / `step()` | Queue accumulates duplicates and agents who are no longer eligible, so capacity collapses (≈5 treated/step with `max_capacity=50` and 2000+ infected) | 430, 452 |
| 3 | Medium | `BaseTest.deliver()` / `BaseScreening.step()` | `outcomes` is never reset, so positives from the last screening round persist on every later timestep | 229, 266 |
| 4 | Medium | `RoutineDelivery.init_pre()` | User-supplied `years` are overwritten by `inclusiverange(start, end)`, so any non-annual `years` (e.g. `[2002, 2006]`) raises a length mismatch | 132 |
| 5 | Medium | `RoutineDelivery.init_pre()` | Crashes with default `start_year`/`end_year` whenever the sim uses date-based start/stop (e.g. `start='2000-01-01'`) | 107-119 |
| 6 | Medium | `CampaignDelivery.init_pre()` | Campaign years outside the sim window silently snap to the first/last timestep, so a 2030 campaign fires in 2010 | 176 |
| 7 | Low | `CampaignDelivery.__init__()` | Documented `interpolate` argument is stored but never used | 165 |

## High severity

### 1. Triage never runs: `self.sim.t in self.timepoints` compares a `Timeline` object — `interventions.py:291`

`BaseTriage.step()` guards delivery with `if self.sim.t in self.timepoints`. `sim.t` is the `ss.Timeline` object, not the integer timestep, so the membership test against the array of timestep indices is always `False` and `deliver()` is never called. Every other delivery class (`BaseScreening.step()`, `BaseVaccination.step()`) uses `sim.ti`. The result is that `ss.routine_triage` and `ss.campaign_triage` are silent no-ops.

```python
screen = ss.routine_screening(product=make_dx('dx1'), prob=0.5, name='screening')
screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']  # docstring pattern
triage = counting_triage(product=make_dx('dx2'), eligibility=screened_pos, prob=0.9, name='triage')  # routine_triage subclass that logs len(step())
sim = ss.Sim(n_agents=2000, start=2000, stop=2010, diseases=dict(type='sis', init_prev=0.3), networks='random', interventions=[screen, triage], copy_inputs=False)
sim.run()
```

Actual:

```
n positive per screen: [314. 369. 476. 579. 674. 744. 794. 855. 906. 922. 753.]
triaged per step: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sim.t type: <class 'starsim.timeline.Timeline'>  sim.t in timepoints: False
```

Expected (same sim, with the test done on `sim.ti`): `triaged per step: [274, 333, 437, 540, 613, 659, 702, 774, 803, 841, 688]`.

Blast radius: every user of `routine_triage` or `campaign_triage` (and any subclass of `BaseTriage` that doesn't override `step()`). Nothing errors, so a screen-and-treat cascade simply loses its triage stage. There are no tests of triage in `tests/`.

**Fix**: change the guard to `if self.sim.ti in self.timepoints:` (matching `BaseScreening.step()`). Note that once this is fixed, finding 3 starts to bite: the documented eligibility pattern will re-triage stale screening positives on every step.

### 2. `treat_num` queue fills with duplicates and ineligible agents, collapsing capacity — `interventions.py:430`, `interventions.py:452`

Each step, `add_to_queue()` appends every eligible agent who accepts, with no check for whether they are already queued, so an agent who stays eligible while waiting is appended again every step. The queue is only pruned of agents who were actually treated (line 452); agents who recover, die, or otherwise stop being eligible stay in the queue forever. Because `get_candidates()` takes `queue[:max_capacity]` from the head, the head soon consists of duplicates and no-longer-eligible agents, which `BaseTreatment.step()` then discards via `intersect(still_eligible)`. Capacity is wasted and the queue grows without bound.

This is the exact example in `docs/user_guide/modules_products.qmd`:

```python
tx_data = sc.dataframe(columns=['disease','state','post_state','efficacy'], data=[['sis','infected','susceptible',0.8]])
tx = logged_treat_num(product=ss.Tx(df=tx_data), eligibility=lambda sim: sim.diseases.sis.infected.uids, max_capacity=50)  # treat_num subclass that logs (n treated, queue length, unique in queue)
sim = ss.Sim(n_agents=5000, diseases=dict(type='sis', init_prev=0.2), networks='random', interventions=tx, verbose=0, copy_inputs=False, rand_seed=1)
sim.run()
```

Actual (first steps, then the last step):

```
(n treated, queue length, unique agents in queue)
(50, 947, 947)
(50, 2168, 1271)
(50, 3722, 1654)
(50, 5671, 2099)
...
(4, 113633, 4897)
n treated per step: [50, 50, 50, 50, 50, 50, 50, 50, 50, 42, 25, 9, 5, 7, 15, 12, 16, 9, 13, 11, 13, 14, 10, 8, 4, 8, 7, 10, 9, 9, 9, 3, 5, 4, 7, 4, 5, 2, 8, 7, 7, 8, 10, 7, 8, 7, 6, 3, 6, 6, 4]
n infected per step: [1321 1704 2149 2616 3095 3518 3899 4162 4343 4365 4138 3832 3553 3342 ... 2204 2215]
```

Expected: 50 treated on every step, since more than 2000 agents are infected (and eligible) throughout. With the queue deduplicated and pruned of ineligible agents (see Fix), the same sim gives `n treated per step: [50, 50, ..., 50]` (all 51 steps).

Blast radius: anyone using `treat_num` with `max_capacity` below the number of eligible agents, which is the whole point of the argument. Treatment volume silently drops by roughly 10x after a few years, and memory/time grow with the queue (113,633 entries for 5000 agents after 50 steps; the `e not in treat_inds` filter is also O(queue × treated)).

**Fix**: in `add_to_queue()`, drop queued agents who are no longer in `check_eligibility()` and only append accepted agents who are not already in the queue (e.g. keep the queue as `ss.uids` and use `setdiff`/`intersect`, preserving order).

## Medium severity

### 3. Screening `outcomes` are never reset, so stale positives persist — `interventions.py:229`, `interventions.py:266`

`BaseTest.deliver()` only assigns `self.outcomes` when someone accepts, and `BaseScreening.step()` only calls `deliver()` on its timepoints. Nothing clears `outcomes` between steps, so after a screening round `sim.interventions.screening.outcomes['positive']` keeps returning the same agents on every later timestep (and on any timepoint where nobody accepts). `BaseTriage.step()` does reset its own `outcomes` at the top of every step, and `BaseTreatment.init_pre()` documents outcomes as "Store outcomes on each timestep", so per-timestep semantics are clearly intended. The triage docstrings build eligibility directly from this attribute (`screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']`).

```python
screen = ss.campaign_screening(product=ss.Dx(df=dx_data), prob=0.5, years=2002, name='screening')
sim = ss.Sim(n_agents=2000, start=2000, stop=2006, diseases=dict(type='sis', init_prev=0.3), networks='random', interventions=screen, analyzers=probe(), copy_inputs=False, verbose=0)
sim.run()  # probe logs (year, n_screened this step, len(screening.outcomes['positive'])) each step
```

Actual:

```
(2000, 0, 0)
(2001, 0, 0)
(2002, 992, 476)
(2003, 0, 476)
(2004, 0, 476)
(2005, 0, 476)
(2006, 0, 476)
```

Expected: `len(outcomes['positive'])` is 0 in 2003-2006, since no screening happened on those steps.

Blast radius: any downstream intervention whose eligibility reads `screening.outcomes` (the documented triage pattern, or a custom treatment). With a campaign screen, or a routine screen with a later `end_year`, the same positives are re-triaged/re-treated every step. Currently masked for triage by finding 1.

**Fix**: reset `self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}` at the start of `BaseScreening.step()` (as `BaseTriage.step()` does), or at the start of `BaseTest.deliver()` plus in the non-timepoint branch.

### 4. `RoutineDelivery` overwrites user-supplied `years`, so non-annual anchor years crash — `interventions.py:132`

The docstring says `years` are the "years over which to interpolate probabilities" and "`prob` ... if array, must match `years`". But after deriving `start_year`/`end_year` from them, line 132 replaces `self.years` with `sc.inclusiverange(start_year, end_year)` (annual steps), and the length check on line 137 then compares that against `prob`. Only the special case where the user's `years` are already every integer year works.

```python
vx = ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=[0.1, 0.5], years=[2002, 2006])
sim = ss.Sim(n_agents=2000, start=2000, stop=2010, diseases='sis', networks='random', interventions=vx, verbose=0)
sim.init()
```

Actual:

```
{'prob': [0.1, 0.5], 'years': [2002, 2006]} -> ValueError Length of years incompatible with length of probabilities: 5 vs 2
{'prob': [0.1, 0.3, 0.5], 'years': [2002, 2004, 2006]} -> ValueError Length of years incompatible with length of probabilities: 5 vs 3
{'prob': array([0.1, 0.2, 0.3, 0.4, 0.5]), 'years': array([2002, 2003, 2004, 2005, 2006])} -> prob per step: [0.1 0.2 0.3 0.4 0.5]
```

Expected: `years=[2002, 2006], prob=[0.1, 0.5]` should interpolate linearly from 0.1 in 2002 to 0.5 in 2006, which `sc.smoothinterp(self.yearvec, years, prob, smoothness=0)` on line 144 would do if given the user's years.

Blast radius: all routine interventions (`routine_vx`, `routine_screening`, `routine_triage`) using a scale-up defined by sparse anchor years, which is the main reason to pass `years` at all.

**Fix**: keep the user's `years` array for interpolation (only use `inclusiverange` when `years` was not supplied), and compare `len(prob)` against the original `years`.

### 5. `RoutineDelivery` crashes on date-based sims with default start/end — `interventions.py:107-119`

When `start_year`/`end_year` are not given they default to `sim.t.start`/`sim.t.stop`. For a sim created with dates (e.g. `start='2000-01-01'`), those are `ss.date` objects, which are not `ss.TimePar`, so they pass through line 116-117 unconverted and `np.isclose(start_year, yearvec)` raises when it tries to subtract a float from a date. Numeric-year sims work only because `sim.t.start` is then `years(2000)`, a `TimePar`.

```python
for start, stop in [(2000, 2010), ('2000-01-01', '2010-01-01'), (ss.date(2000), ss.date(2010))]:
    vx = ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=0.2)
    sim = ss.Sim(n_agents=500, start=start, stop=stop, diseases='sis', networks='random', interventions=vx, verbose=0)
    sim.init()
```

Actual:

```
start=2000: OK, sim.t.start=years(2000), timepoints=[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
start='2000-01-01': sim.t.start=<2000.01.01> -> TypeError Attempted to subtract "2000.0" (<class 'float'>) from a date, which is not suppo
start=<2000.01.01> : sim.t.start=<2000.01.01> -> TypeError Attempted to subtract "2000.0" (<class 'float'>) from a date, which is not suppo
```

Expected: the same timepoints as the numeric-year sim.

Blast radius: every `routine_vx`/`routine_screening`/`routine_triage` with default dates in a sim defined by calendar dates, which is the normal style for day-resolution models.

**Fix**: convert `ss.date` inputs to float years (e.g. `start_year.years` / `ss.date(...).years`, as `CampaignDelivery` effectively does via `sim.t.yearvec`) alongside the `TimePar` conversion on lines 116-117.

### 6. Campaign years outside the sim window fire at the nearest boundary timestep — `interventions.py:176`

`CampaignDelivery.init_pre()` maps campaign years to timesteps with `sc.findnearest(sim.t.yearvec, years_float)` and never checks the result. A campaign year after the sim ends maps to the last timestep, and one before the start maps to the first, so the campaign runs at a time it was never scheduled for. `RoutineDelivery` explicitly rejects out-of-window years ("Years must be within simulation start and end dates"); the campaign docstring says it "delivers only at the specified years".

```python
vx = ss.campaign_vx(product=ss.simple_vx(efficacy=0.9), prob=0.5, years=[2030])
sim = ss.Sim(n_agents=2000, start=2000, stop=2010, diseases='sis', networks='random', interventions=vx, verbose=0, copy_inputs=False)
sim.run()
```

Actual:

```
campaign years: [2030]  timepoints: [10]  -> vaccinated in year(s): [2010.]  n vaccinated: 986
```

Expected: no one vaccinated (or an error, as `RoutineDelivery` gives).

Blast radius: scenario runs where the campaign schedule is fixed (e.g. `years=[2025, 2030]`) but the sim horizon varies; a shorter run silently gets an extra campaign at its final step, inflating coverage/impact.

**Fix**: drop (or raise on) campaign years that fall outside `[yearvec[0], yearvec[-1]]` (with a half-timestep tolerance) before calling `findnearest`, filtering `prob` in step.

## Low severity

### 7. `CampaignDelivery`'s documented `interpolate` argument does nothing — `interventions.py:165`

The docstring documents `interpolate (bool): if True, interpolate probabilities between campaign years (default True)`, and `__init__` stores it as `self.interpolate`, but nothing in Starsim reads it (`grep -rn interpolate starsim/` finds only these two lines in this file). Passing `interpolate=False` gives identical results to `interpolate=True`.

```python
ss.campaign_vx(product=ss.simple_vx(), prob=[0.2, 0.8], years=[2002, 2008], interpolate=False)  # same behaviour as interpolate=True
```

Blast radius: small; campaigns only fire at their listed years, so there is nothing to interpolate, but users who set the flag get no indication it is inert.

**Fix**: remove the argument (with a deprecation warning), or remove it from the docstring.

## Verified clean

Hypothesised and tested, found correct: `RoutineDelivery` annual-to-per-step probability conversion (`1-(1-p)**dt`) and the `adj_factor` extension past `end_year` for `dt=0.5` (covers 2005.0-2007.5 for `start_year=2005, end_year=2007`); interpolation with annual `years` on a half-year timestep; default whole-sim routine delivery on numeric-year sims; `BaseVaccination.step()` indexing of `prob` by position in `timepoints` and the MRO in `routine_vx`/`campaign_vx`/`routine_screening` (the subclass `prob` correctly overrides the `RoutineDelivery`/`CampaignDelivery` placeholder); `BaseTest.check_eligibility()` returning a `BoolArr` (e.g. `sim.people.female`) is handled correctly by `coverage_dist.filter()` (924 female / 0 male screened); `BaseScreening` results (`n_screened`, `n_dx`) are zero before `start_year` and populated after; `Intervention.check_eligibility()` conversion of `BoolArr` to UIDs. `BaseTreatment.get_accept_inds()` uses only `prob[0]`, but `treat_num` has no timepoints to index an array against, so this was not recorded. The `Vx.administer(uids)` vs `BaseVaccination`'s `administer(sim.people, uids)` call signature mismatch belongs to `products.py` and was not recorded here.
