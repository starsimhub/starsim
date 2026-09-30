# `starsim/library/networks/` bug audit

Audit of `spatial.py` (92 lines), `theoretical.py` (121 lines, including Cliff's uncommitted `rng_ints`/`combine_rands` edits, audited as-is), and `household.py` (397 lines) under `starsim/library/networks/`, on branch `rc3.6.2` (HEAD `3d8dc9d5`), Starsim 3.6.1 (editable install), numpy 2.4.6, sciris 3.3.0. Method: line-by-line reading together with `ss.Network`/`ss.DynamicNetwork` in `starsim/networks.py` and `ss.utils.combine_rands()`, then a repro script for every candidate (scripts in the session scratchpad under `library/`). Only defects the repro demonstrated are recorded.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `ErdosRenyiNet.add_pairs()` | Edges use positions in `born_uids` as UIDs, so after any deaths or births edges go to dead agents and high-UID agents never get contacts | 67-68 |
| 2 | High | `DiskNet.step()` | Wall reflection only handles an overshoot of less than 1, so with `v*dt > 1` (including the docstring example at the default dt) agents leave the unit square and the network collapses | 61-78 |
| 3 | Medium | `DiskNet.add_pairs()` | Pairs are built from raw slots `0..len(x)-1` instead of active UIDs: dead agents keep edges and newborns never get any | 85-86 |
| 4 | Medium | `HouseholdNet.add_pairs()` | Overwrites the age of every agent, including unborn fetuses and already-pregnant women created by `ss.Pregnancy(burnin=True)` | 170, 197 |
| 5 | Low | `ErdosRenyiNet.__init__()` | Documented `dur (dur/Dist)` rejects a `Dist`, so the `isinstance(self.pars.dur, ss.Dist)` branch is unreachable | 43, 71 |

## High severity

### 1. `ErdosRenyiNet` treats indices as UIDs — `theoretical.py:67-68`

`idx1, idx2 = np.triu_indices(n=len(born_uids), k=1)` gives positions within `born_uids`, but `p1 = idx1[edge]` and `p2 = idx2[edge]` are appended directly as agent UIDs, never mapped through `born_uids[...]`. That is only correct while the active born population is exactly `0..N-1`. After deaths, edges point at dead UIDs. After births, agents with UID ≥ the number of active born agents can never receive a contact. (The random draws themselves are correctly indexed by `born_uids`.)

```python
sim = ss.Sim(n_agents=1000, rand_seed=1, dt=ss.years(1), start=2000, stop=2010,
             demographics=[ss.Births(birth_rate=ss.freqperyear(40)), ss.Deaths(death_rate=ss.freqperyear(40))],
             diseases='sis', networks=ssl.ErdosRenyiNet(p=0.005))
sim.run(verbose=0)
```

Actual:

```
ErdosRenyiNet: agents alive=994, max uid=1499; edges=2289; edges with a dead endpoint=1300; max uid in edges=974; alive agents with uid>=994 (never eligible)=346
```

Expected: no edges with dead endpoints; every living born agent eligible.

Blast radius: any `ErdosRenyiNet` in a sim with demographics. Here over half the edges are wasted on the dead, and a third of the living population is isolated from the network.

**Fix**: `p1 = born_uids[idx1[edge]]`, `p2 = born_uids[idx2[edge]]`. The `dur` Dist branch should likewise draw with the mapped UIDs.

### 2. `DiskNet` agents escape the square when `v*dt > 1` — `spatial.py:61-78`

Each wall is handled by a single reflection (`x = 2 - x` for `x > 1`, `x = -x` for `x < 0`), which is only valid when the overshoot is less than 1. The default velocity is `ss.freq(0.05, unit=ss.day)`, so `v*dt = 18.25` at the default dt of 1 year (as in the class docstring example) and 1.52 at monthly dt. Positions then run far outside `[0, 1]`, and since edges require distance `< r`, the network empties out.

```python
sim = ss.Sim(n_agents=1000, diseases='sis', networks=ssl.DiskNet(r=0.05), rand_seed=1, stop=2005)  # docstring example, default dt
sim.run(verbose=0)

for dt in [ss.days(7), ss.month]:
    sim = ss.Sim(n_agents=1000, diseases='sis', networks=ssl.DiskNet(r=0.05), rand_seed=1, dt=dt, start=2000, stop=2002)
```

Actual:

```
v*dt = 18.25
x range 0.013452113 830.59766  y range 0.065237164 830.59106
fraction of agents outside unit square: 1.0
edges at end: 10  expected ~ 3923
SIS prevalence: 0.0
dt=7: v*dt=0.350; fraction outside unit square=0.000; edges=3832
dt=1: v*dt=1.521; fraction outside unit square=0.178; edges=2792
```

(The second `dt=1` line is the monthly run; `dt` prints as 1 month.)

Expected: agents stay in the unit square at any dt, giving roughly 3900 edges.

Blast radius: the documented example, and any sim with dt of about 3 weeks or more at the default velocity. The epidemic silently dies out.

**Fix**: reflect with a triangle-wave fold that handles any displacement, e.g. `x = x % 2; x = np.where(x > 1, 2 - x, x)`, flipping the heading's x- or y-component when the number of wall crossings is odd. Alternatively, loop the reflections until everyone is inside.

## Medium severity

### 3. `DiskNet` builds pairs from raw slots rather than active agents — `spatial.py:85-86`

`np.triu_indices(n=len(self.x), k=1)` indexes `self.x.raw`/`self.y.raw` with positions `0..len(x)-1`, where `len(x)` is the number of active agents. Once agents die or are born, those positions no longer correspond to the active UIDs. Dead agents keep receiving edges (at stale positions), and agents in the upper slots are never paired.

```python
sim = ss.Sim(n_agents=1000, rand_seed=1, dt=ss.days(7), start=2000, stop=2005,
             demographics=[ss.Births(birth_rate=ss.freqperyear(40)), ss.Deaths(death_rate=ss.freqperyear(40))],
             diseases='sis', networks=ssl.DiskNet(r=0.05))
sim.run(verbose=0)
```

Actual:

```
len(x)= 985  raw slots= 1500  alive= 985
edges with a dead endpoint: 1271 of 3720 ; max uid in edges: 985
alive agents with uid >= len(x), never given contacts: 182
```

Expected: 0 edges with dead endpoints; all 985 living agents eligible.

Blast radius: any `DiskNet` with demographics.

**Fix**: take `uids = self.x.auids` (or `self.sim.people.auids`), compute distances on `self.x[uids]`/`self.y[uids]`, and emit `uids[i]`, `uids[j]`.

### 4. `HouseholdNet` rewrites the ages of fetuses and pregnant women — `household.py:170, 197`

`pop_size = len(ppl)` counts every agent, including the unborn agents that `ss.Pregnancy(burnin=True)` creates before networks initialize. `ppl.age[all_uids] = ages_flat[gather]` then gives each fetus a DHS household-member age, so it becomes a child or adult. Women who were already pregnant are also reassigned random ages. `dynamic=True` (the default) requires `Pregnancy`, and `init_post()` (lines 129-139) explicitly anticipates women pregnant at initialization, so this is in-contract usage.

```python
sim = ss.Sim(n_agents=3000, rand_seed=1, dt=ss.month, start=2000, stop=2005,
             demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30), burnin=True), ss.Deaths()],
             diseases='sis', networks=[ss.PrenatalNet(), ssl.networks.HouseholdNet(dhs_data='default')])
sim.init()
```

Actual:

```
pregnant at init: 16  of which age<15 or >50: 10  male: 0  ages sample: [58.5 41.5  4.4 73.  68.4 12.4 37.5 43.8]
agents with age<0 (unborn) at init: 0  n_agents total 3016
initial fetuses: ages after HouseholdNet init: [29.6  5.6  0.7 32.1 68.9 26.  12.3 58.2]  parent set: 16  household id set: 16
without HouseholdNet: agents with age<0 at init: 16
```

Expected: the 16 fetuses keep negative ages (as without `HouseholdNet`), and pregnancies stay on women of reproductive age.

Blast radius: `HouseholdNet` with `Pregnancy(burnin=True)`. There are pregnant 4-year-olds and 73-year-olds, and fetuses become household members with contacts before birth. The class docstring warns that ages are overridden, but not that the required companion module's state is corrupted. With `sexes` data, pregnant agents could also become male.

**Fix**: assign DHS ages and sexes only to born agents (`ppl.age >= 0` and not `pregnant`), or at minimum exclude unborn agents from `pop_size`/`all_uids`. Pregnant women should be matched to adult-female DHS slots, or the pregnancies re-drawn after ages are assigned.

## Low severity

### 5. `ErdosRenyiNet` rejects the documented `Dist` duration — `theoretical.py:43, 71`

The docstring says `dur (dur/Dist)`, and `add_pairs()` has an explicit `isinstance(self.pars.dur, ss.Dist)` branch. But the default is `ss.years(0)`, and `update_pars()` refuses to replace a timepar with a distribution:

```python
ssl.ErdosRenyiNet(p=0.01, dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(0.1)))
```

Actual: `TypeError: Updating timepar 0 from <class 'starsim.time.years'> to <class 'starsim.distributions.lognorm_ex'> is not supported`. Expected: variable edge durations.

Blast radius: users following the docstring; loud error, no silent corruption. Scalar `dur` works (edges roughly double with `dur=years(2)`: 5022 vs 10003).

**Fix**: default `dur` to a distribution, e.g. `ss.constant(ss.years(0))`, so both forms are accepted, or drop the Dist branch and the docstring claim.

## Verified clean

ErdosRenyiNet edge density with a static population matches `p` (5100 edges vs 4995 expected, N=1000, p=0.01), and the working-tree `uint64` draw plus `combine_rands()` produces uniform [0, 1] values. Scalar `dur` is correctly converted to timesteps. NullNet builds `n` self-edges with zero beta and validates the size. DiskNet wall reflection formulas are correct for overshoot < 1, and `v * dt` converts units correctly. HouseholdNet: `prob_move_out` accepts float or `bernoulli`, and it and `update_freq` are honoured via the explicit-argument frame inspection. `prob_move_out` changes household counts (1038 vs 1055). Newborns get their mother's household ID and are connected to exactly the living household members (94/94 checked). `ti_move_out_check` prevents repeat move-outs. Truncating the last household makes the total exactly `pop_size`.
