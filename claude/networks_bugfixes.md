# `networks.py` bug audit

Audit of `starsim/networks.py` (all 1404 lines: `Network`, `DynamicNetwork`, `SexualNetwork`, `StaticNet`, `RandomExactNet`, `RandomNet`, `RandomSafeNet`, `MFNet`, `MSMNet`, `PrenatalNet`, `PostnatalNet`, `BreastfeedingNet`, `AgeGroup`, `MixingPools`, `MixingPool`) for real, unambiguous bugs only, at commit `3d8dc9d5` (branch `rc3.6.2`, working tree as-is), with starsim 3.6.1 (editable install), numpy 2.4.6, networkx 3.6.1. Method: line-by-line reading plus the relevant parts of `demographics.py` (`Pregnancy`), `loop.py` (step order), `modules.py` (`update_pars`) and `utils.py` (`plot_args`); every finding below was reproduced with a script in the session scratchpad (`networks/r*.py`), and every "actual" output is verbatim from those runs. Style, docstring typos, performance and contrived edge cases are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `PostnatalNet.add_pairs()` / `DynamicNetwork.end_pairs()` | Every postnatal edge lasts one timestep less than `dur`; `dur` equal to one timestep produces no edges at all | 1057, 446 |
| 2 | High | `MSMNet.step()` | Participation and debut age are redrawn for every male on every timestep | 987 |
| 3 | Medium | `MSMNet.__init__()` | `duration` and `acts` defaults have no time units, so they are per-timestep and change with `dt` (docstring: acts "per year") | 938, 940 |
| 4 | Medium | `MSMNet.add_pairs()` | Pairing is deterministic (lower half of available UIDs paired with upper half), so the network is nearly bipartite rather than random | 968-970 |
| 5 | Medium | `RandomSafeNet.add_pairs()` | Documented integer `dur` crashes the sim on the first step | 810 |
| 6 | Medium | `StaticNet.__init__()` / `init_post()` | Passing a networkx generator only works for `(n, p, seed)` generators; generator-specific arguments (`k`, `m`, `d`) are rejected | 516, 536 |
| 7 | Low | `MFNet.__init__()` | Documented `rel_part_rates` parameter is never used | 845 |
| 8 | Low | `Network.plot()` | Documented `**kwargs` "passed to `nx.draw_networkx()`" always raise | 354-358 |
| 9 | Low | `Network`, `MixingPool` docstrings | Three docstring examples raise when run | 84, 89, 1272 |

## High severity

### 1. Postnatal edges are removed one timestep early — `networks.py:1057`, `networks.py:446`

`PostnatalNet.add_pairs()` is called from `Pregnancy.process_delivery()` during the demographics step, which runs *before* the networks step (loop order: demographics → networks → diseases). The newly-added edge then goes through `DynamicNetwork.end_pairs()` in the same timestep, where `dur` is decremented before any transmission has happened. Networks that form their own edges (`RandomNet`, `MFNet`, etc.) call `add_pairs()` *after* `end_pairs()` inside `step()`, so for them an edge with `dur=N` timesteps is present for N transmission steps; for `PostnatalNet` it is present for N−1. With `dur` equal to one timestep (e.g. `dur=ss.months(3)` at `dt=ss.months(3)`, or any `dur<2*dt`) no mother-infant edge ever exists when transmission runs, so postnatal transmission is silently zero.

```python
class Probe(ss.Intervention):  # runs after networks.step(), before diseases.step()
    def init_pre(self, sim): super().init_pre(sim); self.count = Counter()
    def step(self):
        for inf in np.asarray(self.sim.networks.postnatalnet.p2): self.count[int(inf)] += 1

for months in [1, 3, 6]:
    sim = ss.Sim(n_agents=5000, dt=ss.months(1), start=2000, stop=2010,
                 networks=[ss.PrenatalNet(), ss.PostnatalNet(dur=ss.months(months))],
                 demographics=ss.Pregnancy(fertility_rate=50), diseases='sis', interventions=Probe(), verbose=0).run()
    # tally, per newborn, the number of steps its edge was present
```

Actual:

```
PostnatalNet(dur=months(1)), dt=1 month: timesteps each mother-infant edge is present during transmission -> {0: 640}
PostnatalNet(dur=months(3)), dt=1 month: timesteps each mother-infant edge is present during transmission -> {2: 640}
PostnatalNet(dur=months(6)), dt=1 month: timesteps each mother-infant edge is present during transmission -> {5: 640}
```

Expected: `{1: 640}`, `{3: 640}`, `{6: 640}`. (Also seen: at `dt=ss.years(1)` with `dur=ss.months(6)`, the network never contains an edge at any step.)

Blast radius: every user modelling postnatal/breastfeeding-period transmission with `PostnatalNet(dur=...)` (e.g. HIV/syphilis vertical transmission in STIsim-style models) gets a systematically shortened exposure window — 17% short for a 6-month period at monthly `dt`, 100% short for coarse `dt`. The same mechanism also shortens the *initial* batch of edges of every `DynamicNetwork` with `dur>0` (they are created in `init_post()` and decremented at `ti=0` before transmission; e.g. `RandomNet(dur=ss.months(6))` initial edges last 5 steps), but that is a one-off transient. `BreastfeedingNet` overrides `end_pairs()` and is not affected.

**Fix**: have `PostnatalNet.add_pairs()` store `dur + 1` timesteps (i.e. `p.dur/self.t.dt + 1`, and `+1` on the `Dist` branch) to compensate for the same-step decrement, or give `PostnatalNet` an `end_pairs()` that skips edges added this timestep (e.g. by recording `ti_start` and only decrementing edges with `ti_start < self.ti`).

### 2. `MSMNet` redraws participation and debut every timestep — `networks.py:987`

`MSMNet.step()` calls `self.set_network_states()` with no `upper_age`, so `set_network_states()` takes the "all males" branch and redraws `participant` (Bernoulli, default p=0.1) and `debut` for every male on every step. `MFNet.step()` correctly passes `upper_age=self.t.dt.years` so only newborns are assigned. The result is that MSM participation is not a stable individual trait: each step a fresh random ~10% of men participate, and debut age is re-sampled each step (so a man can flip between "debuted" and "not debuted").

```python
sim = ss.Sim(n_agents=5000, start=2000, stop=2010, networks=ss.MSMNet(), diseases='sis')
sim.init()
msm = sim.networks.msmnet
males = sim.people.male.uids
part0 = msm.participant[males].copy(); deb0 = msm.debut[males].copy()
sim.run_one_step(); sim.run_one_step()
print((part0 != msm.participant[males]).mean(), (deb0 != msm.debut[males]).mean())
```

Actual:

```
frac of males whose participation flipped after 2 steps: 0.18355945730247406
frac of males whose debut age changed after 2 steps: 1.0
MFNet frac flipped: 0.0038
```

Expected: ~0 for both MSMNet lines (as for MFNet, only agents younger than one timestep should be (re)assigned).

Blast radius: anyone using `ss.MSMNet`; the size of the MSM "core" is effectively the whole male population sampled over time, which inflates reach and changes epidemic dynamics. `tests/test_networks.py::test_other` only checks that it runs.

**Fix**: in `MSMNet.step()`, call `self.set_network_states(upper_age=self.t.dt.years)`, matching `MFNet.step()`.

## Medium severity

### 3. `MSMNet` duration and acts defaults are unitless, so they scale with `dt` — `networks.py:938`, `networks.py:940`

`MFNet` uses `duration = ss.lognorm_ex(mean=ss.years(15), ...)` and `acts = ss.poisson(lam=ss.freqperyear(80))`, which are converted to timesteps. `MSMNet` uses `duration = ss.lognorm_ex(mean=2, std=1)` and `acts = ss.lognorm_ex(mean=80, std=20)` with no units. Since `end_pairs()` decrements `dur` by one per step and `net_beta()` uses `acts` per step, these mean "2 timesteps" and "80 acts per timestep". The class docstring says `acts` is the "Number of acts per year".

```python
for dt in [ss.years(1), ss.years(1/12)]:
    for net in [ss.MSMNet(participation=0.5), ss.MFNet()]:
        sim = ss.Sim(n_agents=5000, start=2000, stop=2003, dt=dt, networks=net, diseases='sis', verbose=0).run()
        n = sim.networks[0]
        print(type(n).__name__, sim.t.dt, n.edges.dur.mean(), n.edges.acts.mean())
```

Actual:

```
MSMNet dt=1: mean remaining dur (steps)=1.43, mean acts per step=79.14, n_edges=666
MFNet dt=1: mean remaining dur (steps)=11.00, mean acts per step=80.08, n_edges=1659
MSMNet dt=0.08333333333333333: mean remaining dur (steps)=1.42, mean acts per step=79.11, n_edges=670
MFNet dt=0.08333333333333333: mean remaining dur (steps)=143.01, mean acts per step=6.68, n_edges=1659
```

Expected: MSMNet at monthly `dt` should have ~12x the remaining duration in steps and ~6.7 acts per step, as MFNet does. Instead, switching to monthly `dt` makes MSM partnerships 12x shorter in calendar time and 12x more act-intense.

Blast radius: any `MSMNet` user with `dt` other than one year.

**Fix**: give the defaults units, e.g. `duration = ss.lognorm_ex(mean=ss.years(2), std=ss.years(1))` and `acts = ss.lognorm_ex(mean=ss.freqperyear(80), std=ss.freqperyear(20))` (or a Poisson as in `MFNet`), checking that `lognorm_ex` accepts rate-valued parameters.

### 4. `MSMNet` pairing is deterministic and nearly bipartite — `networks.py:968-970`

`add_pairs()` takes the sorted array of available males and pairs `available_m[:n_pairs]` with `available_m[n_pairs:2*n_pairs]`. There is no random draw: the lowest-UID available man is always paired with the median-UID available man, and men in the lower half of the UID range almost never partner with each other. The docstring says "A network that randomly pairs males".

```python
sim = ss.Sim(n_agents=5000, start=2000, stop=2030, networks=ss.MSMNet(participation=0.5), diseases='sis', verbose=0)
sim.init()
n = sim.networks.msmnet
med = np.median(np.asarray(sim.people.male.uids))
# each step: fraction of edges with both partners below / above the median UID
```

Actual:

```
MSMNet: mean fraction of edges with both partners below median UID = 0.021; both above = 0.014 (random pairing: ~0.25 each)
```

Expected: ~0.25 each under random pairing.

Blast radius: all `MSMNet` users; the contact structure is an artefact of UID ordering (which correlates with age/cohort once births are included), rather than random mixing.

**Fix**: shuffle `available_m` with a module RNG (e.g. an `ss.random()` draw keyed on the available UIDs followed by `argsort`, which is also CRN-friendly) before splitting it into `p1` and `p2`.

### 5. `RandomSafeNet(dur=<int>)` crashes — `networks.py:810`

The docstring documents `dur (int/ss.dur)`. `RandomSafeNet`'s default is the plain number `0` (unlike `RandomExactNet`, whose default `ss.years(0)` causes an integer to be converted to a duration), so an integer `dur` stays a plain number and `self.pars.dur/self.t.dt` evaluates `int / ss.dur`, which returns an `ss.freq`. The edges' `dur` array therefore holds a rate object, and `end_pairs()` fails on the first step.

```python
ss.Sim(n_agents=1000, networks=ss.RandomSafeNet(dur=1), diseases='sis', verbose=0).run()
```

Actual:

```
  File "/home/cliffk/idm/starsim/starsim/networks.py", line 446, in end_pairs
    self.edges.dur = self.edges.dur - 1 # dur is stored in units of self.t.dt (timesteps), so decrement by 1 per step
  File "/home/cliffk/idm/starsim/starsim/time.py", line 1728, in __sub__
    raise TypeError(f'Only rates can be subtracted from rates, not {other}. ...')
TypeError: Only rates can be subtracted from rates, not 1. ...
```

(When the network is added in `ss.Sim(...).init()` and inspected via `r3_random_dur.py`, the failure surfaces as `TypeError: '<' not supported between instances of 'freq' and 'freq'`.) Expected: runs, as `ss.RandomNet(dur=1)` does (edges last 1 year = 12 steps at monthly `dt`). `ss.RandomSafeNet(dur=ss.years(1))` works.

Blast radius: anyone following the docstring and passing a number for `dur` to the CRN-safe random network, which is the network recommended for scenario analysis.

**Fix**: make the default `dur = ss.years(0)` (as in `RandomExactNet`), so numbers are converted on `update_pars()`; or convert a plain number explicitly before dividing by `self.t.dt`.

### 6. `StaticNet` cannot pass arguments to most networkx generators — `networks.py:516`, `networks.py:536`

The docstring says "the graph can be created by passing a networkx generator function to Starsim", with extra keyword arguments forwarded (`ss.StaticNet(graph=nx.erdos_renyi_graph, p=0.0001, seed=True)`). But `__init__` passes `**kwargs` to `update_pars()`, which only accepts the predefined `seed`, `p`, `n_contacts` and raises on anything else; and `init_post()` always passes `p` and `seed`. So only generators with exactly an `(n, p, seed)` signature work; any generator needing its own argument (`k`, `m`, `d`) is impossible to use, and generators without `p` or `seed` fail.

```python
ss.Sim(n_agents=500, networks=ss.StaticNet(graph=nx.watts_strogatz_graph, k=4, p=0.1), diseases='sis').run()
ss.Sim(n_agents=500, networks=ss.StaticNet(graph=nx.barabasi_albert_graph, m=3), diseases='sis').run()
ss.Sim(n_agents=500, networks=ss.StaticNet(graph=nx.barabasi_albert_graph), diseases='sis').init()
ss.Sim(n_agents=500, networks=ss.StaticNet(graph=nx.complete_graph), diseases='sis').init()
```

Actual:

```
StaticNet(graph=watts_strogatz, k=4, p=0.1): ERROR ValueError: 1 unrecognized arguments for staticnet: k
StaticNet(graph=barabasi_albert, m=3): ERROR ValueError: 1 unrecognized arguments for staticnet: m
barabasi_albert_graph ERROR TypeError barabasi_albert_graph() missing 1 required positional argument: 'm'
complete_graph ERROR TypeError complete_graph() got an unexpected keyword argument 'seed'
```

Expected: the generator is called as `gen(n=n_agents, k=4, p=0.1, seed=rng)` etc. The warning text on failure ("networkx X not supported. Try using ss.StaticNet() instead.") is also self-referential. The theoretical-networks docstring in `starsim/library/networks/theoretical.py:91` recommends `ss.StaticNet(nx.empty_graph)`, which also fails this way.

Blast radius: users wanting a small-world, scale-free (BA), regular or other standard graph without building it by hand first; the workaround (pre-build the graph and pass it) works.

**Fix**: store unrecognized kwargs separately (e.g. `self.graph_kwargs`) rather than routing them through `update_pars()`, and only inject `p` and `seed` if the generator's signature accepts them (`inspect.signature`), converting `n_contacts` to `p` only when `p` is accepted.

## Low severity

### 7. `MFNet(rel_part_rates=...)` has no effect — `networks.py:845`

`rel_part_rates` is documented ("Relative participation in the network") and defined as a parameter, but nothing in Starsim reads it (only occurrences are the docstring, signature and `define_pars`).

```python
for r in [1.0, 0.1]:
    sim = ss.Sim(n_agents=3000, networks=ss.MFNet(rel_part_rates=r), diseases='sis', verbose=0, rand_seed=2).run()
```

Actual:

```
rel_part_rates=1.0: n_edges=989, participants=2684, final prevalence=0.0090
rel_part_rates=0.1: n_edges=989, participants=2684, final prevalence=0.0090
```

Expected: fewer participants/edges with `rel_part_rates=0.1`.

Blast radius: users who set it expecting reduced participation get identical results silently.

**Fix**: either apply it (e.g. multiply the participation probability by `rel_part_rates` in `set_participation()`) or remove the parameter and its docstring entry.

### 8. `Network.plot(**kwargs)` rejects every keyword — `networks.py:354-358`

`kwargs` is documented as "passed to `nx.draw_networkx()`", but it is first passed to `ss.plot_args()`, which raises on any key it does not know (e.g. `node_size`), and the original `kwargs` dict (not the leftover) is then also forwarded to `nx.draw_networkx()`, so a key `plot_args()` does know (e.g. `figsize`) is rejected by networkx.

```python
sim = ss.Sim(n_agents=200, networks='random', diseases='sis', verbose=0).init()
sim.networks[0].plot(figsize=(4,4))
sim.networks[0].plot(node_size=5)
```

Actual:

```
plot {'figsize': (4, 4)} ERROR ValueError Received invalid argument(s): figsize
plot {'node_size': 5} ERROR KeyNotFoundError Did not successfully convert all plotting keys: ... Unconverted: node_size
```

Expected: both plot. Only `plot()` with no kwargs works.

**Fix**: split the kwargs first: pop the keys `plot_args()` recognizes into a separate dict for `ss.plot_args()`, and pass only the remainder to `nx.draw_networkx()`.

### 9. Three docstring examples raise — `networks.py:84`, `networks.py:89`, `networks.py:1272`

- `ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand') # Alternate method`: the first positional argument is `name`, so this raises `TypeError: Invalid value for name: must be str, not <class 'dict'>`.
- `ss.Network(**network, index=index, ...)`: a `Network` is not a mapping, so this raises `TypeError: starsim.networks.Network() argument after ** must be a mapping, not Network` (`**network.edges` or `**network.to_dict()` would work).
- `MixingPool` example uses `beta = ss.Rate(0.2)`; `compute_transmission()` then calls `beta.to_prob()` on the abstract base class, raising `NotImplementedError: ss.Rate() does not implement _base_prob; see ss.prob, ss.per, or ss.freq` on the first step (the docstring's own `Args` also says to use a float).

Blast radius: users copying the examples. **Fix**: correct the examples (`ss.Network(**dict(...))`, `ss.Network(**network.to_dict(), ...)`, `beta = 0.2`).

## Verified clean

Tested and found correct: `RandomNet`/`RandomExactNet` edge counts and mean degree (10.0 for `n_contacts=10`), persistence with `dur=ss.years(5)`, and conversion of integer and `ss.dur` `dur` values to timesteps at yearly and monthly `dt`; the `RandomNet.get_edges()` equal-stub fast path; `RandomSafeNet` pairing (9998 edges from 5 edges × 2000 agents, 6 incidental self-edges, distinct random values per stub) and its `ss.dur`/`Dist` durations; `MFNet` debut/participation assignment for agents born during the sim at yearly and monthly `dt` (0.90 participation in both newborn and original cohorts), MFNet duration and acts time-unit conversion; `PrenatalNet`/`BreastfeedingNet` running with `Pregnancy`; `PostnatalNet` `name`/`label` kwargs; `MixingPool` with `BoolArr`-returning `src`/`dst` lambdas (identical results to the `.uids` equivalents), explicit-`ss.uids` `src` pruning on death via `remove_uids()`, and the `MixingPools` docstring example; `StaticNet` with its default generator, `p`, `n_contacts` (mean degree 4.05 for 4), and a pre-built graph; `find_contacts()` docstring example; `from_df()` round-trip. Hypotheses considered and dropped as out of scope or not bugs: `RandomNet(dur=<Dist>)` is rejected by `update_pars()` (not documented for that class); `RandomSafeNet` with a `Dist` `dur` gives all of an agent's source edges the same duration (CRN slotting by UID, undocumented input); `DynamicNetwork.end_pairs()` returns the pre-removal edge count (return value unused); `AgeGroup` caching by `ti` across different sims (contrived).
