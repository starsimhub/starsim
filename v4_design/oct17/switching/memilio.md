# MEmilio: would its users switch?

## Who this user is

A modeler at DLR or HZI (or a group that uses their code) who writes Python scripts against MEmilio's C++ models (`memilio.simulation.osecir`, `osecirvvs`), and sometimes C++ when a model needs to change. Their everyday model is the age-resolved ODE-SECIR(VVS) with the six RKI age groups, put on a graph of Germany's 400 counties (`set_nodes`, `set_edges`) with commuting from the Federal Employment Agency's commuter data, contact matrices by location (home, school, work, other), dampings for NPIs, dynamic NPIs that switch on in a county when its 7-day incidence passes a threshold, school holidays per state, and a `GraphParameterStudy` of 10 to 100 runs whose parameters are drawn from uniform ranges, saved as p05/p25/p50/p75/p95 per county to HDF5 and drawn as maps. Data come from MEmilio's own `memilio-epidata` (RKI cases, DIVI ICU, Destatis populations, commuters). They also have the agent-based model, IDE and LCT models, and surrogate neural networks trained on the ODE. They chose MEmilio because it is fast (C++), validated in papers, built for German policy advice, and does the whole chain from downloaded data to percentile maps. They do not fit by MCMC in MEmilio itself.

## The first day

The two vignettes (`lingua_franca_design/vignettes/memilio/secir`, `secir_ageres`) are models this user recognizes at once: the seven SECIR states, branches with MEmilio's names turned into `p=[...]`, seasonality from `StartDay`, a damping smoothed over the day before (`ss.timeseries(..., mode='cosine')`), and test-and-trace capacity as `np.clip` and `np.cos` of a count. The age-resolved version is 21 lines against 45 and agrees within 1.5 people of 10,000 (the interpolation error of MEmilio's own reference). A whole command takes 0.086 s against 0.47 s, but MEmilio's `simulate()` alone is 0.1 ms, against 0.2 ms for a v4 restart, so this user will read the times as "about the same, and both instant". The translation is more readable than the original: the equations are in the script, where in MEmilio they are in `cpp/models/ode_secir/model.h`. That is a real gain for this user, who today has to read C++ to know what their model does.

What they would notice: the vignette's open finding 4 (the test-and-trace threshold is on a flow, which has to be rewritten as `0.91/2*INS.n`), and that age-specific `RelativeTransmissionNoSymptoms` has no declared form (`deferred.md`, "Gaps found while writing the importers"). Both bite in their real model, where every parameter differs by age.

The front door hurts. `docs/` is entirely Starsim 3, so this user meets v4 only through `v4_design/design/` (5,457 lines of spec), the vignettes, and `skills/import-model/references/memilio.md`. The importer guide is the best page for them and they would not find it from the docs. The overview's own example fails at the scenarios line (Fable 2 `03` finding 1).

## The second week

All scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/memilio/`. The shared setup (`setup.py`) is a synthetic Germany: 400 counties with lognormal populations summing to 83 million, positions, the six RKI age groups, MEmilio's own German contact matrices (`_envs/memilio/src/data/Germany/contacts/baseline_*.txt`, summed over the four locations), gravity commuting to each county's 20 nearest neighbours with 15% commuting out, 50 exposed per age group in 10 seeded counties, parameters at the midpoints of the ranges in `2020_npis_sarscov2_wildtype_germany.py`, and contacts halved from day 30.

### 1. Germany's counties by age, with commuting

MEmilio (`run_memilio.py`): `osecir.MobilityGraph`, 400 nodes, 8,000 edges, `MobilitySimulation(graph, dt=0.5).advance(90)`. v4 (`v4_secir.py`), the part that differs from the vignette:

```python
secir.stratify(age=ages, split=list(d['ages']))
secir.stratify(patch=ss.geo(names=names, pop=d['pop'], x=d['x'], y=d['y']),
               mobility=ss.Mobility(matrix=0.5*d['M'], kind='commute'))   # MEmilio's commuters are away half the day
sim = ss.Sim(secir, networks=ss.contacts(age=d['cm']), total_pop=d['pop'].sum(), stop=ss.days(90))
```

Getting there took three tries. `dur=ss.days(ss.by(age=[...]))` raises `TypeError: Expecting scalar, array, or ss.dur as value, not by(...)` with no hint; `ss.by(age=[ss.days(x) for x in ...])` works. Then `sim.stratify(...)`, the spelling in `best-practices/metapopulation.md`, broke `sim.new()` and therefore `ss.parallel()` (below).

| | MEmilio | v4 |
|---|---|---|
| 50 counties: run | 0.17 s | 0.17 to 0.21 s |
| 400 counties: build | 13.4 s (the Python loop of `add_node`/`add_edge`) | 0.08 s (plus 1.2 s of imports) |
| 400 counties: run, 90 days | 1.54 s | 25 s (33 s under the profiler, all inside the kernel; a second run after `restart()` is no faster) |
| 400 counties: peak memory | 125 MB | 592 MB |
| Model lines | about 45 | about 25 |

Agreement: with commuting turned off, the seeded counties agree to 1e-8 (symptomatic on day 20: 461.25427 against 461.25426) and the national totals to 0.13% of the peak, so the SECIR, the age matrix, and the damping are exact. With commuting, the national curves agree within 0.2% (50 counties) and 0.8% (400) of the peak, but county by county (counties whose peak is over 10 people) the median largest difference is 11% of that county's peak and the worst 25 to 36%. Scaling the time away by 0.5 or 2 makes it worse (31% and 63%), and `kind='visit'` or `'force'` worse still, so 0.5 with `'commute'` is the right translation and the rest is the model: MEmilio moves commuters into the destination node for half of each day and integrates them there, v4 weights exposure by the share of time away. These are two different, both standard, coupling models; the difference is concentrated in counties whose epidemic is started by imports.

Why the run is slow: `Mixing.within_places()` (`solvers.py:167-183`) builds `np.kron(share, term.matrix)` and `np.kron(share, np.eye(size))`, two dense (places × ages)² matrices, and the kernel multiplies by them at every RK4 stage. At 400 × 6 that is 2,400² = 5.8 million entries each; at 1,000 counties I measured two matrices of 288 MB each; at 3,000 places × 6 ages it would be 2.8 GB each (computed, not run). MEmilio's cost grows with the number of edges, v4's with the square of the number of groups. Opus (`notes/dimensions-and-mixing.md` finding 7) found the same pattern for movers (aging, migration); none of the reviews has it for the transmission matrix itself.

### 2. A parameter study with percentiles per county

MEmilio's way is `UncertainValue.set_distribution(ParameterDistributionUniform(lo, hi))` on each age group's value, then `GraphParameterStudy(graph, 0, 90, 0.5, num_runs).run()` and `ensemble_percentile`. In v4 (`v4_study.py`) the parameters are named `ss.par`s and the draws are made by hand:

```python
draw = lambda k: A(rng.uniform(lo[k], hi[k], 6)) if k == 'beta' else A(rng.uniform(lo[k], hi[k]))
msim = ss.parallel(sim, pars=[{k: draw(k) for k in lo} for _ in range(10)])
q = msim.results.secir.n_ISy.select(patch='c007').quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
```

At 50 counties, 10 runs (uniform draws of the transmission probability by age, `TimeExposed`, and `TimeInfectedNoSymptoms`): v4 2.8 s, MEmilio 4.2 s. National peak of symptomatic people, p05/p50/p95: v4 5,475 / 9,696 / 22,619; MEmilio 4,685 / 9,763 / 19,728 (different draws, 10 runs each, so agreement in distribution). Percentiles per county and day come out as a 50 × 5 × 91 array in one line, which is pleasant. At 400 counties the same study would take about 300 s in v4 (10 runs of 25 to 33 s) against about 15 s in MEmilio; I did not run it.

Two things went wrong on the way. First, `ss.parallel(sim, ...)` on a sim stratified with `sim.stratify(...)` fails with `KeyError: 'The model is not stratified by "patch"; its dimensions are []'`: `Sim.stratify()` (`sim.py:535`) records the call in `self.pars.stratify` but not in `self.inputs` (`sim.py:86`), and `new()` rebuilds from `self.inputs` (`sim.py:431`), so every copy loses the stratification. Repro in `newbug.py`, for a plain two-place SIR, on both the restart path and the copy path. Stratifying the disease (`secir.stratify(...)`) works. None of the reviews has this. Second, there is no way to say "draw this parameter from its distribution once per run": `prior=` is used only by calibration (the flepiMoP vignette's open finding 1; Fable `12` lists it).

### 3. Local dynamic NPIs

MEmilio's `DynamicNPIs` (`set_threshold(200.0, dampings)`, `duration=14`, `interval`, `base_value=100000`) act per county: a county whose 7-day incidence per 100,000 passes 200 gets its own damping for at least 14 days. I did not run MEmilio's version. The spec says a rule with memory is `step()` code (`08-interventions.md` §Triggers), and `when=` returns one boolean for the whole sim, so a per-county rule is a class plus a modifier (`v4_npi.py`):

```python
class LocalNPI(ss.Intervention):
    compartmental = True                      # without it, auto runs the model as 10,000 agents
    def step(self):
        ti = self.sim.ti
        new = self.sim.results.secir.symptoms.by('patch')
        inc = np.array([new[c].values[max(0, ti-7):ti].sum() for c in names])/d['pop']*1e5
        self.until[inc > self.threshold] = ti + self.hold
        self.factor = np.where(self.until >= ti, 0.6, 1.0)
effect = ss.modify(beta, multiply=lambda sim: sim.interventions.npi.factor[sim.group_coords[:, 1]])
```

It works: the national peak of symptomatic people falls from 750,843 to 149,298 at 50 counties, and all 50 counties trigger. But it took four attempts, each with a poor message. Without `compartmental = True` (documented only in `implementation.md`'s departures), `solver='auto'` silently chose agents for an 83-million-person model and crashed with `AttributeError: 'Agents' object has no attribute '_uids'`; `solver='ode'` said `E804 ... only works on agents` without saying how to declare otherwise. `multiply=` returning `ss.by(patch=...)` also flipped the model to agents and crashed; returning one value per county raised `operands could not be broadcast together with shapes (300,) (1,50)`; only indexing by the internal `sim.group_coords` worked. The run is then stepped from Python: 1.1 s at 50 counties (0.2 s in the kernel without it) and 60 s at 400.

### 4. Bringing their own scripts

`starsim/interop/dump_memilio.py` on MEmilio's own examples (`imp/`): `ode_secir_ageres.py` dumps and `ss.load(..., format='memilio')` reproduces it, with a clear account of how each flow was read. `ode_secir_mobility.py` (a `MobilitySimulation`) and `ode_secir_parameter_study.py` (a `GraphParameterStudy`) both say `ran no MEmilio simulation, so there is no model to write`, which is wrong (each ran one); `ode_secirvvs_simple.py` crashes the dumper with `TypeError: Unable to convert function return value to a Python type!`. So the importer covers single-node scripts; this user's production scripts are graphs, studies, and SECIRVVS, and none of them import.

## What would win them over

- **The model is in the script.** `S >> E @ beta * contacts * (0.7*INS + risk*ISy)` and branches as `p=[...]` replace parameters set on a C++ model whose equations live in `model.h`. A new compartment, a new branch, or a flow that MEmilio's model does not have (a second vaccine dose, an extra severity stage) is a line, not a C++ change and a rebuild. For a group that keeps writing new `ode_*` model directories in C++, this is the strongest argument.
- **Exact agreement where the models are the same.** The age-resolved SECIR to 1.5 in 10,000, the per-county SECIR without commuting to 1e-8, and the imported examples to 1.6e-7 (`implementation.md`).
- **One model, several solvers.** The same declaration runs as an ODE, a stochastic compartmental model, or agents (`solver=`), which MEmilio does with separate ODE, SDE, and ABM codebases. `stages=` gives MEmilio's LCT models (Erlang stages) as a keyword.
- **Inference they do not have.** NUTS on ODE models with gradients through the run (the flepiMoP report fits four parameters to three weekly series per place in 70 s), and particle MCMC for stochastic compartmental models. MEmilio has none: its source tree has no MCMC, ABC, or likelihood code (searched `_envs/memilio/src`).
- **Small studies are as fast or faster.** At 50 counties a run is 0.2 s in both, and a 10-run study 2.8 s against 4.2 s, with percentiles by county in one call and no Python loop over `add_edge`.
- **`describe()` and the honest importer.** Every guess is marked and every untranslated item is listed, which matches this group's validation culture.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Best point of the tradeoff? | Already in a review? |
|---|---|---|---|---|
| Transmission within places with commuting uses dense (places × ages)² matrices (`solvers.py:181-182`): 16 to 21 times slower than MEmilio at 400 counties × 6 ages, 2 × 288 MB at 1,000, about 2 × 2.8 GB at 3,000 | Performance bug | Yes, medium: apply the age matrix within each place and the time share between places as two small products (`share @ (C ⊗ I)` as a reshape and two `dot`s), never forming the Kronecker product | No; a factored product costs O(places × ages² + edges × ages) | Partly: Opus dims 7 has the same pattern for movers, not for mixing |
| Commuting is a time share, not MEmilio's two-phase exchange; counties differ by a median of 11% of their peak (worst 25 to 36%), national totals within 0.8% | Inherent tradeoff (two standard coupling models) | Could add an exchange mode with a half-day step and day/night mixing (the metapopulation trade-offs already name it) | Yes for most uses; say the size of the difference in the guide and `describe()` | No |
| `sim.stratify()` is lost by `sim.new()`, so `ss.parallel(sim, ...)` and `calibrate()` fail on the best-practices spelling | Bug | Yes, small: record the call in `self.inputs` | n/a | No |
| Per-county threshold NPIs with a hold need a class, `compartmental = True`, and `sim.group_coords` indexing; the run is then stepped from Python (2 to 5 times slower) | Missing feature, docs | Yes, small to medium: a per-group `when=`/`where=` on a result (7-day incidence per place) with `hold=`, tabulated in the kernel | No: "memory is `step()` code" is simple for one global switch, but a per-place hold is the standard case in this field and deserves a keyword | Fable `11` finding 3 and `06` finding 4 (the silent fall-back to agents), not the feature |
| A module or modifier that fails on groups makes `solver='auto'` choose agents silently, at 10,000 agents for 83 million people, then crash | Bug / design | Yes, small: an exception while trying on groups is an error, not a reason to change paradigm | No | Fable `06` finding 4, `11` finding 3; Fable 2 guesses table |
| No draw of uncertain parameters once per run (`prior=` is for calibration only) | Missing feature | Yes, small: `ss.parallel(sim, n_runs=, priors=True)`, values in `msim.run_pars` | No | flepiMoP vignette finding 1; Fable `12` |
| `ss.days(ss.by(...))` raises a `TypeError` with no hint | Bug (message) | Yes, trivial | n/a | No |
| The importer takes single-node ODE scripts only; graphs and parameter studies say "ran no MEmilio simulation", SECIRVVS crashes the dumper | Missing feature | Yes, medium: record `MobilityGraph` nodes and edges as `patch` with a matrix, and `GraphParameterStudy` distributions as priors | Worth it: graphs are what this user runs | Opus and Fable 2 propose moving interop out of the core, which does not change this |
| German data: `ss.geo('Germany', level=2)` needs `starsim_data[spatial]` (`ImportError` here); populations are WorldPop, not Destatis; no commuter matrix, no RKI or DIVI data, no school holidays | Ecosystem | Partly: bundle level 2 for Germany with Destatis populations and the commuter matrix; case data stays MEmilio's epidata (which could be read as `ss.data`) | n/a | No |
| A parameter of the infectious person's group by age, and a transition's flow in a rate | Missing feature | Yes, small to medium | n/a | `deferred.md` lists both; not in the reviews |
| `docs/` is all Starsim 3; the importer guide is in `skills/` | Docs | Yes | n/a | Fable 2 `09` finding 4; Fable `12` |
| Maturity: MEmilio has years of German policy use, papers, an ABM, IDE models, and surrogates | Ecosystem | Over time | n/a | No |

## Verdict

Today: no. Their production run (400 counties × 6 ages, a 100-run study, local NPIs) is 16 to 21 times slower per run than MEmilio, needs hand-written code for the parameter draws and the dynamic NPIs, and none of their scripts import.

After the four reviews' fixes: still no for that workflow, because no review covers the dense mixing matrix, `sim.stratify()` being lost, per-place NPIs, or the graph importer; but closer for new models, where the declared language already beats writing C++.

A clear win would take the factored mixing (so 400 × 6 is as fast as MEmilio), forward prior draws, a per-place threshold modifier with a hold, and a graph-aware importer; then v4 offers what MEmilio cannot: the model in the script, one model under three solvers, and a posterior.

## The demo that would sell it

`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/memilio/demo.py`:

```python
counties = ss.geo('Germany', level=2)                                   # 400 Kreise, Destatis populations, commuter matrix attached
beta = ss.par(ss.by(age=[0.03, 0.06, 0.06, 0.06, 0.09, 0.175]), prior=ss.uniform(0.75, 1.25, relative=True))
S, E, INS, ISy, ISev, ICr, R = ss.states(7)
secir = ss.Disease(infection=S >> E @ beta*(INS + 0.2*ISy), onset=E >> INS @ ss.days(3.3), ...)  # the rest as in the vignette
secir.stratify(age=[0, 5, 15, 35, 60, 80], patch=counties, mobility=ss.Mobility(counties.commuters, kind='commute', away=0.5))
incidence = secir.symptoms.weekly / secir.N                             # a transition's flow, per place, usable in a condition
npi = ss.modify(beta, multiply=0.6, where=incidence > 200e-5, hold=ss.days(14))   # local: each county on its own
sim = ss.Sim(secir, networks=ss.contacts('Germany'), interventions=npi, start='2020-10-01', stop='2021-01-01')
msim = ss.parallel(sim, n_runs=100, priors=True)                        # one kernel call per run, sparse mixing
msim.results.secir.n_ISy.by('patch').quantiles([0.05, 0.25, 0.5, 0.75, 0.95]).to_df().to_parquet('p.parquet')
```

The sell is ten lines for what is about 600 in `2020_npis_sarscov2_wildtype_germany.py`, with the equations visible, finishing in about MEmilio's time. It does not run today. What does: the SECIR by age, the stratification by age and place, `ss.contacts('Germany')` (a named matrix), and `ss.parallel(..., pars=[...])` with quantiles by county (with the disease stratified, not the sim). What does not: `ss.geo('Germany', level=2)` (an `ImportError` without the spatial extra, and no commuter matrix), a transition's flow in a condition, a per-place `where=` on a result with `hold=`, `priors=True`, and the speed at 400 counties.
