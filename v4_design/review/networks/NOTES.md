# STIsim networks on Starsim v4

`port.py` runs in ~12 s (`conda run -n starsim python port.py`). It has three parts. (A) builds StructuredSexual from declarative `ss.MFNet` layers. (B) subclasses `sti.StructuredSexual` so it works with a declared v4 disease. (C–F) probe per-act beta, conditions built on the network, the solver choice, and timing. Each attempt that failed is kept in an `attempt(...)` call, and the error is printed.

Env: starsim 4.0.0 (v4 branch) and stisim 1.7.1, both editable. **stisim imports and its v3 classes run on v4**, through the legacy `ss.Infection` path: `ss.Sim(diseases=sti.HIV(), networks=sti.StructuredSexual())` works. `sti.Sim(..., dt=...)` does not: `dt` is routed as an STI/connector par and raises a ValueError. That is a stisim bug, not a v4 one, but it blocks `sti.Sim` on v4.

## Network base class: v3 → v4

The API has barely changed. `Network.__init__(name=, label=)` became `Network(pars=None, edges=None, **kwargs)`. Standard module args such as `name`, `dt`, and `label` go to `update_pars`, and every other kwarg becomes an edge column. STIsim's `super().__init__(name=name)` still works.

Unchanged from v3: `meta`, `edges`, `end_pairs`, `add_pairs`, `net_beta(disease_beta, inds, disease)`, `find_contacts`, and `remove_uids`.

New in v4: `remove_edges`, `eligible()`, and `DegreeTargets` (mixed into `PartnerNet` and `MFNet`).

**A v4 network subclass needs exactly what a v3 one did.** The friction is all on the disease side (below), not in the network API.

## Feature by feature

| STIsim feature (v3) | v4 | Expressible | Rating |
|---|---|---|---|
| Debut by sex `base.py:144-151` (`debut_f`, `debut_m`) | `MFNet(debut=ss.lognorm_ex(mean=lambda self,sim,uids: np.where(female,20,21), std=3))` | declarative (callable par) | awkward: the lambda needs the 3-argument signature just to give a value per sex. Suggest `ss.by(sex=[20, 21])` as a dist parameter |
| Partner number / concurrency by risk group `mf.py:194-232` | `MFNet(n_contacts=, concurrency=, max_contacts=)`: one pooled target for everyone | partial | ugly. A target per group is impossible: a callable `n_contacts` raises `TypeError: loop of ufunc does not support argument 0 of type function` (`degree_dist` uses `n_contacts.pars` directly) |
| Risk groups `mf.py:172-192` (3 groups, sex-specific shares) | none on `MFNet`. Approximated by per-layer `participation` | partial | ugly. Each layer draws participation independently, so the same person is not high-risk in every layer |
| Risk-assortative matching (`p_matched_stable`, `p_mismatched_casual`, `mf.py:278-291`) | `PartnerNet(match=dict(risk=0.9))` exists, but `MFNet` has no `match=` | impossible on MF | `ValueError: 2 unrecognized arguments for mf: match`. `PartnerNet` can't stand in for MF: `match=dict(female=0.0)` still gives 54% same-sex pairs, because `match` only pushes above chance (clipped at 0) |
| Age-assortative matching with age gaps by age group × risk (`mf.py:246-251`, `matchers.py`, 8 matchers) | none | impossible declaratively | — |
| Partnership type stable/casual/one-off, with durations by age × risk (`mf.py:305-319`) | three `MFNet` layers with their own `duration` | partial | awkward. The type comes from the layer rather than the pair. Durations can't depend on age × risk. Each layer adds a beta key and results |
| Sex work, FSW + clients with entry age and duration (`fsw.py:90-...`) | an `MFNet` layer with participation by sex and a 1-month duration | partial | ugly. No entry or exit ages (participation is lifetime). The degree target is pooled across sexes: realized FSW 1.74 clients/step vs clients 0.61, set by the ratio of participants and not controllable |
| Acts per partnership `base.py:26` | `MFNet(acts=ss.lognorm_ex(mean=ss.freqperyear(80), ...))`; per step it is correct (6.7/month) | declarative | clean |
| Condoms by risk pair × year, with `eff_condom` per disease (`base.py:153-159`, `176-208`) | no condom concept in v4 networks. Tried as a per-layer beta multiplier | impossible declaratively | ugly. `0.008*(1-0.9*ss.timeseries(...))` raises `TypeError: unsupported operand type(s) for *: 'float' and 'timeseries'`. A precomputed `ss.timeseries` as a value in the beta dict raises `TypeError: '>' not supported between instances of 'timeseries' and 'timeseries'` (`solvers.py:3940`, `ref = max(...)` on unevaluated values). Bug: values that vary in time are not evaluated inside a per-network beta dict |
| Directional beta `sti.py:205-229` (`beta_m2f`, `rel_beta_f2m`) | `ss.transmission('S -> A', beta=dict(structuredsexual=[m2f, f2m]))` | declarative | clean, and it works on the v3 network (p1 = male). The ratio form `rel_beta_f2m` is lost, so for calibration f2m becomes an independent number unless written as an expression |
| beta for `maternal`/`msm`/`breastfeeding` by network name | the same dict | declarative | awkward: **every network needs a key**, bookkeeping ones too. `PriorPartners` (beta=0 edges) raises `E403 RouteMismatch: ... a network called "priorpartners" with no value`, so `priorpartners=0` has to go in every disease. Suggest skipping networks whose edges all have beta 0, or a `Network(transmits=False)` |
| Whole StructuredSexual as a subclass | `class StructuredSexual(sti.StructuredSexual)` + a `net_beta` override (~40 lines; reuses the 2,100 v3 lines) | subclass | awkward (2 required patches below). Unmodified it fails: `AttributeError: 'Pars' object has no attribute 'eff_condom'`; with `eff_condom=0.9` on the disease: `TypeError: float() argument must be a string or a real number, not 'BinOp'`. A declared disease's `pars.eff_condom` is an `ss.par` (an Expr), not a number, so a network can't do arithmetic with disease pars without `.value` |
| Partner counts, `lifetime_partners` etc. (`mf.py:12-25`) | kept (subclass) | subclass | fine |
| PriorPartners for PN `layered_networks.py:95-127` | reused unchanged | subclass | fine, apart from the beta-key tax |

## Specific questions

**(3) Per-act transmission: bug 1, the most important finding.** In the engine path for declared diseases (`solvers.py:3949`), a bare per-act beta is treated as a *rate in the sim's unit* and becomes `1 - exp(-beta*dt)` before it reaches `net_beta`. `SexualNetwork.net_beta` then applies `1-(1-p)^acts`, with acts already per step. Results:
- beta=0.5 per act arrives as 0.3935 when `unit='month'` (inferred from `dt=ss.months(1)`) and as 0.0408 when `unit='year'`.
- Same model, same seed, changing only `unit=`: SI prevalence in 2020 is **0.311 with unit=month and 0.072 with unit=year**.

This contradicts `06-routes.md` ("with a route it is a probability per contact (per act...)"). It is invisible for HIV-sized betas at `unit=month` (0.4% error) and large for bacterial STIs (beta ~0.3–0.6: −21% at 0.5). v3's `ss.Infection.infect` passes bare betas through unchanged (`diseases.py:619`).

Fix: on a network, a bare beta must reach `net_beta` as-is, and only a `TimePar` beta should be converted. The port undoes the conversion in `net_beta`: `-np.log1p(-p)/disease.engine.dt`. A modeler should never have to write that.

`acts=` on `MFNet` works. `PartnerNet` still has no `acts=` (open question 18): add it, reusing `SexualNetwork.net_beta`.

Condoms need either an edge column (`condoms`) that `SexualNetwork.net_beta` reads with a disease-level `eff_condom`, or a modifier target such as `ss.modify('mf.edges.condoms', set=ss.timeseries(...), where=...)`. Today neither exists. Suggest moving STIsim's condom `net_beta` (`base.py:153-159`) into `ss.SexualNetwork` with default `condoms=0`.

**(4) Directional beta.** `beta=dict(structuredsexual=[m2f, f2m])` works with the v3 class, and `describe()` echoes it. **Clean.**

**(5) Network-derived conditions.**
- Partner notification: `ss.treat(drug, on=ng.cure, trace=['structuredsexual','priorpartners'], coverage=0.5)` works. Tracing over a list of networks, with recent ex-partners through PriorPartners, is genuinely nice. Three problems:
  1. `on_positive=` takes a single delivery, so a screen can't both treat the index and notify partners. A list raises `TypeError: 'list' object is not callable` (it falls through to the "function of (sim, uids)" branch). Suggest accepting a list.
  2. To trigger PN on treated people only (not natural clearance), the disease needs a dummy transition, `cure = ss.progression('I -> S', rate=ss.peryear(0))`, with `Treatment(transition='cure')`. Model structure invented for an intervention. Suggest `on=tx` (a delivery as an event: "people this delivery reached or cured this step").
  3. **Cascade bug/footgun.** If partners are cured by the same transition, their cure triggers PN of *their* partners: 4.0 partners treated per index vs 0.6 with a second dummy `pn_cure` transition. That is unbounded chain tracing, and nothing warns about it. Suggest that `trace=` deliveries do not re-fire `on=` events, or a `depth=1` option.
  
  `trace` uses current edges only. A lookback window ("partners in the last 6 months") needs a PriorPartners-style network; suggest `trace=dict(net='mf', within=ss.months(6))`.
- "≥2 partners in the last year" (open Q17): not expressible as a lazy condition. `eligible=net.partners_12m >= 2` (a network `FloatArr` before init) builds a `Cond('array', None, ...)`, which is then looked up on the *disease*: `AttributeError: Disease object has no attribute "partners_12m"`. It also warns `Trying to access non-initialized Arr`. `conditions.from_arr` loses the owning module because an Arr has no `.mod` before init. The workaround works: `eligible=lambda sim: sim.networks.structuredsexual.partners_12m >= 2`, plus a network `step()` that counts current + prior-partner edges. The proposed `net.degree` / `ss.partner_of(hiv.I)` would cover current degree. "In the last year" also needs the network to remember ended partnerships, which only STIsim's PriorPartners does.

**(6) Solver and describe.**
- With `total_pop` and no `n_agents`, `solver='auto'` picks abm, with reason `StructuredSexual is an individual-level network`. Correct, for a custom subclass too.
- With `n_agents` the reason is `n_agents was given`, so the network reason is hidden.
- `solver='ode'` raises `E804 ... MFNet has no compartmental approximation yet`, for `MFNet` and for StructuredSexual. But `sim.solvers()` claims `network: MFNet  ode: approx (contact rates)`. **Mismatch between spec and table** (`06-routes.md` says to approximate and not refuse).
- `describe()` for networks is just `mixing  structuredsexual (StructuredSexual)`. Missing: partners, durations, acts, and above all how beta is read (per act; the spec says describe prints "which reading applies, with the resulting units of beta"). That is exactly where bug 1 would have shown up.

**Gotcha found on the way.** With an explicit `unit='month'`, bare `start=2000, stop=2020` mean *months* 2000–2020: 21 points, dates in year 166. With `unit` inferred from `dt=ss.months(1)`, the same numbers are years. The same inputs mean different things depending on whether `unit` was given.

## Timing (10k agents, 2000–2020, monthly, births + deaths)

| Run | s |
|---|---|
| networks only: StructuredSexual subclass + PriorPartners | 0.5 |
| networks only: 3 `MFNet` layers | 0.4 |
| v3 `sti.HIV` + `sti.StructuredSexual` (on starsim 4, legacy path) | 1.3 |
| v4 declared HIV + StructuredSexual subclass | 0.7 |
| v4 declared HIV + 3 `MFNet` layers | 0.8 |
| v4 declared HIV + one targeted `MFNet` | 0.4 |

v3 on starsim 3 couldn't be run: the env only has v4. The v3 vs v4 gap is mostly the disease (`sti.HIV` carries CD4, ART hooks, and age/sex results). Network cost is similar: StructuredSexual ≈ 3 MFNet layers.

## Line counts
- v3 `stisim/networks/`: 2,100 lines.
- v4 declarative approximation: ~25 lines, but it loses risk groups, assortative matching, age gaps, typed durations, condoms, and SW windows.
- v4 subclass: ~40 lines on top of the reused 2,100.

## Ranked problems
1. **Per-act beta becomes `1-exp(-beta*dt)` in the sim's unit** on sexual networks: results depend on `unit=` (0.311 vs 0.072 prevalence). This is a correctness bug.
2. **No condoms / `eff_condom` in v4 networks**, and values that vary in time don't work inside a per-network beta dict (two TypeErrors). Condoms by partnership type and year are central to every STI model.
3. **`MFNet` has no per-group targets or `match=`.** No risk groups, no age assortativity, no sex-asymmetric degree (FSW). StructuredSexual can't be expressed declaratively; it stays a subclass.
4. **Declared-disease pars are `ss.par` Exprs**, so network code that reads `disease.pars.x` breaks (`BinOp` TypeError).
5. **PN ergonomics**: a single `on_positive`, a dummy transition needed for `on=`, silent chain-tracing cascade, and no lookback window.
6. **No lazy conditions on network state** (`from_arr` resolves against the disease). Function-of-sim workaround only.
7. **Every network needs a beta key**, bookkeeping ones included.
8. `solvers()` says `MFNet` can be approximated as an ODE, but running it as an ODE raises E804. `describe()` says nothing about networks or how beta is read.

## Surprisingly nice
- The v3 network API carried over almost unchanged: STIsim's 2,100-line network runs under v4 with a 40-line shim, and also as-is with v3 diseases.
- Directional `beta=dict(net=[m2f, f2m])` is clean and checked (E403 catches typos).
- `trace=` over a list of networks, including PriorPartners, gave working partner notification in one line.
- `MFNet(n_contacts=, concurrency=)` plus `net.check()` with W402 is a good diagnostic: it flagged that the casual layer misses its concurrency target (0.022 vs 0.03) under debut, participation, and births.
- `solver='auto'` picks agents for a custom network subclass, with a readable reason.
- The declared engine is fast: the HIV-like model ran 2× faster than v3 `sti.HIV`.
