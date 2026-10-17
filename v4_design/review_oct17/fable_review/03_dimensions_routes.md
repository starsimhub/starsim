# Dimensions and routes

I reviewed `v4_design/design/05-dimensions.md` (342 lines) and `06-routes.md` (345), the relevant rows of `17-vocabulary.md` and `19-open-questions.md`, and the "Departures" and "Known issues" lists in `v4_design/admin/implementation.md`. Code: the stratification half of `starsim/transitions.py` (`by`, `geo`, `gravity`, `Mobility`, `Strat`, `Transfer`/`transfer`, `stratify`, lines 2045 to 2972 of 2972), `Sim.stratify`/`init_strats`/`init_ragged`/`grid_index` in `starsim/sim.py` (535 to 830 of 2692), `Disease.stratify` in `starsim/diseases.py`, all of `starsim/networks.py` (2235 lines, 23 top-level classes and functions), `Mixing` in `starsim/solvers.py` (46 to 311) and `Compartmental.init_adjust`/`local_count`, `starsim/conditions.py` where it selects by level, and `tests/test_dimensions.py` (526). I grepped every network class across `tests/`, `starsim/library`, `docs/`, and the 16 vignettes under `lingua_franca_design/vignettes` that call `stratify`. I ran five scripts (`scratchpad/dimensions/s1_entrypoints.py` to `s5_routes.py`) covering the entry points, counts across dimensions, an age × place model with Prem contacts and gravity commuting, a strain dimension, sim-level options, and the route table; outputs are quoted below.

## Summary

1. **Make `kind` a property of the value, not of the name, and drop the name list.** `place=`, `location=`, `herd=`, `lineage=` are all `plain` today, and every vignette that needed space wrote `kind='space'` by hand; one wrote `sex=[...], kind='space'` to get per-group counts. `patch=ss.geo(names=[...])` already marks space; add `ss.strains([...])` (or accept `immunity=` as the marker) and keep only `age` by name. Breaking for `patch=['a', 'b']` without `kind=` (it would become plain, which it is for `place=` already). (Finding U1.)
2. **One entry point for a sim-wide dimension, with the full option set, or none.** `ss.Sim(stratify=...)`/`sim.stratify()` silently turn `states=`, `immunity=`, and `inplace=` into dimensions called `states`, `immunity`, `inplace`, skip the `E301` check on `adjust` (a typo is ignored), and give a sim-level `strain` to `S` as well. Either route the sim-level call through `Graph.stratify` for every module, or reduce it to levels only (`sim.stratify(age=[...])`, nothing else) and put every option on the disease. Breaking only for sim-level `adjust`/`where`/`mobility`. (U2, B1, B2.)
3. **Settle the count-alignment rule explicitly rather than by kind.** `I.n` in a rate is the count in the same place for a space dimension and the global total for every other dimension; the spec's "aligned on every shared dimension" is not what runs, and `I.n.sum('age')`/`I.n.select(...)` do not exist. Say "counts are totals, except within a place" in the spec, and add one explicit spelling (`I.n.by('sex')` or similar) so nobody has to lie about `kind`. Not breaking. (U3, S1.)
4. **Collapse the five spellings of an age mixing matrix to `ss.contacts`.** `ss.MixingPools(src={...: ss.AgeGroup}, dst=..., n_contacts=M)`, `ss.MixingPools` with `LevelGroup`, `ss.contacts('Kenya')`, `ss.contacts(M, age=edges)`, and `ss.contacts(age=M)` all produce the same object; `age=` means edges in one form and a matrix in the other. Make `contacts` the only user-facing constructor (`ss.contacts('Kenya')`, `ss.contacts(M, age=edges)`, `ss.contacts(M, by='risk')`), keep `MixingPools` as the class it returns, and retire `AgeGroup`/`LevelGroup` from the public surface. Breaking for `MixingPools(src=...)` users (Starsim 3 docs). (U4, C1.)
5. **Prune the network catalogue.** `networks.py` has 18 network classes; `RandomSafeNet`, `MaternalNet` (an alias), `PostnatalNet`/`BreastfeedingNet`, `MSMNet`, `HybridNet`, and `ClusterNet` are each used by one test or one library module, and none by a vignette. Move the Covasim-specific and the single-use ones to `starsim/library/networks/`, where `DiskNet`/`ErdosRenyiNet`/`NullNet` already live, and merge `RandomExactNet`/`RandomNet`/`RandomSafeNet` into one `RandomNet` with a `method=` option. Breaking (imports move). (C2.)
6. **Stop promising what the routes do not do.** `dependence='density'`/`n_ref`, the sexual-network ODE approximation ("partner-change rates between sex and risk groups"), `E403 RouteMismatch` on a misspelled `beta` key, and `net.check()` on anything but `PartnerNet`/`MFNet` are in the spec and absent or inert in the code. Either implement the two that matter (`E403`; a note when `I.n/N` is written under a contact matrix) or strike the rest from 06. Not breaking. (S2, B3.)
7. **Keep in-place `stratify` and the `where=` ragged model; do not add a fourth entry point.** The four spellings that exist (`sir.stratify`, `ss.stratify(sir)`, `sim.stratify`, `ss.Sim(stratify=)`) are two too many; `ss.stratify()` the function earns nothing (it is one line calling the method) and can go. (U5.)

## Simpler for the user

### U1. The kind of a dimension is inferred from a list of five place names and five strain names

**Issue.** `transitions.py:2430` is the whole mechanism: `kinds = dict(age='age', patch='space', node='space', region='space', district='space', county='space', strain='strain', variant='strain', genotype='strain', serotype='strain', resistance='strain')`. Anything else is `plain`, and `plain` cannot take `mobility=` or `immunity=`. From `s1_entrypoints.py`:

```
place kind: plain
herd kind: plain
location kind: plain
place + mobility ERR: ValueError mobility= is for a spatial stratification, but "place" is of kind "plain"; pass kind="space" if it is a place
place=ss.geo(names=...) kind: space labels ['a', 'b']
lineage kind: plain states: None
lineage + immunity ERR: ValueError immunity= is for a strain dimension, but "lineage" is of kind "plain"; pass kind="strain" if it is one
```

Every vignette and test that used a place name outside the list wrote `kind='space'`: `vignettes/camdl/stratified/model.py:17` (`patch=['north', 'south'], kind='space'`, even though `patch` is in the list, because the author did not trust it), `siminf/sir_herds/model.py:28` and `siminf/sise3/model.py:49` (`herd=..., kind='space'`), and `tests/test_dimensions.py:297,338,383,443,468` (`place=`, `herd=`, all with `kind='space'`). The spec's own list omits `place`, the word the tests use most.

**Why it matters.** A user learns a rule ("the name decides the kind") that fails on the first name they pick that is not in the list, and the failure comes from `mobility=` or `immunity=`, not from the name. The alternative the spec already has, `patch=ss.geo(...)`, marks space unambiguously (`Strat.__init__`, `transitions.py:2729-2730`), and `19-open-questions.md` B6 raises exactly this for strain.

**Proposal.** Put the kind on the value: `patch=ss.geo(names=['a', 'b'])` (exists; a geography with no positions or populations is legal) and `strain=ss.strains(['A', 'B'])` or, simpler, let `immunity=` and `move=`/strain-aware transmissions imply strain. Keep `age` by name only, since it alone is also a property every agent already has. Delete `kinds`, `kind=`, and the two "pass kind=..." error messages. `describe()` keeps printing the kind.

**Cost.** Breaking for `patch=['a', 'b']` without `kind=` or `ss.geo` (becomes plain; three vignettes already pass `kind='space'` and would change to `ss.geo(names=...)`). The spec table in 05 §Kinds shrinks to two rows. Confidence: high.

### U2. The sim-level `stratify` is not the same call as the disease-level one

**Issue.** 05 says "`sim.stratify()` takes the same arguments as `model.stratify()`", and `sim.py:539` repeats it. `Sim.init_strats` (`sim.py:552-560`) pops only `split`, `adjust`, `kind`, `where`, `mobility` and treats every other keyword as a dimension. From `s1_entrypoints.py`:

```
== A. ss.Sim(stratify=dict(age=..., states=[...]))
dims: [('age', ['0-14', '15-64', '65+'], None), ('states', ['S', 'I'], None)]
== A2. ss.Sim(stratify=dict(strain=[...], immunity=...))
dims: [('strain', ['A', 'B'], 'strain'), ('immunity', [[1, 0.3], [0.3, 1]], 'plain')]
```

And from `s4_simlevel.py`: a misspelled transition in a sim-level `adjust` is silently ignored, where the disease-level call raises `E301`; a sim-level `adjust` naming a transition one disease lacks is silently skipped for that disease (`solvers.py:989-1004` has no "unknown name" branch); and a sim-level `strain` applies to `S` (`sim strat states: None`, `n_S has strata? True`) where the disease-level one applies to infected states only (`disease-level strat states: ['I', 'R']`), because the strain guess lives in `Graph.stratify` (`transitions.py:1837-1839`) and not in `Strat`.

**Why it matters.** The sim is where the spec tells people to stratify by age and place. A user who moves a working `flu.stratify(...)` line to `sim.stratify(...)` gets a different model or a silent no-op.

**Proposal.** Two honest options. (a) Make `sim.stratify(**kw)` call `mod.graph.stratify(**kw)` on every declared module and keep one `Strat` per name on the sim, so the checks and guesses are shared; `init_strats` then only reconciles levels (`E308`). (b) Reduce the sim-level call to levels only: `sim.stratify(age=[0, 15, 65])`, with `adjust`, `where`, `states`, `mobility`, `immunity`, and transfers always on the disease. (b) is simpler to explain ("the sim says what the groups are; the disease says what differs between them") and matches how all 16 vignettes are written: none passes an option to the sim-level call. Either way, an unknown keyword at the sim level should be an error, not a dimension.

**Cost.** (b) breaks `sim.stratify(patch=..., mobility=...)`, which nobody uses, and `where=` at the sim level, used once in `test_dimensions.py:508`. Confidence: high that the current state is wrong; medium on (b) over (a).

### U3. "Counts are matched on shared dimensions" is true only for space, and the override spelled out in the spec does not exist

**Issue.** 06 §Counts used literally: "a count is aligned with the person at risk on every dimension the two share and summed over the rest ... `I.n.sum('age')` and `I.n.select(age=...)` override the alignment." The code aligns on the spatial dimension only (`Compartmental.local_count`, `solvers.py:1900-1904`; the agent side likewise keys on `mixing.space`), and `Count` (`transitions.py:257-268`) has no `sum` or `select`. From `s2_counts.py`, an SIR with `S >> I @ 0.5 * I.n / N` and infection seeded in the first level only:

```
plain (risk)                 ode  cum_infection by level: [9827. 9926.]  -> second level reached: True
space (patch, kind=space)    ode  cum_infection by level: [9827.    0.]  -> second level reached: False
age labels                   ode  cum_infection by level: [9827. 9926.]  -> second level reached: True
```

The vignette `atomica/hiv_dyn/model.py:71` shows what a user does when they need the aligned count on a non-space dimension: `hiv.stratify(sex=['females', 'males'], kind='space', split=cascade[0]) # A place, so that .n is the count in each sex`.

**Why it matters.** This is open question B1 "Counts across stratifications", which the implementation has answered differently from the spec: totals everywhere, local within a place. That answer is the right default (homogeneous mixing over risk groups should see the total), but it needs an explicit escape hatch, and the spec text and the missing methods send the user the wrong way.

**Proposal.** Rewrite the rule in 06 as "a count is the total, except that within a space dimension it is the count in the same place". Add one explicit per-group spelling: `I.n.by('sex')` (or `I.n.local('sex')`), returning the count at the person's own level of that dimension, implemented in the same place `local_count` already handles space. Then `kind='space'` for sex becomes unnecessary and U1's removal of `kind=` is safe.

**Cost.** None breaking; one small method and one paragraph. Confidence: high.

### U4. Five ways to say "age mixing matrix"

**Issue.** All of these build an `ss.MixingPools`:

```python
ss.MixingPools(src={'0-15': ss.AgeGroup(0, 15), '15+': ss.AgeGroup(15, None)}, dst=..., n_contacts=M)   # Starsim 3, networks.py:1903-1909
ss.MixingPools(src={...: ss.LevelGroup(age=0)}, ...)                                                      # networks.py:1868
ss.contacts('Kenya')                                                                                      # networks.py:2216
ss.contacts(M, age=[0, 15])            # age= is the lower edges of M's groups                            # networks.py:2222
ss.contacts(age=M)                     # age= is the matrix, for the model's own groups                   # networks.py:2195
```

From `s4_simlevel.py`: `contacts(matrix, age=edges)` gives groups `['0-15', '15+']` and no `dims`; `contacts(age=matrix)` gives groups `['age=0', 'age=1']` and `dims=['age']`; `contacts(age=[0, 15])` alone is an error about a square matrix. The same keyword means edges or a matrix depending on whether the positional argument is present. `MixingPools()` with no arguments raises `TypeError: src must be a provided as a dictionary` while its docstring (`networks.py:1891`) says it defaults to under/over 15.

**Why it matters.** The spec's reason for `ss.contacts` was to have one name for a contact matrix. The Starsim 3 form is still what the user guide teaches (`docs/user_guide/modules_networks.qmd:178-247`, `t5_networks.qmd:135-160`), and the two `contacts` forms differ in a way that is easy to get backwards.

**Proposal.** One constructor with three unambiguous forms: `ss.contacts('Kenya', layer=...)`; `ss.contacts(M, age=[0, 15, 65])` (matrix with its own age edges); `ss.contacts(M, by='risk')` or `ss.contacts(risk=M)` for a matrix over a model dimension (keep the keyword form, but refuse `age=` as a matrix: for age, the edges form is the one that generalizes). Make `MixingPools` the class `contacts` returns, not something a user constructs; drop `AgeGroup` and `LevelGroup` from `__all__` (they exist to serve `contacts`, `networks.py:2205,2226`, and `test_mixingpools.py`). `MixingPool` (singular) stays as the homogeneous pool with `n_contacts`.

**Cost.** Breaking for `MixingPools(src=..., dst=...)` and `AgeGroup` in user code; the user guide's mixing-pool section is rewritten around `contacts`. Confidence: high.

### U5. Four spellings of `stratify`, and the function form earns nothing

**Issue.** `sir.stratify(...)`, `ss.stratify(sir, ...)` (`transitions.py:2970-2972`, one line), `sim.stratify(...)`, `ss.Sim(stratify=...)` (a dict, or, undocumented, a list of dicts as `test_dimensions.py:508` uses). All four work (`s1_entrypoints.py` F).

**Proposal.** Drop `ss.stratify()`; it is listed in 17 as "the same as `sir.stratify`" and buys only a lowercase name. Keep `ss.Sim(stratify=)` as the keyword form of `sim.stratify()` (that pairing is Starsim's convention for every module list). Document the list-of-dicts form or remove it. Not breaking in practice (no vignette or test uses `ss.stratify(`). Confidence: medium.

**Keep** in-place by default. 19-D risk 11 argues it well, `E305` catches the re-run cell, and `inplace=False` covers the comparison. The one surprise I hit: `sim.stratify()` after `sim.init()` is a plain `RuntimeError` (`sim.py:541-543`) rather than a coded error, and `stratify` on a sim does nothing visible until `init()`, so `sim.strats` does not exist until then.

### U6. Moving between levels has five spellings

**Issue.** People move along a dimension by `ss.transfer(risk='low -> high', rate=)` (a `Transfer`, `transitions.py:2896`), `ss.transfer(vaccinated=True, coverage=)` (which returns an `ss.Delivery(move=)`, `transitions.py:2966`, a different class with a different lifecycle), `ss.progression('L -> I', move=dict(strain='A+B -> A'))` (`transitions.py:638-676`), `ss.Mobility(kind='migrate')` (a transfer on the spatial dimension, `solvers.py:1643`), a movement register (`Mobility(data=)`), and aging, which is generated. `s4_simlevel.py` G shows the same function returning two unrelated types depending on which keywords are present.

**Why it matters.** `ss.transfer` returning a `Delivery` is the kind of thing that works until a user passes it to `ss.Disease(...)` expecting a transition. 19-B6 asks "Should `ss.transfer` take the delivery keywords?"; the implementation answered yes, by overloading.

**Proposal.** Keep `ss.transfer` as a transition only. A delivered move is `ss.Delivery(move=dict(vaccinated=True), coverage=0.7, years=[2021])` (or `ss.vaccinate` with a product, which is what most users mean). `Mobility(kind='migrate')` and `Mobility(data=)` are fine as they are: they are the spatial dimension's own form of a transfer and are documented as such.

**Cost.** Breaking for `ss.transfer(..., coverage=)`; one example in 05 and one in 17 change. Confidence: medium.

### U7. The written-out `/N` form silently ignores the mixing, and `describe()` does not say so

**Issue.** 06 says `S >> I @ beta * I.n / N.n` is "the same thing written out for homogeneous mixing". Under a contact matrix it is literal (by rule) and so the matrix is ignored. From `s4_simlevel.py` C, `describe()` prints

```
infection    S -> I   spontaneous
             rate = 0.5 × I.n / (S + I + R).n
...
mixing       contacts (MixingPools): contact matrix by age
```

with no connection drawn between the two lines, and the user who wrote the spec's own example with `init_prev=` gets `ValueError: "dis" has no contact-mediated transition to seed init_prev through` (`s2_counts.py`, `solvers.py:1177`).

**Proposal.** When a rate uses `X.n` for a people state `X` that is also infectious-looking, and a route other than homogeneous mixing is present, print one note: "`infection` uses `I.n` literally, so `contacts` does not apply to it; write `I` for exposure through the routes". And let `init_prev` seed through a transition whose rate mentions a people state's count (06 already says such a transition "is still an infection for the purposes of results").

**Cost.** None. Confidence: high.

### U8. Is "the user never writes `/N`" sound for multi-place and vector models?

Mostly yes, and the implementation shows where it is not. For a person-to-person disease in several places, the bare `I` works and is local to the place (`s2_counts.py`, space rows: the epidemic stays where it is seeded with no mobility), and with commuting both the bare form and the matrix form are handled by `Mixing.by_place`/`within_places` (`solvers.py:100-119, 168-183`). For vector models the user does write `/Nh` (`Iv.n / Nh`, 06 §Vector-borne, `test_dimensions.py:41`): that is right, because bites per person is not a contact rate the routes know, and `Nh = (Sh + Eh + Ih + Rh).n` is aligned per place like any count. Where it is unsound: (i) with a plain dimension the bare `I` is the global fraction and `I.n/N.n` the global ratio, so they agree, but with a space dimension `I.n/N.n` is the local ratio while `(S + I + R).n` is also local, which is correct only because both are counts; a user who writes `I.n / sim.total_pop` gets a different model per place. (ii) `dependence='density'` is in the spec but not the code, so a density-dependent model has to be written with `I.n`, which is what the importer does (`implementation.md` departures: "kept as counts (`beta * I.n`) ... there is no density-dependent mixing"). Recommendation: strike `dependence=`/`n_ref` from 06 and 17 and say that density dependence is written as `beta * I.n`.

## Simpler codebase

### C1. `AgeGroup`, `LevelGroup`, and the `src`/`dst` dict protocol exist only to serve `contacts()`

`networks.py:1842-1884` (45 lines) define two callables that return uids, and `MixingPools` (`1887-2004`, 120 lines) builds one `MixingPool` per cell from them. In a compartmental run none of that is used: `Mixing.age_matrix` (`solvers.py:185-216`) reaches back into `pars.src` to recover the bands, and `dim_matrices` (`237-252`) uses the `dims` attribute that `contacts()` bolts onto the object (`networks.py:2212`). The agent run with places does the same via `init_pools` (`solvers.py:286-310`), bypassing the pools entirely. So the pools-of-pools machinery is the agent-only, no-places path. If `MixingPools` stored a matrix plus the group definition (edges, or a dimension name) and computed group membership itself, the two helper classes, the `dims`/`ref` attributes, and the "indexed [source, destination]" transposition comments (`networks.py:2210,2232`) go away, and the compartmental side reads the matrix directly. Confidence: medium.

### C2. The network catalogue

`networks.py` has 18 network classes plus the two pool classes. Usage outside `networks.py` (files that mention the class; `grep -lw`):

| Class | Lines | tests | library/core | docs | vignettes |
|---|---|---|---|---|---|
| `RandomNet` | 74 (+144 in `RandomExactNet`) | 18 | 10 | 13 | 0 |
| `MFNet` | 203 | 10 | 4 | 4 | 0 |
| `PrenatalNet` | 19 | 8 | 8 | 3 | 0 |
| `PartnerNet` | 193 (+129 `DegreeTargets`) | 3 | 3 | 0 | 2 |
| `MixingPool` | 154 | 4 | 3 | 3 | 0 |
| `StaticNet` | 84 | 1 | 4 | 1 | 0 |
| `RandomSafeNet` | 111 | 1 | 0 | 0 | 0 |
| `ClusterNet` | 88 | 1 | 1 (`HybridNet`) | 1 | 0 |
| `HybridNet` | 70 | 1 | 0 | 1 | 0 |
| `MSMNet` | 71 | 1 | 0 | 2 | 0 |
| `BreastfeedingNet` | 46 | 1 | 0 | 0 | 0 |
| `PostnatalNet` | 42 | 0 | 2 | 1 | 0 |
| `MaternalNet` | 5 (alias) | 0 | 0 | 1 | 0 |
| `contacts` | 74 | 10 | 14 | 10 | 7 |

Observations. The three random networks (`RandomExactNet`, `RandomNet`, `RandomSafeNet`; 329 lines together) differ in how targets are drawn and whether they are CRN-safe, and `RandomSafeNet` even uses a different parameter name (`n_edges`, "n_contacts = 2*n_edges", `networks.py:1195-1200`). One `RandomNet(method='fast' | 'exact' | 'safe')` with one parameter name is what a user can hold. `HybridNet`, `ClusterNet`, and `MSMNet` are Covasim/HIV-specific and belong with `DiskNet`, `ErdosRenyiNet`, `NullNet` in `starsim/library/networks/`, where the user guide already sends people for "additional example networks". `MaternalNet` is a backward-compatibility alias and can go in v4. The abstract tier (`Route`, `Network`, `DynamicNetwork`, `SexualNetwork`) is four classes for what the user guide itself describes as "never directly" and "rarely" used; `Route` is 10 lines, and `17-vocabulary.md:229` says it is not part of the v4 vocabulary while `__init__.py:93` exports it. Cost: imports move; `docs/user_guide/modules_networks.qmd` §Comparison is rewritten (it is Starsim 3's and would have to be anyway). Confidence: high on the move, medium on the merge.

### C3. `Mobility` carries four kinds and two meanings of `away`

`transitions.py:2554-2709`: `kind` is `commute`, `visit`, `migrate`, or `force`; `away` is "the share of their time people spend away" for `commute`/`visit` and "the share of each place's force of infection that leaves it" for `force`; a bare `ss.gravity()` or matrix is commuting and a register is migration. The spec has three kinds (`visit` is an undocumented addition, used by two vignettes, `flepimop/seir_2pop/model.py:35` and `ghana_measles/seir_districts/model.py:28`). `force` exists to reproduce LASER (06 §Commuting), and 19-B9 already suspected LASER's design is "probably `ss.MixingPool` grouped by patch". Proposal: keep `commute`, `visit`, `migrate`; keep `force` but say in its docstring that it is a LASER-compatibility form and not recommended for new models; add `visit` to 06. Not breaking. Confidence: medium.

### C4. `Strat` is the one class that does the work, and it is good; `init_strats` is where the sim-level duplication lives

`Strat.__init__` (`transitions.py:2727-2804`) does level parsing, split normalization, adjust checks, immunity checks, and carriage. `Sim.init_strats` (`sim.py:547-651`) then re-derives a split for age from a named population, resolves joint splits, folds ragged groups, and merges module and sim strats, with a second `E305` and a second `E1102` of its own. The duplication noted in U2 (`states`/`immunity`/strain guess in `Graph.stratify` but not here) follows from having two construction paths. Routing the sim-level call through `Graph.stratify` (U2 option a) removes `sim.py:552-560` and the `opts` list. Confidence: medium.

### C5. `geo` says it has no named geographies, but it does

`implementation.md` "Departures": "`ss.geo` takes the places directly (names, populations, and positions or distances); there are no named geographies." `s3_twodim.py`: `ss.geo('Kenya', level=1)` returns `geo('Kenya', level=1: 47 places)` offline, via `ss.named.geography` (`transitions.py:2468-2473`). The departure note is stale and should be removed; the spec's `ss.geo('Kenya', level=1)` works as written. Not a code change.

## Spec and implementation

### S1. Disagreements beyond the departures list

- 06 §Counts: alignment on "every dimension the two share" and the `I.n.sum('age')`/`I.n.select(age=...)` overrides. Code: space only; no such methods (U3).
- 05 §Stratifying the sim or the disease: "`sim.stratify()` takes the same arguments as `model.stratify()`". Code: five options are recognized, the rest become dimensions (U2, B1).
- 06 §Frequency and density dependence: `dependence=`, `n_ref`. Code: `ss.MixingPool(dependence='density')` is `ValueError: 1 unrecognized arguments for mixingpool: dependence` (`s1_entrypoints.py` H); `solvers.py:62` hard-codes `'frequency'`.
- 06 §Routes table: `ss.MFNet` "approximated by partner-change rates between sex/risk/age groups" under a compartmental solver. Code: `E804 CapabilityError: MFNet has no compartmental approximation yet` (`s1_entrypoints.py` I; `solvers.py:152-153`). The same for `ss.MixingPool` with `src`/`dst` (`s1` J) and `ClusterNet` (`s5` E). 06 says "It does not refuse"; it refuses. I think refusing is the right behavior here (19-B5 shows the approximation is not settled), so the spec should change, not the code.
- 06 §Beta per route: "A key that matches no route raises `E403 RouteMismatch`". Code: `E403` is raised only for a network that has no key (`solvers.py:4132`); a misspelled key (`needls`) passes silently (`s5_routes.py` B).
- 06 §Layers: "`sim.networks.school` reaches a layer that was declared inline". Code: in a compartmental run the routes are moved to `sim.mixed` and `pars.networks` is emptied (`sim.py:383-384`, "the networks are replaced by contact rates between groups, so are not run themselves"), so `sim.networks` is `[]` and `sim.networks.school` does not exist (`s6_networks_attr.py`: `sim.networks keys: [] | hasattr school: False`). The modifier path `'school.pars.n_contacts'` still works because it resolves by name. Either keep the routes in `sim.networks` under every solver (they are still the model's routes, run or not) or change the spec line.
- 06 §Describing a network: `net.check()` "simulates the network alone". Code: it exists on `DegreeTargets` only (`networks.py:950`) and reruns the whole sim, refusing if the sim has no declared model (`networks.py:964`).
- 05 §Kinds: `place` is not in the spec's list either, yet it is the name the tests use; the spec and the code agree with each other and disagree with the users (U1).
- 06 §Mobility: `visit` is missing from the spec; `away=` has a second meaning under `force` (C3).
- 05 §Levels: "`sir.stratify(vaccinated=[False, True])`, a boolean property". Works, and a boolean dimension is also created on demand by a transfer with a boolean target (`sim.py:570-576`); 05 §`ss.transfer` says a transfer naming a missing dimension is `E1102`, but for booleans it is not, which is friendlier and should be in the spec.

### S2. Over-specified

- `dependence`/`n_ref`: one keyword, one table, one paragraph of 06 and two rows of 17 for something no vignette needed and the importer sidesteps. Strike.
- `ss.geo(['Kenya', 'Uganda'])`, `places=`, `year=` fetching, `geo.adjacency`: all present in the code; none used by a vignette. Fine to keep since they are there, but 06 §Geography could be a quarter of its length.
- The sexual-network approximation paragraph (06, "The sexual-network approximation") describes an algorithm that does not exist. Replace with one line: "individual-level sexual networks run as agents; `solver='ode'` is `E804`".

### S3. Under-specified

- What a sim-level option does when a module lacks the thing it names (U2).
- Whether `I.n` is local to a place in an agent-based run with commuting (it is: `Agents.exposure` uses `place_raw()`, `solvers.py:3389-3393`; a user cannot tell from 06).
- `ss.Sim(stratify=[dict(...), dict(...)])`, the list form, which is the only way to give options to two sim-level dimensions.
- `age=['young', 'old']` becomes `plain` with attribute `age_group` (`transitions.py:2764-2766, 2863`); 05 says only that labels and edges are both accepted. Worth one sentence: "age groups given as labels do not age and are stored as `age_group`".

### S4. Open questions the implementation has answered

| 19 item | Answer in the code |
|---|---|
| B1 Counts across stratifications | Totals, except within a space dimension (U3). Needs the spec text changed and one explicit spelling added |
| B1 `sim.people.age` in a compartmental run | The bin midpoint (`Strat.mids`, `transitions.py:2876-2881`) |
| B1 Network-to-group contact rates | Neither: the mean of `n_contacts` (`Mixing.mean`, `solvers.py:122-130`); sexual networks are refused |
| B5 Which approximation for a sexual network | None; `E804` (the honest answer) |
| B6 Marking a dimension as strain-like | Still the name list, plus `kind=`; U1 proposes the value |
| B6 Should `ss.transfer` take delivery keywords | Yes, by returning a `Delivery` (U6 argues against) |
| B9 Agents in patches with aggregate coupling (LASER) | `Mixing.init_pools` runs a mixing pool or contact matrix within each place as agents (`solvers.py:286-310`), and `kind='force'` reproduces LASER's coupling. Both vignettes agree with the originals (`implementation.md:27,44`) |
| B9 Space conveniences (maps, per-patch data) | Not built; `ss.by(pd.DataFrame(...))` covers per-patch values |
| B2 A reservoir per patch alongside people | Works: `test_dimensions.py:303-317` (cholera, upstream/downstream) and `s2_counts.py` space rows |

## Bugs

### B1. Sim-level `stratify` turns unknown options into dimensions

Repro (`s1_entrypoints.py`): `ss.Sim(sir, stratify=dict(age=[0, 15, 65], states=['S', 'I']))` → `dims: [('age', ...), ('states', ['S', 'I'], None)]`; `ss.Sim(sir, stratify=dict(strain=['A', 'B'], immunity=[[1, 0.3], [0.3, 1]]))` → a plain dimension called `immunity` with two matrix rows as its levels. Cause: `sim.py:554` pops a fixed list of five keys. Fix: pop the full set (`states`, `immunity`, `inplace`) or error on any keyword whose value is not a list, array, or `ss.geo`.

### B2. Sim-level `adjust` is not checked

Repro (`s4_simlevel.py` A): `ss.Sim(sir, stratify=dict(age=[0, 15, 65], adjust={'recovry': [2, 1, 0.5]})).init()` → no error; the disease-level call raises `E301`. With two diseases, an `adjust` naming a transition only one has is applied to that one and silently skipped for the other. Cause: `Compartmental.init_adjust` (`solvers.py:989-1004`) has `if name in self.rel_adjust ... elif name in self.adjust` and no else.

### B3. A misspelled `beta` key is not `E403`

Repro (`s5_routes.py` B): `beta=dict(mf=0.008, needls=0.03)` with networks `mf` and `needles` → `no error`. `solvers.py:4132` checks only that every network has a key. The spec's `E403` is for the other direction.

### B4. `MixingPools()` docstring promises a default it does not have

`networks.py:1891` "By default, separates the population into <15 and >15 age groups"; `ss.MixingPools()` → `TypeError: src must be a provided as a dictionary, not <class 'NoneType'>` (`s3_twodim.py`). Either restore the default or fix the docstring (and its typo).

### B5. A joint `split` over a dimension the model lacks is an uncoded `KeyError`

`s3_twodim.py`: `sir.stratify(patch=towns, split=ss.by(pop, dims=['age', 'patch']))` with no age dimension → `KeyError: 'The model is not stratified by "age"'` from `Sim.dim_index` (`sim.py:799-805`) at `init()`. Should be `E1102` with the "Add e.g. sim.stratify(age=[...])" hint the other paths give.

## Keep

- **Stratification as an operator on a finished model, with multiplicative `adjust` only.** The TB resistance example (`test_dimensions.py:55-73`) and summer2's clinical strata (`:149-162`) read as the spec promised, and the ODE/binomial/ABM agreement tests in `test_dimensions.py` are the strongest evidence in the repo that the design works. Do not add overwrite adjustments.
- **Space as a dimension, not a model class.** `s3_twodim.py` runs an age × place SIR with Prem contacts aggregated to three groups, aging, and gravity commuting in nine compartments per state, and `describe()` says exactly what was guessed. The `[guessed: commuting, since kind= was not given]` and `W404` aggregation note are the right level of chattiness.
- **`where=` ragged dimensions.** Age groups for women only, with men "missing, not all ages", is a hard thing to get right and `init_ragged`/`fold`/`grid_index` (`sim.py:653-731`) do it in 80 lines with the semantics 05 spells out; `test_ragged` checks every consequence the spec lists. Later reviewers should not "simplify" the grid-with-fold representation into a true ragged array.
- **Strains carried together (`'A+B'`) and `move=`.** `Strat.init_carriage` (`transitions.py:2806-2832`) is 27 lines for a feature tbsim needs, and `test_carriage` checks it against a hand-written ODE to 1e-3.
- **The movement register.** `Mobility.load`/`events` (`transitions.py:2620-2680`) replace SimInf's select and shift matrices with a table of named columns, and the `to_states` column covers aging between herds. This is the clearest win over the compared frameworks in my area.
- **Refusing rather than approximating sexual networks under an ODE.** The spec said approximate; the code says `E804` with "Run with solver=abm". The code is right; change the spec.
- **`ss.by` with the table form.** `ss.by(pd.DataFrame(age=..., patch=..., value=...))` for sparse initial conditions is used by two vignettes and is the one place where the dataframe form beats a keyword.
- **`ss.contacts('Kenya')` adding the age stratification itself** when none was given (`solvers.py:195-199`, `s5_routes.py` D: 16 groups appear with a note). It is what the user meant.

## Outside my area

- `init_prev` refuses to seed through a transition whose rate uses `I.n` (`solvers.py:1177`), though 06 says such a transition "is still an infection for the purposes of results". Results/transitions reviewer.
- `ss.modify(school.pars.n_contacts, ...)` (the spec's spelling in 06 §Layers) is `E504`; the string path works. Already a listed departure, but 06 still shows the object form. Effects reviewer.
- `Sim.init_crossings` (`sim.py:733-789`) turns each jointly tracked disease into a `Strat` of `kind='disease'` with `holders` and `state_levels` attributes that no other `Strat` has; the stratification machinery is being reused for something that is not a stratification. Solvers reviewer may want to weigh whether `Crossing` should own its own layout.
