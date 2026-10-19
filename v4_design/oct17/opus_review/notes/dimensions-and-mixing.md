# Review: stratification, mixing, networks, space, and metapopulations

Reviewer scope: `v4_design/best-practices/{stratification,population-and-mixing,metapopulation}.md`, `v4_design/design/{05-dimensions,06-routes}.md`, and the corresponding code in `starsim/networks.py`, `named.py`, `people.py`, `transitions.py` (`Strat`, `geo`, `gravity`, `Mobility`, `Transfer`), `solvers.py` (`Mixing`, movers, `Agents.infect`), and `sim.py` (`stratify`, `init_strats`). Every bug claim below was reproduced with a script in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/dims/` (`exN.py` named in each finding), run with `/software/conda/envs/v4/bin/python`.

## Summary

The core idea is right and worth keeping: one `stratify()` operator for age, risk, place, and strain, results read with `.select()`/`.by()`, and mixing given separately through `networks=`. The simple cases work, run fast, and agree between ODE and agents. The problems sit one level out from the simplest example. (1) The spec's own flagship mixing example, layered `ss.contacts` with a `beta` per layer, crashes under both solvers. (2) Several guessed defaults change the model silently and by large amounts: the *name* of a dimension decides whether places mix, a stratified population is 35% over 65, aging with no deaths inflates the elderly by 40% a year in the ODE, and a sex dimension as agents is unrelated to the agents' actual sex. (3) There are two stratify parsers that disagree, and the sim-level/disease-level distinction is cosmetic because dimensions are global anyway. (4) Movement between levels is implemented as dense group-by-group matrices, which makes default aging 100 times slower on a 16-age by 290-place model. Structurally, I would merge the two stratify entry points into one, infer a dimension's kind from its values rather than its name, give mixing pools one counts-based implementation under both solvers (deleting the N² sub-pool modules), and split `ss.Mobility` into "who meets whom" (a route) and "who moves" (a transfer).

## Findings, ranked

### 1. The layered contact-matrix example fails under both solvers

Type: bug

Evidence: the example in `06-routes.md` §Layers (home/school/work/community `ss.contacts` layers, with `beta=dict(home=0.1, school=0.05)`) fails three separate ways (`ex7.py`, `ex7b.py`, `ex7c.py`, `ex7d.py`):

```python
home, school = ss.contacts('Kenya', layer='home'), ss.contacts('Kenya', layer='school')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=dict(home=0.03, school=0.03)), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.001)
ss.Sim(sir, networks=[home, school], stratify=dict(age=[0,5,18,65]), unit='day', stop=150).run()
# ODE (auto):  TypeError: unsupported operand type(s) for *: 'dict' and 'float'   (transitions.py:409, via Compartmental.flow_base)
# n_agents=20000, even with a scalar beta:  AttributeError: Module pool:0.0-5.0->0.0-5.0 already added   (networks.py:1965, people.py:334)
```

- Under any compartmental solver, a `beta` dict keyed by route is never resolved: `Agents.route_beta()` (solvers.py:4125) handles dicts but `Compartmental` evaluates the expression directly. The same crash happens with two plain `ss.MixingPool`s.
- As agents, `MixingPools.init_pre()` names each sub-pool `f'pool:{sk}->{dk}'` without the parent's name, so any two `ss.contacts()` layers collide.
- A `beta` key that matches no route (`dict(home=..., school=..., work=0.1)` with no `work` network) is silently ignored, where the spec says `E403`. Only the reverse case (a route with no key) raises `E403`.

Proposal: resolve per-route `beta` in one place shared by both engines (the `Mixing` terms already carry `name`, so `term.contacts * beta[term.name]`), prefix sub-pool names with the parent (`f'{self.name}:{sk}->{dk}'`, or better, finding 8, have no sub-pools at all), and check route keys in both directions at init. Add the layered example from 06 to `tests/test_mixingpools.py` under `ode` and `abm`.

Cost: small. That it slipped through suggests the spec examples are not run as tests; extracting and running every code block in `design/05` and `06` would catch this class of problem.

### 2. As agents, a `sex` dimension has nothing to do with the agents' sex

Type: bug

Evidence (`ex22.py`, `ex23.py`): `sir.stratify(sex=['female', 'male'])` creates a new random per-agent property `people.sex`, drawn independently of `people.female`. They agree for 48% of agents, i.e. by chance. With an `ss.MFNet` (which pairs on `people.female`) and `adjust={'infection': [0.0, 1.0]}` ("only men can be infected"):

```text
infected women (by people.female): 51   infected men: 51
results select sex=female cum: 0.0   male: 52.0
```

So the results say no women were infected, while the network infected women and men equally. In a compartmental run `sim.py:279` does treat `sex` as standing in for `female`, so the two solvers disagree about what `sex` means. The spec (05 §Dimensions are agent properties) says an existing property is used ("age, female, ..."), and every spec example spells it `sex=['female', 'male']`.

Proposal: map a dimension called `sex` onto `people.female` (levels `'female'`/`'male'`, in either order, by label), with no new attribute and no random draw, and make `split=` on it set `p_female`. More generally, keep a small table of dimension names that alias built-in people properties (`age` to `people.age`, `sex` to `people.female`) in one place, and have the agent engine and the compartmental engine both read it.

Cost: small; a hidden special case, but it is the one users will expect.

### 3. A dimension's name silently decides whether places mix

Type: design

Evidence (`ex5.py`): the same two-level model seeded in the first level, with no network, gives:

```text
region     kind=space  infections by level [4693.    0.]
location   kind=plain  infections by level [4693. 4702.]
place      kind=plain  infections by level [4693. 4702.]
province   kind=plain  infections by level [4693. 4702.]
state      kind=plain  infections by level [4693. 4702.]
district   kind=space  infections by level [4693.    0.]
```

`kinds` (transitions.py:2428) maps only `patch/node/region/district/county` to space. Space changes the default mixing from homogeneous to "within each place only", which is the difference between an epidemic everywhere and an epidemic in one place. The repo's own tests have to write `place=[...], kind='space'` (test_dimensions.py:305, 457), and `location=` is used as a plain dimension in test_mixing. `describe()` prints `region (space)` without marking the kind as guessed. The same applies to strain: `resistance`, `variant`, or `genotype` silently restructure the force of infection and, by default, split initial infections 50/50 between the levels, so `resistance=['DS', 'MDR']` starts with half the cases MDR unless `split=` is given (`ex20.py`).

Proposal: infer kind from what was given, not from the name.

- `age` stays name-inferred (it is always age, and needs numeric edges anyway).
- Space if the levels are an `ss.geo` or `mobility=` is given; otherwise plain. A patch model written as a bare list with no movement is then homogeneous across patches, which is what a user who wrote no mixing would expect. Someone who wants isolated patches writes `sir.stratify(patch=ss.geo(names=[...]))` or `kind='space'`.
- Strain if `immunity=` is given or `kind='strain'`; keep the name list only as a printed suggestion ("'resistance' looks like a strain; pass kind='strain' to give each strain its own force of infection").
- Mark any inferred kind `[guessed]` in `describe()`.

```python
# before: the name does the work, invisibly
sir.stratify(region=['north', 'south'])                 # isolated
sir.stratify(location=['north', 'south'])               # mixed
# after: the value does the work, visibly
sir.stratify(region=['north', 'south'])                 # plain: mixed, like any other dimension
sir.stratify(region=ss.geo(names=['north', 'south']))   # a place: isolated unless mobility= says otherwise
```

Cost: a breaking change for code that relies on `patch=[...]` meaning isolated patches. It is a small one, since `ss.geo(...)` or `kind='space'` is a one-word fix, and the error it removes is silent. Principle 3 ("an error is for when a wrong guess would be silently harmful") cuts against guessing from names here.

### 4. Two stratify parsers that disagree; and the sim/disease distinction is cosmetic

Type: design / bug

Evidence (`ex2.py`, `ex3.py`):

- `Graph.stratify()` (transitions.py:1818) and `Sim.init_strats()` (sim.py:547) parse the same keywords separately. The sim version pops only `split, adjust, kind, where, mobility`, so `ss.Sim(stratify=dict(cd4=[0, 200], states=['I']))` creates two dimensions, `cd4` and `states` (`[('cd4', None), ('states', None)]`). `immunity=` and `ss.transfer` keywords (`acquired=ss.transfer(...)`, documented in 05 for `stratify`) become dimensions in the same way.
- A typo in a sim-level `adjust` is silently ignored (`adjust={'recovry': ...}` runs). `Engine.init_adjust()` only applies names that exist (`elif name in self.adjust`), and the `E301` check exists only in `Graph.stratify`. A sim-level `adjust` naming a transition that only one of two diseases has applies to that one and is silently skipped for the other.
- The one-dimension-per-call rule for options is enforced on the disease form only.
- Stratifying one disease by age stratifies the whole sim: with `flu.stratify(age=[0, 18])` and an unstratified `rsv`, `sim.results.rsv.n_S.by('age')` works and `sim.strats == ['age']`. So "stratify the sim for people properties, the disease for its own dimensions" is a rule about *which `adjust` names refer to*, not about structure. `E308` is the only way the two forms differ.

Proposal:

1. One parser: a `Strat.from_kwargs(**kwargs) -> (strats, transfers)` used by `Graph.stratify`, `Sim.stratify`, and `Sim(stratify=)`, with the same validation everywhere. That deletes the duplicated option list in `init_strats()`.
2. State the actual rule in 05 and drop the two-places framing: *a dimension always belongs to the sim; where you declare it decides only the default `states=` and which module's transitions `adjust` names*. A sim-level `adjust` names transitions as `'flu.infection'`, or by bare name if it is unambiguous, and an unknown name is `E301` naming every module's transitions.
3. With that rule, `sim.stratify()` and `ss.Sim(stratify=)` are two spellings of one call. Keep the keyword (it reads well in the overview example) and keep `sim.stratify()` as its method form, but there is no separate concept to document.

Cost: low. It removes code and a paragraph of spec, and fixes three silent failures.

### 5. Default populations: uniform ages to 100, and named data that does not bring its population

Type: design (guessed defaults)

Evidence (`ex1.py`, `ex1b.py`, `ex10.py`, `ex15.py`):

- The overview's flagship model (`stratify=dict(age=[0, 15, 65]), networks=ss.contacts('Kenya'), total_pop=50e6`) gives Kenya a population that is 15% under 15 and 35% over 65. The true figures are about 40% and 3%. `describe()` marks it `[guessed: ages uniform from 0 to 100]`, but a user who wrote `ss.contacts('Kenya')` has every reason to think Kenya's population is in use. Peak infections in the 65+ group come out at 8.3M, against 1.5M people over 65 in Kenya.
- As agents, the population changes when you stratify: unstratified `ss.People` ages are uniform 0–60 (people.py:234), but adding `stratify=dict(age=...)` makes them uniform 0–100 (max age 60.0 becomes 100.0, and the share over 65 goes from 0 to 0.35). The comment at transitions.py:2803 ("As in ss.People: ages are uniform up to 100") is wrong. A stratification is supposed to describe structure, not change who is in the model.
- `ss.geo('Kenya')` carries county populations summing to 52M and sets the split, but `total_pop` stays at the 10,000 default, so each county has about 200 people. `ss.people('Kenya')` does set `total_pop`.

Proposal:

- Named data implies its population unless one is given: `ss.contacts('Kenya')` brings `ss.people('Kenya')`'s age split if the sim has no people, printed as a guess; `ss.geo(...)` with populations sets `total_pop` to their sum. One named country in a sim, through any loader, means that country's people.
- Use one default age distribution everywhere (agents and compartments, stratified or not). If no data is given, a generic declining pyramid (for example the UN world-average distribution, which `starsim_data` already has) is a far better guess than uniform 0–100 and is still printed as a guess.

Cost: changes default numbers in existing results and tests; no new vocabulary.

### 6. Aging is on by default with no deaths, and the 1/width rate distorts real pyramids

Type: design (guessed default) / ODE–agent mismatch

Evidence (`ex1c.py`): one year of the overview model with `people=ss.people('Kenya')`, no demographics:

```text
ode  start [20.5M 30.2M 1.46M]  end [19.2M 31.0M 2.07M]     # 65+ grows 42% in a year
abm  start [20.6M 30.2M 1.46M]  end [19.2M 31.4M 1.60M]     # 65+ grows 10%
```

The ODE ages people out of 15–64 at `1/50` per year, which is correct only for a uniform age distribution; Kenya's 15–64 group has few people near 64. And with no deaths, the elderly accumulate under both solvers. In the age-specific attack rate this shows up as 1.39 infections per person over 65 (`ex1b.py`), because the denominator a user would naturally take (the starting population) is wrong by the end of the run. The switch was deliberate (implementation.md line 101: "it was off unless there were births or deaths; now on"), but for an epidemic model of a year or less it is all cost: there is a bias, an ODE/agents mismatch, and (finding 7) a large slowdown.

Proposal: (a) default `use_aging` on only when the sim has births or deaths, or when the run is long relative to the narrowest age band (say, more than a fifth of it); otherwise off, with a note. (b) When aging is on in a compartmental run, set the rate out of band `i` from the density at its upper edge, `a_i = n(edge_{i+1}) / N_i`, taken from the split's own interpolation (the same interpolation `init_strats` already does for named populations), rather than `1/width`. That keeps a stationary pyramid stationary, and it is the standard correction.

Cost: (a) reverses a recent decision; (b) is about ten lines in `init_movers`.

### 7. Movement between groups is a dense groups-by-groups matrix: default aging is 100 times slower

Type: performance

Evidence (`ex19.py`, `ex19b.py`): an SIR with 16 age groups by 290 places (4,640 groups), 100 days:

```text
no aging, no migration   0.16 s
aging (the default)     15.92 s
migration only          18.89 s
movers: 2, 344 MB of dense matrices, max RSS 1.0 GB
```

`Compartmental.init_movers()` (solvers.py:1609) builds an `ng × ng` matrix for aging, for each transfer, for migration, and for each sim-level move, even though every one of them moves people along *one* dimension with a rate that depends on that dimension's level (and, for a transfer, possibly on the group). `fold_matrix` adds another `ng × ng` pass. With the spec's own "it is easy to get there by accident" model (4,512 groups per state), every transfer costs 160 MB.

Proposal: represent a mover as `(dim, R)`, where `R` is an `n_d × n_d` rate matrix between that dimension's levels, optionally times a per-group factor. Apply it by reshaping the compartments to the group grid and contracting along that axis (`np.tensordot`, or a small Numba loop over the stride `group_mults[d]`). That costs `O(ng · n_d)` rather than `O(ng²)` in memory and time, and the same structure works for the kernels' whole-run path. Ragged `where=` folding becomes a mask on the source levels.

Cost: a contained refactor of `init_movers`/`mover_rates`/`step_movers` plus the kernel equivalent; no user-facing change.

### 8. Mixing pools have three implementations, and `ss.contacts` builds N² modules with per-agent state

Type: code simplicity / performance

Evidence (`ex8.py`; solvers.py:3995–4024; networks.py:1887–2160): as agents, `ss.contacts('Kenya')` becomes `ss.MixingPools`, which creates 16 × 16 = 256 `ss.MixingPool` sub-modules, each registering a per-agent `FloatArr('eff_contacts')`. `sim.people` ends up with 276 states for a 4-age SIR. `Agents.infect()` then has three code paths for pools:

1. the counts-based `Mixing.pool_terms` path, used only when there is a spatial dimension;
2. a per-sub-pool loop for `MixingPools` without one;
3. the v3 `MixingPool.compute_transmission()` (a probability `beta*disease_beta*trans*acq`, not a hazard) for anything else.

The counts path gives the same answer and is faster. Adding a dummy one-level patch, which switches to path 1, takes the run from 2.6 s to 0.8 s at 50,000 agents (`ex8.py`: attack by age `[0.896 0.996 0.984 0.876]` and `[0.895 0.997 0.986 0.879]`). The compartmental `Mixing.age_matrix` already reads the sub-pool parameters back out of `MixingPools`.

Proposal: for declared models, make a contact matrix a plain route object (`ss.contacts` returns, say, an `ss.Mixing` holding `matrix`, `age` bands or `dims`, `name`, and `ref`), and use the counts-based `Mixing.term_exposure` under both engines, with or without space. `ss.MixingPool(n_contacts=c)` becomes the 1 × 1 case of the same object. Keep `ss.MixingPools`/`MixingPool.compute_transmission` only for v3 class-based diseases, or retire them. This deletes paths 2 and 3, the `AgeGroup`/`LevelGroup` callables for declared models, and the 256 modules; it fixes finding 1's name collision for free; and it makes "the same matrix under every solver" literally the same code.

Also drop `beta=` on `ss.contacts`/`ss.MixingPool` for declared models. It is a second, multiplicative per-route `beta` alongside the disease's `beta=dict(route=...)`, which the spec already makes the way to say "infectiousness depends on the route". Two ways to say one thing is one too many.

Cost: medium refactor of `networks.py` and `Agents.infect()`; the v3 class API is unaffected if the old classes stay for it.

### 9. `ss.gravity` defaults are meaningless, and the spec's own example sends everyone away 100% of the time

Type: API / design

Evidence (`ex4.py`): with `kenya = ss.geo('Kenya', level=1)` (populations about 1e6, distances about 100 km):

```text
gravity(k=1.0, a=1.0, b=1.0, c=2.0)  mean share of time at home 0.0  min -0.0
gravity(k=0.5, a=1.0, b=1.0, c=2.0)  mean share of time at home 0.0  min -0.0
```

`k × pop_j / d²` is around 100 for neighbouring counties, so `Mobility.time_share` caps each row at a total of 1 (transitions.py:2700) and sets the diagonal to 1 − 1 = 0 (and to −0.0 from rounding). Every resident spends all their time elsewhere, and nothing says so. `k=0.5` is the example in 05 §Space, 06 §Mobility, and metapopulation.md. Because `k` has units of `distance^c / pop^(b+a−1)`, there is no default that is right for both a three-town toy (the tests use `k=2e-4`) and Kenya.

Proposal: let the gravity model give only the *shape* (`a, b, c`) and let `away=` give the *amount*, defaulting to a printed guess (for example 10% of time away, or a required argument):

```python
# before
sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))      # silently 100% away
# after
sir.stratify(patch=kenya, mobility=ss.gravity(c=2.0, away=0.1))                  # 10% of time away, split by gravity
```

Keep `k=` for anyone who has calibrated it, but warn (`W` code) whenever the cap binds for any row, naming how many places it binds for. Also stop returning `-0.0` on the diagonal (clip at 0).

Cost: changes the gravity signature; the cap warning alone is a two-line fix.

### 10. `ss.Mobility` mixes two concepts, and movement between levels has six spellings

Type: design

Evidence: `ss.Mobility` (transitions.py:2554) has four kinds plus a register. `commute`, `visit`, and `force` change *who meets whom* (they produce a `time_share` used in `Mixing.by_place`), while `migrate` and `data=` change *where people are* (they become movers or register events). Separately, people move between levels by aging (generated), `ss.transfer(...)` (rate), `ss.transfer(..., coverage=)` (an intervention), `move=` on a transition, `Mobility(kind='migrate')`, and `Mobility(data=)`. `kind='visit'` is used by the flepiMoP importer and the LASER skill but is not in 06 or the vocabulary; `force` exists to match LASER. Principle 9 says a feature that pays off in one framework's use case does not go in, and `force` and `visit` together are two kinds for two frameworks. And 05 says "who mixes with whom is given to the sim, not to `stratify`", while commuting, which is mixing, is given to `stratify`.

Proposal:

- Movement of people is always a transfer. `ss.transfer(patch=M, rate=...)` or `ss.transfer(patch=ss.gravity(...))` is migration (a matrix value means "from row to column"), and `ss.transfer(patch=ss.register('movements.csv'))` is a register. That takes `kind='migrate'` and `data=` off `ss.Mobility`, and aging is described as "a generated transfer", which is what the code already does.
- Coupling of mixing stays as `mobility=` on a spatial stratification (it reads well and needs the geography), with two kinds in the main vocabulary, `commute` and `visit` (`visit` is the asymmetric one, and flepiMoP's model is a mainstream one, not a one-off). `force` becomes an interop-only option documented in the LASER guide.

```python
# before
sir.stratify(patch=kenya, mobility=ss.Mobility(ss.gravity(), kind='migrate'))
sir.stratify(herd=herds, mobility=ss.Mobility(data='movements.csv'))
# after
sir.stratify(patch=kenya, moving=ss.transfer(patch=ss.gravity(away=0.01)))
sir.stratify(herd=herds, sales=ss.transfer(herd='movements.csv'))
```

Cost: a breaking rename for a new feature with few users yet. It shrinks `Mobility` to a mixing concept and reduces "how do people change level?" to one word.

### 11. Partial cross-immunity between strains: `auto` picks an ODE that is off by a factor of two

Type: design / ODE–agent mismatch

Evidence (`ex13.py`): two flu strains with waning and `immunity=[[1, 0.3], [0.3, 1]]`, cumulative infections per person by strain:

```text
no immunity=   ode [0.159 1.51 ]   abm [0.14  1.517]
immunity=      ode [2.066 2.675]   abm [1.285 0.927]
```

The ODE (chosen by `solver='auto'`) gives 1.6 and 2.9 times the agent result, and reverses which strain dominates. The build prints a note, but principle 3 and 00-overview §Four properties say `auto` "never drops individual-level structure; the one thing it does unasked is represent a dwell time by Erlang sub-stages". Keeping only the last strain is exactly dropping structure.

Proposal: either (a) under `auto`, a strain dimension with partial `immunity=` runs as agents, with the reason printed; or (b) better, make the compartmental form exact by tracking the *set* of strains each person has had as a hidden dimension with 2^k levels (4 levels for two strains, 8 for three). That is the textbook history-based multistrain ODE (Gog and Grenfell style), it is small for the k ≤ 3 that covers nearly all real use, and it removes a documented known issue. Above some k, fall back to (a) with a note.

Cost: (a) is one line in the solver choice; (b) is a generated dimension in `init_immunity`, using machinery that already exists.

### 12. The network ODE approximations do not match the spec, and `PartnerNet`'s ignores duration

Type: doc-code mismatch

Evidence (`ex17.py`, `ex18.py`): 06 §Routes says sexual networks run as an ODE are "approximated by partner-change rates between sex/risk/age groups", with per-partnership transmission `1 − (1 − β)^(acts × dur)`. In the code (`Mixing.init`, solvers.py:150):

- `ss.MFNet` raises `E804 ... MFNet has no compartmental approximation yet`.
- `ss.PartnerNet` is approximated by homogeneous mixing with `n_contacts` *contacts per day*, where `n_contacts` is partners at a time. Duration is ignored, so equilibrium SIS prevalence is:

```text
dur=5:   abm 0.670   ode 0.714
dur=50:  abm 0.449   ode 0.714
dur=500: abm 0.168   ode 0.714
```

This only bites when someone asks for `solver='ode'` (auto runs agents), but then the note ("approximated by homogeneous mixing with 0.7 contacts per day") is the only warning, and the number is not a contact rate at all.

Proposal: implement the spec's form for `PartnerNet` and `MFNet`, which are both parameterized by mean partners `n`, duration `D`, and per-act or per-day `β`: partnership formation rate `n/D` per person, and per-partnership transmission `1 − exp(−β·D)` (or the per-act version). It is about ten lines, the same for both. Until then, 06's table should say "E804" for `MFNet` and give the actual formula for `PartnerNet`.

Cost: small.

### 13. `dependence='density'` is in the spec and the vocabulary but not in the code

Type: doc-code mismatch / vocabulary

Evidence (`ex14.py`): `ss.MixingPool(n_contacts=10, dependence='density')` raises `ValueError: 1 unrecognized arguments for mixingpool: dependence`. 06 §Frequency and density dependence specifies it with `n_ref`, and 17-vocabulary lists `dependence` and `n_ref` on `ss.MixingPool`. implementation.md (departures) says "there is no density-dependent mixing", but only in passing, in an interop bullet. `describe()` still prints "frequency-dependent".

Proposal: delete it from 06 and 17 rather than build it. Density dependence is already expressible exactly as `S >> I @ beta * I.n` (06 §Counts used literally), which is what the importers do. That shrinks the vocabulary by two keywords, consistent with principle 9.

Cost: none; it is a documentation change.

### 14. Contact matrices are aggregated by band width, not by population

Type: doc-code mismatch

Evidence (`ex21.py`): 05 and 06 say "the matrix is aggregated to the model's groups, weighted by population". `Mixing.age_matrix` (solvers.py:185) weights each source band by its *width* within the model group (`overlap/width`), even when `ss.people('Kenya')` is given, and treats the open 75+ band as 75–100. So for a 65+ group, the 75+ band gets weight 25/35 = 0.71. Kenya, 3 groups:

```text
pop-weighted  [[9.60 6.16 0.47] [3.42 11.64 0.54] [1.95 4.81 0.67]]
starsim       [[9.64 6.16 0.47] [3.20 10.92 0.58] [1.81 4.31 0.65]]
```

That is 6–10% low on the adult and elderly rows, and it is a bias, not noise.

Proposal: weight by the sim's age split interpolated to the bands (the same `np.interp` that `init_strats` uses for named populations), and fall back to width only when the split is itself the uniform guess.

Cost: small.

### 15. Inconsistent and overloaded keywords

Type: API

Evidence (`ex16.py`, `ex15.py`, networks.py:2161):

- Duration of contacts is `dur` on `RandomNet`, `RandomExactNet`, and `PartnerNet`, but `duration` on `MFNet`. The other spelling raises "unrecognized argument" each way. 17-vocabulary lists `dur=` for networks.
- `ss.contacts(M, age=[0, 15])` means "the lower edges of M's bands", but `ss.contacts(age=[[...]])` means "a matrix over the model's age dimension". The same keyword holds edges in one call and a matrix in the other.
- `ss.geo(['north', 'south'])` is read as a list of countries (`E605: the place "north" was not found ... did you mean Norway`), while `ss.geo(names=['north', 'south'])` is a list of labels.

Proposal: accept `dur` on `MFNet` (keep `duration` as an alias); use `ss.contacts(M, bands=[0, 15])` for the edges, so `age=` always means "matrix over the age dimension"; and have `ss.geo` with a list try countries first but, if none match, treat the list as labels with a note, or require `ss.geo(country=...)` for lookups.

Cost: small, backwards-compatible with aliases.

### 16. Users must remember `sim.groups_changed()`

Type: API

Evidence: implementation.md known issues, and sim.py:807. Code that writes `people.patch` (or any property a dimension is built from) must call `sim.groups_changed(uids)`, or the agents' groups go stale silently. This is the kind of rule principle 1 says the user should not have to know.

Proposal: make the arrays that back dimensions mark the group cache dirty on `__setitem__` (`ss.ModifiableArr` already intercepts writes for `track_writes`). Then `groups_changed()` becomes internal.

Cost: a small overhead on writes to those few arrays only.

### 17. `E305` on an identical re-run

Type: API (minor)

Evidence (`ex11.py`): re-running `sir.stratify(age=[0, 15, 65])` raises `E305`, with a helpful message that says this is what a re-run cell gives. The message is good, but principle 3 says an error is for a guess that would be silently harmful, and an identical repeat has an obvious safe guess.

Proposal: if the dimension exists with identical levels and options, do nothing and print a one-line note; raise `E305` only when the repeat differs.

Cost: trivial.

### 18. Smaller doc–code and describe mismatches

Type: doc-code mismatch

- metapopulation.md §Space is a dimension writes `ss.geo('Kenya', level='county')`, which raises `ValueError: could not parse administrative level from 'county'` (`ex15.py`). 05 and 06 use `level=1`.
- implementation.md, "Departures from the spec", says "`ss.geo` takes the places directly ...; there are no named geographies", which contradicts its own checklist line 117 and the code. It should be deleted.
- With a contact matrix, `describe()` prints `beta = 0.05 per day` (`ex1.py`); per 06 §What beta means, with a route `beta` is a probability per contact, and `describe()` is supposed to say so.
- In a model with space and no network, `describe()` prints the rate as `0.3 × I.n / N  [homogeneous mixing]` and, on the mixing line, "homogeneous within each region". The rate line should be `I.n[place] / N[place]`.
- `kind='visit'` (finding 10) and the gravity formula's `pop[i]^(a−1)` (listed as a departure) should be in 06, not only in implementation.md.
- A first age edge above 0 (`age=[15, 25, 35, 50]`, 05 §Stratifying part of the model) puts everyone under 15 in the 15–24 bin without comment (`ex22.py`). Either say so in 05 or add an implicit `<15` level.

## What is good and should be kept

- One `stratify()` operator for age, risk, place, strain, and boolean properties, with multiplicative `adjust` only. Results are read with `.select()`/`.by()`, never by generated names. This is the right design, and it is pleasant to use.
- The model text does not change when the routes change: the same `ss.Disease` ran unmodified under no network, `MixingPool`, `contacts`, and `RandomNet`, ODE and agents, with closely matching results (`ex6.py`; attack 0.992–0.993 in every case).
- `ss.transfer` is small and correct: coverage on a 16-group model delivers exactly 0.70, not `1 − (1 − f)^P` (`ex9.py`), and a rate transfer matches `1 − e^{−1}` under both solvers.
- Errors such as `E305`, `E307`, and `E308` have excellent messages that name the rule and the fix (`ex11.py`).
- `ss.geo('Kenya')`, `ss.people('Kenya')`, and `ss.contacts('Kenya')` bundled offline, with provenance in the spec and hash, are a real advantage over every framework surveyed. Findings 5 and 9 are about wiring them together, not about the data.
- `PartnerNet`'s degree-target approach (stationary seeking probabilities) is elegant and recovers its targets (`ex17.py`: mean degree 0.687 against 0.70, concurrency 0.216 against 0.22, max 3), and `net.check()` is a good diagnostic.
- Internally, every kind of movement already reduces to one "mover" abstraction in the compartmental engine. Finding 7 is about its representation, not the idea.
