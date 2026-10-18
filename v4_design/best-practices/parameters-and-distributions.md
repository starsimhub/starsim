# Parameters and distributions

## Recommendation

1. **Declare parameters; refuse unknown names.** A misspelled parameter should be an error at build time, not a silent behavioral difference.
2. **A parameter may be a number, a typed quantity, a distribution, a time series, a seasonal curve, a per-group table, or a function.** Every slot accepts all of them.
3. **Keep values with the model by default, and make it possible to separate them.** camdl is right that structural identity should be separable from values; it is wrong that this must be mandatory.
4. **Let a module say what data it needs, as data.** epymorph's requirement contract is the most transferable idea in the review after typed time. Here it is optional metadata, used to show what a model needs and which of it is missing (`sim.requires()`) before anything runs; it is not enforced. A data-entry form generated from it is deferred ([19-open-questions.md](../design/19-open-questions.md#c-deferred-not-in-v1)).
5. **Record where a value came from.** One optional field, `source=`, answers the question every reviewer asks.
6. **Parameters have shapes, and the shape is checked.**

## Why

### Declaration catches typos; no framework without it does

[EpiModel](../approaches/epimodel.md): "`param.net(inf.prob = 0.4, act.rate = 2, my.new.param = 0.7)` is legal… There is no parameter schema and no declaration step; a misspelled parameter name is not an error until something looks for it." Its own review lists "No declaration step of any kind — parameters, attributes, and outputs all spring into existence on first use. Typos become silent behavioral differences" as a limitation.

Against that: [MEmilio](../approaches/memilio.md) gets compile-time checking from C++ tag types (`parameters.set<TimeInfected>(2)` — "the strongest parameter-name checking in the review"); [camdl](../approaches/camdl.md) gets it from its compiler; **Starsim gets a runtime version from `define_pars`**: passing a parameter that was not declared (`ss.SIR(foo=3)`) raises an error naming it. Starsim's mechanism is already the right one for Python. Keep it.

### Where values live

[camdl](../approaches/camdl.md) takes the strongest position in the review:

> **parameter values are never declared in the `parameters` block**. That block carries names, kinds, dimensions, and optional priors only. Values come from a `params.toml`, a CLI `--param`, or a named scenario's `set = { ... }`.

with a factored run identity (model hash, params hash, config hash, scenario hash, fit hash) so that "a reviewer asking 'is this a structural change or a parameter sweep?' reads the level hashes rather than diffing files."

The mechanism is excellent and the *mandatory separation* is wrong for our audience. An epidemiologist writing an SIR should not need two files. The resolution: **values default to living inline, and the split is available**, with the hashes computed either way.

### Requirements as data

[epymorph](../approaches/epymorph.md)'s `AttributeDef` is the standout:

```python
requirements = [
    AttributeDef("beta", type=float, shape=Shapes.TxN, comment="infectivity"),
]
```

From its review: "A model states its data interface *as data*, so it can be checked before anything runs… Data provenance is explicit… Shape checking catches an entire class of broadcasting bug at bind time. Nothing else in the review makes a model's data dependencies declarative in this way. Most frameworks have the model reach out and get what it needs."

[Atomica](../approaches/atomica.md) does the same thing in a different medium: the databook — a data-entry spreadsheet with exactly the fields the model needs, per population, with the right units and labels — is **generated from the framework**.

### Distributions do double duty

A distribution in a parameter slot means one of two things, and the slot says which: in `dur=` or `beta=` it is variation between people (each agent draws its own value), and in `prior=` it is uncertainty for fitting (camdl's `~ prior`). Keeping the two in separate slots is what avoids the ambiguity [EpiModel](../approaches/epimodel.md) has, where "a vector-valued parameter means a time series here and a sensitivity sweep there".

### Shapes

[epymorph](../approaches/epymorph.md)'s `Shapes.Scalar` / `N` / `T` / `TxN` / `NxN` are checked when data is attached. Its review states the complement precisely: "Dimensional types say *what a number means*; shapes say *how many of them there are and along which axis*. A language spanning agents, patches, strata, and time needs both." [SimInf](../approaches/siminf.md)'s `gdata` versus `ldata` (global versus per-node parameters) and [MetaCast](../approaches/metacast.md)'s `universal_params` versus `subpop_params` are two independent arrivals at a coarse version of the same thing. Here the shape is taken from the value (`ss.by(age=[...])` has one value per age group) and checked against the model's stratification, with an error (`ss.ShapeError`) that names the dimension and its levels.

### Named data, not just values

[Epydemix](../approaches/epydemix.md) loads a country's population and contact matrices by name (`load_epydemix_population("United_States", contacts_source="mistry_2021", ...)`). Its review: "A lingua franca that can express a contact matrix but cannot say *which* contact matrix, from *which* source, for *which* place, has moved the problem rather than solved it." epymorph's ADRIOs are the pull-based version with caching and provenance; camdl has `read("data/lga_pop.tsv", column="patch")`; [LASER](../approaches/laser.md) ships a library of migration models. All four say the same thing: **a reference to data is part of the model**, and where it came from belongs in the description of the run and its hash.

## The specification

[04-parameters.md](../design/04-parameters.md) specifies `ss.par` (with `bounds=`, `prior=`, `source=`, and `doc=`), the forms a value can take (a number, a typed quantity, a distribution, `ss.timeseries`, `ss.seasonal`, `ss.by`, or a function), values kept in a file (`ss.Sim(pars='params.csv')`), requirements (`ss.Requires` and `sim.requires()`), and named data (`ss.people`, `ss.contacts`, `ss.geo`, from `starsim_data`). The distributions and the units of bare numbers are in [02-types.md](../design/02-types.md), fitting with bounds and priors in [13-inference.md](../design/13-inference.md), and the run hash in [11-random.md](../design/11-random.md).

## Trade-offs

- **Declaration means adding a parameter to a custom module is two edits** (declare, use). Starsim already works this way and it is worth it. Most parameters live in their transition and are never named at all.
- **A requirements table is only as good as the requirement declarations.** A module that reads something it did not list will be missing from the table (and from a data-entry form generated from it, if one is built). Enforcing the list (Vivarium's `PopulationView`, where a module sees only what it declared) would close that gap, but people forget declarations and they drift from the code, so it is left as optional metadata.
- **A function in a parameter slot is only known to work under the solvers it has been tried with.** `sir.solvers()` reports what happened; nothing is promised in advance.
- **Bundled data goes stale and is large; pulled data needs a network and can change underneath you.** So do some of each: bundle what is small and common, pull everything else with a cache (epymorph's approach), and version both. `starsim_data` keeps each UN revision as a separate store, so a new release adds data and does not change what an existing model loads, and the source goes into the hash so a changed upstream is detectable.
- **Optional `source=` will mostly be empty.** True, and it costs nothing when unused, and the models that matter most are the ones where someone bothered.

## Rejected

- **Open parameter lists** (EpiModel's `...`). The cost is documented.
- **Mandatory value/model separation** (camdl). The capability is right; requiring two files for an SIR is not.
- **String-concatenated parameter names** ([MetaCast](../approaches/metacast.md)'s `parameters['p' + subpop_suffix]`, `'beta_[high,vaccinated]'`). Its review: "the lesson is in the split, not the mechanism." Shapes do this job properly.
- **`Use_Defaults: 1`** ([EMOD](../approaches/emod.md)), where "much of a model's behavior is not visible in its files". We guess defaults aggressively and *print every one*, which is the opposite policy despite sounding similar.
- **New distribution names, for now.** Starsim's names are used (`ss.lognorm_ex`, not a new `ss.lognorm`; `ss.nbinom`, not `ss.neg_binomial`). There was at first no separate `ss.erlang`: an Erlang distribution is exactly a gamma distribution with an integer shape (`ss.gamma(a=3, scale=...)`), and its main use in this field, splitting a state into sub-stages, is `stages=3` on the transition (`ss.erlang()` has since been added, as a friendlier spelling). Starsim's review flags "~20 distributions" as part of the surface-area problem, so the documentation teaches the dozen listed in [17-vocabulary.md](../design/17-vocabulary.md); the others, and any SciPy distribution via `ss.Dist`, remain available. The list itself is not fixed (see open questions).
- **Enforced requirement views** (Vivarium). See the trade-off above.
- **Repeating the unit on every distribution parameter.** Once is enough.

## Open questions

The open questions from this document are in [19-open-questions.md](../design/19-open-questions.md).
