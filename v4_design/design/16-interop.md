# Working with other frameworks

Importing models from, and exporting them to, the other frameworks reviewed in [approaches](../approaches). This is different from changing the solver ([12-backends.md](12-backends.md)), which runs one model in different ways; this moves a model between languages.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md) and the reviews in [approaches](../approaches).

## The priority

People have to want to write models in this language for any of this to matter. An interchange format that nobody writes in ends up as an export target that every tool supports partially. So the order is: first, a model should never be easier to write without the lingua franca than with it; then import and export.

## One build step for every importer

An importer reads another framework's model into an `ss.interop.Source`: its states, parameters, flows as rate expressions in the source's own terms, stratifications, outcomes, and what it could not translate. `ss.interop.build()` turns a source into a sim by the same rules for every format (below), so each framework needs one reader. An exporter writes a sim's compartmental form in another format. There are no direct framework-to-framework converters.

```
EMULSION, odin, SBML, SimInf,                                                        ┌─▶ SBML, odin, Mermaid (ss.interop.export(..., format=))
flepiMoP, summer2, Atomica, camdl,  ─▶ reader ─▶ ss.interop.Source ─▶ build() ─▶ sim ┤
EpiModel, epydemix, MEmilio                                                          └─▶ Python that builds it again (sim.to_code())
```

## Import

<!-- fragment -->
```python
import starsim.interop # Not imported by import starsim
sim = ss.interop.load('model.xml',  format='sbml')
sim = ss.interop.load('model.yaml', format='emulsion')
sim = ss.interop.load('config.yml', format='flepimop')
sim = ss.interop.load('framework.xlsx', format='atomica', databook='databook.xlsx')
sim = ss.interop.load(summer_model)                 # a Python object from another framework; format taken from its type
```

The result is an ordinary sim (`ss.interop.formats` lists the importers; the format is told from the file or object where it can be). `sim.describe()` prints it, it can be stratified, run, and fitted like one written by hand, and `sim.to_code()` writes it as a Python script that builds it again, which is the usual next step. `sim.imported` records how it was read: the format and file, how each rate was read (`choices`), what could not be translated (`incomplete`), and the prompt for finishing it (`prompt`). A framework with no importer can still fill in an `ss.interop.Source` by hand and call `ss.interop.build()`.

An importer SHOULD:

1. **Keep names.** A transition called `infection` in the source is called `infection` here.
2. **Mark what it guessed.** Values the importer filled in are marked `[guessed]` in `describe()`, exactly as for a hand-written model.
3. **Keep going.** Something that does not map onto a built-in piece is kept as a literal expression or a function (below), not dropped and not refused.
4. **List what it could not translate at all**, by name (`ImportIncomplete`), in `sim.imported.incomplete`, and in the code `sim.to_code()` writes. Every key of the source the importer does not read is listed, not dropped.

An incomplete import is not refused. It runs without the listed parts, warning again when it is first run, and it is finished by a person or an AI assistant with the `import-model` skill: the warning ends with the instruction to give the assistant (also kept as `sim.imported.prompt`), and the skill's "Finishing an incomplete import" works through the list and checks the result with the `validate-import` skill. A part that needs a feature Starsim does not have is left listed and reported as a gap, not forced.

### Classifying rates on import

Most frameworks write a transition as a general rate expression: `beta * S * I / N`, `gamma * I`, `mu * N`. There is one transition class here, so the importer's job is to write each one as `src >> dst @ rate` with the right kind of rate. The rules, in order:

1. **Strip the source state.** A flow written as a total (`gamma * I` out of `I`) becomes a per-person rate (`gamma`).
2. **A rate that mentions another state, divided by a population total, is exposure to that state.** `beta * S * I / N` out of `S` becomes `S >> I @ beta * I`. The bare `I` means exposure through the model's routes, which with no network is `I / N` ([06-routes.md](06-routes.md)).
3. **A rate that mentions another state with no division by a total** (`beta * S * I`) is density-dependent mass action. It is kept as counts, `S >> I @ beta * I.n`, which is exact; the choice is printed ("density-dependent, kept as counts"). There is no density-dependent mixing to set.
4. **A rate that mentions only parameters** is a spontaneous transition: `I >> R @ gamma`. With no source it is an inflow from `ss.null` (a birth), and with no destination an outflow to it (a death).
5. **Anything else is kept exactly as written, as a literal expression of counts**: `S >> I @ beta * W.n / (W.n + K)`. Counts (`.n`) are used literally, so any arithmetic the source framework could write can be carried over. This is correct by construction, and it runs under every solver.
6. **A compartment that is not people is an ordinary state.** An environmental reservoir, a vector population, or a continuous per-node variable (SimInf's `v`) becomes an `ss.State` in the same disease as the compartments it is coupled to, with its inflows and outflows written against `ss.null` and the rates that use it written with counts (`W.n`). The library works out that it is not people from the transitions; the importer writes `ss.State(people=False)` anyway, because it knows.
7. **If the source is code that cannot be turned into an expression** (an arbitrary R or Python function), it is kept as a function in the rate slot and the import says so. The model then runs wherever that function runs ([03-states-and-transitions.md](03-states-and-transitions.md#functions-in-any-slot)).

Rules 2 and 3 are guesses about what the source meant, and the difference matters when the population changes. The importer prints which it chose for each transition, and `describe()` shows the result. If an expression matches neither pattern cleanly, rule 5 applies: the literal form loses nothing, it only means the transition will not pick up a network or contact matrix if one is added later.

### What each importer reads

| Source | `format=` | What it reads |
|---|---|---|
| **EMULSION** | `'emulsion'` (YAML) | Each state machine as a disease, with its numbers in EMULSION's time unit; the solver follows the aggregation type (`binomial` for compartments, `abm` for individuals and hybrid models); minimum stays as `after=` |
| **odin** | `'odin'` (the R DSL) | `deriv()` and `update()` equations split into signed terms and paired into flows; `rbinom()` draws as a chain binomial; `lambda`, `if`, contact matrices, seasonality, and counts |
| **SBML** | `'sbml'` (XML) | Species as states, reactions as transitions, classified by the rules above |
| **SimInf** | `'siminf'` (an R script with `mparse()` or a predefined model) | The transitions, run by `gillespie`; several nodes as a stratification of places, with the scheduled events as a movement register (`ss.Mobility(data=...)`); the continuous state `v` as a state that is not people |
| **flepiMoP** | `'flepimop'` (the config YAML) | Compartment products, `proportional_to` rates, the outcomes chain as `ss.outcome`, `seir_modifiers` (every scenario kept in `sim.imported.scenarios`), and drawn parameters with their distributions as priors |
| **summer2** | `'summer2'` (a JSON dump, or the model) | Each flow kind as one transition, `stratify_with` as `stratify`, adjustments as `adjust=`, and the mixing matrix; the derived outputs are worked out from the results by `ss.interop.summer2.outputs()` |
| **Atomica** | `'atomica'` (the framework, databook, and program book workbooks) | Compartments, transitions, characteristics, and the databook's values; the program book's spending as parameters whose coverage moves the parameters they affect |
| **camdl** | `'camdl'` (a `.camdl` file, camdlc's JSON, or a pycamdl model) | The model, with the parameter values of one of its scenarios (`scenario=`), as an ODE or a chain binomial as camdl's backends run it |
| **EpiModel** | `'epimodel'` (a JSON dump from `dump_epimodel.R`, or the R script) | `netsim()`, `dcm()`, and `icm()` runs with the built-in models and networks (`ss.PartnerNet` for the network model's targets); custom modules are listed with their R source |
| **epydemix** | `'epydemix'` (a JSON dump from `dump_epydemix.py`, or the model) | The transitions, the population, and its contact matrices |
| **MEmilio** | `'memilio'` (a JSON dump from `dump_memilio.py`) | The ODE models (`osir`, `oseir`, `oseirdb`, `osecir`), alone, on a mobility graph (as places with commuting), or in a parameter study (its distributions as priors) |

A Starsim 3 model is not imported: its modules written as code still run (with the names removed from Starsim 4 changed, as the migration guide and the `migrate-starsim3` skill say), and turning `step()` logic into declared transitions is done by hand with the `import-model` skill. LASER and the other models MEmilio has are translated by hand, with the skill's guides. Each importer is tested against its framework's examples (`tests/test_interop_*.py`) and in the switching demos (`tests/switching/`).

### What the importers cannot translate yet

Writing the importers found gaps in the language. Each import lists the part it could not translate, and the gaps that remain are tracked in `v4_design/admin/deferred.md`: a transition's current flow cannot be used in a rate (MEmilio's test-and-trace capacity); a count over groups weighted by a parameter that differs between them cannot be written; a parameter of the infectious person's group can be given only as one `rel_trans` for all infectious states together; there is no option for recovery within the step of infection (EpiModel's order); `ss.PartnerNet` has no ERGM terms beyond its targets and matching (triangles, `nodematch(diff=TRUE)`); and an initial number below 1 is always a fraction of the population.

## Export

```python
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
sim = ss.Sim(sir, unit='day')

ss.save('model.py', sim)                     # Python that builds the sim again (sim.to_code())
ss.interop.export(sim, 'sbml', filename='model.xml')
ss.interop.export(sim, 'odin', filename='model.R')
ss.interop.export(sim, 'mermaid', filename='model.md')
```

The code is the complete form of a declared model; `to_code()` writes only the declared parts, and what it does not write (a network or module written as code, named by the class that makes it, a prior that is not a distribution) it lists in the file and as `LossyExport` ([14-model-spec.md](14-model-spec.md#saved-forms)). The model spec (`sim.to_spec()`, [14-model-spec.md](14-model-spec.md)) is the normalized form for `sim.hash()` and `ss.diff()`, and is not read back. Every other format carries less, and each export MUST print what was left out or changed, one line per declaration (`LossyExport`):

```text
>>> ss.interop.export(sim, 'sbml', filename='model.xml')    # stratified by age, with a lognormal dwell time and a random network
LossyExport: exported to SBML: 3 states, 2 transitions, as 13 species and 12 reactions (rates per day)
  changed: the agent-based sim was written as its compartmental form (an ODE)
  changed: the lognorm_ex dwell time "dur" written as 11 exponential stages (Erlang) with its mean, 10 days
  changed: "sir" is stratified (age: 0-14, 15-64, 65+), written as one group, the sum of its groups, with the base values of its parameters
  not exported: network "randomnet" (contacts written as homogeneous mixing)
```

An export does not fail because something cannot be carried; it writes what it can and lists the rest. A model with functions in it exports its declared parts and lists the functions as not exported ([14-model-spec.md](14-model-spec.md#functions)).

Petri nets and the other frameworks' formats are not exported (`v4_design/admin/deferred.md`, "Won't do"). SBML and odin are written from the compartmental form of the model (what `sir.equations()` prints), with the numbers in the sim's unit (SBML's `timeUnits`): a dwell time that is not exponential is written as its Erlang stages, and a stratification as one group, both listed. Both read back with `ss.interop.load()`. `ss.interop.export(sim, format)` returns the text without writing a file.

The diagram is the text of a Mermaid flowchart, from `sir.diagram()`; `format='mermaid'` writes the same diagram for every declared disease in the sim, in one chart. There is no drawn (Matplotlib) diagram: Mermaid is drawn by GitHub, Quarto, and most Markdown viewers. A generated data-entry form (Atomica's databook, `sim.requires(to='form.csv')` in [04-parameters.md](04-parameters.md#requirements)) is deferred from v1.

## Comparing two models

Since both are data, comparing them is mechanical:

```python
sim_a = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir'), unit='day')
sim_b = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.08), recovery=ss.progression('I -> R', dur=ss.gamma(a=3, scale=10/3)), init_prev=0.01, name='sir'), unit='day')
ss.diff(sim_a, sim_b)
```

```text
model:    transition 'recovery' dwell exponential, mean 10 day -> dwell gamma(a=3, scale=3.33333 day)
pars:     beta 0.05/day -> 0.08/day
config:   same
scenario: same
```

The report is in the user's terms (states, transitions, parameters), not a diff of JSON text, and its four lines are the model, parameter, run-setting, and scenario parts of `sim.hash()` ([14-model-spec.md](14-model-spec.md#hashing); the hash also has a part for the data, and the seed). The same function compares an imported model with its hand-written translation.

## The vignette test

Whether this language can express what the other frameworks do is something to test, not assert. [vignettes](../vignettes) translates the [examples](../examples) corpus into the lingua franca, and each translation is checked four ways:

1. **It runs.**
2. **It matches.** Its results agree with the original framework's to a stated tolerance ([18-conformance.md](18-conformance.md)).
3. **It is no longer.** The lingua franca version has no more lines of model code than the original. Where the original calls a built-in written for one situation (`interventions=close_schools_uk()`), the lines inside that built-in count as the original's. This is the hard target.
4. **It is not noticeably slower.** Time is reported as two numbers, the first result and the best of five repeated runs, each against the original's; a vignette fails on time only when it is more than twice as slow and more than a second slower.

Points 3 and 4 are the north star: a model should never be easier to write without the lingua franca than with it. `tests/north_star.py` reads the vignettes and writes the table, failing rows included, and the "Coming from" pages of the docs publish it for each framework. How lines and time are measured is in [18-conformance.md](18-conformance.md#13-north-star).

Where a translation fails any of the four, the vignette says which and why. A list of what does not translate well, with reasons, is a useful result: it says where the design needs work.
