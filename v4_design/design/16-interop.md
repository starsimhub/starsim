# Working with other frameworks

Importing models from, and exporting them to, the other frameworks reviewed in [approaches](../approaches). This is different from changing the solver ([12-backends.md](12-backends.md)), which runs one model in different ways; this moves a model between languages.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md) and the reviews in [approaches](../approaches).

## The priority

People have to want to write models in this language for any of this to matter. An interchange format that nobody writes in ends up as an export target that every tool supports partially. So the order is: first, a model should never be easier to write without the lingua franca than with it; then import and export.

## One build step for every importer

An importer reads another framework's model into an `ss.io.Source`: its states, parameters, flows as rate expressions in the source's own terms, stratifications, outcomes, and what it could not translate. `ss.io.write()` writes a source as Starsim code (a declared `ss.Disease` and its `ss.Sim`) by the same rules for every format (below), and `ss.io.build()` runs that code to make the sim, so each framework needs one reader, and there is one path: the foreign model, then code, then the sim. Starsim never turns a sim back into code. An exporter writes a sim's compartmental form in another format, read from the built model. There are no direct framework-to-framework converters.

```
EMULSION, odin, SBML, SimInf,
flepiMoP, summer2, Atomica, camdl,  ─▶ reader ─▶ ss.io.Source ─▶ write() ─▶ code ─▶ run ─▶ sim ─▶ SBML, odin (ss.export_to(sim, format))
EpiModel, epydemix, MEmilio                                                (sim.imported.code)
```

## Import

```python
sim = ss.import_from('model.xml',  format='sbml') # ss.io is imported when first used, as ss.fit is
sim = ss.import_from('model.yaml', format='emulsion')
sim = ss.import_from('config.yml', format='flepimop')
sim = ss.import_from('framework.xlsx', format='atomica', databook='databook.xlsx')
sim = ss.import_from(summer_model)                 # a Python object from another framework; format taken from its type
sim = ss.io.convert('framework.xlsx', format='atomica', databook='databook.xlsx', filename='model.py') # Read it, and save the code (to run from then on)
```

The result is an ordinary sim (`ss.io.importers` lists the importers; the format is told from the file or object where it can be). `sim.describe()` prints it, it can be stratified, run, and fitted like one written by hand, and its code (`sim.imported.code`, which says where the model came from, how each rate was read, and what was not translated) is the user's to save and change, which is the usual next step. `sim.imported` records how it was read: the format and file, the code (`code`), how each rate was read (`choices`), what could not be translated (`incomplete`), and the prompt for finishing it (`prompt`). A rate is checked before it is written (only arithmetic, comparisons, a few functions, and the names the model declares), so a source's expression is never run as it is; a value the writer cannot write as code (e.g. a network) is given by the importer as `ss.io.Code`, the Python that makes it. A framework with no importer can still fill in an `ss.io.Source` by hand and call `ss.io.build()`.

An importer SHOULD:

1. **Keep names.** A transition called `infection` in the source is called `infection` here.
2. **Mark what it guessed.** Values the importer filled in are marked `[guessed]` in `describe()`, exactly as for a hand-written model.
3. **Keep going.** Something that does not map onto a built-in piece is kept as a literal expression or a function (below), not dropped and not refused.
4. **List what it could not translate at all**, by name (`ImportIncomplete`), in `sim.imported.incomplete`, and at the end of the code it writes. Every key of the source the importer does not read is listed, not dropped.

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

**Named expressions are written once.** A source's named expressions (Atomica's parameter functions and characteristics, EMULSION's and summer2's named quantities) are written out in a rate while it is classified, since the rules read the whole rate. The code then defines each once, as a variable of counts (`ch_prev = inf.n / (sus.n + inf.n + rec.n)`), after those it uses, and writes its name wherever its full form is in a rate or a share. One that is only a state, a parameter, or a number, or that names a transition, stays written out. So the code reads as the source does, and the engine works out each part once (see [10-execution.md](10-execution.md)).

### What each importer reads

| Source | `format=` | What it reads |
|---|---|---|
| **EMULSION** | `'emulsion'` (YAML) | Each state machine as a disease, with its numbers in EMULSION's time unit; the solver follows the aggregation type (`binomial` for compartments, `abm` for individuals and hybrid models); minimum stays as `after=` |
| **odin** | `'odin'` (the R DSL) | `deriv()` and `update()` equations split into signed terms and paired into flows; `rbinom()` draws as a chain binomial; `lambda`, `if`, contact matrices, seasonality, and counts |
| **SBML** | `'sbml'` (XML) | Species as states, reactions as transitions, classified by the rules above |
| **SimInf** | `'siminf'` (an R script with `mparse()` or a predefined model) | The transitions, run by `gillespie`; several nodes as a stratification of places, with the scheduled events as a movement register (`ss.geo.Mobility(data=...)`); the continuous state `v` as a state that is not people |
| **flepiMoP** | `'flepimop'` (the config YAML) | Compartment products, `proportional_to` rates, the outcomes chain as `ss.outcome`, `seir_modifiers` (every scenario kept in `sim.imported.scenarios`), and drawn parameters with their distributions as priors |
| **summer2** | `'summer2'` (a JSON dump, or the model) | Each flow kind as one transition, `stratify_with` as `stratify`, adjustments as `adjust=`, and the mixing matrix; the derived outputs are worked out from the results by `ss.io.summer2.outputs()` |
| **Atomica** | `'atomica'` (the framework, databook, and program book workbooks) | Compartments, transitions, characteristics, and the databook's values (a series without the points equal to those on either side, which change nothing); the program book's spending as parameters whose coverage moves the parameters they affect; a junction whose shares change (with the counts, or in time) as a transition to each destination with its share as a factor of the rate, which is the same flow, since people pass through a junction at once |
| **camdl** | `'camdl'` (a `.camdl` file, or camdlc's JSON) | The model, with the parameter values of one of its scenarios (`scenario=`), as an ODE or a chain binomial as camdl's backends run it |
| **EpiModel** | `'epimodel'` (a JSON dump from `dump_epimodel.R`, or the R script) | `netsim()`, `dcm()`, and `icm()` runs with the built-in models and networks (`ss.DegreeNet` for the network model's targets); custom modules are listed with their R source |
| **epydemix** | `'epydemix'` (a JSON dump from `starsim/io/epydemix.py` run as a script, or the model) | The transitions, the population, and its contact matrices |
| **MEmilio** | `'memilio'` (a JSON dump from `starsim/io/memilio.py` run as a script) | The ODE models (`osir`, `oseir`, `oseirdb`, `osecir`), alone, on a mobility graph (as places with commuting), or in a parameter study (its distributions as priors) |

A Starsim 3 model is not imported: its modules written as code still run (with the names removed from Starsim 4 changed, as the migration guide and the `migrate-starsim3` skill say), and turning `step()` logic into declared transitions is done by hand with the `import-model` skill. LASER and the other models MEmilio has are translated by hand, with the skill's guides. Each importer is tested against its framework's examples (`tests/test_interop_*.py`) and in the switching demos (`tests/switching/`).

### What the importers cannot translate yet

Writing the importers found gaps in the language. Each import lists the part it could not translate, and the gaps that remain are tracked in `v4_design/admin/deferred.md`: a transition's current flow cannot be used in a rate (MEmilio's test-and-trace capacity); a count over groups weighted by a parameter that differs between them cannot be written; a parameter of the infectious person's group can be given only as one `rel_trans` for all infectious states together; there is no option for recovery within the step of infection (EpiModel's order); `ss.PartnerNet` has no ERGM terms beyond its targets and matching (triangles, `nodematch(diff=TRUE)`); and an initial number below 1 is always a fraction of the population.

## Export

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
sim = ss.Sim(sir, unit='day')

ss.export_to(sim, 'sbml', filename='model.xml')
ss.export_to(sim, 'odin', filename='model.R')
```

The code is the complete form of a model ([14-model-spec.md](14-model-spec.md)). The exporters read the built model (each disease's graph and engine), not a serialized form; every other format carries less, and each export MUST print what was left out or changed, one line per declaration (`LossyExport`):

```text
>>> ss.export_to(sim, 'sbml', filename='model.xml')    # stratified by age, with a lognormal dwell time and a random network
LossyExport: exported to SBML: 3 states, 2 transitions, as 13 species and 12 reactions (rates per day)
  changed: the agent-based sim was written as its compartmental form (an ODE)
  changed: the lognorm_ex dwell time "dur" written as 11 exponential stages (Erlang) with its mean, 10 days
  changed: "sir" is stratified (age: 0-14, 15-64, 65+), written as one group, the sum of its groups, with the base values of its parameters
  not exported: network "randomnet" (contacts written as homogeneous mixing)
```

An export does not fail because something cannot be carried; it writes what it can and lists the rest. A model with functions in it exports its declared parts and lists the functions as not exported.

Petri nets and the other frameworks' formats are not exported (`v4_design/admin/deferred.md`, "Won't do"). SBML and odin are written from the compartmental form of the model (what `sir.equations()` prints), with the numbers in the sim's unit (SBML's `timeUnits`): a dwell time that is not exponential is written as its Erlang stages, and a stratification as one group, both listed. Both read back with `ss.import_from()`. `ss.export_to(sim, format)` returns the text without writing a file.

The diagram is a Mermaid flowchart, from `sir.diagram()`, with each arrow labeled with its transition's name and value (`recovery: 10 days`) and, for a transition with `p=`, the share that takes it. It is a string of the flowchart's text (`print()` gives it), which a notebook draws: JupyterLab and Quarto draw it from a Markdown block. There is no drawn (Matplotlib) diagram: Mermaid is drawn by GitHub, Quarto, JupyterLab, and most Markdown viewers.

## Comparing two models

Two models are compared by their results (run both, as the importers' tests do with an import and its hand translation), by `describe()`, or by a text diff of their code; there is no comparison function, since the code is the model.

## The vignette test

Whether this language can express what the other frameworks do is something to test, not assert. [vignettes](../vignettes) translates the [examples](../examples) corpus into the lingua franca, and each translation is checked four ways:

1. **It runs.**
2. **It matches.** Its results agree with the original framework's to a stated tolerance ([18-conformance.md](18-conformance.md)).
3. **It is no longer.** The lingua franca version has no more lines of model code than the original. Where the original calls a built-in written for one situation (`interventions=close_schools_uk()`), the lines inside that built-in count as the original's. This is the hard target.
4. **It is not noticeably slower.** Time is reported as two numbers, the first result and the best of five repeated runs, each against the original's; a vignette fails on time only when it is more than twice as slow and more than a second slower.

Points 3 and 4 are the north star: a model should never be easier to write without the lingua franca than with it. `tests/north_star.py` reads the vignettes and writes the table, failing rows included, and the "Coming from" pages of the docs publish it for each framework. How lines and time are measured is in [18-conformance.md](18-conformance.md#13-north-star).

Where a translation fails any of the four, the vignette says which and why. A list of what does not translate well, with reasons, is a useful result: it says where the design needs work.
