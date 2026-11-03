---
name: convert-model
description: Translate a disease model between Starsim v4 and another framework (EMULSION, odin, SimInf, flepiMoP, summer2, Atomica, SBML, EpiModel, epipack, epydemix, epymorph, LASER, MEmilio, camdl, the Pandemic Simulator, or Starsim v3), in either direction. Use when someone wants to import, export, port, convert, or reimplement a model, or share a Starsim model with people who use another tool, including the parts that do not map one to one; and check a converted model against its original, by running both and comparing their results.
---

# Convert a model to or from Starsim

To bring a model into Starsim, read on. To take one out, see "Exporting a model" below.

A model is imported in two passes. The mechanical pass (`ss.import_from(..., format=...)`) translates everything with an obvious one-to-one mapping and lists what it could not translate. The second pass is yours: translate each listed item by hand into the closest Starsim form, and write the frameworks with no importer by hand from the start. Then check the result against the original (`references/validate.md`).

## Workflow

1. **Find the model and its parts.** Locate the source files (YAML, R, XML, Python, workbooks, data files) and list what the model has: states, flows, parameters and their units, structure (age, place, risk), contacts (mixing, networks), demography, interventions and scenarios, outputs, run settings (time unit, step, duration, population, stochastic or not). Read the framework's guide in `references/` before going further.
2. **Run the mechanical import**, if the framework has one (see the table below):
   ```python
   import starsim as ss
   import sciris as sc
   sim = ss.import_from('model.yaml')                     # the format is told from the file where it can be
   sim = ss.import_from('model.R', format='odin')         # or given
   sc.savetext('model.py', sim.imported.code)              # the model as Starsim code, to edit (the sim is this code run)
   ```
   `ss.import_from()` writes the model as Starsim code (a declared `ss.Disease` and the `ss.Sim`) and runs it to make the sim. It prints how each rate was read and warns `ImportIncomplete` with each piece it could not translate. Both are kept in `sim.imported` (`choices`, `incomplete`) and are comments in the code: the reading at the top, the untranslated pieces at the end.
3. **Check the guesses** in the printed reading and in `sim.describe()` (marked `[guessed]`):
   - A rate with another state divided by the total (`beta*S*I/N`) was read as exposure (`beta * I`, frequency-dependent). One without the total (`beta*S*I`) was kept as counts (`beta * I.n`, density-dependent). Make sure that is what the source meant, especially if the population changes.
   - Units: bare numbers are in the sim's unit (`unit=`). A per-step probability is not a rate: `-log(1 - p)/dt` is the rate that gives it. A probability per time unit is `ss.prob(p, "day")` and the like.
   - Initial numbers: numbers of people are in `init_n=` and shares in `init=` (1.0 is everyone); the rest start in the first state, `first=` (the source's first compartment of people, or a state named `S` or `susceptible`). Check that the source's initial values were numbers where they are in `init_n=`, and that the first state is the right one.
   - The solver: match the source's kind of run (`'ode'` for deterministic, `'binomial'` for discrete-time stochastic compartments, `'gillespie'` for exact event-driven ones, `'abm'` for individuals).
4. **Translate each untranslated piece** (see "Finishing an incomplete import" below) with the closest Starsim form, using `references/idioms.md` (minimum stays, conditions, calendars, capacities, actions on crossing, births and offspring, variables of individuals, scheduled events, interventions, networks, outputs) and the framework's guide. In order of preference: a declared form (a transition, `where=`, `ss.timeseries`, `ss.outcome`, `ss.modify`), then a function in a slot (`rate=fn`), then custom code (a subclass with `step_state()`). Say why in a comment wherever you used a function or custom code, starting the comment with `Workaround:` if it stands in for something Starsim could declare.
5. **Keep names** from the source (states, parameters, transitions) so the two can be compared, and keep the source's values: do not tidy a parameter into a "nicer" one unless the result is identical.
6. **Compare the final model with the mechanical import**: a text diff of `model.py` against `sim.imported.code` lists what you changed. Everything it lists should be one of the untranslated pieces or a deliberate fix.
7. **Validate** as `references/validate.md` says: run both, compare exactly (deterministic) or statistically (stochastic), and explain any difference.
8. **Report** what was translated mechanically, what by hand, each workaround and why, and anything that does not translate well. A gap is a finding to report, not something to hide or force.
9. **Say what the model is for.** If others will run it, its docstring says what it answers, its key assumptions, its known biases, and what it should not be used for, taken from the source where the source says so.

## Finishing an incomplete import

An import that could not translate everything is not refused: `ss.import_from()` warns `ImportIncomplete` with each missing part, the sim runs without them (and warns again when first run), and finishing it is this skill's job. Someone will often arrive with only the warning's instruction ("Finish importing <file> (<format>) into Starsim ..."). Then:

1. **Get the list.** `sim = ss.import_from('<file>', format='<format>')`; `sim.imported.incomplete` is a list of `(name, why)`, one per missing part, and `sim.imported.prompt` is the instruction itself. Save the model's code with `sc.savetext('model.py', sim.imported.code)`: the parts are listed at its end under `Not translated (ImportIncomplete)`, and how each rate was read is in the comments at its top.
2. **Find each part in the source** (the `name` says where: a transition, a state's duration, a level, a parameter, an output), and read the framework's guide in `references/` for the row that covers it.
3. **Translate it in `model.py`** with the closest form, in the order of preference of step 4 of the workflow, using `references/idioms.md`. Keep the source's names. Mark a stand-in for something Starsim could declare with a `Workaround:` comment. Delete the part's line from the `Not translated` list when it is done, so the list is always what is left.
4. **Report a gap instead of forcing it.** If the part is a feature Starsim does not have (e.g. EMULSION's offspring by prototype across several machines), leave it out, keep its line in the list, and say in the report what it changes in the results and what Starsim would need.
5. **Check that nothing else changed**: a text diff of `model.py` against `ss.import_from('<file>', format='<format>').imported.code` should show only the parts you translated.
6. **Validate** the final model against the original (`references/validate.md`), and explain any difference that is left, part by part.
7. **Report** a table of the parts: how each was translated (declared, a function, custom code, or left out as a gap), and the validation's verdict.

## Frameworks

| Framework | Mechanical import | Guide |
|---|---|---|
| EMULSION | `format='emulsion'` (YAML) | `references/emulsion.md` |
| odin | `format='odin'` (the R DSL) | `references/odin.md` |
| SBML | `format='sbml'` (XML) | `references/sbml.md` |
| SimInf | `format='siminf'` (`mparse` and the predefined models) | `references/siminf.md` |
| flepiMoP | `format='flepimop'` (the config YAML) | `references/flepimop.md` |
| summer2 | `format='summer2'` (a JSON dump of the model object) | `references/summer2.md` |
| Atomica | `format='atomica'` (the framework and databook workbooks, and any program book and calibration) | `references/atomica.md` |
| EpiModel | `format='epimodel'` (a JSON dump from `dump_epimodel.R`; built-in models and networks, custom modules listed with their R source) | `references/epimodel.md` |
| epydemix | `format='epydemix'` (a JSON dump from `starsim/converters/epydemix.py` run as a script, or a live model) | `references/epydemix.md` |
| epipack | `format='epipack'` (a JSON dump from `starsim/converters/epipack.py` run as a script) | the user guide's conversion page |
| epymorph | `format='epymorph'` (a JSON dump from `starsim/converters/epymorph.py` run as a script) | the user guide's conversion page |
| LASER, laser-measles | by hand | `references/laser.md` |
| MEmilio | `format='memilio'` (a JSON dump from `starsim/converters/memilio.py` run as a script; the ODE models `osir`, `oseir`, `oseirdb`, and `osecir`, alone, on a `MobilityGraph` (places with commuting), or in a parameter study (distributions as priors)); the other models (`osecirvvs`, IDE, LCT, SDE, agent-based) by hand | `references/memilio.md` |
| camdl | `format='camdl'` (a `.camdl` file, or camdlc's JSON) | `references/camdl.md` |
| Pandemic Simulator | by hand | `references/pandemic_simulator.md` |
| Starsim v3 | upgrade first with the migrate-starsim3 skill (removed names, changed results); then declared transitions by hand | `references/starsim3.md` |

`ss.converters.importers` lists the importers this version has. For a framework with no importer, it can still help to fill in an `ss.converters.Source` (states, parameters, flows as written in the source) and call `ss.converters.build(source)`, which classifies the rates by the same rules, prints how it read each one, and writes the model as code (`ss.converters.write(source)` gives the code alone).

## Exporting a model

An export is also two passes. `ss.export_to()` writes what it can, and lists the rest; what it left out is then yours to translate or report. Each exporter writes the compartmental form of the declared diseases, so it carries less than an import (`ss.converters.exporters` lists them; `starsim convert --to=<format>` from the shell).

| To share | How |
|---|---|
| The model, for SBML tools such as COPASI | `ss.export_to(sim, 'sbml', filename='model.xml')` |
| The model, for odin | `ss.export_to(sim, 'odin', filename='model.R')` |
| The model, for EpiModel, SimInf, summer2, epydemix, epipack, epymorph, MEmilio, or Atomica | `ss.export_to(sim, '<format>', filename=...)`: an R or Python script that builds and runs it in the framework, or for Atomica, a framework workbook |
| The model, for any other framework | By hand, from `sim.diseases.x.equations()`, reading the framework's guide in `references/` from right to left |
| Results | `sim.to_json()`, or `sim.to_df()` |
| Forecasts for a hub | `ss.converters.to_hub()` |
| The model, for Starsim users | The script itself, or `ss.save()` for a sim partway through a run |

1. **Decide whether the target can hold the model.** Exports are compartmental, so a model whose answer depends on individuals (a network, histories, who infected whom) changes meaning, even though the export succeeds. Say so before going on, rather than after.
2. **Export, and read every line of the `ss.errors.LossyExport` warning.** An export doesn't fail for what it can't carry. Lines starting `not exported:` were dropped (a network is written as homogeneous mixing; interventions, code, and functions are left out), and lines starting `changed:` were altered (a sim run as agents is written as an ODE; a stratified disease is written as the sum of its groups, with the base values of its parameters). Each line is a difference between the two models.
3. **Translate what was dropped by hand** in the target's language where it has a form for it, and keep the Starsim names so the two can be compared. Otherwise, leave it out and report it.
4. **Check the units.** The export says its unit of time in its first lines. Bare numbers in the target are in that unit.
5. **Validate.** `ss.import_from()` reads an export back, which checks the file, not the target's reading of it: run the export in the target itself, if it is available, and compare as `references/validate.md` says. Compare with the Starsim model solved the same way (`solver='ode'`), so that the difference isn't the solver.
6. **Report** what was exported, each line of the warning, what you translated by hand, and what the differences do to the question the model is for.

## Rules

- Never drop part of a model silently. Everything not translated is listed (by `ImportIncomplete`, or by you in the report). A model with parts still listed in `sim.imported.incomplete` is not finished, even though it runs.
- The same goes for exports: an export isn't finished while a line of its `LossyExport` warning is unexplained.
- Don't change the model to make it fit. If Starsim cannot say something, translate the closest thing, mark it as a workaround, and report the gap.
- Don't install a framework into someone's environment without asking; the original is run in the framework's own environment (`references/validate.md`).
- The source's expressions are parsed, never run as they are: `ss.import_from()` checks each rate before writing it as code, allowing only arithmetic, comparisons, a few functions, and the names the model declares; a rate that is anything else is listed as not translated.
