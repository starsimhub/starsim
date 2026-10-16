---
name: import-model
description: Translate a disease model from another framework (EMULSION, odin, SimInf, flepiMoP, summer2, Atomica, SBML, EpiModel, epydemix, LASER, MEmilio, camdl, or Starsim v3) into Starsim v4. Use when someone wants to import, port, convert, or reimplement an existing model in Starsim, including the parts that do not map one to one. Pair with the validate-import skill to check the result against the original.
---

# Import a model into Starsim

A model is imported in two passes. The mechanical pass (`ss.load(..., format=...)`) translates everything with an obvious one-to-one mapping and lists what it could not translate. The second pass is yours: translate each listed item by hand into the closest Starsim form, and write the frameworks with no importer by hand from the start. Then check the result with the validate-import skill.

## Workflow

1. **Find the model and its parts.** Locate the source files (YAML, R, XML, Python, workbooks, data files) and list what the model has: states, flows, parameters and their units, structure (age, place, risk), contacts (mixing, networks), demography, interventions and scenarios, outputs, run settings (time unit, step, duration, population, stochastic or not). Read the framework's guide in `references/` before going further.
2. **Run the mechanical import**, if the framework has one (see the table below):
   ```python
   import starsim as ss
   sim = ss.load('model.yaml')                     # the format is told from the file where it can be
   sim = ss.load('model.R', format='odin')         # or given
   sim.to_code('model.py')                         # the model as a Python file, to edit
   ```
   `ss.load()` prints how each rate was read (which rule of `16-interop.md` applied) and warns `W1601 ImportIncomplete` with each piece it could not translate. Both are kept in `sim.imported` (`choices`, `incomplete`) and written into `model.py` as comments: the reading in the docstring, the untranslated pieces at the end.
3. **Check the guesses** in the printed reading and in `sim.describe()` (marked `[guessed]`):
   - A rate with another state divided by the total (`beta*S*I/N`) was read as exposure (`beta * I`, frequency-dependent). One without the total (`beta*S*I`) was kept as counts (`beta * I.n`, density-dependent). Make sure that is what the source meant, especially if the population changes.
   - Units: bare numbers are in the sim's unit (`unit=`). A per-step probability is not a rate: `-log(1 - p)/dt` is the rate that gives it. A probability per time unit is `ss.probperday(p)` and the like.
   - Initial numbers: a value above 1 is a count, below 1 a fraction, `1` (an int) one person, `1.0` everyone.
   - The solver: match the source's kind of run (`'ode'` for deterministic, `'binomial'` for discrete-time stochastic compartments, `'gillespie'` for exact event-driven ones, `'abm'` for individuals).
4. **Translate each untranslated piece** with the closest Starsim form, using `references/idioms.md` (minimum stays, conditions, calendars, capacities, actions on crossing, births and offspring, variables of individuals, scheduled events, interventions, networks, outputs) and the framework's guide. In order of preference: a declared form (a transition, `where=`, `ss.timeseries`, `ss.outcome`, `ss.modify`), then a function in a slot (`rate=fn`), then custom code (a subclass with `step_state()`). Say why in a comment wherever you used a function or custom code, starting the comment with `Workaround:` if it stands in for something Starsim could declare.
5. **Keep names** from the source (states, parameters, transitions) so the two can be compared, and keep the source's values: do not tidy a parameter into a "nicer" one unless the result is identical.
6. **Compare the final model with the mechanical import**: `ss.diff(ss.load('model.yaml'), final_sim)` lists what you changed, in the model's own terms. Everything it lists should be one of the untranslated pieces or a deliberate fix.
7. **Validate** with the validate-import skill: run both, compare exactly (deterministic) or statistically (stochastic), and explain any difference.
8. **Report** what was translated mechanically, what by hand, each workaround and why, and anything that does not translate well. A gap is a finding to report, not something to hide or force.

## Frameworks

| Framework | Mechanical import | Guide |
|---|---|---|
| EMULSION | `format='emulsion'` (YAML) | `references/emulsion.md` |
| odin | `format='odin'` (the R DSL) | `references/odin.md` |
| SBML | `format='sbml'` (XML) | `references/sbml.md` |
| SimInf | `format='siminf'` (`mparse` and the predefined models) | `references/siminf.md` |
| flepiMoP | `format='flepimop'` (the config YAML) | `references/flepimop.md` |
| summer2 | `format='summer2'` (a JSON dump of the model object) | `references/summer2.md` |
| Atomica | `format='atomica'` (the framework and databook workbooks) | `references/atomica.md` |
| EpiModel | `format='epimodel'` (a JSON dump from `dump_epimodel.R`; built-in models and networks, custom modules listed with their R source) | `references/epimodel.md` |
| epydemix | `format='epydemix'` (a JSON dump from `dump_epydemix.py`, or a live model) | `references/epydemix.md` |
| LASER, laser-measles | by hand | `references/laser.md` |
| MEmilio | by hand | `references/memilio.md` |
| camdl | `format='camdl'` (a `.camdl` file, camdlc's JSON, or a pycamdl model) | `references/camdl.md` |
| Starsim v3 | runs unchanged; declared transitions by hand | `references/starsim3.md` |

`ss.interop.formats` lists the importers this version has. For a framework with no importer, it can still help to fill in an `ss.interop.Source` (states, parameters, flows as written in the source) and call `ss.interop.build(source)`, which classifies the rates by the same rules and prints how it read each one.

## Rules

- Never drop part of a model silently. Everything not translated is listed (by `W1601`, or by you in the report).
- Don't change the model to make it fit. If Starsim cannot say something, translate the closest thing, mark it as a workaround, and report the gap.
- Don't install a framework into someone's environment without asking; running the original is the validate-import skill's job, in the framework's own environment.
- The source's expressions are parsed, never run: `ss.load()` builds rates with the model spec's parser, which allows only arithmetic, comparisons, a few functions, and the names the model declares.
