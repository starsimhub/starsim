# Working with other frameworks

Importing models from, and exporting them to, the other frameworks reviewed in [approaches](../approaches). This is different from changing the run method ([12-backends.md](12-backends.md)), which runs one model in different ways; this moves a model between languages.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md) and the reviews in [approaches](../approaches).

## The priority

People have to want to write models in this language for any of this to matter. An interchange format that nobody writes in ends up as an export target that every tool supports partially. So the order is: first, a model should never be easier to write without the lingua franca than with it; then import and export.

## Everything goes through the model spec

An importer reads another framework's model and produces a model spec ([14-model-spec.md](14-model-spec.md)). An exporter reads a model spec and writes another framework's format. There are no direct framework-to-framework converters, so each framework needs at most two adapters.

```
EpiModel ─┐                                      ┌─▶ SBML
summer2  ─┤                                      ├─▶ Petri net
epipack  ─┼─▶ importer ─▶ model spec ─▶ exporter ┼─▶ odin
epydemix ─┤                                      ├─▶ a diagram (Mermaid, Graphviz)
SBML     ─┘                                      └─▶ a data-entry form
```

## Import

```python
sir = ss.load('model.xml',  format='sbml')
sir = ss.load('model.yaml', format='emulsion')
sir = ss.load('config.yml', format='flepimop')
sir = ss.load('framework.xlsx', format='atomica')
sir = ss.load(summer_model)                 # a Python object from another framework; format taken from its type
```

The result is an ordinary model: `sir.describe()` prints it, and it can be stratified, run, and saved like one written by hand.

An importer SHOULD:

1. **Keep names.** A transition called `infection` in the source is called `infection` here.
2. **Mark what it guessed.** Values the importer filled in are marked `[guessed]` in `describe()`, exactly as for a hand-written model.
3. **Keep going.** Something that does not map onto a built-in piece is kept as a literal expression or a function (below), not dropped and not refused.
4. **List what it could not translate at all**, by name (`W1601 ImportIncomplete`).

### Classifying rates on import

Most frameworks write a transition as a general rate expression: `beta * S * I / N`, `gamma * I`, `mu * N`. There is one transition class here, so the importer's job is to write each one as `src >> dst @ rate` with the right kind of rate. The rules, in order:

1. **Strip the source state.** A flow written as a total (`gamma * I` out of `I`) becomes a per-person rate (`gamma`).
2. **A rate that mentions another state, divided by a population total, is exposure to that state.** `beta * S * I / N` out of `S` becomes `S >> I @ beta * I`. The bare `I` means exposure through the model's routes, which with no network is `I / N` ([06-routes.md](06-routes.md)).
3. **A rate that mentions another state with no division by a total** (`beta * S * I`) is density-dependent mass action. It is imported the same way, with the mixing set to `dependence='density'` and `n_ref` chosen so the numbers match; the choice is printed.
4. **A rate that mentions only parameters** is a spontaneous transition: `I >> R @ gamma`. With no source it is a birth, and with no destination a death.
5. **Anything else is kept exactly as written, as a literal expression of counts**: `S >> I @ beta * W.n / (W.n + K)`. Counts (`.n`) are used literally, so any arithmetic the source framework could write can be carried over. This is correct by construction, and it runs under every method.
6. **If the source is code that cannot be turned into an expression** (an arbitrary R or Python function), it is kept as a function in the rate slot and the import says so. The model then runs wherever that function runs ([03-states-and-transitions.md](03-states-and-transitions.md#functions-in-any-slot)).

Rules 2 and 3 are guesses about what the source meant, and the difference matters when the population changes. The importer prints which it chose for each transition, and `describe()` shows the result. If an expression matches neither pattern cleanly, rule 5 applies: the literal form loses nothing, it only means the transition will not pick up a network or contact matrix if one is added later.

### What to expect from each source

| Source | What it gives | How well it imports |
|---|---|---|
| **Starsim v3** | States declared, transitions in `step()` code, interventions, calibration components | Runs unchanged, since existing Starsim modules are valid modules. Turning `step()` logic into declared transitions is assisted, not automatic |
| **epipack** | Reaction tuples | Nearly complete: 3-tuples are spontaneous transitions, 5-tuples are contact-mediated |
| **summer2** | Named flow kinds, stratifications, requested outputs | Good: each flow kind is one transition with the matching rate, `stratify_with` becomes `stratify`, adjustments become `adjust=` |
| **SimInf** | `mparse` transition strings, a continuous state `v`, events | Good for the compartments; `v` becomes an `ss.Compartments` module; scheduled events become `ss.Mobility(data=...)` or interventions, which is the harder part |
| **odin** | `deriv`/`update`/`initial` equations, `data()` with a likelihood | Fair: scalar models classify by the rules above; array-indexed models need their indices turned into stratifications or are kept as literal expressions |
| **EMULSION** | A YAML state machine | Good, and useful: its shipped models already exercise switching between compartmental and individual-based runs |
| **flepiMoP** | Compartment products, `proportional_to` rates, modifiers, an outcomes chain | Fair for structure, good for outcomes, which map directly to `ss.outcome` |
| **EpiModel** | A derivative function (`dcm`) or network modules (`netsim`) | `dcm` models are code, so they import as functions; `netsim` is better, since `inf.prob` and `act.rate` are `beta` and the network's acts |
| **Atomica** | A transition matrix, characteristics, programs, a databook | Good: the framework workbook is close to `sir.transitions` and the databook to `sim.needs(to=...)` |
| **SBML, Petri-net formats** | Species and reactions | Fair: reactions are classified by the rules above; anything else is a literal expression |
| **EMOD, EpiHiper** | JSON campaigns and rule files | Poor for disease structure (it is in compiled code); good for interventions, whose targeting maps onto `eligible=` conditions |
| **epydemix, MetaCast, epymorph** | Python code | Fair; their contact and mobility datasets are more valuable as data providers for `ss.contacts` and `ss.Mobility` |

The two worth doing first are **Starsim v3**, because existing users need it to work, and **EMULSION**, because its example models are a ready-made test of running one model several ways.

## Export

```python
sim.to_json('sim.json')                              # the model spec: complete, and loads back
ss.save('model.xml', sim, format='sbml')
ss.save('model.json', sim, format='petrinet')
ss.save('model.R', sim, format='odin')
ss.save('model.md', sim, format='mermaid')
```

Only the model spec is complete. Every other format carries less, and each export MUST print what was left out or changed, one line per declaration (`W1401 LossyExport`):

```text
>>> ss.save('model.xml', sim, format='sbml')
exported to SBML: 3 states, 2 transitions, 1 stratification (age, written out as 9 species)
not exported: network 'mf' (SBML has no networks)
not exported: function 'clearance' (Python code)
changed: lognormal dwell time on 'recovery' written as 11 Erlang stages
```

An export does not fail because something cannot be carried; it writes what it can and lists the rest. A model with functions in it exports its declared parts and lists the functions as not exported ([14-model-spec.md](14-model-spec.md#functions)).

The diagram is produced by `sir.diagram()`. A generated data-entry form (Atomica's databook, `sim.needs(to='form.csv')` in [04-parameters.md](04-parameters.md#requirements)) is deferred from v1.

## Comparing two models

Since both are data, comparing them is mechanical:

```python
ss.diff(sim_a, sim_b)
```

```text
model:    transition 'hospitalization' added; 'recovery' dwell time exponential(10 days) -> gamma(shape 3, mean 10 days)
pars:     beta 0.05 -> 0.08
config:   same
scenario: baseline -> vaccine
```

The report is in the user's terms (states, transitions, parameters), not a diff of JSON text. The same function compares an imported model with its hand-written translation.

## The vignette test

Whether this language can express what the other frameworks do is something to test, not assert. [vignettes](../vignettes) translates the [examples](../examples) corpus into the lingua franca, and each translation is checked four ways:

1. **It runs.**
2. **It matches.** Its results agree with the original framework's to a stated tolerance ([18-conformance.md](18-conformance.md)).
3. **It is no longer.** The lingua franca version has no more lines of model code than the original.
4. **It is no slower.** It takes no more wall-clock time to run than the original.

Points 3 and 4 are the north star: a model should never be easier to write without the lingua franca than with it, and each main framework's own examples should be implementable in no more lines and no more compute time. How lines and time are measured is in [18-conformance.md](18-conformance.md#13-north-star).

Where a translation fails any of the four, the vignette says which and why. A list of what does not translate well, with reasons, is a useful result: it says where the design needs work.
