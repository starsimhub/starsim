# The model spec

What a model is, written out as data. The model spec is a JSON document that holds everything needed to rebuild a model: its states, transitions, parameters, stratifications, mixing, interventions, results, and run settings.

Justified by [best-practices/model-structure.md](../best-practices/model-structure.md) and [best-practices/principles.md](../best-practices/principles.md).

## What it is for

A model in this language is a set of objects (states, transitions, parameters), not a block of code. The model spec is those objects written to a file. It is what gets:

- **saved and loaded**, so a model can be sent to someone else or archived with a paper;
- **hashed**, so that "is this the same model?" has an answer ([11-random.md](11-random.md));
- **compared**, declaration by declaration (`ss.diff`);
- **drawn and printed** (`diagram()`, `describe()`, `equations()` all read the same structure);
- **translated** to and from other frameworks ([16-interop.md](16-interop.md)).

Nobody is expected to write a model spec by hand. Models are written in Python; the spec is produced from them.

```python
sir.to_json()                  # the model spec of one disease, as a dict
sir.to_json('sir.json')        # ... saved to a file
sim.to_json('sim.json')        # the whole sim: modules, networks, interventions, data, settings
sim2 = ss.load('sim.json')     # back to a sim
```

`to_json()` is Starsim's existing name. It works before the model is in a sim; values that are only resolved inside a sim (the unit of a bare number, a guessed population size) are then written as `null`.

A sim is one run, so its spec holds one seed and no number of runs. The runs of an `ss.MultiSim` share one spec and differ only in `rand_seed`.

## Structure

This is the spec of the model on the first page of [00-overview.md](00-overview.md): an age-stratified SIR with a fitted `beta`, a reported-cases outcome, a contact matrix, data, and one intervention.

```json
{
  "schema": "lingua-franca/1.0",
  "modules": [
    {
      "name": "sir",
      "type": "Disease",
      "dt": null,
      "init_prev": 0.01,
      "states": [
        {"name": "S", "people": true, "init": null, "infectious": false, "rel_sus": 1.0, "rel_trans": 1.0, "label": null},
        {"name": "I", "people": true, "init": null, "infectious": true, "rel_sus": 1.0, "rel_trans": 1.0, "label": null},
        {"name": "R", "people": true, "init": null, "infectious": false, "rel_sus": 1.0, "rel_trans": 1.0, "label": null}
      ],
      "pars": [
        {"name": "beta", "value": {"type": "per", "value": 0.05, "unit": "day"},
         "bounds": [0.01, 0.5], "prior": null, "source": null, "doc": null}
      ],
      "transitions": [
        {"name": "infection", "kind": "contact-mediated", "src": ["S"], "dst": ["I"],
         "rate": ["*", ["par", "beta"], ["state", "I"]],
         "dur": null, "p": null, "stages": null, "label": null},
        {"name": "recovery", "kind": "spontaneous", "src": ["I"], "dst": ["R"],
         "rate": null,
         "dur": {"dist": "expon", "mean": {"type": "dur", "value": 10.0, "unit": "day"}},
         "p": null, "stages": null, "label": null}
      ],
      "stratifications": [],
      "results": [
        {"name": "cases", "type": "outcome", "source": ["*", 0.5, ["result", "infection"]],
         "delay": {"type": "dur", "value": 5.0, "unit": "day"}, "duration": null, "every": {"type": "dur", "value": 1.0, "unit": "week"}, "likelihood": null}
      ],
      "functions": [],
      "guessed": ["states.I.infectious", "transitions.recovery.dur.dist", "pars.beta.value.unit"]
    }
  ],
  "networks": [
    {"name": "contacts", "type": "MixingPools", "layer": null, "dependence": "frequency", "n_ref": null,
     "data": {"provider": "contacts", "place": "Kenya", "source": "prem2021", "version": "2021.1"}}
  ],
  "stratify": [
    {"name": "age", "kind": "age", "levels": [0, 15, 65], "split": null, "adjust": {},
     "states": null, "where": null, "mobility": null, "immunity": null, "transitions": []}
  ],
  "modifiers": [],
  "interventions": [
    {"name": "masks", "type": "modify", "target": "sir.pars.beta", "multiply": 0.6, "add": null, "set": null,
     "where": null, "when": null, "start": "2020-03-15", "stop": null}
  ],
  "data": {"file": "weekly_cases.csv", "sha256": "5b1f0c9e", "columns": ["cases"], "aliases": {}},
  "config": {
    "solver": "auto", "integrator": null, "approx": null, "unit": "day",
    "start": "2020-01-01", "stop": "2021-01-01", "dt": {"type": "dur", "value": 1.0, "unit": "day"},
    "total_pop": 50000000.0, "n_agents": null, "rand_seed": 1
  },
  "fit": null,
  "meta": {
    "library_version": "4.0.0", "created": "2026-10-12T00:00:00Z",
    "solver_chosen": "ode", "solver_reason": "every part has a compartmental form",
    "notes": ["unit: day (from start and stop dates)", "split for age taken from the population"]
  }
}
```

Every field is present in every document; a value that was not given and has no default is `null`, not left out. A fixed shape makes two specs easy to compare and makes a reader's job smaller.

The top-level parts:

| Part | Holds |
|---|---|
| `modules` | Diseases, demographics, and custom modules, each with its states (people or not), parameters, transitions, stratifications, declared results, and functions |
| `networks` | Networks, mixing pools, and mobility, as passed to `networks=`. An empty list means homogeneous mixing with one contact per unit time |
| `stratify` | Stratifications applied to the whole sim (`sim.stratify(age=...)` or `ss.Sim(..., stratify=...)`), in the order applied, in the same form as a module's `stratifications`. They apply to every module; a module's own list holds the dimensions and adjustments that belong to it alone |
| `modifiers` | `ss.modify` objects passed to `connectors=`: interactions that are part of the model |
| `interventions` | Everything passed to `interventions=`: deliveries, `ss.change`, and modifiers that are policy. These define the scenario |
| `data` | The data attached to results by name: where it came from, a hash of its content, and its columns |
| `config` | Run settings. `solver` is what was asked for (`"auto"` unless the user chose one); the solver actually used is in `meta`. `integrator` names the ODE integration library or scheme, or is `null` for the library's choice. `approx` is `null` (approximations are made and noted), `true` (made silently), or `false` (an error). `unit` is the sim's unit, or `null` when time is counted in timesteps |
| `fit` | Calibration settings (data, weights, method, trials), or `null` |
| `meta` | Things that describe the document or the run and do not change the model: versions, the chosen solver and its reason, the build notes |

### States

Every state ([03-states-and-transitions.md](03-states-and-transitions.md#states-that-are-not-people)) carries `people`: `true` if the state holds people, `false` if it holds a plain count (bacteria in water, mosquitoes). It is always written, whether it was given with `ss.State(people=...)` or worked out from the transitions (a state that no transition connects to the module's first state does not hold people). A state inferred to hold people is the ordinary case and is not listed in `guessed`; a state inferred to be a count is, which is the line `describe()` prints for it. A state with `"people": false` is left out of `total_pop` and of the people dimensions in `stratify`, and is never represented as agents.

There is no separate kind of module for such states: a reservoir or a vector population is written in the same module as the disease, with the same transition objects.

### Transitions

There is one transition object, matching the one class `ss.Transition` ([03-states-and-transitions.md](03-states-and-transitions.md)):

| Field | Holds |
|---|---|
| `src` | A list of source state names, or `null` for an inflow (a birth, shedding into a reservoir) |
| `dst` | A list of destination state names, or `null` for an outflow (a death, decay). More than one name is a branch |
| `rate` | A rate expression (below), or `null` when the transition is driven by a dwell time |
| `dur` | A dwell-time distribution, or `null` when the transition is driven by a rate |
| `p` | Branch probabilities: one per destination, or one number for a transition that is one branch of several out of its source state |
| `stages` | The number of sub-stages asked for with `stages=`, or `null` |
| `kind` | `"contact-mediated"` if the rate contains a bare state, otherwise `"spontaneous"` |

JSON `null` in `src` or `dst` is `ss.null`, outside the model: `ss.null >> W @ rate` is written `"src": null, "dst": ["W"]`.

`kind` is derived from `rate` and is written only for the reader's convenience. A loader recomputes it and ignores what the file says.

`dur=ss.days(10)` and `rate=ss.perday(0.1)` say the same thing, so they are written the same way: a constant hazard with no states in it is stored as an exponential `dur` with mean `1 / rate`. Anything else given as a rate (an expression, a time series, a per-period probability, a function) stays in `rate`.

### Outcomes

An `ss.outcome` is a result with `"type": "outcome"`:

| Field | Holds |
|---|---|
| `source` | An expression (below) that is a weighted sum of flows: transitions and other outcomes, each referred to as `["result", name]`. A number times a flow is that fraction of its events, so `0.5 * infection` is `["*", 0.5, ["result", "infection"]]` and there is no separate probability field |
| `delay` | A duration, a distribution, or a function, or `null` |
| `duration` | A duration or a distribution, or `null`. When given, the outcome also has an `n_<name>` result |
| `every` | The reporting period the outcome is totaled by (a duration, or a function), or `null` |
| `likelihood` | A distribution, or `null` |

A result that is a ratio or a sum of states (`prevalence = I / N`) is written the same way, with its expression in `source`. Whether a result was passed as its own keyword or through `results=` or `outcomes=` makes no difference to the spec.

### Stratifications

A model stores its stratifications as a list, in the order they were applied, and does not write out the strata, the aging transfers, or the sub-stages they generate. Those follow from the rules in [05-dimensions.md](05-dimensions.md) and [03-states-and-transitions.md](03-states-and-transitions.md), and a model with 4,512 compartments per state should still be a readable file.

### Guessed values

Each module carries `guessed`: the paths of its fields whose values were guessed, not given. This is what `describe()` marks `[guessed]`. The spec always contains the resolved value, so a reader never has to re-apply the guessing rules; `guessed` only records which values those were.

## Rate expressions

Rates, branch probabilities, result ratios, and conditions are written as nested lists: an operator or a kind, followed by its arguments. A number, a quantity, or a name can stand anywhere an argument can. Nothing in the spec is a string of code to be parsed. [CK: don't particularly like this, it's extremely hard to read]

```json
["*", ["par", "beta"], ["state", "I"]]
["/", ["*", ["par", "beta"], ["n", "W"]], ["+", ["n", "W"], ["par", "K"]]]
["/", ["*", ["par", "a"], ["par", "b"], ["n", "Iv"]], ["n", "Sh", "Eh", "Ih", "Rh"]]
["*", ["par", "beta"], ["+", ["state", "I"], ["*", 0.5, ["state", "A"]]]]
```

These are `beta * I`; cholera's `beta * W.n / (W.n + K)`; malaria's `a * b * Iv.n / Nh` with `Nh = (Sh + Eh + Ih + Rh).n`; and `beta * (I + 0.5 * A)`.

| First item | Arguments | Meaning |
|---|---|---|
| `state` | a state name | A bare state: contact with people in that state, through the model's routes. Its presence is what makes a transition contact-mediated. The state MUST be one that holds people |
| `n` | one or more state names | The plain count in those states, summed (people, or whatever the states hold) |
| `prev` | one or more state names | That count as a fraction of the module's population |
| `par` | a parameter name | A named parameter (`sir.pars.beta`) |
| `attr` | a path such as `people.age` | An attribute of the people or of a module, used in conditions |
| `result` | a result name | A result: the flow of a transition, an outcome, or another declared result. Used in outcome sources and ratios |
| `fn` | a function name | A function recorded in `functions` (below) |
| `+` `-` `*` `/` | two or more expressions | Arithmetic |
| `>` `<` `>=` `<=` `==` `!=` | two expressions | Comparison |
| `&` `\|` `~` | expressions | And, or, not |
| `isin` | an expression and a list | Membership |
| `select` | an expression and an object of coordinates | Selection along a stratification, as `.select(age=0)` |
| `sum` | an expression and a stratification name | Sum over a stratification, as `.n.sum('age')` |

A state in another module is written with its module's name (`hiv.I`). Inside its own module the short name is used.

A lazy condition ([07-effects.md](07-effects.md#conditions)) such as `(ppl.age > 65) & ~sir.vaccinated` is written the same way:

```json
["&", [">", ["attr", "people.age"], 65], ["~", ["attr", "sir.vaccinated"]]]
```

## Values

Typed quantities, using Starsim's type names:

```json
{"type": "dur", "value": 10.0, "unit": "day"}
{"type": "per", "value": 0.05, "unit": "day"}
{"type": "prob", "value": 0.1, "unit": "day"}
{"type": "freq", "value": 2.0, "unit": "week"}
```

A plain JSON number is a number with no time dimension (a probability, a multiplier, a count). A value keeps the unit it was written with. A bare number in a rate or duration slot is written as a quantity in the sim's unit, and is listed in `guessed` if that unit was inferred (from `dt`, the dates, or the typed values) and not given with `unit=`.

If nothing in the model has a unit, time is counted in timesteps: `config.unit` is `null`, a bare number is written as a quantity with `"unit": null` (`{"type": "per", "value": 0.05, "unit": null}` is 0.05 per step), and `start` and `stop` are step numbers. Nothing is listed in `guessed`, because nothing was assumed.

Distributions, with Starsim's names:

```json
{"dist": "lognorm_ex", "mean": {"type": "dur", "value": 10.0, "unit": "day"}, "std": {"type": "dur", "value": 3.0, "unit": "day"}}
{"dist": "gamma", "a": 3, "scale": {"type": "dur", "value": 3.333, "unit": "day"}}
```

Values by stratum, time series, and seasonal curves:

```json
{"by": ["age"], "values": [0.08, 0.05, 0.03]}
{"timeseries": {"2020-03-15": 0.2, "2020-06-01": 0.6}, "mode": "constant"}
{"seasonal": {"mean": 0.05, "amplitude": 0.3, "peak": "January"}}
{"by": ["patch", "patch"], "file": "mobility.npz", "sha256": "9c2e41aa"}
```

Large arrays are stored as a reference to a file plus a hash of its content, so the spec stays readable. Named data (`ss.people('Kenya')`, `ss.contacts('Kenya')`, `ss.geo(...)`) is stored as the provider, place, source, and version that were resolved ([04-parameters.md](04-parameters.md#named-data)).

## Functions

A function in a slot, a function passed as custom per-step behavior, and an overridden `step()` are recorded by name and source text:

```json
{
  "name": "clearance",
  "args": ["sim", "uids"],
  "used_in": ["transitions.discharge.rate"],
  "source": "def clearance(sim, uids):\n    return 0.2 / (1 + sim.people.viral_load[uids])\n"
}
```

and referred to from the slot as `["fn", "clearance"]`. A lambda is given the name of the slot it fills.

**A spec that contains functions is not fully portable.** It can be read, compared, hashed, and drawn anywhere, but it can only be *run* by Python, and only if everything the function refers to (imports, other functions, global variables) is available where it is loaded. The source text is recorded so that a person can see what the function does and so the hash changes when it changes. `ss.load` on a spec with functions rebuilds them from the source text only if asked (`ss.load('sim.json', run_code=True)`); otherwise the functions must be supplied by importing the script that defines them. A spec with no functions is plain data and can be run by any implementation.

`sir.to_json()` prints a one-line note when the spec contains functions, naming them.

## Normalization

The same model can be written in three spellings and in any keyword order. Normalization is what makes all of them produce the same file. It is applied before saving, hashing, or comparing, and applying it twice MUST give the same result as applying it once.

1. **Resolve.** Every guessed default is written out as a value.
2. **Type.** Every bare number in a rate or duration slot becomes a quantity in the sim's unit (`null` in a sim with no unit).
3. **Expand shorthand.** Arrow strings, in any accepted spelling (`'I -> R'`, `'I->R'`, `'I >> R'`), become `src` and `dst` lists, with `ss.null` and a missing end of an arrow both written as `null`; `ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`, and the operator form all become the one transition object; a constant hazard becomes an exponential `dur`; an `init=` dict on the disease is written into the `init` field of each state; `transitions=`, `results=`, and `outcomes=` lists are merged with members passed as keywords; an alias for a solver becomes its name (`'ctmc'` becomes `'gillespie'`).
4. **Order.** Modules, states, and stratifications stay in declared order, because that order means something (the first state receives the remainder of the initial population; modules run in the order given; stratifications apply in sequence). Transitions, parameters, results, and functions are sorted by name. The arguments of `+`, `*`, `&`, and `|` are sorted. Object keys are written in the order shown in this document.
5. **Numbers.** Floating-point values are written with full round-trip precision.

After normalization, the three spellings of the same model ([01-objects.md](01-objects.md#the-equivalence-rule)) MUST produce the same document, once names agree. This is tested ([18-conformance.md](18-conformance.md)).

## Hashing

`sim.hash()` returns an `sc.objdict` with five parts ([11-random.md](11-random.md#run-identity)): four hashes and the seed. Each hash is SHA-256, shortened for display, of one slice of the normalized spec:

| Part | Computed from |
|---|---|
| `model` | `modules`, `networks`, `stratify`, and `modifiers`, with every number replaced by a placeholder: the structure |
| `pars` | Every number in those parts, in document order, converted to the sim's unit, plus the provider, source, and version of any named data: the values |
| `config` | `config`, apart from `rand_seed` |
| `scenario` | `interventions`. Shown as the scenario's name where it has one (`ss.parallel(vaccine=...)`), as a hash otherwise |
| `seed` | `config.rand_seed`, as it is; not a hash |

```text
>>> sim.hash()
objdict(model='8f3a21', pars='c47e90', config='1b6d55', scenario='vaccine', seed=1)
```

Each part is a plain string or number, so `sim.hash().pars` can go in a file name or a column of a data frame. The hash is a by-product of the model being data. It tells apart the many runs that come from one commit (a sweep, a calibration) and is the test for whether two descriptions are the same model; the commit or tag remains the main record of what code was run.

- Changing a rate from 0.05 to 0.08 changes `pars` and not `model`. Adding a state or a transition changes `model`.
- Because values are converted to one unit before hashing, `ss.days(7)` and `ss.weeks(1)` hash the same. Calendar durations (`ss.datedur`) are hashed as written.
- `meta`, `guessed`, `label`, `doc`, and a parameter's `source` do not contribute to any hash. Relabeling a state does not change a model, and neither does writing down a value that would have been guessed.
- `fit` and `data` are not part of `sim.hash()`. Fitting settings have their own hash, on the fit, so changing the number of trials does not look like a change to the model ([13-inference.md](13-inference.md)).
- The runs of an `ss.MultiSim` differ only in `seed`.
- A function contributes its source text to `model`. This is a best effort: a function that depends on something outside its own source can change behavior without changing the hash.
- Two models that are the same mathematically but declared differently (two transitions out of a state versus one branching transition, say) may hash differently. A spurious difference is the safe direction to fail in.

## Versioning

`schema` is `lingua-franca/MAJOR.MINOR`.

- A minor version may add fields. A reader SHOULD ignore fields it does not know within its own major version.
- A major version may remove or change fields, and SHOULD come with a converter from the previous one.
- Loading a document whose major version is newer than the reader raises `E1401 SchemaTooNew`, naming the version needed.

## Round trip

```python
sim.to_json('sim.json')
sim2 = ss.load('sim.json')
```

`ss.load` of a saved spec MUST give a model whose own spec is identical to the one saved, and therefore the same hash. For a spec without functions this holds on any machine. For one with functions it holds when the functions are available (above).

## Export to other formats

The model spec is the only complete format. Exports to other frameworks' formats ([16-interop.md](16-interop.md)) usually cannot carry everything, and each MUST print what was left out or changed, one line per declaration (`W1401 LossyExport`):

```text
>>> ss.save('model.xml', sim, format='sbml')
exported to SBML: 3 states, 2 transitions, 1 stratification (age, written out as 9 species)
not exported: network 'mf' (SBML has no networks)
not exported: function 'clearance' (Python code)
changed: lognormal dwell time on 'recovery' written as 11 Erlang stages
```

## What makes a model checkable

Because the model is data with a fixed shape:

1. A model written by a person or generated by an AI can be checked before it is run, against the rules in this folder.
2. Two descriptions of the same model produce the same document, so "did we build the same thing?" can be tested.
3. A tool that produces or reads models has a schema to work to, with a version.

This depends on the documentation being shipped with the library version and tested against it, which is a release requirement recorded in [18-conformance.md](18-conformance.md).

## Open questions

- Whether `ss.prob` keeps its period in the spec (as above) or is converted to a per-step value at build time. Keeping it is more faithful; see [02-types.md](02-types.md).
- Whether a people property stratified on the only disease in a sim (`sir.stratify(age=...)`) and the same stratification on the sim (`sim.stratify(age=...)`) should normalize to one document. They are the same model; as written they are recorded where they were applied and hash differently.
- Whether an inferred `"people": true` belongs in `guessed`. As written only an inferred count is listed, because that is the guess worth reading.
- Whether the spec should optionally write out the expanded strata and sub-stages for readers that cannot apply the stratification rules themselves. Exporters need the expanded form; it is not clear that the saved file does.
