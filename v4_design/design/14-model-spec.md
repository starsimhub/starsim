# The model spec

What a model is, written out as data. The model spec is a JSON-compatible document that holds the whole model: its states, transitions, parameters, stratifications, mixing, interventions, results, and run settings. It is the normalized form of the model, for hashing and comparing; it is not read back into a sim.

Three things are kept apart, each with its own method:

| What | Method | Holds | Read back |
|---|---|---|---|
| The model spec | `sim.to_spec()`, `sir.to_spec()` | the model as plain data: no results and no state of a run | no; it is hashed (`sim.hash()`), compared (`ss.diff()`), and written as code (`sim.to_code()`) |
| The results as JSON | `sim.to_json()` | the sim's parameters (a module among them as a reference, its type and name), its summary, and its results, for reading elsewhere | no |
| A saved sim (a checkpoint) | `sim.save()`, `ss.save('run.sim', sim)` | the sim itself, as a pickle, with its results; saved partway through a run, with its full state, so that it can continue | yes, with `ss.load()`, by the same code |

Justified by [best-practices/model-structure.md](../best-practices/model-structure.md) and [best-practices/principles.md](../best-practices/principles.md).

## What it is for

A model in this language is a set of objects (states, transitions, parameters), not a block of code. The model spec is those objects written out as plain data. It is what gets:

- **hashed**, so that "is this the same model?" has an answer ([11-random.md](11-random.md));
- **compared**, declaration by declaration (`ss.diff`);
- **written as code**: `sim.to_code()` writes the Python that builds the model again from its spec ([saved forms](#saved-forms)).

Nobody is expected to write a model spec by hand, and nothing reads one back: models are written in Python, the spec is produced from them, and a model is saved and shared as code. There is no non-Python consumer of the spec.

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=10),
    init_prev = 0.01,
)
sim = ss.Sim(sir, stop=100)

sir.to_spec()                  # the model spec of one disease, as a dict
sir.to_spec('sir.json')        # ... saved to a file
sim.to_spec()                  # the whole sim: modules, networks, stratifications, interventions, settings
ss.save('model.py', sim)       # the model as code, which builds it again (sim.to_code())
```

`sir.to_spec()` works before the disease is in a sim; values that are only resolved inside a sim (the unit of a bare number, a guessed population size) are then written as `null`. `sim.to_spec()` initializes the sim first if it is not initialized (`sim.hash()` instead works on a copy, and leaves the sim as it was). `ss.load()` of a saved spec raises an error that says to save the model as code instead.

A sim is one run, so its spec holds one seed and no number of runs. The runs of an `ss.MultiSim` share one spec and differ only in `rand_seed`.

## Structure

This is the spec of the model on the first page of [00-overview.md](00-overview.md) (an age-stratified SIR with a fitted `beta`, a reported-cases outcome, and a contact matrix), with the masks modifier as its intervention, as `sim.to_spec()` writes it:

```json
{
  "schema": "lingua-franca/1.0",
  "modules": [
    {
      "name": "sir", "type": "Disease", "dt": null, "init_prev": 0.01, "first": "S",
      "states": [
        {"name": "S", "people": true, "init": null, "infectious": false, "rel_sus": 1.0, "rel_trans": 1.0, "label": null},
        {"name": "I", "people": true, "init": null, "infectious": true, "rel_sus": 1.0, "rel_trans": 1.0, "label": null},
        {"name": "R", "people": true, "init": null, "infectious": false, "rel_sus": 1.0, "rel_trans": 1.0, "label": null}
      ],
      "pars": [
        {"name": "beta", "value": {"type": "per", "value": 0.05, "unit": "day"}, "bounds": [0.01, 0.5], "prior": null, "source": null, "doc": null},
        {"name": "dur", "value": {"type": "dur", "value": 10.0, "unit": "day"}, "bounds": null, "prior": null, "source": null, "doc": null}
      ],
      "transitions": [
        {"name": "infection", "kind": "contact-mediated", "src": ["S"], "dst": ["I"], "rate": "beta * I",
         "dur": null, "p": null, "stages": null, "label": null, "migration": null, "after": null, "before": null, "acquisition": null, "move": null, "increment": null},
        {"name": "recovery", "kind": "spontaneous", "src": ["I"], "dst": ["R"], "rate": null,
         "dur": {"dist": "expon", "mean": ["par", "dur"]}, "p": null, "stages": null, "label": null, "migration": null, "after": null, "before": null, "acquisition": null, "move": null, "increment": null}
      ],
      "stratifications": [],
      "results": [
        {"name": "cases", "type": "outcome", "source": "0.5 * infection",
         "delay": {"type": "dur", "value": 5.0, "unit": "day"}, "duration": null, "likelihood": null}
      ],
      "functions": [],
      "transfers": null,
      "updates": null,
      "guessed": ["states.I.infectious", "transitions.recovery.dwell"]
    }
  ],
  "networks": [
    {"type": "MixingPools", "name": "contacts", "label": "contacts", "factory": "starsim.networks.MixingPools", "version": null, "pars": {"diseases": null, "beta": 1.0},
     "data": {"provider": "starsim_data", "version": "0.4.0", "place": "Kenya", "source": "prem2021", "layer": "all",
              "location": "Kenya", "study": "Prem et al. 2021", "license": "GPL-3.0"}}
  ],
  "stratify": [
    {"geo": null, "split_from": "people", "unit": null, "name": "age", "kind": "age", "levels": [0.0, 15.0, 65.0],
     "split": [0.39284168924481755, 0.5792660879552572, 0.02789222279992524],
     "adjust": {}, "states": null, "where": null, "mobility": null}
  ],
  "modifiers": [],
  "interventions": [
    {"type": "modify", "name": "masks", "label": "masks", "factory": "starsim.connectors.modify", "version": null, "target": "sir.pars.beta", "multiply": 0.6, "add": null, "set": null,
     "start": "2020-03-15", "stop": null}
  ],
  "config": {
    "solver": "auto", "integrator": null, "approx": null, "unit": "day", "start": "2020-01-01", "stop": "2021-01-01",
    "dt": 1.0, "total_pop": 50000000.0, "n_agents": null, "rand_seed": 1, "crn": true, "precision": 32,
    "substeps": null, "use_aging": null
  },
  "meta": {
    "library_version": "4.0.0", "solver_chosen": "ode", "solver_reason": "every part of the model has a compartmental form",
    "notes": [
      "the ages of the people are those of Kenya, since the contacts \"contacts\" are; give people= for others",
      "people do not age between the age groups, since nobody is born or dies; pass use_aging=True to age them",
      "contact matrix \"contacts\" aggregated from 16 age groups to the model's 3 (0-14, 15-64, 65+)",
      "unit='day' (from the typed values in the model)"
    ]
  }
}
```

The fields of each kind of object are always written, in the same order; a value that was not given and has no default (or a feature that is not used) is `null`, not left out. A fixed shape makes two specs easy to compare. The hash leaves out every field whose value is `null` ([hashing](#hashing)), so a field added for a new feature, written as `null` where it is not used, changes no other model's hash, and the feature needs no code of its own to keep them as they were. `updates` on a module is each with its `name`, the `state` it sets, and its `value` as an expression.

The top-level parts:

| Part | Holds |
|---|---|
| `modules` | Declared diseases, each with its states (people or not), parameters, transitions, stratifications, declared results, and functions. A module written as code is written as its type, name, label, `factory` (the class that makes it, as `module.Class`), `version` (the class's `version` attribute, if it has one), and parameters: it is seen through these, not through its code ([hashing](#hashing)) |
| `networks` | Networks, mixing pools, and mobility, as passed to `networks=`: each as its `type`, `name`, `label`, and `pars` (typed values and distributions written as elsewhere in the spec), and, for named data such as `ss.contacts('Kenya')`, a `data` reference saying where it came from |
| `stratify` | Stratifications applied to the whole sim (`sim.stratify(age=...)` or `ss.Sim(..., stratify=...)`), in the order applied, in the same form as a module's `stratifications`, with the split resolved (`split_from` says where it came from). They apply to every module; a module's own list holds the dimensions and adjustments that belong to it alone |
| `modifiers` | `ss.modify` objects passed to `connectors=`: interactions that are part of the model |
| `interventions` | Everything passed to `interventions=`: deliveries, `ss.change`, and modifiers that are policy. These define the scenario. They are in the order they run: each delivery after those whose events trigger it. A delivery's trigger (`on`) and condition (`where`) are written as the text of the expression (`"screening.positive"`, `"(cytology.since >= years(5))"`), and a cascade as the names of its steps. The properties of people that memory adds are derived, not stored, so the hash depends only on what was written |
| `config` | Run settings. `solver` is what was asked for (`"auto"` unless the user chose one); the solver actually used is in `meta`. `integrator` is the ODE scheme (`"rk4"` or `"euler"`), or `null` for the default. `approx` is `null` (approximations are made and noted), `true` (made silently), or `false` (an error). `unit` is the sim's unit, or `null` when time is counted in timesteps; `dt` is a number in that unit |
| `meta` | Things that describe the document or the run and do not change the model: the library version, the chosen solver and its reason, and the notes made while building it |

The data attached to the sim is not in the spec; `sim.hash()` hashes it separately ([hashing](#hashing)). Fitting settings are not in it either: they are the arguments of `sim.calibrate()`, kept on the fit.

### States

Every state ([03-states-and-transitions.md](03-states-and-transitions.md#states-that-are-not-people)) carries `people`: `true` if the state holds people, `false` if it holds a plain count (bacteria in water, mosquitoes). It is always written, whether it was given with `ss.State(people=...)` or worked out from the transitions (a state that no transition connects to the module's first state does not hold people). A state inferred to hold people is the ordinary case and is not listed in `guessed`; a state inferred to be a count is, which is the line `describe()` prints for it. A state with `"people": false` is left out of `total_pop` and of the people dimensions in `stratify`, and is never represented as agents.

There is no separate kind of module for such states: a reservoir or a vector population is written in the same module as the disease, with the same transition objects.

### Transitions

There is one transition object, matching the one class `ss.Transition` ([03-states-and-transitions.md](03-states-and-transitions.md)):

| Field | Holds |
|---|---|
| `src` | A list of source state names, or `null` for an inflow (a birth, shedding into a reservoir) |
| `dst` | A list of destination state names, or `null` for an outflow (a death, decay). More than one name is a branch, and each has a flow of its own, `<transition>.<state>`, which an outcome's source and a result expression name in that form (`0.8 * cin.X`) and which a data series may be named |
| `rate` | A rate expression (below), or `null` when the transition is driven by a dwell time or a constant rate |
| `dur` | The dwell time: a distribution, or `{"dist": "expon", "mean": ["par", name]}` for a duration and `{"dist": "expon", "rate": ["par", name]}` for a constant rate, referring to the parameter that holds the value; `null` when the transition is driven by a rate expression |
| `p` | Branch probabilities: one per destination, or one number for a transition that is one branch of several out of its source state |
| `stages` | The number of sub-stages asked for with `stages=`, or `null` |
| `kind` | `"contact-mediated"` if the rate contains a bare state, otherwise `"spontaneous"` |
| `acquisition` | `true` or `false` if given with `acquisition=` ([03-states-and-transitions.md](03-states-and-transitions.md#what-counts-as-acquisition)); `null` when it is worked out |

JSON `null` in `src` or `dst` is `ss.null`, outside the model: `ss.null >> W @ rate` is written `"src": null, "dst": ["W"]`.

`kind` is derived from `rate` and is written for the reader's convenience.

A value written inline in a transition is a parameter of the module, named after its slot (`dur`, `beta`; `rate_decay` for the rate of `decay` when the name is taken), so it appears in `pars` and is referred to by name. A constant rate (a number, a typed rate, a function, a time series) is written as an exponential `dur` given by its `rate`, and a duration as one given by its `mean`. `dur=ss.days(10)` and `rate=ss.perday(0.1)` mean the same model but are written differently, so they hash differently. Anything that mentions states or other parameters stays in `rate` as an expression.

### Outcomes

An `ss.outcome` is a result with `"type": "outcome"`:

| Field | Holds |
|---|---|
| `source` | An expression (below) that is a weighted sum of flows: transitions and other outcomes, each referred to as `["result", name]`. A number times a flow is that fraction of its events, so `0.5 * infection` is `["*", 0.5, ["result", "infection"]]` and there is no separate probability field |
| `delay` | A duration, a distribution, or a function, or `null` |
| `duration` | A duration or a distribution, or `null`. When given, the outcome also has an `n_<name>` result |
| `likelihood` | The likelihood's distribution and its parameters (`{"type": "nbinom", "pars": {"n": {"par": "k"}, "p": 0.5}}`, where `{"par": "k"}` is a fitted parameter), or `null` |

A result that is a ratio or a sum of states (`prev = I / (S + I + R)`) is written as `{"name": "prev", "type": "result", "source": "I / (S + I + R)"}`. Whether a result was passed as its own keyword or through `results=` or `outcomes=` makes no difference to the spec.

### Stratifications

A model stores its stratifications as a list, in the order they were applied, and does not write out the strata, the aging transfers, or the sub-stages they generate. Those follow from the rules in [05-dimensions.md](05-dimensions.md) and [03-states-and-transitions.md](03-states-and-transitions.md), and a model with 4,512 compartments per state should still be a readable file. A clock is written with `"kind": "clock"`, its levels as given (not the steps a compartmental run keeps it in), their `"unit"`, and no split: `{"unit": "month", "name": "since_birth", "kind": "clock", "levels": [0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0], "split": null, ...}`.

A transition that moves people along a dimension carries (and otherwise has `null`) `"move"` (the level, arrow, or clock time as given, `{"since_birth": 0.0}`) and `"increment"` (the amount for each counter, `{"parity": 2}`); a transfer along a counter carries `"increment"` as a number, and a delivery `"increment"` as a dict. An entry of `p=` with a `to` axis is written as given (`"p": ["1 - p_use", "p_use * mix * w"]`), its tables among the parameters.

### Guessed values

Each module carries `guessed`: the paths of its fields whose values were guessed, not given. This is what `describe()` marks `[guessed]`. The spec always contains the resolved value, so a reader never has to re-apply the guessing rules; `guessed` only records which values those were.

## Rate expressions

Rates, branch probabilities, result sources, and conditions are written as strings, in the Python syntax they were written in, with the names the module declares:

| Written | In the spec |
|---|---|
| `S >> I @ beta * I` | `"beta * I"` |
| cholera's `beta * W.n / (W.n + K)` | `"beta * W.n / (W.n + K)"` |
| malaria's `a * b * Iv.n / (Sh \| Eh \| Ih \| Rh).n` | `"a * b * Iv.n / (Sh + Eh + Ih + Rh).n"` |
| `beta * (I + 0.5 * A)` | `"beta * (I + 0.5 * A)"` |
| `np.clip(0.5 * I.frac, 0, 0.02)` | `"minimum(maximum(0.5 * I.frac, 0), 0.02)"` |
| `(Q.n < 20) * 0.1` | `"(Q.n < 20) * 0.1"` |
| `ss.outcome(0.5 * infection)` | `"0.5 * infection"` |
| `where=(ss.ppl.age > 65) & ~sir.I` | `"((people.age > 65) & ~sir.I)"` |
| `rate=ss.peryear(1)*expit(b0 + b_urban*ss.ppl.urban)` | `"peryear(1) * expit(b0 + b_urban * people.urban)"` |
| `ss.by(sex=dict(female=0.02, male=0.05))` | `{"by": ["sex"], "values": {"keyed": {"female": 0.02, "male": 0.05}}}` |
| `increment=dict(parity=2), move=dict(since_birth=0)` | `"increment": {"parity": 2}, "move": {"since_birth": 0}` |

What can appear: the names of the module's states, parameters, and transitions (a transition in a result's source is its flow); a bare state, which means contact with people in that state (its presence is what makes a transition contact-mediated); `.n` (the count in a state or a set of states, summed), `.frac` (that count as a fraction of the module's population), `.total` (the count over every place), and `.dwell` (the time each person has spent in a state, in `p=` and rates of the transitions out of it: `"1 - exp(-k * P.dwell)"`); `+`, `-`, `*`, `/`, `**`; comparisons; `&`, `|`, and `~` in conditions; attributes of the people (`people.age`), in conditions and in rates (`ss.ppl.age` in code); and the functions `exp`, `log`, `sqrt`, `absolute`, `sin`, `cos`, `expit` (SciPy's, imported by `to_code()` from `scipy.special`), `minimum`, and `maximum`. A set of states is written as their sum, and `np.clip` as `minimum` and `maximum`. Numbers are written at full precision, with the parentheses the expression needs, so the string means exactly what was written. A state in another module is written with its module's name (`hiv.I`); inside its own module the short name is used.

The spec is not parsed back. The importers read rates from other frameworks' formulas ([16-interop.md](16-interop.md)) with a parser of the same arithmetic (`ss.spec.Parser`: numbers, the declared names, `.n`, `.frac`, `.total`, `.dwell`, arithmetic, comparisons, typed quantities such as `perday(0.5)`, and the functions above), which allows nothing else, so that reading a file runs no code from it.

## Values

Typed quantities, using Starsim's type names:

```json
{"type": "dur", "value": 10.0, "unit": "day"}
{"type": "per", "value": 0.05, "unit": "day"}
{"type": "prob", "value": 0.1, "unit": "day"}
{"type": "freq", "value": 2.0, "unit": "week"}
{"type": "peract", "value": 0.01}
{"type": "peract", "m2f": 0.3, "f2m": 0.15}
```

`peract` is a probability per act (`ss.peract`), with one value or one for each direction of the network; it has no unit. A bare number used as `beta` on a network with acts is written as `{"type": "peract", "value": 0.01}`, with `transitions.<name>.beta` in `guessed`, so the spec is read the same without knowing the network. A source weight (`source=dict(...)`) is written in the rate text, as the operator form is (`"rate": "beta * (P + 0.55*EL)"`). A modifier's `efficacy`, `share`, and `route` are written with it. The acts, sides, and two-mode groups of networks are not in the spec (networks other than contact matrices are not; `to_code()` lists them under `LossyExport`); a contact matrix over several dimensions is (`"contacts": {"matrix": [...], "dims": ["sex", "risk"]}`).

A plain JSON number is a number with no time dimension (a probability, a multiplier, a count). A value keeps the unit it was written with. A bare number in a rate or duration slot is written as a quantity in the sim's unit. Where that unit was inferred (from the typed values or the dates) rather than given with `unit=`, the note that says so is in `meta.notes` (`unit='day' (from the typed values in the model)`).

If nothing in the model has a unit, time is counted in timesteps: `config.unit` is `null`, a bare number is written as a quantity with `"unit": null` (`{"type": "per", "value": 0.05, "unit": null}` is 0.05 per step), and `start` and `stop` are step numbers. Nothing is listed in `guessed`, because nothing was assumed.

Distributions, with Starsim's names:

```json
{"dist": "lognorm_ex", "mean": {"type": "dur", "value": 10.0, "unit": "day"}, "std": {"type": "dur", "value": 3.0, "unit": "day"}}
{"dist": "gamma", "a": 3, "scale": {"type": "dur", "value": 3.333, "unit": "day"}}
```

Values by stratum, time series, and seasonal curves:

```json
{"by": ["age"], "values": [0.08, 0.05, 0.03]}
{"by": ["age", "sex"], "product": {"age": [1.0, 2.0], "sex": [3.0, 4.0]}}
{"by": ["patch", "patch2"], "array": [[1.0, 0.1], [0.1, 1.0]]}
{"timeseries": {"2020-03-15": 0.2, "2020-06-01": 0.6}, "mode": "constant"}
{"seasonal": {"mean": 0.05, "amplitude": 0.3, "peak": "January"}}
```

Arrays are written in full. Named data (`ss.people('Kenya')`, `ss.contacts('Kenya')`, `ss.geo.geography(...)`) is stored as the provider, place, source, and version that were resolved ([04-parameters.md](04-parameters.md#named-data)).

## Functions

A function in a slot is the value of its parameter, written as its name, its source text, and the plain values (numbers, strings) it takes from outside its source (the variables it closes over, the globals it uses, its defaults):

```json
{"name": "rate_discharge", "value": {"fn": "clearance", "source": "def clearance(sim, uids):\n    return 0.2 / (1 + sim.people.viral_load[uids])\n", "values": null, "ref": null, "version": null},
 "bounds": null, "prior": null, "source": null, "doc": null}
```

Like any inline value, the parameter is named after its slot. A function passed to the disease as custom per-step behavior is listed in the module's `functions`, by name, in the same form. A function in a parameter of a module written as code, or in a modifier, is written the same way.

The source is what identifies a function. A function whose source is not available (NumPy's `np.exp`, a function from a compiled library) is identified instead by where it can be imported from, `ref` (`"numpy.exp"`). User code can give a function its own identity, `version` (`fn.version = '2'`), which is used in place of its source, so that the hash changes when the version does and not when a comment does. A function with none of these (a `functools.partial`, a lambda typed at a prompt) cannot be told apart from another, and `sim.hash()` raises an error that names it rather than treat two such models as the same; running and fitting do not need the hash, so they are not affected.

**A spec that contains functions is not portable.** It can be compared and hashed, but the model can only be run by Python, and only if everything the function refers to (imports, other functions, global variables) is available. The source text is recorded so that a person can see what the function does and so that the hash changes when it does. The code written by `sim.to_code()` includes the source text of each function.

## Normalization

The same model can be written in three spellings and in any keyword order. Normalization is what makes all of them produce the same document. `to_spec()` applies it, so it is what is hashed and compared.

1. **Resolve.** Every guessed default is written out as a value.
2. **Type.** Every bare number in a rate or duration slot becomes a quantity in the sim's unit (`null` in a sim with no unit).
3. **Expand shorthand.** Arrow strings, in any accepted spelling (`'I -> R'`, `'I->R'`, `'I >> R'`), become `src` and `dst` lists, with `ss.null` and a missing end of an arrow both written as `null`; `ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`, and the operator form all become the one transition object; a constant rate becomes an exponential `dur`; an inline value becomes a named parameter; an `init=` dict on the disease is written into the `init` field of each state; `transitions=`, `results=`, and `outcomes=` lists are merged with members passed as keywords.
4. **Order.** Modules, states, transitions, and stratifications stay in declared order (modules run in the order given; stratifications apply in sequence; `to_code()` writes the states and transitions as they were given). The state that receives the remainder of the initial population is written as the module's `first`, so that it does not depend on any order; `sim.hash()` sorts the transitions by name, since their order does not change the model. Parameters, results, and functions are sorted by name. Object keys are written in the order shown in this document. The terms of an expression are kept in the order written, so `beta * I` and `I * beta` are written (and hashed) differently.
5. **Numbers.** Floating-point values are written with full round-trip precision.

After normalization, the three spellings of the same model ([01-objects.md](01-objects.md#the-equivalence-rule)), and a class with the same members, MUST produce the same document once names agree (`tests/test_transitions.py::test_forms` checks that their hashes are equal).

## Hashing

`sim.hash()` returns an `sc.objdict` with six parts ([11-random.md](11-random.md#run-identity)): five hashes and the seed. Each hash is SHA-256, shortened to six characters, of one slice of the normalized spec, written as JSON with sorted keys and without the fields that are `null`. It depends only on that data, not on Python's `hash()` (which differs between processes) or on where objects are in memory, so it is the same in every process and on every machine, and is a durable fingerprint of the model's definition:

| Part | Computed from |
|---|---|
| `model` | `modules`, `networks`, `stratify`, and `modifiers`, with every number replaced by a placeholder: the structure, including the text of every expression and function, and where any named data came from |
| `pars` | Every number in those parts, in document order, with typed quantities converted to the sim's unit: the values |
| `config` | `config`, apart from `rand_seed`, including the numerical options that change results (`ss.options.crn` and `ss.options.precision`) |
| `scenario` | `interventions` |
| `data` | The data attached to the sim (as a table), or `None`; separate from `model` and `pars`, so that "the model changed" and "the data changed" can be told apart |
| `seed` | `config.rand_seed`, as it is; not a hash |

```text
>>> sim.hash()
#0. 'model':    '7fe765'
#1. 'pars':     '70b78a'
#2. 'config':   'a19c00'
#3. 'scenario': '4162b6'
#4. 'data':     '7b352e'
#5. 'seed':     1
```

Each part is a plain string or number, so `sim.hash().pars` can go in a file name or a column of a data frame. The hash is a by-product of the model being data. It tells apart the many runs that come from one commit (a sweep, a calibration) and is the test for whether two descriptions are the same model; the commit or tag remains the main record of what code was run.

- Changing a parameter from 0.05 to 0.08 changes `pars` and not `model`. Adding a state or a transition changes `model`, and so does changing a number written inside an expression (`0.5 * infection`), which is part of the expression's text.
- Because values are converted to one unit before hashing, `ss.days(7)` and `ss.weeks(1)` hash the same. Calendar durations (`ss.datedur`) are hashed as written.
- `meta`, `guessed`, `label`, `doc`, and a parameter's `source` do not contribute to any hash. Relabeling a state does not change a model, and neither does writing down a value that would have been guessed.
- Fitting settings are not part of `sim.hash()`, so changing the number of trials does not look like a change to the model. The data is its own part, `data`.
- `init_prev=0.01` and `init=dict(I=0.01)` are the same model, and hash the same.
- The runs of an `ss.MultiSim` differ only in `seed`.
- A function contributes its source text (or its `ref` or `version`, [above](#functions)) to `model`, and the plain values it closes over or has as defaults (numbers, strings) to `pars`. This is a best effort: a function that depends on something else outside its own source (a global, an object) can change behavior without changing the hash. Two different functions in the same slot, with the same parameters, do not hash the same.
- A module written as code contributes its class (`factory`), its `version` if its class gives one, and its parameters, but not its code: a change to its methods does not change the hash unless its `version` changes. Its source is not captured; giving a module a factory and its configuration as parameters is what makes it identifiable.
- Numbers written in rates are kept at full precision, so `beta * (1 + 1e-9) * I` and `beta * I` hash differently. A sim that is not initialized is hashed (from an initialized copy) without being changed.
- Two models that are the same mathematically but declared differently (two transitions out of a state versus one branching transition, say) may hash differently. A spurious difference is the safe direction to fail in.

## Versioning

`schema` is `lingua-franca/MAJOR.MINOR` (now `lingua-franca/1.0`), and `meta.library_version` is the version of Starsim that wrote the spec. Neither is part of any hash. Since nothing reads a spec back, there is no converter between versions; a change to the shape of the spec changes the hashes of the models it affects, and is listed in the changelog.

## Saved forms

A model is saved as code, `ss.save('model.py', sim)` (which writes `sim.to_code()`); running it MUST give a model whose own spec is identical, and therefore the same hash (`tests/test_transitions.py::test_load`). Only the declared parts of a model are written as code: declared diseases (with their functions, from their source), families of strains, deliveries with their products (each as a variable, so a step can refer to another's events), cascades, `ss.modify`, contact matrices (by name or as numbers), transfers, and priors that are distributions. Nothing is added to reconstruct the rest. What `to_code()` does not write (the attached data, a module or network written as code, named by its `factory`, a function whose source is not a plain definition, a prior that is not a distribution) it lists in the code, for adding by hand, and in an `ss.errors.LossyExport` warning; a part of a declared disease that it cannot write (the migration of a transition, a disease's custom functions) is an error rather than a different model. A round trip of every model in the test corpus (`tests/devtests/interop/test_interop.py::test_round_trip`) checks this. A sim with its results is saved with `ss.save('run.sim', sim)` and read with `ss.load('run.sim')`; saved partway through a run (`sim.run(until=...)`), it is a checkpoint that can continue. There is no third form: the spec itself is not read back, and the results as JSON (`sim.to_json()`) are output, not a model.

## Export to other formats

The model spec and the code `sim.to_code()` writes are the only complete forms. Exports to other frameworks' formats ([16-interop.md](16-interop.md)) usually cannot carry everything, and each MUST say what was left out or changed, one line per declaration, in an `ss.errors.LossyExport` warning:

```text
>>> ss.save('model.xml', sim, format='sbml')
exported to SBML: 3 states, 2 transitions, as 13 species and 12 reactions (rates per day)
  changed: the agent-based sim was written as its compartmental form (an ODE)
  changed: the lognorm_ex dwell time "dur" written as 11 exponential stages (Erlang) with its mean, 10 days
  changed: "sir" is stratified (age: 0-14, 15-64, 65+), written as one group, the sum of its groups, with the base values of its parameters
  not exported: network "randomnet" (contacts written as homogeneous mixing)
```

## What makes a model checkable

Because the model is data with a fixed shape, two descriptions of the same model produce the same document, so "did we build the same thing?" can be tested: by `sim.hash()`, or declaration by declaration with `ss.diff(a, b)`. A model written by a person or generated by an AI is checked by building it: the errors in [15-errors.md](15-errors.md) are raised when the model is made or the sim initialized, before it runs. No JSON Schema file is published, since nothing outside Python reads the spec.

## Open questions

- Whether a people property stratified on the only disease in a sim (`sir.stratify(age=...)`) and the same stratification on the sim (`sim.stratify(age=...)`) should normalize to one document. They are the same model; as written they are recorded where they were applied and hash differently.
- Whether the spec should optionally write out the expanded strata and sub-stages. The exporters expand the model from the engine, not from the spec, so nothing needs it now.
- The grammar of the spec is not rewritten for the saved form; see `deferred.md` (review R29). A prior that is a distribution is written as one (`{"dist": "normal", "loc": 0.3, "scale": 0.1}`), and any other prior as text.
