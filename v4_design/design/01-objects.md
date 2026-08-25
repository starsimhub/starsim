# Objects and construction

The class hierarchy, how models are constructed, and the rules that keep the short form and the long form the same language.

Justified by [best-practices/principles.md](../best-practices/principles.md) and [best-practices/model-structure.md](../best-practices/model-structure.md).

## The hierarchy

```
ss.Base                     serialization, hashing, summary, equality
 ├── ss.Module              lifecycle: init → step → finish; owns states, transitions, results
 │    ├── ss.Disease        the common case; states and transitions
 │    ├── ss.Route          contact structure (see 06-routes.md)
 │    ├── ss.Demographics   births, deaths, ageing, migration
 │    ├── ss.Intervention   products and delivery (see 08-interventions.md)
 │    ├── ss.Connector      imperative cross-module code; must declare `writes`
 │    └── ss.Analyzer       observation only; may not write
 ├── ss.Sim                 a model plus a population, a timeline, and a method
 ├── ss.Declaration         anything valid as a keyword in a Module body
 │    ├── ss.State
 │    ├── ss.Transition     transmission | progression | birth | death | transfer | process
 │    ├── ss.Par
 │    ├── ss.Outcome
 │    └── ss.Continuous
 ├── ss.Effect              a declared modification of a named quantity
 ├── ss.Expr                a predicate or arithmetic expression over model state
 ├── ss.Quantity            a typed number (see 02-types.md)
 │    ├── ss.dur  ss.rate  ss.prob  ss.freq  ss.count
 │    └── ss.Dist           a distribution, which is a Quantity-valued random variable
 └── ss.Result              a named time series with units and provenance
```

`ss.Module` MUST be the only module abstraction. Diseases, networks, demographics, interventions, connectors, and analyzers differ in what they declare, never in how they are scheduled or initialized. This is Starsim's design and it is kept unchanged; it is what makes multi-disease models and non-disease health states composable.

## Constructing a model

### Form 2 is canonical

```python
ss.Disease(
    <name> = <declaration>,      # one keyword per state, transition, parameter, or outcome
    ...,
    name  = None,                # module name; defaults to the variable name, then the class name
    dt    = None,                # this module's timestep; defaults to the sim's (10-execution.md)
    phase = 'main',              # execution phase (10-execution.md)
)
```

Every keyword whose value is an `ss.Declaration` becomes a member of the module under that name. **The keyword is the name**, and that name is used everywhere else: in results (`sim.results.infection`), in effects (`ss.Effect(..., sir.beta, ...)`), in adjustments (`adjust={'recovery': ...}`), in the IR, and in error messages.

Names MUST be valid Python identifiers and MUST be unique within a module. A name that collides with a reserved attribute of `ss.Module` (`step`, `init`, `results`, `summary`, `stratify`, `observe`, `name`, `dt`, `phase`, `requires`, `writes`) MUST raise `E201 ReservedName`.

### Form 1: the dict

A `dict` passed where a module is expected is expanded to `ss.Disease(**d)`. Values that are not `ss.Declaration` instances are coerced by these rules, and by no others:

| Value shape | Coerced to | Condition |
|---|---|---|
| `('A -> B', q)` where `q` is a rate or bare number | `ss.transmission('A -> B', beta=q)` | the arrow's source is a susceptible state — that is, a state with no inbound transition |
| `('A -> B', q)` where `q` is a duration | `ss.progression('A -> B', dur=q)` | always |
| `('A -> B', q)` where `q` is a probability | `ss.progression('A -> B', p=q)` | always |
| a number, `ss.Quantity`, or `ss.Dist` | `ss.Par(value)` | always |
| a `dict` | a nested module | always |

The first row is the only inference that depends on model context, and it is deliberately narrow: it resolves the SIR/SIS/SEIR shapes and nothing else. If both rules could apply, or neither, the coercion MUST raise `E202 AmbiguousShorthand` naming the tuple and showing the two explicit spellings. The shorthand is for the common case; it is not a general syntax.

```python
sir = dict(infection=('S -> I', 0.05), recovery=('I -> R', ss.days(10)))
ss.Sim(sir).run()
```

### Form 3: the class

Class attributes that are `ss.Declaration` instances are collected in definition order and are equivalent to the same keywords passed to the constructor. Subclasses inherit their parents' declarations; a subclass may override a declaration by rebinding the name, and may add behavior by defining methods.

```python
class SEIR(ss.Disease):
    infection = ss.transmission('S -> E', beta=0.05, source='I')
    latency   = ss.progression('E -> I', dur=ss.days(3))
    recovery  = ss.progression('I -> R', dur=ss.days(10))
```

Inside a class body, a declaration may reference an earlier one by name, because it is an ordinary Python name at that point:

```python
class SIR(ss.Disease):
    I         = ss.State(init=0.01, infectious=True)
    infection = ss.transmission('S -> I', beta=0.05)
    recovery  = ss.progression('I -> R', dur=ss.days(10))
    prevalence = I / ss.N                       # a characteristic; see 09-results.md
```

### The equivalence rule

This rule is normative and constrains every other document in this folder:

> **Any keyword accepted by the dict form MUST be accepted by the constructor form and the class form, and MUST mean the same thing. Any model expressible in the class form MUST have a constructor form and, unless it defines methods, a dict form. Subclassing adds behavior; it MUST NOT add vocabulary.**

Concretely: a feature MUST NOT be implemented as a method that only subclasses can call, if the same feature is reachable by keyword. A feature MUST NOT require the class form unless it is imperative code. Any proposal that violates this rule is rejected regardless of its other merits.

The corollary for conformance: for every model in the test corpus, the three forms MUST produce byte-identical IR ([14-ir.md](14-ir.md)) and therefore identical model hashes. [18-conformance.md](18-conformance.md) §Form equivalence.

## `ss.Sim`

```python
ss.Sim(
    model,                       # a Module, dict, or list of either
    *,
    # --- population ---
    people    = None,            # ss.People, or None to build from n
    n         = 1e6,             # population size represented (semantic, all backends)
    n_agents  = None,            # simulated agents (ABM only; defaults to n, see below)
    mixing    = None,            # ss.Route or list of routes; defaults to ss.Homogeneous()
    # --- time ---
    start     = None,            # date, year, or 0
    stop      = None,
    dt        = None,            # default timestep; modules may override
    unit      = None,            # unit assumed for bare numbers (02-types.md)
    # --- execution ---
    method    = None,            # 'ode' | 'ctmc' | 'tau' | 'binomial' | 'sde' | 'abm' | 'stochastic'
    n_reps    = 1,
    rand_seed = 0,
    crn_key   = 'slot',
    # --- content ---
    interventions = None,
    pars      = None,            # external parameter values (04-parameters.md)
    results   = None,            # restrict the result set (09-results.md)
    label     = None,
)
```

Only `model` is required. Every other argument has a default that is guessed and printed.

### `n` versus `n_agents`

These are different questions and MUST NOT be conflated.

- **`n` is the size of the population being represented.** It is semantic: it appears in the ODE as the compartment total, in the CTMC as the state space size, and in the ABM as the population the agents stand for. It affects the answer.
- **`n_agents` is simulation resolution.** It exists only in the agent-based backend and is a computational choice, like `dt`.

If `n_agents < n`, every result is scaled by `n / n_agents` and `summary()` MUST print the scale factor and a note that stochastic variance is inflated by it. If `n_agents` is unset it defaults to `n` when `n <= 1e6`, and otherwise raises `E801 ResolutionUnset`, because silently simulating 100 million agents and silently simulating 100 thousand are both wrong answers to an unasked question.

Starsim's current `n_agents` keyword maps to `n_agents`, and models that set it without setting `n` behave as before, with `n = n_agents`.

## Lifecycle

Every module has the same three-phase lifecycle, and the phases are the same in every backend:

| Phase | Method | May do |
|---|---|---|
| Build | `init(sim)` | resolve declarations, allocate state, register effects, check requirements |
| Run | `step()` | read start-of-step state; queue updates ([10-execution.md](10-execution.md)) |
| Finish | `finish()` | finalize results |

A module that declares only transitions, states, parameters, and outcomes MUST NOT need to define any of these. `step()` exists for the escape hatch and for genuinely imperative processes; a module that defines it is flagged in `capabilities()` ([12-backends.md](12-backends.md)).

## The module contract

```python
class MyModule(ss.Module):
    requires = [ss.Needs('contact_matrix', shape='age x age')]   # what I read
    writes   = [hiv.rel_trans]                                   # what I modify
```

- `requires` is checked at build time against the available data; a missing requirement raises `E601 MissingRequirement` naming the module, the attribute, and where it could come from. See [04-parameters.md](04-parameters.md).
- `writes` grants write access. A module without a `writes` entry for a quantity receives a read-only view of it, and an attempt to write raises `E501 UndeclaredWrite`. See [07-effects.md](07-effects.md).

Both are inferred for declarative modules and MUST be written only by modules that define `step()`. A module with no `step()` that declares `requires` or `writes` is not an error, but `summary()` notes that the declarations are redundant.

## Naming conventions

Normative, because consistency here is what makes the vocabulary memorable:

| Kind of name | Convention | Examples |
|---|---|---|
| Classes: things with a lifecycle | `CamelCase` | `ss.Sim`, `ss.Disease`, `ss.RandomNet`, `ss.MixingPool`, `ss.Effect`, `ss.State` |
| Constructors returning values | `lowercase` | `ss.transmission`, `ss.progression`, `ss.transfer`, `ss.days`, `ss.peryear`, `ss.lognorm`, `ss.gravity` |
| Verbs: things that do something | `lowercase` | `ss.vaccinate`, `ss.screen`, `ss.treat`, `ss.change`, `ss.parallel`, `ss.diff`, `ss.export` |
| Methods that declare | plural noun | `Model.results(...)`, `Model.observe(...)`, `Model.stratify(...)` |
| Attributes that hold | plural noun | `sim.results`, `sim.transmissions` |

This follows Starsim's existing split (`ss.SIR` and `ss.RandomNet` versus `ss.bernoulli` and `ss.peryear`) and extends it consistently.

## `summary()`

`summary()` is load-bearing, not decorative. It is the mechanism that makes aggressive default-guessing safe, and it is therefore normative.

Every object with declarations MUST implement `summary()`. It MUST print, for a model:

1. Every state, with its properties and whether it was declared or inferred.
2. Every transition, with its resolved kind, its rate or dwell-time distribution **as resolved** (`dwell ~ Exp(mean=10 days)`, not `dur=10`), and its source states where applicable.
3. Every dimension, with its levels and its kind.
4. Every route, with its denominator convention.
5. Every effect, with its target and combiner.
6. Every value that was **inferred rather than given**, marked as such.
7. Every declaration that costs a capability, marked with `!` and the capability lost.

It MUST NOT print values that were given explicitly and are unremarkable; the summary is a diff against the user's intent, not a dump.

```text
>>> sir.summary()
Disease 'sir' — 3 states, 2 transitions

states       S, I, R                                        [inferred from transitions]
             I infectious                                   [inferred: destination of 'infection']
             init: S=1.0                                    [inferred]

infection    S -> I   transmission, source=I                [source inferred]
             beta = 0.05/day  per contact                   [unit inferred: day]
recovery     I -> R   progression
             dwell ~ Exp(mean = 10 days)                    [inferred: scalar dur]

route        Homogeneous(n_contacts=10), frequency-dependent [both inferred]
results      n_S n_I n_R infection recovery cum_infection cum_recovery
```

`summary(verbose=True)` additionally prints the per-step conversions ([02-types.md](02-types.md)), the compiled transition list, and the compartment or agent counts.

## Equality, copying, and identity

- Two models are **equal** if their normalized IR is identical ([14-ir.md](14-ir.md)). Equality MUST NOT depend on construction form, declaration order, or variable names.
- `copy()` MUST be deep for declarations and shallow for data references; a copied model that fetched `ss.People('Kenya')` MUST NOT re-fetch.
- A model's identity for reproducibility is its four-level hash, not its Python object identity. See [11-random.md](11-random.md).

## Rejected constructions

Listed here because they will otherwise be proposed again:

- **A registry of model types by string** (`ss.Sim(diseases='sir')`). Starsim accepts this today; it is retained only as sugar for the shipped library models, and it MUST resolve to an ordinary `ss.Disease` that `summary()` prints in full. A string that cannot be printed back as a model is a second language.
- **Builder methods that mutate structure after construction** (`m.add_transition_flow(...)`). `stratify()` and `observe()` are the only structural mutators, and both are operators with defined IR semantics. Everything else is constructor keywords.
- **Positional arguments beyond the first.** `ss.transmission('S -> I', 0.05)` is rejected in favor of `beta=0.05`; the arrow is the only positional argument in the language, and the dict shorthand's `(arrow, quantity)` tuple is the only exception.
- **A separate `Model` and `ModelBuilder`.** One class, constructed once.
