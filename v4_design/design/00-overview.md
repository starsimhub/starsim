# Overview

The language on one page, then the object model behind it. Everything here is specified in detail elsewhere in this folder.

## The whole language in one model

```python
import starsim as ss

# --- ingredients: parameters and states -----------------------------------
beta = ss.par(0.05, bounds=[0.01, 0.5])     # per day; it has bounds, so it can be fitted
dur  = ss.days(10)
S, I, R = ss.states(3)                      # named from the variables they are assigned to

# --- the disease: transitions, written like the equations ------------------
infection = S >> I @ beta * I               # "S goes to I at rate beta times I"
recovery  = I >> R @ dur                    # a duration after @ is a mean dwell time
cases     = ss.outcome(0.5 * infection, delay=ss.days(5))   # half of infections are reported, 5 days later; weekly data is compared with the total over each week
sir = ss.Disease(infection, recovery, cases, init_prev=0.01)  # names come from the variable names

# --- a sim: age groups, the mixing that goes with them, and data attached to results by name
kw  = dict(stratify=dict(age=[0, 15, 65]), networks=ss.contacts('Kenya'), total_pop=50e6, start='2020-01-01', stop='2021-01-01')
sim = ss.Sim(sir, data='weekly_cases.csv', **kw)     # the file has a column called 'cases'
sim.describe()                              # initializes the sim, which prints "solver: ode (every part of the model has a compartmental form)" and the unit (day, from the typed values); then the resolved model, with every guessed default marked
sim.run()
sim.plot()                                  # results, with the data drawn over 'cases'

# --- calibration: fit what has bounds to what has data ---------------------
fit = sim.calibrate()

# --- an intervention, and scenarios ----------------------------------------
masks = ss.modify(sir.pars.beta, multiply=0.6, start='2020-03-15')
sims  = ss.parallel(
    baseline = fit.sim,                     # the calibrated sim
    masks    = ss.Sim(sir, pars=fit.pars, interventions=masks, **kw),
)
sims.plot()                                 # ss.parallel runs the sims

# --- chance: a sim is one run; many runs are a MultiSim --------------------
msim = ss.parallel(ss.Sim(sir, solver='tau', **kw), n_runs=100)   # stochastic: tau-leaping, since 50 million people are too many events to take one at a time
msim.plot()                                 # the median and an interval
```

The short form of the same disease, for when nothing needs a name, is two lines:

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=10),
    init_prev = 0.01,
)
ss.Sim(sir, stop=100).run().plot()          # nothing has a unit, so time is in timesteps: 100 steps
```

## What a model is made of

| Thing | Written as | Answers | Specified in |
|---|---|---|---|
| **State** | `ss.State`, `ss.states(3)` | What can someone be? (For a reservoir or a vector population: what is being counted?) | [03](03-states-and-transitions.md) |
| **Transition** | `S >> I @ rate`, or `ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`, `ss.transfer`; all are `ss.Transition` | How do they move between states? | [03](03-states-and-transitions.md) |
| **Parameter** | a number, or `ss.par(...)` when it needs a name, bounds, or a source | How fast, how likely, how long? | [02](02-types.md), [04](04-parameters.md) |
| **Stratification** | `sim.stratify(age=...)` for a property of people; `sir.stratify(strain=...)` for one disease's own dimension | What else varies: age, place, risk, strain? | [05](05-dimensions.md) |
| **Mixing** | `ss.Sim(networks=...)` | Who contacts whom, and how often? | [06](06-routes.md) |
| **Interaction** | `ss.modify(...)`, or Python in a module's `step()` | What changes what? | [07](07-effects.md) |

Interventions ([08](08-interventions.md)) are a product or a change to a quantity, plus a delivery. Results ([09](09-results.md)) are automatic for every state and transition, extra ones are declared with the model, and data attaches to them by name. A sim is one run; many runs are an `ss.MultiSim`, made by `ss.parallel(sim, n_runs=100)` or its shortcut `sim.run(n_runs=100)`. Calibration ([13](13-inference.md)) fits parameters that have bounds to results that have data.

## The object model

```
ss.Sim                          modules, a population, mixing, a timeline, a solver
 ├── modules                    one or more ss.Module
 │     ├── ss.Disease           states people are in, and transitions between them (ss.Infection for a disease written as code)
 │     │     ├── ss.State           declared; people, or a plain count
 │     │     ├── ss.Transition      contact-mediated or spontaneous (derived from its rate)
 │     │     ├── ss.par             named parameters, in sir.pars
 │     │     └── results            n_<state>, one flow per transition, ss.outcome, ratios
 │     ├── ss.Demographics      births, deaths, pregnancy
 │     ├── ss.Intervention      deliveries: ss.vaccinate, ss.screen, ss.treat
 │     └── ss.Connector, ss.Analyzer
 ├── people:   ss.People        size, demography, per-agent attributes
 ├── stratifications            dimensions such as age and place (sim.stratify), with ss.geo.Mobility between places
 ├── networks: ss.Network, ss.MixingPool     none given means homogeneous mixing
 ├── modifiers: ss.modify, ss.change     changes to named quantities while a condition holds, or at given times
 ├── loop                       the order modules run in each step (sim.loop)
 └── results                    sim.results, with data attached by name
ss.MultiSim                     many runs of a sim (ss.parallel); its results have a run dimension
```

`ss.Module` is the single module abstraction: diseases, networks, demographics, interventions, connectors, and analyzers are all modules with the same lifecycle. This is Starsim's, unchanged. [01-objects.md](01-objects.md)

## One vocabulary, several lengths

The same transition can be written two ways, and a module can be built with a constructor or as a class. All of them produce the same model. [01-objects.md §The equivalence rule](01-objects.md#the-equivalence-rule)

```python
# Operators: states declared
S, I, R   = ss.states(3)
infection = S >> I @ 0.3 * I

# Fully explicit
infection = ss.Transition(S, I, rate=0.3 * I, name='infection')

# A class, when there is behavior that is not a transition
class SIR(ss.Disease):
    S, I, R   = ss.states(3)
    infection = S >> I @ 0.3 * I
    recovery  = I >> R @ ss.days(10)

    def step(self):
        super().step()
        self.do_something_bespoke()
```

Subclassing adds behavior. It does not add vocabulary.

## Four properties the design aims for

1. **The model is data.** States and transitions are objects, so the diagram, the equations, the stratified model, the hash, and the saved file are all derived from them. The serialized form is the model spec ([14-model-spec.md](14-model-spec.md)).
2. **The solver is chosen for you, and can be changed.** `solver='auto'` picks the simplest, fastest representation that runs the model faithfully: an ODE if every part has a compartmental form, agents otherwise. `solver='ode' | 'binomial' | 'gillespie' | 'abm'` asks for one. For simple models this is a free run-time choice; for complex models the solver is effectively part of the specification. [12-backends.md](12-backends.md)
3. **Approximations are stated.** `'auto'` never approximates the mixing and never drops individual-level structure; the one thing it does unasked is represent a non-exponential dwell time by Erlang sub-stages, and it says so. When a solver is asked for that the model does not natively fit, the library uses the standard approximation where one exists and says so in one line; it stops only when there is no sensible approximation. `approx=False` makes any approximation an error, and `approx=True` keeps the note in `sim.notes` without printing it. `sir.solvers()` shows which solvers can run a model before anything runs.
4. **Defaults are guessed and printed.** `describe()` shows every guessed dwell-time distribution, unit, and mixing assumption, and which states are people and which are counts. [01-objects.md §`describe()`](01-objects.md#describe)

## What happens at build time

Nothing is computed when a model is written. `sim.init()`, which `sim.run()` (and `sim.describe()`) calls if it has not been called already, does three things in order:

1. **Resolve defaults and check.** Names are assigned (states and transitions take theirs from the variables they are bound to); sources of infection (from the rates), which states are people, initial conditions, dwell-time distributions, and the mixing are filled in where they were not given. Unknown state and parameter names, values outside their bounds, and shapes that do not match their stratification are reported here, before the first timestep, as named errors ([15-errors.md](15-errors.md)). Every guess is recorded so that `describe()` can mark it.
2. **Pick the unit and the solver.** The sim's unit comes from the model ([02-types.md](02-types.md#the-sims-unit)). Under `solver='auto'`, the library checks whether every part has a compartmental form (trying any user functions on groups, and turning a non-exponential dwell time into sub-stages) and chooses an ODE or agents accordingly; under an explicit solver it works out what, if anything, has to be approximated. The choice and its reason are printed (`solver: ode (every part of the model has a compartmental form)`), with a note for the unit if it was inferred and one for each approximation.
3. **Build the population and the results.** The people (agents, or one row per group in a compartmental run), the initial values of every state, the results, and the loop. For a compartmental run with transmission, it prints R0 and the number infectious at the start (`sir: R0 = 3, with 100 infectious at the start`), as a check that the model and its seed are what was meant.

The model spec ([14-model-spec.md](14-model-spec.md)), which is what gets hashed, compared, and saved as code, is built from the resolved model when it is asked for (`sim.to_spec()`, `sim.hash()`, `ss.diff()`, `sim.to_code()`), not at `init()`.

Then `sim.run()` steps through the loop ([10-execution.md](10-execution.md)), in which modules run in a defined, printable order.

## Reading order

- Writing models: this page → [03](03-states-and-transitions.md) → [04](04-parameters.md) → [06](06-routes.md) → [08](08-interventions.md) → [09](09-results.md).
- Implementing the language: [01](01-objects.md) → [02](02-types.md) → [03](03-states-and-transitions.md) → [14](14-model-spec.md) → [10](10-execution.md) → [12](12-backends.md).
- Adding a solver: [14](14-model-spec.md) → [12](12-backends.md) → [10](10-execution.md) → [18](18-conformance.md).
- Reviewing the design: [17-vocabulary.md](17-vocabulary.md) for the total surface, then [19-open-questions.md](19-open-questions.md) for what is not settled.
