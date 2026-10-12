# Overview

The language on one page, then the object model behind it. Everything here is specified in detail elsewhere in this folder.

## The whole language in one model

```python
import starsim as ss

# --- ingredients: parameters and states -----------------------------------
beta = ss.par(0.05, bounds=[0.01, 0.5])     # per day; it has bounds, so it can be fitted
dur  = ss.days(10)
S, I, R = ss.states('S I R')

# --- the disease: transitions, written like the equations ------------------
infection = S >> I @ beta * I               # "S goes to I at rate beta times I"
recovery  = I >> R @ dur                    # a duration after @ is a mean dwell time
cases     = ss.outcome(infection, p=0.5, delay=ss.days(5))    # half of infections are reported, 5 days later
sir = ss.Disease(infection, recovery, cases, init_prev=0.01)  # names come from the variable names

# --- a stratification: age, with the mixing that goes with it --------------
sir = sir.stratify(age=[0, 15, 65])         # returns a new model

# --- a sim, with data attached to results by name --------------------------
kw  = dict(networks=ss.contacts('Kenya'), total_pop=50e6, unit='day', start='2020-01-01', stop='2021-01-01')
sim = ss.Sim(sir, data='weekly_cases.csv', **kw)     # the file has a column called 'cases'
sim.describe()                              # the resolved model, with every guessed default marked
sim.run()                                   # method='auto': prints "method: ode (every part has a compartmental form)"
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
```

The short form of the same disease, for when nothing needs a name, is two lines:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
ss.Sim(sir, unit='day').run().plot()
```

## What a model is made of

| Thing | Written as | Answers | Specified in |
|---|---|---|---|
| **State** | `ss.State`, `ss.states('S I R')` | What can someone be? | [03](03-states-and-transitions.md) |
| **Transition** | `S >> I @ rate`, or `ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`, `ss.transfer`; all are `ss.Transition` | How do they move between states? | [03](03-states-and-transitions.md) |
| **Parameter** | a number, or `ss.par(...)` when it needs a name, bounds, or a source | How fast, how likely, how long? | [02](02-types.md), [04](04-parameters.md) |
| **Stratification** | `sir.stratify(age=...)` | What else varies: age, place, risk, strain? | [05](05-dimensions.md) |
| **Mixing** | `ss.Sim(networks=...)` | Who contacts whom, and how often? | [06](06-routes.md) |
| **Interaction** | `ss.modify(...)`, or Python in a module's `step()` | What changes what? | [07](07-effects.md) |

Interventions ([08](08-interventions.md)) are a product or a change to a quantity, plus a delivery. Results ([09](09-results.md)) are automatic for every state and transition, extra ones are declared with the model, and data attaches to them by name. Calibration ([13](13-inference.md)) fits parameters that have bounds to results that have data.

## The object model

```
ss.Sim                          modules, a population, mixing, a timeline, a method
 ├── modules                    one or more ss.Module
 │     ├── ss.Disease           states people are in, and transitions between them
 │     │     ├── ss.State           declared, or inferred from arrow strings
 │     │     ├── ss.Transition      contact-mediated or spontaneous (derived from its rate)
 │     │     ├── ss.par             named parameters, in sir.pars
 │     │     └── results            n_<state>, one flow per transition, ss.outcome, ratios
 │     ├── ss.Compartments      states that are counts: reservoirs, vectors
 │     ├── ss.Demographics      births, deaths, pregnancy
 │     ├── ss.Intervention      ss.vaccinate, ss.screen, ss.treat, ss.change
 │     └── ss.Connector, ss.Analyzer
 ├── people:   ss.People        size, demography, per-agent attributes
 ├── networks: ss.Network, ss.MixingPool, ss.Mobility     none given means homogeneous mixing
 ├── modifiers: ss.modify       changes to named quantities while a condition holds
 ├── loop                       the order modules run in each step (sim.loop)
 └── results                    sim.results, with data attached by name
```

`ss.Module` is the single module abstraction: diseases, networks, demographics, interventions, connectors, and analyzers are all modules with the same lifecycle. This is Starsim's, unchanged. [01-objects.md](01-objects.md)

## One vocabulary, several lengths

The same transition can be written three ways, and a module can be built with a constructor or as a class. All of them produce the same model. [01-objects.md §The equivalence rule](01-objects.md#the-equivalence-rule)

```python
# Arrow strings: states inferred
infection = ss.transmission('S -> I', beta=0.05)

# Operators: states declared
S, I, R   = ss.states('S I R')
infection = S >> I @ 0.05 * I

# Fully explicit
infection = ss.Transition(S, I, rate=0.05 * I, name='infection')

# A class, when there is behavior that is not a transition
class SIR(ss.Disease):
    S, I, R   = ss.states('S I R')
    infection = S >> I @ 0.05 * I
    recovery  = I >> R @ ss.days(10)

    def step(self):
        super().step()
        self.do_something_bespoke()
```

Subclassing adds behavior. It does not add vocabulary.

## Four properties the design aims for

1. **The model is data.** States and transitions are objects, so the diagram, the equations, the stratified model, the hash, and the saved file are all derived from them. The serialized form is the model spec ([14-model-spec.md](14-model-spec.md)).
2. **The method is chosen for you, and can be changed.** `method='auto'` picks the simplest, fastest representation that can run the model exactly: an ODE if every part has a compartmental form, agents otherwise. `method='ode' | 'ctmc' | 'tau' | 'binomial' | 'sde' | 'abm'` asks for one. For simple models this is a free run-time choice; for complex models the method is effectively part of the specification. [12-backends.md](12-backends.md)
3. **Approximations are stated.** Under `'auto'` nothing is approximated. When a method is asked for that the model does not natively fit, the library uses the standard approximation where one exists and says so in one line; it stops only when there is no sensible approximation. `sir.methods()` shows which methods can run a model before anything runs.
4. **Defaults are guessed and printed.** `describe()` shows every inferred state, source of infection, dwell-time distribution, unit, and mixing assumption. [01-objects.md §`describe()`](01-objects.md#describe)

## What happens at build time

Nothing is computed when a model is written. `sim.init()`, which `sim.run()` calls if it has not been called already, does four things in order:

1. **Resolve defaults.** Names are assigned; states are inferred from arrow strings if none were declared; sources of infection, units of bare numbers, initial conditions, dwell-time distributions, and the mixing are filled in where they were not given. Every guess is recorded so that `describe()` can mark it.
2. **Check.** Unknown state and parameter names, shapes that do not match their stratification, branch probabilities that do not sum to 1, values outside their bounds. Problems that can be found here are reported here, before the first timestep, by code and name ([15-errors.md](15-errors.md)).
3. **Build the model spec.** The resolved model is written out as plain data ([14-model-spec.md](14-model-spec.md)): this is what gets hashed, saved, compared, drawn, and exported.
4. **Pick the method.** Under `method='auto'`, the library checks whether every part has a compartmental form (trying any user functions on groups) and chooses an ODE or agents accordingly; under an explicit method it works out what, if anything, has to be approximated. The choice, the reason, and any approximation are printed.

Then `sim.run()` steps through the loop ([10-execution.md](10-execution.md)), in which modules run in a defined, printable order.

## Reading order

- Writing models: this page → [03](03-states-and-transitions.md) → [04](04-parameters.md) → [06](06-routes.md) → [08](08-interventions.md) → [09](09-results.md).
- Implementing the language: [01](01-objects.md) → [02](02-types.md) → [03](03-states-and-transitions.md) → [14](14-model-spec.md) → [10](10-execution.md) → [12](12-backends.md).
- Adding a run method: [14](14-model-spec.md) → [12](12-backends.md) → [10](10-execution.md) → [18](18-conformance.md).
- Reviewing the design: [17-vocabulary.md](17-vocabulary.md) for the total surface, then [19-open-questions.md](19-open-questions.md) for what is not settled.
