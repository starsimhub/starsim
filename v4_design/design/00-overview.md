# Overview

The language on one page, then the object model behind it.

Everything here is specified in detail elsewhere in this folder; this document exists so that the shape of the whole is visible before the parts.

## The whole language in one model

```python
import starsim as ss

# --- the disease: states and transitions ---------------------------------
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

# --- a dimension: age, with the mixing that goes with it -----------------
sir.stratify(age=[0, 15, 65], mixing=ss.contacts('Kenya'))

# --- the surveillance system, separate from the disease ------------------
sir.observe(
    cases = ss.Outcome(sir.infection, p=0.5, delay=ss.days(5),
                       data='data/weekly_cases.csv', every=ss.weeks(1),
                       likelihood=ss.neg_binomial(k=ss.Par(bounds=[0.1, 100]))),
)

# --- an intervention -----------------------------------------------------
masks = ss.Effect(ss.time > ss.date('2020-03-15'), sir.beta, multiply=0.6)

# --- scenarios, and a run ------------------------------------------------
sims = ss.parallel(
    baseline = ss.Sim(sir, start='2020-01-01', stop='2021-01-01'),
    masks    = ss.Sim(sir, start='2020-01-01', stop='2021-01-01', interventions=masks),
)
sims.run()
sims.plot()
```

Every line above is normative: each construct is specified, each default is printed by `summary()`, and each is available in all four paradigms except where [12-backends.md](12-backends.md) says otherwise.

## The five things a model is made of

| Thing | Class | Answers | Specified in |
|---|---|---|---|
| **State** | `ss.State` | What can someone be? | [03](03-states-and-transitions.md) |
| **Transition** | `ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`, `ss.transfer`, `ss.process` | How do they move between states? | [03](03-states-and-transitions.md) |
| **Dimension** | `Model.stratify()` | What else varies — age, place, risk, strain? | [05](05-dimensions.md) |
| **Route** | `ss.Route` and subclasses | Who contacts whom, how often? | [06](06-routes.md) |
| **Effect** | `ss.Effect` | What modifies what, and how do modifications compose? | [07](07-effects.md) |

Interventions ([08](08-interventions.md)) are built from transfers and effects. Results and the observation model ([09](09-results.md)) are declared alongside, never inside, the disease. Nothing else is a first-class concept.

## The four claims

The specification exists to make four claims checkable rather than aspirational.

1. **The model is data.** A model compiles to the intermediate representation in [14-ir.md](14-ir.md) before it computes anything. Everything downstream — dimension checking, stratification, diagrams, ODE generation, gradients, hashing, format translation — is a function of that representation. [Justification: best-practices/model-structure.md]
2. **The paradigm is a run-time choice.** `run(method='ode' | 'ctmc' | 'tau' | 'binomial' | 'sde' | 'abm')` selects a backend; the model text does not change. [12-backends.md](12-backends.md)
3. **Nothing is silently approximated.** Every conversion is exact, approximate-with-a-named-closure, or refused by name. `Model.capabilities()` reports which, before anything runs. [12-backends.md](12-backends.md), [15-errors.md](15-errors.md)
4. **Defaults are guessed and printed.** `summary()` shows every inferred state, dwell-time distribution, unit, denominator convention, and combination rule. A guess the user can see is not hidden behavior. [01-objects.md](01-objects.md) §Summary

## The object model

```
ss.Sim                          a model, a population, a route set, a timeline, a method
 ├── model:  ss.Module          one or more; ss.Disease is the common case
 │     ├── ss.State             declared or inferred from transitions
 │     ├── ss.Transition        transmission | progression | birth | death | transfer | process
 │     ├── ss.Dimension         added by stratify(); age, patch, strain, or plain
 │     └── ss.Outcome           the observation model
 ├── people: ss.People          size, demography, and per-agent attributes
 ├── mixing: ss.Route           Homogeneous | MixingPool | RandomNet | SexualNet | Network | Mobility
 ├── effects: ss.Effect         cross-module modifications, with declared combiners
 └── results: ss.Result         auto-generated per state and per transition
```

`ss.Module` is the single module abstraction: diseases, networks, demographics, interventions, connectors, and analyzers are all modules with the same lifecycle. This is Starsim's, unchanged. [01-objects.md](01-objects.md)

## Three verbosities, one vocabulary

The same model, written three ways. Any keyword accepted by one form is accepted by the others and means the same thing. [01-objects.md §The equivalence rule](01-objects.md#the-equivalence-rule)

```python
# 1. Dict — for the SIR-shaped majority
sir = dict(infection=('S -> I', 0.05), recovery=('I -> R', ss.days(10)))

# 2. Constructors — the canonical form
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

# 3. Class — when there is behavior that is not a transition
class SIR(ss.Disease):
    infection = ss.transmission('S -> I', beta=0.05)
    recovery  = ss.progression('I -> R', dur=ss.days(10))

    def step(self):
        super().step()
        self.log_something_bespoke()
```

Subclassing adds behavior. It never adds vocabulary.

## What the compiler does

```
model text  ──parse──▶  object graph  ──normalize──▶  IR  ──lower──▶  backend
                             │                        │                  │
                        inference               canonical form       ode / ctmc /
                     (states, kinds,           (hashing, diff,       tau / binomial /
                      units, defaults)          serialization,        sde / abm
                             │                   translation)            │
                             ▼                        ▼                  ▼
                        summary()               hash(), save()       results
```

Four stages, specified in [01](01-objects.md) (parse), [03](03-states-and-transitions.md)–[09](09-results.md) (inference), [13](14-ir.md) (normalize), and [12](12-backends.md) (lower). Errors are raised at the earliest stage that can detect them, which for almost everything is before the first timestep. [15-errors.md](15-errors.md)

## Reading order

- Implementing the language: [01](01-objects.md) → [02](02-types.md) → [03](03-states-and-transitions.md) → [13](14-ir.md) → [10](10-execution.md) → [12](12-backends.md).
- Writing a backend: [13](14-ir.md) → [12](12-backends.md) → [10](10-execution.md) → [17](18-conformance.md).
- Writing models: [00](00-overview.md) → [03](03-states-and-transitions.md) → [06](06-routes.md) → [08](08-interventions.md) → [09](09-results.md).
- Reviewing the design: [17-vocabulary.md](17-vocabulary.md) for the total surface, then [19-open-questions.md](19-open-questions.md) for what is not settled.
