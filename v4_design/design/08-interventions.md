# Interventions

What is delivered, and how it is delivered: products, delivery, follow-ups, parameter changes, triggers, and scenarios.

Justified by [best-practices/interventions.md](../best-practices/interventions.md).

## What plus delivery

An intervention has two parts.

- **What**: a product (a vaccine, a test, a treatment) or a change to a quantity. This part is cleanly separable and reusable.
- **Delivery**: who gets it, when, and how many. These are often not separable from each other ("send drugs to the 100 clinics with the most positive tests, while stocks last"), so delivery is written either with simple keywords or as one function that returns who gets it this step.

```python
bcg = ss.Vaccine(efficacy=0.85)                                                 # what
ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=ppl.age < 5)       # delivery, by keywords
ss.vaccinate(bcg, deliver=my_function)                                          # delivery, by a function
```

Interventions are passed to the sim with `interventions=`.

## Simple cases that need no product

**A change to a parameter over time** is `ss.change` ([below](#sschange)):

```python
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
```

**A change that applies to some people, or under a condition,** is `ss.modify` ([07-effects.md](07-effects.md)):

```python
ss.modify(school.pars.n_contacts, multiply=0.2, start='2020-03-15', stop='2020-06-01')
ss.modify(sir.rel_sus, multiply=0.6, where=sir.has_net)
```

**Vaccination as a move between strata** is `ss.transfer` ([05-dimensions.md](05-dimensions.md#moving-between-coordinates-sstransfer)). The disease does not mention vaccination; protection is declared on the dimension and delivery is a transfer:

```python
sir_vax = sir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.15]})
rollout = ss.transfer(vaccinated=True, coverage=0.7, years=[2021, 2022], eligible=ppl.age > 65)
```

## Products

```python
ss.Vaccine(efficacy=0.85, waning=ss.years(5), cost=12.50)
ss.Vaccine(efficacy=dict(acquisition=0.85, transmission=0.3, severity=0.5))
ss.Test(sensitivity=0.9, specificity=0.98, cost=4.00, delay=ss.days(2))
ss.Treatment(efficacy=0.95, cost=120.0)
```

These are Starsim's `ss.Vx`, `ss.Dx`, and `ss.Tx` under longer names; the short names remain as aliases.

**Efficacy.** A scalar `efficacy` on a vaccine means protection against acquisition. A dict gives efficacy against several things at once. Its keys are `acquisition` (scales `rel_sus`), `transmission` (scales `rel_trans`), `mortality` (scales the disease's death transitions), or the name of any transition in the disease (`severity` in a model whose transition to severe disease has that name); an unknown key is an error listing the valid ones. Each entry multiplies its quantity by `1 − efficacy` for the people who hold the product.

**Waning.** `efficacy` is the effect at full strength and `waning` is how it fades with time since delivery. A duration means exponential decay with that mean (printed); a function of time since delivery returning the remaining fraction, or an `ss.timeseries`, gives any other shape.

**A product is realized as modifiers** on the people who hold it, so it follows the combination rules in [07-effects.md](07-effects.md): two vaccines against acquisition multiply, a vaccine and a stratification adjustment multiply, and all of them are listed by `sim.what_modifies()`.

**Tests.** `sensitivity` and `specificity` are spelled out. A test returns a result per person, which is what follow-ups branch on. By default it detects infection with the disease it is delivered for (the states the model treats as infected, printed by `describe()`); `detects=` takes a [condition](07-effects.md#conditions) to say something else. `delay=` is the time from test to result.

**Treatments.** A scalar `efficacy` is the probability that treatment clears the infection: the person leaves the infected states by the disease's recovery transition (guessed and printed; `transition=` names it when there are several). A dict, as for vaccines, describes a treatment that changes rates without curing, such as antiretroviral therapy (`efficacy=dict(transmission=0.96, mortality=0.9)`).

**Cost.** `cost=` is per delivery. Total cost is then a result of the intervention, which is all v1 needs for cost-effectiveness. Budget optimization is out of scope for v1.

## Delivery

```python
ss.vaccinate(product, **delivery)
ss.screen(test, on_positive=None, on_negative=None, **delivery)
ss.treat(treatment, **delivery)
```

### Delivery keywords

| Keyword | Meaning |
|---|---|
| `start=`, `stop=` | the period of delivery (dates, years, or durations since the start of the sim) |
| `years=` / `dates=` | deliver in rounds at these times (a campaign) |
| `interval=` | deliver in rounds this far apart, between `start` and `stop` |
| `eligible=` | who can receive it: a [condition](07-effects.md#conditions) |
| `coverage=` | the fraction of eligible people reached |
| `capacity=` | the number that can be delivered |
| `when=`, `min_duration=` | a [trigger](#triggers): deliver only while a function of the sim is true |

```python
ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=ppl.age < 5)                # two campaign rounds
ss.vaccinate(bcg, coverage=0.7, interval=ss.years(1), start=2021, eligible=ppl.age < 5)  # a yearly round
ss.vaccinate(bcg, coverage=0.9, start=2021, eligible=ppl.age >= 0.75)                    # routine, at 9 months
```

**Rounds and routine delivery.** With `years=`, `dates=`, or `interval=`, delivery happens in rounds, and in each round every eligible person is offered the product once. With only `start=`/`stop=` (or nothing), delivery is routine: each person is offered the product once, at the first step on which they are eligible. Routine versus campaign is therefore a keyword, not a class.

**`coverage=` versus `capacity=`.** These are different specifications and both are common.

- `coverage=0.7` is a fraction of the eligible people. It accepts everything a parameter accepts, including `ss.by(age=[...])`, a time series, and a function of the sim returning one probability per person.
- `capacity=50_000` is a number available: per round for campaigns, or as a rate for routine delivery (`ss.peryear(50_000)`; a bare number is per unit time in the sim's unit). It may be a time series. When fewer are available than eligible people who would receive one, recipients are chosen at random among them, and delivery stops when the capacity is used up ("while stocks last").
- Given together, coverage is applied first and capacity caps the result.

Each delivery reports the number delivered and the realized coverage as results, in total and per stratum. On a stratified model, coverage is a fraction of all eligible people and capacity is a total, not a number per stratum ([05-dimensions.md](05-dimensions.md#coverage-and-counts-on-a-stratified-model)).

`eligible=` restricts to people; restricting to particular contacts is not needed, because a network already says who is in contact with whom.

### The `deliver=` function

When who, when, and how many depend on each other, write one function. It is called every step (within `start`/`stop` if given) and returns the UIDs of the people who receive the product on that step.

```python
def to_busiest_clinics(sim):
    hiv, ppl = sim.diseases.hiv, sim.people
    waiting  = hiv.diagnosed & ~hiv.on_art                       # diagnosed and not yet treated
    n_waiting = np.bincount(ppl.clinic[waiting], minlength=n_clinics)
    busiest  = np.argsort(n_waiting)[-100:]                      # the 100 clinics with the most
    return (waiting & np.isin(ppl.clinic, busiest)).uids

art = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))
ss.treat(art, deliver=to_busiest_clinics, capacity=ss.peryear(20_000))    # while stocks last
```

`deliver=` replaces `eligible=`, `coverage=`, and the round keywords; `capacity=` may still be given and caps what the function returns. In a compartmental run the function is tried on groups, where it must return a number of people per group; the example above uses a per-agent attribute and is agent-based.

### Follow-ups

```python
ss.screen(ss.Test(sensitivity=0.9, specificity=0.98),
          coverage=0.4, interval=ss.years(3),
          eligible=(ppl.age > 30) & ppl.female,
          on_positive=ss.treat(ss.Treatment(efficacy=0.95), coverage=0.8))
```

`on_positive=` and `on_negative=` name what happens to people with that test result. The value is another delivery, with its own coverage, eligibility, and capacity (so loss to follow-up at each step is explicit), or a function of `(sim, uids)` for anything else. A follow-up delivery can have follow-ups of its own, to any depth: screen, then triage, then confirm, then treat.

```python
confirm = ss.screen(pcr, coverage=0.9, on_positive=ss.treat(drug, coverage=0.8))
ss.screen(rapid_test, coverage=0.4, interval=ss.years(1), on_positive=confirm)
```

A follow-up happens on the step the result is available: the same step by default, later if the test has a `delay`.

### Event-triggered delivery

`on=` delivers to the people something has just happened to, instead of on a schedule:

```python
ss.screen(hiv_test, on=anc.visit, coverage=0.9, on_positive=ss.treat(art))    # test at antenatal visits
ss.treat(drug, on=tb.diagnosis, coverage=0.85)                                # treat on diagnosis
```

`on=` accepts a transition (the people who went through it this step), a boolean state or condition belonging to any module (the people for whom it is true this step), or a function of the sim returning UIDs. `eligible=`, `coverage=`, and `capacity=` then apply to those people. `on_positive=X` is the same as giving `X` an `on=` that is "this test came back positive". The worked example in [07-effects.md](07-effects.md#plain-code-in-step) shows an event-triggered delivery next to the same logic written as `step()` code.

## `ss.change`

The simplest time-varying intervention is a list of times and a list of values:

```python
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
ss.change(school.pars.n_contacts, at=[2020.2, 2020.6], by=[0.2, 1.0])
ss.change(sir.pars.beta, at=[ss.days(30), ss.days(60)], to=[0.02, 0.05])
```

- `target` is any quantity `ss.modify` accepts.
- `at=` takes dates, years, or durations since the start of the sim. A bare number is a year; a duration is written with its unit (`ss.days(30)`).
- `by=` gives multipliers relative to the baseline value (not to the previous entry); `to=` gives absolute values. Exactly one is given, with one entry per time.
- Each value holds from its time until the next; before the first time the baseline applies.

`ss.change` is a modifier ([07-effects.md](07-effects.md)) whose multiplier (or set value) is a step function of time. It combines with other modifiers by the same rule and appears in `what_modifies()`.

## Triggers

"When" is usually `start=`, `stop=`, `years=`/`dates=`, or `interval=`. Anything more complex is a function of the sim returning `True` or `False`, passed as `when=` to a delivery or to `ss.modify`:

```python
def many_cases(sim):
    new = sim.results.sir.new_infections
    return new[max(0, sim.ti - 7):sim.ti].sum() > 1000

ss.modify(school.pars.n_contacts, multiply=0.2, when=many_cases, min_duration=ss.weeks(4))
```

The function is called each step, at the intervention's place in the loop, and can read anything in the sim, including results up to the previous step. `min_duration=` keeps the intervention on for at least that long once the function first returns `True`; without it, a threshold policy switches on and off every few steps as the quantity crosses the threshold, so a threshold trigger SHOULD have one.

## Scenarios

```python
base = ss.Sim(hiv, networks=ss.MFNet(), interventions=art_program)                 # what is really in place
prep = ss.Sim(hiv, networks=ss.MFNet(), interventions=[art_program, prep_rollout])

sims = ss.parallel(baseline=base, prep=prep)
sims.plot()
sims.to_df()                # one tidy table indexed by scenario, replicate, and time
sims.diff('baseline')       # differences from the baseline, with intervals
```

- **A scenario is a sim.** It contains exactly the interventions passed to it. The baseline contains whatever is really in place (antiretroviral therapy in an HIV model, routine immunization in a measles model); it is not a run with everything switched off.
- `ss.parallel` runs the sims, in parallel, and returns them as one object keyed by the names given.
- With common random numbers ([11-random.md](11-random.md)), the scenarios share their random draws, so the difference between two scenarios is the effect of the intervention and not noise.

## How each method realizes an intervention

Conditions, coverage functions, and `deliver=` functions are tried on what the method provides ([07-effects.md](07-effects.md#on-agents-and-on-groups)): agents in an agent-based run, groups in a compartmental one. Nothing is classified in advance; if something asks for per-agent information in a compartmental run, the run stops and says what was asked for.

| Declaration | Compartmental | Agent-based |
|---|---|---|
| `ss.change`, `ss.modify` | a time- or group-varying multiplier on the quantity | a cached per-agent or scalar multiplier |
| `ss.transfer(dim=..., coverage=)` | a flow between the dimension's compartments | select eligible agents, thin to coverage, change the property |
| `ss.vaccinate`, `ss.treat` with a product | adds a "holds this product" dimension to the model (printed) and moves people along it; exponential waning is a flow back, other shapes use stages | records who holds the product and when they got it |
| `ss.screen` with follow-ups | flows thinned by sensitivity and specificity; needs the tested and result states as dimensions, added automatically | a result per agent |
| `eligible=` using stratifications and states | selects groups | selects agents |
| `eligible=` using a per-agent attribute | stops with a message naming the attribute | selects agents |
| `capacity=` | a cap on the total flow, shared across groups in proportion to the eligible | a cap on the number of agents |
| `deliver=fn(sim)` | works if the function returns numbers per group | returns UIDs |
| delivery to specific contacts (contact tracing, partner notification) | not available | available on networks that keep contacts |

## Rejected

- **Four independent parts** (where, when, to whom, what) as the organizing idea. "What" separates cleanly; the rest often depend on each other, so they are keywords for simple cases and one function otherwise.
- **A class per combination of delivery mode and product** (routine screening, campaign screening, routine vaccination, ...). One function per product type, with keywords.
- **Interventions that are off unless a scenario turns them on.** A sim contains its interventions; a baseline usually has some.
- **Interventions as extra compartments written into the disease.** A transfer along a dimension, or a product, does this without touching the disease.
- **A rule language with its own variables, sets, and operations.** Conditions and functions in Python say the same thing in a line.
- **`days=` as the time keyword of `ss.change`.** `at=` takes dates, years, or durations.
- **Budget optimization and optimal control in v1.** `cost=` keeps the door open.
