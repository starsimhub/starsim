# Interventions

What is delivered, and how it is delivered: products, delivery, follow-ups, parameter changes, triggers, and scenarios.

Justified by [best-practices/interventions.md](../best-practices/interventions.md).

## What plus delivery

An intervention has two parts.

- **What**: a product (a vaccine, a test, a treatment) or a change to a quantity. This part is cleanly separable and reusable.
- **Delivery**: who gets it, when, and how many. These are often not separable from each other ("send drugs to the 100 clinics with the most positive tests, while stocks last"), so delivery is written either with simple keywords or as one function that returns who gets it this step.

<!-- fragment -->
```python
bcg = ss.Vaccine(efficacy=0.85)                                                 # what
ss.vaccinate(bcg, coverage=0.7, at=[2021, 2025], where=ppl.age < 5)       # delivery, by keywords
ss.vaccinate(bcg, deliver=my_function)                                          # delivery, by a function
```

Interventions are passed to the sim with `interventions=`.

## Simple cases that need no product

**A change to a parameter over time** is `ss.change` ([below](#sschange)):

<!-- fragment -->
```python
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
```

**A change that applies to some people, or under a condition,** is `ss.modify` ([07-effects.md](07-effects.md)):

<!-- fragment -->
```python
ss.modify('school.pars.n_contacts', multiply=0.2, start='2020-03-15', stop='2020-06-01')
ss.modify(sir.rel_sus, multiply=0.6, where=sir.has_net)
```

**Vaccination as a move between strata** is `ss.transfer` ([05-dimensions.md](05-dimensions.md#moving-between-coordinates-sstransfer)). The disease does not mention vaccination; protection is declared on the dimension and delivery is a transfer:

<!-- fragment -->
```python
sir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.15]})    # 85% protection, declared once
rollout = ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021, 2022], where=ppl.age > 65)
```

## Products

```python
ss.Vaccine(efficacy=0.85, waning=ss.years(5), cost=12.50)
ss.Vaccine(efficacy=dict(acquisition=0.85, transmission=0.3, severity=0.5))
ss.Vaccine(efficacy=0.85, leaky=False)                                          # all or nothing: 85% fully protected, the rest not at all
ss.Test(sensitivity=0.9, specificity=0.98, cost=4.00, delay=ss.days(2))
ss.Treatment(cure=0.95, cost=120.0)
```

They replace Starsim 3's `ss.Vx`, `ss.Dx`, and `ss.Tx`, which remain as base classes for products written as code; called without the arguments of Starsim 3's classes (`df=`, `diseases=`), they make an `ss.Vaccine`, `ss.Test`, or `ss.Treatment`. Starsim 3's delivery classes (`ss.routine_vx`, `ss.campaign_vx`, `ss.routine_screening`, and the rest) are removed; their spellings here are in the migration guide.

**Efficacy.** A scalar `efficacy` on a vaccine means protection against acquisition, and is required. A dict gives efficacy against several things at once. Its keys are `acquisition` (scales `rel_sus`, which acts on every acquisition, including one by a bite rate or an EIR: [03-states-and-transitions.md](03-states-and-transitions.md#what-counts-as-acquisition)), `transmission` (scales `rel_trans`), `mortality` (scales the disease's death transitions), the name of any transition in the disease (`severity` in a model whose transition to severe disease has that name), or one destination of a transition with several (`'progression.C'`, which scales the chance of going to `C`, not the rate of leaving); an unknown key raises `UnknownEffect`, listing the valid ones. Each entry multiplies its quantity by `1 − efficacy` for the people who hold the product. On a transition chosen with `p=` (one of several branches), it scales that branch's probability, and the other branches share the rest. With `leaky=False` the vaccine is all or nothing: a share `efficacy` of those who receive it are fully protected and the rest not at all.

**Holding a product.** Whether someone holds a product is a property of people named `has_<product>` (`has_bcg`); the product's own name is taken by the product module. A delivery never gives a product to someone who already holds it.

**Waning.** `efficacy` is the effect at full strength and `waning` is how it fades. A duration means the product is lost at an exponentially distributed time with that mean, under every solver, after which the person can receive it again. A function of the time since delivery, returning the fraction of protection that remains, fades the efficacy while the person still holds the product (in a compartmental run, people hold it in levels of time since delivery, so `has_<product>` has more than two levels).

**A product is realized as modifiers** on the people who hold it, so it follows the combination rules in [07-effects.md](07-effects.md): two vaccines against acquisition multiply, a vaccine and a stratification adjustment multiply, and all of them are listed by `sim.what_modifies()`.

**Tests.** `sensitivity` and `specificity` are spelled out. A test returns a result per person, which is what follow-ups branch on. By default it detects infection with the disease it is delivered for (the states the model treats as infected, printed by `describe()`); `detects=` takes a [condition](07-effects.md#conditions) to say something else (a function of the sim and UIDs makes the sim agent-based). `sensitivity=` may be a dict with one value for each state, each a number or an `ss.by` (`sensitivity=dict(symptomatic=0.91, asymptomatic=ss.by(age=[0.73, 0.78]))`), and the test then detects being in those states. `delay=` is the time from test to result; in a compartmental run, people wait for their result in levels of one step each.

**Treatments.** `cure=` is the probability that treatment clears the infection, once, when it is given: the person leaves by the disease's recovery transition (`transition=` names it when there are several), from that transition's own states, so a treatment can start from any state (from first-line to second-line treatment, say). The people it moves count towards that transition's flow, and a delivery `on=` that transition includes them from the next step. `efficacy=`, a dict as for vaccines, is what it does while it is held, such as antiretroviral therapy (`efficacy=dict(transmission=0.96, mortality=0.9)`). The two can be given together (`ss.Treatment(cure=0.3, efficacy=dict(transmission=0.9))`); a number for `efficacy=` raises, pointing to `cure=`. With neither, `cure` is 1. Either way a treatment is given to the people who are infected (the states acquisitions lead to and the sources of infection, [03-states-and-transitions.md](03-states-and-transitions.md#what-counts-as-acquisition), so in a malaria model infected by a bite rate, `Eh` and `Ih`) unless `where=` says otherwise.

**Cost.** `cost=` is per delivery. Total cost is then a result of the intervention, which is what cost-effectiveness needs. Allocating a budget between programs is `ss.optimize(sim, maximize=..., at=...)`, which reallocates spending parameters (the programs of an imported Atomica program book, or any parameters named with `spend=`) to maximize or minimize a result, keeping the total; a program declared in Python, rather than imported, is not built ([`deferred.md`](../admin/deferred.md)).

## Delivery

<!-- fragment -->
```python
ss.vaccinate(product, **delivery)
ss.screen(test, on_positive=None, on_negative=None, **delivery)
ss.treat(treatment, **delivery)
```

### Delivery keywords

| Keyword | Meaning |
|---|---|
| `start=`, `stop=` | the period of delivery (dates, durations since the start of the sim, or numbers in the sim's unit, as for `at=` of [`ss.change`](#sschange)) |
| `at=` | deliver in rounds at these times (a campaign), given as for `at=` of `ss.change`: in a sim with dates, a calendar year is written as a date (`'2030'`) |
| `interval=` | deliver in rounds this far apart, between `start` and `stop` |
| `where=` | who can receive it: a [condition](07-effects.md#conditions), as for `where=` of `ss.modify` |
| `coverage=` | the fraction of eligible people reached |
| `capacity=` | the number that can be delivered |
| `on=` | an event in another module that [triggers delivery](#event-triggered-delivery), such as `on=anc.visit` |
| `when=` | a [trigger](#triggers): deliver only on steps where a function of the sim returns `True` |
| `at_age=` | routine delivery to people as they reach an age, e.g. `at_age=ss.months(9)` (a bare number is in years) |
| `move=` | instead of a product, a move between two states for the people reached, e.g. `move='S -> V'` (a one-off campaign); with `capacity=`, it seeds a number of people on a schedule |
| `trace=` | deliver to the contacts, in this network (or these), of the people that `on=` selects or that a test sends to this follow-up, instead of to them (contact tracing, partner notification). Agents only |
| `deliver=` | a function of the sim returning who receives it this step ([below](#the-deliver-function)) |
| `disease=` | the disease the product acts on (default: the only one) |

<!-- fragment -->
```python
ss.vaccinate(bcg, coverage=0.7, at=[2021, 2025], where=ppl.age < 5)                # two campaign rounds
ss.vaccinate(bcg, coverage=0.7, interval=ss.years(1), start=2021, where=ppl.age < 5)  # a yearly round
ss.vaccinate(bcg, coverage=0.9, start=2021, at_age=ss.months(9))                        # routine: each child as they reach 9 months
```

**Rounds and routine delivery.** With `at=` or `interval=`, delivery happens in rounds, and in each round every eligible person is offered the product once. With only `start=`/`stop=` (or nothing), delivery is routine: each person is offered the product once, at the first step on which they are eligible, and the record that they were offered it is a property of people named `offered_<delivery>`. Routine versus campaign is therefore a keyword, not a class. A routine delivery by age is given as people reach the age: on its first step, the people already past it (eligible a step earlier, and not eligible as newborns) are recorded as offered it, so `at_age=ss.months(9)` (the same as `where=ppl.age >= 0.75`) vaccinates children as they turn 9 months and not everyone older on the first day, while `where=ppl.age < 5` reaches every child under 5. A catch-up of those already older is a campaign round.

**`coverage=` versus `capacity=`.** These are different specifications and both are common.

- `coverage=0.7` is a fraction of the eligible people. It accepts everything a parameter accepts, including `ss.by(age=[...])`, a time series, and a function of the sim returning one probability per person.
- `capacity=50_000` is a number available: a bare number is per round for campaigns (or per step for delivery on events or by a function), and a rate (`ss.peryear(50_000)`) is per unit time. Routine delivery has no rounds, so a bare number there raises, asking for a rate. It may be a time series. When fewer are available than eligible people who would receive one, recipients are chosen at random among them, and delivery stops when the capacity is used up ("while stocks last").
- Given together, coverage is applied first and capacity caps the result.

Each delivery reports the number delivered and the realized coverage as results, in total and per stratum. On a stratified model, coverage is a fraction of all eligible people and capacity is a total, not a number per stratum ([05-dimensions.md](05-dimensions.md#coverage-and-counts-on-a-stratified-model)).

`where=` restricts to people; restricting to particular contacts is not needed, because a network already says who is in contact with whom.

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

`deliver=` replaces `where=`, `coverage=`, and the round keywords; `capacity=` may still be given and caps what the function returns. In a compartmental run the function is tried on groups, where it must return a number of people per group; the example above uses a per-agent attribute and is agent-based.

### Follow-ups

<!-- fragment -->
```python
ss.screen(ss.Test(sensitivity=0.9, specificity=0.98),
          coverage=0.4, interval=ss.years(3),
          where=(ppl.age > 30) & ppl.female,
          on_positive=ss.treat(ss.Treatment(cure=0.95), coverage=0.8))
```

`on_positive=` and `on_negative=` name what happens to people with that test result. The value is another delivery, with its own coverage, eligibility, and capacity (so loss to follow-up at each step is explicit), or a function of `(sim, uids)` for anything else. A follow-up happens when the result arrives, so it cannot have a schedule of its own (`years=`, `interval=`, ...), which raises. A follow-up delivery can have follow-ups of its own, to any depth: screen, then triage, then confirm, then treat. In a compartmental run, when a test has follow-ups for both results, each applies only to the people with its result.

<!-- fragment -->
```python
confirm = ss.screen(pcr, coverage=0.9, on_positive=ss.treat(drug, coverage=0.8))
ss.screen(rapid_test, coverage=0.4, interval=ss.years(1), on_positive=confirm)
```

A follow-up happens on the step the result is available: the same step by default, later if the test has a `delay`. Someone waiting for a result is not given the same test again, so a test repeated more often than its delay still returns results (`describe()` says so).

### Event-triggered delivery

`on=` delivers to the people something has just happened to, instead of on a schedule:

<!-- fragment -->
```python
ss.screen(hiv_test, on=anc.visit, coverage=0.9, on_positive=ss.treat(art))    # test at antenatal visits
ss.treat(drug, on=tb.diagnosis, coverage=0.85)                                # treat on diagnosis
```

`on=` accepts a transition (the people who went through it this step), or a function of the sim returning UIDs. A state or condition is not an event, and raises an error that points to `where=`: the people in a state on each step are `where=sir.I, interval=1`. `where=`, `coverage=`, and `capacity=` then apply to those people. `on_positive=X` is the same as giving `X` an `on=` that is "this test came back positive". The worked example in [07-effects.md](07-effects.md#plain-code-in-step) shows an event-triggered delivery next to the same logic written as `step()` code.

### What a delivery reports

Every delivery is a module with results of its own, in total and selectable by stratum like any other result ([09-results.md](09-results.md)): the number delivered each step (`delivered`), the number eligible (`eligible`), the realized coverage (`coverage`, which for several groups is what was delivered over who was eligible in them), and the cost if the product has `cost=` (`cost`).

An intervention that tests also reports what it found. `ss.screen` has a `diagnosed` result: the number of positive results each step, counted on the step the result is available (so after the test's `delay`). For an intervention named `screening` it is `sim.results.screening.diagnosed`. These are flows, so they can be summed, scaled, and compared with data by name.

**Reported cases, when testing is modeled.** `ss.outcome` ([09-results.md](09-results.md)) is a shortcut for a reporting process the model does not represent: "about half of infections are reported, five days later". When the model does represent testing, through one or more interventions, the outcome is not used for cases. Each testing intervention reports the diagnoses it made, and reported cases are their sum:

<!-- fragment -->
```python
symptomatic_testing = ss.screen(pcr, on=sir.symptom_onset, coverage=0.6)
screening           = ss.screen(rapid_test, coverage=0.1, interval=ss.weeks(1))

cases = symptomatic_testing.diagnosed + screening.diagnosed
```

So an outcome and a testing intervention never both produce the same cases: if the model has tests, the tests produce the cases. The sum is a result like any other sum of flows, and it is what the `cases` column of a data file is matched to. It belongs to neither intervention, and whether it is declared on the sim or on the disease that both test for is open ([19-open-questions.md](19-open-questions.md)).

The choice between the three ways of representing an observed quantity:

| The model contains | Use |
|---|---|
| No testing or reporting process | `ss.outcome`, a weighted sum of flows with a delay |
| Testing, as one or more interventions | the interventions' own `diagnosed` results, summed |
| Something that changes what happens to the person (a diagnosed person is treated, or isolates) | a state, set by the intervention (`on_positive=`) |

An intervention marks individuals (a positive result leads to a follow-up, or sets `hiv.diagnosed`); an outcome never does, because it never recorded who they were.

## `ss.change`

The simplest time-varying intervention is a list of times and a list of values:

<!-- fragment -->
```python
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
ss.change('school.pars.n_contacts', at=[2020.2, 2020.6], by=[0.2, 1.0])
ss.change(sir.pars.beta, at=[ss.days(30), ss.days(60)], to=[0.02, 0.05])
```

- `target` is any quantity `ss.modify` accepts.
- `at=` takes dates, durations since the start of the sim (`ss.days(30)`), or numbers, which are times in the sim's unit: in a sim with dates, that long after the start (so a calendar year is written as a date, `'2030'`); in a sim whose `start` and `stop` are years (`start=2020`), the year. A time outside the sim raises `TimeOutsideSim`; so does a `start=` or a round of a delivery, while a `stop=` after the end means it never stops. `start=` is not accepted: the change starts at its first time.
- `by=` gives multipliers relative to the baseline value (not to the previous entry); `to=` gives absolute values. Exactly one is given, with one entry per time.
- Each value holds from its time until the next; before the first time the baseline applies.

`ss.change` returns a modifier ([07-effects.md](07-effects.md)), an `ss.modify` whose multiplier (or set value) is a step function of time, starting at its first time (before it, the quantity has its own value). It combines with other modifiers by the same rule, appears in `what_modifies()`, and takes `stop=`, `when=`, and `where=` like any modifier.

## Triggers

"When" is usually `start=`, `stop=`, `at=`, or `interval=`: the same keywords for a delivery, `ss.change`, and `ss.modify`. Anything more complex is a function of the sim returning `True` or `False`, passed as `when=` to a delivery or to `ss.modify`:

```python
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01)
home, school = ss.RandomNet(n_contacts=3, name='home'), ss.RandomNet(n_contacts=10, name='school')

def many_cases(sim):
    new = sim.results.sir.infection
    return new[max(0, sim.ti - 6):sim.ti + 1].sum() > 100  # the last seven completed steps

closures = ss.modify('school.pars.n_contacts', multiply=0.2, when=many_cases)
sim = ss.Sim(sir, networks=[home, school], interventions=closures, n_agents=2000, stop=ss.days(100)).run()
```

- The function is called each step, at the intervention's place in the loop, and can read anything in the sim, including results up to the previous step. Results are read as in Starsim: storage is preallocated for the whole run, and during a step `sim.ti` is the time point at its start, the last one with results, so the results so far are `[:sim.ti + 1]` ([10-execution.md](10-execution.md)). A shorter spelling for "the steps so far" would be a convenience and is not specified ([19-open-questions.md](19-open-questions.md)).
- It can read any result, including an outcome ([09-results.md](09-results.md)): `when=lambda sim: sim.results.sir.n_hosp[sim.ti] > 500` acts on hospital occupancy without hospitalization being a state.
- There is no built-in vocabulary for calendar conditions. "No school contacts on Fridays" is `ss.modify('school.pars.n_contacts', multiply=lambda sim: 0.0 if sim.now.weekday() == 4 else 1.0)`, in a sim with dates.

**`when=` and `where=` have no memory, except a hold.** The intervention is active on exactly the steps where the function returns `True`. A bare threshold therefore switches on and off as the quantity crosses it, which is rarely the policy meant. The one memory a modifier has is `hold=`: once its condition holds for someone, it stays on for them for at least that long (`hold=ss.days(14)`), which with a condition on a transition's incidence by place is MEmilio's dynamic NPIs, each place on its own (`where=secir.symptoms.incidence(over=ss.days(7), by='patch') > 200e-5`). Any other rule that remembers what it did (reopen only after two weeks below the threshold, close at most three times) is a module with a `step()`, where the memory is an ordinary attribute and can be read, printed, and tested:

```python
class SchoolClosure(ss.Intervention):
    def __init__(self, threshold=100, hold=ss.weeks(4), **kwargs):
        super().__init__(**kwargs)
        self.define_pars(threshold=threshold, hold=hold)
        self.closed      = False
        self.date_closed = None

    def step(self):
        sim = self.sim
        new = sim.results.sir.infection
        many = new[max(0, sim.ti - 6):sim.ti + 1].sum() > self.pars.threshold
        if many and not self.closed:
            self.closed, self.date_closed = True, sim.now
        elif self.closed and not many and sim.now - self.date_closed >= self.pars.hold:
            self.closed = False

closure = SchoolClosure(name='closure')
effect  = ss.modify('school.pars.n_contacts', multiply=0.2, when=lambda sim: sim.interventions.closure.closed)
sim     = ss.Sim(sir, networks=[home, school], interventions=[closure, effect], n_agents=2000, stop=ss.days(100)).run()
```

The class decides and remembers; the modifier applies the effect, so the base contact rate is never overwritten and comes back when `closed` is cleared ([07-effects.md](07-effects.md#plain-code-in-step)). Whether `when=` should exist at all, given how little it can say safely, is open ([19-open-questions.md](19-open-questions.md)).

## Scenarios

<!-- fragment -->
```python
base = ss.Sim(hiv, networks=ss.MFNet(), interventions=art_program)                 # what is really in place
prep = ss.Sim(hiv, networks=ss.MFNet(), interventions=[art_program, prep_rollout])

sims = ss.parallel(baseline=base, prep=prep, n_runs=10)
sims.plot()
sims.to_df()                # one tidy table: quantiles across the runs, by scenario and time
sims.diff('baseline')       # differences from the baseline, run by run, by scenario, module, and result
```

- **A scenario is a sim.** It contains exactly the interventions passed to it. The baseline contains whatever is really in place (antiretroviral therapy in an HIV model, routine immunization in a measles model); it is not a run with everything switched off.
- `ss.parallel` runs the sims, in parallel, and returns them as one MultiSim whose results have a run axis, selectable by scenario (`select(scenario='prep')`). With `n_runs=` each scenario is run several times ([09-results.md](09-results.md)).
- With common random numbers ([11-random.md](11-random.md)), the scenarios share their random draws, so the difference between two scenarios is the effect of the intervention and not noise.

## How each solver realizes an intervention

Conditions, coverage functions, and `deliver=` functions are tried on what the solver provides ([07-effects.md](07-effects.md#on-agents-and-on-groups)): agents in an agent-based run, groups in a compartmental one. Nothing is classified in advance; under `solver='auto'`, something that asks for per-agent information makes the sim agent-based, and under an explicit compartmental solver the run stops and says what was asked for. A delivery at time point `ti` happens between the steps and is counted there. Deliveries on a schedule (routine or in rounds), to people chosen by states and stratifications, of a product people hold or a transfer, are taken by the kernels in a whole compartmental run; the others (tests and their follow-ups, cures, `capacity=`, `when=`) step the run from Python.

| Declaration | Compartmental | Agent-based |
|---|---|---|
| `ss.change`, `ss.modify` | a time- or group-varying multiplier on the quantity | a per-agent or scalar multiplier |
| `ss.Delivery(move=dict(dim=...), coverage=)` | a flow between the dimension's compartments | select eligible agents, thin to coverage, change the property |
| `ss.vaccinate`, `ss.treat` with a product | adds a `has_<product>` dimension to the model (printed) and moves people along it; waning given as a duration is a flow back, and a function of time since delivery uses levels of time since delivery | records who holds the product and when they got it |
| `ss.screen` with follow-ups | flows thinned by sensitivity and specificity; with `delay=`, people wait for their result in levels of one step | a result per agent |
| `when=fn(sim)` | called once per step | called once per step |
| a module with `step()` (a trigger with memory) | works if the code uses only what groups have | works |
| `where=` using stratifications and states | selects groups | selects agents |
| `where=` using a per-agent attribute | stops with a message naming the attribute | selects agents |
| `capacity=` | a cap on the total flow, shared across groups in proportion to the eligible | a cap on the number of agents |
| `deliver=fn(sim)` | works if the function returns numbers per group | returns UIDs |
| `trace=`: delivery to specific contacts (contact tracing, partner notification) | not available | available on networks that keep contacts |

## Rejected

- **Four independent parts** (where, when, to whom, what) as the organizing idea. "What" separates cleanly; the rest often depend on each other, so they are keywords for simple cases and one function otherwise.
- **A class per combination of delivery mode and product** (routine screening, campaign screening, routine vaccination, ...). One function per product type, with keywords.
- **Interventions that are off unless a scenario turns them on.** A sim contains its interventions; a baseline usually has some.
- **Interventions as extra compartments written into the disease.** A transfer along a dimension, or a product, does this without touching the disease.
- **A rule language with its own variables, sets, and operations.** Conditions and functions in Python say the same thing in a line.
- **`min_duration=` on a trigger.** `hold=` is that minimum; other rules with memory are a `step()`.
- **An outcome and a testing intervention both producing cases.** If the model has tests, the tests produce the cases.
- **`days=` as the time keyword of `ss.change`.** `at=` takes dates, years, or durations.
- **Budget optimization and optimal control in v1.** `cost=` keeps the door open. (Allocating a fixed budget between imported programs, `ss.optimize`, was added for Atomica's users; optimal control was not.)
