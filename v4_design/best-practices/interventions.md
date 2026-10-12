# Interventions

## Recommendation

1. **An intervention is *what* plus *delivery*.** What is a product (a vaccine, a test, a treatment) or a change to a quantity. Delivery is who gets it, when, and how many. The "what" separates cleanly; who, when, where, and how many often do not separate from each other, so delivery is either a few simple keywords or one function.
2. **Eligibility is written against the model's real attributes.** The general form is a function of the sim, as in Starsim today. The short form is the same expression written directly on the people and disease objects, which is evaluated later when the sim runs. No separate vocabulary of predicate names.
3. **Keep Starsim's product/delivery split**, and separate efficacy from waning as EMOD does.
4. **Many interventions are an `ss.transfer` between strata or an `ss.modify` on a named quantity.** Do not build a class for what is already one of those.
5. **The simple case stays a one-liner.** Covasim's `days`/`changes` form was simpler than what replaced it, and it comes back as `ss.change(..., at=, by=)`.
6. **Coverage as a fraction and capacity as a number are different things**, and both are needed: `coverage=0.7` and `capacity=50_000`.
7. **Interventions carry a cost.** Nothing else in the review except Atomica does this, and it is what the policy audience needs.
8. **Scenarios live in the model file**, not in a shell script, and a baseline contains whatever is really in place.

## Why

### The split that runs through the whole review

Finding 1 of the [landscape review](../approaches/README.md#cross-cutting-findings): "Structural rigor and intervention expressiveness have not yet appeared in the same framework." [camdl](../approaches/camdl.md), [odin](../approaches/odin.md), [summer2](../approaches/summer2.md), and [epipack](../approaches/epipack.md) have the best model specification and essentially no intervention vocabulary. [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [Starsim](../approaches/starsim.md), and [Atomica](../approaches/atomica.md) have the best intervention vocabulary and weak structural specification.

Closing that split is, per the review, "the most concrete contribution available to this project."

### What separates and what does not

[EMOD](../approaches/emod.md) decomposes a campaign event into four pieces:

| Part | Question | EMOD |
|---|---|---|
| Where | Spatial scope | `Nodeset_Config` |
| When, how many, how often | Schedule and coverage | `Event_Coordinator_Config` |
| To whom | Eligibility | `Targeting_Config` |
| What | The product | `Intervention_Config` + `Waning_Config` |

This is the most complete treatment in the review and it is a useful checklist. As a design it overstates how independent the pieces are. Consider a realistic policy: *distribute treatment drugs to the 100 clinics with the most positive cases, while stocks last.* Where, when, how many, and to whom are all functions of who tested positive. Only "what" (the drug) can be swapped without touching the rest.

So the lingua franca keeps two parts. Starsim already has this split (`ss.Product` and the `RoutineDelivery` / `CampaignDelivery` mixins), and its review says "the product/delivery split is the right decomposition and should survive into the design."

### Eligibility in existing frameworks

[Starsim](../approaches/starsim.md) takes eligibility as a function of the sim that returns who is eligible. That is fully general, works with any custom attribute, and is the form to keep. Its limits, per its review: "it is Python, so it cannot be serialized, diagrammed, or checked."

[EpiModel](../approaches/epimodel.md)'s tabular scenario grammar, `{scenario, param, at, value}`, covers a large fraction of real specifications without any code, and its limit is that "it can only set parameter values, so anything that targets a *subset of agents* (screen 40% of over-50s annually) has to fall back to procedural code." [flepiMoP](../approaches/flepimop.md)'s modifiers have the same shape and the same ceiling.

Two frameworks have eligibility as data. **EMOD:**

```json
"Targeting_Config": {
  "class": "TargetingLogic",
  "Logic": [ [ {"class": "HasIntervention", "Is_Equal_To": 0, "Intervention_Name": "MyVaccine"} ],
             [ {"class": "HasIP", "Is_Equal_To": 1, "IP_Key_Value": "Risk:HIGH"} ] ]
}
```

**EpiHiper** has set algebra over typed node and edge predicates, "the most expressive targeting mechanism in the review", at the cost of a separate rule language. And [`individual`](../approaches/individual.md) shows the same algebra in code, with bitsets:

```r
I <- health$get_index_of("I")
I$and(already_scheduled$not(inplace = TRUE))
```

The aim is to get eligibility that reads like the `individual` version, can be saved like the EMOD version when it is simple, and falls back to a plain function when it is not.

## The proposal

### What and delivery

```python
ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=...)
#            what ─┘  └──────────── delivery ─────────────────┘
```

Delivery has two forms. The **simple keywords** cover most programs:

| Keyword | Meaning |
|---|---|
| `start=`, `stop=` | When delivery begins and ends (dates or years) |
| `years=` / `dates=` | Specific times (a campaign) |
| `interval=` | How often (routine delivery) |
| `eligible=` | Who can receive it |
| `coverage=` | Fraction of the eligible who receive it |
| `capacity=` | Number available, per round or per unit time |
| `on=` | An event in another module that triggers it (e.g. `on=anc.visit`) |

When who, when, where, and how many depend on each other, write **one function** that returns who gets it this step:

```python
def top_clinics(sim):
    """ People awaiting treatment at the 100 clinics with the most positives """
    hiv, ppl = sim.diseases.hiv, sim.people
    waiting  = (hiv.diagnosed & ~hiv.on_art).uids
    counts   = np.bincount(ppl.clinic[waiting], minlength=n_clinics)
    top      = np.argsort(counts)[-100:]
    return waiting[np.isin(ppl.clinic[waiting], top)]

ss.treat(art, deliver=top_clinics, capacity=ss.timeseries('art_stock.csv'))   # while stocks last
```

`deliver=` replaces the who/when keywords; `capacity=` still applies on top, so "while stocks last" does not have to be written by hand.

### Eligibility

**The general form** is a function of the sim. This is Starsim's current form, and it works with anything, including attributes the modeler added:

```python
eligible = lambda sim: (sim.people.age > 65) & ~sim.diseases.sir.vaccinated
```

**The short form** drops the `lambda sim:` when the objects are already in scope:

```python
ppl = ss.People(n_agents=10_000)
sir = ss.Disease(...)

eligible = (ppl.age > 65) & ~sir.vaccinated
```

This is the EMOD `Targeting_Config` above in one line. It works because `ppl.age` and `sir.vaccinated` are state arrays that have not been initialized yet (the sim has not been built). Comparing or combining an uninitialized state array does not compute anything; it builds a condition that remembers the expression and is evaluated each time it is needed once the sim is running. After initialization the same expression evaluates immediately, as now. So `~sir.vaccinated` written before the run means "whoever is unvaccinated at the time", not "whoever is unvaccinated now".

There is nothing new to learn: the names are the attributes that already exist on the objects, a custom attribute works the same way as a built-in one (`ppl.clinic == 12`), and the two forms can be mixed. Because the short form is recorded as an expression, it can be printed by `describe()` and saved in the model spec, which a function cannot.

**In a compartmental run** the same condition is evaluated on groups. It works if everything it uses is a stratification (age groups, sex, vaccination status), and stops with a clear message otherwise:

```text
Cannot evaluate eligibility for 'bcg_campaign' with method='ode':
  ppl.clinic is a per-agent attribute and is not a stratification.
  Fix: stratify by clinic, or run with method='abm'.
```

**When** is `start=`, `stop=`, `years=`/`dates=`, and `interval=`. Anything more complex is a function of the sim, with `min_duration=` so that a policy does not switch on and off every step:

```python
ss.modify(school.pars.n_contacts, multiply=0.2,
          when=lambda sim: sim.results.sir.new_infections[-7:].sum() > 1000,
          min_duration=ss.weeks(4))
```

This covers [EMOD](../approaches/emod.md)'s event broadcast/observe and [EpiHiper](../approaches/epihiper.md)'s triggers ("close schools when weekly cases exceed 1000"). There is no built-in vocabulary for calendar conditions such as weekdays. EpiHiper's "no school contacts on Fridays" (roughly 200 lines of JSON across variables, sets, triggers, and interventions) is a function:

```python
ss.modify(school.pars.n_contacts, multiply=lambda sim: 0.0 if sim.now.weekday() == 4 else 1.0)
```

### Many interventions are already primitives

Before adding an intervention class, check whether it is one of the two primitives that already exist.

**Vaccination can be a transfer between strata** ([stratification.md](stratification.md)). [Epydemix](../approaches/epydemix.md) and [MEmilio](../approaches/memilio.md)'s `secirvvs` add vaccinated copies of the compartments by hand; here it is one line:

```python
sir_vax = sir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.15]})   # 85% protection, declared once
ss.transfer(vaccinated=True, rate=ss.peryear(0.3))
```

For a boolean, write the value people move to. The disease model never mentions vaccination; the vaccinated stratum's protection is declared once on the stratification as a multiplier on the infection hazard, following [EpiHiper](../approaches/epihiper.md)'s state properties. Use `ss.vaccinate` (below) when there is a product with efficacy, waning, and cost, or a campaign with coverage.

**A contact reduction is a modifier** ([composition-and-effects.md](composition-and-effects.md)):

```python
ss.modify(school.pars.n_contacts, multiply=0.2, start='2020-03-15', stop='2020-06-01')
```

which is [Epydemix](../approaches/epydemix.md)'s `add_intervention(layer_name="school", start_date=..., reduction_factor=0.2)` and [MEmilio](../approaches/memilio.md)'s `contact_matrix.add_damping(0.6, SimulationTime(12.5))`, with the network and the dates visible. When it stops, the contact rate returns to its base value.

**A parameter change is a modifier too**, which covers [EpiModel](../approaches/epimodel.md)'s scenario table and [flepiMoP](../approaches/flepimop.md)'s modifiers:

```python
ss.modify(sir.pars.beta, multiply=0.4, start='2020-03-15')
```

### The simple form

From the [Covasim note](../approaches/notes.md#covasim--hpvsim--stisim--fpsim):

> `change_beta(days=[30, 60], changes=[0.5, 1.0])` — a list of days and a list of changes — is a smaller, more legible time-varying-parameter grammar than the class lattice that replaced it. flepiMoP's modifiers and EpiModel's scenario tables are the same shape. **Simplicity here was arguably lost in the generalization.**

It comes back, for any parameter and any time unit:

```python
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
ss.change(sir.pars.beta, at=[ss.days(30), ss.days(60)],   by=[0.5, 1.0])   # the Covasim example
ss.change(hiv.pars.condom_use, at=[2010, 2020], to=[0.3, 0.6])
```

`at=` takes dates, years, or durations since the start of the sim. `by=` gives multipliers relative to the baseline value; `to=` gives absolute values. Three frameworks converged on this shape. It should be one line, and it should be the first thing in the documentation on interventions.

### Products

A product is the thing delivered: a vaccine, a diagnostic, a treatment. [EMOD](../approaches/emod.md) keeps efficacy and its decay as separate objects (`Vaccine_Take`, `Vaccine_Type` as acquisition-, transmission-, or mortality-blocking, and a `Waning_Config` that may be constant, exponential, or a piecewise map). Starsim's `ss.Vx`, `ss.Dx`, and `ss.Tx` are the same things under shorter names.

```python
bcg = ss.Vaccine(
    efficacy = dict(acquisition=0.85, severity=0.5),   # keys: acquisition, transmission, mortality, or a transition name (here `severity`)
    waning   = ss.years(5),                            # a duration means exponential decay with that mean; a function or series gives any shape
    cost     = 12.50,
)

ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=ppl.age < 5)
```

A vaccine usually protects against more than one thing, so `efficacy` is a dict keyed by what is blocked (`acquisition`, `transmission`, `severity`, `mortality`). A single number, `efficacy=0.85`, means acquisition.

### Delivery

Starsim has `RoutineDelivery` (start, end, coverage, interval) and `CampaignDelivery` (specific years, coverage) as mixin classes, combined with each kind of intervention to give `routine_vx`, `campaign_vx`, `routine_screening`, `campaign_screening`, and so on. Here they are keyword arguments, because `years=[2021, 2025]` versus `interval=ss.years(1)` already says which one it is:

```python
ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025])                               # campaign
ss.vaccinate(bcg, coverage=0.9, start=2021, stop=2030, eligible=ppl.age < 1)      # routine
```

**Coverage and capacity.** `coverage=0.7` is a fraction of the eligible population. `capacity=50_000` is a number of doses, tests, or treatment slots, per round for a campaign or per unit time for routine delivery, and it may be a time series. This is [EMOD](../approaches/emod.md)'s `NChooser` (target a number, not a fraction) and it is how real programs are usually constrained. Both can be given: reach 70% of those eligible, but no more than 50,000.

```python
ss.vaccinate(bcg, coverage=0.7, capacity=50_000, years=[2021, 2025])
```

**Follow-ups.** Diagnostics and cascades follow HPVsim's screen → triage → treat pattern. **Starsim today** connects the steps by passing each one an eligibility function that looks up the previous one's outcomes:

```python
screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']
triage = ss.routine_triage(product=my_triage, eligibility=screened_pos, prob=0.9, start_year=2030)
```

**Lingua franca:** a delivery names what happens next, and the follow-up is itself a delivery, so it can have its own coverage and its own follow-up, to any depth:

```python
ss.screen(
    ss.Test(sensitivity=0.9, specificity=0.98),
    coverage    = 0.4,
    interval    = ss.years(3),
    eligible    = (ppl.age > 30) & ppl.female,
    on_positive = ss.screen(                       # triage
        colposcopy,
        coverage    = 0.8,                         # 20% lost to follow-up
        on_positive = ss.treat(ss.Treatment(efficacy=0.95), coverage=0.9),
    ),
)
```

**Triggered by another module.** A delivery can happen when something happens elsewhere in the model, with `on=`:

```python
ss.screen(hiv_test, on=anc.visit, coverage=0.95, on_positive=ss.treat(art))
```

For logic that does not fit these keywords, write a module with a `step()`; see the worked example in [composition-and-effects.md](composition-and-effects.md#complex-interactions-are-a-step).

### Cost

Only [Atomica](../approaches/atomica.md) represents an intervention as something with a unit cost and a coverage function that a budget constrains, which is the representation a health ministry needs. Its `Targetable` column declares which parameters a program may move.

`cost=` on the product is the whole requirement for v1: it makes cost-effectiveness a result, not a post-processing script, and it leaves room for [PyRoss](../approaches/notes.md#pyross)-style optimal control and Atomica-style budget optimization later without committing to either now.

### Scenarios in the file

[flepiMoP](../approaches/flepimop.md) declares `scenarios: [Ro_lockdown, Ro_all]` in the config; camdl has named scenarios paired against baseline with common random numbers; [`epidemics`](../approaches/epidemics.md) returns a scenario grid as one tidy frame with scenario identifiers.

```python
sims = ss.parallel(
    baseline = ss.Sim(hiv, interventions=art),
    prep     = ss.Sim(hiv, interventions=[art, prep]),
    scale_up = ss.Sim(hiv, interventions=[art_90, prep]),
)
sims.plot()
```

(Starsim's `ss.parallel` takes sims positionally today; the keyword labels are a proposed addition.) A baseline is the world as it is, so it contains whatever is really in place (ART, for HIV). Each scenario is a complete sim that says exactly what it contains; nothing is switched on or off implicitly. Because [common random numbers](stochasticity-and-reproducibility.md) are on, the difference between two scenarios is the effect of the intervention and not noise.

## Trade-offs

- **Conditions that evaluate later are an implementation cost, and a possible source of confusion.** The same expression builds a condition before the sim is initialized and returns an array after. The rule is simple to state, but error messages have to be good, and `describe()` should show each condition as written.
- **A function cannot be saved or drawn.** Eligibility or delivery written as a function is printed by name in `describe()` and the model spec records that a function was used, not what it does. This is accepted: the function is the general form and must always work.
- **A condition on a per-agent attribute has no compartmental meaning** unless the attribute is a stratification. `ppl.age > 65` works in both if the model is stratified by age with a boundary at 65; `ppl.clinic == 12` works only with agents. The run stops and names the attribute.
- **`deliver=fn` hides structure.** A delivery function can do anything, so two interventions using one cannot be compared keyword by keyword. Use the keywords when they fit.
- **Products are a vocabulary that will grow.** Vaccines, tests, treatments, bednets, condoms, contact tracing. Most are an `ss.modify` on a named quantity with a coverage and an eligibility, and only the ones with distinct structure (a diagnostic returns a *result*) need a class.

## Rejected

- **Four independent parts** (EMOD's decomposition taken literally). Where, when, how many, and to whom depend on each other in real programs. What + delivery.
- **A vocabulary of global predicate names** (`ss.age > 65`, `ss.weekday == 'Fri'`). It reads well for the built-in attributes and has no answer for custom ones without something like Polars' `pl.col('name')`, and the list of names never stays small. The real attributes on the real objects need no list.
- **A class per combination of intervention type and delivery type.** Starsim's `routine_screening` / `campaign_screening` / `routine_triage` / `campaign_triage` / `routine_vx` / `campaign_vx` is two axes spelled out as classes. Keyword arguments, one function each.
- **Interventions as contact reductions only** (Epydemix, MEmilio). Covers COVID-era scenarios and little else.
- **Interventions as extra compartments written by hand** (Epydemix's answer for vaccination, MEmilio's `secirvvs`). This is what `ss.transfer` between strata replaces, and it is why vaccination need not touch the disease model.
- **`'no -> yes'` arrows for booleans.** `ss.transfer(vaccinated=True, ...)`.
- **A separate rule language** (EpiHiper's variables/sets/triggers/operations). The semantics are the best in the review; the syntax is 200 lines of JSON for one Friday.
- **Edge-level eligibility** (EpiHiper's edge predicates). Networks are enough: to change school contacts, modify the school network.
- **Interventions that are inactive unless a scenario enables them** (camdl). Baselines usually contain real interventions; each scenario lists what it contains.
- **Budget optimization in v1** (Atomica, PyRoss). Do not rule it out; do not build it yet.

## Open questions

- **Who gets it when capacity runs out?** With 50,000 doses and 80,000 people eligible, the default is a random 50,000. Real programs prioritize (oldest first, highest risk first). Probably a `priority=` function or attribute; not settled.
- **Stock that carries over.** `capacity=` per round or per unit time does not by itself express a stock that is delivered in batches and depleted over time. A time series covers the simple cases; a stock that depends on use may need `deliver=fn`.
- **How far do conditions that evaluate later extend?** Comparisons and `&`, `|`, `~` on state arrays are clear. Arithmetic (`ppl.age - hiv.ti_infected`) and indexing are less so. The fallback is always the function form, so the short form can start small.
- **Should `ss.transfer` accept the delivery keywords?** A vaccination campaign with coverage and years but no product could be written as a transfer or as `ss.vaccinate` with a minimal product. One of them should be the documented way.
