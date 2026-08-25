# Interventions

## Recommendation

1. **Adopt EMOD's four-part decomposition: where × when and how many × to whom × what.** It is the most complete treatment in the review and each part is independently substitutable.
2. **Eligibility is a declarative predicate over agent and model state**, built from Python operators — not an opaque callable.
3. **Keep Starsim's product/delivery split**, and separate efficacy from waning as EMOD does.
4. **Most interventions are a `ss.transfer` between strata or an `ss.Effect` on a named quantity.** Do not build a class lattice for what is already two primitives.
5. **The simple case stays a one-liner.** Covasim's `days`/`changes` form was simpler than what replaced it, and that simplicity should come back.
6. **Interventions carry a cost.** Nothing else in the review except Atomica does this, and it is what the policy audience needs.
7. **Scenarios live in the model file**, not in a shell script.

## Why

### The split that runs through the whole review

Finding 1 of the [landscape review](../approaches/README.md#cross-cutting-findings): "Structural rigour and intervention expressiveness have not yet appeared in the same framework." [camdl](../approaches/camdl.md), [odin](../approaches/odin.md), [summer2](../approaches/summer2.md), and [epipack](../approaches/epipack.md) have the best model specification and essentially no intervention vocabulary. [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [Starsim](../approaches/starsim.md), and [Atomica](../approaches/atomica.md) have the best intervention vocabulary and weak structural specification.

Closing that split is, per the review, "the most concrete contribution available to this project."

### The four parts

From [EMOD](../approaches/emod.md), which decomposes a campaign event into four orthogonal pieces:

| Part | Question | EMOD |
|---|---|---|
| Where | Spatial scope | `Nodeset_Config` |
| When, how many, how often | Schedule and coverage | `Event_Coordinator_Config` |
| To whom | Eligibility | `Targeting_Config` |
| What | The product | `Intervention_Config` + `Waning_Config` |

Starsim has parts two and four (`RoutineDelivery` / `CampaignDelivery` mixins, and `ss.Product`). camdl's `transfer(...)` is part four only. Nobody else has more than one.

### Eligibility is the piece everyone is missing

[Starsim](../approaches/starsim.md): "Eligibility is an arbitrary callable returning UIDs, which is fully general and completely opaque: it is Python, so it cannot be serialized, diagrammed, or checked."

[EpiModel](../approaches/epimodel.md)'s tabular scenario grammar is genuinely good — `{scenario, param, at, value}` covers a surprising fraction of real specifications without any code — and its limit is exactly the same: "it can only set parameter values, so anything that targets a *subset of agents* (screen 40% of over-50s annually) has to fall back to procedural code."

[flepiMoP](../approaches/flepimop.md)'s modifiers have the same shape and the same ceiling.

Against that, two frameworks show what the declarative version looks like. **EMOD:**

```json
"Targeting_Config": {
  "class": "TargetingLogic",
  "Logic": [ [ {"class": "HasIntervention", "Is_Equal_To": 0, "Intervention_Name": "MyVaccine"} ],
             [ {"class": "HasIP", "Is_Equal_To": 1, "IP_Key_Value": "Risk:HIGH"} ] ]
}
```

**EpiHiper** goes further with set algebra over typed node *and edge* predicates, computed, named, and reusable — "the most expressive targeting mechanism in the review".

And [`individual`](../approaches/individual.md) shows the imperative version of the same algebra, with bitsets:

```r
I <- health$get_index_of("I")
I$and(already_scheduled$not(inplace = TRUE))
```

Its review draws the conclusion: "the lingua franca should have a **declarative set expression** over agent state — the EpiHiper form — with the bitset algebra as its natural implementation."

## The proposal

### Eligibility as a predicate

Python's operators build the expression tree. It reads like the sentence and it is data.

**EMOD** (the 12-line JSON above) → **lingua franca:**

```python
eligible = (ss.age > 65) & ~sir.vaccinated
```

**EpiHiper's "school edges on Fridays"** (roughly 200 lines of JSON across variables, sets, triggers, and interventions) →

```python
ss.Effect(ss.weekday == 'Fri', school.contacts, multiply=0.0)
```

The predicate vocabulary is small and closed:

```python
ss.age, ss.sex, ss.uid                    # built-in agent attributes
sir.I, sir.vaccinated                     # any state, as a predicate
ss.has(vaccine)                           # EMOD's HasIntervention
ss.time, ss.weekday, ss.year              # model-level observables
sim.results.n_infected > 1000             # EpiHiper's triggers: predicates over results
&  |  ~  >  <  ==                         # composition
```

`ss.age > 65` returns an `ss.Expr`, not a boolean. That is the whole trick, and it is the same trick pandas and SQLAlchemy use, so the syntax is already familiar to the audience.

The lambda escape hatch remains and prints its cost:

```python
eligible = lambda sim, uids: my_complicated_thing(sim, uids)
# ! opaque predicate: not serializable, not diagrammable, not usable in compartmental backends
```

### Most interventions are already primitives

Before adding an intervention class, check whether it is one of the two primitives that already exist.

**Vaccination is a transfer between strata** ([stratification.md](stratification.md)):

```python
ss.transfer(vaccinated='no -> yes', coverage=0.7, years=[2021, 2022], eligible=ss.age > 65)
```

The disease model never mentions vaccination. `rel_sus` for the vaccinated stratum is declared once on the dimension, following [EpiHiper](../approaches/epihiper.md)'s state properties.

**A contact reduction is an effect** ([composition-and-effects.md](composition-and-effects.md)):

```python
ss.Effect(ss.date('2020-03-15') <= ss.time <= ss.date('2020-06-01'), school.contacts, multiply=0.2)
```

which is exactly [Epydemix](../approaches/epydemix.md)'s `add_intervention(layer_name="school", start_date=..., reduction_factor=0.2)` and [MEmilio](../approaches/memilio.md)'s `contact_matrix.add_damping(0.6, SimulationTime(12.5))`, in one line, with the layer and the dates visible.

**A parameter change is an effect too**, which subsumes [EpiModel](../approaches/epimodel.md)'s scenario table and [flepiMoP](../approaches/flepimop.md)'s modifiers:

```python
ss.Effect(ss.time > ss.date('2020-03-15'), sir.beta, multiply=0.4)
```

That is three of the four intervention mechanisms in the entire review, expressed with primitives that exist for other reasons. The class lattice is for what remains.

### Products, delivery, and waning

What remains is the delivery of a *thing* with efficacy characteristics — vaccines, diagnostics, treatments. Starsim's product/delivery split is right and its review says so: "The product/delivery split is the right decomposition and should survive into the design."

EMOD adds the missing piece — efficacy and its decay are separate objects (`Vaccine_Take`, `Vaccine_Type` as acquisition-/transmission-/mortality-blocking, and a `Waning_Config` that may be constant, exponential, or a piecewise map).

```python
bcg = ss.Vaccine(
    blocks     = 'acquisition',                  # or 'transmission', 'severity', 'mortality'
    efficacy   = 0.85,
    waning     = ss.exp_decay(halflife=ss.years(5)),
    cost       = 12.50,
)

ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=ss.age < 5)
```

The delivery half keeps Starsim's two shapes — `RoutineDelivery` (start, end, coverage, interval) and `CampaignDelivery` (specific years, coverage) — but as keyword arguments rather than mixin classes, because `years=[2021, 2025]` versus `interval=ss.years(1)` already says which one it is.

Diagnostics and cascades follow HPVsim's screen → triage → treat pattern, which Starsim's `BaseTest`/`BaseTreatment` classes already generalize:

```python
ss.screen(ss.Test(sens=0.9, spec=0.98), coverage=0.4, interval=ss.years(3),
          eligible=(ss.age > 30) & (ss.sex == 'f'),
          on_positive=ss.treat(ss.Treatment(efficacy=0.95)))
```

### Bring back the simple form

From the [Covasim note](../approaches/notes.md#covasim--hpvsim--stisim--fpsim):

> `change_beta(days=[30, 60], changes=[0.5, 1.0])` — a list of days and a list of changes — is a smaller, more legible time-varying-parameter grammar than the class lattice that replaced it. flepiMoP's modifiers and EpiModel's scenario tables are the same shape. **Simplicity here was arguably lost in the generalisation.**

So it comes back, as sugar over `ss.Effect`:

```python
ss.change(sir.beta, days=[30, 60], to=[0.5, 1.0])
```

Three frameworks converged on this shape. It should be one line, and it should be the first thing in the documentation.

### Triggers

[EMOD](../approaches/emod.md)'s event broadcast/observe and [EpiHiper](../approaches/epihiper.md)'s triggers make reactive policy declarative — "when a new infection is observed, trigger contact tracing", or "close schools when weekly cases exceed 1000". Most frameworks make this a callback.

Since a predicate can already reference results, a trigger needs no new machinery:

```python
ss.Effect(sim.results.new_infections.weekly > 1000, school.contacts, multiply=0.2)
```

The only addition is a per-trigger hysteresis, because policies do not flap:

```python
ss.Effect(..., min_duration=ss.weeks(4))
```

### Cost

Only [Atomica](../approaches/atomica.md) represents an intervention as something with a unit cost and a coverage function that a budget constrains — the representation a health ministry needs. Its `Targetable` column declares which parameters a programme may move, which is also a partial answer to the connector problem.

`cost=` on the product is the whole requirement for v1: it makes cost-effectiveness a result rather than a post-processing script, and it is what keeps the door open to [PyRoss](../approaches/notes.md#pyross)-style optimal control and Atomica-style budget optimization later without committing to either now.

### Scenarios in the file

[flepiMoP](../approaches/flepimop.md) declares `scenarios: [Ro_lockdown, Ro_all]` in the config; camdl has named scenarios paired against baseline with common random numbers; [`epidemics`](../approaches/epidemics.md) returns a scenario grid as one tidy frame with scenario identifiers.

```python
sims = ss.parallel(
    baseline = ss.Sim(sir),
    vaccine  = ss.Sim(sir, interventions=vx),
    both     = ss.Sim(sir, interventions=[vx, masks]),
)
sims.plot()
```

camdl's discipline is worth copying exactly: **interventions are inactive by default and enabled by scenario**, which makes the baseline unambiguous. And because [common random numbers](stochasticity-and-reproducibility.md) are on, the paired difference is the effect rather than the noise.

## Trade-offs

- **The predicate algebra is a real implementation cost** — an expression tree, evaluators for each backend, and error messages. It is the price of eligibility being data, and every framework without it has the documented ceiling.
- **A predicate that references agent-level attributes has no compartmental meaning** unless the attribute is a dimension. `(ss.age > 65)` works in both because age is a dimension; `ss.uid % 7 == 0` works only in an ABM. Capability check, named.
- **`ss.Effect` doing this much work makes it central.** If it is wrong, a lot is wrong. That is an argument for it being small: a condition, a target, a combiner.
- **Products are a vocabulary that will grow.** Vaccines, tests, treatments, bednets, condoms, contact tracing. Resist: most are `ss.Effect` on a named quantity with a coverage and an eligibility, and only the ones with genuinely distinct structure (a diagnostic returns a *result*) need a class.

## Rejected

- **A class lattice per intervention type.** Starsim's `routine_screening` / `campaign_screening` / `routine_triage` / `campaign_triage` / `routine_vx` / `campaign_vx` is a Cartesian product of two axes spelled out as classes. Keyword arguments, one class each.
- **Interventions as contact reductions only** (Epydemix, MEmilio). Covers COVID-era scenarios and nothing else.
- **Interventions as extra compartments** (Epydemix's answer for vaccination, MEmilio's `secirvvs`). This is what `ss.transfer` between strata replaces, and it is why vaccination need not touch the disease model.
- **Opaque eligibility callables as the primary mechanism** (Starsim today). Available as an escape hatch, priced.
- **A separate rule language** (EpiHiper's variables/sets/triggers/operations). The semantics are the best in the review; the syntax is 200 lines of JSON for one Friday. Python operators give the same semantics in one line.
- **Budget optimization in v1** (Atomica, PyRoss). Do not foreclose; do not build.

## Open questions

- **Stock constraints.** "Vaccinate as many as we have doses for" is EMOD's `NChooser` (target a number, not a fraction) and is common in real programmes. Coverage-as-a-number versus coverage-as-a-fraction is a real distinction and probably belongs in `coverage=`.
- **Does eligibility need edge predicates?** EpiHiper's set algebra covers edges as well as nodes, which is what makes "close school contacts" expressible directly rather than as a layer. Layers may be enough.
- **Cascades with state.** "Screen, and if positive, triage, and if positive, treat" is a chain where each step has its own coverage and loss-to-follow-up. `on_positive=` handles depth two; deeper cascades may need a different shape.
