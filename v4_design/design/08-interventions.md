# Interventions

The four-part decomposition, products, delivery, and scenarios. Most interventions are not new objects; they are a transfer or an effect.

Justified by [best-practices/interventions.md](../best-practices/interventions.md).

## The decomposition

Every intervention answers four independent questions, and each part is independently substitutable:

| Part | Question | Spelled as |
|---|---|---|
| **Where** | spatial scope | `patch=` or an expression over the spatial dimension |
| **When, how many, how often** | schedule and coverage | `years=`, `interval=`, `start=`, `stop=`, `coverage=` |
| **To whom** | eligibility | `eligible=` — an `ss.Expr` ([07-effects.md](07-effects.md)) |
| **What** | the product | `ss.Vaccine`, `ss.Test`, `ss.Treatment`, or an `ss.Effect` |

This is EMOD's factoring, which is the most complete in the landscape, expressed as keyword arguments rather than as four nested configuration objects.

## Check the primitives first

Before reaching for an intervention class, check whether the intervention is one of the two primitives that already exist. Three of the four intervention mechanisms in the entire landscape are covered this way.

**Vaccination is a transfer between dimension coordinates:**

```python
sir.stratify(vaccinated=['no', 'yes'], adjust={'infection': [1.0, 0.15]})
ss.transfer(vaccinated='no -> yes', coverage=0.7, years=[2021, 2022], eligible=ss.age > 65)
```

The disease model never mentions vaccination. Protection is declared once on the dimension; delivery is declared once as a transfer. Representing vaccination as extra compartments inside the disease is what this replaces.

**A contact reduction is an effect:**

```python
ss.Effect(ss.date('2020-03-15') <= ss.time <= ss.date('2020-06-01'),
          school.contacts, multiply=0.2)
```

**A parameter change is an effect:**

```python
ss.Effect(ss.time > ss.date('2020-03-15'), sir.beta, multiply=0.4)
```

This last form subsumes tabular scenario grammars (`{scenario, param, at, value}`) and configuration-file modifier blocks entirely, and unlike them it can also target a subset of agents, because the condition is an expression rather than a time window.

## The simple form

The simplest time-varying-parameter grammar in the landscape is a list of times and a list of values, and it should be the first thing in the documentation:

```python
ss.change(sir.beta, days=[30, 60], to=[0.5, 1.0])
ss.change(school.contacts, years=[2020.2, 2020.6], to=[0.2, 1.0])
ss.change(sir.beta, dates=['2020-03-15', '2020-06-01'], to=[0.5, 1.0])
```

`ss.change` is sugar over `ss.Effect` and MUST compile to exactly that: the values are `multiply=` factors applied from each time until the next. It exists because three frameworks converged on this shape and because generality here cost more legibility than it bought.

## Products

What remains after the primitives is the delivery of a *thing* with efficacy characteristics.

```python
ss.Vaccine(
    blocks   = 'acquisition',              # 'acquisition' | 'transmission' | 'severity' | 'mortality'
    efficacy = 0.85,
    take     = 1.0,                        # fraction in whom the product works at all
    waning   = ss.exp_decay(halflife=ss.years(5)),
    cost     = 12.50,
    doses    = 1,
    interval = None,                       # for multi-dose schedules
)

ss.Test(sens=0.9, spec=0.98, cost=4.00, delay=ss.days(2))
ss.Treatment(efficacy=0.95, cost=120.0, blocks='transmission')
```

Three product classes, and adding a fourth requires showing that it has structure the existing three do not — a diagnostic earns its class because it returns a *result* that other things branch on; a bednet does not, because it is `ss.Effect(ss.has(net), sir.rel_sus, multiply=0.6)`.

**Efficacy and waning are separate.** `efficacy` is the effect size at full potency; `waning` is a function of time since receipt. Keeping them separate means a waning function can be reused across products and a product's efficacy can be calibrated without touching its waning. `waning=` accepts `ss.exp_decay(halflife=)`, `ss.linear_decay(to=, over=)`, or an `ss.timeseries`.

**`blocks=` names which quantity the product acts on**, so a product compiles to an `ss.Effect` on a known target and inherits its composition rules. Two vaccines both blocking acquisition compose multiplicatively, as they should, with no extra machinery.

**`cost=` is the whole cost requirement for v1.** It makes cost-effectiveness a result rather than a post-processing script, and it keeps the door open to budget optimization and optimal control later without committing to either now.

## Delivery

```python
ss.vaccinate(product, coverage=, eligible=, years=|interval=, start=, stop=, patch=)
ss.screen(test, coverage=, eligible=, interval=, on_positive=)
ss.treat(treatment, coverage=, eligible=, ...)
```

```python
bcg = ss.Vaccine(blocks='acquisition', efficacy=0.85,
                 waning=ss.exp_decay(halflife=ss.years(5)), cost=12.50)

ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=ss.age < 5)
ss.vaccinate(bcg, coverage=0.7, interval=ss.years(1), start=2021, eligible=ss.age < 5)
```

Routine versus campaign delivery is a keyword, not a class: `interval=` says routine, `years=` says campaign. A Cartesian product of delivery mode × product type spelled out as classes is exactly the vocabulary growth the language is trying to avoid.

`coverage=` is a fraction by default. A **number** of doses is a different and equally real specification, and is written `coverage=ss.count(50_000)`; the distinction matters when supply, not demand, is the constraint. Coverage that cannot be met from the eligible population emits `W1101 CoverageUnreachable` with the achieved fraction.

### Cascades

```python
ss.screen(ss.Test(sens=0.9, spec=0.98),
          coverage=0.4, interval=ss.years(3),
          eligible=(ss.age > 30) & (ss.sex == 'f'),
          on_positive=ss.treat(ss.Treatment(efficacy=0.95), coverage=0.8))
```

`on_positive=` and `on_negative=` chain one delivery to another, each with its own coverage and eligibility, so loss to follow-up at each step is explicit. Chaining is arbitrary depth, though depth beyond three is a sign the cascade wants to be a small module.

## Triggers

Because an expression may reference results, reactive policy is declarative with no new machinery:

```python
ss.Effect(sim.results.infection.weekly > 1000, school.contacts, multiply=0.2,
          min_duration=ss.weeks(4))
```

`min_duration=` is required in practice, not optional: without hysteresis a threshold trigger oscillates at the step frequency, and every framework that offers triggers without it has this bug latent.

## Scenarios in the file

Scenarios are part of the analysis, not a shell script around it.

```python
sims = ss.parallel(
    baseline = ss.Sim(sir),
    vaccine  = ss.Sim(sir, interventions=vx),
    both     = ss.Sim(sir, interventions=[vx, masks]),
)
sims.run()
sims.plot()
sims.to_df()      # one tidy frame, indexed by scenario, replicate, and time
```

Requirements:

- **Interventions are inactive by default and enabled by scenario.** This makes the baseline unambiguous, and it means a model file can carry its full intervention set without any of them being silently on.
- **Common random numbers are on by default across scenarios** ([11-random.md](11-random.md)), so a paired difference is the effect rather than the noise.
- **A set of scenarios is the default unit of a run**, not a single trajectory. `ss.parallel` returns a result object indexed by scenario, replicate, and time, and `sims.diff('baseline')` gives paired differences with intervals.

## Realization per backend

| Declaration | Compartmental | Agent-based |
|---|---|---|
| `ss.transfer(dim=..., coverage=)` | a flow between coordinate compartments at the implied rate | select eligible agents, thin to coverage, change the property |
| `ss.Effect(cond, x, multiply=)` | a time- or group-varying multiplier on `x` | a registered per-agent modifier on `x` |
| `ss.Vaccine(...)` | an effect on the vaccinated coordinate | an effect keyed to agents holding the product |
| `eligible=` rank `group` | selects coordinates | selects agents |
| `eligible=` rank `agent` | **refused** ([12-backends.md](12-backends.md)) | selects agents |
| `ss.Test(...)` returning a result | requires a diagnosed dimension | per-agent, exact |

Eligibility rank is the mechanism that decides whether an intervention survives paradigm conversion, and it is computed and reported before the run.

## Rejected

- **A class lattice per intervention type.** Routine screening, campaign screening, routine triage, campaign triage, routine vaccination, campaign vaccination is a Cartesian product of two axes spelled out as six classes. Keyword arguments; one class each.
- **Interventions as contact reductions only.** Covers one era's scenarios and nothing else.
- **Interventions as extra compartments.** This is what a transfer between dimension coordinates replaces.
- **Opaque eligibility callables as the primary mechanism.** Available as an escape hatch, priced.
- **A separate rule language** with variables, sets, triggers, and operations as configuration. The semantics are the best in the landscape; the syntax is 200 lines of JSON for one Friday. Python operators give the same semantics in one line.
- **Budget optimization and optimal control in v1.** Do not foreclose — `cost=` and declared effects are what keep the door open — but do not build.
