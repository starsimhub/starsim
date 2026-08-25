# Types: time, rates, and quantities

The dimensional type system. Small, inferred from context, and never optional.

Justified by [best-practices/time-and-units.md](../best-practices/time-and-units.md). This is Starsim's existing type system, specified precisely and extended in exactly one way: bare numbers are accepted and typed by the argument slot they appear in.

## The types

There are five, and there MUST NOT be a sixth without a corresponding change to this document.

| Type | Dimension | Means | Constructors |
|---|---|---|---|
| `ss.dur` | time | an elapsed duration | `ss.days`, `ss.weeks`, `ss.months`, `ss.years`, `ss.datedur` |
| `ss.rate` | time⁻¹ | a hazard: events per unit time, continuously | `ss.perday`, `ss.perweek`, `ss.permonth`, `ss.peryear` |
| `ss.prob` | dimensionless, ∈ [0,1] | a probability attached to a period | `ss.probperday`, `ss.probperweek`, `ss.probpermonth`, `ss.probperyear` |
| `ss.freq` | time⁻¹ | a count of events per unit time | `ss.freqperday`, `ss.freqperweek`, `ss.freqpermonth`, `ss.freqperyear` |
| `ss.count` | dimensionless, ≥ 0 | a number of individuals or events | plain numbers in count slots; `ss.N` |

`ss.rate` and `ss.freq` share a dimension and are not the same type. A rate is the parameter of an exponential clock and converts by `1 − exp(−r·dt)`; a frequency is an expected count and converts by `f·dt`. Conflating them is one of the quietest common errors in the field, and the two-type split exists to prevent it. Attempting to add them raises `E101 IncompatibleTypes`.

**Deviation from Starsim, noted deliberately.** Starsim names the hazard type `per`; this specification names it `rate`, because "a rate" is what an epidemiologist calls the thing and "a per" is not a noun. The constructors keep their existing spelling (`ss.perday`, `ss.peryear`), so no model text changes, and `ss.per` remains a recognized name for the same type through the migration. This is the only rename in the type system; everything else is Starsim's, unchanged.

`ss.datedur` is retained and is not redundant with `ss.months`: calendar durations are not fixed multiples of days, and float-year arithmetic gets month lengths wrong.

## Conversion

Conversion is the type's responsibility, never the user's. The four conversions to a step of length `dt` are:

| From | To per-step probability | To per-step expected count |
|---|---|---|
| `ss.rate(r)` | `1 − exp(−r·dt)` | `r·dt` |
| `ss.prob(p, per=T)` | `1 − (1 − p)^(dt/T)` | — |
| `ss.freq(f)` | — | `f·dt` |
| `ss.dur(d)` | via `rate = 1/d` | — |

These formulas are normative. A backend MUST NOT substitute `r·dt` for `1 − exp(−r·dt)`.

**The large-step warning.** When `r·dt > 0.1` for any converted rate, the build MUST emit `W102 LargeStep`, naming the transition, the rate, the timestep, and the relative discrepancy between `r·dt` and `1 − exp(−r·dt)`. Above `r·dt > 1` this becomes an error unless `method` is continuous-time. This is the boundary at which intuitions about "a 10% chance per day" stop being reliable.

**Printing the conversion.** `summary(verbose=True)` MUST show the resolved per-step form for every converted quantity:

```text
timestep: 1 day
infection: beta = 0.05/day  ->  p = 1 - exp(-0.05 x 1) = 0.0488 per step
recovery:  dwell ~ Exp(mean = 10 days)  ->  p = 0.0952 per step
```

This is the transparency odin obtains by making the modeler write `1 - exp(-gamma*dt)` by hand, obtained instead from the model being data.

## Arithmetic

The complete type lattice. Any combination not in this table MUST raise `E101 IncompatibleTypes`.

| Left | Op | Right | Result | Note |
|---|---|---|---|---|
| `dur` | `+` `−` | `dur` | `dur` | |
| `dur` | `×` `÷` | scalar | `dur` | |
| `dur` | `÷` | `dur` | scalar | |
| `rate` | `+` `−` | `rate` | `rate` | competing hazards add |
| `rate` | `×` `÷` | scalar | `rate` | |
| `rate` | `×` | `dur` | scalar | **the expected count over the period, not a probability** |
| `rate` | `×` | `count` | `count`/time | a flow |
| `freq` | `+` `−` | `freq` | `freq` | |
| `freq` | `×` | `dur` | `count` | |
| `prob` | `×` | `prob` | `prob` | joint, assuming independence |
| `prob` | `×` | `count` | `count` | |
| scalar | `÷` | `dur` | `rate` | |
| scalar | `÷` | `rate` | `dur` | |
| `count` | `+` `−` | `count` | `count` | |
| `count` | `÷` | `count` | scalar | |
| `rate` | any | `prob` \| `freq` | **error** | `E101` |
| `freq` | any | `prob` | **error** | `E101` |
| `dur` | `+` | scalar | **error** | `E103 UntypedInTypedExpression` |

The `rate × dur → scalar` row is the one that most often signals a mistake, so `E102` names it specifically when the result lands in a probability slot:

```text
E102 WrongDimension: rate expression for transition 'recovery' has dimension [dimensionless],
     expected [count/time].
       rate = ss.peryear(0.1) * ss.days(3)
                               ^^^^^^^^^^^ multiplying by a duration removes the time dimension
     A rate times a duration is an expected count, not a rate. Did you mean p= rather than rate=?
```

Error messages of this shape are required, not suggested. See [15-errors.md](15-errors.md).

## Bare numbers

**A bare Python number is legal in any typed slot.** The slot supplies the dimension; the sim supplies the unit.

```python
ss.transmission('S -> I', beta=0.05)     # beta= is a rate slot     -> ss.rate(0.05, unit)
ss.progression('I -> R', dur=10)         # dur= is a duration slot  -> ss.dur(10, unit)
ss.progression('I -> R', p=0.1)          # p= is a probability slot -> ss.prob(0.1, per=dt)
```

The slot-to-type mapping is closed and is part of the specification of each constructor; it is collected in [17-vocabulary.md](17-vocabulary.md). A slot that accepts more than one type (`ss.progression` accepts `dur=`, `rate=`, or `p=`) resolves by which keyword was used, never by the value.

**Unit resolution** proceeds in this order, stopping at the first that applies:

1. The quantity is already typed (`ss.perday(0.05)`) — use it.
2. `Sim(unit=...)` was given — use it.
3. `Sim(dt=...)` was given — use the unit of `dt`.
4. `Sim(start=..., stop=...)` are dates — use days.
5. Otherwise assume days, and emit `W101 UnitAssumed` in the summary.

Rule 5 is deliberate. Refusing to run because nobody said "day" would discourage exactly the users this language exists for, and the assumption is printed:

```text
infection: beta = 0.05/day  (rate; unit assumed: day)
```

A model whose bare numbers are all in years and which is silently run in days will produce results that are wrong by a factor of 365 — which is why the assumption appears in the summary, and why `W101` is emitted rather than suppressed. Magnitude heuristics MUST NOT be used to guess the unit: a rate of 0.05 is plausible per day and per year, and a wrong guess dressed up as a clever one is worse than a stated default.

## Time in a `Sim`

```python
ss.Sim(sir, start='2020-01-01', stop='2021-06-30', dt=ss.days(1))   # calendar
ss.Sim(sir, start=2020, stop=2030, dt=ss.months(1))                 # float years
ss.Sim(sir, start=0, stop=100)                                      # relative
```

All three MUST be supported. Calendar dates are the default interpretation of a string; a bare number in `start=` is a year if it is above 1500 and a relative time otherwise, and the interpretation is printed.

A `Timeline` exposes the parallel representations Starsim already provides — `tvec`, `tivec`, `timevec`, `yearvec`, `datevec`, `relvec` — so that output can be requested in the frame the data are in. This is retained unchanged and MUST NOT be extended. Results carry one timevec and convert on request ([09-results.md](09-results.md)).

## Per-module timesteps

A module MAY declare its own timestep:

```python
class Demographics(ss.Demographics):
    dt = ss.years(1)
```

Modules with different timesteps are interleaved by absolute time by the execution loop ([10-execution.md](10-execution.md)). A module that declares no `dt` inherits the sim's. This is Starsim's mechanism and nothing else in the landscape has it; it is kept, and it is the default behavior rather than an option.

**Sub-step offsets.** A module or route MAY declare an offset within the step:

```python
ss.Mobility(ss.gravity(...), kind='commute', at=ss.hours(8), returns=ss.hours(18))
```

Sub-step scheduling is what commuting requires: leave in the morning, mix at the destination, return in the evening. The semantics are specified in [10-execution.md](10-execution.md) §Sub-steps. Sub-step offsets and per-module timesteps are one mechanism — "this process happens at this point in time" — with two spellings, and both resolve to absolute times in the loop.

## Distributions as typed quantities

An `ss.Dist` carries the type of its parameters, so a distribution is legal wherever a quantity of that type is:

```python
ss.lognorm(mean=ss.days(10), std=ss.days(3))     # a dur-valued distribution
ss.lognorm(mean=0.05, std=0.01)                  # dimensionless; typed by its slot
```

The ten distributions in the core vocabulary are `bernoulli`, `uniform`, `normal`, `lognorm`, `expon`, `gamma`, `erlang`, `poisson`, `binomial`, `weibull`, plus `neg_binomial` for observation likelihoods. Anything else is `ss.dist(scipy_frozen)`, which is supported everywhere a `Dist` is and reports `~` capability for backends that need a closed form. Distributions MUST NOT be added to the core vocabulary without removing one; see [17-vocabulary.md](17-vocabulary.md).

**The array-draw rule**, stated once, normatively, because it is ambiguous in most frameworks and silently wrong in some:

> **A distribution placed in a per-agent slot draws once per agent. A distribution evaluated to a value draws once, and that value is shared.**

```python
ss.progression('I -> R', dur=ss.lognorm(mean=ss.days(10)))   # one draw per agent
shared = ss.lognorm(mean=ss.days(10)).rvs()                  # one draw, shared by everyone
```

RNG stream assignment and reproducibility semantics are in [11-random.md](11-random.md).

## What is deliberately not typed

- **Money.** `cost=12.50` is a number with a currency label, not a dimension to check. Cost-effectiveness arithmetic is checked for shape, not for dimension.
- **Space.** Distances are numbers in a unit declared by the geography ([06-routes.md](06-routes.md)), not a dimension in this lattice.
- **A general dimensional algebra** (mass, length, concentration). Epidemiology needs time⁻¹, dimensionless, and counts. `ss.Continuous` reservoirs ([03](03-states-and-transitions.md)) carry a free-text unit that is recorded and not checked.

## Open questions carried forward

- Whether `ss.prob` should carry its period as part of its type in the IR, or normalize to a per-step value at compile time. The former is more faithful; the latter makes the IR smaller. See [19-open-questions.md](19-open-questions.md).
- Continuous-time backends have no `dt` and therefore no conversion at all, which is a real correctness advantage. The model does not need to know; the conversion table above simply goes unused. This is specified in [12-backends.md](12-backends.md) and is the reason continuous-time methods are preferred where they are tractable.
