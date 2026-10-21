# Types: time, rates, and quantities

Durations, rates, probabilities, and frequencies; how bare numbers get their units; and time in a sim.

Justified by [best-practices/time-and-units.md](../best-practices/time-and-units.md). These are Starsim's types and names, with Starsim's names kept: `ss.perday` to `ss.peryear`, `ss.probperday` to `ss.probperyear` (the same as `ss.prob(0.1, 'day')` to `ss.prob(0.1, 'year')`), `ss.freqperday` to `ss.freqperyear`, and the one-unit instances `ss.day` to `ss.year`. In a declared transition a `freq` and a `per` of the same value give the same result ([Conversion](#conversion)). There are two additions: a bare number is accepted in every slot and typed by the slot it appears in, and a model in which nothing has a unit counts time in timesteps.

## The types

| Type | Dimension | Means | Constructors |
|---|---|---|---|
| `ss.dur` | time | an elapsed duration | `ss.days`, `ss.weeks`, `ss.months`, `ss.years`, `ss.datedur` |
| `ss.per` | time⁻¹ | a hazard: events per unit time, continuously | `ss.perday`, `ss.perweek`, `ss.permonth`, `ss.peryear` |
| `ss.prob` | probability per period | the chance of an event within a period | `ss.probperday`, `ss.probperweek`, `ss.probpermonth`, `ss.probperyear` (the same as `ss.prob(0.1, 'day')` and so on) |
| `ss.freq` | time⁻¹ | an expected number of events per unit time | `ss.freqperday`, `ss.freqperweek`, `ss.freqpermonth`, `ss.freqperyear` |

One more value type is not a time value: **`ss.peract(p)`**, a probability per act, for a transmission over a network that defines acts (`ss.MFNet`, `ss.PartnerNet(acts=...)`). It has no unit and no conversion, so it means the same at any step: a partnership with `a` acts per unit time transmits over a step with probability `1 − (1 − p)^(a·dt)`. It takes one value, or one for each of the network's directions (`ss.peract(m2f=0.3, f2m=0.15)`; see [06-routes.md](06-routes.md#beta-per-route-and-direction)); each value is a number in [0, 1] (else `OutOfBounds`), an `ss.par` or an expression of them, an `ss.timeseries`, or an `ss.by`. A typed rate inside it raises `WrongDimension`.

`ss.per`, `ss.prob`, and `ss.freq` share the base class `ss.Rate`; all four types share `ss.TimePar`. `ss.per` is the hazard type and keeps its Starsim name. A typed value needs its unit: `ss.dur(3)` and `ss.per(2)` with no unit raise (they were years in Starsim 3).

`ss.per` and `ss.freq` have the same dimension and are different types. A hazard is the parameter of an exponential clock and converts to a per-step probability by `1 − exp(−r·dt)`; a frequency is an expected count and converts by `f·dt`. Mixing them up is a common quiet error, so adding one to the other raises `IncompatibleTypes`, as Starsim does today.

The difference is only in that conversion, which a module written as code asks for (`to_prob()` for a chance this step, `* dt` for a count this step). A declared transition never asks for it: its rate is always a hazard, which the solver turns into a step itself (the ODE integrates it; the stochastic solvers and agents use `1 − exp(−r·dt)`). So in a declared transition `ss.freqperday(0.1)` and `ss.perday(0.1)` give the same result, while `ss.probperday(0.1)` is the hazard that gives a 10% chance a day, `−ln(0.9) = 0.105` a day, and gives a different result (10 days of recovery from 1,000: 368 left at `perday(0.1)` or `freqperday(0.1)`, 349 at `probperday(0.1)`).

`ss.datedur` is kept alongside `ss.months` because calendar durations are not fixed multiples of days.

Counts are plain numbers. `I.n` is the number of people in a state (the same number as the result `n_I`) and `I.frac` its fraction ([03-states-and-transitions.md](03-states-and-transitions.md)). `P.time`, the time each person has spent in a state, is a number in the sim's unit, as a bare number returned by a function is.

## Conversion

Converting to a timestep is the type's job, not the user's. For a step of length `dt`:

| From | To a per-step probability | To a per-step expected count |
|---|---|---|
| `ss.per(r)` | `1 − exp(−r·dt)` | `r·dt` |
| `ss.prob(p)` per period `T` | `1 − (1 − p)^(dt/T)` | |
| `ss.freq(f)` | | `f·dt` |
| `ss.dur(d)` | through the rate `1/d` | |

A solver MUST NOT substitute `r·dt` for `1 − exp(−r·dt)`.

**Large steps.** In a run that takes whole steps (`'binomial'`, `'abm'`, or the ODE with Euler steps of a given size), when `rate × dt > 0.1` for a dwell time, the build gives the note `ss.LargeStep`, naming the transition, `rate × dt`, the share of people who leave in a step, and what to do (a smaller `dt`, `substeps=`, or `solver='gillespie'`). Under the other solvers, which follow the rates within a step, it is given only for a mean dwell time shorter than the step, which is usually a rate meant as a duration (`rate=10`). It is a note, not an error; `sim.check_dt()` shows how much the results change with a smaller step ([10-execution.md](10-execution.md)).

Continuous-time solvers (`'ode'`, `'gillespie'`) have no `dt` in this sense and do no conversion at all.

## Arithmetic

Starsim's rules, summarized:

| Left | Op | Right | Result | Note |
|---|---|---|---|---|
| `dur` | `+` `−` | `dur` | `dur` | units are converted |
| `dur` | `+` `−` `×` `÷` | number | `dur` | a bare number takes the duration's unit: `ss.days(3) + 2` is 5 days |
| `dur` | `÷` | `dur` | number | |
| `per` | `+` `−` | `per` | `per` | competing hazards add; units are converted |
| `per` | `×` `÷` | number | `per` | |
| `per` | `×` | `dur` | number | the probability of at least one event over the period, `1 − exp(−r·d)`: `ss.perday(0.1) * ss.days(3)` is 0.259 |
| `freq` | `×` | `dur` | number | an expected count, `f·d`: `ss.freqperday(0.1) * ss.days(3)` is 0.3 |
| `prob` | `+` `−` | `prob` | `prob` | through the underlying rates, as competing hazards: `ss.prob(0.1, 'day') + ss.prob(0.1, 'day')` is 0.19 per day, the same as `2 * ss.prob(0.1, 'day')` |
| `prob` | `×` `÷` | number | `prob` | scales the underlying rate, not the probability: `ss.probperyear(0.8) * 0.2` is 0.275 a year, not 0.16 (a relative risk on a probability is a relative risk on its hazard). A probability of 1 is an infinite rate and stays 1; as the rate of a transition it raises, since no solver can step it |
| `prob` | `×` | `dur` | number | the probability over the period |
| rate | `÷` | rate of the same kind | number | |
| number | `÷` | `dur` | `per` | `1 / ss.days(10)` is `ss.perday(0.1)` |
| number | `÷` | `per` or `freq` | `dur` | `2 / ss.perday(0.1)` is `ss.days(20)` |
| `per` | `+` `−` | `prob` or `freq` | error | `IncompatibleTypes` |
| rate | `+` `−` | number or `dur` | error | `IncompatibleTypes`; a number minus a rate too |
| number | `÷` | `prob` | error | `IncompatibleTypes` |
| rate | `×` | rate | error | `WrongDimension`: per time squared |
| rate | `÷` | `dur` | error | `WrongDimension` |
| `dur` | `×` `÷` | `dur` and rate respectively | error | `WrongDimension`: time squared |

Here "rate" is any of `per`, `prob` (with a time unit), and `freq`. Both errors name the two types.

The slot a result lands in is checked too. A rate multiplied by a duration is a plain number, and putting one in a `rate=` slot or after `@` is a mistake, so `WrongDimension` points at it:

```text
WrongDimension: the rate of the transition "recovery" is a plain number (a rate times a duration), not a rate:
       rate = peryear(0.1) * days(3) = 0.00082158
     A rate times a duration is a probability over that duration, not a rate; give the rate itself, e.g. rate=ss.peryear(0.1).
```

The number remembers the rate and the duration it was made from, which is how the slot can tell it from a bare number. The same check applies to the typed values inside an expression, from their time dimensions: in a rate, `ss.perday(0.3) * I * ss.days(3)` is not per unit time and raises, while `R0 * I / ss.days(10)` is and does not; a duration given as `rate=` raises too, as does a rate given as `dur=` (`dur=ss.prob(0.1, 'day')`). A whole duration after `@` is a dwell time, not a rate ([03](03-states-and-transitions.md#what-may-follow-)). A bare number has no dimension of its own, so an expression made of bare numbers is taken at its word.

## Bare numbers

**A bare Python number is accepted in every slot**, with no warning. The slot gives the dimension and the sim gives the unit.

<!-- fragment -->
```python
ss.transmission('S -> I', beta=0.05)      # beta= is a rate slot      -> 0.05 per unit time
ss.progression('I -> R', dur=10)          # dur= is a duration slot   -> 10 units of time
ss.progression('I -> R', rate=0.1)        # rate= is a rate slot      -> 0.1 per unit time
I >> R @ 0.1                              # after @, a bare number is always a rate
```

A bare number after `@` is a rate, never a duration, because "at" reads as "at rate" and `@ 0.1` and `@ 10` cannot both be guessed from their size. A bare-number duration uses the keyword, `dur=10`.

A slot that can be filled in more than one way (`ss.progression` takes `dur=` or `rate=`) is decided by which keyword was used, never by the value. The slots and their types are listed in [17-vocabulary.md](17-vocabulary.md).

A typed value (`ss.perday(0.05)`) keeps its own unit and is converted to the sim's. A bare number takes the sim's unit.

### The sim's unit

**The sim's unit** is taken from the first of these that applies:

| | What is given | The sim's unit |
|---|---|---|
| 1 | `unit='day'` | what it says |
| 2 | typed values in the model (`dur=ss.days(10)`, or a declared module's typed `dt`) | their unit; if they differ and the model has bare numbers, an error asking for `unit=` (which unit the bare numbers are in cannot be guessed) |
| 3 | `start=` or `stop=` as a date, such as `start='2020-01-01'` | days |
| 4 | a typed `dt` or a typed `start`, `stop`, or `dur`, such as `dt=ss.weeks(1)` | its unit |
| 5 | `start=` or `stop=` as a year, such as `start=2020, stop=2030` | years |
| 6 | none of these | none: time is in timesteps |

The model's unit is a property of the model, so its own typed values come before how the run is written; `dt` then sets only the step. Only `start` and `stop` supply a unit from dates: a date on an intervention or in the data does not. A year-like number for `start` or `stop` (`stop=2020`) is a calendar year, since Starsim 3 sims are written that way, unless `unit=` gives another unit. The unit is never guessed from the size of the numbers: 0.05 is plausible per day and per year.

Modules written as code are outside this rule: their typed defaults do not set the unit (they convert themselves), and a bare number given for one of their parameters (`ss.MFNet(duration=5)`) keeps the unit of its default, as in Starsim 3, with a warning when that differs from the sim's. A sim with no declared states keeps Starsim 3's time defaults (years, and 50 of them with no `stop`).

Whenever the unit comes from rules 2 to 4 it is a guess about what the bare numbers meant, so the sim says where it came from (`UnitInferred`), at `init()` and in `describe()`:

```text
>>> ss.Sim(sir, start='2020-03-01', stop='2020-09-01').init()
solver: ode (every part of the model has a compartmental form)
  note: unit='day' (from start and stop dates)
```

and `describe()` shows `unit  day  [from start and stop dates]`.

The default timestep is one unit: one day, one year, or one step. With no `stop`, a sim runs for 50 units (or 50 steps) from its start.

### Time in timesteps

If nothing has a unit (no `unit=`, no unit on any `dt`, no dates, no typed values), time is unitless and counted in timesteps. `beta=0.3` is per step, `dur=10` is ten steps, `stop=100` is 100 steps, results are indexed by step, and nothing is said or assumed about days or years:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
    init_prev = 0.01,
)
ss.Sim(sir, stop=100).run()    # 100 steps
```

This is what a textbook model means. No note is printed, because nothing was guessed; `describe()` shows `unit  none: time is in timesteps`.

- The timestep is one step. With a different `dt`, the bare numbers are per unit of time, a unit with no name: `describe()` then says `time is in units with no name, and a step is 0.1 of them`, `beta = 0.3 per unit`, and `every 0.1 units`, not "per step".
- The times have no dates: `sim.t.datevec` raises, and so does an `ss.seasonal()`, which is a yearly cycle. (The arithmetic is done in days underneath, so a module written as code with typed values, such as `ss.peryear`, takes a step as a day.)
- A unitless sim has no calendar, so it cannot be compared with dated data or given a dated intervention: a date on an intervention is outside the sim (`ss.TimeOutsideSim`). Give the sim a unit or dates.
- This differs from Starsim 3, which assumes years when nothing is specified, and which sims with no declared states still do.

## Time in a sim

```python
ss.Sim(sir, start='2020-01-01', stop='2021-06-30', dt=ss.days(1))   # calendar dates
ss.Sim(sir, start=2020, stop=2030, dt=ss.months(1))                 # years
ss.Sim(sir, stop=100, unit='day')                                   # relative time, in days
ss.Sim(sir, stop=100)                                               # relative time, in steps
```

All four are supported. A string is a calendar date; a bare number in `start=` or `stop=` is read as a year if it looks like one (`2020`) and as a relative time otherwise (`0`), and the reading is printed (`unit='year' (from start and stop years)`). Relative time works with or without a unit.

The sim's `Timeline` provides the parallel time vectors Starsim already has (`tvec`, `yearvec`, `datevec`, `timevec`, and so on), so output can be requested in the frame the data are in. Nothing is added to it; whether it can be reduced is an open question below.

## Per-module timesteps

A module may have its own timestep:

```python
ss.Sim([sir, ss.Births(dt=ss.years(1))], dt=ss.days(1))
```

Modules with different timesteps are interleaved by absolute time in the loop ([10-execution.md](10-execution.md)). A module with no `dt` uses the sim's. This is Starsim's mechanism, kept as it is.

There are no sub-step offsets. Commuting, which the first draft handled with departure and return times inside a step, is specified instead as the fraction of time a person spends in the destination patch; a commuter contributes to and is exposed in each patch in proportion to the time spent there. See [06-routes.md](06-routes.md).

## Distributions

Distributions use Starsim's names: `ss.normal`, `ss.lognorm_ex` (given the mean and standard deviation of the lognormal itself), `ss.lognorm_im` (given those of the underlying normal), `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, `ss.choice`, and the rest of Starsim's set. Any SciPy distribution can be wrapped with `ss.Dist(dist=...)`. There are two additions: `ss.erlang(k=, mean=)`, an Erlang distribution given by its number of stages and its mean, which is `ss.gamma` with an integer shape (`stages=` on the transition says the same thing); and `ss.truncnorm(loc=, scale=, low=, high=)`, a normal distribution truncated to an interval.

A distribution is accepted wherever a quantity of the same type is. **Units are given once per distribution**:

```python
ss.lognorm_ex(mean=10, std=3, unit='days')       # the unit applies to both parameters
ss.lognorm_ex(mean=ss.days(10), std=3)           # std inherits the unit of mean
ss.lognorm_ex(mean=10, std=3)                    # no unit: typed by its slot, like a bare number
```

Putting the unit on one parameter is enough; the other parameters of the same dimension inherit it. Shape parameters (`a` in `ss.gamma`, `c` in `ss.weibull`) have no unit.

**In a compartmental run** there are no individuals to draw for, so a distribution is realized in one of two ways. A dwell-time distribution becomes Erlang sub-stages ([03-states-and-transitions.md](03-states-and-transitions.md#dwell-time)). A distribution on a quantity that varies between people, such as `beta` or `rel_sus`, makes the model agent-based under `solver='auto'`; under an explicit `solver='ode'` the mean is used, with a note, and a user who wants the spread kept bins it into a dimension ([05-dimensions.md](05-dimensions.md), [12-backends.md](12-backends.md)).

Random number streams and reproducibility are in [11-random.md](11-random.md).

## What is not typed

- **Counts.** Numbers of people, events, mosquitoes, or bacteria are plain numbers.
- **Money.** `cost=12.50` is a number.
- **Space.** Distances are numbers in the unit of the geography ([06-routes.md](06-routes.md)).
- **Other physical dimensions** (mass, concentration). A state that is not people, such as a water reservoir, holds a number whose unit is whatever the modeler means by it; the cholera example in [03](03-states-and-transitions.md#cholera-an-environmental-reservoir) uses cells/mL and says so in a comment.

Adding a type is a real cost to every user who has to learn it, so the set should stay at Starsim's four unless there is a strong case.

## Open questions

- Whether `ss.prob` keeps its period in the model spec or is normalized to a per-step value at build time. The first is more faithful; the second is smaller.
- **The rest of the type lattice.** Adding an `ss.per` to an `ss.freq` or `ss.prob` is an error (above). The rules for every other pair of types need writing down once rather than discovering per operator.
- **Can the Timeline be simpler?** Three views seem irreducible: the step index, time in the sim's own unit, and the calendar date. The other vectors (`yearvec`, `relvec`, `tvec`) could become conversions on request, `sim.t.to('year')`. Whether that is a real simplification needs trying in Starsim.
