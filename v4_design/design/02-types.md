# Types: time, rates, and quantities

Durations, rates, probabilities, and frequencies; how bare numbers get their units; and time in a sim.

Justified by [best-practices/time-and-units.md](../best-practices/time-and-units.md). These are Starsim's types and names, unchanged. There are two additions: a bare number is accepted in every slot and typed by the slot it appears in, and a model in which nothing has a unit counts time in timesteps.

## The types

| Type | Dimension | Means | Constructors |
|---|---|---|---|
| `ss.dur` | time | an elapsed duration | `ss.days`, `ss.weeks`, `ss.months`, `ss.years`, `ss.datedur` |
| `ss.per` | time⁻¹ | a hazard: events per unit time, continuously | `ss.perday`, `ss.perweek`, `ss.permonth`, `ss.peryear` |
| `ss.prob` | probability per period | the chance of an event within a period | `ss.probperday`, `ss.probperweek`, `ss.probpermonth`, `ss.probperyear` |
| `ss.freq` | time⁻¹ | an expected number of events per unit time | `ss.freqperday`, `ss.freqperweek`, `ss.freqpermonth`, `ss.freqperyear` |

`ss.per`, `ss.prob`, and `ss.freq` share the base class `ss.Rate`; all four types share `ss.TimePar`. `ss.per` is the hazard type and keeps its Starsim name.

`ss.per` and `ss.freq` have the same dimension and are different types. A hazard is the parameter of an exponential clock and converts to a per-step probability by `1 − exp(−r·dt)`; a frequency is an expected count and converts by `f·dt`. Mixing them up is a common quiet error, so adding one to the other raises `E101 IncompatibleTypes`, as Starsim does today.

`ss.datedur` is kept alongside `ss.months` because calendar durations are not fixed multiples of days.

Counts are plain numbers. `I.n` is the number of people in a state (the same number as the result `n_I`) and `I.frac` its fraction ([03-states-and-transitions.md](03-states-and-transitions.md)).

## Conversion

Converting to a timestep is the type's job, not the user's. For a step of length `dt`:

| From | To a per-step probability | To a per-step expected count |
|---|---|---|
| `ss.per(r)` | `1 − exp(−r·dt)` | `r·dt` |
| `ss.prob(p)` per period `T` | `1 − (1 − p)^(dt/T)` | |
| `ss.freq(f)` | | `f·dt` |
| `ss.dur(d)` | through the rate `1/d` | |

A solver MUST NOT substitute `r·dt` for `1 − exp(−r·dt)`.

**Large steps.** When `r·dt > 0.1` for a converted rate, the build prints `W102 LargeStep`, naming the transition, the rate, the timestep, and the difference between `r·dt` and `1 − exp(−r·dt)`. It is a note, not an error; `sim.check_dt()` shows how much the results change with a smaller step ([10-execution.md](10-execution.md)).

**Seeing the conversion.** `describe(verbose=True)` shows the per-step form of every converted quantity:

```text
timestep: 1 day
infection: beta = 0.05 per day  ->  p = 1 - exp(-0.05 x 1) = 0.0488 per step
recovery:  dwell ~ exponential(mean = 10 days)  ->  p = 0.0952 per step
```

odin gets this transparency by having the modeler write `1 - exp(-gamma*dt)` by hand; here it is printed from the model.

Continuous-time solvers (`'ode'`, `'gillespie'`) have no `dt` in this sense and do no conversion at all.

## Arithmetic

Starsim's rules, summarized:

| Left | Op | Right | Result | Note |
|---|---|---|---|---|
| `dur` | `+` `−` | `dur` | `dur` | units are converted |
| `dur` | `+` `−` `×` `÷` | number | `dur` | a bare number takes the duration's unit: `ss.days(3) + 2` is 5 days |
| `dur` | `÷` | `dur` | number | |
| `per` | `+` `−` | `per` | `per` | competing hazards add |
| `per` | `×` `÷` | number | `per` | |
| `per` | `×` | `dur` | number | the expected count over the period, not a probability |
| `freq` | `×` | `dur` | number | an expected count |
| `prob` | `+` `−` | `prob` | `prob` | |
| number | `÷` | `dur` | a rate | |
| number | `÷` | `per` | `dur` | |
| `per` | `+` `−` | `prob` or `freq` | error | `E101 IncompatibleTypes` |

The slot a result lands in is checked too. A rate multiplied by a duration is a plain number, and putting one in a `rate=` slot is usually a mistake, so `E102 WrongDimension` points at it:

```text
E102 WrongDimension: the rate for transition 'recovery' has no time dimension.
       rate = ss.peryear(0.1) * ss.days(3)
                               ^^^^^^^^^^^ multiplying by a duration removes the "per time"
     A rate times a duration is an expected count, not a rate. Did you mean p= instead of rate=?
```

This check applies only when every value in the expression is typed. An expression made of bare numbers is taken at its word.

## Bare numbers

**A bare Python number is accepted in every slot**, with no warning. The slot gives the dimension and the sim gives the unit.

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
| 2 | a typed `dt`, such as `dt=ss.weeks(1)` | the unit of `dt` |
| 3 | `start=` or `stop=` as a date, such as `start='2020-01-01'` | days |
| 3 | `start=` or `stop=` as a year, such as `start=2020, stop=2030` | years |
| 4 | typed values in the model, and nothing above (`dur=ss.days(10)`) | their unit; if they differ, the smallest, with a note naming the values that disagreed |
| 5 | none of these | none: time is in timesteps |

A module's own typed `dt` counts as a typed value under rule 4. A date on an intervention or in the data supplies days in the same way as a date in `start=`. The unit is never guessed from the size of the numbers: 0.05 is plausible per day and per year.

Whenever the unit comes from rules 2 to 4 it is a guess about what the bare numbers meant, so the sim says where it came from (`W101 UnitInferred`), at `init()` and in `describe()`:

```text
>>> ss.Sim(sir, start='2020-03-01', stop='2020-09-01').run()
note: unit='day', dt=1 day (from start and stop dates)
```

```text
infection: beta = 0.05 per day  [unit from start and stop dates]
```

The default timestep is one unit: one day, one year, or one step.

### Time in timesteps

If nothing has a unit (no `unit=`, no unit on any `dt`, no dates, no typed values), time is unitless and counted in timesteps. `beta=0.05` is per step, `dur=10` is ten steps, `stop=100` is 100 steps, results are indexed by step, and nothing is said or assumed about days or years:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
ss.Sim(sir, stop=100).run()    # 100 steps
```

This is what a textbook model means. No note is printed, because nothing was guessed; `describe()` shows `unit  none: time is in timesteps`.

- The timestep is one step.
- A unitless sim has no calendar, so it cannot be compared with dated data or given a dated intervention. Doing either supplies a unit, at which point the bare numbers that were per step become per day (or whatever the dates imply), and the note says so.
- This differs from Starsim, which today assumes years when nothing is specified.

## Time in a sim

```python
ss.Sim(sir, start='2020-01-01', stop='2021-06-30', dt=ss.days(1))   # calendar dates
ss.Sim(sir, start=2020, stop=2030, dt=ss.months(1))                 # years
ss.Sim(sir, stop=100, unit='day')                                   # relative time, in days
ss.Sim(sir, stop=100)                                               # relative time, in steps
```

All four are supported. A string is a calendar date; a bare number in `start=` is read as a year if it looks like one (`2020`) and as a relative time otherwise (`0`), and the reading is printed. Relative time works with or without a unit.

The sim's `Timeline` provides the parallel time vectors Starsim already has (`tvec`, `yearvec`, `datevec`, `timevec`, and so on), so output can be requested in the frame the data are in. Nothing is added to it; whether it can be reduced is an open question below.

## Per-module timesteps

A module may have its own timestep:

```python
ss.Sim([sir, ss.Births(dt=ss.years(1))], dt=ss.days(1))
```

Modules with different timesteps are interleaved by absolute time in the loop ([10-execution.md](10-execution.md)). A module with no `dt` uses the sim's. This is Starsim's mechanism, kept as it is.

There are no sub-step offsets. Commuting, which the first draft handled with departure and return times inside a step, is specified instead as the fraction of time a person spends in the destination patch; a commuter contributes to and is exposed in each patch in proportion to the time spent there. See [06-routes.md](06-routes.md).

## Distributions

Distributions use Starsim's names: `ss.normal`, `ss.lognorm_ex` (given the mean and standard deviation of the lognormal itself), `ss.lognorm_im` (given those of the underlying normal), `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, `ss.choice`, and the rest of Starsim's set. Any SciPy distribution can be wrapped with `ss.Dist(dist=...)`. `ss.erlang(k=, mean=)` is the one addition: an Erlang distribution given by its number of stages and its mean, which is `ss.gamma` with an integer shape; `stages=` on the transition says the same thing.

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
- **Typed values that disagree.** With `dur=ss.days(10)` in one place, `ss.peryear(0.02)` in another, a bare `beta=0.3`, and no `unit=` or `dt=`, rule 4 picks days and says so, which is a guess about what the bare number meant. An error asking for `unit=` may be the better answer.
- **Can the Timeline be simpler?** Three views seem irreducible: the step index, time in the sim's own unit, and the calendar date. The other vectors (`yearvec`, `relvec`, `tvec`) could become conversions on request, `sim.t.to('year')`. Whether that is a real simplification needs trying in Starsim.
