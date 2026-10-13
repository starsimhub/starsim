# Time and units

## Recommendation

1. **Keep Starsim's time types and their names.** `ss.dur` / `ss.per` / `ss.prob` / `ss.freq` as distinct types, with conversion as the type's responsibility, is the most-validated idea in the review. Nothing is renamed.
2. **Calendar dates by default.** Interventions have dates, data has dates, and converting them to step indices is a recurring source of error.
3. **Do not assume a global timestep.** Per-module timelines interleaved by absolute time already work.
4. **Accept a bare number; it takes the sim's unit.** A type system that makes `beta=0.05` an error will lose the audience. If nothing specifies a unit, assume days and print the assumption.
5. **Print the conversion.** odin's transparency and Starsim's safety are not in conflict when the library can print what it resolved.
6. **Dimension-check rate expressions at build time**, with camdl's error, not a shape mismatch.

## Why: five independent arrivals

This is finding 2 of the [landscape review](../approaches/README.md#cross-cutting-findings), and it is as settled as anything in it:

| Framework | Medium | Vocabulary |
|---|---|---|
| [Starsim](../approaches/starsim.md) | Python class hierarchy | `dur` / `per` / `prob` / `freq` |
| [camdl](../approaches/camdl.md) | Compiler dimensional types | `rate` / `probability` / `count` / `time` |
| [Atomica](../approaches/atomica.md) | A spreadsheet column | Probability / Duration / Rate / Number / Proportion |
| [EMULSION](../approaches/emulsion.md) | Three alternative YAML keys | `rate:` / `proba:` / `duration:` |
| [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) / [Catalyst.jl](../approaches/notes.md#modelingtoolkitjl--catalystjl) | SymPy units / `@unit_checks` | Symbolic units per concept |

Five media, five communities, one small vocabulary. Note especially that Atomica — the only non-programmer-facing tool in the review — made the same distinction. It is not a programmer's nicety.

The negative evidence is just as strong. [EpiModel](../approaches/epimodel.md): "everything is per-step, unitless, and unchecked; rates, probabilities, and counts are all `numeric`. Changing the step size is a manual rescaling exercise." [odin](../approaches/odin.md), [summer2](../approaches/summer2.md), [epipack](../approaches/epipack.md), [Epydemix](../approaches/epydemix.md), [epiworld](../approaches/epiworld.md), [`individual`](../approaches/individual.md), [LASER](../approaches/laser.md), [SimInf](../approaches/siminf.md), [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [flepiMoP](../approaches/flepimop.md), [MEmilio](../approaches/memilio.md): no dimensional types, anywhere.

The specific error this prevents: `0.1` per year could be a hazard, a probability, or an event count, and they convert to a per-timestep quantity by three different formulas (`1 - exp(-r·dt)`, rescaling, `r·dt`). Conflating them is one of the quietest common errors in the field.

## The proposal

### Keep what exists

Starsim already has this and it does not need redesigning:

```python
ss.days(10), ss.weeks(2), ss.months(3), ss.years(5), ss.datedur(months=1)   # durations
ss.peryear(0.1)      # a hazard; converts by 1 - exp(-r·dt)
ss.probperyear(0.1)  # a probability attached to a period; rescales
ss.freqperyear(3)    # a count of events per unit time; scales linearly
```

`ss.datedur` earns its place: it handles the month-length problem that float-year arithmetic gets wrong.

The names stay as they are. `ss.per` is the hazard type (base class `ss.Rate`, with `ss.perday`, `ss.perweek`, `ss.peryear`); it is not renamed to `ss.rate`. Existing Starsim code and the lingua franca use the same words.

### Accept bare numbers

This must work, with no warning:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
sim = ss.Sim(sir, unit='day')
```

The argument slot knows what kind of quantity it wants: `beta=` is a rate, `dur=` is a duration, `p=` is a probability. The only missing information is the *unit*, and the sim's unit supplies it. `sir.describe()` prints what was resolved:

```text
infection: S -> I,  beta = 0.05/day  (rate; unit from sim)
recovery:  I -> R,  dwell ~ Exp(mean = 10 days)
```

[camdl](../approaches/camdl.md)'s verbosity is "the right trade for correctness and a real trade against approachability". For our audience the balance falls the other way, and we can have both by taking the type from the slot and printing the result.

The same rule covers operator-form models: in `I >> R @ 0.1` the number is a rate per unit time in the sim's unit, and in `I >> R @ ss.days(10)` the duration is a mean dwell time.

If nothing specifies a unit (no `unit=`, no `dt=` with a unit, no dates, no typed values), the sim assumes days and says so:

```text
>>> ss.Sim(sir).run()
note: no time unit given; assuming unit='day', dt=1 day
```

Starsim's own default unit is the year, so this is a change from Starsim and is listed under open questions. Examples in this folder do not rely on the default: they give `unit=`, `dt=`, or typed values. [CK: if nothing has any units, then i think the unit should be in timesteps, not 'days'. as soon as anything has any units, we can use those, and print where we got them from.]

### Print the conversion, do not make people write it

**odin** ([approaches/odin.md](../approaches/odin.md)) puts `dt` in scope so the modeler writes the conversion explicitly:

```r
p_SI <- 1 - exp(-(beta * I / N * dt))
```

Its review calls this "more transparent and less safe: nothing stops a modeler writing `p_SI <- beta * I / N * dt`, which is correct only for small `dt`, and nothing marks `beta` as a rate rather than a probability."

**Starsim** makes it the type's job: `beta.to_prob(dt)`.

**Lingua franca**: Starsim's mechanism, odin's transparency, recovered from the printed model rather than from the source:

```text
>>> sim.describe(verbose=True)
timestep: 1 day
infection: beta = 0.05/day  →  p = 1 - exp(-0.05 × 1) = 0.0488 per step
```

odin gets this transparency by having the modeler write the conversion. Here the library does the conversion and shows it.

### Calendar dates

[Epydemix](../approaches/epydemix.md), Starsim, camdl (with `origin`), and [flepiMoP](../approaches/flepimop.md) support real dates. [odin](../approaches/odin.md), [summer2](../approaches/summer2.md), [EpiModel](../approaches/epimodel.md), [epipack](../approaches/epipack.md), [`individual`](../approaches/individual.md), [epiworld](../approaches/epiworld.md), and [LASER](../approaches/laser.md) do not, or do so awkwardly.

```python
sim = ss.Sim(sir, start='2020-01-01', stop='2021-06-30', dt=ss.days(1))
```

and both of these are legal, because both are natural:

```python
ss.Sim(sir, start=2020, stop=2030, dt=ss.months(1))          # float years
ss.Sim(sir, start='2020-03-15', stop='2020-06-01')           # dates
```

Relative time (`start=0, stop=100`) stays available for textbook models. Starsim's `Timeline` already exposes `tvec` / `tivec` / `timevec` / `yearvec` / `datevec` / `relvec`, which is a lot of parallel representations, and does mean output can be requested in whatever frame the data are in. Keep it; do not add to it. [CK: if we can simplify the Timeline, i'd appreciate it! but i'm not sure we can, since dates/times are messy.]

### Per-module timesteps

From the [Starsim review](../approaches/starsim.md): "a within-host progression module can run at `dt = 1 day` while a demographic module runs at `dt = 1 year` in the same sim. The `Loop` interleaves them by absolute time. Nothing else in the landscape does this."

Keep it, and make it the *default behavior* rather than an option: a module that declares `dt=ss.years(1)` gets it, and a module that declares nothing inherits the sim's. Contrast [EpiModel](../approaches/epimodel.md), where the network's dissolution coefficients are calibrated in time steps, so the network and the epidemic are locked to a common step by construction.

[epymorph](../approaches/epymorph.md) has a different time structure again: **sub-step scheduling**. A movement clause declares `leaves=TickIndex(step=0)` and `returns=TickDelta(step=1, days=0)`: leave in the morning, mix at the destination, return in the evening. The lingua franca does not add a sub-step mechanism for this. Commuting is handled by the mixing ([metapopulation.md](metapopulation.md)), and a model that needs a day and a night phase can use a half-day timestep.

### Dimension checking

The check that justifies the whole apparatus, from [camdl](../approaches/camdl.md):

```text
error[E300]: transition 'infection' rate has wrong dimension
= note: rate = ((beta * S) * I)
```

That error is the missing `/N`. In the lingua franca the common case cannot go wrong this way, because the user writes `S >> I @ beta * I` and the mixing supplies the denominator ([population-and-mixing.md](population-and-mixing.md)). Where a user writes a longer expression with typed values, it is still checked:

```python
ss.progression('I -> R', rate=ss.peryear(0.1))
# fine: a rate per person per year
ss.progression('I -> R', rate=ss.peryear(0.1) * ss.days(3))
# TimeError: the rate for 'recovery' has no time dimension; expected [1/time]
#   rate = peryear(0.1) * days(3)
#   A rate times a duration is a probability. Did you mean p= rather than rate=?
```

The last line of the message suggests the fix. [Error messages are part of the language](principles.md). Bare numbers cannot be checked this way, which is one reason to type the values in a model that mixes timescales.

## Trade-offs

- **`ss.perday(0.05)` is more to type than `0.05`.** Which is why bare numbers work; the typed form is for when the model spans timescales or the units are not obvious.
- **Rate-to-probability conversion has corners at large `dt`.** Starsim's review flags this, particularly around calendar durations and `datedur` versus float-year arithmetic. The mitigation is a warning when `rate × dt > ~0.1`, where the exponential conversion starts to matter and people's intuitions stop being reliable.
- **Per-module timesteps make the execution order harder to reason about.** Starsim's loop being an object that can be printed and plotted (`sim.loop`) is what makes this manageable; keep that. [CK: yes, but it's hard to interpret in practice ... we should think about how to make it as clear as possible.]
- **Bare numbers skip the dimension check**, and a default unit can be the wrong one. Both are handled by printing: `describe()` shows every resolved value with its unit, and an assumed unit is announced at run time.
- **Fifteen time classes is a lot** (Starsim's own review says so). Resist adding more. If a new one seems necessary, check whether it is a `datedur` in disguise.

## Rejected

- **Unitless integer steps** (EpiModel, `individual`, LASER, epiworld, SimInf). The cost is documented across five reviews. [CK: actually, i think this would be nice. for theoretical models, forcing it into dates or years feels awkward at best.]
- **Making the user write `1 - exp(-r*dt)`** (odin, `individual`). Transparent and unsafe; we get the transparency from the printout.
- **Units as an optional annotation** ([SBML](../approaches/notes.md#sbml-systems-biology-markup-language)'s `unitDefinition`, which is in the standard and "widely ignored in practice"). Here the kind of quantity comes from the argument slot and the unit from the sim, so every value ends up with both whether or not the user typed them.
- **Renaming Starsim's types** (`ss.per` to `ss.rate`, for example). The current names work and existing code uses them.
- **Guessing the unit from the size of the numbers.** A default that depends on the values is harder to predict than one that does not.
- **A sub-step mechanism** (epymorph's tick offsets). See above.
- **A general dimensional algebra** (mass, length, currency). Epidemiology needs time⁻¹, dimensionless, and counts. Cost per unit of the intervention is a number with a currency label, not a dimension to check.

## Open questions

- What happens when a `ss.freq` and a `ss.per` are added? Probably an error, but the rules for the full type lattice need writing down once rather than discovering per operator. [CK: agree]
- **The default unit.** This document assumes days when nothing specifies a unit, because most textbook models and most outbreak models are written in days. Starsim's own default is the year. One of the two has to give, or the default has to depend on something stated (for example, years when `start=2020` and days when `start='2020-01-01'`). [CK: i think we should use integer timesteps ... what's the most common answer among the models surveyed?]
- Continuous-time methods ([SimInf](../approaches/siminf.md)'s, [EoN](../approaches/eon.md)'s; here `method='ctmc'` and `'ode'`) have no rate-to-probability conversion at all, which is a real correctness advantage. Does the *model* need to know? Probably not: see [paradigm-conversion.md](paradigm-conversion.md).
