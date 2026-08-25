# Time and units

## Recommendation

1. **Keep Starsim's time type system essentially as-is.** `dur` / `per` / `prob` / `freq` as distinct types, with conversion as the type's responsibility, is the most-validated idea in the review.
2. **Calendar dates by default.** Interventions have dates, data has dates, and converting them to step indices is a recurring source of error.
3. **Do not assume a global timestep.** Per-module timelines interleaved by absolute time already work.
4. **Accept a bare number, and guess its unit from context.** A type system that makes `beta=0.05` an error will lose the audience.
5. **Print the conversion.** odin's transparency and Starsim's safety are not in conflict once the model is data.
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

### Accept bare numbers

The one change worth making to Starsim's treatment. This must work:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
sim = ss.Sim(sir, unit='day')
```

The argument slot knows what kind of quantity it wants: `beta=` is a rate, `dur=` is a duration, `p=` is a probability. The only missing information is the *unit*, and the sim's unit supplies it. The summary then prints what was resolved:

```text
infection: S -> I,  beta = 0.05/day  (rate; assumed unit=day)
recovery:  I -> R,  dwell ~ Exp(mean = 10 days)
```

This is the difference between a type system that helps and one that gatekeeps. [camdl](../approaches/camdl.md)'s verbosity is "the right trade for correctness and a real trade against approachability" — for our audience the balance falls the other way, and we can have both by *inferring* the type from the slot and *printing* the result.

Where the guess genuinely cannot be made, ask:

```python
ss.Sim(sir)   # no unit given, parameters are bare numbers
# → assumes unit='day', prints it. Correct 99% of the time for beta≈0.05, dur≈10.
```

Even here, guess by magnitude and print. Nothing about `dur=10` with `beta=0.05` suggests years.

### Print the conversion, do not make people write it

**odin** ([approaches/odin.md](../approaches/odin.md)) puts `dt` in scope so the modeler writes the conversion explicitly:

```r
p_SI <- 1 - exp(-(beta * I / N * dt))
```

Its review calls this "more transparent and less safe: nothing stops a modeller writing `p_SI <- beta * I / N * dt`, which is correct only for small `dt`, and nothing marks `beta` as a rate rather than a probability."

**Starsim** makes it the type's job: `beta.to_prob(dt)`.

**Lingua franca**: Starsim's mechanism, odin's transparency, recovered from the printed model rather than from the source:

```text
>>> sim.summary(verbose=True)
timestep: 1 day
infection: beta = 0.05/day  →  p = 1 - exp(-0.05 × 1) = 0.0488 per step
```

The transparency odin buys with modeler effort, we get from the model being data.

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

Relative time (`start=0, stop=100`) stays available for textbook models. Starsim's `Timeline` already exposes `tvec` / `tivec` / `timevec` / `yearvec` / `datevec` / `relvec`, which is a lot of parallel representations, and does mean output can be requested in whatever frame the data are in. Keep it; do not add to it.

### Per-module timesteps

From the [Starsim review](../approaches/starsim.md): "a within-host progression module can run at `dt = 1 day` while a demographic module runs at `dt = 1 year` in the same sim. The `Loop` interleaves them by absolute time. Nothing else in the landscape does this."

Keep it, and make it the *default behavior* rather than an option: a module that declares `dt=ss.years(1)` gets it, and a module that declares nothing inherits the sim's. Contrast [EpiModel](../approaches/epimodel.md), where the network's dissolution coefficients are calibrated in time steps, so the network and the epidemic are locked to a common step by construction.

[epymorph](../approaches/epymorph.md) adds a case worth supporting: **sub-step scheduling**. A movement clause declares `leaves=TickIndex(step=0)` and `returns=TickDelta(step=1, days=0)` — leave in the morning, mix at the destination, return in the evening. That is a genuinely different time structure from either a global `dt` or per-module timelines, and commuting needs it.

### Dimension checking

The check that justifies the whole apparatus, from [camdl](../approaches/camdl.md):

```text
error[E300]: transition 'infection' rate has wrong dimension
= note: rate = ((beta * S) * I)
```

In the lingua franca most rate expressions do not exist — the route computes the force of infection ([population-and-mixing.md](population-and-mixing.md)), so the missing-`/N` bug is structurally unavailable. Where a user writes an expression, it must still check:

```python
ss.progression('I -> R', rate=ss.peryear(0.1) * sir.I)
# ✓ per-year × count = count per year
ss.progression('I -> R', rate=ss.peryear(0.1) * ss.days(3))
# ✗ TimeError: rate expression for 'recovery' has dimension [dimensionless], expected [count/time]
#   rate = peryear(0.1) * days(3)
#   A rate times a duration is a probability. Did you mean p= rather than rate=?
```

Note the last line. [Error messages are part of the language](principles.md).

## Trade-offs

- **`ss.perday(0.05)` is more to type than `0.05`.** Which is why bare numbers work; the typed form is for when the model spans timescales or the units are not obvious.
- **Rate-to-probability conversion has corners at large `dt`.** Starsim's review flags this, particularly around calendar durations and `datedur` versus float-year arithmetic. The mitigation is a warning when `rate × dt > ~0.1`, where the exponential conversion starts to matter and people's intuitions stop being reliable.
- **Per-module timesteps make the execution order harder to reason about.** Starsim's `Loop` being an inspectable data object is what makes this tolerable; keep that.
- **Fifteen time classes is a lot** (Starsim's own review says so). Resist adding more. If a new one seems necessary, check whether it is a `datedur` in disguise.

## Rejected

- **Unitless integer steps** (EpiModel, `individual`, LASER, epiworld, SimInf). The cost is documented across five reviews.
- **Making the user write `1 - exp(-r*dt)`** (odin, `individual`). Transparent and unsafe; we get the transparency from the printout.
- **Units as an optional annotation** ([SBML](../approaches/notes.md#sbml-systems-biology-markup-language)'s `unitDefinition`, which is in the standard and "widely ignored in practice"). Optional typing is not typing. Ours is inferred from the argument slot, so it is never optional and never typed by hand.
- **A general dimensional algebra** (mass, length, currency). Epidemiology needs time⁻¹, dimensionless, and counts. Cost per unit of the intervention is a number with a currency label, not a dimension to check.

## Open questions

- What happens when a `ss.freq` and a `ss.per` are added? Probably an error, but the rules for the full type lattice need writing down once rather than discovering per operator.
- Sub-step scheduling (epymorph's leave/return) versus per-module timesteps: are these one mechanism or two? They feel like one — "this process happens at this offset within the step" — but the interleaving semantics need specifying.
- Continuous-time backends ([SimInf](../approaches/siminf.md), [EoN](../approaches/eon.md)) have no `dt` and therefore no conversion at all, which is a real correctness advantage. Does the *model* need to know? Probably not: see [paradigm-conversion.md](paradigm-conversion.md).
