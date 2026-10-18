# Time and units

## Recommendation

1. **Keep Starsim's time types and their names.** `ss.dur` / `ss.per` / `ss.prob` as distinct types, with conversion as the type's responsibility, is the most-validated idea in the review. Nothing is renamed; the duplicate spellings are dropped (a probability per period is `ss.prob(0.1, 'day')`, and a count per unit time is `ss.time.freq`, for modules written as code).
2. **Calendar dates wherever they are wanted.** Interventions have dates, data has dates, and converting them to step indices is a recurring source of error, so a sim's start and stop, the times of interventions, and data can all be dates.
3. **Do not assume a global timestep.** Per-module timelines interleaved by absolute time already work.
4. **Accept a bare number; it takes the sim's unit.** A type system that makes `beta=0.05` an error will lose the audience. If nothing in the model has a unit, time is counted in timesteps and no unit is invented.
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

### A textbook model has no unit

Most of the frameworks surveyed (EpiModel, `individual`, LASER, epiworld, SimInf, odin, summer2, and epipack) count time in steps or in an unnamed unit, and that is what a textbook model means: `beta=0.3` per step, `dur=10` steps. Forcing such a model into dates or years adds something its author never said, which is why a model with no unit anywhere counts time in timesteps, and why no default unit is assumed (Starsim 3 assumed years).

### Print the conversion, do not make people write it

[odin](../approaches/odin.md) puts `dt` in scope so the modeler writes the conversion (`p_SI <- 1 - exp(-(beta * I / N * dt))`). Its review calls this "more transparent and less safe: nothing stops a modeler writing `p_SI <- beta * I / N * dt`, which is correct only for small `dt`, and nothing marks `beta` as a rate rather than a probability." Starsim makes the conversion the type's job (`beta.to_prob(dt)`). Both can be had: the library converts, and prints what it resolved, so odin's transparency comes from the printed model rather than from the source. [camdl](../approaches/camdl.md)'s verbosity is "the right trade for correctness and a real trade against approachability"; taking the kind of quantity from the argument slot and printing the result gives most of the correctness without the verbosity.

### Per-module timesteps

From the [Starsim review](../approaches/starsim.md): "a within-host progression module can run at `dt = 1 day` while a demographic module runs at `dt = 1 year` in the same sim. The `Loop` interleaves them by absolute time. Nothing else in the landscape does this." Contrast [EpiModel](../approaches/epimodel.md), where the network's dissolution coefficients are calibrated in time steps, so the network and the epidemic are locked to a common step by construction. [epymorph](../approaches/epymorph.md) has a different structure again, sub-step scheduling (`leaves=TickIndex(step=0)`, `returns=TickDelta(step=1, days=0)`: leave in the morning, mix at the destination, return in the evening). No sub-step mechanism is added for it: commuting is handled by the mixing ([metapopulation.md](metapopulation.md)), and a model that needs a day and a night phase can use a half-day timestep.

### Dimension checking catches the missing `/N`

camdl's `error[E300]: transition 'infection' rate has wrong dimension` with `rate = ((beta * S) * I)` is the missing `/N`, the check that justifies the whole apparatus. Here the common case cannot go wrong that way, because the user writes `S >> I @ beta * I` and the mixing supplies the denominator ([population-and-mixing.md](population-and-mixing.md)); a longer expression with typed values is still checked, and the error suggests the fix ([error messages are part of the language](principles.md)). Bare numbers cannot be checked this way, which is one reason to type the values in a model that mixes timescales.

## The specification

[02-types.md](../design/02-types.md) specifies the time types (`ss.dur`, `ss.per`, `ss.prob`, with `ss.days` to `ss.years`, `ss.perday` to `ss.peryear`, `ss.prob(0.1, 'day')`, `ss.datedur`, and `ss.date`), arithmetic between them, bare numbers and where the unit comes from, and time in a sim. Per-module timesteps and the ordered loop are in [10-execution.md](../design/10-execution.md), and the errors for meaningless arithmetic and wrong dimensions (`ss.IncompatibleTypes`, `ss.WrongDimension`) in [15-errors.md](../design/15-errors.md).

## Trade-offs

- **`ss.perday(0.05)` is more to type than `0.05`.** Which is why bare numbers work; the typed form is for when the model spans timescales or the units are not obvious.
- **Rate-to-probability conversion has corners at large `dt`.** Starsim's review flags this, particularly around calendar durations and `datedur` versus float-year arithmetic. The mitigation is a note when a step is large for a transition (`ss.LargeStep`), where the exponential conversion starts to matter and people's intuitions stop being reliable.
- **Per-module timesteps make the execution order harder to reason about.** Starsim's loop is an object that can be printed and plotted (`sim.loop`), which helps, but the printout is the fully unrolled list of every call at every time and is hard to read. So `describe()` shows the loop as a schedule instead: one line for each step size, with the modules in the order they run (`schedule  every day: randomnet, sir`), and the unrolled list is `sim.loop`.
- **Bare numbers skip the dimension check**, and an inferred unit can be the wrong one. Both are handled by printing: `describe()` shows every resolved value with its unit, and a unit taken from `dt`, the dates, or the typed values is announced when the sim is built. Typed values in different units alongside bare numbers are an error (`ss.IncompatibleTypes`) asking for `unit=`, since the unit of the bare numbers would be a guess.
- **A unitless model cannot be compared with dated data or given a dated intervention.** Doing either supplies a unit, at which point the bare numbers that were "per step" become "per day" (or whatever the dates imply), and the note says so.
- **Starsim's time classes were a lot** (its own review says so), and the duplicate spellings have been dropped. Resist adding more. If a new one seems necessary, check whether it is a `datedur` in disguise.

## Rejected

- **Unitless steps as the only kind of time** (EpiModel, `individual`, LASER, epiworld, SimInf). The cost, documented across five reviews, is that rates, probabilities, and counts are all unchecked numbers and changing the step size means rescaling by hand. A unitless model is fine; having no other kind is not.
- **A default unit when none is given** (an earlier draft assumed days; Starsim assumes years). Either one puts words in the modeler's mouth.
- **Making the user write `1 - exp(-r*dt)`** (odin, `individual`). Transparent and unsafe; we get the transparency from the printout.
- **Units as an optional annotation** ([SBML](../approaches/notes.md#sbml-systems-biology-markup-language)'s `unitDefinition`, which is in the standard and "widely ignored in practice"). Here the kind of quantity comes from the argument slot and the unit from the sim, so every value ends up with both whether or not the user typed them.
- **Renaming Starsim's types** (`ss.per` to `ss.rate`, for example). The current names work and existing code uses them.
- **Guessing the unit from the size of the numbers.** A default that depends on the values is harder to predict than one that does not. (One exception is kept from Starsim: a year-like `start` or `stop`, such as `stop=2030`, is a calendar year.)
- **A sub-step mechanism** (epymorph's tick offsets). See above.
- **A general dimensional algebra** (mass, length, currency). Epidemiology needs time⁻¹, dimensionless, and counts. Cost per unit of the intervention is a number with a currency label, not a dimension to check.

## Open questions

The open questions from this document are in [19-open-questions.md](../design/19-open-questions.md).
