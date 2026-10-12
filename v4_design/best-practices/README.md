# Best practices

Distillation of the modeling landscape into best practices for each type of modeling.

## Purpose

Where `approaches` describes what frameworks do, this folder takes a position on what they *should* do. Each document argues for a particular way of expressing a modeling concept, with reference to how existing frameworks handle it and why one treatment is preferable.

The test for every recommendation is the project's north star: **a model should never be easier to write without the lingua franca than with it.** Concretely, each main framework's own examples should be implementable here in no more lines of model code and no more compute time than the original.

## How to read these

Every document follows the same shape:

1. **Recommendation**: the position, stated as imperatives, at the top.
2. **Why**: what the landscape review shows, with the frameworks named.
3. **The proposal**: the original framework's code, then the same thing in the lingua franca, side by side.
4. **Trade-offs**: what the recommendation gives up.
5. **Rejected**: things that looked good in isolation and should not be built.
6. **Open questions**: what is not yet settled.

Code in the "lingua franca" column is written as `starsim` (`import starsim as ss`), because [the base for the lingua franca is Starsim](../README.md) and because a proposal should be valid Python. Where a name is ambiguous, Starsim's existing name and behavior are used. Where Starsim already does the right thing, the document says so and moves on.

## Contents

### The principles

- [principles.md](principles.md): who the user is, what "simple" commits us to, when to guess a default, why plain Python is always available, and what we are deliberately not building.

### Cross-cutting concerns

- [model-structure.md](model-structure.md): **the central document.** States and transitions; one transition class written three ways (arrow strings, `S >> I @ beta * I`, `ss.Transition`); dwell-time distributions; branching.
- [time-and-units.md](time-and-units.md): Starsim's durations, rates, probabilities, and frequencies, unchanged; bare numbers that take the sim's unit; calendar dates; per-module timesteps.
- [parameters-and-distributions.md](parameters-and-distributions.md): `ss.par`, the forms a parameter can take (number to function), Starsim's distributions, data provenance, and where values live.
- [stratification.md](stratification.md): age, space, risk, strain, and vaccination status as one concept; `stratify` returns a new model; strain as a dimension or a second disease.
- [population-and-mixing.md](population-and-mixing.md): `networks=`; homogeneous mixing by default with the classical `beta`; what a state in a rate means; environmental and vector-borne transmission through counts and `ss.Compartments`.
- [composition-and-effects.md](composition-and-effects.md): `ss.modify` for scaling another module's quantity while a condition holds; plain `step()` code for anything more complex; Starsim's ordered loop.
- [interventions.md](interventions.md): an intervention is *what* plus *delivery*; eligibility written against real attributes; products, coverage versus capacity, cost, and scenarios.
- [results-and-observation.md](results-and-observation.md): one results concept: automatic counts and flows, ratios, reported quantities with `ss.outcome`, and data attached by name.
- [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md): per-distribution streams, common random numbers, run identity, optional caching.
- [calibration.md](calibration.md): `sim.calibrate()` with no likelihood required; the method chosen from the model; comparing models.

### By paradigm

- [compartmental.md](compartmental.md): deterministic ODE, the default for a purely compartmental model; `equations()`.
- [stochastic-compartmental.md](stochastic-compartmental.md): CTMC/Gillespie, tau-leaping, chain-binomial, SDE.
- [metapopulation.md](metapopulation.md): space as a dimension, with or without movement; mobility models and movement data.
- [agent-based.md](agent-based.md): Starsim's architecture, kept; what makes a model agent-based; aggregate quantities inside an agent-based sim.
- [paradigm-conversion.md](paradigm-conversion.md): choosing a run method; `method='auto'` never approximates; an explicitly requested method uses the standard approximation and says so; `methods()`.

## Relationship to other folders

- Draws on the reviews in [approaches](../approaches).
- Informs the specifications in [design](../design), which are based on best practices here.
