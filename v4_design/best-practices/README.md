# Best practices

Distillation of the modeling landscape into best practices for each type of modeling.

## Purpose

Where `approaches` describes what frameworks do, this folder takes a position on what they *should* do. Each document argues for a particular way of expressing a modeling concept, with reference to how existing frameworks handle it and why one treatment is preferable.

## How to read these

Every document follows the same shape:

1. **Recommendation** — the position, stated as imperatives, at the top.
2. **Evidence** — what the landscape review shows, with the frameworks named.
3. **The proposal** — the original framework's code, then the same thing in the lingua franca, side by side.
4. **Trade-offs** — what the recommendation costs, and who pays.
5. **Rejected** — things that looked good in isolation and should not be built.

Code in the "lingua franca" column is written as `starsim` (`import starsim as ss`), because [the base for the lingua franca is Starsim](../README.md) and because a proposal that cannot be spelled in the host language is not a proposal. Where a spelling is speculative it is marked as such. Where Starsim already does the right thing, the document says so and moves on.

## Contents

### The doctrine

- [principles.md](principles.md) — who the user is, what "simple" commits us to, when to guess a default, where the escape hatches go, and what we are deliberately not building.

### Cross-cutting concerns

- [model-structure.md](model-structure.md) — **the central document.** States and transitions as declared data; the closed transition vocabulary; dwell-time distributions; branching.
- [time-and-units.md](time-and-units.md) — durations, rates, probabilities, frequencies; calendar dates; per-module timesteps.
- [parameters-and-distributions.md](parameters-and-distributions.md) — declaration, typing, defaults, distributions, data provenance, and where values live.
- [stratification.md](stratification.md) — age, space, risk, strain, and vaccination status as one concept; stratification as an operator.
- [population-and-mixing.md](population-and-mixing.md) — the route abstraction; separating transmissibility from contact; specifying structure by its generating model.
- [composition-and-effects.md](composition-and-effects.md) — how one module modifies another's quantities without last-write-wins; execution order; the module contract.
- [interventions.md](interventions.md) — the four-part decomposition; eligibility as a declarative predicate; products, waning, and cost.
- [results-and-observation.md](results-and-observation.md) — auto-generated results, characteristics, the reporting cascade, and the likelihood.
- [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md) — per-distribution streams, common random numbers, run identity.
- [calibration.md](calibration.md) — what to fit, how to declare it, and which method to reach for.

### By paradigm

- [compartmental.md](compartmental.md) — deterministic ODE.
- [stochastic-compartmental.md](stochastic-compartmental.md) — CTMC/Gillespie, tau-leaping, chain-binomial, SDE.
- [metapopulation.md](metapopulation.md) — patches, mobility, and movement data.
- [agent-based.md](agent-based.md) — agents, networks, scheduled events.
- [paradigm-conversion.md](paradigm-conversion.md) — the paradigm lattice, which conversions are exact, which need a named closure, and which the compiler must refuse.

## Relationship to other folders

- Draws on the reviews in [approaches](../approaches).
- Informs the specifications in [design](../design), which are based on best practices here.
