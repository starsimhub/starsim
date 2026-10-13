# Best practices

Distillation of the modeling landscape into best practices for each type of modeling.

## Purpose

Where `approaches` describes what frameworks do, this folder takes a position on what they *should* do. Each document argues for a particular way of expressing a modeling concept, with reference to how existing frameworks handle it and why one treatment is preferable.

The test for every recommendation is the project's north star: **a model should never be easier to write without the lingua franca than with it.** Concretely, each main framework's own examples should be implementable here in no more lines of model code and no more compute time than the original. (Where the original calls a built-in written for one situation, the lines inside that built-in count as part of the original; see [principles.md](principles.md).)

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

- [model-structure.md](model-structure.md): **the central document.** States and transitions; one transition class written three ways (arrow strings, `S >> I @ beta * I`, `ss.Transition`); `ss.null` for inflows and outflows; states that are not people (a reservoir, a vector population); dwell-time distributions; branching.
- [time-and-units.md](time-and-units.md): Starsim's durations, rates, probabilities, and frequencies, unchanged; bare numbers that take the sim's unit, and time counted in timesteps when nothing in the model has a unit; calendar dates; per-module timesteps.
- [parameters-and-distributions.md](parameters-and-distributions.md): `ss.par`, the forms a parameter can take (number to function), Starsim's distributions, what data a module `requires`, data provenance, and where values live.
- [stratification.md](stratification.md): age, space, risk, strain, and vaccination status as one concept; `stratify` works in place, with a copy on request; people properties are stratified on the sim and disease-specific dimensions on the disease; strain as a dimension or a second disease.
- [population-and-mixing.md](population-and-mixing.md): `networks=`; homogeneous mixing by default with the classical `beta`; what a state in a rate means (`I` versus `I.n`); a `beta` per network where infectiousness depends on the route; environmental and vector-borne transmission through counts and states that are not people; network models run as compartmental models on request.
- [composition-and-effects.md](composition-and-effects.md): `ss.modify` for scaling another module's quantity while a condition holds; what reading and writing a modified quantity mean (effective value and base); plain `step()` code for anything more complex; write tracking on request; Starsim's ordered loop.
- [interventions.md](interventions.md): an intervention is *what* plus *delivery*; eligibility written against real attributes; `when=` as a function of the sim, and a class for any policy with memory; products, coverage versus capacity, cost, and scenarios.
- [results-and-observation.md](results-and-observation.md): one results concept: automatic counts and flows, ratios, stratified results as one result with dimensions, reported quantities with `ss.outcome`, data attached by name; a sim is one trajectory and a MultiSim is many; the infection log on request.
- [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md): per-distribution streams, common random numbers keyed on agent slots, `sim.hash()` as a supplement to the commit hash, and checking convergence in timestep and number of runs.
- [calibration.md](calibration.md): `sim.calibrate()` with no likelihood required; the normalized absolute error as the default goodness of fit; the fitting method chosen from the model; gradients by automatic differentiation for ODE models; staged fitting; comparing models.

### By paradigm

- [compartmental.md](compartmental.md): deterministic ODE, the default for a purely compartmental model; what `solver='auto'` does; Erlang stages for dwell times; `equations()`.
- [stochastic-compartmental.md](stochastic-compartmental.md): Gillespie, tau-leaping, chain-binomial, and SDE as named solvers; `solver='stochastic'`; reservoirs and vectors as states holding counts; extinction as the headline output.
- [metapopulation.md](metapopulation.md): space as a dimension, with or without movement; mobility models and movement data.
- [agent-based.md](agent-based.md): Starsim's architecture, kept; what makes a model agent-based; states that hold counts (a reservoir, mosquitoes) alongside agents.
- [paradigm-conversion.md](paradigm-conversion.md): choosing a solver; `solver='auto'` never approximates the mixing or drops individual-level structure, and may represent a dwell time with Erlang stages; an explicitly requested solver uses the standard approximation and says so; `approx=`; `solvers()`.

## Relationship to other folders

- Draws on the reviews in [approaches](../approaches).
- Informs the specifications in [design](../design), which are based on best practices here.
