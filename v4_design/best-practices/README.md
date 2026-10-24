# Best practices

Distillation of the modeling landscape into best practices for each type of modeling.

## Purpose

Where `approaches` describes what frameworks do, this folder takes a position on what they *should* do. Each document argues for a particular way of expressing a modeling concept, with reference to how existing frameworks handle it and why one treatment is preferable.

The test for every recommendation is the project's north star: **a model should never be easier to write without the lingua franca than with it.** Concretely, each main framework's own examples should be implementable here in no more lines of model code than the original (the hard target), and in about the same time (a time fails only if it is both more than twice as slow and more than 1 s slower, on the first run or the best of five). Where the original calls a built-in written for one situation, the lines inside that built-in count as part of the original; see [principles.md](principles.md).

## How to read these

These documents are the argument for the design; [design](../design) is the specification, and says each rule once. Every document follows the same shape:

1. **Recommendation**: the position, stated as imperatives, at the top.
2. **Why**: what the landscape review shows, with the frameworks named.
3. **The specification**: a few lines pointing to the chapters of [design](../design) that specify what the recommendation became. The code lives there, not here, so that the two cannot drift apart.
4. **Trade-offs**: what the recommendation gives up.
5. **Rejected**: things that looked good in isolation and should not be built, with the reasons.

What is not yet settled is in [design/19-open-questions.md](../design/19-open-questions.md), not in these documents. Code quoted from other frameworks (and from Starsim 3) stays where it shows what the landscape does. Where Starsim already does the right thing, the document says so and moves on.

## Contents

### The principles

- [principles.md](principles.md): who the user is, what "simple" commits us to, when to guess a default, why plain Python is always available, and what we are deliberately not building.

### Cross-cutting concerns

- [model-structure.md](model-structure.md): **the central document.** States and transitions; one transition class written three ways (arrow strings, `S >> I @ beta * I`, `ss.Transition`); `ss.null` for inflows and outflows; states that are not people (a reservoir, a vector population); dwell-time distributions; branching.
- [time-and-units.md](time-and-units.md): Starsim's durations, rates, and probabilities (`ss.dur`, `ss.per`, `ss.prob`); bare numbers that take the model's unit, and time counted in timesteps when nothing in the model has a unit; calendar dates; per-module timesteps.
- [parameters-and-distributions.md](parameters-and-distributions.md): `ss.par`, the forms a parameter can take (number to function), Starsim's distributions, what data a module `requires`, data provenance, and where values live.
- [stratification.md](stratification.md): age, space, risk, strain, and vaccination status as one concept; `stratify` works in place, with a copy on request; people properties are stratified on the sim and disease-specific dimensions on the disease; strain as a dimension or a second disease.
- [population-and-mixing.md](population-and-mixing.md): `networks=`; homogeneous mixing by default with the classical `beta`; what a state in a rate means (`I` versus `I.n`); a `beta` per network where infectiousness depends on the route; environmental and vector-borne transmission through counts and states that are not people; random and static networks run as compartmental models on request (sexual networks do not yet have a compartmental form).
- [composition-and-effects.md](composition-and-effects.md): `ss.modify` for scaling another module's quantity while a condition holds; what reading a modified quantity gives (the base value, with `.effective` for the value the modifiers make); plain `step()` code for anything more complex; write tracking on request; Starsim's ordered loop.
- [interventions.md](interventions.md): an intervention is *what* plus *delivery*; eligibility written against real attributes; `when=` as a function of the sim, `hold=` for a rule that stays on for a while, and a class for any other policy with memory; products, coverage versus capacity, cost, and scenarios.
- [results-and-observation.md](results-and-observation.md): one results concept: automatic counts and flows, ratios, stratified results as one result with dimensions, reported quantities with `ss.outcome`, data attached by name; a sim is one trajectory and a MultiSim is many (`ss.parallel`, or `sim.run(n_runs=)`); the infection log on request.
- [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md): per-distribution streams, common random numbers keyed on agent slots, a stream of its own for each module's stochastic compartmental run, `sim.hash()` as a supplement to the commit hash, and checking convergence in timestep (`sim.check_dt()`).
- [calibration.md](calibration.md): `sim.fit()` with no likelihood required; the normalized absolute error as the default goodness of fit; the fitting method chosen from the model (and a posterior from NUTS, particle MCMC, or ABC-SMC where one can be had); gradients by automatic differentiation for ODE models; staged fitting; sensitivity; comparing models.

### By paradigm

- [compartmental.md](compartmental.md): deterministic ODE, the default for a purely compartmental model; what `solver='auto'` does; Erlang stages for dwell times; `equations()`.
- [stochastic-compartmental.md](stochastic-compartmental.md): Gillespie (with tau-leaping as its fallback for long runs) and chain-binomial as named solvers; reservoirs and vectors as states holding counts, which stay deterministic; extinction as the headline output (`msim.extinction()`).
- [metapopulation.md](metapopulation.md): space as a dimension, with or without movement; mobility models and movement data.
- [agent-based.md](agent-based.md): Starsim's architecture, kept; what makes a model agent-based; states that hold counts (a reservoir, mosquitoes) alongside agents.
- [paradigm-conversion.md](paradigm-conversion.md): choosing a solver; `solver='auto'` never approximates the mixing or drops individual-level structure, and may represent a dwell time with Erlang stages; an explicitly requested solver uses the standard approximation and says so; `approx=`; `solvers()`.

## Relationship to other folders

- Draws on the reviews in [approaches](../approaches).
- Informs the specifications in [design](../design), which are based on best practices here.
