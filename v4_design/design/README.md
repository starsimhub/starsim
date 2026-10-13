# Design

The specification of the epi modeling lingua franca, following the practices argued for in [best-practices](../best-practices).

## Purpose

Where `approaches` describes what existing frameworks do and `best-practices` takes a position on what they should do, `design` says what this language is: the objects a model is made of, what they mean, how a model is run and fitted, how it is saved, what the library says when something is wrong, and the tests that hold the design to its claims.

The measure of success is the north star from the [root README](../README.md): **a model should never be easier to write without the lingua franca than with it.**

## How to read these

The documents are a specification. MUST, SHOULD, and MAY are used where a requirement is real, and sparingly:

- **MUST** / **MUST NOT**: required for the behavior to be correct.
- **SHOULD**: recommended, with exceptions.
- **MAY**: optional.

The general policy is to guess a sensible default and print it, not to raise an error.

Each document starts with a pointer to the `best-practices` document that justifies it. Code is written against Starsim (`import starsim as ss`), because [the base for the lingua franca is Starsim](../README.md); where the specification differs from Starsim's current behavior, it says so. Decisions that are still guesses are collected in [19-open-questions.md](19-open-questions.md).

## Contents

### Start here

- [00-overview.md](00-overview.md): the whole language in one example, what a model is made of, and what happens at build time.

### The language

- [01-objects.md](01-objects.md): the class hierarchy, capitalization, constructing a model, how things get their names, `ss.Sim` (one sim is one run; many runs are an `ss.MultiSim`), and `describe()`.
- [02-types.md](02-types.md): Starsim's durations, rates, probabilities, and frequencies; how bare numbers get their units, and time in timesteps when nothing has a unit; time in a sim; distributions.
- [03-states-and-transitions.md](03-states-and-transitions.md): the core. States, the one transition class and its three spellings, rates that mention states, dwell times, branching, functions in any slot, `ss.null`, and states that are not people (reservoirs and vector populations).
- [04-parameters.md](04-parameters.md): `ss.par`, the forms a value can take, values by stratum, values kept outside the model, and named data.
- [05-dimensions.md](05-dimensions.md): stratification by age, space, risk, or strain; `stratify` on a model or on the whole sim, in place by default; `ss.transfer`.
- [06-routes.md](06-routes.md): who meets whom. The force of infection, networks and mixing pools, counts used literally, layers, mobility, and geography.
- [07-effects.md](07-effects.md): interactions between modules. `ss.modify`, conditions, and plain code in `step()`.
- [08-interventions.md](08-interventions.md): products, delivery, follow-ups, `ss.change`, triggers, and scenarios.
- [09-results.md](09-results.md): automatic results, stratified results as one result with dimensions, declared results and `ss.outcome`, data attached by name, results of many runs, and the transmission log.

### Running and fitting

- [10-execution.md](10-execution.md): the ordered loop, what runs when, competing hazards, per-module timesteps, and building the sim.
- [11-random.md](11-random.md): random number streams, common random numbers, the run hash, and convergence checks.
- [12-backends.md](12-backends.md): solvers. `solver='auto'`, the solvers, the standard approximations and how they are reported, `solvers()`, and the try-it rule for functions.
- [13-inference.md](13-inference.md): `sim.calibrate()`, goodness of fit, how the fitting method is chosen, sensitivity, and model comparison.

### Machinery

- [14-model-spec.md](14-model-spec.md): the model spec, the JSON form of a model: its structure, rate expressions, functions, normalization, hashing, and versioning.
- [15-errors.md](15-errors.md): errors, warnings, and notes: the message shape and the list of every code.
- [16-interop.md](16-interop.md): importing from and exporting to other frameworks, and the vignette test.

### Discipline

- [17-vocabulary.md](17-vocabulary.md): every public name, counted; the capitalization rule; keywords and slots; what was removed since the first draft.
- [18-conformance.md](18-conformance.md): the test suites, including the north-star test, and the corpus of models they run on.
- [19-open-questions.md](19-open-questions.md): what this revision settled, what is open, what is deferred, and the risks.

## What the specification claims

1. **The model is data.** States, transitions, and parameters are objects, so the diagram, the equations, the stratified model, the hash, and the saved file all come from the same thing. [14-model-spec.md](14-model-spec.md)
2. **The solver is chosen for you and can be changed.** `solver='auto'` picks the simplest solver that represents the model faithfully, an ODE if it can be one and agents otherwise, and says which. It never approximates the mixing; the one thing it does unasked is represent a non-exponential dwell time by Erlang stages, and it says so. `solver=` asks for another. [12-backends.md](12-backends.md)
3. **Approximations are named when they happen.** Asking for a solver the model does not natively fit uses the standard approximation and prints one line saying so; `approx=True` silences the line, and `approx=False` makes the approximation an error instead. [12-backends.md](12-backends.md#approximations)
4. **Defaults are guessed and printed.** `describe()` shows every guess, marked as one. [01-objects.md](01-objects.md#describe)
5. **The vocabulary is small and counted.** [17-vocabulary.md](17-vocabulary.md)
6. **A model is never easier to write without the lingua franca than with it.** For each vignette, the lingua franca version takes no more lines of model code and no more run time than the original framework's. [18-conformance.md](18-conformance.md#13-north-star)

Each claim has a test suite in [18-conformance.md](18-conformance.md).

## Relationship to other folders

- Justified by [best-practices](../best-practices).
- Informed by the reviews in [approaches](../approaches).
- Exercised by [vignettes](../vignettes), which translate the [examples](../examples) corpus into this specification. Each translation is tested for results, length, and speed against its original, and where it falls short the vignette says why.
