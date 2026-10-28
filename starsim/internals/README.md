# Internals

How Starsim works under the hood, for developers rather than modelers. Each module is also at the top level, e.g. `ss.errors` and `ss.graph`.

- `graph.py`: the graph of a declared module (its states, transitions, and parameters), completed with defaults and checked (`ss.graph.Graph`)
- `presentation.py`: how a model is shown: `describe()`, `diagram()`, and `equations()`
- `expressions.py`: the expressions that rates, results, and conditions are made of (`ss.par`, counts, arithmetic), and how they are evaluated and printed
- `gradients.py`: the ODE run in JAX, for gradients (`sim.gradient()`), log-densities (`sim.log_density()`), and many runs at once
- `errors.py`: the errors and warnings that are classes (e.g. `ss.CapabilityError`, `ss.errors.AtBound`), and `ss.note()`
- `debugtools.py`: profiling, debugging, and mock objects for testing
- `cli.py`: the `starsim` command (`starsim init`, `starsim compile`, `starsim ai`)
- `assets/`: the fonts for plots (see its README)
