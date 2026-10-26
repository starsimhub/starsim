# Starsim

Starsim is a framework for modeling diseases in Python. The same model runs as an ODE, as a stochastic compartmental model, or as agents (`solver='ode'`, `'binomial'`, `'abm'`, and others; by default Starsim chooses one and says why in `sim.describe()`).

## Essentials

- Import it as `import starsim as ss`; the example and reference modules (HIV, cholera, measles, household networks, ...) are `import starsim.library as ssl`. Everything public is at the top level (`ss.Sim`, `ss.Disease`); don't import submodules.
- Declare a model as states and transitions, rather than writing it as code, where possible: it can then run under any solver, be described (`sim.describe()`), drawn (`sim.diseases.sir.diagram()`), fitted with gradients, and exported.

```python
import starsim as ss

S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.01), name='sir')
sim = ss.Sim(sir, total_pop=10_000, unit='day', stop=100)
sim.run()
sim.plot()
```

- `S >> I @ rate` is a transition: a rate in terms of a state (`0.3*I`) is transmission, and a duration (`ss.days(10)`) is time spent in the state. Run the same model as agents with `ss.Sim(..., solver='abm')`, and add a contact network with `networks='random'`.
- Durations and rates carry units (`ss.days(10)`, `ss.years(2)`, `ss.peryear(0.1)`), so they are right whatever the time step.
- Check a model before trusting it: `sim.describe()` lists what each part means and what Starsim guessed (marked `[guessed: ...]`); fix every guess that is wrong.
- Results are in `sim.results` (e.g. `sim.results.sir.I`); `sim.to_df()` gives them as a dataframe.

## Looking things up

- The API, with every public name's signature, summary and defaults: `help(ss.Sim)`, or https://docs.starsim.org/llms.txt (for the latest release; it can differ from the version installed, which `ss.__version__` gives).
- Tutorials and the user guide: https://docs.starsim.org.
- Errors from Starsim say what to change; read the whole message before changing code.

## Performance

Starsim compiles its numerical code with Numba once per install. If `import starsim` prints a note that it is not compiled yet, suggest that the user run `starsim compile` once (it takes a few minutes); don't run it yourself unprompted.

## Skills

Load one of these only when the task needs it (`starsim ai show <skill>`). The files a skill refers to (e.g. `references/`) are beside it, in the folder that `starsim ai show` names.

<!-- The list of skills is added here by `starsim ai show` -->
