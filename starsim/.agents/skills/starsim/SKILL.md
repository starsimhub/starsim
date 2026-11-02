---
name: starsim
description: Build, run, fit, and debug disease models with Starsim (agent-based and compartmental, in Python). Use when the user works with `import starsim`, or asks for /starsim.
---

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

## Common mistakes

- `I >> R @ 8` is a rate of 8, not 8 days: give durations with their unit (`ss.days(8)`). See `running_debugging`.
- A plain number is in the sim's unit of time, so `stop=2` in a sim in days is two days. See `running_debugging`.
- With a contact matrix or network, β is per contact, so adding one multiplies R0 by the number of contacts. See `populations_networks`.
- `init_prev` in an SEIR model starts people in E, where nobody is infectious; use `init=dict(I=...)`. See `running_debugging`.
- After a run, read `sim.diseases.x`, not the object you made: the sim copies its modules. See `running_sim`.
- Keep per-person values in `define_states()`, draw from `ss` distributions rather than `np.random`, and call `super()`. See `building_modules`.

## Looking things up

- Read the docs rather than working from memory: Starsim changes between versions. https://starsim.org/v4/llms.txt is the map: every tutorial and user guide page, with a summary and a link to its text, then every public name with its signature, summary, and defaults. It is for the latest release, which can differ from the version installed (`ss.__version__`); for that version's own docs of a name, use `help(ss.Sim)`.
- Pages to start from: `running_debugging` (symptoms, how to check a model, and common pitfalls), `building_diseases` (declared diseases), `building_modules` (modules written as code), `calibration_fitting` (fitting to data), and `running_solvers` (which solver, and why).
- Errors from Starsim say what to change; read the whole message before changing code.

## Performance

Starsim's numerical code is in Rust kernels, already compiled in the wheels on PyPI, or otherwise Numba kernels, compiled once per install. If `import starsim` prints a note that the kernels are not compiled yet, suggest that the user run `starsim compile` once (it takes a few minutes); don't run it yourself unprompted.

## Skills

Load one of these only when the task needs it (`starsim ai show <skill>`). The files a skill refers to (e.g. `references/`) are beside it, in the folder that `starsim ai show` names.

- `import-model`: translate a model from another framework into Starsim, including the parts that do not map one to one
- `validate-import`: check a port against the original, by running both and comparing their results
- `migrate-starsim3`: upgrade Starsim 3 code to Starsim 4

<!-- V4-PREVIEW: restore the link above to docs.starsim.org/llms.txt for normal publishing -->
