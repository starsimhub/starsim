---
name: starsim
description: Build, run, fit, and debug disease models with Starsim (agent-based and compartmental, in Python). Use when the user works with `import starsim`, or asks for /starsim.
---

# Starsim

Starsim is a framework for modeling diseases in Python. The same model runs as an ODE, as a stochastic compartmental model, or as agents (`solver='ode'`, `'binomial'`, `'abm'`, and others; by default Starsim chooses one and says why in `sim.describe()`).

Starsim 4 differs a lot from Starsim 3, and from what you may remember. Check names against the docs rather than working from memory (see "Where to read").

## A first model

```python
import starsim as ss

S, I, R = ss.states('S I R')
infection = S >> I @ 0.3*I      # From S to I, at a rate of 0.3 per day times the share infectious
recovery = I >> R @ ss.days(10) # From I to R, after 10 days on average
sir = ss.Disease(infection, recovery, init=dict(I=0.01), name='sir')
sim = ss.Sim(sir, total_pop=10_000, dur=ss.days(100))
sim.run()
sim.plot()
```

`S >> I @ rate` is a transition: a rate in terms of a state (`0.3*I`) is transmission, and a duration (`ss.days(10)`) is time spent in the state. Results are in `sim.results` (e.g. `sim.results.sir.I`), and `sim.to_df()` gives them as a dataframe.

Comparing scenarios is a matter of changing the base sim. `sim.new()` makes a new sim from the same inputs, with whatever is given changed:

```python
import starsim as ss

S, I, R = ss.states('S I R')
measles = ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), init_prev=0.001, name='measles')
base = ss.Sim(measles, total_pop=20_000, dur=ss.days(120), verbose=0)
vaccination = ss.vaccinate(efficacy=0.9, when=ss.days(5), coverage=0.7)
msim = ss.parallel({'No vaccine': base, 'Vaccine': base.new(interventions=vaccination)})
msim.summarize()
```

## Choosing an approach

- **Declared or coded.** Declare a model as states and transitions where possible: it can then run under any solver, be described, drawn, fitted with gradients, and exported. Write a module as code only for what can't be declared, and declare the rest.
- **Compartments or agents.** Leave `solver` unset at first. Use agents (`n_agents=`, or `solver='abm'`) when the question depends on individuals: a contact network, histories, or who infected whom. Use compartments for large populations and for fitting. Say which was chosen, and why, if it matters to the question.
- **Built-in or new.** Check `import starsim.library as ssl` for an existing disease or network (HIV, cholera, measles, household networks, ...) before writing one, and read what it assumes: these are simple models. Everything else public is at the top level (`ss.Sim`, `ss.Disease`); don't import submodules.
- **An existing model.** To bring one in from another framework, or take one out to another tool, use the `convert-model` skill. For Starsim 3 code, use `migrate-starsim3`.

## How to work

1. **Start with the smallest model** that has the structure of the question, with few people and a short run.
2. **Read what Starsim says.** A run prints what it ran and how, R0, and any notes. `sim.describe()` lists what each part means and what Starsim guessed (marked `[guessed: ...]`); it returns the text, so in a script, `print(sim.describe())`. Fix every guess that is wrong, rather than leaving it.
3. **Check it against what you expect:** R0, the size and timing of the peak, the population over time, and the units of each rate. A model that runs isn't yet a model that is right; many mistakes give no error.
4. **Add one thing at a time,** rerunning each time, so that any change in the results has one cause.
5. **Report with the assumptions.** Say which values came from the user, which from data, and which you chose. For a stochastic model, run several seeds (`ss.parallel(sim, n_runs=10)`) before reporting a difference.

Ask the user, rather than choosing, when a value would change the answer and has no source (e.g. a transmission rate for a specific setting), and before a run or fit likely to take many minutes.

## Default rules

- Give durations and rates with units (`ss.days(10)`, `ss.years(2)`, `ss.peryear(0.1)`), so that they are right whatever the time step. `I >> R @ 8` is a rate of 8, not 8 days.
- A plain number is in the sim's unit of time: `dur=2` in a sim in days is two days. Give the length of the run with units (`dur=ss.days(100)`), or the start and stop as dates; `stop=` is when the run ends, not its length.
- With a contact matrix or network, β is per contact, so adding one multiplies R0 by the number of contacts: rescale β.
- `init_prev` seeds the first infected state, so in an SEIR model nobody starts infectious. Use `init=dict(I=...)` to start people in a named state.
- The population is `total_pop=` under every solver; `n_agents=` is the number of agents, and giving it means a run as agents.
- Use real data by name where it exists: `ss.population('Kenya')`, `ss.contacts('Kenya')`, `ss.Births(birth_rate='Kenya')`.
- Use `ss.vaccinate()`, `ss.screen()`, and `ss.treat()` for programs, rather than writing an intervention as code.
- Use `sim.fit()` to fit: it fits each parameter that has bounds (`ss.par(0.5, bounds=[0.1, 1])`) to each result that has data, with no objective function to write.
- After a run, read `sim.diseases.x`, not the object you made: the sim copies its modules.
- In a module written as code, keep per-person values in `define_states()`, make distributions once in `__init__()` and draw from them rather than from `np.random`, and call `super()` in each method you override.
- Plot with `sim.plot()` and `msim.plot()` before writing plots of your own.

## Where to read

https://starsim.org/v4/llms.txt is the map: every tutorial and user guide page, with a summary and a link to its text, then every public name with its signature, summary, and defaults. It is for the latest release, which can differ from the version installed (`ss.__version__`); for that version's own docs of a name, run `starsim help Sim` (or `help(ss.Sim)`), which for any other term lists the names and pages that mention it. Read the page before writing that part of a model.

| Task | Pages |
| --- | --- |
| Learn Starsim from the start | `intro_concepts`, `t1_intro` |
| Write a disease | `building_diseases`, `building_parameters`, `building_time` |
| Write something as code | `building_modules` |
| Age groups, or any other strata | `building_stratification` |
| What is observed, rather than what happens | `building_outcomes` |
| Results and plots | `building_results`, `building_plotting` |
| Populations and data for a place | `populations_people`, `populations_data`, `populations_demographics` |
| Contacts and networks | `populations_networks` |
| Places and movement between them | `populations_geography` |
| Vaccines, screening, treatment | `interventions_products`, `interventions_interventions` |
| Cost-effectiveness | `interventions_costs` |
| Import or export a model | The `convert-model` skill |
| Scenarios and many runs | `running_scenarios` |
| Which solver, and why | `running_solvers` |
| Something looks wrong, or is slow | `running_debugging` |
| Fit to data | `calibration_fitting`, `calibration_diagnostics` |
| Find the best program | `calibration_optimization` |

Errors from Starsim say what to change; read the whole message before changing code.

## Performance

Starsim's numerical code is in Rust kernels, already compiled in the wheels on PyPI, or otherwise Numba kernels, compiled once per install. If `import starsim` prints a note that the kernels are not compiled yet, suggest that the user run `starsim init compile` once (it takes a few minutes); don't run it yourself unprompted.

## Skills

Load one of these only when the task needs it (`starsim ai show <skill>`). The files a skill refers to (e.g. `references/`) are beside it, in the folder that `starsim ai show` names.

- `convert-model`: translate a model to or from another framework, including the parts that do not map one to one, and check it against the original
- `migrate-starsim3`: upgrade Starsim 3 code to Starsim 4

<!-- V4-PREVIEW: restore the link above to docs.starsim.org/llms.txt for normal publishing -->
