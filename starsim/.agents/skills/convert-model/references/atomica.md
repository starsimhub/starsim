# Atomica

## What the source is

An Atomica model is a framework workbook (compartments, characteristics, parameters with their functions and formats, the transition matrix, interactions) and a databook (initial values and parameter values by population and year), both `.xlsx`, and maybe a program book (programs, costs, and their effects). It is run with `at.Project(framework=..., databook=...).run_sim()`.

## Mapping

| Atomica | Starsim | Mechanical |
|---|---|---|
| A compartment | A state (a source is `ss.dead`; a sink such as the dead stays a state, as in Atomica's results) | yes |
| A characteristic | A named expression and a result (a sum of compartments, maybe over a denominator) | yes |
| A link in the transition matrix | A transition `<from>_to_<to>` (with `_<parameter>` if a link has several) | yes |
| Format `rate`, `probability` | A rate per year | yes |
| Format `duration` | `1 / par` | yes |
| A `duration` marked `Timed` | A fixed dwell time, `ss.constant(ss.years(d))` (a compartmental run approximates it by Erlang stages, which differs from Atomica by about 1e-5) | yes |
| Format `number` | The whole flow, divided among the source's people | yes |
| Format `proportion` (junctions) | The shares of a branch (below) | yes |
| A parameter function | An expression of counts (`ch_prev * transpercontact`); `floor(x)` written as `x` (listed if `x` is not whole) | yes |
| Databook values by year | `ss.timeseries(..., interpolation='linear')`, held constant outside, as Atomica does | yes |
| Initial numbers | Solved from the databook's compartments and characteristics at the first year, as Atomica does: those with a setup weight (by default, those in the databook or with a default), zeros fixed, least squares with the smallest norm | yes |
| Several population types (`Population types` sheet) | The first type as the disease; the others (each would be a disease of its own) and the interactions between them listed | no |
| `SRC_POP_AVG()`, `SRC_POP_SUM()`, where every pair of populations interacts alike | A count in all of them (`I.total`) | yes |
| Those that are not alike, and a flow's value (`x*:ddis`) in a function | Listed (with the transitions that use them): an aggregate over interacting populations is a contact matrix, `ss.contacts(population=...)` | no |
| Several populations | A stratification `population=ss.levels([...], kind='space')` (so `sus.n` is the count in each population), values as `ss.by(population=[...])` | yes |
| Atomica's step | `solver=ss.ode(integrator='euler', substeps=1)`, `dt` in years (Atomica's default 0.25) | yes |
| Programs (a program book, `progbook=`) | A parameter of spending for each program (`spend_<program>`, listed in `sim.imported.programs`), its coverage from spending and unit cost, and each affected parameter moved towards the programs' outcomes by coverage (additive interaction only; others are listed) | yes |
| A calibration (`calibration=`, Atomica's saved y-factors) | Each value scaled by its y-factor | yes |
| Junctions | A branch: `ss.Transition(src, {a: pa, b: pb}, rate=...)` | yes |
| Transfers between populations (rates, probabilities, durations) | `ss.transfer` on the population stratification | yes |
| Interactions (a force of infection between populations) | A contact matrix, `ss.contacts(population=[[...]])` | no |

## What needs care

- Atomica moves `value × dt` (or `dt / duration`, or `number × dt`) each step, capped at everyone: the Euler step. The import uses `solver=ss.ode(integrator='euler', substeps=1)` so that it agrees to rounding error; a finer Starsim run is closer to the continuous model but not to Atomica.
- Run with the same `dt` as the reference (`ss.import_from(..., dt=1)` for the worked examples): at a different step the numbers differ by order `dt`.
- Atomica runs from the first year of data to five years after the last (or to 2035 if the model has births); the import does the same unless `start=` and `stop=` are given.

## Worked example

[`atomica/hiv_dyn/model.py`](https://github.com/starsimhub/lingua_franca/blob/main/docs/models/atomica/hiv_dyn/model.py): the HIV care cascade by sex, with programs and three budget scenarios. `ss.import_from('hiv_dyn_framework.xlsx', databook='hiv_dyn_databook.xlsx', dt=1)` gives the baseline, which agrees with Atomica to 3e-14 in all 78 compartments, flows, and characteristics (the SIR demo to 1e-15). The worked example writes the programs by hand:

```python
coverage = ss.par(lambda sim: np.minimum(budget * spend / cost / eligible.n, 1))
switch   = (1 - programs)*before + programs*after                     # each affected parameter
sims     = dict(baseline=sim, double=sim.new(pars=dict(budget=ss.timeseries({2016: 1, 2020: 2}))))
```

## Running the original

In Atomica's own environment: `P = at.Project(framework=..., databook=..., do_run=False); P.settings.sim_dt = dt; result = P.run_sim()`. For each population, `comp.vals` for each compartment, `link.vals / dt` for each link (people per year), and `pop.get_variable(name)[0].vals` for characteristics; the [worked examples](https://github.com/starsimhub/lingua_franca/tree/main/docs/models/atomica)' `simulate.py` write these in the long format. Stocks agree at the same index. Atomica and Starsim both record a link's flow at the start of its step, so compare Starsim's `res[transition]` with Atomica's `link.vals / dt` at the same index, with no shift, adding the transitions of a link with several parameters. Expect agreement to rounding error; more means a format was translated wrongly.
