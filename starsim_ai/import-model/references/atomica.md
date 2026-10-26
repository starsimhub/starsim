# Atomica

## What the source is

An Atomica model is a framework workbook (compartments, characteristics, parameters with their functions and formats, the transition matrix, interactions) and a databook (initial values and parameter values by population and year), both `.xlsx`, and maybe a program book (programs, costs, and their effects). It is run with `at.Project(framework=..., databook=...).run_sim()`.

## Mapping

| Atomica | Starsim | Mechanical |
|---|---|---|
| A compartment | A state (a source is `ss.null`; a sink such as the dead stays a state, as in Atomica's results) | yes |
| A characteristic | A named expression and a result (a sum of compartments, maybe over a denominator) | yes |
| A link in the transition matrix | A transition `<from>_to_<to>` (with `_<parameter>` if a link has several) | yes |
| Format `rate`, `probability` | A rate per year | yes |
| Format `duration` | `1 / par` | yes |
| A `duration` marked `Timed` | A fixed dwell time, `ss.constant(ss.years(d))` (a compartmental run approximates it by Erlang stages, which differs from Atomica by about 1e-5) | yes |
| Format `number` | The whole flow, divided among the source's people | yes |
| Format `proportion` (junctions) | Listed: a branch with `p=` by hand | no |
| A parameter function | An expression of counts (`ch_prev * transpercontact`); `floor(x)` written as `x` (listed if `x` is not whole) | yes |
| Databook values by year | `ss.timeseries(..., mode='linear')`, held constant outside, as Atomica does | yes |
| Initial numbers | Solved from the databook's compartments and characteristics at the first year, as Atomica does: those with a setup weight (by default, those in the databook or with a default), zeros fixed, least squares with the smallest norm | yes |
| Several population types (`Population types` sheet) | The first type as the disease; the others (each would be a disease of its own) and the interactions between them listed | no |
| `SRC_POP_AVG()`, `SRC_POP_SUM()` and the like, and a flow's value (`x*:ddis`) in a function | Listed (with the transitions that use them): an aggregate over interacting populations is a contact matrix, `ss.contacts(..., dims='population')` | no |
| Several populations | A stratification `population=[...]` with `kind='space'` (so `sus.n` is the count in each population), values as `ss.by(population=[...])` | yes |
| Atomica's step | `integrator='euler', substeps=1`, `dt` in years (Atomica's default 0.25) | yes |
| Programs (the program book) | Coverage from spending, and each affected parameter as baseline plus coverage times effect, by hand | no |
| Junctions | A branch: `ss.Transition(src, [a, b], rate=..., p=[pa, pb])` | no |
| Transfers between populations | `ss.transfer`, or `mobility=` on the population stratification | no |
| Interactions (a force of infection between populations) | A contact matrix, `ss.contacts([[...]], dims='population')` | no |

## What needs care

- Atomica moves `value × dt` (or `dt / duration`, or `number × dt`) each step, capped at everyone: the Euler step. The import uses `integrator='euler', substeps=1` so that it agrees to rounding error; a finer Starsim run is closer to the continuous model but not to Atomica.
- Run with the same `dt` as the reference (`ss.interop.load(..., dt=1)` for the LFD examples): at a different step the numbers differ by order `dt`.
- Atomica runs from the first year of data to five years after the last (or to 2035 if the model has births); the import does the same unless `start=` and `stop=` are given.

## Worked example

`lingua_franca_design/vignettes/atomica/hiv_dyn/model.py`: the HIV care cascade by sex, with programs and three budget scenarios. `ss.interop.load('hiv_dyn_framework.xlsx', databook='hiv_dyn_databook.xlsx', dt=1)` gives the baseline, which agrees with Atomica to 3e-14 in all 78 compartments, flows, and characteristics (the SIR demo to 1e-15); the programs are by hand, as in the vignette:

```python
coverage = ss.par(lambda sim: np.minimum(budget * spend / cost / eligible.n, 1))
switch   = (1 - programs)*before + programs*after                     # each affected parameter
sims     = dict(baseline=sim, double=sim.new(pars=dict(budget=ss.timeseries({2016: 1, 2020: 2}))))
```

## Running the original

In Atomica's environment (`lingua_franca_design/_envs/atomica`): `P = at.Project(framework=..., databook=..., do_run=False); P.settings.sim_dt = dt; result = P.run_sim()`. For each population, `comp.vals` for each compartment, `link.vals / dt` for each link (people per year), and `pop.get_variable(name)[0].vals` for characteristics; LFD's `examples/atomica/*/simulate.py` writes these in the long format. Stocks agree at the same index. Atomica records a link's flow at the start of a step and Starsim at its end, so compare Starsim's `res[transition].values[1:]` with Atomica's `link.vals[:-1] / dt` (`--shift -1`), adding the transitions of a link with several parameters. Expect agreement to rounding error; more means a format was translated wrongly.
