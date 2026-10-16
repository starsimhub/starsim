# flepiMoP

## What the source is

A flepiMoP model is a configuration YAML plus CSV files beside it: `compartments` (a product of named dimensions, e.g. infection stage × vaccination), `subpop_setup` (a geodata CSV of places and populations, and a mobility CSV), `initial_conditions`, `seir` (parameters, transitions with `source`, `destination`, `rate`, `proportional_to`, and `proportion_exponent`, and the integration method), `seir_modifiers` and `scenarios`, `outcomes` (chained probabilities, delays, and durations), and `inference`. It is run with `flepimop simulate config.yml` (gempyor).

## Mapping

| flepiMoP | Starsim | Mechanical |
|---|---|---|
| `compartments` | One state for each combination, named as flepiMoP names them (`S_unvaccinated`) | yes |
| `subpop_setup.geodata` | A stratification `subpop` (kind `'space'`), split by the initial numbers | yes |
| `subpop_setup.mobility` | `ss.Mobility(matrix=0.5*amount/pop, kind='visit')` (travelers spend half the day away: gempyor's `percent_day_away`) | yes |
| `initial_conditions` (`SetInitialConditions`, `Default`) | `init=dict(...)`, by place as `ss.by(subpop=[...])` | yes |
| `seir.parameters` (numbers, arithmetic) | Parameters | yes |
| A parameter drawn from a distribution | Its mean (listed); a prior, and a draw for each run, by hand | no |
| A transition | One flow per combination of levels, written as flepiMoP's total, `rate × source^e × Π (group^e / N)`, then classified (so the force of infection is exposure, `beta * I`) | yes |
| `integration.method` | `rk4` and `euler` → `'ode'`, `stochastic` → `'binomial'` | yes |
| `SinglePeriodModifier`, `StackedModifier` | `ss.modify('<disease>.pars.<par>', multiply=..., start=..., stop=...)`, following `stacked_modifier_method` (`product`, `reduction_product`, `sum`) | yes |
| `scenarios` | The chosen one as `interventions` (default: the first); each scenario's modifiers in `sim.imported.scenarios` | yes |
| `outcomes` from incidence or another outcome | `ss.outcome(p * source, delay=ss.days(d), duration=ss.constant(...))`, with the probability as `p_<outcome>` | yes |
| A modifier for some places only | `where=` on the `ss.modify`, against the place | no |
| `MultiPeriodModifier`, `subpop_groups`, `outcome_modifiers` | One `ss.modify` per period, by hand | no |
| A parameter from a `timeseries` file | `ss.timeseries({date: value, ...})`, `ss.by` over places | no |
| `seeding` (importations on dates) | An `ss.Mobility(data=df)` register (rows with no `from` add people) | no |
| An outcome from a prevalence | A result (`p * I`), or custom code | no |
| `inference` | `sim.calibrate()` with the data | no |

## What needs care

- Several dimensions are imported as a flat product of states. A cleaner Starsim model keeps the stages as states and the other dimension as a stratification, with moves between its levels as `ss.transfer` and differences as `adjust=`; check the two agree with the validate-import skill before replacing one with the other.
- A stop date in flepiMoP is the last day that applies; Starsim's `stop=` is the first day that no longer does, so a modifier's stop is the end date plus one day (the import does this).
- flepiMoP draws parameters for each run ("slot"). To compare exactly, run Starsim with the values that a slot drew (`spar/*.spar.csv`, `hpar/*.hpar.csv`), as the vignette does.

## Worked example

`lingua_franca_design/vignettes/flepimop/vaccine_scenarios/model.py`: two provinces with vaccination and waning, three scenarios of stacked modifiers, and chained outcomes. The import of the two-province SEIR matches flepiMoP's reference to 1e-12 in every compartment of both provinces; the vaccine scenarios, run with the original's draws, match its means to 4e-12 in every stock, flow, and outcome. The draws, by hand:

```python
Ro = ss.par(2.5, prior=sps.truncnorm((1.1 - 2.5)/0.1, (3 - 2.5)/0.1, loc=2.5, scale=0.1))
sims = [sim.new(pars=dict(Ro=v)) for v in Ro.prior.rvs(10, random_state=1)]
```

## Running the original

Copy the project folder to a temporary one (flepiMoP reads and writes relative to it), then, with `_envs/flepimop/bin` first on the PATH (`tools.run(cmd, 'flepimop', cwd=...)`): `flepimop simulate config.yml --write-csv --no-write-parquet --jobs 1 --in-id ref --out-id ref`. The output is in `model_output/<name>_<scenario>/ref/`, one file per slot: `seir/*.seir.csv` (columns `date`, `mc_value_type` — `prevalence` for stocks, `incidence` for flows into each compartment — `mc_name`, then one column per place), `hosp/*.hosp.csv` (one column per outcome), and the drawn values in `spar/` and `hpar/`. A stock on day `k` is the number at the start of day `k` in both. flepiMoP's incidence on day `k` is the number during day `k`, while Starsim's flow at index `k` is the number in the step ending at `k`, so compare Starsim's flows from index 1 with flepiMoP's from day 0 (`--shift -1`). RK4 runs with the same step agree to rounding error; a stochastic run compares statistically.
