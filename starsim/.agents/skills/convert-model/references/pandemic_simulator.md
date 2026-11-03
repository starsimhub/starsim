# Pandemic Simulator

## What the source is

The World Health Organization's [Pandemic Simulator](https://github.com/WHO-Collaboratory/pandemic-simulator-compartment) holds compartmental models in JAX, each a directory with `model.py`, `example-config.json`, and `model.md`. A model is a class with a declared schema (`define_parameters()`) and a hand-written `equation()`. The values are in the JSON config.

**Read `equation()`, not the schema.** Most models compute some flows by hand, and an edge can be mainly a form field, so the schema's edges often differ from what runs. There is no importer for this reason. For a run to compare with, use `python -m compartment.models.<model>.main --mode local --config_file <config> --output_file <file>`.

## Mapping

| Pandemic Simulator | Starsim |
|---|---|
| `add_compartment('S', ...)` | A state |
| `*_total` compartments | `.cum` on a flow's result |
| `add_transmission_parameter(..., frequency_dependent=True)` | `S >> I @ beta*I` |
| An edge with `ValueType.DAYS` | A rate: `1/days`, or `ss.days(days)` as a transition's rate |
| `self._apply_flow(derivs, 'I', 'H', rate*I)` | `I >> H @ rate` |
| A value worked out in `equation()` from parameters | An expression of `ss.par`s, which follows any change to them |
| `FieldConfigs` with `has_variance` | `ss.par(value, bounds=[low, high])`, run with `ss.parallel(sim, priors=True)` |
| An intervention with dates | `ss.modify(target, multiply=1 - adherence*reduction, start=date, stop=date)` |
| An intervention with `start_threshold` | `ss.modify(target, multiply=..., where=I.n >= a)` |
| `end_threshold`, on a stochastic model | `until=I.n <= b` as well |
| `run_simulation()`'s run without interventions | `sim.new(interventions=None)` |
| Age mixing from the ISO3 code | `ss.contacts('Kenya')` |
| `STOCHASTIC = True`, with Poisson draws in `equation()` | `solver='binomial'` or `'gillespie'` |

## Traps

- **Percentages.** `infected_population`, thresholds, adherence, and reductions are percentages. An `infected_population` of 0.0015 is 0.0015%, and the number is rounded to two decimals.
- **The default solver forgets.** With `odeint`, `end_threshold` is ignored, so a measure ends as soon as prevalence falls below `start_threshold`. Leave `until=` out to match the original's results.
- **Dates in float32.** A measure can start up to 1/16 of a day from its date, because the date is compared in JAX's default float32. To compare results closely, run the original with `JAX_ENABLE_X64=1`.
- **Steps.** Output is daily up to 365 days and coarser after that, and an adaptive solver takes its own steps. A measure on a threshold is checked once a step in Starsim, so use a smaller `dt` to compare it closely.
- **The model card.** `set_model_metadata()` has the assumptions, biases, and `not_for` that belong in the docstring.
