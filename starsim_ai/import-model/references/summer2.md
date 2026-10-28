# summer2

## What the source is

A summer2 model is Python: a `summer2.CompartmentalModel` built in code, with compartments, flows of named kinds (`add_infection_frequency_flow`, `add_transition_flow`, `add_death_flow`, `add_crude_birth_flow`, ...), stratifications (`Stratification`, `AgeStratification`, `StrainStratification`, with splits, flow adjustments, infectiousness adjustments, and mixing matrices), initial populations, and requested derived outputs. It is run with `model.run()`.

## Getting the model out

The model is built in summer2's own environment and written to JSON by `starsim/io/summer2.py`, which needs only summer2 (copy it into the summer2 environment):

```bash
python summer2.py my_model.py:build_model model.json   # a function that returns the model
python summer2.py my_model.py model.json               # a module that makes a variable called model
python summer2.py my_model.py:build_model model.json params.json   # with the values of its named Parameters (summer2 needs them to work out the initial population)
```

Then, in Starsim, `sim = ss.import_from('model.json')`. If the source builds several models (summer2's notebooks build one per cell), write a short script that builds one with every derived output wanted, and dump that.

## Mapping

| summer2 | Starsim | Mechanical |
|---|---|---|
| `InfectionFrequencyFlow` | Exposure: `contact_rate * I` | yes |
| `InfectionDensityFlow` | Counts: `contact_rate * I.n` | yes |
| `TransitionFlow`, `DeathFlow` | A transition; an outflow (one for each compartment, `<flow>_<compartment>`) | yes |
| `CrudeBirthFlow`, `ImportFlow`, `AbsoluteFlow` | An inflow (`rate * N`, or a number) | yes |
| `stratify_with()` | `stratify(...)` with `split=`, `states=`, and `adjust=` (`Multiply(x)` is x, `Overwrite(x)` is x/base) | yes |
| Infectiousness adjustments | `adjust={'rel_trans': [...]}` | yes |
| `AgeStratification` | Numeric levels with `kind='age'`; Starsim ages people itself, so summer2's aging flows are left out, and the unit is years | yes |
| Mixing matrices (`Data(...)` or a plain array) | `networks=ss.contacts(dict(age=..., location=...))` (multiplied together, as summer2's Kronecker product) | yes |
| A flow into a stratified compartment with adjustments | The shares as the split (an `ss.by` over an earlier dimension with `source_strata`), and the flow multiplied by their sum | yes |
| Derived outputs | `ss.io.summer2.outputs(sim)`, which works them out from the results | yes |
| A named `Parameter` | A parameter of that name, with the value given in `params.json` (or `ss.import_from(model, pars=...)` for a live model) | yes |
| `get_linear_interpolation_function(x, y)`, `get_sigmoidal_interpolation_function(x, y)`, and products of them with named Parameters | `ss.timeseries(..., mode='linear')` (at the points, or the sigmoid sampled every tenth of a step; the reviewer's TB model by age agrees with summer2 to 7e-6 of the population in every compartment) | yes |
| Another parameter that is a function of time | `ss.timeseries(..., mode='linear')` or `ss.seasonal(...)` | no |
| An adjustment on several earlier dimensions | Rates by hand: `rate=base * ss.by([[...], [...]], dims=['age', 'clinical'])` | no |
| Infectiousness that differs between compartments | `rel_trans=` on each `ss.State` | no |
| `ReplacementBirthFlow` | Births at the death rate: `ss.null >> S @ mu * (S + I + R).n` | no |
| Computed values, tracked values | An `ss.par(lambda sim: ...)`, or a result | no |

## What needs care

- summer2 splits the initial population by the population split, but Starsim's split of a partial stratification is its split on entry; the import starts such a compartment from `ss.by(<counts>, dims=[...])`.
- Strain stratifications are imported as the disease's strains (`ss.Disease(..., strains=ss.Strains(...))`, then `dis.strains`), with the flow adjustments as values by strain; check the infection flows, since summer2 gives each strain its own force of infection.
- summer2's flow outputs are rates: divide Starsim's transition results by `dt`. Its cumulative outputs add up those rates without multiplying by the step.
- summer2's flow output at time `t` is the mean of the instantaneous rates at `t - dt` and `t` (the trapezoid rule), not the number that moved over the step, which is what Starsim's transition result is. They differ where a flow curves sharply within a step: in the TB model with a yearly step, notifications differ by 5% in 1955, early in the epidemic and the scale-up, and by 0.2% or less from 1980, while every compartment agrees. Compare compartments to validate an import; to reproduce summer2's numbers exactly, take the mean of `rate × n_<state>` at the two ends of each step.

## Worked example

`lingua_franca_design/vignettes/summer2/seir-age-clinical/model.py` (SEIR by age with a clinical stratification of `I` entered by age, infectiousness by level, and derived outputs) and `vignettes/summer2/sir-age-location/model.py` (mixing matrices for age and location). The imports agree with summer2 within 0.0005 people of 1,000 (every compartment and derived output) and 0.08 people of 1,000. The derived outputs, in hand-written code:

```python
res = sim.results.sir
incidence_old = res.incidence.select(age='old').values / dt   # summer2's flow outputs are rates
```

## Running the original

In summer2's environment (`lingua_franca_design/_envs/summer2`): build the model, `model.run()`, then `model.get_outputs_df()` (one column per stratified compartment, e.g. `SXage_youngXlocation_urban`) and `model.get_derived_outputs_df()`; LFD's `examples/summer2/*/export.py` writes these in the long format. In Starsim, `res[f'{state}'].select(age=..., location=...)`. A stock at index `k` is the number at summer2's time `k`, with no shift.
