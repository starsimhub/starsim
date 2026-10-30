# epydemix

## What the source is

epydemix is a Python package for age-structured compartmental models, run as a stochastic chain-binomial in daily steps. A model is a script: `EpiModel(compartments=[...])` or `load_predefined_model("SIR", transmission_rate=...)`; `model.add_parameter(parameters_dict={...})` (values can be arrays over time, such as a contact reduction `r`, or by age, such as `IFR` via `convert_to_2Darray`); `model.add_transition(source, target, params, kind)` with `kind="mediated"` (`params=("beta*r", "I")`: a rate expression and the infectious state) or `kind="spontaneous"` (`params="eps"`); `model.override_parameter(start, end, name, value)`; `model.add_intervention(layer, start, end, reduction_factor=...)` on one setting's contacts; `model.set_population(load_epydemix_population("Kenya"))` (bundled populations and contact matrices by setting, downloaded from epydemix-data); `model.run_simulations(start_date, end_date, percentage_in_agents=..., Nsim=...)` or `simulate(model, ..., initial_conditions_dict=..., **parameters)`; and `ABCSampler(...).calibrate(...)` for ABC-SMC.

## Getting the model out

The model is built in epydemix's own environment and written to JSON by `starsim/converters/epydemix.py`, which needs only epydemix:

```bash
python epydemix.py my_model.py model.json               # a script that makes a variable called model
python epydemix.py my_model.py:build model.json         # a function that returns the model, or dict(model=..., start_date=..., **run arguments)
```

Then, in Starsim, `sim = ss.import_from('model.json')`. The dump records the last `simulate()` or `run_simulations()` call made with the model, which gives the dates, the step, the initial numbers by age group (worked out from `percentage_in_agents` as epydemix does), and any parameters passed to the run rather than set on the model. That matters: scripts often pass `beta` (or an `R0`-derived array) only to the run, as the COVID tutorial's `simulate_wrapper` does. If the script runs a calibration (ABC-SMC can take an hour), write a short script that builds the model and returns it with the run's arguments at fixed values, and dump that.

## Mapping

| epydemix | Starsim | Mechanical |
|---|---|---|
| A compartment | A state | yes |
| `kind="mediated"`, `params=("beta*r", "I")` | Exposure: `S >> E @ beta * r * I`, through the contacts | yes |
| `kind="spontaneous"`, `params="eps"` | A rate per day: `E >> I @ eps` | yes |
| Two exits from a state at `r*X` and `r*(1-X)` (`mu*IFR`, `mu*(1-IFR)`) | One branch, `ss.Transition(I, [D, R], rate=mu, p=[IFR, None])`, named `I_to_D_or_R` | yes |
| Other competing exits | Separate transitions from the same state (competing hazards, as epydemix's multinomial) | yes |
| A number | A parameter | yes |
| An array over time (1D) | `ss.timeseries({date: value})`, a point only where it changes, from the run's start date | yes |
| An array by age (`convert_to_2Darray`) | `ss.by([...], dims=['age'])` | yes |
| An array over time and by age | Its first row, by age (listed) | no |
| `override_parameter()` of a number | Folded into the parameter's `ss.timeseries` | yes |
| Age groups (`"0-4"`, `"65+"`) and `Nk` | `stratify(age=[0, 5, ...], split=Nk)`, `use_aging=False` (epydemix does not age people) | yes |
| Groups that are not ages | `stratify(group=[...])` | yes |
| Contact matrices by setting | `networks=ss.contacts(sum of the settings)` | yes |
| Initial numbers by group (recorded run) | `init_n=dict(I=ss.by([...], dims=['age']))`, the state with most people taking the rest (it is made the first state, `first=`) | yes |
| Daily steps of a chain binomial | `solver='binomial'`, `substeps=1`, `unit='day'`, stop a day after `end_date` | yes |
| `add_intervention()` on one setting | Listed; by hand (`ss.modify(dis.S_to_E, multiply=..., start=..., stop=...)` if every setting is reduced alike, or the contacts as two networks) | no |
| A custom transition kind (`register_transition_kind`) | Listed; by hand | no |
| A chain of compartments `D0..D3` at `4/Delta` | Kept as compartments (exact); by hand, `ss.Transition(D, Dead, dur=Delta, stages=4)` reads better | no |
| `beta = R0*mu/rho(C)` (computed in the script) | The value passed to the run is imported; to fit `R0`, write it out: `R0*mu/max(np.linalg.eigvals(matrix).real)` | no |
| Seasonality from a sine with a peak date | The array passed to the run is imported; by hand, `ss.seasonal(mean=(1 + smin)/2, amplitude=(1 - smin)/(1 + smin), peak=14/365)` | no |
| Initial numbers drawn at random in each run | The recorded draw, fixed | no |
| Scenarios (contacts higher from a date) | `ss.modify(dis.S_to_E, multiply=1.3, start='2020-06-01')` | no |
| `ABCSampler` (ABC-SMC) | `ss.par(..., bounds=[...])`, `data=ss.simdata(...)`, `sim.fit()`: a point estimate, no posterior | no |

## What needs care

- **One step a day, with no substeps.** epydemix is a discrete-time model in its own right: in a day, `1 - exp(-rate)` of the people leave a state. Starsim's binomial solver takes substeps for an Erlang dwell time (`stages=`), for every transition, which changes the model (deaths by 2 August 6,039 against 10,525 in the COVID example). The import passes `substeps=1`; keep it if you rewrite the death chain with `stages=`.
- **Initial numbers are rounded down by age.** `percentage_in_agents` spreads the infections over the age groups and rounds each down (10 became 8 in Kenya); the dump records the actual numbers. The COVID tutorial's own initial conditions lose the odd person in each group (`(infected/2).astype(int)` twice), so they add up to 4 fewer than the population; the import makes the sim's population the initial numbers (epydemix still divides by `Nk` in the force of infection, a difference of 6e-7).
- **A branch has one result.** `I_to_D_or_R` counts everyone leaving `I`; the number dying is the flow into `D` (`D0_to_D1` and `D0`), or write the two exits as separate transitions to keep both.
- **Weeks end a day apart.** A Starsim flow summed weekly to a data date covers the week to the end of the day before (the date is the start of that day); epydemix's weeks end on Sunday. Sum the original's daily transitions to the same weeks before comparing.
- **No posterior.** ABC-SMC is not implemented; the Starsim fit is a point estimate inside the original's posterior, and projections from one point can differ a lot from projections averaged over a posterior. Compare at fixed parameter values first; the calibration is a separate, looser check.
- **Results by age group.** `res.I.select(age=0)` is the first group; select run by run in a multisim.
- `ss.simdata()` wants a `str` path, not a `Path`.

## Worked example

`lingua_franca_design/examples/epydemix/sir_kenya` (the SIR in Kenya, dumped from the original script as it is) and `examples/epydemix/covid19` (COVID-19 in Massachusetts, dumped from a short script that builds the model at the posterior medians, in the persistence scenario). Against 200 runs of each original (the same initial numbers, `compare.py`): every series agrees, 30 for Kenya (each compartment and transition in each of the 5 age groups; `frac_z3` 0, every KS p ≥ 0.039, means within 1.1% of the peak) and 40 for COVID-19 (all 9 compartments, 4 transitions, and infections, deaths, and new deaths in each of the 9 age groups; `frac_z3` ≤ 0.025, every KS p ≥ 0.02, means within 5% of the peak, 1.7% for the totals). Final deaths: 10,480 (10,279 to 10,705) against the stored reference's 10,525 (10,324 to 10,737), which draws new initial numbers in each run. The import of the COVID model:

```python
beta = ss.par(ss.timeseries({'2020-02-23': 0.0626, '2020-02-24': 0.0626, ...}))  # The array the run was given
IFR  = ss.par(ss.by([1.61e-05, 6.95e-05, ...], dims=['age']))
r    = ss.par(ss.timeseries({'2020-02-23': 1.0, ...}))

S_to_E       = S >> E @ beta * r * I
E_to_I       = E >> I @ eps
I_to_D0_or_R = ss.Transition(I, [D0, R], rate=mu, p=[IFR, None])
D0_to_D1     = D0 >> D1 @ 4.0 / Delta  # ... to D3_to_D4
```

The hand-written vignette (`vignettes/epydemix/covid19/model.py`) instead writes `beta` from `R0` and `ss.seasonal`, and the death chain with `stages=4`, so that `R0`, `Delta`, and the seasonality can be fitted.

## Running the original

- Environment: `lingua_franca_design/_envs/epydemix` (a `uv` venv with epydemix 1.4.0; needs a network connection the first time a population is loaded). Build with `python _envs/envs.py install epydemix`.
- The examples run the original through a `simulate.py` in the epydemix env (`tools.run(['python', 'simulate.py'], 'epydemix')`), which `exec`s the original script with `run_simulations` wrapped to pass a seed (the scripts set none), and prints JSON: summarized runs plus the population (`Nk`, the contact matrices by setting). For `compare.py`, write every run instead: `simulate(model, rng=rng, **run)` in a loop, with `tr.compartments` and `tr.transitions` (with a `new_` prefix) in the long format with a `run` column.
- Series: each compartment by age (`I_0-4`, ..., `I_total`) at the end of each day, and the number moving during each day. The initial state is not in epydemix's output: write it at time 0, and the end of day `t` (from 0) at time `t + 1`. Then a count at time `k` is Starsim's `n_<state>[k]` (`compare.py run --time index`), and a transition at `k` is Starsim's flow at `k`, with no shift.
- Stochastic: 200 runs; compare statistically, with the same initial numbers on both sides.
