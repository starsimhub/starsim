# epydemix

## What the source is

epydemix is a Python package for age-structured compartmental models, run as a stochastic chain-binomial in daily steps. A model is a script: `EpiModel(compartments=[...])` or `load_predefined_model("SIR", transmission_rate=...)`; `model.add_parameter(parameters_dict={...})` (values can be arrays over time, such as a contact reduction `r`, or by age, such as `IFR` via `convert_to_2Darray`); `model.add_transition(source, target, params, kind)` with `kind="mediated"` (`params=("beta*r", "I")`: a rate expression and the infectious state) or `kind="spontaneous"` (`params="eps"`); `model.set_population(load_epydemix_population("Kenya"))` (bundled populations and contact matrices by setting, downloaded from epydemix-data); `model.run_simulations(start_date, end_date, percentage_in_agents=..., Nsim=...)`; and `ABCSampler(...).calibrate(...)` for ABC-SMC.

## Mapping

| epydemix | Starsim v4 | Fit |
|---|---|---|
| `add_transition(kind="mediated", params=("beta", "I"))` | `ss.transmission('S -> E', beta=beta, source='I')` | Exact |
| `add_transition(kind="spontaneous", params="eps")` | `E >> I @ ss.perday(eps)` | Exact |
| Two competing exits `mu*(1-IFR)` and `mu*IFR` | One branch: `ss.Transition(I, [D, R], rate=mu, p=ss.by(age=IFR))` | Exact |
| A chain of compartments `D0..D3` at `4/Delta` | `ss.Transition(D, Dead, dur=Delta, stages=4)` | Exact (Erlang) |
| A population by age (`Nk`) | `dis.stratify(age=[0, 5, 20, 50, 65], split=Nk)`, `total_pop=sum(Nk)`, `use_aging=False` | Exact |
| Contact matrices by setting (home, school, work, community) | `networks=ss.contacts(age=sum(layers))` | Exact |
| `load_epydemix_population("Kenya")` | No bundled data in Starsim: read the population and matrices the original downloaded (dump them to JSON in the epydemix env) | Data only |
| A parameter that is an array over days (`r`) | `ss.timeseries(dict(zip(dates, values)))` | Exact |
| Seasonality from a sine with a peak date | `ss.seasonal(mean=(1 + smin)/2, amplitude=(1 - smin)/(1 + smin), peak=14/365)` | Exact |
| `beta = R0*mu/rho(C)` | Write it out: `R0*mu/max(np.linalg.eigvals(matrix).real)` | Exact |
| `percentage_in_agents` | `init=dict(I=10)` or `init=dict(E=I0/2, I=I0/2)` | Approximate (see below) |
| Scenarios (contacts higher from a date) | `ss.modify(dis.infection, multiply=1.3, start='2020-06-01')` | Exact |
| `ABCSampler` (ABC-SMC) | `ss.par(..., bounds=[...])`, `data=ss.data(...)`, `sim.calibrate()`: Optuna, a point estimate | Different method: no posterior |
| Stochastic runs (`Nsim`) | `solver='binomial'`, `sim.run(n_runs=...)` | Exact in kind |

## What needs care

- **One step a day, with no substeps.** epydemix is a discrete-time model in its own right: in a day, `1 - exp(-rate)` of the people leave a state. Starsim's binomial solver takes substeps for an Erlang dwell time (`stages=`), for every transition, which changes the model (deaths by 2 August 6,039 against 10,525 in the COVID example). Pass `substeps=1` to match.
- **Initial infections are rounded down by age.** `percentage_in_agents` spreads the infections over the age groups and rounds each down (10 became 8 in Kenya), and in the COVID example splits them at random. Starsim splits in proportion to the population; give the original's actual numbers (dump `model.create_default_initial_conditions(...)`) if the start matters.
- **Turn aging off.** An age stratification turns on aging in a compartmental run; epydemix does not age people, so pass `use_aging=False`.
- **Weeks end a day apart.** A Starsim flow summed weekly to a data date covers the week to the end of the day before (the date is the start of that day); epydemix's weeks end on Sunday. Sum the original's daily transitions to the same weeks before comparing.
- **No posterior.** ABC-SMC is not implemented; the Starsim fit is a point estimate inside the original's posterior, and projections from one point can differ a lot from projections averaged over a posterior. Compare at fixed parameter values first; the calibration is a separate, looser check.
- **Results of many runs by age group.** Select by group run by run if `msim.results.<dis>.<result>.select(age=...)` fails.
- `ss.data()` wants a `str` path, not a `Path`.

## Worked example

`lingua_franca_design/vignettes/epydemix/sir_kenya/model.py` (the SIR in Kenya) and `lingua_franca_design/vignettes/epydemix/covid19/model.py` (COVID-19 in Massachusetts, with calibration and scenarios). The core of the COVID translation:

```python
mu     = ss.perday(1/2.5) # Recovery rate
IFR    = np.array([0.00161, 0.00695, 0.0309, 0.0844, 0.161, 0.595, 1.93, 4.28, 7.80])/100
R0     = ss.par(2.22, bounds=[1.5, 6.0])
Delta  = ss.par(ss.days(20), bounds=[7, 27]) # From recovery to death, in 4 stages
I0     = ss.par(4260, bounds=[10, 9999])
smin   = ss.par(0.8, bounds=[0.6, 1.0])
season = ss.seasonal(mean=(1 + smin)/2, amplitude=(1 - smin)/(1 + smin), peak=14/365) # 15 January
beta   = R0*mu/max(np.linalg.eigvals(matrix).real)*season*r # r: the contact reduction, an ss.timeseries

S, E, I, R, D, Dead = ss.states(6)
covid = ss.Disease(
    infection = ss.transmission('S -> E', beta=beta, source='I'),
    onset     = E >> I @ ss.perday(1/4),
    outcome   = ss.Transition(I, [D, R], rate=mu, p=ss.by(age=IFR)),
    deaths    = ss.Transition(D, Dead, dur=Delta, stages=4),
    init      = dict(E=I0/2, I=I0/2),
)
covid.stratify(age=[0, 10, 20, 30, 40, 50, 60, 70, 80], split=data['Nk'])
sim = ss.Sim(covid, networks=ss.contacts(age=matrix), total_pop=sum(data['Nk']), data=deaths, unit='day',
             start='2020-02-23', stop='2020-08-02', solver='binomial', substeps=1, use_aging=False)
```

## Running the original

- Environment: `lingua_franca_design/_envs/epydemix` (a `uv` venv with epydemix 1.4.0; needs a network connection the first time a population is loaded). Build with `python _envs/envs.py install epydemix`.
- The examples run the original through a `simulate.py` in the epydemix env (`tools.run(['python', 'simulate.py'], 'epydemix')`), which `exec`s the original script with `run_simulations` wrapped to pass a seed (the scripts set none), and prints JSON: summarized runs plus the population (`Nk`, the contact matrices by setting) so the Starsim version can read the same data.
- Series: `results.get_stacked_compartments()` gives each compartment by age (`I_0-4`, ..., `I_total`) at the end of each day, and `get_stacked_transitions()` the number moving during each day (written with a `new_` prefix). The initial state is not in epydemix's output: write it at time 0. With day 1 the first value, a count at time `t` is Starsim's `n_<state>[t-1]` from the start date.
- Stochastic: 100 to 200 runs; compare means, and peaks and their timing.
- Write the summary with `examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`), and `read_results(folder)` reads it back as `{series: (times, values)}`.
