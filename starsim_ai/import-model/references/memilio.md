# MEmilio

## What the source is

MEmilio is a C++ library with Python bindings (`memilio.simulation`) for compartmental and agent-based models, mostly of COVID-19. The ODE models are fixed in C++ (`osecir`: the SECIR/SECIHURD model with states `Susceptible`, `Exposed`, `InfectedNoSymptoms`, `InfectedNoSymptomsConfirmed`, `InfectedSymptoms`, `InfectedSymptomsConfirmed`, `InfectedSevere`, `InfectedCritical`, `Recovered`, `Dead`), so a model is a Python script that sets parameters on `Model(n_age_groups)`: times in each state (`TimeExposed`, `TimeInfectedNoSymptoms`, ...), probabilities of each branch (`RecoveredPerInfectedNoSymptoms`, `SeverePerInfectedSymptoms`, `CriticalPerSevere`, `DeathsPerCritical`), `TransmissionProbabilityOnContact`, `RelativeTransmissionNoSymptoms`, the risks of infection from symptomatic people and `TestAndTraceCapacity`, `Seasonality` with `StartDay`, `ICUCapacity`, populations by age group and state, a `ContactMatrix` with `Damping`s, then `simulate(t0, tmax, dt, model)` (an adaptive Runge-Kutta). The equations are in MEmilio's source (`cpp/models/ode_secir/model.h`), not in the script, so read them there.

## Getting the model out

The ODE models `osir`, `oseir`, `oseirdb`, and `osecir` import mechanically. The model is built in MEmilio's own environment and written to JSON by `starsim/converters/memilio.py`, which needs only MEmilio's Python bindings. It runs the script as it is and records the first model the script simulates (with `simulate()`, `simulate_flows()`, a `Simulation` or `MobilitySimulation` that is advanced, or a `ParameterStudy` or `GraphParameterStudy` that is run):

```bash
python memilio.py ode_secir_simple.py model.json          # a script that runs a simulation
python memilio.py my_model.py:run_simulation model.json   # a function in a script
```

```python
sim = ss.import_from('model.json', format='memilio') # A sim, with how each rate was read in sim.imported
```

The importer writes each model's equations as MEmilio's C++ has them (`cpp/models/ode_*/model.h`), with the parameters, populations, and contacts from the file. It translates, exactly:

- parameters by age group, as `ss.by(..., dims=['age'])` where they differ, and age groups as labels (`'0'`, `'1'`, ...) with `ss.contacts(...)`, the matrix at the start summed over MEmilio's matrices;
- dampings as a factor on the force of infection (`contacts`, an `ss.timeseries` with `mode='cosine'`), from the matrix MEmilio uses at the start and either side of each damping, so the levels and types of the dampings are already combined;
- seasonality (`season`, an `ss.seasonal` with the peak from `StartDay`);
- test and trace (`risk`) and ICU capacity (`critical_share`), as functions of the total counts, along MEmilio's half cosine;
- deaths as outflows (`death`, and `severe_death` with `DeathsPerSevere` or an ICU capacity), since MEmilio leaves the dead out of the total; MEmilio's `Dead` is `death.cum` (plus `severe_death.cum`, plus any who start dead);
- the `...Confirmed` states only if they start with people in them (nothing flows into them);
- a graph (`MobilityGraph`, or the `ModelGraph` of a `GraphParameterStudy`) as a stratification by place (`patch`, one level per node, named by its id), with each node's populations by age group and place, and its edges as commuting: `ss.geo.Mobility(matrix=0.5*M, kind='commute')`, where `M[i, j]` is the edge's coefficient (the share of node `i`'s people who go to `j`). MEmilio moves them there for every other step and back, so they spend half their time there, whatever `dt` is;
- a parameter study's distributions (`UncertainValue.set_distribution(ParameterDistributionUniform(lo, hi))`, or `...Normal`) as priors, so `ss.parallel(sim, n_runs=, priors=True)` is the study (`sim.imported.runs` is its number of runs). As MEmilio's osecir draws them: once for all age groups for `TimeExposed`, `TimeInfectedNoSymptoms`, `RelativeTransmissionNoSymptoms`, and the risks from the symptomatic (from the first group's distribution); for each group on its own for the rest (`ss.par(ss.by(...), prior=...)`, one draw per level); and, where each group has its own range, as a standard draw shifted and scaled by group (`TransmissionProbabilityOnContact_shift + TransmissionProbabilityOnContact_scale*TransmissionProbabilityOnContact_draw`, with `_draw` from `ss.uniform(0, 1)`). The study's sampled dampings (`ContactPatterns.dampings`, `DampingSampling`) replace the matrices' own dampings, as in MEmilio, each as a parameter (`damping`) with its prior.

It lists (`ImportIncomplete`): a parameter that differs by age group where it belongs to the infectious person's group (`RelativeTransmissionNoSymptoms`, `RiskOfInfectionFromSymptomatic`) or is summed over the groups (`RecoveredPerInfectedNoSymptoms` and `TimeInfectedNoSymptoms` in test and trace), for which the mean is used; a damping that cuts some contacts more than others (the mean share is used); dynamic NPIs and school holidays; in a graph, a parameter or contact matrix that differs between nodes (the first node's is used), a finite `ICUCapacity` or `TestAndTraceCapacity` (MEmilio applies it to each node's own numbers, the import to the whole model's), commuting that differs between age groups or states (the mean is used), and commuting dampings; in a study, initial numbers that are drawn, sampled dampings of several levels or types or weighted by group, and distributions of parameters the model does not use as such (e.g. `Seasonality`). `memilio.py` writes `osecirvvs` models (listing `InfectiousnessNewVariant`, which the bindings cannot read, and the daily vaccinations), but there is no importer for them yet. Other models (`osecirvvs`, `omseirs4`, the metapopulation, IDE, LCT, SDE, and agent-based models) are written by hand, as below.

## Mapping

| MEmilio | Starsim v4 | Fit |
|---|---|---|
| `TimeExposed` | `onset = E >> INS @ ss.days(3.2)` | Exact |
| `TimeInfectedNoSymptoms` with `RecoveredPerInfectedNoSymptoms` | `ss.Transition(INS, [ISy, R], dur=ss.days(2), p=[0.91, 0.09])` | Exact |
| `SeverePerInfectedSymptoms`, `CriticalPerSevere` | The same, one branching transition each | Exact |
| `DeathsPerCritical` | Two transitions from `ICr` with the same `dur` and `p=0.7`, `p=0.3` (one to `R`, one to `ss.null`) | Exact |
| `Dead` | An outflow: the dead leave the model, and are counted from the flow (`death.cum`) | Exact |
| `TransmissionProbabilityOnContact` × contacts, with `RelativeTransmissionNoSymptoms` | `infection = S >> E @ beta * contacts * (0.7*INS + risk*ISy)` | Exact |
| `RiskOfInfectionFromSymptomatic` rising to `Max...` with `TestAndTraceCapacity` | An expression of counts: `excess = np.clip((0.91/2*INS.n - 35)/(5*35 - 35), 0, 1)`, `risk = 0.35 - 0.1*np.cos(np.pi*excess)` | Exact |
| `Seasonality` with `StartDay` | `ss.seasonal(mean=beta, amplitude=0.2, peak=(-StartDay/365) % 1)` (or a start date and `peak=0`) | Exact |
| `ContactMatrix` by age | `dis.stratify(age=['0', '1', '2'])` and `networks=ss.contacts(matrix)` | Exact |
| `Damping(coeffs, t, ...)` (a cut in contacts, smoothed over the day before) | `ss.timeseries({0: 1.0, t-1: 1.0, t: 1 - coeff}, mode='cosine')` as a factor of the rate | Exact |
| Populations by age and state | `init_n=dict(E=100, INS=50, ...)` (split evenly over groups by default) and `total_pop=` | Exact |
| `...Confirmed` states that nothing flows into | Leave out | Exact |
| `ICUCapacity` (unlimited by default) | Leave out unless set; otherwise a condition on `ICr.n` | Exact when unlimited |
| `simulate(t0, tmax, dt)` | `ss.Sim(..., stop=ss.days(tmax))` (solver `ode`, RK4 with substeps) | Exact to the reference's interpolation error |
| `MobilityGraph` with `add_node(id, model)` and `add_edge(i, j, MobilityParameters(c))`, run by `MobilitySimulation` | `dis.stratify(patch=ids, split=ss.by(pop/pop.sum(), dims=['age', 'patch']), mobility=ss.geo.Mobility(matrix=0.5*M, kind='commute'))` | Close: a share of time rather than an exchange, and symptomatic commuters not reduced (see below) |
| `UncertainValue.set_distribution(ParameterDistributionUniform(lo, hi))`, `GraphParameterStudy(graph, t0, tmax, dt, n)` | `ss.par(value, prior=ss.uniform(lo, hi))`, then `ss.parallel(sim, n_runs=n, priors=True)` | Exact in distribution (MEmilio's draws given by `pars=` agree run by run to 2%) |

## What needs care

- **The equations are not in the script.** Read the model's `get_derivatives()` in the C++ source for the exact flows (e.g. the flow from `INS` to `ISy` is `(1 - mu_CR)/TimeInfectedNoSymptoms * INS`). The rates are per day.
- **The threshold is on a flow.** Test-and-trace capacity compares the number becoming symptomatic each day with the capacity; Starsim cannot name a transition's current flow in a rate, so write the flow as a count expression (`0.91/2*INS.n`, repeating the probability and the time), with a comment.
- **Inside a function, a declared state's `.n` is symbolic.** Use NumPy functions of count expressions in the rate itself (`np.clip`, `np.cos`), which compile into the kernels; in a function of the sim, read `sim.diseases.<name>.<state>.n`.
- **Dampings are smoothed.** MEmilio smooths every damping along half a cosine over the day before its time: `mode='cosine'` matches.
- **Seasonality's day 0.** MEmilio's day 0 is `StartDay` of a 365-day year; with no dates, `ss.seasonal` starts at the beginning of a year, so the peak is `(-StartDay/365) % 1` (the peak is a fraction of the year from 0 to 1, or a time of year such as `ss.days(305)`).
- **Age groups are labels.** Name them as MEmilio numbers them (`'0'`, `'1'`, `'2'`); Starsim treats strings as labels and does not age anyone.
- **A damping within a step.** The import runs in steps of a day; a value that varies in time is taken at the start, middle, and end of each step, so MEmilio's one-day smoothing of a damping is approximated within the step, and `substeps=` does not help. With `dt=0.1`, the import agrees with a tightly integrated MEmilio run to 1e-7 of the population; with steps of a day, to about 2e-4.
- **Commuting is a share of time, not an exchange.** MEmilio moves commuters into the other node and integrates them there for half of each day; Starsim weights each place's exposure by the time spent there. MEmilio also lets those with symptoms commute less (by `RiskOfInfectionFromSymptomatic`, its `get_mobility_factors()`), which Starsim's `ss.geo.Mobility` cannot say, and which is most of the difference: in a test of three places with that risk at 0.25, the total with symptoms agrees to 1.2% of the peak and each place to 5 to 19%; with the risk at 1, each place agrees to 1 to 3%. On the review's 50 synthetic counties, the total agrees to 0.1% and the counties by a median of 11% (worst 25%) of their peaks.
- **A study's draws are MEmilio's.** In a `GraphParameterStudy`, every node takes the first node's drawn parameters (only `ICUCapacity` and `TestAndTraceCapacity` stay by node), which is what one set of parameters for the whole sim gives. With MEmilio's own draws given to the import (`pars=`), each run's peak agrees to 2%; over 200 runs at 50 counties, the median peak with symptoms is 10,948 against MEmilio's 10,571, in 38 s against 71 s.
- **The reference is interpolated.** MEmilio's adaptive solver returns uneven time points; `interpolate_simulation_result()` gives whole days, with an error of about a person in 10,000. That is the floor for agreement.

## Worked example

`lingua_franca_design/vignettes/memilio/secir/model.py` (the simple SECIR) and `lingua_franca_design/vignettes/memilio/secir_ageres/model.py` (by age, with seasonality, a damping, and test-and-trace). The age-resolved version:

```python
beta     = ss.seasonal(mean=0.05, amplitude=0.2, peak=ss.days(305)) # Highest on 1 Jan, day 305 (day 0 is 2 Mar)
contacts = ss.timeseries({0: 1.0, 29: 1.0, 30: 0.3}, mode='cosine') # Cut by 70% over day 29, as MEmilio's damping
S, E, INS, ISy, ISev, ICr, R = ss.states(7)
excess = np.clip((0.91/2*INS.n - 35)/(5*35 - 35), 0, 1) # People becoming symptomatic each day, beyond the 35 that can be traced
risk = 0.35 - 0.1*np.cos(np.pi*excess) # From 0.25 to 0.45, along half a cosine

secir = ss.Disease(
    infection = S >> E @ beta * contacts * (0.7 * INS + risk * ISy),
    onset     = E >> INS @ ss.days(3.2),
    symptoms  = ss.Transition(INS, [ISy, R], dur=ss.days(2), p=[0.91, 0.09]),
    severe    = ss.Transition(ISy, [ISev, R], dur=ss.days(5.8), p=[0.2, 0.8]),
    critical  = ss.Transition(ISev, [ICr, R], dur=ss.days(9.5), p=[0.25, 0.75]),
    recovery  = ss.Transition(ICr, R, dur=ss.days(7.1), p=0.7),
    death     = ss.Transition(ICr, ss.null, dur=ss.days(7.1), p=0.3),
    init_n    = dict(E=100, INS=50, ISy=50, ISev=20, ICr=10, R=10),
)
secir.stratify(age=['0', '1', '2'])
sim = ss.Sim(secir, networks=ss.contacts(np.full((3, 3), 10/3)), total_pop=10_000, stop=ss.days(50))
```

## Running the original

- Environment: `lingua_franca_design/_envs/memilio` (a `uv` venv with Python 3.11 and the `memilio-simulation` 2.4.1 wheel, and matplotlib). Build with `python _envs/envs.py install memilio`.
- The examples' `run.py` run the original script as it is through a driver in the env: the driver wraps `simulate` to keep its result and time it, calls the script's function with plotting off, interpolates the result to whole days with `memilio.simulation.osecir.interpolate_simulation_result`, sums the age groups, and prints JSON with the state names from `InfectionState`.
- Series: each state's count at each whole day (time 0 is the initial state, the same as Starsim's start), and `Dead`, which Starsim has as the cumulative death flow. MEmilio is deterministic: one run, compared to a tolerance of about the interpolation error (1.5 people in 10,000 here).
- Write the summary with `examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`), and `read_results(folder)` reads it back as `{series: (times, values)}`.
