# camdl

## What the source is

camdl is a compartmental modeling language (a compiler, `camdlc`, in OCaml; a runtime, `camdl`, in Rust) with two front ends. A `.camdl` file declares `time_unit`, `compartments { S, I, R }`, `parameters { beta : rate in [lo, hi] ~ prior(...) }` (with types `rate`, `probability`, `positive`, bounds, priors, and `#'` docs), `let` definitions (`let N = S + I + R`), `dimensions` and `stratify(by = age)`, `tables`, `transitions { infection : S --> I @ beta * S * I / N }` (each rate a total flow, and indexed families such as `infection[a in age]`), `observations` (a projection such as `incidence(infection)`, a schedule, and a likelihood such as `neg_binomial(mean = rho * projected, r = k)`), `interventions` and events, `init`, `scenarios`, and `simulate { from, to }`. `camdlc model.camdl` writes the model as a JSON intermediate representation (IR) with every family and stratification spelled out. The Python front end, `pycamdl`, builds a model as objects (`m.dimensions(...)`, `m.compartments(...)`, `m.stratify(...)`, `m.transitions(...)`, `m.init(...)`) and runs it in Python, without the compiler, so it has no IR. Backends: `ode`, stochastic (chain-binomial, the default), and inference (NUTS, IF2, particle filters).

## Getting the model out

`ss.load('model.camdl', pars=dict(...))` compiles the file with `camdlc` (from the PATH, `CAMDLC`, or `camdlc=`) and reads the IR; `ss.load('model.json')` reads an IR written by `camdlc model.camdl --no-state-grad > model.json`. Give parameter values that the model leaves to the command line (`--param beta=0.4`) as `pars=`; a model's scenarios fill in values too (`scenario=`, default the first). A pycamdl model is written in the same form by `starsim/interop/dump_pycamdl.py`, which needs only pycamdl:

```bash
python dump_pycamdl.py stratified.py model.json        # a module that makes a model called m or model
python dump_pycamdl.py my_model.py:build model.json    # a function that returns one
```

or `ss.load(m)` on a live pycamdl model, if pycamdl can be imported where Starsim is.

## Mapping

| camdl | Starsim | Mechanical |
|---|---|---|
| `compartments` | States (all declared as people) | yes |
| `S --> I @ beta * S * I / N` | `S >> I @ beta * I` (the source taken out; a state over the total is exposure) | yes |
| `I --> R @ gamma * I`, `E --> I @ E / 5 'days` | `I >> R @ gamma`, `E >> I @ 0.2` (time literals in the model's unit) | yes |
| A birth or death (`--> S`, `S -->`) | An inflow or an outflow against `ss.null` | yes |
| `let N = S + I + R` | A named expression, written out where it is used | yes |
| Parameters with bounds, priors (`log_normal`, `normal`, `beta`, `gamma`, `exponential`, `uniform`, `truncated_normal`, `half_normal`; a flat prior is the bounds), and docs | `ss.par(value, bounds=[...], prior=ss.lognorm_im(mean=mu, sigma=sigma), doc=...)` | yes |
| Observations of a flow (`incidence(...)`) with a share (`rho * projected`), weekly (`closing_at(time, 7 'days)`), and a likelihood (`neg_binomial`, `poisson`, `normal` with a fixed sd) | `ss.outcome(rho * infection, likelihood=ss.nbinom(n=k))` (weekly data is compared with the total over each week) | yes |
| `time_unit`, `simulate { from, to }`, `dt` | `unit=`, `start=`, `stop=`, `dt=` | yes |
| `init`, including expressions of parameters and other compartments | `init=dict(...)` (worked out to numbers) | yes |
| A guard, `N > 0 ? I/N : 0` | `(N > 0) * (I / maximum(N, 1e-300))`, which is 0 for an empty group rather than 0/0 | yes |
| `scenarios` | The chosen one's parameter values; all in `sim.imported.scenarios` | yes |
| Dimensions, `stratify()`, families, tables, `sum(b in age, ...)`, aging along a dimension, movement | One state per combination of levels (`S_child_north`), with the IR's rates; exact, but not written as `stratify()` (listed in the notes) | yes, flat |
| An initial number drawn from a distribution | Its mean (listed) | no |
| A forcing function of time (`sinusoidal`, `periodic`, `piecewise`) | A parameter set to 1 (listed): write it as `ss.seasonal(...)` or `ss.timeseries(...)` | no |
| Interventions (off unless enabled) and events (`transfer`, `add` at times) | Listed in `ImportIncomplete`, with the scenarios that enable each (a one-off move between states has no declared form yet) | no |
| A compartment with an ODE (`ode`), a reaction such as `A + B --> C` | Listed | no |
| An observation of a number in states (`prevalence`), a ratio of flows | A result, or listed; give the likelihood when calibrating | no |
| Fitting (NUTS, IF2, PMMH) | `sim.calibrate(method='nuts')` for an ODE whose series all have a likelihood, `method='pmcmc'` or `'if2'` for a stochastic one (the default is a best fit) | by hand |

## What needs care

- **Rewrite a stratified import as `stratify()`.** The IR has the dimensions spelled out, so the import is flat: right, but long. For a model like the stratified SEIR, the declared form is `seir.stratify(age=[0, 5, 50])`, `seir.stratify(patch=[...], kind='space', split=pop, mobility=ss.geo.Mobility(matrix=..., kind='migrate'))`, and `networks=ss.contacts(contacts, bands=[0, 5, 50])` (see the worked example). Check it against the flat import with the validate-import skill before replacing one with the other.
- **A one-off move between states has no declared form.** camdl's `transfer(S --> R, fraction=0.6, at=20)` is listed. The closest declared form is a fully effective vaccine, `ss.vaccinate(ss.Vaccine(efficacy=1.0), at=[20], coverage=0.6, where=...)`, which gives the same infections but leaves the people in `S`; compare infections and `I`, not `S` and `R`.
- **Interventions are off unless enabled**, in camdl as in pycamdl (`--enable vaccinate`, `run(enable=[...])`); events always happen.
- **Give the solver.** The import runs as an ODE (`solver='ode'`, camdl's `--backend ode`); camdl's default backend is chain-binomial (`solver='binomial'`). At a step of a day the binomial import's mean infectious curve is within 0.9% of camdl's 200 runs; at a quarter of a day it is 12% away, since the two schemes differ by order dt.
- **An initial number below 1** is read by Starsim as a fraction of the population, not a number of people.
- A model that divides by an empty group without a guard gives NaN in Starsim, as camdl stops with `DivByZero`.

## Worked example

`lingua_franca_design/vignettes/camdl/sir/sir.py` (an SIR with weekly cases and a negative-binomial likelihood) and `lingua_franca_design/vignettes/camdl/stratified/model.py` (SEIR by age and patch). `ss.load('sir.camdl', pars=dict(beta=0.4, gamma=0.15, rho=0.6, k=10))` gives the vignette's model, with its priors, and agrees with camdl's ODE to camdl's rounding (its compartments are written as whole numbers: within 0.5 people) and to 1e-7 in the daily infections and weekly cases. The pycamdl SEIR by age and patch (24 states, 58 transitions, with aging and movement), written with `dump_pycamdl.py`, agrees with pycamdl's ODE to 3e-16 in every compartment; its vaccination is an intervention, off unless enabled, and listed. The declared form of the stratified model, by hand:

```python
contacts = [[12, 4, 1], [4, 8, 2], [1, 2, 3]]
pop = ss.by([[3000, 1000], [5000, 2000], [2000, 500]], dims=['age', 'patch'])

S, E, I, R = ss.states('S E I R')
seir = ss.Disease(
    infection = ss.Transition(S, E, rate=ss.perday(0.04) * I),
    onset     = ss.Transition(E, I, dur=ss.days(5)),
    recovery  = ss.Transition(I, R, rate=ss.perday(0.1)),
    init      = dict(I=ss.by(age=[10, 0, 0], patch=[1, 0])),
)
seir.stratify(age=[0, 5, 50]) # Children, adults, and elders, who age at 1/5 and 1/45 a year
seir.stratify(patch=['north', 'south'], kind='space', split=pop, mobility=ss.geo.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
sim = ss.Sim(seir, networks=ss.contacts(contacts, bands=[0, 5, 50]), total_pop=13_510, unit='day', stop=150)
```

Over camdl's own test models (85 files), 84 import and run; 54 with nothing listed, and the rest listing forcing functions of time (17), observations of numbers in states (11), ODE compartments (7), and events (7).

## Running the original

- Environment: `lingua_franca_design/_envs/camdl` (`bin/camdl` and `bin/camdlc`, built from the camdl repository at a fixed commit). Another `camdlc` on the PATH may be older and fail on newer syntax; give `camdlc=` to choose.
- A `.camdl` model runs from the command line: `camdl simulate model.camdl --backend ode --no-progress --output-dir <store> -o traj.tsv --obs obs.tsv --param beta=0.4 ...` (or `--scenario baseline`). The stochastic backend is the default; `--replicates 200` gives many runs in one command (see `examples/camdl/sir/run.py`). A pycamdl model runs in Python: `m.run(backend='ode')`, `m.run(n=200, seed=1)`.
- Series: the trajectory TSV has `t`, each compartment (as whole numbers), and `flow_<transition>` (the flow in the step ending at `t`); the observations TSV has each observed series at its emit times. A compartment at `t` is Starsim's `n_<state>[t]`, and a flow at `t` is Starsim's transition result at the same index, with no shift.
- Stochastic: 200 runs; compare means and the 5% to 95% range.
