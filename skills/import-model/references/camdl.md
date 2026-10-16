# camdl

## What the source is

camdl is a compartmental modeling language (a compiler in OCaml, a runtime in Rust) with two front ends. A `.camdl` file declares `time_unit`, `compartments { S, I, R }`, `parameters { beta : rate in [lo, hi] ~ prior(...) }` (with types `rate`, `probability`, `positive`, bounds, priors, and `#'` docs), `let` definitions (`let N = S + I + R`), `transitions { infection : S --> I @ beta * S * I / N }` (each rate a total flow), `observations { cases { covers = closing_at(time, 7 'days); projected = incidence(infection); cases ~ neg_binomial(mean = rho * projected, r = k) } }`, `init { ... }`, and `simulate { from, to }`. The Python front end, `pycamdl`, builds the same model as objects: `m.dimensions(age=[...], patch=[...])`, `m.compartments(...)`, `m.stratify(age)`, `m.parameters(...)`, `m.tables(...)` (arrays over dimensions), indices (`a, b = age.indices('a', 'b')`) whose free use makes families of transitions, `cm.sum(..., b)`, `a.next` (aging along a dimension), `.where(p != q)`, `m.interventions(cm.transfer(..., fraction=, at=))`, `m.init({...})`. Backends: `ode`, stochastic (chain-binomial), and inference (NUTS).

## Mapping

| camdl | Starsim v4 | Fit |
|---|---|---|
| `S --> I @ beta * S * I / N` | `S >> I @ beta * I` (the source taken out; a state over the total is exposure) | Exact |
| `I --> R @ gamma * I` | `I >> R @ gamma` | Exact |
| `E --> I @ E / 5 'days` | `ss.progression('E -> I', dur=ss.days(5))` | Exact |
| `beta : rate in [a, b] ~ log_normal(mu, sigma)` | `ss.par(ss.perday(0.4), bounds=[a, b], prior=ss.lognorm_im(mean=mu, sigma=sigma), doc=...)` | Exact |
| `observations` with `incidence(infection)`, weekly, `neg_binomial(mean = rho * projected, r = k)` | `ss.outcome(rho * infection, every=ss.weeks(1), likelihood=ss.nbinom(n=k))` | Exact |
| Fitting (NUTS) | `sim.calibrate()` (NUTS when every series has a likelihood and the sim is an ODE) | Exact in kind |
| `time_unit = 'days` | `unit='day'` (give it: the unit is otherwise taken from `stop=`) | Exact |
| `m.dimensions(age=...)`, `m.stratify(age)` | `dis.stratify(age=[0, 5, 50])` (ages as bin edges; aging from the bands) | Exact |
| A table by age × age (`contacts[a, b]`) with `cm.sum(contacts[a,b]*I[b,p]/N[b,p], b)` | `networks=ss.contacts(contacts, age=[0, 5, 50])`, acting within each patch | Exact |
| A table by age × patch (`pop[a, p]`) | `split=ss.by([[3000, 1000], ...], dims=['age', 'patch'])` | Exact |
| Aging (`c[a] >> c[a.next] @ c[a] / stay[a]`) | From the age stratification's bands | Exact |
| Movement between patches (`(c[a,p] >> c[a,q] @ move * c[a,p]).where(p != q)`) | `dis.stratify(patch=['north', 'south'], kind='space', split=pop, mobility=ss.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))` | Exact |
| `cm.transfer(S['child', p] >> R['child', p], fraction=0.6, at=[20 'days])` | `ss.vaccinate(ss.Vaccine(efficacy=1.0), years=[20], coverage=0.6, eligible=(ss.People(1).age < 5) & seir.S)` | Same infections; `S` and `R` counts differ (see below) |
| `init { I['child', 'north'] = 10 }` | `init=dict(I=ss.by(age=[10, 0, 0], patch=[1, 0]))` | Exact |
| Backends `ode` / stochastic | `solver='ode'` / `solver='binomial'` | Exact / close (below) |

## What needs care

- **Count lines against the `.camdl` file**, not the `pycamdl` script that drives it.
- **Give the unit.** With `stop=ss.weeks(12)` and no `unit=`, the sim runs in weeks although the rates are per day; camdl's `time_unit` is `unit='day'`.
- **A one-off move between states is a product, not a move.** camdl's `transfer` moves 60% of susceptible children to `R`; the closest declared form is a fully effective vaccine, so they stay in `S` but cannot be infected. The epidemic is the same; the counts in `S` and `R` are not. Compare infections and `I`, not `S` and `R`, or write a function that moves them, marked as a workaround.
- **A condition on age needs a people object.** Write `ss.People(1).age < 5` in a compartmental model.
- **The two chain-binomial schemes differ at a coarse step.** At a step of a day, the stochastic epidemic can rise about a day sooner than camdl's; at a quarter day the gap shrinks. The ODE agrees to 0.02%. Compare the ODE first; compare the stochastic version at peaks and sizes, or at a smaller step.
- **Weekly observations.** `closing_at(time, 7 'days)` sums the flow over the week ending at each time; `every=ss.weeks(1)` does the same.

## Worked example

`lingua_franca_design/vignettes/camdl/sir/sir.py` (an SIR with weekly cases and a negative-binomial likelihood) and `lingua_franca_design/vignettes/camdl/stratified/model.py` (SEIR by age and patch). The stratified one:

```python
contacts = [[12, 4, 1], [4, 8, 2], [1, 2, 3]]
pop = ss.by([[3000, 1000], [5000, 2000], [2000, 500]], dims=['age', 'patch'])

seir = ss.Disease(
    infection = ss.transmission('S -> E', beta=ss.perday(0.04), source='I'),
    onset     = ss.progression('E -> I', dur=ss.days(5)),
    recovery  = ss.progression('I -> R', rate=ss.perday(0.1)),
    init      = dict(I=ss.by(age=[10, 0, 0], patch=[1, 0])),
)
seir.stratify(age=[0, 5, 50]) # Children, adults, and elders, who age at 1/5 and 1/45 a year
seir.stratify(patch=['north', 'south'], kind='space', split=pop, mobility=ss.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
vaccinate = ss.vaccinate(ss.Vaccine(efficacy=1.0), years=[20], coverage=0.6, eligible=(ss.People(1).age < 5) & seir.S)
sim = ss.Sim(seir, networks=ss.contacts(contacts, age=[0, 5, 50]), total_pop=13_510, unit='day', stop=150)
```

## Running the original

- Environment: `lingua_franca_design/_envs/camdl` (`bin/camdl` and `bin/camdlc`, built from the camdl repository at a fixed commit with OCaml and Rust). Build with `python _envs/envs.py install camdl`.
- A `.camdl` model runs from the command line: `camdl simulate model.camdl --backend ode --no-progress --output-dir <store> -o traj.tsv --obs obs.tsv --param beta=0.4 ...` (`tools.run(cmd, 'camdl')`). The stochastic backend (chain-binomial) is the default; `--replicates 200` gives many runs in one command (see `examples/camdl/sir/run.py`). A `pycamdl` model runs in Python: `m.run(n=20, seed=1)`.
- Series: the trajectory TSV has `t`, each compartment, and `flow_<transition>` (the flow in each step, ending at `t`); the observations TSV has each observed series at its emit times. Write the compartments as `ode_S`, ..., the infections as `ode_infection`, and the weekly cases as `ode_cases` (and `stochastic_I`, `stochastic_cases` as summaries over runs). The value at `t` is after the step ending at `t`, as Starsim's `n_<state>[ti]` is.
- Stochastic: 200 runs; compare means and the 5% to 95% range.
- Write the summary with `examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`), and `read_results(folder)` reads it back as `{series: (times, values)}`.
