# LASER (and measles metapop' laser-measles)

## What the source is

LASER (`laser-core`, `laser-generic`, and the measles-specific `laser-measles`) is IDM's Python framework for fast agent-based models over many places. A model is a script: a `GeoDataFrame` of places (name, population, geometry, initial `S`/`E`/`I`/`R`); a `PropertySet` of parameters (`nticks`, `beta`, `gravity_k`, `gravity_a`/`b`/`c`, seasonality); `Model(scenario, parameters)`; and `model.components = [...]`, a list of component classes run in order each tick (`SEIR.Susceptible`, `SEIR.Exposed(model, incubation_dist, infectious_dist, ...)`, `SEIR.Infectious`, `SEIR.Transmission`, births, mortality by a life-table estimator, importation, seasonality). Distributions come from `laser.core.distributions` (`normal`, `gamma`, ...). Each component is a Numba-compiled loop over agents or places; custom components are classes with `step(tick)`. The measles metapop model is a NumPy/Sciris rewrite of a notebook: a chain-binomial over districts.

## Mapping

| LASER | Starsim v4 | Fit |
|---|---|---|
| Places (`GeoDataFrame` with populations and coordinates) | `ss.geo.geography(names=..., pop=..., lat=..., lon=...)` or `x=`/`y=` in km, or `distances=` | Exact |
| `SEIR.Transmission` with `beta` | `S >> E @ ss.perday(beta) * I` | Exact |
| `SEIR.Exposed`/`Infectious` with `normal(loc, scale)` | `E >> I @ ss.normal(7, 1, unit='days')` | Exact as agents; Erlang stages as an ODE |
| `gamma(shape=40, scale=0.25)` | `ss.erlang(k=40, mean=ss.days(10))` | Exact |
| Gravity coupling of the force of infection | `dis.stratify(patch=places, mobility=ss.geo.Mobility(ss.geo.gravity(c=2.5), kind='force', away=3/32))` | Exact (`kind='force'` is LASER's form) |
| Gravity with laser-measles' scaling (mean share exported `k`, capped per place) | Build the matrix in NumPy and pass `ss.geo.Mobility(matrix=gravity, kind='force')` | Exact |
| Gravity mixing (residents make a share of contacts elsewhere) | `ss.geo.Mobility(ss.geo.gravity(c=1.5), kind='visit', away=0.01)` | Exact |
| Initial infections in one place | `init_n=dict(I=ss.by(patch=[0, 100] + [0]*14))` | Exact |
| Births at a crude rate | `ss.birth(S, rate=ss.peryear(cbr/1000))`; vaccinated births as a second `ss.birth(R, ...)` | Exact |
| Deaths at a constant rate | `ss.death(rate=ss.peryear(0.007))` | Exact |
| Deaths from a life table (an age at death drawn once) | A function that writes `sim.people.ti_dead` (workaround) | Exact, but not declared |
| Importation of a fixed number every N days | A function that moves people from `S` to `I` (workaround); or, as a rate per person, `S >> E @ ss.peryear(x) * (S + E + I + R).n / S.n` | Workaround |
| Seasonality as a cosine | `ss.seasonal(mean=..., amplitude=..., peak=...)` | Exact |
| Seasonality as a repeating table (26 fortnights) | `ss.timeseries({14*i: ss.perday(v) for i,v in enumerate(vals)}, period=ss.days(365))` | Exact |
| Age pyramid | `ss.People(n, age_data=np.array([ages, weights]).T)` | Exact |
| Reported cases (a share at onset) | `ss.outcome(0.005*onset)` | Exact |
| Agents | `n_agents=...` (fewer than the population, each standing for several people), or a compartmental solver | Exact in kind |

## What needs care

- **The coupling moves the force of infection, not people.** LASER adds `f[i]·network[i,j]` to place `j`'s force of infection, unweighted by size. Starsim's default commuting moves people; use `kind='force'` to match.
- **`ss.geo.gravity`'s exponent `a` is that of `pop[i]^(a−1)`.** LASER's `a=0` is Starsim's `a=1`. And `ss.geo.Mobility(away=)` gives every place the same share exported: for laser-measles' mean-share scaling and per-place cap, build the matrix yourself.
- **`ss.gamma(40, 0.25)` is not shape 40 and scale 0.25** (the second positional argument is SciPy's `loc`). Use `ss.erlang(k=, mean=)`.
- **The original's order of components changes the durations.** With `Exposed` before `Infectious` in the component list, someone who becomes infectious counts down that same day, so the infectious period is a day shorter than written (`round(normal(8, 2)) − 1`). Check the component order and set the Starsim duration to what the original actually does; say so in a comment.
- **Bare stops are in the sim's unit.** Write `unit='day', stop=years*365`.
- **Speed.** Starsim is much slower per agent than LASER (about 20 to 110 ns per agent-step against 1 ns, on one core against all of them). Run a scaled-down version (fewer places, fewer years, or `n_agents` below the population) for the comparison, and run the original at the same scale.
- **Deaths outside a declared transition** (the life-table function) appear in `sim.results.new_deaths`, not in the disease's results.
- **Importation into `E` from outside** cannot be `ss.null >> E`, which adds people; write it as a rate per susceptible.

## Worked example

`lingua_franca_design/vignettes/laser/hub_spoke_seir/model.py` (an SEIR on a hub and spokes), `lingua_franca_design/vignettes/laser/england_wales/model.py` (measles in England and Wales, with the life-table and importation workarounds), and `lingua_franca_design/vignettes/measles_metapop/seir_districts/model.py` (260 districts). The hub and spoke:

```python
angles = 2*np.pi*np.arange(5)/5
names  = ['Central City'] + [f'Town {i+1}' for i in range(5)] + [f'Interim {i+1}{ab}' for i in range(5) for ab in 'AB']
pop    = [1_000_000] + [100_000]*5 + [25_000]*10
dist   = [0] + [150]*5 + [50, 100]*5 # Distance from the city, in km
angle  = [0] + list(angles) + list(np.repeat(angles, 2))
places = ss.geo.geography(names=names, pop=pop, x=dist*np.cos(angle), y=dist*np.sin(angle))

S, E, I, R = ss.states('S E I R')
seir = ss.Disease(
    infection = ss.Transition(S, E, rate=ss.perday(2/14) * I),
    onset     = ss.Transition(E, I, dur=ss.normal(7, 1, unit='days')),
    recovery  = ss.Transition(I, R, dur=ss.normal(14, 2, unit='days')),
    init_n    = dict(I=ss.by(patch=[0, 100] + [0]*14)), # 100 infectious people in Town 1
)
seir.stratify(patch=places, mobility=ss.geo.Mobility(ss.geo.gravity(c=2.5), kind='force', away=3/32))
sim = ss.Sim(seir, total_pop=sum(pop), n_agents=175_000, stop=ss.days(365))
```

## Running the original

- Environment: `lingua_franca_design/_envs/laser` (a `uv` venv with `laser-generic` 1.1.0 and `laser-core` 1.0.5; Numba compiles on the first run). Build with `python _envs/envs.py install laser`. The England and Wales data (`EnglandWalesMeasles`, `EnglandWalesDistances.npy`) is in the installed `laser-generic` source under `_envs/laser/src/`.
- The examples run a `simulate.py` in the env (`tools.run(['python', 'simulate.py', str(n_runs)], 'laser')`), which builds and runs the model `n_runs` times and prints the time series of each node's compartments as JSON (`model.nodes.S[:, node]`, and so on, indexed by tick).
- Series: counts by place and tick (`S`, `E`, `I`, `R` per node; or summed), and summaries such as total infections, final recovered, peak size and time by place, and the share of weeks with no infections. LASER's `nodes.X[t]` is the count at the start of tick `t` (for comparison with Starsim's `X[t-1]`); check the first row against the initial conditions.
- Stochastic: 5 to 50 runs, depending on cost; compare means and ranges, and run-to-run spread for fade-outs.
- Write the summary with `examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`), and `read_results(folder)` reads it back as `{series: (times, values)}`.
