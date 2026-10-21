# EpiModel

## What the source is

EpiModel is an R package. A model is an R script with three kinds: `dcm()` (a deterministic compartmental model, built in or a custom derivative function), `icm()` (a stochastic individual contact model), and `netsim()` (a stochastic network model, the main use). A network model is built in three steps: `netest()` fits a temporal ERGM from a `formation` formula (`~edges + concurrent + degrange(from = 4)`, `nodematch("risk")`) with `target.stats` and a dissolution model (`dissolution_coefs(~offset(edges), duration = 50, d.rate = ...)`); `netsim()` runs the epidemic with `param.net()` (`inf.prob`, `act.rate`, `rec.rate`, arrival and departure rates), `init.net()` (`i.num`), and `control.net()` (`type = "SIS"`, `nsteps`, `nsims`). Larger models replace or add modules (`infection.FUN`, `progress.FUN`, custom `*.FUN` functions in R that read and write node attributes with `get_attr()`/`set_attr()`), as in the EpiModel Gallery's HIV model, and run scenarios with `use_scenario()`.

## Getting the model out

The script is run in EpiModel's environment by `starsim/interop/dump_epimodel.R` (it needs only EpiModel and jsonlite), with `netest()`, `netdx()`, `netsim()`, `dcm()`, and `icm()` replaced by functions that record their arguments, so nothing is fitted or simulated and it takes about a second. Each call of `netsim()`, `dcm()`, or `icm()` is a run; statements that fail afterwards (plots of the results) are skipped. A saved `netsim()` result or `netest()` (`.rds`) works too.

```bash
Rscript dump_epimodel.R intro_sis.R model.json
```

Then, in Starsim, `sim = ss.load('model.json')`, or `ss.load('intro_sis.R', rscript='.../bin/Rscript')` to dump and load in one step. Choose a run with `run=` (a number, or a scenario's `.scenario.id`); the others are in `sim.imported.scenarios` as the parameters that differ. The script runs as `Rscript` would run it, so `interactive()` is false: the Gallery's HIV model dumps its small test size (300 people, 100 steps), not the 1,500 and 1,040 of an interactive run.

## Mapping

| EpiModel | Starsim | Mechanical |
|---|---|---|
| `netsim()` with `type = "SI"`, `"SIS"`, `"SIR"` | States `S`, `I` (and `R`); `infection = S >> I @ beta * I` with `beta = 1 - (1 - inf.prob)^act.rate` per partnership per step; `rec.rate` to `S` or `R` | yes |
| Probabilities per step | `ss.prob(..., 'day')` (or the `unit=` given: a step is only a name, so `unit='week'` gives the same numbers) | yes |
| `a.rate`, `ds.rate`, `di.rate`, `dr.rate` | An inflow `a_rate * (S + I + R).n` and an outflow from each state | yes |
| `i.num`, `r.num`; the network's size | `init=dict(S=..., I=..., R=...)`, `n_agents=n`, `solver='abm'` | yes |
| `nsteps` | `stop=nsteps - 1` (EpiModel's time 1 is the start) | yes |
| `edges`, `concurrent`, `degrange(from = k)` with target 0 | `ssl.DegreeNet(n_contacts=2*edges/n, concurrency=concurrent/n, max_contacts=k - 1, dur=duration)` | yes (the same targets, not an ERGM fit; marked guessed) |
| `nodematch("attr")` | `match=dict(attr=target/edges)`, and a stratification `attr` split as the attribute is on the network | yes (guessed) |
| Several networks (`netsim(list(est_main, est_cas), ...)`) | One `ssl.DegreeNet` each, `net1`, `net2`, ... | yes |
| `dissolution_coefs(d.rate = ...)` | Nothing: `ssl.DegreeNet` allows for partnerships ended by departures | yes |
| `dcm()` with `type = "SI"`, `"SIS"`, `"SIR"` | The same states with rates, `inf_prob * act_rate * I` (frequency-dependent), `solver='ode'`, `dt` from `control.dcm(dt=)` | yes |
| `icm()` with `type = "SI"`, `"SIS"`, `"SIR"` | The same, with `solver='binomial'`: acts between random pairs as a rate of infection (marked guessed) | yes |
| Scenarios (several calls of `netsim()`, `use_scenario()`) | The chosen one as the sim; the others in `sim.imported.scenarios`, for `sim.new(pars=...)` | yes |
| Other ERGM terms (`nodefactor`, `absdiff`, `gwesp`, `degree`, `offset(...)`, `nodematch(diff = TRUE)`) | Listed; triangles and the like have no counterpart, so note it as a finding | no |
| Partnerships that last longer within a group (`~offset(edges) + offset(nodematch(...))`) | Listed; the first duration is used | no |
| Two groups (parameters ending in `.g2`) | Listed: `stratify(group=[1, 2])` with rates by group, by hand | no |
| A parameter by time since infection (`inf.prob` as a vector), `inter.eff`, `inter.start` | Listed; the first value is used | no |
| A custom module (`infection.FUN = infect`, `type = NULL`) | Listed, with its R source in `sim.imported.modules` (and other functions in `sim.imported.functions`); a model of custom modules only imports its networks, with the parameters in `sim.imported.pars` | no |
| A node attribute that changes (stage, care state) | A state, or a dimension: `hiv.stratify(care=[...], states=[A, C, D])` with `ss.transfer(care='undx -> dx', rate=...)` | no |
| An attribute's effect on a rate (`rel_trans` on ART) | `adjust=dict(rel_trans=[...], aids=[...])` on the stratification, or `rel_trans=` on a state | no |
| A custom module that delivers something (PrEP, testing) | A product and a delivery: `ss.vaccinate(ss.Vaccine(...), coverage=..., where=...)`, `ss.treat(...)`; a function for eligibility on the network (below) | no |
| `dcm()` with a custom derivative function (`new.mod`) | Listed with its source; pair the outflow of one compartment with the inflow of another into a transition, and classify each by the import rules | no |

## What needs care

- **Order within a step.** EpiModel lets people recover in the step in which they are infected and counts after that, so its prevalence is about 5% lower for the same epidemic, and it grows a little more slowly (`icm()` SIS: 460 infected at equilibrium against 487; `netsim()` SIS: 227.5 against 234). Compare new infections a step, which agree (the introduction: 12.0 against 11.7 a step at equilibrium, and the statistical comparison passes; `icm()`: 24.1 against 24.7), or allow for it.
- **Acts per partnership.** EpiModel's chance over a partnership in a step is `1 - (1 - inf.prob*m)^act.rate`, with `m` the infectiousness multiplier of the infected partner. Starsim multiplies `1 - (1 - inf.prob)^act.rate` by `rel_trans`; the difference is small (0.0374 against 0.0370 for an acute partner). `ssl.DegreeNet` has no `acts=`.
- **The network matches the targets better than EpiModel's own.** EpiModel's simulated network can undershoot its targets (casual mean degree 0.28 against 0.30), so agreement on the targets does not imply agreement on the network. Compare the simulated network's diagnostics (edges, mean degree, concurrency) as series too.
- **Conditions on the network are a function.** Eligibility such as "two or more partners across both networks, or a partner with HIV" has no declared form; write a `where=` function that reads `net.edges.p1`/`p2` and mark it as a workaround.
- **Stopping a product at a rate** is written as waning: `ss.Vaccine(efficacy=0.95, waning=-1/np.log(1 - p_stop))`, a mean duration of protection in steps.
- **A care cascade is a dimension, not a disease.** EpiModel's separate attributes (stage, care state, ART history) become one disease with a `care` dimension on the infected states; "off ART after being on it" is a level of its own (`off`).
- **Initial infections.** EpiModel draws who is infected (and, in custom models, their stage) at random; the import gives the expected numbers.

## Worked example

`lingua_franca_design/vignettes/epimodel/sis-network/model.py` (the introduction), which `ss.load()` imports completely (against 100 EpiModel runs: new infections agree statistically, prevalence is 3% higher for the reason above, and the network has 174 partnerships against 177 and 110 people with two or more against 111; a `dcm()` SIR with arrivals and departures agrees with EpiModel's LSODA run to 3e-7 of the largest value), and `lingua_franca_design/vignettes/epimodel/hiv/model.py` (the Gallery HIV model, with the cascade and PrEP), of which only the two networks and the parameters import mechanically: its six modules are R (`infect`, `progress`, `cascade`, `prep`, `dfunc`, `afunc`), listed with their source. The hand translation of those modules:

```python
week = ss.prob # EpiModel uses each rate as the probability of an event in a week (a step)
acts = dict(main=3, casual=1)
cascade = ss.par(0, doc='Whether the care cascade runs')
care = lambda arrow, rate, **kw: ss.transfer(care=arrow, rate=cascade*week(rate), **kw)
S = ss.State()
A = ss.State(rel_trans=5, label='Acute')
C = ss.State(label='Chronic')
D = ss.State(rel_trans=2, label='AIDS')
hiv = ss.Disease(
    infection  = ss.Transition(S, A, rate=({k: week(1 - (1 - 0.0025)**n) for k,n in acts.items()}) * (A + C + D)),
    chronic    = A >> C @ week(1/12),
    aids       = C >> D @ week(1/520),
    aids_death = D >> ss.null @ week(1/104),
    departure  = S + A + C >> ss.null @ week(0.0005),
    arrival    = ss.birth(S, rate=0.00065),
    test       = care('undx -> dx', 0.015, states=['A', 'C']),
    link       = care('dx -> art', 0.1),
    init       = dict(A=2, C=98, D=20),
)
hiv.stratify(care=['undx', 'dx', 'art', 'supp', 'off'], states=[A, C, D], split=[1, 0, 0, 0, 0],
             adjust=dict(rel_trans=[1, 1, 0.3, 0.01, 1], aids_death=[1, 1, 1, 0.1, 1]))
main   = ssl.DegreeNet(n_contacts=0.5, concurrency=0.04, max_contacts=2, dur=200)
casual = ssl.DegreeNet(n_contacts=0.3, concurrency=0.10, dur=26)
sim = ss.Sim(hiv, networks=dict(main=main, casual=casual), n_agents=1500, stop=1040) # A step is a week
```

## Running the original

- Environment: `lingua_franca_design/_envs/epimodel` (R with a private library; `bin/Rscript` puts it first). Build it with `python _envs/envs.py install epimodel`.
- The examples' `run.py` call a small R driver (`reference.R`) that runs the original script and then repeats `netsim()` for `n_runs` simulations, writing every simulation's `as.data.frame(sim)` to CSV. Copy that pattern for a new model: `tools.run(['Rscript', driver, model, tmp, str(n_runs), str(seed), str(n_cores)], 'epimodel')`.
- Series: EpiModel's `s.num`, `i.num`, `si.flow` (new infections in the step), `is.flow`, and the network diagnostics (`edges`, `meandeg`, `concurrent`, `duration`). EpiModel's times start at 1, which is the start: its time `t` is Starsim's `ti = t - 1` (as the vignette's `compare.py` says, index `k` is the number after `k` steps in both), so compare by index or use `--shift 1`. Flows have no value at the first step, where Starsim's are 0.
- Stochastic: use 100 runs for an SIS, 10 to 20 for a large model (each HIV run takes about a minute), and compare means with their spread.
- Write the summary with `examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`, with `low` and `high` the 5% and 95% quantiles over runs), and `read_results(folder)` reads it back as `{series: (times, values)}`.
