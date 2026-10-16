# EpiModel

## What the source is

EpiModel is an R package. A model is an R script with three kinds: `dcm()` (a deterministic compartmental model, often a custom derivative function), `icm()` (a stochastic individual contact model), and `netsim()` (a stochastic network model, the main use). A network model is built in three steps: `netest()` fits a temporal ERGM from a `formation` formula (`~edges + concurrent + degrange(from = 4)`, `nodematch("risk")`) with `target.stats` and a dissolution model (`dissolution_coefs(~offset(edges), duration = 50, d.rate = ...)`); `netsim()` runs the epidemic with `param.net()` (`inf.prob`, `act.rate`, `rec.rate`, arrival and departure rates), `init.net()` (`i.num`), and `control.net()` (`type = "SIS"`, `nsteps`, `nsims`). Larger models replace or add modules (`infection.FUN`, `progress.FUN`, custom `*.FUN` functions in R that read and write node attributes with `get_attr()`/`set_attr()`), as in the EpiModel Gallery's HIV model.

## Mapping

| EpiModel | Starsim v4 | Fit |
|---|---|---|
| `type = "SIS"` / `"SIR"` with `inf.prob`, `act.rate` | `ss.transmission('S -> I', beta=ss.prob(1 - (1 - inf.prob)**act.rate))` | Exact (EpiModel works per partnership per step) |
| `rec.rate` | `ss.progression('I -> S', rate=ss.prob(rec.rate))` | Exact: EpiModel's rates are probabilities per step |
| `formation = ~edges + concurrent + degrange(from = k)`, `target.stats`, `duration` | `ss.PartnerNet(n_contacts=2*edges/n, concurrency=concurrent/n, max_contacts=k-1, dur=duration)` | Approximate: matches the degree targets, not an ERGM fit |
| `nodematch("risk")` | `ss.PartnerNet(match=dict(risk=share_within))` | Approximate (EpiModel's nodematch, recovered to about 0.01) |
| Triangles, other ERGM terms | Not available; note it as a finding | None |
| `dissolution_coefs(d.rate = ...)` | Nothing: `ss.PartnerNet` measures departures and corrects for them | Exact in effect |
| Several networks (main, casual) | `networks=dict(main=..., casual=...)`, `beta=dict(main=..., casual=...)` | Exact |
| `a.rate` (arrivals), `ds.rate`/`di.rate` (departures) | `ss.birth('-> S', rate=...)`, `(S | I) >> ss.null @ ss.prob(...)` | Exact |
| `i.num` | `init=dict(I=10)` | Exact (EpiModel draws stages at random; give expected numbers) |
| A node attribute that changes (stage, care state) | A state, or a dimension: `hiv.stratify(care=[...], states=[A, C, D])` with `ss.transfer(care='undx -> dx', rate=...)` | Exact |
| An attribute's effect on a rate (`rel_trans` on ART) | `adjust=dict(rel_trans=[...], aids=[...])` on the stratification, or `rel_trans=` on a state | Exact |
| A custom module that delivers something (PrEP, testing) | A product and a delivery: `ss.vaccinate(ss.Vaccine(...), coverage=..., eligible=...)`, `ss.treat(...)` | Exact for keyword delivery; a function for eligibility on the network (below) |
| Scenarios (`param.net` variants) | `sim.new(pars=dict(...))`, `sim.new(interventions=...)`, `ss.parallel(**scenarios)` | Exact |
| `dcm()` with a derivative function | Classify each term by the import rules (pair the outflow of one compartment with the inflow of another into a transition); write the declared form | Exact if the terms pair up |

## What needs care

- **Everything is per step.** EpiModel's rates are probabilities per time step (often a week). Write them with `ss.prob` (alias it: `week = ss.prob`) and set `stop=` in steps; don't convert to rates per day.
- **Acts per partnership.** EpiModel's chance over a partnership in a step is `1 - (1 - inf.prob*m)^act.rate`, with `m` the infectiousness multiplier of the infected partner. Starsim multiplies `1 - (1 - inf.prob)^act.rate` by `rel_trans`; the difference is small (0.0374 against 0.0370 for an acute partner). `ss.PartnerNet` has no `acts=`.
- **Order within a step.** EpiModel lets people recover in the step they are infected and counts after that, so its prevalence is about 5% lower for the same epidemic (SIS at equilibrium). Compare new infections a step, or allow for it.
- **Bare numbers as durations of contacts are in years.** In a sim with no unit, give `dur=` in steps explicitly when it matters, or check the network's `net.check()` output.
- **The network matches the targets better than EpiModel's own.** EpiModel's simulated network can undershoot its targets (casual mean degree 0.28 against 0.30), so agreement on the targets does not imply agreement on the network. Compare the simulated network's diagnostics (edges, mean degree, concurrency) as series too.
- **Conditions on the network are a function.** Eligibility such as "two or more partners across both networks, or a partner with HIV" has no declared form; write an `eligible=` function that reads `net.edges.p1`/`p2` and mark it as a workaround.
- **Stopping a product at a rate** is written as waning: `ss.Vaccine(efficacy=0.95, waning=-1/np.log(1 - p_stop))`, a mean duration of protection in steps.
- **A care cascade is a dimension, not a disease.** EpiModel's separate attributes (stage, care state, ART history) become one disease with a `care` dimension on the infected states; "off ART after being on it" is a level of its own (`off`).

## Worked example

`lingua_franca_design/vignettes/epimodel/sis-network/model.py` (the introduction) and `lingua_franca_design/vignettes/epimodel/hiv/model.py` (the Gallery HIV model, with the cascade and PrEP). The key part of the HIV translation:

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
    infection  = ss.transmission('S -> A', beta={k: week(1 - (1 - 0.0025)**n) for k,n in acts.items()}, source=['A', 'C', 'D']),
    chronic    = A >> C @ week(1/12),
    aids       = C >> D @ week(1/520),
    aids_death = D >> ss.null @ week(1/104),
    departure  = (S | A | C) >> ss.null @ week(0.0005),
    arrival    = ss.birth('-> S', rate=0.00065),
    test       = care('undx -> dx', 0.015, states=['A', 'C']),
    link       = care('dx -> art', 0.1),
    init       = dict(A=2, C=98, D=20),
)
hiv.stratify(care=['undx', 'dx', 'art', 'supp', 'off'], states=[A, C, D], split=[1, 0, 0, 0, 0],
             adjust=dict(rel_trans=[1, 1, 0.3, 0.01, 1], aids_death=[1, 1, 1, 0.1, 1]))
main   = ss.PartnerNet(n_contacts=0.5, concurrency=0.04, max_contacts=2, dur=200)
casual = ss.PartnerNet(n_contacts=0.3, concurrency=0.10, dur=26)
sim = ss.Sim(hiv, networks=dict(main=main, casual=casual), n_agents=1500, stop=1040) # A step is a week
```

## Running the original

- Environment: `lingua_franca_design/_envs/epimodel` (R with a private library; `bin/Rscript` puts it first). Build it with `python _envs/envs.py install epimodel`.
- The examples' `run.py` call a small R driver (`reference.R`) that runs the original script and then repeats `netsim()` for `n_runs` simulations, writing every simulation's `as.data.frame(sim)` to CSV. Copy that pattern for a new model: `tools.run(['Rscript', driver, model, tmp, str(n_runs), str(seed), str(n_cores)], 'epimodel')`.
- Series: EpiModel's `s.num`, `i.num`, `si.flow` (new infections in the step), `is.flow`, and the network diagnostics (`edges`, `meandeg`, `concurrent`, `duration`). EpiModel's times start at 1, which is the start: its time `t` is Starsim's `ti = t - 1` (as the vignette's `compare.py` says, index `k` is the number after `k` steps in both), so compare by index or use `--shift 1`. Flows have no value at the first step, where Starsim's are 0.
- Stochastic: use 100 runs for an SIS, 10 to 20 for a large model (each HIV run takes about a minute), and compare means with their spread.
- Write the summary with `examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`, with `low` and `high` the 5% and 95% quantiles over runs), and `read_results(folder)` reads it back as `{series: (times, values)}`.
