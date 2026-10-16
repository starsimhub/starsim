# Starsim v3

## What the source is

A Starsim v3 model is Python: an `ss.Sim` of modules (`diseases=`, `networks=`, `demographics=`, `interventions=`, `connectors=`, `analyzers=`), each a class. A disease subclasses `ss.Infection` (or `ss.SIR`, `ss.SIS`, `ss.SEIR`, `ss.NCD`) and writes its natural history as code: `define_pars(beta=..., dur_inf=ss.lognorm_ex(...), p_death=ss.bernoulli(...))`, `define_states(ss.BoolState('infected'), ss.FloatArr('ti_recovered'), ...)`, `set_prognoses(uids)` (on infection: sample durations and schedule the next event by writing `ti_<event>`), `step_state()` (each step: move agents whose `ti_<event> <= self.ti`), transmission over networks with `beta` per network, `rel_sus`/`rel_trans` arrays, and results from `init_results()`/`update_results()`. Interventions subclass `ss.Intervention` (or use `ss.routine_vx`, `ss.campaign_vx`, `ss.routine_screening`, `ss.treat_num` with products `ss.Vx`, `ss.Dx`, `ss.Tx`); connectors subclass `ss.Connector` and edit one disease's `rel_sus` or rates from another's states; calibration is `ss.Calibration(sim, calib_pars, build_fn, components=[ss.Normal(...), ...])`.

## The first step: it already runs

Starsim v4 runs v3 modules unchanged, as agents: `ss.Sim(diseases=ss.SIR(...), networks=ss.RandomNet(...), n_agents=...)` works in v4, and a v3 module can sit beside a declared one. So port in two stages: first get the v3 model running under v4 and record its results (that is the reference for `validate-import`, with no other environment needed if the v3 code is importable; otherwise run it in a v3 env); then turn the `step()` logic into declared states and transitions, one piece at a time, checking against the reference after each. Only a declared model runs as an ODE or a stochastic compartmental model; a module written as code that works on groups as well as agents says so with `compartmental = True`.

## Mapping

| Starsim v3 | Starsim v4 | Fit |
|---|---|---|
| `BoolState('susceptible')`, `BoolState('infected')`, `BoolState('recovered')` | `S, I, R = ss.states('S I R')` (one state per mutually exclusive condition) | Exact |
| A categorical `state` array with an `IntEnum` (tbsim's `TBS`) | One `ss.State` per value | Exact |
| Transmission in `Infection.infect()` with `beta` | `infection = S >> I @ beta * I`, or `ss.transmission('S -> I', beta=dict(net1=..., net2=...))` over named networks | Exact |
| `set_prognoses` sampling `dur_inf`, `step_state` moving those with `ti_recovered <= ti` | `recovery = I >> R @ dur_inf` (a duration or distribution after `@` is a dwell time) | Exact in distribution |
| `p_death` drawn at infection, death at the end of the same duration | Two transitions from `I` with the same `dur` and `p=`: `ss.Transition(I, R, dur=dur, p=0.99)`, `ss.Transition(I, ss.null, dur=dur, p=0.01)` | Exact in distribution (a branch to `ss.null` inside a list of destinations is not supported) |
| `self.transition(uids, to={A: rate_a, B: rate_b})` (competing exits) | One transition per exit: `X >> A @ rate_a`, `X >> B @ rate_b` | Exact (competing hazards) |
| A rate scaled per agent (`rr_activation[uids]`) | A dimension with `adjust=`, or `ss.modify(tr, multiply=..., where=...)` | Exact |
| A rate that depends on time since entering a state (`exp(-k * tau)`) | Split the state into recent and remote (tbsim's `infection` and `infection_remote`, at rate `k_latent`), or a `fn(sim, uids)` reading `ti_<state>` (agents only) | Approximate, or agents only |
| Per-agent relative risk after recovery | Separate states (tbsim's `cleared`, `recovered`, `treated`) with their own reinfection rates | Exact |
| `define_pars(...)` | `ss.par(value, bounds=[lo, hi], doc=...)`, with typed values (`ss.peryear(...)`, `ss.years(...)`) | Exact |
| `init_prev=ss.bernoulli(0.01)` | `init_prev=0.01`, or `init=dict(I=0.01)` | Exact |
| `ss.Births`, `ss.Deaths` | `ss.birth('-> S', rate=...)`, `ss.death(rate=...)` in the disease, or keep the v3 demographics modules (agents) | Exact |
| `ss.Connector` editing `rel_sus` or a rate from another disease | `ss.modify(tb.latent_asy, multiply=1.9, where=hiv.latent)`, passed as `connectors=` | Exact (and compartmental: the diseases are crossed, `ss.Crossing`) |
| `ss.Vx`/`ss.routine_vx`, `ss.Tx`/`ss.treat_num`, `ss.Dx`/`ss.routine_screening` | Products `ss.Vaccine(efficacy=dict(<transition>=...), waning=...)`, `ss.Treatment(...)`, `ss.Test(sensitivity=..., ...)`, delivered with `ss.vaccinate(...)`, `ss.treat(...)`, `ss.screen(..., on_positive=...)` and keywords (`eligible=`, `coverage=`, `years=`, `interval=`) | Exact for keyword delivery |
| A custom `ss.Intervention.step()` | A delivery's `deliver=` function, a plain function passed to the disease (called each step), or keep it as a v3 module | Agents only, unless it works on groups |
| A parameter change over time in an intervention | `ss.change(dis.pars.beta, at=[...], by=[...])` | Exact |
| `ss.Calibration` with `calib_pars`, `build_fn`, components | Bounds on the parameters (`ss.par(..., bounds=)`), `data=ss.data(df)` on the sim, and `sim.calibrate()`; a likelihood, if wanted, on an outcome (`ss.outcome(..., likelihood=ss.nbinom(n=k))`) | Exact in intent; a different default method |
| Results from `update_results()` (`n_infected`, `prevalence`, `new_infections`) | Automatic: `n_<state>`, each transition's flow and `cum_<name>`, `new_infections`, `prevalence`; extras declared as expressions (`infected = A + C + D`) or `ss.outcome` | Exact, but names differ (below) |
| Analyzers | Keep them (they run on agents), or replace with declared results and `result.by('age')` | Exact |

## What needs care

- **Results have other names.** v3's `n_infected` is `n_I` (or a declared `infected = ...` sum); recoveries are the flow named after the transition. Map the names before comparing, and say which in `compare.py`.
- **Order within a step.** In a declared model, progressions happen in `step_state()`, before interventions; as agents, a delivery comes after the step's progressions and before transmission, while in a compartmental run it comes before all of the step's flows. A delivery on the first step can reach fewer people as agents (tbsim's TPT reached 7% fewer). Compare after the first few steps, or allow for it.
- **Agents and the ODE differ by order `dt`.** A declared model run as agents and as an ODE differs by about `rate × dt` (5% to 10% for TB at a weekly step, more near R = 1). Compare agents with the v3 agents, and the ODE with the agents at a fine step; don't chase the rest.
- **v3 `step()` code that branches on per-agent history** (time since infection, counts of past episodes) has no declared form unless it can become states or a dimension. Prefer new states (as tbsim did for reinfection risk and front-loaded progression) over keeping a function, since a function makes the model agents-only.
- **Drug resistance and other combinatorial attributes** are a dimension (a strain dimension, in tbsim: `drugs=`, with `immunity=` between strains), not hand-built states.
- **Name collisions.** A transition named like a module parameter (`start`, `stop`, `dt`) used to break the sim; check that the names are free, and rename (tbsim used `stop_art`).
- **Change the source as needed.** When porting a real model, change its own API as much as needed to fit v4, and where a feature doesn't fit, change v4 rather than writing a model-specific workaround (in tbsim, drug resistance became a v4 strain dimension).

## Worked example

tbsim's port: `tbsim/tb.py` on the `v1.0.0` branch of https://github.com/starsimhub/tbsim (the v3 version is on `main`). In v3, the latent state's exits were code:

```python
u = self.latent.uids
inf_non, inf_asy = self.progression_rates(u) # Per agent, with rr_activation and exp(-k * tau)
self.transition(u, to={TBS.CLEARED: self.pars.inf_cle, TBS.NON_INFECTIOUS: inf_non, TBS.ASYMPTOMATIC: inf_asy}, rng=self._rng_inf)
newly_cleared = u[self.state[u] == TBS.CLEARED]
self._enter_cleared(newly_cleared, self.pars.rr_reinfection_cleared) # A per-agent relative risk of reinfection
```

In v4 they are declared transitions, the time dependence is a second latent state, the relative risk is a state of its own, and BCG is a product acting on the transitions:

```python
foi = p.beta*(p.trans_asymp*S.asymptomatic + S.symptomatic)
add('transmission', 'susceptible', 'infection', foi)
add('reinfect_cle', 'cleared', 'infection', p.rr_reinfection_cleared*foi) # 'cleared' carries the relative risk
add('clearance', 'infection', 'cleared', p.inf_cle)
add('latent_non', 'infection', 'non_infectious', p.inf_non)
add('latent_asy', 'infection', 'asymptomatic', p.inf_asy)
if front: # The hazard falls with time since infection: recent, then remote
    add('remote', 'infection', 'infection_remote', p.k_latent)
    add('remote_asy', 'infection_remote', 'asymptomatic', p.inf_asy_remote)
T.tb_death = ss.death('symptomatic ->', rate=p.sym_dead, name='tb_death')
T.diagnosis = ss.Transition([S.non_infectious, S.asymptomatic, S.symptomatic], S.treatment, rate=p.theta, name='diagnosis')

bcg = ss.Vaccine(efficacy=dict(latent_asy=0.34, latent_non=0.34, tb_death=0.72), waning=ss.years(10), name='bcg')
sim = ss.Sim(tb, interventions=ss.vaccinate(bcg, eligible=ppl.age < 1, coverage=0.9))
```

## Running the original

- A v3 model runs in v4 as it is (agents), so the reference can come from the same environment: run the v3 version with several seeds, save the results you will compare (`sim.results.<disease>.<name>` for each series, after mapping the names), and summarize. If it doesn't run in v4, use a v3 env (a separate conda env with Starsim 3.x; for a direct comparison in one process, `sc.importbypath()` the v3 package under another name).
- Series: v3's results by name (`n_infected`, `new_infections`, `prevalence`, and the model's own), with `timevec` as the times. Both versions index results by step in the same way, so times line up.
- Stochastic: 10 to 50 seeds (`ss.parallel(sim, n_runs=...)` or `sim.run(n_runs=...)`); compare means and their spread.
- To share the format of the other frameworks' references, write the summary with `lingua_franca_design/examples/tools.py`: `write_results(folder, rows)` writes `reference/results.csv` in the long format (`time, series, value, low, high`, with `low` and `high` the 5% and 95% quantiles over runs), and `read_results(folder)` reads it back as `{series: (times, values)}`.
