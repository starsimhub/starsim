# Starsim v3

## What the source is

A Starsim v3 model is Python: an `ss.Sim` of modules (`diseases=`, `networks=`, `demographics=`, `interventions=`, `connectors=`, `analyzers=`), each a class. A disease subclasses `ss.Disease` or `ss.Infection` (or `ss.SIR`, `ss.SIS`, `ss.SEIR`, `ss.NCD`, which is `ssl.NCD` in Starsim 4) and writes its natural history as code: `define_pars(beta=..., dur_inf=ss.lognormal(...), p_death=ss.bernoulli(...))`, `define_states(ss.BoolState('infected'), ss.FloatArr('ti_recovered'), ...)`, `set_prognoses(uids)` (on infection: sample durations and schedule the next event by writing `ti_<event>`), `step_state()` (each step: move agents whose `ti_<event> <= self.ti`), transmission over networks with `beta` per network, `rel_sus`/`rel_trans` arrays, and results from `init_results()`/`update_results()`. Interventions subclass `ss.Intervention` (or use `ss.routine_vx`, `ss.campaign_vx`, `ss.routine_screening`, `ss.treat_num` with products `ss.Vx`, `ss.Dx`, `ss.Tx`); connectors subclass `ss.Connector` and edit one disease's `rel_sus` or rates from another's states; calibration is `ss.Calibration(sim, calib_pars, build_fn, components=[ss.Normal(...), ...])`.

## The first step: make it run

Starsim v4 runs v3 modules written as code (`ss.Disease` and `ss.Infection` subclasses, the latter renamed `ss.AgentDisease`, networks, demographics, custom interventions, connectors, analyzers) as agents, and a v3 module can sit beside a declared one. The v3 names that v4 has replaced were removed, and `ss.SIR`/`SIS`/`SEIR` are now declared, so first upgrade the model with the migrate-starsim3 skill (`../../migrate-starsim3/SKILL.md`), which maps every removed name to its v4 spelling and lists the changes that alter results. Then port in two stages: record the upgraded model's results (the reference for the comparison in `validate.md`; or run the original in a v3 env); then turn the `step()` logic into declared states and transitions, one disease at a time, checking against the reference after each. Only a declared model runs as an ODE or a stochastic compartmental model; a module written as code that works on groups as well as agents says so with `compartmental = True`.

## Mapping

The mapping of each Starsim 3 job to Starsim 4 (states, transitions, competing exits, death from infection, rates by agent, parameters, initial infections, demographics, connectors, products and deliveries, calibration, and results) is in the migration guide's [Jobs and spellings](https://starsim.org/v4/migration/v3_v4.html#jobs-and-spellings), and what changes results on the way is in its [What changes the numbers](https://starsim.org/v4/migration/v3_v4.html#what-changes-the-numbers). What the guide doesn't cover, for porting a disease written as code:

| Starsim v3 | Starsim v4 | Fit |
|---|---|---|
| A categorical `state` array with an `IntEnum` (tbsim's `TBS`), or several `BoolState`s | One `ss.State` per mutually exclusive value | Exact |
| A rate that depends on time since entering a state (`exp(-k * tau)`) | Split the state into recent and remote (tbsim's `infection` and `infection_remote`, at rate `k_latent`), or a rate that reads the time in the state, `ss.peryear(r)*np.exp(-k*L.time_in)` (agents only; `L.time_in` is the clock, which is right where a time worked out by hand from `ti_L` during a step is one step long) | Approximate, or agents only |
| Per-agent relative risk after recovery | Separate states (tbsim's `cleared`, `recovered`, `treated`) with their own reinfection rates | Exact |
| A custom `ss.Intervention.step()` | A delivery's `who=` as a function, a plain function passed to the disease (called each step), or keep it as a v3 module | Agents only, unless it works on groups |
| Analyzers | Keep them (they run on agents), or replace with declared results and `result.by('age')` | Exact |

## What needs care

- **The guide's traps**: a `beta` dict of bare numbers changes meaning (give `ss.probpermonth(...)` etc.), results have other names, and stocks are one index later. With `compare.py`, compare the stocks and the flows in separate calls (`--series`), the stocks with `--shift` of minus one step, and say in it how the names were mapped.
- **Order within a step.** A declared model takes all of its transitions in `step()`, after the interventions, under every solver, so a delivery sees the people at the start of the step. In Starsim 3's code, progressions (`step_state()`) came before the interventions, so a delivery on an event of a progression happens one step later in the port.
- **Agents and the ODE differ by order `dt`.** A declared model run as agents and as an ODE differs by about `rate × dt` (5% to 10% for TB at a weekly step, more near R = 1). Compare agents with the v3 agents, and the ODE with the agents at a fine step; don't chase the rest.
- **v3 `step()` code that branches on per-agent history** (time since infection, counts of past episodes) has no declared form unless it can become states or a dimension. Prefer new states (as tbsim did for reinfection risk and front-loaded progression) over keeping a function, since a function makes the model agents-only.
- **Drug resistance and other strains** are part of the disease, not hand-built states: `ss.Disease(..., strains=ss.Strains(['DS', 'RIF'], max_strains=2, immunity=ss.immunity(due_to=L, protection=0.79)))`, with resistance emerging on the transition that causes it (`ss.Transition(Tx, I, rate=..., move=DS >> (DS, RIF))`). Other combinatorial attributes are a dimension (`stratify()`).
- **Name collisions.** A transition named like a module parameter (`start`, `stop`, `dt`) used to break the sim; check that the names are free, and rename (tbsim used `stop_art`).
- **Change the source as needed.** When porting a real model, change its own API as much as needed to fit v4, and where a feature doesn't fit, change v4 rather than writing a model-specific workaround (in tbsim, drug resistance became the disease's strains, `ss.Strains`).

## Worked example

tbsim's port: `tbsim/tb.py` on the `v4` branch of https://github.com/starsimhub/tbsim (the v3 version is on `main`). In v3, the latent state's exits were code:

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
symptomatic = ss.State()
T.tb_death = ss.death(symptomatic, rate=p.sym_dead, name='tb_death')
T.diagnosis = ss.Transition([S.non_infectious, S.asymptomatic, S.symptomatic], S.treatment, rate=p.theta, name='diagnosis')

bcg = ss.Vaccine(efficacy=dict(latent_asy=0.34, latent_non=0.34, tb_death=0.72), waning=ss.years(10))
sim = ss.Sim(tb, interventions=ss.vaccinate(bcg, who=ppl.age < 1, coverage=0.9))
```

## Running the original

- A v3 model runs in v4 as it is (agents), so the reference can come from the same environment: run the v3 version with several seeds, save the results you will compare (`sim.results.<disease>.<name>` for each series, after mapping the names), and summarize. If it doesn't run in v4, use a v3 env (a separate conda env with Starsim 3.x; for a direct comparison in one process, `sc.importbypath()` the v3 package under another name).
- Series: v3's results by name (`n_infected`, `new_infections`, `prevalence`, and the model's own), with `timevec` as the times. Compare v3's stocks at index `k` with v4's at `k + 1`, flows at the same index, over several seeds.
- Stochastic: 10 to 50 seeds (`ss.parallel(sim, n_runs=...)`); compare means and their spread.
- Write the summary as a CSV in the long format of `validate.md` (`time, series, value, low, high`, with `low` and `high` the 5% and 95% quantiles over runs).
