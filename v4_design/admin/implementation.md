# Implementation plan

Status of the Starsim v4 implementation of the design spec. The code is in `starsim/` in this repository, on the `v4` branch; section numbers refer to the files in `design/`. Checked items run and have a test in `tests/test_transitions.py`, `tests/test_deliveries.py`, or `tests/test_dimensions.py`.

## Build set

The nine build steps from [examples/README.md](../examples/README.md).

- [x] 1. SIR, in all spellings
- [x] 2. SEIR with an Erlang dwell time
- [x] 3. A branching model with a competing death hazard
- [x] 4. Cholera, with a water reservoir
- [x] 5. Age-stratified SIR with a contact matrix
- [x] 6. Vaccination as a stratification plus a transfer
- [x] 7. A model with outcomes, data, and bounds
- [x] 8. A model with custom code
- [x] 9. A sexual-network model

## Held-out set

- [x] Wave 1: for each framework, the simplest example that shows what it is for. The originals, their environments, and the Starsim versions are kept in the `lingua_franca_design` repository (`examples/`, `_envs/`, and `vignettes/`), not here.
    - [x] camdl: SIR with weekly reported cases, and a negative-binomial likelihood
    - [x] summer2: SIR by age and location, with a mixing matrix for each
    - [x] epydemix: SIR in Kenya's population, with contact matrices by age
    - [x] Atomica: the SIR demo, stepped a year at a time
    - [x] EMULSION: one SIR as compartments and as individuals
    - [x] LASER: SEIR in a hub and spokes with gravity coupling (agrees, with the force of infection moving between places as in LASER)
    - [x] MEmilio: SECIR with severity branches and a change in contacts
    - [x] flepiMoP: SEIR in two provinces whose residents visit each other
    - [x] odin: a stochastic SIR, with its array of replicates as many runs
    - [x] SimInf: SIR in 1,600 herds, with 466,692 scheduled events as a movement register (agrees with the original; one Gillespie run takes 9.5 s, against 0.08 s)
    - [x] EpiModel: SIS over partnerships that form and dissolve (agrees at equilibrium with `ss.PartnerNet` and EpiModel's target statistics; the epidemic grows about 12% more slowly at first)
- [x] Wave 2: the other eight conformance corpus models
    - [x] 4. SIR with a lognormal dwell time
    - [x] 7. Patches with gravity mobility, and with none
    - [x] 9. Two diseases that modify each other (as agents; a compartmental run is listed under 06)
    - [x] 12. Ross-Macdonald malaria, with a seasonal inflow and mosquito species
    - [x] 13. Screening with follow-up treatment, moved in the loop
    - [x] 16. A threshold-triggered intervention, with and without memory
    - [x] 17. TB with resistance as a strain
    - [x] 18. Two diseases with age stratified on the sim
- [x] Wave 3: one complex example from each framework, Ghana measles, and TB
    - [x] camdl: SEIR by age and patch, with contacts by age within each patch, aging, movement, and a vaccination (agrees as an ODE; as chain-binomial, the same peaks a day sooner at a step of a day)
    - [x] Ghana measles: stochastic SEIR in 260 districts with gravity mixing, seasonality, births with vaccination, deaths, importation, and reported cases (the `gm-port` rewrite; agrees, with no change to Starsim)
    - [x] MEmilio: the age-resolved ODE SECIR, with seasonality, a cosine-smoothed cut in contacts, and test-and-trace capacity (agrees within 1.5 people of 10,000, the error of the reference itself; everything declared)
    - [x] summer2: SEIR by age with a clinical stratification of `I` only, entered by age, with infectiousness by level and derived outputs (agrees within 0.0006 people of 1,000)
    - [x] epydemix: COVID-19 in Massachusetts, nine age groups, an Erlang delay to death, seasonality, mobility, reopening scenarios, and calibration (agrees at fixed values within 1%; the calibration is a point estimate, not ABC-SMC)
    - [x] flepiMoP: two provinces with vaccination and waning, three scenarios of stacked modifiers, and chained outcomes (agrees exactly by province with the original's draws)
    - [x] odin: SEIR in 5 age groups with births, deaths, aging, waning, and a fixed 14-day latent period (`delay()`) (agrees within 1.6%, from the stages; the youngest group differs because odin's latent people do not age)
    - [x] Atomica: the HIV care cascade by sex, with programs and three budget scenarios (identical to rounding error)
    - [x] EMULSION: the quickstart's three state machines (age, gestation, health), births on farrowing, minimum stays, a calendar, and a quarantine capacity (agrees within 1.8 animals with EMULSION's effective quarantine window)
    - [x] EpiModel: HIV with a care cascade and PrEP over two dynamic networks, four scenarios (agrees within the noise of the runs; 62 lines against 405, 2.5 s against 54)
    - [x] LASER: measles in England and Wales (53 places within 100 km of Norwich, at their real sizes, for 10 years), with births, life-table deaths, seasonality, importations, and gravity coupling (agrees: 106,466 infections against 106,639; 131 s against 3.0 s)
    - [x] SimInf: SISe3 in 1,600 herds with an environmental pressure and 783,773 events, aging by the register's `to_states` (fully declared; agrees within and between herds, once counts were settled after the register's events, as SimInf updates `phi`, and `phi`'s rates were made those of SimInf's daily update; 12 s a run as Gillespie against 0.1 s)
    - [x] TB: tbsim (branch `v1.0.0`). Done: `tbsim.TB` declared (front-loading, relapse, waning protection), with the care cascade, BCG, TPT, and HIV and nutrition as v4 products, deliveries, and modifiers; drug resistance as a strain dimension of `tbsim.TB` (`drugs=`, any number of drugs, people carrying up to `max_strains` at once), with `tbsim.LSHTMResistant` (the reference structure, stratified the same way) matching `two_strain_ode` to 1e-4 in every compartment, and as agents close to the ODE. migration as `immigration`/`emigration` parameters of `tbsim.TB` (with `ss.immigration()` and `ss.emigration()`). docs, tutorials, and examples ported. A second-line regimen (`second_line=`, its own state and transitions, cure of resistant strains, and relapse), started for people found resistant by a test for resistance in the care cascade (`tbsim.care(tb, dst=tbsim.DST(tb))`; as an ODE and as agents, 18 and 19 people on it on average); strain-aware TPT (`tbsim.TPT(tb=tb, drugs=['INH'])`, an efficacy by level of resistance: the share of the strains carried that its drugs act on); household contact tracing (`tbsim.contact_tracing(tb)`, as agents: the contacts of people starting treatment are tested, and treated or given TPT). Not ported: switching regimen during a course, clearing strains by TPT, and resistance acquired under TPT

## By section of the spec

### 01 Objects

- [x] `ss.Sim(model, ...)`, with modules named from their variables
- [x] `sim.new()`, `sim.describe()`, `sim.solvers()`, `sim.hash()`
- [x] `sim.run(n_runs=)` and `ss.parallel(name=sim, n_runs=)`
- [x] `ss.load()` of a model spec (`ss.from_spec()`), giving the same spec and hash: declared diseases (states, parameters, transitions, outcomes and results, values by group, time series, seasonal values, and functions of NumPy's in rates) and their stratifications, and the run settings. Not yet: networks, modifiers, interventions, Python functions, priors and likelihoods, movement between places, and joint splits, each of which raises `NotImplementedError` saying so
- [x] `sim.R0()` and `disease.R0()`, from the next-generation matrix at the disease-free state (with the sim's mixing, stratification, and places; a count that infection depends on, such as a water reservoir, counts as infected)
- [x] `disease.diagram()`: a Matplotlib figure, or the text of a Mermaid flowchart

### 02 Types

- [x] The unit rules for declared sims (`unit=`, typed `dt`, dates, typed values, timesteps)
- [x] Typed durations and rates in expressions
- [x] `W102 LargeStep`, for the solvers that take whole steps (including Euler with `substeps=` given, where outflows that would take more than everyone in a compartment are scaled down together, as Atomica does)
- [ ] The other unit notes

### 03 States and transitions

- [x] Arrow strings, `>>` and `@`, `ss.Transition`, and the class form
- [x] `ss.null`, births and deaths, branches with `p=`, competing hazards
- [x] People versus count states, inferred
- [x] Counts are real under every solver: deterministic, at their expected values, under `gillespie`, `tau`, and `binomial` (SimInf's environmental pressure died out when it was drawn as whole numbers), and advanced by RK4 over each step as agents (it was biased by 5% and a step early)
- [x] As agents, counts in rates are those at the start of the step for every transition (they depended on the order the states were declared)
- [x] Dwell-time distributions, `stages=`, `ss.erlang`
- [x] Functions as parameter values (the try-it rule)
- [x] NumPy functions of counts and parameters in a rate (`np.minimum`, `np.maximum`, `np.clip`, `np.exp`, `np.log`, `np.sqrt`, `np.abs`, `np.sin`, `np.cos`), and comparisons (`Q.n < 20`, 1 or 0), declared, and run in the kernels
- [x] `where=` on a transition (`ss.progression('J -> A', rate=..., where=ppl.age > 1)`): a modifier that sets its rate to 0 for everyone else, so it runs wherever a modifier with that condition does
- [ ] `states=` on transmission

### 04 Parameters

- [x] `ss.par` with bounds, priors, and docs; parameters named from their slots
- [x] `pars=` by name on the sim
- [x] `ss.timeseries` (constant, linear, cosine, spline) and `ss.seasonal`, in rates, durations, modifier amounts, coverage, and capacity
- [x] A time series inside `ss.by` (or `ss.seasonal`), in the keyword and table forms; tabulated by group for the kernels
- [ ] `requires=` and `sim.requires()`

### 05 Dimensions

- [x] `stratify()` on a disease and on the sim, `split=`, `adjust=` (of a transition, or of `rel_trans` and `rel_sus`)
- [x] `ss.transfer` as a rate (a parameter in it is registered, so `pars=` sets it, and `ss.modify(sir.vaccination, ...)` scales it, under every solver), `ss.by` (one dimension, a product over several, a table, or an array with named axes)
- [x] `split=` over several dimensions at once (`split=ss.by(pop, dims=['age', 'patch'])`); as agents, the other levels are drawn given the age group
- [x] Aging between age groups (it was off unless there were births or deaths; now on with an age stratification, unless `use_aging=False`)
- [x] `states=` on `stratify()`: dimensions of some states only, including states that are not people
- [x] Entering a dimension of some states only with a split that depends on another dimension (`split=ss.by(..., dims=['age', 'clinical'])`, each row the share entering each level), under every solver; an `adjust` that cannot apply, since the people it moves have no level of the dimension, raises `E306`
- [x] Strain dimensions: a force of infection for each strain, the infector's strain passed on, `adjust` by strain
- [x] Space: mixing within each place, counts per place, `ss.geo`, `ss.gravity`, and `ss.Mobility` (commuting, visiting, and migration)
- [x] Dimensions listed by `describe()`, with the states each applies to and the mobility
- [x] Initial conditions for each level of a dimension (`init=dict(I=ss.by(patch=[...]))`)
- [x] Age groups given as labels, kept apart from the age people have; `use_aging=False` in a compartmental run
- [x] Results by time point, as in other frameworks: `n_X[0]` is the start, `n_X[k]` the number at time point k, a flow at k the number during the step that ends at k; no step from `stop` (question 27). The loop skips the steps at the first time point; a declared model's step and a delivery take the time of the step's start (`ss.solvers.interval()`); the kernels and JAX record the start and then each step; a register's events dated at k are in the numbers at k; reporting periods end at their time points
- [x] Strains carried together: levels such as `'A+B'` (each strain carried needs a level of its own); someone at such a level passes on each strain in proportion to its infectiousness (`adjust=dict(rel_trans=...)`), and infection with a strain someone does not carry adds it, or leaves them as they are if they carry it or there is no level for it (a transmission that leaves people in the same state is then not counted). In every solver, and in the kernels and JAX
- [x] `move=` on a transition: a change of level in a dimension as well as of state, from one level (an arrow, which also limits the transition to people at that level) or from any
- [x] `immunity=` between strains: people in a state that has a strain and is not otherwise infected (`R`) can be infected by each strain, as far as their past strains do not protect them. A compartmental run keeps only the last strain (noted); agents keep every strain they have had. Not yet over networks as agents
- [ ] `where=` on `stratify()`
- [x] As agents, a split is assigned exactly (rounded to whole agents) in a random order
- [x] A register's `to_states` column moves the people chosen from each state to the matching one of another list (SimInf's shift matrix), under every solver
- [x] Movement registers (`ss.Mobility(data=)`): people moved between places, added, and removed on given dates, under every solver
- [ ] Named geographies (`ss.geo('Kenya')`)
- [x] A contact matrix by age within each place, in a compartmental run, with commuting, visiting, or a moving force of infection
- [ ] Networks that know about places, as agents (a contact matrix as agents mixes people across places)

### 06 Routes

- [x] Homogeneous mixing, `ss.MixingPool`, `ss.contacts` matrices
- [x] A contact matrix for each of the model's own dimensions, multiplied together (`ss.contacts(age=..., location=...)`)
- [x] A network described by target statistics (edges, concurrency, a cap on partners), and diagnostics of the network that was simulated (`ss.PartnerNet`, `net.check()`); partnerships ended by a partner dying are allowed for, as EpiModel's `d.rate` correction does, so the mean degree holds with births and deaths
- [x] Networks given by name (`networks=dict(main=..., casual=...)`), and an `ss.PartnerNet` named from its variable
- [ ] Other target statistics (mixing by attribute, triangles), and `n_contacts`/`concurrency` on `ss.MFNet`
- [x] Scheduled events that add, remove, or move individuals: one register, where a row with no `from` adds people and one with no `to` removes them. Each event happens at the start of the first step on or after its date, in the order of the table; people are chosen at random from the states it names (by default, everyone), and added to its states (by default, the first). In a compartmental run with several diseases, each draws its own people
- [ ] Bundled populations and contact matrices (`ss.contacts('Kenya')`)
- [x] Coupling between places that moves the force of infection, as LASER does (`ss.Mobility(kind='force')`)
- [x] Networks, as agents and (approximated, with a note) as compartments
- [x] Multi-disease coupling in a compartmental run: diseases that a condition links are crossed (`ss.Crossing`), each a dimension of the sim whose levels are its compartments of people; each engine holds everyone, and takes the people over from the engine that changed them last. A dimension of some states of one disease only (a strain, or ART) is allowed, if not of its first state (tbsim's TB with resistance and HIV with ART, as an ODE, agree with agents). Not yet: a dimension of some states of several of the diseases, a first state that splits on entry, an expression in one disease's rates that counts another's states, and the kernels taking the whole run in one call

### 07 Effects

- [x] `ss.modify` on a parameter, with `multiply=`, `add=`, `set=`, `start=`, `stop=`, `when=`
- [x] `where=` on `ss.modify`
- [x] Targets other than parameters: `rel_sus`, `rel_trans`, a transition
- [x] Time-varying amounts (a function of the sim, `ss.by`)
- [x] Lazy conditions (`ppl.age > 65`, `sir.I & ~sir.vaccinated`), on agents and on groups
- [x] Conditions on another disease's state, when both have a state of the same name
- [x] `sim.what_modifies()`, for declared modifiers
- [x] `W502 OverlappingSet`
- [ ] Network parameters as targets (`school.pars.n_contacts`)
- [ ] Base and effective values on arrays (`.base`), `track_writes=`
- [x] A condition on another module's state in a compartmental run (on a declared disease's state; see 06)
- [x] As agents, a multiplier on a transition whose dwell time has already been drawn: when it changes, what is left of each person's dwell time is rescaled by the change in the rate (a modifier that started during the run had no effect on the people already waiting)
- [ ] A multiplier on a transition that is a `p=` branch
- [x] Clipping of modified values to the quantity's range: a rate, duration, or multiplier made negative is 0, with `W503 ClippedToRange` once and a line in `what_modifies()` (a branch probability was already clipped to [0, 1])

### 08 Interventions

- [x] `ss.change`
- [x] Products: `ss.Vaccine`, `ss.Test`, `ss.Treatment`, with `efficacy=`, `waning=`, `cost=`
- [x] Deliveries: `ss.vaccinate`, `ss.screen`, `ss.treat`
- [x] Delivery keywords: `start=`, `stop=`, `years=`, `dates=`, `interval=`, `eligible=`, `coverage=`, `capacity=`
- [x] `ss.transfer` with delivery keywords
- [x] `on=` (event-triggered delivery) and `when=`
- [x] `deliver=` functions
- [x] Follow-ups: `on_positive=`, `on_negative=`, test `delay=` (agents only)
- [x] Delivery results: `delivered`, `eligible`, `coverage`, `cost`, `diagnosed`
- [x] Both kinds of solver: a product dimension in a compartmental run, per-agent records in an agent-based one
- [x] Scenarios: `sims.diff('baseline')`
- [x] Delivery results by stratum (`delivered`, `eligible`, `coverage`, `cost`, `diagnosed`), under every solver
- [ ] Waning by a function of time in a compartmental run (stages)
- [ ] A test delay in a compartmental run
- [x] Time series for `coverage=` and `capacity=` (`ss.timeseries`)
- [ ] `ss.Vx`, `ss.Dx`, `ss.Tx` as aliases of the new products
- [x] Contact tracing and partner notification: `trace=` on a delivery (`ss.screen(test, on=tb.diagnosis, trace='household')`) delivers to the contacts, in a network (or several), of the people `on=` selects or a test sends to it, with `eligible=` and `coverage=` applying to the contacts; agents only (`E804` otherwise). Ring vaccination of an SIR on a static network cuts infections from 48,800 to 570

### 09 Results

- [x] A count per state and a flow per transition, with the timing convention
- [x] Result expressions (sums and ratios of states)
- [x] `ss.outcome`, with delays, durations, and reporting periods
- [x] `res.select()` and `res.by()` on stratified results, including cumulative ones and outcomes (with their delays, durations, and reporting periods kept for each group); a state has only the dimensions it was stratified by
- [x] `select()` and `by()` on the results of many runs (kept by group in one kernel call unless that would take more than 500 MB)
- [x] A parameter as the weight of a flow in an outcome (`rho * infection`)
- [x] A set of states as their sum (`Ih / (Ia | Is | Ih)`, `(Ia | Is).n`), and a sum of states as the sources of a transition (`(Ia + Is) >> R`)
- [x] `select()` and `by()` chained (`res.n_I.select(age='0-4').by('sex')`); declared sums of states stratified, and ratios of two sums too (each keeps its numerator and denominator by group, adds each up over the groups selected, and then divides)
- [x] A flow into a dimension that its source does not have is recorded at the level people arrive at (summer2's incidence by clinical status), under every solver
- [ ] Flows in ratio results
- [x] Extinction reporting: `msim.extinction()` (not yet in `msim.summarize()`)

### 10 to 12 Execution, random numbers, and solvers

- [x] `ode` (RK4, or any `scipy` integrator), `binomial`, `tau`, `gillespie`, `sde`, `abm`
- [x] `auto` and `stochastic` solver choice, with the reason printed
- [x] Approximations noted, and refused with `approx=False`
- [x] `sim.loop.insert()` of a module, which moves its `step()`
- [x] `sim.check_dt()`
- [x] Numba kernels for the compartmental solvers (`starsim/kernels.py`): the flows and rate expressions are flattened into arrays once, and one kernel, the same for every model, takes a step of `rk4`, `euler`, or `binomial`. Falls back to Python where a rate is a function of the sim
- [x] A whole run in one call to the kernel, when the sim is nothing but compartments (no interventions, modifiers, transfers, or inserted functions); used by `sim.run()` without being asked
- [x] `sim.restart(pars=, rand_seed=)`: run a built sim again without building it again; used by `ss.fit()`
- [x] `sim.run(n_runs=)` as one call to the kernel for all the runs, when the sim is nothing but compartments
- [x] Values that vary in time are tabulated once for a sim, and kept when it is restarted
- [x] The kernel step as one function of loops over single numbers (a step of a three-compartment ODE takes 1.4 µs, from 8.5: passing the program to a Numba function costs about 0.4 µs each time)
- [x] A jump of a distribution only records its index; the generator is moved when it is next needed, which for draws for agents (hashed from the index) is never. Starsim 3 sims give the same results, 35% faster with 200 agents and 6% with 20,000
- [x] Agents: each state counted once a step, not twice; the level of each agent in each dimension kept with the group; exposure by place from two counts
- [x] A distribution whose parameters are plain values keeps what they were processed into, instead of working it out at every draw; `ss.normal` and the lognormals work out their inverse CDF directly (the same arithmetic as SciPy's, to the last bit). With the jumps, a Starsim 3 SIS sim of 200 agents takes 25 ms, from 54, with the same results
- [x] `solver="binomial"` takes a substep for each stage of the dwell time with the most stages (it moves people at most one stage a step, so with whole steps a dwell time in 49 stages of a mean of 7 days came out as 49 days). The LASER example as compartments now gives 1,427,000 recovered, against 1,394,000 as an ODE; it gave 198,000
- [x] A kernel for `tau`: the LASER example takes 6.7 s, from 93 s, and agrees with the ODE
- [x] Kernels for `gillespie` and `sde`, stepped and for a whole run. A Gillespie SIR in 100,000 people takes 0.16 s, from 4.2, and the malaria model in 1,000 people 0.45 s, from 27; the SDE was already cheap (17 ms, from 28). The SDE now takes `substeps=` in Python as well
- [ ] Kernels for the `scipy` integrators
- [x] Gillespie with many groups: where places do not affect each other within a step (people mix within their place, and no transition moves them between groups), each place has its own events over the step, and only its rates are worked out again after each one. The SimInf herds take 3.0 s, from 9.5 (SimInf: 0.08 s); a movement register of 466,692 events is read in 0.9 s, from 41
- [x] A whole run in the kernel with modifiers that depend only on time (no `where=`, `when=`, or function as the amount): the program is worked out for every step before the run. flepiMoP's scenarios with four modifiers no longer need the loop
- [x] A whole run with modifiers whose condition is only on states and stratifications (`where=hiv.acute`, as tbsim's drug resistance and its effects of HIV use), with transfers at a constant rate (e.g. starting ART), and with diseases tracked jointly: each takes each step in turn on one joint array, as they do step by step (the same to 1e-16, and in JAX to 1e-15). So TB with HIV, or with drug resistance, can be restarted, differentiated, and fitted by NUTS. For now the joint run is a loop in Python over the steps, with one kernel call a disease a step
- [ ] A whole run with deliveries, modifiers with `when=` or a function as their amount, and transfers whose rate varies (the run is stepped from Python, with one kernel call a step)
- [x] Many runs with different parameter values: `ss.parallel(sim, pars=dict(beta=[...]))`. With `integrator='jax'`, an ODE takes them all in one call to JAX: 1,000 runs of a year of an SIR in 0.5 s, from 1.2 s one at a time (the JAX run itself is 0.05 s; the rest is setting up each run and working out its results). Otherwise each set is one call to the kernels for all its seeds
- [x] Building a sim with modifiers or values that vary in time no longer builds a second, probe sim to try them on groups (they always work on groups), and a modifier's search for time parameters no longer walks into the modules its condition names: flepiMoP's scenarios build in 26 ms, from 62
- [x] Faster building of a sim: an SIR builds in 6 ms (from 13), and the TB model (30 years of weekly steps) in 22 ms (from 106), against a run of 8 ms. What is left is mostly Sciris's `sc.search()`, which records every object it visits (see `sciris_issues.md`)
- [x] Agents: under homogeneous mixing with one strain, the probability of infection for everyone susceptible and exposed in one compiled pass (the rate is worked out once for each group, and the draw is the one made before, so the same agents are infected); each state counted by group in one pass; the level of each agent in a dimension reused until the property changes. The LASER example takes 3.5 ms a step, from 4.4 (8.5 at the start)
- [x] Agents: each agent's group is worked out again only for the agents whose level changed (births, moves, a new strain), which the code that changes it says (`sim.groups_changed(uids)`), and only dimensions binned from a property that changes on its own (age) are checked every step; and a state's members are read from its array rather than through the living agents. LASER's England and Wales model as agents runs 38% faster (100 days in 2.1 s, from 3.4), with the same results
- [ ] Agents: the same for scheduled exits, exposure by place, and the counts of all the states at once: perhaps another 20%. A pass over 175,000 agents costs 0.1 to 0.2 ms however it is written, and a step needs about fifteen
- Not planned: a whole agent-based run in one kernel, as for compartments. A prototype of the best case (one compiled pass over 175,000 agents a step) took 1.1 ms a step on one thread, against 4.4 ms now, so 4 times at most; it reaches the 1 ns for each agent of LASER only on 20 threads, which Starsim usually spends on running sims side by side. And it would run only models with nothing in Python between steps, which are the ones the compartmental solvers already run
- [ ] The chain-binomial solver with many stages: a substep for each stage (40 a day for LASER's latent period; fewer makes each stage too long). Moving people several stages in one substep, as many as an Erlang chain would in that time, would be exact with one substep
- [ ] Gillespie thinning for time-varying rates (a rate is held constant within a step)
- [x] Values that vary smoothly in time (`ss.seasonal`, and `ss.timeseries` other than `mode='constant'`) read at the time of each stage of an RK4 step, in the kernels (from their values at the start, middle, and end of each step) and in Python (and by the `scipy` integrators). A step function is held for the whole step, as `ss.change()` is. MEmilio's age-resolved SECIR at a step of a day went from 9 people in 10,000 to 1.5, the error of the reference itself
- [x] JAX and gradients (`starsim/gradients.py`): `integrator='jax'` runs the same RK4 (or Euler) steps as the kernels in JAX, from the same program, and matches them to 1e-15. `sim.gradient(results, pars)` gives the derivative of each result with respect to each parameter; it agrees with finite differences to 1e-10, including through a function in a rate (`np.exp(-I.frac)`) and an outcome with a delay. `sim.calibrate()` on an ODE fits by L-BFGS-B with these gradients when JAX is installed: with 8 parameters it recovers the true values in 110 evaluations (6 s), where Nelder-Mead stops at its limit of 1,600 without them. With 3 it is slower than Nelder-Mead (1.7 s against 0.25, including about 1 s to compile), since each gradient costs as much as about 8 runs. `method='asd'` (`sc.asd()`) does no better than Nelder-Mead on these: on the 3-parameter fit it stops after 1,000 evaluations with a mismatch of 1e-3 (Nelder-Mead: 1e-6 in 190), and on the 8-parameter fit at 0.02 after 2,800 (Nelder-Mead: 0.03 after 1,600)

### 13 Inference

- [x] `ss.data`, `sim.calibrate()`, `ss.Fit`
- [x] Fitting a parameter wherever it is used: in an initial count (`init=dict(E=I0/2, I=I0)`), inside `ss.seasonal` or `ss.timeseries` (`mean=(1 + smin)/2`), or shared by several modules or by models built more than once (registered with each, and fitted once)
- [x] Data matched by name to summed results: data by stratum in long form (a column named after a dimension says which stratum a row is for), as one series per result and stratum (`n_I[age=0-14]`) compared with `n_I.select(age='0-14')`, in every fit and in the log-density in JAX. Not yet: `sim.plot()` drawing data by stratum (`fit.plot()` does)
- [x] Likelihoods (`ss.poisson`, `ss.nbinom`, `ss.normal`, `ss.lognorm_ex`, with the model's value as the mean), fitted by maximum likelihood; `ss.compare_models`, with `holdout=` and `folds=`
- [x] NUTS and priors: `sim.calibrate()` (the default when every series has a likelihood; `ss.compare_models()` still fits a point estimate for each fold), with BlackJAX and the log-density as one function in JAX (`ss.gradients.Density`: the numbers in the run from the parameters, the run, the results with data, the likelihood, and the priors), checked against runs of the sim at a few draws; chains on cores of their own; rank-normalized R-hat, bulk ESS, and divergences (`W1304 NotConverged`). Against camdl's ODE NUTS on its SIR (2 parameters, 4 chains of 500 + 500 draws, RK4 with steps of a day): the same posterior; sampling 2.5 s against about 2.2 s, with a bulk ESS of about 310 against 211, so more effective draws per second; but 1.9 s of compiling (0.9 s when it has been compiled before, from the cache in `ss.options.jax_cache`) and 1.7 s of imports, so 5.5 s from the command line against camdl's 2.3 s. The log-density covers several diseases, including ones tracked jointly; each step of the run in JAX takes only the substeps it needs (a loop with a fixed count took the most allowed every time), which makes a gradient of TB with HIV 4.5 times faster (57 ms). TB with HIV (tbsim, as an ODE tracked jointly, with ART started at a rate), fitted to yearly HIV prevalence and symptomatic TB drawn around a run with an HIV incidence of 0.015 and a `beta` of 9: medians 0.0151 (0.0149 to 0.0155) and 9.24 (8.80 to 9.62), from 4 chains of 200 + 200 draws in 2 minutes (5 s of compiling). Outcomes that people stay in (`duration=`) are covered too, with data on `n_<outcome>`. Not yet: in JAX, modifiers of fitted parameters; a parameter inside a distribution (`dur=ss.gamma(a=3, scale=par)`) is not supported in any run yet (a fitted dwell time is `dur=par` with `stages=`) (these are sampled with Python in the loop, one chain at a time); staged fitting (`method=[...]`)
- [x] Particle MCMC (`method='pmcmc'`, the default for a stochastic compartmental sim whose data series all have a likelihood: `sde`, `binomial`, `tau`, or `gillespie`): a random-walk Metropolis sampler (`ss.calibration.pmmh()`, with the proposal's covariance learned in the warm-up and its scale tuned toward 15% accepted) on the likelihood estimated by a bootstrap particle filter (`ss.ParticleFilter`). The particles are stepped together by the kernels (`ss.kernels.advance()`) from one data point to the next, resampled systematically when their weights become uneven, and keep the steps that later data points read (all of them, for a result such as an outcome with a delay). Each disease is filtered on its own. The chains start from the best of the initial values and a spread of points, and run in parallel processes; each uses 100 particles for the first half of its warm-up (estimating its current point again at each step, so a lucky estimate far from the posterior cannot hold it), then enough that the estimate's variance at the middle of where it has been lately is about 1. Defaults: 4 chains of 1,000 + 1,000 draws. An SIR fitted to weekly cases in 10,000 people (`beta` and `dur`, from 0.3 and 10; truth 0.5 and 6): 50 particles, medians 0.517 (0.476 to 0.562) and 6.00 (4.96 to 7.70), R-hat 1.014, bulk ESS about 300, in 88 s. `binomial` and `tau` are as fast; `gillespie` works but is slow (exact events in 10,000 people). Not yet: reporting particle degeneracy; a sim with several diseases tracked jointly, or a stratum of a result that is not a state or a transition
- [x] No ABC-SMC: an agent-based sim is fitted by Optuna instead (see departures)
- [x] Identifiability notes (`W1302`): from the model, two fitted parameters that appear only multiplied together, or one divided by the other, with the product or ratio the data determine; from the fit, two almost perfectly correlated over a posterior (|r| > 0.95, of their logs where searched on a log scale), or, for a fit by gradients, two whose effects on the fitted results at the best fit are almost the same (|r| > 0.999, one gradient). Not yet: fitting a product directly (`pars=[beta * contacts]`)

### 14 to 16 Model spec, errors, and interop

- [x] `sim.to_spec()` and the model hash
- [x] Coded errors and notes (`E204`, `E804`, `W1205`, ...)
- [x] `W1201 SmallCounts` (an ODE or SDE that starts with fewer than 10 infected) and `W1202 ExtinctionPossible` (the number infected falls below one and rises again)
- [ ] The notes `W1203` and `W1204` (`W102` says what `W1204` would)
- [ ] Interop with other formats

## Departures from the spec

- Sims with no declared states keep Starsim 3's time defaults.
- Rates are written as strings in the model spec, and the spec is `sim.to_spec()`; `sim.to_json()` is unchanged from Starsim 3. `ss.load()` reads the strings back with a parser that accepts only arithmetic, comparisons, the functions allowed in rates, typed quantities (`perday(0.5)`), and the names the module declares, so loading a spec runs no code from it.
- The default ODE integrator is fixed-step RK4 with automatic substeps, in the Numba kernels, with or without gradients being asked for; JAX is used only for `integrator='jax'`, `sim.gradient()`, and gradient-based fitting. `integrator='jax'` takes the same RK4 steps (not an adaptive JAX integrator), so its results are the kernels'.
- What JAX differentiates is the run, from the numbers in the program and the compartments at the start to the compartments and flows at every step. The steps either side are Python (the parameters to those numbers, and the run to the results, outcomes, and goodness of fit), and are differentiated by central differences along the direction of each parameter. So a gradient is exact to about 1e-10, not to the last bit, and costs two restarts and two runs' worth of results for each parameter.
- A stratified population has ages from 0 to 100.
- Every agent's dwell time is sampled, including exponential ones.
- The property of people that says they hold a product is named `has_<product>` (e.g. `has_bcg`), and the record that a routine delivery has been offered is `offered_<delivery>`. The product's own name is taken by the product module.
- A product whose `waning=` is a duration is lost at an exponentially distributed time, under every solver, after which the person can receive it again. Only a function of time since delivery fades the efficacy while the person still holds the product.
- A delivery never gives a product to someone who already holds it.
- A bare number in `ss.change(at=...)` or `start=` is a year only if the sim has dates; otherwise it is a time in the sim's unit.
- A modifier acts where the engine uses the quantity; reading `sir.rel_sus` from custom code gives the base value.
- A module written as code that works on groups as well as on agents says so with the class attribute `compartmental = True`; otherwise it makes the model agent-based.
- The peak of `ss.seasonal` given as a month is the middle of that month. In a sim with no dates, time starts at the beginning of a year.
- `ss.geo` takes the places directly (names, populations, and positions or distances); there are no named geographies.
- `ss.gravity` gives `k × pop[i]^(a−1) × pop[j]^b / distance^c`: for commuting, the share of time a person from `i` spends in `j` (capped so that it sums to at most 1, or rescaled to `away=`), and for migration the rate of moving.
- An inflow whose rate uses a count (`xi * I.n`, or `ss.birth`) arrives in each place according to the count there; an inflow at a fixed rate is shared between places by their split.
- A new infection is recorded under the strain acquired, and a state that a dimension does not apply to is counted at that dimension's first level.
- As agents, the number who start in each state is exact (rounded to whole agents), not a chance for each agent.
- A level of a stratification named like an attribute of a result (`low`, `high`) is a column called `<dimension>_<level>` in `result.by()`.
- There is no ABC-SMC: a stochastic sim without a likelihood for every data series, and any agent-based sim, is fitted by Optuna (a best fit, not a posterior), which in practice works better for agent-based models; a stochastic compartmental sim with a likelihood for every series gets a posterior from particle MCMC.

- `n_<state>[ti]` is the number after step `ti` under every solver, so it agrees with the flows of the step (`n_R[ti] - n_R[ti-1] == recovery[ti]`); the number before the first step is not a result. A result's time vector still labels step `ti` with its start, so data compared with a stock is compared with the number one step later than its label says.

## Known issues

- The kernels are compiled the first time they are used after Starsim is installed or changed, which takes about 12 s; after that they are loaded from a cache.
- With `sim.run(n_runs=)` in one call to the kernel, only the states and transitions are kept by group (not outcomes or results given as expressions), and reported outcomes are drawn for all the runs at once, so differ (in the draw, not the distribution) from those of a single run with the same seed.
- The kernels draw random numbers from Numba's generator, so a stochastic compartmental run gives different numbers (from the same distribution) with `engine.use_kernels = False`.
- A typed quantity that is changed in place during a run (the number inside an `ss.perday`, say) is not seen, since a converted value is kept for as long as the parameter holds the same object; assigning a new value is.
- A typed rate cannot be used as a plain number in an expression (`1 - ss.peryear(0.008)`).
- `sim.new(n_agents=None)` does not turn an agent-based sim into a compartmental one; the argument has to be left out.
- A bare number as the duration of a network's contacts is in years, even in a sim that counts time in steps.
- With `stop=ss.weeks(12)` and no `unit=`, the sim runs in weeks even if every rate is per day.
- As agents, a dwell time is sampled when someone enters a state, so an `adjust` on a duration does not follow a person who later moves to another level (through an `ss.transfer`, or by aging). A rate does.
- A strain's initial share is the split of the dimension (equal by default), which applies to everyone who starts in an infected state.
- A function of the sim that compares ages with a number that is not a bin edge runs in a compartmental model without complaint (each group has the age of its midpoint); the lazy form (`ppl.age > 70`) is checked.
- As agents, someone tested on the step they were infected may already have recovered (about `dt/2` of a dwell time), so event-triggered tests find slightly fewer cases than in a compartmental run.
- An ODE and an agent-based run of the same model differ by an amount of order `dt` (about 6% in the prevalence of an SIS model with a daily step).
- As agents, people added by a movement register in a step are made at its end, so the other events of the same step cannot choose them; and people who moved into a place in a step are chosen after those who were already there.
- A movement register cannot add people to a state that splits on entry into branches.
- A condition on a property of people (`ppl.age < 5`) needs a `ppl` object (`ss.People(1)`) even in a compartmental model, which has none.
- Arithmetic on stratified results keeps the values by group only for sums, differences, and scaling by a number; for anything else (a ratio), `select()` says the result is not stratified. (A ratio declared on the disease, `prev=I/N`, is stratified: it keeps its numerator and denominator.)
- With `immunity=` between strains, a compartmental run keeps only each person's last strain (noted), so with partial protection people can go from A to B and back to A; as agents, every strain is kept, but not yet over networks.
- Code that changes a property of people that a dimension is made from (`people.patch`) has to call `sim.groups_changed(uids)`, or the agents' groups are not worked out again (dimensions binned from a property that changes on its own, such as age, are checked every step).
- A state that is not people is deterministic under the stochastic solvers, so a vector population whose own randomness matters cannot be drawn.
- Under the exact solver, places are run one at a time between steps only when people mix within their place and nothing moves them between groups within a step; otherwise every rate is worked out again after each event.
- A plain `duration=` on an outcome is an exponential stay, while `delay=` is an exact shift.
- As agents, TB (LSHTM) comes out 5–10% higher than the ODE with weekly steps, and 20–30% higher with monthly ones; this is the difference of order `dt` above, made larger by an epidemic that is near R = 1.
- `rel_sus` and `rel_trans` on a state are numbers, not `ss.par`s, so they cannot be fitted; the TB models write them into the rates instead.
- As agents, a delivery comes after the step's progressions (which a declared model takes in `step_state()`, before the interventions) and before its transmission, while in a compartmental run it comes before all of the step's flows. So a delivery as agents misses the people who progress in that step: TPT on the first step reached 7% fewer people with a latent infection that leaves at 3.9 per year with weekly steps.
- Once JAX has run in a process (`sim.gradient()`, a gradient fit, or `integrator='jax'`), running sims in parallel by forking processes (`ss.parallel()` of agent-based sims, Optuna with several workers) can deadlock, as JAX warns. Spawning instead would be safe but takes about 1.5 s longer to start.
- Gradients need a sim that the kernels run from start to finish (no interventions, modifiers with conditions, transfers, or movement register), and a rate that is a Python function of the sim has none (one written with NumPy functions of counts does).
- The JAX run is compiled the first time it is used for a model (about 1 s), and again if a step needs more substeps than it has allowed for.
