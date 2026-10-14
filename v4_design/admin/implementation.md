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
- [ ] Wave 3: one complex example from each framework, Ghana measles, and TB (the models still to be chosen, except camdl's)
    - [x] camdl: SEIR by age and patch, with contacts by age within each patch, aging, movement, and a vaccination (agrees as an ODE; as chain-binomial, the same peaks a day sooner at a step of a day)
    - [x] Ghana measles: stochastic SEIR in 260 districts with gravity mixing, seasonality, births with vaccination, deaths, importation, and reported cases (the `gm-port` rewrite; agrees, with no change to Starsim)

## By section of the spec

### 01 Objects

- [x] `ss.Sim(model, ...)`, with modules named from their variables
- [x] `sim.new()`, `sim.describe()`, `sim.solvers()`, `sim.hash()`
- [x] `sim.run(n_runs=)` and `ss.parallel(name=sim, n_runs=)`
- [ ] `ss.load()` of a model spec
- [ ] `sim.diagram()`, `sim.R0()`

### 02 Types

- [x] The unit rules for declared sims (`unit=`, typed `dt`, dates, typed values, timesteps)
- [x] Typed durations and rates in expressions
- [x] `W102 LargeStep`, for the solvers that take whole steps
- [ ] The other unit notes

### 03 States and transitions

- [x] Arrow strings, `>>` and `@`, `ss.Transition`, and the class form
- [x] `ss.null`, births and deaths, branches with `p=`, competing hazards
- [x] People versus count states, inferred
- [x] Dwell-time distributions, `stages=`, `ss.erlang`
- [x] Functions as parameter values (the try-it rule)
- [ ] `states=` on transmission, `where=` on a transition

### 04 Parameters

- [x] `ss.par` with bounds, priors, and docs; parameters named from their slots
- [x] `pars=` by name on the sim
- [x] `ss.timeseries` (constant, linear, spline) and `ss.seasonal`, in rates, durations, modifier amounts, coverage, and capacity
- [ ] A time series inside `ss.by`
- [ ] `requires=` and `sim.requires()`

### 05 Dimensions

- [x] `stratify()` on a disease and on the sim, `split=`, `adjust=`
- [x] `ss.transfer` as a rate, `ss.by` (one dimension, a product over several, a table, or an array with named axes)
- [x] `split=` over several dimensions at once (`split=ss.by(pop, dims=['age', 'patch'])`); as agents, the other levels are drawn given the age group
- [x] Aging between age groups (it was off unless there were births or deaths; now on with an age stratification, unless `use_aging=False`)
- [x] `states=` on `stratify()`: dimensions of some states only, including states that are not people
- [x] Strain dimensions: a force of infection for each strain, the infector's strain passed on, `adjust` by strain
- [x] Space: mixing within each place, counts per place, `ss.geo`, `ss.gravity`, and `ss.Mobility` (commuting, visiting, and migration)
- [x] Dimensions listed by `describe()`, with the states each applies to and the mobility
- [x] Initial conditions for each level of a dimension (`init=dict(I=ss.by(patch=[...]))`)
- [x] Age groups given as labels, kept apart from the age people have; `use_aging=False` in a compartmental run
- [ ] `immunity=` between strains
- [ ] `where=` on `stratify()`
- [x] Movement registers (`ss.Mobility(data=)`): people moved between places, added, and removed on given dates, under every solver
- [ ] Named geographies (`ss.geo('Kenya')`)
- [x] A contact matrix by age within each place, in a compartmental run, with commuting, visiting, or a moving force of infection
- [ ] Networks that know about places, as agents (a contact matrix as agents mixes people across places)

### 06 Routes

- [x] Homogeneous mixing, `ss.MixingPool`, `ss.contacts` matrices
- [x] A contact matrix for each of the model's own dimensions, multiplied together (`ss.contacts(age=..., location=...)`)
- [x] A network described by target statistics (edges, concurrency, a cap on partners), and diagnostics of the network that was simulated (`ss.PartnerNet`, `net.check()`)
- [ ] Other target statistics (mixing by attribute, triangles), and `n_contacts`/`concurrency` on `ss.MFNet`
- [x] Scheduled events that add, remove, or move individuals: one register, where a row with no `from` adds people and one with no `to` removes them. Each event happens at the start of the first step on or after its date, in the order of the table; people are chosen at random from the states it names (by default, everyone), and added to its states (by default, the first). In a compartmental run with several diseases, each draws its own people
- [ ] Bundled populations and contact matrices (`ss.contacts('Kenya')`)
- [x] Coupling between places that moves the force of infection, as LASER does (`ss.Mobility(kind='force')`)
- [x] Networks, as agents and (approximated, with a note) as compartments
- [ ] Multi-disease coupling in a compartmental run

### 07 Effects

- [x] `ss.modify` on a parameter, with `multiply=`, `add=`, `set=`, `start=`, `stop=`, `when=`
- [x] `where=` on `ss.modify`
- [x] Targets other than parameters: `rel_sus`, `rel_trans`, a transition
- [x] Time-varying amounts (a function of the sim, `ss.by`)
- [x] Lazy conditions (`ppl.age > 65`, `sir.I & ~sir.vaccinated`), on agents and on groups
- [x] Conditions on another disease's state, when both have a state of the same name (agents only)
- [x] `sim.what_modifies()`, for declared modifiers
- [x] `W502 OverlappingSet`
- [ ] Network parameters as targets (`school.pars.n_contacts`)
- [ ] Base and effective values on arrays (`.base`), `track_writes=`
- [ ] A condition on another module's state in a compartmental run
- [ ] A multiplier on a transition that is a `p=` branch, or (as agents) one whose dwell time has already been sampled
- [ ] Clipping of modified values to the quantity's range

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
- [ ] Delivery results by stratum
- [ ] Waning by a function of time in a compartmental run (stages)
- [ ] A test delay in a compartmental run
- [x] Time series for `coverage=` and `capacity=` (`ss.timeseries`)
- [ ] `ss.Vx`, `ss.Dx`, `ss.Tx` as aliases of the new products
- [ ] Contact tracing and partner notification

### 09 Results

- [x] A count per state and a flow per transition, with the timing convention
- [x] Result expressions (sums and ratios of states)
- [x] `ss.outcome`, with delays, durations, and reporting periods
- [x] `res.select()` and `res.by()` on stratified results, including cumulative ones; a state has only the dimensions it was stratified by
- [x] `select()` and `by()` on the results of many runs (kept by group in one kernel call unless that would take more than 500 MB)
- [x] A parameter as the weight of a flow in an outcome (`rho * infection`)
- [ ] Flows in ratio results
- [ ] Extinction reporting

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
- [ ] Gillespie with many groups: every rate is worked out again after each event, though in a model where places do not mix within a step only the place of the event changes. The SimInf herds take 9.5 s (about 18 µs an event, over 1,600 herds), against 0.08 s for SimInf, which runs each herd on its own
- [ ] A whole run in the kernel with modifiers, transfers, and deliveries (the run is stepped from Python, with one kernel call a step)
- [ ] Many runs in one call with different parameter values (a restarted run is 0.5 to 1 ms, most of it Python around the kernel)
- [ ] Faster building of a sim (20 to 50 ms, which is now nearly all of a single run)
- [x] Agents: under homogeneous mixing with one strain, the probability of infection for everyone susceptible and exposed in one compiled pass (the rate is worked out once for each group, and the draw is the one made before, so the same agents are infected); each state counted by group in one pass; the level of each agent in a dimension reused until the property changes. The LASER example takes 3.5 ms a step, from 4.4 (8.5 at the start)
- [ ] Agents: the same for scheduled exits, exposure by place, and the counts of all the states at once: perhaps another 20%. A pass over 175,000 agents costs 0.1 to 0.2 ms however it is written, and a step needs about fifteen
- Not planned: a whole agent-based run in one kernel, as for compartments. A prototype of the best case (one compiled pass over 175,000 agents a step) took 1.1 ms a step on one thread, against 4.4 ms now, so 4 times at most; it reaches the 1 ns for each agent of LASER only on 20 threads, which Starsim usually spends on running sims side by side. And it would run only models with nothing in Python between steps, which are the ones the compartmental solvers already run
- [ ] Gillespie thinning for time-varying rates (a rate is held constant within a step)
- [ ] Time-varying values read at the times the ODE integrator evaluates the rates (they are read once a step)
- [ ] JAX and gradients

### 13 Inference

- [x] `ss.data`, `sim.calibrate()`, `ss.Fit`
- [ ] Data matched by name to summed results
- [x] Likelihoods (`ss.poisson`, `ss.nbinom`, `ss.normal`, `ss.lognorm_ex`, with the model's value as the mean), fitted by maximum likelihood; `ss.compare_models`, with `holdout=` and `folds=`
- [ ] Posterior methods (NUTS, particle MCMC, ABC-SMC) and priors
- [ ] Identifiability notes

### 14 to 16 Model spec, errors, and interop

- [x] `sim.to_spec()` and the model hash
- [x] Coded errors and notes (`E204`, `E804`, `W1205`, ...)
- [ ] The notes `W1201` to `W1204`
- [ ] Interop with other formats

## Departures from the spec

- Sims with no declared states keep Starsim 3's time defaults.
- Rates are written as strings in the model spec, and the spec is `sim.to_spec()`; `sim.to_json()` is unchanged from Starsim 3.
- The default ODE integrator is fixed-step RK4 with automatic substeps.
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
- Arithmetic on stratified results keeps the values by group only for sums, differences, and scaling by a number; for anything else (a ratio), `select()` says the result is not stratified.
