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
    - [x] TB: tbsim (branch `v1.0.0`). Done: `tbsim.TB` declared (front-loading, relapse, waning protection), with the care cascade, BCG, TPT, and HIV and nutrition as v4 products, deliveries, and modifiers; drug resistance as a strain dimension of `tbsim.TB` (`drugs=`, any number of drugs, people carrying up to `max_strains` at once), with `tbsim.LSHTMResistant` (the reference structure, stratified the same way) matching `two_strain_ode` to 1e-4 in every compartment, and as agents close to the ODE. migration as `immigration`/`emigration` parameters of `tbsim.TB` (with `ss.immigration()` and `ss.emigration()`). docs, tutorials, and examples ported. A second-line regimen (`second_line=`, its own state and transitions, cure of resistant strains, and relapse), started for people found resistant by a test for resistance in the care cascade (`tbsim.care(tb, dst=tbsim.DST(tb))`; as an ODE and as agents, 18 and 19 people on it on average); strain-aware TPT (`tbsim.TPT(tb=tb, drugs=['INH'])`, an efficacy by level of resistance: the share of the strains carried that its drugs act on); household contact tracing (`tbsim.contact_tracing(tb)`, as agents: the contacts of people starting treatment are tested, and treated or given TPT); switching to second line during a course after a test for resistance (`tbsim.monitoring(tb)`, through `switch_second_line`, with a treatment that moves people out of any state); TPT clearing the strains it acts on and resistance acquired under it (`tbsim.TB(tpt=...)`: transitions whose `where=` is holding a TPT product, so a sim without TPT has none of them; as rates while TPT is held, where the old tbsim had outcomes at the end of a course); as agents, testing for resistance only people treated within some time before (`tbsim.care(..., retreatment=ss.years(2))`, from each agent's time of starting treatment, which a compartmental model does not keep). The ODE and agents agree on all of these (e.g. 516 and 544 people switched; 2,262 and 2,210 resistance acquired under TPT)

## By section of the spec

### 01 Objects

- [x] `ss.Sim(model, ...)`, with modules named from their variables
- [x] `sim.new()`, `sim.describe()`, `sim.solvers()`, `sim.hash()`
- [x] `sim.run(n_runs=)` and `ss.parallel(name=sim, n_runs=)`
- [x] `ss.load()` of a model spec (`ss.from_spec()`) (later removed), giving the same spec and hash: declared diseases (states, parameters, transitions, outcomes and results, values by group, time series, seasonal values, and functions of NumPy's in rates) and their stratifications, and the run settings. Not yet: networks, modifiers, interventions, Python functions, priors and likelihoods, movement between places, and joint splits, each of which raises `NotImplementedError` saying so
- [x] `sim.R0()` and `disease.R0()`, from the next-generation matrix at the disease-free state (with the sim's mixing, stratification, and places; a count that infection depends on, such as a water reservoir, counts as infected)
- [x] `disease.diagram()`: a Matplotlib figure, or the text of a Mermaid flowchart

### 02 Types

- [x] The unit rules for declared sims (`unit=`, typed `dt`, dates, typed values, timesteps)
- [x] Typed durations and rates in expressions
- [x] `W102 LargeStep`, for the solvers that take whole steps (including Euler with `substeps=` given, where outflows that would take more than everyone in a compartment are scaled down together, as Atomica does)
- [x] The other unit notes: `W103 UnitsDisagree`, naming the bare numbers the smallest unit was applied to (`W101` was the only other)

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
- [x] `states=` on transmission: superseded, since the spec has none; which states infect is `source=`, and who can be infected is `where=`

### 04 Parameters

- [x] `ss.par` with bounds, priors, and docs; parameters named from their slots
- [x] `pars=` by name on the sim
- [x] `ss.timeseries` (constant, linear, cosine, spline) and `ss.seasonal`, in rates, durations, modifier amounts, coverage, and capacity
- [x] A time series inside `ss.by` (or `ss.seasonal`), in the keyword and table forms; tabulated by group for the kernels
- [x] `requires` and `sim.requires()`: a module's list of `ss.Requires`, and for a declared model its parameters, as a table of what is given and what is missing, without initializing the sim (so it works when a missing value would stop it). Starsim 3's `self.requires = HIV` (another module) is shown as a required module

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
- [x] `where=` on `stratify()`: people outside it have no level of the dimension (selections, `by()`, conditions, and `adjust=` leave them out; values by level give them the split-weighted mean, or none of a number of people; they do not move between its levels), under every solver; nested `where=` too. In a compartmental run the groups are still a grid, with the people outside counted at the first level and the other groups empty. Contacts by such a dimension are `E307`, which is now also raised for `states=` that leave out the people who transmit or are infected
- [x] As agents, a split is assigned exactly (rounded to whole agents) in a random order
- [x] A register's `to_states` column moves the people chosen from each state to the matching one of another list (SimInf's shift matrix), under every solver
- [x] Movement registers (`ss.Mobility(data=)`): people moved between places, added, and removed on given dates, under every solver
- [x] Named geographies (`ss.geo('Kenya', level=1)`), from `starsim_data`: names, populations, centroids, distances, and which places share a border (`geo.adjacency`), for the first two administrative levels of 184 countries, bundled (no download); `places=` for a subset, several countries at once, and other years or finer levels fetched (`level=2` gives Kenya's 290 sub-counties). The level-1 populations are WorldPop's 2020 1 km rasters summed within each unit when `starsim_data` is built (a populated cell within 10 km of a unit but in none counts for the nearest, since boundaries are simplified at the coast): for 183 countries they come to 0.93 to 1.015 of the UN total (median 0.994); Kiribati, whose raster is empty, shares its total by area, with a warning. Where the places came from is in `describe()`, the model spec, and the hash, and `to_code()` writes `ss.geo('Kenya', level=1)`
- [x] A contact matrix by age within each place, in a compartmental run, with commuting, visiting, or a moving force of infection
- [x] Networks that know about places, as agents: a contact matrix (`ss.contacts()`) and a mixing pool of everyone act within each place, from the agents' counts in each group, as in a compartmental run (`ss.Mixing.init_pools()`), and a random network (`ss.RandomNet`, `ss.RandomExactNet`) pairs people within their place; with commuting or visiting, each contact is in a place drawn from the share of time spent in each. Agents agree with the ODE to within 2% in each place, with and without commuting (the epidemic had reached every place as agents)

### 06 Routes

- [x] Homogeneous mixing, `ss.MixingPool`, `ss.contacts` matrices
- [x] A contact matrix for each of the model's own dimensions, multiplied together (`ss.contacts(age=..., location=...)`)
- [x] A network described by target statistics (edges, concurrency, a cap on partners), and diagnostics of the network that was simulated (`ss.PartnerNet`, `net.check()`); partnerships ended by a partner dying are allowed for, as EpiModel's `d.rate` correction does, so the mean degree holds with births and deaths
- [x] Networks given by name (`networks=dict(main=..., casual=...)`), and an `ss.PartnerNet` named from its variable
- [x] Other target statistics, and `n_contacts`/`concurrency` on `ss.MFNet`: `ss.PartnerNet(match=dict(risk=0.9))`, the share of partnerships between people at the same level of a property (EpiModel's nodematch; recovered to within 0.01), and `ss.MFNet(n_contacts=, concurrency=, max_contacts=)`, the same targets as `ss.PartnerNet` (shared in `ss.DegreeTargets`), between men and women, among people who take part after their debut; without `n_contacts`, `ss.MFNet` is unchanged. Triangles are not covered (they need an ERGM fit)
- [x] Scheduled events that add, remove, or move individuals: one register, where a row with no `from` adds people and one with no `to` removes them. Each event happens at the start of the first step on or after its date, in the order of the table; people are chosen at random from the states it names (by default, everyone), and added to its states (by default, the first). In a compartmental run with several diseases, each draws its own people
- [x] Bundled populations and contact matrices (`ss.contacts('Kenya')`)
    - [x] Contact matrices by name: `ss.contacts('Kenya', layer='home', source='prem2021')`, from `starsim_data` (`prem2021` is bundled, for 177 countries; `prem2017` and `mistry2021` are fetched on first use); a country a source lacks is told which sources have it. A named matrix is written in the model spec as where it came from, so `from_spec()` and `to_code()` rebuild it
    - [x] Populations by name: `ss.people('Kenya', year=2024)`, the UN World Population Prospects' population by 5-year age group and sex (bundled for 187 countries, every ten years from 1950 to 2100, interpolated in between). It gives the sim its size (`total_pop`, unless one is given) and the split of an age stratification that has none of its own; as agents, their ages and the share who are female. Without `n_agents=`, it does not make the model agent-based; without `year=`, it is the year the sim starts, if it has dates. It is written in the model spec as where it came from, so `from_spec()` and `to_code()` rebuild it
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
- [x] Network parameters as targets: `ss.modify('school.pars.n_contacts', ...)`, as a path, since a network's `pars` hold plain values. In a compartmental run it scales the network's contact rate (or matrix), and the run is stepped from Python; as agents, each contact happens with that probability in a step (in a mixing pool, the hazard is scaled). With `where=`, for the people who would be infected. ODE and agents agree to within the difference of order `dt` (3.5% at a step of a day, 1.4% at a quarter)
- [x] Base and effective values on arrays, `track_writes=`: as agents, `sir.rel_sus.effective` (an `ss.ModifiableArr`), since reading gives the base (see departures); `ss.Sim(track_writes=True)` makes `rel_sus` and `rel_trans` record which module assigned them, listed by `sim.what_modifies()`, at no cost when off
- [x] A condition on another module's state in a compartmental run (on a declared disease's state; see 06)
- [x] As agents, a multiplier on a transition whose dwell time has already been drawn: when it changes, what is left of each person's dwell time is rescaled by the change in the rate (a modifier that started during the run had no effect on the people already waiting)
- [x] A multiplier on a transition that is a `p=` branch: it multiplies the branch's probability, and the branches with no modifier share the rest in proportion (with every branch modified, or more than 1 in all, they are scaled to sum to 1, with `W503`). A product's efficacy against the transition, and `where=` on it, do the same (people excluded take the other branches; they stayed in the state). Under every solver, stepped from Python in a compartmental run
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
- [x] Waning by a function of time in a compartmental run: holding the product has 10 levels of time since delivery, which everyone moves up every few steps (exact when the protection settles within 9 steps; otherwise each level is several steps, with a note, and `approx=False` makes it agents). ODE and agents agree within 1.5%
- [x] A test delay in a compartmental run: people wait for their result in levels of one step each (`pending_<delivery>`), as many as agents wait, and go to the follow-up when it comes; the same diagnoses as agents (25,051 and 24,773 with a follow-up for each result)
- [x] Time series for `coverage=` and `capacity=` (`ss.timeseries`)
- [x] `ss.Vx`, `ss.Dx`, `ss.Tx` as aliases of the new products: called with keyword arguments and without those of Starsim 3's classes (`df=`, `diseases=`), they make an `ss.Vaccine`, `ss.Test`, or `ss.Treatment`; Starsim 3's classes are kept, for their own arguments and as base classes
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
- [x] Flows in ratio results: a transition divided by something is its flow (`incidence = infection / S`), under every solver, by group, in a whole run, and in the model spec
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
- [x] Kernels for the `scipy` integrators: the right-hand side is the kernels' step, in a mode that returns the rates of change (`ss.kernels.derivs()`), with values that vary in time interpolated at the time asked for. The same results as in Python (to the last bit for constant rates), about twice as fast; for a small model, `solve_ivp`'s own overhead is most of what is left
- [x] Gillespie with many groups: where places do not affect each other within a step (people mix within their place, and no transition moves them between groups), each place has its own events over the step, and only its rates are worked out again after each one. The SimInf herds take 3.0 s, from 9.5 (SimInf: 0.08 s); a movement register of 466,692 events is read in 0.9 s, from 41
- [x] A whole run in the kernel with modifiers that depend only on time (no `where=`, `when=`, or function as the amount): the program is worked out for every step before the run. flepiMoP's scenarios with four modifiers no longer need the loop
- [x] A whole run with modifiers whose condition is only on states and stratifications (`where=hiv.acute`, as tbsim's drug resistance and its effects of HIV use), with transfers at a constant rate (e.g. starting ART), and with diseases tracked jointly: each takes each step in turn on one joint array, as they do step by step (the same to 1e-16, and in JAX to 1e-15). So TB with HIV, or with drug resistance, can be restarted, differentiated, and fitted by NUTS. For now the joint run is a loop in Python over the steps, with one kernel call a disease a step
- [x] A whole run with deliveries on a schedule (routine, or in rounds), to people chosen by states and stratifications only, with a coverage that is a number or `ss.by`, of a product that people hold or a transfer (`ss.Delivery.tabulated`): each is a shift of a share of each compartment and group to another group at the start of the steps it happens on (`ss.kernels.apply_shifts()`), in the kernels' `run()`, `run_many()` (with the results of the deliveries of each run), and `advance()` (the particle filter), and in JAX (as matrices; its results are worked out again from the compartments), so these sims can be restarted, differentiated, and fitted by NUTS. As an ODE, the results are those of the stepped run to 1e-15, and the gradient agrees with central differences to 3e-7; `run(n_runs=)` gives each run's results exactly. A step costs what it does without deliveries: tbsim with BCG takes 19 ms rather than 545 ms, and with TPT 13 ms rather than 360 ms. A product's effect is constant (`Effect.constant`), so the tables are no longer worked out step by step for it. Still stepped from Python: tests and their follow-ups, treatments that cure, capacity, `when=`, deliveries in a model of several diseases, modifiers with `when=` or a function as their amount, and transfers whose rate varies (see `deferred.md`)
- [x] Many runs with different parameter values: `ss.parallel(sim, pars=dict(beta=[...]))`. With `integrator='jax'`, an ODE takes them all in one call to JAX: 1,000 runs of a year of an SIR in 0.5 s, from 1.2 s one at a time (the JAX run itself is 0.05 s; the rest is setting up each run and working out its results). Otherwise each set is one call to the kernels for all its seeds
- [x] Building a sim with modifiers or values that vary in time no longer builds a second, probe sim to try them on groups (they always work on groups), and a modifier's search for time parameters no longer walks into the modules its condition names: flepiMoP's scenarios build in 26 ms, from 62
- [x] Faster building of a sim: an SIR builds in 6 ms (from 13), and the TB model (30 years of weekly steps) in 22 ms (from 106), against a run of 8 ms. What is left is mostly Sciris's `sc.search()`, which records every object it visits (see `sciris_issues.md`)
- [x] Agents: under homogeneous mixing with one strain, the probability of infection for everyone susceptible and exposed in one compiled pass (the rate is worked out once for each group, and the draw is the one made before, so the same agents are infected); each state counted by group in one pass; the level of each agent in a dimension reused until the property changes. The LASER example takes 3.5 ms a step, from 4.4 (8.5 at the start)
- [x] Agents: each agent's group is worked out again only for the agents whose level changed (births, moves, a new strain), which the code that changes it says (`sim.groups_changed(uids)`), and only dimensions binned from a property that changes on its own (age) are checked every step; and a state's members are read from its array rather than through the living agents. LASER's England and Wales model as agents runs 38% faster (100 days in 2.1 s, from 3.4), with the same results
- [x] Agents: every state counted by group (or by place) in one compiled pass; dead agents removed by compaction rather than `np.isin` (which sorted every UID each step); `true()`, `false()`, and `count()` read from the full array rather than gathering the active agents' values; the deaths of a step counted in one pass; and each agent's group updated in place rather than copied. LASER's England and Wales model (668,000 agents, 100 days) takes about 2.2 s, from 4.1 on the same (busy) machine, with the same results. What is left is spread out: the infection pass, exposure by place, and the members of each state, each 1 to 3 ms a step
- Not planned: a whole agent-based run in one kernel, as for compartments. A prototype of the best case (one compiled pass over 175,000 agents a step) took 1.1 ms a step on one thread, against 4.4 ms now, so 4 times at most; it reaches the 1 ns for each agent of LASER only on 20 threads, which Starsim usually spends on running sims side by side. And it would run only models with nothing in Python between steps, which are the ones the compartmental solvers already run
- [x] The chain-binomial solver with many stages: people move through as many stages as they pass in a step (a Poisson number, competing with the state's other exits; `ss.kernels.draw_chain()`), so it takes whole steps. A cohort through 49 stages matches the ODE to sampling error (a substep for each stage did not quite), and the LASER hub-and-spoke example takes 0.33 s, from 15 s. Its final size is 1.4% above the ODE's, from holding infection for a whole day, as daily chain-binomial models do
- [x] Time-varying rates in the stochastic solvers: Gillespie and tau-leaping hold values that vary smoothly in time for a quarter of a step at a time (at its middle, from the same start, middle, and end values the ODE interpolates), with events exact within each quarter, by the memorylessness of the waits; chain-binomial and SDE take them at the middle of each substep. With quarter-year steps, a seasonal SIS goes from 5–6% too many infections to within 1% of a daily ODE, with a note (a departure: not thinning)
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
- [x] The notes `W1203` and `W1204`: not needed. Tau leaps are chosen so that no compartment loses more than 3% of its people in one, so rates cannot change quickly within a leap, and `W102` says what `W1204` would
- [x] Interop with other formats: import first (`ss.load(source, format=...)`, `starsim/interop/`), then export
    - [x] The shared core: each importer reads its format into an `ss.interop.Source` (states, parameters, named expressions, and flows written as totals or per person), and `ss.interop.build()` classifies every rate by the rules of 16 (the source taken out of a total; another state over the total is exposure; without the total, mass action kept as counts; otherwise kept as written, with counts), prints how each was read, marks the guesses in `describe()`, and lists what was not translated as `W1601 ImportIncomplete` (also in `sim.imported`). The import parses the source's expressions with the model spec's parser, so it runs no code from the file
    - [x] `sim.to_code()` and `ss.to_code(spec)`: the sim as a Python file in the operator form, which builds the same spec again; an imported sim's file also says how each rate was read and lists what was not translated, as comments where a person or the import skill takes over
    - [x] EMULSION (`format='emulsion'`): each state machine a disease; rates, per-step probabilities, fixed stays (a duration with one way out at probability 1, or `amount-all-but: 0`), deaths by `autoremove`, initial numbers by prototype or compartment, and extra outputs that are expressions of counts as results; minimum stays, `cond`, `when`, conditional `escape`, actions, offspring, variables of individuals, and calendars are listed as not translated
    - [x] odin (`format='odin'`): `deriv` and `update` equations multiplied out and paired into flows, `rbinom`/`rpois` draws as rates under the binomial solver, `user()` defaults, replicate arrays as `n_runs`, arrays with flows from level to level as an age stratification, and `delay()` as a fixed dwell; its history, chained draws, `rmultinom`, mixing within a level, `interpolate`, and time are listed. The SEIR in age groups agrees within 6.3% in `I` with the delay's history added by hand, as the vignette does; the stochastic SIR arrays within 4.7 of 1,001
    - [x] SBML (`format='sbml'`): species, parameters, assignment rules, function definitions, and reactions with a net change of one down and one up, from MathML; events, rate and algebraic rules, other stoichiometries, and compartments of other sizes are listed. An SIR equals the hand-written model to 1e-6
    - [x] SimInf (`format='siminf'`, read from the R script without running R): `mparse()` and the predefined models (`SIR`, `SIS`, `SEIR`, `SISe`, `SISe3`), nodes as a `space` stratification, `ldata` by node, scheduled events as a movement register (with `E` and `N`), and the SISe environment as a state that is not people; data computed in R (`u0_SIR()`, `events_SIR()`) is listed and passed in as files. Over 96 runs of the herds SIR, every compartment's mean is within about 1.4 standard errors of SimInf's; over 24 runs of SISe3, within about 1
    - [x] flepiMoP (`format='flepimop'`): compartments as the product of their dimensions, places with commuting from the geodata and mobility files, initial conditions by place, transitions with `proportional_to`, the integration method, modifiers and scenarios as `ss.modify`, and outcomes from incidence or other outcomes; distributions (at their means), seeding, inference, timeseries parameters, and modifiers for some places only are listed. The two-province SEIR matches to 1e-12, and the vaccine scenarios, with the original's draws, to 4e-12
    - [x] summer2 (`format='summer2'`, from a JSON file written in summer2's environment by `starsim/interop/dump_summer2.py`): every flow kind but replacement births, stratifications with splits and adjustments, age, mixing matrices as `ss.contacts`, flows into partial strata, and derived outputs (`ss.interop.summer2.outputs()`); agrees within 0.0005 and 0.08 people of 1,000
    - [x] Atomica (`format='atomica'`, the framework and databook workbooks): compartments, characteristics, the transition matrix by format, parameter functions, values by year, initial numbers solved as Atomica does, populations, and its Euler step; programs, junctions, transfers, and interactions are listed. The SIR agrees to 1e-15 and the HIV baseline to 3e-14 in all 78 series
    - [x] camdl (`format='camdl'`, from camdlc's JSON form, which a `.camdl` file is compiled to, or from a pycamdl model, live or dumped by `starsim/interop/dump_pycamdl.py`): compartments, parameters with bounds and priors, `let` definitions, transitions as totals, observations as outcomes with likelihoods, and scenarios; a stratified model comes in as one state per combination of levels (exact, with the dimensions noted). Forcing functions, events, ODE compartments, other reactions, and some likelihoods are listed. The SIR agrees with camdl's ODE within its rounding (0.5 people; weekly cases within 4e-8), and with 200 chain-binomial runs within 0.9%; pycamdl's SEIR by age and patch within 3e-16; 84 of camdl's 85 test models import and run, 54 with nothing listed
    - [x] EpiModel (`format='epimodel'`, from a JSON file written in EpiModel's environment by `starsim/interop/dump_epimodel.R`, which runs the script with `netest`, `netsim`, `dcm`, and `icm` only recording their arguments, so in about 1 s): the built-in SI, SIS, and SIR models as `netsim`, `dcm`, or `icm`, with per-step probabilities, arrivals and departures, each network as an `ss.PartnerNet` from `edges`, `concurrent`, `degrange`, and `nodematch` (marked as not an ERGM fit), and each other run as a scenario; custom modules are listed, with their R source kept in `sim.imported.modules`. The introduction's SIS network agrees with 100 EpiModel runs in new infections (prevalence is 2.9% higher, as EpiModel lets people recover in the step they are infected), and a `dcm` SIR within 3e-7; from the HIV model, the two networks, the parameters, and the four scenarios import, and its seven custom modules are listed
    - [x] epydemix (`format='epydemix'`, from a JSON file written in epydemix's environment by `starsim/interop/dump_epydemix.py`, or a live model): mediated and spontaneous transitions, two exits at `r*X` and `r*(1-X)` as a branch, parameters over time and by age, the population and contacts by setting, the initial numbers the run was given, with `substeps=1` and no aging, as epydemix does. Against 200 runs of each original, every series passes the statistical comparison: the Kenya SIR's 30 (means within 1.05% of the peak) and COVID-19 in Massachusetts' 40 (within 1.7% for totals)
    - [x] MEmilio (`format='memilio'`, from a JSON file written in MEmilio's environment by `starsim/interop/dump_memilio.py`, which runs the script as it is and records the first model it simulates): the ODE models `osir`, `oseir`, `oseirdb`, and `osecir`, with their equations as MEmilio's C++ has them; parameters by age group, age groups with the contact matrix, dampings as a factor on the force of infection (smoothed over the day before, as MEmilio does), seasonality, test-and-trace and ICU capacity as functions of the counts, and deaths as outflows (MEmilio leaves the dead out of the total). A parameter that differs by age where it belongs to the infectious group or is summed over groups, dampings that differ between groups, dynamic NPIs, and school holidays are listed; other models are by hand. Against MEmilio integrated tightly, the SECIR examples and a SECIR with parameters by age, ICU capacity, and two dampings agree within 1.6e-7 of the population with `dt=0.1`, and within 2.4e-5 to 2.4e-3 with steps of a day (a damping's one-day smoothing falls within a step)
    - [x] `ss.diff(a, b)`: two models compared in their own terms, in the four lines of `sim.hash()`; transitions are matched by name and then by their endpoints, and compared with their parameters written in as values, so an import and a hand translation whose names differ are reported as the same where they are, and a parameter is compared by name only where the same transitions use it
    - [x] Skills for what does not map one to one (`import-model`, with a guide for each framework, including those with no mechanical importer: LASER and Starsim v3), and `validate-import`, which runs the original and the port and compares them, exactly or statistically (`skills/validate-import/scripts/compare.py`: exact for one run each, z scores of the means and Kolmogorov-Smirnov tests of the peak, its time, and the final value for runs on both sides, and the same quantiles against a summary); in `skills/` in this repository. The EMULSION SIR's import agrees with 200 runs of EMULSION in every test
    - [x] Export (`ss.save(filename, sim, format=...)`, or `ss.interop.export()` for the text), with `W1401 LossyExport` listing what each format cannot carry: Mermaid (each disease's states and transitions, labelled with their rates, a subgraph for each disease, and its stratifications in the title), SBML (Level 3 Version 1), and odin (`deriv()` equations, each flow a named variable), the last two from the compartmental form (as `equations()` prints it), with the numbers in the sim's unit (SBML's `timeUnits`, which the SBML importer now reads): a dwell time that is not exponential as its Erlang stages, and a stratification summed into one group (listed). An SEIRD with births, deaths, a branch, and a lognormal dwell (11 compartments, 21 flows) goes out and back in through SBML and odin within 7e-12, and runs in odin 1.2.7 within 4e-6 of Starsim; libsbml finds no errors. Petri nets: won't do (Cliff, 2026-10-16); stratified groups as separate species: not yet (see `deferred.md`)

## Departures from the spec

- Sims with no declared states keep Starsim 3's time defaults.
- The unit of a model's bare numbers comes from the model (`unit=`, else its typed values, else dates, else a typed `dt` or `stop`), as the review's Decision lang 1 recommends, with two departures from that recommendation: a year-like number for `start` or `stop` is still a calendar year (`stop=2020`), since Starsim 3 sims are written that way; and the typed defaults of modules written as code do not set the unit, since they convert themselves and would otherwise disagree with most declared models.
- Rates are written as strings in the model spec, and the spec is `sim.to_spec()`; `sim.to_json()` is unchanged from Starsim 3. The spec is for reading by people and other tools; it is not loaded back (`ss.load()` of a spec and `ss.from_spec()` were removed).
- The default ODE integrator is fixed-step RK4 with automatic substeps, in the Numba kernels, with or without gradients being asked for; JAX is used only for `integrator='jax'`, `sim.gradient()`, and gradient-based fitting. `integrator='jax'` takes the same RK4 steps (not an adaptive JAX integrator), so its results are the kernels'.
- What JAX differentiates is the run, from the numbers in the program and the compartments at the start to the compartments and flows at every step. The steps either side are Python (the parameters to those numbers, and the run to the results, outcomes, and goodness of fit), and are differentiated by central differences along the direction of each parameter. So a gradient is exact to about 1e-10, not to the last bit, and costs two restarts and two runs' worth of results for each parameter.
- A stratified population has ages from 0 to 100.
- Every agent's dwell time is sampled, including exponential ones.
- The property of people that says they hold a product is named `has_<product>` (e.g. `has_bcg`), and the record that a routine delivery has been offered is `offered_<delivery>`. The product's own name is taken by the product module.
- A product whose `waning=` is a duration is lost at an exponentially distributed time, under every solver, after which the person can receive it again. Only a function of time since delivery fades the efficacy while the person still holds the product.
- A delivery never gives a product to someone who already holds it.
- A bare number for a time (`ss.change(at=...)`, a delivery's `start=`) is a time in the sim's unit, counted from the start, even in a sim with dates; it is a calendar year only in a sim whose own `start`/`stop` are year-like numbers. A time outside the sim raises `E1101`.
- A modifier acts where the engine uses the quantity; reading `sir.rel_sus` from custom code gives the base value, and `sir.rel_sus.effective` the value with the modifiers applied (agents only). So `.base` is not needed.
- In a compartmental run, a product whose protection fades by a function of time is held at levels of time since delivery (`stage 1`, ...), so `has_<product>` has more than two levels.
- A network's parameter is the target of a modifier as a path (`'school.pars.n_contacts'`), since `school.pars.n_contacts` is a plain value.
- `W1203` and `W1204` are not used: tau leaps adapt to the rates, and `W102` covers large steps.
- Under `gillespie` (and `tau`), a value that varies smoothly in time is held for a quarter of a step at a time, at its middle, with a note, rather than by thinning, which needs a bound on every rate over the step; a rate can be any expression of the parameters.
- A module written as code that works on groups as well as on agents says so with the class attribute `compartmental = True`; otherwise it makes the model agent-based.
- The peak of `ss.seasonal` given as a month is the middle of that month. In a sim with no dates, time starts at the beginning of a year.
- `ss.gravity` gives `k × pop[i]^(a−1) × pop[j]^b / distance^c`: for commuting, the share of time a person from `i` spends in `j` (capped so that it sums to at most 1, or rescaled to `away=`), and for migration the rate of moving.
- An inflow whose rate uses a count (`xi * I.n`, or `ss.birth`) arrives in each place according to the count there; an inflow at a fixed rate is shared between places by their split.
- A new infection is recorded under the strain acquired, and a state that a dimension does not apply to is counted at that dimension's first level.
- As agents, the number who start in each state is exact (rounded to whole agents), not a chance for each agent.
- A level of a stratification named like an attribute of a result (`low`, `high`) is a column called `<dimension>_<level>` in `result.by()`.

- An imported rate with another state and no division by the total (`beta * S * I`, rule 3 of 16) is kept as counts (`beta * I.n`), which is exact, rather than given `dependence='density'` and an `n_ref`: there is no density-dependent mixing.
- Results follow the convention of other frameworks under every solver: `n_<state>[ti]` is the number at time point `ti` (after `ti` steps, so `n_<state>[0]` is the number at the start), and a flow at `ti` (`recovery[ti]`) is the number during the step that ends there (0 at the start), so `n_R[ti] - n_R[ti-1] == recovery[ti]`; a sim with `stop=T` ends at `T`.

## Known issues

- The kernels are compiled the first time they are used after Starsim is installed or changed, which takes one or two minutes (a line says so); after that they are loaded from a cache.
- With `sim.run(n_runs=)` in one call to the kernel, only the states and transitions are kept by group (not outcomes or results given as expressions), and reported outcomes are drawn for all the runs at once, so differ (in the draw, not the distribution) from those of a single run with the same seed.
- A typed rate cannot be used as a plain number in an expression (`1 - ss.peryear(0.008)`); the error says to convert it first (`rate.to_prob(dt)`).
- A bare number given for a parameter of a module written as code (`ss.MFNet(duration=5)`, `ss.SIS(dur_inf=10)`) keeps the unit of its default, as in Starsim 3, while one in a declared model is in the sim's unit. It warns when the two units differ (`duration` read in years in a sim in days).
- As agents, a dwell time is sampled when someone enters a state, so an `adjust` on a duration does not follow a person who later moves to another level (through an `ss.transfer`, or by aging). A rate does.
- A strain's initial share is the split of the dimension (equal by default), which applies to everyone who starts in an infected state.
- As agents, someone tested on the step they were infected may already have recovered (about `dt/2` of a dwell time), so event-triggered tests find slightly fewer cases than in a compartmental run.
- An ODE and an agent-based run of the same model differ by an amount of order `dt` (about 6% in the prevalence of an SIS model with a daily step).
- As agents, people added by a movement register in a step are made at its end, so the other events of the same step cannot choose them; and people who moved into a place in a step are chosen after those who were already there.
- A movement register cannot add people to a state that splits on entry into branches.
- Arithmetic on stratified results keeps the values by group only for sums, differences, and scaling by a number; for anything else (a ratio), `select()` says the result is not stratified. (A ratio declared on the disease, `prev=I/N`, is stratified: it keeps its numerator and denominator.)
- With `immunity=` between strains, a compartmental run keeps only each person's last strain (noted), so with partial protection people can go from A to B and back to A; as agents, every strain is kept, but not yet over networks.
- Code that changes a property of people that a dimension is made from (`people.patch`) has to call `sim.groups_changed(uids)`, or the agents' groups are not worked out again (dimensions binned from a property that changes on its own, such as age, are checked every step).
- A state that is not people is deterministic under the stochastic solvers, so a vector population whose own randomness matters cannot be drawn.
- Under the exact solver, places are run one at a time between steps only when people mix within their place and nothing moves them between groups within a step; otherwise every rate is worked out again after each event.
- A plain `duration=` on an outcome is an exponential stay, while `delay=` is an exact shift.
- As agents, TB (LSHTM) comes out 5–10% higher than the ODE with weekly steps, and 20–30% higher with monthly ones; this is the difference of order `dt` above, made larger by an epidemic that is near R = 1.
- `rel_sus` and `rel_trans` on a state are numbers, not `ss.par`s, so they cannot be fitted; the TB models write them into the rates instead.
- Once JAX has run in a process (`sim.gradient()`, a gradient fit, or `integrator='jax'`), running sims in parallel by forking processes (`ss.parallel()` of agent-based sims, Optuna with several workers) can deadlock, as JAX warns. Spawning instead would be safe but takes about 1.5 s longer to start.
- Gradients need a sim that the kernels run from start to finish (no interventions other than deliveries the kernels take, no modifiers with `when=`, transfers whose rate varies, or movement register), and a rate that is a Python function of the sim has none (one written with NumPy functions of counts does).
- The JAX run is compiled the first time it is used for a model (about 1 s), and again if a step needs more substeps than it has allowed for.

## Review of 2026-10-17

The work from [the review's plan](../review_oct17/implementation.md), by phase. Finding numbers are those of the review.

### Phase 1: unblock

- [x] R can build a model (N17): every frame lookup goes through `ss.utils.getframe()`, which gives None when there is no frame. Tested with no Python frame (`tests/test_transitions.py`) and from R (`tests/test_r.py`, which runs `tests/r/sir.R` through reticulate; a CI workflow, `test_r.yml`, installs R and reticulate to run it).
- [x] A line says when the kernels are being built (X8).
- [x] Distinct runs after `init()` (N5): `ss.parallel()` builds each run again with `sim.new(rand_seed=)` (or restarts it), and builds again a sim that has run, so `ss.parallel(baseline=fit.sim)` works (P1b); an initialized sim with no inputs raises.
- [x] `sim.stratify()` survives `sim.new()` (N18).
- [x] `sim.save()` after a run (R15): `Loop.shrink()` no longer makes the plan, the engines have a `shrink()`, and copying a sim no longer leaves its parameters and states referring to the original's modules (the custom `__deepcopy__` of `ss.par` and `ss.State` kept the old module when the parameter was reached before its module, through `sim.pars.<name>`).
- [x] The headline examples run (P1a, P1b): the two-line SIR is `beta=0.3` with `init_prev=0.01` throughout the spec, best practices, and docstrings; `AlreadyRunError` names `sim.new()`; `msim.plot()` of declared runs draws the median and a 95% interval of each scenario (`MultiResults.plot()`).
- [x] Portable test paths (P12): `tests/paths.py` finds the design repo from `STARSIM_LFD` (and camdl from `STARSIM_CAMDL`), next to this repo by default, and the tests that need it skip with a message naming the variable; `test_samples` uses `tmp_path`.

### Phase 2: contract tests

- [x] The model corpus, `tests/corpus.py`: the 18 conformance models and five of the users' patterns, each a function that returns a sim. The older test files keep their own models.
- [x] The spec's code runs (`tests/test_spec.py`): every block in `design/00` to `09`, file by file, with fragments marked `<!-- fragment -->` and the known library failures listed in the test, which fails when one is fixed.
- [x] The likelihoods match the run, and simulation-based calibration for each method (`tests/test_contracts_inference.py`; longer runs with `STARSIM_SLOW=1`).
- [x] Streams, agreement between solvers (the simple models by default, the whole corpus with `STARSIM_SLOW=1`), the order of competing exits, conservation, exact identities, and the guesses `describe()` marks (`tests/test_contracts.py`).
- [x] The thirteen switching demos (`tests/switching/`, run with `pytest switching`): 19 tests pass and 31 are expected failures, each naming what blocks it.

### Phase 3: the silent wrong answers

- [x] All the items, by three agents in worktrees and in the main checkout, apart from: `sde` (X5), which goes with the solver menu in phase 5; X7 (agents' `dur=` against `rate=`, and seeded people leaving a step late), which is for Cliff; and W11, deferred. Each agent's notes, with the decision options taken (all the recommended ones), are in the session's scratchpad (`wt/WORKLOG_*.md`).
- X3: each engine draws from its own `np.random.Generator` (stream 0 for Python, 1 for the kernels, 2 for a register's events at the start), seeded from `[rand_seed, crc32(name), stream]` and passed into every kernel that draws (a typed list for `run_many()`), so a stepped run gives the same draws as a whole one. Numba's generators have no hypergeometric draw, so the kernels have their own (one draw per person moved, which is cheap for a register's events). Reseeding Numba's global generator each step was measured at 4.8 µs a call, too slow.
- X19: a distribution's seed is a hash of its name and the sim's seed together, which moved every agent-based baseline once (`tests/baseline.yaml` regenerated).
- N1: the particle filter took the rows of the tables as one per step when a smooth value makes three. N2: the JAX density wrote one value into every row of the table for a flow whose rate is a fitted value times a time series.
- Fixed along the way: copying a sim left its parameters and states referring to the original's modules; `next_generation()` left the compartments at the disease-free state.

### Phase 4: the guessing rules

- [x] The principle restated (a guess fills in what is missing and never reinterprets what was given) in `best-practices/principles.md`, `design/README.md`, and the guess table in `03`.
- [x] The unit rule (L1), keyword numbers (L3, `E602`), the people anchor (L4), the default seed and the R0 line at `init()` (L5), `source=` for SEIR-shaped strings (L8, `E212`), a dwell shorter than the step (L9), initial values by size and one hash for `init`/`init_prev` (L17), bare times in the sim's unit and outside the span (D2, `E1101`), a dimension's first level (D3), a dimension's kind from its value (D4), the world's ages by default (D11), aging only with births or deaths (D12a), the printed step unit (L18, partly), the typed-arithmetic table and `E102` (L19), and a set of states reused for several diseases (L20, partly). L23 is deferred; the departures and the parts for Cliff are in "Departures from the spec" and `deferred.md`.

### Phase 7, started early: inference

- [x] ABC-SMC (`method='abc-smc'`, `ss.calibration.abc_smc()`), staged fitting (`method=` a list, `W1303`), `sim.sensitivity()` (Latin hypercube and partial rank correlations), `ss.binom()` and `ss.bernoulli()` as likelihoods, `ss.parallel(sim, priors=True)` (`ss.calibration.prior_draws()`), `fit.project()`, `msim.to_df(quantiles=)`, `msim.to_hub()`, and the spread of Optuna's best trials (Cliff's note in `13`). The flepiMoP importer gives a drawn parameter its distribution as its prior. IF2 is deferred until N27.

### Phase 5: one system (first round)

- [x] Errors and notes as classes (P6, P5, Decision 13): `starsim/errors.py`, 37 errors, 6 warnings, and 10 notes, each `ss.<Name>`; the catalog in `design/15-errors.md` is generated from their docstrings and checked by `test_spec.py::test_error_catalog`; `verbose` is one integer.
- [x] One language (X11, L31, P3, D23, N19, R13; Decision 2): declared `ss.SIR`, `ss.SIS`, and `ss.SEIR`; `ss.Disease` declared only, `ss.Infection` for code; the duplicated Starsim 3 vocabulary removed, mapped in `skills/migrate-starsim3`; v4 verbs (`ss.modify`, conditions, `calibrate()`, `sim.new()`) on modules written as code. Kept for Cliff: the 11 intervention classes and `ss.Calibration` that downstream packages subclass.
- [x] One `ss.Result` (a MultiSim is its results; `ss.parallel` the one way, with `sim.run(n_runs=)` its shortcut; one name per flow, `.cum`), the Python solvers deleted and the solver menu (`auto`, `ode`, `binomial`, `gillespie`, `abm`), `auto`'s reasons in one record, one time index during a step and one delivery slot, and one compartmental program per crossing (X25), by agents R, S, T, and X25. For Cliff: `sim.people = None` in a compartmental run, the dating of infections, S's thresholds.

### Phase 6: docs

- [x] "Starsim 4 in ten minutes" (`docs/ten_minutes.qmd`), the Starsim 3 to 4 migration guide (`docs/migration/v3_v4.md`), the R path on the same page, and `tests/north_star.py` (lines, first run, and best of five against the originals).
- [x] The tutorials and user guide ported (35 pages, all run), and a "Coming from" page for each of 13 frameworks; the bugs they found fixed or recorded (W15).

### Phase 7: features for the seats

- [x] Inference: see above; also vector parameters (`ss.par(ss.by(patch=...), bounds=)`, fitted level by level, `pars={'beta[a]': ...}`), a modifier's amount as an `ss.par`, typed bounds, `calibrate(gof=fn)` without data, and `sim.log_density()` (reverse-mode gradients).
- [x] `ss.optimize()` over spending, and the Atomica importer's program books (Atomica's baseline exactly, and its optimum within 0.014%); `.total` counts over every place.
- [x] `ss.modify(hold=)` and `tr.incidence(over=, by=)` (MEmilio's dynamic NPIs, matched exactly); `PartnerNet(n_contacts=ss.by(...), acts=)` and the duration in `net.check()`; `ss.Mobility(scale='mean', cap=)`; `ss.death(lifetable=)`; seeding as a delivery with `move=` and `capacity=`; `ss.Vaccine(leaky=False)`; `at_age=`; `after=`, `before=`, and `offspring=` on transitions (agents); a thin R layer in rstarsim (`R/declare.R`).
- [ ] Not done: the importers as a companion package (for Cliff), odin's fixed delay, the data for US states and German districts (written up in `starsim_data_issues.md`), offspring by prototype, and a declared `ss.program()`. (IF2 and a fittable `away=` were done later, by N27 and PD2.)

### Phase 8: performance

- [x] N23 (factored mixing within places with commuting; sparse movers), N24 (results by group at a reporting interval, `ss.Sim(results=[...])`), N28 (`ss.contacts()` as numbers), by agent P.
- [x] N25 and N26 prototyped: both targets reachable. [x] N25, N26, N27 (and IF2) built in (memory at 48 bytes per agent, per-event Gillespie by place, particles in the kernels, IF2); agent speed at LASER's scale is still open (`v4_design/oct19/implementation.md`, Stage 5).

### Phase 9: surface and structure

- [x] The time vocabulary (L28): `ss.prob(0.1, 'day')`, `ss.time.freq`, units required.
- [~] The saved form (R21): the spec is not read back (Decision 10); `ss.save('model.py', sim)`. R29's cleaning of the spec waits.
- [~] Dependencies (P14): `seaborn` dropped, `[data]` and `[interop]` extras; `starsim_data` as a dependency needs it on PyPI.
- [ ] The rest after phase 5.
