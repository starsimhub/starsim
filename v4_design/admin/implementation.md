# Implementation plan

Status of the Starsim v4 implementation of the design spec. The code is in `/home/cliffk/idm/starsim` on the `v4` branch; section numbers refer to the files in `design/`. Checked items run and have a test in `tests/test_transitions.py` or `tests/test_deliveries.py`.

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

- [ ] Wave 1: one simple example from each framework (MEmilio, EpiModel, epydemix, summer2, SimInf, odin, EMULSION, flepiMoP, Atomica, LASER, camdl)
- [ ] Wave 2: the other eight conformance corpus models (4, 7, 9, 12, 13, 16, 17, 18)
- [ ] Wave 3: one complex example from each framework, Ghana measles, and TB

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
- [ ] The unit notes `W102` and friends

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
- [ ] `ss.timeseries`, `ss.seasonal`
- [ ] `requires=` and `sim.requires()`

### 05 Dimensions

- [x] `stratify()` on a disease and on the sim, `split=`, `adjust=`
- [x] `ss.transfer` as a rate, `ss.by`
- [x] Aging between age groups
- [ ] Strain, space, and mobility dimensions

### 06 Routes

- [x] Homogeneous mixing, `ss.MixingPool`, `ss.contacts` matrices
- [x] Networks, as agents and (approximated, with a note) as compartments
- [ ] Multi-disease coupling in a compartmental run

### 07 Effects

- [x] `ss.modify` on a parameter, with `multiply=`, `add=`, `set=`, `start=`, `stop=`, `when=`
- [x] `where=` on `ss.modify`
- [x] Targets other than parameters: `rel_sus`, `rel_trans`, a transition
- [x] Time-varying amounts (a function of the sim, `ss.by`)
- [x] Lazy conditions (`ppl.age > 65`, `sir.I & ~sir.vaccinated`), on agents and on groups
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
- [ ] Time series for `coverage=` and `capacity=` (`ss.timeseries`)
- [ ] `ss.Vx`, `ss.Dx`, `ss.Tx` as aliases of the new products
- [ ] Contact tracing and partner notification

### 09 Results

- [x] A count per state and a flow per transition, with the timing convention
- [x] Result expressions (sums and ratios of states)
- [x] `ss.outcome`, with delays, durations, and reporting periods
- [x] `res.select()` and `res.by()` on stratified results
- [ ] Flows in ratio results
- [ ] Extinction reporting

### 10 to 12 Execution, random numbers, and solvers

- [x] `ode` (RK4, or any `scipy` integrator), `binomial`, `tau`, `gillespie`, `sde`, `abm`
- [x] `auto` and `stochastic` solver choice, with the reason printed
- [x] Approximations noted, and refused with `approx=False`
- [ ] Gillespie thinning for time-varying rates
- [ ] `sim.check_dt()`
- [ ] JAX and gradients

### 13 Inference

- [x] `ss.data`, `sim.calibrate()`, `ss.Fit`
- [ ] Data matched by name to summed results
- [ ] Likelihoods, `ss.compare_models`
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

## Known issues

- `result.by()` fails for a stratification with a level called `low` or `high`, since those are attributes of a result.
- A function of the sim that compares ages with a number that is not a bin edge runs in a compartmental model without complaint (each group has the age of its midpoint); the lazy form (`ppl.age > 70`) is checked.
- As agents, someone tested on the step they were infected may already have recovered (about `dt/2` of a dwell time), so event-triggered tests find slightly fewer cases than in a compartmental run.
- An ODE and an agent-based run of the same model differ by an amount of order `dt` (about 6% in the prevalence of an SIS model with a daily step).
