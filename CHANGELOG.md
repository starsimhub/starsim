# What's new

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output are flagged with the term "Regression". Changes that may require update to downstream code are flagged with the term "Migration".

## Version 4.0.0 (in development)

This version adds a way to write a model as declared states and transitions, which can then be run as an ODE, as a stochastic compartmental model, or as agents. Models written as code (`ss.SIR`, custom `step()` methods, etc.) still run, but their results are one step later than in Starsim 3 (see *Regression* below), so they agree with Starsim 3 statistically, not number for number. The migration guide (`docs/migration/v3_v4.md`) maps Starsim 3 code to Starsim 4.

### New features
- `ss.CodedDisease` is Starsim 3's `ss.Disease`: a disease written as code with no states of its own; `ss.Infection` subclasses it.
- `python -m starsim.migrate <path>` rewrites Starsim 3 code for Starsim 4 in place (renames, the network's name with `ss.PrenatalNet`, a fractional `verbose`, `annualize()`, the coded bases, and the time index in `step()`, `update_results()`, and analyzers), and lists what needs a person; the migrate-starsim3 skill starts from it.
- Removed methods of Starsim 3 (`Result.annualize()`, `MultiSim.reduce()`, `mean()`, `median()`, `reset()`) raise an error that says what to write instead, rather than a bare `AttributeError`.
- *Delays*: a fixed delay (`ss.constant` as a dwell time) is exact in compartmental runs, as a conveyor of whole steps rather than 50 Erlang stages: odin's 14-day SEIR agrees with odin within 0.25% at a step of a day (1.6% with 200 stages before), in milliseconds rather than seconds. `after=` and `before=` (minimum stays) run as compartments by the same conveyor, and the chain binomial agrees with agents step for step. `ss.update(count, value)` sets a count each step to a value worked out from the model at the start of the step, as odin's `update()` of a variable that is not a flow (a lag, a running maximum), in the kernels, in JAX, and as agents.
- *Inference*: `method='if2'` gives the maximum-likelihood values of a stochastic compartmental sim by iterated filtering (Ionides et al. 2015, as pomp's `mif2()` and camdl): each particle of the filter carries its own parameter values, perturbed by a shrinking random walk; four chains from different starts, with `fit.converged` when they end at the same log-likelihood. On camdl's SIR it gives camdl's estimates in about the same time. Parameters of how the data are observed, and of the starting values, are not fitted by IF2 yet (the error points to `method='pmcmc'`).
- `ss.People.age < 15` (and `.female`, `.alive`) is a condition written from the class, with no people made.
- *Docs*: "Starsim 4 in ten minutes" (`docs/ten_minutes.qmd`), the whole workflow on one page, with using Starsim from R; a Starsim 3 to 4 migration guide (`docs/migration/v3_v4.md`); the tutorials and user guide ported to Starsim 4; and a "Coming from" page for each of 13 frameworks (`docs/coming_from/`), with its failing rows.
- *Programs and budgets*: the Atomica importer reads a program book (`progbook=`, `programs_start=`): each program's spending is a parameter, its coverage is the people its spending reaches over everyone it targets, and the parameters it affects move towards its outcomes with coverage. `ss.optimize(sim, maximize=..., at=..., start=...)` reallocates spending between programs to maximize (or minimize) a result, keeping the total; on Atomica's HIV demo it finds Atomica's optimum in about as long. `(S + I).total` is the number in some states in the whole population (every place, where `.n` is the same place).
- *Inference*: `method='abc-smc'` gives a posterior without a likelihood; `method=` a list runs staged fitting, each stage starting from the last; `sim.sensitivity()` gives partial rank correlations over samples within the bounds; `ss.binom()` and `ss.bernoulli()` are likelihoods; `ss.parallel(sim, priors=True)` draws the parameters from their priors; `fit.project(scenarios)` runs scenarios with the posterior's draws; `msim.to_df(quantiles=)` and `msim.to_hub()` write runs as quantiles and as a forecast hub submission; Optuna fits give the spread of their best trials as an indication of uncertainty; `sim.log_density()` gives the log-density in JAX as a function, with its gradient by reverse mode. A parameter with a value per level (`ss.par(ss.by(patch=[...]), bounds=)`) is fitted level by level, with the same bounds and prior (`fit.pars.beta` is an `ss.by`; one level can be set by name, `pars={'beta[a]': 0.3}`); so are a modifier's amount given as an `ss.par` (which can then be set by name), and typed values with typed bounds (`ss.par(ss.days(8), bounds=[ss.days(3), ss.weeks(4)])`).
- **States and transitions**: `ss.Disease` can now be built from `ss.State` and `ss.Transition` objects, written as arrow strings (`ss.transmission('S -> I', beta=0.05)`, `ss.progression('I -> R', dur=10)`), operators (`S >> I @ beta * I`), or explicitly. Also `ss.states()`, `ss.null` (for inflows and outflows), `ss.birth()`, `ss.death()`, `ss.immigration()` (a number arriving per unit time, at ages like the population's) and `ss.emigration()` (as `ss.death()`, but agents are recorded as emigrants), and `ss.par()` (a named parameter with bounds). States that no transition connects to the first state (e.g. a water reservoir) are counts, not people, and are real numbers under every solver. A rate can use NumPy functions of counts (`np.clip(0.5 * I.frac, 0, 0.02)`) and comparisons (`Q.n < 20`); a set of states is their sum (`Ih / (Ia | Is | Ih)`); `where=` on a transition limits it to some people.
- **Solvers**: `ss.Sim(solver=...)` runs these models with `'ode'`, `'gillespie'`, `'tau'`, `'binomial'`, `'sde'`, or `'abm'`; the default, `'auto'`, uses an ODE if every part of the model has a compartmental form, and agents otherwise. Dwell times that are not exponential are run as Erlang stages in a compartmental run, with a note. `ss.erlang()` and `ss.truncnorm()` are new distributions. Values can vary in time (`ss.timeseries()`, `ss.seasonal()`); the ODE takes them at each stage of a step. The first run after installing says that the solver kernels are being built (one or two minutes, once).
- **Sim**: the first argument of `ss.Sim` can be the model (a module or a list of modules). If nothing in a declared model has a unit, time is counted in timesteps; `unit=` sets the unit of bare numbers (if the typed values have different units, the smallest is used, with a note, `ss.UnitInferred`, naming the bare numbers). New methods: `sim.describe()`, `sim.solvers()`, `sim.hash()`, `sim.to_spec()` (read back with `ss.load()`, for declared diseases and their stratifications), `sim.new()`, `sim.R0()`, and (for diseases) `describe()`, `diagram()`, `equations()`, `R0()`, and `transitions`.
- **Stratification**: `sir.stratify(age=[0, 15, 65], adjust=...)`, `sim.stratify()`, and `ss.transfer()` split a model by a property of people; `ss.contacts()` makes mixing pools from a contact matrix; `result.select()` and `result.by()` read stratified results; `ss.by()` gives a value per level, for one or several dimensions (as a product, a table, or an array with named axes, any of whose values can be a time series, with arithmetic), and `split=ss.by(...)` splits the population over several dimensions at once (or, for a dimension of some states only, gives the share entering each level for each level of another, e.g. severity by age); `adjust=` can also scale `rel_trans` and `rel_sus` by level; `immunity=` on a strain dimension gives the protection of each strain against the others; `where=` stratifies some groups only (`sir.stratify(age=[15, 25, 35, 50], where=dict(sex='female'))`), and people outside them have no level of it, so selecting a level leaves them out. People age between age groups.
- **Strains carried together**: a strain dimension can have levels for people who carry several strains at once (`strain=['A', 'B', 'A+B']`). Someone who carries several passes on each in proportion to its infectiousness, and someone infected with a strain they do not carry carries it as well, e.g. through a transmission that leaves people in their state (`superinfection = L >> L @ beta*I`). A transition can also change a person's level with `move=` (`ss.progression('L -> I', rate=..., move=dict(strain='A+B -> A'))`).
- **Places**: a stratification by place (`sir.stratify(patch=ss.geo(...))`) mixes people within each place, with commuting, visiting, or migration between places from `ss.gravity()` or a matrix (`ss.Mobility()`), or with the force of infection itself moving between places, as in LASER (`kind='force'`). A contact matrix by age acts within each place; as agents, so do a contact matrix, a mixing pool of everyone, and a random network, as in a compartmental run (people contact others where they live, or where they commute to). `ss.Mobility(data=...)` is a register of people moved between places (or between states, with `to_states`), added, and removed on given dates, as in SimInf's scheduled events. `ss.Mobility(scale='mean', cap=)` scales movement so that places' share away is `away=` on average and at most `cap=`, as LASER scales its gravity coupling.
- **Partnerships**: `ss.PartnerNet()` is a network of partnerships that form and dissolve, described by targets for the number of partners people have at a time (`n_contacts`, `concurrency`, `max_contacts`), the share of partnerships between people at the same level of a property (`match=dict(risk=0.9)`), and how long partnerships last, as in EpiModel (allowing for partnerships ended by deaths); `net.check()` compares the network it makes with the targets. `ss.PartnerNet(n_contacts=ss.by(risk=[...]))` gives each level its own number of partners, as EpiModel's nodefactor, and `net.check()` also reports how long partnerships last; with `acts=`, beta is per act, as EpiModel's act rate. `ss.MFNet` takes the same targets for the number of partners (`n_contacts`, `concurrency`, `max_contacts`), between men and women; without them it is unchanged. Networks can be given by name (`networks=dict(main=..., casual=...)`).
- **Requirements**: a module can list the data it needs (`requires = [ss.Requires('contact_matrix', shape='age x age', source='Prem et al. 2021')]`), and `sim.requires()` shows them for the whole sim (a declared model's are its parameters), and which are missing, without initializing the sim.
- **Results and fitting**: a transition divided by something in a result is its flow (`incidence = infection / S`), by group too. `ss.outcome()` declares reported quantities (with a weight, delay, duration, and reporting period), which are stratified like the flows they come from; `ss.data()` matches data to results by name; `sim.calibrate()` fits every parameter that has bounds to every result that has data, wherever it is used (in a rate, an initial count, or a seasonal value), returning an `ss.Fit`. A likelihood on an outcome (`likelihood=ss.nbinom(n=k)`) or given to `sim.calibrate(likelihood=...)` makes the fit maximize the likelihood, fitting the likelihood's own parameters too; `ss.compare_models()` compares models by how well they predict held-out data. A stochastic compartmental sim (`solver='sde'`, `'binomial'`, `'tau'`, or `'gillespie'`) whose data series all have a likelihood is fitted by particle MCMC (`method='pmcmc'`): a random-walk Metropolis sampler on the likelihood as estimated by a particle filter (`ss.ParticleFilter`), which samples the exact posterior; the particles are stepped together by the kernels, each chain chooses how many it needs in its warm-up, and the chains run in parallel. An agent-based sim is fitted by Optuna, as before (a best fit, not a posterior).
- **Modifiers and conditions**: `ss.modify()` changes a named quantity (a parameter, a transition's rate, `sir.rel_sus`, or `sir.rel_trans`) while a condition holds, for everyone or for the people selected by `where=`; `ss.change()` changes one at given times; `sim.what_modifies()` lists what acts on a quantity. `hold=` keeps a modifier on for someone for a while once its condition holds for them, and `tr.incidence(over=, by=)` is a condition on how many went through a transition over a window, as a share of the people at the same level of a stratification, so that each place can apply a rule on its own (MEmilio's dynamic NPIs). A modifier that depends only on time, and on who it applies to only through states and stratifications (`where=hiv.acute`), is worked out before the run, as is a transfer at a constant rate, so a compartmental run stays in one call to the kernels. Conditions are written against the people and modules themselves (`(ppl.age > 65) & sir.S`) and are evaluated when the sim runs (`ss.Cond`), or are functions of the sim. A condition in one disease on a state of another (`ss.modify(hiv.rel_trans, multiply=3, where=syph.I)`) also works in a compartmental run: the states of the two diseases are then tracked jointly, as a dimension of the sim for each (so `hiv.results.n_I.by('syph')` splits HIV by syphilis state), and the two have the same people, as agents do. A dimension of some states of one of them (e.g. a strain of its infected states) is allowed. As an ODE, diseases tracked jointly take a whole run together (each taking each step in turn, as they do step by step), so they can be restarted, differentiated, and fitted by NUTS. A modifier on a transition chosen with `p=` (one of several branches out of a state) changes its probability, and the other branches share the rest; a vaccine's efficacy against such a transition (`efficacy=dict(severe=0.75)`) does the same. `ss.modify('school.pars.n_contacts', multiply=0.2)` changes the contacts of a network, under every solver (as agents, each contact then happens with that probability in a step). As agents, `sir.rel_sus.effective` gives the values with the active modifiers applied (reading `sir.rel_sus` gives the base), and with `ss.Sim(track_writes=True)`, `sim.what_modifies()` also lists the modules that assigned `rel_sus` or `rel_trans` directly.
- **Interventions**: `ss.Vaccine`, `ss.Test`, and `ss.Treatment` are products, delivered by `ss.vaccinate()`, `ss.screen()`, and `ss.treat()` in rounds, routinely, on an event (`on=sir.infection`), or by a function, with `eligible=`, `coverage=`, and `capacity=`; a test's `on_positive=` and `on_negative=` name follow-ups. `trace=` delivers to the contacts, in a network, of the people selected, for contact tracing and partner notification (agents only). `ss.transfer()` with delivery keywords moves people along a stratification; a transfer can be the target of `ss.modify()`. A delivery's results can be selected by stratum. These run under every solver: holding a product is a per-agent property in an agent-based run and a stratification in a compartmental one. A test's `delay=` and a product whose protection fades by a function of time (`waning=lambda t: ...`) now work in a compartmental run too: people wait for their result, or hold the product, in levels of one or more steps. `ss.death(lifetable=ss.lifetable(ages, deaths))` draws each person's age at death once from a life table as agents, and is a death rate for each age group in a compartmental run stratified by age; a delivery with `move=` and `capacity=` seeds a number of people on a schedule. As agents, a transition can wait for a minimum stay in its source state (`after=`), apply only during one (`before=`, as EMULSION's escape), and give each person who goes through it newborns (`offspring=(NB, 3)`); the EMULSION importer translates minimum stays as `after=`. `ss.vaccinate(move='S -> V')` moves the people reached from one state to another (a one-off campaign), `ss.Vaccine(leaky=False)` fully protects a share `efficacy` of those who receive it and leaves the rest unprotected, and `at_age=` delivers routinely as people reach an age. `ss.Vx`, `ss.Dx`, and `ss.Tx` called without the arguments of Starsim 3's classes (`df=`, `diseases=`) make an `ss.Vaccine`, `ss.Test`, or `ss.Treatment`.
- **Scenarios and runs**: `ss.parallel(sim, n_runs=100)`, `sim.run(n_runs=100)`, and `ss.parallel(baseline=..., vaccine=...)` return a MultiSim whose results have a run dimension (`msim.results.sir.n_I.quantiles()`, or by group, `msim.results.sir.n_I.select(age='0-4')`); `msim.diff('baseline')` gives the difference of each scenario from a baseline, and `msim.extinction()` how often each disease died out. `ss.parallel(sim, pars=dict(beta=[0.2, 0.3, 0.4]))` runs a sim with each set of parameter values (`msim.run_pars` has the values of each run).
- **Importing models**: `ss.load(file, format=...)` reads a model from EMULSION, odin, SBML, SimInf, flepiMoP, summer2, Atomica, camdl, EpiModel, epydemix, or MEmilio (its ODE models) into a sim (the format is told from the file where it can be), prints how each rate was read, and lists what it could not translate (`ss.ImportIncomplete`, also in `sim.imported`). `sim.to_code()` writes any sim as a Python file that builds it again, and `ss.diff(a, b)` compares two models in their own terms. The `import-model` and `validate-import` skills (in `skills/`) cover what does not map one to one, and check a port against its original.
- **Exporting models**: `ss.save('sir.xml', sim, format='sbml')` writes a sim as SBML, odin, or a Mermaid diagram, and lists what the format cannot carry (`ss.LossyExport`); SBML and odin exports read back with `ss.load()`.
- **Named data**: `ss.people('Kenya', year=2024)` gives a country's population (its size, age structure, and sex ratio, from the UN World Population Prospects), for agents or as the size and age split of a compartmental model; `ss.geo('Kenya', level=1)` gives a country's administrative units (names, populations, centroids, distances, and borders) for a spatial stratification, and `ss.contacts('Kenya', layer='home')` its contact matrix by age, both from `starsim_data`, bundled for every country at the first two levels (finer levels and other years are fetched). Where the data came from is in `describe()`, the model spec, and the hash.
- **JAX and gradients**: an ODE can be run in JAX (the same RK4 steps as the kernels; JAX is optional, `pip install starsim[jax]`), which `sim.gradient()`, `sim.log_density()`, gradient fits, and `ss.parallel()` with many sets of values use without being asked. `sim.gradient()` gives the derivative of results with respect to parameters, by automatic differentiation of the run, and `sim.calibrate()` on an ODE now fits by L-BFGS-B with these gradients when JAX is installed (Nelder-Mead otherwise; `method='asd'` uses `sc.asd()`). `ss.parallel(sim, pars=...)` with many sets of values takes all the runs in one call to JAX. `sim.calibrate()` on an ODE whose data series all have a likelihood now gives a posterior, from the No-U-Turn sampler (BlackJAX; `method='nuts'`, or `method='l-bfgs-b'` for the maximum likelihood, as before): the log-density (the run, the results with data, the likelihood, and the priors) is one function in JAX, differentiated in forward mode, so the whole of the sampling is compiled, and the chains run on cores of their own (Starsim gives JAX a device for each core when it imports it). The priors are the parameters' `prior=` (uniform over their bounds by default), and the fit has the draws (`fit.samples`), each parameter's median, 95% interval, rank-normalized R-hat, and bulk effective sample size, and a band over the posterior in `fit.plot()`. The log-density covers several diseases, including diseases tracked jointly, and outcomes that people stay in; a model it does not cover yet is sampled with Python in the loop, which is much slower. Data by stratum is given in long form, with columns named after dimensions (`age`) saying which stratum each row is for; each result and stratum is a series (`n_I[age=0-14]`), compared with `res.n_I.select(age='0-14')`. A fit notes parameters that the data cannot tell apart (`ss.Unidentifiable`): ones that appear in the model only multiplied together (or one divided by the other), with the product the data determine; ones almost perfectly correlated over a posterior; and, for a fit by gradients, ones that change the fitted results in almost the same way at the best fit. What JAX compiles is kept for later sessions, in `ss.options.jax_cache` (`~/.cache/starsim/jax` by default; `''` for none). In the JAX run, flows are added up with matrices rather than at indices, which makes a gradient about twice as fast.

- *Births and deaths*: births, deaths, and migration of everyone are declared once for the population, in `ss.Sim(demographics=[ss.birth(rate=...), ss.death(rate=...)])`, and run under every solver; the same people are born into and die from every disease. Births or deaths of everyone declared in more than one disease raise `DuplicateDeclaration`. `birth_rate=` and `death_rate=` on `ss.Sim` are removed; `demographics=True` is declared births and deaths at 20 and 10 per 1,000 per year.
- *Deliveries*: a delivery's schedule and target use the keywords of `ss.change` and `ss.modify`: `at=` (was `years=`/`dates=`) and `where=` (was `eligible=`); `on=` takes events only. `ss.vaccinate()`, `ss.treat()`, and `ss.screen()` can make their own product from its keywords (`ss.vaccinate(efficacy=0.9, coverage=0.7, at=[2025])`). A move delivered to some people is `ss.Delivery(move=dict(...), coverage=..., at=[...])`; `ss.transfer()` is movement at a rate, part of the model.
- *Several sources*: a transition with several sources is written with a sum, `S + I + R >> D @ mu` (no parentheses needed); `|` on states is a condition (`where=sir.I | sir.R`), and `(S | I) >> D` or `0.3*(Ia | Is)` raises an error that says to write the sum. `ss.states()` always returns a list (`S, = ss.states(1)`). Arrow strings keep `|` (`'S | I -> D'`).
- *Fits and plots*: `fit.divergences`, `fit.acceptance`, and `fit.band` are `fit.diagnostics.divergences`, `.acceptance`, and `.band`; the column of `fit.trials` that holds what was minimized is `objective` (it was `mismatch`; `fit.measure` says what it is). `sim.plot()` and `msim.plot()` are drawn by one function, `sim.results.plot()`: `msim.plot()` has `sim.plot()`'s titles and figure size, and `sim.plot()` matches data columns to results as `calibrate()` does. Inside, `fit()` is a `Problem` and a registry of methods (`ss.calibration.methods`), and the code for data and results is in `results.py`.
- *Code layout*: Starsim's internal modules are reorganized, with every `ss.` name unchanged: `solvers.py` is a package (`engine`, `mixing`, `compartmental`, `crossing`, `agents`, `choose`), `transitions.py` is split into `expressions.py`, `transitions.py`, `graph.py`, `presentation.py`, and `dimensions.py` (with `ss.by`, `ss.timeseries`, and `ss.seasonal` in `parameters.py`, and `ss.geo`, `ss.gravity`, and `ss.Mobility` in `networks.py`), and dates are in `dates.py`; `ss.solvers.X` and `ss.transitions.X` still find every name they did. A sim's stratifications and groups are `sim.groups` (`ss.dimensions.Groups`), and the solver is chosen by `ss.solvers.choose_solver(sim)`; `sim.strats`, `sim.n_groups`, `sim.group_levels`, `sim.group_coords`, `sim.dim_index()`, and `sim.level_raw()` still work, and the other group attributes and methods (e.g. `sim.groups_changed()`) are now on `sim.groups`.
- *Namespace*: the engines and other internals are reached through their submodules (`ss.solvers.Compartmental`, `ss.solvers.Agents`, `ss.solvers.Crossing`, `ss.transitions.Expr`, `ss.transitions.Graph`, `ss.transitions.Strat`, `ss.conditions.Cond`, `ss.diseases.InfectionLog`, `ss.run.MultiResults`), not `ss.`. What `ss.` exports is the vocabulary of the specification, the classes and helpers that modules written as code use, and the error classes, checked by `tests/test_spec.py`.
- *One spelling each*: `ss.stratify(model, ...)` is removed (use `model.stratify(...)`), and stratifying again in the same way does nothing, with a note. `ss.RandomExactNet` is `ss.RandomNet(method='exact')` and `uniform_targets=True` is `method='uniform'`; `ss.RandomSafeNet`, `ss.ClusterNet`, `ss.HybridNet`, and `ss.MSMNet` moved to the library (`ssl.`); `ss.MaternalNet` and the `ss.Route` export are removed; `ss.MFNet(duration=)` is `dur=`. `ss.contacts(matrix, age=edges)` is `ss.contacts(matrix, bands=edges)`. `track_writes=` is removed (`sim.what_modifies()` lists the declared modifiers), and `.effective` in a compartmental run raises an error that says only agents have it. The `migrate-starsim3` skill maps each old spelling.

### Changes that may affect results
- A fixed delay (`ss.constant` as a dwell time) in a compartmental run is a conveyor of whole steps, not 50 Erlang stages, so it is exact rather than approximate.
- The particle filter draws each particle's random numbers from a stream of its own, so its estimates for a given seed differ from before (statistically the same), and do not depend on the number of threads.
- With several declared diseases, births or deaths reach every disease under the compartmental solvers (before, the other diseases had a fixed population); `demographics=True` with a declared disease runs births and deaths as its transitions (the same model, other random draws).
- *Time names*: the time vocabulary is `ss.dur`, `ss.per`, and `ss.prob`, with `ss.days` to `ss.years` and `ss.perday` to `ss.peryear`; a probability per period is `ss.prob(0.1, 'day')` (and prints so). `ss.probperday` to `ss.probperyear`, `ss.freq` and `ss.freqperday` to `ss.freqperyear`, and the one-unit instances `ss.day` to `ss.year` stay. Removed: the Starsim 2 names `ss.rate`, `ss.time_prob`, and `ss.rate_prob`, deprecated in Starsim 3. A name that Starsim no longer exports raises an error that says what to write instead. `ss.dur(3)` and `ss.per(2)` need a unit (they were years).
- *Docs*: the design specification (`v4_design/design/`) matches the code: departures folded in, promised-but-absent features struck or moved to `v4_design/admin/deferred.md`, settled open questions closed; its code blocks (every chapter), error catalog, and vocabulary counts are checked by `tests/test_spec.py`. The best-practices documents keep only their argument, with links to the design for the specification.
- *Saved forms*: a model is saved as code with `ss.save('model.py', sim)` (which writes `sim.to_code()`), and a sim with its results with `ss.save('run.sim', sim)`; the model spec (`sim.to_spec()`, and `disease.to_spec()`, which `disease.to_json()` also gives) is the normalized form for `sim.hash()` and `ss.diff()` and is no longer read back (`ss.from_spec()` and `ss.load()` of a spec are removed). An outcome's likelihood is written in the spec as its distribution and parameters (not a string), so `to_code()` writes it too.
- *Results by group*: results by group are kept at every step only while they fit in about 100 MB a module (500 MB over the runs of `sim.run(n_runs=)`); beyond that they are kept every week, four weeks, quarter, and so on (a flow added up over the period, a state at its end), with a note, and totals are still kept at every step. `ss.Sim(results=['n_I', 'cases'])` keeps only the named results by group. A stochastic run with mixing between places, or with results kept every few steps, draws in a different order, so its runs differ from before (equally likely); across many runs, outcomes by group are worked out from the flows by group.
- *Dependencies*: `seaborn` is no longer installed with Starsim (only Starsim 3's `ss.Calibration` plots use it, and import it when they are made); `pip install starsim[data]` adds `starsim_data` for named data, and `starsim[interop]` what the importers need beyond the core (`openpyxl`).
- *Many runs*: a MultiSim is its results: `msim.results.<module>.<result>` is an `ss.Result` with a run axis first, however the runs were made, with `select(run=, scenario=, <swept parameter>=)`, `quantiles()`, `median()`, `mean()`, `diff()`, `resample()`, `to_df()`, and `plot()`; `len(msim)` is the number of runs. `ss.parallel()` is the one way to run many sims, and `sim.run(n_runs=)` its shortcut for one; `pars=` takes arrays. Removed: `msim.reduce()`, `msim.mean()`, `msim.median()`, `msim.reset()`, `ss.MultiResult`, and `ss.multi_run()` (`iterpars=` is `pars=`).
- *Flow names*: a flow has one name, its transition's (`sim.results.sir.infection`), and its running total is `.cum`, computed when asked for; `new_infections`, `cum_infections`, and `prevalence` are kept on `ss.SIR`, `ss.SIS`, and `ss.SEIR`, and not added to a disease declared with `ss.Disease()`; a data column `new_infections` matches the flow of a declared disease's one transmission.
- *Typed arithmetic*: arithmetic on a finished transition is about its flow, never its rate: `0.5 * infection` given to a disease is an outcome (half its flow), not a second transition at half the rate (L25).
- *Time during a step*: in every module, `sim.ti` is the time point at the start of the step (the first step has `ti == 0`, as in Starsim 3), and `sim.advance()` moves it to the end before results are recorded; a result written during a step goes at `ti + 1`. Code that tested `sim.ti == k` in an intervention fires one step earlier, and a disease written as code that is seeded at init progresses one step later than in earlier v4 builds (as in Starsim 3). `ss.solvers.interval()` is removed. A declared disease takes all of its transitions in `step()`, after the interventions, under every solver, so a delivery reaches the same people as agents as in compartments (screening as agents reached 7% fewer before). Runs as agents draw different numbers (from the same distribution).
- *Solvers*: the solvers are `auto`, `ode`, `binomial`, `gillespie`, and `abm`; `solver='gillespie'` estimates the number of events from an ODE run and takes tau leaps above a million, with a note (`approx=False` keeps it exact, and refuses a value that varies smoothly in time); `'tau'`, `'sde'`, `'stochastic'`, and `'ctmc'` are removed, with an error naming what to use. `integrator=` is the ODE's scheme only (`'rk4'` or `'euler'`); `ss.parallel(pars=...)` takes 300 or more ODE runs in one call to JAX by itself. Every compartmental run is taken by the kernels (the Python solvers and `use_kernels` are removed), including a function in a rate, a beta for each route, and `where=` on contacts (up to 20x faster); a function that gives an expression of the model's states is built into the program. `solver='auto'` tries function interventions and analyzers and `step()` overrides on groups, and runs an ODE when they work (`compartmental = True` or `False` overrides it); its reasons are one record that `sim.solvers()`, `describe()`, and the `CapabilityError` read, and `sim.kernel_reason()` says why a run is not taken whole.
- *Diseases tracked jointly*: diseases whose states are tracked jointly in a compartmental run (`ss.Crossing`, e.g. HIV and syphilis linked by `ss.modify(..., where=)`) run as one program in the kernels: every rate is worked out from the people at the start of the step (the ODE agrees with a fine step to 1e-7, where it was up to 1% off at quarter-year steps), and stochastic solvers draw each disease's events in turn from those rates. Whole runs under every solver, many runs at once, the particle filter, and JAX take crossed diseases as they take one, and crossed runs are faster (binomial 67 to 19 ms). `ss.gradients.JointRun` is `CrossRun`.
- *One language*: `ss.SIR`, `ss.SIS`, and `ss.SEIR` are declared from states and transitions, so they run under every solver; Starsim 3's keywords still work, `infected`/`susceptible` are aliases of `I`/`S` for this release, and results are named after the states (`n_I`, not `n_infected`). `ss.SIS` has no waning immunity. Starsim 3's coded versions are `ssl.CodedSIR`, `ssl.CodedSEIR`, and `ssl.CodedSIS`. `ss.Disease` is declared only; a disease written as code subclasses `ss.Infection` (as `ss.NCD` now does). A sim whose diseases are all declared counts time from 0 by default. Removed, with their v4 spellings in `skills/migrate-starsim3`: `ss.routine_vx`, `ss.campaign_vx`, `ss.simple_vx`, `ss.routine_screening`, `ss.campaign_screening`, `ss.routine_triage`, `ss.campaign_triage`, `ss.seasonality`, the short names `ss.Vx(...)`/`ss.Dx(...)`/`ss.Tx(...)` for the new products, and the Starsim 2 names `ss.rate`, `ss.time_prob`, and `ss.rate_prob`. On modules written as code, `ss.modify()` changes `rel_sus` and `rel_trans`, a condition can use their arrays (`where=hiv.cd4 < 200`), `sim.calibrate()` fits their parameters given as `ss.par` with bounds, and `sim.new()` works.
- *Errors and notes*: errors are exception classes, named as before but without numbers (`ss.UnknownState`, `ss.CapabilityError`, ...), subclasses of `ss.CodedError` (a `ValueError`; lookups are also `KeyError`s), and can be pickled; `ss.error()` is removed. Warnings are classes too (`ss.ImportIncomplete`, `ss.NotConverged`, ...), subclasses of `ss.StarsimWarning`, and notes are subclasses of `ss.Note` (`ss.note(msg, category=ss.LargeStep, sim=)`). `verbose` is one integer for every solver: 0 prints nothing, 1 (the default) the solver, notes, and about ten progress lines a run, and 2 everything; `verbose=0.1` and `'brief'` raise an error.
- *Random numbers*: each module's stochastic compartmental run draws from a generator of its own (from the sim's seed and the module's name), so its draws do not change when other modules are added, when the run is stepped, or when another sim runs in between; stochastic results differ from before. Tau-leaping draws the number leaving each compartment once, so the order of competing flows does not matter.
- *Units and defaults*: the unit of a model's bare numbers comes from the model (`unit=`, else its typed values, else dates), not from `dt` or a typed `stop`; typed values with different units alongside bare numbers raise `ss.IncompatibleTypes`. With no initial values, 1% start infected (marked as a guess), and `init()` prints R0 and the number infectious at the start. A keyword number on `ss.Disease` sets the parameter of that name (`ss.UnknownParameter` if there is none).
- *Distributions*: a distribution for a parameter other than a dwell time (`beta=ss.lognorm_ex(...)`) gives each agent their own value and makes `solver='auto'` choose agents; a compartmental run uses its mean, with a note.
- *Results and fitting*: every result has a kind (flow, stock, or ratio) that decides how it is compared with data, resampled, and summarized; `ss.outcome(every=)` and `annualize()` are removed (use `resample('year')`). A stochastic fit scores each point by the mean of 4 runs with the same seeds (`n_runs=`); optimizers use the priors when every series has a likelihood; parameter values are checked at `init()` (`ss.UnvaluedParameter`, `ss.OutOfBounds`).
- *Dimensions*: an explicit contact matrix is used whatever its dimension is called; `sir.age < 5` compares ages on a stratified model; contact matrices aggregate by population, and aging follows the population's pyramid; `ss.gravity()` gives the shape of movement and `away=` its amount; a strain dimension with partial immunity runs as agents under `auto`; `ss.StaticNet` in a compartmental run uses its graph's mean degree; a routine delivery by age reaches people as they reach the age.
- *Importers*: EMULSION's `proba` is per time unit; flepiMoP runs through `end_date`; Atomica's initial numbers follow its setup weights.
- *Guesses*: a bare number for a time (`at=`, `start=`, a time series key) is a time in the sim's unit, counted from the start, and one outside the sim raises `ss.TimeOutsideSim`; a dimension's kind comes from its value (`kind=` overrides it); a boolean dimension, or one a transfer moves people along, starts at its first level; with no population given, ages are the world's; people age only with births or deaths (or `use_aging=True`); an initial value of 1 or more is a number of people, whether written `1` or `1.0`; the string form of an SEIR's transmission needs `source=` (`ss.SourceNeeded`).
- *Typed arithmetic*: `number / dur` is a rate and `number / rate` a duration; `-` works wherever `+` does; adding probabilities combines their rates; meaningless arithmetic raises `ss.IncompatibleTypes`, and a rate slot that is not per unit time (or a `dur=` that is not a duration) raises `ss.WrongDimension`.
- *Regression*: results use the convention of other modeling frameworks, under every solver and for modules written as code too: `n_<state>[0]` is the number at the start, `n_<state>[ti]` the number at time point `ti` (after `ti` steps), and a flow at `ti` (`recovery[ti]`) the number during the step that ends there (0 at the start). So `n_R[ti] - n_R[ti-1] == recovery[ti]`. A sim with `stop=T` ends at `T`: it takes one step fewer than before (none from `T`). The step that ends at time point `ti` goes from `ti-1`: modules written as code see time point `ti` (the end of the step), as before; a declared model takes values that vary in time at its start, and a delivery (`ss.vaccinate()`, `ss.treat()`, ...) at time point `ti` happens at `ti`, between the steps, and is counted there. The events of a movement register dated at `ti` are in the numbers at `ti`.
- In a compartmental run, when a test had follow-ups for both results, the second applied to some of the people the first had moved (e.g. a vaccine for negatives also reached people the treatment of positives had just cured).
- A modifier on a transition chosen with `p=`, and `where=` on one, changed how long people took to leave by it, not how many left by it; people that `where=` excluded stayed in the state. They now change its probability, and the other branches take the rest.
- The stochastic compartmental solvers follow a value that varies smoothly in time (`ss.seasonal`, `ss.timeseries`) within a step, as the ODE does: Gillespie and tau-leaping hold it for a quarter of a step at a time (at its middle), with events exact within each quarter, and the chain-binomial and SDE solvers take it at the middle of each substep. They held it at its value at the start of the step; with steps of a quarter of a year, a seasonal SIS had 5–6% too many infections.
- As agents, people age in a model whose births are declared (`ss.birth()`), as with demographics modules (they stayed the age they were born at, unless `use_aging=True` was given).
- Under Gillespie by place (`solver='gillespie'` with places that do not affect each other within a step), counts such as an environmental pressure are settled after the step's register events, as SimInf does, so people who move count towards the place they arrive at. In the SimInf SISe3 example, infection now spreads between herds as fast as in SimInf (it was about 25% slower).

### Bugfixes
- A result key such as `plot(key='cpr')` matches the result named `cpr` when it is also part of other names.
- `ss.MultiSim(sims)` of sims that have run has their results across runs (`msim.results`) without running them again.
- A coded disease whose people were infected but whose `new_infections` are all 0 warns that its `set_prognoses()` should call `super()`.
- `ss.Deaths` and `ss.Pregnancy` run with `people_results=False` (their crude rates need the number alive, so they are left out).
- `sim.loop.insert()` after `sim.init()` runs the function once per step (it ran twice).
- A time at the sim's stop (`ss.change(at=2010)` with `stop=2010`, a delivery round, a modifier's start) is refused, since no step starts there; it was accepted and never applied.
- A run warns when a result of a module written as code looks written to Starsim 3's time index (a flow at the first time point, or a last point left at 0), naming the results and `python -m starsim.migrate`.
- An `ss.Infection` subclass with no `beta` runs, transmitting nothing (it crashed); `init_prev` given as a number on a subclass with no default distribution is a share (it crashed).
- Optimizer fits (L-BFGS-B, Nelder-Mead) search from the best three distinct points of the scan and report `converged=False` when the best two disagree; a two-parameter fit could stop at a worse optimum and say it converged. NUTS chains start from distinct scan points rather than all from the initial values, so they no longer agree on a wrong mode. The JAX log-density sets its substep cap from the corners of the bounds (it could be 37 log units off far from the initial values). `ss.perday(...) * ss.timeseries(...)` works in either order.
- A minimum stay (`after=`) or limit (`before=`) was left out of the model spec, so two models that differed only in one had the same hash; and agents ignored one on a transmission, which now raises a `CapabilityError` under every solver.
- Giving a declared model's parameter a new `ss.par` (e.g. `sim.new(pars=dict(beta=ss.par(0.1, bounds=[0.05, 1])))`) updates the parameter its transitions use, so `describe()`, the model spec, and calibration all see the new bounds.
- A bare number in `demographics=` (e.g. `ss.death(rate=0.01)`) in a model whose typed values have different units raises `IncompatibleTypes` asking for `unit=`, as other bare numbers do, rather than being read in the first unit.
- `ss.birth()` with no rate raises `UndrivenTransition` when it is declared, rather than `UnvaluedParameter` later.
- `sim.sensitivity()` left out the bounded parameters of networks; it takes the same parameters as `calibrate()`, and `ss.parallel()` builds each run (rather than restarting one) when a swept value is a network's parameter that a compartmental run turns into contact rates.
- `sim.add_module()` of a delivery after `init()` failed with a `KeyError` (`has_vaccine`): a delivery of a product the sim already has now works, and one of a new product (whose dimension is laid out at `init()`) raises a `CapabilityError` that says to give it when the sim is made.
- A state, transition, or result named after a method of the model's class (`describe=`, `equations=`) replaced the method; it raises `ReservedName`, as names of every module's attributes do.
- The people not given a starting state went to the first state made (`R, I, S = ss.states(3)` started everyone in `R`) when no transition was contact-mediated; they go to the state that nothing moves people into.
- A rate that counts another disease's state (`beta=0.15*(1 + 5*syph.I.n/N)` in HIV) counted the disease's own state of that name; it now counts the other disease's, under every solver.
- `sim.run(n_runs=)` of diseases tracked jointly ran each on its own (23% fewer HIV infections at the end); a time series held for each step next to a smooth value was taken mid-step in stepped runs (1.4% off); and with a movement register, diseases tracked jointly lost the events dated at the start (6% to 44% off).
- Found by porting the docs: `ss.parallel` workers no longer hang unpickling time factors; `describe()` and `equations()` work on `ss.SIR`, `ss.SIS`, and `ss.SEIR` (and before a sim); a value that varies in time can be a keyword of a disease (`ss.SIS(beta=ss.seasonal(...))`) and can be multiplied by a number; people given to a compartmental run give its age split and size; a parameter written inside a likelihood is named for its outcome (`cases_n`); `float()` of a parameter; `to_df()` of a sim with no calendar gives the times as numbers, not dates in year 0; `plot()` of a module the sim copied says to plot the sim's; `solver='auto'` runs a mixing pool between some people, or with a number of contacts that differs between people, as agents; a fit's seed defaults to the sim's, so it gives the same answer each time; and a parameter swept by its full name (`pars={'sir.beta': [...]}`) can be selected by its own (`res.select(beta=0.3)`).
- A delivery's coverage for several groups (`res.coverage.select(age=['0-4', '5-14'])`) is what was delivered over who was eligible in them; it added the groups' coverages.
- `ss.MixingPool` with `src=`/`dst=` a function returning a boolean array no longer crashes once agents have died; `ssl.HIV`'s deaths and cholera's environment are recorded at the time points they belong to.
- Many runs of a compartmental sim report all of their runs (before, `msim.summarize()` and `msim.compare()` used the one sim left after the last run); a ratio by group across runs is the ratio of its parts in each run (before, the numerator as agents, and an error for restarted runs).
- Importers: `dump_memilio.py` writes MEmilio's graphs (`MobilitySimulation`) and parameter studies, and no longer crashes on `osecirvvs`; `ss.load(..., format='memilio')` reads a graph as places with commuting and a study's distributions as priors, for `ss.parallel(sim, n_runs=, priors=True)`; `dump_summer2.py` names Parameters that have no value instead of failing inside summer2.
- `ss.parallel(sim, priors=True)` draws each level of a parameter with a value per level, as `calibrate()` fits it, instead of one value for all levels; and `ss.by([...], dims=['age'])` gives its values, so `pars={'beta[0]': ...}` works.
- Small dead ends say what to do: `calibrate(method='optuna')` takes `n_trials=` as well as `trials=`, and names the keywords it takes; a data file separated by tabs or semicolons is read as such; a data column that could be the result of several modules says so; `n_contacts=ss.by(...)` on a network other than `ss.PartnerNet` says which takes it (N21).
- `sim.calibrate(gof=fn)` with a loss that is a function of the sim no longer needs data.
- `sim.restart(pars=...)` with a value that varies in time for a parameter that was constant (or the reverse) gave results for the old value; it now says to use `sim.new()`.
- As agents, people born of `ss.Births` or `ss.Pregnancy` were in no state of a declared disease, so could never be infected; a deep copy of a partly run sim with a declared disease or a delivery ran the original's modules; and a declared module's random numbers depended on what else was declared.
- In a rate, nothing divided by anything is nothing (the infectious share of an empty group is 0, not NaN), as in camdl; a run whose numbers of people stop being numbers now stops with an error saying why, rather than returning NaN; and a rate per person that grows with a count of people (`beta * I.n`, often meant as `I.frac`) is noted as density-dependent.
- A dimension called `sex` with `kind=` given (e.g. a place, as in the Atomica HIV model) is no longer taken to be `people.female`.
- A parameter inside a value that varies in time (`ss.seasonal(mean=beta)`) no longer clashes with the name given to the value itself.
- As agents, a modifier of a parameter for some people (`where=`, or an amount by group) in a transmission rate with mixing by group no longer fails with a shape error.
- The notes on the unit and on states treated as counts are kept in `sim.notes` at `verbose=0`, and `approx=True` keeps the approximation notes without printing them (both were dropped); a "did you mean" no longer repeats the whole list when nothing is close.
- The particle filter used the wrong values of a parameter that varies smoothly in time, and the JAX log-density took a fitted value times a time series as constant in time.
- A modifier on a parameter that is a function, `ss.timeseries`, or `ss.seasonal` had no effect; `ss.change()` ignored `stop=`, `when=`, and `where=`.
- A Poisson or negative binomial likelihood of counts that are not whole numbers was `-inf`; a fit that cannot start now raises, naming the data series.
- `sir.pars.beta = x` replaced the parameter with a number; `sim.new(data=)`, short NUTS chains, `compare_models()` (which mixed measures), `to_code()` (which now lists what it could not write), and `sim.hash()` (which now covers modifiers, functions, full precision, and the data; every hash changes) are fixed.
- Deliveries: a waning product is given once; someone waiting for a test result is not retested; a follow-up with its own schedule, `set=`/`add=` on per-person values, and a bare `capacity=` on a routine delivery raise; `ss.Treatment(cure=)` is the probability of cure, and `ss.Vaccine` requires `efficacy=`.
- Dimensions: `stratify(immunity=...)` makes a dimension, every `stratify` takes the same keywords, a `sex` dimension is `people.female` as agents, and a `beta` for each route runs under every solver, with its keys checked (`ss.RouteMismatch`).
- Writing models: `@` with a second driver, arithmetic on a transition driven by a duration, `ss.by` arithmetic over different numbers of levels, `stages=` with `rate=`, a branch written `'I -> (R, D)'`, and a state in an arrow string that nothing enters (usually a typo) raise; `>>` and `@` return new transitions; `ss.transmission()` requires `beta=`; `ss.Sim(a, b)` takes both models; people are anchored on the states the infection moves them between, whatever order the states were made in; `describe()` marks more guesses.
- A compartmental solver asked of a model written as code raises; a function that fails when tried on groups raises, naming it, unless it asked for something only agents have.
- Importers: odin (`lambda`, `if`, contact matrices, seasonality, counts), camdl, Atomica (14 of its 16 library workbooks agree), summer2, and flepiMoP read more; an incomplete import ends with a ready request for an AI assistant with the `import-model` skill (`sim.imported.prompt`); a Starsim 3 `beta` dict of bare numbers warns below a step of one unit.
- The note that a step is large for a transition says how far whole steps can be from the ODE, and what to do.
- A model can be built with no Python frame above it, as from R through reticulate (it raised `ValueError: call stack is not deep enough`).
- `ss.parallel(sim, n_runs=...)` of a declared sim that had been initialized gave the same run every time; each run is now built again with its seed. A sim that has run (e.g. `fit.sim`) can be passed to `ss.parallel()`, and `AlreadyRunError` names `sim.new()`.
- `sim.stratify()` is kept by `sim.new()`, and so by `calibrate()` and `ss.parallel()`.
- `sim.save()` works after a run, under every solver; the parameters and states of a copied sim referred to the original's modules.
- `msim.plot()` of runs of a declared model shows the median and a 95% interval of each result for each scenario; it raised "No legend handles found".
- As agents, a result declared as a sum of states (`active=AS + SY`) is now in people, not agents (it was not multiplied by the population scale).
- A subclass of `ss.Disease` can declare its states as local variables in `__init__` (they were not found by name).
- In a compartmental run, conditions on age (`eligible=ppl.age < 5`) kept working only until the groups' ages had drifted out of their bins, since each group's age was increased at every step; a routine delivery to children stopped after half the width of the age group.
- A delivery on a transition (`on=sir.recovery`) now includes the people that a treatment moved along it (`ss.Treatment(transition=...)`), from the next step, in whatever order the interventions run; they were not included, under any solver. A compartmental run's results by group include them too.
- The note that a transition is fast compared with the timestep is no longer given for an inflow, whose rate is a number of people per unit time.
- A treatment that moves people along a transition (`ss.Treatment(transition=...)`) moves them out of the transition's own states, so it can start from any state (e.g. from first-line to second-line treatment); it moved people out of the infected states only.
- A test that detects a function of the sim and UIDs (`ss.Test(detects=func)`) is known to need agents when the sim is made (`ss.CapabilityError`), so `solver='auto'` chooses them; it stopped the run in a compartmental one.
- A function of the sim can be combined with a condition (`where=func & tb.infection`), as when a transition with `where=` a function also moves people along a dimension.
- `ss.StaticNet` given a graph or a generator stopped a sim, since it kept it as `graph`, which is how the sim finds a declared model; it is now `nx_graph`.
- A likelihood declared on an outcome is used for its data when the module's label differs from its name (as in tbsim); it was not found.
- Data indexed by dates (e.g. by `sim.results.timevec`) are matched to a sim that starts in a year but whose time is not absolute (they were read as microseconds, so lay outside the sim).
- Data are matched to a sim's time points to within half a step (it was 1e-9), so times that differ only by rounding (a sim stopped at `ss.weeks(12)` in days ends at 83.99999989) are the same time point.
- A function passed to a disease as a keyword (`seeding=fn`) is called each step in a compartmental run (it was skipped when the kernels took the whole run), and `solver='auto'` tries it on groups; one that asks for agents (`sir.S.uids`) runs the model as agents, or raises `ss.CapabilityError` under a compartmental solver.
- `calibrate()` fits a bounded parameter of a network (`ss.RandomNet(n_contacts=ss.par(4, bounds=[2, 10]))`) under a compartmental solver, where the network becomes contact rates; it was left out of the fit.
- `ss.modify(when=...)` is part of the model spec, with the function's source, so of `sim.hash()` and `sim.to_code()`; modifiers that differed only in `when=` had the same hash.
- A number of contacts given as a rate (`ss.MixingPool(n_contacts=ss.perday(10))`) is contacts per unit time, as a bare number is: a compartmental run raised a `TypeError` and as agents there was almost no transmission; a typed `beta` with a typed `n_contacts` raises `ss.WrongDimension`.
- `ss.Vaccine(waning=ss.timeseries(...))` (or another product's) raises when it is made, since waning is a function of the time since delivery; the run crashed.
- Arithmetic and `isin()` on the arrays of people made before a sim (`ppl.age + 1 > 3`, `ppl.age.isin([1, 2])`) are lazy conditions, as comparisons are; they raised `IndexError` or selected nobody.
- `sim.to_spec()` (and so `sim.to_code()` and `ss.diff()`) of a sim that is not initialized works on a copy, as `sim.hash()` does; it initialized the sim, which then could not be initialized again.
- An `AttributeError` inside a modifier's `when=` function (or any module property) is reported as itself, not as the module having no attribute `active`.
- `describe()` no longer prints "frequency-dependent" as if it were a choice of mixing, and gives no mixing line for a model with nothing contact-mediated; the default `stop=` and a parameter started at the middle of its bounds are marked as guesses, and a transition bound to two names says so.
- Messages name only current spellings (`ss.prob(0.1, 'day')`, `ss.time.freq`); a weighted destination without parentheses (`S >> 0.5*I + 0.5*R @ 0.1`) says to add them; "did you mean" lists only the closest names, and matches a parameter's name without its module.
- `ss.by()` with the wrong number of values for a dimension raises `ss.ShapeError` when the sim is built, not when it runs; `sim.solvers()` lists a module that needs agents once; data partly outside the simulated period is noted, since those times are not compared; a year-like `start=` read in the `unit=` given (`unit='day', start=2000`) is noted.
- The people not given a starting state no longer go to a state that only newborns enter (`offspring=`), which was taken as one that nothing moves people into (99 of 100 sows started as newborns).

### Performance
- The particle filter reads each data point from its first contributing step, not from step 0, so its cost is linear in the length of the data (3,200 daily points: 0.7 s, from 3.3 s).
- The exact solver on diseases tracked jointly (one program over the product of their compartments) keeps only the compartments that hold anyone, and a rate until its inputs change, rather than working out every flow of the product after each event: `ss.SIR()` and `ss.SIS()` with births and deaths over 50 years take 0.7 s rather than 20 s at 10,000 people, with the same results.
- An ABC-SMC fit no longer starts with a scan over the bounds that it did not use (10 fewer runs per parameter).
- The particle filter (`ss.ParticleFilter`, particle MCMC) runs its particles in parallel in the kernels on `ss.options.threads` threads (one in the workers of `ss.parallel()` and of a calibration): odin's 4-group SEIR takes 2.9 µs per particle-step on one thread, from 6 µs, and 0.8 µs on eight; camdl's SIR fit by particle MCMC takes 79 s, from 188 s. It works out an outcome (e.g. `cases`) at each data point from only the steps it depends on: 2.5 times faster on outcomes.
- The compartmental kernels no longer pass the program to a function at every step or event, which cost about 0.5 µs a call: the Gillespie solver is 3 to 6 times faster, and chain-binomial and ODE runs of a small model about 25% faster, with the same results.
- The exact solver by place (`solver='gillespie'` where places mix only within themselves) steps only the places that can have an event, works out each place's rates without passing the whole program for each flow, and steps many places at once on Numba's threads: SimInf's 16,000 herds with 4.7 million register events run in 3.1 s on one thread and 1.7 s on 20 (30 s before), and the particle filter on one herd takes 0.6 ms per particle run (2.7 ms before). The places draw from a stream each, so results do not depend on the number of threads.
- A movement register is read and resolved once for each distinct value rather than row by row, and places are grouped by one sort: building the 16,000-herd sim takes 2.0 s (39 s before).
- Large agent-based sims of declared diseases are 5 to 13 times faster: the number in each state is counted once and then kept up to date with every change to the states (whatever makes it, including a function's `mod.I[uids] = True`); infections are drawn in the same pass that works out each agent's chance; deaths from a life table are filed under the step they happen in when each is drawn; and the passes over the population (here and in `People`) are split between threads above 200,000 agents (`ss.options.threads`; the workers of `ss.parallel()` and of a calibration use one). LASER's England and Wales model, 3.3 million agents for two years, runs in about 8 s rather than 45 s (or 106 s with the life table declared), with the same results.
- Agent-based sims use less memory: UIDs, slots, and parents are stored as 32-bit integers (read as `ss.uids`, which stay 64-bit), as are each agent's group; a state has scheduled-exit arrays only if it has scheduled exits; and arrays grow by 20% at a time rather than 50%. LASER's England and Wales model at 3.3 million agents takes 1.1 GB rather than 1.6.
- A contact matrix within places, with commuting, is applied as the time share between places times the matrix between levels, rather than as their dense product, and movement between groups (aging, transfers, migration) is kept as sparse rates: MEmilio's 400 counties by 6 ages run in under a second rather than 25 to 80 s, with memory linear in the places. `ss.contacts()` keeps its matrix as numbers, so a sim over 774 places builds in a tenth of a second (170 s and 10 GB before). The Ghana measles model by age and district keeps 84 MB of results a run, not 2.3 GB.
- Deliveries on a schedule (routine or in rounds) to people chosen by states and stratifications, with a coverage that is a number or `ss.by`, of a product that people hold (a vaccine, TPT) or a transfer, are taken by the kernels in a whole compartmental run, so the run is no longer stepped from Python: 15–30 times faster (tbsim with BCG, 19 ms rather than 545 ms), with the same results. Such sims can now also be run many times in one call (`sim.run(n_runs=)`), restarted, run in JAX, differentiated (`sim.gradient()`), and fitted by NUTS. A product's effect no longer makes the program be worked out step by step before the run.
- A compartmental run stepped from Python (e.g. with deliveries, or a modifier with `when=`) works out where each flow's arrivals go only when a branch probability has changed: 20–30% faster.
- Large agent-based sims are faster: dead agents are removed without sorting every agent's ID, `arr.true()`, `arr.false()`, and `arr.count()` read the values of the active agents without gathering them first, the deaths in a step are counted in one pass, each agent's group is updated in place, and (for declared models) every state is counted by group in one pass. LASER's England and Wales model as 668,000 agents runs about 40% faster, with the same results; Starsim 3 sims give the same results, no slower.
- `solver='binomial'` moves people through as many stages of a dwell time as they pass in a step, with one draw per compartment, so it takes whole steps instead of a substep for each stage. This is exact where one stage a substep was not. The LASER hub-and-spoke example (49 stages) runs in 0.33 s, from 15 s. *Regression*: chain-binomial results change, since the steps are whole; `substeps=` gives shorter ones.
- `sim.calibrate()` on a sim with dates is 2–3 times faster: the data and the sim's times are no longer converted to years at every evaluation.
- Building a sim (`sim.init()`) is 2–5 times faster: the search for distributions and time parameters no longer walks results, timelines, or the copy of the inputs kept for `sim.new()`, nor (on its second pass) everything it has already searched; dates are made from years directly (all at once for a timeline); and the plan of the integration loop is made only when the loop runs (a sim run by the kernels in one call never needs it). A 30-year TB model with weekly steps builds in 22 ms (from 106), and an agent-based sim with HIV and births in 52 ms (from 114).


## Version 3.7.2 (2026-10-06)
- `ss.ClusterNet` (and so `ss.HybridNet`) and `ss.MSMNet` now shuffle agents using `float64` random numbers and a stable sort. Previously, tied `float32` random numbers were ordered differently on different CPUs, so the same seed could give slightly different results on different machines.
- *GitHub info*: PR [1435](https://github.com/starsimhub/starsim/pull/1435)


## Version 3.7.1 (2026-10-03)

### New features
- `sim.add_module()` adds a module (e.g. an intervention) to a sim that has already been initialized or partly run.
- `ss.RandomNet(uniform_targets=True)` (and `ss.HybridNet(uniform_targets=True)`) chooses the target of each edge uniformly from the eligible agents, as in Covasim v3, rather than in proportion to their number of contacts. This gives less variable numbers of contacts (e.g. a variance of 15 rather than 30 with `n_contacts=ss.poisson(20)`).
- `sim.people.dead` now covers every agent ever created (indexed by UID), so `people.dead.sum()` is the number of deaths and `people.dead.uids` are the agents who died. Dead agents are still removed from all other arrays. Previously, it was always `False` after each timestep. (Note: `ti_dead` is unchanged, since this is used to schedule deaths.)

### Performance
- Network transmission, `ss.find_contacts()`, and removing dead agents from networks now use Numba for speed (up to 3x faster for some sims). Random numbers for transmission are now only drawn for edges that can transmit. `Network.find_contacts()` is a further 1.3–4x faster, since it no longer creates a set; `ss.find_contacts()` has a new `as_array` argument to do the same.

### Changes that may affect results
- *Regression*: with `ss.options.crn = False`, transmission results differ stochastically, since fewer random numbers are drawn. Results with CRN (the default) are unchanged.
- *Regression*: `ss.RandomSafeNet` now avoids ties by using `float64`, fixing a bug which gave agent 0 too many contacts.

### Changes that may require migration
- *Migration*: sim and module parameters are now locked when the sim is initialized, so setting a parameter that doesn't exist (e.g. a typo such as `sim.pars['rand_sed'] = 2`) raises an error instead of silently doing nothing. To add a new parameter after initialization, use `pars.update(my_par=..., create=True)`.

### Bugfixes
- `dist.shrink()` now removes the SciPy distribution and the random number generator, so modules with many distributions no longer fail the size check in `sim.shrink()`.
- `ss.Results()` can be created without a module, so results can be merged like dicts, e.g. `sc.mergedicts(sim.results, other)`.
- `network.validate()` no longer fails for a network with no edges, and `sim.current_scale` no longer fails if called during initialization with `rescale=True`.
- *GitHub info*: PR [1433](https://github.com/starsimhub/starsim/pull/1433)


## Version 3.7.0 (2026-10-02)
This release adds features needed to port Covasim to Starsim (Covasim v4), and which are also useful for other disease models: 2D results and agent arrays (e.g. by variant), CRN-safe choice of exactly *N* agents, dynamic rescaling, Covasim's "hybrid" network (household, school, workplace, and community networks), and more MultiSim methods.

### New features
- 2D results: `res = ss.Result(..., columns=['wild', 'alpha'])` has shape `(npts, ncols)`. `res.alpha` (or `res.col('alpha')`) returns a 1D result for that column; scaling, `to_df()`, `sim.summarize()`, `resample()`/`annualize()`, `MultiSim.reduce()` and plotting all handle one column/line per result column.
- 2D agent arrays: `ss.FloatArr('sus_imm', columns=n_variants)` (and the same for `BoolArr`/`IntArr`) grow with the population like other states. Columns can be named, e.g. `columns=['wildtype', 'mutant']` allows `arr.mutant` (or `arr['mutant']` or `arr.col('mutant')`, and e.g. `arr.mutant += 0.1` updates the column), and more dimensions are supported via e.g. `columns=(2,3)`.
- `ss.choose_n(n, weights=None)` chooses exactly `n` agents without replacement, and is CRN-safe: if the candidates change slightly, most of the same agents are chosen. It can be used for exact-count seeding, e.g. `ss.SIR(init_prev=ss.choose_n(20))`, or anywhere else an `ss.bernoulli()` can be used.
- Distributions accept a `round` option at creation, e.g. `ss.lognorm_ex(ss.days(5), ss.days(2), round='nearest')`; `round=True` (or `'stochastic'`) rounds stochastically as before, and `round='nearest'` rounds deterministically (also accepted by `dist.rvs()`).
- Dynamic rescaling, as in Covasim: with `ss.Sim(rescale=True)`, the population scale factor starts at 1 and increases to `pop_scale` as agents stop being naive, making some of them naive again via the new `disease.make_naive()`; the per-timestep scale is in `sim.results.pop_scale`, and `sim.current_scale` gives the current value. Controlled by the new sim parameters `rescale`, `rescale_threshold` and `rescale_factor`. See the new [Population rescaling](https://docs.starsim.org/user_guide/advanced_rescaling.html) user guide page.
- `ss.Infection.infect()` is split into `infect_route()`, which computes transmission along one network, and `finalize_infections()`, which removes duplicate infections, so multi-strain diseases can call `infect_route()` once per strain.
- New networks: `ss.ClusterNet` (fully connected clusters, e.g. households) and `ss.HybridNet` (Covasim's household, school, work and community networks, which become four networks `h`, `s`, `w` and `c` when added to a sim). `ss.RandomNet` and `ss.RandomExactNet` have new `dynamic` (set `False` for a static network) and `age_range` parameters.
- New network methods: `net.remove_edges()` (indices or a boolean mask), `net['p1']` to get an edge column, and `net.eligible()`; `net.from_df()` also accepts a dict of arrays, and returns the network.
- New MultiSim methods: `combine()` (merge the sims into one larger sim), `ss.MultiSim.merge()` and `split()`, and `compare()`/`plot_compare()`.
- The infection log (`ss.infection_log`) now records the network of each infection, and extra data per infection (e.g. `log.add_entries(uids, sources, t, variant=variants)`); it stores entries as arrays, so it is much faster.
- The docs now publish a machine-readable index of the whole API, including default module parameters, for LLMs and other tools: <https://docs.starsim.org/llms.txt> (also `llms-full.txt`, with examples, and `api.json`). This replaces the root `llms.txt`, which was a copy of `CLAUDE.md`.

### Changes that may affect results
- *Regression*: stochastic rounding (`dist.rvs(round=True)`) is now CRN-safe, so results that use it (e.g. `ss.NCD`) will differ.
- *Regression*: results inside nested `ss.Results` are now scaled by `pop_scale` (previously only top-level results were).
- *Regression*: cumulative infections are now summed after scaling, which may give differences at the level of floating-point precision.

### Changes that may require migration
- *Migration*: comparing `sim.t` (an `ss.Timeline`) to a number, or using it as an index, now raises an error; use `sim.ti` instead (e.g. `if sim.ti == 10`). Setting `self.t` on a module also gives a clearer error.
- *Migration*: module names must be unique across all module types. Two modules with the same default name (e.g. two `ss.RandomNet()`s) are now renamed with a warning (`randomnet`, `randomnet_1`) rather than raising an error; explicitly set names that clash still raise an error.
- *Migration*: `ss.InfectionLog` is no longer a NetworkX graph; use `log.to_df()`, or `log.to_graph()` for a graph.
- *Migration*: `init_prev` values greater than 1 now raise an error (previously they infected everyone); use `ss.choose_n(n)` to seed a fixed number of infections.
- *GitHub info*: PR [1431](https://github.com/starsimhub/starsim/pull/1431)


## Version 3.6.2 (2026-09-30)
This release contains many LLM-assisted bugfixes. Most changes should have relatively little impact on users, although results may differ slightly for some results in some projects.

### Changes that may affect results
- *Regression*: `ss.Deaths` treated death rates given per month/week/day as per year.
- *Regression*: `ss.SIR` recovery/death and `ss.Pregnancy` losses were mistimed when the module had its own `dt`; CBR, CMR, and burn-in fetus ages also used the sim's `dt` instead of the module's.
- *Regression*: `ss.Pregnancy` burn-in started one step late and its events were counted at `ti=0`; fetal deaths caused by maternal death are now classified by gestational age; fertility data are now looked up at the mean pregnancy duration rather than the shortest.
- *Regression*: `ss.NCD` agents with a zero-length prognosis never died.
- *Regression*: `ss.randint` UID draws now use 64-bit uniforms (previously correlated for wide ranges, e.g. `ssl.ErdosRenyiNet`) and floor correctly for negative `low`; CRN uniforms are never exactly 0; zero-scale distributions (e.g. `ss.normal(std=0)`) return the constant instead of NaN; `ss.years(ss.poisson(x))` now scales with `dt`.
- *Regression*: rate/probability conversions now use `np.expm1()`/`np.log1p()` instead of `1 - np.exp()`/`np.log(1 - x)`, which is more accurate for small probabilities and fixes results differing between x86 and ARM machines (issue [1426](https://github.com/starsimhub/starsim/issues/1426)).
- *Regression*: `ss.prob.to_prob()` with no argument now uses the module's `dt`, and adding/subtracting rates with different units now gives the right value.
- *Regression*: modules given time arguments via `super().__init__(**kwargs)` now inherit the sim's `dt`.
- *Regression*: `ss.PostnatalNet` edges ended one step early; `ss.MSMNet` redrew participation every step, paired deterministically, and had unitless `duration`/`acts`.
- *Regression*: `ssl.ErdosRenyiNet` and `ssl.DiskNet` could connect dead agents; `ssl.DiskNet` agents could leave the unit square; `ssl.HouseholdNet` re-aged unborn agents.
- *Regression*: `ss.routine_triage`/`ss.campaign_triage` never delivered; `ss.treat_num` queued duplicate and ineligible agents; screening outcomes persisted between rounds.
- *Regression*: `ssl.Cholera` environmental transmission was effectively off, and agents could be infected twice; `ssl.Ebola` bodies stopped transmitting after one timestep, rather than at burial (deaths are now recorded at burial).
- *Regression*: fixed `ssl.mnch` `fetal_infection` never applying damage during pregnancy, `NeonatalSepsis` killing non-neonates, and `treat_pregnant` not cancelling deaths or applying `tx_growth_reversal` as a fraction.
- *Regression*: `sim.summarize()` now uses the final value for `cumulative` results (e.g. `births_cumulative`).
- *Regression*: date-based sims with sub-daily timesteps (e.g. `dt=ss.days(0.25)`) now step exactly; previously steps were slightly too long in leap years (0.2507 days).
- *Regression*: `ss.options.set('defaults')` no longer switches to 64-bit precision, and `STARSIM_PRECISION` is now respected.
- *Regression*: `ss.Normal` calibration likelihoods computed the default variance incorrectly and misapplied per-timepoint `sigma2`; `step_containing()` picked the next step rather than the containing one.
- *Regression*: `MultiSim(debug=True)` now runs all `n_runs`; functions added via `sim.loop.insert()` now run on the right sim after the sim is copied.

### Other bugfixes
- Campaign years outside the sim now raise an error, rather than running at the first/last timestep; `DateArray.to_date(inplace=True)` on a float array now raises an error rather than doing nothing; the unused `interpolate` argument of `ss.CampaignDelivery` and `col_names` argument of `Result.resample()` have been removed.
- Fixed crashes: comparing an `Arr` with a NumPy array; `Arr.astype()`; `ss.Births` with `crn=False` and dataframe rates; short sims with `ss.Pregnancy` burn-in; `ss.BreastfeedingNet` with deliveries during burn-in; saving an uninitialized sim; `demographics=True` with `birth_rate`/`death_rate`; `ss.RandomSafeNet(dur=<number>)`; `ss.rand_raw` under CRN; multi-disease `ss.Dx`/`ss.Tx` dataframes; `ss.RoutineDelivery` with date-based sims or non-annual `years`; `ssl.HIV` with a custom name; nested `ss.options.context()`; `Module.create()`; `Results.annualize()`/`append()`; several calibration paths (`die=False`, float-year `combine_reps`, `Binomial` `p` column, interrupted calibrations).
- Fixed plotting: `MultiSim.plot()` deleting `auto_plot=False` results and ignoring `fig`/`alpha`; `sim.plot('new_deaths')` with several matching keys; `People.plot_ages()` percentages and `bins`; `Result.plot(fig=...)`; `Loop.plot()`/`plot_cpu()`; date-axis ticks with Sciris 3.4.
- `ss.MFNet(rel_part_rates=...)` now scales participation (previously it had no effect).
- `ss.Pregnancy(dur_pregnancy=...)` now accepts a number, duration, or any distribution, not only an `ss.choice`.
- Added `DateArray.days_since()`, which gives the exact number of days since a date; `ss.DateArray()` now reads strings as dates (previously as years).
- `for p in sim.people` now iterates over active agents only; `timeline.relvec` is now numeric; `ss.StaticNet` passes extra arguments to networkx generators; `Network.plot()` passes kwargs to `nx.draw_networkx()`; `ss.SIR(name=...)` sets the label; `ss.Samples` no longer adds an `Unnamed: 0` column.
- Fixed diagnostics (`sim.set_diagnostics()`) never recording random variates; `sim.init_dists()` no longer changes dist seeds when called again; and other small fixes to time, distribution, and debugging utilities.
- Fixed several docstring examples.
- *Migration*: `ss.Network` now takes `pars` as its first argument, like other modules, plus an optional `edges` dict or dataframe; standard module arguments (`name`, `label`, `dt`, etc.) are passed to `update_pars()` and other keyword arguments are still treated as edges. `ss.Network('name')` no longer works; use `ss.Network(name='name')`.
- *GitHub info*: PR [1427](https://github.com/starsimhub/starsim/pull/1427)


## Version 3.6.1 (2026-08-29)
This release contains two performance improvements:
- `nb_indexer()` (used for `ss.Arr` UID indexing) is now written as an explicit loop rather than `arr[inds]`, due to a regression in Numba 0.66.
- `ss.RandomNet.get_edges()` now gathers edge targets from the agent array rather than the (much larger) edge array when every agent has the same number of contacts.
- *GitHub info*: PR [1423](https://github.com/starsimhub/starsim/pull/1423)


## Version 3.6.0 (2026-08-28)
This release delivers substantial framework-overhead performance improvements, focused on simulation initialization and per-timestep execution. The gains are largest for lightweight/ODE-style simulations (e.g. a bare 10-year daily sim now initializes roughly 35x faster, and runs ~2x faster with `people_results=False`); simulations with many modules see large initialization speedups from cheaper module timeline copies, and simulations run with common random numbers disabled (`ss.options.crn = False`) are ~1.2-1.3x faster than with CRN. It also contains several bugfixes, a new `ss.SEIR` base class, and a new syntax for defining state aliases.

### Initialization performance
- When a module shares the sim's timeline (the common case), its time vectors were previously deep-copied. They are now copied as containers that share those (immutable) elements.
- `ss.Timeline` now builds only the necessary vectors ( `tvec`, `tivec`, and `yearvec`); the human-friendly `datevec`, `timevec`, and `relvec` are created lazily on first access (as cached properties). This avoids constructing date sequences that may not be required by all sims.
- When every module shares the sim's timeline, plan entries are now generated directly in execution order with no sort; otherwise a numeric `np.lexsort` on the canonical time vectors is used. Previously the plan was sorted using a `(date/dur, int)` object key, which required a slow comparison step.
- `LoopEntry` now uses `@dataclass(slots=True)`, reducing the memory footprint of the 10⁴–10⁵ entries in a typical plan (`starsim/loop.py`).

### Per-timestep performance
- `sim.run()` no longer records timing for every plan entry, since most runs do not use them. Pass `sim.run(profile=True)` to record per-entry CPU timing. The plan metadata via `loop.to_df()` is still accessible.
- A `people_results` parameter was added (default `True`). Setting `ss.Sim(..., people_results=False)` skips the automatic per-timestep People-level results, saving 40% of runtime in a bare sim. Disease/module dynamics are unaffected, though some modules (e.g. demographics) rely on `results.n_alive`, so this optimization can only be used for some sims.
- A module's `update_results` is now omitted from the integration plan when the call does nothing.
- Common random numbers (CRN) can now be skipped with `ss.options.crn = False`. With CRN disabled, both distribution jumping and state copying are skipped, increase run performance by 1.2-1.3x. Note that CRN should still be used in most cases; this pathway is intended for sims that are already not CRN-safe (e.g., use `RandomFastNet`), and where performance is most important.

### SEIR models and infection incidence
- Added `ss.SEIR`, which extends `ss.SIR` with a latent, non-infectious exposed state. Its `exposed` and `infectious` states are the literal E and I compartments; `infected` means "has the infection" and is derived as E plus I, so `n_infected` and `prevalence` include latent infections while transmission depends on `infectious` alone. `ti_exposed` is the time of acquisition and `ti_infectious` is the time of becoming infectious; `ti_infected` is deliberately *not* defined for `ss.SEIR`, since it is too easily confused with `ti_infectious` (accessing it raises an `AttributeError`). `ss.SIR` continues to schedule its outcomes from `ti_infected`, since for an SIR model the two coincide; `ss.SEIR` overrides `set_progression()` to schedule from `ti_infectious` instead.
- `ss.SIR.set_prognoses()` now splits into two methods that subclasses can override separately: `set_infection()` (make the agents infectious, which `ss.SEIR` delays by the latent period) and `set_progression()` (schedule recovery, death, and any other stages, relative to the onset of infectiousness).
- `ss.SIR.step_state()` and `ss.SIR.step_die()` now clear the infection via a new `clear_infection()` method, which `ss.SEIR` overrides (since its `infected` is derived and so can't be written to).
- *Regression/bugfix*: `new_infections` is now recorded by `ss.Infection.set_prognoses()` as infections happen, rather than being inferred afterwards from `ti_infected`. Previously, a model that set `ti_infected` to a future time (as any SEIR-type model must) had its infections counted at the wrong timestep, and the seed infections subtracted on the first timestep produced a negative first value, e.g. `new_infections[0] = -196` for measles seeded with 196 initial cases (issue [1389](https://github.com/starsimhub/starsim/issues/1389)). This fixes user-written models as well as the library ones. Seed infections are still excluded from incidence, since they are prevalent rather than incident.
  - *Migration*: an `ss.Infection` subclass that overrides `set_prognoses()` without calling `super()` now leaves `new_infections`, `cum_infections` and `incidence` permanently zero, where previously the base class recomputed them from `ti_infected` afterwards. Add `super().set_prognoses(uids, sources)` to the override, or override `update_results()` to compute incidence some other way.
- `ss.mock_sim()` now includes `initialized = True`, since a mock sim stands in for an initialized one; `set_prognoses()` uses this to tell seeding apart from transmission.
- Which compartments `ss.SIR.plot()` shows is now set by a `plot_states` class attribute, so a subclass that adds a compartment can show it in one line (`ss.SEIR` sets `plot_states = ['n_susceptible', 'n_exposed', 'n_infected', 'n_recovered']`).
- `ssl.Measles`, `ssl.Cholera` and `ssl.Ebola` now subclass `ss.SEIR`. `Measles` supplies only parameter values; `Ebola` and `Cholera` also override `set_progression()`.
- *Migration*: `ssl.Cholera.dur_exp2inf` and `ssl.Ebola.dur_exp2symp` are now both called `dur_exp`, matching `ss.SEIR`. `ssl.Cholera` and `ssl.Ebola` no longer accept `dur_inf`, which they never used.
- *Regression/bugfix*: `ssl.Measles` called `ss.SIR.set_prognoses()`, which set `infected = True` without clearing it, so exposed agents were simultaneously infectious: `dur_exp` had no effect on transmission and the model behaved as an SIR model. It also sampled `dur_inf` and `p_death` twice per infection. Both are fixed. `ssl.Cholera` and `ssl.Ebola` were not affected by this, since neither inherited `ss.SIR.set_prognoses()`. Note that `ssl.Measles.infected` continues to mean exposed plus infectious, as it did in v3.5.x; it is `infectious` that now correctly excludes the latent period.
- *Migration*: for `ss.SEIR` and its subclasses (`ssl.Measles`, `ssl.Cholera`, `ssl.Ebola`), the I compartment is now `infectious` rather than `infected`, and `n_exposed` and `n_infectious` are the disjoint compartment counts. Code that read `infected`/`n_infected` to mean "currently transmitting" should use `infectious`/`n_infectious`; code that scheduled events off `ti_infected` should use `ti_infectious` (or `ti_exposed`, for the time of acquisition).
- *Regression/bugfix*: `ssl.Ebola` set `rel_trans` in `set_prognoses()`, at which point newly exposed agents are neither severe nor dead, and never updated it as agents progressed — so `sev_factor` and `unburied_factor` had essentially no effect. It is now updated in `step_state()` as agents change state.
- `ssl.Cholera.asymptomatic` is now defined via `define_aliases()` rather than as a property; its meaning (infectious but not symptomatic, and therefore shedding much less) is unchanged.
- *Bugfix*: `ssl.Cholera` recorded environmental infections with `np.full_like(..., np.nan)` on an integer array, which warns and writes the minimum integer value; it now uses `ss.dtypes.int_nan`.
- *Regression/bugfix*: `prevalence` counted only the I compartment in SEIR-type models, silently excluding every latent infection. Since `n_infected` is auto-generated from the `infected` state and `prevalence` is computed from it, an `ss.SEIR` model seeded with 218 measles cases and run for less than the latent period reported `n_infected = 0` and `prevalence = 0` at every timestep; during exponential growth, when E is disproportionately full, the majority of infections could be excluded (at peak incidence: ~8% of infected agents for `ssl.Measles`, ~87% for `ssl.Cholera`, ~89% for `ssl.Ebola`). `n_infected` and `prevalence` now count E plus I, matching standard epidemiological usage and the library docstrings, which already described `exposed` as "infected but not yet infectious".
- *Regression/bugfix*: `ssl.Cholera` and `ssl.Ebola` scheduled their extra stages relative to `ti_infected`, and tested membership of the I compartment with `self.infected`. Both now use `ti_infectious` and `self.infectious`. In particular, `ssl.Cholera.asymptomatic` (which drives environmental shedding) was `infected & ~symptomatic`, which under the corrected naming would have made every exposed agent shed.
- Since SEIR is now built in, the worked example in [Tutorial 4](https://docs.starsim.org/tutorials/t4_diseases.html) extends `ss.SIR` into an SIRS model with waning immunity instead, and the [Diseases](https://docs.starsim.org/user_guide/modules_diseases.html) user guide has a new section on the infected/infectious distinction.

### State aliases
- Added `Module.define_aliases()`, which defines a state name that resolves to something else: either the name of another state (e.g. `self.define_aliases(infectious='infected')`), or a callable taking the module as its only argument (e.g. `self.define_aliases(asymptomatic=lambda self: self.infected & ~self.symptomatic)`). A callable alias behaves like a property: it is recomputed on each access, so it can be read but not written to.
- An alias is only consulted if the attribute isn't otherwise defined, so it acts as a default that a subclass replaces just by defining a state, or another alias, of the same name.
- Any keyword arguments to `define_states()` other than `check`, `reset` and `lock` are passed to `define_aliases()`, so aliases can be declared alongside the states they refer to.
- *Migration*: `ss.Infection.infectious` is no longer a property; it is now an alias of `infected`, defined via `define_aliases()`. Reading `disease.infectious` is unchanged, and overriding it with a property still works.
- `Module` now defines `__getattr__` to resolve aliases; a missing attribute still raises `AttributeError` as before.
- A callable alias now generates an automatic `n_<name>` result, in the same way an `ss.BoolState` does, since unlike a string alias it is a derived quantity that isn't counted anywhere else. This is what lets `ss.SEIR` derive `infected` from `exposed` and `infectious` while keeping `n_infected` and `prevalence`. The new `Module.derived_state_names` property lists them. *Regression*: modules with a callable alias (e.g. `ssl.Cholera.asymptomatic`) gain a corresponding result.
- Writing to a callable alias now raises an error rather than being silently discarded. Previously `self.asymptomatic[uids] = True` would appear to work, but modified a temporary array.
- `Module.define_states(reset=...)` now accepts a state or alias name (or list of names) as well as `True`, so a subclass can replace or remove individual inherited states and aliases rather than all of them. Removing a name without replacing it leaves it undefined, which is how `ss.SEIR` drops `ti_infected`. *Migration*: `reset=True` now clears inherited aliases as well as states, so `ss.SIR` re-declares `infectious` explicitly.
- *Migration*: `define_states(check=False)` is replaced by `define_states(overwrite=True)`, which replaces an existing state or alias of the same name. `check=False` did not actually replace anything: it skipped the "already present" check but still appended, so the state ended up listed twice in `state_list` and `_auto_states`, and initialization then failed with `Another result named "n_infected" already exists`. `overwrite=True` removes the old entry first. `check=False` also disabled type validation, which is now always on. `define_aliases()` takes `overwrite` too.
- `define_aliases()` now raises if an alias would be shadowed by an existing state of the same name, rather than silently defining an alias that is never consulted; pass `overwrite=True` to replace the state instead.
- *Migration*: `Module.define_results()` no longer takes a `check` argument, which was accepted but never used. Its duplicate-name check now also covers derived states, so defining a result that collides with one generated by a callable alias gives the same explanatory error as for a `BoolState`.

### Timepars and timesteps
A timepar only means something once it has been resolved against a timestep, but there were two boundaries at which the unit was silently discarded, producing plausible-looking numbers that were wrong by whatever the unit ratio was. Both now raise, and there are explicit methods for doing the conversion.

- *Migration*: `float()` on an `ss.TimePar` now raises a `TypeError`, since it discarded the unit: `ss.years(1)`, `ss.months(1)` and `ss.days(1)` all gave `1.0`, as did `ss.peryear(1)`, `ss.permonth(1)` and `ss.perday(1)`, despite differing by a factor of 365. Use `.value` for the raw value, `.years` for the value in years, and `.to_dt()` to convert to timesteps (or `.to_prob()`, `.to_events()` and `.to()` for rates). Single-input NumPy ufuncs on a rate (e.g. `np.exp(rate)`) raise the same error, apart from the unit-independent ones (`np.isnan`, `np.isinf`, `np.isfinite`, `np.sign`, `np.signbit`); `float()` on an `ss.date` is unchanged. Any code this breaks was already computing the wrong number: `np.log(2)/ss.years(1)` and `np.log(2)/ss.years(12)` floated to the same value, so a half-life expressed that way had no effect at all. Durations remain plottable: `ss.DateConverter` is now registered for `ss.dur` as well as `ss.date` and `ss.DateArray`, so matplotlib converts them itself rather than via `float()`. This also fixes plotting a list or pandas column of durations, which previously fell back to the bare magnitude. *Migration*: `TimePar.to_numpy()` was removed (use `to_array()`, which is unchanged), since it made matplotlib unpack a timepar to a plain number before it could find the converter.
- Added `TimePar.to_dt()`, which resolves a timepar against the timestep: for a duration it returns the number of timesteps (`ss.years(100).to_dt(ss.months(1))` is `1200.0`), and for a rate the per-timestep value. This is the conversion that must happen before a duration is used in timestep arithmetic, e.g. `self.ti_clearance[uids] = self.ti_infected[uids] + self.pars.dur_persist.to_dt()`. See the new [Durations and timesteps](https://docs.starsim.org/user_guide/advanced_time.html#sec-to-dt) section of the time user guide.
- *Regression/bugfix*: three internal places used `float()` where they needed years. Only the first changes simulation output: `ss.MFNet.step()` passed `float(self.t.dt)` as the upper age for setting network states, so a monthly timestep was treated as one *year*, re-rolling `debut` and `participant` across a 12x too wide age band each step (a 40-year monthly sim shifted `cum_infections` by ~4%; a 10-year one was unchanged, since the wrongly-included agents are all far below debut age). `ss.DateArray.subdaily` and `Timeline.now('yearvec')` were also affected, but only outside the simulation period or in ways that did not propagate to results.
- `ss.Rate.to()` now accepts a duration unit as well as a rate class, and picks the matching rate subclass: `ss.peryear(1).to('month')` gives `permonth(0.0833)` and `ss.probperyear(0.5).to('month')` gives `probpermonth(0.0561)`. Previously only `.to(ss.permonth)` or `.to('permonth')` worked, while `dur.to('month')` worked fine -- an asymmetry that was easy to trip over.
- *Regression/bugfix*: converting an `ss.prob` between units now goes via the underlying rate, so it agrees with `prob.to_prob()`. Previously the value was rescaled linearly, which is right for `ss.per` and `ss.freq` but not for a probability: `ss.probperyear(0.5).to(ss.probpermonth)` gave `probpermonth(0.0417)` rather than the correct `probpermonth(0.0561)`, and scaling upwards raised `ValueError: Probabilities must be in [0, 1], not 182.5` instead of saturating towards 1. This applies to `.to()`, to direct cross-unit construction (`ss.probpermonth(ss.probperyear(0.5))`), and to conversions between an `ss.prob` and an `ss.per` or `ss.freq`. Models that converted probabilities this way were understating event rates; those that used `.to_prob()` are unaffected. `ss.Rate` also gained a `rate` property (an alias to `value`, except for `ss.prob`, where it is `-log(1-value)`), which is what the conversion uses.
- *Migration*: `Module.link_rates()` is now `Module.link_timepars()`, since it links durations as well as rates (previously only rates were given a `default_dur`, so only rates could resolve themselves against `dt` with no argument).

### Other bugfixes
- *Regression/bugfix*: `prevalence` was computed as `n_infected/len(sim.people)`, but `len(sim.people)` counts the agents in `auids`, which still includes agents who died on the current timestep — they aren't removed until `people.finish_step()`, so that analyzers can inspect them. Their disease states have already been cleared by then, so the numerator excluded them while the denominator included them, and `prevalence` was systematically understated by exactly the number of deaths that timestep (3.5% relative in an SIR run with `p_death=0.2`). It now uses the number of living agents, so `prevalence == n_infected/n_alive` exactly, as users expect. This affects `ss.Infection` and `ss.NCD`, and applies to every v3 release: the expression dates from v3.0.0, and the loop order that makes it wrong is unchanged since then, so `prevalence` from any previous version is understated for a model with mortality. Added `People.n_alive`, which is what `results.n_alive` records; use it rather than `len(people)` for any per-timestep denominator.
- *Regression/bugfix*: `results.cum_deaths` is now a true running cumulative sum; previously it recalculated the total on each timestep, resulting in an off-by-one timestep error.
- *Regression/bugfix*: `ss.RandomNet` now samples edge targets from the list of sources rather than the agent list, so it produces different (and, for stratified networks, correct) realizations than Starsim v3.5.x; `ss.RandomExactNet` and `ss.RandomSafeNet` are unaffected for contiguous UIDs.
- *Regression/bugfix*: `ss.RandomSafeNet` now forms edges between UIDs rather than positional indices.
- *Bugfix:* `sim.to_json()` now works for sims containing modules that have different timelines (and thus different results timesteps).
- *Bugfix:* `ss.Births()` now accepts a `Time` data column, matching `ss.Deaths` and `ss.Pregnancy` (the previous label of `Year` also still works, however).
- *GitHub info*: PR [1408](https://github.com/starsimhub/starsim/pull/1408)


## Version 3.5.2 (2026-07-14)
- Starsim v3.5.0 added several optimizations affecting how random numbers are drawn. However, these changes introduced several `dtype`-related bugs that have been fixed in this release:
  - Fixed a bug where `ss.random()` always returned `float64` regardless of the configured type `ss.dtypes.float`.
  - Fixed `ss.poisson()` sometimes returning `float64` and sometimes returning an integer. It now always returns `ss.dtypes.int`.
  - Fixed `ss.multi_random.rvs()` returning an incorrect number of samples due to type mismatches.
  - *Regression*: Because the default CRN mechanism precision has changed to consistently match `ss.dtypes.float`, the exact random values produced for agent-indexed draws are different from v3.5.1.
- `ss.uids()` now performs additional validation to more consistently ensure it only contains integers.
  - *Migration*: Previously `ss.uids([1.5])` would convert the input to an integer and thus return `ss.uids([1])`. However, because this usage would often reflect an error, this operation will now raise a `TypeError` to prevent inadvertent conversion of floats silently succeeding. To maintain previous functionality, convert the input to integers before calling `ss.uids()`. 

- `Timeline.now()` returns durations rather than bare floats, for simulations that use duration-based units.
- Made the `folder` argument of `ss.Samples` optional, and moved it from first to third.
  - *Migration:* Positionally-created Samples should use kwargs instead, e.g. `ss.Samples(resultsdir, outputs)` should be changed to `ss.Samples(outputs, folder=resultsdir)`.

- *GitHub info*: PR [1393](https://github.com/starsimhub/starsim/pull/1393)


## Version 3.5.1 (2026-07-09)
- Added `ss.Arr.isin()`, for comparing an array against multiple values. For example, code like `recent = (self.ti_infected == self.ti) | (self.ti_infected == self.ti-1)` could now be `recent = self.ti_infected.isin([self.ti, self.ti-1])`.
- `ss.poisson` now uses precomputed CDFs for a 60x performance gain in some cases.
- `ss.choice` (with the default `replace=True`) now uses the inverse-CDF/hash path for a ~10x performance gain in some cases.
  - *Regression*: because the `ss.choice` random-number path changed, the exact values produced for `ss.choice` UID draws differ from previous versions (results remain statistically equivalent). Models using `ss.choice` (including `ss.Pregnancy`) will produce different stochastic realizations, and stored regression baselines should be regenerated.

- Fixed a bug in `ss.library.HouseholdNet` move-out timing that only manifested when the `ss.Pregnancy` module ran on a coarser `dt` than the sim (e.g. `ss.Pregnancy(dt=ss.months(3))`). Eligibility was determined by comparing `pregnancy.ti_delivery` (an index in the pregnancy module's *own* timeline) against the sim timestep; with mismatched dt, pregnant non-heads became re-eligible far too often, greatly over-fragmenting households and suppressing household transmission (e.g. ~40% fewer infections in a TB model with quarterly pregnancy). Move-out is now evaluated exactly once per pregnancy (keyed to `ti_pregnant`), which is dt-agnostic and bit-identical when the pregnancy dt equals the sim dt.
  - *Regression*: This fix will change the results obtained from using this module.

- *GitHub info*: PR [1388](https://github.com/starsimhub/starsim/pull/1388)


## Version 3.5.0 (2026-07-05)
This release provides several major performance improvements, focused on random number generation, networks, and pregnancy. On a representative large simulation (50,000 agents, 100 years, SIS on a random network with births and deaths), the random number changes alone make the default configuration roughly 1.4x faster, or about 2x faster with common random numbers disabled (`ss.options.crn = False`); the network, transmission, and pregnancy changes described below provide further ~1.3x speedups on top of this, for a total of ~1.8-2.6x speedups.

### Common random numbers
- Common random numbers (CRN) work by giving each agent a "slot" and drawing one random number per slot, so that the same agent makes the same decision regardless of what other agents do. Previously this was implemented by drawing `slots.max()+1` numbers and keeping only those at the requested slots. Because newborn agents are assigned slots spread over a wide range (up to `slot_scale` times the population size), this could mean generating up to 5x more random numbers than were actually needed. The default CRN path now generates only the number of values needed, via a counter-based hash (`ss.distributions.hash_uniforms()`, a Numba-compiled [splitmix64](https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64) algorithm keyed by the distribution seed and draw index) that maps each slot directly to a uniform value. This preserves CRN reproducibility while removing the redundant draws.
- Added a new `crn` option (`ss.options.crn`, default `True`). Setting `ss.options.crn = False` skips the slot machinery entirely. This is not CRN-safe across scenarios, but is statistically valid and considerably faster; it is useful when reproducibility across counterfactual scenarios is not required.
- Sped up `ss.multi_random` (used for pairwise transmission probabilities) by including a dedicated fast path for the common two-distribution case that avoids constructing a typed list on each call.
- *Migration*: the `single_rng` option has been removed. Its purpose was to demonstrate the "single centralized RNG" behavior of other agent-based modeling frameworks; use `ss.options.crn = False` instead, which is both faster and demonstrates the same stochastic-branching problem. (Note, however, that it does *not* use a single RNG; this option is no longer available.)
- *Regression*: because the default CRN mechanism changed, the exact random values produced for agent-indexed draws are different from previous versions. Results remain statistically equivalent and CRN reproducibility is preserved, but stored regression baselines will need to be regenerated.

### RandomNet and HouseholdNet
- *Regression*: `ss.RandomNet` is now a faster (~1.8x), approximate network in which the target end of each edge is drawn independently. The per-agent degree is therefore only approximately Poisson rather than exact. The previous exact behavior is available as the new `ss.RandomExactNet`; switch to it if you rely on the exact degree distribution.
- Edge generation for `ss.RandomExact` networks (what were previously called `ss.Random`) now uses an in-place Fisher-Yates shuffle (`ss.networks.fisher_yates_shuffle()`) instead of `np.random.permutation`, about 1.3x faster.
- Networks whose edges last a single timestep (`dur=0`, e.g. `ss.RandomNet`) now regenerate edges via a zero-copy fast path that avoids concatenating onto the previous edge list each step.
- `ss.library.HouseholdNet` initialization is now fully vectorized across household assignment, edge construction, and female-head-of-household selection, for a performance improvement of 10-20x for large populations.
- `ss.library.HouseholdNet.add_births()` was also vectorized, for a performance improvement of ~1.3x in high-turnover simulations.

### Pregnancy
- The `ss.Pregnancy` conception logic was optimized, for a ~20% performance improvement.
- *Regression*: the default `slot_scale` for `ss.Pregnancy` was increased from 5 to 100. Newborns are assigned a random slot in the range `[n_agents, slot_scale·n_agents]`; a larger range reduces the chance that two newborns share a slot (and therefore make identical random draws). Previously a larger `slot_scale` also meant proportionally more random draws, but with the new hash-based CRN the number of draws no longer depends on the slot range, so the default was raised to reduce these collision artifacts at no performance cost. This changes results for models with pregnancy.

### Other performance optimizations

- The disease transmission step now uses a Numba-compiled kernel (`_nb_transmit`) that identifies transmitting edges in a single branchless pass and returns its result as a zero-copy view, about 1.4x faster at typical transmission densities. Effective transmissibility and susceptibility are computed directly on the raw state arrays, avoiding intermediate `Arr` wrapper allocations.
- Extracting the UIDs of a boolean state (`Arr.true()`, `Arr.uids`, and boolean filtering) now uses a branchless Numba compaction kernel above a threshold number of elements.
- The `numba_indexing` threshold (the array size above which Numba is used instead of NumPy for indexing) was lowered from 5000 to 2000, since Numba wins for both gather and compaction above ~2000 elements.

### Other changes

- `starsim.library` now exports its contents at the top level, so classes can be accessed either via their submodule (e.g. `ss.library.networks.HouseholdNet`, `ss.library.diseases.Cholera`) or directly (e.g. `ss.library.HouseholdNet`, `ss.library.Cholera`), like core Starsim.
- Added `tests/benchmark_large.py` and associated benchmark data for tracking performance on larger simulations.

### Regression information
- The default random number values for agent-indexed draws have changed (see "Common random numbers" above); results are statistically equivalent but not bit-for-bit identical, and baselines should be regenerated.
- `ss.RandomNet` is now approximate; use `ss.RandomExactNet` for the previous exact-degree behavior.
- The `single_rng` option has been removed; use `ss.options.crn = False` instead.
- The default `slot_scale` for `ss.Pregnancy` was increased from 5 to 100, which changes results for models with pregnancy.
- *GitHub info*: PR [1378](https://github.com/starsimhub/starsim/pull/1378)


## Version 3.4.0 (2026-06-21)
This release introduces the **Starsim library** (`starsim.library`), which absorbs the former standalone `starsim_examples` package. This release also adds memory, reproducibility, and usability improvements along with numerous bugfixes.

### The Starsim library
- Example and reference modules that previously lived in the separate `starsim_examples` package are now bundled with Starsim in `starsim.library`, imported as `import starsim.library as ssl`. (or just used as `ss.library`). Like core Starsim, the library exports everything at the top level:
  - Diseases: `ss.library.Cholera`, `ss.library.Ebola`, `ss.library.HIV`, `ss.library.Measles`
  - Maternal, neonatal, and child health: `ss.library.FetalHealth`, `ss.library.MaternalInfections`, `ss.library.NeonatalSepsis`
  - Networks: `ss.library.HouseholdNet`, `ss.library.SpatialNet`, and the theoretical/abstract networks
- *Migration*: `ss.FetalHealth` and `ss.HouseholdNet` are no longer available at the top level. Instead of `ss.FetalHealth()` and `ss.HouseholdNet()`, use `ss.library.FetalHealth()` and `ss.library.HouseholdNet()` (or `ssl.FetalHealth()` and `ssl.HouseholdNet()` with `import starsim.library as ssl`).
- *Migration*: `ss.MaternalNet` has been removed. It was a thin alias for the prenatal network; replace `ss.MaternalNet(...)` with `ss.PrenatalNet(...)`.
- *Migration*: the `gonorrhea` and `syphilis` example diseases have been removed from the package (they are not part of the new library). Full, maintained STI models are available in [STIsim](https://github.com/starsimhub/stisim). Replace `ss.Gonorrhea()` with `import stisim as sti; sti.Gonorrhea()` and the same for syphilis.
- Standalone example scripts ([LASER](https://laser.idmod.org/) comparisons, language translations, network embedding) have moved out of the package into `docs/examples`.

### Memory and reproducibility
- `ss.shrink()` (previously `ss.utils.shrink()`) is now available at the top level. Calling `sim.run(shrink=True)` (the new default) performs a lightweight shrink that clears the circular `Sim` references that otherwise block garbage collection. A full `sim.shrink()` additionally clears `sim.people`, taking a typical sim from ~MB to ~KB.
- Shrinking was refactored to be more robust, including for distributions and mixing pools, with clearer error messages when a module fails to shrink below its expected size.
- Distribution run caches are now cleared automatically after each `sim.run()`.
- *Regression*: distributions now count the call and jump/reset their random state even when sampling zero agents, so common-random-number (CRN) behavior no longer depends on whether any random numbers were actually drawn. This is more correct, but may produce numerically different results in models where a distribution is sometimes sampled with an empty set of agents.

### Bugfixes and improvements
- Whereas before only interventions and analyzers could be provided as pure functions, now any module can; the function becomes the `step()` method, with the sim being the argument to the function. See [Modules.from_func](https://docs.starsim.org/api/modules.html#starsim.modules.Module.from_func) for details.
- Fixed the `n_female` result not being calculated.
- `dist.set()` now raises a clear error when given a parameter name that is not valid for that distribution, rather than silently ignoring it.
- Improved the error message shown when a module overrides misses a required `super()` call.
- Fixed `ss.MultiSim` not respecting `n_runs` in some cases.
- Fixed `sim.loop.plot()`, and added and tested `sim.loop.plot_step_order()` for visualizing multi-timestep execution order.
- Fixed `MultiSim.plot('single_key')` raising `AttributeError: 'Axes' object has no attribute 'flatten'` on a reduced multisim (i.e. after `reduce()`, `mean()`, or `median()`) when plotting a single result key.
- `ss.years()` now accepts dates and date strings (and iterables of them), converting them to decimal years (e.g. `ss.years('2020-07-01')`).
- Made `ss.parse_age_range()` more robust when parsing `'X to Y'`-style ranges.
- Fixed the string representation of an uninitialized `ss.People` object.
- Pinned `numpy>2`.
- Applied an engineering uplift across the codebase: added and corrected docstrings, normalized line endings, improved error messages, and fixed assorted minor issues.

### Documentation
- Added a "getting started" guide and reorganized the API reference.
- Clarified the documentation on inclusive vs. exclusive time.
- Added new user-guide and tutorial pages, and updated guidance on common modeling gotchas (transmission `beta` as a probability, getting agent UIDs from states, custom-module conventions, etc.).
- *GitHub info*: PR [1344](https://github.com/starsimhub/starsim/pull/1344)


## Version 3.3.4 (2026-05-21)
- Fixed `ss.CampaignDelivery` failing to initialize on date-based timelines: `init_pre()` now compares against the float `sim.t.yearvec` instead of the date-typed `sim.timevec`, which caused `sc.findnearest` to crash because subtracting `ss.date` objects produces `datedur`.
- Fixed `ss.CampaignDelivery` missing a `coverage_dist` placeholder in `__init__`, which caused `AttributeError` on the first delivery. This affected `ss.campaign_screening`, `ss.campaign_triage`, and `ss.campaign_vx`.
- Fixed `ss.RoutineDelivery` off-by-one when `dt = 1`: `adj_factor` is now `0` (was `1`) for `dt >= 1`, so the intervention no longer runs for one extra year past `end_year`. The previous behavior also caused an `IndexError` when a per-year `prob` array was supplied.
- Aligned `RoutineDelivery.yearvec` with `self.timepoints` by slicing `sim.t.yearvec` directly, ensuring `prob` from `sc.smoothinterp` always matches the timepoint length.
- **Regression**: `ss.routine_screening`, `ss.routine_triage`, and `ss.routine_vx` with `dt = 1` will now stop one year earlier than before (at `end_year` rather than `end_year + 1`).
- Added test coverage for `RoutineDelivery` and `CampaignDelivery` in `tests/test_interventions.py`.
- *GitHub info*: PR [1325](https://github.com/starsimhub/starsim/pull/1325)


## Version 3.3.3 (2026-04-15)
- Added `ss.parse_age_range()` and `ss.apply_age_range()` to parse age range specifications like `'[5, 10), [10, 15)'`.
- Optimized `ss.RandomNet()` performance.
- Fixed `ss.HouseholdNet.add_births()` to assign household IDs exactly once at delivery, rather than using an age-window heuristic tied to `update_freq`. Newborns are now identified by having a parent, age >= 0, and no household ID yet assigned. Moved HouseholdNet's `add_births()` call before the `update_freq` gate so it runs every timestep, ensuring correct behavior with any update frequency or when the network timestep exceeds gestation.
- Inspired by [Starsim.jl](http://epirecip.es/Starsim.jl/dev/), added examples of Starsim models translated into pure Python, Numba, Jax, Julia, and Rust in `starsim_examples/translations`. (Thanks to [Simon Frost](https://github.com/sdwfrost) for the Julia conversion.)
- Added code coverage, linting, and performance benchmarks to CI/CD.
- *GitHub info*: PR [1308](https://github.com/starsimhub/starsim/pull/1308)


## Version 3.3.2 (2026-04-04)
- Added a `copy_sim` argument to `ss.MultiSim.run()` to handle large parallel runs of a single sim (e.g. `ss.MultiSim(sim, n_runs=100)`), where multiprocess pickling isn't guaranteed to implicitly copy the sim.
- Removed (unused) `_locked` attribute from `ss.TimePar`.
- Converted tutorials and user guide from Jupyter notebook (JSON) to Quarto notebook (Markdown) format.
- Added GitHub Actions workflows for test coverage, docs build, and linting.
- Changed from implicit (`*`) to explicit imports in `__init__.py`, and removed `__all__` from individual modules.
- *GitHub info*: PR [1293](https://github.com/starsimhub/starsim/pull/1293)


## Version 3.3.1 (2026-04-02)
- Added `uids.concatenate()`, which unifies `uids.concat()` and `uids.cat()` into a single method that works as both an instance method (`x.concatenate(y)`) and an unbound class method (`ss.uids.concatenate(x, y)` or `ss.uids.concatenate([x, y])`). `None` values are silently filtered. 
- `uids.concat()` and `uids.cat()` are deprecated but provide a `DeprecationWarning` for backwards compatibility, to be removed in a future release.
- Added `+` and `+=` operators to `ss.uids` for concatenation (e.g. `a + b` is equivalent to `a.concatenate(b)`). Note that `+` preserves duplicates, unlike `|` which deduplicates.
- Comparison operations on `ss.uids` (e.g. `a == b`) now correctly return a plain `np.ndarray` instead of a `uids` instance.
- Scalar reductions on `ss.uids` (e.g. `x.max()`, `x.min()`) now return a plain Python `int`, consistent with element indexing (`x[0]`).
- Numeric aggregation operations that are not meaningful for UID arrays (`sum()`, `mean()`, `std()`, `var()`, `prod()`, `cumsum()`, `cumprod()`) now raise `TypeError`.
- Added `ndim` to `ss.TimePar`, fixing a bug when trying to assign scalar TimePars (e.g. `ss.years(1990)`) to `pandas` dataframes.
- *GitHub info*: PR [1279](https://github.com/starsimhub/starsim/pull/1279)


## Version 3.3.0 (2026-03-27)
- Made scalar time parameters behave more consistently by removing the ability to index them if they are scalars, and also having `np.iterable()` return `False` for such parameters
- **Backwards-compatibility notes:** This change may cause simulation results to be numerically different, as it will cause distributions to use a more efficient sampling method for affected scalar parameters (in such cases, there is likely to be a performance increase)

### Pregnancy loss and neonatal death classification
- Added background pregnancy loss via `p_loss` parameter in `Pregnancy`. Per-timestep hazard means early losses are naturally more common.
- Prenatal deaths are now classified by gestational age: miscarriage (<20w) vs stillbirth (>=20w), controlled by `loss_threshold` parameter.
- Preterm (<37w) and very preterm (<32w) births are now classified and tracked, with configurable thresholds via `preterm_threshold` and `very_preterm_threshold`.
- Added passive neonatal death detection: deaths of agents aged 0-28 days are classified as neonatal deaths.
- New results: `miscarriages`, `stillbirths`, `nnds`, `n_preterm`, `n_very_preterm`, `preterm_rate`.
- New states on newborns: `preterm`, `very_preterm`, `neonatal_death`.
- Removed unused `_p_miscarriage` and `_p_stillbirth` placeholder distributions.
- *Migration*: `n_pregnancies_this_step` and `n_births_this_step` counters removed; use results directly (e.g. `new_pregnancies`).

### Generic congenital outcome framework
- Added `set_congenital()` to the base `Infection` class. Diseases opt in by defining `birth_outcome_keys` and `birth_outcomes` in pars.
- Added `step_congenital()` helper to process scheduled congenital events at delivery time.
- Added `_assign_congenital_outcomes()` override point for state- or GA-dependent outcome probabilities.
- Death outcomes (`miscarriage`, `nnd`, `stillborn`) trigger `request_death`; non-lethal outcomes set a bool state.

### FetalHealth improvements
- Simplified `apply_timing_shift`, `apply_growth_restriction`, `reverse_timing_shift`, `reverse_growth_restriction` — removed redundant pregnancy filtering and type coercions.
- `n_exposures` state tracks disease exposures during pregnancy (connectors increment directly via `fh.n_exposures[uids] += 1`).
- Added `interp_fn` parameter for customizing birth weight interpolation.
- Negative growth penalties (growth boost, e.g. GDM macrosomia) are now additive rather than using diminishing returns.

### MNCH examples
- Added `starsim_examples/mnch/` with reusable maternal/newborn/child health examples:
  - `CongenitalDisease`: simple SIR with congenital outcomes via the generic framework.
  - `NeonatalSepsis`: SIR that infects/kills newborns, useful for testing NND detection.
  - `fetal_infection`: connector that damages fetal health when mothers are infected.
  - `treat_pregnant`: intervention that treats infected pregnant women.

- *GitHub info*: PR [1269](https://github.com/starsimhub/starsim/pull/1269)


## Version 3.2.2 (2026-03-11)
- Added `people.born`, which tracks agents that are alive (`sim.people.alive`) and born (`sim.people.age >= 0`).
- Fixed a bug that prevented `ss.Arr` objects from being modified in-place by functions involving other `ss.Arr` objects.
- Fixed an initialization bug in `ss.HouseholdNet`.
- *GitHub info*: PR [1227](https://github.com/starsimhub/starsim/pull/1227)


## Version 3.2.1 (2026-03-06)
- Fixed a bug introduced in v3.2.0 that changed how modules were parsed. Also introduced a `custom` argument to `ss.Sim()`, to specify custom modules.
- Moved `HouseholdNet` from `starsim_examples` into `starsim` and merged it with `EvolvingHouseholdNet`. By default, the network is dynamic. To use the static network, use `ss.HouseholdNet(dynamic=False)`.
- The calibration database is now created by default in a temporary folder, to avoid collisions with multiple calibrations running simultaneously; to restore previous behavior, set either `study_name` or `db_name`, e.g. `db_name='starsim.db'`.
- *GitHub info*: PR [1200](https://github.com/starsimhub/starsim/pull/1200)


## Version 3.2.0 (2026-03-03)

### Sim module changes
- Changed `sim.modules` to `sim.custom` for storage of custom modules that do not match standard types (e.g., diseases, interventions). Passing modules in via a `modules` argument is still supported, such modules will automatically be moved to the appropriate attribute upon initialization. 
- Updated `sim.modules` to now return an iterator over all modules.
- Removed `sim.module_list` which can be replaced with `list(sim.modules)` if an exact replacement is required. In many cases, `sim.module_list` was used only to iterate over the modules, in which case it can be directly replaced with `sim.modules`.
- Removed `sim.module_dict` which can be replaced with `sim.get_modules(as_dict=True)` if required. `sim.module_dict` was removed as an attribute to help avoid ambiguity as to where modules are stored in the `Sim`. In cases where the `module_dict` was used to retrieve modules by name, `sim.get_module` can be used instead. 
- Added `sim.get_modules()` to search for modules, and `sim.get_module()` to retrieve exactly one module. Also added `ndict.get()` (e.g. `sim.diseases.get()`) with similar functionality.

### Other changes
- Fixed a bug in `ss.Timeline()` that led to array length mismatches for certain combination of `start`, `stop`, and `dt`.
- Changed the default start to be the year 2000 regardless of other input (previously, if e.g. `dt='month'`, the start would change to being zero-based); now, only if `start=0` is a timepoint-based (non-calendar-based) timeline assumed.
- Fixed a bug introduced in v3.1.0 that prevented `People.filter()` from working.
- Provided methods to annualize results for plotting (e.g. `sim.results.sir.prevalence.annualize()`).
- Added `sim.people.keys()` as an alias for `sim.people.states.keys()`, and `sim.dists.keys()` as an alias for `sim.dists.dists.keys()`.
- Added option for `ss.MultiSim` to be constructed from other MultiSims and from tuples of sims (previously, only lists of sims were supported).
- *GitHub info*: PR [1186](https://github.com/starsimhub/starsim/pull/1186)


## Version 3.1.1 (2026-02-18)
- Added `sim.set_diagnostics()` (and the associated `ss.Diagnostics()` class), which allows for tracking of every agent in every state and every random number generated by the sim.
- Fixed bug with pregnancy burn-in and NumPy 2.4.
- Fixed bug with pregnancy results calculation for sims with duration less than 1 year.
- *GitHub info*: PR [1145](https://github.com/starsimhub/starsim/pull/1145)


## Version 3.1.0 (2026-02-12)
- *Migration information*: See `docs/migration/v3.0_v3.1.md` for a detailed, LLM-friendly migration guide.
- Added new functionality to the `Pregnancy` module:
    - Renamed `make_p_fertility()` -> `make_p_conceive()`
    - Added variable durations of pregnancies and tracking of pre-term birth outcomes
    - Added variable breastfeeding durations
    - Refactored the logic of how pregnancies and births are handled with new helper methods for better extensibility
    - Added results on age-specific fertility rates and total fertility rates
- Previously `People.(module_name).(state_name)` could be used to access module states (e.g., `people.sir.infected`). However, this mechanism fails to pass on property attributes that are dynamically computed (e.g., `SIR.infectious`). Now `People.(module_name)` simply contains a reference to the module, so all attributes can be accessed. 
- Add support for operating on `ss.BaseArr` with time parameters (e.g., multiplying a state by a duration)
- Add support for creating an `ss.dur` from an `ss.BaseArr` 
- Changed date plotting to convert `ss.date` and `ss.DateArray` values to years internally. This facilitates adding extra data to plots when the data is in years.
- Added `ss.parse_age_range` utility function to standardize formats for age ranges in data/input files.
- Implemented `__bool__` for `ss.bernoulli` which returns `False` if there is no possibility of returning any `True` values.
- Added automatic result creation for `ss.BoolState` instances in `People`, matching the behavior already present in modules. `People` now has `auto_state_list` property and `init_results()` method. 
- Implemented operators for `ss.BoolArr` and `ss.uids` where the latter is treated as a `BoolArr`. This makes operations like `people.sir.infected & people.female` and `people.sir.infected & people.female.uids` equivalent.
- Implemented in-place operators for `ss.BoolArr` and `ss.uids`. The former are true in-place operations that reuse existing memory and preserve references, and thus they can be safely used with `BoolState` instances in modules directly.
- Fixed/added operators `+` and `-` for `ss.DateArray` when operating on `ss.date` and `ss.dur` instances
- Fixed bug when multiplying an `ss.prob` by an array of durations
- Renamed arguments to `MaternalNet.add_pairs()` from `mother_inds` to `mother_uids` and `unborn_inds` to `unborn_uids` to clarify that these inputs should be agent UIDs.
- Removed `dur` argument to `MaternalNet.add_pairs()` as the edge lifespan is now managed by the `Pregnancy` module with edges being removed upon delivery.
- Removed `Network.prenatal` and `Network.postnatal` attributes as these are now tracked by type (`ss.PrenatalNet` and `ss.PostnatalNet`)
- *GitHub info*: PR [1065](https://github.com/starsimhub/starsim/pull/1065)


## Version 3.0.6 (2026-01-28)
- Fixed bug in `sim.loop.plot_cpu()` when using pandas v3
- *GitHub info*: PR [1118](https://github.com/starsimhub/starsim/pull/1118)


## Version 3.0.5 (2025-10-22).
- Fixed bugs with products being added/stepped multiple times.
- Fixed bug that prevented `Result.resample(output_form='series')` from being used.
- *GitHub info*: PR [1095](https://github.com/starsimhub/starsim/pull/1095)


## Version 3.0.4 (2025-10-22)
- Added additional calibration examples for workflows to re-identify known parameters.
- Removed `p_death` from the parameters for the `ss.Deaths` module, as `Deaths.make_p_death()` always overrides this value and is therefore the means by which the death probability should be set or changed. 
- *GitHub info*: PR [1055](https://github.com/starsimhub/starsim/pull/1055)


## Version 3.0.3 (2025-10-07)
- Swapped order of `age` and `female` states in People to allow `female` dist to use `age` state during population initialization.
- *GitHub info*: PR [1051](https://github.com/starsimhub/starsim/pull/1051)


## Version 3.0.2 (2025-08-25)
- Additional minor updates following the v3 release.
- Cleans up some of the logic for converting rates to probabilities in the demographic modules
- Adds a `request_removal` method to People, which can be used to remove people from the population for reasons other than death (e.g. migration)
- Fixes a bug that prevented `ss.prob` from accepting arrays
- Adds a `year` property to Timelines, so `sim.t.now('year')` can be rewritten as `sim.t.year`
- *GitHub info*: PR [1032](https://github.com/starsimhub/starsim/pull/1032)


## Version 3.0.1 (2025-08-04)
- Minor updates following the v3 release.
- The logic for initializing `ss.Timeline` has been changed. Now, `start` and `stop` will not be converted to dates unless at least one argument is a date (e.g. `start=ss.years(2000)` will remain `ss.years()`). `ss.Timeline` is also now much more robust about handling different types of input, and choosing sensible defaults for missing inputs (e.g. with `stop=ss.days(20)`, the new default is `start=ss.days(0)`).
- Subtracting dates now yields a `relativedelta`, e.g. `ss.date(2025) - ss.date(2015)` now gives `ss.datedur(years=10)` (previously it converted to days).
- `ss.DateArray` now has a `unit` attribute, allowing more reliable conversion to years.
- Timepars now have a `disp()` method.
- *GitHub info*: PR [1029](https://github.com/starsimhub/starsim/pull/1029)


## Version 3.0.0 (2025-08-02)

### Summary
Starsim v3 includes a reimplementation of how time is handled, an extensive new suite of debugging tools, and smaller changes, including an extensive user guide in addition to the tutorials. Please also see `docs/migration_v2v3` for a detailed (and LLM-friendly) migration guide for porting existing Starsim code over to the new version. If a point below says "See the migration guide", that indicates that additional information (and a conversion script where possible) is provided in that guide. Otherwise, it is (generally) a non-breaking change.

### Time
Time is now based on precise datetime stamps (specifically, `pd.Timestamp`). In Starsim v3.0, the supported time units are days, weeks, months, and years. Days or years are always handled exactly and weeks and months are defined in terms of them, i.e. a week is exactly 7 days and a month is exactly 1/12th of a year. The following changes have been made to the API:

- While durations work similarly as in Starsim v2, rates work differently. The base class, `ss.Rate`, cannot be used directly. Instead, you must use one of the three derived classes. `ss.freq()` is the closest to `ss.rate()` in Starsim v2, and is a simple inverse of `ss.dur()`. `ss.per()` is the equivalent of `ss.rate_prob()` in Starsim v2, but is the primary probability-based rate that should be used (e.g. for beta, birth rates, death rates, etc.). Finally, `ss.prob()` is the equivalent to `ss.time_prob()` in Starsim v2, but whereas `time_prob` was preferred in v2, `per` (equivalent to `rate_prob` in v2) is preferred in v3.
- In addition to these base classes, each of them is available for each time unit. For durations, singletons are available (`ss.day`), as well as e.g. `ss.days()`, `ss.years()` etc. For rates, `ss.peryear()` is derived from `ss.per`, while `ss.probperyear()` is the equivalent for `ss.prob` and `ss.freqperyear()` is the equivalent for `ss.freq`. `ss.prob` can also be unitless (`ss.per` and `ss.freq` cannot be).
- `ss.beta()` has been removed; use `ss.prob()` instead for a literal equivalent, although in most cases `ss.per()` is preferable, e.g. `ss.peryear()`.
- `ss.rate()` has been removed; use `ss.freq()` instead for a literal equivalent, although in most cases `ss.per()` is preferable, e.g. `ss.peryear()`.
- `unit` has been removed as an argument; use `dt` instead, e.g. `ss.Sim(dt=1, unit='years')` is now `ss.Sim(dt=ss.year)` (or `ss.Sim(dt='years')` or `ss.Sim(dt=ss.years(1))`).
- Although `ss.dur()` still exists in Starsim v3.0, it is preferable to use named classes instead, e.g. `ss.years(3)` instead of `ss.dur(3, 'years')`.
- `ss.Time()` is now called `ss.Timeline()` and its internal calculations are handled differently.
- `ss.time_ratio()` has been removed; time unit ratio calculations (e.g. months to years) are now handled internally by timepars.
- `t.abstvec` has been removed; in most cases, `t.tvec` should be used instead (although `t.yearvec`, `t.datevec`, or `t.timevec` may be preferable in some cases).
- Multiplication by `dt` no longer happens automatically; call `to_prob()` to convert from a timepar to a unitless probability (or `to_events()` to convert to a number of events instead).

For full details, see the migration guide.

### Distributions
- Distributions now have a `scale_type` attribute, which determines how they scale with time: some distributions, like `ss.normal()`, can be scaled either before or after random numbers are drawn; others, like `ss.poisson()`, can only be scaled before. See `ss.scale_types` for details.
- Distributions that can be scaled post-draw now take an `unit` argument, e.g. `ss.normal(mean=5, std=2, unit=ss.years(1))`. This is equivalent to `ss.years(ss.normal(mean=5, std=2))` or `ss.normal(mean=ss.years(5), std=ss.years(2))`. Note that not all distributions can be scaled this way (you will get an error if you try to scale a non-scalable distribution.)
- Distributions now have a random-number-safe `randround()` method
- There are two new distributions, `ss.beta_dist()` and `ss.beta_mean()` (not called `ss.beta()` to distinguish from the beta transmissibility parameter).

### Sim and People changes
- Sims now take an optional `modules` argument. These are run first, before anything else in the integration loop. If you want, you can supply everything directly as a module, e.g. `ss.Sim(modules=[ss.Births(), ss.RandomNet(), ss.SIR()])` is equivalent to `ss.Sim(demographics=ss.Births(), networks=ss.RandomNet(), diseases=ss.SIR())`. You can also add your own custom modules, not based on an existing Starsim module type, and specify the order they are called in.
- `sim.modules` has been renamed `sim.module_list`.
- `ss.People` has a new `filter()` method, which lets you chain operations, e.g.: `ppl = sim.people; f = ppl.filter(ppl.female & (ppl.age>5) & ~ppl.sir.infected)`
- `ss.People` has new methods `plot()` (all variables) and `plot_ages()` (age pyramid by sex).

### Module changes

#### Base module updates
- Module methods have a new decorator, `@ss.required()`, which flag that it is an error for the method not to be called. This is used to prevent the user from accidentally forgetting to call `super().method()`.
- Modules can now be used like dictionaries for accessing user-defined states, e.g. `module['my_custom_state']` is an alias for `module.my_custom_state`.
- `module.states` has been renamed `module.state_list`. `module.statesdict` has been renamed `module.state_dict`. There is also a `module.auto_state_list` property, referring specifically to `ss.BoolState` attributes.
- Built-in modules now have function signatures that look like this example for `ss.Births()`: `def __init__(self, pars=None, rel_death=_, death_rate=_, rate_units=_, **kwargs):`. Although `_` is simply `None`, this notation is short-hand for indicating that (a) the named arguments are the available parameters for the module, (b) their actual values are set by the `define_pars()` method.
- Modules now have a much briefer `__repr__`, typically one line. The full module information can still be obtained via `module.disp()`.

#### Changes to networks
- `key_dict` has been removed from `ss.Network()`; modify the `network.meta` dictionary directly instead, e.g. `ss.DynamicNetwork` has `self.meta.dur = ss_float` in its `__init__()` method, while `ss.SexualNetwork` has `self.meta.acts = ss_int`.
- `ss.Network()` now has `plot()` and `to_edgelist()` methods.
- `ss.RandomNet()` now only adds "missing" edges, fixing a bug in which the longer the edge duration, the more edges the network had.
- There is a new network, `ss.RandomSafe()`, whch is similar to `ss.Random()` but random-number safe (at the cost of being slightly slower).
- For `ss.MixingPool`, the argument `contacts` has been renamed `n_contacts`.

#### Changes to other modules
- There is a new built-in analyzer `ss.dynamics_by_age()`.
- There is a new built-in connector `ss.seasonality()`.
- `ss.Births()` is now random-number safe.
- `ss.Deaths()` now has a default rate of 10 per 1000 people per year (instead of 20). Births is still 20. This means that with `demographics=True`, the population grows at roughly the correct global average rate.
- `ss.sir_vaccine()` has been renamed `ss.simple_vx()`.

### Debugging tools
Starsim v3 comes with a new set of tools for debugging (in `debugtools.py`): both understanding simulations and what is happening at different points in time, and understanding and profiling code performance to find possible improvements to efficiency. These include:
- `sim.profile()`: profiles a run of a sim, showing module by module (and optionally line by line) where most of the time is being spent. See `ss.Profile()` for details.
- `ss.Debugger()`: steps through one or more simulations, and raises an exception when a particular condition is met (e.g., if the results from two sims start to diverge).
- `ss.check_requires():` use this function to check if a sim contains a required module (e.g., an HIV-syphilis connector would require both HIV and syphilis modules to be present in the sim).
- `ss.check_version()`: for checking which version of Starsim is installed.

#### Loop debugging tools
- `ss.Loop` now has an `insert()` method that lets you manually insert functions at an arbitrary point in the simulation loop.
- `ss.Loop` now has a `plot_step_order()` method that lets you validate that the steps are being called in the expected order.

#### Profiling tests
- In the tests folder, there is a `benchmark_tests.py` script for benchmarking the performance of the tests. Each test is also now timed (via the `@sc.timer` decorator).
- There are also various profiling scripts, e.g. `profile_sim.py`.
- Baseline and performance benchmark files have been converted from JSON to YAML.

#### Mock functions
Starsim components are intended to work together as part of an `ss.Sim` object, which handles initialization and coordination across modules, distributions, time parameters, etc. But sometimes, it's useful to build a very simple example to test a component in isolation. Starsim v3 comes with "mock" components, which allow you to work with e.g. a module without incorporating it into a full sim. These have the essential structure of the real thing (e.g., `sim.t.dt`), but without the complexity of the full object. These mock objects are:

- `ss.mock_sim()`: generates a mock sim; useful for testing modules
- `ss.mock_module()`: generates a mock module; useful for testing distributions
- `ss.mock_people()`: generates a mock `ss.People` object; used by `ss.mock_sim()`
- `ss.mock_time()`: generates a mock `ss.Timeline` object; used by `ss.mock_sim()` and  `ss.mock_module()`

Distributions now have a `mock()` method, that allows you to immediately start using them to generate random numbers, e.g. `ss.normal(5,2).mock().rvs(10)`.

### Other changes

#### File structure
- All diseases except for SIR, SIS, and NCD have been moved to a separate `starsim_examples` folder. This is installed together with Starsim, but must be imported separately, e.g. `import starsim_examples as sse; hiv = sse.HIV()` instead of `import starsim as ss; hiv = ss.HIV()`. See the migration guide for details.
- `ss.register_modules()` lets you register external modules as Starsim modules, to allow calling by string; e.g. `ss.register_modules(sse)` (from above) will let you do `ss.Sim(diseases='hiv')`, since `sse.HIV` is registered as a known Starsim module.
- Files have been reorganized:
    - `analyzers.py` and `connectors.py` have been created (split out from `modules.py`);
    - `calibration.py` and `calib_components.py` have been combined into `calibration.py`;
    - `disease.py` has been renamed `diseases.py` and `diseases/sir.py` and `diseases/ncd.py` have been incorporated into it;
    - `timeline.py` has been split out from `time.py`.

#### Plotting
- Plotting defaults have been updated; you can use these defaults via `with ``ss.style()`.
- Result plotting has been improved, in terms of module labels and correct x-axis labels.
- `ss.MultiSim()` plotting has been improved to display legends correctly.
- Plotting arguments are now handled by `ss.plot_args()`, which will parse different arguments among figure, plot, scatter, and other functions, e.g. `ss.plot_args(dpi=150, linewidth=3)` will set the figure DPI and the plot line width.
- Sim results are now automatically skipped during plotting if they were never updated (e.g., `n_deaths` if there were no deaths).

#### Array and results updates
- Array indexing has been reimplemented, using Numba instead of NumPy for large operations; this should be about 30% faster. An unnecessary array copy operation was also removed, for a further ~50% efficiency gain. (Note that although array indexing is now much faster, it was not typically the slowest step, so "real world" performance gains are closer to 10-20%.)
- There is a new class, `ss.IntArr`, although in most cases `ss.FloatArr` is still preferred due to better handling of NaNs.
- `ss.State` has been renamed to `ss.BoolState`. See the migration guide for details.
- `ss.options._centralized` has been renamed `ss.options.single_rng`. Although there is a very small performance benefit, use of this option is not recommended.
- `ss.set_seed()` has been removed; the seed should be set automatically by the distributions. If you want to set the seed for a custom (not random-number-safe) non-distribution random number, call `np.random.seed()` manually.
- Distributions and modules now define their own `shrink()` methods (for saving small files). Note that if you define a custom module that stores a lot of data, you may want to define your own `shrink()` method to remove this data.
- `ss.Result()` objects now have `disp()` and `to_str()` methods, and they can also be treated as dicts (e.g. `res['low']` instead of `res.low`).
- You can now create an `ss.MultiSim` comprised of sims of different lengths; however, the sims will be truncated to be of the same length, and they are _not_ temporally aligned. In general, it is still inadvisable to use `msim.reduce()` with sims of different lengths.

#### New options

Starsim v3 comes with several new options, set via `ss.options`:

- `ss.options.check_method_calls`: whether to check that required module methods are called (default `True`)
- `ss.options.install_fonts`: whether to install custom fonts for plotting (default `True`)
- `ss.options.numba_indexing`: threshold at which to switch from NumPy to Numba indexing (default 5000 indices)
- `ss.options.style`: plotting style to use; options are "starsim", "fancy", "simple", or any Matplotlib style
- `ss.options.warn_convert`: whether to warn when automatically converting time parameters

`ss.options` also now has a `help` method that will print detailed help on a given option or all options, e.g. `ss.options.help(detailed=True)`.

#### Migration summary
- An agentic LLM such as Cursor or Claude Code should be able to perform most of the migrations from v2 to v3 automatically. Tell it to read `llms.txt`, then `docs/migration/v2_v3/README.md`. However, there will still be a few things to manually double check, especially around time conversions (such as whether you want `ss.beta()` ported literally to `ss.probperyear()`, or "upgraded" to `ss.peryear()`).
- *GitHub info*: PR [927](https://github.com/starsimhub/starsim/pull/927)


## Version 2.3.2 (2025-07-16)
- Fix argument passing in `Infection.infect`. This will be the final Starsim v2.x release.
- *GitHub info*: PR [1008](https://github.com/starsimhub/starsim/pull/1008)

## Version 2.3.1 (2025-02-25)
- Updated `ss.Sim.shrink()` to remove additional objects, resulting in a smaller sim size.
- `ss.Calibration.save_csv()` has been replaced by `ss.Calibration.to_df()` (to save to a CSV, use `ss.Calibration.to_df().to_csv()`.
- `ss.Result.shape` has been renamed `ss.Result._shape`, so `ss.Result.shape` now correctly returns the actual size of the array.
- Results by default convert all result keys to lowercase; use `keep_case=True` to turn off this behavior.
- Fixed a bug with an [`ss.date`](`starsim.time.date`) object converting to a `pd.Timestamp` upon copy.
- *GitHub info*: PR [865](https://github.com/starsimhub/starsim/pull/865)

## Version 2.3.0 (2025-02-14)
- The calibration class has been completely redesigned. Calibration now relies on "components", which capture mismatch with a particular data type (e.g., new infections). The new approach also adds additional statistical rigor for calculating mismatches.
- `ss.MixingPool` has been updated to be more modular, and behave more like `ss.Network`; in particular, `compute_transmission()` rather than `step()` is called to determine new infections.
- `ss.Result` now has a `summarize_by` argument, which determines how a result should be summarized as a scalar (e.g., mean for a prevalence, sum for a count, last entry for a cumulative count).
- Fixed a bug with time parameters incorrectly pulling the parent unit from the Sim, rather than the parent module.
- *GitHub info*: PR [831](https://github.com/starsimhub/starsim/pull/831)

## Version 2.2.0 (2024-11-18)
- Starsim is now available for R! See <https://r.starsim.org> for details.
- The `Calibration` class has been completely rewritten. See the calibration tutorial for more information.
- A negative binomial distribution is now available as `ss.nbinom()`.
- `ss.Births()` now uses a binomial draw of births per timestep, rather than the expected value.
- Added `ss.load()` and `ss.save()` functions, and removed `ss.Sim.load()`.
- *GitHub info*: PR [778](https://github.com/starsimhub/starsim/pull/778)

## Version 2.1.1 (2024-11-08)
- Adds improved Jupyter support for plotting (to prevent plots from appearing twice); you can disable this by setting `ss.options.set(jupyter=False)`.
- Adds `auto_plot` to `Result` objects, to indicate if it should appear in `sim.plot()` by default.
- Adds `copy()` to the Sim and modules.
- Networks now store their length on each timestep as a result.
- Improves `sim.shrink()`, with typical size reductions of &gt;99%.
- Adds additional plotting options `show_module` (include the module name in the plot title), `show_label` (use the simulation label as the figure title), and `show_skipped` (shows results even if `auto_plot=False`).
- *GitHub info*: PR [745](https://github.com/starsimhub/starsim/pull/745)


## Version 2.1.0 (2024-11-07)

### Summary

- Time in simulations is now handled by an `ss.Time()` class, which unifies how time is represented between the `Sim` and each module.
- In addition to networks, there is now a new way of implementing disease transmission via mixing pools.

### Time

- Time handling now performed by the `ss.Time()` class. This has inputs similar to before (`start`, `stop`, `unit`, `dt`, with `dur` still available as a sim input). However, instead of the previous `timevec` and `abs_tvec` arrays, there are now multiple ways of representing time (including `datevec` and `yearvec`), regardless of what the inputs were.
- Dates are now represented in a native format, `ss.date`, that is based on `pd.Timestamp`.

### Mixing pools

- Adds a new approach to disease transmission called mixing pools. A mixing pool is a "mean field" coupling wherein susceptible agents are exposed to the average infectious agent. The user can create a single mixing pool using the `ss.MixingPool` class, or create many pools using `MixingPools`. Such mixing pools could be used to simulate contact matrices, for example as published by Prem et al.
- There is a new `ss.Route` class, which is the base class for `ss.Network` and `ss.MixingPool`.

### Other changes

- Demographic modules have been updated to fix various bugs around different time units.
- The method for hashing distribution trace strings into seeds has changed, meaning that results will be stochastically different compared to Starsim v2.0.
- Fixed a bug with how timepars were updated in parameters.
- There is a new `ss.Base` class, which both `ss.Sim` and `ss.Module` inherit from.
- Results now print as a single line rather than the full array. The latter is available as `result.disp()`.
- `sim.to_df()` now works even if different modules have different numbers of timepoints.
- The `timepars` module has been renamed to `time`.
- In demographics modules, `units` has been renamed `rate_units`.
- There are two new options, `ss.options.date_sep` and `ss.options.license`. The former sets the date separator (default `.`, e.g. `2024.04.0.4`), and the latter sets if the license prints when Starsim is imported.
- *GitHub info*: PR [724](https://github.com/starsimhub/starsim/pull/724)

## Version 2.0.0 (2024-10-01)

### Summary

Version 2.0 contains several major changes. These include:
module-specific timesteps and time-aware parameters (including a
day/year `unit` flag for modules, and `ss.dur()` and `ss.rate()` classes
for parameters), and changes to module types and integration (e.g.
renaming `update()` and `apply()` methods to `step()`;).

### Time-aware parameters and modules

- Added `ss.dur()`, `ss.rate()`, and `ss.time_prob()` classes, for automatic handling of time units in simulations. There are also convenience classes `ss.days()`, `ss.years()`, `ss.perday()`, `ss.peryear()`, and `ss.beta()` for special cases of these.
- `ss.dur()` and `ss.rate()`, along with modules and the sim itself, have a `unit` parameter which can be `'day'`, `'week'`, `'month'`, or `'year'` (default). Modules now also have their own timestep `dt`. Different units and timesteps can be mixed and matched. Time parameters have a `to()` method, e.g. `ss.dur(1, 'year').to('day')` will return `ss.dur(365, unit='day')`.
- The `ss.Sim` parameter `n_years` has been renamed `dur`; `sim.yearvec` is now `sim.timevec`, which can have units of days (usually starting at 0), dates (e.g. `'2020-01-01'`), or years (e.g. `2020`). `sim.abs_tvec` is the translation of `sim.timevec` as a numeric array starting at 0, using the sim's units (usually `'day'` or `'year'`). For example, if `sim.timevec` is a list of daily dates from `'2022-01-01'` to `'2022-12-31'`, `sim.abs_tvec` will be `np.arange(365)`.
- Each module also has its own `mod.timevec`; this can be different from the sim if it defines its own time unit and/or timestep. `mod.abs_tvec` always starts at 0 and always uses the sim's unit.
- There is a new `Loop` class which handles the integration loop. You can view the integration plan via `sim.loop.to_df()` or `sim.loop.plot()`. You can see how long each part of the sim took with `sim.loop.plot_cpu()`.
- There are more advanced debugging tools. You can run a single sim timestep with `sim.run_one_step()` (which in turn calls multiple functions), and you can run a single function from the integration loop with `sim.loop.run_one_step()`.

### Module changes

- Functionality has been moved from `ss.Plugin` to `ss.Module`, and the former has been removed.
- `ss.Connector` functionality has been moved to `ss.Module`. `ss.Module` objects can be placed anywhere in the list of modules (e.g., in demographics, networks, diseases, interventions), depending on when you want them to execute. However, `ss.Connector` objects are applied after `Disease.step_state()` and before `Network.step()`.
- Many of the module methods have been renamed; in particular, all modules now have a `step()` method, which replaces `update()` (for demographics and networks), `apply()` (for interventions and analyzers), and `make_new_cases()` (for diseases). For both the sim and modules, `initialize()` has been renamed `init()`.
- All modules are treated the same in the integration loop, except for diseases, which have `step_state()` and `step_die()` methods.
- The Starsim module `states.py` has been moved to `arrays.py`, and `network.py` has been moved to `networks.py`.

### State and array changes

- `ss.Arr`, `ss.TimePar`, and `ss.Result` all inherit from the new class `ss.BaseArr`, which provides functionality similar to a NumPy array, except all values are stored in `arr.values` (like a `pd.Series`).
- Whereas before, computations on an `ss.Arr` usually returned a NumPy array, calculations now usually return the same type. To access the NumPy array, use `arr.values`.
- There is a new `ss.State` class, which is a subtype of `ss.BoolArr`. Typically, `ss.State` is used for boolean disease states, such as `infected`, `susceptible`, etc., where you want to automatically generate results (e.g. `n_infected`). You can continue using `ss.BoolArr` for other agent attributes that you don't necessarily want to automatically generate results for, e.g. `ever_vaccinated`.

### Results changes

- Results are now defined differently. They should be defined in `ss.Module.init_results()`, not `ss.Module.init_pre()`. They now take the module name, number of points, and time vector from the parent module. As a result, they are usually initialized via `ss.Module.define_results(res1, res2)` (as opposed to `mod.results += [res1, res2]` previously). `define_results()` automatically adds these properties from the parent module; they can still be defined explicitly if needed however.
- Because results now store their own time information, they can be plotted in a self-contained way. Both `ss.Result` and `ss.Results` objects now have `plot()` and `to_df()` methods.

### Demographics changes

- Fixed a bug in how results were defined for `ss.Births` and `ss.Deaths`.
- The `ss.Pregnancy` module has been significantly rewritten, including: (1) Agents now have a `parent` which indicates the UID of the parent; (2) Women now track `child_uid`; (3) On neonatal death, the pregnancy state of the mother is corrected; (4) Pregnancy rates now adjusted for infecund rather than pregnant; (4) Pregnancy now has a burn-in, which defaults to `True`; (5) Pregnancy has a `p_neonatal_death` parameter to capture fetal and neonatal death if the mother dies.
- Slots now has a minimum, default of 100, to account for small initial population sizes that grow dramatically over time.

### Computational changes

- There have been several performance improvements. The default float type is now `np.float32`. Transmission is now handled by a specialized `Infection.compute_transmission()` method. Several additional functions now use Numba, including `fastmath=True`, which leverages Intel's short vector math library.
- A new `ss.multi_random()` distribution class has been added, that allows random numbers to be generated by two (or more) agents. It largely replaces `ss.combine_rands()` and is 5-10x faster.
- A new `ss.gamma()` distribution has also been added.
- Distributions have a new `jump_dt` method that jumps by much more than a single state update.
- `ss.parallel()` and `ss.MultiSim.run()` now modify simulations in place by default. Instead of `sims = ss.parallel(sim1, sim2).sims; sims[0].plot()`, you can now simply do `ss.parallel(sim1, sim2); sim1.plot()`.

### Other changes

- Data can now be supplied to a simulation; it will be automatically plotted by `sim.plot()`.
- `ss.Calibration` has been significantly reworked, and now includes more flexible parameter setting, plus plotting (`calib.plot_sims()` and `calib.plot_trend()`). It also has a `debug` argument (which runs in serial rather than paralell), which can be helpful for troubleshooting issues.
- `MultiSim` now has display methods `brief()` (minimal), `show()` (moderate), and `disp` (verbose).
- `sim.export_df()` has been renamed `sim.to_df()`.
- Most classes now have `to_json()` methods (which can also export to a dict).
- Fixed a bug in how the `InfectionLog` is added to disease modules.
- `Sim.gitinfo` has been replaced with `Sim.metadata` (which includes git info).
- `Infection.validate_beta()` is now applied on every timestep, so changes to beta during the simulation are now honored.
- `sim.get_intervention()` and `sim.get_analyzer()` have been removed; use built-in `ndict` operations (e.g., the label) to find the object you're after.
- `requires` has been removed from modules, but `ss.check_requires()` is still available if needed. Call it manually from `init_pre()` if desired, e.g. a PMTCT intervention might call `ss.check_requires(self.sim, ['hiv', 'maternalnet'])`.
- For networks, `contacts` has been renamed `edges` except in cases where it refers to an *agent's* contacts. For example, `network.contacts` has been renamed `network.edges`, but `ss.find_contacts()` remains the same.
- Networks now have a `to_graph()` method that exports to NetworkX.
- `ss.diff_sims()` can now handle `MultiSim` objects.
- `Sim._orig_pars` has been removed.
- `ss.unique()` has been removed.

### Regression information

- Note: the list here covers major changes only; in general, Starsim v1.0 scripts will not be compatible with Starsim v2.0.
- Results from Starsim v2.0 will be stochastically (but not statistically) different from Starsim v1.0.
- All duration and rate parameters should now be wrapped with `ss.dur()` and `ss.rate()`. Events that represent probabilities over time (i.e. hazard rates) can also be wrapped with `ss.time_prob()`, although this is similar to `ss.rate()` unless the value is relatively large.
- `ss.Plugin` has been removed. Use `ss.Module` instead.
- `init_results()` is now called by `init_pre()`, and does not need to be called explicitly.
- `default_pars()` has been renamed `define_pars()`.
- `add_states()` has been renamed `define_states()`
- `initialize()` has been renamed `init()`.
- `Demographics.update()` has been renamed `Demographics.step()`.
- `Network.update()` has been renamed `Network.step()`.
- `Disease.update_pre()` has been renamed `Disease.step_state()`.
- `Disease.make_new_cases()` has been renamed `Disease.step()`.
- `Disease.update_death()` has been renamed `Disease.step_die()` (which is now called by `People.step_die()`).
- `Infection._set_cases()` has been renamed `Infection.set_outcomes()`.
- `Intervention.apply(sim)` has been renamed `Intervention.step()`; ditto for `Analyzer`.
- `Module.step()` no longer takes `sim` as an argument (e.g., replace `intervention.apply(sim)` with `intervention.step()`).
- All modules now have methods for `start_step()`, `finish_step()`, `init_results()`, and `update_results()`.
- `Network.contacts` has been renamed `Network.edges`.
- `sim.get_intervention()` and `sim.get_analyzer()` have been removed; simply call directly instead (e.g. replace `sim.get_intervention('vaccine')` with `sim.interventions['vaccine']`).
- `requires` is no longer an attribute of modules; call the `ss.check_requires()` function directly if needed.
- `People.resolve_deaths()` has been renamed `People.check_deaths()`
- `ss.unique()` has been removed.
- *GitHub info*: PR [626](https://github.com/starsimhub/starsim/pull/626)

## Version 1.0.3 (2024-09-26)
- Fixes a bug in which some intervention parameters (e.g. eligibility) do not get set properly.
- *GitHub info*: PR [639](https://github.com/starsimhub/starsim/pull/639)

## Version 1.0.2 (2024-09-25)
- Fixes a bug in which random numbers drawn from auto-jumped distributions would overlap with random numbers drawn from subsequent timesteps.
- *GitHub info*: PR [639](https://github.com/starsimhub/starsim/pull/639)

## Version 1.0.1 (2024-07-22)
- Adds a new distribution, `ss.rand_raw()`, that samples raw integers from the random number bit generator, for use with calculating transmission. This version is roughly 20-30% faster than the previous implementation.
- Adds interpolation to age-standardized fertility rate (ASFR) data.
- Adds flexibility to ART initiation.
- *GitHub info*: PR [593](https://github.com/starsimhub/starsim/pull/593)

## Version 1.0.0 (2024-07-10)
- Official release of Starsim!
- Adds a `Calibration` class, based on [Optuna](https://optuna.org), to facilitate the calibration of Starsim models.
- Adds `mean()`, `median()`, and `plot()` methods to `MultiSim`.
- Adds `low` and `high` attributes to `Result` objects.
- Adds a `flatten()` method to `Results`, allowing nested `Results` objects to be turned into flat dictionaries.
- Removes duplicate UIDs among new infections, and adds a `unique()` method to `ss.uids`.
- Fixes a bug that prevented `ss.lognorm_im()` from using callable parameters.
- Updates the default `Sim` string representation to be a single line; the more verbose version is available via `sim.disp()`.
- *GitHub info*: PR [581](https://github.com/starsimhub/starsim/pull/581)

## Version 0.5.10 (2024-07-03)
- Adds two new common-random-number-safe networks. The first is an Erdős-Rényi network that is similar to `RandomNet` but parameterized differently. The second is a 2D spatial network with connectivity between agents within a given radius; these agents can also optionally move.
- *GitHub info*: PR [575](https://github.com/starsimhub/starsim/pull/575)

## Version 0.5.9 (2024-06-30)
- Added a `ss.histogram()` distribution, which allows generating new random values from an empirical histogram.
- When binned age data is provided to specify the initial ages for new agents, the ages are now distributed throughout the year/bin rather than new agents being assigned integer ages
- Initial age data is now accepted as a `pd.Series` rather than a `pd.DataFrame` where the index corresponds to the age values, thereby avoiding the need for specific dataframe column names to be used to specify the age and value
- *GitHub info*: PR [572](https://github.com/starsimhub/starsim/pull/572)

## Version 0.5.8 (2024-06-30)
- Revert to making infection logging disabled by default. However, the infection log will now always be created so disease subclasses can override logging behaviour where required (e.g., to capture additional metadata)
- **Backwards-compatibility notes:** Logging has been moved from an argument to `Disease` to `pars`. Existing code such as `Disease(log=True)` should be changed to `Disease(pars={'log':True})`. The 'log' option can be added to the pars passed to any subclass e.g., `ss.HIV(pars={...,log=True})`.
- *GitHub info*: PR [573](https://github.com/starsimhub/starsim/pull/573)

## Version 0.5.7 (2024-06-27)
- Implemented a new `ss.combine_rands()` function based on a bitwise-XOR, since the previous modulo-based approach could introduce correlations between pairs of agents.
- *GitHub info*: PR [546](https://github.com/starsimhub/starsim/pull/546)

## Version 0.5.6 (2024-06-22)
- `ss.Infection.make_new_cases()` now returns the index of the network associated with each transmission event
- If a `People` object is provided to the `Arr` constructor, the arrays will be pre-initialized to index the current UIDs in the `People` object. This enables construction of temporary `Arr` instances that can be used to perform intermediate calculations (e.g., inside `Intervention.apply()` or within a module update step)
- Deprecated `Arr(raw=...)` argument to simplify initialization, as in practice the `raw` variable is not directly set, and this update also introduces a new pathway for initializating the <span class="title-ref">raw</span> attribute
- `ss.uids.to_numpy()` now returns a view rather than a copy
- `ss.bernoulli.filter()` now supports `ss.BoolArr` as an input, where the filtering will operate on the `uids` returned by `ss.BoolArr.uids`
- `ss.uids()` supports construction from `set` objects (via `np.fromiter()`)
- *GitHub info*: PR [565](https://github.com/starsimhub/starsim/pull/555)

## Version 0.5.5 (2024-06-19)
- Added labels to `Result` and state (`Arr`) objects.
- Added Numba decorator to `find_contacts` to significantly increase performance.
- Fixed bug when comparing `uids` and `BoolArr` objects.
- *GitHub info*: PR [562](https://github.com/starsimhub/starsim/pull/555)

## Version 0.5.4 (2024-06-18)
- Adjusted `RandomNet` to avoid connections to unborn agents and use random rounding for half edges
- Adds `get_analyzers` and `get_analyzer`
- Refactor how data is pre-processed for births/pregnancy/death rates, giving about a 10% decrease in run time for the STIsim HIV model
- `BoolArr.uids` is automatically called when doing set operations on `uids` with a `BoolArr`
- *GitHub info*: PR [555](https://github.com/starsimhub/starsim/pull/555)

## Version 0.5.3 (2024-06-10)
- `ss.uids` class implements set operators to facilitate combining or otherwise operating on collections of UIDs
- `FloatArr.isnan` and `FloatArr.notnan` return `BoolArr` instances rather than UIDs (so as to facilitate logical operations with other `BoolArr` instances, and to align more closely with <span class="title-ref">np.isnan</span>)
- `Arr.true()` and `Arr.false()` are supported for all `Arr` subclasses
- `BoolArr.isnan` and `Boolarr.notnan` are also implemented (although since `BoolArr` cannot store NaN values, these always return `False` and `True`, respectively)
- *GitHub info*: PR [544](https://github.com/starsimhub/starsim/pull/544)

## Version 0.5.2 (2024-06-04)
- Renames `network.contacts` to `network.edges`.
- For modules (including diseases, networks, etc.), renames `initialize()` to `init_pre()` and `init_vals()` to `init_post()`.
- Renames `ss.delta()` to `ss.constant()`.
- Allows `Arr` objects to be indexed by integer (which are assumed to be UIDs).
- Fixes bug when using callable parameters with `ss.lognorm_ex()` and `ss.lognorm_im()`.
- Fixes bug when initializing `ss.StaticNet()`.
- Updates default birth rate from 0 to 30 (so `demographics=True` is meaningful).
- Adds `min_age` and `max_age` parameters to the `Pregnancy` module (with defaults 15 and 50 years).
- Adds an option for the `sir_vaccine` to be all-or-nothing instead of leaky.
- Updates baseline test from HIV to SIR + SIS.
- Fixes issue with infection log not being populated.
- *GitHub info*: PR [527](https://github.com/starsimhub/starsim/pull/527)

## Version 0.5.1 (2024-05-15)
- Separates maternal transmission into prenatal and postnatal modules.
- *GitHub info*: PR [509](https://github.com/starsimhub/starsim/pull/509)

## Version 0.5.0 (2024-05-14)

### Summary

All inputs to the sim and modules now use a `ss.Pars()` class, which
handles updating and validation. It is now not necessary to ever use
`pars=` (although you still can if you want), so what was previously:

`sim = ss.Sim(pars=dict(diseases='sir', networks='random'))`

is now just:

`sim = ss.Sim(diseases='sir', networks='random')`

Updates happen recursively, so distributions etc. can be flexibly
updated.

This has significantly changed how modules are initialized; what was
previously:

    def __init__(self, pars=None, **kwargs):
    
        pars = ss.omergeleft(pars,
            dur_inf = 6,
            init_prev = 0.01,
            p_death = 0.01,
            beta = 0.5,
        )
    
        par_dists = ss.omergeleft(par_dists,
            dur_inf = ss.lognorm_ex,
            init_prev = ss.bernoulli,
            p_death = ss.bernoulli,
        )
    
        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

is now:

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            beta = 0.5,
            init_prev = ss.bernoulli(0.01),
            dur_inf = ss.lognorm_ex(6),
            p_death = ss.bernoulli(0.01),
        )
        self.update_pars(pars, **kwargs)

### Parameter changes

- Added a `ss.Pars` class (and a `ss.SimPars` subclass) that handles parameter creation, updates, and validation.
- Initialization has been moved from `sim.py` to `parameters.py`; `ss.Sim.convert_plugins()` has been replaced by `ss.SimPars.convert_modules()`.
- The key method is `ss.Pars.update()`, which performs all necessary validation on the parameters being updated.

### Initialization changes

- Previously, the people were initialized first, then the states were initialized and the values populated, then the modules were initialized, and finally the distributions are initialized. This led to circular logic with the states being initialized based on uninitialized distributions. Now, states and modules are *linked* to the `People` and `Sim` objects, but further initialization is not done at this step. This ensures all distributions are created but not yet used. Next, distributions are initialized. Finally, the initial values are populated, and everything is initialized.
- New methods supporting these changes include `ss.link_dists()`, `dist.link_sim()`, `dist.link_module()`, `sim.init_vals()`, `people.init_vals()`, `module.init_vals()`,

### Module changes

- Whereas modules previously initialized a dict of parameters and then called `super().__init__(pars, **kwargs)`, they now call `super().__init__()` first, then `self.default_pars(par1=x, par2=y)`, then finally `self.update_pars(pars, **kwargs)`.
- What was previously e.g. `ss.Module(pars=dict(par=x))` is now `ss.Module(par=x)`.
- `par_dists` has been removed; instead, distributions are specified in the default parameters, and are updated via the `Pars` object.
- Modules now contain a link back to the `Sim` object. This means that all methods that used to have `sim` as an argument now do not, e.g. `self.update()` instead of `self.update(sim)`.
- `ss.module_map()` maps different module types to their location in the sim.
- `ss.find_modules()` finds all available modules (including subclasses) in Starsim.
- Removed `ss.dictmerge()` and `ss.dictmergeleft` (now handled by `ss.Pars.update()`).
- Removed `ss.get_subclasses()` and `ss.all_subclasses()` (now handled by `ss.find_modules()`).
- Modules can no longer be initialized with a `name` key; it must be `type` (e.g. `dict(type='sir')` rather than `dict(name='sir')`.
- Added `to_json()` and `plot()` methods to `Module`.
- Removed `connectors.py`; connectors still exist but as an empty subclass of `Module`.

### People and network changes

- `BasePeople` has been removed and merged with `People`.
- Time parameters (`ti`, `dt`, etc.) have been removed from `People`. Use `sim.ti`, `sim.dt` etc. instead. One consequence of this is that `people.request_death()` now requires a `sim` argument. Another is that network methods (e.g. `add_pairs()`) now take `sim` arguments instead of `people` arguments.
- `SexualNetwork` is now a subclass of `DynamicNetwork`.
- Removed `ss.Networks` (now just an `ss.ndict`).
- Network connectors have been removed.
- `Person` has been implemented as a slice of `sim.people[i]`.
- There is a new parameter `use_aging`; this defaults to `True` if demographic modules are supplied, and `False` otherwise.

### Other changes

- Boolean arrays have new methods `true()`, `false()`, and `split()`, which return the UIDs for the `True` values (alias to `arr.uids`), `False` values, and both sets of values, respectively. `ss.bernoulli.split()` has been added as an alias of `ss.bernoulli.filter(both=True)`.
- All inputs to a sim are now copied by default. To disable, use `ss.Sim(..., copy_inputs=False)`.
- There is a new `Plugin` class, which contains shared logic for Interventions and Analyzers. It has a `from_func()`, which will generate an intervention/analyzer from a function.
- Diseases no longer have a default value of `beta=1` assigned; beta must be defined explicitly if being used.
- Individual diseases can now be plotted via either e.g. `sim.plot('hiv')` or `sim.diseases.hiv.plot()`.
- Distributions can be created from dicts via `ss.make_dist()`.
- A new function `ss.check_sims_match()` will check if the results of two or more simulations match.
- `ndict` values can be accessed through a call; e.g. `sim.diseases()` is equivalent to `sim.diseases.values()`.
- Merged `test_dcp.py` and `test_base.py` into `test_other.py`.
- Renamed `test_simple.py` to `test_sim.py`.
- Renamed `test_dists.py` to `test_randomness.py`.
- *GitHub info*: PR [488](https://github.com/starsimhub/starsim/pull/488)

## Version 0.4.0 (2024-04-24)
- Replace `UIDArray`, `ArrayView`, and `State` with `Arr`, which has different subclasses for different data types (e.g. `FloatArr`, `BoolArr`, and `IndexArr`). States are usually represented by `BoolArr` (e.g. `sir.infected`), while other agent properties are represented by `FloatArr` (e.g. `sir.rel_trans`).
- Arrays that had previously been represented using an integer data type (e.g. `sir.ti_infected`) are now also `FloatArr`, to allow the use of `np.nan`. Integer arrays are supported via `IndexArr`, but these are only intended for use for slots and UIDs.
- `Arr` objects automatically skip over dead (or otherwise removed) agents; the "active" UIDs are stored in `sim.people.auids`, which is updated when agents are born or die. This array is linked to each `Arr`, so that e.g. `sim.people.age.mean()` will only calculate the mean over alive agents. To access the underlying Numpy array, use `sim.people.age.raw`.
- `FloatArr` has `isnan`, `notnan`, and `notnanvals` properties. `BoolArr` has logical operations defined. For example, `~people.female` works, but `~people.ti_dead` does not; `people.ti_dead.notnan` works, but `people.female.notnan` does not.
- UIDs used to be NumPy integer arrays; they are now `ss.uids` objects (which is a class, but is lowercase for consistency with `np.array()`, which it is functionally similar to). Indexing a state by an integer array rather than `ss.uids()` now raises an exception, due to the ambiguity involved. To index the underlying array with an integer array, use `Arr.raw[int_arr]`; to index only the active/alive agents, use `Arr.values[int_arr]`.
- Dead agents are no longer removed, so `uid` always corresponds to the position in the array. This means that no remapping is necessary, which has a significant performance benefit (roughly 2x faster for large numbers of agents).
- Renamed `omerge` to `dictmerge` and `omergeleft` to `dictmergeleft`.
- *GitHub info*: PR [456](https://github.com/starsimhub/starsim/pull/456)

## Version 0.3.4 (2024-04-18)
- Default duration of edges in `ss.RandomNet` changed from 1 to 0; this does not matter if `dt=1`, but does matter with smaller `dt` values.
- Removed `ss.HPVNet`.
- `new_deaths` now counted for cholera.
- Crude birth and death rates now take `dt` into account.
- The ability to use a centralized random number generator has been restored via `ss.options(_centralized=True)`; this option not advised, but can be used for testing.
- *GitHub info*: PR [473](https://github.com/starsimhub/starsim/pull/473)

## Version 0.3.3 (2024-04-16)
- Changed Ebola model transmission logic.
- Fixed bug with module names not being preserved with multiple initialization.
- *GitHub info*: PR [463](https://github.com/starsimhub/starsim/pull/463)

## Version 0.3.2 (2024-04-08)
- Change to syphilis model to permit latent transmission.
- *GitHub info*: PR [450](https://github.com/starsimhub/starsim/pull/450)

## Version 0.3.1 (2024-03-31)
- Added SIS model.
- Fixes distribution initialization.
- Allows interventions and analyzers to be functions.
- Tidies up tests.
- Performance improvements in `UIDArray` (~3x faster for large numbers of agents).
- *GitHub info*: PR [428](https://github.com/amath-idm/stisim/pull/428)

## Version 0.3.0 (2024-03-30)

### New RNGs & distributions

- Replaces `ss.SingleRNG()`, `ss.MultiRNG()`, `ss.ScipyDistribution()`, and `ss.ScipyHistogram()` with a single `ss.Dist()` class. The `starsim.random` and `starsim.distributions` submodules have been removed, and `starsim.dists` has been added.
- The `ss.Dist` class uses `np.random.default_rng()` rather than `scipy.stats` by default, although a `scipy.stats` distribution can be supplied as an alternative. This is up to 4x faster (including, critically, for Bernoulli distributions).
- Also removes `ss.options.multirng` (the new version is equivalent to it being always on).
- Removes duplicate logic for transmission (`make_new_cases()`)
- Adds new custom distributions such as `ss.choice()` and `ss.delta()`.
- These distributions can be called directly, e.g. `dist = ss.weibull(c=2); dist(5)` will return 5 random variates from a Weibull distribution.
- Instead of being manually initialized based on the name, the `Sim` object is parsed and all distributions will be initialized with a unique identifier based on their place in the object (e.g. `sim.diseases.sir.pars.dur_inf`), which is used to set their unique seed.

### Other changes

- This PR also fixes bugs with lognormal parameters, and makes it clear whether the parameters are for the *implicit* normal distribution (`ss.lognorm_im()`, the NumPy/SciPy default, equivalent to `ss.lognorm_mean()` previously) or the "explicit" lognormal distribution (`ss.lognorm_ex()`, equivalent to `ss.lognorm()` previously).
- Renames `ss.dx`, `ss.tx`, `ss.vx` to`ss.Dx`, `ss.Tx`, `ss.Vx`.
- Removed `set_numba_seed()` as a duplicate of `set_seed()`.
- *GitHub info*: PR [392](https://github.com/amath-idm/stisim/pull/392)

## Version 0.2.10 (2024-03-18)
- SIR duration of infection now accounts for dt
- Reworked sir\_vaccine to modify rel\_sus instead of moving agents from susceptible to recovered.
- n\_years no longer necessarily an integer
- *GitHub info*: PR [389](https://github.com/amath-idm/stisim/pull/389)

## Version 0.2.9 (2024-03-18)
- Renames and extends the multirng option in settings, now called 'rng', which set how random numbers are handled in Starsim with three options:
    - "centralized" uses the centralized numpy random number generator for all distributions.
    - "single" uses a separate (SingleRNG) random number generator for each distribution.
    - "multi" uses a separate (MultiRNG) random number generator for each distribution.
- *GitHub info*: PR [349](https://github.com/amath-idm/stisim/pull/349)

## Version 0.2.8 (2024-03-13)
- Add `ss.demo()` to quickly create a default simulation.
- *GitHub info*: PR [380](https://github.com/amath-idm/stisim/pull/380)

## Version 0.2.7 (2024-03-09)
- Update `StaticNet` with defaults and correct argument passing
- *GitHub info*: PR [339](https://github.com/amath-idm/stisim/pull/339)

## Version 0.2.6 (2024-02-29)
- Make random number streams independent for SIR
- *GitHub info*: PR [307](https://github.com/amath-idm/stisim/pull/307)

## Version 0.2.5 (2024-02-29)
- Improve logic for making new cases with multi-RNG
- *GitHub info*: PR [337](https://github.com/amath-idm/stisim/pull/337)

## Version 0.2.4 (2024-02-27)
- Improve `sim.summarize()`
- Improve `sim.plot()`
- Improve SIR model defaults
- *GitHub info*: PR [320](https://github.com/amath-idm/stisim/pull/320)

## Version 0.2.3 (2024-02-26)
- Removes `STI` class
- Changes default death rate from units of per person to per thousand people
- Allows `ss.Sim(demographics=True)` to enable births and deaths
- Fix pickling of `State` objects
- Rename `networks.py` to `network.py`, and fix HIV mortality
- *GitHub info*: PRs [305](https://github.com/amath-idm/stisim/pull/305), [308](https://github.com/amath-idm/stisim/pull/308), [317](https://github.com/amath-idm/stisim/pull/317)

## Version 0.2.2 (2024-02-26)
- Add the `Samples` class
- *GitHub info*: PR [311](https://github.com/amath-idm/stisim/pull/311)

## Version 0.2.1 (2024-02-22)
- Only remove dead agents on certain timesteps
- *GitHub info*: PR [294](https://github.com/amath-idm/stisim/pull/294)

## Version 0.2.0 (2024-02-15)
- Code reorganization, including making `networks.py` and `disease.py` to the top level
- Networks moved from `People` to `Sim`
- Various classes renamed (e.g. `FusedArray` to `UIDArray`, `STI` to `Infection`)
- Better type checking
- Added `MultiSim`
- Added cholera, measles, and Ebola
- Added vaccination
- More flexible inputs
- *GitHub info*: PR [235](https://github.com/amath-idm/stisim/pull/235)

## Version 0.1.8 (2024-01-30)
- Transmission based on number of contacts
- *GitHub info*: PR [220](https://github.com/amath-idm/stisim/pull/220)

## Version 0.1.7 (2024-01-27)
- Performance enhancement for disease transmission, leading to a 10% decrease in runtime.
- *GitHub info*: PR [217](https://github.com/amath-idm/stisim/pull/217)

## Version 0.1.6 (2024-01-23)
- Adds template interventions and products for diagnostics and treatment
- Adds syphilis screening & treatment interventions
- *GitHub info*: PR [210](https://github.com/amath-idm/stisim/pull/210)

## Version 0.1.5 (2024-01-23)
- Renamed `stisim` to `starsim`.
- *GitHub info*: PR [200](https://github.com/amath-idm/stisim/pull/200)

## Version 0.1.4 (2024-01-23)
- Adds a syphilis module
- *GitHub info*: PR [206](https://github.com/amath-idm/stisim/pull/206)

## Version 0.1.3 (2024-01-22)
- Read in age distributions for people initializations
- *GitHub info*: PR [205](https://github.com/amath-idm/stisim/pull/205)

## Version 0.1.2 (2024-01-19)
- Functionality for converting birth & fertility data to a callable parameter within SciPy distributions
- *GitHub info*: PR [203](https://github.com/amath-idm/stisim/pull/203)

## Version 0.1.1 (2024-01-12)
- Improving performance of MultiRNG
- Now factoring the timestep, `dt`, into transmission calculations
- *GitHub info*: PRs [204](https://github.com/amath-idm/stisim/pull/204)

## Version 0.1.0 (2023-12-10)
- Allows SciPy distributions to be used as parameters
- Optionally use multiple random number streams and other tricks to maintain coherence between simulations
- Adding functionality to convert death rate data to a callable parameter within a SciPy distribution
- *GitHub info*: PRs [170](https://github.com/amath-idm/stisim/pull/170) and [202](https://github.com/amath-idm/stisim/pull/202)

## Version 0.0.8 (2023-10-04)
- Enable removing people from simulations following death
- *GitHub info*: PR [121](https://github.com/amath-idm/stisim/pull/121)

## Version 0.0.7 (2023-09-08)
- Refactor distributions to use new Distribution class
- *GitHub info*: PR [112](https://github.com/amath-idm/stisim/pull/112)

## Version 0.0.6 (2023-08-30)
- Changes agent IDs from index-based to UID-based
- Allows states to store their own data and live within modules
- *GitHub info*: PR [88](https://github.com/amath-idm/stisim/pull/88)

## Version 0.0.5 (2023-08-29)
- Refactor file structure
- *GitHub info*: PRs [77](https://github.com/amath-idm/stisim/pull/77) and [86](https://github.com/amath-idm/stisim/pull/86)

## Version 0.0.2 (2023-06-29)
- Adds in basic Starsim functionality
- *GitHub info*: PR [17](https://github.com/amath-idm/stisim/pull/17)

## Version 0.0.1 (2023-06-22)
- Initial version.
