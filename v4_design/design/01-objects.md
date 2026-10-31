# Objects and construction

The class hierarchy, how models are constructed, how things get their names, and the rule that keeps the short form and the long form the same language.

Justified by [best-practices/principles.md](../best-practices/principles.md) and [best-practices/model-structure.md](../best-practices/model-structure.md).

## The hierarchy

```
ss.Sim                      modules, a population, a timeline, and a solver; one run
ss.MultiSim                 many runs of a sim, or several sims; made by ss.parallel
ss.People                   the population: size, demography, per-agent attributes
ss.Module                   lifecycle: init → step → finalize; owns states, transitions, parameters, results
 ├── ss.Disease             any disease: states that people are in and the transitions between them (declared), code, or both
 │    └── ss.Infection      adds transmission written as code (beta over networks, rel_sus, rel_trans), as in Starsim 3
 ├── ss.Network             who contacts whom: ss.RandomNet, ss.SexualNet, ...
 ├── ss.MixingPool          group-level mixing; also ss.MixingPools, which ss.contacts makes
 ├── ss.Demographics        births, deaths, pregnancy, migration
 ├── ss.Product             ss.Vaccine, ss.Diagnostic, ss.Treatment (08-interventions.md)
 ├── ss.Intervention        deliveries: ss.vaccinate, ss.screen, ss.treat (08-interventions.md)
 ├── ss.Connector           cross-module code in step(); ss.modify and ss.change make one
 └── ss.Analyzer            extra results
ss.State                    something a person can be in, or a count of something that is not people
ss.Transition               a movement between states; built by >> and @, or by
                            ss.transmission, ss.progression, ss.birth, ss.death, ss.transfer
ss.par                      a named parameter with bounds, prior, and provenance
ss.dur, ss.per, ss.prob      time types (base classes ss.TimePar, ss.Rate); 02-types.md
ss.geo.Mobility                 movement between the places of a stratification (06-routes.md)
ss.Dist                     distributions: ss.normal, ss.lognorm_ex, ...
ss.Result                   a named time series, with a dimension per stratification and, for many runs, a run axis
```

`ss.Module` is the only module abstraction. Diseases, networks, demographics, interventions, connectors, and analyzers differ in what they contain, not in how they are initialized or scheduled. This is Starsim's design, kept unchanged; it is what makes multi-disease models and non-disease health states composable. Existing Starsim modules, written as subclasses with `step()`, remain valid modules.

## Naming conventions

**Modules, containers, and base classes are CamelCase; the concrete pieces you put inside a module are lowercase.** This is Starsim's existing rule (`ss.Dist` versus `ss.normal`, `ss.Rate` versus `ss.perday`, `ss.RandomNet` versus `ss.bernoulli`), applied to the new names. Products (`ss.Vaccine`, `ss.Diagnostic`, `ss.Treatment`) are CamelCase because they are modules, as Starsim's `ss.Product` is; the deliveries that hand them out (`ss.vaccinate`, `ss.screen`, `ss.treat`) are pieces and are lowercase. No product and delivery differ only by case.

| | Examples |
|---|---|
| CamelCase | `ss.Sim`, `ss.MultiSim`, `ss.People`, `ss.Module`, `ss.Disease`, `ss.Network`, `ss.RandomNet`, `ss.MixingPool`, `ss.geo.Mobility`, `ss.Intervention`, `ss.Connector`, `ss.Analyzer`, `ss.fit.Fit`; the two graph classes `ss.State` and `ss.Transition`; the products `ss.Vaccine`, `ss.Diagnostic`, `ss.Treatment` |
| lowercase | time (`ss.days`, `ss.perday`), distributions (`ss.normal`, `ss.lognorm_ex`), `ss.par`, `ss.states`, `ss.null`, `ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`, `ss.transfer`, `ss.modify`, `ss.outcome`, `ss.by`, `ss.timeseries`, `ss.seasonal`; deliveries `ss.vaccinate`, `ss.screen`, `ss.treat`, `ss.change`; loaders `ss.people`, `ss.contacts`, `ss.data`; functions `ss.parallel`, `ss.fit.compare_models` |

Methods that describe the model are `describe()`, `diagram()`, and `equations()`. Methods about results keep their Starsim names: `plot()`, `summarize()`, `to_df()`.

## Constructing a model

There are two ways to construct a module, and inside either one, three ways to spell a transition ([03-states-and-transitions.md](03-states-and-transitions.md)).

### The constructor form

```python
ss.Disease(
    *transitions,            # positional: transitions and outcomes, named from their variable names
    transitions = None,      # the same things as lists, for longer models: transitions,
    results     = None,      #   extra results (ratios and sums of states),
    outcomes    = None,      #   and ss.outcome objects; each named from its variable
    name = None,             # the module's name
    dt   = None,             # this module's timestep; defaults to the sim's
    init_prev = None,        # initial prevalence (03-states-and-transitions.md)
    init = None,             # the whole starting distribution: dict(S=0.90, I=0.01, R=0.09)
    **members,               # one keyword per state, transition, parameter, result, or function
)
```

The operator form declares the ingredients first and then passes the transitions positionally (or as keywords, which name them):

```python
beta = 0.3
dur  = ss.days(10)

S, I, R   = ss.states(3)
infection = S >> I @ beta * I
recovery  = I >> R @ dur
sir = ss.Disease(infection, recovery, init_prev=0.01)
```

A two-transition model can take its results and outcomes the same way, positionally or as keywords. A longer one keeps the three kinds of thing apart with the grouped keywords, and the names still come from the variables:

```python
prevalence = I / (S + I + R)
cases = ss.outcome(0.5 * infection, delay=ss.days(5))
hosp  = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))

sir = ss.Disease(
    transitions = [infection, recovery],
    results     = [prevalence],
    outcomes    = [cases, hosp],
    init_prev   = 0.01,
)
```

The grouped keywords and the short forms build the same module and may be mixed. `transitions`, `results`, and `outcomes` are therefore reserved and cannot be used as member names (a `ValueError`).

What a keyword's value is decides what it becomes:

| Value | Becomes |
|---|---|
| an `ss.State` | a state |
| an `ss.Transition` | a transition |
| an `ss.par`, number, typed quantity, or distribution | a parameter in `sir.pars` |
| an `ss.outcome`, or a sum or ratio of states such as `S + I + R` or `I / N` | a result ([09-results.md](09-results.md)). Arithmetic on states never makes a new state |
| a plain function | custom behavior, called every step ([03](03-states-and-transitions.md#custom-per-step-behavior)) |

### Names

The keyword is the name. When there is no keyword, a transition's name is looked up in this order:

1. the keyword it was passed under (`ss.Disease(infection=...)`);
2. its own `name=` argument (`ss.Transition(S, I, rate=beta * I, name='infection')`);
3. the name of the variable it is bound to in the code that called the constructor (`infection = S >> I @ beta * I` then `ss.Disease(infection, recovery)`);
4. a fallback built from its endpoints: `S_to_I`.

Rule 3 is found by looking for the object among the caller's local and global variables, once, when `ss.Disease(...)` (or `ss.Sim(...)`, for modules) is called; nothing else looks, and a string does not name a variable. It is a convenience and it can miss: a transition written inline (`ss.Disease(S >> I @ beta * I)`) or built in a loop has no variable, so it gets the fallback name, and `describe()` shows the names that were used. If two fallback names collide, the later ones get a numeric suffix (`S_to_I_2`). An object bound to two variables takes the fallback name too.

States and `ss.par` objects are named by the same lookup, with one difference for states: there is no fallback. A state from `S, I, R = ss.states(3)` or `S = ss.State()` has no name until the module is built, and gets it from rules 1 to 3; one that cannot be matched to a keyword, a `name=`, or exactly one variable raises a `ValueError`, which says to use `ss.states('S I R')` or `ss.State('S')`. A state from `ss.states('S I R')`, or declared in a class body, already has its name. A module's name (for a plain `ss.Disease`, a delivery, a modifier, or a product) is its `name=`, else the variable it was assigned to when it was made (`campaign = ss.vaccinate(...)`, read from the instruction that stores it, so a loop variable or `sim.new()` that later holds it does not change it), else the variable it is bound to where it is passed to `ss.Sim` (unless that is a name the sim already uses, as `interventions` is in `def run(interventions): ss.Sim(interventions=interventions)`), else the lowercased class name as in Starsim; this is why `sir = ss.Disease(...)` gives `sim.diseases.sir` and `sim.results.sir`. A product that a delivery makes from its keywords (`ss.vaccinate(efficacy=0.9, ...)`) is named for its kind (`vaccine`) if that name is free, else for its delivery (`campaign_product`), without a warning. A product with a default name that its delivery took from their shared variable (`bcg = ss.vaccinate(BCG(), ...)`, where `BCG` is a subclass of `ss.Vaccine`) is named for the delivery and its kind (`bcg_vaccine`).

The name is then used everywhere: `sir.infection`, the results, `ss.modify` and `ss.change` targets, random streams, and messages. Names are Python identifiers and are unique within a module. Two members given the same explicit name raise a `ValueError`, and a name that collides with an attribute of `ss.Module` (`step`, `init_pre`, `results`, `pars`, `describe`, `stratify`, `name`, `dt`, and so on) raises a `ValueError`.

### The class form

Class attributes that are states, transitions, parameters, or outcomes are collected in definition order and mean the same as the same keywords passed to the constructor. Subclasses inherit them, may replace one by rebinding its name, and may add behavior by defining methods; rebinding a declared parameter to something that is not one (a method, or a number) raises `CapabilityError`, since the transitions read the base class's parameter (a function as its value is given in `__init__()`, as any value is). A member that is an expression with a value for each person (it reads `ss.ppl`, a clock, an attribute, or a parameter by level), e.g. `age = ss.ppl.age` or `duration = affected.time`, is a helper used through the objects, not a result, and a parameter under another name (`A = saturation_age`) is the same parameter. A result of a parameter is an expression of it under a name of its own (`eir_now = 1*eir`); a parameter given by keyword under its own name and read by a transition, as if it were a result, raises a `ValueError`.

```python
class SEIR(ss.Disease):
    beta = ss.par(0.05, bounds=[0.01, 0.5])

    S = ss.State()                           # in a class body a state learns its own name
    E = ss.State()
    I = ss.State()
    R = ss.State()
    infection  = S >> E @ beta * I
    latency    = E >> I @ ss.days(3)
    recovery   = I >> R @ ss.days(10)

    def step(self):
        super().step()                       # the declared transitions
        self.count_something_bespoke()       # anything else
```

Use the class form when there is behavior that is not a transition, or when the model is going into a library. A module that only has states, transitions, parameters, and results never needs to define a method.

### Declared in `__init__()`

A model whose structure depends on its arguments (a state or transition that only some versions have, or transitions made in a loop, one for each strain or drug) cannot be written in a class body, which runs once for the class. Its `__init__()` declares the states, transitions, and results on `self`, with `define_states()`, `define_transitions()`, and `define_results()`, which take them positionally (named by `name=` or by their variable), as lists, or by keyword:

```python
class TB(ss.Disease):
    def __init__(self, relapse=False, **kwargs):
        super().__init__(init=dict(infection=0.05), **kwargs)
        p = ss.pars(tb_pars())
        self.define_states(
            ss.State('susceptible', label='Susceptible'),
            ss.State('infection',   label='Latent infection'),
            ss.State('active',      label='Active TB'),
            ss.State('recovered',   label='Recovered'),
        )
        foi = p.beta*self.active
        self.define_transitions(
            transmission = self.susceptible >> self.infection @ foi,
            progression  = self.infection >> self.active @ p.progression,
            recovery     = self.active >> self.recovered @ p.recovery,
        )
        if relapse: # Only some versions of the model have it
            self.define_states(ss.State('relapsing', label='Will relapse'))
            self.define_transitions(...)
        self.define_results(prevalence=self.active.frac)
```

Each is an attribute at once (`self.active`), for the declarations that follow. They are the same members as the same keywords given to `ss.Disease` and build the same model; `overwrite=True` replaces one of the same name (e.g. a transition whose rate is changed for each strain). They are given while the module is made, and built together when it is first used (by the sim, or by `tb.transitions`, `describe()`, ...); declaring more after that raises `CapabilityError`. `define_states()` also still defines the arrays of a module written as code (`ss.BoolState`), and `define_results()` its `ss.Result` objects: what is given says which.

`ss.Disease` is the base of every disease, and whether each part of it is declared or written as code is a question about that part, not about the class. A disease may declare states and transitions (in the operator or the class form), which its engine runs under any solver; it may instead define per-agent arrays (`define_states()`) and change them in its own `step_state()`, `step()`, and other methods, as in Starsim 3; or it may mix the two in one class, e.g. a declared natural history and a coded `ss.FloatArr` that its `step()` updates after `super().step()` has taken the declared transitions. The seam is the engine: each lifecycle method of `ss.Disease` runs its engine's part, and a disease that declares nothing has an engine with nothing to run, so a coded override calls `super()` as for any module, and no method asks which kind of disease it is. That engine still gives the disease's code what a declared engine gives its own as agents, by the same code (`ss.solvers.Transmission`): the effects on its `rel_sus` and `rel_trans` (`dis.current_value()`), transmission over a network with who infected whom, and a delivery's conditions and cure; where the two differ (a beta as a probability in the step rather than a rate), the caller says so; strains are passed on, and protection against them drawn, the same way for both. A disease with no declared states and no transmission (malnutrition, contraception as a behavior, Starsim 3's `class X(ss.Disease)` with its own states and `step()`) is a plain `ss.Disease`, with no `susceptible` or `infected` states forced on it, and `describe()` prints its class, arrays, and parameters. `ss.Infection(ss.Disease)` adds transmission written as code and nothing else: the `susceptible` and `infected` states, `rel_sus`, `rel_trans`, `ti_infected`, `init_prev`, transmission over the networks for its `beta` (`transmit()`, `transmit_route()`), `set_prognoses()` for those infected, `infect(uids)` to infect some agents now from code (their outcomes set as transmission sets them, with no source, as for a declared disease), and the standard `prevalence` and `new_infections` results; with no `beta` it is not transmitted (as `ssl.NCD`). `isinstance(x, ss.Disease)` is true of both, and both are diseases to the sim (`sim.diseases`, the infection log, deaths, `sim.rescale()`). What cannot work is refused: a class that declares states or transitions and also defines one of `ss.Infection`'s transmission methods (`set_prognoses()`, `transmit()`, ...) without subclassing `ss.Infection` raises `ss.CapabilityError`, since the declared engine never calls it. What needs structure (`describe()` in full, `equations()`, `diagram()`, `R0()`, `ss.move()`, `current_state()`, `stratify()`) works on the declared part. A disease with nothing declared runs as agents if it keeps a value for each person (`define_states()`, as Starsim 3's do), and otherwise beside the compartments, as any module written as code does; a declared one with code of its own follows the try-it rule of [12-backends.md](12-backends.md) (`compartmental = False` if its code needs agents). Starsim 3's `ss.Disease` and `ss.Infection` keep their meaning, so `python -m starsim.migrate` does not rewrite either. The verbs that act on arrays reach a disease written as code (`ss.modify` of an `ss.Infection`'s `rel_sus` and `rel_trans`, conditions on its arrays, fitting its parameters given as `ss.par` with bounds, `sim.new()`).

Code can also read and write the states of a declared disease run as agents, as Starsim 3 code does: indexing (`sir.I[uids] = True`), `set()`, and in-place operators (`sir.I |= ...`) are tracked, so the counts (`sir.I.n`), the results, and other modules later in the same step see the change. Writes through the NumPy array behind a state (`sir.I.raw`) are not tracked, and the counts and results do not see them; `STARSIM_CHECK_COUNTS=1` checks the counts at every step. The time each agent entered a state (`ti_X`) is kept only where the engine needs it, for a state declared `ss.State(keep_time=True)` (for code that reads it, e.g. the years since onset), or for every state if `ss.options.keep_times = True`; reading another raises an `AttributeError`. A disease written as code takes declared products and deliveries that act on the states it has (e.g. `ss.vaccinate()` on `rel_sus`).

### Library models

Starsim's `dict(type='sir', beta=0.1)` and `ss.SIR(beta=0.1)` are kept as the way to ask for a model from the library. A library model is an ordinary module: `ss.SIR`, `ss.SIS`, and `ss.SEIR` are declared in the class form, with Starsim 3's keywords, so `describe()` prints them in full and `equations()` works on them. Starsim 3's versions written as code are `ssl.AgentSIR`, `ssl.AgentSIS`, and `ssl.AgentSEIR`.

The dict *tuple* shorthand from the first draft, `dict(infection=('S -> I', 0.05))`, is not part of the language. Guessing the kind of transition from the type of a bare value was ambiguous (it misclassifies SIRS, for example), and the operator form is as short.

### The equivalence rule

> **Anything that can be written in one form can be written in the others, with the same keywords meaning the same thing. Subclassing adds behavior; it does not add vocabulary.**

Concretely:

- The three transition spellings (arrow-string constructors, operators, explicit `ss.Transition`) build the same objects. A keyword accepted by a shorthand constructor (`dur=`, `rate=`, `p=`, `stages=`) is accepted by `ss.Transition` and means the same thing.
- The constructor form and the class form build the same module. A feature MUST NOT be reachable only as a method that subclasses can call if it could be a keyword, and MUST NOT require the class form unless it is imperative code.
- For every model in the test corpus, the forms MUST give the same results, once names agree. [18-conformance.md](18-conformance.md).

## `ss.Sim`

```python
ss.Sim(
    modules,                     # a module, transitions, or a list of them; diseases=, demographics=, etc. also work
    # --- who ---
    networks  = None,            # networks and mixing pools alike; none means homogeneous mixing
    people    = None,            # an ss.People, e.g. ss.people('Kenya', year=2024)
    stratify  = None,            # people dimensions for every module: dict(age=[0, 15, 65]) (05-dimensions.md)
    total_pop = None,            # size of the population represented
    n_agents  = None,            # number of simulated agents; giving it makes the run agent-based
    # --- when ---
    start     = None,            # a date, a year, or 0
    stop      = None,            # none: 50 units (or steps) after the start
    dt        = None,            # default timestep; modules may have their own
    unit      = None,            # unit for bare numbers; none anywhere means time is in timesteps (02-types.md)
    # --- how ---
    solver    = 'auto',          # 'auto' | 'ode' | 'binomial' | 'gillespie' | 'abm'
    approx    = None,            # None notes each approximation; True keeps the notes without printing them; False makes them errors
    rand_seed = 1,
    verbose   = 1,               # 0 prints nothing; 1 what is run and how, the notes, progress, and how the run ended; 2 everything
    # --- what else ---
    interventions = None,
    connectors    = None,
    analyzers     = None,
    data      = None,            # a file, a dataframe, or ss.simdata(...), with columns named like results (09-results.md)
    pars      = None,            # parameter values by name (04-parameters.md)
    results   = None,            # the results to keep by group, e.g. ['I', 'cases']; the totals are always kept (09-results.md)
    label     = None,
    **kwargs,                    # less common options, e.g. integrator (12-backends.md)
)
```

Only the model is required; every other argument has a default that is guessed and printed. The keywords are Starsim's (`networks`, `n_agents`, `total_pop`, `rand_seed`, `data`), with `unit`, `solver`, `approx`, `stratify`, and `integrator` added; `results` now names the results kept by group. One difference from Starsim today: the first positional argument is the model rather than a `pars` dict. A dict in that position is still read as Starsim reads it.

Transitions given to the sim (`ss.Sim(infection, recovery)`, `ss.Sim([infection, recovery])`, or `ss.Sim(diseases=[infection, recovery])`) are one disease, as `ss.Disease(infection, recovery)` is, named from its states (`sir`). The results of that disease, or of one the sim names from its states (`ss.Sim(ss.Disease(...))`), are also at the top of the sim's results (`sim.results.I` is `sim.results.sir.I`), if there is one such disease. Each module is also an attribute of the sim (`sim.sir` is `sim.diseases.sir`), unless the sim has an attribute of that name, which warns.

`networks=` accepts networks and mixing pools alike. There is no `ss.Homogeneous`: homogeneous mixing is what happens when no network is given, and `ss.MixingPool(n_contacts=10)` sets a contact rate. See [06-routes.md](06-routes.md).

### `total_pop` and `n_agents`

- **`total_pop` is the size of the population being represented.** It is the total of the compartments in an ODE, and the population the agents stand for in an agent-based run.
- **`n_agents` is how many agents are simulated.** It matters only in an agent-based run, and is a computational choice like `dt`.

Defaults, all printed by `describe()`:

- Neither given: `total_pop` is 10,000 (Starsim's default population).
- Only `n_agents`: `total_pop = n_agents`, as in Starsim.
- Only `total_pop`, in an agent-based run: `n_agents` is `total_pop` or 10,000, whichever is smaller.
- When `n_agents < total_pop`, results are scaled by `total_pop / n_agents` and `describe()` prints the scale factor.

`total_pop` counts people. States that are not people (a reservoir, a vector population; [03-states-and-transitions.md](03-states-and-transitions.md#states-that-are-not-people)) are not part of it and are never agents.

### `solver`

`solver='auto'` picks the simplest, fastest representation that runs the model faithfully: if every piece has a compartmental form the model runs as a deterministic ODE, otherwise as agents. The choice and the reason are printed. A model becomes agent-based by containing something agent-based (an individual-level network, `n_agents=`, a per-agent function or attribute, a module written as code that keeps a value for each person) or by asking (`solver='abm'`); being written as code is not itself a reason, so a sim of modules written as code with nothing per person has no agents. `'auto'` never approximates the mixing and never drops individual-level structure. The one thing it does unasked is represent a non-exponential dwell time by Erlang sub-stages, so a lognormal duration does not by itself force an agent-based run, and it says so. When a solver is asked for explicitly and the model does not natively fit it, the library uses the standard approximation where there is one and says so in one line.

The names are `'ode'`, `'binomial'` (chain-binomial), `'gillespie'` (the exact continuous-time stochastic solver, which stays exact however many events a run takes), `'tau'` (tau-leaping: close to exact, and much faster for a run of many events), and `'abm'`. Any other name raises an error that lists them. `solver=` may also be given to `sim.run()`, before the sim is initialized; once it is, the solver is fixed, and `sim.new(solver=...)` makes a fresh copy with another. `sim.solvers()` (and `sir.solvers()` for one module) prints which solvers can run the model, before anything runs. See [12-backends.md](12-backends.md).

`approx=` controls what happens when something is approximated: by default a one-line note is printed (and kept in `sim.notes`), `approx=True` says the approximation is intended and keeps the note without printing it, and `approx=False` makes any approximation an error (`CapabilityError`); under `solver='auto'` it means a model with a non-Erlang dwell time runs as agents instead ([12-backends.md](12-backends.md#approximations)).

`integrator=` is the scheme of the ODE, `'rk4'` (the default) or `'euler'` (for matching frameworks that take Euler steps); it is specified in [12-backends.md](12-backends.md). JAX is not chosen by it: `sim.gradient()`, `sim.log_density()`, gradient fits, and `ss.parallel()` with many sets of values use JAX by themselves.

### One sim is one run

A sim is one trajectory, whatever the solver. Several runs are a MultiSim, as in Starsim:

```python
sim  = ss.Sim(sir, solver='gillespie', total_pop=1000)
msim = ss.parallel(sim, n_runs=100)            # or its shortcut, sim.run(n_runs=100)

msim.plot()                                    # the median and an interval
msim.plot(lines=True)                          # every run
msim.results.sir.I.quantiles([0.025, 0.5, 0.975])
msim.results.sir.I.values                    # the full array, runs by time
```

There is no replicate count on `ss.Sim`: `sim.run(n_runs=100)` is a shortcut for `ss.parallel(sim, n_runs=100)`, and returns the MultiSim, not the sim. `ss.parallel(baseline=..., vaccine=...)` runs several named sims and returns one object indexed by scenario, run, and time. Results are specified in [09-results.md](09-results.md).

### Methods of a sim

| Method | Does |
|---|---|
| `sim.init()` | builds the model: resolves defaults, checks, picks the unit and the solver ([00-overview.md](00-overview.md#what-happens-at-build-time)) |
| `sim.run(solver=None, n_runs=None)` | runs it, calling `init()` first if needed (and again from the start, if it has run); returns the sim, or with `n_runs=` a MultiSim of that many runs |
| `sim.new(**kwargs)` | a fresh, uninitialized copy of the sim as it was before it was initialized, with any keywords changed (`sim.new(solver='abm')`); setting `sim.pars` on an initialized sim raises, pointing here, and other changes made to it belong to that run only (its data and `jax` are kept, as are values set by `restart()`) |
| `sim.describe()` | prints the resolved model, with every guess marked ([below](#describe)) |
| `sim.stratify(age=..., ...)` | stratifies every module by a property of people; the same as `stratify=` ([05-dimensions.md](05-dimensions.md)) |
| `sim.what_modifies(x)` | lists what changes a quantity ([below](#the-module-contract)) |
| `sim.plot()`, `sim.summarize()`, `sim.to_df()` | results, as in Starsim ([09-results.md](09-results.md)) |
| `sim.fit()` | fits parameters that have bounds to results that have data ([13-inference.md](13-inference.md)) |

Age, sex, and place are properties of people, and every module in a sim has to agree on them, so they are stratified once, on the sim. A dimension that belongs to one disease (strain, drug resistance) is stratified on that disease with `sir.stratify(...)`. The sim works on its own copies of the modules it is given, as in Starsim, so one `sir` can be passed to several sims.

## Lifecycle

Every module has the same lifecycle, and it is Starsim's:

| When | Method | What happens |
|---|---|---|
| Build (`sim.init()`) | `init_pre(sim)`, `init_results()`, `init_post()` | resolve defaults, allocate state and results, set initial conditions |
| Each step | `start_step()` | start the step (e.g. move the random number streams on) |
| Each step, diseases written as code | `step_state()` | progress people between states, before networks and interventions |
| Each step | `step()` | read the current state and change it; a declared disease takes all of its transitions here |
| Each step, diseases | `step_die(uids)` | take out the people who died this step |
| Each step | `update_results()` | record results, at the end of the step |
| Each step | `finish_step()` | tidy up |
| End | `finalize()` | finalize results |

A module that only has states, transitions, parameters, and results does not define any of these.

`step()` reads and writes the sim's current state directly. Modules run in a defined order each step (Starsim's: start of step, custom modules, demographics, disease state updates, connectors, networks, interventions, diseases (transmission, and every transition of a declared disease), deaths, the move of time to the end of the step, results, analyzers, end of step), modules of one type run in the order given, and the order is a data object, `sim.loop`, that can be printed, plotted, and inserted into. Step `k` runs from time point `k-1` to `k`, and throughout it, in every module, `sim.ti` is `k`, where its results go (`self.results.x[self.ti]`), and `sim.now` is its start. There are no queued updates and no named phases: a module sees whatever the modules before it did this step, so infection, testing, and treatment can all happen within one step if ordered that way. [10-execution.md](10-execution.md) specifies the loop.

## The module contract

A module is not required to declare what it reads or writes. Declarations of that kind get forgotten and drift from the code, so the library observes instead:

```python
sim.what_modifies(hiv.rel_trans)     # the ss.modify() modifiers on it
```

Modifiers made with `ss.modify` are always listed, because they are declared; direct writes by code are not tracked. Modules may read and write other modules' state directly. Scaling a quantity through `ss.modify` rather than overwriting it SHOULD be preferred, because modifiers combine by a stated rule and the base value is never lost ([07-effects.md](07-effects.md)); this is guidance, not something the library enforces.

## `describe()`

`describe()` prints the resolved model as text. It is what makes guessing defaults safe: every guess is shown, and marked as a guess. It replaces the first draft's model-level `summary()`; `sim.summarize()` is Starsim's results summary and stays about results.

Every module implements `describe()`, and `sim.describe()` covers the whole sim. For a model it prints:

1. every state, with its properties, and which states are counts and not people;
2. every transition, with its name, its kind, and its rate or dwell-time distribution as resolved (`time in I ~ exponential(mean = 10 days)`, not `dur=10`; a distribution as it was given, `erlang(k=4, mean=8 days)`, not by its internal parameters), its source of infection where there is one, and its rate with bare states expanded (`0.3 × I.n / N` under homogeneous mixing);
3. every parameter that is documented or can be fitted, with its value and its bounds (`beta = 1.9 per day, between 1 and 3`); there is no separate table of parameters;
4. every outcome, with its definition (`cases   outcome: 0.5 × onset, after 4 days (exponential)`);
5. every stratification, with its levels, the share of people in each (`split: 0-14 0.3, 15-64 0.6, 65+ 0.1`), and its adjustments (`adjust: rel_sus × 1, 0.03`);
6. the mixing: each network or pool, and how a state in a rate is read through it (homogeneous mixing is frequency-dependent: `I` is `I.n / N`);
7. every modifier, with its target and how it combines;
8. the solver and why it was chosen, and anything that was approximated;
9. which parts restrict the choice of solver (custom functions, per-agent attributes);
10. the notes given at `init()`;
11. the sim's unit and where it came from, and the loop as a schedule: one line per group of modules in the order they run, with how often each runs (`every day: sir, random network; every year: births, deaths`). The fully unrolled list stays available as `sim.loop` ([10-execution.md](10-execution.md)).

How the model is run, which is about the solver rather than the model (`plan: sir: as compiled code, the whole run in one call`), and each rate and dwell time per step are shown only with `describe(verbose=True)`.

Under a solver that takes whole steps (agents and the chain binomial), a model with transmission also gets two `timing` lines: an infection is dated at the end of its step, and a state's count includes the step people enter it ([02-types.md](02-types.md)). The solvers that follow the rates within a step (the ODE, Gillespie, tau-leaping) do not show them.

Everything guessed rather than given is marked `[guessed]`, with the rule that produced it. The description is in sections (the sim, each model, mixing and interventions, results and running, and the notes): in color in a terminal, as tables in a notebook, and as plain text otherwise (and from `describe(output=True)`), with each guess at the end of its line.

```text
>>> sim = ss.Sim(sir, stop=100)
>>> sim.describe()
Sim: 1 module, 10,000 people [guessed], steps 0 to 100
  solver       ode                                             [guessed: every part of the model has a compartmental form]
  unit         none: time is in timesteps                      [nothing in the model has a unit]

Disease 'sir': 3 states, 2 transitions
  states       S, I, R
               I infectious
               init: init_prev = 0.01; the rest in S

  infection    S -> I   transmission, from I
               beta = 0.3 per step
               rate = 0.3 × I.n / N                            [homogeneous mixing]
  recovery     I -> R   spontaneous
               time in I ~ exponential(mean = 10 steps)          [guessed: scalar duration]

Mixing and interventions
  mixing       homogeneous, 1 contact per step                 [guessed: no network given]

Results and running
  results      S, I, R, and one flow per transition
  schedule     every step: sir
```

A run says, as it starts, what is run and how, in one line (`Sim: sir, 10,000 people, as an ODE, steps 0 to 100`), with R0 under it for a compartmental run and then the notes; as it goes, a bar redrawn in place (in a terminal or a notebook; elsewhere, a line every tenth of the run); and as it ends, how long it took, with each disease's infections and their peak (`Done in 0.03 s; sir: 9,309 infections; at most 3,036 infectious, on 27`). A MultiSim says how many runs of which scenarios it ran, and its runs say nothing, unless `verbose=` is given.

Give the sim a unit (`unit='day'`, a typed `dt`, or dates) and the same lines read `0.3 per day` and `mean = 10 days`, with the source of the unit shown ([02-types.md](02-types.md#bare-numbers)). `sim.describe(output=True)` returns the text instead of printing it.

## Equality, copying, and identity

- Two models written in different forms are the same model if they give the same results; there is no hash or comparison function ([14-model-spec.md](14-model-spec.md)), and `==` on two modules is Python's identity.
- `copy()` is deep for the model and shallow for data references, so a copied model that fetched `ss.people('Kenya')` does not fetch it again.
- `stratify()` changes the model in place and returns it, so it can be chained or passed straight to `ss.Sim`. `sir.stratify(age=..., inplace=False)` returns a stratified copy and leaves `sir` alone, for comparing the two side by side. Stratifying twice by the same dimension is an error ([05-dimensions.md](05-dimensions.md)).

## Not included

- **The dict tuple shorthand** and its coercion rules (above).
- **Builder methods that change structure after construction** (`m.add_transition_flow(...)`). A model's states and transitions are given once, while it is made (as keywords, in a class body, or with `define_states()` and `define_transitions()` in its `__init__()`); `stratify()` is the one operator applied afterwards.
- **States named in strings** (arrow strings, `ss.transmission('S -> I', beta=0.3)` and `ss.progression('I -> R', dur=10)`). A transition takes the states themselves, so nothing is inferred from a name, including the source of infection, which is in the rate (`S >> I @ beta * I`).
- **Required `reads`/`writes` declarations, read-only views, and queued updates.** See the module contract and lifecycle above.
- **A separate `Model` and `ModelBuilder`.** One class, constructed once.
- **A module class for states that are not people** (the earlier `ss.Compartments`). A reservoir or a vector population is ordinary states in the disease ([03-states-and-transitions.md](03-states-and-transitions.md#states-that-are-not-people)).
- **A replicate count on the sim** (the earlier `n_reps=`). A sim is one run; many runs are a MultiSim.
