# Solvers: choosing, approximating, and building

How a model is run: the solvers, how the default is chosen, what happens when a model is asked to run in a way it does not natively fit, and how the runnable model is built.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md), [best-practices/compartmental.md](../best-practices/compartmental.md), [best-practices/stochastic-compartmental.md](../best-practices/stochastic-compartmental.md), [best-practices/agent-based.md](../best-practices/agent-based.md), and [best-practices/metapopulation.md](../best-practices/metapopulation.md).


## The default: `solver='auto'`

> **`'auto'` picks the simplest, fastest representation that runs the model without approximating its mixing or dropping anything individual-level: a deterministic ODE if every piece of the model has a compartmental form, otherwise agents. The choice and the reason are printed.**

```python
import starsim as ss

sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
    init_prev = 0.01,
)
```

```text
>>> ss.Sim(sir, unit='day', stop=100).run()
solver: ode (every part of the model has a compartmental form)
sir: R0 = 3, with 100 infectious at the start
```

```text
>>> ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), unit='day', stop=100).run()
solver: abm (RandomNet is an individual-level network)
Initializing sim with 10000 agents
```

The solver is set with `solver=`, given to `ss.Sim(...)`, or to `sim.run(...)` before the sim is initialized (the second wins). After `init()` the solver is fixed; `sim.new(solver=...)` makes a fresh copy with another, which is how one sim is run several ways. The printed lines can be hidden with `verbose=0`.

`'auto'` chooses between the ODE and agents. It never picks a stochastic compartmental solver; a user who wants one asks for it by name ([The solvers](#the-solvers)).

**The one approximation `'auto'` makes** is to represent a non-exponential dwell time by Erlang stages, and it says so (a fixed delay, `ss.constant`, is not approximated: it is a conveyor of whole steps, [below](#a-fixed-delay-and-a-minimum-stay)):

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=ss.weibull(c=1.4, scale=ss.days(8))),
    init_prev = 0.01,
)
```

```text
>>> ss.Sim(sir, unit='day', stop=100).run()
solver: ode (every part of the model has a compartmental form)
  note: recovery has a weibull dwell time, approximated by 2 Erlang stages (mean 7.29139 days kept; std 5.16 instead of 5.28 days)
sir: R0 = 2.19, with 100 infectious at the start
```

A lognormal recovery time does not by itself turn a two-line model into an agent-based one. `'auto'` never approximates the mixing (a network stays a network) and never drops individual-level structure (per-person attributes, histories, custom per-agent code); a model with any of those runs as agents.

### What makes a model agent-based

A model runs as agents under `'auto'` if it contains any of these:

- an individual-level network (`ss.RandomNet`, `ss.MFNet`, `ss.PartnerNet` and other sexual networks, `ss.StaticNet`, a maternal network), or a mixing pool between some people only (`src=`, `dst=`) or with a number of contacts that differs between people;
- `n_agents=` given to the sim or to `ss.People`;
- a per-agent function (`fn(sim, uids)`) or a per-agent attribute that is not a stratification (viral load, an individual's partner count);
- a distribution on a per-person quantity other than a dwell time (`beta`, `rel_sus = ss.lognorm_ex(mean=1, std=0.3)`) that has not been binned into a dimension;
- custom code (a function intervention or analyzer, or a `step()` override) that does not work on groups when it is tried on them (see [Functions](#functions-try-it)); a module written as code says outright that it works on groups, or does not, with the class attribute `compartmental = True` or `False`;
- something that needs individual histories: contact tracing, transmission trees, the infection log (`analyzers='infection_log'`).

Or the user asks: `solver='abm'` runs any model as agents, for example to study elimination with a simple model.

Everything else has a compartmental form: homogeneous mixing, an `ss.MixingPool` of everyone and contact matrices, stratifications, exponential and Erlang dwell times (exactly), fixed delays and minimum stays (`ss.constant`, `after=`, `before=`: by a conveyor of whole steps), and other dwell times (by stages, with the note above), `ss.modify` with conditions on strata, interventions whose eligibility uses only strata, states that are not people (reservoirs, vectors), and functions that use only counts, parameters, time, and strata. A state that is not people never makes a model agent-based.

The reasons are kept in one record, which `sim.solvers()`, `describe()`, and the `CapabilityError` all read.

For simple models the solver is a free choice made when the sim is built. For complex models it is in effect part of the specification: a model built around a sexual network is an agent-based model, and running it another way gives a different, coarser model.


## The solvers

```python
ss.Sim(sir, solver='auto')        # the default
ss.Sim(sir, solver='ode')         # deterministic compartmental
ss.Sim(sir, solver='binomial')    # chain-binomial, discrete time
ss.Sim(sir, solver='gillespie')   # exact stochastic, continuous time; tau-leaping, with a note, for a run that would take too long one event at a time
ss.Sim(sir, solver='abm')         # agent-based
```

The solvers of earlier drafts raise an error that names what to use: `'tau'` and `'stochastic'` (use `'gillespie'`, which takes tau leaps by itself when that pays), `'sde'` (removed: it resolved competing exits in declaration order at coarse steps, agreed worst with the ODE, and had no user; use `'gillespie'` or `'ode'`), and `'ctmc'` (use `'gillespie'`).

The model object is the same in every case, and whatever the solver, a sim is one trajectory; many runs are `ss.parallel(sim, n_runs=100)` ([09-results.md](09-results.md#one-run-many-runs)). The exact solver is called `'gillespie'` because that is the name the field uses for it. An integro-differential solver (`'ide'`, for arbitrary dwell times without stages) is deferred.

| Solver | Time | State | Good for | Loses accuracy when |
|---|---|---|---|---|
| `ode` | continuous | real-valued | large populations, fast fitting | counts are small or extinction matters |
| `gillespie` | continuous | integer | exact stochastic dynamics (for rates that are constant within each quarter of a step) | values vary quickly within a step; for a run that would take more than about a second one event at a time, it takes tau leaps, which lose accuracy when rates change quickly within a leap |
| `binomial` | discrete | integer | simple stochastic models on a fixed step | `rate * dt` is large |
| `abm` | discrete | individuals | networks, heterogeneity, histories | `rate * dt` is large; cost grows with `n_agents` |

Some of these conditions are checked and reported as one-line notes, each naming the quantity involved and what would not have the problem: after an ODE run, `ss.SmallCounts` (it starts with only a few infected) and `ss.ExtinctionPossible` (the number infected falls below one and rises again); at `init()`, for a solver that takes whole steps (`binomial`, `abm`), `ss.LargeStep` (a transition's `rate * dt` is large), which also says how far whole steps can be from the ODE and to run `sim.check_dt()`. They are notes, not errors: the run goes ahead and the user is told.

`'gillespie'` counts the events a run would take from an ODE run at `sim.init()` (milliseconds) and estimates how long taking them one at a time would take: about 0.1 µs an event for one group, and about 0.05·n^1.5 µs for n groups that affect each other (strata that mix, or places that people commute between), since each event changes the rates of the groups the mixing takes it to. Below about a second it takes them one at a time; above, it takes tau leaps and says so. Places that do not affect each other within a step are run by place, which is always exact, since it is faster than tau leaps (SimInf's 16,000 herds and 4.7 million events take 2 to 3 s):

```text
>>> ss.Sim(sir, total_pop=1000, solver='gillespie', unit='day', stop=100).run()
solver: gillespie (exact; an ODE run gives about 1,900 events)
```

```text
>>> ss.Sim(sir, total_pop=10_000_000, solver='gillespie', unit='day', stop=100).run()
  note: solver="gillespie" takes tau leaps: an ODE run gives about 1.9e+07 events, which would take about 2.8 s one at a time, more than 1 s; approx=False keeps it exact
solver: tau (tau-leaping for solver="gillespie": an ODE run gives about 1.9e+07 events, about 2.8 s one at a time)
```

Small counts, where extinction questions arise, are runs of few events, so they get the exact solver. `approx=False` keeps the exact solver however long it takes. `sim.solver` records which solver produced the result (`'tau'` when it took tau leaps). The tau leaps choose their own size from the rates, so a step that is large for them needs no note of its own.

## Approximations

> **`'auto'` approximates nothing except a non-exponential dwell time, which it represents by Erlang stages and says so. When a solver is requested explicitly and the model does not natively fit it, the library uses the standard approximation where one exists and says so in one line. It stops with an error only when there is no sensible approximation.**

```text
>>> ss.Sim(hiv, networks=ss.RandomNet(n_contacts=4), solver='ode', unit='year', stop=40).run()
  note: RandomNet approximated by homogeneous mixing with 4 contacts per year (its mean number of contacts)
solver: ode (requested)
  note: acute has a lognorm_ex dwell time, approximated by 6 Erlang stages (mean 0.25 years kept; std 0.102 instead of 0.1 years)
```

A network model run as an ODE is a different, coarser model, not an invalid one, and the user who typed `solver='ode'` has asked for it. A network of partnerships (`ss.MFNet`, `ss.PartnerNet`) has no approximation yet, so it raises `CapabilityError` under a compartmental solver.

`approx=` says what to do about approximations:

| `approx=` | Effect |
|---|---|
| not given (the default) | Approximations are made and each prints a one-line note |
| `approx=True` | Approximations are made and the notes are kept in `sim.notes` without being printed: the user has said they are expected |
| `approx=False` | Any approximation is an error (`CapabilityError`, naming the piece and the approximation that would have been used) |

- Under `solver='auto'`, `approx=False` rules out the Erlang stages too: a model with a dwell time that has no exact compartmental form then runs as agents, where the dwell time is sampled directly, and the printed reason names the transition.
- `verbose=0` hides everything printed at run time, the notes included.
- Whatever is printed, the notes are kept in `sim.notes` (and in the model spec's metadata), so they survive into saved output; `sim.solvers()` shows what each solver would approximate.

### The standard approximations

| The model has | Solver | Approximated by | Reported |
|---|---|---|---|
| A random or static network (`ss.RandomNet`, `ss.StaticNet`) | compartmental, explicit | homogeneous mixing with the network's mean number of contacts (for a static network, the mean degree of its graph) | the contact rate and where it came from |
| A network of partnerships (`ss.MFNet`, `ss.PartnerNet`) | compartmental, explicit | none yet: `CapabilityError` | |
| A dwell time that is not exponential, Erlang, or fixed (`ss.constant`) | compartmental, explicit or `'auto'` | Erlang stages matching the mean and, as closely as possible, the standard deviation; `stages=k` sets the number directly | the number of stages, and the mean and standard deviation that result |
| A distribution on a per-person quantity that is not a dwell time (`beta`, `rel_sus`) | compartmental, explicit | its mean. To keep the variation, the user bins it into a dimension (`sir.stratify(rel_sus=[...])`), which is exact for the bins and needs no note | the mean used |
| A continuous agent attribute used by the model (age, CD4 count) | compartmental, explicit | the bins of a stratification by it (a function then sees each group's midpoint); without one, `CapabilityError`, saying to stratify by it | |
| A function in a `rate=`, `p=`, `where=`, or `when=` slot | compartmental | nothing: it is evaluated on groups instead of agents, if it can be ([Functions](#functions-try-it)) | only if it fails |
| `n_agents < total_pop` | `abm` | scaling: each agent stands for `total_pop / n_agents` people and count results are multiplied up | the scale factor |
| A compartmental model with nothing agent-based | `abm` | nothing needed: agents are sampled into the states, with homogeneous mixing | `n_agents` if defaulted |
| A very large population | `gillespie` | tau-leaping, when the events estimated by an ODE run would take more than about a second one at a time (never by place); `approx=False` keeps the exact solver | the estimated number of events and time |
| A value that varies smoothly in time (`ss.seasonal`, `ss.timeseries`) | `gillespie` | held for a quarter of a step at a time, at its middle; `approx=False` makes it an error | that it is held |

"Explicit" means the user named the solver. Under `'auto'` only the dwell-time row applies; the other rows make the model agent-based instead.

How network rates are obtained: a random network gives its mean number of contacts (`n_contacts`, or the mean of its distribution), and a static network the mean degree of its graph, as a homogeneous contact rate, so that the per-contact `beta` keeps its meaning. This discards concurrency, clustering, and partnership persistence, and the one-line note is the reminder.

How stages are chosen: `k = round(mean^2 / std^2)`, at least 1, with each stage exponential with mean `mean / k`. The mean is kept exactly, and the note prints the standard deviation that results next to the one asked for. A distribution more variable than an exponential gets one stage and a note that its extra variance is not represented. `ss.gamma` with an integer shape is already Erlang and is exact. The same expansion is used by every compartmental solver, stochastic ones included.

### A fixed delay and a minimum stay

A fixed delay (`ss.constant`, as odin's `delay()`) has no good Erlang form: its stages converge as one over their number, and their ODE becomes stiff (200 stages for odin's 14-day latent period took 1.9 s and were still 1.6% off). Every compartmental solver takes it as a conveyor of whole steps instead ([03-states-and-transitions.md](03-states-and-transitions.md#a-fixed-delay)): `K + 1` compartments for a delay of `K` steps, moved along one at the end of each step, with no rate between them. The people due to leave leave in the next step:

| Solver | When the people due to leave a conveyor leave |
|---|---|
| `ode` | at a constant rate over the step, so the release of the people who arrived over a step `K` steps before is that of the delay differential equation, to second order (odin's SEIR within 0.25% at a step of a day, 0.06% at half a day, in 8 ms); the state's other exits act on them until they leave, with the release lowered to what they leave, so a competing exit acts over the whole delay |
| `binomial` | at the start of the step, once the step's rates are worked out from the people as they were, as agents' scheduled exits come first; so it agrees with agents step for step |
| `gillespie` (and its tau leaps) | at the middle of the step, the mean of the ODE's release |
| `abm` | each at the start of the step their delay, rounded to the nearest step, ends in |

The stock at each time point is the same in each: exact for a delay of whole steps. A delay that is not a whole number of steps is rounded to one, with a note. A minimum stay (`after=`) and a limit (`before=`) use the same conveyor, with each transition acting only on the people who have stayed long enough (or not too long); a stay on a transmission, or out of a state that splits on entry, has no compartmental form yet. The kernels, the particle filter, and the run in JAX all take conveyors; a model without one pays nothing for them.

### What has no approximation

These stop a compartmental run with `CapabilityError`:

- contact tracing, and anything else that needs to know who a specific person's contacts were;
- transmission trees and individual histories (the infection log, `analyzers='infection_log'`);
- custom code that only works on agents: a function taking `uids`, a `deliver=fn(sim) -> uids` delivery, a `step()` override that reaches for a per-agent attribute that is not a stratification.

```text
CapabilityError: cannot run 'hiv' with solver='ode'.
     The intervention 'partner_notification' traces the contacts of people who test positive,
     and a compartmental run has no individual contacts to trace.
     Options: use solver='abm', or replace it with a screening rate among partners' groups.
```

## `solvers()`: what can run this model

```python
sir.solvers()     # for one disease
sim = ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), unit='day', stop=100)
sim.solvers()     # for the whole sim: networks, interventions, and custom code included
```

For an HIV model with a sexual network, a lognormal dwell time, a function in a rate, and partner notification (`trace=`):

```text
>>> sim.solvers()
                                 ode                                       binomial, gillespie                       abm
network: MFNet                   no (no approximation)                     no (no approximation)                     yes
function: care_seeking           no (the model does not run as ...)        no (the model does not run as ...)        yes
dwell time: acute (lognorm_ex)   approx (Erlang stages)                    approx (Erlang stages)                    yes
module: partner_notification     no ("partner_notification" traces ...)    no ("partner_notification" traces ...)    yes

auto would choose: abm (MFNet is an individual-level network)
```

The table is deliberately coarse: one row per ingredient that matters (each network, each non-exponential dwell time, each custom function, each module that needs agents, `n_agents=`), one column per solver, and each entry is `yes`, `approx` (with how), or `no` (with why). Ordinary transitions and parameters that run everywhere are not listed. The stochastic compartmental solvers (`binomial`, `gillespie`) share one column. `sim.solvers()` initializes the sim if it has not been, since functions are filled in by trying them on groups; when something else already makes the model agent-based, they are not tried. `sir.solvers()` on a disease alone tries its functions on the disease's own groups. The last line is what `'auto'` will do, and why; `sim.kernel_reason()` says, separately, why a compartmental run is taken a step at a time rather than in one call to the kernels.

## Functions: try it

Any slot accepts a function ([03-states-and-transitions.md](03-states-and-transitions.md)), and a condition may be a function of the sim ([08-interventions.md](08-interventions.md)). The library does not analyze functions in advance. It calls them with what the solver has:

- In an agent-based run, arrays have one entry per agent.
- In a compartmental run, the same names return arrays with one entry per group (one row per stratum): in a model stratified by age, `sim.people.age` is the age group's midpoint, and a state's count (`sir.I.n`) is the number in that state.

A function that uses only things that exist in both (state counts, parameters, time, stratification coordinates) works in both without change:

```python
def care_seeking(sim):
    return 0.1 + 0.4 * (sim.people.age > 65)     # works on agents and on age groups
```

A function that asks for something agent-only stops the compartmental run with a message naming what it asked for:

```text
CapabilityError: people.n_partners was asked for, but it is a per-agent attribute and not a stratification, so it does not exist in a compartmental run.
     Stratify by it (e.g. sim.stratify(n_partners=[...])), or run with solver="abm".
```

So a lambda in an ODE model is fine, as long as what it does makes sense for groups. Under `'auto'`, functions are tried on groups once at `sim.init()`; if one fails, the model runs as agents and the printed reason says which function.

## How a model is built

At `sim.init()` the library turns the declarations into something runnable ([10-execution.md](10-execution.md) §Building the sim). No model code is interpreted line by line during the run.

- **Compartmental solvers.** The declared transitions become a flat list of flows, each with a source, a destination, and a rate expression over the array of groups. The right-hand side of the ODE (or the propensities of the stochastic solvers) is generated from that list. Declaration order does not matter, because every flow is evaluated from the same state.
- **Every compartmental solver** runs in compiled kernels (Numba), from one flat program of the flows; there is no second copy in Python. The ODE takes fixed RK4 steps with automatic substeps, whether or not gradients are asked for. A value that is a function of the sim is called once a step and held within it, unless it gives an expression of the model's states (`I.n`), which is put into the program in its place.
- **A whole run in one call.** A compartmental sim with nothing that needs a step of its own (deliveries on a schedule and modifiers that depend only on time and states are worked out before the run) is run from start to finish in one call to the kernels; otherwise the kernels take it a step at a time from Python. `sim.kernel_reason()` says why a run is stepped.
- **Gradients, and many runs with different values,** run through JAX, which runs the same program (the same RK4 steps) as array code; it is optional (`pip install starsim[jax]`). `sim.gradient()` differentiates the run by forward-mode automatic differentiation, and the steps either side of it (parameters into the program, and the run into results) by central differences, so its gradients are exact to about 1e-10. `sim.log_density()` and NUTS have the whole log-density (the run, the results with data, the likelihood, and the priors) as one function in JAX. That is what makes gradient-based optimizers and NUTS available for ODE models ([13-inference.md](13-inference.md#gradients)). Gradients are for `solver='ode'` only, for a sim that the kernels run from start to finish, and for rates written as expressions (a rate that is a Python function of the sim has none); the stochastic solvers and agents have none. JAX is chosen by `sim.gradient()`, `sim.log_density()`, gradient fits, and `ss.parallel()` with 300 or more sets of values, never by `integrator=` (`integrator='jax'` is an error).
- **The agent-based solver** is Starsim's: each declared transition becomes vectorized array operations on the agents in its source state, called from the loop.

`integrator=` is the scheme of the ODE: `'rk4'` (the default, with enough substeps, chosen at the start of each step, that no compartment turns over by more than a quarter in one, nor grows by more than a quarter through its inflows, as an epidemic's infected do: within about 1e-4 of the exact solution on the textbook models at a daily step) or `'euler'` (for matching frameworks that take Euler steps, as the Atomica importer does). Any other value is an error at `sim.init()`. It has nothing to say about which solver runs the model, which is `solver=`. `substeps=` sets the number of substeps per step, for the ODE schemes and for `binomial` (whose default is 1: it takes whole steps, moving people through as many stages of a dwell time as they pass in a step).

## States that are not people

A state that is not people (bacteria in water, mosquitoes; [03-states-and-transitions.md](03-states-and-transitions.md)) is an ordinary `ss.State` in the same disease. The library works out that it is not people because no transition connects it to the disease's first state, prints that guess in `describe()`, and `ss.State(people=False)` says it explicitly. It holds a plain count, is left out of `total_pop`, and is not split by people dimensions (age, sex, risk) when the model is stratified.

| Solver | What such a state is |
|---|---|
| `ode` | One more real-valued compartment (or one per patch). Nothing distinguishes it from the others except that it is not counted in `N` |
| `gillespie` (and its tau leaps), `binomial` | A real-valued count, whose flows are their expected values over each step (or each event's interval, under `gillespie`), while people are drawn. This is SimInf's split between continuous-time events for animals and a deterministic environmental pressure, and it keeps the cost of a reservoir independent of its size |
| `abm` | A count, never agents. People are agents and the reservoir stays a number, advanced over each step as an ODE (by RK4) with the people as they were at its start ([10-execution.md](10-execution.md#states-that-are-not-people)) |

Rates that use the count are written with `.n` (`beta * W.n / (W.n + K)`, `shed * I.n`) and are used exactly as written under every solver: there is no contact rate and no network in them. A count set each step by `ss.update` is worked out from the model at the start of the step and set at its end, in the kernels (and in JAX) for the compartmental solvers and after the transmissions for agents; the exact solver by place falls back to the exact solver, since an update can read every place. Under `'auto'` a model with such states runs as an ODE unless something else makes it agent-based; adding `n_agents=` makes the people agents and leaves the count a count.

## Compartmental specifics

```python
sir.equations()     # the ODE system, with symbolic parameter names
sir.R0()            # optional numeric helper, for simple models
```

```text
>>> sir.equations()
dS/dt = -beta * S * I / N
dI/dt =  beta * S * I / N - I / dur
dR/dt =  I / dur
```

`equations()` is a method of the disease and prints parameter names, not numbers. Inline parameters take the name of their slot (`beta`, `dur`), qualified by the transition's name where two would clash (`dur_recovery`). A dwell time represented by stages prints one equation per stage (`dI_1/dt`, ...). A stratified disease prints the equations of one group, not an indexed form over the groups. Nothing in the declaration is ODE-specific: with a network or mixing pool supplying contacts, a contact-rate symbol appears in the infection term and `beta` is per contact.

`R0()` returns a number computed from the next-generation matrix at the disease-free state, for models where that is well defined; it says so when it is not. The infected compartments include the counts infection passes through (a reservoir, mosquitoes infected and infectious), and the other counts that an inflow fills are at their disease-free equilibrium (mosquitoes not infected: emergence over mortality); each passage through a vector is a generation, so for Ross-Macdonald `R0()` is the square root of the classical value. Symbolic analysis beyond `equations()` (symbolic Jacobians, fixed points, eigenvalues) is not part of v1. Gradients of results with respect to parameters are a different thing and are in scope: they come from automatic differentiation of the ODE run ([How a model is built](#how-a-model-is-built)).

Stages multiply compartments, and with stratification they multiply quickly ([05-dimensions.md](05-dimensions.md) §Size); `describe()` lists the dimensions and their levels, and the stages of each dwell time.

## Stochastic compartmental specifics

Time-varying rates MUST work under every solver. Under `gillespie` (and its tau leaps) a value that varies smoothly in time is held for a quarter of a step at a time, at its middle, so the solver is exact for rates that are constant within each quarter; this is said in a note, and `approx=False` makes it an error. The chain-binomial solver takes such a value at the middle of each substep.

The continuous-time solver has no `dt` and so no rate-to-probability conversion. Its results are recorded on the sim's time points, so that results from every solver have the same time axis.

Non-exponential dwell times need sub-states here exactly as under an ODE: the Erlang expansion is the same one, `stages=k` sets it by hand, and a dwell time that is not Erlang is approximated by fitted stages with the same one-line note.

A sim is one trajectory. The distribution comes from many runs, `msim = ss.parallel(sim, n_runs=500)` (or `sim.run(n_runs=500)`), and `msim.extinction()` says how often each disease died out ([09-results.md](09-results.md#one-run-many-runs)), since it is often the reason for choosing such a solver. Each module's stochastic compartmental run draws from a generator of its own, so pairing between scenarios is weak ([11-random.md](11-random.md)).

## Agent-based specifics

The agent-based solver is Starsim's, and its architecture is kept: typed per-agent arrays, persistent UIDs with a live view so that births and deaths do not invalidate references, vectorized operations, and one module abstraction.

| Declaration | In an agent-based run |
|---|---|
| `ss.State` (people) | a boolean array over agents |
| `ss.State` that is not people | a count (optionally per patch) stepped alongside the agents; never agents |
| a contact-mediated transition | for each contact on each network, compute the transmission probability and draw |
| a transition with `dur=` | sample a duration on entry and schedule the exit |
| competing rates out of one state | one draw per agent ([10-execution.md](10-execution.md)) |
| `sir.stratify(risk=[...])` | an agent property; continuous ones stay continuous |
| `ss.modify(target, multiply=, where=)` | applied where the quantity is used; `sir.rel_sus` reads the base value and `sir.rel_sus.effective` the value with the active modifiers |
| `ss.outcome(...)` | a thinned, delayed draw from the source events |

- **Custom code** is ordinary Starsim: boolean state arrays, `dist.filter(uids)` to select with a probability, and direct reads and writes of other modules' state ([07-effects.md](07-effects.md)).
- **Aggregate quantities inside an agent-based sim** (an environmental reservoir, a mosquito population, wastewater) are states that are not people ([above](#states-that-are-not-people)): ordinary states of the disease that hold counts, written with the same transition syntax. This is how vector-borne and environmental transmission are written alongside agents.
- **Within-host state** (viral load, CD4 count) is a float agent property updated in `step()`. It can be a stratification when binned.
- **Contact tracing** applies only to networks, which keep their contacts; mixing pools do not.
- **Memory.** Arrays grow as agents are added, as in Starsim. Preallocating for the whole run is an optimization to benchmark before building, and would not change anything the user writes.
- **Small `n_agents`** is not warned about. Scaling to `total_pop` is reported as in the table above.


## Agreement between solvers

The aim is that **simple models agree across solvers within a stated tolerance**, and this is tested rather than assumed: each simple model of the corpus is run under `ode`, `binomial`, `gillespie`, and `abm` at a fine step, and the stochastic solvers' means must be within a tolerance of the ODE (5%, or a wider one stated with its reason) (`tests/test_contracts.py`; [18-conformance.md](18-conformance.md)). Agents and stepped solvers differ from the ODE by an amount of order `dt`, which is why the step is fine. Entries marked `approx` are not tested for agreement.

How large the difference of order `dt` is, measured on the current solvers against the exact solver (`gillespie` at `dt=0.1`), over 1,000 runs (200 as agents) of models of 1,000 people with daily rates (the stats review's `solvers_vs_gillespie.py`, re-run on 2026-10-18):

| Model | Solver | `dt` = 1 day | `dt` = 7 days |
|---|---|---|---|
| SIR, R0 3, mean infectious 10 days | `binomial` | peak +4%, 1.7 days late; final size +1% | peak +32%, 10 days late; final size +5% |
| | `abm` | peak +2%, 3.4 days late; final size unchanged | peak +12%, 24 days late; final size −0.4% |
| SEIR, Erlang latent (3 stages, 4 days) and infectious (2 stages, 6 days) periods | `binomial` | peak +3%, 3 days late; final size +1.5% | peak +26%, 21 days late; final size +5% |
| | `abm` | peak −7%, 3 days late; final size unchanged | peak +22%, 15 days late; final size +4% |
| SIS that can die out, 200 people, R0 1.5 | `binomial` | extinction within a year 11% (exact: 15%) | extinction 1% |
| | `abm` | extinction 18% | extinction 23% |
| SIR by age, contact matrix, peak on day 7 | `binomial` | peak +1%, 2 days late | peak +3%, 14 days late |
| | `abm` | peak +4%, 2.5 days late | peak +5%, 15 days late |

As agents, an exponential dwell given as a duration (`dur=10`) and the same dwell given as a rate (`rate=0.1`) are realized differently, and so differ at a coarse step: the duration is drawn when the agent enters the state and rounded to the nearest step, while the rate is a chance 1 − exp(−r·dt) of leaving at the start of each step. On an SIR of 100,000 people (`beta` 0.3 per day, contact-mediated, 10 days infectious), the peak is +4.7% of the ODE's with `dur=` and −6% with `rate=` at a daily step (final size 0% and −1%), and +16% and −44% at a weekly step (final size −0.3% and −13%). A chance r·dt/(1 + r·dt) for both would put each within 1%, but only for contact-mediated transmission: a rate written on counts (`S >> I @ beta*I.frac`) and states that are counts see the people who leave at the start of a step, and move the other way (decisions: item 18). Taking those rates and counts after the exits too, as transmission is, puts the SIRs within 1% at a daily and a weekly step, but r·dt/(1 + r·dt) slows each state's clock by 1 + r·dt (someone who enters a state during a step is counted in it for 1/r plus one step), which only a single-rate SIR does not show: as agents at a daily step it put an SEIR's I peak 12% low (3 and 5 days, `dur=`; it is 1% low), the cholera model's I peak 11% high (it is 2%), and a state filled at a capped rate 15% high (it is 2%), so it is not taken yet (open question: item 18).

At `dt=0.1` every difference is within about 1% (peaks a few tenths of a day late). On a seasonal SIRS of 100,000 people (mean `beta` 0.3, amplitude 0.5, 8 days infectious, 60 days immune), `binomial` at a daily step has 19% more infectious on average at monthly points than the daily ODE, up to 53% at one point, and 8.5% more infected-days (between 7.4% and 8.7% over five seeds); with `substeps=4` or `dt=0.25` that falls to 5% on average (15% at most), and `gillespie` is within 1.5%. The `ss.LargeStep` note gives the cause for each transition (`rate × dt`); `sim.check_dt()` measures the effect on the model at hand.

For complex models, exact equivalence between solvers is neither provable nor the aim. An agent-based HIV model and its ODE approximation are expected to differ; what the library guarantees is that it reports what was approximated.

## Deferred: mixing solvers within one model

Some questions need most of a model to run as smooth flows while a few rare transitions happen as whole, random events: for example, importation of a single case into a population where the ODE says the epidemic is over. This would be spelled as a `solver=` keyword on the transition (`ss.transmission('S -> I', beta=1e-6, solver='gillespie')`), defaulting to following the run. It is not in v1. Until then, such questions use a stochastic solver for the whole model.

## Rejected

- **A separate model class per solver.** The same disease should be writable once.
- **Stochasticity as a property of the model.** It is a choice of solver.
- **Refusing to approximate.** A user who asks for an ODE of a network model wants the standard coarse version. Say what was done; `approx=False` is there for those who want the stop.
- **Approximating the mixing under `'auto'`.** The default never replaces a network by rates or drops per-person structure; that happens only when the user names a solver. Erlang stages for a dwell time are the one exception, because sending an otherwise simple model to agents over one lognormal is the bigger surprise.
- **Approximating without saying so.** The note is one line. `approx=True` or `verbose=0` hides it, and it is still stored with the results.
- **A separate module class for quantities that are not people.** They are ordinary states that hold counts.
- **The diffusion approximation as the default stochastic solver.** It fails at small counts, which is where stochasticity matters.
- **A menu of network-to-ODE closures.** One standard approximation per ingredient is enough. Someone who needs the network's exact behavior can run agents.
- **Agent-based as the default for every model.** A two-line SIR should run as the three equations it is, instantly. Models that need agents get them automatically.
- **Analyzing user functions in advance.** Trying them is simpler and gives a clearer message.
- **Making the user write the derivative function.** It is positional, unchecked, and cannot be stratified automatically.
