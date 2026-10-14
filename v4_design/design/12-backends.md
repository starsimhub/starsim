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
)
```

```text
>>> ss.Sim(sir, unit='day', stop=100).run()
solver: ode (every part of the model has a compartmental form)
```

```text
>>> ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), unit='day', stop=100).run()
solver: abm (RandomNet is an individual-level network)
n_agents: 10000 (default)
```

The solver is set with `solver=`, given either to `ss.Sim(...)` or to `sim.run(...)` (the second wins, which is convenient for running one sim several ways). The printed line can be hidden with `verbose=0`.

`'auto'` chooses between the ODE and agents. It never picks a stochastic compartmental solver; a user who wants one asks for it by name, or for `solver='stochastic'` ([The solvers](#the-solvers)).

**The one approximation `'auto'` makes** is to represent a non-exponential dwell time by Erlang stages, and it says so:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=ss.weibull(c=1.4, scale=ss.days(8))),
)
```

```text
>>> ss.Sim(sir, unit='day', stop=100).run()
solver: ode (every part of the model has a compartmental form)
  note: recovery has a Weibull dwell time, approximated by 2 Erlang stages (mean 7.3 days kept; std 5.2 instead of 5.3 days)
```

A lognormal recovery time does not by itself turn a two-line model into an agent-based one. `'auto'` never approximates the mixing (a network stays a network) and never drops individual-level structure (per-person attributes, histories, custom per-agent code); a model with any of those runs as agents.

### What makes a model agent-based

A model runs as agents under `'auto'` if it contains any of these:

- an individual-level network (`ss.RandomNet`, `ss.MFNet` and other sexual networks, `ss.StaticNet`, a maternal network);
- `n_agents=` given to the sim or to `ss.People`;
- a per-agent function (`fn(sim, uids)`) or a per-agent attribute that is not a stratification (viral load, an individual's partner count);
- a distribution on a per-person quantity other than a dwell time (`beta`, `rel_sus = ss.lognorm_ex(mean=1, std=0.3)`) that has not been binned into a dimension;
- custom code (a `step()` override or a function module) that does not work on groups (see [Functions](#functions-try-it));
- something that needs individual histories: contact tracing, transmission trees, the infection log (`analyzers='infection_log'`).

Or the user asks: `solver='abm'` runs any model as agents, for example to study elimination with a simple model.

Everything else has a compartmental form: homogeneous mixing, `ss.MixingPool` and contact matrices, stratifications, exponential and Erlang dwell times (exactly) and other dwell times (by stages, with the note above), `ss.modify` with conditions on strata, interventions whose eligibility uses only strata, states that are not people (reservoirs, vectors), and functions that use only counts, parameters, time, and strata. A state that is not people never makes a model agent-based.

For simple models the solver is a free choice made at run time. For complex models it is in effect part of the specification: a model built around a sexual network is an agent-based model, and running it another way gives a different, coarser model.


## The solvers

```python
ss.Sim(sir, solver='auto')        # the default
ss.Sim(sir, solver='ode')         # deterministic compartmental
ss.Sim(sir, solver='gillespie')   # exact stochastic, continuous time; 'ctmc' is accepted as an alias
ss.Sim(sir, solver='tau')         # tau-leaping
ss.Sim(sir, solver='binomial')    # chain-binomial, discrete time
ss.Sim(sir, solver='sde')         # diffusion approximation
ss.Sim(sir, solver='abm')         # agent-based
ss.Sim(sir, solver='stochastic')  # pick among gillespie/tau/binomial and say which
```

The model object is the same in every case, and whatever the solver, a sim is one trajectory; many runs are `ss.parallel(sim, n_runs=100)` ([09-results.md](09-results.md#one-run-many-runs)). The exact solver is called `'gillespie'` because that is the name the field uses for it; `'ctmc'`, the name of the process it simulates, is accepted as an alias and means the same thing. An integro-differential solver (`'ide'`, for arbitrary dwell times without stages) is deferred.

| Solver | Time | State | Good for | Loses accuracy when |
|---|---|---|---|---|
| `ode` | continuous | real-valued | large populations, fast fitting | counts are small or extinction matters |
| `gillespie` | continuous | integer | exact stochastic dynamics | never (it is exact); slow when the number of events is very large |
| `tau` | continuous | integer | large stochastic models | rates change quickly within a leap |
| `binomial` | discrete | integer | simple stochastic models on a fixed step | `rate * dt` is large |
| `sde` | continuous | real-valued | large populations with noise | any compartment is small |
| `abm` | discrete | individuals | networks, heterogeneity, histories | `rate * dt` is large; cost grows with `n_agents` |

These conditions are checked during the run and reported as one-line notes (`W1201 SmallCounts`, `W1202 ExtinctionPossible`, `W1203 LeapConditionMarginal`, `W1204 DiscretizationCoarse`), each naming the quantity involved and a solver that would not have the problem. They are notes, not errors: the run goes ahead and the user is told.

`'stochastic'` chooses for the user and says what it chose:

```text
>>> ss.Sim(sir, total_pop=1000, solver='stochastic', unit='day', stop=100).run()
solver: gillespie (exact; about 2,000 events at this population size)
```

```text
>>> ss.Sim(sir, total_pop=10_000_000, solver='stochastic', unit='day', stop=100).run()
solver: tau (exact Gillespie would need about 2e7 events)
  note: tau-leaping is an approximation; step chosen adaptively, leap condition 0.03
```

The choice SHOULD lean toward the exact solver when counts are small, because that is when extinction questions arise. A user who asks for `solver='gillespie'` gets the exact solver however long it takes; it is never swapped for tau-leaping. The result records which solver produced it.

## Approximations

> **`'auto'` approximates nothing except a non-exponential dwell time, which it represents by Erlang stages and says so. When a solver is requested explicitly and the model does not natively fit it, the library uses the standard approximation where one exists and says so in one line. It stops with an error only when there is no sensible approximation.**

```text
>>> ss.Sim(hiv, networks=ss.MFNet(), solver='ode', unit='year', stop=40).run()
solver: ode (requested)
  note: MFNet approximated by partner-change rates between sex/risk groups (4 groups)
  note: 'acute' has a lognormal dwell time, approximated by 4 Erlang stages (mean and std matched)
```

There are many valid ODE models of sexually transmitted infections. A network model run as an ODE is a different, coarser model, not an invalid one, and the user who typed `solver='ode'` has asked for it.

`approx=` says what to do about approximations:

| `approx=` | Effect |
|---|---|
| not given (the default) | Approximations are made and each prints a one-line note |
| `approx=True` | Approximations are made and the notes are not printed: the user has said they are expected |
| `approx=False` | Any approximation is an error (`E804 CapabilityError`, naming the piece and the approximation that would have been used) |

- Under `solver='auto'`, `approx=False` rules out the Erlang stages too: a model with a dwell time that has no exact compartmental form then runs as agents, where the dwell time is sampled directly, and the printed reason names the transition.
- `verbose=0` hides everything printed at run time, the notes included.
- Whatever is printed, the notes are stored with the results, so they survive into saved output, and they remain available from `sim.describe()` and `sim.solvers()`.

### The standard approximations

| The model has | Solver | Approximated by | Reported |
|---|---|---|---|
| A random network (`ss.RandomNet`, `ss.StaticNet`) | compartmental, explicit | contact rates between groups, with the network's mean number of contacts | the contact rate, or the matrix if stratified |
| A sexual or other structured network (`ss.MFNet`, ...) | compartmental, explicit | partner-change (or contact) rates between the groups the network distinguishes: sex, risk group, and age group if the model is stratified by age | the groups and the rate matrix |
| A dwell time that is not exponential or Erlang | compartmental, explicit or `'auto'` | Erlang stages matching the mean and, as closely as possible, the standard deviation; `stages=k` sets the number directly | the number of stages, and the mean and standard deviation that result |
| A distribution on a per-person quantity that is not a dwell time (`beta`, `rel_sus`) | compartmental, explicit | its mean. To keep the variation, the user bins it into a dimension (`sir.stratify(rel_sus=[...])`), which is exact for the bins and needs no note | the mean used |
| A continuous agent attribute used by the model (age, CD4 count) | compartmental, explicit | bins: the stratification's bin edges if it has one, otherwise default bins | the bin edges |
| A function in a `rate=`, `p=`, `eligible=`, or `when=` slot | compartmental | nothing: it is evaluated on groups instead of agents, if it can be ([Functions](#functions-try-it)) | only if it fails |
| `n_agents < total_pop` | `abm` | scaling: each agent stands for `total_pop / n_agents` people and count results are multiplied up | the scale factor |
| A compartmental model with nothing agent-based | `abm` | nothing needed: agents are sampled into the states, with homogeneous mixing | `n_agents` if defaulted |
| A very large population | `gillespie` | not substituted: the exact solver runs. `solver='stochastic'` picks tau-leaping and says so | |

"Explicit" means the user named the solver. Under `'auto'` only the dwell-time row applies; the other rows make the model agent-based instead.

How network rates are obtained: the library builds the network once for a sample population at `sim.init()`, counts contacts per unit time between each pair of groups, and uses those as the mixing rates. For a network with partnership durations, the rate is partnerships formed per unit time times acts per partnership, so that the per-contact `beta` keeps its meaning. This discards concurrency, clustering, and partnership persistence, and the one-line note is the reminder.

How stages are chosen: `k = round(mean^2 / std^2)`, at least 1, with each stage exponential with mean `mean / k`. The mean is kept exactly, and the note prints the standard deviation that results next to the one asked for. A distribution more variable than an exponential gets one stage and a note that its extra variance is not represented. `ss.gamma` with an integer shape is already Erlang and is exact. The same expansion is used by every compartmental solver, stochastic ones included.

### What has no approximation

These stop a compartmental run with `E804 CapabilityError`:

- contact tracing, and anything else that needs to know who a specific person's contacts were;
- transmission trees and individual histories (the infection log, `analyzers='infection_log'`);
- custom code that only works on agents: a function taking `uids`, a `deliver=fn(sim) -> uids` delivery, a `step()` override that reaches for a per-agent attribute that is not a stratification.

```text
E804 CapabilityError: cannot run 'hiv' with solver='ode'.
     The intervention 'partner_notification' traces the contacts of people who test positive,
     and a compartmental run has no individual contacts to trace.
     Options: use solver='abm', or replace it with a screening rate among partners' groups.
```

## `solvers()`: what can run this model

```python
hiv.solvers()     # for one disease
sim.solvers()     # for the whole sim: networks, interventions, and custom code included
```

```text
>>> sim.solvers()
                                 ode                               stochastic                        abm
network: MFNet                   approx (partner-change rates)     approx (partner-change rates)     yes
dwell time: acute (lognormal)    approx (4 Erlang stages)          approx (4 Erlang stages)          yes
function: care_seeking           yes (uses age, sex)               yes (uses age, sex)               yes
partner_notification             no (traces individual contacts)   no (traces individual contacts)   yes

auto would choose: abm (MFNet is an individual-level network)
```

The table is deliberately coarse: one row per ingredient that matters (each network, each non-exponential dwell time, each custom function, anything individual-level), one column per solver, and each entry is `yes`, `approx` (with how), or `no` (with why). Ordinary transitions and parameters that run everywhere are not listed. The stochastic compartmental solvers (`gillespie`, `tau`, `binomial`, `sde`) share one column unless they differ for this model. For functions the entries are filled in by trying them on groups, which happens at `sim.init()`; before that (and for `hiv.solvers()` on a disease that is not yet in a sim) a function's entry is `?`. The last line is what `'auto'` will do, and why.

## Functions: try it

Any slot accepts a function ([03-states-and-transitions.md](03-states-and-transitions.md)), and a condition may be a function of the sim ([08-interventions.md](08-interventions.md)). The library does not analyze functions in advance. It calls them with what the solver has:

- In an agent-based run, arrays have one entry per agent.
- In a compartmental run, the same names return arrays with one entry per group (one row per stratum): `sim.people.age` is the age group's midpoint, and a state's count (`sir.I.n`) is the number in that state.

A function that uses only things that exist in both (state counts, parameters, time, stratification coordinates) works in both without change:

```python
def care_seeking(sim):
    return 0.1 + 0.4 * (sim.people.age > 65)     # works on agents and on age groups
```

A function that asks for something agent-only stops the compartmental run with a message naming what it asked for:

```text
E804 CapabilityError: function 'care_seeking' asked for people.n_partners, which is a per-agent
     attribute and not a stratification, so it is not available with solver='ode'.
     Options: stratify by it (hiv.stratify(n_partners=[0, 1, 5])), or use solver='abm'.
```

So a lambda in an ODE model is fine, as long as what it does makes sense for groups. Under `'auto'`, functions are tried on groups once at `sim.init()`; if one fails, the model runs as agents and the printed reason says which function.

## How a model is built

At `sim.init()` the library turns the declarations into something runnable ([10-execution.md](10-execution.md) §Building the sim). No model code is interpreted line by line during the run.

- **Compartmental solvers.** The declared transitions become a flat list of flows, each with a source, a destination, and a rate expression over the array of groups. The right-hand side of the ODE (or the propensities of the stochastic solvers) is generated from that list. Declaration order does not matter, because every flow is evaluated from the same state.
- **Small models** are integrated with SciPy when there is no gradient to compute. This has no start-up cost.
- **Large models, and any model being fitted with gradients,** run through JAX: JAX runs the generated right-hand side as fast array code, and the exact gradient of every result with respect to every parameter comes with it by automatic differentiation. That is what makes gradient-based optimizers, NUTS, and gradient-based sensitivity available for ODE models ([13-inference.md](13-inference.md#gradients)). Gradients are for `solver='ode'` only; the stochastic solvers and agents have none. Without JAX installed, SciPy is used and gradients are by finite differences.
- **User functions** in a rate are differentiable if they are written with array operations that JAX can trace. One that is not makes the run fall back to SciPy, with a one-line note, and the fit falls back to a method that needs no gradient.
- **The agent-based solver** is Starsim's: each declared transition becomes vectorized array operations on the agents in its source state, called from the loop.

Which integration library is used is printed with the solver (`solver: ode (SciPy, 3 compartments)`) and can be set with `integrator=`: `integrator='scipy'` or `integrator='jax'`. `integrator=` names the library or scheme that integrates the ODE; it has nothing to say about which solver runs the model, which is `solver=`.

## States that are not people

A state that is not people (bacteria in water, mosquitoes; [03-states-and-transitions.md](03-states-and-transitions.md)) is an ordinary `ss.State` in the same disease. The library works out that it is not people because no transition connects it to the disease's first state, prints that guess in `describe()`, and `ss.State(people=False)` says it explicitly. It holds a plain count, is left out of `total_pop`, and is not split by people dimensions (age, sex, risk) when the model is stratified.

| Solver | What such a state is |
|---|---|
| `ode`, `sde` | One more real-valued compartment (or one per patch). Nothing distinguishes it from the others except that it is not counted in `N` |
| `gillespie`, `tau`, `binomial` | A real-valued count, whose flows are their expected values over each step (or each event's interval, under `gillespie`), while people are drawn. This is SimInf's split between continuous-time events for animals and a deterministic environmental pressure, and it keeps the cost of a reservoir independent of its size |
| `abm` | A count, never agents. People are agents and the reservoir stays a number, advanced over each step as an ODE (by RK4) with the people as they were at its start ([10-execution.md](10-execution.md#states-that-are-not-people)) |

Rates that use the count are written with `.n` (`beta * W.n / (W.n + K)`, `shed * I.n`) and are used exactly as written under every solver: there is no contact rate and no network in them. Under `'auto'` a model with such states runs as an ODE unless something else makes it agent-based; adding `n_agents=` makes the people agents and leaves the count a count.

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

`equations()` is a method of the disease and prints parameter names, not numbers. Inline parameters take the name of their slot (`beta`, `dur`), qualified by the transition's name where two would clash (`dur_recovery`). A stratified disease prints in indexed form (`dS[a]/dt = -beta * S[a] * sum_b(C[a,b] * I[b] / N[b])`). Nothing in the declaration is ODE-specific: with a network or mixing pool supplying contacts, a contact-rate symbol appears in the infection term and `beta` is per contact.

`R0()` returns a number computed from the next-generation matrix at the disease-free state, for models where that is well defined; it says so when it is not. Symbolic analysis beyond `equations()` (symbolic Jacobians, fixed points, eigenvalues) is not part of v1. Gradients of results with respect to parameters are a different thing and are in scope: they come from automatic differentiation of the ODE run ([How a model is built](#how-a-model-is-built)).

Stages multiply compartments, and with stratification they multiply quickly. `describe()` reports the total number of compartments ([05-dimensions.md](05-dimensions.md) §Size).

## Stochastic compartmental specifics

Time-varying rates MUST work under every solver. Under `gillespie` this needs the thinning algorithm for inhomogeneous rates, which the library uses without being asked; freezing the rate over an interval is not acceptable without a note.

Continuous-time solvers have no `dt` and so no rate-to-probability conversion. Results are resampled onto the sim's output grid by default, so that results from every solver have the same time axis.

Non-exponential dwell times need sub-states here exactly as under an ODE: the Erlang expansion is the same one, `stages=k` sets it by hand, and a dwell time that is not Erlang is approximated by fitted stages with the same one-line note.

A sim is one trajectory. The distribution comes from many runs, `msim = ss.parallel(sim, n_runs=500)`, and extinction is reported by `msim.summarize()` for every solver with integer state ([09-results.md](09-results.md#one-run-many-runs)), since it is often the reason for choosing such a solver. Common random numbers pair at the level of the run, not the agent ([11-random.md](11-random.md)).

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
| `ss.modify(target, multiply=, where=)` | a cached effective array, recomputed when the set of affected agents changes |
| `ss.outcome(...)` | a thinned, delayed draw from the source events |

- **Custom code** is ordinary Starsim: boolean state arrays, `dist.filter(uids)` to select with a probability, and direct reads and writes of other modules' state ([07-effects.md](07-effects.md)).
- **Aggregate quantities inside an agent-based sim** (an environmental reservoir, a mosquito population, wastewater) are states that are not people ([above](#states-that-are-not-people)): ordinary states of the disease that hold counts, written with the same transition syntax. This is how vector-borne and environmental transmission are written alongside agents.
- **Within-host state** (viral load, CD4 count) is a float agent property updated in `step()`. It can be a stratification when binned.
- **Contact tracing** applies only to networks, which keep their contacts; mixing pools do not.
- **Memory.** Arrays grow as agents are added, as in Starsim. Preallocating for the whole run is an optimization to benchmark before building, and would not change anything the user writes.
- **Small `n_agents`** is not warned about. Scaling to `total_pop` is reported as in the table above.


## Agreement between solvers

The aim is that **simple models agree across solvers within a stated tolerance**, and this is tested rather than assumed: for each simple model in the corpus, [18-conformance.md](18-conformance.md) runs every solver that `solvers()` marks `yes` and compares the results (stochastic solvers by their mean over many runs, at large population size). Entries marked `approx` are tested differently: the test records how far the approximation is from the agent-based result, so that the size of the difference is known and regressions are caught, without asserting that the two are equal.

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
