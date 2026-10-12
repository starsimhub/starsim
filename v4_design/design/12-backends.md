# Run methods: choosing, approximating, and building

How a model is run: the methods, how the default is chosen, what happens when a model is asked to run in a way it does not natively fit, and how the runnable model is built.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md), [best-practices/compartmental.md](../best-practices/compartmental.md), [best-practices/stochastic-compartmental.md](../best-practices/stochastic-compartmental.md), [best-practices/agent-based.md](../best-practices/agent-based.md), and [best-practices/metapopulation.md](../best-practices/metapopulation.md).

## The default: `method='auto'`

> **`'auto'` picks the simplest, fastest representation that runs the model exactly: a deterministic ODE if every piece of the model has a compartmental form, otherwise agents. The choice and the reason are printed.**

```python
import starsim as ss

sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
)
```

```text
>>> ss.Sim(sir, unit='day', stop=100).run()
method: ode (every part of the model has a compartmental form)
```

```text
>>> ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), unit='day', stop=100).run()
method: abm (RandomNet is an individual-level network)
n_agents: 10000 (default)
```

Under `'auto'` nothing is approximated. The printed line can be hidden with `verbose=0` and the choice overridden with `method=`, given either to `ss.Sim(...)` or to `sim.run(...)` (the second wins, which is convenient for running one sim several ways).

### What makes a model agent-based

A model runs as agents under `'auto'` if it contains any of these:

- an individual-level network (`ss.RandomNet`, `ss.MFNet` and other sexual networks, `ss.StaticNet`, a maternal network);
- `n_agents=` given to the sim or to `ss.People`;
- a per-agent function (`fn(sim, uids)`) or a per-agent attribute that is not a stratification (viral load, an individual's partner count);
- a dwell-time distribution with no exact compartmental form (anything other than exponential or an Erlang given as `ss.gamma` with an integer shape or as `stages=k`);
- custom code (a `step()` override or a function module) that does not work on groups (see [Functions](#functions-try-it));
- something that needs individual histories: contact tracing, transmission trees.

Or the user asks: `method='abm'` runs any model as agents, for example to study elimination with a simple model.

Everything else has a compartmental form: homogeneous mixing, `ss.MixingPool` and contact matrices, stratifications, exponential and Erlang dwell times, `ss.modify` with conditions on strata, interventions whose eligibility uses only strata, `ss.Compartments` modules, and functions that use only counts, parameters, time, and strata.

For simple models the run method is a free choice made at run time. For complex models it is in effect part of the specification: a model built around a sexual network is an agent-based model, and running it another way gives a different, coarser model.

## The methods

```python
ss.Sim(sir, method='auto')        # the default
ss.Sim(sir, method='ode')         # deterministic compartmental
ss.Sim(sir, method='ctmc')        # exact stochastic, continuous time (Gillespie)
ss.Sim(sir, method='tau')         # tau-leaping
ss.Sim(sir, method='binomial')    # chain-binomial, discrete time
ss.Sim(sir, method='sde')         # diffusion approximation
ss.Sim(sir, method='abm')         # agent-based
ss.Sim(sir, method='stochastic')  # pick among ctmc/tau/binomial and say which
```

The model object is the same in every case. An integro-differential method (`'ide'`, for arbitrary dwell times without stages) is deferred.

| Method | Time | State | Good for | Loses accuracy when |
|---|---|---|---|---|
| `ode` | continuous | real-valued | large populations, fast fitting | counts are small or extinction matters |
| `ctmc` | continuous | integer | exact stochastic dynamics | the number of events is very large (slow, not wrong) |
| `tau` | continuous | integer | large stochastic models | rates change quickly within a leap |
| `binomial` | discrete | integer | simple stochastic models on a fixed step | `rate * dt` is large |
| `sde` | continuous | real-valued | large populations with noise | any compartment is small |
| `abm` | discrete | individuals | networks, heterogeneity, histories | `rate * dt` is large; cost grows with `n_agents` |

These conditions are checked during the run and reported as one-line notes (`W1201 SmallCounts`, `W1202 ExtinctionPossible`, `W1203 LeapConditionMarginal`, `W1204 DiscretizationCoarse`), each naming the quantity involved and a method that would not have the problem.

`'stochastic'` chooses for the user and says what it chose:

```text
>>> ss.Sim(sir, total_pop=1000, method='stochastic', unit='day', stop=100).run()
method: ctmc (exact; about 2,000 events at this population size)
```

```text
>>> ss.Sim(sir, total_pop=10_000_000, method='stochastic', unit='day', stop=100).run()
method: tau (exact simulation would need about 2e7 events; leap size chosen adaptively)
```

The choice SHOULD lean toward the exact method when counts are small, because that is when extinction questions arise.

## Approximations

> **`'auto'` never approximates. When a method is requested explicitly and the model does not natively fit it, the library uses the standard approximation where one exists and says so in one line. It stops with an error only when there is no sensible approximation.**

```text
>>> ss.Sim(hiv, networks=ss.MFNet(), method='ode', unit='year', stop=40).run()
method: ode (requested)
approximated: MFNet -> partner-change rates between sex/risk groups (4 groups)
approximated: lognormal dwell time in 'acute' -> 4 Erlang stages (mean and std matched)
```

There are many valid ODE models of sexually transmitted infections. A network model run as an ODE is a different, coarser model, not an invalid one, and the user who typed `method='ode'` has asked for it.

Two switches:

- `approx=False` makes any approximation an error (`E804 CapabilityError`, naming the piece and the approximation that would have been used).
- `verbose=0` hides the notes. They remain available from `sim.describe()` and `sim.methods()`.

### The standard approximations

| The model has | Requested method | Approximated by | Reported |
|---|---|---|---|
| A random network (`ss.RandomNet`, `ss.StaticNet`) | compartmental | contact rates between groups, with the network's mean number of contacts | the contact rate, or the matrix if stratified |
| A sexual or other structured network (`ss.MFNet`, ...) | compartmental | partner-change (or contact) rates between the groups the network distinguishes: sex, risk group, and age group if the model is stratified by age | the groups and the rate matrix |
| A non-exponential dwell time | compartmental | Erlang stages matching the mean and standard deviation; `stages=k` sets the number directly | the number of stages |
| A continuous agent attribute used by the model (age, CD4 count) | compartmental | bins: the stratification's bin edges if it has one, otherwise default bins | the bin edges |
| A function in a `rate=`, `p=`, `eligible=`, or `when=` slot | compartmental | nothing: it is evaluated on groups instead of agents, if it can be ([Functions](#functions-try-it)) | only if it fails |
| `n_agents < total_pop` | `abm` | scaling: each agent stands for `total_pop / n_agents` people and count results are multiplied up | the scale factor |
| A compartmental model with nothing agent-based | `abm` | nothing needed: agents are sampled into the states, with homogeneous mixing | `n_agents` if defaulted |

How network rates are obtained: the library builds the network once for a sample population at `sim.init()`, counts contacts per unit time between each pair of groups, and uses those as the mixing rates. For a network with partnership durations, the rate is partnerships formed per unit time times acts per partnership, so that the per-contact `beta` keeps its meaning. This discards concurrency, clustering, and partnership persistence, and the one-line note is the reminder.

How stages are chosen: `k = round(mean^2 / std^2)`, at least 1, with each stage exponential with mean `mean / k`. A distribution more variable than an exponential gets one stage and a note that its extra variance is not represented.

### What has no approximation

These stop a compartmental run with `E804 CapabilityError`:

- contact tracing, and anything else that needs to know who a specific person's contacts were;
- transmission trees and individual histories (`sim.transmissions`);
- custom code that only works on agents: a function taking `uids`, a `deliver=fn(sim) -> uids` delivery, a `step()` override that reaches for a per-agent attribute that is not a stratification.

```text
E804 CapabilityError: cannot run 'hiv' with method='ode'.
     The intervention 'partner_notification' traces the contacts of people who test positive,
     and a compartmental run has no individual contacts to trace.
     Options: use method='abm', or replace it with a screening rate among partners' groups.
```

## `methods()`: what can run this model

```python
hiv.methods()     # for one disease
sim.methods()     # for the whole sim: networks, interventions, and custom code included
```

```text
>>> sim.methods()
                               ode      ctmc     tau      binomial sde      abm
MFNet (sexual network)         approx   approx   approx   approx   approx   yes     approx: partner-change rates by sex/risk group
MixingPool (age matrix)        yes      yes      yes      yes      yes      yes
dwell 'acute' (lognormal)      approx   approx   approx   approx   approx   yes     approx: 4 Erlang stages
dwell 'aids' (gamma, shape 3)  yes      yes      yes      yes      yes      yes
eligible (age, sex)            yes      yes      yes      yes      yes      yes     uses stratifications only
function 'care_seeking'        ?        ?        ?        ?        ?        yes     tried at run time
partner_notification           no       no       no       no       no       yes     no: needs individual contacts
------------------------------------------------------------------------------------------
whole model                    no       no       no       no       no       yes     auto -> abm
```

The table is deliberately coarse: one row per ingredient that matters (each network, each non-exponential dwell time, custom functions, anything individual-level), one column per method, and each entry is `yes`, `approx` (with how), `no` (with why), or `?` for a function that has not been tried yet. Ordinary transitions and parameters that run everywhere are not listed. The last row is what `'auto'` will do.

## Functions: try it

Any slot accepts a function ([03-states-and-transitions.md](03-states-and-transitions.md)), and a condition may be a function of the sim ([08-interventions.md](08-interventions.md)). The library does not analyze functions in advance. It calls them with what the run method has:

- In an agent-based run, arrays have one entry per agent.
- In a compartmental run, the same names return arrays with one entry per group (one row per stratum): `sim.people.age` is the age group's midpoint, and a state such as `sir.I` is the count in each group.

A function that uses only things that exist in both (state counts, parameters, time, stratification coordinates) works in both without change:

```python
def care_seeking(sim):
    return 0.1 + 0.4 * (sim.people.age > 65)     # works on agents and on age groups
```

A function that asks for something agent-only stops the compartmental run with a message naming what it asked for:

```text
E804 CapabilityError: function 'care_seeking' asked for people.n_partners, which is a per-agent
     attribute and not a stratification, so it is not available with method='ode'.
     Options: stratify by it (hiv.stratify(n_partners=[0, 1, 5])), or use method='abm'.
```

So a lambda in an ODE model is fine, as long as what it does makes sense for groups. Under `'auto'`, functions are tried on groups once at `sim.init()`; if one fails, the model runs as agents and the printed reason says which function.

## How a model is built

At `sim.init()` the library turns the declarations into something runnable ([10-execution.md](10-execution.md) §Building the sim). No model code is interpreted line by line during the run.

- **Compartmental methods.** The declared transitions become a flat list of flows, each with a source, a destination, and a rate expression over the array of groups. The right-hand side of the ODE (or the propensities of the stochastic methods) is generated from that list. Declaration order does not matter, because every flow is evaluated from the same state.
- **Small models** are integrated with SciPy. This has no extra dependencies and no start-up cost.
- **Large models, and any model being fitted with gradients,** use JAX: JAX runs the generated right-hand side as fast array code, and the gradient of every result with respect to every parameter comes with it, which is what makes gradient-based calibration and sensitivity available ([13-inference.md](13-inference.md)). JAX is an optional dependency; without it, SciPy is used and gradients are by finite differences.
- **User functions** in a JAX run have to be traceable by JAX. One that is not makes the run fall back to SciPy, with a one-line note.
- **The agent-based method** is Starsim's: each declared transition becomes vectorized array operations on the agents in its source state, called from the loop.

Which solver is used is printed with the method (`method: ode (SciPy, 3 compartments)`) and can be set with `solver=`.

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

`R0()` returns a number computed from the next-generation matrix at the disease-free state, for models where that is well defined; it says so when it is not. Symbolic analysis beyond `equations()` (Jacobians, fixed points) is not part of v1.

Stages multiply compartments, and with stratification they multiply quickly. `describe()` reports the total number of compartments ([05-dimensions.md](05-dimensions.md) §Size).

## Stochastic compartmental specifics

Time-varying rates MUST work under every method. Under an exact continuous-time method this needs the thinning algorithm for inhomogeneous rates; freezing the rate over an interval is not acceptable without a note.

Continuous-time methods have no `dt` and so no rate-to-probability conversion. Results are resampled onto the sim's output grid by default, so that results from every method have the same time axis.

Extinction is reported for every method with integer state ([09-results.md](09-results.md)), since it is often the reason for choosing such a method.

## Agent-based specifics

The agent-based method is Starsim's, and its architecture is kept: typed per-agent arrays, persistent UIDs with a live view so that births and deaths do not invalidate references, vectorized operations, and one module abstraction.

| Declaration | In an agent-based run |
|---|---|
| `ss.State` | a boolean array over agents |
| a contact-mediated transition | for each contact on each network, compute the transmission probability and draw |
| a transition with `dur=` | sample a duration on entry and schedule the exit |
| competing rates out of one state | one draw per agent ([10-execution.md](10-execution.md)) |
| `sir.stratify(risk=[...])` | an agent property; continuous ones stay continuous |
| `ss.modify(target, multiply=, where=)` | a cached effective array, recomputed when the set of affected agents changes |
| `ss.outcome(...)` | a thinned, delayed draw from the transition's events |
| `ss.Compartments` | counts (optionally per patch) stepped alongside the agents |

- **Custom code** is ordinary Starsim: boolean state arrays, `dist.filter(uids)` to select with a probability, and direct reads and writes of other modules' state ([07-effects.md](07-effects.md)).
- **Aggregate quantities inside an agent-based sim** (an environmental reservoir, a mosquito population, wastewater) are `ss.Compartments` modules: states that are counts, with the same transition syntax, living in the same sim. This is how vector-borne and environmental transmission are written alongside agents.
- **Within-host state** (viral load, CD4 count) is a float agent property updated in `step()`. It can be a stratification when binned.
- **Contact tracing** applies only to networks, which keep their contacts; mixing pools do not.
- **Memory.** Arrays grow as agents are added, as in Starsim. Preallocating for the whole run is an optimization to benchmark before building, and would not change anything the user writes.
- **Small `n_agents`** is not warned about. Scaling to `total_pop` is reported as in the table above.

## Agreement between methods

The aim is that **simple models agree across methods within a stated tolerance**, and this is tested rather than assumed: for each simple model in the corpus, [18-conformance.md](18-conformance.md) runs every method that `methods()` marks `yes` and compares the results (stochastic methods by their mean over replicates, at large population size).

For complex models, exact equivalence between methods is neither provable nor the aim. An agent-based HIV model and its ODE approximation are expected to differ; what the library guarantees is that it reports what was approximated.

## Deferred: mixing methods within one model

Some questions need most of a model to run as smooth flows while a few rare transitions happen as whole, random events: for example, importation of a single case into a population where the ODE says the epidemic is over. This would be spelled as a `method=` keyword on the transition (`ss.transmission('S -> I', beta=1e-6, method='ctmc')`), defaulting to following the run. It is not in v1. Until then, such questions use a stochastic method for the whole model.

## Rejected

- **A separate model class per run method.** The same disease should be writable once.
- **Refusing to approximate.** A user who asks for an ODE of a network model wants the standard coarse version. Say what was done; `approx=False` is there for those who want the stop.
- **A menu of network-to-ODE closures.** One standard approximation per ingredient is enough. Someone who needs the network's exact behavior can run agents.
- **Agent-based as the default for every model.** A two-line SIR should run as the three equations it is, instantly. Models that need agents get them automatically.
- **Analyzing user functions in advance.** Trying them is simpler and gives a clearer message.
- **Making the user write the derivative function.** It is positional, unchecked, and cannot be stratified automatically.
