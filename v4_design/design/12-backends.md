# Solvers: choosing, approximating, and building

How a model is run: the solvers, how the default is chosen, what happens when a model is asked to run in a way it does not natively fit, and how the runnable model is built.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md), [best-practices/compartmental.md](../best-practices/compartmental.md), [best-practices/stochastic-compartmental.md](../best-practices/stochastic-compartmental.md), [best-practices/agent-based.md](../best-practices/agent-based.md), and [best-practices/metapopulation.md](../best-practices/metapopulation.md).


## The default: `solver='auto'`

> **`'auto'` picks the simplest, fastest representation that runs the model without approximating its mixing or dropping anything individual-level: a deterministic ODE if every piece of the model has a compartmental form, otherwise agents. The choice is printed as the run starts, and `describe()` gives the reason.**

```python
import starsim as ss

S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=10),
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

The solver is set with `solver=`, given to `ss.Sim(...)` or to `sim.run(...)` (the second wins). A sim built for one solver is built again for another when `sim.run(solver=...)` asks for it, which is how one sim is run several ways; `sim.new(solver=...)` makes a fresh copy with another instead. The printed lines can be hidden with `verbose=0`.

`'auto'` chooses between the ODE and agents. It never picks a stochastic compartmental solver; a user who wants one asks for it by name ([The solvers](#the-solvers)).

**The one approximation `'auto'` makes** is to represent a non-exponential dwell time by Erlang stages, and it says so (a fixed delay, `ss.constant`, is not approximated: it is a conveyor of whole steps, [below](#a-fixed-delay-and-a-minimum-stay)):

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=ss.weibull(c=1.4, scale=ss.days(8))),
    init_prev = 0.01,
)
```

```text
>>> ss.Sim(sir, unit='day', stop=100).run()
solver: ode (every part of the model has a compartmental form)
  note: recovery has a weibull dwell time, approximated by 2 Erlang stages (mean 7.29139 days kept; std 5.16 instead of 5.28 days)
sir: R0 = 2.19, with 100 infectious at the start
```

A lognormal recovery time does not by itself turn a simple model into an agent-based one. `'auto'` never approximates the mixing (a network stays a network) and never drops individual-level structure (per-person attributes, histories, custom per-agent code); a model with any of those runs as agents.

### What makes a model agent-based

A model runs as agents under `'auto'` if it contains any of these:

- an individual-level network (`ss.RandomNet`, `ss.SexualNet` and other networks of partnerships, `ssl.StaticNet`, a maternal network), or a mixing pool between some people only (`src=`, `dst=`) or with a number of contacts that differs between people;
- `n_agents=` given to the sim or to `ss.People`;
- a per-agent function (`fn(sim, uids)`) or a per-agent attribute that is not a stratification (viral load, an individual's partner count);
- a distribution on a per-person quantity other than a dwell time (`beta`, `rel_sus = ss.lognorm_ex(mean=1, std=0.3)`) that has not been binned into a dimension;
- a module written as code that needs agents: a disease, network, demographic module, intervention or connector written as code, unless it says it works on groups with the class attribute `compartmental = True`, any module with `compartmental = False`, and an analyzer or plain `ss.Module` that keeps a value for each person (`define_states()`), which only agents have. Otherwise an analyzer, a plain `ss.Module` (e.g. an ODE of its own in NumPy), a function given as a module, and the functions and `step()` of a declared module run beside the compartments without changing the solver; code is never run to find out (decided 2026-10-20), so if one of them asks for something only agents have, the run stops and says so (see [Functions](#functions-try-it));
- something that needs individual histories: contact tracing, transmission trees, the infection log (`analyzers='infection_log'`), or what each person remembers of the strains they have had scaled by an attribute of their own (`ss.immunity(due_to=strains.last, scale='nab')`, [below](#strains));
- strains with courses of their own that affect each other (`immunity=`, or a shared state such as one cancer per person), whatever `approx=` (decided 2026-10-20: the approximation from each strain's shares is gone; below);
- a clock: the time in a state or a set of states (`X.time`, `(A | B).time`, `X.progress`), the time since an event, or a dimension of kind `'clock'`; and the values of each person declared in a module, `ss.attr` (decided 2026-10-20; fixed delays, minimum stays, and Erlang stages are the compartmental forms of timing), apart from a trait (a distribution drawn once for each person, with no modifier), which a compartmental solver named takes at its mean with a note, as it does a parameter that is a distribution (`approx=False` refuses);
- a table by bins that split the model's groups (`ss.by(dict(age=np.arange(100), value=...))` on 5-year age groups), which agents read for each person; a compartmental run does not average it over each group, but stops and says to give the values by the model's groups;
- an initial value (`init_prev=` or `init=`) by a property of people that is not a dimension, or a function of the agents;
- a delivery that follows each person's own history (decided 2026-10-20): a queue (`queue=`), a wait from the trigger (`delay=`) or for a test's result, how often or how long ago an event happened (`.count`, `.since`; whether it did, `.ever`, is a level of people), protection that fades by a function of the time since delivery, events combined (`a.positive | b.positive`), or one delivery for several diseases (`disease=[...]`).
- each person's mother or children: `ss.mother()`, fetuses (`ss.Pregnancy(fetus=True)`), and `ss.Lactation` (a maternal or postnatal network is above); a compartmental solver named explicitly stops with a `CapabilityError` for these, before the run (decided 2026-10-20);
- `ss.Pregnancy`, which has only a small compartmental form, used when a compartmental solver is named ([below](#pregnancy-as-compartments)).

Or the user asks: `solver='abm'` runs any model as agents, for example to study elimination with a simple model.

Everything else has a compartmental form: homogeneous mixing, an `ss.MixingPool` of everyone and contact matrices, stratifications, exponential and Erlang dwell times (exactly), fixed delays and minimum stays (`ss.constant`, `after=`, `before=`: by a conveyor of whole steps), and other dwell times (by stages, with the note above; several out of one state for the same people race, as a compartment for each combination of their stages: [03](03-states-and-transitions.md)), `ss.modify` with conditions on strata, interventions whose eligibility uses only strata (deliveries on a schedule or triggered by one event, with coverage and capacity; tests with immediate results; cures; exponential waning; `.ever`), states that are not people (reservoirs, vectors), and functions that use only counts, parameters, time, and strata. A state that is not people never makes a model agent-based.

#### Pregnancy as compartments

`ss.Pregnancy` runs as agents under `'auto'`. A compartmental solver named explicitly runs a small form of it, with a note that says what it leaves out (decided 2026-10-20): its own states (`NP`, `P`), conception, abortion and maternal deaths, the delivery at the end of `dur_pregnancy` (a fixed delay, or Erlang stages for a distribution), and `n` children for each delivery (`ss.birth(per=preg.delivery)`, also into another module's state). Its losses do not read the gestational age (`P.time`): the loss rate is split between miscarriage and stillbirth by the share of the mean gestation below `loss_threshold`. Its burn-in (`burnin=True`) puts the women pregnant at the start evenly over gestation, less the abortions and losses they would have had. It has no fetuses, no links between mothers and children, no lactation, no preterm risk, and no deaths at delivery. Against agents (`burnin=False`, weekly to three-monthly steps), deliveries, births and the number pregnant agree within 3%, and people are conserved exactly; with burn-in, the number pregnant at the start agrees within 10%. A module declares such a form by defining `compartmental_form(sim)`, which is called only when a compartmental solver is named; solver choice knows nothing else about pregnancy. It is a module's approximation of itself; a compartmental form written outside Starsim (a shape of waning, a dwell more variable than an exponential as a mixture of exponential branches, an outcome that depends on the time spent as bins of it) is written as declared states instead ([08-interventions.md](08-interventions.md#products), decided 2026-10-22).

The reasons are kept in one record, which `describe()` and the `CapabilityError` read.

For simple models the solver is a free choice made when the sim is built. For complex models it is in effect part of the specification: a model built around a sexual network is an agent-based model, and running it another way gives a different, coarser model.


## The solvers

```python
ss.Sim(sir, solver='auto')        # the default
ss.Sim(sir, solver='ode')         # deterministic compartmental
ss.Sim(sir, solver='binomial')    # chain-binomial, discrete time
ss.Sim(sir, solver='gillespie')   # exact stochastic, continuous time
ss.Sim(sir, solver='tau')         # tau-leaping: stochastic, close to exact, for a run of many events
ss.Sim(sir, solver='abm')         # agent-based
```

The diffusion approximation of earlier drafts (`'sde'`) is gone: it resolved competing exits in declaration order at coarse steps, agreed worst with the ODE, and had no user. An unknown solver raises an error that lists the solvers.

The model object is the same in every case, and whatever the solver, a sim is one trajectory; many runs are `ss.parallel(sim, n_runs=100)` ([09-results.md](09-results.md#one-run-many-runs)). The exact solver is called `'gillespie'` because that is the name the field uses for it. An integro-differential solver (`'ide'`, for arbitrary dwell times without stages) is deferred.

| Solver | Time | State | Good for | Loses accuracy when |
|---|---|---|---|---|
| `ode` | continuous | real-valued | large populations, fast fitting | counts are small or extinction matters |
| `gillespie` | continuous | integer | exact stochastic dynamics (for rates that are constant within each quarter of a step) | values vary quickly within a step |
| `tau` | continuous | integer | large stochastic runs, close to exact | rates change quickly within a leap |
| `binomial` | discrete | integer | simple stochastic models on a fixed step | `rate * dt` is large |
| `abm` | discrete | individuals | networks, heterogeneity, histories | `rate * dt` is large; cost grows with `n_agents` |

Some of these conditions are checked and reported as one-line notes, each naming the quantity involved and what would not have the problem: after an ODE run, `ss.errors.SmallCounts` (it starts with only a few infected) and `ss.errors.ExtinctionPossible` (the number infected falls below one and rises again); at `init()`, for a solver that takes whole steps (`abm`, and `binomial` with `substeps=` given), `ss.errors.LargeStep` (a transition's `rate * dt` is large), which also says how far whole steps can be from the ODE and to run `sim.check_dt()`. They are notes, not errors: the run goes ahead and the user is told.

`'gillespie'` is always exact: it takes the events one at a time, however many there are, and `sim.solver` is the solver that was asked for. Places that do not affect each other within a step (mixing only within each place, with no commuting, and one place if the model has none) are run place by place: after each event, only the rates of its place are worked out again, so an event costs about 0.1 µs however many places there are, and a place where nothing can happen is skipped until something changes it (SimInf's 16,000 herds and 5 million events take about 4 s). Diseases tracked jointly, whose large program has mostly empty compartments, are stepped by the compartments that hold anyone. Groups that affect each other (strata that mix by a contact matrix, or places that people commute between) have every rate worked out again after each event: about 4 µs an event for 20 groups, and 10 to 60 µs for 100. For millions of events in groups like these, `'tau'` (tau leaps, each short enough that the rates change little over it) or `'binomial'` takes a fraction of a second, at a small bias that is noted in [the approximations](#the-standard-approximations):

```text
>>> ss.Sim(sir, total_pop=1000, solver='gillespie', unit='day', stop=100).run()
solver: gillespie (requested)
```

Small counts, where extinction questions arise, are runs of few events, for which the exact solver is fast. The tau leaps choose their own size from the rates, so a step that is large for them needs no note of its own.

## Approximations

> **`'auto'` approximates nothing except a non-exponential dwell time, which it represents by Erlang stages and says so. When a solver is requested explicitly and the model does not natively fit it, the library uses the standard approximation where one exists and says so in one line. It stops with an error only when there is no sensible approximation.**

```text
>>> ss.Sim(hiv, networks=ss.RandomNet(n_contacts=4), solver='ode', unit='year', stop=40).run()
  note: RandomNet approximated by homogeneous mixing with 4 contacts per year (its mean number of contacts)
solver: ode (requested)
  note: acute has a lognorm_ex dwell time, approximated by 6 Erlang stages (mean 0.25 years kept; std 0.102 instead of 0.1 years)
```

A network model run as an ODE is a different, coarser model, not an invalid one, and the user who typed `solver='ode'` has asked for it. A network of lasting partnerships (`ss.SexualNet`, `ssl.DegreeNet`) uses the partner-change model of [`06-routes.md`](06-routes.md) ("Networks of partnerships"), with a note; `'auto'` keeps it as agents.

`approx=` says what to do about approximations:

| `approx=` | Effect |
|---|---|
| not given (the default) | Approximations are made and each prints a one-line note |
| `approx=True` | Approximations are made and the notes are kept in `sim.notes` without being printed: the user has said they are expected |
| `approx=False` | Any approximation is an error (`CapabilityError`, naming the piece and the approximation that would have been used) |

- Under `solver='auto'`, `approx=False` rules out the Erlang stages too: a model with a dwell time that has no exact compartmental form then runs as agents, where the dwell time is sampled directly, and the printed reason names the transition.
- `verbose=0` hides everything printed at run time, the notes included.
- Whatever is printed, the notes are kept in `sim.notes` (and in the model spec's metadata), so they survive into saved output; `sim.describe()` shows what would be approximated as compartments.

### The standard approximations

| The model has | Solver | Approximated by | Reported |
|---|---|---|---|
| A random or static network (`ss.RandomNet`, `ssl.StaticNet`) | compartmental, explicit | homogeneous mixing with the network's mean number of contacts (for a static network, the mean degree of its graph) | the contact rate and where it came from |
| A network of lasting partnerships (`ss.SexualNet`, `ssl.DegreeNet`) | compartmental, explicit | the partner-change model ([`06-routes.md`](06-routes.md)): partnerships formed at `n/D̄` per person, each transmitting with its chance over its whole length, from partners drawn from the population; its sides, and the property it matches or counts partners by, must be dimensions. Against agents (20 runs of 10,000, prevalence at 10 and 20 years, ODE over agents): risk groups with 6-month partnerships 1.00 and 0.99; HIV-like, 5-year monogamous partnerships 1.11 and 1.25 (infection kept within pairs, which it spreads); gonorrhea-like with Poisson partners at a time, 0.19 and 0.13 (concurrency, which it leaves out; 1.0 and 1.5 near the threshold with one partner at a time); a curable infection that passes within a partnership faster than it is cleared (SIS cleared in a year, 80 acts a year at 0.08 and 0.04 per act, 6-month partnerships one at a time; one run of 20,000), 0.50 and 0.52 (reinfection within a partnership, which it leaves out: it counts at most one transmission in each) | the rate partnerships form, and under the route in `describe()`, the chance over a partnership in each direction |
| A modifier of a transmission on a route with a share for each edge (`share=lambda net: ...`) | compartmental, explicit | none: `CapabilityError` (a share that is a number, a time series, or by a dimension is exact) | |
| A dwell time that is not exponential, Erlang, or fixed (`ss.constant`) | compartmental, explicit or `'auto'` | Erlang stages matching the mean and, as closely as possible, the standard deviation; `stages=k` sets the number directly | the number of stages, and the mean and standard deviation that result |
| A distribution on a per-person quantity that is not a dwell time (`beta`, `rel_sus`) | compartmental, explicit | its mean. To keep the variation, the user bins it into a dimension (`sir.stratify(rel_sus=[...])`), which is exact for the bins and needs no note | the mean used |
| A continuous agent attribute used by the model (age, CD4 count) | compartmental, explicit | the bins of a stratification by it (a function then sees each group's midpoint); without one, `CapabilityError`, saying to stratify by it | |
| A function in a `rate=`, `p=`, `where=`, or `when=` slot | compartmental | nothing: it is evaluated on groups instead of agents, if it can be ([Functions](#functions-try-it)) | only if it fails |
| `n_agents < total_pop` | `abm` | scaling: each agent stands for `total_pop / n_agents` people and count results are multiplied up | the scale factor |
| A compartmental model with nothing agent-based | `abm` | nothing needed: agents are sampled into the states, with homogeneous mixing | `n_agents` if defaulted |
| Many events | `tau` | tau leaps, each short enough that the rates change little over it: close to the exact solver, but biased (in 1e7 people, the peak of an SIR about 0.7% high, several times the run-to-run spread) | nothing: it is asked for by name |
| A value that varies smoothly in time (`ss.seasonal`, `ss.timeseries`) | `gillespie` | held for a quarter of a step at a time, at its middle; `approx=False` makes it an error | that it is held |

"Explicit" means the user named the solver. Under `'auto'` only the dwell-time row applies; the other rows make the model agent-based instead.

How network rates are obtained: a random network gives its mean number of contacts (`n_contacts`, or the mean of its distribution), and a static network the mean degree of its graph, as a homogeneous contact rate, so that the per-contact `beta` keeps its meaning. This discards concurrency, clustering, and partnership persistence, and the one-line note is the reminder.

How stages are chosen: `k = round(mean^2 / std^2)`, at least 1, with each stage exponential with mean `mean / k`. The mean is kept exactly, and the note prints the standard deviation that results next to the one asked for. A distribution more variable than an exponential gets one stage and a note that its extra variance is not represented; one with a coefficient of variation above 3 (e.g. a Weibull with shape below about 0.4, whose one stage keeps a mean dominated by a tail few people reach) makes `'auto'` run the model as agents, naming the transition, and a compartmental solver that is asked for warns; to keep it, write it as a mixture of exponential branches decided on entry (`ss.Transition(I, R, p=0.1, dur=ss.days(820))` and `ss.Transition(I, R, p=0.9, dur=ss.days(20))`: a mean of 100 days and a CV of 2), which every solver runs exactly. `ss.gamma` with an integer shape is already Erlang and is exact. The same expansion is used by every compartmental solver, stochastic ones included.

### A fixed delay and a minimum stay

A fixed delay (`ss.constant`, as odin's `delay()`) has no good Erlang form: its stages converge as one over their number, and their ODE becomes stiff (200 stages for odin's 14-day latent period took 1.9 s and were still 1.6% off). Every compartmental solver takes it as a conveyor of whole steps instead ([03-states-and-transitions.md](03-states-and-transitions.md#a-fixed-delay)): `K + 1` compartments for a delay of `K` steps, moved along one at the end of each step, with no rate between them. The people due to leave leave in the next step:

| Solver | When the people due to leave a conveyor leave |
|---|---|
| `ode` | at a constant rate over the step, so the release of the people who arrived over a step `K` steps before is that of the delay differential equation, to second order (odin's SEIR within 0.25% at a step of a day, 0.06% at half a day, in 8 ms); the state's other exits act on them until they leave, with the release lowered to what they leave, so a competing exit acts over the whole delay |
| `binomial` | at the start of the step, once the step's rates are worked out from the people as they were, as agents' scheduled exits come first; so it agrees with agents step for step |
| `gillespie`, `tau` | at the middle of the step, the mean of the ODE's release |
| `abm` | each at the start of the step their delay ends in, rounded to whole steps at random (keeping its mean) |

The stock at each time point is the same in each: exact for a delay of whole steps. A delay that is not a whole number of steps is rounded to one, with a note. A minimum stay (`after=`) and a limit (`before=`) use the same conveyor, with each transition acting only on the people who have stayed long enough (or not too long); a stay on a transmission, or out of a state that splits on entry, has no compartmental form yet. The kernels, the particle filter, and the run in JAX all take conveyors; a model without one pays nothing for them.

Clocks are not conveyors (decided 2026-10-20): a rate, `p=`, or condition that reads the time in a state or since an event, and a dimension of kind `'clock'`, run as agents, and a compartmental solver that is asked for raises `CapabilityError` before the run. The duration-dependent hazards they would have given are written with Erlang stages or fixed delays, and an outcome that depends on the time spent (HPV's chance of CIN given the time infected) as bins of that time: a split on entry into a few states, each with its fixed dwell and its own chance of each outcome (`ss.Transition(I, [I1, I2, I3], dur=0, p=w)`, then `ss.Transition(I1, [C, R], dur=ss.constant(d1), p=[f(d1), None])`, ...), exact for the bins.

### Strains

`ss.Disease(strains=ss.Strains(...))` ([05-dimensions.md](05-dimensions.md#strains)) is one front end; the engine chooses how to run what is declared, and does not tell the modeler except in `describe()`. Representations need not be equivalent, and are not promised to be: a model that asks for something a solver cannot do gets a `CapabilityError`, never a silent change in coinfection, memory, or counts.

| What is declared | Representation | Solvers |
|---|---|---|
| Every state shared, complete protection (the default), or protection due to a shared state | One level per strain carried: levels × states; protection by level and strain | All, with gradients under the ODE |
| Every state shared, sets carried (`max_strains` above 1; superinfection by partial protection due to an infected state; emergence by `move=`) | Sets carried × states, the sets made up to `max_strains`; an outcome for each strain (`per='strain'`) as a branch by level and a change of level | All, with gradients under the ODE |
| Protection due to memory (`strains.last`, `strains.ever`) | Each agent's last strain, or the strains had, as bits | A dimension of the disease's people: none and each strain (last), or each set of strains (ever, `2^k` levels), moved along as people leave the states they carry strains in; with `scale=` (an attribute), agents only |
| Courses of their own, independent | One declared disease per strain, side by side: k copies of the states, not `2^k` | All (below) |
| Courses of their own, coupled (`immunity=`, a shared state) | Agents, one disease per strain | Agents; a compartmental solver raises `CapabilityError` |

Under networks, protection against the incoming strain is drawn for each pair that would transmit (HP14), with a random stream of its own, so it is evaluated only for those reached; when several strains reach one person in a step, each contact is as likely as the others to be the one that infects. A product's efficacy by strain multiplies each strain's force of infection under every solver.

### Strains with courses of their own, side by side

Strains with courses of their own are not crossed (unless a disease outside them is tracked jointly with them, [07-effects.md](07-effects.md)): strains that do not affect each other run side by side, with the sum of their compartments rather than the product. Each member's results are exactly as they would be alone; the union results are `1 − Π_k (1 − f_k)` of each group, with `f_k` member `k`'s share of it (the flows are sums, as agents); and the seeding (`init_prev=`, `split=`) seeds `init_prev × split_k` of each group in each member, so some are seeded with two (at 10% and 4 members, about 0.4%), which is noted.

Strains that affect each other (`immunity=`, a shared state) run as agents: `auto` runs them so, saying why, and a compartmental solver raises `CapabilityError`, whatever `approx=` (decided 2026-10-20; the approximation from each member's shares, 5 to 8% low against the exact form at 2 members, is gone). As compartments, the couplings are written as `ss.modify(..., where=)` between members, which tracks their states jointly (exactly, with the product of their compartments).

### What has no approximation

These stop a compartmental run with `CapabilityError`:

- contact tracing, and anything else that needs to know who a specific person's contacts were;
- transmission trees and individual histories (the infection log, `analyzers='infection_log'`);
- custom code that only works on agents: a function taking `uids`, a `deliver=fn(sim) -> uids` delivery, a `step()` override that reaches for a per-agent attribute that is not a stratification;
- protection due to the strains a person remembers scaled by an attribute of their own (`ss.immunity(due_to=strains.last, scale='nab')`);
- strains with courses of their own that affect each other (`immunity=`, a shared state), a clock, an attribute (`ss.attr`) other than a trait, and a table by bins that split the model's groups;
- a value read from each person's own properties that is not a dimension, including an initial value (`init_prev=ss.by(dict(age=[0, 15], value=[0, 0.1]))`).

```text
CapabilityError: cannot run 'hiv' with solver='ode'.
     The intervention 'partner_notification' traces the contacts of people who test positive,
     and a compartmental run has no individual contacts to trace.
     Options: use solver='abm', or replace it with a screening rate among partners' groups.
```

## What can run this model

```python
sim = ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), unit='day', stop=100)
sim.describe()    # networks, interventions, and custom code included
```

Under the solver, `describe()` lists one line per ingredient that matters (each network, each non-exponential dwell time, each function, each module that needs agents, `n_agents=`), saying whether it runs as compartments: `yes`, `approx` (with how), or `no` (with why). For an HIV model with a sexual network, a lognormal dwell time, and partner notification (`trace=`):

```text
solver       abm                                    [guessed: SexualNet is an individual-level network]
             network: sexualnet: approx as compartments (the partner-change model)
             dwell time: acute (lognorm_ex): approx as compartments (Erlang stages)
             module: partner_notification: no as compartments ("partner_notification" traces contacts, which only a network of agents keeps)
```

Ordinary transitions and parameters that run everywhere are not listed. Nothing is run to fill it in (decided 2026-10-20): a module says whether it works on groups with `compartmental = True` or `False`, and functions are listed as running on groups, to be checked when they run (see below). When a solver was asked for, the last line says what `'auto'` would choose, and why; `sim.kernel_reason()` says, separately, why a compartmental run is taken a step at a time rather than in one call to the kernels. Near the end, `plan` says what runs where: the declared modules in the kernels (the whole run in one call, or one step at a time and why, with diseases tracked jointly as one program) or as agents; modifiers that the kernels take into their tables; and the modules that run in Python at their place in each step. It describes the run; it is not a separate set of capabilities to declare.

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

So a lambda in an ODE model is fine, as long as what it does makes sense for groups. Under `'auto'`, functions are not tried in advance: a function that does not work on groups stops the run with the error above (at `sim.init()` for values, which are first evaluated there), and the fix is `solver='abm'`, a stratification, or `compartmental = False` on the module.

## How a model is built

At `sim.init()` the library turns the declarations into something runnable ([10-execution.md](10-execution.md) §Building the sim). No model code is interpreted line by line during the run.

- **Compartmental solvers.** The declared transitions become a flat list of flows, each with a source, a destination, and a rate expression over the array of groups. The right-hand side of the ODE (or the propensities of the stochastic solvers) is generated from that list. Declaration order does not matter, because every flow is evaluated from the same state.
- **Every compartmental solver** runs in the same kernels, from one flat program of the flows: as plain Python, or compiled by Numba. Both are the same code, and give the same results with the same seed (the draws are the same code, `ss.perflib.draws`, compiled or not), so the only difference is speed. Compiling takes about a minute, once per install, so it is opt-in: by default (`ss.options.compile = 'auto'`), the kernels are compiled first if the runs in a session would take longer as plain Python than compiling takes (judged from their steps, substeps, groups, flows and mixing; a stiff ODE's substeps are judged from its fastest rates at the start), and otherwise the run is taken as plain Python. Either way it says so in one line, whatever `verbose` is, since a minute of silence looks like a hang: when it compiles, and (once a session, for a run of more than a second) how long a plain run would take each way. `ss.compile()` compiles them ahead of time (in a Docker image, in CI, or before a calibration); `'always'` and `'never'` force either. Whatever is compiled says so as it starts, once a session for each part, in one line with how long it takes on a laptop (unless `ss.options.verbose` is 0): `Compiling the compartmental solvers with Numba, estimated time: 30-90 s`. For Numba this is said only when it compiles, not when it loads from its cache, for the parts that take a second or more (the kernels' entry points, the agent solver's passes, and each function or method of `ss.jit`); JAX traces again in each session, so its parts (a run, the log-density, NUTS) say so each session, though its own cache on disk makes them quicker after the first. The compiled kernels are kept in Numba's cache, beside the package, or in the user's cache folder if that cannot be written to. Plain Python is fast enough for small models (a one-group SIR over a year in days takes about 0.02 s, 16 age groups about 0.1–0.3 s) and slow for large ones (50 places, 1.5–6 s; 260 districts over 16 years, about 10 minutes, so that one compiles). The ODE takes fixed RK4 steps with automatic substeps, whether or not gradients are asked for. A value that is a function of the sim is called once a step and held within it, unless it gives an expression of the model's states (`I.n`), which is put into the program in its place.
- **A whole run in one call.** A compartmental sim with nothing that needs a step of its own (deliveries on a schedule and modifiers that depend only on time and states are worked out before the run) is run from start to finish in one call to the kernels; otherwise the kernels take it a step at a time from Python. `sim.kernel_reason()` says why a run is stepped.
- **Deliveries.** `ss.Delivery` is one class, which users subclass: its schedule, coverage, capacity, conditions, products and results are shared, and its own methods are how it runs as agents, where a selection is a boolean per agent. In a compartmental run it also makes a small adapter, `ss.solvers.CompartmentalDelivery` (`delivery.compartments`, in `solvers/deliveries.py`), for what differs there: a selection is the share of the people in each state and group, so a capacity takes whole groups in the order of the priority (pro rata within a tie), moving people moves shares of the compartments between groups and carries the step's events along, and a cure or a test acts on shares. The adapter also gives the kernels the deliveries they take in a whole run.
- **Gradients, and many runs with different values,** run through JAX, which runs the same program (the same RK4 steps) as array code. JAX and BlackJAX are dependencies, imported only when first used, so `import starsim` does not wait for them. The run in JAX and the kernels' ODE are two implementations of the same steps, kept in agreement by a test (to 1e-9 on every test model). Both stay, since the kernels' rates are needed anyway for the stochastic solvers, and only the kernels take step-by-step runs, families of strains, and movement registers. A first run in JAX also takes 0.5–0.7 s a session to import and compile, though with many places it is faster than the kernels (0.5 s against 4.2 s for Ghana's 260 districts). `sim.gradient()` differentiates the run by forward-mode automatic differentiation, and the steps either side of it (parameters into the program, and the run into results) by central differences, so its gradients are exact to about 1e-10. `sim.log_density()` and NUTS have the whole log-density (the run, the results with data, the likelihood, and the priors) as one function in JAX. That is what makes gradient-based optimizers and NUTS available for ODE models ([13-inference.md](13-inference.md#gradients)). Gradients are for `solver='ode'` only, for a sim that the kernels run from start to finish, and for rates written as expressions (a rate that is a Python function of the sim has none); the stochastic solvers and agents have none. JAX is chosen by `sim.gradient()`, `sim.log_density()`, gradient fits, `ss.parallel()` with 300 or more sets of values, and for a single run by `ss.solvers.choose_engine()` (as for the chain binomial, below: JAX if the kernels would take more than half a second, e.g. Ghana's 260 districts, 0.5 s against 4 s, agreeing to about 1e-14), never by `integrator=` (`integrator='jax'` is an error).
- **The chain binomial and tau-leaping in JAX** (`ss.perflib._chain`): the same rates (`ss.gradients.Run.rates()`) with all of a step's draws in one native call (the exact one-uniform draws in Numba, called through an XLA FFI custom call), as one compiled `lax.scan`, `vmap`-able over seeds and parameter sets with the same draws as single runs. It follows `ss.kernels.run()` draw for draw: stages of a dwell time, splits, maps, movers, credits, reservoirs, conveyors, deliveries at the start of a step (their numbers reached and eligible counted as the kernels count them), movement registers at its end, substeps (a call each), tau-leaping (a `lax.while_loop` of leaps, whose length JAX works out from the rates), and diseases tracked jointly (the crossing's program). It agrees with the kernels in distribution, not by seed. Which engine runs a compartmental model (stochastic, or the ODE) is `ss.solvers.choose_engine()`: `sim.jax = True` or `False`, or by default (None) JAX if the run would take the kernels more than half a second (`kernel_seconds()`), since JAX takes 0.3–0.5 of the kernels' time once started but about half a second to start in a session. The choice depends on the model only, so a seed gives the same run whenever it is run, and the runs of a `MultiSim` are taken together in JAX (in batches, reduced to the results kept) whenever a single run would be. A sim it cannot run (`reason()`, e.g. code between steps, or a handler that reports the wrong XLA FFI API version) runs on the kernels.
- **The agent-based solver** is Starsim's: each declared transition becomes vectorized array operations on the agents in its source state, called from the loop.

`integrator=` is the scheme of the ODE: `'rk4'` (the default, with enough substeps, chosen at the start of each step, that no compartment turns over by more than a quarter in one, nor grows by more than a quarter through its inflows, as an epidemic's infected do, nor does a count that a rate reads, over all its compartments (a fixed delay's first slot fills from empty in each step); a compartment holding less than 1e-9 of everyone is only starting to fill, and does not count: within about 1e-4 of the exact solution on the textbook models at a daily step) or `'euler'` (for matching frameworks that take Euler steps, as the Atomica importer does). Any other value is an error at `sim.init()`. It has nothing to say about which solver runs the model, which is `solver=`. `substeps=` sets the number of substeps per step, for the ODE schemes and for `binomial`. The chain binomial's default is enough substeps that rate × dt is at most 0.1 for the fastest rate out of a compartment of people known at the start (at most 100; a fixed delay, a conveyor of whole steps, does not count), since each substep is off the ODE by an amount of order rate × dt (ST55: an acute stage of 1.7 months at monthly steps took the share ever infected from 9% to 51%; CV25: a daily SEIR peaked 4 days late). It moves people through as many stages of a dwell time as they pass in a substep. `substeps=1` takes whole steps, as discrete-time chain-binomial models do (the importers of odin, camdl, EMULSION, EpiModel, epydemix and flepiMoP set it).

## States that are not people

A state that is not people (bacteria in water, mosquitoes; [03-states-and-transitions.md](03-states-and-transitions.md)) is an ordinary `ss.State` in the same disease. The library works out that it is not people because no transition connects it to the disease's first state, prints that guess in `describe()`, and `ss.State(people=False)` says it explicitly. It holds a plain count, is left out of `total_pop`, and is not split by people dimensions (age, sex, risk) when the model is stratified.

| Solver | What such a state is |
|---|---|
| `ode` | One more real-valued compartment (or one per patch). Nothing distinguishes it from the others except that it is not counted in `N` |
| `gillespie`, `tau`, `binomial` | A real-valued count, whose flows are their expected values over each step (or each event's interval, under `gillespie`), while people are drawn. This is SimInf's split between continuous-time events for animals and a deterministic environmental pressure, and it keeps the cost of a reservoir independent of its size |
| `abm` | A count, never agents. People are agents and the reservoir stays a number, advanced over each step as an ODE (by RK4) with the people as they were at its start ([10-execution.md](10-execution.md#states-that-are-not-people)) |

Rates that use the count are written with `.n` (`beta * W.n / (W.n + K)`, `shed * I.n`) and are used exactly as written under every solver: there is no contact rate and no network in them. A count set each step by `ss.update` is worked out from the model at the start of the step and set after the transmissions, as agents only (decided 2026-10-20): `auto` runs a model with one as agents, and a compartmental solver raises. Under `'auto'` a model with states that are not people runs as an ODE unless something else makes it agent-based; adding `n_agents=` makes the people agents and leaves the count a count.

## Compartmental specifics

```python
sir.equations()     # the ODE system, with symbolic parameter names
sir.R0()            # optional numeric helper, with the mixing of its sim if it is in one
```

```text
>>> sir.equations()
dS/dt = -beta * S * I / N
dI/dt =  beta * S * I / N - I / dur
dR/dt =  I / dur
```

`equations()` is a method of the disease and prints parameter names, not numbers. Inline parameters take the name of their slot (`beta`, `dur`), qualified by the transition's name where two would clash (`dur_recovery`). A dwell time represented by stages prints one equation per stage (`dI_1/dt`, ...). A stratified disease prints the equations of one group, not an indexed form over the groups. Nothing in the declaration is ODE-specific: with a network or mixing pool supplying contacts, a contact-rate symbol appears in the infection term and `beta` is per contact.

`R0()` returns a number computed from the next-generation matrix at the disease-free state, for models where that is well defined; it says so when it is not. By definition R0 is in a population with no immunity: at the disease-free state everyone is in the first state, each group keeping its size, whoever the model starts immune (`init=dict(R=0.9)`). The line a run prints as it starts gives R0, and also the reproduction number at the start, with the share susceptible, when it differs by more than 1% (`measles: R0 = 15.2 (R = 1.52 at the start, with 10% susceptible), with 100 infectious at the start`). The infected compartments include the counts infection passes through (a reservoir, mosquitoes infected and infectious), and the other counts that an inflow fills are at their disease-free equilibrium (mosquitoes not infected: emergence over mortality); each passage through a vector is a generation, so for Ross-Macdonald `R0()` is the square root of the classical value. Symbolic analysis beyond `equations()` (symbolic Jacobians, fixed points, eigenvalues) is not part of v1. Gradients of results with respect to parameters are a different thing and are in scope: they come from automatic differentiation of the ODE run ([How a model is built](#how-a-model-is-built)).

Stages multiply compartments, and with stratification they multiply quickly ([05-dimensions.md](05-dimensions.md) §Size); `describe()` lists the dimensions and their levels, and the stages of each dwell time.

## Stochastic compartmental specifics

Time-varying rates MUST work under every solver. Under `gillespie` and `tau` a value that varies smoothly in time is held for a quarter of a step at a time, at its middle, so the solver is exact for rates that are constant within each quarter; this is said in a note, and `approx=False` makes it an error. The chain-binomial solver takes such a value at the middle of each substep.

The continuous-time solver has no `dt` and so no rate-to-probability conversion. Its results are recorded on the sim's time points, so that results from every solver have the same time axis.

Non-exponential dwell times need sub-states here exactly as under an ODE: the Erlang expansion is the same one, `stages=k` sets it by hand, and a dwell time that is not Erlang is approximated by fitted stages with the same one-line note.

A sim is one trajectory. The distribution comes from many runs, `msim = ss.parallel(sim, n_runs=500)` (or `sim.run(n_runs=500)`), and how often each disease died out is read from their results ([09-results.md](09-results.md#one-run-many-runs)), since it is often the reason for choosing such a solver. Each module's stochastic compartmental run draws from a generator of its own, so pairing between scenarios is weak ([11-random.md](11-random.md)).

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
- **Compiling custom code: `@ss.jit`** (prototype, 2026-10-22; [oct22/jax/jit.md](../oct22/jax/jit.md)). A decorator for a function or a module's method that compiles it with Numba (or traces a function with JAX when it is given JAX arrays, e.g. under `jax.grad()`), with no change in what it does: a method's reads of `self` (states, parameters, `self.ti`, results) are passed in, each state as its raw array with the active agents, indexed, reduced and turned into UIDs as `ss.Arr` does, and its draws (`d.rvs(uids)`, `d.filter(uids)`) use the distribution's common random numbers, so a compiled method gives the same results as the method run as Python. A method is compiled in parts: each run of statements that can be compiled is one compiled function, and the statements that can't (a call into the sim such as `people.request_death()`, Starsim's own `super().set_prognoses()`, a draw inside a loop) run as written, in order, between them; a call to another method of `self` or to `super()` in modelers' code is compiled into the caller when that method compiles whole. Durations and rates converted with constant arguments (`.to_prob()`, `.to_dt()`, `.years`) are worked out before the call, UIDs are combined as `ss.uids` does (`a - b` removes, `|` unites), `d.set(p=array)` of an `ss.bernoulli` is used by the draws after it and applied in Python after the compiled part, and `.sum()`/`.mean()` add pairwise as NumPy does, so results are identical to the last bit. `@ss.jit` on a module class compiles its step methods (`step()`, `step_state()`, `step_die()`, `set_prognoses()`, `set_congenital()`, `update_results()`), including those it inherits from modelers' classes. A method none of which compiles says why in one line (`ss.errors.NotCompiled`, once a sim). Whole-array code compiled runs at about NumPy's speed; the speed is in loops over agents (see jit.md). The machinery is in `starsim/perflib/`, which no user needs to read; `ss.jit` is its only name at the top level.
- **Memory kept for reuse.** On Linux with glibc, `import starsim` sets malloc to keep large freed blocks in the heap for reuse rather than giving them back to the system, so each step's new arrays take no page faults (15–30% of a sim with a random network); the process keeps its peak memory until it ends. `ss.options.keep_memory` (`STARSIM_KEEP_MEMORY=0` before import) turns it off, and `ss.perflib.release_memory()` gives back what is free.
- **Aggregate quantities inside an agent-based sim** (an environmental reservoir, a mosquito population, wastewater) are states that are not people ([above](#states-that-are-not-people)): ordinary states of the disease that hold counts, written with the same transition syntax. This is how vector-borne and environmental transmission are written alongside agents.
- **Within-host state** (viral load, CD4 count) is a float agent property updated in `step()`. It can be a stratification when binned.
- **Contact tracing** applies only to networks, which keep their contacts; mixing pools do not.
- **Memory.** Arrays grow as agents are added, as in Starsim. Preallocating for the whole run is an optimization to benchmark before building, and would not change anything the user writes.
- **Small `n_agents`** is not warned about. Scaling to `total_pop` is reported as in the table above.


## Agreement between solvers

The aim is that **simple models agree across solvers within a stated tolerance**, and this is tested rather than assumed: each simple model of the corpus is run under `ode`, `binomial`, `gillespie`, and `abm` at a fine step, and the stochastic solvers' means must be within a tolerance of the ODE (5%, or a wider one stated with its reason) (`tests/devtests/test_contracts.py`; [18-conformance.md](18-conformance.md)). Agents and stepped solvers differ from the ODE by an amount of order `dt`, which is why the step is fine. Entries marked `approx` are not tested for agreement.

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

As agents, an exponential dwell given as a duration (`dur=10`) and the same dwell given as a rate (`rate=0.1`) are realized differently, and so differ at a coarse step: the duration is drawn when the agent enters the state and rounded to whole steps (at random, keeping its mean), while the rate is a chance 1 − exp(−r·dt) of leaving at the start of each step. On an SIR of 100,000 people (`beta` 0.3 per day, contact-mediated, 10 days infectious), the peak is +4.7% of the ODE's with `dur=` and −6% with `rate=` at a daily step (final size 0% and −1%), and +16% and −44% at a weekly step (final size −0.3% and −13%). Each solver keeps its simplest, most natural reading of a dwell and a rate, and the differences between them are documented, not removed: they are what different methods give (decision 16, 2026-10-19). Three other conventions for agents were built and measured, and each fixed some models and broke others: a chance r·dt/(1 + r·dt) for both forms puts contact-mediated SIRs within 1%, but moves rates written on counts (`S >> I @ beta*I.frac`) the other way; taking those rates after the start-of-step exits too fixes the SIRs at a daily and a weekly step, but slows each state's clock by 1 + r·dt, which put an SEIR's I peak 12% low, the cholera model's 11% high, and a state filled at a capped rate 15% high (they are 1%, 2%, and 2% off as is); and reading every rate at the start of a step and every exit at its end did better on those, but put a fixed 10-day SIR 7.4% low (it is 0.4%). With the textbook's fast rates (Keeling and Rohani's program 6.3 at a daily step), agents peak 25% low and 9 days late, and `binomial` 23% high; a smaller `dt` is the remedy.

At `dt=0.1` every difference is within about 1% (peaks a few tenths of a day late). On a seasonal SIRS of 100,000 people (mean `beta` 0.3, amplitude 0.5, 8 days infectious, 60 days immune), `binomial` at a daily step has 19% more infectious on average at monthly points than the daily ODE, up to 53% at one point, and 8.5% more infected-days (between 7.4% and 8.7% over five seeds); with `substeps=4` or `dt=0.25` that falls to 5% on average (15% at most), and `gillespie` is within 1.5%. The `ss.errors.LargeStep` note gives the cause for each transition (`rate × dt`); `sim.check_dt()` measures the effect on the model at hand.

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
- **Agent-based as the default for every model.** A simple SIR should run as the three equations it is, instantly. Models that need agents get them automatically.
- **Analyzing user functions in advance.** Trying them is simpler and gives a clearer message.
- **Making the user write the derivative function.** It is positional, unchecked, and cannot be stratified automatically.
