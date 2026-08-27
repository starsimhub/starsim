# Agent-based

## Recommendation

1. **Keep Starsim's architecture: typed per-agent state arrays, persistent UIDs, vectorized operations, one module abstraction.** It is the best agent-based specification surface in the review and does not need replacing.
2. **The disease declaration is the same as in every other paradigm.** Agent-based is what happens when the route is a network and the backend is agents.
3. **Adopt queued updates so that within-step order does not matter.** *[CK: I'm not sure about this]*
4. **"Select a set, thin it stochastically, act" is the fundamental agent-based operation.** Make it a primitive.
5. **Dwell times become scheduled events.** This is the ABM realization of the same declaration that becomes Erlang stages in an ODE.
6. **Agent properties and stratification dimensions are the same thing.** *[CK: well, stratification dimensions are agent properties. The reverse isn't necessarily true.]*
7. **Preallocation, dead-flagging, and columnar storage are implementation, not semantics.**

## Why

### Starsim is already right about most of this

From its [review](../approaches/starsim.md): typed per-agent state arrays with declared dtypes, defaults and labels; persistent UIDs with an `auids` live view so births and deaths do not invalidate references; one `ss.Module` abstraction for diseases, networks, demographics, interventions, connectors, and analyzers; the `Route` abstraction unifying networks and mixing pools; declared results; the execution loop as an inspectable data object; per-agent common random numbers.

The gaps are the ones this folder addresses elsewhere: transitions are code ([model-structure.md](model-structure.md)), connectors are unconstrained ([composition-and-effects.md](composition-and-effects.md)), eligibility is opaque ([interventions.md](interventions.md)), and the compartmental limit is approximate ([compartmental.md](compartmental.md)). None of them are agent-based problems.

### The floor, and what everyone builds on it

[`individual`](../approaches/individual.md) is the minimal set of primitives: typed per-agent variables backed by bitsets, scheduled targeted events, processes, and a render. No epidemiology at all, and `malariasimulation` is built on it.

Its review draws the conclusion this document turns on: "Every agent-based framework in this review adds a domain vocabulary to roughly these five primitives, and every one adds a *different* vocabulary. That divergence — Starsim's modules, epiworld's viruses and tools, EMOD's campaigns, EpiHiper's rules — is the thing a lingua franca is supposed to make unnecessary."

### Order independence

`individual` is the only agent-based framework in the review where within-step process order does not matter, because processes queue updates and the loop applies them after every process has run. Its review calls this "the strongest correctness property of any agent-based framework in the review", and Starsim's own review flags the counterpart as a limitation: "results still depend on whether the connector runs before or after the network updates, and nothing declares that dependency." *[CK: I could be convinced, but I think this is fine? I think modules running in a preordained order makes more sense than someone not getting diagnosed because the test fell on timestep 3.001 and the infection fell on 2.999. If that's what "queue updates" means. If it's like Starsim's request_death (where agents are flagged to die but do not actually die until the end of the timestep), I think that's ok, but I worry that in custom code users might not do that and that could break things.]*

[Vivarium](../approaches/vivarium.md) makes ordering declared (named lifecycle phases with priorities); [Mesa](../approaches/notes.md#mesa) makes activation order "a first-class modelling decision the user selects", and names the options; Starsim makes it inspectable. `individual` makes it irrelevant, which is better than all three where it is achievable.

### The one operation

From the `individual` review: "`S$sample(rate = pexp(q = foi * dt))` is what nearly every agent-based process reduces to. A language primitive of this shape — *for the agents matching this predicate, with this probability, do this* — would cover most interventions, most progressions, and most transmission."

The same algebra appears three more times: [EpiHiper](../approaches/epihiper.md)'s declarative set algebra over node and edge predicates, [EMOD](../approaches/emod.md)'s `Targeting_Config`, and [Mesa](../approaches/notes.md#mesa)'s `AgentSet` with `select` / `shuffle` / `do` / `agg`. And [LASER](../approaches/laser.md)'s own architecture document reaches for it: "declarative behavior is encouraged, with step functions optionally described in SQL-like syntax."

Five arrivals. It is a primitive.

## The proposal

### The same model

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.lognorm(mean=ss.days(10), std=ss.days(3))), # [CK: you shouldn't need to declare the units twice in one dist unless they differ]
)

sim = ss.Sim(sir, n_agents=50_000, mixing=ss.RandomNet(n_contacts=ss.poisson(10))) # [CK: not sure I agree with 'mixing' instead of 'networks', but I could be convinced]
sim.run()
```

Identical to the [compartmental](compartmental.md) declaration except for the route and the agent count. The lognormal dwell time is the difference agent-based execution buys: it is sampled directly rather than approximated by Erlang stages.

What the backend does with each declaration:

| Declaration | ABM realization |
|---|---|
| `ss.State('S')` | `ss.BoolState` per agent |
| `ss.transmission(...)` | Enumerate route contacts, compute per-contact probability, draw |
| `ss.progression(dur=...)` | Sample a time, write `ti_recovered`, fire when due |
| `sir.stratify(risk=[...])` | A per-agent `risk` property |
| `ss.Effect(cond, x, multiply=)` | A registered modifier on `x`, resolved before `x` is read |
| `ss.Outcome(...)` | A delayed, thinned draw from the transition's event stream |

*[CK: I don't understand why some of these are capitalized and others aren't. I can also see users forgetting that it's sir.stratify() and not ss.stratify(sir) (although I suppose both could easily be supported).]*

*[CK: I don't like the name "Effect".]*

*[CK: I'm not sure I understand either why transmission and progression need to be separate instead of e.g. `ss.flow()`, or whether this framework is flexible enough to support very complex transmission functions.]*

### The select-thin-act primitive

**`individual`** ([approaches/individual.md](../approaches/individual.md)):

```r
I <- health$get_index_of("I")
I$and(already_scheduled$not(inplace = TRUE))
rec_times <- rgeom(n = I$size(), prob = pexp(q = gamma * dt)) + 1
recovery_event$schedule(target = I, delay = rec_times)
```

**Lingua franca** — the same operation, using the [predicate algebra](interventions.md#eligibility-as-a-predicate) that already exists for eligibility:

```python
ss.act(who=(sir.I & ~ss.scheduled('recovery')), p=0.1, do=...)
```

and in practice most uses of it never appear, because they *are* the declared transitions. `ss.act` is the shape of the escape hatch, not a thing users write for SIR. Bitsets are the natural implementation, exactly as `individual` shows.

*[CK: I think the name "act" may be too brief. I think `p` is likely to be part of `who` in most cases. I'm not sure about "do" either.]*

### Queued updates

```python
class Custom(ss.Module):
    def step(self):
        newly = (sir.I & (ss.age < 5)).uids
        self.queue(sir.severe, newly, True)     # applied after every module has stepped
```

Every module sees start-of-step state; updates land together. Starsim's `Loop` remains for the cases that genuinely need sequencing, and it remains inspectable, printable, and insertable-into. What changes is that ordering stops being *load-bearing by default* — it becomes something you reach for deliberately, and `loop.plot_step_order()` shows what you did.

The cost, per `individual`'s review: within-step causal chains must be written as explicit sub-steps. `sim.substep()` or an explicit phase is how, and making the chain visible is the point.

*[CK: Hmm, I'm not sure I see how this would work. How do you do tie-breaks in the queued events? Don't you need an explicit ordering anyway? Don't you then just have two definitions of ordering -- the module order and whatever you use to solve tiebreaks in the queue?]*

### Scheduled events for dwell times

Starsim's `ti_*` arrays, `individual`'s `TargetedEvent`, and [LASER](../approaches/laser.md)'s `SortedQueue` are the same mechanism, chosen for the same reason: avoid scanning every agent for something that applies to few.

This is where the paradigm-independence claim is most concrete. One declaration:

```python
ss.progression('I -> R', dur=ss.gamma(shape=2, scale=ss.days(5)))
```

| Backend | Realization |
|---|---|
| ABM | Sample a time per agent, schedule the event |
| ODE | Erlang sub-compartments |
| IDE | Memory kernel |
| CTMC | Sub-states |

[EpiHiper](../approaches/epihiper.md) declares it; [MEmilio](../approaches/memilio.md) needed four model families for it. See [paradigm-conversion.md](paradigm-conversion.md).

### Agent properties are dimensions

[EMOD](../approaches/emod.md)'s Individual Properties — user-defined categorical attributes declared in demographics, targetable by interventions and transitionable between values — are the agent-based spelling of a stratification dimension, and treating them as one concept is the unification proposed in [stratification.md](stratification.md):

```python
sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
# ABM: a per-agent 'risk' property, targetable by ss.Effect and by eligibility predicates
# ODE: S_low, S_high, ... with the adjustment applied
```

Continuous properties (age, viral load, CD4 count) are ordinary agent state and only get binned when a compartmental backend or an output requests it. **Bin at the boundary, not in the model.**

### The virus/tool duality, considered and folded in

[epiworld](../approaches/epiworld.md) models an agent as carrying two collections: **viruses** (transmissible, with their own rates) and **tools** (protective, with multiplicative effects on susceptibility, transmissibility, recovery, and death). Its review: "Multi-disease and multi-intervention composition fall out of this automatically… It is less general — a tool is a bundle of four multipliers, not arbitrary behaviour — and that is exactly why it composes."

It is a genuinely good factoring, and it is what [`ss.Effect`](composition-and-effects.md) already does with a wider target list and a declared combiner. A tool is `ss.Effect(ss.has(mask), sir.rel_sus, multiply=0.7)`. Adopting epiworld's *constraint* (a fixed set of modifiable quantities, multiplicative composition) without adopting its *vocabulary* (a fifth first-class object) is the right trade — see [principles.md](principles.md) on vocabulary size.

### Population

Starsim's `ss.People` with persistent UIDs and an `auids` live view is right, and LASER's alternative — fixed capacity, preborn rows activated at the right timestep, dead rows flagged, no runtime allocation — is a performance strategy for the same semantics. *[CK: I am OK with LASER's version also being an option, although I'd be surprised if it actually had a significant performance benefit, since I think for a typical sim, even over 50 years, Starsim arrays only rescale a handful of times. Worth checking if it actually makes a difference before implementing it.]*

From the [LASER review](../approaches/laser.md): "A lingua franca that has to compile to both needs to know which of these are *semantic* (typed time, state ownership) and which are purely *implementation* (preallocation, dead-flagging). Most are implementation — which is encouraging, and worth confirming deliberately."

And the constraint LASER imposes on its users is information the language already has: "If the model declares its maximum population, its node set, and its output shape — which a declarative language naturally does — a backend can preallocate without the modeller thinking about it."

## Trade-offs

- **Queued updates cost a second pass** over the changed set. Cheap next to transmission, and it buys order-independence. *[CK: as above ... I'm not convinced, but could be.]*
- **Agent-based execution is the expensive one.** Starsim's review notes performance is "Python-bound for anything not vectorized… a custom `step()` written naively is slow, and there is no compilation step to catch that." Declared transitions are compiled and vectorized; only escape hatches are at risk, and the profiler should point at them by name.
- **Some things only exist here.** Partnership histories, contact tracing, transmission trees, and heterogeneous individual dwell times have no compartmental analogue. That is a reason to use this paradigm, and the capability list should say so positively rather than only as refusals elsewhere.
- **Stochastic noise at small agent counts** is the mirror of the SDE problem: 500 agents is not a small population, it is a noisy one. Worth a warning when `n_agents` is far below the population being represented. *[CK: disagree, I think ... no one would run 500 agents, expect it to look like India, and be surprised when it doesn't.]*

## Rejected

- **A separate agent-based specification vocabulary.** This is the divergence `individual`'s review identifies as the thing a lingua franca should make unnecessary.
- **Integer state indices** (epiworld). Names.
- **Order as the primary mechanism** (LASER's component list, EpiModel's `module.order`). Queue by default; order deliberately. *[CK: unconvinced]*
- **Unconstrained shared state** (LASER's `LaserFrame`, Starsim's connectors). See [composition-and-effects.md](composition-and-effects.md). *[CK: unconstrained should be available as an escape hatch, not a default.]*
- **A fifth object for protective measures** (epiworld's tools). `ss.Effect` covers it.
- **Exposing preallocation to the user** (LASER's `calc_capacity`). The compiler has the information. *[CK: I'd prefer if we don't talk about compilers for Python.]*
- **String status attributes** (EpiModel's `status == "i"`). "Multi-disease models, co-infection, and orthogonal state dimensions have to be encoded by hand into extra attributes, and nothing coordinates them."

## Open questions

- **What is the agent-based analogue of `ss.Continuous`?** Environmental reservoirs are per-patch, not per-agent, and an ABM with patches needs both. *[CK: good question -- we need to figure this out!]*
- **Within-host state.** Viral load, CD4, and immune waning are continuous per-agent trajectories that are not transitions and not dimensions. Starsim handles them as `FloatArr` plus imperative code, which is honest and unstructured. Is there a declarative form worth having, or is this correctly the escape hatch? *[CK: I think more native handling of within-host would be nice. But also, these are very obviously agent properties -- so what I would is not the implementation, but rather your assertion that agent properties == stratifications. In any case, I'm not sure I agree with your premise. CD4 count absolutely could be a stratification w.r.t. treatment eligibility.]*
- **Contact tracing** needs the contact history, which is a route property that not every route retains. [EoN](../approaches/eon.md)'s conditional link transmission and [epipack](../approaches/epipack.md)'s cascading link events are the two treatments in the review; both are network-only. *[CK: that's fine, obviously we're not going to do contact tracing from e.g. environmental transmission.]*
