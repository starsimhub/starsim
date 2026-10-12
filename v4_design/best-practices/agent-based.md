# Agent-based

## Recommendation

1. **Keep Starsim's architecture: typed per-agent state arrays, persistent UIDs, vectorized operations, one module abstraction.** It is the best agent-based specification surface in the review and does not need replacing.
2. **The disease declaration is the same as in every other run method.** A model is agent-based when it contains something agent-based (an individual-level network, `n_agents=`, per-agent custom code) or when the user asks for `method='abm'`.
3. **Keep Starsim's ordered loop.** Modules run in a defined, printable order. Only two things are resolved jointly: removals are deferred to the end of the step, and competing hazards within a module are drawn together.
4. **"Select a set, thin it with a probability, act on the result" is the basic agent-based operation, and Starsim already has it:** boolean state arrays plus `dist.filter(uids)`. Keep that idiom; do not add a new primitive for it.
5. **Dwell times become scheduled events.** This is the agent-based realization of the same declaration that becomes Erlang stages in an ODE.
6. **Every stratification dimension is an agent property. Not every agent property is a dimension.** A property becomes a dimension when you stratify by it.
7. **Storage layout is implementation, not semantics.** Preallocation and similar optimizations should be benchmarked before they are built.

## Why

### Starsim is already right about most of this

From its [review](../approaches/starsim.md): typed per-agent state arrays with declared dtypes, defaults and labels; persistent UIDs with an `auids` live view so births and deaths do not invalidate references; one `ss.Module` abstraction for diseases, networks, demographics, interventions, connectors, and analyzers; the `Route` abstraction unifying networks and mixing pools; declared results; the execution loop as an inspectable data object; per-agent common random numbers.

The gaps are the ones this folder addresses elsewhere: transitions are written as code where they could be declared ([model-structure.md](model-structure.md)), connectors overwrite each other's values ([composition-and-effects.md](composition-and-effects.md)), delivery is a lattice of classes ([interventions.md](interventions.md)), and the compartmental limit is approximate ([compartmental.md](compartmental.md)). None of them are agent-based problems.

### The floor, and what everyone builds on it

[`individual`](../approaches/individual.md) is the minimal set of primitives: typed per-agent variables backed by bitsets, scheduled targeted events, processes, and a render. No epidemiology at all, and `malariasimulation` is built on it.

Its review draws the conclusion this document turns on: "Every agent-based framework in this review adds a domain vocabulary to roughly these five primitives, and every one adds a *different* vocabulary. That divergence — Starsim's modules, epiworld's viruses and tools, EMOD's campaigns, EpiHiper's rules — is the thing a lingua franca is supposed to make unnecessary."

### Order within a step

`individual` is the only agent-based framework in the review where within-step process order does not matter, because processes queue updates and the loop applies them after every process has run. [Vivarium](../approaches/vivarium.md) declares order with named lifecycle phases and priorities; [Mesa](../approaches/notes.md#mesa) makes activation order a modeling decision the user selects; [Starsim](../approaches/starsim.md) runs modules in a fixed order and makes that order inspectable.

Queued updates remove one kind of surprise and add another. If infection, testing, and treatment each see only start-of-step state, the fastest possible treatment is two timesteps after infection, which is an artifact of the method. With modules run in a known order (infection, then testing, then treatment), the whole chain can happen in one step. A queue also needs its own rule for which update wins when two conflict, so it does not remove ordering; it adds a second one.

So the lingua franca keeps Starsim's ordered loop, and takes from `individual` only the part Starsim already has: removals are requested during the step and applied at the end. See [composition-and-effects.md](composition-and-effects.md#execution-order).

### The one operation

From the `individual` review: "`S$sample(rate = pexp(q = foi * dt))` is what nearly every agent-based process reduces to. A language primitive of this shape — *for the agents matching this predicate, with this probability, do this* — would cover most interventions, most progressions, and most transmission."

The same operation appears in [EpiHiper](../approaches/epihiper.md)'s set algebra over node and edge predicates, [EMOD](../approaches/emod.md)'s `Targeting_Config`, [Mesa](../approaches/notes.md#mesa)'s `AgentSet` with `select` / `shuffle` / `do` / `agg`, and Starsim's boolean arrays with `dist.filter()`. Since Starsim's version is already short and already in use, it is the one to keep.

## The proposal

### The same model

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.lognorm_ex(mean=10, std=3, unit='days')),
)

sim = ss.Sim(sir, n_agents=50_000, networks=ss.RandomNet(n_contacts=ss.poisson(10)), unit='day')
sim.run()
```

Identical to the [compartmental](compartmental.md) declaration except for the network and the agent count. The unit of the lognormal is given once, for the whole distribution. The lognormal dwell time is what agent-based execution gives you: it is sampled directly, where an ODE run would approximate it with Erlang stages.

What an agent-based run does with each declaration:

| Declaration | Agent-based realization |
|---|---|
| a state (`ss.State`, or `'S'` in an arrow) | `ss.BoolState` per agent |
| `ss.transmission(...)` | Go through the network's contacts, compute the per-contact probability, draw |
| `ss.progression(dur=...)` | Sample a time, write `ti_recovered`, fire when due |
| `sir.stratify(risk=[...])` | A per-agent `risk` property |
| `ss.modify(x, multiply=, where=)` | A modifier on `x`; the effective value is base × active modifiers |
| `ss.outcome(...)` | A delayed, thinned draw from the transition's events |

Three notes on the names in this table:

- **Capitalization follows Starsim's existing rule** (`ss.Dist` vs `ss.normal`, `ss.Rate` vs `ss.perday`): modules, containers, and base classes are CamelCase (`ss.Sim`, `ss.Disease`, `ss.Network`, and the two graph classes `ss.State` and `ss.Transition`); the concrete pieces you put inside a module are lowercase (`ss.transmission`, `ss.progression`, `ss.modify`, `ss.outcome`, `ss.days`, `ss.normal`).
- **`sir.stratify(...)` and `ss.stratify(sir, ...)` are the same thing**, and both are supported. Both return a new stratified model and leave `sir` as it was.
- **There is one transition class, `ss.Transition`.** `ss.transmission` and `ss.progression` are shorthand constructors for it that read well in the two-line SIR. In the operator form there is no distinction to make: `S >> I @ beta * I` is contact-mediated because its rate mentions a state, and `I >> R @ ss.days(10)` is not. Every slot (`rate=`, `beta=`, `dur=`, `p=`) also accepts a function of the sim, so a transmission function can be as complex as it needs to be; see [model-structure.md](model-structure.md).

### Select, thin, act

**`individual`** ([approaches/individual.md](../approaches/individual.md)):

```r
I <- health$get_index_of("I")
I$and(already_scheduled$not(inplace = TRUE))
rec_times <- rgeom(n = I$size(), prob = pexp(q = gamma * dt)) + 1
recovery_event$schedule(target = I, delay = rec_times)
```

**Starsim today, and the lingua franca unchanged:**

```python
class Severity(ss.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(p_severe=ss.bernoulli(p=0.1))
        self.update_pars(**kwargs)

    def step(self):
        ppl = self.sim.people
        sir = self.sim.diseases.sir
        candidates = (sir.infected & (ppl.age < 5) & ~sir.severe).uids   # select
        severe     = self.pars.p_severe.filter(candidates)              # thin
        sir.severe[severe] = True                                        # act
```

Selecting is boolean arithmetic on state arrays. Thinning is `dist.filter(uids)`, which returns the subset that passed and uses each agent's own random number stream. The probability can itself depend on the agent (`ss.bernoulli(p=fn)`), which is why a separate `p` argument on a new primitive would add little: in most real cases the probability is part of the selection. Acting is whatever comes next: set a state, set a `ti_*` time, call another module.

Most uses of this never appear in user code, because they are what declared transitions and [deliveries](interventions.md) do internally. It is what a custom `step()` is made of.

### What happens in what order

Each step runs Starsim's loop: start of step; demographics; disease state updates and progression; connectors; networks; interventions; transmission; deaths; results; analyzers; end of step. Modules of one type run in the order they were given. `sim.loop` is a data object that can be printed, plotted (`sim.loop.plot_step_order()`), and inserted into.

Two things are deliberately not left to module order:

- **Removals are deferred.** A module that kills or removes agents calls `sim.people.request_death(uids)`, as in Starsim today. The agents are flagged, and are actually removed at the end of the step, after every module has run. This is the supported way to remove agents in custom code, and it is the one place where custom code should not act immediately.
- **Competing hazards within a module are drawn together.** If an infected agent can recover or die, both written as rates out of the same state, one multinomial draw decides which happens. The order in which the transitions were declared does not bias the result.

Everything else takes effect immediately, so a module sees the changes made by the modules that ran before it in the same step.

### Scheduled events for dwell times

Starsim's `ti_*` arrays, `individual`'s `TargetedEvent`, and [LASER](../approaches/laser.md)'s `SortedQueue` are the same mechanism, chosen for the same reason: avoid scanning every agent for something that applies to few.

One declaration serves every run method:

```python
ss.progression('I -> R', dur=ss.gamma(a=2, scale=5, unit='days'))
```

| Run method | Realization |
|---|---|
| `'abm'` | Sample a time per agent, schedule the event |
| `'ode'` | Erlang sub-compartments (two stages here; exact for an integer shape, a stated approximation otherwise) |
| `'ctmc'`, `'tau'`, `'binomial'` | Sub-states |

`stages=k` on the transition asks for `k` sub-stages in every method, including agent-based. [EpiHiper](../approaches/epihiper.md) declares dwell-time distributions; [MEmilio](../approaches/memilio.md) needed four model families for them. See [paradigm-conversion.md](paradigm-conversion.md).

### Dimensions are agent properties

[EMOD](../approaches/emod.md)'s Individual Properties, user-defined categorical attributes declared in demographics, targetable by interventions and transitionable between values, are the agent-based spelling of a stratification dimension. Treating the two as one concept is the unification proposed in [stratification.md](stratification.md):

```python
sir_risk = sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
# Agents:       a per-agent 'risk' property, usable in eligibility conditions and in ss.modify(where=)
# Compartments: S_low, S_high, ... with the adjustment applied
```

The relationship runs one way. Every dimension is an agent property; most agent properties (a viral load, a time of last test, a household ID) are never dimensions. A property becomes a dimension when you stratify by it, and a continuous property is binned for that purpose:

```python
hiv_cd4 = hiv.stratify(cd4=[0, 200, 350, 500])    # e.g. for treatment eligibility by CD4 count
```

In an agent-based run `cd4` and `age` stay continuous per-agent values, and the bins are used for whatever was declared on them (eligibility, adjustments, results). In a compartmental run the bins are the groups.

### The virus/tool duality

[epiworld](../approaches/epiworld.md) models an agent as carrying two collections: **viruses** (transmissible, with their own rates) and **tools** (protective, with multiplicative effects on susceptibility, transmissibility, recovery, and death). Its review: "Multi-disease and multi-intervention composition fall out of this automatically… It is less general — a tool is a bundle of four multipliers, not arbitrary behavior — and that is exactly why it composes."

The useful part of this is multiplicative modifiers that do not overwrite each other, and that is what [`ss.modify`](composition-and-effects.md) provides, for any named quantity instead of four. An epiworld tool is:

```python
ss.modify(sir.rel_sus, multiply=0.7, where=masks.wearing)
```

No separate "tool" object is needed.

### Population

Starsim's `ss.People` with persistent UIDs and an `auids` live view stays. [LASER](../approaches/laser.md)'s alternative (fixed capacity, rows for future births allocated up front and activated at the right timestep, dead rows flagged, no allocation during the run) is a performance strategy for the same semantics, so it could be offered as an option without changing any model.

Whether it is worth building is an empirical question. Starsim's arrays grow in chunks, and over a typical 50-year run they are resized only a handful of times, so the gain may be small. Benchmark a few representative models first, and build it only if the difference is significant. If it is built, the user should not have to compute a capacity by hand as with LASER's `calc_capacity`: the library can estimate it from the declared demographics at build time.

### Aggregate quantities alongside agents

Some things in an agent-based model are not agents: a mosquito population, an environmental reservoir, a wastewater concentration. They are counts, possibly one per patch. These are `ss.Compartments` modules: states that are counts, written with the same transition syntax, living in the same sim as the agent-based modules.

The closest thing in the review is [SimInf](../approaches/siminf.md)'s `u`/`v` split: `u` holds the integer compartment counts, `v` holds continuous per-node variables such as environmental contamination, and its built-in `SISe` models couple an SIS disease to a continuous environmental compartment that is updated by a function after each time step. Its review recommends adopting the split and notes that Starsim's `FloatArr` plays a similar role per agent but that nothing does per node. In the lingua franca, with people as agents and the reservoir as a count:

```python
beta, K, xi = 0.5, 1e5, 10            # per day; half-saturation concentration; shedding per infected person per day
S, I, R, W  = ss.states('S I R W')

cholera = ss.Disease(                 # people
    infection = S >> I @ beta * W.n / (W.n + K),
    recovery  = I >> R @ ss.days(5),
)
water = ss.Compartments(              # the reservoir: a count, not agents
    shedding = None >> W @ xi * I.n,
    decay    = W >> None @ ss.days(30),
)

sim = ss.Sim([cholera, water], n_agents=10_000, unit='day')
```

`I.n` and `W.n` are plain counts, used literally in the rates. Each module reads the other's current count; the people are simulated as agents and the reservoir as a number. Vector-borne transmission has the same shape, with mosquito states in the `ss.Compartments` module and `a * b * Iv.n / Nh` in the human infection rate.

## Trade-offs

- **Results depend on module order**, by an amount that shrinks with the timestep. `sim.check_dt()` shows how much. This is accepted in exchange for within-step chains.
- **Agent-based execution is the expensive one.** Starsim's review notes that a custom `step()` written naively is slow. Declared transitions are vectorized by the library; only custom code is at risk, and the profiler should point at it by name.
- **Some things only exist here.** Partnership histories, contact tracing, transmission trees, and per-individual dwell times have no compartmental analogue. That is a reason to use agents, and `sir.methods()` should say so positively and not only as "no" in the other columns.
- **Mixing agents and counts in one sim needs care at the boundary.** An `ss.Compartments` module is deterministic or stochastic by its own setting, and the agent side is always stochastic. The combination is well defined, but it should be printed.

## Rejected

- **A separate agent-based specification vocabulary.** This is the divergence `individual`'s review identifies as the thing a lingua franca should make unnecessary.
- **Integer state indices** (epiworld). Names.
- **Queued updates for every state change** (`individual`). They remove within-step chains and still need a tie-break rule. Deferred removals and joint draws for competing hazards are kept.
- **A new select-thin-act primitive.** Boolean arrays and `dist.filter(uids)` already do this in three lines.
- **Direct writes as the only way for modules to interact** (LASER's `LaserFrame`, Starsim's connectors today). Direct access stays available for anything that needs it; `ss.modify` is the first thing to reach for when scaling someone else's quantity. See [composition-and-effects.md](composition-and-effects.md).
- **A fifth object for protective measures** (epiworld's tools). `ss.modify` covers it.
- **Making the user compute array capacity** (LASER's `calc_capacity`). If preallocation is built, the library sizes the arrays.
- **A warning for small `n_agents`.** Nobody runs 500 agents expecting the result to look like a country of a billion. `total_pop` and `n_agents` are both printed, and that is enough.
- **String status attributes** (EpiModel's `status == "i"`). "Multi-disease models, co-infection, and orthogonal state dimensions have to be encoded by hand into extra attributes, and nothing coordinates them."

## Open questions

- **Within-host state.** Viral load, CD4 count, and immune waning are continuous per-agent trajectories. Today they are float agent properties updated in `step()`, which works, and they can be stratifications once binned (CD4 for treatment eligibility). More native support would be worth having, for example declaring a trajectory as a function of time since infection and letting the library update it. The shape of that is not settled.
- **Per-patch `ss.Compartments` with agents.** A reservoir per patch, with agents assigned to patches, needs the count state to be indexed by the same `patch` dimension the agents use. This should follow from [stratification.md](stratification.md) and needs a worked example.
- **Contact tracing** needs the contact history, which only routes that keep individual contacts (networks) can provide. That is the intended scope: nobody traces contacts through an environmental reservoir or a mixing pool. [EoN](../approaches/eon.md)'s conditional link transmission and [epipack](../approaches/epipack.md)'s cascading link events are the two treatments in the review; both are network-only. What is open is how much history a network should keep by default.
