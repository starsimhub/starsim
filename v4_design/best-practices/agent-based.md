# Agent-based

## Recommendation

1. **Keep Starsim's architecture: typed per-agent state arrays, persistent UIDs, vectorized operations, one module abstraction.** It is the best agent-based specification surface in the review and does not need replacing.
2. **The disease declaration is the same as under every other solver.** A model is agent-based when it contains something agent-based (an individual-level network, `n_agents=`, per-agent custom code) or when the user asks for `solver='abm'`.
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

So the lingua franca keeps Starsim's ordered loop, and takes from `individual` only the part Starsim already has: removals are requested during the step and applied at the end. See [10-execution.md](../design/10-execution.md).

### The one operation

From the `individual` review: "`S$sample(rate = pexp(q = foi * dt))` is what nearly every agent-based process reduces to. A language primitive of this shape — *for the agents matching this predicate, with this probability, do this* — would cover most interventions, most progressions, and most transmission."

The same operation appears in [EpiHiper](../approaches/epihiper.md)'s set algebra over node and edge predicates, [EMOD](../approaches/emod.md)'s `Targeting_Config`, [Mesa](../approaches/notes.md#mesa)'s `AgentSet` with `select` / `shuffle` / `do` / `agg`, and Starsim's boolean arrays with `dist.filter()`. Since Starsim's version is already short and already in use, it is the one to keep. Most uses of it never appear in user code, because they are what declared transitions and deliveries do internally; it is what a custom `step()` is made of.

### Counts alongside agents

Some things in an agent-based model are not agents: a mosquito population, an environmental reservoir, a wastewater concentration. The closest precedent in the review is [SimInf](../approaches/siminf.md)'s `u`/`v` split: `u` holds the integer compartment counts and `v` continuous per-node variables such as environmental contamination, which its `SISe` models couple to an SIS disease. Its review notes that Starsim's `FloatArr` plays that role per agent but nothing does per node. Here such a quantity is an ordinary state of the disease that no person can be in, so it holds a count (one per patch in a model stratified by place) while the people are agents.

## The specification

The disease declaration is the one in [03-states-and-transitions.md](../design/03-states-and-transitions.md), including dwell times (sampled for each agent, exponential ones too) and states that are not people. What makes a model agent-based, and what an agent-based run does with each declaration, is in [12-backends.md](../design/12-backends.md). Dimensions as agent properties are in [05-dimensions.md](../design/05-dimensions.md); `ss.modify` and plain code in `step()` in [07-effects.md](../design/07-effects.md); the order of calls in a step, competing hazards, and deferred removals in [10-execution.md](../design/10-execution.md).

## Trade-offs

- **Results depend on module order**, by an amount that shrinks with the timestep. `sim.check_dt()` shows how much. This is accepted in exchange for within-step chains.
- **Agent-based execution is the expensive one.** Starsim's review notes that a custom `step()` written naively is slow. Declared transitions are vectorized by the library; only custom code is at risk, and the profiler should point at it by name.
- **Some things only exist here.** Partnership histories, contact tracing (`trace=` on a delivery), transmission trees, and per-individual dwell times have no compartmental analogue. That is a reason to use agents, and `sir.solvers()` should say so positively and not only as "no" in the other columns.
- **Mixing agents and counts in one sim needs care at the boundary.** The agent side is always stochastic, while a state that holds a count is advanced deterministically under every solver. That makes no difference for a reservoir of billions of bacteria and a real one for a few hundred mosquitoes, whose own randomness cannot yet be drawn.
- **Preallocation is a performance strategy, not semantics.** [LASER](../approaches/laser.md) allocates rows for future births up front. Starsim's arrays grow in chunks and are resized only a handful of times in a 50-year run, so the gain may be small: benchmark representative models before building it, and if it is built, the library sizes the arrays from the declared demographics.
- **Which states are people is a guess.** It is right for every example here, and `describe()` marks the states it took to be counts so that a wrong guess is visible.

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

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
