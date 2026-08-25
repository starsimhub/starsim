# individual

|  |  |
|---|---|
| **Language** | R, with a C++ (Rcpp) core |
| **Paradigms** | Agent-based, discrete-time, state-and-event |
| **Specification style** | Construct variables, events, and processes; hand them to a simulation loop |
| **Version reviewed** | 0.1.19 |
| **Licence** | MIT |
| **Code** | <https://github.com/mrc-ide/individual> |
| **Docs** | <https://mrc-ide.github.io/individual/> |
| **Paper** | [10.21105/joss.03539](https://doi.org/10.21105/joss.03539) |
| **Downstream** | `malariasimulation` and much of the `malariaverse` are built on it |

## What it's for

`individual` is a toolkit for building fast individual-based models in R. It contains **no epidemiology at all**: no compartments, no diseases, no networks, no transmission. It provides typed per-agent variables backed by bitsets, a scheduled-event mechanism, and a simulation loop, and the modeller writes everything else.

It is in this review as the **lower bound**: the minimal set of primitives an agent-based epidemiological model needs, with nothing above them. Every other agent-based framework here can be read as `individual` plus a domain vocabulary, and seeing which vocabulary each one adds is informative.

## How a model is specified

The full SIR model from the package tutorial:

```r
# State: a categorical variable over the population
health <- CategoricalVariable$new(categories = c("S","I","R"), initial_values = health_states_t0)

# A process: a function of time that queues state updates
infection_process <- function(t) {
  I <- health$get_size_of("I")
  foi <- beta * I / N
  S <- health$get_index_of("S")               # a Bitset
  S$sample(rate = pexp(q = foi * dt))         # thin it stochastically, in place
  health$queue_update(value = "I", index = S)
}

# An event: something scheduled to happen to specific individuals later
recovery_event <- TargetedEvent$new(population_size = N)
recovery_event$add_listener(function(t, target) {
  health$queue_update("R", target)
})

recovery_process <- function(t) {
  I <- health$get_index_of("I")
  already_scheduled <- recovery_event$get_scheduled()
  I$and(already_scheduled$not(inplace = TRUE))       # bitset algebra
  rec_times <- rgeom(n = I$size(), prob = pexp(q = gamma * dt)) + 1
  recovery_event$schedule(target = I, delay = rec_times)
}

# Output
health_render <- Render$new(timesteps = steps)
health_render_process <- categorical_count_renderer_process(health_render, health, health_states)

simulation_loop(
  variables = list(health),
  events    = list(recovery_event),
  processes = list(infection_process, recovery_process, health_render_process),
  timesteps = steps
)
```

## Core abstractions

There are exactly five, and this is the point:

| Concept | `individual` name | Notes |
|---|---|---|
| **Variable** | `CategoricalVariable`, `IntegerVariable`, `DoubleVariable`, `RaggedInteger`, `RaggedDouble` | Typed per-agent state. Categorical variables are stored as bitsets |
| **Bitset** | `Bitset` | A set of individuals, with `and`, `or`, `not`, `xor`, `sample`, `size`, and in-place variants |
| **Process** | `function(t)` | Runs every timestep; reads state, queues updates |
| **Event** | `Event`, `TargetedEvent` with listeners | Scheduled with a delay; fires on specific individuals |
| **Render** | `Render` | Collects time series output |

Plus a set of "prefab" processes — `bernoulli_process`, `fixed_probability_multinomial_process`, `multi_probability_bernoulli_process`, `infection_age_process`, `categorical_count_renderer_process`, `update_category_listener`, `reschedule_listener` — which are the named, parameterised versions of the most common patterns.

## The two-phase update

The most important semantic detail. Processes do not mutate state; they **queue** updates (`health$queue_update(...)`), which the loop applies after every process has run. Every process in a timestep therefore sees the same start-of-step state, and the result does not depend on the order of processes within a step.

This is a deliberate and consequential design choice, and `individual` is the only agent-based framework in the review that makes it. Starsim's `Loop`, EpiModel's `module.order`, and Vivarium's priorities all make the ordering explicit *and* load-bearing. `individual` makes it not matter.

## Bitsets as the population algebra

`Bitset` is the second load-bearing idea. Sets of individuals are bitsets, and selecting a cohort is set algebra:

```r
I <- health$get_index_of("I")
I$and(already_scheduled$not(inplace = TRUE))   # infectious AND NOT already scheduled
```

Then `$sample(rate = p)` thins the set stochastically in place. The combination — set algebra to select, stochastic thinning to apply — covers a remarkable proportion of what agent-based epidemiological code actually does, and it does so at C++ speed with a compact and readable R surface.

This is the same operation EpiHiper expresses as declarative set algebra over network predicates and EMOD expresses as `Targeting_Config`. Here it is an imperative API rather than a declaration, but the underlying algebra is identical.

## Time

Unitless integer timesteps. The tutorial is explicit about the convention: `dt` is a scalar the modeller multiplies rates by, `steps = tmax/dt`, and transition probabilities are computed as `pexp(q = rate * dt)`.

That last idiom is worth noting — using the exponential CDF to convert a rate to a per-step probability is correct and is written out in the model each time, exactly as in odin. No dimensional types.

## Strengths

- **Minimal and complete.** Five primitives, and `malariasimulation` — one of the most detailed disease models in the field — is built on them.
- **Order-independent processes** via queued updates. The strongest correctness property of any agent-based framework in the review.
- **Bitset algebra as the cohort-selection mechanism**: fast, expressive, and a direct match for how modellers think about who is eligible for what.
- **Stochastic thinning in place** (`$sample(rate = p)`) as the counterpart to selection.
- **`TargetedEvent` with delays** — schedule a future state change for a specific set of individuals — which is the natural way to express dwell times and is the same mechanism Starsim uses with `ti_*` arrays.
- **Prefab processes** covering the common patterns as named, parameterised units.
- **Checkpointing** (`save_object_state` / `restore_object_state`) built in.
- **Genuinely fast** for R, via the Rcpp/bitset core.
- **Deliberately domain-free**, which makes it reusable and makes its abstractions honest.

## Limitations

- **No epidemiology.** No disease, network, transmission, intervention, or demographic vocabulary. Every model is written from primitives, and two models of the same thing may share no structure.
- **The model is entirely code.** Processes are R closures. Nothing is introspectable, serialisable, diagrammable, or convertible.
- **Transmission has no support at all.** The tutorial's `foi <- beta * I / N` is homogeneous mixing computed by hand; contact structure is the modeller's problem.
- **Unitless time**, with the `dt` convention documented rather than enforced.
- **No results vocabulary** beyond `Render` collecting numbers a process pushes into it.
- **No inference, no calibration, no scenario machinery.**
- **R-bound**, with the C++ core not intended as a standalone library.

## Implications for the lingua franca

1. **Adopt the queued-update, order-independent step semantics.** This is `individual`'s most valuable contribution and it is directly relevant to a gap identified in Starsim: if every process sees start-of-step state and updates are applied at the end, execution order stops being hidden behaviour. The cost is that within-step causal chains ("infect, then immediately test the newly infected") need to be expressed explicitly as multiple sub-steps — which is the right trade, because it makes the chain visible.
2. **Set algebra over the population is a core primitive, and three frameworks agree.** `individual`'s `Bitset`, EpiHiper's declarative `sets`, and EMOD's `Targeting_Config` are the same concept at different levels of declarativeness. The lingua franca should have a **declarative set expression** over agent state — the EpiHiper form — with the bitset algebra as its natural implementation.
3. **"Select a set, then thin it stochastically" is the fundamental agent-based operation.** `S$sample(rate = pexp(q = foi * dt))` is what nearly every agent-based process reduces to. A language primitive of this shape — *for the agents matching this predicate, with this probability, do this* — would cover most interventions, most progressions, and most transmission.
4. **Scheduled targeted events are the right dwell-time mechanism for agent-based execution.** `recovery_event$schedule(target = I, delay = rec_times)` is the ABM realisation of EpiHiper's declared `dwellTime` distribution and camdl's `via erlang`. **One declaration — "the dwell time in I is Gamma(α, β)" — should compile to a scheduled event in an agent-based backend, Erlang sub-compartments in an ODE backend, and a memory kernel in an IDE backend.** That is a concrete, checkable instance of the paradigm-independence claim, and it is worth prototyping early.
5. **`individual` defines the floor, and the gap above it is the deliverable.** Every agent-based framework in this review adds a domain vocabulary to roughly these five primitives, and every one adds a *different* vocabulary. That divergence — Starsim's modules, epiworld's viruses and tools, EMOD's campaigns, EpiHiper's rules — is the thing a lingua franca is supposed to make unnecessary.
