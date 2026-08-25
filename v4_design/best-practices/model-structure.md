# Model structure: states and transitions

The central document. Everything else in this folder depends on the position taken here.

## Recommendation

1. **Transitions are declared data, not imperative code.** This is the single largest gap between Starsim and this project's goals, and closing it is what makes everything downstream possible.
2. **A transition is either spontaneous or contact-mediated, and the distinction is structural** — not implicit in whether a rate expression happens to mention another state.
3. **The transition vocabulary is small, named, and closed, with one escape hatch that states its cost.**
4. **Dwell time is a declared distribution per transition, defaulting to exponential**, and the backend decides how to realize it.
5. **A state is one concept.** "Compartment" is what a state is called when the backend is an ODE; "boolean per-agent array" is what it is called when the backend is an ABM. The declaration is identical.
6. **Infer the state list from the transitions.** Nobody should have to declare `S`, `I`, `R` and then write `S -> I`.
7. **The disease model is a labelled directed graph over states.** Say so, and render it.

## Why: what "transitions are code" costs

Every framework in the review that can do something interesting with a model does it because the model is data. Every framework that cannot, cannot.

From the [Starsim review](../approaches/starsim.md), stated as a limitation:

> **Transitions are still code.** `define_states` declares the state space, but there is no object representing "S → I at rate β·I/N". Progression is scheduled imperatively by writing `ti_recovered` in `set_prognoses()`. The model is therefore not introspectable as structure, cannot be diagrammed or dimension-checked as a transition system, and cannot be automatically converted to ODEs.

The same finding in [EpiModel](../approaches/epimodel.md) ("There is no object representing 'S → I at rate λ'. A model cannot be introspected, diagrammed, dimension-checked, or converted to another paradigm without parsing R"), [`individual`](../approaches/individual.md) ("The model is entirely code. Processes are R closures. Nothing is introspectable, serialisable, diagrammable, or convertible"), [MetaCast](../approaches/metacast.md) ("`y_deltas[states_index['S']] += -infections` is not a specification; it is code"), and [LASER](../approaches/laser.md).

And the payoff, wherever the model *is* data:

| Capability | Framework | Enabled by |
|---|---|---|
| Compile-time dimension checking | [camdl](../approaches/camdl.md) | Named transitions with typed rate expressions |
| ODEs, Jacobian, fixed points, epidemic threshold, all derived | [epipack](../approaches/epipack.md) | Reaction tuples as SymPy |
| Automatic stratification of a finished model | [summer2](../approaches/summer2.md) | Named flow kinds the framework understands |
| `incidence(transition)` as an observable | [camdl](../approaches/camdl.md) | Transitions have names |
| Analytic gradients for NUTS, by source-to-source differentiation | [camdl](../approaches/camdl.md) | Symbolic rate expressions |
| Automatic model diagrams | [epiworld](../approaches/epiworld.md), [EMULSION](../approaches/emulsion.md), [EoN](../approaches/eon.md) | The model is a graph |
| Translation between six model formats | [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) | A closed Template taxonomy |
| One model text on three backends | [camdl](../approaches/camdl.md), [epipack](../approaches/epipack.md) | The dynamics operator is chosen after the structure is declared |

Starsim already proved half of this internally: `define_states` buys automatic `n_<state>` results, serialization, and validation. Declaring the transitions the same way buys the rest of the table.

## The proposal

### The simplest thing

**camdl** ([approaches/camdl.md](../approaches/camdl.md)):

```camdl
compartments { S, I, R }
let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}
```

**SimInf** ([approaches/siminf.md](../approaches/siminf.md)):

```r
mparse(transitions = c("S -> beta * S * I / N -> I",
                       "I -> gamma * I -> R"),
       compartments = c("S", "I", "R"), ...)
```

**Starsim today** — 15 lines of `define_pars` / `define_states` plus a `step_state()` method that writes `ti_recovered`.

**Lingua franca:**

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.perday(0.05)),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
```

Two lines, four facts. States `S`, `I`, `R` are inferred. `I` is inferred infectious because it is the destination of the transmission. `N` never appears, because the force of infection is the route's job and not the modeler's — see [population-and-mixing.md](population-and-mixing.md). The transitions are named `infection` and `recovery` because they are keyword arguments, which means `sim.results.infection` and `incidence(infection)` exist without anyone declaring a result.

The dict form is the same thing with the constructors elided:

```python
sir = dict(infection=('S -> I', ss.perday(0.05)), recovery=('I -> R', ss.days(10)))
```

`ss.Disease` reads a positional `(arrow, quantity)` pair as `transmission` if the quantity is a rate the pathogen carries and `progression` if it is a duration — see the inference rules below, and note that this shorthand is for the SIR-shaped 90%, not a general syntax.

### The arrow string

`'S -> I'` is the one place a string is load-bearing, and it is deliberate. It carries *structure* (two names and a direction), never an *expression*. The convergence is strong: camdl's `S --> I`, SimInf's `S -> ... -> I`, [Atomica](../approaches/atomica.md)'s transition matrix, [EoN](../approaches/eon.md)'s `H.add_edge('E','I')`, [EpiHiper](../approaches/epihiper.md)'s `entryState`/`exitState`, [Epydemix](../approaches/epydemix.md)'s `source`/`target`. Everyone draws the same arrow.

What must never happen is [flepiMoP](../approaches/flepimop.md)'s `rate: ["Ro * gamma"]` or [EMULSION](../approaches/emulsion.md)'s `value: 'transmission_I * total_I / total_population'`. Rates are Python objects and Python expressions.

Equivalent longhand for anyone who dislikes the arrow, or for programmatic construction:

```python
ss.transmission(src='S', dst='I', beta=ss.perday(0.05))
```

### Two kinds, because the mathematics is different

Four independent frameworks arrived at this split. [EoN](../approaches/eon.md) uses two directed graphs (`H` spontaneous, `J` neighbour-induced). [epipack](../approaches/epipack.md) uses 3-tuples and 5-tuples. [EpiHiper](../approaches/epihiper.md) has separate `transitions` and `transmissions` arrays. [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) formalizes it as `NaturalConversion` versus `ControlledConversion`.

```python
ss.progression('I -> R', dur=ss.days(10))                       # autonomous
ss.transmission('S -> I', beta=ss.perday(0.05))                 # requires a contact
ss.transmission('S -> E', beta=ss.perday(0.05), source='I')     # SEIR: contact source is not the destination
```

The `source=` default is "the destination state" — right for SIR, SIS, SIRS; wrong for SEIR, where naming `I` is the actual epidemiology and should be written down. This is the one place we ask for a word that a lazier design would guess.

Multi-source transmission (competing or cooperating pathogens, multi-strain) takes a list: `source=['I_wild', 'I_alpha']`, which is MIRA's `GroupedControlledConversion`.

### The vocabulary, in full

Small enough to hold in one screen. This is the whole of it:

| Constructor | Meaning | Prior art |
|---|---|---|
| `ss.transmission(arrow, beta=, source=)` | Contact-mediated | EoN `J`, epipack 5-tuple, EpiHiper `transmissions`, MIRA `ControlledConversion` |
| `ss.progression(arrow, dur= \| rate= \| p=)` | Spontaneous | EoN `H`, epipack 3-tuple, EpiHiper `transitions`, MIRA `NaturalConversion` |
| `ss.birth(arrow, rate=)` / `ss.death(arrow, rate=)` | Source and sink | camdl inflow/outflow, summer2 `add_crude_birth_flow` / `add_death_flow`, SimInf `@` |
| `ss.transfer(<dim>='a -> b', rate= \| p=)` | Move between *strata*, not disease states | [MetaCast](../approaches/metacast.md) transfer dicts, camdl `transfer(...)` |
| `ss.process(fn)` | The escape hatch: arbitrary Python | [`individual`](../approaches/individual.md)'s processes, Starsim's `step()` |

`ss.transfer` is what vaccination, ageing, and migration are: the individual does not change disease state, they change dimension coordinate. Keeping it separate from `ss.progression` is [MetaCast](../approaches/metacast.md)'s insight and it is why interventions can be written without touching the disease model ([interventions.md](interventions.md)).

Deliberately absent: [summer2](../approaches/summer2.md)'s eight named flow kinds, [flepiMoP](../approaches/flepimop.md)'s `proportional_to` / `proportion_exponent` pairs. summer2 needs `add_infection_frequency_flow` versus `add_infection_density_flow` because it has no route abstraction; we have one, and frequency-versus-density is a property of the route, not of the transition. flepiMoP's structured rate form is analysable and, per its own review, "opaque — getting frequency- versus density-dependence right through parallel lists of compartment sets and exponent strings is error-prone".

### The escape hatch, priced

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.perday(0.05)),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
    weirdness = ss.process(my_function),
)
```

```text
>>> sir.summary()
...
weirdness  custom process
  ! not stratifiable, not convertible to ODE/CTMC, excluded from R0 and Jacobian
```

This is [camdl](../approaches/camdl.md)'s capability-check discipline — "where a paradigm × feature combination is not supported, say so by name" — applied to the user's own code. [`epidemics`](../approaches/epidemics.md) does the same thing for a different reason and explains it: you cannot rate-intervene on `infectiousness_rate` because that parameter sets the number of Erlang sub-compartments.

### Dwell time is a distribution, and it is the sharpest test of paradigm independence

This is finding 7 of the [landscape review](../approaches/README.md#cross-cutting-findings): "Dwell-time distribution is a first-class axis that most frameworks silently fix." ODE means exponential, and nearly every framework assumes it without saying so. [MEmilio](../approaches/memilio.md) needed four separate model families (ODE / LCT / GLCT / IDE) to span the axis. [EpiHiper](../approaches/epihiper.md) declares a distribution per transition. camdl has `via erlang` / `via hyper_erlang`. `individual` and Starsim schedule sampled times.

**EpiHiper** ([approaches/epihiper.md](../approaches/epihiper.md)):

```json
{ "id": "I_to_R", "entryState": "I", "exitState": "R",
  "probability": 1, "dwellTime": { "gamma": { "alpha": 1.0, "beta": 5.0 } } }
```

**Lingua franca:**

```python
ss.progression('I -> R', dur=ss.gamma(shape=1.0, scale=ss.days(5)))
```

and the compiler realizes it per backend, which is the concrete, testable instance of the project's central claim:

| Backend | Realization | Precedent |
|---|---|---|
| ODE (exponential) | `rate = 1/mean` | Everyone |
| ODE (non-exponential) | Erlang sub-compartments, generated | camdl `via erlang`, [MEmilio](../approaches/memilio.md) LCT |
| IDE | Memory kernel | MEmilio IDE |
| CTMC / Gillespie | Exponential clock, or sub-states | [SimInf](../approaches/siminf.md) |
| Agent-based | Sample a time, schedule the event | [`individual`](../approaches/individual.md)'s `TargetedEvent`, Starsim's `ti_*` |

Three equivalent spellings, following [EMULSION](../approaches/emulsion.md)'s three transition keys (`rate:`, `proba:`, `duration:`) and Starsim's three rate types:

```python
ss.progression('I -> R', dur=ss.days(10))          # mean dwell time      → Exp(mean=10 days)
ss.progression('I -> R', rate=ss.perday(0.1))      # hazard               → Exp(mean=10 days)
ss.progression('I -> R', p=ss.probperday(0.1))     # per-period probability
```

A scalar `dur=` means *exponential with that mean*, not "exactly 10 days". This is the guess that most needs printing, so `summary()` shows `recovery: I -> R, dwell ~ Exp(mean=10 days)` — the visible expansion that [odin](../approaches/odin.md) achieves by making the modeler write `1 - exp(-gamma*dt)` by hand, without making them write it.

### Branching

**Atomica** ([approaches/atomica.md](../approaches/atomica.md)) uses a *junction*: a compartment flagged `Is Junction` with zero residence time, splitting inflow by proportion. **epymorph** ([approaches/epymorph.md](../approaches/epymorph.md)) has `fork(...)`. camdl has branching transitions.

**Lingua franca** — one arrow, several destinations:

```python
ss.progression('I -> R | D', dur=ss.days(10), p=[0.99, 0.01])
```

Probabilities must sum to 1 (or one may be omitted and inferred). This is more legible than a zero-residence-time pseudo-compartment and it keeps `D` an ordinary state.

### States carry properties, not multipliers

**EpiHiper** ([approaches/epihiper.md](../approaches/epihiper.md)) puts `infectivity` and `susceptibility` on the *state*:

```json
{ "id": "V",      "susceptibility": 0.1 },
{ "id": "wanedV", "susceptibility": 0.5 }
```

That two-scalar declaration *is* the vaccine model — no multipliers threaded through transmission code, and a new state composes without editing anything else. Starsim's `rel_sus` / `rel_trans` are the per-agent version of the same idea.

**Lingua franca** — keep both, because they answer different questions:

```python
ss.State('V',     rel_sus=0.1)    # a property of being in this state
ss.State('wanedV', rel_sus=0.5)
```

with per-agent `rel_sus` retained for heterogeneity that is not a state (age, comorbidity, an intervention someone received). The state-level version is the declarative default; the agent-level version is what [composition-and-effects.md](composition-and-effects.md) governs.

### Explicit state declaration, when you want it

Inference covers SIR. Declare states explicitly when they carry properties, labels, or initial conditions:

```python
sir = ss.Disease(
    S = ss.State(init=0.99),
    I = ss.State(init=0.01, infectious=True, label='Infectious'),
    R = ss.State(),
    infection = ss.transmission('S -> I', beta=ss.perday(0.05)),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
```

States and transitions share one namespace of keyword arguments, so the name on the left is the name used everywhere else. [odin](../approaches/odin.md)'s rule — the state set is whatever has an `initial()`, and `deriv()` versus `update()` selects the paradigm — is the same factoring: the state space is shared across paradigms, only the dynamics operator differs.

### Order does not matter

[odin](../approaches/odin.md) topologically sorts its equations; [SimInf](../approaches/siminf.md) resolves its dependency graph; [summer2](../approaches/summer2.md) builds a `networkx` DAG; camdl compiles to a flat IR. From the odin review:

> Order-independence is a requirement, not a nicety. Any lingua franca whose statements have to be read in order has failed the "no hidden behavior" goal, because execution order *is* hidden behavior.

Declared transitions are a set. Within a timestep they all see start-of-step state and are applied together — [`individual`](../approaches/individual.md)'s queued-update semantics, which is "the strongest correctness property of any agent-based framework in the review". Only `ss.process` escape hatches need an ordering, and that is what Starsim's `Loop` is for ([composition-and-effects.md](composition-and-effects.md)).

### Render it

Because the model is a graph, the pictures are nearly free, and they are the artifact non-modelers actually read. [epiworld](../approaches/epiworld.md) has `ModelDiagram`; [EMULSION](../approaches/emulsion.md) puts `fillcolor` on states because the engine draws the model; [EoN](../approaches/eon.md) literally executes a `networkx.DiGraph`; [Atomica](../approaches/atomica.md)'s transition matrix is an adjacency matrix that a domain expert fills in by hand.

Three projections, all derived from the same object:

```python
sir.plot()      # state-transition diagram
sir.summary()   # the resolved model as text, including every inferred default
sir.to_df()     # the transition matrix — Atomica's grid, for free
```

The tabular projection is how the [accessibility benchmark](principles.md) gets met without adopting spreadsheets as a language: since the model is data, a form, a grid, or a web UI is a serialization question rather than a language question.

## Trade-offs

- **A closed vocabulary means some models take the escape hatch.** summer2 pays this and it is why its stratification works. The mitigation is that the hatch is one call, not a fork of the framework, and that its cost is printed rather than discovered.
- **Inference means the printed summary is load-bearing.** If someone writes an SEIR and does not read the summary, they may not notice that `E` was assumed infectious. Mitigation: `source=` defaults to the destination, so SEIR is *wrong in an obvious way* (the summary says `E infectious`) rather than subtly. Consider warning when a transmission destination has an onward progression before any transmission sources it — the SEIR shape is detectable.
- **Rate expressions in Python are checkable at build time, not compile time.** Later than camdl. Earlier than every Python framework here.
- **Transitions-as-data does not make imperative code go away.** Within-host viral load, partnership formation, and care cascades will still be written as `step()` methods. The claim is only that the *disease structure* need not be.

## Rejected

- **A general reaction primitive `A + B → C + D`** ([epipack](../approaches/epipack.md)). Maximally expressive and, as summer2 shows from the other side, unstratifiable — the framework cannot transform a flow whose meaning it does not know. The two named kinds cover the same epidemiological ground with the meaning attached.
- **Integer state indices** ([epiworld](../approaches/epiworld.md)'s `target_states = 2L`). Named, checked references only.
- **Requiring `N` in the rate expression.** camdl is right that `beta * S * (I/N)` should type-check, and right that "no hidden multiplication" is safer — but a modeler who writes `/N` by hand can write `/N` wrongly, and the denominator convention is a route property. See [population-and-mixing.md](population-and-mixing.md).
- **A separate `Junction` state kind** ([Atomica](../approaches/atomica.md)). The branching arrow says the same thing without adding a state that is not a state.
- **Mandatory ontology identifiers** ([MIRA](../approaches/notes.md#askem-model-representation-amr--mira)). Optional `ss.State('I', ontology='ido:0000511')` for anyone doing model merging; never required.

## Open questions

- Should `ss.transmission` be able to fire from a state that is not itself a disease state — environmental reservoirs, vectors? [SimInf](../approaches/siminf.md)'s `u`/`v` split (discrete compartments alongside continuous node-level variables) is the clean precedent, and environmental transmission is common enough to deserve an answer.
- Where exactly does the shorthand `('S -> I', rate)` stop being helpful and start being ambiguous? It needs a written rule, not a heuristic.
- Multi-strain: HPVsim added genotype as a state *dimension* rather than duplicating the module; summer2 has a special-cased `StrainStratification`. Is a strain a dimension ([stratification.md](stratification.md)) or a second disease? Probably a dimension, but the force-of-infection restructuring summer2 special-cases has to be derivable.
