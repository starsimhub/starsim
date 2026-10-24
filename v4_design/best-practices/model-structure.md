# Model structure: states and transitions

The central document. Everything else in this folder depends on the position taken here.

## Recommendation

1. **Transitions are declared data wherever they can be, and plain Python wherever they cannot.** This is the single largest gap between Starsim and this project's goals. Declaring a transition does not mean giving up functions: any part of a transition may be a function when the logic is complex.
2. **There is one kind of transition, and its rate says which mathematics applies.** A rate that mentions a state (`beta * I`) is contact-mediated; a rate that does not is spontaneous. The distinction is structural, read from what was written, and the user never has to pick a category.
3. **States are declared like ingredients, then transitions are written like the method.** The two-line shorthand may infer them; anything longer should list them.
4. **Three spellings, one object.** Arrow strings for the shortest form, operators (`S >> I @ beta * I`) for the form that reads like the equations, and `ss.Transition(...)` for the fully explicit form.
5. **Dwell time is a declared distribution per transition, defaulting to exponential.** The solver decides how to realize it, and the user can say how explicitly.
6. **A state is one concept.** "Compartment" is what a state is called when the model runs as an ODE; "boolean per-agent array" is what it is called when it runs with agents. The declaration is identical.
7. **The disease model is a labeled directed graph over states.** Say so, and draw it.

## Why: what "transitions are code" costs

Every framework in the review that can do something interesting with a model does it because the model is data. Every framework that cannot, cannot.

From the [Starsim review](../approaches/starsim.md), stated as a limitation:

> **Transitions are still code.** `define_states` declares the state space, but there is no object representing "S → I at rate β·I/N". Progression is scheduled imperatively by writing `ti_recovered` in `set_prognoses()`. The model is therefore not introspectable as structure, cannot be diagrammed or dimension-checked as a transition system, and cannot be automatically converted to ODEs.

The same finding in [EpiModel](../approaches/epimodel.md) ("There is no object representing 'S → I at rate λ'. A model cannot be introspected, diagrammed, dimension-checked, or converted to another paradigm without parsing R"), [`individual`](../approaches/individual.md) ("The model is entirely code. Processes are R closures. Nothing is introspectable, serializable, diagrammable, or convertible"), [MetaCast](../approaches/metacast.md) ("`y_deltas[states_index['S']] += -infections` is not a specification; it is code"), and [LASER](../approaches/laser.md).

And the payoff, wherever the model *is* data:

| Capability | Framework | Enabled by |
|---|---|---|
| Build-time dimension checking | [camdl](../approaches/camdl.md) | Named transitions with typed rate expressions |
| ODEs derived from the model | [epipack](../approaches/epipack.md) | Reaction tuples as SymPy |
| Automatic stratification of a finished model | [summer2](../approaches/summer2.md) | Named flow kinds the framework understands |
| `incidence(transition)` as an observable | [camdl](../approaches/camdl.md) | Transitions have names |
| Analytic gradients for fitting | [camdl](../approaches/camdl.md), [summer2](../approaches/summer2.md) | Symbolic or traceable rate expressions |
| Automatic model diagrams | [epiworld](../approaches/epiworld.md), [EMULSION](../approaches/emulsion.md), [EoN](../approaches/eon.md) | The model is a graph |
| Translation between six model formats | [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) | A closed Template taxonomy |
| One model text run several ways | [camdl](../approaches/camdl.md), [epipack](../approaches/epipack.md), [EMULSION](../approaches/emulsion.md) | The dynamics are chosen after the structure is declared |

Starsim already proved half of this internally: `define_states` buys automatic `n_<state>` results, serialization, and validation. Declaring the transitions the same way buys the rest of the table.

That table leans heavily on camdl, which is the framework that took "the model is data" furthest. What we take from it is the payoff. What we do not take is its strictness: camdl refuses anything it cannot prove, and this project's users need a tool that lets them write the complicated transition anyway.

### One kind of transition is enough

Four frameworks independently split transitions into two kinds: [EoN](../approaches/eon.md) uses two directed graphs (`H` spontaneous, `J` neighbor-induced), [epipack](../approaches/epipack.md) 3-tuples and 5-tuples, [EpiHiper](../approaches/epihiper.md) separate `transitions` and `transmissions` arrays, and [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) `NaturalConversion` versus `ControlledConversion`. The split is real, because the mathematics differs, but the rate already says which kind a transition is (`I >> R @ gamma` has no state in it; `S >> I @ beta * I` does). That is how the equations are written, and it is why camdl and SimInf can use one transition syntax.

### Dwell time is an axis most frameworks fix silently

Finding 7 of the [landscape review](../approaches/README.md#cross-cutting-findings): "Dwell-time distribution is a first-class axis that most frameworks silently fix." ODE means exponential, and nearly every framework assumes it without saying so. [MEmilio](../approaches/memilio.md) needed four model families (ODE, LCT, GLCT, IDE) to span the axis; [EpiHiper](../approaches/epihiper.md) declares a distribution per transition; camdl has `via erlang` and `via hyper_erlang`; `individual` and Starsim schedule sampled times. Declaring the distribution on the transition and letting each solver realize it (Erlang stages in a compartmental run, a sampled time for agents) spans the axis with one model. A scalar duration means an exponential dwell with that mean, which is the guess that most needs printing; [odin](../approaches/odin.md) gets the same visibility by making the modeler write `1 - exp(-gamma*dt)` by hand.

### States carry properties

[EpiHiper](../approaches/epihiper.md) puts `infectivity` and `susceptibility` on the state (`{"id": "V", "susceptibility": 0.1}`), and that two-number declaration is the vaccine model: no multipliers threaded through transmission code, and a new state composes without editing anything else. Starsim's per-agent `rel_sus` and `rel_trans` answer a different question (heterogeneity that is not a state), so both are kept.

### The model as a table

Because the model is data, a grid of transitions (Atomica's transition matrix, which a domain expert fills in by hand), a form, or a web UI is a question of serialization, not of language. That is how the [accessibility benchmark](principles.md) is met without adopting spreadsheets as a language.

## The specification

[03-states-and-transitions.md](../design/03-states-and-transitions.md) specifies states, the one transition class and its three spellings (arrow strings, `S >> I @ beta * I`, and `ss.Transition`), what may follow `@`, rates that mention states, the source of infection, dwell times, branching, functions in any slot, `ss.null`, states that are not people, and what is guessed. Names from variables, `describe()`, `diagram()`, `equations()`, and `transitions` are in [01-objects.md](../design/01-objects.md); what a state in a rate means under each kind of mixing is in [06-routes.md](../design/06-routes.md).

## Trade-offs

- **Two ways to write an arrow is one more than the minimum.** The string form is the shortest path to a running model and is what Starsim users will expect; the operator form reads like the equations and makes the source explicit. Both build the same object, and `describe()` prints one canonical form.
- **Operators have precedence rules the user does not see.** `S >> I @ beta * I` works without parentheses, but `S >> p * A + (1 - p) * B @ rate` does not, and needs them around the destinations. The error for the second should say so.
- **Inference means the described model matters.** The guessed source of a transmission is right for SIR and SIS and wrong for an SEIR whose author forgot `source=`, and the structure cannot tell them apart; so an SEIR-shaped string form without `source=` is an error (`ss.SourceNeeded`), and every other guess is marked `[guessed]` in `describe()`. The operator form, where the source is written down, avoids the question.
- **A function in a rate is opaque.** The library cannot print it as an equation or differentiate it symbolically. It can still run it wherever it works.
- **Whether a state is people is inferred.** A reservoir or a vector population is recognized because no transition connects it to the states people move through. That is right for every example we have, and it is a guess: a disease that deliberately keeps two unconnected sets of people states would be misread, and has to say `ss.State(people=True)`. `describe()` marks which states are counts.

## Rejected

- **Separate classes for spontaneous and contact-mediated transitions as the user-facing choice.** The split is real and the library tracks it; the user's rate expression already says which is which.
- **A general reaction primitive `A + B → C + D`** ([epipack](../approaches/epipack.md)). `S >> I @ beta * I` covers the same ground while keeping "who moves" separate from "who causes it", which is what stratification and networks need.
- **Requiring `/N` in the rate.** camdl is right that `beta * S * (I/N)` should type-check, but a modeler who writes `/N` by hand can write it wrongly, and the denominator depends on the mixing. It is allowed (via counts, `I.n / N`) and never needed in the common case.
- **A dict shorthand with positional tuples**, `dict(infection=('S -> I', 0.05))`. The first draft proposed it; deciding whether `0.05` is a transmission rate or a recovery rate from context turned out to need rules that misfire (SIRS, for one). The two-line constructor form is as short and is not ambiguous.
- **A separate module class for quantities that are not people** (an earlier draft's `ss.Compartments`). The reservoir's states use the same syntax as the disease's, so a second class added a name (and a confusing one, since `S`, `I`, and `R` are compartments too) without adding anything to say.
- **A bare number after `@` as a duration.** `I >> R @ 10` meaning "after 10 days" is how some people think, but `@ 0.1` and `@ 10` cannot both be guessed safely from magnitude. A bare number is a rate; a duration is typed or given as `dur=`.
- **Integer state indices** ([epiworld](../approaches/epiworld.md)'s `target_states = 2L`). Named, checked references only.
- **A separate `Junction` state kind** ([Atomica](../approaches/atomica.md)). The weighted arrow says the same thing.
- **Mandatory ontology identifiers** ([MIRA](../approaches/notes.md#askem-model-representation-amr--mira)). Never required; an optional `ontology=` on `ss.State` was considered for model merging and dropped as a small thing that did not earn a name ([17-vocabulary.md](../design/17-vocabulary.md)).

## Open questions

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
