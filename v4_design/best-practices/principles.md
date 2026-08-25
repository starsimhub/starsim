# Principles

The doctrine the rest of this folder applies. Read this first; every other document assumes it.

## Recommendation

1. **The user is an epidemiologist, not a programmer.** Optimize the model text for someone who can read an equation and write a `for` loop, and who will never read the source of the library.
2. **The model text should look like the algorithm in the modeler's head.** If a step of the model takes one sentence to say, it should take about one line to write.
3. **Guess the default.** When the right answer is obvious 99% of the time, take it and say so in the model summary. An error is for when a wrong guess would be silently harmful.
4. **One language, two verbosities.** A dict and a class are the same vocabulary written at different lengths, not two worlds. Never delete the short form to enable a feature that only the long form needs — support both.
5. **Complexity belongs in the library.** The declared surface may be small and the compiler enormous.
6. **Every escape hatch is allowed, and every escape hatch is priced.** Arbitrary Python is always available; the model summary says what guarantees it just cost you.
7. **Nothing is a string that could be an object.** Python, not YAML in Python.
8. **A feature that only pays off in one framework's use case does not go in.** Bloat is a correctness problem, because a vocabulary nobody can hold is a vocabulary nobody uses correctly.

## Why: the audience is the binding constraint

The landscape divides sharply on who it is for, and the division predicts everything else about the design.

[Atomica](../approaches/atomica.md) is the accessibility benchmark: a national HIV or TB model, with its compartments, parameters, derived indicators, and funded intervention programmes, defined by a health ministry analyst in two Excel workbooks and no code. [`epidemics`](../approaches/epidemics.md) states the trade explicitly in its scope document — it "trades away some flexibility in defining model structures for a gain in the ease of defining epidemic scenario components", because its users are "public health practitioners rather than research-focussed modellers". [EMULSION](../approaches/emulsion.md) exists so that a model can be discussed with veterinarians and economists during development rather than translated for them afterwards.

At the other end, [MEmilio](../approaches/memilio.md) requires C++17 templates and CMake, [camdl](../approaches/camdl.md) requires an OCaml/Rust toolchain, and [EpiHiper](../approaches/epihiper.md) requires C++, MPI, and PostgreSQL — and EpiHiper's "close schools on Fridays" is roughly 200 lines of JSON.

Starsim sits in the middle and its own review flags the cost: "API surface is large. Six module categories, four state types, ~20 distributions, ~15 time classes, three rate types, and a `Loop` API. Each piece is justified; the total is a lot to hold, and the framework leans heavily on AI tooling to make it navigable — which is a signal about the specification burden, not a solution to it."

That last sentence is the one to take seriously. **Total vocabulary size is a first-class design constraint, and every recommendation in this folder has to pay for its share of it.**

## The two-language problem is a design failure, and it is ours to fix

Starsim today has a declarative dict surface:

```python
sim = ss.Sim(diseases=dict(type='sir', beta=0.05, init_prev=0.01), networks='random')
```

and a class surface, and — as its review says — "there is no path from the first to the second, and the boundary is where most users get stuck." Vivarium has the same split (YAML lists components, Python defines them). EMOD has it (JSON configures, C++ implements). EMULSION has it (YAML, plus "code add-ons" that forfeit the model-as-document property).

The rule that follows:

> **Any keyword accepted by the dict form is accepted by the class form and means the same thing, and any model expressible in the class form has a dict form.** Subclassing adds *behavior*; it never adds *vocabulary*.

Concretely, this is the target — the same model, three ways, with no re-learning between them:

```python
# 1. Shortest: the model is a dict
sir = dict(infection=('S -> I', ss.perday(0.05)), recovery=('I -> R', ss.days(10)))

# 2. Named constructors, when the shorthand runs out
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.perday(0.05)),
    recovery  = ss.progression('I -> R', dur=ss.lognorm(mean=ss.days(10), std=ss.days(3))),
)

# 3. A class, when there is behavior that is not a transition
class SIR(ss.Disease):
    infection = ss.transmission('S -> I', beta=ss.perday(0.05))
    recovery  = ss.progression('I -> R', dur=ss.days(10))

    def step(self):
        super().step()
        self.log_something_bespoke()
```

Form 3 uses form 2's vocabulary verbatim. Nothing is re-spelled.

## Guessing defaults

The rule is not "never error". It is: **an error is a claim that no guess is safe, and that claim has to be true.**

Guess, and print it:

| Situation | Guess | Evidence |
|---|---|---|
| The state list | Inferred from the transitions that mention states | Nobody writes `S -> I` and then wants to be told `I` was not declared |
| Which state is infectious | The contact source of the transmissions; defaults to the destination | Right for SIR/SIS; SEIR must say `source='I'`, which is the actual modeling content |
| Initial conditions | Everyone in the first state; `init_prev` seeds the rest | [odin](../approaches/odin.md) infers the state set from `initial()`; this goes one step further |
| Dwell-time distribution | Exponential, when a scalar duration or a rate is given | It is what every ODE means, and what [EMULSION](../approaches/emulsion.md) already assumes; the difference is that we *print* it |
| Number of agents, timestep, units | Sensible values keyed to the timescale of the parameters given | [Epydemix](../approaches/epydemix.md) and Starsim both do this today |
| Effect composition | Multiplicative | See [composition-and-effects.md](composition-and-effects.md) |
| Result set | Counts for every state, flows for every named transition | Starsim's automatic `n_<state>` results; [camdl](../approaches/camdl.md)'s `incidence(transition)` |

Refuse, because a guess would be wrong quietly:

- A rate expression whose dimensions do not check ([camdl](../approaches/camdl.md)'s missing-`/N` error).
- A stratified coverage that would deliver `1 − (1 − f)^P` instead of `f` (camdl E239).
- A mixing matrix on a partial stratification ([summer2](../approaches/summer2.md) enforces this and is right to).
- A paradigm × feature combination the backend cannot execute — say so by name rather than approximating (camdl's capability checks; [`epidemics`](../approaches/epidemics.md)'s refusal to let a rate intervention touch a parameter that sets Erlang stage count).

And in every case, **print the resolved model**. `sim.summary()` should show the states, the transitions with their inferred kinds and dwell-time distributions, the routes, and the defaults that were filled in. A guess the user can see is not hidden behavior; a guess they cannot see is.

## Error messages are part of the language

camdl's E237/E238/E239 encode hard-won knowledge about stratified coverage and deliver it exactly when the mistake is made. This matters more for us than for camdl, because a substantial fraction of lingua franca models will be written by an AI that will otherwise produce the wrong answer with complete confidence.

The requirement: **every refusal names the rule, shows the offending expression, and states the fix.** Not `ValueError: shape mismatch`.

## AI-native means four concrete things

The [approaches README](../approaches/README.md) frames this project as AI-native. In practice, from the frameworks that tried:

1. **Documentation shipped with the version, offline and doc-tested** — camdl's `camdl docs <topic>` and `AGENTS.md`. An agent that reads stale docs writes broken models.
2. **A machine-drivable contract** — [Epydemix](../approaches/epydemix.md) ships a CLI with a documented `AGENT.md` rather than exposing its Python API; camdl does the same. Worth noting that both chose a CLI over an API.
3. **The model is data, so the model can be checked.** Everything in [model-structure.md](model-structure.md) follows from this.
4. **A canonical printed form.** If prose maps deterministically to a specification, two agents given the same prose should produce the same text. That requires a normalizer, not just a parser.

## What we are deliberately not building

Each of these looked good in isolation. Each costs more than it returns.

| Rejected | Where it comes from | Why not |
|---|---|---|
| A standalone DSL | [camdl](../approaches/camdl.md), [odin](../approaches/odin.md) | Both are excellent and both cost a toolchain. Python is the constraint, and the type system that makes camdl work is expressible in Python objects. |
| YAML or JSON as the authoring surface | [EMULSION](../approaches/emulsion.md), [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [flepiMoP](../approaches/flepimop.md) | Four independent instances of the same mistake: expressions end up in strings, nothing is type-checked, and errors come from a parser that knows nothing about epidemiology. A data format is the right thing to compile *to* and the wrong thing to write *in*. [SBML](../approaches/notes.md#sbml-systems-biology-markup-language)'s history closes the argument. |
| Spreadsheet authoring | [Atomica](../approaches/atomica.md) | Binary, undiffable, formula cells that break silently. But the *projection* is worth having: see [model-structure.md](model-structure.md) on the transition matrix. |
| Mandatory ontology identifiers on states | [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) | `identifiers={"ido": "0000511"}` on every compartment is real value for automated model merging and real friction for every user who is not merging models. Optional annotation, never required. |
| Symbolic fixed-point and eigenvalue analysis as a headline feature | [epipack](../approaches/epipack.md) | Genuinely delightful, and intractable for exactly the stratified models where the derivation is hardest by hand. Ship it as an opt-in that says when it gives up. |
| `Overwrite` adjustments | [summer2](../approaches/summer2.md) | Multiplicative adjustments compose; overwrites do not, and their result depends on stratification order. |
| Additive effect composition as the default | [`epidemics`](../approaches/epidemics.md) | Chosen there for legibility and documented honestly, but two 60% reductions summing to 120% has to be clamped somewhere. Multiplicative by default, declarable per effect. |
| Positional or integer references | [epiworld](../approaches/epiworld.md)'s `target_states = 2L`, [EpiModel](../approaches/epimodel.md)'s derivative-vector ordering | A wrong index is a silently different model. Names, always, checked. |
| Optimal control and budget optimization in v1 | [PyRoss](../approaches/notes.md#pyross), [Atomica](../approaches/atomica.md) | The right long-term capability and not a v1 vocabulary. The obligation is only to *not foreclose* it: interventions with declared costs and effects make it possible later. |
| A separate specification for each paradigm | [MEmilio](../approaches/memilio.md)'s twenty-plus model families, [EpiModel](../approaches/epimodel.md)'s three parallel APIs | This is the problem, not a solution. See [paradigm-conversion.md](paradigm-conversion.md). |

## Trade-offs we are accepting

- **Python's dynamism means some checks happen at build time rather than compile time.** camdl catches a dimensional error before anything runs; we catch it when the `Sim` is constructed. That is later than ideal and far earlier than any Python framework in the review manages today.
- **A closed transition vocabulary means some models need the escape hatch.** [summer2](../approaches/summer2.md) pays this price and it is why its stratification operator works at all. We pay it too, and print what it costs.
- **Guessing defaults means the printed summary is load-bearing.** If people do not read it, some guesses will go unnoticed. Mitigation: the summary is short, it is printed by default on `run()`, and it highlights only what was inferred rather than everything.
