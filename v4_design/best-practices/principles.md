# Principles

The principles the rest of this folder applies. Read this first; every other document assumes it.

## The test

Two statements define success, and every recommendation in this folder should be checked against them:

1. **A model should never be easier to write without the lingua franca than with it.**
2. **For each of the main frameworks (MEmilio, EpiModel, Epydemix, and the rest), its own examples can be written in the lingua franca in no more lines of code, and run in no more time, than the original.**

The second is measurable, and [vignettes](../vignettes) is where it gets measured: lines of model code and wall-clock time, original against translation.

## Recommendation

1. **The user is an epidemiologist, not a programmer.** Optimize the model text for someone who can read an equation and write a `for` loop, and who will never read the source of the library.
2. **The model text should look like the algorithm in the modeler's head.** If a step of the model takes one sentence to say, it should take about one line to write. `S >> I @ beta * I` is the target.
3. **Guess the default.** When the right answer is obvious 99% of the time, take it and say so when the model is described. An error is for when a wrong guess would be silently harmful.
4. **One vocabulary, several lengths.** The two-line form, the operator form, and the class form are the same words written at different lengths. Never delete the short form to enable a feature that only the long form needs; support both.
5. **Complexity belongs in the library.** What the user writes may be tiny and what the library does with it may be large.
6. **Plain Python is always available, and it is not second class.** Real models have parts that are easiest to say as a function or a `step()` method. Those parts work everywhere they can, and the library says where they cannot.
7. **Expressions are Python, never strings.** A string may name a state or draw an arrow (`'S -> I'`); it never carries a formula.
8. **Approximate rather than refuse, and say what was approximated.** A coarser model that runs is more useful than a refusal, as long as the user is told.
9. **A feature that only pays off in one framework's use case does not go in.** A vocabulary nobody can hold is a vocabulary nobody uses correctly.

## Why: the audience is the binding constraint

The landscape divides sharply on who it is for, and the division predicts everything else about the design.

[Atomica](../approaches/atomica.md) is the accessibility benchmark: a national HIV or TB model, with its compartments, parameters, derived indicators, and funded intervention programs, defined by a health ministry analyst in two Excel workbooks and no code. [`epidemics`](../approaches/epidemics.md) states the trade explicitly in its scope document: it "trades away some flexibility in defining model structures for a gain in the ease of defining epidemic scenario components", because its users are "public health practitioners rather than research-focussed modellers". [EMULSION](../approaches/emulsion.md) exists so that a model can be discussed with veterinarians and economists during development rather than translated for them afterwards.

At the other end, [MEmilio](../approaches/memilio.md) requires C++17 templates and CMake, [camdl](../approaches/camdl.md) requires an OCaml/Rust toolchain, and [EpiHiper](../approaches/epihiper.md) requires C++, MPI, and PostgreSQL. EpiHiper's "close schools on Fridays" is roughly 200 lines of JSON.

Starsim sits in the middle and its own review flags the cost: "API surface is large. Six module categories, four state types, ~20 distributions, ~15 time classes, three rate types, and a `Loop` API. Each piece is justified; the total is a lot to hold, and the framework leans heavily on AI tooling to make it navigable — which is a signal about the specification burden, not a solution to it."

That last sentence is the one to take seriously. **Total vocabulary size is a design constraint, and every recommendation in this folder has to justify its share of it.** Several things proposed in the first draft of this folder did not survive that test, and they are listed under [what we are not building](#what-we-are-deliberately-not-building).

## One vocabulary, several lengths

Starsim today has a declarative surface for library models:

```python
sim = ss.Sim(diseases=dict(type='sir', beta=0.05, init_prev=0.01), networks='random')
```

and a class surface for everything else, and, as its review says, "there is no path from the first to the second, and the boundary is where most users get stuck." Vivarium has the same split (YAML lists components, Python defines them). EMOD has it (JSON configures, C++ implements). EMULSION has it (YAML, plus "code add-ons" that forfeit the model-as-document property).

The rule that follows:

> **Anything you can say in a short form means the same thing in a longer form, and moving from one to the next never means re-learning the words.** Subclassing adds *behavior*; it never adds *vocabulary*.

The same model, three ways:

```python
# 1. Two lines: arrow strings, states inferred
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)

# 2. Operators: declare the states, then write the transitions the way they are said aloud
beta = 0.05
dur  = ss.days(10)

S, I, R   = ss.states('S I R')
infection = S >> I @ beta * I
recovery  = I >> R @ dur
sir = ss.Disease(infection, recovery)

# 3. A class, when there is behavior that is not a transition
class SIR(ss.Disease):
    S, I, R   = ss.states('S I R')
    infection = S >> I @ 0.05 * I
    recovery  = I >> R @ ss.days(10)

    def step(self):
        super().step()
        self.log_something_bespoke()
```

All three build the same `ss.Transition` objects, and [model-structure.md](model-structure.md) gives the rules. Form 1 is the fastest way to a running model. Form 2 is the one to teach: it lists the ingredients before the method, it reads like the equations, and it makes the source of infection explicit. Form 3 uses form 2's vocabulary verbatim.

## Guessing defaults

The rule is not "never error". It is: **an error is a claim that no guess is safe, and that claim has to be true.**

Guess, and print it:

| Situation | Guess | Evidence |
|---|---|---|
| The state list, in the two-line form | The states named in the arrows | Nobody writes `'S -> I'` and then wants to be told `I` was not declared. (When states *are* declared, an undeclared name is a typo and is reported.) |
| Which state is infectious, when not written | States flagged `infectious`, else the destination of the transmission | Right for SIR and SIS. It cannot be checked, so it is printed; the operator form `beta * I` makes it explicit |
| Initial conditions | Everyone in the first state; `init_prev` or one `init=` seeds the rest | [odin](../approaches/odin.md) infers the state set from `initial()`; nobody should have to write both 0.99 and 0.01 |
| Dwell-time distribution | Exponential, when a scalar duration or a rate is given | It is what every ODE means, and what [EMULSION](../approaches/emulsion.md) already assumes; the difference is that we *print* it |
| Units of a bare number | The sim's unit | Starsim and [Epydemix](../approaches/epydemix.md) both do this today |
| Mixing, when no network is given | Homogeneous, so `beta` is the textbook transmission rate | It is what an unadorned SIR means |
| How the model is run | The simplest method that represents it exactly: an ODE if it can be one, agents otherwise | See [paradigm-conversion.md](paradigm-conversion.md) |
| How modifiers combine | Multiplicatively | See [composition-and-effects.md](composition-and-effects.md) |
| Results | A count for every state, a flow for every named transition | Starsim's automatic `n_<state>` results; [camdl](../approaches/camdl.md)'s `incidence(transition)` |
| Goodness of fit | Normalized root-mean-square error against any data supplied | See [calibration.md](calibration.md); nobody should need a likelihood to fit a model |

Stop, because a guess would be wrong quietly:

- A rate expression whose dimensions do not check ([camdl](../approaches/camdl.md)'s missing-`/N` error).
- A stratified coverage that would deliver `1 − (1 − f)^P` instead of `f` (camdl E239).
- A mixing matrix on a partial stratification ([summer2](../approaches/summer2.md) enforces this and is right to).
- A misspelled state or parameter name, when names were declared.

Note what is *not* on that list: asking for a run method the model does not exactly fit. That gets the standard approximation and a one-line note, not an error; see principle 8 and [paradigm-conversion.md](paradigm-conversion.md).

And in every case, **describe the resolved model**. `sim.describe()` shows the states, the transitions with their dwell-time distributions, the networks, the run method, and the defaults that were filled in. A guess the user can see is not hidden behavior; a guess they cannot see is. (`sim.summarize()` and `sim.plot()` keep their Starsim meanings: they are about results.)

## Plain Python is part of the language

The first draft of this folder treated arbitrary Python as an escape hatch to be discouraged. That was wrong for two reasons. Real transitions are often complicated (a probability of death that depends on age, sex, comorbidity, and time since treatment), and real interactions between modules are often more complicated still ("find people who are HIV positive, filter by an age- and sex-dependent probability, and schedule an HIV test at their next antenatal visit"). No closed vocabulary covers these, and pretending otherwise produces a language that handles SIR and nothing a working modeler needs.

So every place a number can go, a function can go, and any module can have a `step()` method that does whatever it needs. What the declared forms buy is that the library can do more with them (draw the diagram, write the equations, run them as an ODE, stratify them). What a function costs is only whatever it actually prevents: a function that works on group totals runs in a compartmental model; one that reaches for individual agents needs agents. The library finds out by trying it, and says so plainly when it does not work.

## Error messages are part of the language

camdl's E237/E238/E239 encode hard-won knowledge about stratified coverage and deliver it exactly when the mistake is made. This matters more for us than for camdl, because a substantial fraction of lingua franca models will be written by an AI that will otherwise produce the wrong answer with complete confidence.

The requirement: **every error names the rule, shows the offending expression, and states the fix.** Not `ValueError: shape mismatch`.

## AI-native means four concrete things

The [approaches README](../approaches/README.md) frames this project as AI-native. In practice, from the frameworks that tried:

1. **Documentation shipped with the version, offline and doc-tested**: camdl's `camdl docs <topic>` and `AGENTS.md`. An agent that reads stale docs writes broken models.
2. **A machine-drivable contract**: [Epydemix](../approaches/epydemix.md) ships a CLI with a documented `AGENT.md` rather than exposing its Python API; camdl does the same. Worth noting that both chose a CLI over an API.
3. **The model is data, so the model can be checked.** Everything in [model-structure.md](model-structure.md) follows from this.
4. **A canonical printed form.** If prose maps deterministically to a specification, two agents given the same prose should produce the same text. That requires a normalizer, not just a parser.

## What we are deliberately not building

Each of these looked good in isolation. Each costs more than it returns.

| Rejected | Where it comes from | Why not |
|---|---|---|
| A standalone DSL | [camdl](../approaches/camdl.md), [odin](../approaches/odin.md) | Both are excellent and both cost a toolchain. Python is the constraint, and the type system that makes camdl work is expressible in Python objects. |
| YAML or JSON as the authoring surface | [EMULSION](../approaches/emulsion.md), [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [flepiMoP](../approaches/flepimop.md) | Four independent instances of the same mistake: expressions end up in strings, nothing is type-checked, and errors come from a parser that knows nothing about epidemiology. A data format is the right thing to export *to* and the wrong thing to write *in*. [SBML](../approaches/notes.md#sbml-systems-biology-markup-language)'s history closes the argument. |
| Spreadsheet authoring | [Atomica](../approaches/atomica.md) | Binary, undiffable, formula cells that break silently. But the *projection* is worth having: see [model-structure.md](model-structure.md) on the transition table. |
| Refusing to run a model in a paradigm it does not exactly fit | [camdl](../approaches/camdl.md)'s capability checks | Right for a compiler whose users want proofs; wrong for epidemiologists, who routinely and validly write ODE models of sexually transmitted infections. Approximate, and say so. |
| Order-independent execution through queued updates | [`individual`](../approaches/individual.md) | Elegant, but a defined module order is easier to reason about and lets infection, testing, and treatment happen in one step. Starsim's ordered, inspectable loop stays. |
| A closed list of things one module may modify in another | [epiworld](../approaches/epiworld.md)'s tools, [EpiHiper](../approaches/epihiper.md)'s state properties | Too restrictive to be useful for real multi-disease models. Any named quantity can be modified; anything more complex is a `step()` method. |
| A predicate mini-language with its own built-in names | [EMOD](../approaches/emod.md)'s `Targeting_Config`, Polars-style column objects | Eligibility is written against the model's real attributes (`people.age`, `sir.vaccinated`), so custom attributes work exactly like built-in ones. |
| Mandatory ontology identifiers on states | [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) | `identifiers={"ido": "0000511"}` on every compartment is real value for automated model merging and real friction for every user who is not merging models. Optional annotation, never required. |
| Symbolic fixed-point and eigenvalue analysis | [epipack](../approaches/epipack.md) | Delightful for textbook models, and intractable for exactly the stratified models people actually build. Printing the equations is worth having; the rest is not a v1 feature. |
| A ladder of analytic network approximations | [EoN](../approaches/eon.md) | A decade of mathematics for SIS and SIR on networks. Anyone who needs an exact network result can run agents, which is one keyword away. |
| `Overwrite` adjustments | [summer2](../approaches/summer2.md) | Multiplicative adjustments compose; overwrites do not, and their result depends on stratification order. |
| Additive composition as the default | [`epidemics`](../approaches/epidemics.md) | Chosen there for legibility and documented openly, but two 60% reductions summing to 120% has to be clamped somewhere. Multiplicative by default, additive on request. |
| Positional or integer references | [epiworld](../approaches/epiworld.md)'s `target_states = 2L`, [EpiModel](../approaches/epimodel.md)'s derivative-vector ordering | A wrong index is a silently different model. Names, always, checked. |
| Optimal control and budget optimization in v1 | [PyRoss](../approaches/notes.md#pyross), [Atomica](../approaches/atomica.md) | The right long-term capability and not a v1 vocabulary. The obligation is only to *not foreclose* it: interventions with declared costs make it possible later. |
| A separate specification for each paradigm | [MEmilio](../approaches/memilio.md)'s twenty-plus model families, [EpiModel](../approaches/epimodel.md)'s three parallel APIs | This is the problem, not a solution. See [paradigm-conversion.md](paradigm-conversion.md). |

## Trade-offs we are accepting

- **Checks happen when the sim is built, not earlier.** camdl catches a dimensional error before anything runs; we catch it at `sim.init()`. That is later than a compiled language manages and far earlier than any Python framework in the review manages today.
- **The declared forms cover the common structure, not everything.** [summer2](../approaches/summer2.md) accepts the same limit and it is why its stratification operator works at all. Whatever falls outside is written as a function, and the library reports what it can no longer do with that part.
- **Guessing defaults means people need to look at the described model.** If they do not, some guesses will go unnoticed. Mitigation: the description is short, it marks what was guessed, and the one-line notes about the run method and any approximation are printed on `run()`.
- **Approximating instead of refusing means two runs of "the same model" can differ for reasons the user did not think about.** Mitigation: `method='auto'` never approximates, every approximation is named when it happens, and `approx=False` turns approximations into errors for anyone who wants that.
