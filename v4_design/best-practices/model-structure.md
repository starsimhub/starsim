# Model structure: states and transitions

The central document. Everything else in this folder depends on the position taken here.

## Recommendation

1. **Transitions are declared data wherever they can be, and plain Python wherever they cannot.** This is the single largest gap between Starsim and this project's goals. Declaring a transition does not mean giving up functions: any part of a transition may be a function when the logic is complex.
2. **There is one kind of transition, and its rate says which mathematics applies.** A rate that mentions a state (`beta * I`) is contact-mediated; a rate that does not is spontaneous. The distinction is structural, read from what was written, and the user never has to pick a category.
3. **States are declared like ingredients, then transitions are written like the method.** The two-line shorthand may infer them; anything longer should list them.
4. **Three spellings, one object.** Arrow strings for the shortest form, operators (`S >> I @ beta * I`) for the form that reads like the equations, and `ss.Transition(...)` for the fully explicit form.
5. **Dwell time is a declared distribution per transition, defaulting to exponential.** The run method decides how to realize it, and the user can say how explicitly.
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

**Starsim today**: 15 lines of `define_pars` / `define_states` plus a `step_state()` method that writes `ti_recovered`.

**Lingua franca:**

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
```

Two lines, four facts. States `S`, `I`, `R` are inferred from the arrows. `I` is taken to be infectious because it is where the transmission leads. `N` never appears, because dividing by the population is the library's job and not the modeler's (see [population-and-mixing.md](population-and-mixing.md)). The transitions are named `infection` and `recovery` because those are the keywords, so results for both exist without anyone declaring them.

### The same model, written the way it is said

```python
beta = 0.05           # per day: a bare number takes the sim's unit
dur  = ss.days(10)

S, I, R   = ss.states('S I R') # [CK: would really like to avoid having to duplicate "S I R"]
infection = S >> I @ beta * I
recovery  = I >> R @ dur

sir = ss.Disease(infection, recovery)
```

Read `S >> I @ beta * I` as "S goes to I at rate beta times I". `>>` is a valid Python operator and `->` is not, which is the only reason the arrow looks different from the string form.

- `ss.states('S I R')` returns three `ss.State` objects, the same idiom as `sympy.symbols('x y z')`.
- `a >> b` makes a transition from `a` to `b`, and `@` attaches whatever drives it.
- `ss.Disease(infection, recovery)` picks up the names `infection` and `recovery` from the variables, so nothing is typed twice. (Keywords work too, and win if both are given.)

What may follow `@`:

| After `@` | Means | Example |
|---|---|---|
| A duration | Mean time spent in the source state | `I >> R @ ss.days(10)` |
| A rate, or a bare number | A hazard per unit time | `I >> R @ ss.perday(0.1)`, `I >> R @ 0.1` |
| A distribution of durations | The dwell-time distribution | `I >> R @ ss.lognorm_ex(mean=10, std=3, unit='days')` |
| An expression containing a state | A force of infection | `S >> I @ beta * I` |
| A function | Called to get the rate | `I >> D @ death_rate` (see [below](#when-a-transition-is-complicated)) |

A bare number after `@` is a rate, because "at" reads as "at rate". For a bare-number duration use the keyword form, `ss.progression('I -> R', dur=10)`.

### Fully explicit

Everything above builds the same object, and it can be built directly:

```python
ss.Transition(S, I, rate=beta * I, name='infection')
ss.Transition('I', 'R', dur=ss.days(10), name='recovery')
```

`ss.Transition(src, dst, rate= | dur= | p=, ...)` takes states or their names. This is the form for programmatic construction and for anyone who dislikes operators. The shorthand constructors are thin wrappers around it:

| Shorthand | Is |
|---|---|
| `ss.transmission('S -> I', beta=b, source='I')` | `ss.Transition('S', 'I', rate=b * I)` |
| `ss.progression('I -> R', dur=d)` | `ss.Transition('I', 'R', dur=d)` |
| `ss.birth('-> S', rate=r)` | `ss.Transition(None, 'S', rate=r * N.n)`: an inflow rate is a total count per unit time, not a per-capita hazard, so the per-capita birth rate is scaled by the population |
| `ss.death('I ->', rate=r)` | `ss.Transition('I', None, rate=r)`; the arrow is optional, and `ss.death(rate=r)` applies to every state |
| `ss.transfer(risk='low -> high', rate=r)` | A move between strata; see [stratification.md](stratification.md) |

On capitalization, which follows Starsim's existing habit (`ss.Dist` and `ss.normal`, `ss.Rate` and `ss.perday`): modules and base classes are CamelCase (`ss.Disease`, `ss.State`, `ss.Transition`); the concrete pieces you put inside a module are lowercase (`ss.transmission`, `ss.par`, `ss.days`).

### Declaring states

A recipe lists the ingredients and then the method, and that convention exists for a reason: the list is where you see the whole model at a glance, and it is what catches a misspelled name in the method. So:

```python
S, I, R = ss.states('S I R')                     # the short way
E = ss.State('E', label='Exposed')               # one at a time, when a state has properties
I = ss.State('I', init=0.01, infectious=True)
```

**Declare states or let them be inferred; both work.** Inference from arrow strings is what makes the two-line model possible, and it stays. Once any state is declared, all of them must be, and an arrow that names an undeclared state is an error with the nearest match suggested.

Initial conditions are given only where they are not zero. `ss.State('I', init=0.01)` puts 1% in `I` and the rest in the first state; `init_prev=0.01` on the disease does the same thing, as in Starsim. Nobody should have to write both 0.99 and 0.01.

[odin](../approaches/odin.md)'s rule that the state set is whatever has an `initial()` is the same idea from the other side: the states are shared by every way of running the model, and only the dynamics differ.

### One kind of transition

Four frameworks independently split transitions into two kinds. [EoN](../approaches/eon.md) uses two directed graphs (`H` spontaneous, `J` neighbor-induced). [epipack](../approaches/epipack.md) uses 3-tuples and 5-tuples. [EpiHiper](../approaches/epihiper.md) has separate `transitions` and `transmissions` arrays. [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) formalizes it as `NaturalConversion` versus `ControlledConversion`.

The split is real, because the mathematics differs. But the user does not need two classes to express it, because the rate already says which one it is:

```python
I >> R @ gamma          # no state in the rate: spontaneous
S >> I @ beta * I       # a state in the rate: someone in I is doing the infecting
```

This is exactly how the equations are written, and it is why camdl and SimInf can use one transition syntax. The library reads the kind from the rate and uses it for everything that depends on it (diagrams, stratification, running on a network).

**A bare state in a rate means exposure to that state through the model's contact structure.** With no network given, mixing is homogeneous, so `beta * I` is the textbook β·I/N and `beta` is the classical transmission rate. Add a contact matrix or a network and the same line means "per contact with someone in `I`", with the network supplying who meets whom. The disease does not change. This is the unification argued in [population-and-mixing.md](population-and-mixing.md).

### The source of infection

In the operator form the source is written down: `S >> E @ beta * I` says people leave `S` for `E` because of contact with `I`. Nothing is guessed.

In the string form it is a keyword:

```python
ss.transmission('S -> I', beta=0.05)                 # source not given
ss.transmission('S -> E', beta=0.05, source='I')     # SEIR: given
```

When `source=` is omitted, the library uses the states declared `infectious=True` if there are any, and otherwise the destination. That guess is right for SIR and SIS and wrong for an SEIR whose author forgot the keyword, and **there is no way to tell those apart from the structure**: both are a chain out of `S`. So the guess is not checked; it is printed (`describe()` shows `infection: S -> E, infected by E [guessed]`), and the documentation recommends the operator form, where the question does not arise, for anything beyond SIR and SIS.

Several sources are a sum: `S >> I @ beta * (I + 0.5 * A)` for an asymptomatic state half as infectious, or `source=['I', 'A']` in the string form. This is MIRA's `GroupedControlledConversion`.

### Transmission that is not person to person

`I.n` is the plain number of people in a state, and a rate may be any expression of counts and parameters, used as written. That covers the cases where there is no contact for `beta` to be per:

```python
# Cholera: infection from a water reservoir, with dose response
W = ss.states('W')
water = ss.Compartments(
    shedding = ss.Transition(None, W, rate=shed * I.n),   # an inflow rate is a total per unit time
    decay    = ss.Transition(W, None, rate=ss.perday(0.3)),
)
infection = S >> I @ beta * W.n / (W.n + K)
```

```python
# Malaria: humans are infected by mosquitoes, mosquitoes by humans
Sv, Ev, Iv = ss.states('Sv Ev Iv')
mosquitoes = ss.Compartments(
    biting     = Sv >> Ev @ a * c * Ih.prev,
    incubation = Ev >> Iv @ ss.days(10),
    death      = ss.death(rate=ss.perday(0.1)),
)
Nh = (Sh + Eh + Ih + Rh).n
human_infection = Sh >> Eh @ a * b * Iv.n / Nh
```

(`Sh`, `Eh`, `Ih`, `Rh` are the human disease's states, `Ih.prev` the fraction of humans infectious, and `Nh` the human population size; `a`, `b`, `c` are the biting rate and the two transmission probabilities.)

`ss.Compartments` is a module whose states are counts rather than agents: a water reservoir, a mosquito population, a wastewater signal. It uses the same transition syntax and can sit in the same sim as agent-based modules, per patch if the model is spatial. [SimInf](../approaches/siminf.md)'s split between discrete compartments and continuous node-level variables is the precedent, and its `SISe` models are this pattern. [CK: "Compartments" is a bad name since just sounds like SIR compartments.]

### Branching, merging, and everything in between

**One to many.** TB's fast and slow progression, or recovery versus death:

```python
p = 0.1
S >> (p * Lfast + (1 - p) * Lslow) @ beta * I       # weights on the destinations
ss.Transition(S, [Lfast, Lslow], p=[p, 1 - p], rate=beta * I)   # the same, explicit
ss.transmission('S -> Lfast | Lslow', beta=0.05, p=[0.1, 0.9])  # the same, as a string
```

In real models the split is rarely two constants. `p` takes anything a parameter can take: a table by stratum (`ss.by(age=[0.02, 0.1, 0.3])`) or a function of the sim that returns one value per agent or per group. Where the branches also take different amounts of time, write them as two transitions that each carry `p=` and their own `dur=`: the branch is chosen on leaving the source state, then each waits its own time, which is what Starsim's `set_prognoses` does by hand.

```python
recovery = ss.progression('I -> R', p=0.99, dur=ss.days(10))
death    = ss.progression('I -> D', p=0.01, dur=ss.days(18))
```

[Atomica](../approaches/atomica.md) does this with a *junction* (a zero-residence-time compartment), [epymorph](../approaches/epymorph.md) with `fork(...)`. The weighted arrow says the same thing without a state that is not a state.

**Many to one.** `(S | I | R) >> D @ mu`, or `ss.death(rate=mu)`, which applies to every state.

**Many to many.** Vaccination, aging, and migration move people between strata whatever their disease state. That is `ss.transfer`, kept separate from disease transitions following [MetaCast](../approaches/metacast.md), and it is why interventions can be written without touching the disease model ([stratification.md](stratification.md), [interventions.md](interventions.md)). Anything that fits none of these is written out one transition at a time.

### When a transition is complicated

Declared does not mean simple. Every slot (`rate`, `dur`, `p`, `beta`) accepts a function:

```python
def death_rate(sim):
    age = sim.people.age
    return 0.001 * np.exp(0.08 * age)

death = I >> D @ death_rate
```

The transition is still an edge in the graph: it is drawn, it has a result, it can be stratified. Only its rate is opaque.

A function is called with whatever the run provides. In an agent-based run `sim.people.age` is one value per agent. In a compartmental run it is one value per age group, if age is a stratification. A function that only uses things that exist in both (state counts, parameters, time, stratification coordinates) therefore works in both, and one that reaches for something only agents have stops a compartmental run with a message naming what it asked for. The library does not try to analyze the function in advance; it tries it.

When even the endpoints are not fixed, pass a plain function as a keyword and it is called every step, or override `step()` in a class:

```python
sir = ss.Disease(infection, recovery, weirdness=my_function)
```

```text
>>> sir.describe()
...
weirdness   custom function, called each step
```

Within-host viral load, partnership formation, and care cascades are normally written this way, and that is fine.

### Dwell time is a distribution

Finding 7 of the [landscape review](../approaches/README.md#cross-cutting-findings): "Dwell-time distribution is a first-class axis that most frameworks silently fix." ODE means exponential, and nearly every framework assumes it without saying so. [MEmilio](../approaches/memilio.md) needed four separate model families (ODE / LCT / GLCT / IDE) to span the axis. [EpiHiper](../approaches/epihiper.md) declares a distribution per transition. camdl has `via erlang` / `via hyper_erlang`. `individual` and Starsim schedule sampled times.

**EpiHiper** ([approaches/epihiper.md](../approaches/epihiper.md)):

```json
{ "id": "I_to_R", "entryState": "I", "exitState": "R",
  "probability": 1, "dwellTime": { "gamma": { "alpha": 1.0, "beta": 5.0 } } }
```

**Lingua franca:**

```python
I >> R @ ss.gamma(a=2, scale=5, unit='days')
# [CK: or this form:]
ss.progression('I -> R', dur=ss.gamma(a=2.0, scale=ss.days(5)))
# [CK: string form should accept I -> R, I->R, I>>R, etc]
```

and each run method realizes it its own way:

| Run method | Realization | Precedent |
|---|---|---|
| ODE, exponential | `rate = 1/mean` | Everyone |
| ODE or stochastic compartmental, non-exponential | Erlang sub-stages, generated | camdl `via erlang`, [MEmilio](../approaches/memilio.md) LCT |
| Agent-based | Sample a time, schedule the event | [`individual`](../approaches/individual.md)'s `TargetedEvent`, Starsim's `ti_*` |

The user can also say how, rather than leave it to the method: `stages=3` uses three sub-stages everywhere, including in an agent-based run, which is the way to make an agent-based and a compartmental version of a model match stage for stage.

Three equivalent ways to give the timing, following [EMULSION](../approaches/emulsion.md)'s three transition keys (`rate:`, `proba:`, `duration:`) and Starsim's time types:

```python
ss.progression('I -> R', dur=ss.days(10))          # mean dwell time   → Exp(mean = 10 days)
ss.progression('I -> R', rate=ss.perday(0.1))      # hazard            → Exp(mean = 10 days)
ss.progression('I -> R', rate=ss.probperday(0.1))  # probability per period
```

A scalar `dur=` means *exponential with that mean*, not "exactly 10 days". This is the guess that most needs printing, so `describe()` shows `recovery: I -> R, dwell ~ Exp(mean = 10 days)`. [odin](../approaches/odin.md) achieves the same visibility by making the modeler write `1 - exp(-gamma*dt)` by hand; we print it instead.

### States carry properties

**EpiHiper** ([approaches/epihiper.md](../approaches/epihiper.md)) puts `infectivity` and `susceptibility` on the *state*:

```json
{ "id": "V",      "susceptibility": 0.1 },
{ "id": "wanedV", "susceptibility": 0.5 }
```

That two-number declaration *is* the vaccine model: no multipliers threaded through transmission code, and a new state composes without editing anything else. Starsim's `rel_sus` / `rel_trans` are the per-agent version of the same idea.

**Lingua franca**: keep both, because they answer different questions:

```python
V      = ss.State('V',      rel_sus=0.1)    # a property of being in this state
wanedV = ss.State('wanedV', rel_sus=0.5)
```

with per-agent `rel_sus` retained for heterogeneity that is not a state (age, comorbidity, an intervention someone received), which is what [composition-and-effects.md](composition-and-effects.md) covers.

### Order

Transitions out of the same state are resolved together, as competing hazards, so the order in which a disease's transitions are listed does not favor the first one. Across modules, order is Starsim's loop and it does matter, deliberately: see [composition-and-effects.md](composition-and-effects.md).

### Show it

Because the model is a graph, the pictures are nearly free, and they are what non-modelers actually read. [epiworld](../approaches/epiworld.md) has `ModelDiagram`; [EMULSION](../approaches/emulsion.md) puts `fillcolor` on states because the engine draws the model; [EoN](../approaches/eon.md) literally executes a `networkx.DiGraph`; [Atomica](../approaches/atomica.md)'s transition matrix is an adjacency matrix that a domain expert fills in by hand.

```python
sir.diagram()       # the state-transition diagram
sir.describe()      # the resolved model as text, with every guessed default marked
sir.equations()     # the ODE system, with parameter names
sir.transitions     # the transitions as a table: Atomica's grid
```

These names are deliberately not `plot()`, `summary()`, or `to_df()`, which in Starsim are about results and stay that way.

The table is how the [accessibility benchmark](principles.md) gets met without adopting spreadsheets as a language: since the model is data, a form, a grid, or a web UI is a serialization question, not a language question.

## Trade-offs

- **Two ways to write an arrow is one more than the minimum.** The string form is the shortest path to a running model and is what Starsim users will expect; the operator form reads like the equations and makes the source explicit. Both build the same object, and `describe()` prints one canonical form.
- **Operators have precedence rules the user does not see.** `S >> I @ beta * I` works without parentheses, but `S >> p * A + (1 - p) * B @ rate` does not, and needs them around the destinations. The error for the second should say so.
- **Inference means the described model matters.** If someone writes an SEIR in the string form and forgets `source=`, the only signal is the `[guessed]` line.
- **A function in a rate is opaque.** The library cannot print it as an equation or differentiate it symbolically. It can still run it wherever it works.
- **`ss.Compartments` is a second kind of population.** It is justified by vector-borne and environmental diseases, which are a large share of the global burden and which no amount of contact structure describes.

## Rejected

- **Separate classes for spontaneous and contact-mediated transitions as the user-facing choice.** The split is real and the library tracks it; the user's rate expression already says which is which.
- **A general reaction primitive `A + B → C + D`** ([epipack](../approaches/epipack.md)). `S >> I @ beta * I` covers the same ground while keeping "who moves" separate from "who causes it", which is what stratification and networks need.
- **Requiring `/N` in the rate.** camdl is right that `beta * S * (I/N)` should type-check, but a modeler who writes `/N` by hand can write it wrongly, and the denominator depends on the mixing. It is allowed (via counts, `I.n / N`) and never needed in the common case.
- **A dict shorthand with positional tuples**, `dict(infection=('S -> I', 0.05))`. The first draft proposed it; deciding whether `0.05` is a transmission rate or a recovery rate from context turned out to need rules that misfire (SIRS, for one). The two-line constructor form is as short and is not ambiguous.
- **Integer state indices** ([epiworld](../approaches/epiworld.md)'s `target_states = 2L`). Named, checked references only.
- **A separate `Junction` state kind** ([Atomica](../approaches/atomica.md)). The weighted arrow says the same thing.
- **Mandatory ontology identifiers** ([MIRA](../approaches/notes.md#askem-model-representation-amr--mira)). Optional `ss.State('I', ontology='ido:0000511')` for anyone doing model merging; never required.

## Open questions

- **Should a bare number after `@` be a rate or a duration?** This document says rate. `I >> R @ 10` meaning "after 10 days" is tempting and is how some people think, but `@ 0.1` and `@ 10` cannot both be guessed safely from magnitude.
- **Multi-strain.** HPVsim added genotype as a state dimension; summer2 has a special-cased `StrainStratification`; Starsim would use two disease modules. Both should be expressible: a dimension when the strains share a structure (TB with drug-sensitive and drug-resistant strains, plus a transfer for acquired resistance), separate diseases when they do not. See [stratification.md](stratification.md).
- **How much should `ss.Compartments` be able to do?** Counts with transitions is enough for reservoirs and simple vector models. Whether it needs its own stratification (mosquito species, larval stages by habitat) or should stay deliberately small is not settled. [CK: not sure I entirely understand, but yes will need to support complex vector models, certainly different species. But could just use regular states for that perhaps?]
- **Picking up names from variables.** `ss.Disease(infection, recovery)` finds the names by looking at the caller's variables. That is convenient and slightly magical; the fallback when it fails (two names for one object, a transition built inline) is a name made from the states, `S_to_I`. [CK: ok]
