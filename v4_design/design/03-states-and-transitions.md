# States and transitions

The core of the language. A model is a set of states and the transitions between them, written so that it reads like the equations.

Justified by [best-practices/model-structure.md](../best-practices/model-structure.md).

## Three spellings, one model

There is one transition class, `ss.Transition`, and three ways to write one. They build the same objects and the same model spec ([14-model-spec.md](14-model-spec.md)).

```python
import starsim as ss

# 1. Shorthand constructors with arrow strings (states inferred): the two-line SIR
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
    init_prev = 0.01,
)

# 2. Operators (states declared first, like ingredients before the method)
beta = 0.3                       # per day (bare numbers take the sim's unit)
dur  = ss.days(10)

S, I, R   = ss.states(3)         # named from the variables when the disease is built
infection = S >> I @ beta * I    # "S goes to I at rate beta times I"
recovery  = I >> R @ dur         # a duration after @ is a mean dwell time
sir = ss.Disease(infection, recovery, init_prev=0.01)   # names taken from the variable names

# 3. Fully explicit
sir = ss.Disease(
    ss.Transition(S, I, rate=beta * I, name='infection'),
    ss.Transition('I', 'R', dur=ss.days(10), name='recovery'),
    init_prev = 0.01,
)
```

The first is the short form for SIR-shaped models. The second is the recommended form for everything else, and the one documentation examples beyond the two-line SIR use. The third is for building models programmatically.

## `ss.State`

<!-- fragment -->
```python
ss.State(
    name       = None,      # taken from the keyword or the variable name if omitted
    init       = None,      # initial fraction or count; see below
    people     = None,      # is this state people? None: worked out from the transitions (see States that are not people)
    infectious = False,     # is this state a source of infection by default?
    rel_sus    = 1.0,       # susceptibility multiplier while in this state
    rel_trans  = 1.0,       # infectiousness multiplier while in this state
    label      = None,      # human-readable, for plots and forms
)
```

A state is one concept. It is a compartment when the model runs as an ODE and a boolean per-agent array when it runs as agents; the declaration is the same.

`ss.states(3)` returns three `ss.State` objects (`ss.states(1)` returns one state, not a list). They have no names yet: a state takes its name from the variable it is assigned to, found when the module is built by the same lookup that names transitions ([01-objects.md](01-objects.md#names)), so `S I R` is typed once. Use `ss.State(...)` directly when a state needs properties:

```python
S, R = ss.states(2)
I    = ss.State(init=0.01)
V    = ss.State(rel_sus=0.2, label='Vaccinated')
```

`ss.states('S I R')` (the `sympy.symbols` idiom) returns one named state per name, in order, and `ss.State('S')` one named state. Use them when the names are wanted before the module is built, or when the lookup cannot work: states made in a loop or a comprehension, kept in a list or dict, or bound to two variables. A state has no fallback name, because a made-up name for a state would end up in results and in other modules' references. A state that cannot be matched to a keyword, a `name=`, or exactly one variable raises `UnnamedState`:

```text
UnnamedState: state number 2 of the module has no name. It is used in ['S -> <unnamed state>', '<unnamed state> -> R'], and no variable or keyword holding it was found.
     Give the names up front: S, I, R = ss.states('S I R').
```

Inside a class body a state is declared without repeating its name, because Python tells a class attribute what it was called (`__set_name__`):

```python
class SIR(ss.Disease):
    S = ss.State()
    I = ss.State(init=0.01)
    R = ss.State()
```

At module level there is no such hook, which is why the lookup happens when the module is built: `ss.Disease(infection, recovery)` looks among the caller's variables for each state its transitions use. An arrow string or a `source='I'` that names a declared state is resolved at the same moment, after the names are known. Picking names up from variables is convenient and slightly magical; it stays under review in [19-open-questions.md](19-open-questions.md).

`rel_sus` and `rel_trans` on a state are properties of being in that state, so a partially protective vaccine state is two numbers on a declaration rather than multipliers threaded through transmission code. They multiply with the per-agent `rel_sus` and `rel_trans` that [07-effects.md](07-effects.md) describes.

A state belongs to the module whose transitions lead into or out of it. A state that appears only inside another module's rate (for example one disease's `I` in another disease's progression rate) is a reference, not a member. One set of states and transitions can build several diseases: when a disease is given states or transitions that another disease already has, they are copied (together, so they still refer to each other), and parameters stay shared. Two members of one module with the same explicit name raise `DuplicateDeclaration`.

**The first state** (the anchor) is where people are unless they are put elsewhere: the source of the first contact-mediated transition; with none, the earliest state of the states connected to those given initial values. It does not depend on the order in which the states were made.

### Declare states or infer them

Both work.

- **Inferred.** If no state is declared, every name that appears in an arrow string (or as a string in `ss.Transition`) becomes a state with default properties. This is what makes the two-line SIR possible.
- **Declared.** States created with `ss.states(...)` or `ss.State()`, whether used in operators or passed as keywords (`ss.Disease(S=ss.State(), I=ss.State(init=0.01), ...)`), are the ingredients list. If any state of a module is declared, every state MUST be: an arrow string or `source=` naming an undeclared state raises `UnknownState` with the closest declared names. This is typo protection; in the operator form Python's own `NameError` does the same job.

### Initial conditions

Give `init` only where it is nonzero. Nobody has to write both 0.99 and 0.01.

```python
I = ss.State(init=0.01)                                    # 1% start in I; the other 99% are placed for you
sir = ss.Disease(infection, recovery, init_prev=0.01)      # the same thing, without declaring I specially
sir = ss.Disease(infection, recovery, init=dict(S=0.90, I=0.01, R=0.09))   # the whole starting distribution, in one place
```

1. A value below 1 is a fraction of the population; a value of 1 or more is a number of people, whether written `1` or `1.0`. The interpretation is printed by `describe()`.
2. `init` accepts the parameter ladder ([04-parameters.md](04-parameters.md)), for example `ss.by(age=[0.0, 0.02, 0.01])`.
3. Whatever is left over goes to the first state ([above](#ssstate)). This applies to people only. A state that is not people ([below](#states-that-are-not-people)) starts at zero unless it has an `init`, and its `init` is always a count, never a fraction.
4. `init_prev=x` on the disease seeds a fraction `x` through its contact-mediated transition, so the seeded individuals start wherever a new infection starts (`I` in an SIR, `E` in an SEIR). A disease with no contact-mediated transition (environmental or vector-borne, below) uses `init` on a state instead. With no initial values at all and a contact-mediated transition, `init_prev=0.01` is guessed, since nobody infected would mean no epidemic, and `init()` prints R0 and the number infectious at the start.
5. In an agent-based run the number who start in each state is exact (rounded to whole agents), not a chance for each agent; in a compartmental run it is used exactly.
6. `init=dict(...)` on the disease gives the starting value of several states at once. Its keys are state names and each value means what the same value would mean as `ss.State(init=...)`. It is for when the whole distribution is wanted; if the values do not account for everyone, the remainder still goes to the first state.

## `ss.Transition`

<!-- fragment -->
```python
ss.Transition(
    src,                # a state (object or name), several states, or ss.null for an inflow
    dst,                # a state (object or name), a list of states (branching), or ss.null for an outflow
    rate   = None,      # a hazard: number, ss.per, ss.prob, any expression of states and parameters, or a function
    dur    = None,      # a dwell time: a duration or a distribution of durations (instead of rate)
    p      = None,      # branch probability (see Branching)
    stages = None,      # number of sub-stages used to realize the dwell time (with dur, not rate)
    where  = None,      # who it applies to: a condition, as for ss.modify (see below)
    move   = None,      # a change of level in a dimension as well as of state (05-dimensions.md)
    after  = None,      # a minimum stay in the source state before anyone leaves by it (agents only)
    before = None,      # for a rate: it applies only until people have been in the source state this long (agents only)
    offspring = None,   # each person who goes through it has newborns in this state: a state, or (state, number) (agents only)
    name   = None,
    label  = None,
)
```

At most one of `rate` and `dur` is given, and `stages=` goes with `dur=`. `dur=ss.days(10)` and `rate=ss.perday(0.1)` say the same thing and produce the same model spec.

`after=`, `before=`, and `offspring=` are for models that count time in a state or have litters, as EMULSION's minimum durations, escapes, and births do (`weaning = ss.Transition(NB, W, rate=ss.perday(0.5), after=ss.days(21))`, `farrowing = ss.Transition(G, L, dur=ss.days(115), offspring=(NB, 10))`). They need agents.

**The kind of a transition is derived, not chosen.** If the rate mentions a bare state, the transition is *contact-mediated* and that state is the source of infection; otherwise it is *spontaneous*. `describe()` prints the kind of every transition.

**`ss.null` stands for outside the model.** A transition from it is an inflow (births, shedding into a reservoir) and a transition to it is an outflow (deaths, decay). It is a named object and not `None` because Python evaluates `None @ 0.1` before it ever sees the arrow, and `None` cannot answer; `ss.null` can, so `I >> ss.null @ 0.1` works with a bare number. (`I >> None`, with no `@`, also makes an outflow, but the language writes `ss.null`.) `ss.null` is not a state: it has no count, no result, and no name in the model.

A transition's name is used for its results, for addressing it (`sir.infection`), and in messages. How the name is found is in [01-objects.md](01-objects.md#names).

## Arrow strings and the shorthand constructors

```
arrow       ::= [source] arrow_op [dest_list]
arrow_op    ::= "->" | "-->" | ">>"
dest_list   ::= name ("|" name)*
source      ::= name
name        ::= python_identifier
```

Strings are forgiving: whitespace is ignored and the arrow may be spelled `->`, `-->`, or `>>`, so `'I -> R'`, `'I->R'`, `'I >> R'`, and `'I>>R'` all parse to the same thing. `describe()` prints one canonical form, `I -> R`. An arrow string carries names and a direction and nothing else; rates are Python objects, never text (the model spec writes them as text, [14-model-spec.md](14-model-spec.md), but that is the saved form, not something a user writes). An empty source is an inflow and an empty destination an outflow. Branches are separated by `|`; `'I -> (R, D)'` raises `UnknownState`, saying so.

The lowercase constructors return an `ss.Transition`. Each is a shorter way of writing the explicit form:

| Shorthand | Is the same as |
|---|---|
| `ss.transmission('S -> I', beta=b)` | `ss.Transition('S', 'I', rate=b * I)` |
| `ss.transmission('S -> E', beta=b, source='I')` | `ss.Transition('S', 'E', rate=b * I)` |
| `ss.progression('I -> R', dur=d)` | `ss.Transition('I', 'R', dur=d)` |
| `ss.progression('I -> R \| D', dur=d, p=[0.99, 0.01])` | `ss.Transition('I', ['R', 'D'], dur=d, p=[0.99, 0.01])` |
| `ss.birth('-> S', rate=r)` | `ss.Transition(ss.null, 'S', rate=r * N.n)`, with `N` everyone alive in the module: an inflow rate is a total per unit time, so the per-capita birth rate is scaled by the population |
| `ss.death('I ->', rate=r)` | `ss.Transition('I', ss.null, rate=r)`; the arrow is optional, and `ss.death(rate=r)` applies to every state that is people |
| `ss.death(lifetable=ss.lifetable(ages, deaths))` | deaths of everyone from a life table (below) |
| `ss.immigration('-> S', rate=r)` | `ss.Transition(ss.null, 'S', rate=r)`, a number arriving per unit time, whose agents arrive at ages drawn from the population's |
| `ss.emigration('I ->', rate=r)` | as `ss.death()`, whose agents are recorded as emigrants |
| `ss.transfer(risk='low -> high', rate=r)` | a transition between coordinates of a stratification (below) |

`ss.progression` takes `dur=` or `rate=`, and optionally `p=`, `stages=`, `where=`, `move=`, `after=`, `before=`, and `offspring=`. `ss.transmission` takes `beta=` (required) and optionally `source=`, `p=`, and `where=`. The word `transmission` is what stands for "times the source": `ss.transmission('S -> I', beta=b)` is `S >> I @ b * I`, not `S >> I @ b`. There is no `ss.infection` alias; the keyword names the event (`infection=`) and the constructor names the mechanism.

**`beta` may differ by route and direction.** Infectiousness belongs to the disease and contact to the network, and where the pathogen transmits differently through different kinds of contact the disease gives one value per network, as in Starsim:

<!-- fragment -->
```python
infection = ss.transmission('S -> I', beta=dict(mf=[0.008, 0.004], needles=0.03))
sim = ss.Sim(hiv, networks=[ss.MFNet(), needles])
```

The keys are network names. A pair is the two directions of a directed network (here male to female, then female to male). A single number applies to every network. The operator form takes the same dict as its `beta` (`S >> I @ beta * I`). How each network uses its value is in [06-routes.md](06-routes.md).

## The operators

| Expression | Returns | Reads as |
|---|---|---|
| `a >> b` | an `ss.Transition` from `a` to `b` | "a goes to b" |
| `b @ x` | an `ss.Transition` into `b` driven by `x`, with no source yet | "into b, at x" |
| `a >> b @ x` | the complete transition | "a goes to b at x" |
| `S + I + R >> D` | a sum of states as the source | "any of S, I, R goes to D" (several sources); `S \| I` on states is a condition, for `where=` |
| `p * A + (1 - p) * B` | a weighted sum of states | a split between destinations, or a weighted source of infection |
| `I.n` | a count | the number of people in `I`; the same number as the result `n_I` |
| `I.frac` | a fraction | `I.n` divided by the number of people in `I`'s module |
| `I.total` | a count | the number in `I` in the whole population; in a model stratified by place, `I.n` in a rate is the number in the same place |
| `S + I + R` | a sum of states | a source of infection or a denominator in a rate; a result when given to the module by name. Never a new state |
| `ss.null >> S`, `I >> ss.null` | an inflow, an outflow | "from outside", "to outside" |

**Python precedence.** `@` and `*` bind more tightly than `>>`, and equally with each other, so `S >> I @ beta * I` parses as `S >> ((I @ beta) * I)`. This works without parentheses because the transition returned by `I @ beta` forwards arithmetic (`*`, `/`, `+`, `-`) to its rate: multiplying it by `I` gives a transition into `I` at rate `beta * I`, and `S >>` then supplies the source. `(S >> I) @ beta * I` builds the same transition. Three consequences are worth knowing:

- Several sources are a sum of states, `S + I + R >> D @ mu`, which needs no parentheses since `+` binds more tightly than `>>`. `S | I` on states is a condition (for `where=`), and `(S | I) >> D` raises an error that says to write the sum.
- A weighted destination needs parentheses for the same reason `+` does in arithmetic: `S >> (p * A + (1 - p) * B) @ beta * I`.
- For an outflow, `I >> ss.null @ 0.02` parses as `I >> (ss.null @ 0.02)`, and `ss.null` handles the `@` itself, so a bare number works there as it does after a state.
- A weighted destination without its parentheses, `S >> p * A + (1 - p) * B @ rate`, cannot be parsed as intended, and raises a `TypeError`.

Arithmetic on a finished transition is about its flow, never its rate: `0.5 * infection` given to a disease is half of its flow (an outcome, `cases = 0.5 * infection`, [09-results.md](09-results.md)), not a second transition at half the rate, and `infection / S` is its flow per person in `S` (a result). A rate is halved where it is written, or with `ss.modify`. `>>` and `@` return new transitions, so a transition can be reused as a starting point; `@` with a second driver raises an error, and so does forwarding arithmetic to a duration (`S >> (I @ ss.days(10)) * I` raises `WrongDimension`, which says to divide by the duration instead).

### What may follow `@`

| After `@` | Means | Example |
|---|---|---|
| a duration | a mean dwell time | `I >> R @ ss.days(10)` |
| a distribution of durations | a dwell-time distribution | `I >> R @ ss.lognorm_ex(mean=10, std=3, unit='days')` |
| an `ss.per` rate, or a bare number | a hazard per unit time (a bare number is in the sim's unit) | `I >> R @ ss.perday(0.1)`, `I >> R @ 0.1` |
| an `ss.prob` | a probability per period, converted as in [02-types.md](02-types.md) | `I >> R @ ss.prob(0.1, 'day')` |
| an expression containing a state | a force of infection | `S >> I @ beta * I` |
| an expression of counts and parameters | a hazard that depends on the model's state | `S >> I @ beta * W.n / (W.n + K)` |
| a function | called each step to get the rate | `I >> R @ my_rate` |

**A bare number after `@` is a rate**, because "at" reads as "at rate". It is never a duration: `I >> R @ 10` meaning "after 10 days" is how some people think, but `@ 0.1` and `@ 10` cannot both be guessed safely from their size. A duration after `@` is typed (`ss.days(10)`); a bare-number duration uses the keyword form, `ss.progression('I -> R', dur=10)`.

A transition that reaches build time with nothing driving it and no `p=` raises `UndrivenTransition`.

## Rates that mention states

There are two ways to mention a state in a rate, and they answer different questions.

- **`I.n` is how many people are in `I`.** It is a number, the same one that appears in the results as `n_I`. `xi * I.n` says each infectious person sheds `xi`, and nothing more is done to it.
- **A bare `I` is contact with people in `I`**: the chance that a contact is with someone in `I`. With no network that is `I.n / N`, so `beta * I` and `beta * I.n / N` are the same model. With a contact matrix or a network it is the same quantity worked out for each person from whom they actually meet, which cannot be written as one formula. That is why the bare form exists.

So write the bare state when people catch the infection from people, and `.n` when the formula needs the number itself (a reservoir, a vector, a threshold).

### A bare state means exposure

**A bare state inside a rate means "exposure to that state through the model's routes".** In `S >> I @ beta * I`, each person in `S` moves to `I` at a hazard of `beta` times their exposure to people in `I`.

- With no network given, mixing is homogeneous with one contact per unit time, so exposure to `I` is `I.n / N`. Then `beta * I` is the textbook `beta * S * I / N` and `beta` is the classical transmission rate. The user never writes `/ N`.
- When networks or mixing pools are given (`ss.Sim(sir, networks=...)`), they supply who meets whom and how often, and `beta` is per contact. The exposure of each person is then computed from their own contacts. The full definition is in [06-routes.md](06-routes.md).

The model text is the same in both cases. Exposure through a bare state is always frequency-dependent (a share of contacts); density dependence is written with the count, `beta * I.n`, which is what the importers write for a rate such as `beta * S * I`, and a rate per person that grows with a count of people is noted as density-dependent, since `I.frac` is often what was meant. `describe()` prints every rate with its bare states expanded, including the denominator, so what a rate resolved to can always be checked:

```text
infection    S -> I   contact-mediated, source I
             rate = 0.3 × I.n / N                            [homogeneous mixing]
```

Several sources are a sum of exposures: `S >> E @ beta * (I + 0.5 * A)` says asymptomatic people are half as infectious, and `S >> I @ beta * I + beta_vax * I_vax` gives each source its own coefficient. A rate may multiply the exposure by anything that contains no bare state, such as a parameter, a seasonal curve, or a count (`beta * I * (1 - 0.3 * H.frac)`). The rate MUST be linear in the bare states, because exposure through a network is a sum over contacts; `I ** 2` or `I / (I + K)` raises `NonlinearExposure`, which says to use `I.n` if a literal count was meant.

Three things follow from the rule and are worth stating:

- **A rate with no state in it is not transmission.** `S >> I @ beta` is a constant hazard, like `I >> R @ gamma`: everyone in `S` moves to `I` at rate `beta` whether or not anyone is infectious. It is legal and it is spontaneous. In the string form the word `transmission` supplies the source, so `ss.transmission('S -> I', beta=beta)` is `S >> I @ beta * I`.
- **A bare state must be people.** There is no contact with a reservoir to work out, so `S >> I @ beta * W`, with `W` a state that is not people, raises `ExposureToCount`, which suggests `W.n`.
- **A sum of states is not a state.** `N = S + I + R` makes no new state and nobody is "in" `N`. In a rate it is a source of infection (bare) or a denominator (`N.n`); given to the module by name it is a result ([09-results.md](09-results.md)).

States that are a source of any contact-mediated transition are marked infectious.

### `I.n` is a count, `I.frac` a fraction

`I.n` is the plain number of people in a state (the number represented, after any scaling from `n_agents` to `total_pop`), and `I.frac` is that number as a fraction of the module's people. For a state that is not people, `W.n` is simply its value. A sum of states has them too: `N = S + I + R` then `N.n`. Rates may be arbitrary expressions of counts and parameters, and they are used literally, with no routes involved:

<!-- fragment -->
```python
S >> I  @ beta * W.n / (W.n + K)      # environmental: saturating dose-response to a reservoir W
Sh >> Eh @ a * b * Iv.n / Nh          # vector-borne: bites per person from infectious mosquitoes
```

NumPy's elementary functions of an expression are expressions too, so a rate can saturate, switch, or ramp without becoming a function of the sim: `np.minimum`, `np.maximum`, `np.clip`, `np.exp`, `np.log`, `np.sqrt`, `np.abs`, `np.sin`, and `np.cos`. For example, MEmilio's extra risk from symptomatic people once testing and tracing is overwhelmed is `0.35 - 0.1 * np.cos(np.pi * np.clip((0.455 * INS.n - 35) / 140, 0, 1))`. A transition for some people only takes `where=`, a condition as for `ss.modify` ([07-effects.md](07-effects.md#conditions)): `ss.progression('J -> A', rate=ss.perday(1), where=ss.People.age > 1)`. It is the same as a modifier that sets the transition's rate to 0 for everyone else; on a branch chosen with `p=` it changes the branch's probability instead, and the other branches share the rest ([Branching](#branching)). A comparison is 1 or 0, so a rate can stop at a capacity: `I >> Q @ 0.5 * (Q.n < 20)`. A bare state may not appear inside a function or a comparison, since the force of infection must stay linear in the infectious.

This is how environmental and vector-borne transmission are written. There is no separate route kind for them and no separate class for the reservoir or the vector; worked examples are under [States that are not people](#states-that-are-not-people) below. A transition whose rate uses only counts is spontaneous: every member of the source state has the same hazard.

### Source inference in the string form, and its limits

In the operator and explicit forms the source of infection is in the rate, so nothing is inferred. In the string form it is a keyword:

```python
ss.transmission('S -> E', beta=0.05, source='I')
```

If `source=` is omitted, the source is the states declared `infectious=True`, and if there are none, the destination. The destination is right for SIR, SIS, and SIRS and wrong for SEIR, and the two have the same shape, so the guess is only made where it cannot be the SEIR one: when the destination leads on to a state that people then leave for somewhere other than the susceptible state (`S -> E`, `E -> I`, `I -> R`; or `S -> I`, `I -> H`, `H -> R`), `source=` must be given (`SourceNeeded`). Otherwise the guess is printed by `describe()` (`infection  S -> I  contact-mediated, source I  [guessed: source is the destination]`).

## Dwell time

A scalar duration means an exponentially distributed dwell time with that mean, not "exactly 10 days". That is what every ODE means; here `describe()` prints it:

```text
recovery   I -> R   dwell ~ exponential(mean = 10 days)     [guessed: scalar duration]
```

Any `ss.Dist` of durations is accepted:

```python
I >> R @ ss.days(10)                                   # exponential
I >> R @ ss.erlang(k=3, mean=ss.days(10))              # Erlang with 3 stages, mean 10 days; the same as ss.gamma(a=3, scale=ss.days(10 / 3))
I >> R @ ss.lognorm_ex(mean=10, std=3, unit='days')    # lognormal
I >> R @ ss.weibull(c=2, scale=ss.days(11))
```

How it is realized follows from the solver:

| Dwell time | As agents | As compartments |
|---|---|---|
| exponential (scalar duration or a rate) | sample a time per agent when they enter the state, and schedule the move | one rate, `1 / mean` |
| `ss.gamma` with an integer shape `k` | sample | `k` generated sub-stages at rate `k / mean`, exact |
| any other distribution | sample | Erlang sub-stages matched to its mean and variance, with a one-line note |

When the distribution has no exact compartmental form, a compartmental run uses Erlang stages matched to the mean and variance (`k = round((mean / std)²)`, at least 1) and says so in one line (`ss.StagesUnderAuto`: `note: recovery has a lognorm_ex dwell time, approximated by 11 Erlang stages (mean 10 days kept; std 3.02 instead of 3 days)`). This is the one approximation `solver='auto'` makes without being asked: a non-exponential dwell time does not by itself make a model agent-based. A user who wants the distribution sampled exactly asks for `solver='abm'`. `approx=True` hides the note. With `approx=False` nothing is approximated: `'auto'` runs the model as agents, where the dwell time is sampled directly, and an explicit compartmental solver raises `CapabilityError`. The approximation policy as a whole is in [12-backends.md](12-backends.md).

**The user can say how explicitly** with `stages=k`: a compartmental run uses `k` sub-stages with the given mean, and agents draw their dwell time from the Erlang distribution with `k` stages and that mean, which is the way to make an agent-based and a compartmental run of a model match.

```python
ss.progression('I -> R', dur=ss.days(10), stages=3)
ss.Transition(I, R, dur=ss.days(10), stages=3)
```

Generated sub-stages appear in `describe()` and are summed in results, so `n_I` means what the user meant. `ss.erlang(k=3, mean=ss.days(10))`, `ss.gamma` with an integer shape, and `stages=3` give the same model and the same results, but are written differently in the model spec (a gamma dwell time, or an exponential one with `stages: 3`), so their hashes differ.

## Branching

One source, several destinations. The three spellings:

<!-- fragment -->
```python
S >> (p * Lfast + (1 - p) * Lslow) @ beta * I                           # weights on the destinations
ss.Transition(S, [Lfast, Lslow], p=[p, 1 - p], rate=beta * I)           # explicit
ss.transmission('S -> Lfast | Lslow', beta=beta, p=[p, 1 - p])          # arrow string
```

A branching transition is one named transition with several destinations. Its result is the total flow, and the flow into each destination shows in the counts of the destination states; a flow per branch is had by writing each branch as its own transition with `p=` (below).

**`p` is always a branch probability**: the chance of taking this branch, decided when the person enters the source state (or, for a contact-mediated transition, when they are infected). A per-period probability that governs timing is not `p=`; it is a typed rate, `rate=ss.prob(0.1, 'day')`.

- `p` accepts everything a parameter accepts ([04-parameters.md](04-parameters.md)): a constant, `ss.by(age=[...])`, a time series, or a function of the sim returning one value per agent or per group. In real models the split is rarely two constants.
- The probabilities out of one state MUST sum to 1. One entry MAY be left out (`None` in a list, or one transition with no `p=`) and gets the remainder, which `describe()` marks as guessed; with two destinations a single value is the first one's probability. A constant split that does not sum to 1, or `ss.null` as one of several destinations (a branch out of the model is a transition of its own), raises `BadBranchSplit`.
- A modifier on a branch ([07-effects.md](07-effects.md)), a vaccine's efficacy against it, and `where=` on it change its probability, and the other branches share the rest; they do not change how long people take to leave.

**Each branch may have its own dwell time**: decide the branch on entry, then wait. Write two transitions out of the same state, each with `p=` and its own `dur=`:

```python
ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    severe    = ss.progression('I -> H', p=0.1, dur=ss.days(5)),
    mild      = ss.progression('I -> R', p=0.9, dur=ss.days(10)),
)
```

In the operator form a single weighted destination says the same thing: `I >> 0.1 * H @ ss.days(5)` and `I >> 0.9 * R @ ss.days(10)`.

The rule that covers every case: **transitions out of one state that carry `p=` split on entry; transitions that carry only rates compete as hazards** ([below](#competing-hazards)). The two can be mixed: a death hazard out of `I` competes with whichever branch the person was assigned. In a compartmental run, a split with per-branch dwell times is realized as one generated sub-compartment per branch, summed in results. A branch with `p=` and no `rate=` or `dur=` happens immediately on entry, and `describe()` says so.

A junction (a zero-duration pseudo-compartment that only splits its inflow) is not needed and is not part of the language.

## Many-to-one

<!-- fragment -->
```python
S + I + R >> D @ mu           # one transition, several sources
ss.death(rate=mu)               # every state that is people, to outside the model
```

A transition with several sources is one named transition, whose result is the total flow from all of them. The same movement for every combination of a stratification (many-to-many) is `ss.transfer`.

## Births, deaths, and migration

```python
ss.birth('-> S', rate=ss.peryear(0.02))       # crude birth rate: 0.02 per person per year, into S
ss.death('I ->', rate=ss.peryear(0.1))        # disease-specific mortality
ss.death(rate=ss.peryear(0.01))               # background mortality from every state
ss.immigration('-> S', rate=ss.peryear(500))  # 500 people arriving per year, into S
ss.emigration(rate=ss.peryear(0.01))          # people leaving, from every state

N = S + I + R
ss.null >> S @ ss.peryear(0.02) * N.n         # operator forms of the first two
I >> ss.null @ ss.peryear(0.1)
```

- An outflow rate is per person in the source state, like any other transition.
- An inflow has no source state to be "per person" of, so in the operator and explicit forms its rate is the total number entering per unit time. `ss.birth(rate=r)` is the epidemiologist's crude birth rate and is shorthand for `r * N.n`, where `N` is everyone alive in the module.
- `ss.immigration()` and `ss.emigration()` are an inflow and an outflow like any other in a compartmental run. As agents, immigrants arrive at ages drawn from the population's age distribution (people born with `ss.birth()` are newborns), and emigrants are recorded as having left (`sim.results.new_emigrants`) rather than as having died.
- `ss.death(lifetable=ss.lifetable(ages, deaths))` takes deaths from a life table (`deaths` is a life table's d(x)) instead of a rate: as agents, each person's age at death is drawn from it once, given their age; in a compartmental run, which then needs a stratification by age, it is a death rate for each age group.
- As agents, people age in a model whose births are declared, as they do with demographics modules.
- In an agent-based run, removals are requested during the step and carried out at its end (Starsim's `request_death`), so a person who dies this step is still visible to the modules that run after the disease ([10-execution.md](10-execution.md)).

Births, deaths, and migration of everyone are declared once for the population, in `ss.Sim(demographics=[ss.birth(rate=...), ss.death(rate=...)])`: there they run under every solver and add and remove the same people from every disease. The sim makes them transitions of one disease (the first, or the first with the state births name; with no arrow, people are born into each disease's first state), whose people are everyone's: as agents the diseases share the people, and in a compartmental run with several declared diseases their states are tracked jointly (a crossing, see [07-effects.md](07-effects.md)). Births or deaths of everyone declared in more than one disease are `DuplicateDeclaration`, since they would count people twice. `demographics=True` is births and deaths at 20 and 10 per 1,000 people per year, declared the same way; Starsim 3's `birth_rate=` and `death_rate=` (per 1,000, which made the model agents) are removed. With no declared disease, these run as `ss.Births` and `ss.Deaths`, which stay as modules for demography from data (by year, age, and sex) and pregnancy, as agents.

## `ss.transfer`

Movement between coordinates of a stratification ([05-dimensions.md](05-dimensions.md)) without a change of disease state:

```python
ss.transfer(risk='low -> high', rate=ss.peryear(0.1))                         # named coordinates: an arrow
ss.transfer(vaccinated=True, rate=ss.peryear(0.2))                            # boolean coordinate: the target value
ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.05), states=['on_treatment'])
```

- The keyword is the stratification; the value is an arrow over its coordinates. For a boolean coordinate write the value people move to, not an arrow.
- A transfer that happens all the time is part of the model: it is given to the disease as a keyword (`ss.Disease(..., vax=ss.transfer(vaccinated=True, rate=...))`) or to `stratify()`.
- A transfer applies to **every disease state at once** unless `states=[...]` restricts it. The third line is acquired drug resistance, which happens only while on treatment.
- It takes `rate=`, `dur=`, and `p=` with the same meanings as any transition, and any of them accepts the parameter ladder.
- A boolean coordinate that the model does not have yet is created, with everyone starting `False`, and printed by `describe()`. A named stratification that does not exist raises `MissingDimension`.

Vaccination, aging between groups, risk-group turnover, and migration are all transfers, which is why a disease model does not have to mention vaccination to be vaccinated against. Delivering a transfer to a target coverage, to an eligible group, or in particular years is an intervention; see [08-interventions.md](08-interventions.md).

## Functions in any slot

Imperative transitions are fine. Any slot (`rate=`, `dur=`, `p=`, `beta=`, and anything after `@`) accepts a function:

```python
def severity(sim):                       # works on agents and on age groups
    return 0.01 + 0.1 * (sim.people.age > 65)

def clearance(sim, uids):                # agent-only: uses a per-agent attribute
    return 0.2 / (1 + sim.people.viral_load[uids])

ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05, source='I'),
    course    = ss.progression('I -> H | R', p=severity, dur=ss.days(7)),
    discharge = ss.progression('H -> R', rate=clearance),
)
```

- `fn(sim)` returns a scalar or an array with one value per member of the population it is shown. `fn(sim, uids)` is for agent-only logic and returns one value per requested agent. The library tells them apart by the number of arguments.
- A bare number returned from a function is in the sim's unit; typed values are converted as usual.

**The try-it rule.** A function is tried on whatever the solver provides. In an agent-based run it sees agents. In a compartmental run it sees groups: `sim.people` has one row per stratum, an attribute that is a stratification (`age`, `sex`, `vaccinated`) has one value per group, and state counts are available as `I.n`. If the function only uses things that exist in both (state counts, parameters, time, stratification coordinates), it works in both. If it reaches for something agent-only (`uids`, or an attribute that is not a stratification), the compartmental run stops with `CapabilityError` naming the function and what it asked for:

```text
CapabilityError: the function "clearance" cannot run with solver="ode".
     It takes uids, which exist only per agent. Run with solver="abm".
```

A function that fails on groups for any other reason raises its own error, naming it, rather than being taken to need agents.

There is no static analysis of the function and no declaration of what it reads. The library tries it and reports what happened. `solver='auto'` does the same trial at build time and chooses agents if the function does not work on groups.

## Custom per-step behavior

A plain function passed as a keyword to a module is called once per step, after that module's transitions:

```python
def seed_cluster(sim):
    sir = sim.diseases.sir
    if sim.ti == 30:                     # on step 30, infect five susceptible agents
        uids = sir.S.uids[:5]
        sir.S[uids] = False
        sir.I[uids] = True

sir = ss.Disease(infection, recovery, seeding=seed_cluster)
```

The function is called with the sim, after the module's transitions, every step; there is nothing to wrap it in. To pass a function as a named *parameter* rather than as behavior, wrap it in `ss.par(fn)`.

The other way is the class form: subclass `ss.Disease`, declare the states and transitions in the class body, and override `step()` ([01-objects.md](01-objects.md#the-class-form)). A disease written entirely as code, with per-agent arrays and no declared states, subclasses `ss.Infection`, as in Starsim 3, and runs as agents. Both are ordinary, supported ways to write a model. What they cost is the choice of solver: custom code runs as agents unless it happens to work on groups, by the same try-it rule, and `describe()` and `solvers()` say which.

## States that are not people

Some of what a disease model tracks is not people: bacteria in water, a mosquito population, a wastewater signal. These are ordinary `ss.State` objects, declared and used like any other, with the same transition syntax, in the same `ss.Disease` as the states people are in. There is no separate class for them. It is how environmental and vector-borne transmission are written.

**How they are recognized.** A person starts in the disease's first state or is seeded into another, and moves only along transitions. So a state is people if transitions connect it to the disease's first state, in either direction (a maternal-immunity state `M` with `M >> S` is people, although nobody moves from `S` into it), and it is not people if nothing connects them. In the cholera model below no transition joins `W` to `S`, `I`, or `R`, and in the malaria model none joins the mosquito states to the human ones. A transition to or from `ss.null` is not a connection, so a state that is only filled by an inflow and emptied by an outflow is not people. This is a guess, so `describe()` prints it:

```text
states       S, I, R
             W a count, not people                           [guessed: no transition connects W to S]
```

The build also says it once as a note (`ss.StateNotPeople`).

**Saying it outright.** `ss.State(people=False)` declares a count and `ss.State(people=True)` declares people; either one overrides the guess and removes the `[guessed]` mark. `people=True` is needed by a disease that deliberately keeps two sets of people states with no transition between them.

**What follows.** A state that is not people:

- holds a plain count, a number, whatever the solver. It is never represented as agents;
- is not part of `total_pop`, of `N`, or of `I.frac`'s denominator, and `n_agents` and the scale factor between agents and people do not apply to it;
- starts at zero unless given an `init`, which is a count;
- has no age, sex, or risk group and does not sit on networks. Stratifying by a property of people leaves it unsplit; it is split by `patch`, or by dimensions of its own given with `states=` (`malaria.stratify(species=['gambiae', 'funestus'], states=[Sv, Ev, Iv])`). See [05-dimensions.md](05-dimensions.md#states-that-are-not-people);
- is used in a rate as a count, `W.n`. Used bare it raises `ExposureToCount` (below);
- is not touched by `ss.death(rate=...)` with no arrow, which applies to people; its own outflow is written out (`W >> ss.null @ rate`);
- has results like any other state (`n_W`, and a flow for each of its transitions).

```text
ExposureToCount: the rate for 'infection' uses the bare state W, and W is not people.
       rate = beta * W
     A bare state means contact with people in that state, and there is no contact with a count.
     Write W.n for the number itself. If W is people, declare it with ss.State(people=True).
```

### Cholera: an environmental reservoir

Codeço's SIR-with-water model. People are infected by the concentration of bacteria in the water, with a saturating dose-response; infectious people shed into the water; bacteria decay.

```python
beta = 0.5        # per day: infection hazard at saturating exposure
K    = 1e5        # cells/mL giving half the maximum hazard
xi   = 10         # cells/mL shed per infectious person per day

S, I, R, W = ss.states(4)                     # W is bacteria in the water

cholera = ss.Disease(
    infection = S >> I @ beta * W.n / (W.n + K),
    recovery  = I >> R @ ss.days(5),
    shedding  = ss.null >> W @ xi * I.n,      # inflow: total per day
    decay     = W >> ss.null @ ss.days(30),   # mean lifetime in water
    init      = dict(I=0.001),
)
sim = ss.Sim(cholera, total_pop=100_000, unit='day', stop=365)
```

No transition leads from `S` to `W`, so `W` is a count: `total_pop=100_000` is the people in `S`, `I`, and `R`. Nothing here is agent-specific, so `solver='auto'` runs it as an ODE. Add `n_agents=10_000` or a network and the people become agents while `W` stays a count; the model text does not change. The corresponding SimInf model keeps `W` in its per-node continuous state `v` and updates it in a C post-timestep function; here it is two more transitions.

### Malaria: a mosquito population

Ross-Macdonald. Humans are infected at a rate proportional to the number of infectious mosquitoes per person; mosquitoes are infected in proportion to human prevalence.

```python
a = 0.3           # bites per mosquito per day
b = 0.5           # probability an infectious bite infects a human
c = 0.5           # probability a bite on an infectious human infects the mosquito

Sh, Eh, Ih, Rh = ss.states(4)
Sv, Ev, Iv     = ss.states(3)
Nh             = (Sh + Eh + Ih + Rh).n

malaria = ss.Disease(
    infection  = Sh >> Eh @ a * b * Iv.n / Nh,
    latency    = Eh >> Ih @ ss.days(12),
    recovery   = Ih >> Rh @ ss.days(100),
    waning     = Rh >> Sh @ ss.days(365),
    emergence  = ss.null >> Sv @ ss.perday(5_000),      # an inflow rate is a total per unit time
    biting     = Sv >> Ev @ a * c * Ih.n / Nh,
    incubation = Ev >> Iv @ ss.days(10),
    mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
    init       = dict(Ih=0.01, Sv=50_000),
)
sim = ss.Sim(malaria, n_agents=5_000, unit='day', stop=3 * 365)
```

Nobody can move from `Sh` into `Sv`, `Ev`, or `Iv`, so the mosquito states are counts. Here `n_agents=` makes the humans agents; the mosquitoes are three numbers. `Ih=0.01` is a fraction of the people and `Sv=50_000` is a count of mosquitoes. A seasonal mosquito population is `emergence = ss.null >> Sv @ ss.seasonal(mean=5_000, amplitude=0.8, peak='April')`. Several mosquito species are a stratification of the three vector states.

### How counts are advanced

In a compartmental run this changes nothing: every state, human or mosquito, is a number, advanced by the sim's solver with every other flow. In an agent-based run it is what lets the people be agents while the mosquitoes stay three numbers. The counts are advanced once per step with the rest of their disease, from the current counts (`I.n` is then the number of people the agents in `I` stand for), and their flows are applied as expected values, with the conversions in [02-types.md](02-types.md), because these quantities are usually large and are not individuals. Whether those flows should be random draws by default is open. How each solver advances them is in [12-backends.md](12-backends.md#states-that-are-not-people).

## Competing hazards

Within one module, transitions out of the same state that are driven by rates compete, and they are resolved jointly: the total exit hazard is the sum, the probability of leaving in a step is `1 − exp(−total × dt)`, and the destination is chosen in proportion to each hazard. A discrete-time solver MUST implement this as one multinomial draw per state (or its per-agent equivalent), not as sequential binomial draws, so that the order in which transitions are declared inside a module does not bias results.

This is a statement about one module's transitions out of one state. Across modules, and for custom `step()` code, the sim runs an ordered loop in which each module sees the current state left by the ones before it; that order is defined, printable, and specified in [10-execution.md](10-execution.md).

## What is guessed

Every one of these is a default that `describe()` prints and marks, and each row is tested both ways (`tests/test_contracts.py`): a model that makes the guess has it marked, and one that does not, not. The guesses about time (a bare time, the unit) are in [02-types.md](02-types.md), and those about dimensions (a dimension's kind and first level, the population's ages, aging) in [05-dimensions.md](05-dimensions.md).

| Guessed | Rule |
|---|---|
| The set of states | Every name in an arrow string, when none is declared |
| A transmission's source, in the string form | The states declared `infectious=True`, else the destination; required if the destination leads on as in an SEIR (`SourceNeeded`) |
| Which states are infectious | The sources of contact-mediated transitions |
| Initial conditions | Whatever is not given goes to the first state (the source of the first transmission, [above](#ssstate)); with nothing given and a transmission, `init_prev=0.01` |
| Which states are people | Every state connected to the first state by transitions; a state connected to none of them holds a count |
| Dwell-time distribution | Exponential, from a scalar duration or a rate |
| Number of sub-stages | From the distribution, unless `stages=` is given |
| Units of bare numbers | [02-types.md §Bare numbers](02-types.md#bare-numbers) |
| Mixing | Homogeneous, one contact per unit time, when no network is given |
| Names of transitions | [01-objects.md §Names](01-objects.md#names) |
| Size of the population | 10,000 people, when none of `total_pop`, `n_agents`, and `people=` is given |
| Length of the run | 50 units of time (or steps, with no unit), when neither `stop=` nor `dur=` is given |
| Starting value of a fitted parameter | The middle of its bounds (geometric when they span a factor of 10 or more), for `ss.par(bounds=[...])` with no value |

## Looking at the model

Because the model is a set of state and transition objects, these are derived rather than written. Names like `plot`, `summarize`, and `to_df` are kept for results, as in Starsim.

```python
sir.describe()      # the resolved model as text, with every guessed default marked
sir.diagram()       # the state-transition diagram
sir.equations()     # the ODE system, with symbolic parameter names
sir.transitions     # the table of transitions; printing it shows the state-by-state matrix
sir.solvers()       # which solvers can run this model
sir.to_spec()       # the model spec (14-model-spec.md)
```

`describe()` is specified in [01-objects.md](01-objects.md#describe). `sim.describe()` and `sim.solvers()` cover the whole sim.

`equations()` is a method of the disease (or any module with transitions), not of the sim, and it prints parameter names, not numbers:

```text
>>> sir.equations()
dS/dt = -beta * S * I / N
dI/dt =  beta * S * I / N - I / dur
dR/dt =  I / dur
```

It writes the equations the model has under homogeneous mixing or the mixing it was given, with generated sub-stages shown, and says which parts (if any) have no ODE form. It is the only symbolic output in v1.

`transitions` is a table with one row per transition, and printing it also shows the matrix of sources by destinations:

```text
>>> sir.transitions
name        from  to   driven by          kind
infection   S     I    beta * I           contact-mediated, source I
recovery    I     R    dur = 10 days      spontaneous

      S    I            R
S     .    infection    .
I     .    .            recovery
R     .    .            .
```

The matrix is Atomica's transition matrix, and it is the natural grid for a spreadsheet or form view of the model. An empty column for a state that also starts empty means nothing can ever be in it. When the states were inferred from arrow strings that is usually a typo, so building the sim raises `UnknownState`, with the closest names; declared states, or an `init` for the state, say it is meant. (A reservoir is not this case: its column holds an inflow from `ss.null`.)

`solvers()` prints a coarse table: one row per ingredient that matters (each network, non-exponential dwell times, custom functions), one column per solver, and entries `yes`, `approx (how)`, or `no (why)`. It is specified in [12-backends.md](12-backends.md).
