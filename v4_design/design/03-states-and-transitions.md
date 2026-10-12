# States and transitions

The core of the language. A model is a set of states and the transitions between them, written so that it reads like the equations.

Justified by [best-practices/model-structure.md](../best-practices/model-structure.md).

## Three spellings, one model

There is one transition class, `ss.Transition`, and three ways to write one. They build the same objects and the same model spec ([14-model-spec.md](14-model-spec.md)).

```python
import starsim as ss

# 1. Shorthand constructors with arrow strings (states inferred): the two-line SIR
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)

# 2. Operators (states declared first, like ingredients before the method)
beta = 0.05                      # per day (bare numbers take the sim's unit)
dur  = ss.days(10)

S, I, R   = ss.states('S I R')
infection = S >> I @ beta * I    # "S goes to I at rate beta times I"
recovery  = I >> R @ dur         # a duration after @ is a mean dwell time
sir = ss.Disease(infection, recovery)   # names taken from the variable names

# 3. Fully explicit
sir = ss.Disease(
    ss.Transition(S, I, rate=beta * I, name='infection'),
    ss.Transition('I', 'R', dur=ss.days(10), name='recovery'),
)
```

The first is the short form for SIR-shaped models. The second is the recommended form for everything else, and the one documentation examples beyond the two-line SIR use. The third is for building models programmatically.

## `ss.State`

```python
ss.State(
    name       = None,      # taken from the keyword or the variable name if omitted
    init       = None,      # initial fraction or count; see below
    infectious = False,     # is this state a source of infection by default?
    rel_sus    = 1.0,       # susceptibility multiplier while in this state
    rel_trans  = 1.0,       # infectiousness multiplier while in this state
    label      = None,      # human-readable, for plots and forms
)
```

A state is one concept. It is a compartment when the model runs as an ODE and a boolean per-agent array when it runs as agents; the declaration is the same.

`ss.states('S I R')` returns one `ss.State` per name, in order (the same idiom as `sympy.symbols`); with a single name it returns that one state. Use `ss.State(...)` directly when a state needs properties:

```python
S, R = ss.states('S R')
I    = ss.State(init=0.01)
V    = ss.State(rel_sus=0.2, label='Vaccinated')
```

Inside a class body a state is declared without repeating its name, because Python tells a class attribute what it was called (`__set_name__`):

```python
class SIR(ss.Disease):
    S = ss.State()
    I = ss.State(init=0.01)
    R = ss.State()
```

At module level there is no such hook, so `ss.states('S I R')` names the states once, as `sympy.symbols` does. `S, I, R = ss.states()` with no argument SHOULD also work, by reading the names from the unpacking assignment in the caller's bytecode (as the `varname` package does); it is one step more magical than the name-from-variable lookup the rest of the language relies on, and where it cannot find the names (a REPL, `exec`, an unsupported Python version) it raises an error that says to pass the string ([19-open-questions.md](19-open-questions.md#b12-declaring-states-without-repeating-their-names)).

`rel_sus` and `rel_trans` on a state are properties of being in that state, so a partially protective vaccine state is two numbers on a declaration rather than multipliers threaded through transmission code. They multiply with the per-agent `rel_sus` and `rel_trans` that [07-effects.md](07-effects.md) describes.

A state belongs to the module whose transitions move people into or out of it. A state that appears only inside another module's rate (for example a water reservoir `W` in a disease's infection rate) is a reference, not a member. A state that is an endpoint of transitions in two modules raises `E206 DuplicateDeclaration`. States are ordered by when they were created, so "the first declared state" is the first name in the first `ss.states(...)` call.

### Declare states or infer them

Both work.

- **Inferred.** If no state is declared, every name that appears in an arrow string (or as a string in `ss.Transition`) becomes a state with default properties. This is what makes the two-line SIR possible.
- **Declared.** States created with `ss.states()` or `ss.State()`, whether used in operators or passed as keywords (`ss.Disease(S=ss.State(), I=ss.State(init=0.01), ...)`), are the ingredients list. If any state of a module is declared, every state MUST be: an arrow string or `source=` naming an undeclared state raises `E204 UnknownState` with the closest declared names. This is typo protection; in the operator form Python's own `NameError` does the same job.

### Initial conditions

Give `init` only where it is nonzero. Nobody has to write both 0.99 and 0.01.

```python
I = ss.State(init=0.01)                                    # 1% start in I; the other 99% are placed for you
sir = ss.Disease(infection, recovery, init_prev=0.01)      # the same thing, without declaring I specially
```

1. A value below 1 is a fraction of the population; a value above 1 is a count. A Python `int` of exactly 1 is one individual and a `float` of exactly 1.0 is everyone. The interpretation is printed by `describe()`.
2. `init` accepts the parameter ladder ([04-parameters.md](04-parameters.md)), for example `ss.by(age=[0.0, 0.02, 0.01])`.
3. In a module whose states are people, whatever is left over goes to the first state: the first declared state, or, when states are inferred, the source of the first transition. In an `ss.Compartments` module nothing is left over; a state with no `init` starts at zero.
4. `init_prev=x` on the disease seeds a fraction `x` through its contact-mediated transition, so the seeded individuals start wherever a new infection starts (`I` in an SIR, `E` in an SEIR). A disease with no contact-mediated transition (environmental or vector-borne, below) uses `init` on a state instead.
5. In an agent-based run a fraction is realized as one Bernoulli draw per agent, as in Starsim; in a compartmental run it is used exactly.

## `ss.Transition`

```python
ss.Transition(
    src,                # a state (object or name), several states, or None for an inflow
    dst,                # a state (object or name), a list of states (branching), or None for an outflow
    rate   = None,      # a hazard: number, ss.per, ss.prob, any expression of states and parameters, or a function
    dur    = None,      # a dwell time: a duration or a distribution of durations (instead of rate)
    p      = None,      # branch probability (see Branching)
    stages = None,      # number of sub-stages used to realize the dwell time
    name   = None,
    label  = None,
)
```

At most one of `rate` and `dur` is given. `dur=ss.days(10)` and `rate=ss.perday(0.1)` say the same thing and produce the same model spec.

**The kind of a transition is derived, not chosen.** If the rate mentions a state, the transition is *contact-mediated* and that state is the source of infection; otherwise it is *spontaneous*. A transition with `None` as its source is an inflow and with `None` as its destination an outflow. `describe()` prints the kind of every transition.

A transition's name is used for its results, for addressing it (`sir.infection`), and in messages. How the name is found is in [01-objects.md](01-objects.md#names).

## Arrow strings and the shorthand constructors

```
arrow       ::= [source] arrow_op [dest_list]
arrow_op    ::= "->" | "-->" | ">>"
dest_list   ::= name ("|" name)*
source      ::= name
name        ::= python_identifier
```

Whitespace is ignored and the arrow may be spelled `->`, `-->`, or `>>`, so `'I -> R'`, `'I->R'`, and `'I >> R'` are the same string. An arrow string carries names and a direction and nothing else; rates are Python objects, never text. An empty source is an inflow and an empty destination an outflow.

The lowercase constructors return an `ss.Transition`. Each is a shorter way of writing the explicit form:

| Shorthand | Is the same as |
|---|---|
| `ss.transmission('S -> I', beta=b)` | `ss.Transition('S', 'I', rate=b * I)` |
| `ss.transmission('S -> E', beta=b, source='I')` | `ss.Transition('S', 'E', rate=b * I)` |
| `ss.progression('I -> R', dur=d)` | `ss.Transition('I', 'R', dur=d)` |
| `ss.progression('I -> R \| D', dur=d, p=[0.99, 0.01])` | `ss.Transition('I', ['R', 'D'], dur=d, p=[0.99, 0.01])` |
| `ss.birth('-> S', rate=r)` | `ss.Transition(None, 'S', rate=r * N.n)`, with `N` everyone alive in the module |
| `ss.death('I ->', rate=r)` | `ss.Transition('I', None, rate=r)` |
| `ss.transfer(risk='low -> high', rate=r)` | a transition between coordinates of a stratification (below) |

`ss.progression` takes `dur=` or `rate=`, and optionally `p=`. `ss.transmission` takes `beta=` and optionally `source=` and `p=`. As in Starsim, `beta` may be a dict keyed by network name when it differs by network ([06-routes.md](06-routes.md)).

## The operators

| Expression | Returns | Reads as |
|---|---|---|
| `a >> b` | an `ss.Transition` from `a` to `b` | "a goes to b" |
| `b @ x` | an `ss.Transition` into `b` driven by `x`, with no source yet | "into b, at x" |
| `a >> b @ x` | the complete transition | "a goes to b at x" |
| `S \| I \| R` | a set of states | "any of S, I, R" (several sources) |
| `p * A + (1 - p) * B` | a weighted sum of states | a split between destinations, or a weighted source of infection |
| `I.n` | a count | the number of people in `I` |
| `I.prev` | a fraction | `I.n` divided by the number of people in `I`'s module |
| `None >> S`, `I >> None` | an inflow, an outflow | |

**Python precedence.** `@` and `*` bind more tightly than `>>`, and equally with each other, so `S >> I @ beta * I` parses as `S >> ((I @ beta) * I)`. This works without parentheses because the transition returned by `I @ beta` forwards arithmetic (`*`, `/`, `+`, `-`) to its rate: multiplying it by `I` gives a transition into `I` at rate `beta * I`, and `S >>` then supplies the source. `(S >> I) @ beta * I` builds the same transition. Three consequences are worth knowing:

- `|` binds more loosely than `>>`, so a set of sources needs parentheses: `(S | I | R) >> D @ mu`. Without them Python would try to combine a state with a transition, which raises an error saying to add the parentheses.
- A weighted destination needs parentheses for the same reason `+` does in arithmetic: `S >> (p * A + (1 - p) * B) @ beta * I`.
- For an outflow, `I >> None @ x` relies on `x` to handle the `@`, which typed values (`ss.peryear(0.02)`, `ss.days(30)`, `ss.par(...)`) do and a bare Python number cannot. With a bare number write `(I >> None) @ 0.02` or `ss.death('I ->', rate=0.02)`.

Arithmetic on a finished transition also applies to its rate, so `infection * 0.5` is the same transition at half the rate.

### What may follow `@`

| After `@` | Means | Example |
|---|---|---|
| a duration | a mean dwell time | `I >> R @ ss.days(10)` |
| a distribution of durations | a dwell-time distribution | `I >> R @ ss.lognorm_ex(mean=10, std=3, unit='days')` |
| an `ss.per` rate, or a bare number | a hazard per unit time (a bare number is in the sim's unit) | `I >> R @ 0.1` |
| an `ss.prob` | a probability per period, converted as in [02-types.md](02-types.md) | `I >> R @ ss.probperday(0.1)` |
| an expression containing a state | a force of infection | `S >> I @ beta * I` |
| an expression of counts and parameters | a hazard that depends on the model's state | `S >> I @ beta * W.n / (W.n + K)` |
| a function | called each step to get the rate | `I >> R @ my_rate` |

A transition that reaches build time with nothing driving it and no `p=` raises `E209 UndrivenTransition`.

## Rates that mention states

### A bare state means exposure

**A bare state inside a rate means "exposure to that state through the model's routes".** In `S >> I @ beta * I`, each person in `S` moves to `I` at a hazard of `beta` times their exposure to people in `I`.

- With no network given, mixing is homogeneous with one contact per unit time, so exposure to `I` is `I / N`. Then `beta * I` is the textbook `beta * S * I / N` and `beta` is the classical transmission rate. The user never writes `/ N`.
- When networks or mixing pools are given (`ss.Sim(sir, networks=...)`), they supply who meets whom and how often, and `beta` is per contact. The exposure of each person is then computed from their own contacts. The full definition is in [06-routes.md](06-routes.md).

The model text is the same in both cases. Whether mixing is frequency- or density-dependent is an option on the mixing (`dependence='frequency'` by default) and is printed by `describe()`.

Several sources are a weighted sum: `S >> E @ beta * (I + 0.5 * A)` says asymptomatic people are half as infectious. A rate may multiply the exposure by anything that contains no bare state, such as a parameter, a seasonal curve, or a count (`beta * I * (1 - 0.3 * H.prev)`). The rate MUST be linear in the bare states, because exposure through a network is a sum over contacts; `I ** 2` or `I / (I + K)` raises `E208 NonlinearExposure`, which says to use `I.n` if a literal count was meant.

States that are a source of any contact-mediated transition are marked infectious.

### `I.n` is a count, `I.prev` a fraction

`I.n` is the plain number of people in a state (the number represented, after any scaling from `n_agents` to `total_pop`), and `I.prev` is that number as a fraction of the module's population. A sum of states has them too: `N = S + I + R` then `N.n`. Rates may be arbitrary expressions of counts and parameters, and they are used literally, with no routes involved:

```python
S >> I  @ beta * W.n / (W.n + K)      # environmental: saturating dose-response to a reservoir W
Sh >> Eh @ a * b * Iv.n / Nh          # vector-borne: bites per person from infectious mosquitoes
```

This is how environmental and vector-borne transmission are written. There is no separate route kind for them; worked examples are under [`ss.Compartments`](#sscompartments) below. A transition whose rate uses only counts is spontaneous: every member of the source state has the same hazard.

### Source inference in the string form, and its limits

In the operator and explicit forms the source of infection is in the rate, so nothing is inferred. In the string form it is a keyword:

```python
ss.transmission('S -> E', beta=0.05, source='I')
```

If `source=` is omitted, the source is the states declared `infectious=True`, and if there are none, the destination. The destination is right for SIR, SIS, and SIRS and wrong for SEIR. **This guess cannot be checked**: an SIR and an SEIR with a forgotten `source=` have the same shape, so no warning can tell them apart. It is therefore printed by `describe()` (`infection  S -> E  source: E [guessed]`), and anything that is not SIR- or SIS-shaped SHOULD use the operator form or an explicit `source=`.

## Dwell time

A scalar duration means an exponentially distributed dwell time with that mean, not "exactly 10 days". That is what every ODE means; here `describe()` prints it:

```text
recovery   I -> R   dwell ~ exponential(mean = 10 days)     [guessed: scalar duration]
```

Any `ss.Dist` of durations is accepted:

```python
I >> R @ ss.days(10)                                   # exponential
I >> R @ ss.gamma(a=3, scale=ss.days(10 / 3))          # Erlang with 3 stages, mean 10 days
I >> R @ ss.lognorm_ex(mean=10, std=3, unit='days')    # lognormal
I >> R @ ss.weibull(c=2, scale=ss.days(11))
```

How it is realized follows from the run method:

| Dwell time | As agents | As compartments |
|---|---|---|
| exponential (scalar duration or a rate) | sample a time per agent and schedule the move | one rate, `1 / mean` |
| `ss.gamma` with an integer shape `k` | sample | `k` generated sub-stages at rate `k / mean`, exact |
| any other distribution | sample | no exact form, so `method='auto'` runs the model as agents |

When a compartmental method is asked for explicitly and the distribution has no exact form, the library uses Erlang stages matched to the mean and variance (`k = round((mean / std)²)`, at least 1) and says so in one line (`recovery: lognormal dwell time approximated by 11 Erlang stages`). With `approx=False` this is `E804 CapabilityError` instead. The approximation policy as a whole is in [12-backends.md](12-backends.md).

**The user can say how explicitly** with `stages=k`: use `k` sub-stages with the given mean, in every method, agents included.

```python
ss.progression('I -> R', dur=ss.days(10), stages=3)
ss.Transition(I, R, dur=ss.days(10), stages=3)
```

Generated sub-stages appear in `describe()` and are summed in results, so `n_I` means what the user meant. `ss.erlang` is not in the vocabulary; `ss.gamma` with an integer shape, or `stages=`, says the same thing.

## Branching

One source, several destinations. The three spellings:

```python
S >> (p * Lfast + (1 - p) * Lslow) @ beta * I                           # weights on the destinations
ss.Transition(S, [Lfast, Lslow], p=[p, 1 - p], rate=beta * I)           # explicit
ss.transmission('S -> Lfast | Lslow', beta=beta, p=[p, 1 - p])          # arrow string
```

A branching transition is one named transition with several destinations. Its result is the total flow, and `.sel(to='Lfast')` selects one branch ([09-results.md](09-results.md)).

**`p` is always a branch probability**: the chance of taking this branch, decided when the person enters the source state (or, for a contact-mediated transition, when they are infected). A per-period probability that governs timing is not `p=`; it is a typed rate, `rate=ss.probperday(0.1)`.

- `p` accepts everything a parameter accepts ([04-parameters.md](04-parameters.md)): a constant, `ss.by(age=[...])`, a time series, or a function of the sim returning one value per agent or per group. In real models the split is rarely two constants.
- The probabilities out of one state MUST sum to 1. One entry MAY be left out (`None` in a list, or one transition with no `p=`) and gets the remainder; with two destinations a single value is the first one's probability. A constant split that does not sum to 1 raises `E205 BadBranchSplit`.

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

```python
(S | I | R) >> D @ mu           # one transition, several sources
ss.death(rate=mu)               # every state of the module, to outside the model
```

A transition with several sources is one named transition; `.sel(source='I')` selects the flow from one source. The same movement for every combination of a stratification (many-to-many) is `ss.transfer`.

## Births and deaths

```python
ss.birth('-> S', rate=ss.peryear(0.02))       # crude birth rate: 0.02 per person per year, into S
ss.death('I ->', rate=ss.peryear(0.1))        # disease-specific mortality
ss.death(rate=ss.peryear(0.01))               # background mortality from every state

None >> S @ ss.peryear(0.02) * N.n            # operator forms of the first two
I >> None @ ss.peryear(0.1)
```

- An outflow rate is per person in the source state, like any other transition.
- An inflow has no source state to be "per person" of, so in the operator and explicit forms its rate is the total number entering per unit time. `ss.birth(rate=r)` is the epidemiologist's crude birth rate and is shorthand for `r * N.n`, where `N` is everyone alive in the module.
- In an agent-based run, removals are requested during the step and carried out at its end (Starsim's `request_death`), so a person who dies this step is still visible to the modules that run after the disease ([10-execution.md](10-execution.md)).

Demographic transitions may be declared inside a disease. For anything beyond the simplest case a `ss.Demographics` module (`ss.Births`, `ss.Deaths`, `ss.Pregnancy`) keeps them in one place and lets them run at their own timestep.

## `ss.transfer`

Movement between coordinates of a stratification ([05-dimensions.md](05-dimensions.md)) without a change of disease state:

```python
ss.transfer(risk='low -> high', rate=ss.peryear(0.1))                         # named coordinates: an arrow
ss.transfer(vaccinated=True, rate=ss.peryear(0.2))                            # boolean coordinate: the target value
ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.05), states=['on_treatment'])
```

- The keyword is the stratification; the value is an arrow over its coordinates. For a boolean coordinate write the value people move to, not an arrow.
- A transfer applies to **every disease state at once** unless `states=[...]` restricts it. The third line is acquired drug resistance, which happens only while on treatment.
- It takes `rate=`, `dur=`, and `p=` with the same meanings as any transition, and any of them accepts the parameter ladder.
- A boolean coordinate that the model does not have yet is created, with everyone starting `False`, and printed by `describe()`. A named stratification that does not exist raises `E1102 MissingDimension`.

Vaccination, aging between groups, risk-group turnover, and migration are all transfers, which is why a disease model does not have to mention vaccination to be vaccinated against. Delivering a transfer to a target coverage, to an eligible group, or in particular years is an intervention; see [08-interventions.md](08-interventions.md).

## Functions in any slot

Imperative transitions are fine. Any slot (`rate=`, `dur=`, `p=`, `beta=`, and anything after `@`) accepts a function:

```python
def severity(sim):                       # works on agents and on age groups
    return 0.01 + 0.1 * (sim.people.age > 65)

def clearance(sim, uids):                # agent-only: uses a per-agent attribute
    return 0.2 / (1 + sim.people.viral_load[uids])

ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    outcome   = ss.progression('I -> H | R', p=severity, dur=ss.days(7)),
    discharge = ss.progression('H -> R', rate=clearance),
)
```

- `fn(sim)` returns a scalar or an array with one value per member of the population it is shown. `fn(sim, uids)` is for agent-only logic and returns one value per requested agent. The library tells them apart by the number of arguments.
- A bare number returned from a function is in the sim's unit; typed values are converted as usual.

**The try-it rule.** A function is tried on whatever the run method provides. In an agent-based run it sees agents. In a compartmental run it sees groups: `sim.people` has one row per stratum, an attribute that is a stratification (`age`, `sex`, `vaccinated`) has one value per group, and state counts are available as `I.n`. If the function only uses things that exist in both (state counts, parameters, time, stratification coordinates), it works in both. If it reaches for something agent-only (`uids`, or an attribute that is not a stratification), the compartmental run stops with `E804 CapabilityError` naming the function and what it asked for:

```text
E804 CapabilityError: 'discharge' cannot run with method='ode'.
     Its rate function 'clearance' takes uids and reads people.viral_load, which exists only per agent.
     Run with method='abm', or stratify by viral_load (for example hiv.stratify(viral_load=[0, 3, 5])).
```

There is no static analysis of the function and no declaration of what it reads. The library tries it and reports what happened. `method='auto'` does the same trial at build time and chooses agents if the function does not work on groups.

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

The function is wrapped automatically. `ss.func(fn, ...)` exists only to attach options to it (a label, or its own `dt`). To pass a function as a named *parameter* rather than as behavior, wrap it in `ss.par(fn)`.

The other way is to subclass and override `step()`, as in Starsim ([01-objects.md](01-objects.md#the-class-form)). Both are ordinary, supported ways to write a model. What they cost is method coverage: custom code runs as agents unless it happens to work on groups, by the same try-it rule, and `describe()` and `methods()` say which.

## `ss.Compartments`

An `ss.Compartments` module holds states that are **counts**, not people: an environmental reservoir, a mosquito population, a wastewater signal. It uses the same states and the same transition syntax as `ss.Disease`, and lives in the same sim as agent-based modules. It is how environmental and vector-borne transmission are written. (The name is under review, since "compartments" also describes the states of an SIR; see [19-open-questions.md](19-open-questions.md#b2-how-far-should-sscompartments-go-and-what-is-it-called).)

It differs from `ss.Disease` in three ways: its states are never individual agents, whatever the run method; they are not part of the human population (they do not count toward `N`, age, or sit on networks); and they may be stratified only by dimensions that make sense for a count, usually space (`water.stratify(patch=...)`).

### Cholera: an environmental reservoir

Codeço's SIR-with-water model. People are infected by the concentration of bacteria in the water, with a saturating dose-response; infectious people shed into the water; bacteria decay.

```python
beta = 0.5        # per day: infection hazard at saturating exposure
K    = 1e5        # cells/mL giving half the maximum hazard
xi   = 10         # cells/mL shed per infectious person per day

S, R = ss.states('S R')
I    = ss.State(init=0.001)
W    = ss.states('W')                         # bacteria in the water

cholera = ss.Disease(
    infection = S >> I @ beta * W.n / (W.n + K),
    recovery  = I >> R @ ss.days(5),
)
water = ss.Compartments(
    shedding = None >> W @ xi * I.n,          # inflow: total per day
    decay    = W >> None @ ss.days(30),       # mean lifetime in water
)
sim = ss.Sim([cholera, water], total_pop=100_000, unit='day', stop=365)
```

Nothing here is agent-specific, so `method='auto'` runs it as an ODE. Add `n_agents=10_000` or a network and the people become agents while `W` stays a count; the model text does not change. The corresponding SimInf model keeps `W` in its per-node continuous state `v` and updates it in a C post-timestep function; here it is two more transitions.

### Malaria: a mosquito population

Ross-Macdonald. Humans are infected at a rate proportional to the number of infectious mosquitoes per person; mosquitoes are infected in proportion to human prevalence.

```python
a = 0.3           # bites per mosquito per day
b = 0.5           # probability an infectious bite infects a human
c = 0.5           # probability a bite on an infectious human infects the mosquito

Sh, Eh, Rh = ss.states('Sh Eh Rh')
Ih         = ss.State(init=0.01)
Sv         = ss.State(init=50_000)
Ev, Iv     = ss.states('Ev Iv')
Nh         = (Sh + Eh + Ih + Rh).n

malaria = ss.Disease(
    infection = Sh >> Eh @ a * b * Iv.n / Nh,
    latency   = Eh >> Ih @ ss.days(12),
    recovery  = Ih >> Rh @ ss.days(100),
    waning    = Rh >> Sh @ ss.days(365),
)
mosquitoes = ss.Compartments(
    emergence  = None >> Sv @ ss.perday(5_000),
    infection  = Sv >> Ev @ a * c * Ih.prev,
    incubation = Ev >> Iv @ ss.days(10),
    mortality  = (Sv | Ev | Iv) >> None @ ss.days(10),
)
sim = ss.Sim([malaria, mosquitoes], n_agents=5_000, unit='day', stop=3 * 365)
```

Here `n_agents=` makes the humans agents; the mosquitoes are three numbers. A seasonal mosquito population is `emergence = None >> Sv @ ss.seasonal(mean=5_000, amplitude=0.8, peak='April')`.

### How counts are advanced

In a compartmental run an `ss.Compartments` module is simply more compartments, advanced by the sim's method. Inside an agent-based sim it is advanced deterministically by default (expected flows each step, with the conversions in [02-types.md](02-types.md)), because these quantities are usually large; `ss.Compartments(..., method='binomial')` (or another stochastic compartmental method from [12-backends.md](12-backends.md)) makes its flows random draws instead. It takes its place in the loop with the diseases ([10-execution.md](10-execution.md)), in the order given.

## Competing hazards

Within one module, transitions out of the same state that are driven by rates compete, and they are resolved jointly: the total exit hazard is the sum, the probability of leaving in a step is `1 − exp(−total × dt)`, and the destination is chosen in proportion to each hazard. A discrete-time method MUST implement this as one multinomial draw per state (or its per-agent equivalent), not as sequential binomial draws, so that the order in which transitions are declared inside a module does not bias results.

This is a statement about one module's transitions out of one state. Across modules, and for custom `step()` code, the sim runs an ordered loop in which each module sees the current state left by the ones before it; that order is defined, printable, and specified in [10-execution.md](10-execution.md).

## What is guessed

Every one of these is a default that `describe()` prints and marks.

| Guessed | Rule |
|---|---|
| The set of states | Every name in an arrow string, when none is declared |
| A transmission's source, in the string form | The states declared `infectious=True`, else the destination |
| Which states are infectious | The sources of contact-mediated transitions |
| Initial conditions | Whatever is not given goes to the first state |
| Dwell-time distribution | Exponential, from a scalar duration or a rate |
| Number of sub-stages | From the distribution, unless `stages=` is given |
| Units of bare numbers | [02-types.md §Bare numbers](02-types.md#bare-numbers) |
| Mixing | Homogeneous, one contact per unit time, frequency-dependent, when no network is given |
| Names of transitions | [01-objects.md §Names](01-objects.md#names) |
| Results | One count per state and one flow per named transition ([09-results.md](09-results.md)) |

## Looking at the model

Because the model is a set of state and transition objects, these are derived rather than written. Names like `plot`, `summary`, and `to_df` are kept for results, as in Starsim.

```python
sir.describe()      # the resolved model as text, with every guessed default marked
sir.diagram()       # the state-transition diagram
sir.equations()     # the ODE system, with symbolic parameter names
sir.transitions     # the table of transitions; printing it shows the state-by-state matrix
sir.methods()       # which run methods can run this model
sir.to_json()       # the model spec (14-model-spec.md)
```

`describe()` is specified in [01-objects.md](01-objects.md#describe). `sim.describe()` and `sim.methods()` cover the whole sim.

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

The matrix is Atomica's transition matrix, and it is the natural grid for a spreadsheet or form view of the model. An empty column for a state that also starts empty means nobody can ever be in it, which is usually a typo; the build notes it as `W207 UnreachableState`.

`methods()` prints a coarse table: one row per ingredient that matters (each network, non-exponential dwell times, custom functions), one column per method, and entries `yes`, `approx (how)`, or `no (why)`. It is specified in [12-backends.md](12-backends.md).
