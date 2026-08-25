# States and transitions

The core of the language. A model is a labelled directed graph over states, declared as data.

Justified by [best-practices/model-structure.md](../best-practices/model-structure.md).

## `ss.State`

```python
ss.State(
    name       = None,      # taken from the keyword if omitted
    *,
    init       = None,      # initial share (fraction) or count; see below
    infectious = False,     # does occupancy of this state transmit?
    rel_sus    = 1.0,       # susceptibility multiplier while in this state
    rel_trans  = 1.0,       # infectiousness multiplier while in this state
    label      = None,      # human-readable, for plots and forms
    ontology   = None,      # optional identifier, e.g. 'ido:0000511'
    dtype      = bool,      # storage type in an agent-based backend
)
```

A state is one concept. "Compartment" is what a state is called when the backend is an ODE; "boolean per-agent array" is what it is called when the backend is agents. The declaration is identical and MUST NOT differ by backend.

`rel_sus` and `rel_trans` on a state are properties of *being in that state*, which is EpiHiper's formulation: a two-scalar declaration on a state is a complete vaccine model, with no multipliers threaded through transmission code. Per-agent `rel_sus` and `rel_trans` are retained separately for heterogeneity that is not a state ([07-effects.md](07-effects.md)); the two multiply.

`ontology` is optional and MUST NOT be required by any operation. It exists so that model merging and translation ([16-interop.md](16-interop.md)) can be automated by those who need it, without imposing an identifier on every user who does not.

### Initial conditions

`init` may be a fraction (`0.01`), a count (`ss.count(500)`), or a distribution. The resolution rules:

1. States with an explicit `init` take it.
2. If the remaining fractions sum to less than 1, the balance goes to the **entry state**: the state with no inbound transition. If there is more than one such state, `E203 AmbiguousEntryState` is raised.
3. If no state has an `init`, the entry state gets everything and `summary()` prints `init: S=1.0 [inferred]`.

`init_prev` is retained as sugar for the common seeding case and is equivalent to setting `init` on the transmission destination.

## The arrow

The arrow string is the one place where a string is load-bearing. It carries **structure** — names and a direction — and MUST NOT carry an expression. Rates are Python objects.

```
arrow       ::= source_list ws "->" ws dest_list
source_list ::= name
dest_list   ::= name (ws "|" ws name)*
name        ::= python_identifier
ws          ::= " "*
```

Whitespace around `->` and `|` is optional and normalized away. Exactly one source; one or more destinations. A destination list of length > 1 is a branching transition (below).

Every constructor that takes an arrow MUST also accept the equivalent longhand, for programmatic construction:

```python
ss.transmission('S -> I', beta=0.05)
ss.transmission(src='S', dst='I', beta=0.05)          # identical
ss.progression(src='I', dst=['R', 'D'], p=[0.99, 0.01], dur=ss.days(10))
```

States named in an arrow that were not declared are created implicitly with default properties. This is the state-inference rule and it is why a two-line SIR is possible.

## The transition vocabulary

Closed, small, and semantically meaningful. It is closed because stratification, ODE generation, and format translation all require the framework to know what a transition *means*; an open rate expression cannot be transformed automatically.

| Constructor | Kind | Mediated by |
|---|---|---|
| `ss.transmission(arrow, beta=, source=, ...)` | contact-mediated | a route |
| `ss.progression(arrow, dur=\|rate=\|p=, ...)` | spontaneous | nothing |
| `ss.birth(arrow, rate=\|p=, ...)` | inflow from outside | nothing |
| `ss.death(arrow, rate=\|p=, ...)` | outflow to outside | nothing |
| `ss.transfer(<dim>='a -> b', rate=\|p=\|coverage=, ...)` | movement between dimension coordinates | nothing |
| `ss.process(fn, ...)` | arbitrary Python | — |

There are six. Adding a seventh requires removing one; see [17-vocabulary.md](17-vocabulary.md).

### `ss.transmission`

```python
ss.transmission(
    arrow,                  # 'S -> I'
    *,
    beta,                   # per-contact transmission rate or probability
    source = None,          # which state(s) infect; defaults to the destination
    route  = None,          # which route this transmission uses; defaults to all
    scale  = None,          # 'jump' to force exact event simulation (12-backends.md)
    p      = None,          # branch split, if the arrow branches
)
```

**`beta` is per contact.** It is the probability, or hazard, of transmission across one contact between an infectious and a susceptible individual. It is a property of the pathogen and the pair. The number of contacts is supplied by the route. This factoring is what makes mass-action, contact-matrix, and edge-based transmission one declaration; the force of infection is specified in [06-routes.md](06-routes.md) and never appears in model text.

**`source` defaults to the destination state.** This is correct for SIR, SIS, and SIRS. SEIR MUST name it:

```python
ss.transmission('S -> E', beta=0.05, source='I')
```

The default is chosen so that omitting it in an SEIR is wrong *visibly* — `summary()` will print `E infectious`, which is an obvious error — rather than subtly. A build in which a transmission destination has an onward progression and is not itself the source of any transmission MUST emit `W204 LikelyLatentState`, naming the transition and showing the `source=` spelling.

`source` may be a list, for competing or cooperating pathogens and for multi-strain models: `source=['I_wild', 'I_alpha']`.

### `ss.progression`

```python
ss.progression(
    arrow,
    *,
    dur  = None,     # dwell time: a duration or a dur-valued distribution
    rate = None,     # hazard
    p    = None,     # per-period probability, OR the branch split (see below)
)
```

Exactly one of `dur`, `rate`, `p` specifies the *timing*. The three are equivalent spellings of the same fact and MUST be interconvertible: `dur=ss.days(10)` and `rate=ss.perday(0.1)` produce identical IR.

**A scalar `dur` means an exponential dwell time with that mean.** It does not mean "exactly 10 days". This is what every ODE means and what most frameworks assume without saying; the difference here is that `summary()` prints the expansion:

```text
recovery   I -> R   progression
           dwell ~ Exp(mean = 10 days)     [inferred: scalar dur]
```

### Dwell time

Dwell time is a first-class axis, declared once and realized per backend. This is the sharpest test of the paradigm-independence claim.

```python
ss.progression('I -> R', dur=ss.days(10))                        # exponential
ss.progression('I -> R', dur=ss.erlang(k=3, mean=ss.days(10)))   # Erlang
ss.progression('I -> R', dur=ss.gamma(shape=1.4, scale=ss.days(5)))
ss.progression('I -> R', dur=ss.lognorm(mean=ss.days(10), std=ss.days(3)))
```

| Backend | Realization | Exactness |
|---|---|---|
| ODE, exponential | `rate = 1/mean` | exact |
| ODE, Erlang(k) | `k` generated sub-compartments at rate `k/mean` | exact |
| ODE, other | nearest Erlang or hyper-Erlang fit, discrepancy reported | approximate, named |
| IDE | memory kernel | exact |
| CTMC / tau | exponential clock, or Erlang sub-states | exact where the ODE form is |
| Chain-binomial | per-step probability, or sub-states | exact where the ODE form is |
| Agent-based | sample a time per agent, schedule the event | exact for any distribution |

Generated sub-compartments MUST appear in `summary()` and MUST be aggregated in results, so that `n_I` means what the user meant. Where no exact realization exists, the backend MUST raise `E802 DwellTimeUnrepresentable` and name the alternatives ([15-errors.md](15-errors.md)).

### Branching

One arrow, several destinations:

```python
ss.progression('I -> R | D', dur=ss.days(10), p=[0.99, 0.01])
```

Branch probabilities MUST sum to 1. One entry MAY be omitted (given as `None`) and is inferred as the remainder. A branching transition is a single named transition in the IR with a destination vector, not several transitions; `sim.results.progression` is the total flow and `sim.results.progression.sel(to='D')` selects a branch.

**The `p=` disambiguation rule**, stated normatively because it is the one genuinely ambiguous keyword in the language:

> If the arrow has **more than one destination**, `p=` is the branch split and MUST be a sequence summing to 1. Timing then comes from `dur=` or `rate=`. If the arrow has **one destination**, `p=` is the per-period probability that governs timing.

A single-destination transition given a sequence `p=`, or a branching transition given a scalar `p=` without `dur=` or `rate=`, raises `E205 AmbiguousBranchSplit`.

A junction — a zero-residence-time pseudo-compartment that splits inflow — is not part of the language. The branching arrow says the same thing without adding a state that is not a state.

### `ss.birth` and `ss.death`

```python
ss.birth('-> S', rate=ss.peryear(0.02))          # crude birth rate into S
ss.birth('-> S', rate=ss.peryear(0.02), replacement=True)   # births balance deaths
ss.death('I ->',  rate=ss.peryear(0.1))          # disease-specific mortality
ss.death(rate=ss.peryear(0.01))                  # background: applies to all states
```

An arrow with an empty source is an inflow; an empty destination is an outflow. `ss.death` with no arrow applies to every state, which is summer2's `add_universal_death_flows`.

Demographic transitions declared inside a disease module are legal but discouraged; a `ss.Demographics` module keeps them where they belong and lets them run at their own timestep.

### `ss.transfer`

Movement between coordinates of a dimension, without a change of disease state:

```python
ss.transfer(vaccinated = 'no -> yes', rate=ss.peryear(0.2))
ss.transfer(patch      = 'urban -> rural', p=0.01)
ss.transfer(vaccinated = 'no -> yes', coverage=0.7, years=[2021, 2022], eligible=ss.age > 65)
```

The keyword is the dimension name; the value is the arrow over that dimension's coordinates. The transfer applies to **all disease states at once** unless `states=` restricts it.

This is what vaccination, ageing, and migration are, and keeping it distinct from `ss.progression` is why the disease model never has to mention vaccination. It is the mechanism underneath most of [08-interventions.md](08-interventions.md).

`coverage=` is available on `ss.transfer` and means "reach this fraction of the eligible population over this period", which the compiler converts to the appropriate rate or per-step probability. The conversion is printed. A `coverage=` on a stratified transfer MUST NOT be applied per stratum: doing so delivers `1 − (1 − f)^P` instead of `f`, and attempting it raises `E303 StratifiedCoverage` ([15-errors.md](15-errors.md)).

### `ss.process` — the escape hatch, priced

```python
ss.process(fn, reads=[...], writes=[...])
```

`fn` is called as `fn(module, sim, uids)` and may do anything. It is always available, and its cost is always printed:

```text
>>> sir.summary()
...
weirdness  custom process
  ! not stratifiable, not convertible to ODE/CTMC/tau/SDE, excluded from R0 and jacobian()
```

An `ss.process` MUST cause `capabilities()` to report `✗` for every non-agent-based method, unless it declares `reads` and `writes` that the compiler can verify are confined to quantities with compartmental analogues — in which case it reports `~`. There is no third option: the compiler either knows what the code touches or it does not.

## `ss.Continuous`

A continuous per-node quantity that is not a compartment: an environmental reservoir, accumulated exposure, or vector abundance.

```python
cholera = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.02, source='W'),
    recovery  = ss.progression('I -> R', dur=ss.days(5)),
    W         = ss.Continuous(shed=ss.perday(10), decay=ss.perday(0.3), unit='cells/mL'),
)
```

`source='W'` resolves to the `ss.Continuous` declared in the same module: a `source` name resolves first against states, then against continuous quantities, and an unresolved name raises `E204 UnknownState`. `W` is shed into by infectious individuals and decays. It is SimInf's `v` alongside its `u`, and it is the language's answer to environmental transmission, where there is no contact for `beta` to be *per*. The force-of-infection form for an environmental source is specified in [06-routes.md](06-routes.md) §Environmental.

A `ss.Continuous` quantity is per-node, not per-agent, and therefore MUST NOT be split by an agent-level dimension. Which dimensions apply is declared: `ss.Continuous(..., by=['patch'])`, defaulting to the spatial dimension if one exists and to the whole population otherwise.

## Inference rules

Collected here in full, because every one of them is a guess that `summary()` must print.

| Inferred | Rule | Marked in summary as |
|---|---|---|
| The state set | Every name appearing in any arrow | `[inferred from transitions]` |
| Which states are infectious | The `source` of each transmission | `[inferred: destination of 'infection']` |
| A transmission's `source` | The destination state | `[source inferred]` |
| The entry state | The state with no inbound transition | `[inferred]` |
| Initial conditions | Everything in the entry state | `[inferred]` |
| Dwell-time distribution | Exponential, from a scalar `dur` or a `rate` | `[inferred: scalar dur]` |
| Units | [02-types.md §Bare numbers](02-types.md#bare-numbers) | `[unit inferred: day]` |
| The transition kind, in dict shorthand | [01-objects.md §Form 1](01-objects.md#form-1-the-dict) | `[inferred: transmission]` |
| The result set | One count per state, one flow per transition | listed |

Nothing else is inferred. In particular, the language MUST NOT infer a dimension's kind from anything except its name, MUST NOT infer a route, and MUST NOT infer a `source=` list of more than one state.

## Order independence

Declared transitions are a **set**, not a sequence. Within a timestep, every declared transition sees start-of-step state and all are applied together. Declaration order MUST NOT affect results, and MUST NOT affect the IR ([14-ir.md](14-ir.md) normalizes it away).

Competing transitions out of the same state — two progressions from `I`, or a progression and a death — are resolved by competing hazards, not sequentially: the total exit hazard is the sum, and the split is proportional. In a discrete-time backend this MUST be implemented as a multinomial draw, not as sequential binomials, because sequential binomials give the first-listed transition an advantage that depends on declaration order.

Only `ss.process` escape hatches need an execution order, and that is what the `Loop` is for ([10-execution.md](10-execution.md)).

## Projections

Because the model is a graph, these are derived rather than written:

```python
sir.plot()          # state-transition diagram
sir.summary()       # the resolved model as text, with every inferred default
sir.to_df()         # the transition matrix: source x destination, with the driving parameter
sir.equations()     # the ODE system (12-backends.md)
sir.to_dict()       # the IR (14-ir.md)
```

`to_df()` is Atomica's transition matrix, and it is how the accessibility benchmark is met without adopting spreadsheets as a language: since the model is data, a grid, a form, or a web UI is a serialization question. It also makes structural errors visible at a glance — an empty row is an absorbing state, an empty column is unreachable — and the build SHOULD warn about both (`W206 AbsorbingState`, `W207 UnreachableState`) unless the state is named `D` or declared `absorbing=True`.
