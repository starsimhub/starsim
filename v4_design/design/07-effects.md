# Effects, expressions, and composition

How one module modifies a quantity owned by another, how modifications compose, and how conditions are written.

Justified by [best-practices/composition-and-effects.md](../best-practices/composition-and-effects.md).

## `ss.Effect`

```python
ss.Effect(condition, target, **combiner)     # conditional
ss.Effect(target, **combiner)                # unconditional
```

```python
ss.Effect(syphilis.active, hiv.rel_trans, multiply=2.0)
```

Read as: *while syphilis is active, multiply HIV transmissibility by 2.* This replaces the arbitrary-Python connector for the common case, and it is the single most load-bearing construct in the language after the transition vocabulary.

| Argument | Type | Meaning |
|---|---|---|
| `condition` | `ss.Expr`, optional | when and to whom the effect applies |
| `target` | a modifiable quantity | what is modified |
| combiner | exactly one of `multiply=`, `add=`, `set=`, `min=`, `max=` | how |

Additional keywords:

```python
ss.Effect(..., min_duration=ss.weeks(4))   # hysteresis: once active, stay active this long
ss.Effect(..., name='school_closure')      # for results and diagnostics; defaults to the variable name
```

An effect declared at module scope is registered with the sim when its module is added. An effect passed to `Sim(interventions=...)` is registered directly.

### What this buys

- Two effects on the same target compose by a **declared rule** rather than by execution order.
- `sim.what_modifies(hiv.rel_trans)` has an answer, and MUST list every effect with its combiner and its source location.
- The effect is serializable, diffable, hashable, and diagrammable.
- The dependency `syphilis` → `hiv.rel_trans` is **derived**, so execution order does not need to be declared ([10-execution.md](10-execution.md)).

## Targets

The set of modifiable quantities is closed. This constraint is what makes effects composable and analysable; widening it beyond this table requires an amendment to this document.

| Target | Meaning |
|---|---|
| `<disease>.rel_sus` | susceptibility to acquiring |
| `<disease>.rel_trans` | infectiousness when infected |
| `<disease>.<transition>.beta` | per-contact transmissibility of one transition |
| `<disease>.<transition>.rate` / `.dur` | progression speed |
| `<disease>.<transition>.p` | branch probability, e.g. severity |
| `<route>.contacts` | contact rate on a route or layer |
| `<route>.matrix` | the mixing matrix itself |
| `<mobility>.matrix` | movement rates |
| any declared `ss.Par` | anything the modeler named |

The last row is what recovers generality without recovering the hazard: a quantity the modeler named is a quantity the modeler can be told about.

A target that is not in this set raises `E504 UnmodifiableTarget`, naming the quantity and, where one exists, the modifiable quantity that stands in for it.

## Combiners

Effects on one target resolve in four stages, in this order. The order is normative, so that a result never depends on registration order.

1. **Replace.** At most one active `set=`. Two active `set=` effects on one target raise `E502 IncompatibleCombiners`, naming both and their source locations.
2. **Scale.** All active `multiply=` compose by product.
3. **Shift.** All active `add=` compose by sum.
4. **Clamp.** `min=` then `max=`. A `min` above a `max` raises `E505 InvalidClamp`.

```
value = (set_value if any else base) × Π multipliers + Σ addends
value = min(max(value, min_clamp), max_clamp)
```

**Multiplicative is the default** because it is the defensible modeling assumption and because it commutes. Additive composition is available because it is a legitimate modeling choice that at least one framework made deliberately and documented — but two 60% reductions summing to 120% has to be clamped somewhere, and clamping is a modeling decision that should be visible.

Mixing `multiply=` and `add=` on the same target is legal and emits `W503 MixedCombiners`, naming the convention above, because the result depends on it.

```text
E502 IncompatibleCombiners: 'hiv.rel_trans' has two active effects with 'set':
       ss.Effect(art.on_treatment, hiv.rel_trans, set=0.05)     [stisim.py:31]
       ss.Effect(cure.taken,       hiv.rel_trans, set=0.00)     [stisim.py:44]
     'set' is exclusive. Use multiply= for both, or make one conditional on the other.
```

## `ss.Expr`

A condition is an expression object, not a boolean. `ss.age > 65` returns an `ss.Expr`; it is never evaluated to `True` or `False` at declaration time. This is the same mechanism dataframe and ORM libraries use, so the syntax is already familiar to the audience, and it is what makes eligibility data rather than code.

### Atoms

```python
ss.age, ss.sex, ss.uid                       # built-in agent attributes
ss.time, ss.year, ss.weekday, ss.date(...)   # model-level observables
sir.I, sir.vaccinated                        # any state or dimension coordinate
sim.results.infection.weekly            # any result
ss.has(product)                              # has received a named product
ss.scheduled('recovery')                     # has a pending scheduled transition
ss.N                                         # total population
```

### Operators

```python
&  |  ~                       # and, or, not
>  <  >=  <=  ==  !=          # comparison
+  -  *  /                    # arithmetic, for building derived quantities
a <= x <= b                   # chained comparison, for time windows
.isin([...])                  # membership
```

```python
(ss.age > 65) & ~sir.vaccinated
ss.date('2020-03-15') <= ss.time <= ss.date('2020-06-01')
sim.results.infection.weekly > 1000
ss.weekday == 'Fri'
```

The last two are worth noting. A predicate over results is a **trigger** and needs no additional machinery — reactive policy is declarative for free. And a 200-line JSON specification of "reduce school contacts on Fridays" becomes `ss.Effect(ss.weekday == 'Fri', school.contacts, multiply=0.0)`.

### Rank

Every expression has a **rank**, which determines where it can be used and which backends can evaluate it:

| Rank | Depends on | Usable in | Compartmental backends |
|---|---|---|---|
| `scalar` | time, results, parameters | anywhere | yes |
| `group` | dimension coordinates (+ scalar) | anywhere | yes |
| `agent` | per-agent attributes that are not dimensions | agent-based only | no |

`ss.age > 65` is rank `group` when `age` is a dimension and rank `agent` when it is not — so the same expression is compartmentally meaningful in a model stratified by age and not in one that is not. `ss.uid % 7 == 0` is always rank `agent`. The rank is computed at build time, reported in `capabilities()`, and is the mechanism by which the compiler decides whether an intervention survives paradigm conversion.

### Evaluation semantics

- Conditions are evaluated against **start-of-step state**, like everything else in a step ([10-execution.md](10-execution.md)).
- An expression referencing a result refers to the most recently completed value of that result, never to a partially accumulated one.
- `min_duration=` makes an effect latch: once the condition first becomes true, the effect stays active for at least that duration regardless of the condition. Policies do not flap, and without this, threshold triggers oscillate at the step frequency.

### The escape hatch

```python
eligible = lambda sim, uids: my_complicated_thing(sim, uids)
```

Always available, and priced:

```text
! opaque predicate: not serializable, not diagrammable, rank=agent,
  excluded from every non-agent-based backend
```

## Effects on a per-agent versus per-group quantity

`rel_sus` and `rel_trans` exist at two levels and they multiply:

```
rel_sus[agent] = rel_sus[state] × Π (effects with agent-rank conditions matching this agent)
rel_sus[group] = rel_sus[state] × Π (effects with group-rank conditions matching this group)
```

A state-level `rel_sus` declared on `ss.State('V', rel_sus=0.1)` is the declarative default and is available in every backend; an effect-driven modifier is what handles heterogeneity that is not a state. Both are visible in `what_modifies()`.

## The imperative escape hatch

```python
class Complicated(ss.Connector):
    requires = [ss.Needs('something')]
    writes   = [hiv.rel_trans]

    def step(self):
        ...
```

- `writes` grants write access to named quantities. A module without a matching `writes` entry receives a read-only view, and a write attempt raises `E501 UndeclaredWrite` naming the quantity and the module.
- `requires` grants read access, and is checked against available data at build time ([04-parameters.md](04-parameters.md)).

This is capability-based access control for model state. It costs nothing at run time — it is compile-time information — and it closes the largest hole in the no-hidden-behavior goal: a module that writes another module's state without saying so.

Both fields are inferred for declarative modules and MUST be written only by modules that define `step()`.

## Execution order

Order is settled in this priority, and each level exists because the one above it is insufficient:

1. **Declared transitions are simultaneous.** All see start-of-step state; all are applied together. The declarative part of a model has no order dependence at all.
2. **Effects are resolved before the quantities they modify are read.** This is derivable from the declarations, so it is derived, not declared.
3. **Imperative code is ordered by the `Loop`**, which remains a data object: printable, `to_df()`-able, plottable, and insertable-into.
4. **Phases are named**, so that a module can say `phase='cleanup'` rather than competing for a position in a list.

The cost of level 1 is that within-step causal chains — "infect, then immediately test the newly infected" — must be written as explicit sub-steps. That is the right trade, because it makes the chain visible. See [10-execution.md](10-execution.md).

### Cycles

Two effects that each depend on the other's target form a cycle. Within a single step this raises `E503 EffectCycle`, naming both effects and the quantities involved. Across steps it is fine and common — HIV raising syphilis susceptibility while syphilis raises HIV transmissibility is a real model — and is resolved by the start-of-step rule, since each reads the other's previous value.

## Rejected

- **Last-write-wins on shared state.** This is the specific defect being fixed. `rel_trans[coinf] = 2.0` in one module silently overwriting whatever another module wrote is the failure mode; two effects with a declared combiner is the fix.
- **Additive composition as the global default.**
- **Execution order as the primary mechanism.** Order should settle what is genuinely ambiguous, not what is derivable.
- **A full value-pipeline vocabulary** — producers, modifiers, post-processors, lifecycle phases, population views, resource graphs — as user-facing surface. The semantics are right and are adopted wholesale; the vocabulary is one line.
- **Arbitrary state access without declaration.** Fast, unanalysable, and free to fix.
- **A fifth first-class object for protective measures** (a "tool" carrying four multipliers). `ss.Effect` covers it with a wider target list and a declared combiner; adopting the constraint without the vocabulary is the right trade.
