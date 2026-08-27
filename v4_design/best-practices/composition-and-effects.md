# Composition and effects

How one module modifies a quantity owned by another, and what runs when.

## Recommendation

1. **A cross-module effect is a declaration, not a mutation.** `ss.Effect(when, what, multiply=)` replaces the arbitrary-Python connector for the common case.
2. **Declare the combination rule, because it is a modeling decision.** Two interventions each reducing transmission by 40% might compose multiplicatively or additively, and the answer changes.
3. **Multiplicative by default.**
4. **A module declares what it reads and what it writes.** Silent writes to another module's state are the largest hole in the "no hidden behavior" goal. *[CK: agree, but let users do this if they want to, but provide good alternatives so they almost never do have to.]*
5. **Declared transitions are order-independent within a step.** Only imperative code needs an execution order, and Starsim's `Loop` already makes that order inspectable. *[CK: I'm not sure how or if this would work in practice.]*
6. **Keep one module abstraction.** `ss.Module` for diseases, networks, demographics, interventions, connectors, and analyzers is right.

## Why

### The Starsim connector problem, stated plainly

From the [Starsim review](../approaches/starsim.md), a limitation:

> **Connectors are unconstrained.** Cross-module interaction is "arbitrary Python that mutates another module's state". This is why multi-disease works at all, and it is also the largest hole in the "no hidden behavior" goal: `rel_trans[coinf] = 2.0` in one module silently overwrites whatever another module wrote.

The worked example in that review is exactly the failure mode:

```python
class HIVSyphConnector(ss.Connector):
    def step(self):
        coinf = (self.sim.diseases.syphilis.active).uids
        self.sim.diseases.hiv.rel_trans[coinf] = 2.0
```

Add a second connector that also writes `hiv.rel_trans` — a risk factor, a vaccine, a seasonal term — and the result depends on which ran last. Nothing declares that either one touches it. [LASER](../approaches/laser.md) has the same hazard in its strongest form: "All components write columns of one `LaserFrame`. This is fast and it is exactly the hazard Vivarium's `PopulationView` and value pipelines exist to prevent."

### Vivarium solved this

From the [Vivarium review](../approaches/vivarium.md) — the entry exists in this collection for this one mechanism:

```python
builder.value.register_rate_producer("diarrhea.incidence_rate", source=self._base_rate)
builder.value.register_value_modifier("diarrhea.incidence_rate", modifier=self._apply_risk_effect)
```

> A component that produces a rate does not know who modifies it. A component that modifies a rate does not know who produces it — only its name. **Multiple modifiers compose by a declared rule**, not by whoever ran last.

And in the model specification, a component whose entire declared purpose is "this modifies that":

```yaml
- RiskEffect('child_growth_failure', 'infected_with_diarrhea.incidence_rate')
```

[epiworld](../approaches/epiworld.md) arrives at a more constrained version: a **tool** is a bundle of multiplicative effects on susceptibility, transmissibility, recovery, and death that an agent carries. Its review: "It is more constrained than Starsim's connectors and, for exactly that reason, does not have Starsim's last-write-wins problem."

[EMOD](../approaches/emod.md)'s `Waning_Config` is the same idea for a single product's efficacy over time.

### And `epidemics` proved the combination rule must be statable

[`epidemics`](../approaches/epidemics.md) composes interventions **additively** — `C × (1 − (X + Y))`, not `C × (1 − X)(1 − Y)` — because additive was judged easier for users to understand, and it **wrote the decision down in a design-principles vignette**. Its review: "Multiplicative composition is the more defensible modelling assumption and the more common one; the package chose the other and documented the choice and its rationale."

That is finding 9 of the [landscape review](../approaches/README.md#cross-cutting-findings): the combination rule is a modeling decision that must be statable. `epidemics` is the proof that a global default silently chosen would be wrong for someone.

## The proposal

### One line for the common case

**Starsim today** (7 lines, opaque, last-write-wins):

```python
class HIVSyphConnector(ss.Connector):
    def step(self):
        coinf = self.sim.diseases.syphilis.active.uids
        self.sim.diseases.hiv.rel_trans[coinf] = 2.0
```

**Lingua franca:**

```python
ss.Effect(syphilis.active, hiv.rel_trans, multiply=2.0)
```

Read as: *while syphilis is active, multiply HIV transmissibility by 2.* The first argument is a [predicate over agent or model state](interventions.md#eligibility-as-a-predicate); the second names the target quantity; the third is the effect and its combination rule. *[CK: I'm worried this isn't flexible enough. Very rarely are "effects" as simple as "one source state, one destination state, one multiplier". It's usually more like "Find people who are HIV positive, filter through a complex age/sex probability condition, and then schedule them for an HIV test on their next ANC visit." There might be 5 source states and 6 target states, with some things multipliers and other boolean flips etc. I don't see how you'd represent something like that in this framework.]*

What this buys, all of it for free:

- Two effects on `hiv.rel_trans` compose by the declared rule instead of by execution order.
- `sim.what_modifies(hiv.rel_trans)` has an answer — Vivarium's "you can ask what modifies this and get an answer".
- The effect is serializable, diffable, and diagrammable.
- The compiler knows `syphilis` must run before `hiv` transmission, so the ordering is *derived* rather than declared. *[CK: but what if HIV prevalence affects syphilis transmision -- which it in fact does? I don't think this is workable in practice.]*

### The combination rule

```python
ss.Effect(a, x, multiply=0.6)     # default: composes as 0.6 × 0.6 = 0.36
ss.Effect(a, x, add=-0.4)         # epidemics-style additive
ss.Effect(a, x, set=0.0)          # exclusive; errors if another 'set' applies
ss.Effect(a, x, min=0.1)          # clamp
```

Multiplicative is the default because it is the defensible assumption and because it commutes, so results do not depend on order. When two effects use incompatible rules on the same target, that is an error with both effects named — not a silent precedence.

```text
EffectError: 'hiv.rel_trans' has two effects with incompatible combiners:
  ss.Effect(syphilis.active, hiv.rel_trans, multiply=2.0)     [stisim.py:14]
  ss.Effect(art.on_treatment, hiv.rel_trans, set=0.05)        [stisim.py:31]
  'set' is exclusive. Use multiply= for both, or make one conditional on the other.
```

### What can be a target

A closed set of named, modifiable quantities keeps this analysable. Following [EpiHiper](../approaches/epihiper.md)'s state properties and [epiworld](../approaches/epiworld.md)'s four tool effects: *[CK: as noted above, I think this will be too restrictive to be useful.]*

| Target | Meaning |
|---|---|
| `<disease>.rel_sus` | Susceptibility to acquiring |
| `<disease>.rel_trans` | Infectiousness when infected |
| `<disease>.<transition>.rate` / `.dur` | Progression speed |
| `<disease>.<transition>.p` | Branching probability (e.g. severity) |
| `<route>.contacts` | Contact rate on a mixing layer |
| any declared `ss.Par` | Anything the modeler named |

epiworld's tool is exactly the first, second, and a slice of the third; its review notes the constraint "is why it composes". We take the same constraint and widen it to any named parameter, which recovers the generality without recovering the hazard.

### The escape hatch is still there

```python
class Complicated(ss.Connector):
    writes = [hiv.rel_trans]     # declared
    def step(self):
        ...
```

`writes = [...]` is [Vivarium](../approaches/vivarium.md)'s `PopulationView` — "a component states which columns it touches, and the framework enforces it. This is capability-based access control for model state, and nothing else in the review has it." A connector without a `writes` declaration gets a read-only view and a clear error, not a silent success. *[CK: that's fine, but the user has to remember to write `writes`. Or, worse, they write it, then someone else updates `step()` and forgets to update `writes`.]*

It pairs with [epymorph](../approaches/epymorph.md)'s declared `AttributeDef` requirements: reads declared as `requires`, writes declared as `writes`, both checked, both feeding the dependency graph.

### Execution order

Four positions in the review:

| Framework | Treatment |
|---|---|
| [`individual`](../approaches/individual.md) | **Irrelevant.** Processes queue updates; the loop applies them after every process has run, so every process sees start-of-step state |
| [Vivarium](../approaches/vivarium.md) | **Declared.** Named lifecycle phases (`time_step_prepare` → `time_step` → `time_step_cleanup` → `collect_metrics`) with per-component priorities |
| [Starsim](../approaches/starsim.md) | **Inspectable.** `ss.Loop` is a data object: printable, `to_df()`-able, plottable, insertable-into |
| [Mesa](../approaches/notes.md#mesa) | **A user-selected modeling decision**, with the options named |
| Everyone else | Whatever order the code happens to run |

Synthesis, in priority order:

1. **Declared transitions are simultaneous.** All see start-of-step state; all are applied together. This is `individual`'s queued-update semantics, "the strongest correctness property of any agent-based framework in the review", and it means the whole declarative part of a model has no order dependence at all. *[CK: I don't like this. Consider infection onset, testing, and treatment. If you specify the order as infection.step() -> testing.step() -> treatment.step(), you can get the whole chain happening in one timestep. Otherwise, you create an artificial two-timestep delay for the fastest possible treatment.]*
2. **Effects are resolved before the quantities they modify are read**, which is derivable from the declarations. *[CK: it's not clear to me if this solves another issue we run into, which is that you might have an agent-heterogeneous `rel_sus`, modify the array by some factor with an intervention, and then return to the original state. Currently, we sloppily copy `rel_sus` into eg `orig_rel_sus`, and then restore it when the intervention ends. Would be nice to handle this natively. But we also don't want to (I assume) be multiplying together 5 different `rel_sus` arrays on each timestep, when most of them are just 1.]*
3. **Imperative code is ordered by Starsim's `Loop`**, which stays as it is — it is one of Starsim's better ideas and it directly serves the no-hidden-behavior goal.
4. **Phases exist and are named** (Vivarium's contribution), so a module can say `phase='cleanup'` rather than fighting over a position in a list. *[CK: maybe, I'd have to see the implementation. Sounds like phases are just different locations in a list, though.]*

The cost, stated by `individual`'s review: within-step causal chains ("infect, then immediately test the newly infected") must be written as explicit sub-steps. That is the right trade, because it makes the chain visible.

### One module abstraction

Starsim's `ss.Module` — the same lifecycle for diseases, networks, demographics, interventions, connectors, and analyzers — is what makes multi-disease and non-disease health states composable, and it is why FPsim (contraception, pregnancy, birth outcomes) fits in the same framework as HIV. Keep it.

What changes is only that the module contract gains two fields:

```python
class MyModule(ss.Module):
    requires = [...]   # epymorph's AttributeDef: what I read
    writes   = [...]   # Vivarium's PopulationView: what I modify
```

Neither is required for a module that only declares transitions, because those are derivable.

## Trade-offs

- **A closed target list means some effects need the escape hatch.** Accepted, and the escape hatch declares its writes, so the analysis survives.
- **Declared reads and writes are boilerplate.** Mitigation: only imperative modules need them; declarative ones are inferred. If a module has no `step()`, it should never need either field.
- **Derived ordering can be ambiguous.** Two modules that each read the other's output is a cycle, and the right response is an error naming both — which epymorph's and Vivarium's dependency graphs already do. *[CK: disagree -- this is fine, it rarely matters, and convention can break the deadlock, e.g. run syphilis before HIV because we care more that the HIV results are accurate.]*
- **`individual`'s simultaneity is not free at scale.** Queued updates mean a second pass over the changed set. In practice this is cheap relative to the transmission step, and it is what buys order-independence. *[CK: I'm not sure it's worth it.]*

## Rejected

- **Last-write-wins on shared state** (Starsim connectors today, LASER components). This is the specific defect being fixed.
- **Additive composition as the global default** (`epidemics`). Documented honestly there; still clamps at 100% and still surprises. Multiplicative default, `add=` available.
- **Execution order as the only mechanism** (EpiModel's `module.order`, LASER's component list, flepiMoP's pipeline). Order should settle what is genuinely ambiguous, not what is derivable. *[CK: hm, not convinced]*
- **Vivarium's full pipeline machinery as user-facing vocabulary.** Value pipelines, combiners, post-processors, lifecycle phases, population views, lookup tables, artifacts, and the resource system are, per its own review, "a lot of machinery before the first line of epidemiology". Take the semantics; expose one line.
- **Arbitrary state access without declaration.** Fast, unanalysable, and — as LASER's review notes — declared read/write sets "cost nothing at run time; they are compile-time information". *[CK: keep, as an escape hatch]*

## Open questions

- **Should `ss.Effect` be time-varying?** `ss.Effect(vaccinated, sir.rel_sus, multiply=ss.waning(0.1, halflife=ss.days(180)))` collapses effects and [EMOD](../approaches/emod.md)'s `Waning_Config` into one thing, which is appealing and may be doing too much. *[CK: as noted above, I think this barely scratches the surface of what "ss.Effect" would need to be doing.]*
- **Where do effects between coordinates of one dimension go?** Cross-immunity between strains is an effect from `strain=alpha` on `strain=delta`'s susceptibility, which is a shape this mechanism does not obviously have. See [stratification.md](stratification.md).
- **Does the effect graph need cycle-breaking, or should cycles simply be errors?** HIV raising syphilis susceptibility while syphilis raises HIV transmissibility is a real, non-pathological cycle over one timestep. Probably: cycles are fine across steps, errors within one. *[CK: disagree, cycles are fine. It shouldn't matter (very much).]*
