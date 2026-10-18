# Composition and effects

How one module changes a quantity owned by another, and what runs when.

## Recommendation

1. **The simple case is one line.** `ss.modify(target, multiply=, where=)` says "while this condition holds, scale that quantity". It replaces the hand-written connector for the common case.
2. **Anything more complex is ordinary Python in a module's `step()`.** This is a first-class way to write an interaction, not a fallback. Real interactions often have several sources, several targets, probabilities, and scheduling, and a short `step()` is the clearest way to say that.
3. **A modifier never overwrites the base value.** A modified quantity has a base value and an effective value (base with the active modifiers applied), and the modifiers act where the library uses the quantity. Reading it from custom code gives the base, and assigning to it sets the base; `.effective` gives the value with the modifiers applied (agents only). When a modifier ends, the original value comes back by itself.
4. **Say how modifiers combine, because it is a modeling decision.** Multiplicative by default; `add=` and `set=` are available, applied in a fixed order (`set`, then `multiply`, then `add`).
5. **Do not require modules to declare what they read and write.** Declarations get forgotten and drift from the code. `sim.what_modifies(x)` always lists the modifiers declared on a quantity; code that writes to it directly is not a modifier, and is not tracked.
6. **Keep Starsim's ordered loop.** Modules run in a defined, printable order each step. Within-step chains (infection, then testing, then treatment) work because of that order, and two modules that depend on each other are fine.
7. **Keep one module abstraction.** `ss.Module` for diseases, networks, demographics, interventions, connectors, and analyzers is right.

## Why

### The Starsim connector problem

From the [Starsim review](../approaches/starsim.md), a limitation:

> **Connectors are unconstrained.** Cross-module interaction is "arbitrary Python that mutates another module's state". This is why multi-disease works at all, and it is also the largest hole in the "no hidden behavior" goal: `rel_trans[coinf] = 2.0` in one module silently overwrites whatever another module wrote.

The worked example in that review:

```python
class HIVSyphConnector(ss.Connector):
    def step(self):
        coinf = (self.sim.diseases.syphilis.active).uids
        self.sim.diseases.hiv.rel_trans[coinf] = 2.0
```

There are three separate problems here, and they have different fixes:

- **Clobbering.** Add a second connector that also writes `hiv.rel_trans` (a risk factor, a vaccine, a seasonal term) and the result depends on which ran last. [LASER](../approaches/laser.md) has the same hazard: "All components write columns of one `LaserFrame`."
- **Restoring.** The connector above never sets `rel_trans` back when syphilis clears. Code that does restore it usually copies the array first (`orig_rel_sus = rel_sus.copy()`), changes it, and copies it back later, which breaks as soon as anything else changed the array in between.
- **Visibility.** Nothing tells you which modules touch `hiv.rel_trans`.

The freedom to write arbitrary Python is not one of the problems. It is the reason multi-disease models can be written at all, and it stays.

### Vivarium's value pipelines

From the [Vivarium review](../approaches/vivarium.md):

```python
builder.value.register_rate_producer("diarrhea.incidence_rate", source=self._base_rate)
builder.value.register_value_modifier("diarrhea.incidence_rate", modifier=self._apply_risk_effect)
```

> A component that produces a rate does not know who modifies it. A component that modifies a rate does not know who produces it — only its name. **Multiple modifiers compose by a declared rule**, not by whoever ran last.

This is the mechanism that fixes clobbering and restoring together: the producer keeps the base value, modifiers are applied on top, and removing a modifier leaves the base intact. [epiworld](../approaches/epiworld.md) has a narrower version: a **tool** is a bundle of multiplicative effects on susceptibility, transmissibility, recovery, and death that an agent carries. [EMOD](../approaches/emod.md)'s `Waning_Config` is the same idea for a single product's efficacy over time.

What we do not take from Vivarium is the rest of the machinery (see [Rejected](#rejected)), or epiworld's fixed list of four targets. A fixed list is too restrictive: real models modify contact rates, care-seeking probabilities, progression rates, and parameters nobody anticipated.

### The combination rule has to be stated

[`epidemics`](../approaches/epidemics.md) composes interventions **additively**, `C × (1 − (X + Y))` rather than `C × (1 − X)(1 − Y)`, because additive was judged easier for users to understand, and it wrote the decision down in a design-principles vignette. Its review: "Multiplicative composition is the more defensible modeling assumption and the more common one; the package chose the other and documented the choice and its rationale."

That is finding 9 of the [landscape review](../approaches/README.md#cross-cutting-findings): the combination rule is a modeling decision. Whatever the default is, it should be printed and changeable. Multiplicative is the default here because it is the usual modeling assumption and because it commutes, so the result does not depend on the order of the modifiers; `set=` is the one kind that does not, and two at once are noted.

### Custom code is the rest

Most real interactions are not "one condition, one target, one multiplier". A typical one: *find people who are HIV-positive and undiagnosed, select some of them with a probability that depends on age and sex, and test them at their next antenatal care visit; start those who test positive on treatment.* There may be five source states and six targets, some multipliers and some boolean flips. Delivery keywords cover the cases that are deliveries; the rest is a short `step()` written with Starsim's boolean arrays and `dist.filter(uids)`, as today. Two pieces of guidance, neither enforced: write booleans and events directly (`hiv.diagnosed[positive] = True` has no combination problem), and to scale a number someone else owns, declare a modifier once and have `step()` set and clear the flag it depends on.

### Declared reads and writes drift

A `writes = [...]` list is easy to forget, and worse, easy to leave unchanged when someone edits `step()` later, at which point it is wrong and trusted. Writes can be observed instead: the arrays know when they are assigned to, so with tracking on they record which module wrote to them during a run. That is [Vivarium](../approaches/vivarium.md)'s "you can ask what modifies this and get an answer" without its `PopulationView` access control, and it cannot drift from the code because it comes from the code running.

### Execution order

Four positions in the review:

| Framework | Treatment |
|---|---|
| [`individual`](../approaches/individual.md) | **Queued.** Processes queue updates; the loop applies them after every process has run, so every process sees start-of-step state |
| [Vivarium](../approaches/vivarium.md) | **Declared.** Named lifecycle phases (`time_step_prepare` → `time_step` → `time_step_cleanup` → `collect_metrics`) with per-component priorities |
| [Starsim](../approaches/starsim.md) | **Ordered and inspectable.** `ss.Loop` is a data object: printable, `to_df()`-able, plottable, insertable-into |
| [Mesa](../approaches/notes.md#mesa) | **A user-selected modeling decision**, with the options named |
| Everyone else | Whatever order the code happens to run |

The reasons for an ordered loop over queued updates:

- **Within-step chains work.** With infection, testing, and treatment ordered that way, someone can be infected, tested, and treated in one step. With queued updates every link in the chain waits a step, which puts an artificial delay of two timesteps on the fastest possible treatment.
- **Mutual dependence is fine.** HIV affects syphilis and syphilis affects HIV. Each module reads the other's current value and the order breaks the tie. Nothing needs to detect or reject the cycle.
- **There is one ordering, not two.** A queue of simultaneous updates still needs a rule for which update wins when two conflict, and that rule is a second ordering on top of the module order.
- **Named phases are positions in the list.** Vivarium's phases are a way of naming where in the order something runs. `sim.loop` already shows the positions and allows inserting at any of them.

## The specification

[07-effects.md](../design/07-effects.md) specifies `ss.modify` (targets, how modifiers combine, base and effective values, `sim.what_modifies()`), conditions, and plain code in `step()`. The order of calls in a step, competing hazards, and deferred removals are in [10-execution.md](../design/10-execution.md), deliveries triggered by another module's events in [08-interventions.md](../design/08-interventions.md), and the one module abstraction in [01-objects.md](../design/01-objects.md).

## Trade-offs

- **Direct writes can still clobber each other.** Two modules that both assign `hiv.rel_trans[uids] = ...` behave as they do in Starsim today. The mitigation is that the better option (`ss.modify`) is one line; nothing prevents it.
- **Observed writes are only known after a run with tracking turned on**, and only for code paths that ran. A module that writes to a quantity only in 2035 does not show up in a run that stops in 2030.
- **Base and effective values are an implementation cost.** Every modifiable array needs a base, a cached effective value, and invalidation when a modifier's active set changes. Unmodified quantities must cost the same as they do now, which should be benchmarked.
- **Results depend on module order**, by an amount that shrinks with the timestep. This is accepted in exchange for within-step chains and for a loop that is easy to explain. `sim.check_dt()` measures it.
- **Custom code reads the base, not what the library uses.** Reading and writing both act on the base, so a read-modify-write (`x[uids] *= 0.5`) does what it looks like and never counts a modifier twice. The cost is that code which wants the modified value has to ask for it (`.effective`), and that is available only as agents.
- **Two `set=` modifiers active at once: the last one wins.** That is the one place where modifiers depend on order, which is why it is noted (`ss.OverlappingSet`).
- **`set` then `multiply` then `add` is a convention.** Someone will want a different order. A function passed to `multiply=`, or a `step()`, covers it.

## Rejected

- **A fixed list of modifiable quantities** (epiworld's four tool effects). It makes analysis easier and it is too restrictive to be useful for real models.
- **Required `reads`/`writes` declarations with enforced views** (Vivarium's `PopulationView`). People forget them, and they drift from the code. Observed writes give the same answer without the upkeep.
- **Queued updates for everything** (`individual`). Removes order dependence and removes within-step chains with it. Deferred removals and joint resolution of competing hazards are the parts worth having.
- **Ordering derived from declared dependencies, with cycles as errors.** Mutual dependence between diseases is normal, and a convention settles the order.
- **Named lifecycle phases as user vocabulary** (Vivarium). The loop is already a list that can be printed and inserted into.
- **Additive composition as the global default** (`epidemics`). Documented well there, but it clamps at 100% and surprises people. Multiplicative default, `add=` available.
- **Vivarium's full pipeline machinery as user-facing vocabulary.** Value pipelines, combiners, post-processors, population views, lookup tables, artifacts, and the resource system are, per its own review, "a lot of machinery before the first line of epidemiology". Take the base-plus-modifiers idea and expose one function.
- **Forbidding direct access to another module's state.** It is how complex interactions get written, and it stays available.

## Open questions

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
