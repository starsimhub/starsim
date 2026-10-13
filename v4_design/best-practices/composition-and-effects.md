# Composition and effects

How one module changes a quantity owned by another, and what runs when.

## Recommendation

1. **The simple case is one line.** `ss.modify(target, multiply=, where=)` says "while this condition holds, scale that quantity". It replaces the hand-written connector for the common case.
2. **Anything more complex is ordinary Python in a module's `step()`.** This is a first-class way to write an interaction, not a fallback. Real interactions often have several sources, several targets, probabilities, and scheduling, and a short `step()` is the clearest way to say that.
3. **A modifier never overwrites the base value.** A modified quantity has a base value and an effective value (base with the active modifiers applied). Reading it gives the effective value; assigning to it sets the base. When a modifier ends, the original value comes back by itself.
4. **Say how modifiers combine, because it is a modeling decision.** Multiplicative by default; `add=` and `set=` are available.
5. **Do not require modules to declare what they read and write.** Declarations get forgotten and drift from the code. `sim.what_modifies(x)` always lists the modifiers declared on a quantity, and a sim run with `track_writes=True` also records which modules wrote to it directly. Tracking is off by default.
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

That is finding 9 of the [landscape review](../approaches/README.md#cross-cutting-findings): the combination rule is a modeling decision. Whatever the default is, it should be printed and changeable.

## The proposal

### One line for the simple case

**Starsim today:**

```python
class HIVSyphConnector(ss.Connector):
    def step(self):
        coinf = self.sim.diseases.syphilis.active.uids
        self.sim.diseases.hiv.rel_trans[coinf] = 2.0
```

**Lingua franca:**

```python
ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)
```

Read as: *multiply HIV transmissibility by 2 where syphilis is active.* The first argument is the quantity to change; `multiply=` (or `add=`, `set=`) is the change; `where=` is the condition, written against real attributes in the same way as [eligibility](interventions.md#eligibility); `start=` and `stop=` limit it in time, and `when=` takes a function of the sim for a condition that dates cannot express ([interventions.md](interventions.md#eligibility)). Modifiers are passed to the sim like any connector:

```python
sim = ss.Sim(
    diseases   = [hiv, syphilis],
    networks   = ss.MFNet(),
    connectors = [
        ss.modify(hiv.rel_trans,    multiply=2.0, where=syphilis.active),
        ss.modify(syphilis.rel_sus, multiply=2.5, where=hiv.infected),
    ],
)
```

The two modifiers above depend on each other (HIV affects syphilis and syphilis affects HIV). That is a real feature of the epidemiology and it is not a problem: each reads the other's current state, and the [loop order](#execution-order) says which one is read first.

What the one-line form gives you:

- Two modifiers on `hiv.rel_trans` combine by a stated rule instead of by which ran last.
- When syphilis clears, `hiv.rel_trans` goes back to its base value with no extra code.
- `sim.what_modifies(hiv.rel_trans)` lists it.
- It is data, so it can be saved in the model spec, compared, and drawn on a diagram.
- It works in a compartmental run if the condition is a state or a stratification (the modifier is applied to the matching groups).

### What can be modified

Any named numeric quantity: any parameter or float state of any module. There is no fixed target list.

```python
ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)          # a per-agent state
ss.modify(sir.pars.beta, multiply=ss.seasonal(amplitude=0.3, peak='January'))  # a parameter, everywhere
ss.modify(school.pars.n_contacts, multiply=0.2, start='2020-03-15', stop='2020-06-01')
ss.modify(hiv.progression.rate, multiply=0.1, where=hiv.on_art)       # a named transition's rate
```

`multiply=`, `add=`, and `set=` accept everything a parameter accepts (a number, `ss.seasonal(...)`, `ss.timeseries(...)`, `ss.by(age=...)`, a function), so a modifier can vary over time or between people. This covers [epiworld](../approaches/epiworld.md)'s tools, [EpiHiper](../approaches/epihiper.md)'s state properties, and [Epydemix](../approaches/epydemix.md)'s contact reductions with one function.

### Base and effective values

This is the answer to the copy-and-restore problem. **Starsim today**, for an intervention that halves susceptibility for a while when `rel_sus` already differs between agents:

```python
def start_protection(self, uids):
    self.orig_rel_sus[uids] = sir.rel_sus[uids]    # remember what it was
    sir.rel_sus[uids] *= 0.5

def end_protection(self, uids):
    sir.rel_sus[uids] = self.orig_rel_sus[uids]    # wrong if anything else changed it meanwhile
```

**Lingua franca:**

```python
ss.modify(sir.rel_sus, multiply=0.5, where=campaign.protected)
```

`sir.rel_sus` keeps its base values (heterogeneous or not). What the transmission step reads is the effective value: base, with every currently active modifier applied. Turning `campaign.protected` off for an agent removes the factor of 0.5 and nothing else, whatever other modifiers have come and gone in the meantime.

This does not mean multiplying five arrays together on every timestep. The effective array is cached and recomputed only when a modifier's active set or value changes. A modifier that is switched off, or one whose condition did not change this step, costs nothing. A quantity with no modifiers is just its base array, as now.

### Reading and writing a modified quantity

Custom code reads and writes these arrays directly, as it does in Starsim today, so what a read and a write mean once modifiers exist has to be stated. Three rules:

- **Reading gives the effective value**: the base, times (or plus) every modifier active for that agent right now. This is what transmission uses, so it is what any other code should see.
- **Assigning sets the base.** The modifiers still apply on top of the new base.
- **`.base` reads the base**, without modifiers.

Take one agent with syphilis, and the modifier `ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)`:

```python
hiv.rel_trans[uid]            # 2.0   effective: base 1.0 × 2.0 for syphilis
hiv.rel_trans.base[uid]       # 1.0

hiv.rel_trans[uid] = 0.5      # a direct write: the base is now 0.5
hiv.rel_trans[uid]            # 1.0   effective: base 0.5 × 2.0; the syphilis modifier still applies
hiv.rel_trans.base[uid]       # 0.5

# ... later, syphilis clears ...
hiv.rel_trans[uid]            # 0.5   the base, with nothing on top
```

A direct write therefore never removes or cancels a modifier, and a modifier never destroys what was written. Code that ran before modifiers existed keeps working: with no modifiers on a quantity, base and effective are the same array.

One line to avoid is `hiv.rel_trans[uids] *= 0.5` on a quantity that has active modifiers. Python reads the effective value, halves it, and assigns the result as the new base, so the modifiers end up counted twice. Write `hiv.rel_trans.base[uids] *= 0.5`, or better, declare a modifier.

### Combining modifiers

```python
ss.modify(x, multiply=0.6, where=a)     # default: two of these give 0.6 × 0.6 = 0.36
ss.modify(x, add=-0.4, where=a)         # epidemics-style additive
ss.modify(x, set=0.0, where=a)          # replace the base value
```

When several modifiers apply to the same quantity for the same agent, they are applied in a fixed order: `set`, then `multiply`, then `add`. Multiplicative is the default because it is the usual modeling assumption and because it commutes, so the result does not depend on the order of the modifiers.

`set=` is the one kind that does not commute. If two `set=` modifiers are active for the same agent at the same time, only one value can stand: the modifier that comes later in the [loop order](#execution-order) wins, and a warning names both. Everything else (two multipliers, a multiplier and an addition) combines without a warning.

`sim.what_modifies(x)` prints the modifiers and the rule:

```text
>>> sim.what_modifies(hiv.rel_trans)
hiv.rel_trans (base: 1.0 per agent)
  modify  multiply=2.0   where=syphilis.active     [stisim.py:14]
  modify  multiply=0.04  where=hiv.on_art          [stisim.py:31]
  written directly by: RiskBehavior.step()         [risk.py:52]
  effective = base × 2.0 × 0.04 where both apply
```

The `modify` lines are always there, because modifiers are declared. The `written directly by` line appears only if the sim was run with `track_writes=True` ([below](#seeing-who-writes-what)).

### Complex interactions are a `step()`

Most real interactions are not "one condition, one target, one multiplier". A typical one: *find people who are HIV-positive and undiagnosed, select some of them with a probability that depends on age and sex, and test them at their next antenatal care (ANC) visit; start those who test positive on treatment.* There may be five source states and six targets, some of them multipliers and some of them boolean flips. Nothing about that fits a one-line modifier, and it should not have to.

Where the interaction is a delivery triggered by an event in another module, the [delivery keywords](interventions.md#delivery) cover it. `on=` says what triggers it, `eligible=` says who, and `coverage=` takes a function when the probability varies between people:

```python
def p_offer(sim, uids):
    ppl = sim.people
    return np.where(ppl.female[uids] & (ppl.age[uids] < 25), 0.9, 0.6)

anc_testing = ss.screen(
    hiv_test,
    on          = anc.visit,                      # at each ANC visit
    eligible    = hiv.infected & ~hiv.diagnosed,
    coverage    = p_offer,
    on_positive = ss.treat(art),
)
```

Where it is not, or where you simply prefer to see the logic, write the module. This is the same thing in plain code:

```python
class ANCTesting(ss.Intervention):
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            p_offer     = ss.bernoulli(p=lambda mod, sim, uids: p_offer(sim, uids)),   # Starsim's callable pars take (module, sim, uids)
            sensitivity = ss.bernoulli(p=0.98),
        )
        self.update_pars(**kwargs)
        self.define_states(ss.BoolState('due_test'))

    def step(self):
        hiv = self.sim.diseases.hiv
        anc = self.sim.demographics.anc

        # Find, filter, and flag for a test at the next visit
        candidates = (hiv.infected & ~hiv.diagnosed & ~self.due_test).uids
        self.due_test[self.pars.p_offer.filter(candidates)] = True

        # Test those who are flagged and have a visit on this step
        tested   = (self.due_test & anc.visit).uids
        positive = self.pars.sensitivity.filter(tested)
        hiv.diagnosed[positive] = True
        hiv.on_art[positive]    = True
        self.due_test[tested] = False
```

This is Starsim as it is today: boolean state arrays combined with `&`, `|`, and `~`; `dist.filter(uids)` to select from a set with a probability (each agent has its own random number stream, so [common random numbers](stochasticity-and-reproducibility.md) hold); and direct reads and writes of other modules' state. The lingua franca does not add a layer over this. Its job here is to make sure the things `step()` needs are easy to reach and that what the code did can be seen afterwards.

Two pieces of guidance for custom code, neither of them enforced:

- **For booleans and events, write directly.** `hiv.diagnosed[positive] = True` has no combination problem.
- **For scaling a number someone else owns, use a modifier and flip a flag.** Declare `ss.modify(hiv.rel_trans, multiply=0.04, where=self.suppressed)` once, and have `step()` set and clear `self.suppressed`. The base value stays intact, other modules' modifiers still apply, and there is nothing to restore. Writing `hiv.rel_trans[uids] = 0.04` directly is allowed, and sets the base value ([above](#reading-and-writing-a-modified-quantity)).

A module with a custom `step()` runs in an agent-based sim. It also runs in a compartmental sim if the code happens to use only things that exist for groups (state counts, parameters, stratifications); otherwise that run stops and names what the code asked for.

### Seeing who writes what

Modules are not asked to declare what they read or write. A `writes = [...]` list is easy to forget, and worse, easy to leave unchanged when someone edits `step()` later, at which point it is wrong and trusted.

Writes can be observed instead. The state arrays already know when they are assigned to, so with tracking on they record which module wrote to them during a run:

```python
sim = ss.Sim(diseases=[hiv, syphilis], networks=ss.MFNet(), track_writes=True).run()
sim.what_modifies(hiv.rel_trans)
```

`sim.what_modifies(hiv.rel_trans)` then lists the modifiers on a quantity together with the modules seen writing it directly (the example output above). Tracking is off by default, because recording the writer on every assignment has a cost and most runs never ask; it is a debugging switch, turned on when the question is "who is changing this?". Without it, `what_modifies()` still lists every declared modifier. This is [Vivarium](../approaches/vivarium.md)'s "you can ask what modifies this and get an answer" without its `PopulationView` access control, and it cannot drift from the code because it comes from the code running.

[epymorph](../approaches/epymorph.md)'s `AttributeDef`, where a module lists the data it needs, is kept as optional metadata (`requires=`): useful for generating a data-entry form or a checklist of inputs, never enforced as a restricted view.

### Execution order

Four positions in the review:

| Framework | Treatment |
|---|---|
| [`individual`](../approaches/individual.md) | **Queued.** Processes queue updates; the loop applies them after every process has run, so every process sees start-of-step state |
| [Vivarium](../approaches/vivarium.md) | **Declared.** Named lifecycle phases (`time_step_prepare` → `time_step` → `time_step_cleanup` → `collect_metrics`) with per-component priorities |
| [Starsim](../approaches/starsim.md) | **Ordered and inspectable.** `ss.Loop` is a data object: printable, `to_df()`-able, plottable, insertable-into |
| [Mesa](../approaches/notes.md#mesa) | **A user-selected modeling decision**, with the options named |
| Everyone else | Whatever order the code happens to run |

The lingua franca keeps Starsim's. Each step runs: start of step; demographics; disease state updates and progression; connectors; networks; interventions; transmission; deaths; results; analyzers; end of step. Within a type, modules run in the order they were given. The order is a data object (`sim.loop`) that can be printed, plotted with `sim.loop.plot_step_order()`, and inserted into. `sim.describe()` shows it as a schedule grouped by how often things run ("every day: sir, random network; every year: births, deaths"), which is easier to read than the unrolled list of every call ([time-and-units.md](time-and-units.md)).

The reasons for an ordered loop over queued updates:

- **Within-step chains work.** With infection, testing, and treatment ordered that way, someone can be infected, tested, and treated in one step. With queued updates every link in the chain waits a step, which puts an artificial delay of two timesteps on the fastest possible treatment.
- **Mutual dependence is fine.** HIV affects syphilis and syphilis affects HIV. Each module reads the other's current value and the order breaks the tie. The convention is to run last the disease whose results matter most, since it sees the most up-to-date state. Nothing needs to detect or reject the cycle.
- **There is one ordering, not two.** A queue of simultaneous updates still needs a rule for which update wins when two conflict, and that rule is a second ordering on top of the module order.
- **Named phases are positions in the list.** Vivarium's phases are a way of naming where in the order something runs. `sim.loop` already shows the positions and allows inserting at any of them.

Three things keep the ordered loop from causing surprises:

1. **Competing transitions inside one module are resolved together.** Transitions out of the same state that are written as rates compete as hazards in one multinomial draw, so the order in which they were declared does not bias the result.
2. **Removals are deferred to the end of the step**, as Starsim's `people.request_death()` does. Agents flagged to die stay in the arrays until every module has stepped, so no module sees a population that changed size halfway through a step. Custom code that removes people should call `request_death()` and not delete them itself.
3. **The size of any ordering effect shrinks with the timestep** (it is O(dt)), and `sim.check_dt()` reruns the model with a halved timestep and shows how much the results move.

### One module abstraction

Starsim's `ss.Module`, with the same lifecycle for diseases, networks, demographics, interventions, connectors, and analyzers, is what makes multi-disease and non-disease health states composable, and it is why FPsim (contraception, pregnancy, birth outcomes) fits in the same framework as HIV. Keep it unchanged. The module contract gains no required fields.

## Trade-offs

- **Direct writes can still clobber each other.** Two modules that both assign `hiv.rel_trans[uids] = ...` behave as they do in Starsim today. The mitigation is that the better option is one line and, with `track_writes=True`, `what_modifies()` shows both writers; nothing prevents it.
- **Observed writes are only known after a run with tracking turned on**, and only for code paths that ran. A module that writes to a quantity only in 2035 does not show up in a run that stops in 2030.
- **Base and effective values are an implementation cost.** Every modifiable array needs a base, a cached effective value, and invalidation when a modifier's active set changes. Unmodified quantities must cost the same as they do now, which should be benchmarked.
- **Results depend on module order**, by an amount that shrinks with the timestep. This is accepted in exchange for within-step chains and for a loop that is easy to explain. `sim.check_dt()` measures it.
- **Reading the effective value and writing the base is asymmetric.** It is the rule that keeps both modifiers and existing code working, and its cost is the `*=` case above: a read-modify-write on a modified quantity does not do what it looks like.
- **Two `set=` modifiers resolve by loop order.** That is the one place where modifiers depend on order, which is why it warns.
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

- **In-place arithmetic on a modified quantity.** A plain assignment sets the base, and that much is settled. `x[uids] *= 0.5` reads the effective value and writes the base, which double-counts the active modifiers. Whether that should warn, or whether in-place operators can be made to act on the base, needs deciding against the implementation.
- **Triggers on `ss.modify`.** `when=` is kept only as a function of the sim that is asked each step and remembers nothing; `min_duration=` is gone, and a policy with memory is a `step()` or an intervention class ([interventions.md](interventions.md#eligibility)). Even in that reduced form a trigger on a one-line modifier is fragile: it reads results while the run is in progress, its answer depends on where in the loop it is asked, and a threshold crossed for a single step switches the modifier on and off. Whether `when=` should stay, or every trigger should be an intervention, is open.
- **Where do modifiers between coordinates of one stratification go?** Cross-immunity between strains is a modifier from `strain=alpha` on `strain=delta`'s susceptibility. When strains are separate diseases it is an ordinary `ss.modify`; when strain is a stratification the spelling is not settled. See [stratification.md](stratification.md).
- **How much does write tracking cost?** Recording the writing module on each assignment is cheap but not free, so it is off by default (`track_writes=False`) and a normal run does no extra work. If benchmarks show the cost is negligible, the default could flip.
- **Is `on=` enough for event-triggered logic in custom modules?** Deliveries can be triggered by another module's events. Whether a custom `step()` needs a matching helper for "at this person's next visit", or whether a boolean flag as in the example above is enough, should be decided against a few real models.
