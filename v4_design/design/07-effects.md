# Interactions between modules: `ss.modify`, conditions, and plain code

How one module changes a quantity that belongs to another, how conditions ("who" and "when") are written, and what order things run in.

Justified by [best-practices/composition-and-effects.md](../best-practices/composition-and-effects.md).

There are two ways to write an interaction, and both are first-class:

- **`ss.modify`**, for the common case: while a condition holds, scale (or shift, or set) a named quantity.
- **Ordinary Python in a module's `step()`**, for everything else. Modules may read and write each other's state directly.

## `ss.modify`

```python
ss.modify(target, multiply=None, add=None, set=None,     # exactly one of these three
          where=None, start=None, stop=None, when=None, min_duration=None, name=None)
```

```python
ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)
```

Read as: *for people with active syphilis, multiply HIV transmissibility by 2.*

| Argument | Meaning |
|---|---|
| `target` | the quantity to change |
| `multiply=`, `add=`, `set=` | what to do to it; exactly one |
| `where=` | who it applies to: a [condition](#conditions). Omitted means everyone |
| `start=`, `stop=` | the period it applies in (dates, years, or durations since the start of the sim). Omitted means always |
| `when=` | a function of the sim returning `True` or `False`, for anything `start`/`stop` cannot say; `min_duration=` keeps it on for at least that long once triggered ([08-interventions.md](08-interventions.md#triggers)) |
| `name=` | used in `describe()` and `what_modifies()`; defaults to the variable or keyword name |

A modifier is passed to the sim like any other module, in `connectors=` (interactions that are part of the biology) or `interventions=` (policy). The two lists differ only in where the condition is evaluated in the loop ([below](#execution-order)).

```python
coinfection = [
    ss.modify(hiv.rel_trans,      multiply=2.0, where=syphilis.active),
    ss.modify(hiv.rel_sus,        multiply=2.7, where=syphilis.active),
    ss.modify(syphilis.rel_sus,   multiply=1.5, where=hiv.infected & ~hiv.on_art),
]
sim = ss.Sim(diseases=[hiv, syphilis], networks=ss.MFNet(), connectors=coinfection)
```

### Targets

The target is **any named numeric quantity**: any parameter of any module, and any float state. There is no fixed list.

```python
hiv.rel_trans                 # a float state (one value per agent or per group)
sir.pars.beta                 # a parameter
sir.recovery.dur              # a slot of a named transition
school.pars.n_contacts        # a network's or mixing pool's parameter, scalar or matrix
'sir.pars.beta'               # the same parameter as a string path
```

A target is a reference to the quantity, not a copy of its current value. State arrays and parameters held in `pars` are objects, so `ss.modify(sir.pars.beta, ...)` refers to the parameter; the string path is the same reference spelled as text, for when the object is not in scope. If the target cannot be resolved to a named numeric quantity when the sim is built (a plain number, a boolean state, a name that does not exist), `E504 UnmodifiableTarget` is raised with the closest matching names. Boolean flips and anything else that is not "change a number" belong in `step()` code.

### How modifiers combine

When several modifiers are active on one target, the effective value is

```
effective = (set value if any, else base) × Π multipliers + Σ addends
```

that is, **set, then multiply, then add**. The rule does not depend on the order in which the modifiers were declared or run.

- `multiply=` is the usual choice. Multipliers commute, and two 40% reductions giving a 64% reduction is the standard assumption.
- `add=` exists because additive combination is a legitimate modeling choice (`epidemics` uses it deliberately). Mixing `multiply=` and `add=` on one target is allowed; the formula above says what happens.
- `set=` replaces the base. If two `set=` modifiers are active for the same people at once, the one later in the loop order is used and `W502 OverlappingSet` names both; use `multiply=` or make one conditional on the other to avoid it.
- A result outside the quantity's natural range (a probability above 1, a negative rate) is clipped to the range, and `what_modifies()` reports that it happened.

Stratification adjustments ([05-dimensions.md](05-dimensions.md)) and product efficacies ([08-interventions.md](08-interventions.md)) are multipliers of this kind and combine with `ss.modify` by the same rule.

### Base and effective values

Every modifiable quantity has a **base** value and an **effective** value.

- The base is what the owning module declared or last assigned. **A modifier never overwrites it.**
- The effective value is the base with all currently active modifiers applied. It is what the library uses, and what any module gets when it reads the quantity.
- When a modifier stops applying (the condition becomes false, `stop=` passes), the effective value returns to the base. Nothing has to be saved and restored; the `orig_rel_sus` copies seen in Starsim interventions are not needed.

For code that touches a modifiable array directly:

```python
hiv.rel_trans[uids]            # reads the effective values
hiv.rel_trans.base[uids]       # reads the base values
hiv.rel_trans[uids] = 0.5      # assigns the base
hiv.rel_trans[uids] *= 0.5     # scales the base; active modifiers are not folded in
```

**Implementation.** The effective array is cached. It MUST be recomputed only when something that feeds it changes: a base value is assigned, a modifier's active set (the people or groups its condition selects) changes, or a time-varying multiplier takes a new value. A modifier outside its `start`/`stop` period is not evaluated at all, and a modifier with a constant multiplier and an unchanged active set costs one comparison per step. A quantity with no modifiers has no cache and behaves exactly as in Starsim.

### Time-varying multipliers

`multiply=`, `add=`, and `set=` accept everything a parameter accepts ([04-parameters.md](04-parameters.md)): a number, a typed quantity, `ss.timeseries(...)`, `ss.seasonal(...)`, `ss.by(age=[...])`, or a function of the sim returning a scalar or one value per agent or group.

```python
ss.modify(sir.pars.beta, multiply=ss.seasonal(mean=1.0, amplitude=0.3, peak='January'))
ss.modify(school.pars.n_contacts, multiply=ss.timeseries({'2020-03-15': 0.2, '2020-06-01': 0.6, '2020-09-01': 1.0}))
ss.modify(sir.rel_sus, multiply=lambda sim: 0.5 if sim.results.sir.prevalence[sim.ti - 1] > 0.05 else 1.0)
```

Protection that wanes with time since a dose is a property of a product (`waning=`, [08-interventions.md](08-interventions.md#products)), which is realized as a modifier of this kind.

`ss.change` ([08-interventions.md](08-interventions.md#sschange)) is a shorthand for a modifier whose multiplier is a step function of time.

### `sim.what_modifies()`

```python
sim.what_modifies(hiv.rel_trans)
```

```text
hiv.rel_trans   base: 1.0 per agent
  modifiers
    syph_trans   multiply=2.0   where=syphilis.active     active for 1,204 agents   [stisim.py:14]
    art          multiply=0.04  where=hiv.on_art          active for 8,811 agents   [stisim.py:31]
  direct writes observed
    AcuteHIV.step()    assigned base for 310 agents (last at 2024-06-01)            [hiv.py:212]
```

It MUST list every modifier on the target, with its operation, condition, period, and where it was declared, and every module that was observed assigning the base value during the run. Writes are observed, not declared: the arrays record which module assigned them. This costs little and is on by default; `ss.Sim(..., track_writes=False)` turns it off. Before the sim has run, the second section is empty.

Scaling a shared quantity through `ss.modify` (or `*=`) is preferred to overwriting it, so that two modules do not clobber each other. This is guidance, and `what_modifies()` is how to check; nothing is enforced.

## Conditions

A condition says who something applies to. It is used for `where=` here and `eligible=` in [08-interventions.md](08-interventions.md). There are no global condition names (`ss.age`, `ss.sex`, `ss.time` and the like do not exist): conditions are written against the real attributes of the people and modules in the sim.

### A function of the sim

Always available, and works with any attribute, including custom ones:

```python
eligible = lambda sim: (sim.people.age > 65) & ~sim.diseases.sir.vaccinated
```

The function is called when the condition is needed and returns a boolean array (or UIDs). This is Starsim's existing form. A function may also take `(sim, uids)`; that form is agent-only.

### The short form: lazy conditions

With the objects in scope, the same condition can be written without the lambda:

```python
ppl = ss.People(n_agents=10_000)
sir = ss.Disease(...)

eligible = (ppl.age > 65) & ~sir.vaccinated
```

This works because a state array that has not been initialized builds an expression instead of computing a value. The exact rule:

- **An array is uninitialized** until the sim it belongs to has been initialized (`sim.init()`, or the start of `sim.run()`). `ppl.age`, `ppl.female`, and every state of a module (`sir.vaccinated`, `hiv.cd4`, `syphilis.active`) are such arrays; so are properties built from them.
- **On an uninitialized array, these operations are lazy**: comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`), logical operators (`&`, `|`, `~`), arithmetic (`+`, `-`, `*`, `/`), and `.isin([...])`. The result is a lazy condition: an unevaluated expression that remembers which arrays it refers to, by module and name. Combining a lazy condition with a number or with another lazy condition is lazy.
- **A lazy condition is evaluated each time it is needed during a run**, against the sim it is running in, using the current values at that point in the loop. It refers to arrays by module and name, so the same condition works in every copy of a sim made by `ss.parallel` or calibration.
- **On an initialized array, the same operations evaluate immediately**, as in Starsim today. Inside `step()`, `self.sim.people.age > 65` is an ordinary boolean array.
- Using a lazy condition where Python needs a single truth value (`if cond:`, `and`, `or`, `not`) is an error that says to use `&`, `|`, `~` instead. Anything the operators above cannot express is written as a function of the sim.
- A boolean state on its own is a condition (`where=syphilis.active`), and a lazy condition prints as the expression it was built from, which is what `describe()` shows.

No new vocabulary is involved: the objects and attribute names are the ones the user already has.

### On agents and on groups

A condition is evaluated on whatever the run method provides.

- **Agent-based run.** Each array is one value per agent, and the condition is a boolean per agent.
- **Compartmental run.** Each array is one value per group (one row per stratum), and the condition is a boolean per group. This works when everything the condition uses is something the model is stratified by, or a state: `ppl.age > 65` works when age is a dimension with a bin edge at 65, and `sir.vaccinated` works when `vaccinated` is a state or a dimension.

The rule is **try it and report**. Nothing is analyzed in advance. If a condition or function in a compartmental run asks for something that exists only per agent (`uids`, an attribute the model is not stratified by, an age threshold that cuts through a bin), the run stops with a message that names what was asked for and what would fix it:

```text
E804 CapabilityError: condition 'eligible' uses people.age > 70, but the model's age
     bins are [0, 15, 65] and 70 falls inside the 65+ bin.
     Fix: add 70 as a bin edge, sir.stratify(age=[0, 15, 65, 70]), or run with method='abm'.
```

A condition on another module's state (`where=syphilis.active` on an HIV quantity) needs the two modules' states tracked jointly. In an agent-based run that is free. In a compartmental run the library crosses the two modules' states, the model grows to the product, and `describe()` shows the new size.

## Plain code in `step()`

Many real interactions are not "one condition, one quantity, one multiplier". They have several sources and targets, mix scaling with boolean changes, and schedule things for later. These are written as ordinary Python in a module's `step()`, using the tools Starsim already has: boolean state arrays, `.uids`, and `dist.filter(uids)` to select people with a probability. This is a normal way to write a model, not a fallback.

The example: HIV-positive women who are not yet diagnosed are tested when they attend antenatal care (ANC), with a probability that depends on age and sex; a positive test marks them diagnosed.

```python
def p_test(sim):                                   # per-agent probability of being tested at a visit
    ppl = sim.people
    return np.where(ppl.age < 25, 0.9, 0.6) * ppl.female

def diagnose(sim, uids):                           # what a positive test does
    hiv = sim.diseases.hiv
    hiv.diagnosed[uids]    = True                  # write the HIV module's state directly
    hiv.ti_diagnosed[uids] = sim.ti
```

**As an event-triggered delivery** ([08-interventions.md](08-interventions.md#event-triggered-delivery)):

```python
hiv_test = ss.Test(sensitivity=0.99, specificity=0.99)

anc_testing = ss.screen(hiv_test,
    on          = anc.visit,                        # delivered to people at an ANC visit this step
    eligible    = hiv.infected & ~hiv.diagnosed,    # a lazy condition
    coverage    = p_test,
    on_positive = diagnose,                         # a delivery, or a function of (sim, uids)
)
```

**As a module with `step()`**:

```python
class ANCTesting(ss.Intervention):
    def __init__(self, sensitivity=0.99, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            tested   = ss.bernoulli(p=lambda self, sim, uids: p_test(sim)[uids]),
            detected = ss.bernoulli(p=sensitivity),
        )

    def step(self):
        hiv = self.sim.diseases.hiv
        anc = self.sim.demographics.anc
        at_visit = (anc.visit & hiv.infected & ~hiv.diagnosed).uids   # HIV-positive, undiagnosed, at ANC today
        tested   = self.pars.tested.filter(at_visit)                  # thin by the age/sex probability
        positive = self.pars.detected.filter(tested)                  # thin by test sensitivity
        diagnose(self.sim, positive)
```

The two are the same model. The first is shorter and also runs in a compartmental method if age, sex, ANC attendance, and diagnosis are all stratifications or states. The second can do anything: decide who is referred at the moment of infection and test them at their next visit (add a `referred` boolean state, set it in one line, test `self.referred & anc.visit`), prioritize by clinic, draw down a stock. Code like this is agent-based unless it happens to work on groups, by the same try-it rule as conditions.

A function can be used in place of a class when no state is needed: a plain function passed as a keyword to a module is called every step ([03-states-and-transitions.md](03-states-and-transitions.md)).

### No required declarations

A module does not declare what it reads or writes. Declarations of that kind get forgotten and drift away from the code, and an out-of-date declaration is worse than none. What a module actually wrote is recorded and shown by `what_modifies()`.

`ss.Needs` and `requires=` remain as optional metadata that says what input data a module expects, for generating data-entry forms ([04-parameters.md](04-parameters.md)). They are never enforced as access control: a module can read anything in the sim.

## Execution order

Order is Starsim's ordered loop ([10-execution.md](10-execution.md)). Each step, modules run in a fixed sequence by type (demographics; disease state updates; connectors; networks; interventions; transmission; deaths; results; analyzers), and within a type in the order they were given. The loop is a data object (`sim.loop`) that can be printed, plotted, and inserted into.

What this means for interactions:

- **Each module sees the current values when it runs.** A chain can therefore complete within one step: infection, testing, and treatment all happen in the same step if the modules are ordered that way.
- **A modifier follows the state it depends on.** A `where=` condition on a state array is kept current: when the state changes, the effective value is recomputed at the next read ([10-execution.md](10-execution.md#where-ssmodify-is-evaluated)). A `when=` or `where=` given as a function is called once per step, at the modifier's place in the loop: among the connectors if it was passed in `connectors=`, among the interventions if in `interventions=`.
- **Mutual dependence is fine.** HIV raises syphilis susceptibility while syphilis raises HIV transmissibility; each reads the other's current value, and the order decides which goes first. There are no cycle errors. Where the order matters, the convention is to run last the module whose accuracy matters most.
- **Any effect of ordering is of order `dt`.** `sim.check_dt()` reruns with a smaller step and shows how much the results change.

## Rejected

- **A closed list of modifiable quantities.** Too restrictive for real models; any named numeric quantity can be a target.
- **Required `writes`/`requires` declarations with enforced read-only views.** They depend on people remembering to keep them current. Observed writes give the same information without that.
- **Global condition names and an expression type the user has to learn.** Conditions use the sim's own attributes, lazily when written before the sim exists.
- **Analyzing conditions in advance to decide what can run where.** Try the condition on what the method provides and report what it asked for.
- **Queued updates, named phases, and ordering derived from declarations.** The ordered loop is simpler, allows within-step chains, and handles mutual dependence.
- **Last-write-wins as the only way to share a quantity.** `ss.modify` keeps the base and combines by a stated rule; direct writes are still allowed and are visible.
- **Additive combination as the default.** Available as `add=`.
- **A separate object for protective measures** (a "tool" carrying fixed multipliers). A product's efficacy is a modifier on the people who hold it.
