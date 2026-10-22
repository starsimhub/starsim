# Interactions between modules: `ss.modify`, conditions, and plain code

How one module changes a quantity that belongs to another, how conditions ("who" and "when") are written, and what order things run in.

Justified by [best-practices/composition-and-effects.md](../best-practices/composition-and-effects.md).

There are two ways to write an interaction, and both are first-class:

- **`ss.modify`**, for the common case: while a condition holds, scale (or shift, or set) a named quantity.
- **Ordinary Python in a module's `step()`**, for everything else. Modules may read and write each other's state directly.

## `ss.modify`

<!-- fragment -->
```python
ss.modify(target, multiply=None, add=None, set=None,     # exactly one of these three (or efficacy=, which is multiply=1 - efficacy)
          efficacy=None, share=None, route=None,
          where=None, start=None, stop=None, when=None, hold=None, name=None)
```

<!-- fragment -->
```python
ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)
```

Read as: *for people with active syphilis, multiply HIV transmissibility by 2.*

| Argument | Meaning |
|---|---|
| `target` | the quantity to change |
| `multiply=`, `add=`, `set=` | what to do to it; exactly one |
| `efficacy=` | for protection (a condom, a needle program, a microbicide), the share of the risk it removes: the same as `multiply=1 - efficacy`; giving both raises |
| `share=` | with `multiply=` or `efficacy=`, the share of acts (or, on a route without acts, of contacts) it acts on: since hazards add, the hazard is multiplied by `1 − share·(1 − multiply)`, exactly. A number, an `ss.timeseries`, an `ss.by`, or, with `route=` and as agents only, a function of the network returning one value per edge (`lambda net: net.edges.condoms`) |
| `route=` | for a modifier of a transmission, the network or networks (by name) it applies on, e.g. condoms on the sexual network and not the needle-sharing one; a name that is not a route raises `RouteMismatch`, and a target that is not a transmission raises. `where=` is then of the people who would be infected. Under the compartmental solvers it multiplies the contacts through that route's term of the mixing, by group |
| `where=` | who it applies to: a [condition](#conditions). Omitted means everyone |
| `start=`, `stop=` | the period it applies in: from the step at `start` up to, but not including, the step at `stop`. Times are dates, durations since the start of the sim, or numbers in the sim's unit, as for `at=` of [`ss.change`](08-interventions.md#sschange). A `start` outside the sim raises `TimeOutsideSim`; a `stop` after the end never stops it. Omitted means always |
| `when=` | a function of the sim returning `True` or `False`, for anything `start`/`stop` cannot say. The modifier applies on exactly the steps where it returns `True` |
| `hold=` | once `where=` (or `when=`) selects someone, the modifier stays on for them for at least this long (a bare number is in the sim's unit). This is the one memory a modifier has; any other rule with memory is `step()` code ([08-interventions.md](08-interventions.md#triggers)) |
| `name=` | used in `describe()` and `what_modifies()`; defaults to the variable name, else to what it does (`beta_multiply`) |

Condoms and similar per-act protection are modifiers of each disease's transmission on a route, with the share of acts they are used in; the efficacy differs by pathogen, so there is one modifier for each disease:

<!-- fragment -->
```python
use = ss.timeseries({2000: 0.05, 2020: 0.3})                                       # share of acts with a condom
condoms = [ss.modify(ng.infection, efficacy=0.9, share=use, route='sexualnet'),    # gonorrhea
           ss.modify(ct.infection, efficacy=0.4, share=use, route='sexualnet')]    # chlamydia
nsp = ss.modify(hiv.infection, multiply=0.2, share=0.5, route='needles', start=2005)   # needle and syringe programs
```

A modifier is passed to the sim like any other module, in `connectors=` (interactions that are part of the biology) or `interventions=` (policy). Which list holds it does not change what it does, `when=` included: a modifier is evaluated where the engine reads the quantity it acts on in a step, whichever list it is in (the lists differ in order only for modules with a `step()` of their own). There is no separate `modifiers=` list.

<!-- fragment -->
```python
coinfection = [
    ss.modify(hiv.rel_trans,      multiply=2.0, where=syphilis.active),
    ss.modify(hiv.rel_sus,        multiply=2.7, where=syphilis.active),
    ss.modify(syphilis.rel_sus,   multiply=1.5, where=hiv.infected & ~hiv.on_art),
]
sim = ss.Sim(diseases=[hiv, syphilis], networks=ss.SexualNet(), connectors=coinfection)
```

### Targets

The target is a named numeric quantity of a declared module, given as the object or as its path:

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(infection=ss.Transition(S, I, rate=0.05 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
targets = [
    sir.pars.beta,              # a parameter
    sir.recovery.dur,           # a transition's inline value, which is a parameter too
    sir.recovery,               # a transition: scales its rate, or its probability if it is a branch chosen with p=
    'hpv.progression.C',        # one destination of a transition with several: scales its probability
    sir.rel_sus,                # each person's (or group's) susceptibility; likewise sir.rel_trans
    'sir.pars.beta',            # the same parameter as a string path
    'school.pars.n_contacts',   # the contacts of a network or mixing pool, as a path
]
```

A target is a reference to the quantity, not a copy of its current value: `ss.modify(sir.pars.beta, ...)` refers to the parameter, and the string path is the same reference spelled as text, for when the object is not in scope. A transfer ([05-dimensions.md](05-dimensions.md#moving-between-coordinates-sstransfer)) is a target as a transition is. The contacts of a network are a target only as a path, since `school.pars.n_contacts` is a plain value. It works under every solver; as agents, a multiplier below 1 makes each contact happen with that probability in a step. Of a disease written as code (an `ss.Infection`), only `rel_sus` and `rel_trans` can be targets: its parameters are plain values that its own code reads, so modifying one raises `CapabilityError`. It has no transmission to name, so a modifier on a route is of its `rel_sus` (`ss.modify(hiv.rel_sus, efficacy=0.8, share=lambda net: net.edges.condoms, route='mf')`), of the people who would be infected over it, as for a declared disease's transmission; the modifiers of a network's contacts apply to it as well.

A modifier on a transition chosen with `p=`, one of several branches out of a state, changes its probability, and the other branches share the rest. If the target cannot be resolved when the sim is built (a plain number, a boolean state, a name that does not exist), `UnmodifiableTarget` is raised, listing the quantities that can be modified. Boolean flips and anything else that is not "change a number" belong in `step()` code.

### How modifiers combine

When several modifiers are active on one target, the effective value is

```
effective = (set value if any, else base) × Π multipliers + Σ addends
```

that is, **set, then multiply, then add**. The rule does not depend on the order in which the modifiers were declared or run.

- `multiply=` is the usual choice. Multipliers commute, and two 40% reductions giving a 64% reduction is the standard assumption.
- `add=` exists because additive combination is a legitimate modeling choice (`epidemics` uses it deliberately). Mixing `multiply=` and `add=` on one target is allowed; the formula above says what happens.
- `set=` replaces the base. If two `set=` modifiers are active for the same people at once, the one later in the loop order wins (connectors before interventions, and within a list the order given), with the note `OverlappingSet`. Use `multiply=`, or make one conditional on the other, to avoid it.
- A rate, duration, or multiplier that the modifiers make negative is set to 0 where it is, and branch probabilities that they make sum to more than 1 are scaled to sum to 1, with the note `ClippedToRange` (once per quantity); `what_modifies()` says so too.
- `add=` and `set=` are for parameters only. On a per-person quantity (`sir.rel_sus`, `sir.rel_trans`) or a transition, the modifier acts on a multiplier of each person's own value, which `set=` or `add=` would replace, so they raise and only `multiply=` is accepted.

Stratification adjustments ([05-dimensions.md](05-dimensions.md)), product efficacies ([08-interventions.md](08-interventions.md)), and protection between strains (`ss.immunity`, below) are multipliers of this kind and combine with `ss.modify` by the same rule.

### Protection between strains: `ss.immunity`

Protection from several strains a person holds is combined by a rule before it multiplies, which a set of modifiers cannot do (each has one target and one condition; HPVsim's `(1 − clip(Σ))·(1 − vaccine)` is 0.35 where modifiers give 0.2). `ss.immunity` is that rule:

<!-- fragment -->
```python
ss.immunity(protected=None, rel_sus=None, rel_trans=None, dur=None, combine='max')
```

The quantity the protection lowers is the keyword, and the protection is its value: `rel_sus=` (being infected), `rel_trans=` (infecting others), and `dur=dict(In=m, Ic=m)` (time spent in states, by the state's name: every exit of the state is faster by `1/(1 − protection)`, HPVsim's severity immunity). Several keywords may be given together. A value is a matrix, `m[k][g]` the protection that holding strain `k` gives against strain `g` (rows from, columns against; 1 is complete), or one number, the same from any strain against every strain. For each strain `g` and person, the protection is the combination over the strains `k` they hold of `m[k][g]`, by `combine=`:

| `combine=` | Protection |
|---|---|
| `'max'` (the default) | the strongest |
| `'sum'` | the sum, at most 1 (HPVsim) |
| `'multiply'` | independent sources: `1 − Π(1 − p)` |
| a function | of one array of protections, with the sources on the first axis (0 for those not held), returning the protection; e.g. the strongest plus half the rest, `lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))`. It needs agents and has no gradient (`sim.gradient()` and a NUTS fit raise `CapabilityError`) |

The quantity is multiplied by `1 − protection`, and several `ss.immunity` objects, `ss.modify`, and product efficacies on the same quantity multiply, so a vaccine's `1 − efficacy` multiplies it, which is HPVsim's rule exactly. In a family of diseases (`ss.Strains`, [05-dimensions.md](05-dimensions.md#strain)), the strains are the members and `protected=` names the states (or a named sum of states) of each member whose holders are protected (`protected='nab'`); the protection is worked out from the members' states at the start of each step, before anything moves. In a strain dimension (`stratify(strain=..., immunity=ss.immunity(rel_sus=m, combine=...))`), a person holds every strain they have had, so there is no `protected=`, and the protection is on `rel_sus` only; a bare matrix is the matrix with the default rule. A protection on a time spent that the matrix can bring to 1 raises (an infinite rate). `what_modifies()` lists each protection with the modifiers of its quantity.

### Base and effective values

A per-person quantity that can be modified (`rel_sus` and `rel_trans`) has two values for each person:

- the **base** value: the quantity's own value, as the module that owns it declared it or last assigned it;
- the **effective** value: the base with every modifier that is active right now applied on top. This is the number the model uses.

A quantity with no active modifier has the same base and effective value. Three rules cover everything code can do with one:

1. **Reading gives the base value.** `sir.rel_sus[uids]` returns the quantity's own value, as code written for Starsim 3 expects when it reads and writes it.
2. **Assigning sets the base.** `sir.rel_sus[uids] = 0.8` changes the quantity's own value. Any active modifiers still apply on top of the new base.
3. **`.effective` reads the effective value.** `sir.rel_sus.effective[uids]` returns what the transmission step will use, with the active modifiers applied.

These are arrays of agents, so `.effective` exists only in an agent-based run; in a compartmental run the modifiers act on the groups' rates inside the solver, and nothing reads them from code.

A modifier never changes the base, which is why nothing has to be restored when it ends. One person, followed through a campaign that halves susceptibility (`ss.modify(sir.rel_sus, multiply=0.5, where=campaign.protected)`):

<!-- fragment -->
```python
sir.rel_sus[uid] = 0.8              # assign: the base is 0.8
sir.rel_sus.effective[uid]          # 0.8   nothing modifies it yet

# ... the person becomes protected, so the modifier is active for them
sir.rel_sus.effective[uid]          # 0.4   effective = base 0.8 × 0.5
sir.rel_sus[uid]                    # 0.8   the base is untouched

sir.rel_sus[uid] = 1.0              # assign: the base is now 1.0
sir.rel_sus.effective[uid]          # 0.5   the modifier still applies

# ... protection ends
sir.rel_sus.effective[uid]          # 1.0   back to the base, with nothing saved or restored
```

The `orig_rel_sus` copies seen in Starsim interventions, which remember a value in order to put it back later, are not needed.

An in-place operation acts on the base, like an assignment: `sir.rel_sus[uids] *= 0.5` halves the base values. It does not read the effective value and write that back, which would fold the active modifiers into the base permanently.

A quantity with no modifiers behaves exactly as in Starsim.

### Time-varying multipliers

`multiply=`, `add=`, and `set=` accept everything a parameter accepts ([04-parameters.md](04-parameters.md)): a number, a typed quantity, an `ss.par` (which can then be fitted, or set by name with `pars=`), `ss.timeseries(...)`, `ss.seasonal(...)`, `ss.by(age=[...])`, or a function of the sim returning a scalar or one value per agent or group. A value that is set or added is in the target's own type.

```python
school   = ss.RandomNet(n_contacts=10, name='school')
seasonal = ss.modify(sir.pars.beta, multiply=ss.seasonal(mean=1.0, amplitude=0.3, peak='January'))
closures = ss.modify('school.pars.n_contacts', multiply=ss.timeseries({'2020-03-15': 0.2, '2020-06-01': 0.6, '2020-09-01': 1.0}))
caution  = ss.modify(sir.rel_sus, multiply=lambda sim: 0.5 if sim.results.sir.n_I[sim.ti] > 200 else 1.0)
sim = ss.Sim(sir, networks=school, interventions=[seasonal, closures, caution], start='2020-01-01', stop='2020-12-31', n_agents=2000)
sim.run()
```

A modifier that depends only on time, and on who it applies to only through states and stratifications (`where=hiv.acute`), is worked out before the run, so a compartmental run stays in one call to the kernels; one with `when=`, `hold=`, or a function as its amount is applied step by step.

Protection that wanes with time since a dose is a property of a product (`waning=`, [08-interventions.md](08-interventions.md#products)).

`ss.change` ([08-interventions.md](08-interventions.md#sschange)) is a shorthand that returns an `ss.modify` whose amount is a step function of time (`ss.timeseries(..., mode='constant')`), starting at its first time, so `stop=`, `when=`, and `where=` work as for any modifier.

### `sim.what_modifies()`

```python
sim.what_modifies(sir.pars.beta)
```

For `rel_sus` in an agent-based sim:

```text
sir.rel_sus
  modifiers
    masks            multiply=0.6   where=sir.S   start=5
```

Modifiers are declared, so they are always listed: every `ss.modify`, `ss.change`, and product efficacy on the target, with its operation, condition, and period. The list MUST be complete; it is available before the sim has run (`what_modifies()` initializes the sim if needed). If the modifiers took the quantity out of its range during a run, a line says so. Code that assigns the base value directly (`hiv.rel_trans[uids] = ...`) is not a modifier, and is not tracked.

Scaling a shared quantity through `ss.modify` (or `*=`) is preferred to overwriting it, so that two modules do not clobber each other. This is guidance, and `what_modifies()` is how to check; nothing is enforced.

## Conditions

A condition says who something applies to. It is used for `where=` here and in the deliveries of [08-interventions.md](08-interventions.md). There are no global condition names (`ss.age`, `ss.sex`, `ss.time` and the like do not exist): conditions are written against the real attributes of the people and modules in the sim.

### A function of the sim

Always available, and works with any attribute, including custom ones:

```python
eligible = lambda sim: (sim.people.age > 65) & ~sim.diseases.sir.vaccinated
```

The function is called when the condition is needed and returns a boolean array (or UIDs). This is Starsim's existing form. A function may also take `(sim, uids)`; that form is agent-only.

### The short form: lazy conditions

With the objects in scope, the same condition can be written without the lambda:

<!-- fragment -->
```python
ppl = ss.People(n_agents=10_000)
sir = ss.Disease(...)

eligible = (ppl.age > 65) & ~sir.vaccinated
```

This works because a state array that has not been initialized builds an expression instead of computing a value. The exact rule:

- **An array is uninitialized** until the sim it belongs to has been initialized (`sim.init()`, or the start of `sim.run()`). `ppl.age`, `ppl.female`, and every state of a module (`sir.vaccinated`, `hiv.cd4`, `syphilis.active`) are such arrays; so are properties built from them. The people's own properties can also be had with no people made: `ss.ppl.age > 65` is the same condition as `ppl.age > 65`, and is the usual form in a compartmental model, which has no people object. `ss.ppl.<name>` works for any property, checked when the sim is built. `ss.People` is the class only (`sim.people = ss.People(...)`): Starsim 3's `ss.People.age` is now `ss.ppl.age`.
- **On an uninitialized array, these operations are lazy**: comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`), logical operators (`&`, `|`, `~`), arithmetic (`+`, `-`, `*`, `/`), and `.isin([...])`. The result is a lazy condition: an unevaluated expression that remembers which arrays it refers to, by module and name. Combining a lazy condition with a number or with another lazy condition is lazy.
- **A lazy condition is evaluated each time it is needed during a run**, against the sim it is running in, using the current values at that point in the loop. It refers to arrays by module and name, so the same condition works in every copy of a sim made by `ss.parallel` or calibration.
- **On an initialized array, the same operations evaluate immediately**, as in Starsim today. Inside `step()`, `self.sim.people.age > 65` is an ordinary boolean array.
- Using a lazy condition where Python needs a single truth value (`if cond:`, `and`, `or`, `not`) is an error that says to use `&`, `|`, `~` instead. Anything the operators above cannot express is written as a function of the sim.
- A boolean state on its own is a condition (`where=syphilis.active`), and a lazy condition (an `ss.Cond`) prints as the expression it was built from, which is what `describe()` shows.
- **A condition in a rate is 1 or 0** for each person (as compartments, the share of each group it holds for), and a property is the person's value, so `0.3*(ss.ppl.age > 65)` and `b*ss.ppl.urban` are rates ([03](03-states-and-transitions.md)). Arithmetic with a parameter makes an expression, not a condition.
- `tr.incidence(over=, by=)` is a condition on how many went through a transition over a window, as a share of the people at the same level of a stratification, so that each place can apply a rule on its own: `sir.infection.incidence(over=ss.days(7), by='patch') > 200e-5`.

No new vocabulary is involved: the objects and attribute names are the ones the user already has.

### On agents and on groups

A condition is evaluated on whatever the solver provides.

- **Agent-based run.** Each array is one value per agent, and the condition is a boolean per agent.
- **Compartmental run.** Each array is one value per group (one row per stratum), and the condition is a boolean per group. This works when everything the condition uses is something the model is stratified by, or a state: `ppl.age > 65` works when age is a dimension with a bin edge at 65, and `sir.vaccinated` works when `vaccinated` is a state or a dimension.

The rule is **try it and report**. Nothing is analyzed in advance. Under `solver='auto'`, a condition or function that fails on groups makes the sim agent-based. Under an explicit compartmental solver, one that asks for something that exists only per agent (`uids`, an attribute the model is not stratified by, an age threshold that cuts through a bin) stops the run with a message that names what was asked for and what would fix it:

```text
CapabilityError: a condition uses people.age and 70, but the model's age bins are [0, 15, 65] and 70 falls inside a bin.
     Fix: add 70 as a bin edge, stratify(age=[0, 15, 65, 70]), or run with solver="abm".
```

The lazy form is checked this way. A function of the sim that compares ages with a number that is not a bin edge is not: it runs, with each group at the age of its midpoint.

A condition on another module's state (`where=syphilis.active` on an HIV quantity) needs the two modules' states tracked jointly. In an agent-based run that is free. In a compartmental run the two diseases' states are tracked jointly, each as a dimension of the other (so `hiv.results.n_I.by('syph')` splits HIV by syphilis state), and the model grows to the product. The crossed diseases then run as one model ([12-backends.md](12-backends.md)), and each disease's results are its view of the product. Diseases are also tracked jointly when people are born or die ([03-states-and-transitions.md](03-states-and-transitions.md)) or die of one of them (a transition to `ss.null`), since those people leave every disease; the members of a family (`ss.Strains`) already have people who die of one die in the others, so dying of a member crosses the family only with diseases outside it. The product is counted before anything is made, including the stages of dwell times, the slots of fixed delays, and a copy of each count for each combination of the others: above `ss.options.max_joint` compartments (10,000 by default, counting the joint states in each group of the other dimensions), `auto` runs agents, and a compartmental solver that was asked for raises `CapabilityError` naming the limit, which can be raised where there is the memory. `describe()` shows why the diseases are tracked jointly, the product's size, the memory of a whole run, and what is assumed: the diseases start independent of each other, and step together at one timestep.

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

<!-- fragment -->
```python
hiv_test = ss.Diagnostic(sensitivity=0.99, specificity=0.99)

anc_testing = ss.screen(hiv_test,
    who      = anc.visit & hiv.infected & ~hiv_test.positive.ever,  # people at an ANC visit this step (an event), HIV-positive and not yet diagnosed (conditions)
    coverage = p_test,
)
diagnosed = hiv_test.positive.ever                  # remembered per person by the test's outcome, so no state is written
treatment = ss.treat(art, who=anc_testing.positive) # what a positive test leads to is a step of its own
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

The two are the same model. The first is shorter and also runs under a compartmental solver if age, sex, ANC attendance, and diagnosis are all stratifications or states. The second can do anything: decide who is referred at the moment of infection and test them at their next visit (add a `referred` boolean state, set it in one line, test `self.referred & anc.visit`), prioritize by clinic, draw down a stock. Code like this is agent-based unless it happens to work on groups, by the same try-it rule as conditions.

Two pieces of guidance for custom code, neither of them enforced:

- **For booleans and events, write directly.** `hiv.diagnosed[positive] = True` has no combination problem.
- **For scaling a number another module owns, use a modifier and flip a flag.** Declare `ss.modify(hiv.rel_trans, multiply=0.04, where=self.suppressed)` once, and have `step()` set and clear `self.suppressed`. The base stays intact, other modules' modifiers still apply, and there is nothing to restore. Writing `hiv.rel_trans[uids] = 0.04` directly is allowed, and sets the base.

The same pattern is how a rule with memory is written: `step()` keeps whatever it needs to remember and sets a flag, and a modifier reads the flag ([08-interventions.md](08-interventions.md#triggers)).

A function can be used in place of a class when no state is needed: a plain function passed as a keyword to a module is called every step ([03-states-and-transitions.md](03-states-and-transitions.md)).

### No required declarations

A module does not declare what it reads or writes. Declarations of that kind get forgotten and drift away from the code, and an out-of-date declaration is worse than none. What modifies a quantity through `ss.modify` is declared, and `what_modifies()` lists it; code that writes a quantity directly is not tracked.

## Execution order

Order is Starsim's ordered loop ([10-execution.md](10-execution.md)). Each step, modules run in a fixed sequence by type (demographics; disease state updates; connectors; networks; interventions; transmission; deaths; results; analyzers), and within a type in the order they were given. A declared disease takes all of its transitions in its `step()`, after the interventions, under every solver. The loop is a data object (`sim.loop`) that can be printed, plotted, and inserted into.

What this means for interactions:

- **Each module sees the current values when it runs.** A chain can therefore complete within one step: infection, testing, and treatment all happen in the same step if the modules are ordered that way.
- **A modifier follows the state it depends on.** A `where=` condition is evaluated where the quantity is used, against the current values ([10-execution.md](10-execution.md#where-ssmodify-is-evaluated)). A `when=` function is a function of the sim as it is at that moment and nothing else; apart from `hold=`, the modifier keeps no record of what it returned on earlier steps.
- **Mutual dependence is fine.** HIV raises syphilis susceptibility while syphilis raises HIV transmissibility; each reads the other's current value, and the order decides which goes first. There are no cycle errors. Where the order matters, the convention is to run last the module whose accuracy matters most.
- **Any effect of ordering is of order `dt`.** `sim.check_dt()` reruns with a smaller step and shows how much the results change.

## Rejected

- **A closed list of modifiable quantities.** Too restrictive for real models; any parameter, transition, transfer, `rel_sus`/`rel_trans`, or network contact rate can be a target, by name.
- **Required `reads`/`writes` declarations with enforced read-only views.** They depend on people remembering to keep them current. Observed writes give the same information without that.
- **Global condition names and an expression type the user has to learn.** Conditions use the sim's own attributes, lazily when written before the sim exists.
- **Analyzing conditions in advance to decide what can run where.** Try the condition on what the solver provides and report what it asked for.
- **Queued updates, named phases, and ordering derived from declarations.** The ordered loop is simpler, allows within-step chains, and handles mutual dependence.
- **Last-write-wins as the only way to share a quantity.** `ss.modify` keeps the base and combines by a stated rule; direct writes are still allowed, and are not tracked.
- **Additive combination as the default.** Available as `add=`.
- **`min_duration=` on `ss.modify`, or any other memory inside a modifier.** A modifier is a function of the current state of the sim, plus `hold=` (stay on for at least this long once selected, as MEmilio's dynamic NPIs do). Any other rule that remembers is a few lines of `step()`, where the memory is visible.
- **Write tracking on by default.** It has a cost on every assignment, and most runs never ask the question.
- **A separate object for protective measures** (a "tool" carrying fixed multipliers). A product's efficacy is a modifier on the people who hold it.
- **`add=` on `rel_sus` for cross-immunity.** It would replace each person's own value; the combination rule is in `ss.immunity`.
- **`ss.immunity(matrix, reduces='rel_sus')`.** The quantity lowered is the keyword (`rel_sus=matrix`), so several quantities can be given at once and none needs a second name.
