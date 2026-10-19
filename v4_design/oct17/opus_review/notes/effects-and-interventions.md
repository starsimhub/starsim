# Review: composition, effects, interventions, products, and deliveries

Scope: `ss.modify`, `ss.change`, conditions (`starsim/conditions.py`), connectors (`starsim/connectors.py`), the v4 deliveries (`ss.Delivery`, `ss.vaccinate`, `ss.screen`, `ss.treat`, `ss.transfer` with delivery keywords) and products (`ss.Vaccine`, `ss.Test`, `ss.Treatment`), the v3 interventions and products that still ship beside them, and how the engines apply all of this. Docs: `best-practices/composition-and-effects.md`, `best-practices/interventions.md`, `design/07-effects.md`, `design/08-interventions.md`. Every bug claim below was reproduced with a script in the scratchpad (`effects/t*.py`), run with `/software/conda/envs/v4/bin/python`.

## Summary

The core ideas are right and should be kept: one-line `ss.modify` with a stated combination rule (set, then multiply, then add) and a kept base value; lazy conditions written against real attributes; "what plus delivery" with keywords for the simple case and one `deliver=` function for the tangled case; a product's efficacy realized as ordinary modifiers, so `what_modifies()` lists vaccines next to connectors; and one delivery class whose behavior is chosen by keywords, not one class per combination. The ODE and agent runs agree closely on everything I tried that should agree (`ss.modify`, `ss.change`, `ss.screen` with an `ss.treat` follow-up, `on=` triggers, capacity).

The biggest problem is not inside the new machinery but at its edge: **none of it works with the v3-style diseases** (`ss.SIR`, `ss.SIS`, `ss.SEIR`, and every disease in `starsim.library`, including HIV), while the full v3 intervention and product stack (`routine_vx`, `campaign_screening`, `treat_num`, `simple_vx`, `ss.Dx` with dataframes, and so on, about 530 lines) still ships and is what those diseases have to use. So a user has two intervention languages and which one works depends on how the disease was written; the error messages do not say so. Second, several guessed defaults are wrong in ways that are silently harmful, against principle 3: a bare-number time in a dated sim is silently a year (so the docstring examples become no-ops, or deliver at step 0); a full-coverage routine delivery of a waning vaccine re-vaccinates everyone every step (and coverage 0.99 does not, a 100-fold difference in infections); and the vaccination-as-a-stratification example in the spec starts half the population vaccinated. Third, there are real bugs (`set=` and `add=` on `rel_sus`, `rel_trans`, or a transition act on a multiplier of 1 rather than on the base, so `set=1.0` on a base of 0.5 leaves it at 0.5; `ss.change` silently drops `start=`/`stop=`/`when=`; an `AttributeError` inside a `when=` function is reported as "modify object has no attribute active"; `ss.Vaccine()` crashes with a `TypeError`), and the design docs still describe a read/write rule (reading gives the effective value) that the code deliberately reversed.

The vocabulary can shrink: `ss.change` is a special case of `ss.modify` with a step-function amount; the `connectors=` versus `interventions=` choice for a modifier makes no observable difference except for `when=`; the products-as-modules appear in the schedule and results with nothing in them; `ss.Treatment` switches between two unrelated behaviors on the type of `efficacy`; and `ss.Vx`/`ss.Dx`/`ss.Tx` change class depending on their arguments.

## Findings, by importance

### 1. The v4 effects and deliveries do not work with any v3-style disease, and the v3 intervention stack is still the only way to intervene on those

Type: design (and API, error messages)

Evidence (`effects/t2.py`, `effects/t22.py`):

```python
sir = ss.SIR()
ss.modify(sir.pars.beta, multiply=0.5, start=2005)
# E504 UnmodifiableTarget: the target of ss.modify() must be a named quantity, such as sir.pars.beta, ... not 0.1 (peryear). A plain number has no name to refer to.
ss.modify('sir.pars.beta', multiply=0.5)      # in a sim with networks='random'
# E504 UnmodifiableTarget: the target of the modifier "m", sir.pars.beta, was not found. Quantities that can be modified: randomnet.pars.n_contacts; did you mean randomnet.pars.n_contacts?
ss.modify(sir.rel_sus, multiply=0.5)
# E504 ... not <FloatArr "rel_sus", len=0, []> (FloatArr). A plain number has no name to refer to.
ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, years=[2005])
# E506 UnknownDisease: "v" needs to know which disease it is for, and the sim has no declared diseases. Use disease=.
ss.change('sir.pars.beta', at=[2005], by=[0.5])   # E504, as above
ss.modify('hiv.rel_trans', multiply=2.0, where=...)  # with starsim.library.HIV: E504, "did you mean mfnet.pars.n_contacts?"
ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=0.5, start_year=2005)   # works
```

Every lookup goes through `sim.declared` (`connectors.py:239-262`, `conditions.py:226-231`, `interventions.py` `Delivery.init_pre`), so a module that is not a declared `ss.Disease` is invisible. All of `starsim/library/diseases/` (Ebola, Measles, Cholera, HIV) subclass `ss.SEIR`/`ss.Infection`, so the flagship example of `07-effects.md` (`ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)`) cannot be written with the library's own HIV. The errors are actively misleading: the parameter is "a plain number", the only modifiable quantity is the network's contacts, and "declared" is a word the user has never been told.

The other direction half works: v3 interventions (`ss.routine_vx`) and custom `ss.Intervention` subclasses run on a declared disease as agents. So the code has two complete intervention systems (`interventions.py:77-532` and `products.py:12-226` for v3, `interventions.py:535-1266` and `products.py:229-533` for v4) whose domains overlap only partly.

Proposal: make the target lookup and the `Selector` work on any module, not only declared ones, at least as agents. The pieces are small: a parameter of any module is `mod.pars[name]`; `rel_sus`/`rel_trans` of a v3 `Infection` are the arrays its `infect()` already reads, and wrapping them as `ss.ModifiableArr` (as `solvers.py:3176` already does for declared diseases) makes `ss.modify` apply; a delivery's product effects for a v3 disease are the same `rel_sus`/`rel_trans` modifiers; `Treatment` cure for a v3 disease is the disease's own `clear_infection()`/`set_prognoses()` hook (or an error naming it). Then retire the v3 delivery classes (`RoutineDelivery`, `CampaignDelivery`, `BaseScreening`, `routine_screening`, `campaign_screening`, `routine_triage`, `campaign_triage`, `BaseTreatment`, `treat_num`, `BaseVaccination`, `routine_vx`, `campaign_vx`, `simple_vx`) into a compatibility module with deprecation warnings that print the v4 equivalent:

```python
# Before (v3, still the only thing that works on ss.SIR)
ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=0.5, start_year=2005)
# After (works on ss.SIR and on ss.Disease alike)
ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, start=2005)
```

Until that is done, the errors should say what is actually wrong: "`sir` is a class-based disease (`ss.SIR`); `ss.modify` currently works only on diseases built from transitions (`ss.Disease(infection=..., recovery=...)`). Use a custom `ss.Intervention`, or ..." and `parse_target` should not call a typed rate "a plain number".

Cost: real work in `Selector` (it assumes `mod.graph`), in `parse_target`, and in the engine-side application for v3 `Infection.infect()`; Treatment-cure needs a hook per v3 disease. But the alternative is shipping two intervention languages indefinitely, which is the opposite of principle 9's vocabulary budget, and most existing Starsim users' diseases are v3-style.

### 2. A bare-number time in a dated sim is silently a year, so the docstring examples do nothing, and a campaign before the start is delivered at step 0

Type: bug (guessed default that is silently harmful)

Evidence (`effects/t25.py`), sim from 2020-01-01 to 2020-06-01, base 9,312 infections:

```python
ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0])           # 9,312: no effect (years 30 and 60 are before the start)
ss.modify(sir.pars.beta, multiply=0.5, start=30, stop=60)      # 9,312: no effect
ss.change(sir.pars.beta, at=[2030], by=[0.5])                  # 9,312: no effect (after the end)
ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, years=[30])  # delivered 5,000 doses at step 0
ss.change(sir.pars.beta, at=[ss.days(30), ss.days(60)], by=[0.5, 1.0])   # 8,675: what was meant
```

`connectors.to_time()` turns any number into a date when the sim is absolute, and `to_ti()` returns 0 for a time before the start and `len(tvec)` for one after the end, with no message. The docstring examples of `ss.modify` and `ss.change` (`start=30`, `at=[30, 60]`) are written for undated sims, and copied into a dated sim they silently become no-ops. v3's `RoutineDelivery` raised "Years must be within simulation start and end dates"; v4 lost that check.

Proposal: in `to_ti()`, a time before the start or after the end of the sim is `E5xx TimeOutsideSim` naming the time, the sim's span, and the fix ("did you mean `ss.days(30)`?"), with one exception: `stop=` after the end is fine and silent (it means "never stops"). A start before the sim's start for a modifier could be allowed with a note ("applies from the start"), but a campaign round before the start should be an error, since delivering it at step 0 is never what was meant.

Cost: a few lines; some existing tests that rely on clamping may need `start=None`.

### 3. A routine delivery with full coverage of a waning product re-vaccinates everyone every step; coverage 0.99 does not

Type: bug

Evidence (`effects/t6.py`), 10,000 people, 100 steps, `ss.Vaccine(efficacy=0.9, waning=10)`:

| Delivery | Doses (ODE) | Infections (ODE) | Doses (agents) | Infections (agents) |
|---|---|---|---|---|
| `ss.vaccinate(vx, start=0)` | 104,211 | 73 | 104,042 | 39 |
| `ss.vaccinate(vx, start=0, coverage=0.99)` | 9,900 | 9,196 | 9,896 | 9,159 |

The spec says routine delivery offers each person the product once, at the first step they are eligible (`08-interventions.md` §Rounds and routine delivery). `Delivery.remembers` (`interventions.py`, the `remembers` property) skips the `offered_<delivery>` record when coverage, capacity, and `when=` are all absent and people hold the product, on the reasoning that everyone eligible holds it, so the "not already holding it" filter is enough. With exponential waning (`Held.loses`), people stop holding it, become eligible again, and are re-vaccinated on the next step. So the default (no `coverage=`) is a continuous re-vaccination program, and a 1% change in coverage changes the epidemic 100-fold.

Proposal: `remembers` must also be true when the product can be lost (`self.holds and self.product.loses`), so the shortcut applies only to products held for good. If re-vaccination after waning is wanted, that is a schedule (`interval=`) or an explicit `revaccinate=True`/`eligible=` that says so.

Cost: one more level of `offered_<delivery>` in compartmental runs of waning products (a doubling of groups for that delivery), which is what the spec already implies.

### 4. `ss.change` silently ignores `start=`, `stop=`, and `when=`

Type: bug

Evidence (`effects/t7.py`): base 9,309 infections; `ss.change(sir.pars.beta, at=[0], by=[0.3], when=lambda sim: False)` and `ss.change(..., stop=10)` both give 492, the same as with no `when=`/`stop=`. `change.__init__` passes `**kwargs` through to `modify.__init__`, which stores them, but `change.active` (`connectors.py:384`) is `return self.sim.ti >= self.tis[0]` and never reads them.

Proposal: either reject them (`ss.change() takes at= and by=/to=; for a period or a trigger, use ss.modify(start=, stop=, when=)`) or honor them by calling `super().active` as well. Better, see finding 9: make `ss.change` a thin constructor for an `ss.modify`, so there is only one `active`.

Cost: trivial.

### 5. The design docs say reading a modified quantity gives the effective value; the code gives the base

Type: doc-code mismatch

Evidence: `07-effects.md` §Base and effective values, rules 1-3 ("Reading gives the effective value ... `.base` reads the base") and the worked example; `composition-and-effects.md` §Reading and writing a modified quantity says the same and warns about `*=`. The code (`implementation.md` §Departures: "reading `sir.rel_sus` from custom code gives the base value, and `sir.rel_sus.effective` the value with the modifiers applied (agents only). So `.base` is not needed.") does the opposite, and `effects/t5.py` confirms it:

```text
read [1. 1. 1. 1.]   eff [0.5 0.5 1. 0.5]          # where=risk=='high' halves rel_sus
after assign 0.8: read [0.8 ...]   eff [0.4 0.4 0.8 0.4]
after *= 0.5:     read [0.4 ...]   eff [0.2 0.2 0.4 0.2]
```

The code's choice is the better one: reading and writing the same thing removes the `*=` trap that the docs spend a paragraph and an open question on, and existing v3 code keeps working. But two best-practices/design pages and the worked example teach the opposite rule, which an AI writing models will follow.

Proposal: update `07-effects.md` and `composition-and-effects.md` to the implemented rule (read and assign the base; `.effective` for the value in use), delete the `*=` open question, and say what `.effective` does in a compartmental run (today it is agents only; it should at least raise a clear error there).

Cost: docs only.

### 6. The vaccination-as-a-stratification example starts half the population vaccinated

Type: design (guessed default), doc-code mismatch

Evidence (`effects/t19.py`), the spec's own example (`08-interventions.md` §Simple cases, `best-practices/interventions.md`):

```python
sir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.15]})
rollout = ss.transfer(vaccinated=True, coverage=0.7, years=[10])
```

`describe()` prints `dimension vaccinated (plain): False, True [guessed: equal across levels]`, so 5,000 of 10,000 people are vaccinated at time 0; the rollout then reaches 3,500 (70% of the other half). Infections: 776, against 3,285 for the "same" intervention written as `ss.vaccinate(ss.Vaccine(efficacy=0.85), coverage=0.7, years=[10])`, and 4,589 with the stratification and no rollout at all. The guess is printed, but for a dimension whose levels are `[False, True]` and that a transfer moves people into, "equal across levels" is almost never right; `Delivery.dims()` itself makes the same dimension with `split=[1.0, 0.0]` when it creates it, but loses to the user's `stratify()`.

Proposal: for a boolean dimension (levels `[False, True]`) with no `split=` and no population data, guess everyone at `False` (and print that). More generally, when an `ss.transfer` or delivery moves people into a level of a dimension, a missing split means "everyone starts at the first level". Fix the two doc examples to show `split=` or rely on the new guess.

Cost: changes the default for one class of dimension; any model that relied on 50/50 for a boolean dimension must now say `split=[0.5, 0.5]`, which is clearer anyway.

### 7. An `AttributeError` inside a `when=` function (or any property it reaches) is reported as "modify object has no attribute active"

Type: bug (error message)

Evidence (`effects/t21.py`): the `SchoolClosure` example from `08-interventions.md` §Triggers, run as written:

```text
AttributeError: modify object has no attribute "active"
```

The real error is that `sim.interventions.closure` does not exist (see finding 22: a custom `ss.Intervention` is named `schoolclosure`, not after its variable). `modify.active` is a property; when the user's `when=` raises `AttributeError` inside it, Python falls back to `Module.__getattr__` (`modules.py:304`), which raises a new `AttributeError` about `active` and discards the original. The same applies to every property on `modify` and `Delivery` that runs user code or looks things up (`tabulated`, `dis`, `eng`, `selector`; `due()` is a method, so it is not affected).

Proposal: make the call sites that run user code methods, not properties (`modify.is_active()`), or wrap user calls (`when`, function amounts, `deliver=`, conditions) in a helper that re-raises with the function's name: "the when= function of 'effect' raised AttributeError: 'ndict' object has no attribute 'closure'". The second is better, because it also covers user functions called from methods, and it is what principle "errors name the rule, show the expression, state the fix" asks for.

Cost: small.

### 8. `set=` and `add=` on a per-person quantity act on a multiplier of 1, not on the base

Type: bug (doc-code mismatch with the stated combination rule)

Evidence (`effects/t30.py`, `t31.py`), agents, with a custom module assigning `rel_sus = 0.5` (the base) for everyone:

```text
set 0.3       base [0.5 0.5]  effective [0.15 0.15]     # the rule says 0.3
add 0.2       base [0.5 0.5]  effective [0.6 0.6]       # the rule says 0.7
multiply 0.5  base [0.5 0.5]  effective [0.25 0.25]     # correct

infections: base 1.0 18,586;  base 0.5 8,594;  base 0.5 + set=1.0 8,594 (rule: 18,586);  base 0.5 + add=0.5 16,584 (rule: 18,586)
```

`Engine.mult_raw()` (`solvers.py:957-964`) evaluates `combine(mods, 1.0, self)` and the result is then multiplied into the base, so `set=x` becomes "multiply by x" and `add=a` becomes "multiply by 1 + a". The rule in `07-effects.md` §How modifiers combine ("(set value if any, else base) × Π multipliers + Σ addends") and in `modify`'s own docstring holds only for parameters (`Engine.modified()`, which passes the real value). For `rel_sus`/`rel_trans` and transition rates, `set=` and `add=` are silently a different operation from what they say.

Proposal: either apply set and add to the base for these quantities (pass the base array into `combine` and divide out afterwards only where the engine needs a pure multiplier), or, simpler and arguably better, restrict `set=`/`add=` to parameters and raise for per-person states and transitions ("`set=` on `sir.rel_sus` would replace each person's own value; use `multiply=`"). Given that `set=` already needs a warning (W502) and the best-practices doc says multiplicative is the default for good reasons, I would take the second: it shrinks the surface and removes a silent wrong answer.

Cost: the first option costs an extra array per modified quantity per step; the second removes a feature nobody can currently be relying on correctly.

### 9. `ss.change` could be one spelling of `ss.modify`, and the two disagree before the first time

Type: API (vocabulary), code simplicity

Evidence: `connectors.py:339-400` subclasses `modify`, overrides `active` and `current` (mutating `self.amount` on each call), and has its own `describe` and `to_json`. `ss.modify(sir.pars.beta, multiply=ss.timeseries({20: 0.2, 60: 1.0}))` holds 0.2 before step 20 (constant extrapolation), whereas `ss.change(sir.pars.beta, at=[20, 60], by=[0.2, 1.0])` holds the base before step 20 (`effects/t12.py`, `t13.py`: same values from step 20 on). Two spellings of "a step function of time" that behave differently on the first segment is a trap, and finding 4 is a symptom of the subclass not quite being a modifier.

Proposal: keep the name `ss.change` (it is Covasim's form and reads well), but implement it as a function returning an `ss.modify` whose amount is `ss.timeseries(dict(zip(at, by)), mode='constant', before=1.0)` (or `before=None` meaning "no modifier", for `to=`). That deletes the subclass and its overrides, makes `start=`/`stop=`/`when=` work for free, and gives one documented rule for what happens before the first time. Alternatively drop `ss.change` and teach `ss.modify(target, multiply={'2020-03-15': 0.5, '2020-06-01': 1.0})`, where a dict amount is a step function; but `ss.change` is the shorter, more readable spelling and earns its place.

```python
# Before: a subclass with its own active/current
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
# After: the same call, returning
ss.modify(sir.pars.beta, multiply=ss.timeseries({'2020-03-15': 0.5, '2020-06-01': 1.0}, mode='constant', before=1.0))
```

Cost: `ss.timeseries` needs a `before=` (or the modifier needs `start=` set to the first time); `isinstance(x, ss.change)` stops working, which only internal code uses.

### 10. A plain function or a custom module forces agents, against "try it and report"

Type: design, doc-code mismatch

Evidence (`effects/t28.py`, `t21.py`, `t13.py`):

```python
def knock(sim):
    if sim.ti == 20: sim.diseases.sir.pars.beta = 0.1
ss.Sim(sir, interventions=knock, total_pop=10000).run()
# solver: abm (Intervention only works on agents)     -- and the model is now 10,000 agents
```

The `SchoolClosure` class of `08-interventions.md` §Triggers likewise gives `solver: abm (SchoolClosure only works on agents)`, though it reads only `sim.results` and sets a flag, and a custom analyzer under `solver='ode'` is `E804 ... only works on agents` until it has `compartmental = True`. `sim.py:368` and `sim.py:979`/`1181` decide by the class attribute `compartmental`, not by trying. The spec says the opposite twice: "A module with a custom `step()` runs in an agent-based sim. It also runs in a compartmental sim if the code happens to use only things that exist for groups" (`composition-and-effects.md`) and "Under `'auto'`, functions are tried on groups once at `sim.init()`" (`12-backends.md` §Functions: try it). The reason printed names the class (`Intervention`), not the function (`knock`).

Proposal: either implement the rule as written (run the module's `step()` on groups and fall back to agents if it touches `uids`, `auids`, or a per-agent array; the `Selector` already raises `E804` from exactly those places), or change the spec to say that custom modules are agent-only unless marked `compartmental = True`, and document the flag in the user-facing docs. The first is better for users and consistent with conditions; the second is honest and cheap. Either way, the reason should name the function or module (`knock`, `schoolclosure`). A switch from an ODE to 10,000 agents because of a one-line policy function is a large, surprising change to the model.

Cost: trying `step()` on groups has side effects (it mutates state), so it would have to be tried on a throwaway copy at init, or tried live on the first step with a fallback that restarts; the class-flag approach avoids that, which is probably why it was chosen. If so, say so in the spec.

### 11. `connectors=` versus `interventions=` for a modifier makes no observable difference, and the connector base class is empty

Type: API (vocabulary), code simplicity

Evidence (`effects/t15.py`): `ss.modify(sir.pars.beta, multiply=0.6, start=30)` gives 8,741.03 infections as a connector and 8,741.03 as an intervention (ODE), and 8,427 both ways as agents. The spec says the two lists "differ only in where the condition is evaluated in the loop", but a `where=` on states is evaluated where the engine reads the quantity, so only `when=` and function amounts can tell the difference. Meanwhile `ss.Connector` (`connectors.py:14-21`) is `pass`, `modify` subclasses it and sets `has_product = False # So it can be passed as an intervention`, and the only stock connector, `ss.seasonality`, does the thing the effects design exists to prevent: `disease.rel_trans[:] = rel_beta` (`connectors.py:91`), overwriting whatever else set it.

Proposal: tell users that a modifier can go in either list and it does not matter (or accept a `modifiers=` list and stop asking), and keep the connectors/interventions distinction only for modules with a `step()`, where loop position is real. Replace `ss.seasonality` with a one-line `ss.modify`, which is what it should teach:

```python
# Before: overwrites rel_trans every step, clobbering any other modifier
ss.Sim(..., connectors=ss.seasonality(scale=0.2, shift=0.0))
# After: combines with everything else, and appears in what_modifies()
ss.Sim(..., connectors=ss.modify(sir.pars.beta, multiply=ss.seasonal(amplitude=0.2, peak='January')))
```

Cost: `ss.seasonality` is public v3 API; keep it as a thin wrapper that returns the modifier, with its `plot()`.

### 12. Products are modules: they appear in the schedule and the results with nothing in them, and are stepped by their deliveries

Type: code simplicity, API

Evidence (`effects/t9.py`): a screen with a treat follow-up gives `sim.modules` = `screening, screening_positive, test, treatment, sir`, `describe()` prints `schedule every step: screening, screening_positive, test, treatment, sir`, and `sim.results.test` and `sim.results.treatment` contain only `timevec`. Waning is stepped by the delivery calling `product.step_waning()` with a `_ti_waned == sim.ti` guard so that two deliveries of one product do not wane it twice (`products.py:388-402`); likewise `_ti_staged`, and `Held.init_pre`/`Test.init_pre` guard with `linked` because "both the sim and the deliveries initialize a product".

Proposal: make a product a plain object owned by the sim's modifier table, not an `ss.Module`; give the sim (or the first delivery of each product) the one waning step, and drop products from the schedule and the results. If a product needs states (`ti_delivered`), those belong on the people (`has_<product>` already is). This removes the guards, two lines of `describe()` noise, and empty results.

Cost: products currently get random-number streams and states through the module machinery; those would move to the delivery or the sim. v3 `ss.Product` subclasses (`Dx` with dataframes) are modules and would stay so in the compatibility layer.

### 13. `ss.Treatment` is two unrelated products chosen by the type of `efficacy`, and a non-curing treatment is given to everyone

Type: API, design (guessed default)

Evidence: `products.py:431-478`. `ss.Treatment(efficacy=0.95)` cures (moves people out of the infected states along a guessed recovery transition, then is not held); `ss.Treatment(efficacy=dict(transmission=0.96))` is held for good and scales rates, and cannot also cure. The default eligibility is "infected" only for the curing kind (`Delivery.default_eligible`), so `ss.treat(ss.Treatment(efficacy=dict(transmission=0.96)), coverage=0.5, interval=5)` on an SIR delivers ART to 9,999.99 of 10,000 people, susceptible and recovered included (`effects/t18.py`). And the "did you mean" in the error for an unknown key lists every option (`E505 ... Valid keys are acquisition, transmission, infection, recovery; did you mean acquisition, transmission, infection, recovery?`), because `closest()` returns the whole list when nothing is close.

Proposal: one rule for both: a `Treatment` is for people in the infected states by default (eligible=), whatever its efficacy; `cure=` (a probability, default 0) and `efficacy=` (effects while held, like a vaccine) are separate keywords, so a treatment can do both (cure some, suppress the rest):

```python
# Before
ss.Treatment(efficacy=0.95)                                    # cures
ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))  # held, does not cure; given to anyone unless eligible= says otherwise
# After
ss.Treatment(cure=0.95)
ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))  # held, given to the infected by default
ss.Treatment(cure=0.3, efficacy=dict(transmission=0.9))        # both, which is not possible today
```

And `closest()` should return nothing (and the message should omit "did you mean") when no option is close.

Cost: `efficacy=<number>` on a treatment would change meaning; keep it as an alias for `cure=` with a note for one release.

### 14. Product defaults and aliases crash or surprise

Type: bug, API

Evidence (`effects/t6.py`, `t15.py`):

```python
ss.vaccinate(ss.Vaccine(), coverage=0.5, years=[10])   # TypeError: float() argument must be a string or a real number, not 'NoneType'
ss.Dx()                                                # AttributeError: 'NoneType' object has no attribute 'state'
type(ss.Vx())               # Vx (the v3 class)
type(ss.Vx(efficacy=0.9))   # Vaccine, and isinstance(ss.Vx(efficacy=0.9), ss.Vx) is False
```

`Held.__init__` defaults `efficacy=None`, which reaches arithmetic. `Vx`/`Dx`/`Tx.__new__` (`products.py:37`, `109`, `173`) return a different class depending on whether keyword arguments were given, so the alias is a class whose instances are not instances of it.

Proposal: `ss.Vaccine()` with no efficacy is either `efficacy=1.0` with a printed guess, or an error naming the keyword; I would pick the error, since a vaccine of unknown efficacy is not a safe guess. Make `ss.Vx`, `ss.Dx`, `ss.Tx` plain functions (or `Vaccine` aliases) once the v3 classes move to a compatibility module (finding 1); until then, `ss.Dx()` with no arguments should error with the two spellings.

Cost: small.

### 15. Lazy conditions on people need a throwaway `ss.People(1)`, and passing it to the sim gives a one-agent model

Type: API, design

Evidence (`effects/t23.py`; `implementation.md` §Departures: "A condition on a property of people (`ppl.age < 5`) needs a `ppl` object (`ss.People(1)`) even in a compartmental model"):

```python
ppl = ss.People(1)
v = ss.vaccinate(ss.Vaccine(efficacy=0.85), coverage=0.7, years=[10], eligible=ppl.age < 5)
sim = ss.Sim(sir, people=ppl, interventions=v, total_pop=1e5).run()
# solver abm, len(sim.people) == 1, pop_scale 100000, 0 doses delivered
```

The condition only records `Cond('people', 'age')`; the object is never consulted, so a `People` from a different sim (or none at all) works just as well, which shows the object is a placeholder. A user who follows the natural reading and passes their `ppl` gets a one-agent sim scaled to 100,000 with no warning.

Proposal: export one placeholder, `ss.people`, whose attributes are lazy (`ss.people.age < 5`, `ss.people.female`), meaning "the people of whatever sim this runs in". This is one name, mirrors `sim.people` inside functions, and is not the closed predicate vocabulary that the design rejects (it has no built-in names of its own: `ss.people.clinic` works for a custom attribute). Keep `ppl.age` working for those who have a real `People`. Separately, a sim given `people=ss.People(1)` and `total_pop=1e5` should warn that it has one agent.

```python
# Before
ppl = ss.People(1)                          # never used for anything else
eligible = (ppl.age > 65) & ~sir.vaccinated
# After
eligible = (ss.people.age > 65) & ~sir.vaccinated
```

Cost: one new public name; principle 9 asks every name to earn its place, and this one removes a dummy object from every example with an age condition.

### 16. Delivery results: realized coverage is 0 on steps without delivery, so its summary is meaningless; `deliver=` reports nobody eligible

Type: API (results)

Evidence (`effects/t9.py`, `t20.py`): `ss.screen(test, coverage=0.4, interval=7, ...)` summarizes as `screening_coverage 0.059` and its follow-up `ss.treat(..., coverage=0.8)` as `screening_positive_coverage 0.119`, because `coverage` is 0 on the six steps in seven with no round and `summarize()` averages it. With `deliver=`, `eligible` is never tallied, so `eligible` and `coverage` are 0 throughout while `delivered` is not.

Proposal: `coverage` is NaN on steps where nobody was eligible (and `eligible` is NaN under `deliver=`, or the function's return is counted as eligible), and the summary of `coverage` is total delivered over total eligible, not a mean of ratios. The follow-up's auto-generated name (`screening_positive`) also reads like a count of positives in `screening_positive_delivered`; `screening_treat` or the product's name would read better.

Cost: small; NaN in results may need care in plotting.

### 17. Interventions written for v3 that check `sim.ti == 0` never fire

Type: design (compatibility), doc

Evidence (`effects/t32.py`, `t31.py`): an intervention's `step()` is called with `ti = 1, 2, ..., n` in both declared and v3-style sims; a custom module that assigns at `ti == 0` silently does nothing (`effects/t31.py` before the fix: "base=0.5" gave the same 18,586 infections as base 1.0). This follows from results by time point (`implementation.md` §By section: "The loop skips the steps at the first time point"), which is a reasonable choice, but v3 interventions commonly use `if self.ti == 0` or `sim.ti == 0` to initialize.

Proposal: say it in the migration guide (a comparison with 0 cannot be detected, so it cannot warn), document that the first call to `step()` has `ti == 1`, and point to `init_post()` as the place for one-time setup. Alternatively call modules' `step()` once at `ti == 0` as v3 did, with deliveries skipping it; but that reintroduces the off-by-one that the time-point change fixed.

Cost: docs, unless the loop changes.

### 18. The documented `cases = a.diagnosed + b.diagnosed` does not exist, and the user's objects are not the sim's

Type: doc-code mismatch

Evidence (`effects/t14.py`): `symptomatic_testing.diagnosed + screening.diagnosed` from `08-interventions.md` §What a delivery reports gives `AttributeError: screen object has no attribute "diagnosed"`. It is listed as open in `19-open-questions.md`, but 08 presents it as working, and it is the recommended way to get reported cases when testing is modeled. Separately, the sim copies its inputs (`copy_inputs=True`, as in v3), so after `ss.Sim(..., interventions=[screening])`, `screening.results` is empty and `sim.interventions[0] is screening` is `False`; the doc's style of naming things by variable invites users to read results off their own objects.

Proposal: either implement lazy results on deliveries (`screening.diagnosed` as a reference by name, like a lazy condition, resolved in the sim's results) and an `outcome`-like way to declare a sum, or mark the example in 08 as not yet available. Consider giving the user's object a back-reference to the sim's copy after `run()` (or making `screening.results` raise "this is the template; the results are in sim.results.screening").

Cost: lazy results are moderate work; the error message is trivial.

### 19. There is no all-or-nothing vaccine in v4

Type: design (missing capability)

Evidence: v3's `ss.simple_vx(leaky=False)` (`products.py:188-224`, tested by `tests/test_interventions.py::test_sir_vaccine_all_or_nothing`) protects a fraction `efficacy` fully and the rest not at all; `ss.Vaccine` is leaky only, and neither design doc mentions the distinction. Leaky and all-or-nothing vaccines with the same efficacy give different epidemics, and this is a standard modeling choice that would otherwise be a reason to keep the v3 product.

Proposal: `ss.Vaccine(efficacy=0.85, take=...)` is one option, but a smaller one is a keyword on the efficacy kind: `ss.Vaccine(efficacy=0.85, leaky=False)`, realized as holding the product with efficacy 1 for a share `efficacy` of recipients and with efficacy 0 for the rest (in a compartmental run, two levels of `has_<product>`; as agents, a draw at delivery). That keeps the v3 word.

Cost: one keyword and one extra level in compartmental runs.

### 20. A `deliver=` function that returns a state fails in a compartmental run with a `TypeError`

Type: bug (error message)

Evidence (`effects/t20.py`): `ss.treat(drug, deliver=lambda sim: sim.diseases.sir.I, capacity=50)` works as agents and, under `solver='ode'`, raises `TypeError: float() argument must be a string or a real number, not 'CountState'` from `delivered_by_function()`. `Selector.to_selection()` already knows how to turn a `CountState` into a selection, but `delivered_by_function()` only sends `ss.uids`, `ss.Arr`, and `ss.Cond` there.

Proposal: route `CountState` (and `ss.State`) through `to_selection()` as well, so "everyone in I" works under both solvers; and when the return is neither a selection nor a number per group, say so with the expected shapes.

Cost: one line.

### 21. Modifiers are recomputed every step, and `when=` or a function amount makes a compartmental run about 10 times slower

Type: performance

Evidence (`effects/t17.py`, `t24.py`): as agents with 200,000 people and 200 steps, 0, 1, and 5 modifiers with `where=ppl.age > x` take 0.60, 0.75, and 1.25 s, so each costs about 0.65 ms a step whether or not anything it depends on changed (`Engine.cache` is cleared every step, `solvers.py:1400` and others). In an ODE of 1,000 steps: no intervention 12 ms, `ss.change` 37 ms, `ss.modify` with an `ss.timeseries` 57 ms, `ss.modify(when=...)` or a function amount 137 ms, `ss.screen` 201 ms. The spec (`07-effects.md` §Implementation) says the effective array "MUST be recomputed only when something that feeds it changes" and "a modifier with a constant multiplier and an unchanged active set costs one comparison per step".

Proposal: for agents, keep the effective multiplier between steps and invalidate on the events the engine already tracks (`groups_changed`, state changes of the states a condition names, a new value of a time-varying amount); for compartmental runs, a `when=` that is a pure function of time could be tabulated as `ss.change` is. Or relax the MUST in the spec: the current costs are acceptable for most models, and the simpler code may be worth it.

Cost: invalidation logic is a classic source of bugs; I would relax the spec rather than add it, unless a real model shows the cost.

### 22. Smaller doc-code and API notes

Type: doc-code mismatch, API

- `08-interventions.md` §`ss.change`: "A bare number is a year". The code makes it a year only in a dated sim (`implementation.md` departures). Finding 2 is the consequence; the doc should state the implemented rule and the new error.
- `07-effects.md` §`sim.what_modifies()` shows `[stisim.py:14]` source locations and "active for 1,204 agents"; the output has neither, and prints float noise (`multiply=0.19999999999999996` for an efficacy of 0.8, `effects/t15.py`). Round to the precision given (`1 - 0.8` printed as `0.2`).
- `ss.modify(sir.pars.beta, multiply=ss.seasonal(peak='January'))` in a sim with no dates and no unit runs without a note, treating steps as days (`effects/t26.py`). A calendar-based amount in a sim without dates deserves a note or an error.
- `ss.vaccinate` raises `TypeError` for an `ss.Treatment` (`kind` check in `Delivery.__init__`). The three classes `vaccinate`, `treat`, and `screen` differ only in this check, so they could be one `ss.deliver(product, ...)` with the three names as readable aliases that do not reject anything; or keep the check, which catches real mistakes cheaply. I would keep the names and the check; it is fine as is.
- Custom `ss.Intervention` subclasses are not named after their variable (`closure = SchoolClosure()` is `schoolclosure`), while v4 objects are (`effect`, `screening`), so the spec's own trigger example (`sim.interventions.closure.closed`) fails (finding 7). Naming from the variable should apply to every module or none.

## What is good and should be kept

- `ss.modify(target, multiply=, where=, start=, stop=)` as the one-line interaction, with the base kept and the combination rule stated. It replaced most hand-written connectors in the tests, and the coinfection example (two declared diseases modifying each other) agrees between ODE and agents (`effects/t29.py`: 20,000 and 17,778 against 20,000 and 18,182).
- Reading and assigning the base, with `.effective` for the value in use (finding 5): simpler than the documented rule, and keeps v3 code correct.
- Lazy conditions: `(ppl.age > 65) & ~sir.vaccinated` prints as itself, works on groups when age is a dimension (with the precise `E804` when a threshold cuts a bin), and refers to modules by name so it survives copies. `__bool__` raising with the fix is exactly right.
- One `Delivery` class with keywords for routine, rounds, `on=`, `when=`, `deliver=`, `capacity=`, and follow-ups to any depth; the `Selector` abstraction (a boolean per agent, or a fraction per state and group) lets the same code run under both solvers, and the results agree closely (screen-and-treat: 8,041 ODE, 8,120 binomial, 7,851 agents).
- Product efficacy realized as ordinary modifiers, so `what_modifies(sir.rel_sus)` lists a vaccine next to a `where=` modifier.
- `W502` on overlapping `set=`, `W503` on clipping, and `when=` called exactly once per step (`effects/t7.py`).
- `capacity=` as a separate concept from `coverage=`, with "while stocks last" for free.
