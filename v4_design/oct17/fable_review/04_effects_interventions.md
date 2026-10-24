# Effects and interventions

I reviewed `v4_design/design/07-effects.md` (290 lines) and `08-interventions.md` (283), the relevant rows of `17-vocabulary.md` and `19-open-questions.md`, and the departures and known issues in `v4_design/admin/implementation.md`. Code: `starsim/connectors.py` (400 lines: `Connector`, `seasonality`, `Target`, `parse_target`, `combine`, `modify`, `change`), `starsim/conditions.py` (446: `Cond`, `Selector`), `starsim/interventions.py` (1266: 529 lines of Starsim 3 classes, 737 of v4's `Delivery`/`vaccinate`/`treat`/`screen`), `starsim/products.py` (533: 226 lines of Starsim 3's `Dx`/`Tx`/`Vx`/`simple_vx`, 307 of v4's `Effect`/`Held`/`Vaccine`/`Treatment`/`Test`), `ModifiableArr` in `starsim/arrays.py:915-958`, `Sim.what_modifies` in `sim.py:493`, and the modifier plumbing in `solvers.py` (`modified`, `contact_mult`, `mult_raw`, `clip`). I ran seven scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/effects/` (`s1_modify.py` through `s7_transfer.py`): a vaccine with waning under `ode` and `abm`, a threshold trigger as `when=` and as the spec's `SchoolClosure` class, a screen-confirm-treat cascade with a delayed test, modifiers with conditions by object and by path, lazy conditions against functions of the sim, and vaccination as `ss.transfer` against `ss.vaccinate`. I also counted where each public name is used in `starsim/library`, `tests/`, `docs/`, and the vignettes.

## Summary

1. **Delete the Starsim 3 intervention and product vocabulary** (`routine_vx`, `campaign_vx`, `routine_screening`, `campaign_screening`, `routine_triage`, `campaign_triage`, `treat_num`, `Base*`, `RoutineDelivery`, `CampaignDelivery`, `Dx`, `Tx`, `Vx`, `simple_vx`: 19 names, 755 lines). No library module, no vignette, and no v4 test uses them; the two worlds do not interoperate (an old delivery cannot carry a new product and vice versa); and the `ss.Vx(efficacy=)`-returns-a-`Vaccine` alias trick is already producing crashes (`ss.Dx()`). Before deleting, make `ss.vaccinate` work on a code-written disease such as `ss.SIR`, which today it refuses (`E506`). Breaking: five docs pages and `tests/test_interventions.py`.
2. **Decide for the implementation on base versus effective, and say so in the spec.** Reading gives the base and the engine applies modifiers where the quantity is used; `.effective` exists only as agents and `.base` does not exist at all. This is the simpler model (one array, no cache invalidation on read); the spec's three rules should be replaced by one sentence. Then either drop `.effective` (one test uses it) or keep it as a debugging aid. Not breaking.
3. **Drop `track_writes` and the direct-writes half of `what_modifies()`.** The open question said "drop it if the vignettes never use it"; they do not, and nor does anything else but one test. It is 40 lines of frame inspection in `arrays.py` for a question nobody has asked. Keep the declared-modifiers half, which is 30 lines and useful. Breaking only for the `track_writes=` keyword.
4. **One way to name a target.** The object form (`sir.pars.beta`, `sir.rel_sus`, `sir.recovery`) works for everything but a network's parameter, where the user is told to write a string, with an error message that names an `ss.constant`. Make network parameters addressable as objects (or accept that the string form is the serialization form, not a user-facing spelling) and remove the "two ways" from the spec. Medium cost.
5. **Shrink the lazy-condition promise to what is built.** Arithmetic on uninitialized arrays crashes with an `IndexError` and a float state (`sir.rel_sus`) is a `Target` before the sim exists, so cannot be compared at all. Comparisons, `&`, `|`, `~`, `isin` on states, people attributes, and dimensions are what works, and they are enough; the spec should list exactly those. Also document `sir.age == '15-24'` (a stratification reached through the module) as the way to write an age condition in a compartmental model instead of `ss.People(1).age < 5`.
6. **Make `ss.vaccinate` the one documented way to vaccinate, and fix the trap in the other.** The spec's `stratify(vaccinated=[False, True])` plus `ss.transfer` example puts half the population in `vaccinated=True` at the start (the split is guessed equal) and then reaches only the other half; `ss.vaccinate` gives a tenfold different answer for what reads as the same model. Either a dimension that is only ever entered by a transfer should default to everyone at the first level, or the transfer-with-delivery-keywords form should go.
7. **Keep follow-ups' names.** `confirm = ss.screen(pcr, ...)` passed as `on_positive=` is renamed `screening_positive`, and its own follow-up `screening_positive_positive`; the user's `sim.results.confirm` then does not exist. Products also appear in `sim.results` as modules with no results. Small, not breaking.

## Simpler for the user

### 1. Two intervention vocabularies, one of them unused

Today `import starsim as ss` exposes, for interventions and products: `Intervention`, `RoutineDelivery`, `CampaignDelivery`, `BaseTest`, `BaseScreening`, `routine_screening`, `campaign_screening`, `BaseTriage`, `routine_triage`, `campaign_triage`, `BaseTreatment`, `treat_num`, `BaseVaccination`, `routine_vx`, `campaign_vx`, `Product`, `Dx`, `Tx`, `Vx`, `simple_vx` (Starsim 3: 20 names) and `Delivery`, `vaccinate`, `treat`, `screen`, `Vaccine`, `Treatment`, `Test` (v4: 7 names), plus `modify`, `change`, `Cond`, `Connector`, `seasonality` (`starsim/__init__.py:108-130`). A user reading the API index sees both `ss.routine_vx(product=ss.simple_vx(efficacy=0.5), prob=0.5, start_year=2021)` and `ss.vaccinate(ss.Vaccine(efficacy=0.5), coverage=0.5, start=2021)` and has to learn which one they are in.

Where each is used (grep over the repo and the vignettes):

| Name | library | tests | docs | vignettes |
|---|---|---|---|---|
| `routine_vx` | 0 | 2 files (both Starsim 3 tests) | 5 | 0 |
| `campaign_vx`, `routine_triage`, `campaign_triage` | 0 | 0 | 2-3 | 0 |
| `routine_screening`, `campaign_screening`, `treat_num` | 0 | 1 | 3 | 0 |
| `Dx`, `Tx`, `Vx`, `simple_vx` | 0 | 2 | 1-4 | 0 |
| `vaccinate`, `Vaccine` | 0 | 1 (`test_deliveries.py`, 22 tests) | 0 | 2-3 |
| `modify` | 0 | 5 | 0 | 7 |

The library's own interventions (`ssl.ART` in `library/diseases/hiv.py:129`, `treat_pregnant` in `library/mnch/fetal_health.py:348`) subclass `ss.Intervention` directly and use neither vocabulary.

The two worlds are also closed to each other. From `s4_cascade.py`:

```text
=== 5. old API + new product mixed: routine_vx(product=ss.Vaccine(...))?
FAILS: KeyError 'No module named "None" with declared transitions was found; these are '
=== 6. new delivery on an old (code) disease: ss.vaccinate on ss.SIR?
FAILS: CodedError E506 UnknownDisease: "vaccinate" needs to know which disease it is for, and the sim has no declared diseases. Use disease=.
```

And the aliasing promised by the spec ("`ss.Vx`, `ss.Dx`, `ss.Tx` ... the short names remain as aliases", 08 line 53) is done with `__new__` dispatch on which keywords were passed (`products.py:37-40, 109-112, 173-176`), which makes `ss.Vx(efficacy=0.9)` a `Vaccine`, `ss.Vx()` a `Vx`, and `ss.Dx()` an `AttributeError: 'NoneType' object has no attribute 'state'` (`s4_cascade.py` section 7).

**Proposal.** Remove the 19 Starsim 3 names and keep `ss.Intervention` and `ss.Product` as base classes for custom code. Before removing, close the real gap: `ss.vaccinate`, `ss.screen`, and `ss.treat` should work on a code-written disease that has `rel_sus`/`rel_trans` and `infected` (that is, on an `ss.SIR`) in an agent-based run. `Delivery` already has an agent path for everything (`uids()`, `give()`, `test()`); what stops it is `init_pre` insisting on `sim.declared` (`interventions.py:725-732`) and the product's `init_pre` reading `dis.graph` (`products.py:367-369`). Keep the names `Vx`, `Dx`, `Tx` out entirely rather than as `__new__` aliases.

**Cost.** Breaking for `docs/tutorials/t6_interventions.qmd`, `docs/user_guide/modules_interventions.qmd`, `modules_products.qmd`, `basics_sim.qmd`, `solutions.qmd` (all teach `simple_vx`/`routine_vx`), `tests/test_interventions.py` (10 tests), and any downstream repo that imports `ss.routine_vx` (stisim, tbsim, hpvsim-style models). Those downstream models are the real cost; a migration note mapping `routine_vx(product=p, prob=0.5, start_year=2021)` to `vaccinate(p, coverage=0.5, start=2021)` covers most uses, but `treat_num`'s queue (people who accepted wait for capacity) has no equivalent: `capacity=` drops people who are not reached rather than queuing them. If queuing matters it is one keyword (`queue=True`) on `Delivery`, not a class.

**Confidence:** high that the duplication should go; medium on the order (make `vaccinate` work on code diseases first).

### 2. "What plus delivery" is right; the keyword set is wider than the spec says

The split holds up: `ss.Vaccine(efficacy=0.9, waning=ss.days(30), cost=1.5)` is reusable across deliveries and sims (`s2_vaccine.py` section 5: two sims with the same `Vaccine` object give identical results, no leaked state), and `ss.vaccinate(bcg, coverage=0.8, years=[10])` reads as the spec intends. `describe()` for a delivery is one good line:

```text
rounds: bcg: vaccine with efficacy against acquisition 0.9; protection lost after a mean of days(30); delivery in rounds; adds has_bcg as a property of people
```

Two things make the surface larger than the spec's table. First, `Delivery.__init__` takes `trace=`, `move=`, and `disease=` in addition to the twelve documented keywords (`interventions.py:568-569`); `trace=` (contact tracing) and `disease=` are useful and should be in 08's table; `move=` exists only so that `ss.transfer(vaccinated=True, coverage=...)` can be a `Delivery` (`transitions.py:2966`), which is finding 6. Second, `deliver=` is a function that "replaces `eligible=`, `coverage=`, and the round keywords" and errors if given with them (`interventions.py:585-587`), but `capacity=`, `when=`, `start=`, `stop=`, `on=` still apply. That is the right call (the function says who; the schedule says when), and the spec says so; it is just more to know than "keywords or one function".

**Proposal.** No structural change. Add `trace=` and `disease=` to the spec's table; remove `move=` by removing finding 6's transfer form. **Cost:** none. **Confidence:** high.

### 3. Reading a modified value: the code's answer is the better one

The spec (07 lines 76-110) promises three rules: reading gives the effective value, assigning sets the base, `.base` reads the base. The code does the opposite (`implementation.md`, departures): reading gives the base, `.effective` gives the modified value, and only as agents. `s1_modify.py` section 3:

```text
=== 3. reading a modified array: base vs .effective (ABM)
type: ModifiableArr
rel_sus[R]       = [1. 1. 1.]
rel_sus.effective[R] = [0.4 0.4 0.4]
=== 3b. the same in a compartmental run
type: Target | has effective? False
```

`arr.base` is `None` (`s6_misc.py` section 2), so the vocabulary table's `base` entry (17 line 117) names nothing.

Which is right? The spec's version has a real argument: Starsim 3 code that reads `sir.rel_sus[uids]` to compute something (a custom transmission step, an analyzer) silently ignores modifiers under the code's version. But the spec's version requires every read to go through a cache that is invalidated on every state change that any active condition depends on, and the spec itself needed a paragraph on when the cache "MUST be recomputed". The code's version has one array, no invalidation, and the modifiers are applied by the engine at the point of use (`solvers.py:3971, 3994` for agents; `mult_raw` for groups), which is also the only version that makes sense in a compartmental run where there is no per-person array at all. The one case the spec's version protects, custom code reading `rel_sus`, is exactly the code that the spec says should "use a modifier and flip a flag" instead (07 line 256).

**Proposal.** Keep the code's semantics and rewrite 07's "Base and effective values" to one paragraph: a modifier never touches the array; the engine applies active modifiers where it uses the quantity; `sir.rel_sus` is the module's own value. Remove `.base` from the vocabulary. Keep `.effective` only if `what_modifies()` stays (it is the way to check a modifier is doing what you think as agents); otherwise drop it with `ModifiableArr`. **Cost:** none to users (nothing used `.base`); the spec loses its nicest worked example (the `orig_rel_sus` paragraph) but that example's point, that nothing has to be restored, still holds. **Confidence:** medium-high.

### 4. `track_writes` and the direct-writes section of `what_modifies()`

`ss.Sim(track_writes=True)` makes `rel_sus` and `rel_trans` record which module assigned them, by walking the call stack from `__setitem__` until it finds a frame whose `self` is an `ss.Module` (`arrays.py:943-958`). `sim.what_modifies()` then prints `Halve.step() assigned 100 values (last at step 10) [file.py:LINE]`. It works (test `test_effective_and_writes`). It is used by that test and nothing else: zero vignettes, zero library modules, zero docs. Open question 77 in `19-open-questions.md` set the bar: "Drop it if the vignettes never use it."

The declared-modifiers half is different: it is available before a run, it is complete by construction, and the output is short and right (`s1_modify.py` section 4; `s2_vaccine.py` section 2 shows a product listed as `bcg multiply=0.0999... where=has_bcg (acquisition efficacy of vaccine)`, which is how a user would discover that a vaccine is a modifier).

**Proposal.** Drop `track_writes`, `ModifiableArr.writes`, `record()`, and the direct-writes branch in `sim.py:514-523`; keep `what_modifies()` for declared modifiers. Format the multiplier (`0.09999999999999998` should print as `0.1`). **Cost:** the keyword, one test, and one paragraph of 07 (which can keep its guidance "scale through `ss.modify` rather than overwrite" without the enforcement story). **Confidence:** high.

### 5. Two spellings of a target, and the one the spec leads with does not always work

The spec lists `school.pars.n_contacts` as an object target (07 line 53) and `'sir.pars.beta'` as "the same parameter as a string path ... for when the object is not in scope" (line 54-57). In the code, a network's parameters are plain values, so the object form fails with a misleading message, and the departures list says to use the string (`s1_modify.py` section 2):

```text
object FAILS: CodedError E504 UnmodifiableTarget: the target of ss.modify() must be a named quantity, such as sir.pars.beta, sir.rel_sus, sir.recovery, or "sir.pars.beta", not ss.constant(v=5) (constant). A plain number has no name ...
path OK: school.pars.n_contacts
```

The string path also needs `parse_target` to guess the kind from the path's pieces (`'pars' in parts`, last part in `['rel_sus', 'rel_trans']`, `connectors.py:134-137`), which is a second resolution scheme beside the object one, and it is resolved again by name at `init_pre` (`connectors.py:239-263`) because the sim works on copies.

Both spellings resolve to `(module name, kind, name)` at init, and the object form is only a convenient way to get those three strings (every object target is reduced to its name at `__init__`, `connectors.py:221`). So the two ways are not two mechanisms; the string is the canonical form and the object is sugar. The problem is only that the sugar does not cover networks.

**Proposal.** Make a network's `n_contacts` (and any parameter a user might modify) an `ss.par` with a `module` back-reference, as the declared diseases' `pars` already are (`diseases.py:98-100`), so the object form works everywhere. Then present one spelling in the spec and docs (the object), and mention the string form only as what `to_json()`/the model spec write. Alternatively, if making network pars into `ss.par` objects is too invasive for Starsim 3 networks, keep both but fix the error message to say "a network's parameter is a plain value; write it as a path, `'school.pars.n_contacts'`". **Cost:** the first is a change to `Network.define_pars`; the second is one line. **Confidence:** medium.

### 6. Conditions: the lazy form is narrower than the spec says, and both forms are needed

What works: comparisons on `ss.People(n).age`, `&`, `|`, `~` on states (`sir.S & ~sir.R`), `.isin`, a dimension by label (`sir.age == '15-24'`), and the same condition under `ode` and `abm` giving the same selection (`s5_conditions.py` section 2: lazy and lambda give identical results under both solvers). The E804 message when an age threshold cuts a bin is exactly the one the spec wrote. What does not (`s5_conditions.py` section 3):

```text
arith People(100) FAILS: IndexError index 0 is out of bounds for axis 0 with size 0
arith rel_sus FAILS: TypeError unsupported operand type(s) for *: 'Target' and 'int'
TypeError: '>' not supported between instances of 'Target' and 'float'
```

The spec (07 line 176) says arithmetic (`+ - * /`) on an uninitialized array is lazy; `Arr._math` has no lazy branch (only `_boolmath` and `__invert__` do, `arrays.py:453, 1033`), so `ppl.age - 5` indexes an empty array. And a float state before the sim exists is a `connectors.Target` (`diseases.py:97`), not an array, so `sir.rel_sus > 0.5` is a `TypeError`. The lazy form is therefore: comparisons and logic on people attributes, states, transitions (as events), and dimensions. That is enough for every condition in the spec and the vignettes.

Are both forms needed? Yes. The function form is the only general one and costs nothing. The lazy form earns its place not by being shorter (`sir.R` against `lambda sim: sim.diseases.sir.R` saves little) but because it is analyzable: `is_static()` (`conditions.py:142`) is what lets the kernels take a whole run with a condition in it, and the compartmental `Selector` can cross two diseases' states (`s5_conditions.py` section 5: `where=syph.I` on an HIV modifier runs as an ODE with 4 groups). A function of the sim can only be tried.

One verbosity point: a condition on age in a compartmental model has to be written `ss.People(1).age < 5` (the camdl vignette does this; `implementation.md` lists it as a known issue), because there is no `ppl` object. But a stratification is reachable from the module once declared: `test_dimensions.py:502` writes `where=sir.age == '15-24'`. That is the better spelling and should be the documented one.

**Proposal.** Rewrite 07's lazy rule to list exactly what is lazy (comparisons, `&`, `|`, `~`, `isin`) on exactly what (people attributes, states, transitions, dimensions); drop the arithmetic claim or implement it (B7 in 19 already doubted it). Document `sir.age == '15-24'` / `sir.age.isin([...])` for age in compartmental models. Make `Target` raise a clear error on comparison ("a float quantity cannot be a condition; compare a state or a stratification"). **Cost:** spec text; one `__gt__` on `Target`. **Confidence:** high.

### 7. Vaccination as a transfer: the spec's example vaccinates half the population at time zero

08 lines 37-42 give, as an alternative to `ss.vaccinate`:

```python
sir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.15]})
rollout = ss.transfer(vaccinated=True, coverage=0.7, years=[2021, 2022], eligible=ppl.age > 65)
```

Run against `ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.8, years=[10])` on the same SIS model (`s7_transfer.py`):

```text
disease  adjust={'infection': [1.0, 0.1]}: delivered=8000 eligible=10000 coverage=0.80  n_S by group at 11: [1934. 17710.]  n_I[11]=355 n_I[29]=160
--- vaccinate 0.8 efficacy 0.9:
n_I[11]=1319 n_I[29]=857
```

`stratify(vaccinated=[False, True])` splits the population equally between the two levels at the start (`describe()` prints `dimension vaccinated (plain): False, True [guessed: equal across levels]`), so 10,000 people are already "vaccinated" on day 0 and the transfer finds only 10,000 eligible. The two spellings the spec presents as alternatives differ by a factor of five in infections. The product path does this right because `Delivery.dims()` builds its `has_<product>` dimension with `split=[1.0, 0.0, ...]` (`interventions.py:672`).

**Proposal.** Pick `ss.vaccinate` as the one documented way (open question B6 asks for exactly this choice) and remove the delivery-keyword form of `ss.transfer` (`transitions.py:2938-2966`, `Delivery(move=...)`, `self.changes`/`self.sources`/`shift` for moves). If the transfer form stays, a dimension named in an `ss.transfer` with delivery keywords must default to `split=[1, 0, ...]`, and `describe()` should mark it. **Cost:** removing the form loses "vaccination as a move between strata" for models that want the compartments visible, but `has_<vaccine>` is a stratification too, and results can be selected by it. **Confidence:** high on the trap, medium on which form to keep.

### 8. Follow-up deliveries lose their names, and products show up as result-less modules

`s4_cascade.py` section 1:

```python
confirm   = ss.screen(pcr, coverage=0.9, on_positive=ss.treat(drug, coverage=0.8))
screening = ss.screen(rapid, coverage=0.4, interval=ss.days(7), start=20, on_positive=confirm)
```

```text
interventions: ['screening', 'screening_positive', 'screening_positive_positive']
confirm.name is now: screening_positive | results keys: ['timevec', 'screening', 'rapid', 'screening_positive', 'pcr', 'screening_positive_positive', 'drug', 'sis']
product result keys: ['timevec']
```

`Sim._add_followups` renames a follow-up `<parent>_<suffix>` when `_default_name` is still set (`sim.py:130-134`), and `_default_name` is still set because `_name_modules` only names the modules passed directly to the sim (`sim.py:109-120`), not the ones reached through `on_positive=`. The user wrote `confirm` and `sim.results.confirm` fails. The products `rapid`, `pcr`, `drug` are added to `sim.modules` (`sim.py:1391-1393`) so that they initialize and can be looked up by name, and so each gets an empty `Results` entry.

**Proposal.** Name follow-ups by variable too (walk `on_positive`/`on_negative` in `_name_modules`), falling back to `<parent>_positive` only for inline ones. Keep products out of `sim.results` (they have no results; `define_results` on a product could be skipped, or `Sim.results` could skip modules with none). **Cost:** none. **Confidence:** high.

### 9. Triggers: `when=` is fine, but the spec's own example does not run

`when=` as a bare threshold flickers as the spec warns: 28 on/off flips in 200 steps (`s3_trigger.py` section 1). The spec's remedy is a class with memory, and it gives one (08 lines 216-235). Run as written, it fails (`s3_trigger.py` section 2):

```text
FAILED: AttributeError modify object has no attribute "active"
```

Two things went wrong, and the message says neither. The class was not named `closure` (its default name is `schoolclosure`), so `sim.interventions.closure` raised `AttributeError` inside the `modify.active` property, and `Module.__getattr__` (`modules.py:300-307`) re-raised it as "no attribute `active`" (see Bugs). With `name='closure'` and `compartmental = True` it runs as an ODE and holds the closure for four weeks (section 2b). Without `compartmental = True`, any custom intervention class makes the whole sim agent-based (`implementation.md`, departures), which the spec's example does not mention.

The class is 15 lines for "close for four weeks once a threshold is passed". That is acceptable, and the spec's argument for not adding `min_duration=` is sound: a memory keyword would be a second place for logic that belongs in code. What is not acceptable is the two silent requirements (the name, the class attribute).

**Proposal.** Fix the error masking (Bugs 1). In 08's example, write `closure = SchoolClosure(name='closure')` or refer to the object (`when=lambda sim: closure.closed` does not survive copying; the spec should say why the name is needed). Add `compartmental = True` to the example with a comment. Consider inferring `compartmental` for an `Intervention` whose `step()` touches no per-agent array, but that is static analysis the spec rejects; the one-line attribute is fine if documented. **Cost:** spec text. **Confidence:** high.

### 10. Waning: the departure is the right call, and `describe()` already says it

The spec says a duration for `waning=` means "exponential decay with that mean"; the code makes it an exponentially distributed time at which protection is lost outright, under every solver (`products.py:388-402` as agents, `Delivery.dim_moves()` as a flow back for groups), and only a function of time fades the efficacy. `s2_vaccine.py` section 3 confirms: effective `rel_sus` is either `0.1` or `1.0`, and 2.7% still hold the vaccine at `t=110` against an expected `exp(-110/30) = 2.6%`. The two are different models of the population-average protection only in the second order, the all-or-nothing form is one flow in a compartmental run against ten stages for the fading form (`levels=[11]` in section 1), and the product's `describe()` line says "protection lost after a mean of days(30)", which is unambiguous. Keep it; update 08 line 57.

## Simpler codebase

### 1. 755 lines of Starsim 3 interventions and products

`interventions.py:1-529` and `products.py:1-226` are the classes in finding 1. They have their own eligibility protocol (`check_eligibility`, three versions: `interventions.py:56, 237, 291`), their own year handling with a `TODO: Refactor to be more agnostic about the types` (`interventions.py:116`), their own results (`n_screened`, `n_dx`), and dataframe-driven products (`Dx(df=...)` with `hierarchy`). None of it is shared with `Delivery`. Removing it also removes the three `__new__` alias methods. See finding 1 for the cost.

### 2. `Delivery` is one 730-line class doing six jobs

`Delivery` (`interventions.py:535-1220`) handles: the schedule (`due`, rounds, routine with an `offered_<name>` dimension); eligibility, coverage, and capacity (`offer`, `receive`, `cap`); giving each of three product kinds and a transfer (`give`); tests with delayed results, as agents (`ti_result`) and as groups (a `pending_<name>` dimension with `2m+1` levels and `move_up` every step: `dims`, `step_results`, `test`, lines 673-676, 1143-1149, 1157-1168); staged waning for groups (`step_stages`, `move_up`); contact tracing (`trace`); and the kernel path (`tabulated`, `shifts`, `store_shifts`, `many_shifts`, lines 992-1054). It is long because the design covers two solvers and a cascade, not because the code is careless; most methods are short and named well. Two pieces could go or move:

- The compartmental realization of a delayed test result (about 60 lines across `dims`, `test`, `step_results`, `result_steps`, `pending`) exists so that `ss.Test(delay=ss.days(2))` runs as an ODE. A delay on a test changes when a follow-up happens by a few steps; as groups it adds `2m+1` compartments per test to the model. Making `delay=` agents-only (an `agent_reasons()` entry, which is the established mechanism) would remove the code and the `pending_*` dimensions from `describe()`. Judgement call; cost is one feature in one solver.
- The transfer path (`move=`, `self.changes`, `self.sources`, the arrow parsing in `init_pre` lines 757-764, the `product is None` branches) goes with finding 7.

### 3. `Effect` is a modifier written twice

`products.Effect` (`products.py:229-268`) has `op`, `active`, `tabulated`, `apply(value, engine)`, `describe()`, and `name`, so that `connectors.combine()` can treat it like an `ss.modify`. It is a second implementation of the modifier protocol by duck typing, and it carries its own waning logic (`apply`, lines 259-263) and its own `describe`. A product's efficacy could be an actual `ss.modify(dis.rel_sus, multiply=<fn or value>, where=<has_product dim>)` appended to `sim.modifiers` by `Held.init_pre`, with the waning factor as the amount (a function of the sim returning per-agent values is already supported by `Selector.per_person`). `combine()`'s `mod.op == 'set'` loop and `what_modifies()` then have one type to know about. **Cost:** `Effect.constant`/`tabulated` would become properties of `modify` that look at the amount, which they nearly are already (`connectors.py:265-275`). **Confidence:** medium.

### 4. `parse_target` and `Target`

`parse_target` (`connectors.py:122-138`) accepts five kinds of object and a string, and for the string guesses the kind from the path's pieces; `modify.init_pre` (lines 239-263) then redoes the resolution against the built sim with a second set of rules (networks, `graph.modifier_kind`). `Target` (lines 110-119) exists only to give `sir.rel_sus` a name before the sim exists. If every modifiable quantity were an object with `.module` and `.name` (finding 5), `parse_target` would be four lines and `Target` could be a `ModifiableArr` that is simply not initialized yet, which would also make `sir.rel_sus > 0.5` a lazy condition for free.

### 5. `seasonality` does what the spec says not to do

`ss.seasonality` (`connectors.py:24-106`) writes `disease.rel_trans[:] = rel_beta` every step, overwriting the base value of every agent, which is the exact pattern 07 line 106 and 256 argue against and which `what_modifies()` was built to catch. With `ss.seasonal` (`transitions.py`, a `TimeValue`) in the language, the same thing is `ss.modify(sis.rel_trans, multiply=ss.seasonal(mean=1.0, amplitude=0.2, peak=...))`, one line, combining correctly with other modifiers and working under every solver. Nothing in the repo uses `ss.seasonality` (zero library, test, docs, or vignette hits). Delete it, or reduce it to a one-line factory returning that modifier. **Confidence:** high.

### 6. Three "initialize once" guards

`Product.init_pre` (`products.py:19-24`), `Held.init_pre` (`359-362`, a `linked` flag), and `Test.init_pre` (`521-525`, the same flag) all exist because a product is initialized both by the sim (it is in `sim.modules`) and by each delivery that carries it (`Intervention.init_pre`, `interventions.py:34-38`). If products are not modules of the sim (finding 8) or deliveries do not initialize them, the guards go.

### 7. Three-way time conversion in `connectors.to_time`

`to_time`/`to_ti` (`connectors.py:141-161`) turn a date, a year, a duration, or a bare number into the sim's time, and `Delivery.init_pre` reuses them. `to_ti` is a linear scan over `tvec` per call (`for ti,now in enumerate(sim.t.tvec)`), and `modify.active` calls `to_time` twice on every evaluation (lines 282-283) rather than once at init. Not a correctness problem; it is the kind of thing that belongs on `Timeline` (`sim.t.index(time)`), which other reviewers may be looking at.

## Spec and implementation

Beyond the departures `implementation.md` lists:

- **Lazy arithmetic is not implemented** (07 line 176 lists `+ - * /` as lazy); `Arr._math` has no lazy branch. Finding 6, Bugs 3.
- **A float state is not an array before the sim exists.** 07 line 175 says `hiv.cd4` is an uninitialized array on which comparisons are lazy; in a declared disease the only per-person float quantities are `rel_sus`/`rel_trans`, and before the sim exists they are `Target` objects (`diseases.py:97`), not arrays, so no comparison on them is lazy (or possible). Finding 6.
- **`when=` is not called "at the modifier's place in the loop."** 07 line 275 says a `when=` or `where=` function is called once per step at the modifier's place in the loop; `modify.step()` is `pass` (`connectors.py:325-326`) and `active`/`apply` run inside `combine()` when the engine first reads the quantity in a step, cached until `start_step()`/`step()` clears it (`solvers.py:1400, 2855`). The open question at 19 line 79 (evaluated on read, or at the modifier's place) is thereby answered: on first read in the step, once per step. The difference is observable only for code that changes a state between the modifier's slot and the transmission step, and the code's behavior (see the current state) is the more useful one. The spec should say this and drop the loop-position story.
- **`connectors=` versus `interventions=` for a modifier makes no difference** (`s1_modify.py` section 8: identical results), consistent with the above. The spec's "the two lists differ only in where the condition is evaluated" (07 line 34) is therefore not true either; they differ in nothing. Say that, or drop one.
- **`.base` does not exist**; 17 line 117 lists it. Finding 3.
- **Follow-up naming** is not specified anywhere and the code's rule surprises. Finding 8.
- **Delivery keywords** `trace=`, `disease=` are not in 08's table; `move=` is an implementation detail that leaks. Finding 2.
- **`sim.results.sir.new_infections`** in the spec's `SchoolClosure` example does exist (as an alias of the `infection` flow), so that part of the example is fine.
- **Routine delivery (19 line 81)** is implemented as "offered once at first eligibility" by adding an `offered_<delivery>` dimension (so compartments) whenever coverage or capacity is less than full (`Delivery.remembers`). The alternative, a rate, would need no dimension; the trade is explicit now and worth a line in 08.
- **Two `set=` modifiers (19 line 78)**: implemented as the warning, said once (`connectors.py:166-170`). Settled.
- **`what_modifies()` (19 line 77)**: no vignette uses it. Settled by the open question's own criterion: drop the tracking half.
- **Where modifiers are clipped (07 line 72)**: implemented with `W503` and recorded in `sim.clipped`, reported by `what_modifies()`. Settled, and good.
- **Open question B7 (`on=` enough for event-triggered logic)**: the vignettes and tests use `on=sir.infection` and `on=tb.diagnosis` and a function follow-up; nothing has needed "at this person's next visit". Leaning yes.

## Bugs

### 1. `Module.__getattr__` masks errors raised inside properties

Any `AttributeError` raised while computing a property on a module is re-raised as "object has no attribute `<property>`" (`modules.py:300-307`), hiding the real error. Repro (`s6_misc.py` section 1):

```python
effect = ss.modify(sis.pars.beta, multiply=0.2, when=lambda sim: sim.interventions.closure.closed)  # no module named 'closure'
ss.Sim(sis, interventions=effect, solver='ode', **kw).run()
```

```text
AttributeError : modify object has no attribute "active"
  File ".../starsim/connectors.py", line 166, in combine
    active = [mod for mod in mods if mod.active]
  File ".../starsim/modules.py", line 307, in __getattr__
    raise AttributeError(f'{type(self).__name__} object has no attribute "{attr}"')
```

The user's mistake is a wrong module name in a `when=` lambda; the message points at `modify.active`. Fix: in `__getattr__`, check whether `attr` is a property on the class and, if so, re-raise the original exception (or compute it with `object.__getattribute__` and let it propagate). This will bite every `when=`/`where=`/`deliver=` function that has a typo. (Outside my files, but every trigger in my area goes through it.)

### 2. A dimension used as a vaccination target starts half full

Finding 7: `make().stratify(vaccinated=[False, True])` plus `ss.transfer(vaccinated=True, coverage=0.8, years=[10])` reports `eligible=10000` of 20,000 at the round and `n_S by group at 11: [1934, 17710]`, because the split is guessed equal. Repro: `s7_transfer.py`. The spec's own example (08 lines 39-41) has this behavior.

### 3. Arithmetic on an uninitialized people array raises `IndexError`

```python
ss.People(100).age - 5
```

```text
IndexError: index 0 is out of bounds for axis 0 with size 0   (arrays.py:566 in _math -> _index)
```

The spec says this is lazy. Either implement the lazy branch in `_math` as in `_boolmath` (`arrays.py:453`) or raise a clear error. Repro: `s5_conditions.py` section 3.

### 4. `ss.Dx()` with no arguments crashes

```text
Dx() FAILS: AttributeError 'NoneType' object has no attribute 'state'
```

`Dx.__new__` returns a `Test` only when `kwargs` is non-empty and has no `df`; with no kwargs it falls through to `Dx.__init__(df=None)`, which does `df.state.unique()` (`products.py:37-45`). Goes away with finding 1. Repro: `s4_cascade.py` section 7.

### 5. Two inline products of one kind warn about renaming

Two `ss.Test(...)` created inline in one sim are both named `test`, and the sim renames the second `test_1` with a `RuntimeWarning` (`parameters.py:490`), seen in `s4_cascade.py` section 2. A product that was never named should take its delivery's name (`<delivery>_test`) rather than its class name, or not be registered as a module.

## Keep

- **`ss.modify` itself**: one target, one of `multiply`/`add`/`set`, `where`, `start`/`stop`, `when`. The combination rule (set, then multiply, then add; order-independent) is stated once and implemented once (`connectors.py:164-178`); `W502` for overlapping sets and `W503` with clipping are the right level of policing. Open question 19 line 208 ("`ss.modify` does a lot of work ... should stay small") has been honored: it is 150 lines.
- **The base is never overwritten.** `test_triggers` asserts `sir.pars.beta.value == 0.3` after a run with a modifier; `s1_modify.py` sections 1 and 3 confirm. Whatever is decided about `.effective`, this property is the one to protect.
- **Error messages with the closest names.** `E504 ... Quantities that can be modified: sir.pars.beta, sir.pars.dur, sir.infection, sir.recovery, sir.rel_sus, sir.rel_trans; did you mean sir.pars.beta ...` and `E804 ... add 5 as a bin edge, stratify(age=[0, 5, 15, 65]), or run with solver="abm"` are as good as the spec promised.
- **Conditions evaluated on groups, with crossing.** `where=syph.I` on an HIV modifier runs as an ODE by crossing the two diseases' states (`s5_conditions.py` section 5), and the same source gives the same selection as agents. This is the payoff of the lazy form and should not be traded for "functions only".
- **`describe()` for products and deliveries**, one line each, naming the dimension the delivery adds.
- **Delivery results**: `delivered`, `eligible`, `coverage`, `cost`, `diagnosed`, by stratum, under every solver; and a follow-up being a module with its own results, so loss to follow-up is visible as `screening_positive.delivered / screening.diagnosed`.
- **`efficacy=dict(acquisition=..., transmission=..., mortality=..., <transition>=...)`** with an `E505` listing the valid keys. One spelling for a vaccine that does several things, and a treatment that changes rates (`art = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))`) is the same object.
- **`tests/test_deliveries.py`** runs every scenario under `ode`, `binomial`, and `abm` and asserts they agree (`check_close`). It is the conformance test for this whole area and found none of the bugs above only because they are at the edges it does not touch (naming, transfer splits, Starsim 3 classes).
- **The decision to reject `min_duration=`** and keep memory in `step()`. The 15-line class is honest about what a policy with memory is.

## Outside my area

- `Module.__getattr__` masking property errors (Bugs 1) is in `modules.py` and affects every property on every module.
- The guessed-equal default `split` on a stratification (Bugs 2) is a dimensions (05) question: a dimension that nothing initializes should perhaps default to the first level, with "equal across levels" reserved for dimensions that are explicitly populated.
- `sim.results` lists every module including ones with no results (products here), a results (09) question.
- `modify.active` calls `connectors.to_time` twice per evaluation and `to_ti` is a linear scan; a `Timeline.index(time)` would serve both this and `Delivery.init_pre` (time, 02/10).
- `implementation.md` notes that as agents a delivery runs after the step's progressions but before transmission while as groups it runs before all flows, so agent-based deliveries miss people who progress in that step (7% for TPT). That is execution order (10), but it is the largest solver disagreement an intervention user will see.
