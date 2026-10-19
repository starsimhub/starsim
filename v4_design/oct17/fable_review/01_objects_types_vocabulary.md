# Objects, types, names, and the total vocabulary

I reviewed spec sections `00-overview.md`, `01-objects.md` (353 lines), `02-types.md` (200), `17-vocabulary.md` (284), and `19-open-questions.md`, against `starsim/__init__.py` (220 lines), `starsim/time.py` (2612), `starsim/named.py` (184), `starsim/parameters.py` (661), and the naming, `ss.par`, `Expr`, `CodedError`/`note`, and `Graph.collect`/`resolve_*`/`describe` parts of `starsim/transitions.py` (lines 1-260, 428-620, 1075-1575, 1927-2010), plus `Sim.init_unit`/`Sim.describe` in `sim.py` (lines 300-400, 1026-1140) and `Disease.__init__`/`init_graph`/`set_pars` in `diseases.py` (lines 18-215). I ran seven scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/objects/` (`s1_describe.py` to `s7_setbeta.py`, outputs in the matching `.out` files): `describe()` on the overview example and on the MEmilio, summer2, and Atomica vignettes; sixteen variations of names-from-variables; twenty-five variations of the unit rules; every row of the arithmetic table in 02; and the ways of setting a parameter. State/Transition semantics and the `@` operator are another reviewer's area and are not covered here.

## Summary

1. **The public surface is 253 `ss.` names, not the 94 that 17-vocabulary.md counts.** About 60 are v4 internals (`ss.Graph`, `ss.Engine`, `ss.Crossing`, `ss.CountState`, `ss.Strat`, `ss.Cond`, `ss.Expr`, `ss.Fit`, `ss.ParticleFilter`, `ss.make_engine`, ...) and about 40 are Starsim 3 aliases and base classes that now duplicate a v4 name (`ss.Vx`/`ss.Vaccine`, `ss.routine_vx`/`ss.vaccinate`, `ss.rate`/`ss.freq`, `ss.seasonality`/`ss.seasonal`). Move internals into their submodules and drop the duplicates. Breaking for Starsim 3 code that uses the aliases; a one-line alias shim can soften it.
2. **Assigning to `sir.pars.beta` silently destroys the parameter.** `sir.pars.beta = 0.1` replaces the `ss.par` with a float, so bounds and units are lost and `ss.modify(sir.pars.beta)` then fails with `E504 ... not 0.6 (float)`. `sir.pars.update(beta=0.1)` warns "No known mechanism for handling par → float". Make `Pars.__setitem__` set `.value` when the entry is an `ss.par`, as `set_pars` already does. Not breaking.
3. **Drop `ss.states(3)` and keep `ss.states('S I R')`.** States are the one object with no fallback name, so the frame lookup raises `E210` whenever states are built in one function and the module in another, or a state has an alias (`Inf = I`). Six characters buy a sharp edge and the only user-visible frame magic that can fail hard. Breaking for the spec's examples and tests.
4. **The time arithmetic the code does is not the arithmetic 02 specifies.** `ss.perday(0.1) * ss.days(3)` is 0.259 (a probability), where 02's table says 0.3 (an expected count); `2 / ss.perday(0.1)` raises an empty `NotImplementedError` where 02 says a duration; `E101` is a plain `TypeError`; `E102 WrongDimension` is not implemented, so `rate=ss.peryear(0.1) * ss.days(3)` is accepted as a rate. Write the lattice once (02's own open question) and make spec and code agree; for `per × dur` the cheaper fix is the spec's table, since 46 call sites rely on the code's meaning.
5. **"Time in timesteps" is days under the hood, and typed values in code modules do not count.** `ss.Sim([ss.SIS(), sir], stop=100)` reports `unit=None` and converts `ss.SIS`'s `beta=ss.peryear(0.05)` with `dt=days(1)` to 0.000137 per step, with no note. Extend rule 4 of 02 to every module's typed values, or refuse the mix. Also `stop=2000` alone infers years and a start of 1950.
6. **Trim the time constructors from 17 to 9.** `probper*` and `freqper*` (8 names) are used in two network defaults and nowhere in v4; `ss.prob(0.1, 'day')` and `ss.freq(80, 'year')` already spell them. Breaking (about 20 uses in tests and docs).
7. **`describe()` is good and should be kept as is; it lacks `verbose=True`, which 01, 02, and 10 all promise**, and it says `beta = 2 per step` in a sim whose step is `dt=0.1`. Small, not breaking.

## Simpler for the user

### 1. The vocabulary is 253 names, not 94

**Issue.** `17-vocabulary.md` counts 94 `ss.` names and says the count exists "so that the size of the language is something that can be looked at". `dir(ss)` after `import starsim as ss` has 285 public names: 32 submodules, 179 classes, 65 functions, 9 other (`s5_timetypes.py`, and the count run in the shell). Of the 253 non-module names, roughly:

- **The 94 in the spec**, less the ones that do not exist (`ss.func`; see finding 7).
- **About 60 v4 internals** that a user never types: `Graph`, `Strat`, `Transfer`, `Engine`, `Compartmental`, `Agents`, `CountState`, `Crossing`, `Mixing`, `make_engine`, `solver_names`, `Cond`, `Expr`, `CodedError`, `error`, `note`, `MultiResult`, `MultiResults`, `Fit`, `fit`, `ParticleFilter`, `ModelComparison`, `mismatch`, `loglik`, `compare`, `from_spec`, `to_code`, `diff_sims`, `check_sims_match`, `demo`, `Delivery`, `CalibComponent`, `BetaBinomial`, `Binomial`, `DirichletMultinomial`, `GammaPoisson`, `Normal`, `linear_interp`, `linear_accum`, `step_containing`, `link_dists`, `make_dist`, `dist_list`, `scale_types`, `Dists`, `module_map`, `module_types`, `register_modules`, `find_modules`, `required`, `standardize_data`, `standardize_netkey`, `validate_sim_data`, `plot_args`, `show`, `return_fig`, `shrink`, `apply_age_range`, `parse_age_range`, `find_contacts`, `load_fonts`, `style`, `dtypes`, `root`, `Dataset`, `Samples`, `Person`, `AgeGroup`, `InfectionLog`, `PregnancyPars`. `17-vocabulary.md:211` lists `ss.Expr` under "Removed since the first draft", yet `__init__.py:134` exports it.
- **About 40 Starsim 3 names that now have a v4 twin**: `Vx`/`Dx`/`Tx`/`simple_vx` beside `Vaccine`/`Test`/`Treatment`; `routine_vx`, `campaign_vx`, `routine_screening`, `campaign_screening`, `routine_triage`, `campaign_triage`, `treat_num`, `BaseTest`, `BaseScreening`, `BaseTriage`, `BaseTreatment`, `BaseVaccination`, `RoutineDelivery`, `CampaignDelivery` beside `vaccinate`/`screen`/`treat`; `rate`, `time_prob`, `rate_prob` (deprecated since Starsim 2, each prints a six-line warning; `time.py:2384-2414`) beside `freq`/`prob`/`per`; `seasonality` (a Starsim 3 connector, `connectors.py:24`) beside `seasonal` (a v4 parameter form, `transitions.py:2369`); `Route` beside `Network`; `year`/`month`/`week`/`day` instances beside `years`/`months`/`weeks`/`days` classes; `Infection` beside `Disease`; the debugging helpers `mock_sim`, `mock_people`, `mock_module`, `mock_time`, `Profile`, `Debugger`, `Diagnostics`, `check_requires`, `check_version`, `metadata`.

**Why it matters.** Tab completion on `ss.` is the way most people discover the API. With 253 entries, `ss.s<tab>` offers `save, scale_types, screen, seasonal, seasonality, show, shrink, simple_vx, single_run, solver_names, standardize_data, standardize_netkey, states, step_containing, stratify, style`. The spec's principle ("a vocabulary nobody can hold in their head is a vocabulary nobody uses correctly") is defeated at the namespace, whatever the document says. `ss.seasonal` versus `ss.seasonality` is the worst case: two lowercase names, both about seasonality, one a parameter form and one a module.

**Proposal.**
- Keep internals importable but off the top level: `ss.solvers.Engine`, `ss.calibration.ParticleFilter`, `ss.transitions.Graph`, `ss.debugtools.mock_sim`. Submodules are already exported, so nothing becomes unreachable.
- Remove the deprecated trio `ss.rate`, `ss.time_prob`, `ss.rate_prob` (no uses in `starsim/` or `tests/`; the only mentions in `docs/` are the v2-to-v3 migration guide, which is about removing them).
- Rename or remove `ss.seasonality` (make it a thin wrapper around `ss.modify(..., multiply=ss.seasonal(...))` in the library, or drop it).
- Decide for each legacy intervention name whether it is an alias of the v4 one (then make it a real alias and do not export it) or a different thing (then it needs a line in 17). The vocabulary document should count what `dir(ss)` shows, and a test should assert the two agree.

**Cost.** Breaking for Starsim 3 scripts that use `ss.routine_vx`, `ss.Vx`, `ss.rate`, and friends; an alias module (`starsim/legacy.py`) imported on demand would keep them working with one warning. Medium effort; the exports are all in one file.

**Confidence.** High on the count and the duplicates; medium on exactly which legacy classes can go (another reviewer covers interventions).

### 2. Assigning to `sir.pars.beta` silently destroys the parameter

**Issue.** For a declared module, `sir.pars` is an `ss.Pars` whose values are the `ss.par` objects themselves (`diseases.py:98-100` writes the values with `define_pars`, then overwrites each entry with the `par` via `dict.__setitem__` to bypass the type dispatch). Starsim's idiom for changing a parameter, `sir.pars.beta = 0.1`, goes through `sc.objdict.__setattr__` and replaces the object with a float. From `s6_followups.py` and `s7_setbeta.py`:

```text
after attr_assign a.pars.beta = 0.1 float
bounds after sir.pars.beta = 0.2: LOST (now a float)
typed par after sir.pars.beta = 0.1: 0.1 (unit lost silently?)
pars.update warns: ["No known mechanism for handling <class 'starsim.transitions.par'> → <class 'float'>"]
modify after attr assign ERROR: CodedError E504 UnmodifiableTarget: the target of ss.modify() must be a named quantity, such as sir.pars.beta, ... not 0.6 (float).
```

The run still uses the new value, because `par.raw` (`transitions.py:218-224`) reads back through `mod.pars[self.name]` and unwraps whatever it finds. So the five ways of setting beta (`sir.pars.beta = x`, `sir.pars.beta.value = x`, `sir.set_pars(beta=x)`, `sir.pars.update(beta=x)`, `ss.Sim(sir, pars=dict(beta=x))`) all give the same `n_R[-1] = 9901.4`, but three of them have different side effects: the first drops bounds and unit, the fourth warns, and the first breaks `ss.modify`. The `E504` message is also wrong in a way the user cannot act on: it says to use `sir.pars.beta`, which is exactly what they did.

**Why it matters.** `sir.pars.x = value` is the one parameter idiom every Starsim user knows, the overview uses `sir.pars.beta` as the target of `ss.modify`, and `04-parameters.md` makes bounds the thing that decides what `calibrate()` fits. A user who sets a starting value by assignment before calibrating loses the bounds without a word.

**Proposal.** In `Pars.__setitem__` (`parameters.py:73-77`): if the existing entry is an `ss.par` and the new value is not, set `.value` and return. Then delete `Disease.set_pars` (`diseases.py:105-115`), which exists only to do this, and the `par`-to-float branch becomes unreachable in `Pars.update`. Also let `par.raw` stop reading through `mod.pars` (`transitions.py:218-224`); with the entry always the same object, `self.value` is enough.

**Cost.** None to users; removes code. The one behavior that changes is that `sir.pars.beta = ss.par(0.2, bounds=...)` would need to replace the object, which the `isinstance(new, par)` check keeps.

**Confidence.** High.

### 3. Two parameter systems: `ss.par` entries and Starsim 3 entries in the same `sir.pars`

**Issue.** `sir.pars` holds `ss.par` objects for a declared module and `ss.peryear`/`ss.Dist` objects for a code module:

```text
sir.pars: #0. 'dur': par(days(10))  #1. 'beta': par(0.05, bounds=[0.01, 0.5])
ss.SIR pars types: {'init_prev': 'bernoulli', 'beta': 'peryear', 'dur_inf': 'lognorm_ex', 'p_death': 'bernoulli'}
```

So `sim.diseases.sir.pars.beta.value` is right for one and `sim.diseases.sir.pars.beta.value` is the raw number without its unit for the other; `.bounds` exists for one; `ss.modify('sir.pars.beta')` works for both but by different routes; and a bare number keeps "the unit of its default" in the second (`parameters.py:38-60`, the `_bare` bookkeeping and its warning) and the sim's unit in the first (`02-types.md`). `implementation.md` lists this as a known issue; I think it is a design fork rather than an issue, because the two kinds of module will coexist for years.

**Why it matters.** The user has to know which kind of module they are holding before they know what `pars.x` is. The spec's answer, "`ss.par` when it needs a name, bounds, or a source", is fine for writing a model; the asymmetry is in reading one back.

**Proposal.** Either (a) make `ss.par` the one parameter wrapper for both kinds, so `ss.SIR(beta=0.05)` also stores `par(peryear(0.05))` and `.bounds`/`.source` exist everywhere (the `Pars.update` dispatch on `TimePar`/`Dist` moves into `par`), or (b) say plainly in 04 that code modules keep Starsim 3's parameter objects and `ss.par` is for declared models, and make `sir.pars.beta` on a declared module return the *value* (number, `TimePar`, or `Dist`), with the `par` reachable as `sir.beta` (the state and transition attributes already live there, `diseases.py:90-91`). Option (b) is smaller and makes `sir.pars` look the same for both kinds; it needs `ss.modify` to accept `sir.beta` as a target, which it does for transitions already.

**Cost.** (a) touches every Starsim 3 module's `define_pars`; (b) changes `sir.pars.beta` from the `par` to its value, which the spec's `ss.modify(sir.pars.beta, ...)` example relies on, so the spec and the modifier target rules move too. Either is breaking for v4 code written so far; nothing in Starsim 3 breaks.

**Confidence.** Medium. The asymmetry is certain; which way to close it is a judgment.

### 4. Bare numbers bound to variables are not parameters; only `ss.par` is

**Issue.** `01-objects.md:75-81` shows the operator form as

```python
beta = 0.05
dur  = ss.days(10)
infection = S >> I @ beta * I
recovery  = I >> R @ dur
```

and `describe()` then shows `beta = 0.05 per day`, which looks as if the variable name was used. It was not. `resolve_pars` (`transitions.py:1393-1427`) names an anonymous literal after its *slot*; the names agree here by coincidence. From `s6_followups.py`:

```text
rate from variable gamma                 -> pars ['beta', 'rate']      # gamma = 0.1; I >> R @ gamma
gamma as ss.par                          -> pars ['beta', 'gamma']
beta=0.05; contacts=ss.timeseries(...)   -> pars ['dur']               # S >> I @ beta * contacts * I: neither is a parameter at all
```

In the MEmilio vignette, `contacts = ss.timeseries({0: 1.0, 29: 0.55, 30: 0.1})` is reported by `describe()` as `beta = timeseries(...)`, and in the third case two numbers that the user clearly meant as parameters are invisible to `pars`, `calibrate`, and `ss.modify`, because `beta * contacts` is a `BinOp` and only a lone `par` coefficient is recognized as the beta slot (`transitions.py:1407-1415`).

**Why it matters.** The rule the spec states is simple (a number is anonymous, `ss.par` has a name), but the spec's own example and the slot-naming heuristic make it look like variable names are picked up, so the user learns the wrong rule and is surprised later. The heuristic also produces `rate_S_to_I`, `dur_E1_to_E2` when it has to disambiguate (`s2_names.out`, cases 4 and 13), which are names nobody chose.

**Proposal.** Pick one of two consistent rules and say it in 01 and 04. (i) *Only `ss.par` is a parameter*: a literal in a rate stays a literal, is printed as its value in `describe()`, and is not in `sir.pars`; the slot-naming code goes. The two-line short form then has no `sir.pars` at all, which is honest, and `ss.par(0.05)` is the price of a name. (ii) *Every number in a standard slot is a parameter*: keep the slot names, but then apply the same lookup to products of literals, and change the 01 example to `beta = ss.par(0.05)` so nobody infers variable-name capture. I lean to (i): it deletes `resolve_pars`'s heuristics and removes "literal" as a concept the user meets (`par.literal` is set or tested in twelve places in `transitions.py`).

**Cost.** (i) breaks `ss.transmission('S -> I', beta=0.05)` followed by `ss.modify(sir.pars.beta, ...)` and `calibrate` on the short form; users write `beta=ss.par(0.05)` instead. The constructor keywords `beta=`/`dur=`/`rate=` could keep registering a `par` (they are explicitly named by the keyword, so there is no guess), which preserves the short form; only inline arithmetic literals would be anonymous.

**Confidence.** Medium-high.

### 5. States should carry their names; drop `ss.states(3)`

**Issue.** Names from variables work in more places than I expected (`s2_names.out`): module scope, inside a function, in a `transitions=[...]` list, in a class body, under `exec`, and for the module name at `ss.Sim`. They fail in two common patterns, and only states fail hard:

```text
6 made in one fn, Disease in another   ERROR: E210 UnnamedState: state number 1 of the module has no name. It is used in ['<unnamed state> -> <unnamed state>'], and no variable or keyword holding it was found.
7 two names for one transition           transitions=['recovery', 'S_to_I']      # infection also bound to inf2: silent fallback
8 two names for one state   ERROR: E210 UnnamedState: state number 2 ...         # Inf = I
```

Case 6 is a factory (`def ingredients(): S, I, R = ss.states(3); ...; return [infection, recovery]` built by `ss.Disease(*ingredients())` in another function), which is how tests, notebooks, and any model-generating code are written. The error names the state by position and shows the transitions as `<unnamed state> -> <unnamed state>`, so there is nothing to search for. A transition in the same situation falls back to `S_to_I` and `describe()` marks it `[built from its endpoints]`, which is the right behavior.

**Why it matters.** `19-open-questions.md` B12 asks exactly this ("how often that happens ... is to be found out"). The answer is: whenever construction is split across frames, and states are the only object for which the answer is an error.

**Proposal.** Remove the integer form of `ss.states` and keep `ss.states('S I R')` and `ss.State('S')`; in a class body, `S = ss.State()` keeps learning its name via `__set_name__` (`transitions.py:465-469`), which is Python's mechanism and never misses. Then `resolve_states` no longer needs `find_name`, `E210` goes, and the `frame=` argument is used only for transitions and modules, both of which have fallbacks. Keep the lookup for transitions: the fallback makes it safe, and the overview's `sir = ss.Disease(infection, recovery)` reads well.

**Cost.** Six characters per model. Breaking for the spec's examples (`00`, `01`, `03`), the docstrings, and whichever tests use `ss.states(3)`. The equivalence rule in 01 is unaffected.

**Confidence.** High.

### 6. Time types: the arithmetic the code does is not the table in 02

**Issue.** Running every row of the table in `02-types.md:55-67` (`s5_timetypes.out`):

| Expression | 02 says | Code does |
|---|---|---|
| `ss.perday(0.1) * ss.days(3)` | a number: the expected count (0.3) | `0.2592`, the probability `1 - exp(-0.3)` (`time.py:2203-2204`, `per.__mul__ = to_prob`) |
| `ss.freqperday(0.1) * ss.days(3)` | an expected count | `0.3` |
| `1 / ss.days(10)` | "a rate" | `freq(1/10)`, a frequency, not a hazard (`time.py:1234`) |
| `2 / ss.perday(0.1)` | a `dur` | `NotImplementedError` with an empty message (`time.py:2211-2212`) |
| `2 / ss.freqperday(0.1)` | | `days(20)` |
| `ss.perday(0.1) + ss.freqperday(0.1)` | `E101 IncompatibleTypes` | plain `TypeError`, not an `ss.CodedError` |
| `ss.perday(0.1) + ss.peryear(36.5)` | `per` | `perday(0.2)` |
| `ss.perday(0.1) - ss.peryear(36.5)` | `per` | `TypeError: Can only subtract rates of identical types` (`time.py:1794`) |
| `ss.perday(0.1) / ss.days(2)` | | `NotImplementedError` (empty) for `per`, `Exception('Cannot divide a rate by a duration')` for `freq` |
| `rate=ss.peryear(0.1) * ss.days(3)` in a slot | `E102 WrongDimension` | accepted; the transition runs at `0.00082` per step |

Also: `ss.dur(3)`, `ss.per(0.1)`, `ss.freq(2)`, and `ss.Rate(2)` default to years with a warning (`time.py:1078-1082`, `1650-1653`), which contradicts v4's "no default unit"; `ss.prob(0.3)` is a unitless probability that errors when converted; `ss.years(1) == ss.days(365)` is `True`.

**Why it matters.** The first row is the one that matters: `per × dur` is the textbook "rate times time", and a user who writes `ss.perday(0.1) * ss.days(3)` to get an expected number of events (as 02 tells them they will) gets a probability. The `per`/`freq` split exists precisely so that this cannot happen quietly, and here the type that is supposed to be the hazard converts to a probability on multiplication while the spec says the opposite. The rest are gaps in a lattice that 02 and B13 already say "need writing down once".

**Proposal.** Decide the table once and make spec and code agree. For `per × dur` keep the code's meaning (a probability) and correct the row at `02:62`; the alternative, making `per × dur` an expected count so that `.to_prob(dt)` is the only way to a probability, is cleaner but costs more (below). For the rest, follow 02: `number / per` is a duration, `number / dur` is a `per` (a hazard, which is what `1/d` means for a dwell time), and `-` works wherever `+` does. Add `E102` in `resolve_pars` (a `par` in a `rate`/`dur` slot whose value is a plain float that came from `TimePar` arithmetic can be detected by having `TimePar.__mul__` return a tagged float, or simply by checking the slot when both operands were typed, as 02 describes). Make the `IncompatibleTypes` error a `CodedError`. Remove the unit-defaulting in `dur()`/`Rate()` constructors (require a unit; the warning already says the default is a guess).

**Cost.** Changing `per × dur` is breaking for Starsim 3 code that relies on `rate * dt` being a probability: `grep` finds about 46 `* self.t.dt`-style sites in `starsim/` against 13 explicit `.to_prob(` calls, and user code outside the repo will have the same ratio. Each site is a one-word fix (`.to_prob(dt)`), but there are many of them, and a silent change of meaning in `rate * dt` is exactly the error the types exist to prevent. If the code's behavior is kept instead, 02's table and the `per`/`freq` justification ("a hazard ... converts by `1 − exp(−r·dt)`; a frequency ... by `f·dt`"; `02:18`) are already consistent with it, and only the arithmetic row at `02:62` is wrong; that is the cheaper fix, and I would take it.

**Confidence.** High that spec and code disagree; medium on which to change.

### 7. Trim the time constructors from 17 to 9 (and the type names)

**Issue.** 27 `TimePar` classes and instances are exported (`s5_timetypes.out`): 4 base types, 17 unit constructors, 4 singletons (`ss.year` etc.), `TimePar`, `Rate`. The spec counts these as "two choices, not 22 things", which is true of `days`/`perday`. It is not true of `probperweek` and `freqpermonth`, which a user has to know exist before they can choose them. In the code, `freqper*` is used only in two network defaults (`networks.py:1489`, `1674`), `probper*` only in `demographics.py` through `ss.prob.array_to_prob`, and neither appears in any v4 vignette or in `transitions.py`. The per-unit subclasses are one-liners (`time.py:2290-2305`), but they carry the `class_map`/`reverse_class_map`/`get_unit_class`/`normalize_unit` plumbing (`time.py:699-780`, `2310-2378`) whose job is to turn strings and classes back into each other.

**Proposal.** Keep the four types. Keep the eight everyday constructors (`days`..`years`, `perday`..`peryear`) and spell the other two types with a unit argument, which already works: `ss.prob(0.1, 'day')`, `ss.freq(80, 'year')`. Remove `probper*`, `freqper*`, and the four singletons `ss.year`/`ss.month`/`ss.week`/`ss.day` (`ss.years(1)` is one character longer and cannot be confused with the class). That is 9 names instead of 21.

**Cost.** Breaking; about 20 uses of `freqper*`/`probper*` in `tests/` and `docs/`, two in `networks.py`. `to('month')` on a `prob` (`time.py:1738-1742`) builds the class name from a prefix table and would simplify to constructing `prob(self, unit=...)`.

**Confidence.** Medium. The spec explicitly chose to keep Starsim's set; this is a judgment that the set was already too large.

### 8. The bare-number unit rules: learnable in the middle, surprising at the edges

**Issue.** Rules 1 to 3 of `02-types.md` behave as written (`s3_units.out`, cases a to e, h, j, t): `unit=` wins, then a typed `dt`, then dates give days and years give years, then typed values in the model. The edges:

- **Unitless time is days in disguise.** With no unit, `pars.dt = days(1)`, `pars.start = days(0)` (`sim.py:1063-1073`, `unit_base = unit or 'day'`). This leaks: `ss.Sim([ss.SIS(), sir], stop=100)` reports `sim.unit = None` and no note, and `ss.SIS`'s `beta = peryear(0.05)` is converted with `dt = days(1)` to `0.000137` per step (`s6_followups.out`, C). Rule 4 reads typed values only from declared modules (`sim.py:1053-1054`), so a code module's typed defaults neither set the unit nor trigger `W103`.
- **A stop that looks like a year.** `ss.Sim(sir, stop=2000)` infers `unit='year'` and a start of **1950** (`sim.py:1038`, `assume_cal_year` is applied to `stop` too; the start then comes from Starsim 3's default duration of 50). `stop=1950` gives 1900 to 1950; `stop=1899` gives 1900 timesteps. `start=0, stop=2000` gives 2001 years.
- **`dt=0.1` in a unitless sim.** The summer2 vignette uses `dt=0.1, stop=20`. `describe()` says `beta = 2 per step` and `schedule every 0.1 steps: seir`. The spec says a unitless sim's "timestep is one step" (`02-types.md:139`); here the step is a tenth of the unit, and "per step" in the output means "per unit", which is the opposite of what the same words mean in the overview's example.
- **`unit='day'` plus year-looking numbers.** `ss.Sim(sir, unit='day', start=2020, stop=2030)` runs 11 days (`sim.py:1037`, deliberately); 02 says a bare `2020` "is read as a year if it looks like one" without this exception.
- **Two notions of unit.** `sim.unit` is `None` for a Starsim 3 sim (`ss.Sim(ss.SIR(), stop=10)`) whose `sim.t.dt` is `years(1)`; `sim.pars.unit` is also `None`. The user-facing `sim.unit` means "the unit for bare numbers in declared models", not "the unit the sim runs in".

**Proposal.** (a) Count typed values in every module under rule 4, including code modules' defaults, so a mixed sim gets a unit or `W103`, never a silent `days(1)`. (b) Apply the year heuristic to `start` only; a `stop` alone is a duration. (c) In a unitless sim, either forbid `dt != 1` or print `per unit` / `every 0.1 units` rather than `per step`; the cleanest is to call the unit `'step'` in `describe()` only when `dt == 1`. (d) Add the `unit=` exception to 02. (e) Give `sim.unit` the sim's actual unit for every sim (`'year'` for a Starsim 3 sim), with a separate flag for "bare numbers are in timesteps".

**Cost.** (a) can change the unit of existing mixed sims from timesteps to years; it is a sim that is currently mis-converting, so the change is a fix. (b) breaks `ss.Sim(stop=2020)` meaning "until 2020", which Starsim 3 allowed; `start=` is needed. Others are not breaking.

**Confidence.** High on the behaviors (all reproduced); medium on (c).

### 9. `describe()`: good, and missing the `verbose` it promises

**Issue.** `describe()` on the overview example and three vignettes (`s1_describe.out`) is clear, short, and marks guesses the way 01 promises: `[guessed: source of 'infection']`, `[inferred from the arrows]`, `[guessed: scalar duration]`, `[from start and stop dates]`. It is the best part of this area. Gaps against the spec:

- `sim.describe(verbose=True)` raises `TypeError: Sim.describe() got an unexpected keyword argument 'verbose'` (`sim.py:1076`). `01-objects.md:336`, `02-types.md:39-45`, and `10-execution.md:128` all promise it, and it is the only place the per-step conversions (`p = 1 - exp(-0.05 x 1) = 0.0488 per step`) were to be shown. There is no other way to see them.
- A bare rate on a spontaneous transition prints without a unit (`rate = 10`, `transitions.py:1566-1567`), while beta prints `0.05 per step` and a hazard from `rate=` prints `0.5 per step` (`dwell ~ exponential(rate = ...)`). A `par` in the rate slot that is not the whole rate gets no unit either.
- `init: sus = 700, inf = 285.71429, rec = 14.28571; the rest in sus` when the three sum to the population; "the rest" is zero.
- The `note` lines at the bottom repeat what was printed at `init()`, in the same wording; that is by design (`sim.notes`) but reads as duplication when the two are a screen apart.
- Item 7 of the 01 list ("which parts restrict the choice of solver") is not printed in any of the four runs; all were ODE, so it may only appear for agents (unverified).

**Proposal.** Add `verbose=` to `Graph.describe`/`Disease.describe`/`Sim.describe` with the per-step line under each transition; print units for every rate-slot par; drop "the rest in X" when nothing is left. Keep everything else exactly as it is.

**Confidence.** High.

### 10. Names the spec lists that do not exist, and vice versa

From `s6_followups.out`, G, checked with `hasattr`:

- Missing: `ss.func` (`17:51`, `03:425`); `sim.sensitivity` and `sim.crn_report` (`17:119`); `sir.solvers()` (`01:219` says "`sir.solvers()` for one module"; only `sim.solvers` exists); `state.prev` (`17:117` says `n`, `prev`; the code and `02-types.md:22` say `.frac`, so the spec disagrees with itself); `result.quantiles` on a plain result (only meaningful on a MultiSim result, so probably fine).
- Present but not in 17: `ss.immigration`, `ss.emigration` (in 03, not counted in 17); `ss.Expr` (listed as removed); the `describe`-adjacent `sim.to_spec` (the departures list says the spec is `to_spec()`, 17 still says `to_json`).
- `implementation.md` says "`ss.geo` takes the places directly; there are no named geographies", but `named.py:67` implements `geography()` and `transitions.py:2470` calls it from `ss.geo`; the departures list is stale.

**Proposal.** Fix 17 to match `dir(ss)` and add a test that diffs them; delete the `ss.func` line or add the function; choose `frac` or `prev` once.

## Simpler codebase

### 1. `time.py` is 2612 lines for four types

Where the length goes, by section:

- `datedur` (`time.py:1301-1623`, about 320 lines) is a second duration class with its own arithmetic, used in 11 places outside the file, nearly all in `timeline.py`. 02 keeps it "because calendar durations are not fixed multiples of days"; `ss.months(1).days` is 30.4167 and `ss.datedur(months=1).days` is 30.4167 (`s5_timetypes.out`), so for the one thing the user sees they agree. Whether `Timeline` could do calendar stepping with `pd.DateOffset` directly and `datedur` go is worth a try (B13).
- `prob.to_prob` (`time.py:2056-2099`) is a near-verbatim copy of `Rate.to_prob` (`1850-1892`), differing in using `_rate` and a `unit is None` branch.
- `Rate.__mul__` (`1763-1766`) exists only to raise a `NotImplementedError` whose text is the Starsim 2 deprecation message; `warn_deprecation` and the three deprecated functions (`2384-2414`) are 30 lines for names with no users.
- `_defer_ops` (`808-822`) rewrites every subclass's arithmetic dunders at class creation so that `TimePar * Expr` returns `NotImplemented`. A single `isinstance(other, Expr)` check at the top of each `__mul__`/`__add__` would be plainer than a class-decorating loop.
- `class_map`/`reverse_class_map`/`get_unit_class`/`normalize_unit`/`get_dur_class`/`get_rate_class`/`get_timepar_class` (`699-716`, `2310-2378`) convert among strings, classes, and instances; the `TODO` at `2333` and `2343` ask whether they are needed. With finding 7 (nine constructors), a dict of four entries would do.
- The plotting support (`MockAxis`, `FloatYearLocator`, `FloatYearFormatter`, `DateConverter`, `2418-2612`) is 200 lines that belong with plotting, not types.

### 2. The two parameter paths

`Graph.pars` (an `objdict` of `par`, `transitions.py:1128`) and `Module.pars` (an `ss.Pars`) are kept in step by `diseases.py:98-100`: `define_pars` with the values, then `dict.__setitem__` of each `par` over its own value to dodge `Pars.update`'s dispatch. `par.raw` (`transitions.py:218-224`) then reads back through `mod.pars[self.name]` to notice when the user has replaced it. `Pars.update` (`parameters.py:88-135`) has eight type branches and the `_bare`/`note_bare`/`forget_bare`/`bare_units` bookkeeping (`parameters.py:38-60`) exists to warn about the legacy "bare number keeps the default's unit" rule. Finding 2 removes the read-back; finding 3 removes one of the two kinds of entry; finding 8(a) would make the `_bare` warning unnecessary (the sim's unit would be the typed default's unit, or `W103`).

### 3. `__init__.py` exports internals by hand

The `from .x import (...)` lists are the only place the public surface is defined, and modules such as `transitions.py` export their whole `__all__` including `Graph`, `Strat`, `Transfer`, `CodedError`, `error`, `note`. If each submodule's `__all__` were split into public and internal (or `__init__.py` imported only the public list), the vocabulary document could be generated and tested rather than maintained. The import-timing scaffold (`__init__.py:5-16`, `208-217`) is fine.

### 4. `named.py` is the right size

184 lines, three loaders, provenance recorded as `ref` and written into the spec. Nothing to cut. The `unknown()` helper raising `E605` with the provider's own suggestion is a good pattern.

## Spec and implementation

Beyond what `implementation.md` lists:

- **`per × dur`** is a probability in the code and an expected count in 02 (finding 6). One of them has to change; the departures list does not mention it.
- **`E101`** is specified as a coded error and raised as a plain `TypeError`; **`E102`** is specified with a worked error message (`02:71-76`) and is not implemented at all (a `rate=` slot accepts `ss.peryear(0.1) * ss.days(3)`).
- **`describe(verbose=True)`** is promised in three sections and does not exist (finding 9).
- **`sir.solvers()`** is promised in 01 and 17 and does not exist; `sim.solvers()` does.
- **`ss.func`** (17, 03) does not exist. **`I.prev`** (17) versus **`I.frac`** (02, code): the spec is internally inconsistent; the code has `frac`.
- **`ss.Expr`** is listed as removed and is exported. `ss.immigration`/`ss.emigration` are in 03 and the code but not in 17's count.
- **Unit from `unit=` overriding year-looking numbers** (`sim.py:1037`) is in the code and not in 02. **`stop` alone** triggering the year heuristic, with a start of 1950, is in neither.
- **Rule 4 reads typed values only from declared modules** (`sim.py:1053`); 02 says "typed values in the model".
- **"The timestep is one step"** (`02:139`) is not true when `dt=` is given in a unitless sim; `describe()` then says "every 0.1 steps".
- **`implementation.md`**'s "there are no named geographies" is stale: `named.geography` exists and `ss.geo` calls it (`transitions.py:2470`).
- **Vocabulary count**: 94 specified, 253 exported.
- **Open questions now answered by the code**: B1 "typed values that disagree" is the smallest unit with `W103` (verified, `s3_units.out` case i: `disagree=['day', 'year']`, unit day); B11 "keep the period of an `ss.prob` in the spec" is answered yes (`quantity_json`, `transitions.py:1975-1978`, writes `type='prob'` with its unit); B12 "how robust is picking up names from variables" is answered in finding 5 (robust across frames for transitions and modules, which have fallbacks; fails for states in factory code and for aliases); B13 "can the Timeline be simpler" was not attempted (`timeline.py` still has `datevec`, `timevec`, `relvec` as properties, 604 lines; not reviewed further).
- **Spec over-specification**: 17's keyword tables and slot table are useful as a reference but duplicate 01-09; they will drift (they already have, see above). A generated table from the signatures would be more honest than a hand-counted one.

## Bugs

1. **`stop=2000` alone gives a 1950 start.** `ss.Sim(sir, stop=2000).init()` -> `unit='year' source='start and stop years' npts=51 start=years(1950) stop=years(2000)` (`s6_followups.out`, D). `stop=1950` gives 1900 to 1950. The year heuristic should not apply to `stop` when `start` is absent, or the start should be reported as guessed.
2. **`E504` after attribute assignment blames the user for the right thing.** `sir.pars.beta = 0.6; ss.modify(sir.pars.beta, multiply=0.5)` -> `E504 UnmodifiableTarget: the target of ss.modify() must be a named quantity, such as sir.pars.beta, ... not 0.6 (float)` (`s7_setbeta.out`). Cause in finding 2.
3. **`sir.pars.update(beta=0.1)` warns** `No known mechanism for handling <class 'starsim.transitions.par'> → <class 'float'>; using default` (`parameters.py:131-134`) for an ordinary operation.
4. **Empty error messages in rate arithmetic.** `2 / ss.perday(0.1)` and `ss.perday(0.1) / ss.days(2)` raise `NotImplementedError` with no message (`time.py:2206-2212`); `ss.perday(0.1) - ss.peryear(36.5)` raises `Can only subtract rates of identical types ... not 36.5` although `+` of the same pair works (`time.py:1794`).
5. **A code module's typed parameter is converted with a day step in a "unitless" sim with no note**: `ss.Sim([ss.SIS(), sir], stop=100)`: `sim.unit=None`, `sis.pars.beta=peryear(0.05)` -> `0.000137` per step, `warnings: []` (`s6_followups.out`, C).
6. **`describe()` says "every 0.1 steps"** for `ss.Sim(seir, dt=0.1, stop=20)` and labels `beta` "per step" when it is per unit (`s1_describe.out`, summer2).
7. **The `E206` message for reusing transitions names the same module twice**: `state "S" is used by transitions in two modules, "disease" and "disease"` (`s2_names.out`, case 15), because the default name is assigned before the check.
8. **`E210`'s message cannot be acted on**: `state number 1 of the module has no name. It is used in ['<unnamed state> -> <unnamed state>']` (`s2_names.out`, case 6).

## Keep

- **`describe()`'s format and its `[guessed: ...]` marks.** Twelve-character key column, one line per fact, the guess and its rule in brackets. It made every finding above easy to see, which is the point of it. Do not make it longer by default.
- **`CodedError` and `ss.note`** (`transitions.py:31-73`): stable codes in messages, notes stored on `sim.notes` and printed once. Extend to the time errors rather than replace.
- **The capitalization rule is kept consistently in practice.** Every new name in `__init__.py` follows it (`ss.Disease`/`ss.transmission`, `ss.Vaccine`/`ss.vaccinate`, `ss.Mobility`/`ss.gravity`, `ss.People`/`ss.people`). The only confusion is `seasonality`/`seasonal`, which is a duplicate rather than a capitalization problem.
- **Names from variables for transitions and modules**, with the `S_to_I` fallback marked in `describe()`. It works across frames, under `exec`, in lists, and in class bodies; the fallback is what makes it safe. (States are the exception; finding 5.)
- **`__set_name__` on `State` and `par`** (`transitions.py:212-215`, `465-469`) and the deep copy of class members per instance (`diseases.py:88`; `s2_names.out` case 9 shows two instances share neither state nor par). This is the right way to do the class form.
- **Unit rules 1 to 3 and the printed source** (`unit='day' (from start and stop dates)`). Learnable in one reading.
- **`ss.null`** as both source and sink; B12 asks for a better word and I do not have one.
- **`named.py`**: loaders return plain data plus `ref`, and the provider is imported lazily.
- **The `@`-is-a-rate rule** (02) and `I >> R @ 10` being refused as a duration; the spec's reasoning is right.
- **`ss.erlang(k=, mean=)`** as the one added distribution name.

## Outside my area

- `ss.Sim(sir, pars={'sir.beta': 0.1})` fails with `E602 ... matches no parameter. Parameters are disease.beta` when `sir` is built inline, because the module is named `disease` until `ss.Sim` sees a variable; the error could say that the module's name is `disease` and why. (Parameters reviewer.)
- `ss.Sim(sir, start=2000, stop=2050, dt=ss.days(1))` builds 18,251 time points with `start=years(2000)` and `dt=days(1)`; the mixed-unit timeline is correct but `sim.t.start` being `years(2000)` while `dt` is days is the kind of thing the Timeline question (B13) is about. (Execution/time reviewer.)
- `Disease(name=<Transition>)`, `Disease(pars=<Transition>)`, and `Disease(init=<Transition>)` give three unrelated `TypeError`s (`s2_names.out`, case 16) rather than `E201 ReservedName`; `results=<Transition>` is accepted because `results` is a grouped keyword. (States/transitions reviewer.)
- The Atomica vignette's force of infection, written with counts, is classified `spontaneous` and its `rate` line has no unit; B1's "a third kind" question is live for any model that writes its own `/N`. (States/transitions reviewer.)
