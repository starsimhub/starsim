# FPsim: the port rechecked against v4 as of October 21

Written 2026-10-21, at commit `ee5d1fb0` on the `v4-simplify` branch of `ssv4`, run on a clean detached worktree of it (the working tree had uncommitted changes, which weren't tested). This rechecks [oct17/fpsim_review.md](../oct17/fpsim_review.md) (FP1–FP39), written at `1a5b7a77`, 220 commits and the simplification ago, and the October 18 status check ([oct17/status_oct18/fp.md](../oct17/status_oct18/fp.md)). FPsim is unchanged (3.6.0, `1817b043`). Starsim 3.7.2 ran on the same interpreter for comparison. Five agents took the same five parts as in October (pregnancy, contraception, interventions, population, framework): each reran the October scripts, updated the declared ports to the current vocabulary, and probed for new problems. Scripts, logs and each part's findings are in the `fpsim` repo under `v4port/oct21/<part>/`, and FPsim's side is in `fpsim/v4_port_oct21.md`. I reran the headline new findings myself (FP40–FP45, FP48–FP54, FP56, FP66); all reproduced with the numbers given. The machine was loaded (1-minute load 18–52) for most of the work, so timings within about 30% are noise.

New findings continue the numbering at FP40. Duplicates of findings from the other rechecks today are marked (MA34 is in [malaria_recheck.md](malaria_recheck.md)).

## In brief

**Much of the October review is fixed.** Of its 39 findings, 22 are fixed, 11 partly, 2 removed with the feature (one with a working replacement, one folded into `describe()`), 3 open and 1 changed.
- **Run tools work on `fp.Sim`** (FP7, FP8): `new()`, sweeps by parameter name (now labeled, R2), `ss.parallel(baseline=..., n_runs=).diff()`, `MultiSim` and `sim.calibrate(gof=)` all work on an unchanged `fp.Sim`. `calibrate()` recovers an exposure factor of 1.58 against a true 1.5 in 4 trials.
- **The silent engine answers of October are fixed**: a change to a branch's `p=` parameter under the ODE and binomial (FP13: 0.800 and 0.806, was 0.200), a parameter read inside a function (FP14), a remainder beside a time-series branch (FP15), `where=` on a dwell exit (FP16), a competing hazard during a short dwell (FP17: 0.0808 against 0.0800, was 0), numeric dimension levels (FP19), stratifying by an existing property (FP20), a bare `dt` (FP21, with a warning), `at=stop` (FP5, now refused, naming the last step), `resample('year')` on dated sims (FP3), and `loop.insert()` twice (FP6).
- **The behavior vocabulary exists, and the declared ports need no functions of the sim**: counters (`increment=`, FP24), clocks and time since an event (FP25), `expit` over `ss.ppl` properties (FP26), `p=` tables with a `to` axis (FP27), `ss.transfer(p=, dur=)` (FP29), `ss.Deaths(lifetable=)` by year and sex (FP33), `dur=0` splits (HP16) and rate × time series (FP31). The declared pregnancy went from 8 per-step functions to 0 (115 → 99 lines), the declared contraception from 16 runtime lambdas and 2 caches to 0 (92 → 91 lines), and education to 77 lines (91 in October, FPsim's 213). They agree with FPsim as well as or better than in October.

**What stops the port being a joy now, in five groups:**

1. **FPsim's pregnancy module can't run on v4, and nothing says so until it crashes** (FP40–FP42). Starsim 3's coded `ss.Pregnancy` is gone; v4's is declared. `FPmod` subclasses it and overrides 9 of its methods; 8 of the 11 v3 states and 13 of the 17 v3 methods it uses are gone. Unchanged, it fails at creation; a subclass that overrides removed methods without crashing is silently ignored (151 conceptions where the override says 0). The migration guide never mentions pregnancy and still says coded modules "run unchanged". FPsim has to rewrite `FPmod` on the declared `ss.Pregnancy`; the seams for that exist (`conception_rate()`, `ss.modify` of its transitions, `since_birth`, callbacks, `ss.Lactation`) and work. The end-of-run warning for v3-style result indices (FP1's fix) has been removed, while the skill still says a run warns.
2. **New silent wrong answers in births and the people they make** (FP43–FP50): `ss.Pregnancy(embryos_per_pregnancy=1)` makes no children; `ss.birth(per=)` makes no children when the parent's transition leaves the model, and loses about 2.5% when the parent's transitions have a dwell; `ss.mother(ss.ppl.age)` reads 0 for every newborn; `ss.histogram` as a dwell or trait draws uniform(0, 1); traits are never drawn for people born to a coded module; and `move_to()` in `init_post()` gives people a past they never had.
3. **New silent wrong answers in the solvers and modifiers** (FP51–FP56): the ODE loses 9–11% of people when two modules are crossed and one has a fixed dwell; `ss.change` or `ss.modify` on a `p=` table does nothing (so FPsim's `update_methods` and `method_switching` have no declared form that works); a Weibull dwell with shape below 1 becomes one exponential stage with an enormous mean under the ODE, which `auto` picks (mCPR 0.09% against 0.37% as agents); a modifier on a declared module's plain parameter (`pregnancy.rel_fertility`) is accepted and does nothing; a dwell under half a step ends in the step it began; and `sim.hash()` of a coded sim changes when it runs.
4. **The declared forms are slow because of one function** (FP57): `ss.by.lookup()` rebuilds its table from pandas rows on every call. It makes declared deaths 107 times slower than `fp.Deaths`, declared contraception 9.4 times slower than FPsim's (1.4 in October), and the declared pregnancy 2.6 times slower than `FPmod`. Caching `keys()` alone (6 lines) brings the pregnancy level with `FPmod`.
5. **Crashes and friction in the new vocabulary** (FP58–FP72): arithmetic on `ss.by` tables, a modifier amount over another module's states, `ss.attr(..., on=X.exit)`, dwell expressions over `ss.ppl`, `to_code()` with an unnamed table, rows over `to` that the spec says may omit destinations, int32 UIDs, and gaps in the migration script.

**Counts:** 33 new findings (FP40–FP72): 16 silent, 7 crashes, 3 speed, the rest friction and docs. Groups 2 and 3 are each 1 to 20 lines to fix, except FP51 and FP52 (S–M). Group 1 needs a decision (below). Group 4 is about 15 lines.

## The port today

| Task | Oct 17 (`1a5b7a77`) | Oct 21 (`ee5d1fb0`) |
|---|---|---|
| FPsim unchanged, its 69 tests | 55 of 61 fail on two removed names | 66 of 69 fail at sim creation, on a list given for a distribution (`dur_breastfeed=[20.27, 4.27]`, S5.9); behind it, `FPmod` can't run on the declared `ss.Pregnancy` (FP40) |
| With a shim | 2 lines; 55 of 61 pass | 6 items, one of which vendors Starsim 3's 846-line `ss.Pregnancy` back; 61 of 69 pass (below) |
| `python -m starsim.interop.migrate` on FPsim | n/a | 16 changes in 8 files (`verbose`, `ti - 1` in analyzers), 13 items by hand (the pregnancy subclass, `reduce()`); misses `to_df(resample=)` (FP67) and FPmod's results written in helpers (FP42) |
| FPsim's interventions against Starsim 3, 6 seeds | Same effects | Same effects (e.g. `change_par` births −49.3% against −49.1%), with the shim |
| Run tools on `fp.Sim` | `new()` returns an empty sim | Work |
| A change to the switching matrix | Ignored under the ODE (FP13) | Applied when written as weights; ignored, as agents and ODE, when written as a `p=` table (FP52) |
| Contraceptive efficacy on conception | `UnmodifiableTarget` | `ss.modify(preg.conception, ...)` on v4's declared `ss.Pregnancy` works (1164 → 602); FPsim's coded modules still unreachable (FP23) |
| Parity, time since birth, the regression, the switching table | Functions, agents only | Declared; contraception also runs as an ODE, but wrongly (FP53) |
| Declared forms against FPsim's code, speed | 1.4–1.8 times slower (deaths 13–30) | 2.6–9.4 times slower (deaths 107), from FP57 |

**The suite with the shim** (61 of 69 pass; on Starsim 3 all 69 pass). The shim has six items: a list for a distribution calls `.set(*list)`; FPsim's classes may redefine an inherited state (`ti_live_birth`); Starsim 3's `ss.Pregnancy` is put back; FPsim's 2D array accepts an int32 UID (FP66); `verbose=1/12` becomes 1; and `to_df(resample=)` sets the date index (FP67). The eight failures:

| Test | Failure | Cause |
|---|---|---|
| `test_baseline` | IMR 213 → 0, MMR 657 → 0, TFR 5.1 → 0, ASFR 142 → 0; crude birth rate 4.3 → 40.4; pop growth +15% | The vendored v3 pregnancy and `FPmod` write results at v3's indices (FP1, now with no warning, FP42). The crude birth rate is FPsim's bug 24 fixed by FP3: the summary's last bin is now a year, and 40.4 is near the data's 34.5 |
| `test_fertility_rates` | ASFR all 0 | As above |
| `test_pregnant_women` | 0 stillbirths in the cohort | X13 and FP1 |
| `test_change_people_state`, `test_add_method_year_boundaries[2]` | FPsim's own interventions timed at `stop` never fire, silently | FPsim's bug 14: v4 refuses `ss.change(at=stop)` (FP5), but FPsim's interventions check the year themselves |
| `test_multisim` | `'MultiSim' object has no attribute 'reduce'` | FP12 |
| `test_to_df` | `'dataframe' object has no attribute 'timevec'` | The shim's `set_date_index` item moves `timevec` into the index; without it, `Experiment` crashes instead |
| `test_age_restricted_initiation_basic` | Oldest user 32.05 > 32 | Flaky on Starsim 3 too (FPsim's bug 31) |

## Status of the October findings

| ID | Finding | Status | Evidence (scripts under `fpsim/v4port/oct21/<part>/`) |
|---|---|---|---|
| FP1 | A coded flow at `self.ti` gives 0 | Open, worse | Ratio 0.0 at every step and no warning: the end-of-run check added on Oct 19 was removed in `7c0c8c96` (FP42). The migration script fixes a write inside a function named `step` (0.088–0.118), not one in a helper it calls, which is where `FPmod` writes (`pregnancy/repro_step_result_*`) |
| FP2 | `finish_step()` results never reach `update_results()` | Partly (documented) | Still 100 per step where 105 is expected; the skill has the row and the script lists it |
| FP3 | `resample('year')` on dated sims | Fixed | Dated deaths by year [1048, 946, 977], as undated (`population/repro_calendar_resample_v4.log`) |
| FP4 | Resampled stocks are year-end values | Fixed (docs) | CHANGELOG and skill rows. The skill says `kind='ratio'` averages a stock "as annualize() did"; it is the mean of the 13 points ending each 1 January, 0.1–0.3% from v3's |
| FP5 | `at=stop` accepted, never applied | Fixed | Refused with `TimeOutsideSim`, naming 2009.917 (`interventions/repro_at_stop_v4.log`); the message calls an `ss.change`'s target "r_multiply" |
| FP6 | `loop.insert()` after `init()` twice | Fixed | 10 calls for 11 points |
| FP7 | `sim.new()` on `fp.Sim` empty | Fixed | `new()` rebuilds `fp.Sim` with 4 modules; `ss.parallel` of a run `fp.Sim` gives 127 and 117 births (`framework/repro_new_empty_v4.log`) |
| FP8 | `calibrate()` on `fp.Sim` crashes | Fixed | 1.580 for a true 1.5 in 4 trials (`framework/calib_coded_v4.log`). `sim.calibrate(gof=<FPsim's mismatch>)`: best 1.713 in 20 trials, 213 s, against `fp.Calibration`'s 287 s on v4 and 378 s on v3 |
| FP9 | `to_spec()`, `hash()`, `diff()` on an array duration | Partly | Work on `fp.Sim`; the minimal case `ss.normal(loc=ss.months(np.array([3, 6])))` still raises `only 0-dimensional arrays ...` (the Oct 19 "already worked" holds only for FPsim's case) |
| FP10 | Removed names break packages | Changed | All 38 `ss.` names FPsim uses exist. It breaks on behavior changes instead: the six shim items, of which the script handles one (`verbose`) and flags one (pregnancy), and nothing covers the int32 UIDs (FP66) |
| FP11 | `to_code()` writes `ss.probperyear` | Fixed | The written code runs. On `fp.Sim`, `to_code()` writes an empty `ss.Sim` with a `LossyExport` comment per coded module, honestly labeled |
| FP12 | `msim.reduce()` | Open, changed | A bare `AttributeError` (fine under no baggage). The script's suggested replacement `low, mid, high = res.quantiles([...])` raises `too many values to unpack` when the scenarios are labeled, since `quantiles()` is per scenario (6 rows for 3); v3's pooled band (113 to 355) has no spelling (`framework/probe_reduce_v4.log`). The Oct 19 note that `quantiles()` pools every run holds only for unlabeled runs |
| FP13 | ODE and binomial ignore a change to a `p=` parameter | Fixed | 0.795 abm, 0.800 ODE, 0.806 binomial (was 0.200, 0.198). A `p=` table is a new case (FP52) |
| FP14 | A function reading a parameter sees its base value | Fixed | `float(par)` and the declared `expit(...)` both go 3,570 → 0 starts |
| FP15 | A remainder beside a time-series branch competes | Fixed | C 0.296 abm, 0.300 ODE (was 0.153) |
| FP16 | `where=` on a dwell exit checked only at entry | Fixed | Share left 0.00 → 1.00 (`contraception/probes/probe_small_adapted_v4.log` 3b) |
| FP17 | A competing hazard during a k-step dwell acts k − 1 steps | Fixed | 1, 2, 3, 6 months: 0.0808, 0.1518, 0.2208, 0.3912 against 0.0800, 0.1535, 0.2212, 0.3935. It now also acts in the step the exit falls due, so the declared stillbirths are 31% above `FPmod`'s, which excludes that step (a choice of convention, not a bug) |
| FP18 | `offspring=` doesn't age or set parents | Removed (replacement works) | `ss.birth(NB, per=tr, n=)`: everyone ages, every newborn has a parent. New problems in it: FP44–FP47 |
| FP19 | Numeric dimension levels | Fixed | Counts [1808, 1808, 1793, 1796, 1785]; `adjust=` reaches quintile 5 |
| FP20 | Stratifying by an existing property overwrites it | Fixed | Urban stays 0.900; `split=` raises `BadSplit`; a float `wealth` is binned |
| FP21 | Bare `dt` read in months | Fixed | 241 points and a `UnitInferred` warning |
| FP22 | A function returning a bare time is read in the sim's unit | Fixed (a note) | The note shows only at `verbose>=1`, so it is silent by default |
| FP23 | Modifiers can't reach coded modules | Partly | The wrong "did you mean dummy.rel_sus" is gone; a coded `ss.Infection`'s parameters get a clear `CapabilityError`. Still: with no declared module, every `ss.change`/`ss.modify` crashes with `min() iterable argument is empty` (in `sc.suggest` with an empty list, which `did_you_mean`'s `default=` doesn't guard); a coded `Demographics` or `Connector` parameter gives a bare "not found"; `ss.Delivery` on coded states says "no declared diseases" or lists an empty set; and no v4 form changes a coded module's plain parameter. The 23-line `coded_change` prototype still matches FPsim's `change_par` exactly |
| FP24 | No counter | Fixed | `increment='parity'` and `increment=dict(parity=2)` for twins |
| FP25 | No time since an event | Fixed (agents) | A clock dimension reset on live births, read by `ss.by` tables; also `preg.since_birth` and `(A | B).time` |
| FP26 | No declared regression | Partly | `expit` over `ss.ppl` properties, coded arrays in conditions, age tables and `ss.par` coefficients work, and `ss.change('contra.b0')` acts (2452 → 255). A year trend still needs a hand-made `ss.timeseries`, since time can't appear in an expression |
| FP27 | No tabular `p=` | Partly | `p=[1 - p_use, p_use*mix*w]` with a `to` axis works; a function returning a row is refused clearly. Changing the table does nothing (FP52), and a row can't omit a destination (FP64) |
| FP28 | No delivery to a mix; no modifier over several transitions | Open | `Delivery(move=dict(to=[...], p=...))` gives a misleading `MissingDimension ... sim.stratify(to=[...])`; `ss.modify([tr1, tr2])` is refused clearly |
| FP29 | `ss.transfer` rejects `p=` and `dur=` | Fixed | `p=[0.3, 0.7]`, `dur=1y` gives [11, 524, 1188] |
| FP30 | Distribution by level; `ss.histogram` as a dwell | Partly | Both are accepted, but the histogram's draws are wrong (FP48); `ss.attr(ss.by(urban={...histograms}))` crashes with `float() argument ... not 'histogram'` |
| FP31 | Rate × time series in three spellings | Fixed | All five spellings run |
| FP32 | `ss.par` in a modifier amount can't be changed | Partly | Works only when the parameter is also given to a module by keyword (1,359 → 2,434 conceptions). Otherwise by object it says "does not belong to a module yet; build the module first" (it was built), and by name "not found". The Oct 19 "already worked" isn't borne out |
| FP33 | Life tables by year and sex | Partly | `ss.Deaths(lifetable={year: dict(female=, male=)})` equals `ss.Deaths(death_rate=table)` run for run. `ss.death(rate=ss.peryear())` runs with no declared disease; `ss.death(lifetable=)` and `ss.death(rate=ss.by(age=...))` are still refused there, with the old message, which suggests `ss.death(rate=...)` |
| FP34 | Declared demography and choice slower; `inspect` per call | `inspect` fixed; speed worse | 0 `inspect` calls (the arity is cached). Speed: FP57 |
| FP35 | `ss.choice` of durations as a dwell crashes | Fixed | Mean stay 3.493 months against 3.5 |
| FP36 | `Transition(A + B, ...)` crashes | Fixed | |
| FP37 | `solvers()` after `init()` crashes | Removed | `describe()` carries the solver plan per transition |
| FP38 | `describe()` hides conditions and names | Partly | No `<lambda>` left; `where=` and `expit` print. Still: `ss.par` scalars and tables print as values (`by(table of 36 rows by age)`) though named; the regression prints twice per transition (154 lines); `fp.Sim` is "2,000 people [guessed]"; and a behavior model is a "DiseaseBase" whose states are "infected ... cured by `ss.treat`" (FP71) |
| FP39 | Small frictions | Partly | Fixed: the `verbose` message; `Delivery` without `disease=` when one declared module has the states; reused distributions (not reproduced). Open: `ever='True -> False'` (`'"True" is not a level'`); `Delivery` without `disease=` beside a second declared module (`UnknownDisease`, about 3 lines); `diff()` and `msim.results` silently drop results the baseline lacks; `stratify(age=[0, 15, 120])` makes an empty `120+`; counts by age use ages from before the step's aging (164 under 10 where there are none) |

**Known findings FPsim hits again:** X13 (no `sim.ti_step`; the script rewrites analyzers to `ti - 1`), D23 (`stratify=` on `fp.Sim` still makes nothing, now with a note), N20 (partly: `ss.par(<dict>, bounds=)` is refused, but the message's advice, one `ss.par` per entry, crashes with `UFuncTypeError` in a coded module, so `exposure_age` can't be fitted), HP1 (fixed: one multi-destination transition decides at exit, by the spec), HP2 (partly: newborns still wait a step, FP47), HP15 (fixed for sex and age; values per person still need `move_to()`, FP50), HP16, HP33, D15, R2 (all fixed). Decision 10 (one default age distribution) is listed as decided, but at `ee5d1fb0` `ss.People()` with no data is still uniform 0–60 and only a sim with a declared module gets the world's ages (mean 29.90 against 33.86); `fp.People` passes Senegal's pyramid, so FPsim is unaffected.

## What should change in v4

### 1. FPsim's pregnancy (a decision for Cliff)

- **FP40 (dead end): Starsim 3's coded `ss.Pregnancy` is gone, and the migration guide doesn't say so.** v4's `ss.Pregnancy` is a declared `ss.Disease` (`NP`, `P`) whose `step()` runs the engine. `FPmod` subclasses Starsim 3's and overrides 9 of its methods; of what it uses, 8 of 11 states (`gestation`, `pregnant`, `ti_delivery`, `breastfeeding`, ...) and 13 of 17 methods (`make_p_conceive`, `process_delivery`, `progress_pregnancies`, ...) are gone, and the new class's states (`ti_live_birth`, `parity`) collide with `FPmod`'s own. Unchanged, `FPmod` fails at creation; past that, on `fecund`. The script flags it (8 lines of `fpmod.py`) and the skill has a row, but `docs/migration/v3_v4.md` never mentions pregnancy, and its overview says modules written as code "run unchanged". The rewrite looks feasible: `conception_rate()` overridden with FPsim's per-woman hazard, losses by age as `ss.modify(preg.miscarriage, multiply=ss.by(age=...))`, twins as `ss.choice`, postpartum by `since_birth` or `ss.Lactation`, all checked (`pregnancy/probe_fpmod_on_declared_v4.log`). Infant mortality has no slot in `ss.Pregnancy` and needs a declared newborn transition or code. **For Cliff:** keep a coded pregnancy base (e.g. Starsim 3's, about 850 lines, in `ssl`) for packages like FPsim, or treat the rewrite as the port (my recommendation, since the declared version has the seams FPsim needs and the constitution says no baggage). Either way: a guide row, and FP41.
- **FP41 (silent): a subclass of `ss.Pregnancy` that overrides Starsim 3 methods is silently ignored.** One overriding `make_p_conceive` to return 0 still has 151 conceptions, and the override is called 0 times (on v3: 0 conceptions, 47 calls) (`pregnancy/repro_override_ignored.py`). Fix: `__init_subclass__` on `ss.Pregnancy` that raises when a subclass defines a removed method, naming the seam to use (`conception_rate()`, `ss.modify`, the callbacks); about 10 lines. `ssl.AgentSIS` already does this for `ss.SIS` (MA23).
- **FP42 (silent): the warning for v3-style result indices was removed, and the migration misses `FPmod`'s pattern.** The end-of-run check for a coded flow non-zero at index 0 or zero at the last point (FP1's fix of Oct 19) was removed in `7c0c8c96`; `skills/migrate-starsim3/SKILL.md:88` still says "a run warns". `migrate_ast()` rewrites indices only inside a function named `step`, and every `FPmod` result is written in a helper (`_post_delivery`, `record_ages`, `handle_abortion`, ...): migrated, the repro's ratio is still 0.0 (`pregnancy/repro_step_result_helper.py`). Fix: restore the check (about 15 lines), or follow `self.<method>()` calls from `step()` in the script (about 20); correct the skill.

### 2. Births, newborns and people's values

- **FP43 (silent): `ss.Pregnancy(embryos_per_pregnancy=1)` makes no children.** A number becomes `ss.choice(a=1)`, which draws from `range(1)`: 1,181 deliveries, 0 births; `=2` the same. The default, `ss.constant(1)` and `ss.choice([1, 2])` are right (`pregnancy/repro_preg_embryos.py`). v3 crashed on this. Fix: a number is `ss.constant(n)`, and `ss.choice` refuses a scalar `a=`; about 5 lines.
- **FP44 (silent): `ss.birth(NB, per=tr)` makes no children when `tr` leaves the model** (to `ss.null`): 3,149 events, 0 children; to a state, 3,149 (`pregnancy/repro_birth_per_null.py`). A maternal death at delivery is this case. Fix: make the children before removing the parent, or refuse; S.
- **FP45 (silent): `ss.mother(ss.ppl.age, born=True)` reads 0 for every newborn.** All mothers are 34 to 64, yet `p = 0.1*(1 + (mother's age < 20))` kills 0.199 (0.100 expected), and `> 20` kills 0.099 (0.200). `ss.mother(ss.ppl.age < 20, born=True)` and a state condition work (`pregnancy/repro_mother_newborn3.py`). This is how infant mortality by mother's age is written. Fix: evaluate a property expression on the mother's UID, as conditions are; S.
- **FP46 (silent): `ss.birth(per=)` loses about 2.5% of children when the parent's transitions have a dwell** (2.39–2.63% for three combinations of dwell and rate; 0% with rates only). Not id reuse in `Agents.fresh()`; cause not found. The declared `ss.Pregnancy` doesn't show it (`pregnancy/repro_birth_per_count.py`). S to M.
- **FP47 (silent, small): newborns placed by `ss.birth(per=)` wait a step before a `dur=0` split out of their state** (822 of 39,256 still in NB at the end; HP2's fix covers people arriving by a transition, not births). 3.6% of the declared pregnancy's newborns never face the infant split. S.
- **FP48 (silent): `ss.histogram` in the declared engine draws uniform(0, 1).** `numeric_dist()` rebuilds a distribution as `type(dist)(**pars)`, and a histogram's values and bins aren't in `pars`. As a trait, mean 0.49 where the histogram's is 6.0; as a dwell, graduation 1.59 years after the start where about 7 is expected, with or without `unit=`; drawn directly, 6.02 (`population/probe_hist_attr.py`). FPsim's education objective is a histogram. Fix: copy the distribution (or keep its parameters in `pars`), and a contract test that the engine's draws match `rvs()`; about 10 lines. `ss.attr(ss.random())` crashes in the same function (`got multiple values for keyword argument 'dtype'`).
- **FP49 (silent): traits (`ss.attr(dist)`) are never drawn for people born to a coded module:** 1,327 of 1,327 NaN under `ss.Births`, 0 under `ss.birth` or `ss.Pregnancy` (`population/repro_traits_newborns.py`). With `FPmod`, 350 of 828 girls under 12 "completed" school at 9. Fix: draw traits for new people in `People.grow()`, as states are; about 5 lines.
- **FP50 (silent): people placed in a state by `move_to()` in `init_post()` get a past they never had.** `sim.init()` then backfills their history through the transitions into that state: an attribute recorded on one of them reads 1.505 where it should be 0.505, a counter 1 where it should be 0, and a write from `init_post()` is overwritten (`population/repro_move_to_attr.py`). FPsim's initial education is placed this way. Fix: don't backfill people placed by `move_to()`, or document it and offer a placement that sets attributes; S, but it needs a decision.

### 3. Solvers and modifiers

- **FP51 (silent): the ODE loses people when two modules are crossed and one has a fixed dwell.** A method module scales a pregnancy module's conception, and pregnancy has a fixed 9-month dwell (a conveyor): after 20 years 4,532.6 of 5,000 people are left with 2 methods (−9.3%), 4,432.7 with 3 (−11.3%), and births are 540 a year against 600 as agents. With an exponential dwell, or uncrossed, all 5,000 stay (`interventions/repro_joint_ode_min.py`). Only a note says the states are tracked jointly. Fix: the conveyor's stages in the crossing seem to drop moves the other module makes during a stage; plus a conservation contract test for every crossing. S to M.
- **FP52 (silent): `ss.change` or `ss.modify` on a `p=` table does nothing.** Row B 0.2 → 0.6 from 2005: shares stay 0.201 (agents) and 0.200 (ODE), by object or by name, with no message (`interventions/repro_to_tables.py`, `contraception/probes/probe_table_change.py`). The plan lists "`ss.change` on a table with a `to` column" as left, but it is accepted silently. It is how `update_methods`, `method_switching` and `add_method` would be declared; a modifier on one destination (`contra.from_none.pill`) does work. Fix: refuse now (S), implement next (M).
- **FP53 (silent): under the ODE, a Weibull dwell with shape below 1 is one exponential stage at its mean.** With FPsim's c = 0.093 the mean is about 2.6e7 times the scale; the note says only "approximated by 1 Erlang stages (each group keeps its mean)", and `solver='auto'` picks the ODE: mCPR goes 0.51 → 0.09% against agents' 0.89 → 0.37% (`contraception/ode_try2_v4.log`). (c = 0.093 is FPsim's bug 6; with it fixed, ODE 1.29% against agents 1.10%.) Fix: under `auto`, run agents when a dwell's coefficient of variation is above about 3, and refuse or warn under an explicit ODE (or fit a hyperexponential for CV > 1); S to M. The same ODE takes 75–318 s for its 10,240 groups against 9.5 s as agents.
- **FP54 (silent): a modifier on a declared module's plain parameter is accepted and does nothing.** `ss.modify('pregnancy.rel_fertility', multiply=0)` and `ss.change(..., to=0)`, by either path: 1,164 conceptions with or without (0 when given at construction), because `conception_rate()` reads it in code (`contraception/probes/probe_coded_par_modify.py`). Fix: apply it through the `float(par)` path (FP14's), or refuse; S.
- **FP55 (silent or a misleading crash): a dwell under half a step ends in the step it began.** Dwells round to the nearest step, so a self-loop re-decides at once: in the declared contraception 749 of 774 debuts left `never` in the step they entered it (mCPR at 2000 0.80% against FPsim's 0.49%), and in pure form `A -> [A, B]` with a mean of 0.2 months raises `TriggerLoop` saying "give one of them a rate or a duration", which it has (`contraception/probes/probe_short_dwell_loop.py`). FPsim rounds with a minimum of one step. Fix: a scheduled dwell lasts at least one step (or a `min_steps=`), and fix the message; S.
- **FP56 (silent): `sim.hash()` and `ss.diff()` of a coded sim read its live parameters,** which coded modules change as they run (`FPmod`'s `p_loss.set(p=...)` each step), so a sim's hash changes when it runs (model 27349d → 050a81, pars f57f2b → 5c7060), and a diff of two `fp.Sim`s that differ only in `exposure_factor` also lists `p_loss` and "modifiers changed" (`framework/probe_hash_run.py`). Fix: hash the parameters as they were at `init()`, or the recorded call; M.

### 4. Speed

- **FP57: `ss.by.lookup()` rebuilds its table on every call** with `iterrows()`, calling `keys()` for each row and dimension, which rescans the column (O(rows²)). It is 12.2 of 14.3 s in the declared pregnancy, 85 of 118 s in the declared contraception, and nearly all of the declared deaths (one age × sex × trend table: 83 s against `fp.Deaths`' 0.78 s). Caching `keys()` per table (6 lines) takes the pregnancy from 5.23 to 2.08 s, with identical results (`FPmod` on v3: 1.96 s); building the grid once per table and only indexing each step (about 15 lines) should do the rest. This is the one fix that decides whether the declared forms are usable for FPsim.
- **FP58: the declared method × pregnancy toy as agents is 2.1 times slower than at `1a5b7a77`** (3.09 against 1.44 s, back to back; methods alone unchanged). Rate exits are drawn as dwells and redrawn whenever their modifier changes, which is every method switch (C17's fix): `schedule()` 8,264 calls against 2,880 (`interventions/profile_toy_v4.log`). Fix: redraw only the exits whose rate changed, or use per-step hazards for rate exits under modifiers; M. A measured cost of a correctness fix, so for `deferred.md`.
- **FP59: `ss.years(array)` scans every element in Python for dates** (`time.py:1309`): 1.9 of 9.7 s in the declared education, 0.9 of 27.8 s in FPsim's Kenya run (as in v3). Check the dtype first; 1 line.

### 5. Crashes and friction in the vocabulary

- **FP60 (friction): `ss.by` tables don't combine as rates need.** `by + by`, `by - by`, `np.minimum(by, 1)` and `np.log(by)` raise; `by * by`, `1 - by`, `by ** x` work. Wrapping each in `ss.par()` works but needs a `name=` for each (9 in an `FPmod`-style rate). Make `ss.by` an expression node and name an unnamed table after its slot; about 20 lines.
- **FP61 (crash): a modifier amount over another module's states** (`ss.modify(preg.conception, multiply=1 - 0.9*meth.on)`) is accepted and crashes in the run (`Disease object has no attribute "on"` as agents, `KeyNotFoundError` as ODE): the states are looked up in the target. It is efficacy in one modifier instead of ten. Resolve as `where=` does; S.
- **FP62 (crash): `ss.attr(I.time, on=I.exit)`** raises `UFuncTypeError ... bitwise_or` (`agents.py:1023`, `np.isin` against a scalar); `on=` a transition works. 1 line.
- **FP63 (crash): dwell expressions over `ss.ppl` or attributes** (`dur=np.maximum(6 - ss.ppl.age, 0)*ss.years(1)`) raise `IndexError` or a broadcast error once people are born, since leaves are evaluated over arrays of different lengths. S to M.
- **FP64 (spec drift): a `p=` row over `to` must name every destination** (`BadBranchSplit`), where spec 03 says an unnamed one gets 0; and a table naming a label that isn't a destination raises `UnknownState`, so one availability table can't be shared by rows with different destinations (the toy needed self-loops at weight 0). Follow the spec; S.
- **FP65 (crash): `sim.to_code()` with an unnamed `ss.by` table** writes its repr (`by(table of 4 rows by age, to)`), a `SyntaxError`. Write tables inline or name them; S.
- **FP66 (crash, migration): UIDs are int32 (`ss.dtypes.uid`) while `ss.dtypes.int` is int64,** so code that checks a single UID against `ss.dtypes.int` (FPsim's `arrays.py:64`) fails when looping over `ss.uids`. Neither the script, the skill nor the guide mentions it. A row, or name `ss.dtypes.uid` in the docs for `uids`; S.
- **FP67 (migration): `to_df(resample=...)` no longer puts the times in the index, and `use_years=True` is ignored.** The script doesn't flag it (the skill has the row), and FPsim's `Experiment` crashes far away with `IndexError` at the five lines the plan lists (`experiment.py` 89, 90, 107, 139; `plotting.py` 453; verified). With `use_years=True` the index is 0, 1, 2, ... and `timevec` holds period ends, while `Result.resample(use_years=True)` labels the year covered. Flag it in the script (one pattern), and honor or refuse `use_years`; S.
- **FP68 (friction): the S5.9 message for a list given to a distribution keeps a unit trap.** It suggests `pars.dur_breastfeed.set(20.27, 4.27)`, which reads the mean in the default's unit (years), as v3 did silently; FPsim's breastfeeding has lasted about 20 years in every Senegal run (241 steps drawn, on v3). Say the unit in the message (`ss.months(...)`); S.
- **FP69 (friction): `init=dict(A=1.0)` is one person.** Duplicate of MA34; it broke four of FPsim's October probes too.
- **FP70 (friction): `ss.Delivery(move=)` takes only an arrow string of names, while `ss.Transition` takes only state objects;** `move=(m.none, m.pill)` and `move=m.none >> m.pill` crash with `AttributeError: items`. Accept a state pair or a transition and infer the module; S.
- **FP71 (docs): `describe()` of a declared model that isn't a disease** prints `DiseaseBase 'meth'`, calls every method state "infected ... counted in prevalence, and cured by `ss.treat`", and leaving `none` "an acquisition [guessed]". With FP38's rest; S.
- **FP72 (small things):** `ss.diff()` prints and returns None unless `output=True`, easily read as "no difference"; `ss.choice(a=ss.weeks(array)).pars.a` loses its unit after `init()`; the CHANGELOG still describes the removed `ss.Requires` (line 86); a note on an empty age group is printed only when the sim, not the module, is stratified.

## Declared ports, October against now

| Part | Oct 17 | Now | Agreement now | Speed now | Still code or a workaround |
|---|---|---|---|---|---|
| Pregnancy | 115 lines, 8 per-step functions | 99 lines, 0 | Against `FPmod` on v3, 8 seeds: conceptions z = 1.8, births 0.5, abortions −0.1, miscarriages 1.6, maternal and infant deaths −0.4, 0.6; stillbirths +31% (z = 8.9, the FP17 convention) | 2.6× `FPmod` (5.23 s), level with a 6-line cache (FP57) | Named `ss.par` wrappers (FP60), the comparison inside `ss.mother` (FP45), a state for births at a maternal death (FP44) |
| Contraception | 92 lines, 16 lambdas, 2 caches | 91 lines, 0, 0 | Full sim, 8 seeds: mCPR ×0.985, births ×1.003, other outcomes within 0.5% of FPsim on v4 | 9.4× FPsim (96.6 s) (FP57) | The year trend as a time series (FP26); `after=` for the minimum step (FP55); FPsim's efficacy bridge (FP23); men leaving `young` by a transition (HP15) |
| Education | 91 lines | 77 lines | Within 1 SE of FPsim's fixed version on 6 seeds (attainment 4.304 against 4.295–4.308); in school +8% | Step 4.3× the coded one (FP57, FP59) | Initial states (FP50), the dropout rate reading FPsim's parity (a table works but 2.5× slower), two dwell functions (FP63), the objective's draw (FP48), traits for `FPmod`'s newborns (FP49) |
| Deaths | 200 time series, 4.8 s | 1 table × 1 series, 6 lines; `ss.Deaths(lifetable=)` | Crude death rate 8.67 (table), 8.84 (`lifetable=`), against `fp.Deaths` 8.74 | 83 s (107×) as declared; `ss.Deaths(lifetable=)` 2.2 s | Nothing, once FP57 is fixed |
| Interventions toy | 84 lines | 70 lines with `p=` tables | Baseline and four of six scenarios identical to the 85-line straight port; `update_methods` and `method_switching` do nothing (FP52) | 2.1× slower than Oct 17 (FP58) | Coverage as a share of current users; FPsim's coded parameters (FP23); one-modifier efficacy (FP61) |

`calibrate()` now reaches the declared regression's intercept in every trial (in October it couldn't be fitted at all); with mCPR near 1% the noise left it at −4.08 against a true −2.30 after 16 trials.

## Speed

| What | v3 | v4 | Ratio |
|---|---|---|---|
| Kenya, 10,000 agents, 2000–2020, best of 4 (shim, so v3's pregnancy) | 14.21 s | 13.05 s | 0.92 (Oct 17, on a quiet machine: 0.79) |
| FPsim interventions baseline, best of 3 | 5.6–6.9 s | 4.5–5.0 s | about 0.75 |
| `import starsim` / with `fpsim` | 3.95 / 4.64 s | 3.53 / 3.88 s | 0.89 / 0.84 |
| `fp.Deaths` / FPsim's education step | 1.12 / 0.95 s | 0.78 / 0.71 s | 0.70 / 0.75 |

Results by seed are identical to October's on both versions for the coded runs. `import starsim` loads neither JAX nor Optuna. In FPsim's own code on v4, distribution draws are 25% of the run (`rvs` 6.9 s, `process_pars` 3.9 s over 31k calls, the new lazy sync 1.3 s).

## Keep

- **`new()` rebuilding the subclass from its call** (FP7): it made every run tool work on `fp.Sim` with no change to FPsim.
- **The behavior vocabulary**: counters, clocks, `expit` over `ss.ppl`, `p=` tables with a `to` axis normalized by v4, `dur=0` and `ss.on()`. Together they removed every function from the declared pregnancy and contraception.
- **The fixes to dwell timing** (FP16, FP17) and to branch modifiers under every solver (FP13, FP15): FPsim's semi-Markov choice is right as agents and as an ODE.
- **Refusals that name the fix**: `TimeOutsideSim` at `stop`, `BadSplit`, `MissingDimension`, `ShapeError` with level names, `CapabilityError` on the way to compartments, the S5.9 message.
- **`ss.Deaths(lifetable=)` by year and sex**, and `ss.Deaths('senegal')`.
- **The declared `ss.Pregnancy`'s seams** (`conception_rate()`, `ss.modify` of its transitions, `since_birth`, callbacks, `ss.Lactation`): enough for FPsim to rebuild `FPmod` on.

## Suggested order

1. **FP40 (the decision) and FP41, FP42.** Whether FPsim rewrites `FPmod` or v4 keeps a coded base decides what the port is. Whichever: refuse overrides of removed methods, restore the index check or extend the script, a guide row.
2. **FP57.** About 15 lines, and the declared forms go from 2.6–107 times slower to about level.
3. **The silent answers FP43–FP56**, each with a contract test: FP43, FP44, FP45, FP48, FP49, FP54, FP55 are a few lines each; FP52 (refuse first), FP51, FP53, FP50 and FP56 are S to M.
4. **FP23's rest and FP60–FP65**: the empty-suggestion crash, coded parameters, `ss.by` arithmetic, the crashes.
5. **Migration rows and the script** (FP12's hint, FP66, FP67, FP68), then FP58, FP59, FP71, FP72.

## Index of new findings

| ID | Finding | Kind | Part's ID |
|---|---|---|---|
| FP40 | Starsim 3's coded `ss.Pregnancy` removed; `FPmod` can't run; not in the guide | Dead end | FW1, pregnancy |
| FP41 | Overrides of removed `ss.Pregnancy` methods silently ignored | Silent | PG3 |
| FP42 | The v3-index warning removed; the script misses writes in helpers; the skill still promises a warning | Silent | PG1 |
| FP43 | `embryos_per_pregnancy=1` makes no children | Silent | PG2 |
| FP44 | `ss.birth(per=)` of a transition to `ss.null` makes no children | Silent | PG6 |
| FP45 | `ss.mother(ss.ppl.age)` reads 0 for newborns | Silent | PG7 |
| FP46 | `ss.birth(per=)` loses about 2.5% of children with dwells | Silent | PG4 |
| FP47 | Newborns wait a step before a `dur=0` split | Silent | PG8 |
| FP48 | `ss.histogram` draws uniform(0, 1) in the engine; `ss.attr(ss.random())` crashes | Silent | PO1, PO3 |
| FP49 | Traits not drawn for people born to a coded module | Silent | PO7 |
| FP50 | `move_to()` in `init_post()` backfills a false history | Silent | PO6 |
| FP51 | ODE loses 9–11% of people when crossed with a fixed dwell | Silent | IV1 |
| FP52 | `ss.change`/`ss.modify` on a `p=` table does nothing | Silent | IV2, CT5 |
| FP53 | Weibull with shape < 1 as one exponential stage under the ODE, chosen by `auto` | Silent | CT4 |
| FP54 | A modifier on a declared module's plain parameter does nothing | Silent | CT1 |
| FP55 | A dwell under half a step ends in its first step; misleading `TriggerLoop` | Silent | CT2 |
| FP56 | `hash()`/`diff()` of a coded sim change when it runs | Silent | FW4 |
| FP57 | `ss.by.lookup()` rebuilds its table every call | Speed | PG9, CT3, PO2 |
| FP58 | Declared toy 2.1× slower than Oct 17 (rate exits redrawn) | Speed | IV7 |
| FP59 | `ss.years(array)` scans elements in Python | Speed | FW7, PO9 |
| FP60 | `ss.by` arithmetic; named wrappers needed | Friction | PG5 |
| FP61 | Modifier amount over another module's states crashes in the run | Crash | IV3 |
| FP62 | `ss.attr(..., on=X.exit)` crashes | Crash | PO4 |
| FP63 | Dwell expressions over `ss.ppl` or attributes crash with births | Crash | PO5 |
| FP64 | Rows over `to` must name every destination, against the spec | Spec drift | IV6 |
| FP65 | `to_code()` with an unnamed table | Crash | CT6 |
| FP66 | int32 UIDs against int64 `ss.dtypes.int`, undocumented | Crash | FW2 |
| FP67 | `to_df(resample=)` not flagged by the script; `use_years` ignored | Migration | FW3, PO8 |
| FP68 | The S5.9 message keeps a unit trap (20-year breastfeeding) | Friction | PG11 |
| FP69 | `init=dict(A=1.0)` is one person | Friction | PG10, CT7 (duplicate of MA34) |
| FP70 | `Delivery(move=)` takes names only | Friction | IV4 |
| FP71 | `describe()` calls a behavior model a disease | Docs | IV8, PG12 |
| FP72 | `diff()` returns None; `ss.choice` loses its unit; stale CHANGELOG line; age-group note | Friction | FW5, FW6, PO10, FP39 |
