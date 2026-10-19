# FPsim: what v4 should change to make the port a joy

Written 2026-10-18, at commit `1a5b7a77` on the `v4` branch of `ssv4`. That is later than the `03741ba2` used for [stisim_review.md](stisim_review.md) and [hpvsim_review.md](hpvsim_review.md), and v4 changed a lot between them (`0b30ab26`, "major bug review"): the time convention within a step, and the removal of several Starsim 3 names. The `ssv4` working tree was being edited during the work, so everything ran on a clean worktree of `1a5b7a77`. FPsim 3.6.0 (`1817b043`, the `v4` branch of `fpsim`) was used as a real model to test v4. Five agents each took one part of FPsim:
- pregnancy (`FPmod`, a subclass of `ss.Pregnancy`);
- contraception (methods, durations of use, the regression-based choice model);
- interventions and scenarios;
- population (people's covariates, deaths, education, location data);
- the framework (`fp.Sim`, calibration, `MultiSim`, the unchanged test suite and speed).

Each agent ran its part unchanged under v4, wrote a v4 version where one made sense, and compared the two over several seeds against Starsim 3.7.2 on the same interpreter. Scripts and outputs are in the `fpsim` repo under `v4port/<part>/`, with FPsim's own bugs in `v4port/README.md`. I reran the main new findings myself, and all of them reproduced.

FPsim is the first test model that is not an infection. It is demography and behavior: conception, pregnancy outcomes, postpartum, contraceptive choice and switching, education. So it tests how far v4's declared vocabulary reaches beyond diseases, and how well v4 keeps a model written as code first-class (which the [README](README.md) says it must). New findings get IDs FP1–FP39. Everything else is cross-referenced to the HPVsim review (HP), the STIsim review (ST), the merged findings (L, D, X, R, P), and [new-findings.md](new-findings.md) (N). Duplicates get one line each.

## In brief

**What works:**
- **Unchanged FPsim runs on v4 once two removed names are spelled the v4 way, and it is faster.** With a two-line shim (`ss.probperyear` and fractional `verbose`), 55 of 61 tests pass. Every FPsim intervention gives the same effect as on v3 over 6 seeds, and the headline numbers agree over 4 to 40 seeds (Kenya births 12,475 against 12,385; mCPR 0.441 against 0.440). It runs 1.2 to 1.5 times faster, as HPVsim did, because v3's per-call `Dist.jump()` is gone:
  - a Kenya sim in 4.25 s against 5.38 s;
  - the suite's tests in 74 s against 99 s, summed;
  - `fp.Calibration` in 84 s against 129 s.
- **v4's run tools could replace most of FPsim's framework code**, once `sim.new()` works on a subclassed sim (FP7, a 16-line fix):

  | FPsim | Lines | v4 | Lines | Agreement |
  |---|---|---|---|---|
  | `fp.Calibration` | 350 | `sim.calibrate(gof=<FPsim's mismatch>)` | 12 | Best exposure factor 1.49 against 1.58; mismatch 5.6 against 5.4; 86 s against 84 s |
  | `fp.Scenarios` | 213 | `ss.parallel(baseline=..., scen=..., n_runs=)` and `diff()` | 1 per scenario | Adds the repeats `fp.Scenarios` refuses |
  | Parameter routing in `fp.Sim` | 111 | Name-based `pars=` | 0 | Only `rate_units` is in two modules |
  | `fp.Deaths` | 76 | `ss.Deaths(death_rate=table)` | 8 | Identical run for run |

- **Declared versions of the parts work, at a fifth to a half of the lines, and agree with FPsim where they model the same thing:**

  | Part | Declared | FPsim (v3) | Agreement |
  |---|---|---|---|
  | Pregnancy (life course, outcomes, offspring) | 115 lines | 484, plus 502 in `ss.Pregnancy` | Conceptions and births within 1–2%; stillbirths within 10% |
  | Contraceptive choice (`StandardChoice`) | 92 | 468 | mCPR within 3–6% on a fixed population; births within 0.6% in the full sim |
  | Education | 91 | 213 | Within 1.6 SE of FPsim once FPsim's dropout bug is fixed |
  | Seven intervention scenarios, on a declared toy | 21 | 543, plus 230 for `Scenarios` | Same direction everywhere; same size for three of six |

- **They gain what FPsim never had:** a flow for every decision, counts by method and state, results by urban and age with no analyzer, `describe()` of every transition, an ODE of the method module (0.1 s against 1.3 s), and refusals that name why a model can't be an ODE. Common random numbers make "add a method mid-run" exact: a method declared closed changes nothing until it opens.
- **Several FPsim bugs can't happen in the declared forms,** because per-person rates are evaluated by UID and first decisions come after initialization: dropout using another woman's probability, a first contraceptive decision that ignores education, and newborns first deciding at age 10.

**What stops the port being a joy, in five groups:**

1. **v4's new time convention silently breaks unchanged coded modules** (FP1–FP6). `sim.advance()` now moves `ti` between `step()` and `update_results()`, and `finish_step()` runs after the results. `FPmod` was written to v3's convention, so its infant mortality rate is 0 at every step (it was 213), its total births drop every stillbirth, and FPsim's outcome test checks an empty cohort and passes. The annual resample also bins by calendar year, so the first year has 11 steps (FP3). No error or warning, and `ss.Pregnancy` itself had to be patched for the same thing in two places.
2. **Removed Starsim 3 names break every package, with no migration path** (FP10–FP12). Unchanged FPsim fails 55 of 61 tests on two removals alone. At this commit unchanged HPVsim fails most of its tests too (from `ss.year`, `ss.routine_vx`, `ss.MaternalNet` and `annualize`), where at `03741ba2` it passed 359 of 368. The migration guide says modules written as code "run unchanged", and v4's own `to_code()` still writes the removed `ss.probperyear`.
3. **The subclassed sim: `sim.new()` on an `fp.Sim` silently returns an empty sim** (FP7). What was a refusal or a crash for STIsim and HPVsim (ST10, HP12) is now a silent wrong answer: `ss.parallel` of an FPsim sim runs empty sims, and `calibrate()` crashes with an unrelated message. This blocks every run tool for FPsim, and its fix is the one that covers all three packages.
4. **Eight new silent wrong answers in the declared engine** (FP13–FP20), which an FPsim-style model hits at once:
   - the ODE and binomial solvers ignore a change to a parameter used in a branch share;
   - a function in a rate sees a parameter's base value, so `ss.change` on it does nothing;
   - a remainder branch beside a time-series branch becomes a competing exit;
   - `where=` on a dwell is checked only at entry;
   - a competing hazard during a k-step dwell acts for k − 1 steps;
   - `offspring=` doesn't turn aging on;
   - a dimension with numeric levels empties its top level;
   - stratifying by an existing property overwrites it.

   Each is 1 to 20 lines to fix.
5. **Behavior models need vocabulary that diseases didn't** (FP24–FP33). Parity is a counter. Postpartum rules depend on time since an event. Contraceptive choice is a logistic regression times a switching matrix by age and postpartum state, decided when a spell ends. Without these, the declared versions are agents-only and opaque to `describe()` and `calibrate()`. So pregnancy and contraceptive choice should stay code for now. What a coded module needs from v4 is smaller, and mostly missing: a modifier that reaches its `rel_sus` (D23, FP23), and a stated place to count a step's events (FP1).

**Counts:** 39 new findings (FP1–FP39), about 25 of them related to an existing finding, plus about 20 known findings that FPsim hits again. Two known findings are fixed at `1a5b7a77`: ST1 (newcomers start in the first state) and HP13 (`diff()` on coded sims). Groups 1 to 4 are small fixes plus docs: about a week of v4 work. Group 5 is design work, shared with HPVsim's per-person values (HP19) and branching rules (HP1, HP18).

## What a joyful FPsim looks like

The target, in proposed spellings (the names marked `# proposed` don't exist today). Pregnancy stays coded, and v4 reaches into it. Contraception, education and deaths are declared. Interventions, scenarios and calibration are v4's.

```python
import starsim as ss, fpsim as fp

loc = fp.data('senegal')                                                  # FPsim's loaders, listed by ss.Requires
methods = loc.methods                                                     # names, efficacy, fitted durations, switching tables

contra = ss.Disease(
    states  = ss.states(['young', 'never', 'none'] + methods.names),
    choose  = ss.Transition(['never', 'none'] + methods.names, ['none'] + methods.names,
                            dur    = methods.dur_use,                    # fitted distributions by method and age
                            p      = ss.logistic(intercept=ss.par(loc.intercept, bounds=[-3, 3]),        # proposed: a declared regression (FP26)
                                                 terms=dict(parity=..., urban=..., edu=...), age=ss.spline(knots, coefs), trend=...)
                                     * ss.table(methods.switching, axes=['from', 'to', 'age', 'postpartum']),  # proposed: a tabular parameter (FP27)
                            decide = 'exit', where=fp.fecund & ~fp.pregnant),                         # proposed: decided and checked at exit (HP1, FP16)
    restart = ss.Transition(methods.names, 'choose', on=fp.delivery, delay=ss.months([1, 6])),       # proposed: on an event of a coded module (ST13, FP25)
    name = 'contra',
)
edu = ss.Disease(..., attainment=ss.value(accumulate=['in_school']), objective=ss.by(urban=[hist0, hist1]))  # proposed (HP19, FP30)
preg = fp.FPmod(location='senegal')                                       # stays coded; counts its events with self.count() (FP1)
efficacy = ss.modify(preg.rel_sus, multiply=1 - methods.efficacy, by=contra)   # reaches a coded module (D23, FP23)

sim = fp.Sim(location='senegal', connectors=[contra, edu, efficacy], data=loc.targets)
fit = sim.calibrate(gof=fp.mismatch, workers=8)                             # works once new() replays fp.Sim (FP7)
ss.parallel(baseline = fit.sim,
            more_use = fit.sim.new(interventions=ss.change(contra.pars.intercept, at=2025, by=0.5)),
            new_inj  = fit.sim.new(interventions=ss.change(contra.pars.open_new_inj, at=2025, to=1)),
            n_runs=20).diff('baseline').plot()
```

Today the parts of this that run need these workarounds:
- `FPmod` patched to count at `ti + 1` and form its ratios at the end (FP1, FP2);
- the two removed names spelled the v4 way (FP10);
- `fp.Sim` rebuilt by hand, or `ss.Sim.__new__` patched, for every run tool (FP7);
- ten lambdas per transition, plus a cache, for each switching row (FP27);
- a function for the regression, which rules out the ODE and `calibrate()` (FP26);
- "as soon as" transitions at `rate=1e6` with `where=` lambdas for postpartum events (ST18, FP25);
- a per-step function for parity, which can't tell twins from singles (FP24);
- a 6-line override of `FPmod.get_contra_eff` for efficacy (FP23);
- a coded `step()` for years of schooling (HP19);
- a closed method opened with `ss.change` for `add_method`.

## The port today, and after the changes

| Task | Today | After the changes below |
|---|---|---|
| FPsim unchanged | 55 of 61 tests fail on two removed names; with them fixed, infant mortality is 0 and stillbirths drop out of births (FP1, FP2) | Runs, with an error naming each removed name's v4 spelling and a warning on index-0 flows |
| Annual outputs (ASFR, crude birth rate, population) | The first year has 11 steps; stocks are year-end values where v3 gave means (FP3, FP4) | A period includes its end; the change is in the migration guide |
| Interventions at `stop` | Accepted and never applied (FP5) | Refused, naming the last step |
| Scenarios, sweeps, calibration | `new()` returns an empty sim (FP7); `calibrate()` crashes (FP8); `hash()`/`diff()` crash on FPsim's durations (FP9) | One fix covers `fp.Sim`, `sti.Sim` and `hpv.Sim` |
| A change to the switching matrix under an ODE | Ignored (FP13) | Applied |
| A change to a regression coefficient | Ignored if the regression is a function (FP14) | Applied |
| "A method available from 2010" as data | Every share wrong (FP15) | Right |
| "Re-decide when the spell ends, if eligible" | Eligibility checked only at entry (FP16) | Checked at exit |
| Miscarriage during a 1-step trimester | Never happens (FP17) | Acts for the whole dwell |
| Births by `offspring=` | Nobody ages; newborns have no mother (FP18) | Aging on; `parent` set |
| Wealth quintile, urban as dimensions | The top quintile is empty; urban overwritten (FP19, FP20) | Right |
| Contraceptive efficacy on conception | `UnmodifiableTarget` with a wrong suggestion (FP23) | `ss.modify(preg.rel_sus, ...)` |
| Parity, postpartum timing, regression, switching matrix | Functions, agents only (FP24–FP27) | Declared, readable by `describe()`, fittable by `calibrate()` |

## What should change in v4

### 1. The time convention: make coded modules safe under it

`0b30ab26` changed when `ti` moves. In v4 now, `sim.ti` is the start of the step during `step()`, and `sim.advance()` moves it to the end before `update_results()` and the analyzers. `Module.finish_step()` runs after the results at that same index, and no longer moves `ti` itself, as v3's did. The changelog states this ("a result written during a step goes at `ti + 1`"). Declared modules and `ss.Pregnancy` follow it: the base class was patched in two places (maternal deaths at `ti + 1`, ASFR from `ti_delivery == ti - 1`). Coded modules written for v3 don't, and they go wrong silently:

- **FP1 (silent): a flow written during `step()` at `self.ti`, and a ratio formed in `update_results()` at `self.ti`, gives 0.** `FPmod._post_delivery()` writes `infant_deaths[self.ti]` at the step's start index, and `update_results()` divides `infant_deaths[ti]` by `births[ti]` at the end index, which hasn't been written yet. So `imr` is 0 at every step. The deaths are counted, one index early: index 0 is non-zero, which the v4 convention says it never is, and the last index is always 0. FPsim's `abortions`, `short_intervals`, `method_failures` and `switchers` are one step out of line with its `births` in the same way. A minimal repro gives a ratio of about 0.1 per step on v3 and 0.0 at every step on v4 (`pregnancy/repro_step_result_index.py`). With `FPmod` patched to write at `ti + 1`, IMR is 178.9 ± 6.0 over 40 seeds against v3's 176.3 ± 6.1. Fix:
  - at finalize, warn when a flow result of a coded module is non-zero at index 0 ("was it written during `step()` at `self.ti`? A step's events go at `ti + 1`"), about 15 lines;
  - give modules a `self.count(name, n)` buffer that is written at `update_results()`, as `ss.Pregnancy._counts` already does, about 20 lines;
  - a row in the migration guide.
- **FP2 (silent): what a coded module records in `finish_step()` never reaches `update_results()`.** `ss.Pregnancy.finish_step()` counts stillbirths at `ti`. In v3 that went to the next index, which the next `update_results()` read. In v4 it lands after the results are taken, so `FPmod.total_births = births + stillbirths` never includes them: 1,185 against 1,280 for one seed (`pregnancy/repro_finish_step_result.py`: 100 per step on v4, 105 on v3). FP1's check doesn't catch this one. Fix: say in the migration guide that `finish_step()` now runs after the results, and that derived results go in `finalize_results()`.
- **X13 again (duplicate):** an analyzer comparing `ti_*` stamps with `sim.ti` matches nobody, since the stamps are the step's start and `sim.ti` its end. `test_pregnant_women`'s cohort is 0 against 316 under v3, so "all resolved" passes vacuously, and the stillbirth assertion is just the first to fail. STIsim found the same `- 1`. A `sim.ti_step` (the start of the step just taken) would make it hard to get wrong.
- **FP3 (silent): `resample('year')` on a dated sim bins by the calendar year of each time point.** The step that ends on 1 January counts in the next year: the first year has 11 steps, a one-point "year" is added at the end, and a stock's year-end value is 1 December's. Deaths by year are 960, 963, 964, 84 dated, against 1,048, 946, 977 undated, which are the true sums (`population/repro_calendar_resample.py`). This contradicts the docstring ("periods include their end"). Every FPsim annual output goes through it. Fix: give each time point the year of `t − ε` in `Result.periods()`, 3 lines, plus a test that dated and undated sims resample alike.
- **FP4: unchanged code's resampled stocks change meaning, from the annual mean to the year-end value, with no "Regression" line** in the changelog or a row in the migration guide. This is R3's agreed change, and its consequence for FPsim is `pop_size_mean` +1.1% and `pop_growth_rate_mean` −9% in the baseline test. Fix: the line and the row ("for v3's annual means, use `kind='ratio'`").
- **FP5 (silent): `ss.change(at=stop)`, `ss.modify(start=stop)` and `ss.Delivery(at=[stop])` are accepted, but no step starts at `stop`, so they never apply.** v3 ran a step at `stop`, which is why two FPsim tests fail on v4: `add_method(year=2002)` and `change_people_state(years=2010)` both time themselves at `stop`. `at=stop + 0.5` raises `TimeOutsideSim`; `at=stop` should too, naming the last step's start. 5 lines. Related to N10.
- **FP6 (silent): `sim.loop.insert()` after `sim.init()` runs the function twice per step**, because the plan is now built lazily and replays the insertions, including the new one. 20 calls for 11 time points (`pregnancy/repro_loop_insert_twice.py`). Fix: build the plan before appending, 1 line.
- **N10 again (duplicate):** v4 takes one step fewer than v3 over the same dates, which makes FPsim's total births 2.2% lower (0.6% per step), and shifts a rising mCPR by about 3.5%.

FP1 and FP2 are the most consequential findings here. They turn unchanged Starsim 3 code that looks right into wrong results, and the only FPsim test that notices is the one exact baseline comparison. 14 of FPsim's 60 tests assert nothing, and most of the rest check a direction or a wide band, so a suite passing under v4 doesn't show the model is the same. The bar for "coded modules run unchanged" should be a warning whenever v3-convention code writes where v4 won't read.

### 2. Removed Starsim 3 names: an error that names the fix, and the migration guide

- **FP10: removed names break every package, with a bare `AttributeError` and no row in the migration guide.** FPsim fails 55 of 61 tests on two: `ss.probperyear` (now `ss.prob(x, 'year')`, bit for bit the same) and `verbose=1/12` or `-1` (now 1 or 0). The `verbose` error is good, and says what to use. The `ss.probperyear` one is `module 'starsim' has no attribute 'probperyear'`. At this commit, unchanged HPVsim also fails most of its tests, from `ss.year` (about 200 errors), `ss.probperyear`, `ss.routine_vx`, `ss.MaternalNet` and `annualize` on a result; at `03741ba2` it passed 359 of 368. The migration guide's overview says modules written as code "run unchanged". Only the `migrate-starsim3` skill lists some of these. Fix: a module-level `__getattr__` in `starsim/__init__.py` that names the v4 spelling for every removed name (`probper*`, `freqper*`, `ss.day` to `ss.year`, `routine_vx`, ...), and a row for each in `docs/migration/v3_v4.md`. About 25 lines. Not aliases, since the spec chose one spelling.
- **FP11 (silent at write time): v4 still writes and recommends `ss.probperyear`.** `sim.to_code()` writes `ss.par(ss.probperyear(0.5))`, which fails when run. The spec reader looks up `ss.probperyear` and `ss.freqperyear`, so it fails on any `prob` or `freq` value with a unit. Three messages in `time.py` say "Are you using ss.prob() instead of e.g. ss.probperyear()?". Fix: write `ss.prob(0.5, 'year')`, and add a test that runs `exec(sim.to_code())` for a model with each time type. About 8 lines. Related to R10.
- **FP12: `msim.reduce()` is a dead end, and a faithful port changes its meaning.** v3's `reduce()` pooled the runs of `test_multisim`'s three exposure scenarios into one band. v4 keeps each sim as its own scenario, so per-scenario quantiles have low equal to high, and the ported assertion would fail. Fix: `MultiSim.__getattr__` for `reduce`, `mean` and `median` that names `msim.results.<module>.<result>.quantiles()` and says scenarios are kept apart. About 8 lines. Related to R2 and R20.

### 3. The subclassed sim

- **FP7 (silent): `sim.new()` on an `fp.Sim` returns an empty plain `ss.Sim`** (10,000 agents, default dates, no modules), which runs. So `ss.parallel(<an initialized or run fp.Sim>, n_runs=2)` returns two empty sims and no error, and `fp.Sim(...).new(interventions=...)` fails only when a result is read. v3 raised `AlreadyRunError` for a sim that had run, and ran an initialized one correctly. The cause is ST10's and HP12's: `fp.Sim` calls `super().__init__(pars=None)` and builds everything afterwards, so `sim.inputs == {}`, which gets past `new()`'s check. What was a refusal or a crash for those packages is now silent. Fix: record `type(self)` and the call in `ss.Sim.__new__`, and replay them in `new()` (`framework/newfix.py`, 16 lines). With it, every run tool works on unchanged FPsim: `new()`, sweeps by parameter name, `ss.parallel(...).diff()`, and `calibrate()` (which recovers an exposure factor of 1.509 against a true 1.5). It covers `fp.Sim`, `sti.Sim` and `hpv.Sim` with no change to the packages, and gives `describe()` its `given` (ST10's guessed dates and agents). Meanwhile, refuse `new()` when `inputs` is empty on a subclass, 2 lines. The spec should state the contract ("a sim is rebuilt by calling its class again with its arguments, with overrides"), with a conformance test of a three-line subclass that builds after `super().__init__()`.
- **FP8: `fp.Sim.calibrate()` crashes with `AttributeError: 'NoneType' object has no attribute 'name'`**, because the probe sim is FP7's empty sim and the data column matches no result. Add a check that names the unmatched column and the results there are, 3 lines. Related to R13 and R16.
- **FP9: `to_spec()`, and so `hash()` and `ss.diff(sim1, sim2)`, crash on a distribution whose duration is an array** (`only 0-dimensional arrays can be converted to Python scalars`). FPsim's location durations are like this, so these are unavailable for every `fp.Sim`. Write the value as a list, 2 lines. Related to R8 and R29.
- **D23 again (duplicate):** `stratify=` on `fp.Sim` silently makes nothing (0 of 72 results), as for HPVsim.
- **N20 again (duplicate, new instance):** FPsim fits a 13-knot `exposure_age` curve with a smoothness penalty. Two spellings crash, and `ss.par(<the whole dict>, bounds=)` is accepted and would fit the dict as one scalar. v4 should resolve `ss.par` inside a coded module's nested dicts and lists (named `exposure_age.rel_exp[3]`), refuse `bounds=` on a non-numeric value, and give a way to state a prior or penalty across several parameters.

`fp.Experiment` should stay FPsim code. Its targets are distributions over people at the end of a run (age by parity, the birth-spacing histogram, method mix by age), which don't fit "data attached to a result by name". `gof=` as a function of the sim is the right seam, and it already works.

### 4. Fix the silent wrong answers in the declared engine

FPsim-style models hit these at once, because they change shares more than rates, and depend on eligibility and on short dwells.

- **FP13 (silent): the ODE and binomial solvers ignore `ss.change` or `ss.modify` on a parameter used in a branch `p=`.** A share set to go from 0.2 to 0.8 stays at 0.200 (ODE) and 0.198 (binomial), while agents give 0.793. A modifier on a rate parameter is applied by all three, and so is one on the branch transition itself (`interventions/repro_ode_branch_change.py`). In the declared toy, opening a new method gives it a share of 0.58 as agents and 0.00 as an ODE. Every FPsim matrix change is of this kind. Fix: in the check for one whole-run kernel call, treat a modifier on any parameter in a `p=` expression as a branch modifier, so the run is stepped or `p` is tabulated per step. About 10 lines, plus a contract test that a modified `p=` parameter gives the same shares under every solver.
- **FP14 (silent): a function in a rate or `p=` that reads a parameter gets its base value, so `ss.change` on that parameter does nothing.** FPsim's choice model is a logistic regression in code, so the natural v4 form of `change_initiation_prob` (a change to the intercept) is a no-op: 3,531 starts after 2005 with and without the change, where 0 are expected; the same parameter used directly as the rate goes from 4,573 to 0 (`interventions/repro_change_par_in_function.py`). Fix: have `float(par)` and `par.value` return the effective value during a run, or pass `pars` to functions; at least, warn when a modified parameter is read by nothing the engine evaluates. About 20 lines. Related to D23.
- **FP15 (silent): `p=None` (the remainder) beside a branch whose `p` is a time series becomes a competing timed exit, so every share is wrong.** With B = 0.2 × (0 before 2005, 1 after), C = 0.3 and D the remainder, C gets 0.153 where 0.30 is expected (`interventions/probe_branch_ts.py`). Constants with a remainder are right. This is the natural spelling of "a method available from 2010". Fix: take a `p=None` sibling as the remainder whenever any sibling has `p=`, worked out at run time, about 10 lines. Related to HP21.
- **FP16 (silent): `where=` on a transition with a dwell is evaluated only when the dwell is drawn.** People who aren't eligible at entry never leave, even after they become eligible: of people eligible from age 30 with a 1-year dwell, 0.00 of those aged 29.0–29.9 at the start have left after 4 years (`contraception/probes/probe_small.py` 3b). FPsim's rule is "re-decide when the duration ends, if fecund and not pregnant", so the declared contraception lets women over 50 go on deciding (7,150 decisions at 35 and over, against FPsim's 6,170). Fix: re-check `where=` when the exit falls due, and hold the person until it holds; or refuse `where=` on a dwell exit. About 20 lines. Related to ST6, which is the `p=` case.
- **FP17 (silent): a competing hazard out of a state with a fixed dwell of k steps acts for k − 1 steps as agents**, not in the step of entry and not in the step the exit is due. A 1-step dwell blocks it entirely: at a hazard of 1 a year, the share leaving by it is 0.000 for a 1-month dwell (0.080 expected), and 0.153 for 3 months (0.221) (`pregnancy/repro_dwell_hazard_steps.py`). FPsim's trimesters are 1 to 3 steps, so miscarriage and stillbirth would be a third to all too low. Fix: draw the hazards before the scheduled exits, so k steps of dwell give k steps of exposure. About 10 lines, plus a test against the ODE. Related to HP2 and L14.
- **FP18 (silent): `offspring=` doesn't turn aging on**, so in a model whose births are `offspring=`, nobody ages: mean age 33.99 before and after 5 years, and 9,521 newborns all aged 0. Newborns from `offspring=` also have no parent (`people.parent == -1`), so nothing mother–child can be declared: infant mortality by mother's age, breastfeeding, MTCT, `MaternalNet`. Fix: count `offspring=` among the births that turn aging on, 1 line; set `parent` in `add_offspring()`, about 3 lines.
- **FP19 (silent): a dimension with numeric levels stores each person's level index but groups people as if the levels were bin edges.** `stratify(wealthquintile=[1, 2, 3, 4, 5], split=[0.2]*5)` stores 0–4 on people, counts [3,616, 1,793, 1,796, 1,785, 0], and `adjust=` for level 5 reaches nobody (`population/repro_dimension_values.py`). String levels are right. FPsim's regression multiplies `wealthquintile` by a coefficient, so it would shift too. Fix: treat numeric levels of a non-age dimension as labels, store and group consistently. About 20 lines. Related to N3.
- **FP20 (silent): stratifying by a property `People` already has replaces its values**, though the spec says "if it already exists, stratifying uses it". FPsim sets `urban` from data; `stratify(urban=[False, True])` makes everyone rural, and a continuous `wealth` becomes level indices. Fix: bin an existing continuous property, map an existing categorical one, and refuse `split=` for either. About 20 lines.
- **FP21 (silent): a bare `dt=1/12` with `start=2000, stop=2020` becomes 1/12 of a month once a typed duration is in months:** 2,881 steps instead of 241, 12 times slower, and `describe()` says "every 0.0833333 months". FPsim's durations are in months and its step is 1/12 of a year, so a declared FPsim falls straight in. It is a counterexample to L1's merged proposal ("a bare number is in the model's unit, from the typed values"). Fix: with year-valued dates, a bare `dt` is in years, or warn when the units differ. About 5 lines.
- **FP22 (silent): a function-valued time parameter that returns a bare number is read in the sim's unit.** FPsim's duration callbacks return months, so copied verbatim they make every duration of use 12 times too long. Warn, as v4 already does for bare-number parameters. Related to L1 and L19.
- **HP1 again (duplicate, two new instances):** FPsim's choice must be decided when a spell ends, from the covariates and the matrix in force then (a year trend of 0.046 in log-odds, durations of up to 100 months). It works only because a multi-destination transition with a dwell happens to draw its destination at exit, which the spec doesn't promise. Spelled as separate `p=` transitions (the import skill's mapping), a matrix change reaches women only at their next entry: about 21 years late for implants. Behavior models need the exit rule documented as much as HPV does.

### 5. Reaching coded modules, and the vocabulary behavior models need

**Coded modules.** Pregnancy and contraceptive choice should stay code for now (below), and v4 should make that easy:
- **FP23: v4's modifiers can't reach a coded demographics module, and the errors mislead.** `ss.modify('pregnancy.rel_sus', ...)` raises `UnmodifiableTarget ... did you mean dummy.rel_sus?`, a wrong suggestion; with no declared module in the sim, `ss.change` and `ss.modify` crash with `min() iterable argument is empty` from `did_you_mean`; and `ss.Delivery(move=...)` says "the sim has no declared diseases" for a model that has none by design. `ss.Pregnancy` never reads `rel_sus`, so a contraceptive's efficacy has no path to conception; the workaround is a 6-line override of `FPmod.get_contra_eff`. Fix: D23(a) for any module's plain parameters and `rel_sus` (a 23-line prototype, `interventions/coded_change.py`, is identical to FPsim's `change_par` and `change_initiation_prob` at every step over 3 seeds), `Pregnancy.make_p_conceive` reading `rel_sus` as `.effective`, a guard for empty options in `did_you_mean` (1 line), and a delivery message that doesn't assume a disease. About 50 lines. Duplicate of D23; the crashes and the suggestion are new.
- **ST11 again (duplicate):** there is no public way for coded code to put agents in a declared state. The declared education sets flags and `ti_<state>` itself, and calls `engine.clear_exit()`.

**Vocabulary.** The declared pregnancy, contraception and education each agree with FPsim, but only with functions in every slot that matters, which makes them agents-only, opaque to `describe()`, and not fittable by `calibrate()`. What they need:
- **FP24: a counter.** Parity is a count of births that conception depends on. There is no relative move (`move=dict(parity=+1)`), so a parity dimension needs one copy of each delivery branch per level (3 branches × 8 levels). The workaround, a per-step function reading `ti_PP`, can't tell twins (+2) from singles, since the branch taken isn't recorded per person. Medium.
- **FP25: rates that depend on time since an event.** Postpartum sexual activity, lactational amenorrhea and birth-spacing preference depend on months since the last live birth, which outlives the postpartum state. Today this is a function reading `ti_PP`, and the third-trimester stillbirth hazard reads the engine's `ti_exit_P3`. `ss.time_since(transition)` in rates and `ss.by()` would declare it; under compartments it needs HP18's tunnel states. Related to HP18 and ST13 (events of another module as triggers, which postpartum decisions need too).
- **FP26: a declared regression.** FPsim's probability of use is a logistic function of an intercept, a 3-knot age spline and its interaction with ever-use, parity, urban, wealth, education and a year trend. As a function, `describe()` prints `<lambda>` ten times per transition and its coefficients can't be `ss.par`s, which is exactly what FPsim calibrates (`prob_use_intercept`, `prob_use_trend_par`). Proposed: `ss.logistic(intercept=ss.par(...), terms=dict(...), age=ss.spline(knots, coefs))` over people's attributes and states, printed as a formula and evaluated on groups when the covariates are dimensions. About 150 lines.
- **FP27: a tabular parameter, and a `p=` that gives a whole row.** FPsim's switching matrix is 10 × 9 per age group and postpartum state. Today `p=` takes one function per destination (a function returning the row crashes with `input operand has more dimensions than allowed`), so each transition needs ten lambdas and a cache, and `update_methods(method_choice_pars=...)` becomes about 70 `ss.change` calls on scalar weights. Proposed: `ss.table(..., axes=['from', 'to', 'age', 'postpartum'])`, addressable by entry or row with a renormalization rule declared once. Related to HP22.
- **FP28: a delivery to a mix of destinations, and a modifier over several transitions.** `change_initiation` without a target method ("start a method from the usual mix") has no form except one delivery per method with thinned coverage, and `add_method` without parameters in `p=` takes 20 modifiers. `move=dict(to=[...], p=...)` and a modifier over a set of transitions (HP22's proposal) would cover both.
- **FP29: `ss.transfer` rejects `p=` and `dur=`** (`takes one dimension ... not ['method', 'p']`), though `05-dimensions.md` says transfers take both. A method as a dimension is the natural way to give a declared pregnancy its efficacy (`adjust={'conception': [1, 1 − eff, ...]}`), but needs a dwell and a choice of destination, which only `ss.Transition` has. Parse `p=` as an argument (a few lines), then allow `'a -> b | c'` with a dwell (medium).
- **FP30: a distribution that varies by level, and `ss.histogram` as a dwell.** FPsim's education objective is a histogram of years that depends on urban. `dur=ss.by(urban=[h0, h1])` raises `float() argument must be ... not 'histogram'`, and `ss.histogram` refuses a unit. About 40 lines. Related to N21. (HP19 again for the rest of education: years of schooling accumulated over several spells, frozen on leaving. This is the fourth model to need a per-person value, after HPVsim's immunity, STIsim's CD4 and odin's auxiliary variable.)
- **FP31: a death rate that trends in time, written as rate × trend, has three spellings that fail**: `ss.peryear(r) * ts` (`IncompatibleTypes`), and `ts * ss.peryear(r)` and `ss.by(age=...) * ts`, which raise a `DuplicateDeclaration` that doesn't name the cause. `idioms.md` recommends "rate * period". Make them expressions, as `ss.par` already is. About 15 lines. Related to N20 and N21.
- **FP32: a parameter inside a modifier's amount can't be changed.** In `multiply=1 - ss.par(0.983, name='eff_inj')` the parameter belongs to no module, so `ss.change` on it raises `UnmodifiableTarget`, and `update_methods(eff=)` becomes a second, stacked modifier. Give an `ss.par` amount a path. Small.
- **FP33: `ss.death(lifetable=)` has no year or sex, and `ss.death` needs a declared disease to host it.** An age at death drawn once can't follow a mortality trend (crude death rate 7.1 in 2000 against FPsim's 11.2, and 11.4 in 2019 against 9.5). Document it as a stationary table, let `ss.Deaths` take a life table, and name the transition in the refusal (today it says `"->" in demographics= needs a declared disease`). Related to the life-table row in [new-findings.md](new-findings.md).

**Known findings FPsim hits again, one line each:**
- HP15: initial states can't depend on sex or age, so men need their own state and the data-driven start is coded `init_post()`.
- ST18: "until delivery plus one month" and "until six months postpartum" have no dwell form; each is an as-soon-as transition at `rate=1e6` with a `where=` lambda, which then prints three notes about the ODE in a model that can only run as agents (ST6, ST16, ST26).
- N19: conditions on `FPmod`'s coded arrays have to be lambdas, which work.
- HP20: a branching flow has no destination axis, so new users of each method aren't a result.
- D15: mCPR among women 15 to 49 needs an age stratification and a `male` state; sex isn't a dimension unless declared.
- HP16, HP2: an immediate split crashes, and pass-through states hold people for a step (about 1% of the conception gap).
- HP33: `ss.Deaths(death_rate='senegal')` fails, though `starsim_data` holds Senegal's deaths in `ss.Deaths`'s own format.
- ST26, HP29: contraception and pregnancy as two declared modules with a cross-module `where=` refuse the ODE; the method module alone runs as one.
- R2: a sweep labels its runs scenario `''`.

### 6. Small things

- **FP34: declared demography and choice are slower than the coded modules they replace.** Deaths by single year of age × sex × year take 4.8 s against 0.37 s for `fp.Deaths` (13 times; 200 time series evaluated each step, plus regrouping as people age), education 1.8 times, and contraception 1.6 times, 65% of it in `finish_moves` working out branch shares. One cause is cheap: a function-valued distribution parameter is introspected with `inspect.signature` on every call (21,924 calls, 0.6 s of 5.1 s); cache the arity, 5 lines. Related to N29 and HP30.
- **FP35:** a dwell given as `ss.choice` of durations (the form `ss.Pregnancy`'s own `dur_pregnancy` takes) crashes the declared engine (`loop of ufunc does not support argument 0 of type months`), and `ss.choice(..., unit=)` is refused. About 5 lines.
- **FP36:** `ss.Transition(A + B, C, rate=...)` crashes in its error path (`argument of type 'BinOp' is not iterable`, in `did_you_mean`); the operator form and a list work. A few lines.
- **FP37:** `disease.solvers()` after `sim.init()` crashes (`Cannot add object "ab" since already present`, or `Cannot modify attribute "rel_sus"`). Small.
- **FP38: `describe()` hides what an FPsim-style model depends on:** a function `where=` isn't printed (none of 10 conditional transitions shows its condition), function-valued scales print as `<function ... at 0x...>`, `p=` prints values not parameter names (so which entry is `w_inj_impl` can't be seen), it doesn't say when a branch is decided, and for an `fp.Sim` it says "10,000 people [guessed]" and "[guessed: ages as the world's population in 2024]" though FPsim gave both. Related to L27 and ST10.
- **FP39: friction:**
  - `move=dict(ever='True -> False')` raises `'"T" is not a level of "ever"'`;
  - `Delivery(move='none -> impl')` needs `disease=` though only one module has those states;
  - `diff()` between sims whose states differ silently drops the results the baseline lacks;
  - `stratify(age=[0, 15, 120])` makes a third level `120+`, unlike `np.digitize`;
  - counts by age group use ages from before the step's aging (164 people under 10 counted after 10 years with nobody under 10, at `dt=1`);
  - an initialized distribution from another sim used as a default fails much later with an index error;
  - `check_verbose` accepts 3 or 7 while its message says 0, 1 or 2.

## Before removing the v3 vocabulary (decision 2)

FPsim shows what removal costs when the coded-module API is meant to keep working. Two removed names fail 55 of 61 of its tests, and five fail most of HPVsim's at this commit, though the fixes are a handful of lines in each package. The cost isn't the edit, it's finding it: a bare `AttributeError`, no migration-guide row, and v4's own `to_code()` and messages still using the old name (FP10, FP11). More serious is the time convention (FP1, FP2): it changes what correct-looking v3 code computes, silently. Before more of the v3 surface goes:
- every removed name should raise an error that names its v4 spelling, and be a row in the migration guide;
- the time convention should come with a check that catches v3-convention flows (index 0 non-zero) and a sentence in the guide about `finish_step()`;
- a release gate: the unchanged suites of FPsim, HPVsim and STIsim, with the exact baseline comparisons each has, run at every commit that changes the coded-module API. FPsim's 55 passes hid IMR going to 0; only `test_baseline` caught it.

## Speed and inefficiencies

| What | v3 | v4 | Ratio |
|---|---|---|---|
| Kenya, 10,000 agents, 2000–2020, best of 4 after a warm-up | 5.38 s | 4.25 s | 1.27 |
| Unchanged suite, summed best per-test time over the 55 tests that pass on both | 99.3 s | 74.1 s | 1.34 |
| `fp.Calibration`, 20 trials, 1,000 agents | 129 s | 84 s | 1.5 (one run each) |
| Interventions baseline, 5,000 agents, best of 3 | 5.35 s | 3.76 s | 1.42 |
| `fp.Deaths`, 20,000 agents × 240 steps | 0.49 s | 0.37 s | 1.3 |
| Declared deaths by age × sex × year | | 4.8 s | 13 times slower than `fp.Deaths` (FP34) |
| Declared contraception against FPsim's on v4 | | 6.3 s against 4.0 s | 1.6 times slower (FP34) |
| Declared education against FPsim's on v4 | | 0.98 s against 0.55 s | 1.8 times slower |
| Method module as an ODE against agents, 5,000 agents × 20 years | | 0.1 s against 1.3 s | 13 times faster |

The machine was shared (load 5 to 44 on 20 cores), so differences under about 30% are noise. The gain on unchanged FPsim is in Starsim's distributions: `Dist.rvs` cumulative time is 1.93 s against 3.09 s in one Kenya run, since v3's per-call `Dist.jump()` (1.14 s over 29,720 calls) is gone. FPsim's own code takes about the same time on both. No v4 regression in speed showed up. The declared versions are slower than the coded ones because every slot is a function evaluated per agent; FP26, FP27 and the arity cache (FP34) would remove most of that.

## Keep

- **Coded modules as first-class.** `FPmod`, `StandardChoice`, FPsim's interventions and `fp.Experiment` run unchanged with the same answers once FP1, FP2 and FP10 are handled. Declared and coded modules work together in one `fp.Sim`: a declared education reads the coded pregnancy through `where=` and feeds the coded contraception regression, and a declared contraception drives coded conception through common random numbers that keep the two independent.
- **Name-based `pars=`, `ss.parallel(baseline=..., n_runs=)` with `diff()`, and `calibrate(gof=)`.** Together they replace about 680 lines of FPsim's framework and add repeats to its scenarios.
- **`ss.change` and `ss.modify` keeping the base.** FPsim keeps `orig_val` and `'reset'` by hand in every class, and its list form crashes (FPsim bug list).
- **Common random numbers for "add a method".** A method declared closed is identical at every step to a sim without it, which makes a mid-run addition exact.
- **Declared results.** A flow per decision, counts by state and level, ratios such as mCPR as declared results readable by age, with no analyzer.
- **Multi-destination transitions with a dwell, self-loops and function-valued scales.** These express FPsim's semi-Markov choice in 92 lines, and the dwell is rounded to whole steps exactly as FPsim does.
- **The unit warning for bare numbers** (`dur_postpartum was given as a number with no unit, so it is read in months`). FPsim's location files trip it, and v3 accepted them silently.
- **`ss.people('Senegal')` and `starsim_data`'s WPP deaths**, which match FPsim's inputs and could replace its 35-line spline loader.

## Suggested order

1. **FP7 (`new()` replays the subclass's call).** One 16-line fix that unblocks every run tool for FPsim, STIsim and HPVsim.
2. **FP1, FP2, FP3 and FP10 (the time convention and removed names).** A warning on index-0 flows, a `count()` buffer, the resample fix, errors that name each removed name's v4 spelling, migration-guide rows, and the downstream suites as a release gate. These decide whether "coded modules run unchanged" is true.
3. **FP13–FP20 (silent engine answers).** 1 to 20 lines each, with a contract test each.
4. **FP23 (D23 for coded demographics) and FP14.** With these, FPsim's interventions are one-line `ss.change` and `ss.modify` calls on its coded modules.
5. **FP5, FP6, FP8, FP9, FP11, FP12, FP21, FP22, FP34–FP39.** Small fixes and messages.
6. **Design pass on behavior vocabulary** (FP24–FP30), together with HP19 (per-person values) and HP1/HP18 (when a branch is decided). Until then, pregnancy and contraceptive choice stay code.

## Index of new findings

| ID | Finding | Kind | Section | Agent's ID | Cross-reference |
|---|---|---|---|---|---|
| FP1 | A coded flow written in `step()` at `self.ti` and divided in `update_results()` gives 0 (IMR 213 → 0) | Silent | 1 | pregnancy FP1 | Related to N10, X13 |
| FP2 | What `finish_step()` records never reaches `update_results()` (stillbirths drop out of births) | Silent | 1 | pregnancy FP2 | Related to N10, X13 |
| FP3 | `resample('year')` on dated sims puts 1 January in the next year | Silent | 1 | population P3 | Related to R3, N10 |
| FP4 | Resampled stocks change from annual mean to year-end, undocumented | Docs | 1 | population P4 | Related to R3 |
| FP5 | `at=stop` is accepted and never applies | Silent | 1 | interventions V4 | Related to N10 |
| FP6 | `loop.insert()` after `init()` runs twice per step | Silent | 1 | pregnancy FP4 | New |
| FP7 | `sim.new()` on `fp.Sim` returns an empty sim | Silent | 3 | framework F1, interventions V7 | Related to ST10, HP12 |
| FP8 | `calibrate()` on `fp.Sim` crashes with an unrelated message | Crash | 3 | framework F2 | Related to R13, R16 |
| FP9 | `to_spec()`, `hash()`, `diff()` crash on an array duration | Crash | 3 | framework F4 | Related to R8, R29 |
| FP10 | Removed v3 names break packages with a bare error and no migration row | Dead end | 2 | framework F6 | New |
| FP11 | `to_code()`, the spec reader and messages still use `ss.probperyear` | Silent at write | 2 | framework F3 | Related to R10 |
| FP12 | `msim.reduce()` is gone, and a port changes its meaning | Dead end | 2 | framework F5 | Related to R2, R20 |
| FP13 | ODE and binomial ignore a change to a `p=` parameter | Silent | 4 | interventions V1 | New |
| FP14 | A function reading a parameter sees its base value | Silent | 4 | interventions V2 | Related to D23 |
| FP15 | A remainder beside a time-series branch becomes a competing exit | Silent | 4 | interventions V3 | Related to HP21 |
| FP16 | `where=` on a dwell exit is checked only at entry | Silent | 4 | contraception C1 | Related to ST6 |
| FP17 | A competing hazard during a k-step dwell acts for k − 1 steps | Silent | 4 | pregnancy FP5 | Related to HP2, L14 |
| FP18 | `offspring=` doesn't turn aging on, and newborns have no parent | Silent | 4 | pregnancy FP6 | New |
| FP19 | Numeric dimension levels: the top level is empty | Silent | 4 | population P1 | Related to N3 |
| FP20 | Stratifying by an existing property overwrites it | Silent | 4 | population P2 | New |
| FP21 | A bare `dt` is read in months when durations are | Silent | 4 | interventions V9 | Related to L1 |
| FP22 | A function returning a bare time is read in the sim's unit | Silent | 4 | contraception C12 | Related to L1, L19 |
| FP23 | Modifiers can't reach a coded demographics module; misleading errors | Dead end | 5 | contraception C3, interventions V6 | Duplicate of D23; crashes new |
| FP24 | No counter (parity) | Vocabulary | 5 | pregnancy FP10 | New |
| FP25 | No rate by time since an event | Vocabulary | 5 | pregnancy FP11 | Related to HP18, ST13 |
| FP26 | No declared regression | Vocabulary | 5 | contraception C4 | New |
| FP27 | No tabular parameter or row-valued `p=` | Vocabulary | 5 | contraception C5, interventions V12c | Related to HP22 |
| FP28 | No delivery to a mix, no modifier over several transitions | Vocabulary | 5 | interventions V12a, b | Related to HP22 |
| FP29 | `ss.transfer` rejects `p=` and `dur=` | Spec drift | 5 | contraception C6 | New |
| FP30 | No distribution by level; `ss.histogram` can't be a dwell | Vocabulary | 5 | population P6 | Related to N21 |
| FP31 | Rate × time series fails in three spellings | Friction | 5 | population P5 | Related to N20, N21 |
| FP32 | A parameter in a modifier's amount can't be changed | Vocabulary | 5 | interventions V10 | New |
| FP33 | Life-table deaths have no year or sex | Vocabulary | 5 | population P9 | Related to new-findings |
| FP34 | Declared demography and choice are 1.6–13 times slower; `inspect` per call | Speed | 6 | population P8, P15; contraception C7, C8 | Related to N29, HP30 |
| FP35 | `ss.choice` durations as a dwell crash | Crash | 6 | pregnancy FP7 | New |
| FP36 | `ss.Transition(A + B, ...)` crashes | Crash | 6 | pregnancy FP8, contraception C9 | New |
| FP37 | `solvers()` after `init()` crashes | Crash | 6 | contraception C10 | New |
| FP38 | `describe()` hides function conditions and parameter names | Friction | 6 | contraception C11, interventions V13, population P13 | Related to L27, ST10 |
| FP39 | Small frictions (boolean moves, `disease=`, `diff()` drops results, age edges, stale age counts, reused distributions, `verbose=3`) | Friction | 6 | various | New |
