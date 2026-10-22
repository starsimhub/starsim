# tbsim v1.0.0 against today's Starsim v4

Written 2026-10-21. A recheck of the tbsim port (`/home/cliffk/idm/tbsim`, branch `v1.0.0`, `28d1bed`) against `ssv4` at `ee5d1fb0` plus the uncommitted tree, in the `v4` env. The tbsim side of this (what tbsim must change) is in tbsim's `v4_port_oct21.md`; this file is what Starsim should take from it. Research only: neither repo was changed. The fixes were made in a scratch copy of tbsim.

Method: run tbsim's tests as they are; run `python -m starsim.interop.migrate` on a copy; fix each break by hand in the copy, rerunning the suite, until it passed; then run the seven example scripts and search the docs for old spellings.

## Summary

- **As it is, tbsim does not run**: 49 of 54 tests fail, all at the first error (`ss.State(infectious=True)`).
- **After the migrate script and about 75 lines of hand changes**, 53 of 54 pass, and all seven examples run. The one left fails because its tolerance is tight for 5,000 agents (the seeds spread ±22%), which isn't a Starsim problem. The 12 changes that were needed are listed in tbsim's file.
- **The silent wrong answer that [strains.md](strains.md) predicted is real, and confirmed end to end.** Without `kind='strain'`, `tbsim.LSHTMResistant` differs from its reference ODE by 100% in the `DS+R` compartments, and six resistance tests fail on numbers, none with an error. With `kind='strain'`, all of it passes again (the ODE is within 1.1e-4 of the reference). This is the strongest case for strains.md's item 1: infer `kind='strain'` from `'+'` levels.
- **The migrate script has two bugs that break or silently change working code**: it renames a downstream function's own `eligible=` parameter, and it deletes `on_positive=`/`on_negative=`, leaving code that runs without its follow-ups. Of the 12 kinds of change tbsim needed, it made or flagged only two.
- **Smaller, for v4**: let `interventions=` take nested lists (cascade helpers now return several steps); give transitions event memory (`.since`); add hints to three bare errors; make `Results.flatten()` keep case as `to_df()` does.
- **Working as intended, and the v4 checks earned their keep**: the NaN check on modifiers found a real tbsim bug (BMI is NaN for children under 2), and the other new refusals (bounds on values, a bare year in a dated sim, `|` of states in a rate) all had clear messages.

## 1. The strain dimension that silently becomes plain (SWA)

tbsim's `add_resistance()` returns `stratify(resistance=[...], split=, adjust=, states=)` with no `kind=` (`tbsim/resistance.py:240`). Since S7.1 removed the name hints, `kind` is `'strain'` only with `immunity=` (`dimensions.py:46`), so the dimension is plain: co-carriage is off, superinfection does nothing, and new infections take their level from the split rather than from their source.

What it did to tbsim's own suite, with every other fix in place:

| Test | Without `kind='strain'` | With it |
|---|---|---|
| `test_models.py::test_resistant_ode` (ODE against `two_strain_ode.R`) | largest difference 1.0 (`L_AB` 0.0 against 2.0; `N_AB` 0.2 against 14.4; the `B` compartments about half) | 1.1e-4 |
| `test_compartmental.py::test_two_strain_abm_vs_ode` | "without treatment, the less-fit resistant strain should be out-competed" fails | passes |
| `test_resistance.py`: `test_selection`, `test_superinfection`, `test_interventions`, `test_monitoring`, `test_second_line` | fail on numbers (e.g. superinfections `0.0 > 0`) | pass |

Nothing warns. The spec still promises a note for names like `resistance` (`05-dimensions.md:175`, as strains.md found), and none is printed. Recommendation: do strains.md's item 1 (infer `kind='strain'` when every part of a `'+'` label is itself a level, about 3 lines), with a test that a co-carriage dimension without `kind=` superinfects. tbsim will pass `kind='strain'` as well.

## 2. The migrate script

### Bugs

1. **It renames a downstream function's own parameter.** The rule `\beligible=` → `where=` (`interop/migrate.py:50`) also matches `def care(tb, ..., eligible=None, ...)` (`tbsim/interventions/care.py:114`), so the signature becomes `where=None` while the body still reads `eligible`: `UnboundLocalError` in every use of `tbsim.care()` (6 tests). Fix: rewrite keywords only in calls to Starsim's deliveries (`ss.screen`, `ss.treat`, `ss.vaccinate`, `ss.Delivery`, `ss.transfer`), never in a `def`.
2. **It deletes `on_positive=X` and `on_negative=X`, leaving code that runs without its follow-ups.** In `tbsim_examples/run_tb_interventions.py` it removed both keywords and listed "to do by hand" items; the result runs, screens, and never confirms or treats (`confirm` and `start_tx` are left unused). The script already knows the replacement (its message says "give it `who=confirm.positive`"), so it can write it: add `who=<screen>.positive` to the follow-up's call and add the follow-up to the interventions. If that's too hard to do safely, leave the keyword in place, so the code fails with v4's own error instead of running wrong.
3. **It applies Starsim 3 rewrites to code written for a v4 prerelease.** tbsim v1.0.0 is v4 code, but its `HouseholdStats.step()` got `if ti == 0:` → `if (ti - 1) == 0:` (`analyzers.py:693`), which moves the "initial" age-mixing matrix to after the first step. `ss.Disease` → `ss.Infection` and `verbose=0.02` → `1` were right. Whether code is Starsim 3 or a v4 prerelease can't be told from the text. A `--prerelease` flag that applies only the prerelease renames would avoid it. This one is minor.

### What it didn't cover

Of the 12 kinds of change tbsim needed, the script made one (`ss.Test`, and `eligible=` in calls) and flagged one by hand (`on_positive=`). For the S8.2 migration rows (each is a v4-prerelease or Starsim 3 spelling found in tbsim):

| Old | New | How it fails now |
|---|---|---|
| `ss.State(..., infectious=True)` | drop it: infectiousness comes from the transmission rate | bare `TypeError` (see §3) |
| `ss.death('X ->')`, `ss.birth('-> X')`, `ss.immigration(f'-> {key}')` | the state: `ss.death(S.X)` | good message |
| `A \| B` of states in a rate (e.g. a prevalence denominator) | `A + B` | good message |
| `ss.library.HouseholdNet` | `import starsim.library as ssl`; `ssl.HouseholdNet` | bare `AttributeError` (see §3) |
| `ss.transfer(art=True, coverage=, years=, eligible=)` | `ss.Delivery(move=dict(art=True), coverage=, at=, where=)` | good message (but `eligible=` → `where=` was rewritten first) |
| `years=[...]` on a delivery | `at=[...]` | good message |
| `ss.Treatment(efficacy=1.0, transition='diagnosis')` | `ss.Treatment(moves='diagnosis')` | good message |
| `.diagnosed` of a screen | `.positive` | flagged by hand |
| `res.cum_<flow>` | `res.<flow>.cum` | `KeyNotFoundError` |
| `ss.Sim(copy_inputs=False)` | drop it, and read modules from the sim | `KeyNotFoundError` with the keys listed |
| `ss.utils.match_result_keys(res, key=None)` | `res.flatten()` | bare `AttributeError` |
| `pytest.raises(ss.CodedError, match='E804')` | `pytest.raises(ss.CapabilityError)` (codes are gone) | the regex doesn't match |
| a co-carriage dimension without `kind=` | `kind='strain'` | **silent** (§1) |

## 3. Error messages that need a hint

Everything else tbsim hit said what to do. These three didn't:

- `ss.State('X', infectious=True)`: `TypeError: State.__init__() got an unexpected keyword argument 'infectious'`. Suggest: "infectious= is gone: a state is infectious when it is in a transmission rate (e.g. `rate=beta*I`)".
- `ss.library`: `module "starsim" has no attribute "library"`. Suggest adding "import it: `import starsim.library as ssl`". This one is also order-dependent: `ss.library.X` works if anything in the process has already imported `starsim.library`, so code can pass in one test file and fail in another.
- `ss.utils.match_result_keys`: a bare `AttributeError`; one line pointing to `Results.flatten()` would do.

## 4. Smaller findings for v4

1. **`interventions=` should take nested lists.** With follow-ups as steps of their own, a helper for a cascade returns several steps: `tbsim.care()` now returns `[screen, treat]`, or up to five with `dst=` and `retreatment=`. `ss.Sim(interventions=[tbsim.care(tb), tbsim.monitoring(tb)])` raises "was expecting interventions entry 0 to be ... Intervention ... but was list", so users must write `[*tbsim.care(tb), *tbsim.monitoring(tb)]`. Flattening one level of lists in `interventions=` (and the other module lists) costs a line, and makes every downstream cascade helper simple to use. Recommended.
2. **Transitions have no event memory.** `ss.conditions.as_cond(tb.diagnosis).since` raises ".since is the memory of an event (a delivery, an outcome of one ..., a product, or an ss.Event), not of tb.diagnosis", though a transition is an event in `who=` and `on=`. tbsim's `treated_within()` (who started treatment in the last two years) reads `tb.ti_treatment`, which is now `TimeNotKept` unless `ss.options.keep_times = True` is set globally. With `.since` on transitions it would be a condition, `tb.diagnosis.since <= ss.years(2)`, with the memory kept only for the transitions that are asked about. Worth deciding; if not, the `TimeNotKept` message could suggest a delivery's `.since` instead.
3. **`Results.flatten()` lowercases keys** (`sir_n_s`), while `to_df()` now keeps case (`sir_n_S`). One spelling for both.
4. **The given module silently holds no results.** As decided (S2.6), a sim copies its inputs. tbsim's analyzer test read `az.results` from the analyzer it gave, got an empty odict, and failed with a `KeyError` far from the cause. A cheap guard: a module that was copied into a sim and never run raises on `.results`, saying to use `sim.analyzers.<name>`. Optional.
5. **Arrow strings**: `ss.Delivery(move='S -> I')` and `ss.transfer(risk='low -> high')` take arrows, while `ss.death`, `ss.birth` and `ss.immigration` now refuse `'X ->'`. That may be deliberate (names of levels against state objects), but it's a distinction users will trip on. Note it in the migration skill at least.
6. **A naming question: `ss.Infection` as the base of every disease written as code.** tbsim's `Malnutrition` must now subclass `ss.Infection` (as must the `ss.Infection` docstring's own example, `Anemia`), and `isinstance(x, ss.Disease)` is false for it. Not a bug, but "malnutrition is an infection" reads oddly to modelers.

## 5. Changes that are right, for the migration skill

- **Bounds are checked on values**: `tbsim.TB(beta=0)` (used to turn transmission off) raises `OutOfBounds` because tbsim gives `beta` the bounds `[1, 50]`. tbsim should widen them to `[0, 50]`.
- **The unit is the finest of the typed values**: HIV's `ss.perday` rates with TB's `ss.peryear` made the sim step daily (3,650 steps over 10 years) with a note. It's correct and noted, but costly by surprise; tbsim will give HIV's rates per year.
- **A bare year in a sim with dates is a time since the start**: `ss.vaccinate(..., start=1980)` in a `tbsim.Sim` (whose defaults are dates) raises `TimeOutsideSim` with the hint "write it as a date, '1980'". In a sim with numeric years, `at=[2005]` is the year. This is clear when it fails, but the same code means different things in `ss.Sim(start=2000)` and `tbsim.Sim()`.
- **A NaN modifier raises** (`ClockOutsideDomain`): it found a real tbsim bug, where `lonnroth_bmi_rr()` is NaN for the 3% of people under 2 (the LMS table has `Length`, not `Height`, there). Before, those people's progression rates were silently NaN.

## Not checked

- Speed against the v4 that tbsim was ported on (the full tbsim suite takes about 4.5 minutes at `-n 3` today).
- Rendering tbsim's docs: they were searched for old spellings (counts in tbsim's file), not built.
