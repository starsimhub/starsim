# STIsim on v4, rechecked at `ee5d1fb0`

Written 2026-10-21. This rechecks the October 18 STIsim review ([oct17/stisim_review.md](../oct17/stisim_review.md), at `03741ba2`, IDs ST1–ST27) against `ee5d1fb0`, the committed head of `v4-simplify`. While this was written, `v4-simplify` was merged into `v4` as `88208f57` with one more commit (`30c98710`, decision 18's `ss.SexualNet`); the main checks were rerun there, and [the last section](#update-v4-at-88208f57) gives the differences. STIsim 1.7.1 (`f9af9e8`) was the test model. Five agents each rechecked one part of STIsim (networks; HIV; syphilis; curable STIs and BV; the framework of sim, test and treat, calibration and analyzers) and reran every old script adapted to the new spellings, against Starsim 3.7.2 and against the migrated coded module on v4. I reran each headline finding against the pinned commit.

Scripts and findings are in the `stisim` repo under `v4port/<part>/oct21/` (each folder has a `FINDINGS.md`). That repo's `v4_port_oct21.md` is the companion for STIsim's side. New findings continue the ST numbering (ST28–ST52), and each gives the agent's ID (LEAD-, NET-, HIV-, SYPH-, SB-, FW-) for the evidence.

## In brief

**Most of what blocked a declared STIsim on October 18 is fixed.** Of the 27 ST findings, 19 are fixed and verified by running, 3 are closed by decision, and 5 remain (ST8, ST9 in part, ST16, ST17 in part, ST21 in part):

- people born during a run are placed in a state (ST1);
- per-act transmission is right, in both directions (ST2, ST3);
- cures reach every infected state (ST5);
- `where=` and function `p=` branches work (ST6, ST7);
- `new()` keeps the subclass, so an unchanged `sti.Sim` gets `new()`, sweeps, `diff()`, `ss.modify` on coded arrays and `calibrate()` (ST10, with N19, N20 and R13);
- coded code can move people (`move_to`, ST11);
- pregnancy events, memory of test results and `ss.mother()` replace four syphilis functions (ST13, ST14, ST18);
- `ss.stay` (ST19), two-dimension contact matrices (ST23), disjoint non-exponential exits under the ODE (ST24), and `calibrate()` on the ODE in 31 s rather than 1,063 s (ST27).

The declared ports lost most of their workarounds. Syphilis with care is 85 lines rather than 130, and it agrees with the coded module within about 1.5 SE.

**Unchanged STIsim no longer runs, and the migration covers about half of what it needs:**

- `import stisim` fails, because `ss.Calibration` is gone (ST40).
- After `python -m starsim.interop.migrate`, 106 of 198 tests pass.
- Four kinds of break that the script doesn't touch account for most of the rest (ST41–ST44). With them fixed by hand, and ST28 worked around, 181 pass. The remaining 17 fail on known causes.

**There are twelve new silent wrong answers, two of which affect unchanged STIsim:**

- **ST28: nobody ages in an `sti.Sim`.** A parameter set on `sim.pars` before `init()` is overwritten for `use_aging` (`dimensions.py:719`). Children never reach sexual debut, the network shrinks, and HIV prevalence after 20 years is 0.051 against 0.115 on Starsim 3 and 0.109 on `03741ba2`. The repro needs three lines of plain `ss.Sim`. Any model that sets `use_aging` this way is hit.
- **ST29: a dwell slowed by a small `adjust=` factor never resumes** after a transfer to a faster level. With `adjust=1e-6`, 1.3% leave where 73.5% should. The `03741ba2` workaround stopped working, and only `adjust=0` is safe.
- Six in the declared language (ST30–ST35):
  - resistance acquired `on=drug.failed` doesn't change the level;
  - `init` with 1.0 at a level means one person;
  - `ss.constant(0)` waits a step under the ODE;
  - bare numbers from a rate function are read in the inferred unit;
  - a neonatal death `on=preg.delivery` is counted as a stillbirth;
  - a delivery's results are one index off from flows.
- Four in fitting and migration (ST36–ST39):
  - annual data is compared one year off;
  - a bounded network par passed through `sti.Sim` loses its bounds and isn't fitted;
  - Optuna reports "Converged" for a fit that returned its start;
  - the migration script misses a stored `ti`.

**A speed trap in the new default.** With `compile='auto'`, compartmental kernels run as plain Python unless they are already compiled, or unless the estimated run time exceeds about 20 s. The estimate ignores substeps, so stiff STI ODEs run uncompiled: syphilis takes 266 s against 1.72 s compiled, and SEIS 83 s against 1.19 s (ST50). Compiled, they are as fast as at `03741ba2`.

**Unchanged STIsim, once migrated, is as fast and as right as before.** It is 1.5 to 1.7 times faster than on Starsim 3: five diseases take 4.5–5.2 s against 7.5–7.6 s, and Zimbabwe 10.4 s against 16.1 s. That is the same speed as at `03741ba2` in an interleaved A/B on a quiet machine. With the aging workaround, every slice matches Starsim 3 within noise: Zimbabwe over 10 seeds, syphilis over 15, the curable STIs and BV over 6, and the analyzers.

**Counts:** 26 new findings (ST28–ST53): 12 silent wrong answers, 5 migration dead ends (one of them only at `88208f57`), 8 crashes or performance traps, and 1 missing piece. About 20 more points of friction or stale docs are listed in [Small things](#small-things).

## Status of the October 18 findings

| ID | Status | Evidence (in `stisim:v4port/<part>/oct21/`) |
|---|---|---|
| ST1 newborns in no state | **Fixed** | `hiv/repro_newborns_state.py`: 0 of 535 (`ss.Pregnancy`) and 0 of 826 (`ss.Births`) children in no state (was 738 of 738); syphilis embryos start in the first state |
| ST2 per-act probability | **Fixed** | `networks/beta_per_act_dt.py`: `ss.peract` 0.384/0.377 monthly/weekly (SE 0.005); a bare number on a network with acts is per act; `ss.probpermonth` there is `WrongDimension` |
| ST3 per-direction pair | **Fixed** | `syphilis/probe_direction_beta2.py`: f2m 0.282/0.475/0.729 at 2/4/8 acts (exact 0.278/0.478/0.728) |
| ST4 extra step on entry | **Won't fix (decided)**; people act for the right dwell | `syphilis/probe_generation.py`: generation exactly 4 steps; only `n_` counts carry the extra step. It still costs trichomoniasis 14% of its infections against v3 (`seis_bv/tv_gap.py`): men infected by persistently infected partners are reinfected as soon as they clear, so counts go as 1/duration (6.21 against 5.73 person-months per infection). Worth a line in the docs, since it changes infection counts, not prevalence |
| ST5 cure from one transition's sources | **Fixed** | `framework/test_and_treat.py`: CT 0.900, NG 0.898 of treated cured with no `transition=` (was 47% and 74%) |
| ST6 `where=` with `p=` | **Fixed** | `syphilis/probe_dur_function2.py`: 2,000 of 2,000 move, each at their own step |
| ST7 function `p=` remainder | **Fixed** | `seis_bv/branch_remainder.py`: 0.897 for every kind of `p=` (was 0.440) |
| ST8 `update_results()` before the first step | **Still**; marked done in `oct19/implementation.md` | `framework/first_step_update_results.py`: `STITreatment`, `GonorrheaTreatment`, `SymptomaticTesting`, `SyphTx` crash with `AttributeError: ... no attribute "outcomes"`. There is no guard and no message about `ti=0`; the migrate script and skill (`SKILL.md` line 91) mention `ti=0` only for analyzers' `step()`, not interventions' `update_results()` |
| ST9 coded code reading a declared `ss.par` | **Partly fixed** | `float(par)` and `disease.value(name)` exist, but `array*(1 - disease.pars.eff_condom)` still builds an expression and crashes (`networks/eff_condom_par.py`). A bare unused keyword now raises `UnknownParameter`, with a good message naming `define_pars()` |
| ST10 subclass `new()` | **Fixed** | `framework/sti_sim_v4.py`: `type(sim.new())` is `stisim.sim.Sim`, identical results; sweeps, `diff()`, `calibrate()` work on an unchanged `sti.Sim`. `sim.given` is still empty, which is the cause of ST28 and of `describe()`'s "[guessed]" |
| ST11 coded code moving people | **Fixed** (`move_to`) | `framework/syndromic_bridge.py`; but moves count in no flow (Small things) |
| ST12 stock targets | **Won't fix (decided)** | Spec 08 "Rejected" (2026-10-20, "STIsim keeps its own"); ART to a target stays two `deliver=` functions, agents only |
| ST13 pregnancy events | **Fixed** | `who=preg.conception`, `on=ss.state('pregnancy.delivery')`, `on=ss.state('pregnancy.birth')` all work (`syphilis/compare_care2.py`); but see ST34 |
| ST14 test results per person | **Fixed** | `who=preg.birth & ss.mother(anc.positive.since <= ss.years(1), born=True)`: 53.8 newborn tests for 70.6 ANC positives |
| ST15 `on=` a union | **Fixed** | `who=ng.careseek \| ct.careseek` (`framework/test_and_treat.py`) |
| ST16 delivery outcome as an event | **Spelled, but broken** | `on=drug.failed` with `move=` is accepted, and its flow counts, but nobody moves (ST30) |
| ST17 route-specific infectiousness | **Partly fixed** | `dict(route=b)*(A + 0.5*B)` runs. Different weights per route crash with `UFuncTypeError` (ST49); syphilis no longer needs them (`ss.mother()` replaces the maternal route) |
| ST18 exact dwell until an event | **Fixed** | `dur=ss.constant(fn)` exact; `dur=fn` warns |
| ST19 branch to stay | **Fixed** | `ss.stay` works; not yet in `03-states-and-transitions.md` |
| ST20 stratified automatic `prevalence` | **Fixed by design** | Declared diseases have no automatic `prevalence`; a declared one selects by age and sex |
| ST21 condoms on networks | **Partly fixed** | `condoms` on `ss.SexualNetwork` edges and `ss.modify(..., share=)`; no condom use by partnership type or risk pair |
| ST22 two-mode network | **Fixed at `ee5d1fb0`; regressed at `88208f57`** | `PartnerNet(between=('fsw', 'client'))` worked at `ee5d1fb0`; decision 18 (`88208f57`) removed `PartnerNet`, and `between=` with it (not found in `networks.py`) |
| ST23 matrix over two dimensions | **Fixed** | `networks/riskgroup_ode.py`: 0.0818 both ways |
| ST24 one non-exponential exit | **Fixed for disjoint `where=`**; competing exits agents only (decided, `support_matrix.md`) | `seis_bv/disjoint_where.py`; BV's exits, disjoint through `adjust=` zeros, are still refused because the check can't see it |
| ST25 `ss.constant(0)` crash | **No crash, now silently a step late** under compartments (ST32) | `seis_bv/zero_dwell.py` |
| ST26 substeps, "immediately" | **Mostly fixed** | BV ODE 1.6 s per simulated year compiled (was 10 s); `dur=0` immediate under every solver; no note on substep counts; `idioms.md` "As soon as" stale |
| ST27 `calibrate()` on an ODE | **Fixed** | `seis_bv/seis_calibrate.py`: 31 s (was 1,063 s); Optuna on agents 34 s (was 549 s) |

Other old findings, rechecked:

- **Fixed:**
  - N7, the bare beta per act;
  - N19, conditions on coded arrays (`where=hiv.cd4 < 500`);
  - N20, `ss.par` inside `ss.peract` in a beta dict;
  - R13, `ss.par` on coded pars, with bounds kept;
  - D11/D12, age stratification no longer changes the epidemic;
  - D15, a sex dimension as agents;
  - D20, callable `capacity=` per step;
  - L4, an unconnected seeded state;
  - X1, a lognormal on a transfer;
  - X13, the same `ti` everywhere within a step;
  - D23, most verbs on coded modules (`what_modifies()` and `describe()` remain);
  - X7, `adjust=0` now follows the person, but see ST29.
- **Still open:**
  - N15, the network can't be declared: 30.8% prevalence against 9.1%, and man–woman partnerships can't take `match=` or `ss.by` degrees;
  - L1, `dt=ss.months(1)` without `unit=` still gives prevalence 1.000 against 0.149, with only a note;
  - D29, the 12 accuracy series per test;
  - #38, `ss.modify` with `fn(sim, uids)`;
  - #35, continuous CD4 on ART, which is now worse: declared CD4 < 50 in 2010 is 0.008 against 0.042 in v3;
  - #36, network conditions;
  - #49, the syphilis ODE: as fast as before compiled, but 160 times slower by default (ST50).
- **Closed by decision:** D27, inline product names (`deferred.md`, NR13).

## New findings

### Silent wrong answers

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| ST28 (LEAD-1) | **A sim parameter set on `sim.pars` after `ss.Sim()` and before `init()` is overwritten for `use_aging`, so nobody ages in any `sti.Sim`.** `sti.Sim.init()` sets `use_aging = True` and calls `ss.Sim.init()`; `dimensions.py:719` then sets it to `sim.has_demography()` because `'use_aging' not in sim.given`. Regression since `03741ba2` | `stisim:v4port/lead/oct21/repro_pars_before_init.py` (plain `ss.Sim`): `use_aging` False after init, age change 0.0 over 5 years (5.0 at `03741ba2`, 6.0 on v3; `rand_seed` set the same way is kept). HIV-only `sti.Sim`, 5,000 agents, 20 years, 5 seeds: prevalence 0.1154 ± 0.0088 (v3), 0.1085 ± 0.0099 (`03741ba2`), **0.0512 ± 0.0032** (`ee5d1fb0`), 0.1065 ± 0.0048 with `sim.given.append('use_aging')`. Children under 10 at the start are 4.9 on average after 20 years, none have debuted, partnerships fall from 1,414 to 1,111 (`hiv_diag.py`, `net_debut_diag.py`). Not narrowed by bisection: STIsim doesn't run on the first 17 first-parent commits after `03741ba2` | Test `sim.pars.use_aging is None` (as `SimPars.validate_demographics()` does), not `given`. A few lines. Add a test that a par set before `init()` survives |
| ST29 (HIV-1) | **A dwell slowed by `adjust=` with a small nonzero factor never resumes** after a transfer to a faster level. The likely cause is that exits beyond the 16-bit `StepArr` range become "never" and are not rescaled. `adjust=0` and factors such as 0.5 work | `hiv/probe_dwell_follow.py`: share leaving `I` by 2002, 0.735 expected: 0.723 at 1e-3, 0.400 at 2e-4, 0.238 at 1e-4, **0.013 at 1e-6**; the `03741ba2` workaround (a no-op `ss.modify`) no longer changes anything. The declared ART model as written leaves 291 people off ART stuck in latency | Keep far exits rescalable (store a remaining fraction, or a float), or raise when an exit overflows. Related to X7 and S5.6 |
| ST30 (SB-1) | **`move=` on a transition triggered `on=<product>.failed` is dropped**: the flow is counted, the state move happens, the level stays. This is the plan's spelling for resistance at treatment failure | `seis_bv/move_probe.py` (rerun): acquisition flows 644 / 627 / 615 / 614 as agents and ODE, R level 0 throughout; with `rate=` instead of `on=` the level moves (3,990 / 3,962) | Apply `move=` on event-triggered transitions as on rate-driven ones. Plan S7.4 |
| ST31 (SB-3, SYPH-1) | **In `init=`, a value of 1.0 is one person** (`count = vals >= 1`), so "everyone here starts in X" has no spelling. At `03741ba2`, 1.0 meant everyone. BV's `init=dict(CST1=ss.by(stable=[1.0, 0, 0]), CST3=ss.by(stable=[0, 1.0, 0]), ...)` now starts women whose stable CST is 3 in CST 1; the old syphilis probes rerun unchanged seed one person instead of everyone | `seis_bv/bv_init.py` (rerun): CST1 995, CST3 0, against 240 / 755 with 0.999999; the error otherwise offered ("Say where the rest start, e.g. M=0.9993") misleads. Inconsistent with `init_prev=1.0` | A value at most 1 is a share, at least inside an `ss.by` and for floats (`1.0` vs `1`), or a typed `ss.count()`. The migration skill should mention it |
| ST32 (SB-2) | **`ss.constant(0)` is a one-step delay under the ODE and binomial solvers**, with no note; as agents, and with `dur=0`, it is immediate | `seis_bv/zero_dwell.py`: n_I at 5 years 42,088 (ODE) and 31,679 (binomial) against 51,280 (agents) and 49,889 (`dur=0`); endemic level 18% low | Treat a zero constant as `dur=0`. Was ST25's crash |
| ST33 (SB-4) | **A rate function returning bare numbers is read in the model's inferred unit**, which is weeks once any typed value in the model is in weeks | `seis_bv/bv_exact_unit.py`: BV CST 4 0.967 against 0.605 with `unit='year'`; the only signal is a `UnitInferred` note | Read a function's bare numbers per year, as elsewhere, or require a typed return. Related to FP21, L1 |
| ST34 (SYPH-2) | **A neonatal death `on=preg.delivery` removes the fetus before it is born**, so `ss.Pregnancy` counts a stillbirth; `on=ss.state('pregnancy.birth')` gives a birth and then a neonatal death. Nothing says which event is before birth for the child | `syphilis/probe_nnd_timing.py`: 81 forced NNDs; `on=delivery` gives `stillbirth` 82, `neonatal_deaths` 0; `on=birth` gives 1 and 81 | A line in the `ss.Pregnancy` docstring and `08-interventions.md`; or a warning when a fetal death fires on `delivery` |
| ST35 (SYPH-6) | **A delivery's results and a transition's flow land at different indices for the same step**: a delivery on `preg.conception` gave `[7, 5, 8, 12, 6, 6, 0]` against the `conception` flow `[0, 7, 5, 8, 12, 6, 6]`. Sums agree; cascade tables and dated comparisons are a step off | `syphilis/probe_ti_offset.py` | One convention for where a step's counts go (X13's single convention, for deliveries) |
| ST36 (FW-2) | **Annual flow data is compared one year off by `calibrate()`**: a row `year=2010` is matched with the flow over (2009, 2010], while STIsim's `to_df(resample='year')` and UNAIDS-style data label calendar years [2010, 2011). Stocks are interpolated at Jan 1, where STIsim used the year's mean or end | `framework/calibrate_v4.py D`: at the true parameters, calendar-year data scores 0.591 for `hiv.new_infections` (the series shifted a year); the Zimbabwe data has a `year` column | `ss.data(df, period='year')`, or read an integer `year` column as calendar periods (flows over [Y, Y+1), stocks mid-year), with a note |
| ST37 (FW-3) | **A bounded parameter passed to `sti.Sim` for its network is silently not fitted.** `sti.Sim` builds its network pars as a full `ss.Pars` of defaults, so `Pars.update()` unwraps the `ss.par` into that object's `_bounded`, and the network gets a float. Disease pars route fine. Half of this is STIsim's routing, but v4's `Pars.update(other_pars)` dropping bounds silently is the trap | `framework/probe_bounded_routing.py` (rerun): `sti.Sim(f1_conc=ss.par(...))` fitted `[]`; a `StructuredSexual(f1_conc=ss.par)` instance fits; a two-parameter fit reported one, at its bound | Carry `_bounded` through `Pars.update(Pars)`, or warn when a bounded par is dropped |
| ST38 (FW-4) | **Optuna fits of agent sims report "Converged" when they returned the start or a lucky trial**: "best improved < 1% over the last quarter" holds whenever an early trial stays best, and the mismatch rerun at the best values can be worse than before the fit | `framework/calibrate_v4.py A`: 2,000 agents, 30 trials, truth 0.05: best 0.0201, mismatch 0.367 before and 0.474 after, "Converged: True … improved by 0.0%". Zimbabwe, 10 trials: returned the initial values though trial 3 scored better; the enqueued start was proposed twice | Don't call a fit converged when its best is the start or an early trial, and report the rerun mismatch beside the trial score. Related to `13-inference.md` (smoothed best), R5 |
| ST39 (FW-1) | **The migration script rewrites `== ti` but not a stored `ti`**: STIsim's partnership analyzer records `ti_expired=ti` (now the step's end) and compares it later with `ti_formed` stamped in `step()` (the start), so every duration is one step long | `framework/analyzers_shift_check.py`: recorded minus true duration +1 for 5,261 of 5,352 (0 for 4,631 of 4,773 on v3) | Flag assignments of `ti` in analyzers and `update_results()`, as comparisons are; a line in `SKILL.md` |

### Migration dead ends

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| ST40 (LEAD-2, FW-5) | **`ss.Calibration` was removed with no replacement class, so `import stisim` fails** (`class Calibration(ss.Calibration)`), as HPVsim's does (`sexual_network.md`). The script lists it "to do by hand" with good advice, but a package that can't import can't be migrated piece by piece. What STIsim's calibration becomes works: `ss.par(bounds=)`, `data=`, `sim.calibrate()`, with `build_fn` no longer needed, `check_fn` as `gof=`, `make_calib_sims` as `ss.parallel(fit.sim.new(), pars=sets)`. Missing: worker processes, Optuna storage and resume, results saved per trial, `prune_fn` | `framework/calibrate_v4.py` (5.1 s per evaluation against 5.45 s per trial on v3) | Either keep a thin `ss.fit.Calibration` that maps the old arguments onto `calibrate()` (against "no baggage", but it is the one thing that stops two downstream packages importing), or have the script comment out the subclass and point at the mapping. `n_workers` and `storage=` are worth having anyway |
| ST41 (LEAD-3) | **`ss.Infection` is no longer a subclass of `ss.Disease`, and there is no public base for "any disease"** (`DiseaseBase` isn't exported). The script rewrites `class X(ss.Disease)` to `ss.Infection` but not `isinstance(x, ss.Disease)`, so `sti.Sim` refuses every coded disease (66 test failures); changing the check to `ss.Infection` refuses declared ones. `ss.Infection` now also defines `susceptible` and `infected`, so `sti.SimpleBV`, which defines them too, crashes (`Cannot add "susceptible" ... already present`, SB-11) | Lead test runs; `seis_bv/v3_extras.py` | Export one base (`ss.DiseaseBase`, or make `isinstance(x, ss.Disease)` true for both), and have the script rewrite the check and flag redefined states |
| ST42 (LEAD-4, HIV-3, SB-12) | **Starsim 3's `ss.Pregnancy` attributes are gone, the script rewrites none, and the docs say they are kept.** `pregnant`, `breastfeeding`, `ti_pregnant`, `ti_delivery`, `tri1_uids`–`tri3_uids`, `set_ptb()`; results `stillbirths` → `stillbirth`, `miscarriages` → `miscarriage`. 40 test failures after the script. `migrate.py:45` says the new pregnancy has "Starsim 3's arguments, attributes, results, and callbacks", as does the skill's row 38, which its row 39 contradicts. `breastfeeding` needs `ss.Lactation`, which exists only with a `BreastfeedingNet`. BV's preterm link has no declared form and is lost in code unless the woman's risk is redrawn by hand (`preg.preterm_risk(uids)`, which can schedule a delivery already past) | Lead test runs (`lead_sti2`: 179 pass after these renames); `seis_bv/bv_ptb.py` | Rewrite `pregnant` → `P`, `ti_pregnant` → `ti_P`, `ti_delivery` → `np.ceil(ti_exit_P)` and the result names in the script; correct the message and row 38; a public `preg.redraw(uids)` keeping the elapsed gestation, or a modifier of the preterm chance |
| ST43 (LEAD-5) | **Removed members of `ss.Infection` are not flagged by the script**: `compute_transmission()` (STIsim's syphilis `infect()` calls it; the replacement `ss.networks.nb_transmit()` or `route.transmit()` has another argument order) and the class attribute `prenatal_death_keys` (still on `ssl.CongenitalDisease`) | `syphilis.py:256`, `syphilis.py:848` | Add both to the script's "to do by hand" list, with the replacements. A check of every removed public name of `ss.Infection` and `ss.Pregnancy` against the downstream packages would catch the rest |

### Crashes and performance traps

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| ST44 (LEAD-6, FW-6) | **A data column that matches no result stops the sim at `init()`**: `hivsim.demo('zimbabwe')` attaches data with `hiv.p_art` and raises `UnknownDataColumn ... The closest results are n_on_art, p_on_art, restart`. In v3 data on a sim was for plotting. 4 STIsim tests fail on it. Not in the skill or the script | `framework/calibrate_v4.py C` | A note at `init()`, and the error only when fitting (or `ignore_extra=True` by default for plotting); drop near matches like `restart` |
| ST45 (NET-2) | **STIsim's MSM networks inherit the man–woman `sides` new on `ss.SexualNetwork`**, so a declared disease over them raises `RouteMismatch: "msm" has a partnership from 587 to 2577, whose sides give no direction it has (m2f, f2m)`. Contradicts `sexual_network.md`'s "STIsim needs no change"; rule 5 there misses it | `networks/msm_sides.py` | The error could suggest `sides = None`; or the default sides apply only when edges are man–woman |
| ST46 (NET-1) | `dict(net=ss.peract(m2f=, f2m=))`, the spelling in `ss.peract`'s docstring, crashes under `solver='ode'`: `TypeError: '>' not supported between instances of 'peract' and 'int'` (`solvers/mixing.py:128`) | `networks/peract_dict_ode.py` | One line in the "carries no transmission" check |
| ST47 (SB-5) | As agents, a transmission rate with `ss.by(...)` over `ss.contacts(sex=...)` raises `ValueError: operands could not be broadcast together with shapes (5000,) (250,)` (`agents.py:2585`); the ODE runs. This blocks one text for agents and ODE for a sex-specific SEIS | `seis_bv/contacts_by_crash.py` | Broadcast group values to agents in `infect` |
| ST48 (SYPH-3) | A split with one destination (`ss.Transition(S, [F], p=[1.0], rate=...)`) becomes `p=1` on the transition, and three such out of one state raise `BadBranchSplit: ... sum to 3, not 1` | `syphilis/probe_nnd_timing.py` | A one-destination split is a plain transition |
| ST49 (SYPH-4) | Different source weights per route (`dict(random=b*(A + 0.5*B), maternalnet=b2*(A + B))`) crash with `UFuncTypeError: Cannot cast ufunc 'divide' output from dtype('O')` | `syphilis/probe_small_errors2.py` | A form, or a clear error (ST17's remainder) |
| ST50 (SB-6, #49) | **`compile='auto'` runs stiff ODEs as plain Python**, because its estimate (`plain_seconds()`, `npts × size`) ignores substeps. In a new install, a default run gives no hint that compiling would help | Syphilis ODE (273 compartments, 30 years): 266 s by default against 1.72 s with `STARSIM_COMPILE=always` (1.64 s at `03741ba2`); exponential dwells 0.55 s against 0.04 s. SEIS 83 s against 1.19 s; BV first run 84 s. I reran syphilis on both commits | Include substeps in the estimate (they are known at `init()`), and print a note when a plain run passes about 20 s: "this would take about 2 s compiled; ss.compile() once" |
| ST51 (FW-7) | `ss.fit.compare()` assumes every result is on the sim's timeline: with a module whose own `dt` is a year (36 points in a 421-step sim), `calibrate()` crashes with a bare broadcast `ValueError` (`results.py:1588`) | `framework/calibrate_v4_C_dt_year.log` | Compare on each result's own timeline |

### Missing

- **ST52 (NET-5, N15).** No declared man–woman network can have risk matching or degree by risk: `between=` excludes `match=` and `ss.by`, `ss.MFNet` takes neither, and `sides='sex'` adds same-sex partnerships. The error for `DegreeNet(between=..., n_contacts=ss.by(...))` says "which ssl.DegreeNet takes", on a `DegreeNet`. STIsim's network stays coded (30.8% prevalence declared against 9.1%).
- Accuracy results per test (D29) and condom use by partnership type (ST21) remain from October 18. The syphilis slice adds the per-person decay of latent infectiousness, which has no compartmental form, and its constant-weight approximation is wrong with care: 0.238 against 0.156 infected (SYPH-10).

### Small things

- **Messages:**
  - `what_modifies('hiv.rel_sus')` on a coded HIV raises `KeyError: No module named "hiv" with declared transitions` (HIV-8).
  - `hiv.equations()` raises a bare `AttributeError` (it was a clear `ValueError`).
  - `ss.modify(hiv.pars.beta_m2f)` raises `UnmodifiableTarget "not 0.05 (float)"`, while the string form gives a clear `CapabilityError`.
  - `ss.modify('hiv.infection', efficacy=, share=, route=)` calls `infection` "a parameter" (HIV-9).
  - `ss.par(...) * ss.peract(...)` loses the parameter (HIV-10).
  - `ssl.DegreeNet(match=dict(female=0.0))` is clipped to random mixing silently (NET-4).
- **Defaults:**
  - With `ss.Pregnancy` in the sim, a delivery's default `disease=` is no longer the only disease, so every syphilis care step needs `disease='syph'` (SYPH-5); the default could skip demography modules.
  - Typed day or week dwells make the sim's unit days, so every bare `fertility_rate=`/`death_rate=` warns although `dt` is monthly (SYPH-9).
  - `solver: abm (...)` prints at every calibration trial even with `verbose=False` (FW-10).
- **Spellings:**
  - `who=ss.mother(event)` is refused, so "treat the fetus when the mother is treated" needs memory (SYPH-7).
  - A clock in a modifier made before the sim isn't found without `ss.options.keep_times = True` (SYPH-8).
  - `fit.compare()` needs time as the index (FW-10).
  - Sweeping a declared `beta` dict replaces the whole dict (FW-9).
  - `disease.move_to()` is counted in no flow; a `via=` would credit one (FW-8).
  - `adjust=` on a branch now scales its probability, not its dwell, and only a note says so; the spec agrees, but old workarounds change meaning silently (SB-7).
- **Baggage:** `uids.cat()` warns `DeprecationWarning` (the constitution's "no baggage"), and the script doesn't rewrite it. `verbose=1/12` raises (the script rewrites it).
- **Stale docs:**
  - `ss.transmission` is in `00-overview.md:61` and `01-objects.md:25,42` but gone from the code (NET-8);
  - `deferred.md:42` says `ss.MFNet` under the ODE raises; it now uses the partnership approximation (NET-9);
  - spec 03's two sections on immediate branches contradict each other, and the engine follows "Each branch" (SB-9);
  - `idioms.md` "As soon as" still recommends a fast rate;
  - `ss.stay` isn't in spec 03.
- **HIV-6:** ART stays about 17 below its target (578 against 595), probably because dropout now runs after the deliveries in a step.

## Update: `v4` at `88208f57`

`88208f57` is `ee5d1fb0` plus `30c98710` ("final simplification changes": 22 files, 358 lines added and 706 removed, mostly decision 18's one sexual network). Rerun there (`stisim:v4port/lead/oct21/`):

- **ST28 is still there**: `repro_pars_before_init.py` gives `use_aging` False and no aging.
- **ST53 (migration dead end, new; was NET-6).** `ss.SexualNetwork`, `ss.MFNet` and `ss.PartnerNet` are gone, leaving `ss.SexualNet`, so `import stisim` fails again (`AttributeError: module "starsim" has no attribute "SexualNetwork". Did you mean: 'SexualNet'?`).
  - STIsim's network base (`networks/base.py:51`) and BV's `isinstance(net, ss.SexualNetwork)` (`bv.py:523,560`) use the old name.
  - The updated script rewrites `ss.MFNet` and `ss.MSMNet` but not `ss.SexualNetwork`. It should, since it is the base class of every downstream sexual network (STIsim's, HPVsim's).
  - The script also flags STIsim's own strings `'msm'` (its network names in `validate_beta`, `msm.py`) as Starsim 3 network names, which they are not. That is 5 false positives.
- **With the rename by hand, STIsim behaves as at `ee5d1fb0`.**
  - Tests: 179 pass and 19 fail. They are the same 17 failures plus two new ones:
    - `test_parameter_docs_coverage`: STIsim's network now carries `ss.SexualNet`'s pars `concurrency`, `debut`, `dur`, `max_contacts`, `n_contacts`, `participation`, `sides`, which STIsim doesn't use. This is `sexual_network.md`'s point that unused base pars show in subclasses.
    - `test_syph_epi`: a 72 against 66 comparison of small counts, which is seed noise.
  - HIV-only prevalence after 20 years: 0.1030 ± 0.0034 (5 seeds), against 0.1065 ± 0.0048 at `ee5d1fb0`.
  - Five diseases: 4.32 s against 4.16 s, the same speed.
  - So STIsim's own partnership formation is untouched, as the design intended. I didn't measure the network statistics or the debut draws (NET-6's double `set_network_states()`).
- ST45 (MSM networks inheriting man–woman sides) now has the fix `sides='mm'` in v4's own vocabulary. STIsim's MSM classes still need to set it.
- ST22 regressed: `between=` (sex workers and clients as a two-mode network, lasting or one-off) went with `ss.PartnerNet`.
- The other findings were not rerun at `88208f57`. Of the files they touch, only `networks.py`, `solvers/mixing.py` (ST46) and `sim.py` changed.

## Suggested order

1. **ST28** (a few lines), and a test that pars set before `init()` survive. It is the only finding here that changes unchanged downstream results, and it hides behind a plausible epidemic.
2. **The migration set** (ST40–ST43, ST39, ST44): roughly a day of script rules and doc fixes, so that STIsim (and HPVsim) import and the script gets them to about 180 of 198 tests.
3. **The declared silent wrong answers** ST29–ST35. Each is small, apart from ST29's storage of far exits.
4. **ST50** (substeps in the compile estimate, a few lines), **ST8** (it is marked done, but isn't), and the fitting set ST36–ST38.
5. Then the crashes (ST45–ST51), ST53 (a line in the script), and small things as they come up.

## Index of new findings

| ID | Short | Kind |
|---|---|---|
| ST28 | `use_aging` set before `init()` is overwritten; nobody ages in `sti.Sim` | Silent wrong answer (regression) |
| ST29 | A small `adjust=` factor never resumes after a transfer | Silent wrong answer (regression of a workaround) |
| ST30 | `move=` on an `on=<product>.failed` transition is dropped | Silent wrong answer |
| ST31 | `init` value 1.0 is one person | Silent wrong answer (regression) |
| ST32 | `ss.constant(0)` waits a step under compartments | Silent wrong answer |
| ST33 | A rate function's bare numbers are read in the inferred unit | Silent wrong answer |
| ST34 | Neonatal death `on=preg.delivery` counts as a stillbirth | Silent wrong answer |
| ST35 | Delivery results and flows one index apart | Silent offset |
| ST36 | Annual data compared one year off | Silent wrong answer (fitting) |
| ST37 | Bounded pars lose their bounds through `Pars.update()` | Silent dead end (fitting) |
| ST38 | Optuna "Converged" at the start | Silent wrong answer (reporting) |
| ST39 | The script misses a stored `ti` | Silent wrong answer (migration) |
| ST40 | `ss.Calibration` removed; `import stisim` fails | Migration dead end |
| ST41 | No public base for any disease; `isinstance(x, ss.Disease)`; duplicated states | Migration dead end |
| ST42 | Starsim 3 pregnancy attributes gone, not rewritten, docs say kept | Migration dead end |
| ST43 | Removed `ss.Infection` members not flagged | Migration dead end |
| ST44 | Unmatched data column stops the sim | Crash (behavior change) |
| ST45 | MSM networks inherit m/f `sides` | Crash |
| ST46 | `ss.peract` dict under the ODE | Crash |
| ST47 | `ss.by` over `ss.contacts()` as agents | Crash |
| ST48 | One-destination split | Crash |
| ST49 | Per-route source weights | Crash |
| ST50 | `compile='auto'` ignores substeps (160x slower ODE by default) | Performance trap |
| ST51 | `fit.compare()` with a module on its own timeline | Crash |
| ST52 | No declared man–woman network with risk matching | Missing |
| ST53 | `ss.SexualNetwork` removed (decision 18), not rewritten by the script | Migration dead end (at `88208f57`) |
