# Starsim v4: the plan from October 22

Written 2026-10-22. It replaces the open parts of three documents, which keep the evidence and history:

- [oct21/plan.md](../oct21/plan.md): its unchecked items only.
- [oct21/review_synthesis.md](../oct21/review_synthesis.md): the downstream rechecks, with Cliff's comments answered. This plan also covers [tbsim_recheck.md](../oct21/tbsim_recheck.md), which the synthesis left out.
- [oct21/strain_plan.md](../oct21/strain_plan.md): strains, with Cliff's comments answered.

Finding IDs (MA, HP, FP, CV, ST, TB §) point to the reviews. The [constitution](../constitution.md) governs, and where this plan differs from it, the plan is changed. Sizes: S is under a day of agent work, M a few days. **SWA** marks a silent wrong answer.

## Definition of done

This plan is done when all of these hold, and not before or beyond:

1. **A1–A3 are built**: the disease base class, the `self.ti` timing, and ODE forms written outside Starsim.
2. **B1–B8 are done**: the silent wrong answers and crashes from the reviews.
3. **C (strains) is done** to its own checklist, including the three test builds.
4. **D (speed, compiling, calibration) and E (networks, downstream-facing) are done.**
5. **Every open question below is answered by Cliff or moved to [admin/deferred.md](../admin/deferred.md)** (the first six are answered; see Decisions).
6. **F1 and F2 are done**: measured against S0.3 with the new benchmarks, and the records updated.
7. **Checks pass**: the full suite, `tests/switching` apart from known timing flakes, and the perf guard, with no benchmark slower than its baseline. [CK: i mean, a little bit of flexibility here is ok, we can optimize performance later; and something going from 1.2 s to 1.5 s is more of an issue than something going from 1.2 ms to 1.5 ms]

These are not part of it: downstream ports and migration (F3), the docs (F4), S1.7 (stepwise kernels, only if needed), and S7.9 (the docstring pass, optional). A new idea goes to deferred.md. If one is added here instead (only for an SWA, or because it blocks an item), the estimate goes down and the update says why.

**Estimate**: about 98% done (2026-10-22, 10:50: up from 97% because Cliff answered question 10 (keep int64 `auids`), so `test_laser_memory`'s target moved to 55 bytes and item 7 is met apart from the known `test_odin_filter` flake; what is left is Cliff's answers to the other open questions below (item 5)). Weights: S 1, M 3. Total about 60 units. Update this line with every edit. It should only go up.

## Order

1. **Now**: B0 (ST28 aging, FP57 `ss.by` speed, the missing benchmarks).
2. **Architecture**: A1–A3. Several B items depend on them.
3. **Correctness**: B1–B8.
4. **Strains**: C.
5. **The rest**: D and E (in parallel with B where the files don't overlap), then F.

## A. Architecture (Cliff's comments call for a rethink)

- [x] **A1. `ss.Disease` is the base of every disease again (M)**. *Done 2026-10-22*: one seam (an engine that does nothing when nothing is declared, `ss.solvers.Undeclared`), mixes run both parts, the migration rewrite is removed (v3's two classes mean the same again). Left: FP71's wording for declared non-transmitting states (with FP38). See [worklogs/a1_disease_base.md](worklogs/a1_disease_base.md). (theme 4; MA22, MA41, ST41, HP53, TB §4.6, FP71). Cliff: "this is an architectural mess. the whole point is that declared and coded parts should be able to live peacefully together." So:
  - `ss.Disease` is the one base. It takes declared states and transitions, coded `step()`/`set_prognoses()` overrides, or a mix of both in one class. "Declared or coded" becomes a question about each part, not about the class.
  - `ss.Infection(ss.Disease)` adds transmission and nothing else.
  - `isinstance(x, ss.Disease)` is true for both. Malnutrition is a plain `ss.Disease`. v3's `class X(ss.Disease)` works as before. `DiseaseBase` goes away.
  - This partly reverses S5.8, which made two classes. Keep S5.8's real gains: the branches on `graph is None` stay gone. Each part's lifecycle runs if it was declared, and a coded override runs if it was given.
  - Spec first (00, 01, 03). Revert the migration script's `ss.Disease` → `ss.Infection` rewrite, except where the class transmits.
- [x] **A2. `x[self.ti]` must just work (M)**. *Done 2026-10-22*: the loop advanced `ti` mid-step; it now advances first, so `self.ti` is the step's index in every phase, and the loop records index 0 itself. CV2, CV3, CV6, ST8, ST35, ST39 and MA41 are fixed. See [a2_timing.md](a2_timing.md) and [worklogs/a2_timing.md](worklogs/a2_timing.md). (theme 3; CV2, CV6, FP1/FP42, ST35, ST39, ST8, MA41). Cliff: "nothing could be simpler than 'we're in a loop, here's the value at the current loop index', so if that's broken, ... something more systematic has gone awry." So treat it as a loop bug, not a user trap:
  - Find why `self.ti` means the start of the step in `step()` but the end of it in `update_results()`. Make it mean the same in every phase of one step, for coded and declared modules, deliveries and transitions alike (that also fixes ST35).
  - Then CV6 (`run(until=)` one short), ST8 (interventions' results before the first step), ST39 (a stored `ti` makes durations one step), and MA41 (nothing writes the initial point) should fall out or become one-liners.
  - `.now` is fine as an extra for corner cases, but it is not the fix. Restore the end-of-run warning for results written at the wrong index only if a case remains.
  - Start with a short write-up of the current convention and the proposed one, for Cliff, before changing the loop.
- [x] **A3. ODE forms written outside Starsim (M)**. *Done 2026-10-22*: a declared state chain is the extension point and already ran under every solver; `move=` now takes several sources (re-dosing), and MA15 is fixed. See [worklogs/a3_ode_forms.md](worklogs/a3_ode_forms.md). (theme 7; MA38, HP18, HP28). Cliff: "if the user can write it to look like an ODE, we should be able to consume it as an ODE", and Starsim can't hard-code every waning shape.
  - First check what malariasim can already write. Box-then-decay as two declared states (a fixed dwell with `after=`, then an exit rate) uses only forms that S3 kept, so it may run under the ODE today. If it does, the fix is the spelling: a product's waning written as states. If it doesn't, find out why.
  - Then the general route. A module can give `compartmental_form(sim)` (S3.2, used by `ss.Pregnancy`). Decide whether that, or a declared state chain, is the public extension point for shapes like "decay at one rate, then another", and write one example from outside Starsim.
  - HP18 (an outcome that depends on the drawn dwell) and HP28 (a CV above 1): say whether the same route covers them. If not, they stay agents only, and FP53's guard (B3) covers `auto`.
  - Don't restore the 40-line box-then-exponential special case unless this route fails.

## B. Correctness and crashes

Cliff on tests: "can probably skip the regression tests; the test suite should not become a creaky history of every random bug." Write tests only for real gaps: the contract tests named below, plus one where a fix has no other cover.

- [x] **B0. First (S each)**. *Done 2026-10-22*: ST28 fixed for every setting read through `sim.given`; `ss.by` tables built once and indexed with NumPy (FPsim-style deaths 4.19 → 0.17 s; CV11 2.3 → 1.4 s, the rest in deferred.md); the four benchmarks are in `tests/benchmark_models.py` and the perf guard. See [worklogs/b0b6_by.md](worklogs/b0b6_by.md).
  - **ST28**: nobody ages in any `sti.Sim` (`use_aging` overwritten, `dimensions.py:719`).
  - **FP57/CV11**: `ss.by.lookup()` rebuilds its table from pandas rows on every call, which makes declared models 3–107× slower. Build the grid once and index it. Cliff: "do NOT use pandas in any hot path!" Grep the other step-time paths for pandas while there.
  - **Benchmarks**: add malariasim's `fseis10`, a declared model with counts, a coded model, and one with an `ss.by` age table to S0.3's set.
- [x] **B1. Nothing accepted then ignored (S, plus S per case)** (theme 1). *Done 2026-10-22*: `ss.connectors.check_applied()` at init raises `ss.errors.Unapplied`; CV1 and MA36 implemented; FP52, FP54, HP6, D23 refused (FP52's implementation is in deferred.md); FP41, HP46 and ST30 are with B4 and C. See [worklogs/b1_unapplied.md](worklogs/b1_unapplied.md). At `init()`, check every modifier, change and product, and raise unless something reads it. Then implement the cases (CV1 in `infect_route()`, ST30, MA36) or refuse them (FP52 until built, FP54, HP6, FP41, HP46, D23, ST37).
- [x] **B2. `init=` (S)**. *Done 2026-10-22*: `init=` takes shares, `init_n=` counts (decision 1); see [worklogs/b2b5b8_init_units.md](worklogs/b2b5b8_init_units.md). (theme 2; MA34, HP48, FP69, ST31). Up to 1 is a share and above 1 is a count, as with `init_prev=`. Cliff: an int is still read as a share (`init=[1, 0, 0]` means everyone), with a warning to write a float. Functions are accepted as with `init_prev=`. Change the note to "the other 9,999 people". For exactly one person, `init_n=` (decision 1).
- [x] **B3. Solvers agree and conserve (S each)** (theme 6). *Done 2026-10-22*: FP51 (conveyor slots lost people in crossings), FP53, ST32 and MA6 (counts substepped) fixed; MA37 was `R0()` called on the user's object rather than the sim's copy (docstring note). Two contract tests added. See [worklogs/b3_solvers.md](worklogs/b3_solvers.md). Two contract tests: people are conserved in every crossing (FP51), and agents match the ODE on the spec's own models, with the analytic `R0()` (MA37). FP53: under `auto`, run agents when a dwell's CV is above about 3, and warn under an explicit ODE. Also ST32 (`ss.constant(0)` waits a step under compartmental solvers) and MA6 (chain-binomial mosquitoes die out at a 30-day step).
- [x] **B4. Births and newborns (S each, one batch)** (theme 8). *Done 2026-10-22*: all fixed at the cause (FP46 was a parent passing a transition twice in a step); ST34 is intended and now documented. See [worklogs/b4_births.md](worklogs/b4_births.md). FP43–FP50, HP37, ST34. Add one contract test that the engine's draws match `dist.rvs()` (FP48). For FP50, don't backfill a history for people placed by `move_to()`. FP46 (2.5% of births lost, cause unknown) needs a diagnosis first.
- [x] **B5. Units and step boundaries (S)** (theme 15). *Done 2026-10-22*, including spec 02's timing contract and two `describe()` lines: ST33, HP47, FP68, S7.8 (a country's matrix is contacts a day), FP55 (stochastic rounding, about 2%).
  - Refuse bare numbers returned by a rate function when the model has typed values (ST33).
  - Fix HP47 and FP68's message.
  - **S7.8's bug** (SWA, carried over): a bare `ss.contacts()` matrix is read in the sim's unit, so with monthly steps, acts per year are read as acts per month.
  - Add a "timing contract" section to spec 02, with `describe()` lines for CV5 and ST4.
  - FP55: drawn dwells round stochastically to whole steps; an explicit 0 stays instant (decision 2).
- [x] **B6. Expression algebra (S–M)** (theme 13). *Done 2026-10-22*: `ss.by` is an expression node; `test_transitions.py::test_algebra` runs 198 tiny cases in 8 s; all listed crashes fixed (ST49 refused under the ODE), plus a silent wrong answer (newborns entered states with another agent's age). make `ss.by` an expression node (FP60). Add one table-driven test: each node type × each operator × each solver, on a tiny model. That covers HP47, HP49–HP51, FP61–FP63, ST46, ST47 and ST49.
- [x] **B7. The remaining SWAs from the reviews (S each)**. *Done 2026-10-22*: CV4, HP38–HP40, HP42–HP44, MA35, ST29 and FP56 fixed, plus a new one (a delivery's `move=` didn't rescale dwells adjusted by that dimension); the rest point to their items. See [worklogs/b7_swas.md](worklogs/b7_swas.md). Was: CV3 (a delayed result released at the start of a step doesn't block a retest in the same step), CV4, HP38–HP42, HP44, MA35, ST29, and any others in the synthesis's list that A1–B6 don't cover. Check each still reproduces, then fix it or point it at the item that did.
- [x] **B8. Leftovers from the October 21 plan (S each)**. *Done 2026-10-22*: the first state is `first=` or a state named `S`/`susceptible`, else `sim.init()` raises.
  - **Where people go when no state is named** (decision 3). Cliff leans explicit: a state named `S` or `susceptible` works; otherwise require the state given to `ss.birth()` or `first=`. Raise at `sim.init()` when it is needed and missing, and apply the same rule to initial placement.
  - **MA15** (*done with A3*): re-dosing a product whose waning is a function gives a wrong answer. It needs a spelling, which may come from A3.

## C. Strains (S7.4)

*Done 2026-10-22*: P1–P5 built, the three test builds run and meet their checks (tbsim: the ODE identical to the old spelling, 10 lines against tbsim's 90; HPV: bit-for-bit the old family; Covasim: max |z| 1.51 against Covasim 4.0.1), apart from Covasim's check 3, which needs CV13. See [worklogs/c_strains.md](worklogs/c_strains.md).

[strain_plan.md](../oct21/strain_plan.md) stays the detailed plan: its ownership model (shared or own states), the event table, the Covasim mapping, the phases P1–P5, and the test builds. Its definition of done is part of this plan's item 3. Cliff's comments change it as below. Update strain_plan.md to match before starting P1.

**Settled by Cliff's comments:**

- **Spelling (D1)**: strains are part of the disease, not a `stratify()` call: `ss.Disease(..., strains=ss.Strains(...))`, which gives `tb.strains`, not `tb.resistance`. Internally a strain set is still a kind of dimension.
- **Several ways to give per-strain values**:
  - by strain, as dicts: `ss.Strains(dict(DS=dict(rel_trans=1, cure=1), RIF=dict(rel_trans=0.575, cure=1/3)))`;
  - Covasim's layout (variant parameters plus a cross-immunity matrix, as in `parameters.py:get_variant_pars()` and `get_cross_immunity()`);
  - lists by parameter, as now.

  This is one container read three ways, not three mechanisms.
- **Members defined separately (D2)**: allowed as a convenience, `ss.Strains([hpv16, hpv18])`, because real definitions are long. They share one natural history. Letting a member's structure differ (an extra state) waits in deferred.md unless it turns out no harder, since it complicates results.
- **`due_to=` replaces `protected=`** on `ss.immunity`. For memory (D3), use objects, not strings: `due_to=R` (a state), `due_to='nab'` (an attribute, scaling each strain's row), `due_to=covid.strains.last`, and `due_to=flu.strains.ever`. Cliff rejected `protected='ever'`. Covasim's form: `ss.immunity(due_to=covid.strains.last, scale='nab', curve=ve)`.
- **No arrow strings**: write `move=DS >> RIF` (replacement) and `move=DS >> (DS, RIF)` (emergence) on strain objects. Cliff said TBC; the build confirms it. The same goes for the last arrow strings elsewhere: `ss.Delivery(move='S -> V')` (17 uses) and `ss.transfer(risk='low -> high')` (TB §4.5).
- **Full carriage (D4)**: at `max_strains`, a new strain replaces the oldest, and the new strain's clock starts fresh. This is counted as a flow. It is Cliff's rule, replacing "not established".
- **D6** (a set takes its worst strain's value) and **D7** (try both spellings in the tbsim build) are accepted. D8 (`curve=`) is accepted.

**Settled since (decisions 3–5)**: transmission from a mixed carrier (D5), emergence on the transition that causes it, and the cure example.

**Answer to Cliff's 2^k question** (table row 3): own courses with no coupling are independent, so each strain gets its own copy of the states. That is k copies side by side, which the ODE handles. 2^k appears only when courses interact, because then the combination a person carries matters. So row 3 is fine, and row 4 (coupled) is agents only.

## D. Speed, compiling, calibration

- [x] **D1. Hot spots (M)**. *Done 2026-10-22*: `series()` tabulated once per run, counts' per-step work hoisted out of RK4 (malaria hybrid 1.8× faster), `ss.years(array)` fixed; the copy-on-write check costs ~12 ns and stays, and the coded slowdown was int32 `auids` (now int64). Stepping counts through the kernels went to deferred.md. See [worklogs/d1d2_speed.md](worklogs/d1d2_speed.md). Was: counts stepped in Python as agents, and `series()` re-tabulated at every derivative call (MA39, MA28); `Arr.__setitem__`'s copy-on-write check (10–15% on every coded model); `ss.years(array)` (FP59, 1 line). FP58 goes to deferred.md as a measured cost.
- [x] **D2. Compile automatically when it pays (S)**. *Done 2026-10-22* (theme 11; ST50, MA39). Count substeps in the estimate. Cliff's rule:
  - if compiling takes less time than the run would save, compile automatically, with a note;
  - if not, run plain and print a note with the estimated plain and compiled times.

  Never compile silently, even at `verbose=0`.
- [x] **D3. Calibration (S each, plus M)** (theme 10). *Done 2026-10-22*, HP54 included (a `strain` column in long data maps rows to each strain's results; [worklogs/fix2_hp54_nuts.md](worklogs/fix2_hp54_nuts.md)): see [worklogs/d3_calibration.md](worklogs/d3_calibration.md).
  - fix the silent ones: ST36, ST37, ST38;
  - fix the crashes: ST44/FP8/CV10, ST51;
  - read an integer `year` column as a calendar period;
  - add `n_workers=` and `storage=` (M);
  - add fitting of a coded intervention's settings (CV10);
  - add mapping a data column to strain members (HP54; after C);
  - MA39: three searches for a one-parameter deterministic fit.

## E. Networks and what downstream packages hit

- [x] **E1. `ss.SexualNet`, all four items (S–M)**. *Done 2026-10-22*: see [worklogs/e1e2_networks.md](worklogs/e1e2_networks.md). (theme 14; Cliff: "do all four"):
  - HP45: don't pair on behalf of subclasses that pair their own.
  - ST45: the error suggests `sides='mm'`.
  - ST22/ST52: two-group partnerships (sex workers and clients) and risk matching come back, with `between=` (decision 6).
  - `test_parameter_docs_coverage`: accept or fix base parameters that subclasses don't use.
- [x] **E2. Small fixes downstream code hits (S each)**. *Done 2026-10-22*, except `.since` (the delivery memory is on `EventSource`, which declared diseases are not) and the optional `.results` guard, both moved to deferred.md:
  - `sim.current_scale` becomes a public property again (CV7).
  - `verbose` fractions mean progress every 1/verbose of the run (CV9). On Cliff's question: Starsim 3's default was `0.1` (`settings.py`, "an update every 10 timesteps"). v4 made `verbose` 0/1/2 with a default of 1, and refuses fractions, so every v3-era package that sets 0.1 breaks.
  - `interventions=` (and the other module lists) flatten one level of nested lists (TB §4.1).
  - `Results.flatten()` keeps case, as `to_df()` does (TB §4.3).
  - Hints on three bare errors: `infectious=`, `ss.library` without its import, and `match_result_keys` (TB §3).
  - `.since` on transitions (TB §4.2), so that `tb.diagnosis.since <= ss.years(2)` works without `keep_times`. Do it if it uses the delivery memory as is; otherwise defer it.
  - A module copied into a sim and never run raises on `.results`, pointing to `sim.analyzers.<name>` (TB §4.4). This one is optional.
- [x] **E3. Decisions to record (S)**:
  - *Done*: the reversal of `oct17/decisions_oct18.md` item 3: no `ss.Calibration`; `sim.calibrate()` replaces it.
  - *Done*: Starsim 3's coded `ss.Pregnancy` is ported by rewriting onto the declared one: add FP41's `__init_subclass__` refusal, ST42's renames, and a declared newborn transition for infant mortality.

## F. Measure and close

- [x] **F1. Measure against S0.3 (S)**. *Done 2026-10-22*: nothing slower on the seconds scale, `by_age` 8.5 → 1.4 s, agents 19% faster; perf guard 14/14; Ghana still 2.2× the NumPy loop; memory 45.7 → 51.2 bytes per agent (question 10). See [measure.md](measure.md): as S8.1 of the October 21 plan, with B0's new benchmarks, and against hand-written code (Ghana measles in NumPy, LASER).
- [x] **F2. Records (S)**. *Done 2026-10-22*: LFD's `design/` and `admin/deferred.md` are straight copies again; deferred.md has an "October 22" section, `CHANGELOG.md` brief lines, [note_for_robyn.md](note_for_robyn.md), the Stale list below, and the strain documents and the October 21 plan are marked superseded:
  - LFD's `design/` is a straight copy of `v4_design/design/`.
  - deferred.md is updated, `CHANGELOG.md` gets brief lines, and the note for Robyn is written (A12).
  - The [Stale](../oct21/plan.md#stale) list carries over and gets what this plan adds (at least A1's `ss.Infection` rewrite in the docs and skill, and C's spellings).
- **F3. Downstream and migration** (after the design is settled; not in the definition of done):
  - tbsim, STIsim, HPVsim, FPsim, Covasim and malariasim run again.
  - The script rules: rewrite keywords only in calls to Starsim's deliveries, never in a `def`; don't delete `on_positive=`/`on_negative=`; rename `ss.SexualNetwork` and `copy_inputs=`; consider a `--prerelease` mode (TB §2).
  - TB §2's migration table becomes skill rows.
  - The downstream fixes in other repos (oct21 plan, "Where things stand") are written up.
- **F4. Docs**: once the implementation is final.

## Stale

Out of date after this plan's changes, and left for F3–F4 (constitution, development principle 1). Carried over from the [October 21 list](../oct21/plan.md#stale), plus what this plan added. Add to it rather than fixing each on the way. The worklogs list file and line references.

From October 21, still true:
- [best-practices/principles.md](../best-practices/principles.md): arrow strings, `ss.transmission`/`ss.progression`, and `init_prev` with a 1% default.
- `tests/switching/test_emulsion.py`: xfail reasons name `offspring=`.
- The docs (`docs/`), tutorials, user guide, "Coming from" pages, release notes and LFD's vignettes, wherever they use removed or renamed names (`ss.geo.*`, `ss.errors.X`, `ss.debug.*`, `'mf'` network names).
- rstarsim: lowercase `to_df()` columns, `disease()` passing `beta=`/`source=`, and now `init = list(I = 10)` (needs `init_n`). Its tests are marked `xfail` (Cliff, 2026-10-22).
- STIsim and HPVsim subclass `ss.SexualNetwork`, now `ss.SexualNet`.
- `CHANGELOG.md`: says JAX is optional and that fits use Nelder-Mead without it.

Reversed by A1: the docs no longer need "a coded disease subclasses `ss.Infection`"; anything written that way since S5.8 (docs, the skills outside the repo) goes back to `ss.Disease`.

Added by this plan:
- **Timing (A2)**: the migrate-starsim3 skill's rows on `self.ti`, `ti_entry` and the promised warning; `skills/import-model/references/starsim3.md`; `docs/migration/v3_v4.md` and the user guide pages that explain start/end-of-step `ti`.
- **`init=`, `init_n=`, `first=` (B2, B8)**: tutorials and user guide pages that give counts in `init=` or rely on the first declared state; the migration script and skills; spec 05's `ss.contacts(matrix, age=...)` example (now `bands=`).
- **Strains and arrow strings (C)**: every use in the docs of `stratify(..., kind='strain')`, `immunity=` on a stratification, `'A+B'` levels, `ss.Strains(name, members, exclusive=)`, `protected=`, and arrow strings in `ss.transfer()`/`move=`; the import-model references (`summer2.md`, `epimodel.md`) and the migrate skill; `starsim/interop/migrate.py` has no rewrite for them yet. tbsim could rename `tbsim.Strains`, and `resistance.py` can become `ss.Strains`.
- **Pregnancy (E3)**: `docs/migration/v3_v4.md` never mentions the pregnancy rewrite; the migrate skill's rows on Starsim 3's pregnancy attributes.
- **Products (A3)**: malariasim's v4 port can write its box-then-decay waning as states to regain the ODE; the skills on products don't mention re-dosing or `move=` from several sources.
- **Calibration (D3)**: the calibration skill and tutorials (`n_workers=`, `storage=`, calendar-year data, the note for an unmatched column).
- **Networks (E1)**: skills outside the repo that mention `sides=` or `keep_case=`; `docs/tutorials/solutions.qmd` uses `ss.MFNet(dur=...)`, which now means an exponential duration.
- **Superseded documents**: `oct21/strains.md` and the two strain proposals (by `strain_plan.md`); synthesis theme 7's "restore the 40 lines" (by A3).

## Decisions (Cliff's answers, 2026-10-22)

Cliff answered the first six questions, and later question 10 in the plan. Each is now settled and the items above follow it.

1. **Exactly one person in `init=` (B2)**: no new top-level name for a corner case. `init=` takes shares (an int too, with a warning to write a float), and `init_n=` takes counts. `init_prev=` keeps its current rule.
2. **Short drawn dwells (B5, FP55)**: an explicit `dur=0` stays instant. A drawn dwell rounds stochastically to whole steps (0.2 steps is 0 steps 80% of the time and 1 step 20% of the time), which keeps the mean. The `TriggerLoop` message is fixed too.
3. **Transmission from a carrier of two strains (D5)**: three options. The default `'one'` picks one strain per transmission, in proportion to relative transmissibility; `'fittest'` is tbsim's `transmission_independent=False`; `'each'` stays for co-carriage models.
4. **Emergence (C)**: mechanistic, on the transition that causes it, not a `where=`. For example, treatment failure carries the strain change: `ss.Transition(Tx, I, prob=p_fail, move=DS >> (DS, RIF))`. Cliff's sketch was `treatment(p_fail=0.3, on_fail=DS >> (DS, RIF))`, as an example rather than an API.
5. **The cure example (C)**: rewrite it as in the recommendation (cure is treatment ending; per-strain success from `adjust=dict(cure=...)`).
6. **`between=` only (E1)**: a pair of groups, with `'mf'` (the default) and `'mm'` as shorthands, per-group `n_contacts`, and `match=` for risk matching.
7. **int64 `auids` (D1, open question 10)**: keep int64 for speed; don't trade speed for a few bytes per agent. Cliff: "DEFINITELY keep int64 for speed rather than trying to save a few extra bytes! record that as important". `test_laser_memory`'s target moves from 50 to 55 bytes per agent (it measures 51); the two ways back under 50 (one `people.scale` value, `rel_sus`/`rel_trans` made on first write) are not wanted for their own sake.

## Open questions for Cliff

New questions from the build go here, each with the guess that the build follows meanwhile.

1. **Products whose waning is written as states (A3).** Today: a protection module, `move=` and `ss.modify(where=)`, which needs no new code. Alternative: sugar such as `ss.Vaccine(efficacy=0.8, waning=prot)`. Going with the first; add the sugar if malariasim finds it verbose.
2. **Re-dosing (A3, MA15).** Products whose waning is a function are now re-dosed (the new dose restarts the waning, as in malariasim). Products with exponential waning still skip holders: equivalent, since the loss is memoryless, but counts and costs differ. Going with the smaller change.
3. **Solver for a mixed disease (A1).** A declared disease with per-agent arrays of its own runs as agents (a named compartmental solver raises) unless it sets `compartmental = True` or gives `compartmental_form`. One whose only code is a `step()` keeps the existing rule (it runs beside the compartments). The alternative, any coded override forcing agents, would change how existing declared classes with a small `step()` run.
4. **`sim.rescale()` with a plain coded `ss.Disease` (A1)** requires it to define `not_naive()`/`make_naive()`, rather than guessing from a `ti_infected` array as `ss.Infection` does.
5. **A stock in calendar-year data (D3, ST36)**: compared with the year's mean, which is what a value "for 2010" usually means. The alternative is the year's end, as `res.resample('year')` takes.
6. **A data column that matches no result (D3, ST44)**: a note at `init()`, and an error only if the fit compares it. The alternative keeps the error, with `ignore_extra=True` as the default.
7. **Fitting a coded module's settings (D3, CV10)**: found by scanning the module's attributes for `ss.par` at init (skipping Starsim's own modules), so `test_prob(symp_prob=ss.par(...))` just works. The alternative is a `custom_fn(sim, values)` hook on `calibrate()`, which is more to learn.
8. **Which `ss.SexualNet` subclasses pair their own (E1, HP45)**: those that override `step()`, as sexual_network.md recommended, so STIsim and HPVsim need no change. The catch: one that overrides `step()` and still calls `super().step()` lacks the base's parameters. A class flag would be clearer but needs a line downstream.
9. **`between=` groups are names only (E1)**, not conditions, since the names give the directions (`fsw2client`). One-off contacts between two groups (the old `PartnerNet` without `dur=`) did not come back; short partnerships are the nearest.
10. *Answered: see Decisions, item 7.*
11. **Compile notes (D2)**: a plain run's note prints only for runs estimated at 1 s or more, once per process; the constants are `compile_seconds = 60` and `speedup = 50`.
12. **HP6 and D23 (B1)**: only an efficacy on a transition with several destinations is refused; a plain `multiply=` keeps its note, since it may be meant to change how long people stay. A sim stratification used by nothing (only coded modules) is now an error rather than a note.
13. **Placement fires enter events (B4, HP37)**, including for the first state: `ss.attr(..., on=S.enter)` is set at the start for everyone starting in `S`. If the first state shouldn't count as entered by placement, it is one condition in `fire_placed()`.
14. **Infant mortality (E3)**: `ss.Pregnancy(infant_rate=)` (default 0, takes `ss.by()`) and a declared transition `infant_mortality` (`infant_death` was already the mother's event). A death `on=preg.delivery` stays intrapartum by design (ST34), now documented rather than warned about.
15. **The library's coded diseases lose your October 19 decision (a) (A2)**. With `ti_entry` gone, `ssl.AgentSIR`/`SIS`/`SEIR` (and Cholera, Ebola, Measles) and `ssl.HIV` use the natural code, which has Starsim 3's timing: an infection of d steps transmits for d − 1 (peak infected −5% at daily steps, −10% at yearly). Restoring (a) is a few lines but puts `self.ti - 1` back into the examples. Going with the natural code; say if (a) should come back. [Cliff, 2026-10-22: asks whether the cause is that coded models apply changes in turn during a step, where ghana-measles works out every change from the state at the start and applies them together; if so, consider splitting a coded step into working out the changes and applying them. Yes, that is the cause (see the reply in this session); to be designed, not built yet.]
16. **`self.now` is the step's start and `self.ti` its end (A2)**: inputs and dates are read at the start (`tvec[ti - 1]`), as in Starsim 3; results go at `ti`. Declared progressions are still dated at the step's start (`ti - 1`) and infections at `ti`, so a time in a state worked out by hand during a step needs `- 1`; the clocks (`X.time`, `.since`) are right.
17. **The partner-change ODE runs low, not high (B3)**. With lasting partnerships and fast transmission within them (6.7 a year here, against recovery at 1 a year), the ODE gives 8% infected against 59% as agents, because it counts at most one transmission per partnership. The spec says it runs high for curable infections, which is wrong here. Options: (a) correct the note and spec; (b) a pair-formation ODE (size M); (c) fold reinfection into P. Recommendation: (a) now, (b) to deferred.md. *Done as (a)*: the note and spec now say when it runs low and high.
18. **Expression semantics (B6)**: an `ss.attr` written inline in a rate is that module's own attribute; a state in a modifier's amount reads as 1 for people in it, as `where=` does; a value by group over a contact matrix applies to the person infected, as agents and the ODE alike.
19. **Multi-disease outcome totals (B7, HP39)**: a treatment with `disease=[a, b]` now treats the infected of every listed disease, but outcomes still count per disease (the total is the sum by disease, as `test_syndromic` defines it). Per-person totals are about 8 lines as agents, but break that identity and have no compartmental form. Going with per disease.
20. **`ss.by(age={0: ..., 15: ...})` (B7, HP42)**: numeric keys are lower bin edges, as in the table form, so integer properties are binned too (parity 3 falls in a "2+" bin). `ss.State(people=None)` is the new default, so that an explicit `people=True` can skip MA35's check, rather than adding a keyword.
21. **`ss.SexualNet(dur=<one value>)`**: a mean with a constant end rate (`ss.expon`), as `ssl.DegreeNet` already reads it and as the partner-change ODE handles exactly. Before, it kept the default's 1-year spread (a CV of 2 at 6 months); scaling the default's CV instead would make it nearly fixed. Also fixed: `n_contacts=ss.constant(1)` left almost nobody partnered, and `acts=` now refuses a rate (`ss.peryear`) and asks for `ss.freqperyear`.
22. **The first state (B8)**: required for any disease with states of people (raised at `sim.init()`), not only when something places people, since in practice births, seeding and rescaling all need it. A state named `S` that means something else (STIsim's CT uses `S` for cleared) becomes the first state; accepted, since `describe()` shows it. A state named by `ss.birth()` is not taken as the first state (it started a herd model with everyone as newborns). `init=dict(S=0.1, I=1e-4)` still puts the rest in the one state without a value; say if that guess should go too.
23. **Units (B5)**: a function's bare number is refused only when the sim's unit was inferred from the model's typed values; a bare matrix keeps the sim's unit (with `ss.contacts(..., unit=)`), and only a country's matrix defaults to contacts a day.
24. **Levels of a plain dimension in `move=` and `ss.transfer()` (C)**: a pair, `risk=('low', 'high')`, since levels are strings, not objects (level objects would be a new concept for one corner). Several source states are `move=U + B + W >> B` (`U | B` is a condition).
25. **Strain sets need `max_strains=` above 1 (C)**: with the default 1, `ss.immunity(due_to=L, rel_sus=0.79)` replaces the strain carried rather than adding one, and `DS >> (DS, RIF)` needs `max_strains=2`. This keeps the levels known when the strains are made.
26. **Strain results and values (C)**: `res.n_I.by('strain')` counts a person carrying DS and RIF in both (a set is `select(strain='DS+RIF')`); per-strain values are multipliers keyed by transition name (Covasim's `rel_beta` is given as the transition it scales); a superinfection below a full set restarts the course's clock as agents (unchanged).
27. **Seeds as agents lose a step (fix2)**: agents seeded at the start are dated a step before it, so they lose a step of infection (0.48 still infected after the first yearly step against 0.607 exact). The same on the original HEAD, and the code says it is deliberate; changing it moves every agent run with initial prevalence. Left as is.
28. **Large steps (fix2)**: a treatment `interval=` under half a step now gets a note (a 30-day interval in a yearly sim was one round a year, which explained a 7,180 vs 609 ODE/agents gap); there is no such note yet for a transmission rate times `dt` well above 1. A data column `n_I[strain=DS]` for strains carried together counts those carrying DS alone (as `select()`), not everyone carrying DS (as `by('strain')`).
29. **After the merge review (fix3)**: strain values for a `per='strain'` transition whose `p=` names the chances (`p='cure'`) multiply its rate, as a strain value multiplies its transition elsewhere (rather than an error); infections made from code (`dis.infect()`) are dated at the step's end like transmissions, so they take no share of that step's competing rates.
