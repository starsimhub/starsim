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
5. **Every open question below is answered by Cliff or moved to [admin/deferred.md](../admin/deferred.md).**
6. **F1 and F2 are done**: measured against S0.3 with the new benchmarks, and the records updated.
7. **Checks pass**: the full suite, `tests/switching` apart from known timing flakes, and the perf guard, with no benchmark slower than its baseline.

These are not part of it: downstream ports and migration (F3), the docs (F4), S1.7 (stepwise kernels, only if needed), and S7.9 (the docstring pass, optional). A new idea goes to deferred.md. If one is added here instead (only for an SWA, or because it blocks an item), the estimate goes down and the update says why.

**Estimate**: about 5% done (2026-10-22: Cliff's comments are in and the scope is set; nothing is built). Weights: S 1, M 3. Total about 60 units. Update this line with every edit. It should only go up.

## Order

1. **Now**: B0 (ST28 aging, FP57 `ss.by` speed, the missing benchmarks).
2. **Architecture**: A1–A3. Several B items depend on them.
3. **Correctness**: B1–B8.
4. **Strains**: C.
5. **The rest**: D and E (in parallel with B where the files don't overlap), then F.

## A. Architecture (Cliff's comments call for a rethink)

- [ ] **A1. `ss.Disease` is the base of every disease again (M)** (theme 4; MA22, MA41, ST41, HP53, TB §4.6, FP71). Cliff: "this is an architectural mess. the whole point is that declared and coded parts should be able to live peacefully together." So:
  - `ss.Disease` is the one base. It takes declared states and transitions, coded `step()`/`set_prognoses()` overrides, or a mix of both in one class. "Declared or coded" becomes a question about each part, not about the class.
  - `ss.Infection(ss.Disease)` adds transmission and nothing else.
  - `isinstance(x, ss.Disease)` is true for both. Malnutrition is a plain `ss.Disease`. v3's `class X(ss.Disease)` works as before. `DiseaseBase` goes away.
  - This partly reverses S5.8, which made two classes. Keep S5.8's real gains: the branches on `graph is None` stay gone. Each part's lifecycle runs if it was declared, and a coded override runs if it was given.
  - Spec first (00, 01, 03). Revert the migration script's `ss.Disease` → `ss.Infection` rewrite, except where the class transmits.
- [ ] **A2. `x[self.ti]` must just work (M)** (theme 3; CV2, CV6, FP1/FP42, ST35, ST39, ST8, MA41). Cliff: "nothing could be simpler than 'we're in a loop, here's the value at the current loop index', so if that's broken, ... something more systematic has gone awry." So treat it as a loop bug, not a user trap:
  - Find why `self.ti` means the start of the step in `step()` but the end of it in `update_results()`. Make it mean the same in every phase of one step, for coded and declared modules, deliveries and transitions alike (that also fixes ST35).
  - Then CV6 (`run(until=)` one short), ST8 (interventions' results before the first step), ST39 (a stored `ti` makes durations one step), and MA41 (nothing writes the initial point) should fall out or become one-liners.
  - `.now` is fine as an extra for corner cases, but it is not the fix. Restore the end-of-run warning for results written at the wrong index only if a case remains.
  - Start with a short write-up of the current convention and the proposed one, for Cliff, before changing the loop.
- [ ] **A3. ODE forms written outside Starsim (M)** (theme 7; MA38, HP18, HP28). Cliff: "if the user can write it to look like an ODE, we should be able to consume it as an ODE", and Starsim can't hard-code every waning shape.
  - First check what malariasim can already write. Box-then-decay as two declared states (a fixed dwell with `after=`, then an exit rate) uses only forms that S3 kept, so it may run under the ODE today. If it does, the fix is the spelling: a product's waning written as states. If it doesn't, find out why.
  - Then the general route. A module can give `compartmental_form(sim)` (S3.2, used by `ss.Pregnancy`). Decide whether that, or a declared state chain, is the public extension point for shapes like "decay at one rate, then another", and write one example from outside Starsim.
  - HP18 (an outcome that depends on the drawn dwell) and HP28 (a CV above 1): say whether the same route covers them. If not, they stay agents only, and FP53's guard (B3) covers `auto`.
  - Don't restore the 40-line box-then-exponential special case unless this route fails.

## B. Correctness and crashes

Cliff on tests: "can probably skip the regression tests; the test suite should not become a creaky history of every random bug." Write tests only for real gaps: the contract tests named below, plus one where a fix has no other cover.

- [ ] **B0. First (S each)**:
  - **ST28**: nobody ages in any `sti.Sim` (`use_aging` overwritten, `dimensions.py:719`).
  - **FP57/CV11**: `ss.by.lookup()` rebuilds its table from pandas rows on every call, which makes declared models 3–107× slower. Build the grid once and index it. Cliff: "do NOT use pandas in any hot path!" Grep the other step-time paths for pandas while there.
  - **Benchmarks**: add malariasim's `fseis10`, a declared model with counts, a coded model, and one with an `ss.by` age table to S0.3's set.
- [ ] **B1. Nothing accepted then ignored (S, plus S per case)** (theme 1). At `init()`, check every modifier, change and product, and raise unless something reads it. Then implement the cases (CV1 in `infect_route()`, ST30, MA36) or refuse them (FP52 until built, FP54, HP6, FP41, HP46, D23, ST37).
- [ ] **B2. `init=` (S)** (theme 2; MA34, HP48, FP69, ST31). Up to 1 is a share and above 1 is a count, as with `init_prev=`. Cliff: an int is still read as a share (`init=[1, 0, 0]` means everyone), with a warning to write a float. Functions are accepted as with `init_prev=`. Change the note to "the other 9,999 people". For exactly one person, see Q1.
- [ ] **B3. Solvers agree and conserve (S each)** (theme 6). Two contract tests: people are conserved in every crossing (FP51), and agents match the ODE on the spec's own models, with the analytic `R0()` (MA37). FP53: under `auto`, run agents when a dwell's CV is above about 3, and warn under an explicit ODE. Also ST32 (`ss.constant(0)` waits a step under compartmental solvers) and MA6 (chain-binomial mosquitoes die out at a 30-day step).
- [ ] **B4. Births and newborns (S each, one batch)** (theme 8): FP43–FP50, HP37, ST34. Add one contract test that the engine's draws match `dist.rvs()` (FP48). For FP50, don't backfill a history for people placed by `move_to()`. FP46 (2.5% of births lost, cause unknown) needs a diagnosis first.
- [ ] **B5. Units and step boundaries (S)** (theme 15).
  - Refuse bare numbers returned by a rate function when the model has typed values (ST33).
  - Fix HP47 and FP68's message.
  - **S7.8's bug** (SWA, carried over): a bare `ss.contacts()` matrix is read in the sim's unit, so with monthly steps, acts per year are read as acts per month.
  - Add a "timing contract" section to spec 02, with `describe()` lines for CV5 and ST4.
  - FP55: Cliff isn't sure why a dwell would need a minimum of one step. See Q2. Go ahead with the proposal there unless he objects.
- [ ] **B6. Expression algebra (S–M)** (theme 13): make `ss.by` an expression node (FP60). Add one table-driven test: each node type × each operator × each solver, on a tiny model. That covers HP47, HP49–HP51, FP61–FP63, ST46, ST47 and ST49.
- [ ] **B7. The remaining SWAs from the reviews (S each)**: CV3 (a delayed result released at the start of a step doesn't block a retest in the same step), CV4, HP38–HP42, HP44, MA35, ST29, and any others in the synthesis's list that A1–B6 don't cover. Check each still reproduces, then fix it or point it at the item that did.
- [ ] **B8. Leftovers from the October 21 plan (S each)**:
  - **Where people go when no state is named** (decision 3). Cliff leans explicit: a state named `S` or `susceptible` works; otherwise require the state given to `ss.birth()` or `first=`. Raise at `sim.init()` when it is needed and missing, and apply the same rule to initial placement.
  - **MA15**: re-dosing a product whose waning is a function gives a wrong answer. It needs a spelling, which may come from A3.

## C. Strains (S7.4)

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

**Open (see Q3–Q5)**: transmission from a mixed carrier (D5), the spelling of emergence (no `Tx → Tx`), and a clearer cure example.

**Answer to Cliff's 2^k question** (table row 3): own courses with no coupling are independent, so each strain gets its own copy of the states. That is k copies side by side, which the ODE handles. 2^k appears only when courses interact, because then the combination a person carries matters. So row 3 is fine, and row 4 (coupled) is agents only.

## D. Speed, compiling, calibration

- [ ] **D1. Hot spots (M)**: counts stepped in Python as agents, and `series()` re-tabulated at every derivative call (MA39, MA28); `Arr.__setitem__`'s copy-on-write check (10–15% on every coded model); `ss.years(array)` (FP59, 1 line). FP58 goes to deferred.md as a measured cost.
- [ ] **D2. Compile automatically when it pays (S)** (theme 11; ST50, MA39). Count substeps in the estimate. Cliff's rule:
  - if compiling takes less time than the run would save, compile automatically, with a note;
  - if not, run plain and print a note with the estimated plain and compiled times.

  Never compile silently, even at `verbose=0`.
- [ ] **D3. Calibration (S each, plus M)** (theme 10):
  - fix the silent ones: ST36, ST37, ST38;
  - fix the crashes: ST44/FP8/CV10, ST51;
  - read an integer `year` column as a calendar period;
  - add `n_workers=` and `storage=` (M);
  - add fitting of a coded intervention's settings (CV10);
  - add mapping a data column to strain members (HP54; after C);
  - MA39: three searches for a one-parameter deterministic fit.

## E. Networks and what downstream packages hit

- [ ] **E1. `ss.SexualNet`, all four items (S–M)** (theme 14; Cliff: "do all four"):
  - HP45: don't pair on behalf of subclasses that pair their own.
  - ST45: the error suggests `sides='mm'`.
  - ST22/ST52: two-group partnerships (sex workers and clients) and risk matching come back, with the spelling in Q6.
  - `test_parameter_docs_coverage`: accept or fix base parameters that subclasses don't use.
- [ ] **E2. Small fixes downstream code hits (S each)**:
  - `sim.current_scale` becomes a public property again (CV7).
  - `verbose` fractions mean progress every 1/verbose of the run (CV9). On Cliff's question: Starsim 3's default was `0.1` (`settings.py`, "an update every 10 timesteps"). v4 made `verbose` 0/1/2 with a default of 1, and refuses fractions, so every v3-era package that sets 0.1 breaks.
  - `interventions=` (and the other module lists) flatten one level of nested lists (TB §4.1).
  - `Results.flatten()` keeps case, as `to_df()` does (TB §4.3).
  - Hints on three bare errors: `infectious=`, `ss.library` without its import, and `match_result_keys` (TB §3).
  - `.since` on transitions (TB §4.2), so that `tb.diagnosis.since <= ss.years(2)` works without `keep_times`. Do it if it uses the delivery memory as is; otherwise defer it.
  - A module copied into a sim and never run raises on `.results`, pointing to `sim.analyzers.<name>` (TB §4.4). This one is optional.
- [ ] **E3. Decisions to record (S)**:
  - The reversal of `oct17/decisions_oct18.md` item 3: no `ss.Calibration`; `sim.calibrate()` replaces it.
  - Starsim 3's coded `ss.Pregnancy` is ported by rewriting onto the declared one: add FP41's `__init_subclass__` refusal, ST42's renames, and a declared newborn transition for infant mortality.

## F. Measure and close

- [ ] **F1. Measure against S0.3 (S)**: as S8.1 of the October 21 plan, with B0's new benchmarks, and against hand-written code (Ghana measles in NumPy, LASER).
- [ ] **F2. Records (S)**:
  - LFD's `design/` is a straight copy of `v4_design/design/`.
  - deferred.md is updated, `CHANGELOG.md` gets brief lines, and the note for Robyn is written (A12).
  - The [Stale](../oct21/plan.md#stale) list carries over and gets what this plan adds (at least A1's `ss.Infection` rewrite in the docs and skill, and C's spellings).
- **F3. Downstream and migration** (after the design is settled; not in the definition of done):
  - tbsim, STIsim, HPVsim, FPsim, Covasim and malariasim run again.
  - The script rules: rewrite keywords only in calls to Starsim's deliveries, never in a `def`; don't delete `on_positive=`/`on_negative=`; rename `ss.SexualNetwork` and `copy_inputs=`; consider a `--prerelease` mode (TB §2).
  - TB §2's migration table becomes skill rows.
  - The downstream fixes in other repos (oct21 plan, "Where things stand") are written up.
- **F4. Docs**: once the implementation is final.

## Open questions for Cliff

Each has a recommendation, which I'll follow if Cliff doesn't object.

1. **Exactly one person in `init=` (B2).** `ss.count` doesn't exist yet. Recommendation: add `ss.count(1)`. It has other uses (introducing a strain with `n_imports`, seeding by number), so it isn't single-purpose. A separate `init_n=` keyword would be the alternative. `np.int64(1)` is too clunky, as Cliff said.
2. **A minimum of one step for dwells (B5, FP55).** Cliff asked why zero-time dwells shouldn't be allowed. They should be, when written: `dur=0` stays instant. The problem is a *drawn* dwell that is short but positive (a mean of 0.2 steps). It rounds to zero, so a self-loop re-decides in the same step: 749 of 774 women left `never` at once, or a `TriggerLoop` raises with a wrong hint. Recommendation: an explicit 0 stays instant, and a drawn positive dwell lasts at least one step (as in v3 and FPsim). The message is fixed too.
3. **Transmission from a carrier of two strains (C, D5).** Cliff: "does this mean someone with DS+RIF would pass BOTH along? surely not?" Under `'each'`, yes: each strain has its own chance at each contact, so both can pass. Recommendation: three options. The default `'one'` picks one strain per transmission, in proportion to each strain's relative transmissibility. `'fittest'` is tbsim's `transmission_independent=False`. `'each'` is kept only because co-carriage models use it today.
4. **Emergence without `Tx → Tx` (C).** Recommendation: a strain change is its own declaration, not a state transition. For example, `tb.strains.change(DS >> (DS, RIF), rate=0.05, where=Tx)`, where `where=` limits it to people on treatment. The tbsim build tries it against `move=` on the treatment-failure transition (ST30) and keeps the simpler.
5. **The cure example (C).** Cliff found `ss.Transition(Tx, R, prob=0.9, per='strain')` unclear. Recommendation: rewrite the examples so cure is treatment ending (a duration on `Tx`). The per-strain chance of success then comes from `adjust=dict(cure=...)`, and `per='strain'` means each strain carried succeeds or fails on its own. Spontaneous clearance is a separate transition.
6. **`sides=` vs `between=` (E1).** `sides=` says which sexes pair (`'mf'`, `'mm'`). `between=(a, b)` says that each partnership has one person from each of two named groups (e.g. `('fsw', 'client')`). Man–woman is just `between=('male', 'female')`, so one keyword covers both. Recommendation, following Cliff's preference: keep only `between=`. It takes a pair of groups, with `'mf'` (the default) and `'mm'` as shorthands, and per-group `n_contacts`. Risk matching is a `match=` on top of it.
