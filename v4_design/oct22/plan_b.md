# Starsim v4: the second plan from October 22

Written 2026-10-22, after [plan.md](plan.md) reached about 98%. It comes from the five downstream reviews written this afternoon, all of `ssv4` at `598831ac`:

- [reviews/tbsim.md](reviews/tbsim.md) (TR, TS, TC, TH, and TB §);
- [reviews/covasim.md](reviews/covasim.md) (CV19–CV30);
- [reviews/fpsim.md](reviews/fpsim.md) (FP73–FP93);
- [reviews/hpvsim.md](reviews/hpvsim.md) (HP62–HP68);
- [reviews/stisim.md](reviews/stisim.md) (ST54–ST61, plus HS- and CB- from its part notes).

There is no malariasim review from today (I1 runs it). The reviews were made at `598831ac`. `fd56c1fd` ("update stale docs") came after and changed `starsim/interop/migrate.py` and the migrate-starsim3 skill: it renames `ss.SexualNetwork`, removes `copy_inputs=`, splits `on_positive=` follow-ups out instead of deleting them, and moves counts to `init_n=`. So G starts by checking which migration findings are still there.

The [constitution](../constitution.md) governs, and where this plan differs from it, the plan is changed. Sizes: S is under a day of agent work, M a few days. **SWA** marks a silent wrong answer. Tests only for real gaps (Cliff: "the test suite should not become a creaky history of every random bug"): the contract tests named below, plus one where a fix has no other cover.

## What the reviews say, in short

- **Speed and lines mostly hold up.** As agents, v4 is 1.2–1.6× faster than v3 for tbsim at 50k+ agents, Covasim, STIsim and FPsim (repeated runs). The declared parts are 7–16× shorter. The exceptions are fixed costs: about 1 ms per step for agents (tbsim at 1k–10k), building a sim (1.5–2.5× v3), the declared HPV strains (1.43× slower), and the declared FPsim contraception (4.2× slower).
- **The biggest group of new silent wrong answers comes from the step clock after A2.** Four symptoms in four reviews: a coded intervention never steps at `ti == 0` (CV19); seeds transmit a step less, and a one-step dwell never transmits, even in v4's own `ss.SIR` (HP62); declared progressions are dated `ti - 1`, so coded checks like `ti_P == self.ti` find nobody, and STIsim's ANC testing silently never happens (FP73, ST54); and `preg.P.time` from code is an array of steps with a sentinel (ST60). Plan questions 15, 16 and 27, and deferred.md's "date a declared progression at the step's index", are all this one issue.
- **`ss.Strains` has three silent wrong answers** (TR1–TR3), plus HP64 (cross-immunity weaker than HPVsim's, cause unconfirmed) and HP66 (a test result over a family drawn once per genotype).
- **The chain binomial is biased with no note** when a stay is about a step or shorter: +6% for tbsim weekly, 4 days late for Covasim daily, and 10–450% for STIsim monthly (TC1, CV25, ST55). Automatic substeps and the compile estimate are both badly off (TR4: 500× slower for the same answer; ST57, FP85, HP68, CV26: estimates 5–21× low).
- **`solver='auto'` still picks the ODE for models it can't run**, once silently (FP83, 3× low), and twice with a crash (TC2, TC3).
- **The "no hacks" target fails mostly on missing public seams**: the ports reach into `engine.selector`, `engine.fresh()`, `strat.states`, `graph.strats`, and raw arrays.
- **The migration still fails at `import`** for STIsim, HPVsim and Covasim (`ss.Calibration` with the hint "Did you mean: 'emigration'?"), and the script has known bugs. This was outside plan.md's definition of done; it moves the "no hacks" and lines scorecards for three models, so its v4 side is in this one.

Boxes in [targets.md](../targets.md) the reviews say can be ticked now: tbsim's "fewer lines" and "not expected to work out of the box"; Covasim's "faster as ABM"; FPsim's "no single-model code"; STIsim's "faster as ABM" and "not expected to work out of the box". (STIsim's "no single-model code" holds, but shares a box with "no hacks", which doesn't.)

## Definition of done

This plan is done when all of these hold, and not before or beyond:

1. **A (the step clock) is designed with Cliff and built**, with its contract tests, and its records corrected.
2. **Every SWA listed in B–E is fixed, refused with a clear message, or moved to deferred.md by Cliff.** HP64 and FP91 are diagnosed first.
3. **The crashes and wrong refusals in B–E are fixed.**
4. **F is done**: the new benchmarks are in `tests/benchmark_models.py` and the perf guard, and F1–F6 are built or measured and set aside with numbers.
5. **G (the v4 side of migration) is done**: the script, the removed-name messages, and the skill rows. Edits to downstream repos are not part of it.
6. **Every question under "Open questions" is answered by Cliff or moved to deferred.md.**
7. **I is done**: the six reviews rerun (malariasim included), `targets.md` ticked where they say so, and the records updated.
8. **Checks pass**: the full suite, `tests/switching` apart from known timing flakes, and the perf guard, with nothing meaningfully slower (Cliff: 1.2 → 1.5 s matters, 1.2 → 1.5 ms doesn't).

Not part of it: changes in tbsim, Covasim, FPsim, HPVsim or STIsim themselves (written up instead, per the October 22 [stale_other_repos.md](stale_other_repos.md)); the docs (plan.md F4); each model's own bugs; and anything in deferred.md that isn't named below. A new idea goes to deferred.md. If one is added here instead (only for an SWA, or because it blocks an item), the estimate goes down and the update says why.

**Estimate**: about 95% done (2026-10-22, 17:45: A2, A4, the strains-by-count refusal and two `tests/switching` failures done; up from 90%; total about 97). Left: I1 (rerun the six reviews, for Cliff to launch one per session with `/v4.review`), and Cliff's answers to the questions below. Weights: S 1, M 3. Total about 95 units. Update this line with every edit. It should only go up.

## Order

1. **First**: F7 (benchmarks, so the speed items have a baseline), A1 (the write-up, for Cliff), G1 (recheck the migration findings after `fd56c1fd`), and the open questions.
2. **The SWAs**: A2–A3 (after Cliff answers), B1–B3, C1, D1, E1.
3. **Crashes and wrong refusals**: B2's crash, C4, C5, D2.
4. **Speed**: F1–F6.
5. **Vocabulary and the compartmental goal**: B4, B5, C2, C3, C6, C7, D3.
6. **Migration**: G2–G6.
7. **Close**: I.

B, C, D and E touch mostly different files and can run in parallel ([[parallel-worktree-agents]]).

## A. The step clock after A2

- [x] **A1. Write-up for Cliff (S)**. *Done*: [step_clock.md](step_clock.md). HP62 turned out not to be the seed date (CR4 needs it), but where a seed leaves relative to the network pass; an agent has it. Symptom 3 is the one question. One page, before changing anything, as for A2. The four symptoms (CV19, HP62, FP73/ST54, ST60), each with its repro, and the options:
  - **Declared moves dated at `ti`** (deferred.md's "date a declared progression at the step's index", promoted). Then `ti_X == self.ti` means "this step" for code and engine alike. Reopens decision 16 and question 16. M.
  - **Seeds count from the first step** (HP62, and question 27, "left as is", which HP62 shows is wrong for v4's own `ss.SIR`): a seed with a dwell of k steps transmits for k steps. Contract test: a seed with a one-step dwell transmits exactly once.
  - **Coded schedules from Starsim 3 (CV19)**: v3's `days=0` was the first step, which is now `ti == 1`. Options: a clear name for the step's start index (`self.t.ti_start`), called out in the migration row; or interventions' `step()` also called at `ti == 0`, as analyzers are. The guess is the first, since a call at `ti == 0` is a step that isn't one.
  - **Question 15** (Cliff: split a coded step into working out the changes and applying them, as Ghana measles does). Say whether it removes any of the above, or is separate.
- [x] **A2. Build what Cliff picks (M)**. *Done 2026-10-22*, as option A (the guess; [worklogs_b/a2_stamps.md](worklogs_b/a2_stamps.md)): every `ti_X` reads as the index of the step the move happened in, and `ti_exit_X` as the step the exit is taken in; the engine keeps its own times (`StepArr.internal()`), so results are bit-identical on every benchmark and `baseline.yaml` is unchanged. STIsim's `ti_P == ti` finds each step's conceptions (0 before). Contract test `test_clocks.py::test_stamps`. HP62's part: see [step_clock.md](step_clock.md) symptom 1.
- [x] **A3. `.time` from code (S; ST60, SWA)**. *Done*: a duration in the sim's unit, NaN outside the state (not for those being split as they leave); 34 in the first trimester against 34 by hand (5,946 before). `preg.P.time < ss.weeks(13)` is True for 5,957 of 6,000 agents. Make it duration-aware (or NaN outside the state) so the comparison is right or raises; give `ss.Pregnancy` a gestational age and `since_delivery` if A2 doesn't make them one-liners (FP73).
- [x] **A4. Records (S)**. *Done*: the skill's rows (and CV19's), specs 02 and 10, `a2_timing.md`: `a2_timing.md:29` ("needs no index rewrite at all"), the migrate skill's timing rows (FP92; recheck lines 89, 91, 96 and the gestation formula at 39 after `fd56c1fd`), `docs/migration/v3_v4.md:35`, plan.md questions 16 and 27.

## B. Strains

*Done 2026-10-22*: see [worklogs_b/b_strains.md](worklogs_b/b_strains.md). TR1–TR3, HP66, HP65 and CV23 fixed (L carrying A+B 37,479 → 0; ASCUS 0.346 → 0.102; women in the men's state 789 → 0); `'fittest'` removed, `'each'` fixed and tested; HP64 is about half explained by `dur=` protection applied during the stay rather than at infection (question 8); TR5–TR9 deferred, with what each takes; F5 4.40 → 3.64 s.

- [x] **B1. Three SWAs in `ss.Strains` (M; TR1–TR3)**:
  - **TR1**: states after clearance keep their strain, so a reinfection adds to it (37,479 of 37,500 latent people carry A+B, expected 0). Carry strains only in states that can lead back to disease without a new infection, with `ss.Strains(states=)` to override, and list them in `describe()`.
  - **TR2**: `ss.immunity(due_to=N)` misreads a non-infectious disease state. Decide "infected" by whether the state carries strains.
  - **TR3**: nobody can start out carrying a set. `split=` by set label (`{'DS': 0.5, 'DS+INH': 0.25}`), and `init=` over a strain dimension exact or an error (today it gives A 250, A+B 0, for 500 and 500).
  - Then check tbsim's reference model without its workarounds (`strat.states`, `strat.split` set by hand) against `two_strain_ode`.
- [x] **B2. Families (S each; HP66 SWA, HP65 SWA, CV23)**:
  - **HP66**: a state shared by name across members is one condition, drawn once (ASCUS 0.346 against 0.10); `ss.state('hpv.Ca')` on a family works.
  - **HP65** (was HP41, handed between B7 and C and tracked in neither): seeding goes through every transmission, not the first; a family with no transmission gets a clear message, not `IndexError` (`agents.py:515`).
  - **CV23**: well-mixed agents with strains and a modifier reading an attribute crash, from a rate cached for a subset (`solvers/agents.py:2737–2742`).
- [x] **B3. HP64: diagnose first (S)**. Cross-immunity is 0.168 against HPVsim's 0.250 (z −7.9). Confirm or reject the hypothesis (`ss.immunity(dur=)` rescales the remaining stay by the current protection; HPVsim shortens the drawn duration once, at infection) with more seeds. If confirmed, the fix becomes a question for Cliff (an option to apply at entry, or documenting the rule).
- [x] **B4. Strain features with one user or none (S; F7 of tbsim, CV30, HPVsim's single-model list)**:
  - remove `transmit='fittest'` (no user; the plan's decision 3 maps it to tbsim by mistake, and tbsim's rule is `'one'`); correct decision 3;
  - tests for `transmit='each'` and `Transition(alone=True)`, or replace `alone=True` with `move=(A, B) >> A` if that says the same;
  - `ss.Strains(cross_immunity=)` and `covasim_matrix()`: plan.md C settled Covasim's layout as one of the three ways to give values, so keep it, with a test and a docstring example;
  - one non-HPV test for the own-course family, `ss.immunity(due_to=<attribute>, combine='sum', dur=)`, and `cure=` by state (e.g. two dengue serotypes).
- [x] **B5. What tbsim writes as copies of transitions (M; TR5–TR10)**. Scope: what removes tbsim's ~55 lines of set-to-set copies, no more. `per='strain'` with acquiring resistance at the same event, and with part of the cured relapsing (TR5); several strain moves per transition, and a move with chances (TR6); `ss.immunity` values as `ss.par` so they can be fitted (TR7); values for carrying several strains (TR8); unordered sets when nothing replaces a strain (16 levels → 10 for two drugs; TR9); a public spelling for `StrainMove(before, after)` with tuples (TR10). Whatever is harder than it looks goes to deferred.md with what it would take.

## C. Solvers: the binomial, substeps, `auto`, and the compile estimate

*Done 2026-10-22* (C6 in part): see [worklogs_b/c_solvers.md](worklogs_b/c_solvers.md). The binomial takes substeps so the fastest starting rate × dt ≤ 0.1 (ST55 0.505 → 0.146 against the ODE's 0.091; importers of discrete-time models ask for `substeps=1`); an ODE growth rule that skipped a fixed delay's first slot gave a negative flow (RF1, found here) and is fixed; TR4 265 → 3.8 s; whether to compile is decided by timing one plain step (within 2.6×, was 5–21× low); FP83, TC2, TC3, FP89, CB-9 and CV24 fixed. Left (deferred): TC1's residual +4.7%, CB-2, and sparse maps for FP84 (a map over 2 GB is now refused with its size).

- [x] **C1. The binomial with short stays (M; ST55, TC1, CV25, CB-2; SWA)**. Choose the binomial's substeps from the rates, as the ODE's are, and note when it runs at one substep with rate × dt above about 0.1. CB-2 (a fixed stay under a step is rounded up under the ODE and binomial): promote deferred.md's "split the inflow between floor and ceiling slots". Contract test: binomial ≈ ODE for a stay of one step.
- [x] **C2. Too many automatic substeps (S; TR4, CV26)**. 16 strain levels take 265 s with automatic substeps and 0.5 s with `substeps=1`, equal to 1e-9. Find why so many are chosen, say how many in a note, and cap them by the accuracy they buy.
- [x] **C3. The compile estimate (S–M, counts 2; ST57, FP85, HP68, CV26)**. `plain_seconds()`/`kernel_seconds()` are 5–21× low, so stiff ODEs run plain for minutes. Calibrate them on measured runs (per compartment × substep, including the derivative's Python overhead), and add a check that the estimate is within 2× on F7's models. Cliff's D2 rule stays. (D2 in plan.md is marked done; this reopens it.)
- [x] **C4. `auto` and the refusals (S each, one batch)**:
  - **FP83 (SWA)**: `dwell_cv()` is NaN for a dwell by group, so FP53's guard is skipped and FPsim's mCPR is 3× low under the ODE. A Weibull's, gamma's or log-normal's CV depends only on its shape; use that. Also deferred.md's "FP53's guard for transfers and dwells by group".
  - **TC2**: a property that isn't a dimension (`where=ss.ppl.age < 1`) is a reason to run as agents.
  - **TC3**: per-agent reads in a compartmental run raise a `CapabilityError`, not `AttributeError: 'CountState' object has no attribute 'raw'`.
  - **FP89**: a modifier's amount `1 - 0.9*m.on` resolves states by their own module.
  - **CB-9**: `compile=True` raises at `init()`, not at `run()`.
- [x] **C5. JAX on a CUDA machine (S; CV24, HS-2)**. The chain binomial's FFI kernel is CPU-only, and `choose_engine()` picks JAX for long runs, so it fails with no setting. Run that kernel on the CPU device, or don't choose JAX when the default device isn't the CPU.
- [x] **C6. A dense `np.kron` in crossings (M; FP84)**. 3,264 × 18 cells asks for 25.7 GiB. Apply the factors one dimension at a time, and have the `max_joint` refusal say what raising it costs.
- [x] **C7. Covasim's strain features under the ODE (M + S + S; CV29)**. The Covasim target names strains. In order of value: a "last strain" level kept after recovery, giving `strains.last` and `p=` a compartmental form (deferred.md, promoted; M); `ss.Delivery(move=S >> E, strain='alpha')` for an import on a date (deferred.md, about 30 lines; S); a distribution-valued `ss.attr` run at its mean with a note, as `spread_pars()` does (S). Clocks as Erlang stages wait in deferred.md. See question 4. *Done 2026-10-22* ([worklogs_b/c7_strain_memory.md](worklogs_b/c7_strain_memory.md)): what people remember of strains (`last`, `ever`) is a dimension of a compartmental run, so `ss.immunity(due_to=strains.last)` and `.ever` run under every solver (KR 4.1 matches the book's equations to 1e-7); `ss.Delivery(move=S >> E, strain=)`; a trait runs at its mean with a note. Found and fixed: with `strains.ever`, protection on the course counted the current infection's own strain (a first infection symptomatic 5.9% of the time, not 60%). Covasim well mixed with every cut feature back but the viral load: ODE 59,796 against agents 62,323 infections, peak day 46.0 against 46.5.

## D. Declared and coded vocabulary

*Done 2026-10-22*: D1 and D2 in [worklogs_b/d_vocab.md](worklogs_b/d_vocab.md), D3 in [worklogs_b/de_seams_calibration.md](worklogs_b/de_seams_calibration.md). Every SWA fixed (FP75 had a second cause: `ss.Deaths` also killed fetuses); `{net: beta}*0.9` can't be fixed (a plain dict) and is deferred. New public names, for Cliff to check: `ss.pars()`, `ss.Vaccine.held` (a product class's `.held`), `Disease.strats`, `Disease.add_callback()`, `.ever`/`.count`/`.since` on transitions, `Disease.count()`, `sim.groups.level()`. FP86 is done as agents only (deferred for the ODE).

- [x] **D1. SWAs (S each)**:
  - **FP74**: the automatic `n_<state>` results of a coded `BoolState` in a declared module read 0. Count them.
  - **FP75**: a pregnant woman's death from another cause counts as a loss only with `fetus=True`. Record it in `step_die()` either way.
  - **FP76**: an infinite hazard moves nobody. It is certain within the step.
  - **FP80** (and MA37's `R0()`): a callback added before `sim.init()` is lost, because the sim runs a copy. Carry callbacks across the copy; look for other per-object state lost the same way.
  - **FP87**: a histogram inside `ss.by` as a dwell draws uniform(0, 1); send each level through FP48's rebuild (and fix the trait form's crash).
  - **FP88**: a derived `ss.attr` is NaN until the first step, so placing people from it places nobody. Evaluate on read.
  - **CV20**: `priority=[due, ss.ppl.age]` reads age as "age > 0". A non-boolean item is a score, or refused.
  - **CB-6**: `sim.groups.level_raw()` gives levels to people outside a dimension's `where=`.
- [x] **D2. Crashes and misleading errors (S, one batch, counts 3)**: FP77 (call rate functions with the UIDs wanted, not raw arrays with sentinels); FP90 (`ss.Delivery` takes its module from `move=`, not `where=`); CV27 (a delivery whose `who=` names a coded disease via `ss.state()` loses it; `mod.pars['beta'] = ss.par(...)` stored raw); CV28 (a clock used only inside an `ss.attr` isn't registered); CB-5 (`split=ss.by(sex=)` bare reshape error); HP52 (`disease=` over coded diseases); HP55's rest (cured women in men's clearance flow, scalar `cure=` curing cancer, the `to="S"` hint); `describe()` printing a `dist*expr` dwell as exponential; the three small `TypeError`s (`np.abs(ss.normal())`, `{net: beta}*0.9`, `dist*1.0`); the `ti_delivery` hint pointing at `ti_delivered` rather than `ti_exit_P` (ST59).
- [x] **D3. Public seams for what the ports do through internals (S each, counts 8)**:
  - **TH1**: "holds a product of this kind" as a condition (`ss.held('tpt')` or `where=<Product>.held`), for tbsim's `holds_tpt()`.
  - **TH2**: `self.define_pars(defaults, **kwargs)` public, for diseases whose structure depends on their options (tbsim's 28-line `make_pars()`; STIsim's `define_pars(eff_condom=)` too).
  - **TH4**: `tb.strats.age.edges`, rather than scanning `graph.strats` by name.
  - **FP78**: `preg.add_callback(event, fn)` for any transition, so losses aren't read with `engine.fresh()`.
  - **FP79**: the v3-override guard refuses only the methods v3's `step()` called as hooks, not `fecund`, `updates_pre` or `_get_uids`.
  - **FP86**: a spelling for "re-decide no sooner than the next step" after decision 2 (`min_steps=1`, or a self-loop's dwell lasting at least a step).
  - **TH5**: `.since` on transitions (deferred.md, about 40 lines; asked for by tbsim twice).
  - **The compartmental run's missing parts**: a public current count (HS-5, for ART to a target), `n_alive` and `new_deaths` (HS-3); and ST9 (an `ss.par` read by coded network code). HS-9 (`UndrivenUnused` on every run without care) goes with these.

## E. Calibration

*Done 2026-10-22*: see [worklogs_b/de_seams_calibration.md](worklogs_b/de_seams_calibration.md). The per-trial solver line is STIsim's (`separate_pars()` takes the sim's `verbose`), written up for STIsim.

- [x] **E1. Silent fits (S each)**: CV21 (`sim.calibrate()` drops module parameters changed after the sim is built; copy the uninitialized sim as `ss.parallel()` does); CV22 (one `ss.par` held by two attributes fitted as two; register by object); ST56 (`Module.update_pars()` drops bounds, so `sti.Sim(f1_conc=ss.par(...))` fits nothing; plan.md D3 marked ST37 done).
- [x] **E2. Crashes and friction (S)**: CV10's default goodness of fit on a `date` column; `ss.fit.compare(sim, df)` with a `year` column (not the index) read as times 0–11; "solver: abm (n_agents was given)" printed at every trial with `verbose=False`.

## F. Speed

*Done 2026-10-22* (F1 and F3 in part): see [worklogs_b/f_speed.md](worklogs_b/f_speed.md). Four benchmark models and 8 perf-guard cases added; `ss.by` and `p=` rows (contraception 0.61× the time, bit-identical); scalar competing draws and lighter event batches, plus one network pass for compatible transmissions (TB 0.82× at 1k agents, 0.89× at 10k; draws change, distributions don't); `Dist.rvs()` 13–30% faster. F3: 44–83% of `sim.init()` is Sciris's `sc.search`; a three-line Sciris change is proposed in the worklog (10–25% off build time), not made. The rest of F1's per-step cost is deferred.

- [x] **F7. Benchmarks first (S)**: add to `tests/benchmark_models.py` and the perf guard a model shaped like the declared HPV (strains, `ss.attr`, `p=` of a clock; HP63), tbsim's lshtm case at 1k and 10k agents (TS4), the declared FPsim contraception (FP82), the 16-level strain ODE (TR4), and a build-time benchmark (TS6, FP81).
- [x] **F1. Fixed per-step cost for agents (M; TS4, HP63)**. `Agents.step_state` is 1.16 s of 2.76 s at 1k agents, mostly `draw_competing` (~12 calls a step) and `finish_moves` (4.9 of 8.2 s in the declared HPV). Profile by parts first ([[preferences/working-style]]: time parts before trusting cProfile), then batch.
- [x] **F2. One pass per network (S–M, counts 2; TS5)**. tbsim's four contact transitions each pass over the network (1,200 calls in 300 steps, 0.61 s against v3's 0.31 s). Group transitions that share a network into one pass.
- [x] **F3. Building a sim (S–M, counts 2; TS6, FP81)**. 1.5–2.5× v3, mostly `sc.search` in `link_dists` and `link_timepars`. If the cost is in Sciris, write it up as a Sciris issue for Cliff rather than working around it. Locate the first run's extra 15% in FPsim (likely compiling) while there.
- [x] **F4. `ss.by` and branch probabilities (S + M; FP82, CV11)**. Key `ss.by.lookup()`'s cache on a tuple, not `str()` of an array (1 line; 37.9 → 29.2 s). Then stop re-evaluating branch probabilities as expressions for every transition at every step (33 of 69 s): cache a table's lookup within a step, or compile each row once.
- [x] **F5. Strain protection (M; HP63)**: compute each member's attributes once per step, not once per pair (1.8 s, 12,800 `nan_to_num` calls).
- [x] **F6. `Dist.rvs()` per call (S–M, counts 2; HP67)**: 78,628 calls at ~100 µs in `hpv.demo()`, 8 of 19 s. Find what the 100 µs is.

## G. Migration (the v4 side)

*Done 2026-10-22*: see [worklogs_b/g_migrate.md](worklogs_b/g_migrate.md). `fd56c1fd` had fixed the `SexualNetwork` rename, `copy_inputs=`, TB §2.1 and §2.2; the rest is built (removed names say what replaced them, for all 76 of Starsim 3.7.2's root names that v4 lacks; `MultiSim.make_run()` with `run_options`; the `uids.cat` shims removed). The perf guard needs one quiet rerun after the merges.

- [x] **G1. Recheck after `fd56c1fd` (S)**: which of ST59, HP53's two rules, TB §2.1 (a `def` parameter rewritten) and TB §2.2 (`on_positive=` deleted) remain. Tick what is done here.
- [x] **G2. Removed names (S)**: a table in `starsim/__init__.py`'s `__getattr__`, so `ss.Calibration` says "use `sim.calibrate()`" (CV7, CV8, ST59, HP53), and no closest-name hint for a removed name.
- [x] **G3. The script (S–M, counts 2)**: flag `compute_transmission()` and `prenatal_death_keys` with their replacements (ST43); rewrite `preg.breastfeeding`; the rules TB F9 lists as uncovered (`cum_*`, `years=`, `ss.transfer(coverage=)`, `ss.Treatment(efficacy=1.0, transition=)`, arrow strings in `ss.birth`/`ss.death`/`ss.immigration`, `'+'` levels); flag `ss.People(n)` with no `age_data` (now the world's ages: −19% infections in `hpv.Sim()`) and `MultiSim.median()` over different labels; `uids.cat` → `uids.concatenate`, then remove the shim (LEAD-8, "no baggage"); and whatever A2 changes.
- [x] **G4. Coded packages' run code (M; CV9, H8)**: `make_run()` as a `MultiSim` method, with run options a subclass can declare, so Covasim's 112 copied lines go. MultiSim's `verbose` handling with it.
- [x] **G5. Attributes hidden by a subclass's `__getattr__` (S; CV8)**: make `auids`, `uid`, `slot` and `parent` properties (`people.py:163–167`).
- [x] **G6. The skill (S)**: TB §2's rows; ST58's period rule (question 3); the A rows.

## I. Measure and close

- [ ] **I1. Rerun the six reviews (M)**, malariasim included, starting each from the ports the reviews left (`v4port/oct22/` in each repo). Tick `targets.md` where they say so, starting with the boxes listed at the top.
- [x] **I2. Diagnose what's untraced (S)**. *Done*: FP91 was an SWA in v4 ([worklogs_b/fp91_traits.md](worklogs_b/fp91_traits.md)): traits drawn in `people.grow()` before the caller set the newcomers' ages, so fetuses got adults' attainment (414 of 1,157 girls born 2000–2005). Traits of people grown without `age=` are now drawn at the next declared step; in school 0.0260 against FPsim's 0.0274 (z 1.3, was 6). STIsim's migration and HPVsim's immigration had the same exposure. FPsim's infant deaths and TFR don't go through this path: deferred.: FP91 (the declared education's share in school moved 27% since `ee5d1fb0`; candidates are decisions 2 and 13, so it may be an SWA in v4). The rest go to deferred.md as untraced: FPsim's infant deaths and TFR, Covasim's `test_nabs` and `test_vaccine_target_eff`, STIsim's HIV-6 and #35, HPVsim's network age-gap SD.
- [x] **I3. Records (S, counts 2)**. *Done* for the first round; synced again at the end:
  - deferred.md gets the older findings the reviews say are tracked nowhere: FP2, FP9, FP12, FP22, FP23, FP28, FP32, FP39, FP64, FP67, FP70, FP72; CB-4 and CB-7 (sex work and age preference on `ss.SexualNet`), CB-11 (BV's ODE 10× agents); HPVsim's family results without members' transfers;
  - plan.md: decision 3's `'fittest'` mapping, and notes that D2 and ST37 were reopened here;
  - `targets.md` ticks (now, and after I1);
  - LFD's `design/` and `admin/` synced, brief `CHANGELOG.md` lines, and the Stale list carried over;
  - the downstream bugs (each review's "own bugs" section) and the skill-table notes, in `stale_other_repos.md`.

## Open questions for Cliff

Each has the guess the build follows meanwhile.

1. **The step clock (A1).** Answered by the write-up. Guess: declared moves dated at `ti`, seeds count from the first step, and a named start index for coded schedules.
2. **Interacting strains under the ODE (HPVsim and Covasim targets).** Spec 12 makes interacting strains in `ss.Strains` agents only, but both targets say "including strains". Options: accept "partly" for HPVsim; bring back per-member shares as an approximation that scales; or change the targets. Guess: accept partly for HPVsim, and do C7 for Covasim, whose cases have exact forms.
3. **Sparse year data (ST58).** v4 holds each label until the next (right for UN 5-year labels); v3 took the nearest year, so STIsim's decade data moves 1990s mortality to 2000 (deaths +7.8%). Guess: keep the rule, note when the data's years are more than about 2 apart, and add a skill row. *Built (2026-10-22)*: `ss.Births`/`ss.Deaths` note when user data's labels are more than 5 years apart during the run (`note_sparse()`), plus a skill row. Found on the way: Starsim's own named-place UN rates are every ten years (`ss.data`), so `ss.Deaths(death_rate='Kenya')` also holds 1990's rates until 2000; the note skips them, since it would fire on every such sim. If those are single-year estimates, the better fix is to interpolate between labels more than 5 years apart (or bundle annual rates); your call.
4. **C7's scope.** Guess: the "last strain" level, `strain=` on a delivery, and an `ss.attr` at its mean; clocks as Erlang stages wait.
5. **Two competing non-exponential exits under the ODE (ST61).** STIsim's SEIS and BV need it, and it is inside the STIsim goal; `support_matrix.md` says agents only (ST24). An exact form races the two Erlang clocks as k1 × k2 compartments. Guess: build it for two exits with a size cap (S–M), unless you'd rather the goal name an exception. *Built as guessed (2026-10-22)*: see [worklogs_b/st61_race.md](worklogs_b/st61_race.md). The two clocks race as k1 × k2 compartments (cap 400 per group, then a clear refusal); two gamma dwells give 0.6117 exact, ODE 0.6117, binomial 0.6107, agents 0.6104; STIsim's SEIS with both exits kept non-exponential is within 6–8% of agents (18% and 9.5% with the port's exponential swap).
6. **`verbose=-1` and `'brief'` at run time (CV9: 14 of Covasim's test failures).** The script rewrites them, and the October 20 "Won't do" rules out compatibility layers. Guess: script only, and the error names the script.
7. **Migration in the definition of done.** plan.md left it out; here its v4 side (G) is in, since it is most of the "no hacks" gap for three models. Say if it should wait.
8. **HP64: protection against severity applied when it changes, or once at infection (B3).** HPVsim fixes it at infection; v4 applies it as it changes during the stay. Applying it at entry closes about half the gap (prevalence 0.034 → 0.048, against HPVsim's 0.063). Left unchanged.
9. **F5 reads members' attributes at the start of each step**, as rates are, which moves the second genotype about 5% against the modifier spelling (a test tolerance went from 3% to 5%). The alternative is live reads without the speedup (4.40 → 3.64 s).
10. **A test over a family (HP66)** gives one result per person, from their most severe state.
11. **The new public names (D3)**: `ss.pars(defaults, **kwargs)` (named like `ss.states()`) rather than a public `define_pars()`; `ss.Vaccine.held` on the product class rather than `ss.held('tpt')`; FP86 as the default (a self-loop's dwell is at least a step, as agents) rather than `min_steps=`.
12. **FP80**: adding a callback to a module after it was given to a sim raises, naming the sim's copy, rather than forwarding it.
13. **HP55**: `cure=dict(...)` cures exactly the states named; people in other states count as `unneeded`, not `failed`.
14. **CV20**: a `priority=` list can mix conditions and scores, ranked item by item, rather than refusing scores.
15. **`ss.Deaths` no longer applies to fetuses** (FP75's second cause); a fetus dies with its mother or by the pregnancy's own losses.
16. **The binomial's default (C1)**: substeps so that rate × dt ≤ 0.1, rather than whole steps with a louder note; a stricter threshold would fix TC1 (+4.7% for tbsim weekly) at about 4× the cost.
17. **Substeps (C2)**: a compartment counts toward the growth rule only above 1e-9 of everyone, rather than a cap on substeps.
18. **Compiling (C3)**: decided by timing one plain step, rather than an estimate from the program's size.
19. **Traits of people grown by code without `age=` (FP91)** are drawn at the next declared step, so code reading them in the same step sees the default. The alternative: draw at `grow()` and have the migration script flag `grow()` without `age=`.
20. **Strain memory (C7)** is a dimension in compartmental runs only (agents keep per-person arrays); `last` over a set of strains remembers the newest; a trait under `auto` still runs agents.
21. **Stamps (A2)**: someone placed at the start reads `ti_X == 0` (keeps `ti_I > 0` as "infected during the run"); `self.ti - ti_X` equals the clock for a move at a step's start, and is one more for a contact infection or anyone placed at the start, so the clocks stay the way to write it; `ti_exit_X` and `set_delivery()` are one more than before (the step the exit is taken in). Cost: one int8 per agent per stamped state.
22. **A rate on counts with strains (`beta*I.frac`, `beta*I.n`)** is refused (`ss.errors.StrainsByCount`, naming `beta*I`), since new cases took their strain from the split rather than from who infected them (silently 50/50 where `beta*I` gives 11/89; [worklogs_b/frac_strains_by_count.md](worklogs_b/frac_strains_by_count.md)). The alternative is to make the rate form pass strains on, which `beta*I` already does.
23. **`sim.new(solver=...)` resets `substeps=`** to the new solver's default unless it is given too (an importer's whole steps for the binomial had become one 7-day RK4 step for the ODE; [worklogs_b/switching.md](worklogs_b/switching.md)). The alternative: keep it, with a `LargeStep` note.
24. **JAX or the kernels (`auto`)** is decided by timing one compiled kernel step, which errs toward the kernels (up to 10× low for models with time-varying values), rather than a size estimate that grew with the square of the places (MEmilio at 100 counties: 0.15 s, not 0.8 s).

## Stale

Out of date after this plan's changes, and left for the docs pass. Add to it rather than fixing each on the way.

- `docs/migration/v3_v4.md`'s timing text, notably line 50 ("a declared disease dates a move between states at the start of the step (`ti - 1`) ... one step long"); the skill's rows are corrected (A4).
- `oct21/strain_plan.md:125` and plan.md decision 3 on `'fittest'` (B4).
- tbsim's resistance workarounds, FPsim's six port workarounds, and STIsim's `ti_P == ti - 1` hand fixes, once A, B1 and D are built (in `stale_other_repos.md`).
- Docs and skills outside the repo that mention `transmit='fittest'` (none found in `docs/`).
- `design/12-backends.md:363`'s "With the textbook's fast rates ... `binomial` 23% high" (measured with whole steps), and any doc, tutorial or skill that says the chain binomial takes whole steps by default (C1).
- Spec 10 or 12, if they describe seeds as "dated a step before the start" in general: a seed read only by contact is now dated at the start (HP62).
- The migrate-starsim3 skill's FPsim and pregnancy rows: losses with `preg.add_callback(preg.stillbirth, fn)` rather than `engine.fresh()`, `preg.P` properties no longer refused by name, `ti_delivery` → `ti_exit_P`, `preg.breastfeeding` → `preg.lactation.L`, `conception_rate(sim, uids)` (D).
- Skills and docs that say a coded module's `n_<state>` isn't counted in a declared module, or that apply `np.abs` where a value is read (HPV's W3); `docs/` pages on `priority=` (lists of conditions only) (D1).
- `plan.md` E2 and its `.since` item: done here (D3).
- `docs/migration/v3_v4.md`: the new script rewrites and flags, the `make_run()` route for packages with their own run code, and the behavior rows (ages when none are given, reductions by scenario, sparse rates by year) (G, ST58).
- The user guide pages on `ti_X` and stamps (`modules_diseases.qmd`, `advanced_indexing.qmd`), and `skills/import-model/references/starsim3.md` if it mentions `ti - 1` for declared stamps (A2).
- Skills and docs that say memory immunity of strains is agents only, or that a strain is introduced with `dis.infect()` (C7).
- `oct20/simplify/support_matrix.md` and `oct21/stisim_port.md` (ST24: "competing exits agents only") are historical records, left as they are (ST61).
- Docs and skills that write a rate on counts (`beta*I.frac`) in a disease with strains (now refused).
