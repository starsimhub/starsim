# Status of every known issue, and the next work plan

Written 2026-10-18, after the first implementation plan ([implementation.md](implementation.md), phases 1 to 9) was worked through. It checks every known issue against the current `v4` working tree (commit `1a5b7a77` plus the uncommitted work since), including the reviews added on 2026-10-18 (stats, fuzz, HPVsim, STIsim, FPsim, malaria), and proposes the next plan.

## How the status was checked

Seven agents each took one source, re-ran its repro scripts (or wrote a minimal one) against the current tree rather than trusting the CHANGELOG, and gave each finding a status: fixed, partly, open, needs Cliff, documented, not a v4 issue, or unclear. The model reviews also ran each package's own test suite against the current Starsim. Their tables, one row per finding with the evidence, are in [status_oct18/](status_oct18/):

| Source | Table | Rows | Fixed | Partly | Open | Needs Cliff | Other |
|---|---|---|---|---|---|---|---|
| The plan, `deferred.md`, known issues, departures, N findings, the ranked L/D/X/R/P findings, W items, next reviews, the switching demos | [reg.md](status_oct18/reg.md) | 366 | 143 | 41 | 80 | 52 | 41 documented, 4 unclear, 3 won't do, 2 not v4 |
| Statistical and scaling audit | [stats.md](status_oct18/stats.md) | 27 | 12 | 5 | 4 | 2 | 3 unclear, 1 not v4 |
| Invariance fuzzing (plus a fresh batch) | [fuzz.md](status_oct18/fuzz.md) | 29 | 6 | 4 | 12 | 2 | 4 not v4, 1 unclear |
| HPVsim (HP1 to HP36, and 3 new) | [hpv.md](status_oct18/hpv.md) | 39 | 4 | 2 | 28 | 5 | |
| STIsim (ST1 to ST27) | [sti.md](status_oct18/sti.md) | 27 | 2 | 3 | 22 | 0 | |
| FPsim (FP1 to FP39) | [fp.md](status_oct18/fp.md) | 39 | 1 | 4 | 29 | 5 | |
| malariasim (MA1 to MA33, and cross-references) | [mal.md](status_oct18/mal.md) | 50 | 1 | 4 | 36 | 3 | 5 not v4, 1 unclear |

The repro scripts the agents ran were in a session scratch folder and are not kept; each table names the review script it re-ran.

## In one page

**The first plan's work holds up.** Of the 74 wrong-answer findings in the first review's tiers 1 and 2, 65 are fixed, 7 partly, 1 open (L16, `pars=` routing), and 1 waits on a decision (X7). The two inference bugs the stats review traced (the particle filter's time running slow, and the JAX density's per-step tables) are fixed, and so are N1 to N3 on the seats' own scripts. The switching demos stand at 41 of 51 passing (1 skipped, 11 expected failures, of which 9 are open items and 2 wait on decisions).

**The four model reviews have barely been acted on, and they are now the bulk of the open work.** Of their 155 findings, 8 are fixed (by other work) and 115 are open. They were written against `03741ba2` or `1a5b7a77`, and almost every repro gives the same numbers today.

**Unchanged downstream packages mostly do not run on the current Starsim:**

| Package | Its tests passing, at review time | Now, unchanged | Now, with the shims named | What the failures are |
|---|---|---|---|---|
| HPVsim | 359 of 368 | 340 of 368 | 161 of 169 in the 12 failing files, after renames and two analyzer edits | 20 renames or removals (`Result.annualize` 12, `ss.routine_vx` 4, `msim.mean`/`median` 3, `ss.MaternalNet` 1), 5 the documented time convention, 3 not v4 |
| STIsim | 192 of 198 | 80 of 198 (26 not collected) | 171 after three renames; 186 after the time-index edit the migration skill gives | `ss.MaternalNet` (renaming it also changes the network's name, which STIsim reads), a coded `ss.Disease` subclass now refused, `verbose=1/12` now refused |
| FPsim | 6 of 62 | 21 of 62 | 55 of 62 after a one-line `verbose` shim | 37 of 40 failures are `verbose=1/12` alone |
| malariasim | 161 of 284 | 161 of 284 | 273 with a 27-line shim, 284 with the full one | 72 from `ss.Disease` being declared-only, 47 from `ss.SIS` being declared, 4 from analyzers running at the initial point |

`verbose` given as a fraction (`1/12`, print every 12 steps) breaks FPsim and STIsim and is in neither the CHANGELOG nor the migration skill; it should be accepted again.

**Silent wrong answers that are new, or worse than when they were found** (the first thing to fix):

| ID | What happens | Since |
|---|---|---|
| HPN1 | `where=A.Ca` on a transition in disease B uses B's own `Ca` whenever B has a state of that name, which every genotype in a family does: a cancel flow of 0 against 1,734 | regressed since `03741ba2` (2,495 then) |
| F1, F1b | The state everyone starts in is the source of the first contact-mediated transition, so declaring `breakthrough = V >> I` first starts everyone in `V`; and `to_code()` writes transitions in alphabetical order, so a round-tripped SIR with breakthrough starts everyone in `V` (peak infected 2,198 to 25), with no warning, in 17 of 60 fuzzed models | new since `03741ba2` (0 of 60 then) |
| ST10, FP7, HP12 | `sim.new()` and `ss.parallel()` on a subclassed sim (`sti.Sim`, `fp.Sim`, `hpv.Sim`, and `fit.sim`) return an empty `ss.Sim` with no error | was refused at `03741ba2` |
| MA23 | An `ss.SIS` subclass that overrides `set_prognoses()` runs with the override ignored | was a crash |
| ST9 | A declared disease refuses an unused keyword parameter, so it cannot carry the `eff_condom` that STIsim's network reads | new |
| HP29 | Three HPV genotypes as an ODE did not finish in 600 s (25.5 s at `03741ba2`); to confirm on a quiet machine | possibly regressed |

Several of these came with this session's changes (the anchor rule, L35's parameter merge, the declared `ss.SIS`), which is why the first phase below adds the four packages' suites as a gate.

**The largest open groups, across sources:**
1. **Agent timing and branching.** How agents realize dwell times, branch choices and their timing, `where=` with `p=`, and remainders (X7, L14, C17, C15, ST4, ST6, ST7, FP16, FP17, HP1, HP6, HP16, HP18, HP20, HP21, MA7). One decision (decide at entry or at exit; exponential `dur=` as a hazard or not) settles most of it.
2. **Coded modules and subclassed sims.** `new()` on subclasses, the step-boundary convention for code written to Starsim 3, modifiers reaching coded parameters, results for coded modules, and the removed bases (FP1 to FP8, FP23, ST8 to ST11, HP12, HP17, MA22 to MA26, D23).
3. **What counts as an infection, and vector-borne models.** Acquisition guessed from the rate's shape: `rel_sus` and a vaccine's efficacy do nothing for an EIR-driven infection, `ss.treat` reaches no one, and counts given starting values become the agents (MA1 to MA5, MA11, MA12); counts as agents have their own integrator (MA6 to MA9, MA28).
4. **Deliveries, products, and the care cascade.** Cures naming one transition (ST5, HP25), one product over several diseases (HP7), tests with two results (HP24), events and remembered results (ST12 to ST16, ST20), and deliveries in the kernels (DF1, PL2, PL3, PL7).
5. **Per-act transmission on sexual networks** (ST2, ST3, N7, HP3): a per-act probability has no spelling, so female-to-male transmission is 0.43 against Starsim 3's 0.73 at 8 acts.
6. **Inference robustness** (A2, R5, A4, A1, W13, SC3): two-parameter fits stop at a worse optimum and say they converged; NUTS chains start together; the JAX density's substep cap; the stochastic fit's scoring; and a one-line cause of the particle filter's quadratic cost with daily data.
7. **The saved form** (F1b, C06, C07, C09, R10, R15, R24, R29): `to_code()` writes dates, time-series keys, and transition order wrongly, and has no writer for deliveries or contact matrices.
8. **Random streams** (C12a, C12c, C19: X3's remaining parts): state order changes draws in `binomial` and `gillespie`; the whole-run and stepped paths draw differently; and `ss.parallel` outcomes differ from the matching single runs.
9. **Scale.** Dense place-by-place arrays at build time (SC1, SC4), exact Gillespie with interacting groups (SC6), per-call overheads in the agent engine (FP34, MA17, MA27), and memory per agent (N25: 125 to 150 bytes against 50).

## Decisions that gate the work

These need an answer before the phase that depends on them. All are also in [decisions_oct18.md](decisions_oct18.md), which this list extends.

| Decision | Gates | Recommendation |
|---|---|---|
| **D-A. Agent dwell semantics** (X7, HP1): does a transition with several destinations and a dwell decide at exit, from the dwell drawn; and is an exponential `dur=` realized as a hazard each step, like `rate=`? | Phase C | Decide at exit, documented; exponential `dur=` as a hazard (agent results change, toward the ODE) [CK: ok] |
| **D-B. The starting state** (F1): the source of the first contact-mediated transition, the state nothing flows into, or required when ambiguous? | Phase A | The state nothing flows into among those connected to the seeded states, with the anchor written explicitly in `to_spec()` and `to_code()`; a note when two candidates exist [CK: ok, and also a very clear way to write it explicitly] |
| **D-C. What counts as acquisition** (MA11, MA12): transitions out of the anchor state are acquisition, whatever their rate's shape (so `rel_sus`, efficacy, and `ss.treat` apply to EIR-driven infections) | Phase E | Yes [CK: ok, but give a way to override] |
| **D-D. Compatibility for Starsim 3 packages** (answered by Cliff, 2026-10-18): packages need not run unchanged; porting should make their code simpler, and the migration must be clear, with a script, a skill, or both. So no compatibility aliases; removed names raise errors that say what to write, and a migration script rewrites the mechanical changes | Phase B | (decided) [CK: can have SOME compatibility aliases, but need very good rationales] |
| **D-E. The tau-leap switch** (N26, CROSS, stats): 1e6 events everywhere today; exact runs of independent places are faster than leaping | Phase G | No switch for independent places; for interacting groups, base it on the cost per event after SC6 [CK: ok] |
| **D-F. Design passes**: the care cascade (events, remembered results, `on=` over several events), a family of diseases (`ss.Diseases`: cross-immunity, union results, joint seeding), and a behavior vocabulary (counters, time since an event, declared regressions, tables) | Phase H | One design document each before code; the cascade first (STIsim and HPVsim both need it) [CK: ok] |
| **D-G. Stochastic fit scoring** (W13) | Phase D | Mean of the runs' errors, ten runs a point for agent fits [CK: NO, duplicating points is a waste of runs -- with eg optuna you don't need this and you can run 10x faster] |

## Proposed work plan

The phases are ordered by harm: silent wrong answers first, then what makes unchanged packages run, then the larger semantic and design work. Each phase lists its items, their IDs, and a size (S under a day of agent work, M a few days, L more). Phases A and B can run in parallel; C to G can run in parallel once A and B are merged, each in its own worktree as before.

### Phase A: silent wrong answers and regressions (S each; first)

- [x] **HPN1.** A condition naming another disease's state uses that disease's state, as rates already do (the same fix as the cross-disease count fixed this round, in `conditions.py:299-303`); test with two diseases with same-named states.
- [x] **F1, F1b.** Apply D-B; write the anchor explicitly in the spec and in `to_code()`, and keep the declaration order of transitions in `to_code()`; a round-trip test over the corpus and the fuzzer's models.
- [x] **ST10, FP7, HP12.** `Sim.new()` rebuilds `type(self)` with the recorded constructor call, and raises when a subclass has no inputs to rebuild from (FPsim's 16-line prototype, `v4port/framework/newfix.py`).
- [x] **MA23.** An override of a coded method on a declared class (`set_prognoses()` on an `ss.SIS` subclass) raises, naming the coded base to subclass instead.
- [x] **ST9.** A declared disease accepts a keyword parameter it does not use when a network or another module reads it, or says which module should own it.
- [~] **HP29.** (Partly, by PA: a quadratic lookup in `Crossing.build()` removed; three genotypes now run in 358 s against 22 s at `03741ba2`; the rest is the crossing's exposure terms, Phase G.) Re-time three genotypes as an ODE on a quiet machine; if it regressed, find the cause (likely the crossing, now that same-named states are kept distinct).
- [x] **WA1.** A split branch whose rate is a fitted parameter times a time-varying value records its flow (it records 0 while 2,500 people move).
- [x] **C02 remainder.** A `sim.pars` value set after construction survives `new()` and `ss.parallel` (it is lost: a peak of 2,600 against 25,293), or `new()` raises.
- [x] **C11, C11c.** Times compared against the step grid with a tolerance (`ss.months(7)` at `dt=1/12` is delivered on step 8), and days added to dates as days (`2000-01-01` plus 200 days is a day late).
- [x] **ST5, HP25.** A cure never guesses one transition when several infected states exist: it cures from every infected state, or raises asking for `to=`; `describe()` names what it does.
- [x] **MA10.** A value by level of a dimension the state does not have raises rather than taking level 0.
- [x] **MA18.** Newborns from `ss.Births` are placed at each dimension's split, not all in the first level.
- [x] **HP4.** Results of a weighted agent population are weighted by `people.scale` (`n_alive` is 10,000 against 5,050).
- [x] **ST6, ST7.** `where=` with `p=` fires (it never does), and a function `p=` gets the remainder at run time (0.445 instead of 0.9).
- [x] **SC3 residual.** One line, `calibration.py:1384`, makes the particle filter linear in data length, with the same log-likelihood.
- [x] **A2, R5.** A fit restarted from the best two or three scan points, with disagreement reported as `converged=False` (today it stops at 58.86 against 54.25 and says it converged).
- [x] **A4.** NUTS chains start from dispersed points (or an optimum, as the spec says), not all from `x0`.

Done by PA. For Cliff: F1 applies D-B among the contact-mediated sources (taken literally it started everyone in a vaccinated or maternal-immunity state), and `ss.Disease(first=...)` names it explicitly; ST9 was done by the fallback (the error points to `define_pars()`), since passing plain values to coded modules changes what `mod.pars` holds; HP4 weights results, but rates (`I.n`) still count agents. [CK, Oct 18: F1 as applied is right (not everyone in the vaccinated state); the ST9 fallback is fine; HP4 rates: agreed to weight the rates (`I.n`) when the weights differ, and count agents when they are uniform (to do); `ss.CodedDisease`: removed, `ss.Disease` runs as code when it declares nothing, and `ssl.CodedSIR`/`SIS`/`SEIR` became `ssl.AgentSIR`/`AgentSIS`/`AgentSEIR` (done by PJ).]

### Phase B: a clear migration that simplifies downstream code (S to M; with A)

- [x] **A migration script** (`python -m starsim.migrate <path>`, consistent with the migrate-starsim3 skill, which points to it for the mechanical part): renames, keyword changes, the network's name with `PrenatalNet`, `verbose` fractions, `ti` to `ti - 1` in `update_results()` where it is safe; it flags what needs a person or an assistant with the skill. No compatibility aliases.
- [x] **Measured on HPVsim, STIsim, FPsim, and malariasim** (copies): the script, then the skill's remaining steps, then their suites; the migration diff should leave each package's code simpler (lines removed against added). Where a port makes a package more complex, that is a v4 problem.
- [x] **The step-boundary convention for code written to Starsim 3** (FP1, FP2, FP5, FP6, X13, N10): a warning at the end of a run for a coded flow that is non-zero at index 0 or zero at the last point; refusing `at=stop`; migration rows for `finish_step()` and year-end stocks.
- [~] **Analyzers at the initial point** (MA24), and `ss.Deaths` with `people_results=False` (MA32). (MA32 done; MA24 for Cliff: analyzers only after each step fixes 7 downstream tests but breaks `ss.dynamics_by_age` and 2 Starsim tests, so the script flags each analyzer instead.)
- [x] **`new_infections` counted where coded overrides cannot skip it**, or a warning when a coded disease's infections are all zero while its states change.
- [~] **A downstream gate.** (Written, not pushed: `.github/workflows/test_downstream_v4.yml`.) The four packages' suites in CI on the `v4` branch, after their migration.

Done by PB: after the script and the skill, HPVsim 369 of 373, STIsim 188 of 198, FPsim 64 of 69, malariasim 284 of 284 (failures left are the packages' own); migration diffs net −8, 0, +1, +6 lines. For Cliff: the name `ss.CodedDisease` (Starsim 3's bare coded `ss.Disease`, which `ss.Infection` now subclasses), and MA24.

### Phase C: agent timing and branching (M; after D-A)

- [~] One pass in the agent engine: exits decided as D-A says; dwell times rescaled when a duration parameter or a time-varying rate changes (C17: 825, 4,323, and 4,755 against the ODE's 498); transmission entry dated as progression entry (ST4); branch choice evaluated relative to the scheduled exit (FP16, FP17); the flow recorded per destination of one transition (HP20, which `deferred.md` set aside on an assumption that does not hold for HPV); an efficacy keyed on one destination acting on its probability (HP6); `ss.given_dwell` (HP18).
- [~] Contract tests: a dwell entered by transmission equals one entered by progression; agents agree with the ODE at a fine step for every dwell form; branch shares agree across solvers (FP13: 0.20 against 0.79 today).
- [ ] MA7's agents part: a fixed incubation period as agents is exact, not exponential (counts as agents through the compartmental kernels, Phase E).

Done by PC (round 15): D-A documented (one transition with several destinations decided at exit); C17, FP16, FP17, FP13, HP6 (a destination as a target, `'tr.C'`); `test_dwell_forms` (eight dwell forms, agents against the ODE). For Cliff: ST4 (the extra step is in counting, not in behavior; the proposed fix puts a fixed-dwell SIR's peak 12% low) and item 18 (option B fits contact-mediated SIRs within 1% but puts rates on counts and count states 7% to 15% off; B′ would fix both, about M). Deferred: HP20 (results with a destination axis), HP18 (`ss.given_dwell`, design), MA7's agents part (with Phase E). Details: the PC worklog in the handoff_oct18b scratchpad.

### Phase D: inference (S to M)

- [x] **A1.** The JAX density's substep cap set from the fastest rate over the bounds, or the density rebuilt when a draw hits it (37 log-likelihood units off today).
- [x] **W13** (after D-G), and the scale of an all-zero series (SW4, DF21).
- [x] The two coverage checks the stats agent could not re-run (two-parameter NUTS, PMCMC), and X8.
- [x] `ss.perday(...) * ss.timeseries(...)` accepted in either order.
- [x] **Mixing parameters fittable** (DF34, `away=`): rebuild mixing when a fitted parameter changes it.

Done by PD2 (round 15): W13 per Cliff (one run a point for agent fits, four for stochastic compartmental); DF21 (an all-zero series scaled by the model's mean); DF34 (`away=`, `cap=`, gravity parameters fittable; a fit recovers `away=0.02` under ode, binomial, agents); X8 (no change: one kernel build covers every solver, 56 to 60 s cold); PMCMC coverage 0.950 ± 0.022; two-parameter NUTS 0.907/0.917 ± 0.011 after fixing chains trapped in a worse mode (1.75% of fits), the rest unexplained (exact grid posteriors under-cover b by about 2.5 SE too); L26's message.

### Phase E: vector-borne models and counts (M; after D-C)

- [x] Acquisition as D-C says, fixing `rel_sus`, scalar efficacy, `ss.treat`'s default, which states are people, and the R0 matrix together (MA1 to MA5), with a conformance test that the spec's bednet idiom changes the spec's Ross-Macdonald model.
- [~] Counts as agents through the compartmental kernels (substeps, conveyors, modifiers), so mosquitoes do not go extinct at a monthly step (MA6, MA8, MA9, MA28).
- [ ] MA7's ODE offset (the conveyor gives 5,279 against the exact 5,020).
- [x] Typed values converted once at `init()`, series tabulated per step, function parameters evaluated once per step (MA17, MA27, FP31, FP34: `Engine.call()` calls `inspect.signature` on every call).

Done by PE (round 15): acquisition by D-C (MA1 to MA5; `test_contracts.py::test_acquisition`; the override is `acquisition=True/False` on a transition), MA17/MA27/FP31/FP34 (typed values once, functions once a step), and MA6, MA8, MA9 in the agents' count step. For Cliff: the carve-out (a transition out of the first state alongside a contact-mediated one is not an acquisition, so an SIRV's vaccination stays a vaccination; taken literally the rule counts `V` as infected, peak 0.401 to 0.510). Deferred: MA7 (the conveyor's last slot is exempt from other exits: a kernel change that moves every ODE with a fixed delay and a competing exit), MA28 (counts through the compartmental kernels as agents), `prevalence`/`new_infections` for vector models.

### Phase F: the saved form and random streams (M)

- [x] One pass over `to_code()`'s writers (C06, C07, C09a, C09b, R10, R24, R29): times as `ss.days(...)` or numbers, a writer for deliveries and `ss.contacts`, the declaration order (with F1b), and priors in the spec; one round-trip test over the corpus and the fuzzer (build, `to_code`, exec, compare hash and results, save and load), plus the 1% size check in `save()` (C16). (Done by PF, plus three silent writer bugs found by a corpus round trip (transfers missing from the spec, unitless durations, `start` in years); `test_round_trip`; save after run fixed (C16). R24 had no repro.)
- [x] X3's remaining parts (C12a, C12c, C19): one stream per transition in the compartmental engines; the same draws on the whole-run and stepped paths; each run's outcome draws from its own generator in `ss.parallel`. (Done by PF: flows sorted by name at build, stepped and whole runs take the same rate values, each run's outcomes from its own seed. Deferred: deliveries taken in Python when stepped or with several diseases, and outcomes drawn after a whole run, still draw differently from a stepped run; both need design changes.)
- [x] **L16, R14.** `ss.Sim(pars=...)`: model parameters by name, sim settings only by their own names, unknown keys raise. (Done by PF; a model parameter wins over a sim setting of the same name.)

### Phase G: scale (M to L)

- [x] Sparse place-by-place arrays from `Mobility` onward (SC1) and one BLAS thread in workers (SC4), with W17 (workers by `forkserver` once JAX or Numba threads exist). (Done by PG. The multi-threaded-fork warning remains, from OpenBLAS's idle pools; silencing it needs forkserver everywhere, DC23.)
- [x] A propensity tree or dependency graph for exact Gillespie with interacting groups (SC6), then the tau-leap rule from D-E. (Done by PG for the trees: flat cost per event, 0.2 µs at 300 strata, 4 µs at 300 places; the tau-leap rule waits on D-E.)
- [~] Memory per agent (N25), after decisions item 21; the lean path for small declared models (N29). (Memory done by PM, round 15: 117 to 81 bytes per agent at the peak, results identical, speed unchanged; then 81 to 48 by PMEM (round 15, Cliff's item 9: int16 group arrays, no stored `people.uid`, entry times only when read); previously for Cliff: group arrays as int16 and shared, about 19 to 5 bytes; dropping `people.uid`, 4.8; entry times only when read, 2.4 each. N29 open.)
- [ ] Crossings whose places mix within a step (CROSS's deferred item), and the ODE's repeated sums of exposures in a crossing. (The ODE's repeated sums are done by PG with HP29: three HPV genotypes in 9 s, from 383 s, bit for bit.)

### Phase H: design passes (L; after D-F)

Designs written (round 15), in [designs/](designs/README.md), with the decisions for Cliff and where the four documents disagree; code waits on his answers.

- [ ] **The care cascade** (ST12 to ST16, ST20, HP24, HP26, D27, D29): events (`test.positive`, `tx.failed`), remembered results per person, `on=` over several events, stock targets with a priority, and stratified results; deliveries in the kernels (DF1, PL2, PL3, PL7) as part of it.
- [ ] **A family of diseases** (HP22, HP23, HP15, HP29): cross-immunity with a combination rule, union results, joint seeding, and the size of the crossing.
- [ ] **Per-act transmission on sexual networks** (ST2, ST3, N7, HP3): a typed per-act probability applied per act in each direction, and named directions for a `beta` pair; with ST17, ST21 to ST23 (route weights, a condom attribute, a two-mode network, a contact matrix over several dimensions).
- [ ] **Behavior vocabulary** (FP24 to FP28, FP30, HP19): counters, time since an event, declared regressions, tables and row-valued `p=`. Until then, pregnancy and contraceptive choice stay code, which is fine.
- [ ] **Dimensions over people's existing properties** (FP19, FP20, FP39, D15), and units of bare numbers in distributions (FP21, FP22, FP35).

### Phase I: records and docs (S; any time)

- [x] Records that contradict the code: decisions item 13 (`sim.people` in a compartmental run is a one-row `People`, not `None`); departure 3 (`ss.load()` of a spec was removed); known issue 13 (fixed); the xfail reasons of `test_odin_agents` and `test_ghana_measles_fit`; the plan's `sde` and X5 rows; CHANGELOG lines on `ss.Vx`/`ss.Dx`/`ss.Tx` and on mixed units, which contradict each other; the import-model idioms page's `ss.perday(1e9)` for "as soon as", which hangs an ODE (ST26 wants a move instead).
- [x] The register groups from [reg.md](status_oct18/reg.md) that are small: messages and printing (X18, L38, L26), delivery reporting (D27, D29, D30, D31), error trigger tests and golden `describe()` output (P6, DF27 to DF29), and the standalone bugs X15, KI3, KI8, DF11. (Done by PI except: L26 `ss.death(p=)` still a plain `TypeError`; D31 waits on decisions 2 and 3; DF28 and DF29 wait for `describe()` and the kernels to settle.)
- [x] X4: state the coarse-step bias sizes in the docs, after re-running the solver comparison on the current solvers. (Done: table in spec 12, paragraph in the user guide.)
- [ ] The two planned reviews not yet done ([next-reviews.md](next-reviews.md)): the canon (textbook models against their known answers) and a naive user.

### Phase K: Cliff's answers in decisions_oct18.md (S to M)

The `[CK:]` answers in [decisions_oct18.md](decisions_oct18.md), by item number there.

- [x] Behavior (PK): 12, a declared disease and one written as code both date entry into a state at the end of the step; 14, `ss.Births`/`Deaths`/`Pregnancy` with `unit=None` by default (per 1,000 above 1, per person below 1, a warning when ambiguous and `unit` not given), documented; 18, an exponential `dur=` as agents realized as a hazard each step (D-A's second part); 19, Nelder-Mead up to two parameters; 20 (W13, D-G), ten runs a point for agent fits; 30, `msim.summarize()` reports extinction when at least one run died out; HP4's rates weighted by `people.scale` when the weights differ. (Done by PK except 18, reverted for Cliff: sampling an exponential `dur=` as a hazard drawn at the start of a step moves agents away from the ODE (SIR peak at a daily step from +4.8% to -9.8%); the exit chance `r·dt/(1+r·dt)` for both `dur=` and `rate=` fits the ODE within about 1%, but changes every agent model with `rate=` exits. Patch at `wt/scratch-pk/agents_with_item18.diff`. `test_epimodel_fit` now passes with ten runs a point.)
- [x] Namespace and downstream (PL): 3, keep every intervention class a current downstream package uses; 7, move the second tier (`code_names`) out of `ss.`; 8, make the `init_*` phases private unless downstream models override them; 10, check that `ssl.AgentSIS` is Starsim 3's SIS with waning (if so, item 10 is done); 27, the Ghana "Coming from" page becomes "Coming from NumPy"; 13, a short write-up of the alternatives to a one-row `People` in a compartmental run, for Cliff. (Done by PL: 3 and 10 needed no change; 7 moved 18 helpers; 8 kept 4 phases that downstream overrides, made 11 private; 13 is [people_compartmental.md](people_compartmental.md), for Cliff.)
- [x] Memory and the tau-leap rule (PM): 21 (N25), 16-bit timers and `ti_X` as step numbers with fast, obvious conversions; 17 (D-E), the recommendation: no switch for independent places, and for interacting groups a rule from the cost per event after SC6. (Done by PM: 142 to 114 bytes per agent at the peak, speed unchanged; the rest of the way to 50 needs People-level changes, for Cliff. Tau leaps above an estimated 1 s of exact run (`ss.solvers.max_seconds`); independent places always exact. Exact Gillespie with every place reaching every other still grows as about n^1.5.)

## Acceptance for this plan

- No silent wrong answer open in [status_oct18/](status_oct18/) after Phases A and C.
- HPVsim, STIsim, FPsim, and malariasim migrated by the script and the skill, each with simpler code than before and its suite passing, in CI on `v4`.
- The switching demos at their current 41 of 51 or better; the full suite and the corpus hashes unchanged except where a phase changes results on purpose, which its CHANGELOG line says.
- This file's tables re-run at the end by the same seven checks.
