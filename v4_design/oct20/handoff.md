# Handoff, October 20 (post-merge plan)

Work from [implementation.md](implementation.md), in the `ssv4` working tree on `v4-ck-oct20`, one session, no subagents (Cliff's weekly quota was at 95%). Nothing is committed. Kept up to date as items finish; each item's details are beside it in the plan.

## Done

- **The plan**: [implementation.md](implementation.md), from the merge's loose ends ([merge_tradeoffs2.md](merge_tradeoffs2.md)), what is left of Robyn's Phase B ([review/V4_PORT_PLAN.md](../review/V4_PORT_PLAN.md)), and the open items of [oct19/implementation.md](../oct19/implementation.md). Cliff answered its two questions: Stage B continues here, and tests with several results get `positive`/`negative` (B9).
- **Stage A, A1 to A11.** Code: `starsim/interventions.py` (`holding()`, `serve()`, `take_limited()`, `cure_crossed()`, `force_part()`, `shift()`), `starsim/migrate.py`. Tests: `test_programs.py` (`test_target_limits`, `test_people`), `test_treatment.py::test_syndromic`, `test_products.py` (`test_hpv_vaccines`, `test_hpv_screen_treat`), `test_other.py`. Also `skills/migrate-starsim3/SKILL.md` and `v4_design/oct17/designs/care_cascade.md:350`.
- **B2's rest**: credits of no-wait chains drawn nested (`crd_parent`; `kernels.py`: `credit()`, `credit_one()`, the workspace's 10th array; `compartmental.py`; `crossing.py`). `test_transitions.py::test_cascade_binomial` (no longer xfail).
- **B9**: `positive`/`negative` on tests with several results (`products.py`, `interventions.py: report()`), spec 08 (synced to LFD), `test_products.py::test_outcomes`.
- **B1, first part**: `X.time` in an exit of `X` as given dwell; given dwell's slot timing fixed for the ODE, Gillespie and binomial (`compartmental.py: slot_times()`); a clear refusal (and `auto` to agents) for a state followed by its time in a crossing (`choose.py: crossing_groups()`); `stages=` checked (`transitions.py`). `tests/test_conveyor.py` rewritten to v4's spelling (runs; the set clock is a strict xfail); xfail reasons in `test_pregnancy.py` and `test_triggers.py` updated.
- **B4** (second session of the day, after Cliff's "keep going"): births driven by a flow as compartments, so the declared pregnancy runs as compartments; and the rest of B1's crossing case. Details beside B4 in the plan. Code: `compartmental.py` (`add_per_births()`, `birth_map()`, `births_of()`, the layout of a state with immediate branches and a timed remainder, `dwell_slots()`), `engine.py` (`inflow_split()`), `choose.py` (`init_crossings()`), `expressions.py` (`Since` refused as a clock), `pregnancy.py` (`ss.ppl`).
- **B5's rest**: burn-in as compartments (`backfill_spreads()`), and a unit bug in `ss.choice` durations (`transitions.py: numeric_pars()`) that affected agents too. `test_burnin_ode` passes.
- **B6, first part** (after Cliff's "start on b6 and b7"): drawn attributes with `bins=` as levels; details beside B6 in the plan. `test_hpv_nab_ode` passes. Tests: 595 passed, 11 skipped, 6 xfailed.
- **B7, lactation**: details beside B7 in the plan. Tests: 596 passed, 11 skipped, 5 xfailed.
- **B7, fetal levels**: details beside B7 in the plan. Code: `pregnancy.py` (`fetal_specs()`, `route()`, `add_fetal_dims()`, `preg_event()`, `unmother()`), `compartmental.py` (`fetal_births()`, crediting in `record_flows()`), `engine.py` (`mother()`), `mixing.py`, `choose.py`. Tests: 598 passed, 11 skipped, 3 xfailed.
- **B7, transmission in utero over `MaternalNet`** as a fetal transition; `BreastfeedingNet` tried and reverted (see B7 in the plan). Tests at the end of the session: 598 passed, 11 skipped, 3 xfailed; `tests/switching -n 3`: 44 passed, 1 skipped, 8 xfailed (one run had a failure that did not recur, when the machine was loaded).
- `CHANGELOG.md`: lines for each of the above.

Tests: before any change, 580 passed, 11 skipped, 18 xfailed. After A1 to A9: 581 passed. After B2: 584 passed, 17 xfailed. After B1's first part: 588 passed, 11 skipped, 13 xfailed. After B4: 593 passed, 11 skipped, 8 xfailed. After B5 (latest): 594 passed, 11 skipped, 7 xfailed; `tests/switching -n 3`: 44 passed, 1 skipped, 8 xfailed (as on October 19).

## Left for Cliff

- A6's reading (a product given once with several diseases is bivalent for every delivery of it).
- A12: Robyn's `admin/implementation.md` and `V4_PORT_PLAN.md` still list `priority=`, `queue=` and doses as waiting on a design; not edited, since they are her files.
- B1's slot timing is a convention change in given dwell (decided 2026-10-19 as "rates keep the half step"); the numbers are in the plan.

## Next

The three strict xfails left, each a design question as much as code: `BreastfeedingNet` as compartments (mixing by a disease's dimension in a crossing), the HIV CD4 attributes (`test_hiv_cd4_ode`: derived, recorded, `progress`), and the set clock (`test_syphilis_clock`). Then Stage C. The session was large; the fetal-level and synchronized-flow changes deserve a review before they are built on.
