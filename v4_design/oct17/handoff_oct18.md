# Handoff, October 18

Where the work on [implementation_oct18.md](implementation_oct18.md) stands at the end of the session of October 18, and what waits on Cliff. Nothing since commit `1a5b7a77` is committed.

## State of the tree

- Full suite (`cd tests && pytest test_*.py -n 6`, with `STARSIM_LFD` set): 389 passed, 11 skipped.
- `pytest switching -n 6`: 41 passed, 1 skipped, 10 xfailed, 1 xpassed. `test_siminf_scale` is a build-time limit that fails under load; it passes alone. `test_epimodel_fit` now passes (ten runs a point, item 20), but its marker stays, since it depends on the seed.
- `starsim/library/diseases/coded.py` was renamed to `agents.py`, and `docs/coming_from/ghana_measles.qmd` was replaced by `numpy.qmd`: git shows the old files deleted and the new ones untracked, so they need a `git add` when committing.
- `CHANGELOG.md` (4.0.0) has a line for every change below. The spec docs and `admin/implementation.md` are synced to `lingua_franca_design`.
- Each agent's worklog, with details, numbers, and tests, is kept in the session's scratchpad, `wt/WORKLOG_<name>.md`.

## Done this session

Phases A, B, F, G, I, and K, and the inference items of D, are done except the parts that wait on a decision (see the checkboxes in the plan).

- **Phase A (PA), silent wrong answers.** Conditions on another disease's states; the starting state no longer depends on declaration order (`ss.Disease(first=...)`); `new()` and `ss.parallel()` keep subclasses; time-varying branch rates in compartmental runs; times on the step grid, cures, newborns by dimension, weighted results, and more.
- **Phase B (PB), migration.** `python -m starsim.migrate <path>` and the migrate-starsim3 skill; removed names raise with what to write instead. Downstream after migration: HPVsim 369/373, STIsim 188/198, FPsim 64/69, malariasim 284/284.
- **Inference (PD).** The contract tests and fixes in Phase D.
- **Phase F (PF), the saved form and random streams.** `to_code()` writes deliveries, contact matrices, transfers, `where=`, priors, and times correctly; three silent writer bugs found by a corpus round trip (`test_round_trip`); save after a run works again. Draws no longer depend on the order the states are declared in, and each run of `n_runs=`/`ss.parallel()` draws its outcomes from its own seed (same distributions, KS p > 0.18). `ss.Sim(pars=...)` routes each key by name. Still open: a stepped run, or one with several diseases, draws deliveries and outcomes differently from a whole run (a design change for deliveries; outcomes a step at a time cost 4.7x).
- **Phase G (PG), scale.** Three HPV genotypes as an ODE in 9.4 s, from 383 s, bit for bit (HP29). Sparse movement arrays (first run at 6,000 places 0.08 to 0.19 s, from about 0.4 s). Workers with one BLAS thread, started by forkserver once Numba's or JAX's threads exist. Exact Gillespie for strata at a flat cost per event (0.2 µs at 300 strata, from 5.1).
- **Phase I (PI), records and small bugs.** Records that contradicted the code; messages; delivery naming and coverage (a weekly screen at 0.4 reports 0.40, not 0.06); every error class raised by a test; `check_dt()`, in-place typed values, `describe(verbose=True)`; the coarse-step bias table in spec 12 and the user guide (an SIR's peak at a daily step +4% as `binomial`, +2% as `abm`; up to +32% at a weekly step).
- **One `ss.Disease` (PJ).** `ss.CodedDisease` is removed: an `ss.Disease` that declares nothing runs as code, as in Starsim 3, and `ss.Infection` subclasses `ss.Disease`. The library's Starsim 3 diseases are `ssl.AgentSIR`, `ssl.AgentSIS`, and `ssl.AgentSEIR` (they run only as agents). "Coded" is gone wherever it meant "written as code".
- **Phase K, Cliff's answers in [decisions_oct18.md](decisions_oct18.md).**
  - PK: entry into a state dated at the end of the step for diseases written as code too (12); `rate_units=None` for births, deaths, and pregnancy, per 1,000 above 1 and per person below, with a warning when ambiguous (14); Nelder-Mead for one or two parameters (19); ten runs a point for agent fits (20); extinction in `msim.summarize()` when any run died out (30); rates weighted by `people.scale` when the weights differ, unchanged when uniform (HP4).
  - PL: every intervention class downstream uses is still there (3); 18 internal helpers moved out of `ss.` (7); 11 `init_*` phases private, 4 kept public because downstream overrides them (8); `ssl.AgentSIS` is Starsim 3's SIS with waning (10); "Coming from NumPy" replaces the Ghana page (27); the write-up for item 13.
  - PM: 16-bit step numbers for agents' entry and exit times, 142 to 114 bytes per agent at the peak, speed unchanged; agents put straight into a state now leave it (21). Tau leaps when the exact run would take more than about a second (`ss.solvers.max_seconds`), independent places always exact (17).

## Waiting on Cliff

1. **Item 18, agents' exponential dwells.** Realizing `dur=` as a hazard drawn at the start of each step, as approved, moves agents away from the ODE: an SIR's peak at a daily step goes from +4.8% to −9.8% of the ODE's, and the final size from −0.6% to −4.1%. PK reverted it; the patch is in the scratchpad at `wt/scratch-pk/agents_with_item18.diff`. Options: (A) leave it as now, documented; (B, recommended) exit chance `r·dt/(1+r·dt)` for both `dur=` and `rate=`, which fits the ODE within about 1% (peak) and 0.5% (final size), but changes every agent model with `rate=` exits; (C) item 18 as approved. [CK: i'm ok with r·dt/(1+r·dt) if it's not substantially slower; otherwise, revert to the original +4.8 version and document it.]
2. **`ss.MaternalNet`.** Decision 1 says to keep it as an alias ("ok"), but D-D says no compatibility aliases. It is removed now (the error names `ss.PrenatalNet`, and the migration script renames it); it is used 70 times in 9 downstream repos. [CK: i said maybe MaternalNet could be a stand-in for when the user doesn't want to distinguish between PrenatalNet and PostnatalNet. i guess fine to leave in, we can decide what to do with it later.]
3. **Memory below 114 bytes per agent** (target about 50): 32-bit agent IDs, 16-bit death and removal times, `slot`, `parent`, and `scale` only when used, `rel_sus`/`rel_trans` only when something sets them, and no deep copy of a `People` in `sim.inputs` (about 20 bytes per agent). [CK: these all sound like a good idea.]
4. **D-A, the first part.** Does a transition with several destinations and a dwell decide at exit, from the dwell drawn? (Gates Phase C.) [CK: i don't strong preference, but i think so yes.]
5. **D-C.** Do all transitions out of the starting state count as acquisition, so `rel_sus`, efficacy, and `ss.treat` apply to infections driven by a bite rate? (Gates Phase E.) [CK: this should be the default, but there should be a clear and documented way to override it.]
6. **D-F.** One design document each for the care cascade, a family of diseases, and a behavior vocabulary, before code, the cascade first. (Gates Phase H.) [CK: ok, depends on those designs!]
7. **Item 13.** [people_compartmental.md](people_compartmental.md) compares five alternatives; it recommends keeping one row per group (it is one row only with no stratification), with `people.n` and a repr that says "groups". [CK: yes keep B, the current implementation]
8. **Smaller ones.** Whether to add a fixed-seed fuzz run to the repo's tests (the fuzzer is outside the repo); `test_sbc[pmcmc]` passes only at some seeds (moved from seed 1 to 3); where malariasim lives (`/home/cliffk/idm/malariasim` does not exist, and `/home/cliffk/idm/malaria` has no Starsim code). [CK: no don't add fuzz tests unless they're small and fast. the malaria code is in /home/cliffk/idm/ngm/starsim-ngm/malariasim]

## Also still open in the plan

- Phase C (after D-A), Phase E (after D-C), Phase H (after D-F).
- Phase D: the two coverage checks not re-run (two-parameter NUTS, PMCMC) and X8; mixing parameters fittable (DF34); the scale of an all-zero series (SW4, DF21).
- Phase G: the lean path for small declared models (N29); crossings whose places mix within a step; exact Gillespie when every place reaches every other (grows as about n^1.5).
- Phase I: the canon and naive-user reviews; L26 (`ss.death(p=)` gives a plain `TypeError`); DF28 and DF29 (wait for `describe()` and the kernels to settle).
- The multi-threaded-fork warning remains (OpenBLAS's idle pools); silencing it needs forkserver everywhere (DC23).
- The CI workflow for downstream suites (`.github/workflows/test_downstream_v4.yml`) is written, not pushed.

## How the work was run

Each round: a worktree per agent under the session scratchpad's `wt/`, built from `git diff --binary HEAD` of the main checkout plus its untracked files, with that baseline staged in the worktree's index; rules in `wt/PREAMBLE14.md`. To merge: copy `WORKLOG.md` out first, run `wt/merge.py wt/<name>` (it does not apply deletions; delete those by hand), clear `starsim/__pycache__/*.nb*` after kernel changes, run the suite and switching, add the worklog's CHANGELOG lines, tick the plan, then `git worktree remove --force`.
