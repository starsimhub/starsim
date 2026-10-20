# Handoff: Starsim v4 review and ports (2026-10-19)

This folder is `starsim/v4_design/review/` on branch `v4-rs`; on the original machine it was `~/gf/v4_review/`. Absolute paths in these notes refer to that machine.

## Where everything lives

| Repo | Branch | State |
|---|---|---|
| starsim | `v4-rs` (off `v4`) | **Green**: 304 tests pass. 94 commits ahead of `v4`, covering bug fixes and proposal steps 1–8. |
| starsim | `v4-rs-events` | Done: synchronized event flows, compartmental fetal levels, lactation. Passes on its own. |
| starsim | `v4-rs-events-merge` | **WIP**: `v4-rs` + `v4-rs-events` merged, with 2 failing tests (below). The merge agent's partial fix is in the last commit and is untested. |
| stisim | `v4-port` | Bacterial STIs and care ported and tested; v4-rs workarounds dropped. The last commit is WIP: BV, GUD, connectors and syphilis, untested. |
| hpvsim | `v4-port` | One WIP commit with the port in progress, untested. |
| fpsim | `v4-port` | 4 commits; the port is in progress. The tree was clean at handoff. |

`v4-port` branches were created from `origin/main` and track it. Set a real upstream before pushing (`git push -u origin v4-port`).

On this machine they were worktrees: `../stisim-v4`, `../hpvsim-v4`, `../fpsim-v4`. On a new machine, plain checkouts of the branches are fine.

## Docs (this folder, `v4_review/`)

- **`REVIEW.md`**: the original review, covering what is ugly or missing in v4 for STIsim and HPVsim.
- **`*/NOTES.md`, `*/port.py`**: per-model scratch ports and notes, including bugs found in v3.
- **`proposals/00-README.md`**: index and reconciliation of proposals 01–04, the **8 open decisions for Robyn and Cliff**, and the build order.
- **`proposals/01`–`04`**: time and within-host state; programs and treatment; transition timing; pregnancy and births.
- **`PORT_PLAN.md`**: principles, order, and dependencies. It ends with 2 more open decisions.
- **`port_requests.md`**: engine requests logged by the ports. The STIsim requests 1–9 are done. **The 11 HPVsim requests are open** (below).
- **`repros/`**: minimal repros for the first blocking bugs, now fixed.
- **Implementation log:** `starsim/v4_design/admin/implementation.md` on `v4-rs`. It is the source of truth for what is built, what departs from the proposals, and the known issues.

## Pending decisions, implemented behind switches

- **Numbers are people:** `ss.results.people_now`. Set it to `False` to revert.
- **Routine delivery once per eligibility spell:** `ss.Delivery.spells`. Set it to `False` to get once per life.

## Next steps, in order

1. **Finish the events merge** (`v4-rs-events-merge`). Two tests fail after merging; each passes on its own branch:
   - `test_pregnancy.py::test_syphilis`: `mtct: 483 as an ODE, 1475 as agents`.
   - `test_pregnancy.py::test_hiv_vertical`: breastfeeding transmission is 0.

   Likely an interaction with changes on `v4-rs` since the fork (331b4013). The candidates:
   - `burnin=True` default on `ss.Pregnancy`, with fetal levels for women pregnant at the start;
   - the age-split default (from people's ages, uniform 0–60) together with the `breastfed` dimension and births into the youngest groups;
   - flows at index 0.

   Fix the code, then fast-forward `v4-rs` to the result.
2. **Fix the HPVsim engine requests** (`port_requests.md` § HPVsim). Most important:
   - #1, init time: `Dists.init` recurses through `dist.sim`, so a 2-genotype init takes 39 s.
   - #2, per-act beta in two directions is linearized. This is a correctness bug for every STI model with directional beta.
   - #3, `ss.outcome` of a branch, and `.select(to=)`.
   - #5, a dwell mean as an attribute expression.
   - #11, the E212 check on unused array slots.
3. **Resume the ports:**
   - **STIsim:** test and finish the WIP commit (BV, GUD, connectors, syphilis), then HIV/HIVsim, then the network's ODE form.
   - **HPVsim:** test and finish the WIP commit, which covers natural history, products and interventions, and `hpv.Sim`, with an ABM/ODE agreement test on 1–2 genotypes.
   - **FPsim:** FPmod on the declared `ss.Pregnancy`, contraception, interventions, results, the compartmental rewrite, and comparison with v3.

## Gotchas

- **Load:** this machine crashed running 5 agents at once, each running pytest with `-n auto`. Cap the parallel test workers (`-n 4`), or run fewer agents at a time.
- **Run port tests serially.** `pytest -n auto` with tests that use `ss.parallel` (stisim `tests/test_sim.py`) deadlocks once JAX has run (fork after JAX). The symptom is 0% CPU indefinitely.
- **Test times:** the starsim suite takes about 30 s with xdist. The STIsim bacterial tests take about 11 s serially.
- **Imports:** ports import starsim from the `v4-rs` checkout, which is installed editable in the conda env `starsim`. Run each port with `PYTHONPATH=<port checkout>` so the port's own package is imported, and check `pkg.__file__`.
- **Commits:** no Co-Authored-By or other trailers (Robyn's rule).
- **Behaviour changes on `v4-rs`** that shift existing results:
  - `ss.Pregnancy` burn-in is on by default;
  - an age stratification takes its split from people's ages;
  - `ss.constant` dwells become exact conveyors in compartmental runs;
  - a sibling transition with no `p=` and no driver is undriven, not the remainder;
  - branches summing below 1 mean "the rest stay".
- **Known issues** are listed in `implementation.md`. Among them: the chain-binomial solver misses entry-step hazards (−24% for BV at monthly steps); the joint-space maps are dense, so large crossings use a lot of memory; HPV genotypes crossed are 8⁴ states per group.

## v3 bugs found (independent of v4, no issues opened yet)

- **stisim:**
  - `gud_syph` is dead code.
  - `treat_BV` never clears `infected`, and calls a nonexistent `rel_sus_ptb`.
  - BV's p×RR can exceed 1, risk factors are redrawn every step, and newborns are flagged with BV.
  - `SyphTx` never clears `exposed`.
  - `set_congenital` makes infected fetuses immune for life, and `treat_fetus` double-counts.
  - `max_capacity` is never used.
  - `UpdateDrugs` reads a nonexistent attribute.
  - VMMC treats an annual flow as a stock target.
  - `sti.Sim(dt=)` routing.
- **fpsim:**
  - 5 methods never get their fitted durations (`'lnorm'` vs `'lognorm'`).
  - Calibrated fecundity bounds are never applied.
  - A user's contraception module drops the calibrated pars.
  - Everyone starts nulliparous.
  - `change_initiation` compares years exactly.

Details are in `bv_gud_connectors/NOTES.md`, `syphilis/NOTES.md`, `hiv/NOTES.md`, `stisim_bacterial/NOTES.md` and `fpsim/NOTES.md`.
