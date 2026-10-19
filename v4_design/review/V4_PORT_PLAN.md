# Porting v4-rs onto v4: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the 94 feature commits of `v4-rs` onto Cliff's current `v4`, so that there is one Starsim 4 line and the ports (STIsim, HPVsim, FPsim) build on it.

**Architecture:** Branch `v4-rs-port` off `origin/v4` (worktree `~/starsim-port`). Phase A replays the `v4-rs` commits in order, ported to v4's layout and names; a feature that needs compartmental support v4 does not have raises `E804` as compartments, and its compartmental tests are `xfail` with a reason naming the Phase B task. Phase B rebuilds those compartmental features on v4's kernel-only stepping, one task each, removing the xfails.

**Tech stack:** Python 3.11+, NumPy, Numba (`starsim/kernels.py`), JAX (`starsim/gradients.py`), pytest.

**Spec:** [HANDOFF.md](HANDOFF.md), [PORT_PLAN.md](PORT_PLAN.md), [proposals/](proposals/00-README.md), and the implementation log `v4_design/admin/implementation.md` as it is on `v4-rs` (`git show v4-rs:v4_design/admin/implementation.md`), which says what each commit built and how it is tested. v4's own rules are in `v4_design/oct17/implementation.md` ("How to work through it").

## Global constraints

- **Push and PR often.** Push `v4-rs-port` after every task. Open a draft PR against `v4` after Task 1, and update it after each task. Before each task, `git fetch origin` and check `git log HEAD..origin/v4`: if v4 has moved, merge it in first (a task of its own if it conflicts). The branches must never again drift for days.
- **Commits:** no Co-Authored-By or other trailers (Robyn's rule), so no `cherry-pick -x`. A replayed commit keeps its author and message; the mapping old → new SHA goes in the log at the end of this file.
- **Green after every commit:** the tests the commit touches pass. After every task, the whole suite passes, with only the xfails this plan adds (each naming its Phase B task).
- **v4's rules apply:** a test with every fix; no fix slows the kernels (run `tests/perf_guard.yaml` and the benchmarks before and after each Phase B task); changes to what a solver means go to Cliff with the numbers; each breaking change gets a line in `CHANGELOG.md`.
- **v4's names win.** Use v4's names in ported code and tests: `ss.PrenatalNet` (not `MaternalNet`; beta key `prenatalnet`), a delivery's `where=` (not `eligible=`), `at=` (not `years=`/`dates=`), `solver='gillespie'` (not `'tau'`), internals through submodules (`ss.solvers.Compartmental`, `ss.conditions.Cond`, `ss.transitions.Expr`). The full list: `v4_design/oct17/removed_names.md`. `python -m starsim.migrate` does the mechanical renames in tests.
- **Where Cliff fixed the same bug,** keep v4's fix and drop ours, unless our test still fails on v4; say so in the log.
- **Pending decisions stay switches:** `ss.results.people_now` and `ss.Delivery.spells` (HANDOFF.md).
- **Running tests:** `cd ~/starsim-port/tests && PYTHONPATH=~/starsim-port python -m pytest ...`, with `-n 4` at most if xdist is installed (it is not, in either conda env); never `-n auto` (the machine has crashed under load). Check `ss.__file__` points into `~/starsim-port`.
- **Known environmental failure:** `test_calibration.py::test_onepar_custom` fails on this machine with `sqlite3 database is locked`, with or without our changes. Not a regression.
- **Style:** the Starsim style guide; comments describe the software, not this port (no "ported from v4-rs").

## Where the old code lives in v4

| Old (`v4-rs`) | v4 |
|---|---|
| `solvers.py`: `Engine`, `StockCtx`, `CountState`, `convert` | `solvers/engine.py` |
| `solvers.py`: `Compartmental` | `solvers/compartmental.py`; stepping is `ss.kernels.one_step` over the compiled program (no `step_ode`, `derivs`, `shift_conveyors`, `step_per`) |
| `solvers.py`: `Agents` | `solvers/agents.py` (intact) |
| `solvers.py`: `Crossing` | `solvers/crossing.py`; `Crossing.step()` steps every engine in it together (no `claim`/`refresh`/`changed`) |
| `solvers.py`: `Mixing`; `make_engine` | `solvers/mixing.py`; `solvers/choose.py` |
| `transitions.py`: `Expr`, `par`, `evaluate`, ... | `expressions.py` |
| `transitions.py`: `Graph` (`describe`, `diagram`, `equations` → `presentation.py`) | `graph.py` |
| `transitions.py`: `by`, `timeseries`, `seasonal`, `TimeValue` | `parameters.py` |
| `transitions.py`: `Strat`, `Transfer`, `transfer` | `dimensions.py` |
| `transitions.py`: `CodedError`, `note` | `errors.py` |
| `transitions.py`: `geo`, `gravity`, `Mobility` | `networks.py` |
| `time.py`: `date`, `DateArray`, `datedur` | `dates.py` |

Anything else: `python /tmp/port/map.py <oldfile>` (an AST map of the fork's definitions to v4's files; recreate it from this table if `/tmp` was cleared).

## How to replay one commit (every Phase A step)

1. `git cherry-pick <sha>`.
2. For each conflict:
   - **`solvers.py` deleted by v4:** `git show <sha> -- starsim/solvers.py` and apply each hunk by hand to the v4 file that now holds its class (table above). `git rm starsim/solvers.py` once done.
   - **Content conflict:** keep v4's structure and names, and add the commit's behavior.
   - **A compartmental part with no v4 equivalent** (a Python stepping path that v4 removed): don't rebuild it here. Raise `ss.error('E804', ...)` where the compartmental engine would need it, with the same message form as v4's other `E804`s, and mark the commit's compartmental tests `@pytest.mark.xfail(reason='Phase B, task Bn: <feature> as compartments', strict=True)`.
3. Rename to v4's names (`python -m starsim.migrate tests/<file>` for tests).
4. Run the tests the commit touches. Pass, or xfail as above.
5. `git cherry-pick --continue` (or `git commit -C <sha>` after `-n`), keeping the original message.
6. Add a row to the log: old SHA, new SHA, and anything dropped, changed, or deferred.

## Phase A: replay

Each task is a run of consecutive commits (in `v4-rs` order, since later ones build on earlier ones). It ends with the whole suite green (bar this plan's xfails), a push, and the PR updated. "(B)" marks a commit with a compartmental part that likely goes to Phase B.

### Task 1: early fixes (commits 1–13)

`6e10bfe1 7b9a8397 70f4bee8 a32a1dda 8b07bc44 882a2306 b685c5b4 3d447991 064babf8 7c7ab0ed 9570bc05 d15a4780 b4f6086e`

Births placed in declared modules; per-act beta; the sex dimension; owning modules of arrays; capacity in rounds; branch remainder; ratio NaN; results from class bodies; a set of states as a source; lists of follow-ups; typed probability as coverage; a bare start year; `(sim)` against `(sim, uids)`. Several look like bugs Cliff's phase 3 may have fixed (check `v4_design/oct17/implementation.md` and `admin/implementation.md` on v4 first).

- [ ] Replay each commit.
- [ ] Full suite green; push; open the draft PR against `v4` ("Port v4-rs onto v4"), whose description links this plan.

### Task 2: undriven transitions, treatment, events, and clocks (14–24)

`2f61f27a f702a332 e58451c8 58641964 eec567c2 0546e094 e1fad3c3 d02080e7 421ac2af e5e172f5 635b0fc2`

`e58451c8` (`ss.Treatment(moves=)`, as compartments through `force()`) needs checking against v4's `products.Treatment`; `e5e172f5` is agents only (`E804` as compartments already).

- [ ] Replay each commit; full suite green; push; update PR.

### Task 3: programs, `ss.on`, the rest stay, delivery memory (25–45)

`546fba17 fb5d3c6c 5ba93f75 102f6432 86d8d0e1 50011fc7 d769486a bf1675d7(B) e6075676 e52ef390(B) 17f7ca50(B) 9955193e 07fdfef7 ce4d80fa aa86237f 6bb88d2f d4126b5a 5bbeaafa 824232d3(B) c99f5438 ada9334d`

- `bf1675d7`: water-filling over cells.
- `e52ef390`: the `waiting_` staged dimension.
- `17f7ca50`: vanishing compartments and loop-time moves for `ss.on()` (B2).
- `824232d3`: the step conveyor. Its compartmental half goes to B1, since v4 has its own conveyor (`ss.kernels.convey()`), and the agents half (clocks) stays here.
- `546fba17` may already be in v4 (`test_samples` uses `tmp_path`).

- [ ] Replay each commit; full suite green; push; update PR.

### Task 4: follow-ups, products over diseases, declared pregnancy, attributes (46–61)

`d8b1d689(B) 8419867d c5058c4e 8db7df80 50bb85ce(B) fe356e4a 270f35ee 4266fedc 46f5d303 2f0c10fc a6e2c3de ef453e07 da61f0ca 2e9fc5fc(B) 820932c0 b5b0a17f`

- `d8b1d689`: `then=` as compartments.
- `50bb85ce`: the conveyor with `ss.stay` (B1).
- `2e9fc5fc`: births driven by a flow as compartments (B4).
- `ef453e07`: the declared `ss.Pregnancy` goes in `starsim/pregnancy.py`. v4's `demographics.Pregnancy` is renamed `ss.LegacyPregnancy`, as on `v4-rs`.

- [ ] Replay each commit; full suite green; push; update PR.

### Task 5: pregnancy events, `ss.spread`, the later fixes (62–90)

`1f2f77f9 82057114(B) c856e39b d40c3218 ccdf8309 195c7859 16643b84 c9920adf(B) 314e523c(B) 1341164c fc7c8bd5(B) 584b4373 31e7089c ac5ef437 331b4013 3704190c 5b1a2a62 182944f6 b35d005c e924ee52(B) ab75d5ad(B) dc83aea1 7efdc3c5 958b820a 79ecc00b 06374473 15626732(B) e34a9e76 a6642333`

- `82057114`, `c9920adf`, `314e523c`: attributes as compartments (B6).
- `fc7c8bd5`: `ss.spread` as compartments (B5).
- `e924ee52`, `ab75d5ad`: synchronized flows (B3).
- `15626732`: burn-in as compartments (B5).
- `3704190c`: condoms. Check it against v4's `eff_condom`, which `graph.py` mentions.

- [ ] Replay each commit; full suite green; push; update PR.

### Task 6: the events branch and the burn-in fix (91–98)

`6ba95487(B) 49140978(B) c27c2c6a(B) 40491a51 14c21d45(B)`

These are compartmental almost entirely (B7). Replay their agent-side parts, docstrings, and tests (with the compartmental tests xfail).

- `14c21d45`: its births-at-the-first-step half belongs to B4/B5. Its crossing-ownership half has no counterpart in v4 (no `claim`), so drop it and say so in the log. Keep the new assertion in `test_burnin`, xfail as compartments until B5.

- [ ] Replay each commit.
- [ ] The merge `28207463` resolved `v4-rs` against `v4-rs-events` in `pregnancy.py`, `sim.py`, `solvers.py`, `transitions.py` and the tests, and that resolution is in no single commit. Check that `tests/` matches the `v4-rs` tip (`git diff v4-rs -- tests/`) except for renames, xfails, and v4's own tests. Port any difference in behavior.
- [ ] Merge `implementation.md`: v4's file is the base, with the `v4-rs` entries added and marked `[B]` where compartmental support is deferred.
- [ ] Full suite green; push; update PR; tell Robyn Phase A is done, so the ports can move to `v4-rs-port`.

## Phase B: rebuild the compartmental features on the kernels

Each task:
1. Removes its xfails, which are the failing tests.
2. Adds the kernel operations or program fields it needs to `starsim/kernels.py` and `solvers/compartmental.py`, and to `gradients.py` where the JAX path must agree, or makes JAX raise `E804` for it.
3. Runs the perf guard before and after.
4. Ends with the suite green, a push, and the PR updated.

Each task starts by reading the design in the `v4-rs` implementation log and the `v4-rs` code (`git show v4-rs:starsim/solvers.py`) for the behavior, then fits it to v4's program.

- **B1 Conveyors:** reconcile `v4-rs`'s step conveyor (clocks, fixed dwells, `stages='step'`, `ss.stay`; `824232d3`, `50bb85ce`) with v4's `convey()`. Keep one conveyor, Cliff's if it can carry clocks and `ss.stay`, and put the choice to Cliff if the two differ in meaning.
- **B2 Triggers:** `ss.on()` as compartments (`17f7ca50`): vanishing source compartments with per-group weights, and loop-time moves with a note.
- **B3 Synchronized flows:** `ss.on(other.tr)` as compartments, in the kernels' crossing step and in `force()` (`e924ee52`, `ab75d5ad`).
- **B4 Flow-driven births:** `ss.birth(per=)` as compartments, including births at the first step after back-dating (`2e9fc5fc`, `14c21d45`).
- **B5 Back-dating:** `ss.spread` and `ss.Pregnancy(burnin=)` as compartments (`fc7c8bd5`, `15626732`). Back-dating must not disturb a crossing's other engines; that is the bug `14c21d45` fixed on `v4-rs`, and `test_hiv_vertical` covers it.
- **B6 Attributes and clocks as compartments:** `82057114`, `c9920adf`, `314e523c`, and clocks.
- **B7 Fetal levels:** `ss.mother`, delivery routing, prenatal transmission, fetal deaths, `ss.Lactation` and `BreastfeedingNet` as mass action (`6ba95487`, `49140978`, `c27c2c6a`).
- **B8 Programs as compartments:** water-filling by priority, the `waiting_` dimension, and `then=` (`bf1675d7`, `e52ef390`, `d8b1d689`), if Phase A found they needed removed stepping paths.

## Review focus

1. **Ports on the new names.** STIsim and HPVsim use `ss.MaternalNet` and `eligible=`. Ported library code and tests must use v4's names. The ports will hit the errors and `python -m starsim.migrate`, so check the migration script covers what the `v4-rs` features added (e.g. `ss.mother` with `PrenatalNet`). Test: `python -m starsim.migrate` on the `v4-rs` `tests/test_pregnancy.py`, then run it.
2. **`solver='auto'` with a deferred feature.** A model using a Phase B feature must choose agents, not crash in the kernels. Test, in Task 2 and again per Phase B task: `ss.Sim(solver='auto', ...)` with each deferred feature gives `sim.solver == 'abm'`.
3. **`sim.new()` and `restart()`.** v4 replays `sim.inputs` in `sim.new()`. Keywords and modules added by `v4-rs` (`burnin=`, `ss.Capacity`, `then=`, `queue=`) must survive it. Test, in Task 5: `sim.new()` of a sim with each gives the same results under `abm`.
4. **Save and load.** v4's engines `shrink()` before saving (R15). New per-agent arrays and engine state from `v4-rs` (event logs, clocks, fetal links, delivery memory) must shrink too. Test, in Task 6: a save and load round trip of the pregnancy and programs acceptance models under `abm`.
5. **JAX paths.** `sim.gradient()` and `sim.log_density()` on a model with a `v4-rs` feature must agree with the run or raise `E804`, never silently ignore it. Test, in each Phase B task: the gradient of a corpus model with the feature, against finite differences, or `E804`.

## Log

| # | Old | New | Notes |
|---|---|---|---|
| docs | c327d2a1, 3458ebee | 5721ecf6, a839ee7b | Review docs and handoff |
