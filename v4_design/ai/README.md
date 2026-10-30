# AI skills for reviewing v4

Skills that measure v4 against [targets.md](../targets.md). Each one is a folder with a `SKILL.md`; the folders are linked from `v4_design/.claude/skills/` so that Claude Code finds them (run Claude Code from `ssv4` or `v4_design`).

| Skill | Checks | Report |
|---|---|---|
| `/v4.review <model>` | One Starsim-family model against its v3 version: `covasim`, `fpsim`, `hpvsim`, `malariasim`, `mighti`, `stisim`, `tbsim`, or `typhoidsim`. Run each in its own session | `<date>/reviews/<model>.md` |
| `/v4.others` | Ghana measles, LASER, and the models from the approaches survey | `<date>/reviews/others.md` |
| `/v4.canon` | The textbook canon (Keeling and Rohani, Vynnycky and White) | `<date>/reviews/canon.md` |
| `/v4.naive` | The naive-user benchmark | `<date>/reviews/naive.md` |

`<date>` is the day's folder in `v4_design/`, e.g. `oct22`. The files a skill reuses (models, benchmarks, harnesses) are in its `references/` folder and are maintained there: when the v4 API changes, update them in place, so the next run starts from a working copy.

## Rules for every review

These apply to every skill here, and to every subagent a skill launches (pass them on in the brief).

**What is measured.** A review measures v4 and changes nothing in `starsim/`, `tests/`, `docs/`, or `v4_design/design/`. Make a snapshot of the current working tree, so that it doesn't change under the review and includes Cliff's uncommitted work: `git -C /home/cliffk/idm/ssv4 worktree add --detach <scratchpad>/ssv4 HEAD`, then `git -C /home/cliffk/idm/ssv4 diff HEAD | git -C <scratchpad>/ssv4 apply`, and copy any untracked files under `starsim/` across. Run with `PYTHONPATH=<scratchpad>/ssv4` and the `v4` env (`/software/conda/envs/v4/bin/python`). Record the commit, the branch, and whether there were uncommitted changes, at the top of the report. Remove the worktree at the end (`git -C /home/cliffk/idm/ssv4 worktree remove --force <scratchpad>/ssv4`).

**Git.** Use `git -C <dir>`, never `cd <dir> && git`. Never commit, push, stash, or touch the index of any repo.

**Lines of code.** Count model code with [shared/count_lines.py](shared/count_lines.py) (`code` excludes blank lines, comments, and docstrings). Also leave out imports and plotting, as the examples' line counts do, and say what files were counted on each side.

**Timing.** The machine is shared. Check the load first (`uptime`, `nproc`) and keep it under half the cores; at most 8 processes per agent, with one BLAS thread each (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1` for parallel workers). Compare like with like (same model size, same output), interleave the two sides (A, B, A, B), repeat at least three times, and report the median with the load at the time. Separate the first run (which includes Numba compilation) from repeated runs.

**Background work.** Run long jobs with `run_in_background`, or wait on a PID with a timeout (`while kill -0 $PID 2>/dev/null; do sleep 5; done`). Never wait with `pgrep -f` or on a marker in a log. Before reporting, check that no shell or wait loop you or a subagent started is still running.

**Subagents.** At most about 5 at a time. Brief each with its scope, the snapshot's `PYTHONPATH`, where its scratch files go (the scratchpad, not the repo), the rules above, and the file it writes its findings to (incrementally, so nothing is lost if it stops). Rerun every headline finding yourself before it goes in the report.

## The report

Write the report to `v4_design/<date>/reviews/<name>.md` (make the folder if needed). Markdown with no hard wrapping, sentence-case headings. In this order:

1. **Header.** Date, the `ssv4` commit and branch (and whether dirty), the versions of the other side, the machine and its load.
2. **Scorecard.** One row per target line in `targets.md` that the skill covers: the target, its status (**met**, **partly met**, **not met**, or **not measured**), and the evidence in one line (a number, with its comparison). Say which boxes in `targets.md` could now be ticked; don't edit `targets.md` (it is Cliff's).
3. **Since the last report.** Only if an earlier report of the same name exists (`ls -d v4_design/*/reviews/<name>.md`): list what changed, i.e. targets newly met or lost, and numbers that moved. Otherwise leave this section out; no baseline is kept on purpose.
4. **Findings.** What v4 should change to meet the targets that aren't met, ranked by how much each would move the scorecard, each with a minimal repro (a short script in the scratchpad, quoted inline or summarized with its output). Cross-reference `deferred.md` and the plan in the latest dated folder, and say for each finding whether it is already tracked.
5. **Details.** The per-model or per-task tables behind the scorecard.

Keep it short: the scorecard and findings are what Cliff reads; the details are for checking them.
