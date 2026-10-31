---
name: v4.canon
description: Review Starsim v4 against the textbook canon (the 42 models of Keeling and Rohani and Vynnycky and White) for the targets "nothing should be ugly" and "nothing should be slow", and write one report to v4_design/<date>/reviews/canon.md. Use when Cliff runs /v4.canon or asks to rerun the canon review.
---

# Canon review

Checks the "Canon" targets in `v4_design/targets.md`: each textbook model should be written in v4 as the book writes it (nothing ugly) and run fast at the default settings (nothing slow). Read [../README.md](../README.md) first: its rules (snapshot, timing, subagents, report shape) apply here.

## References

- `references/models/`: the canon, one file per chapter (`kr_ch2.py` to `kr_ch8.py`, `vw.py`, `vw_stoch.py`), with the harness in `common.py` (a tight SciPy reference for each model, and scoring at the default settings and at 64 substeps) and `solvers.py` (one model under every solver). Run a chapter with `python kr_ch2.py` (or with model names); `kr_ch6.py` takes `closed|noise|tau|ccs|imports`. These are maintained here: fix them in place when the API changes.
- `references/oct17_review.md`: the first canon review (2026-10-19), with every model's result and the CR findings. Its relative links point at `oct17/canon_review/`. It is the baseline for the first run of this skill.

## Steps

1. **Snapshot** v4 as in the README, and run from `references/models/` with `PYTHONPATH=<snapshot>`.
2. **Bring the models up to date.** Run every chapter. Where a model fails because the API changed (not because v4 is wrong), update it to the current spelling, the one a user would write after reading the docstrings, and note the change. If the model now needs something a user would not think to write (e.g. `first="X"` because the states aren't called `S`), that is an "ugly" finding, not just an update.
3. **Time it.** Add timing to the harness if it isn't there yet (build and run separately, the median of three runs after a first warm-up run, and the SciPy reference's time on the same grid). Keep the change in `references/models/common.py`.
4. **Score each model** on the two targets:
    - **Ugly**, if any of these: the code doesn't read like the book's equations (count the lines against the book's right-hand side); it needs a setting other than the defaults to give the right answer (e.g. `substeps=`, `people=True`, an explicit initial value that should be implied); it can't be written at all; or a run prints a warning or note that is wrong or is noise for this model (e.g. a step-count warning on a fast ODE). Also read `describe()` for each model and flag anything that a reader of the book would not recognize.
    - **Slow**, if the default run takes over 1 s, or over 10 times the SciPy reference's time for an ODE model. These thresholds are provisional; report the times so that Cliff can set them.
    - Also report accuracy against the reference at the default settings (the oct17 bar: 1e-6 for an ODE, a distributional test for a stochastic model), since a model that is fast but wrong at the defaults is not met either.
5. **Fan out if useful.** The chapters are independent; up to four subagents (e.g. KR 2–4, KR 5–6, KR 7–8, VW) can each run and score their chapters into a findings file in the scratchpad. Brief them with the README's rules and step 4's definitions.
6. **Check** each failure with a minimal repro, and cross-reference the oct17 CR findings, `deferred.md` (including Won't do: e.g. the SDE models of KR 6.1 and 6.2 are gone by decision, so mark them "not expressible, by decision" rather than as a failure), and the latest plan.
7. **Write** `v4_design/<date>/reviews/canon.md` in the README's shape. The details section is one table with a row per model: model, lines in v4, lines of the reference right-hand side, auto solver, error at the defaults, time at the defaults (and the SciPy time), ugly (and why), slow, and the change since the last report.
8. **Clean up** the snapshot worktree and any background shells.
