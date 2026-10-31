---
name: v4.naive
description: Rerun the naive-user benchmark on Starsim v4 (simulated newcomers with only the docs, llms.txt, the migration pages, or guesses, doing 16 tasks with hidden reference answers), score it, and write one report to v4_design/<date>/reviews/naive.md. Use when Cliff runs /v4.naive or asks to rerun the naive-user review.
---

# Naive-user review

Checks the "Naive user" targets in `v4_design/targets.md`: every final answer right, and no silent wrong first attempts (code that runs cleanly and gives a wrong number). Read [../README.md](../README.md) first: its rules (snapshot, subagents, report shape) apply here.

## References

- `references/benchmark/`: the task sheets (`tasks.md`, set 1, T1 to T10; `tasks2.md`, set 2, H1 to H6), the data for T8, and `score.py`, which scores one or more `answers.json` files.
- `references/reference/`: the hidden reference answers, computed with SciPy and NumPy only. **Never show these to a persona.**
- `references/expert/`: expert v4 solutions, one per task, which show that each answer can be reached in v4.
- `references/oct17_review.md`: the first run (2026-10-19), with its personas, scores, and NR findings. Its relative links point at `oct17/naive_review/`. It is the baseline for the first run of this skill.

Keep the task set fixed, so that runs can be compared. If Cliff asks for harder tasks (the oct17 review's "What to run next" lists some), add them as a new set (`tasks3.md`, with its references), and score the old sets separately.

## Steps

1. **Snapshot** v4 as in the README. Generate `llms.txt` into the scratchpad from the snapshot (`cd <snapshot>/docs && python make_api.py`, then move the outputs out, or point it at a scratch folder), without touching `ssv4/docs/`.
2. **Check the expert solutions still reach the references** (run each in `references/expert/` and compare with `references/reference/answers*.txt`). Update an expert solution whose API changed, in place. A task that v4 can no longer do is a finding in itself.
3. **Run the four personas** as subagents, at most two at a time, each on both task sets (all 16 tasks), in a scratch folder per persona (`<scratchpad>/naive/<persona>/`). Each persona may read only what its user would have, and never the Starsim source, the tests, `v4_design/`, the references, or earlier transcripts:
    - **Careful novice:** the `.qmd` sources in the snapshot's `docs/`.
    - **Assistant:** the generated `llms.txt` only.
    - **Starsim 3 user:** `docs/migration/v3_v4.md`, and its own Starsim 3 habits.
    - **Habitual user:** `docs/ten_minutes.qmd` once, then guesses.
   
   All of them may read docstrings (`help()`) and what Starsim prints. Tell each to write one script per task, an `answers.json` in the format `score.py` reads, and a `log.md` recording every first spelling it tried, every error and note it saw, and every first attempt whose number it later changed, with that number.
4. **Score** with `python references/benchmark/score.py <scratchpad>/naive/*/answers.json`. Count the silent wrong first attempts from the logs: a first attempt that ran without an error and whose number would have failed `score.py`.
5. **Reproduce** each problem the personas met in a minimal script, and check it against the NR findings, `deferred.md`, and the latest plan.
6. **Write** `v4_design/<date>/reviews/naive.md` in the README's shape. The details are a table of persona by task (right or wrong, and whether the first attempt was silently wrong), the first spellings each persona tried for the main concepts (as in the oct17 review's "First spellings" table), and tool calls and minutes per persona. Copy each persona's `log.md` and `answers.json` to `v4_design/<date>/reviews/naive/<persona>/` so the report's evidence is kept.
7. **Clean up** the snapshot worktree and any background shells.
