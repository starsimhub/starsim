# Reviewer brief

You are one of several parallel reviewers of Starsim v4, in the repo `/home/cliffk/idm/ssv4` (branch `v4`; use `/software/conda/envs/v4/bin/python` to run anything). Cliff, the author, asked for a comprehensive review of the design (`v4_design/best-practices/` and `v4_design/design/`) and its implementation (`starsim/`), looking for improvements. His words: "treat nothing as sacrosanct -- use your judgement about what an improvement means, even if it's structural, a breaking change, etc. ... overall, aim for simplicity -- first and foremost, simplicity for the user, but also simplicity of the codebase (knowing that these two things are often in opposition)."

## Rules

- **Read-only** on the repo, except your one output file in `opus_review/` named in your task. Do not edit code, docs, or tests. Do not run git commands that change anything. Scratch files go in `/tmp/claude-1000/-home-cliffk-idm-ssv4/ca97d20f-10b5-44a7-9912-4c092dbb024e/scratchpad/`.
- **Do not open** `codex_review/` or `fable_review/` (other reviews; Cliff wants this one independent). Do not read `v4_design/admin/transcripts/`.
- Context you may want: `v4_design/admin/implementation.md` (what is done, departures from the spec, known issues) and `v4_design/admin/deferred.md` (set aside, and won't-do). Don't re-propose a "won't do" item without a new reason.
- Running small examples with the v4 python to check a claim is encouraged and cheap. Running the whole test suite is not needed (the files in `tests/` are the spec of what works; `tests/test_transitions.py`, `test_dimensions.py`, `test_deliveries.py` are the v4 ones).
- Don't rewrite the spec or the code. Describe the change, with a before/after snippet where the point is about what the user writes.

## What counts as a finding

Rank by value to the user, then to the maintainer. Good findings are:

1. **User-facing simplicity**: vocabulary that could be smaller, two ways of saying one thing, a keyword that could be inferred, names that mislead, a default that is wrong 20% of the time, an error the user will hit that the library could prevent, a surface that differs between the short form and the long form.
2. **Spec vs implementation**: where the code departs from the spec silently (not in `implementation.md`'s departures list), where the spec says something the code cannot do, where the spec contradicts itself or the best practices, and where the spec specifies something that need not be specified at all.
3. **Codebase simplicity**: duplication across solvers/kernels/JAX/Python paths, abstractions that only one caller uses, special cases that could be the general case, features whose cost in code is out of proportion to their use (say what could be cut and what would be lost), modules that are too big to hold in one head, Starsim 3 compatibility that v4 could drop.
4. **Bugs** found along the way (with a repro), but bugs are not the main aim; a bug audit is a separate exercise.

Not findings: style nits, lint, docstring typos, things already in the departures/known-issues/deferred lists (unless you disagree with the decision, in which case say why).

## Output format

Write Markdown to your output file. No hard-wrapped prose (one paragraph per line). Sentence case headings. Structure:

```
# <Area>: review

One paragraph: what you read, and the overall verdict in two or three sentences.

## Top findings
Numbered, most valuable first, at most ~10. Each: a bold one-line claim, then 2–6 sentences: what is there now (with `file:line` or doc section), why it costs the user or the maintainer, what you propose, and what it would break. Include a before/after code snippet if the point is about what the user writes.

## Other findings
Shorter bullets, same shape, as many as earn their place.

## Things that are good
A few bullets. Cliff will cut things based on this review, so say what should not be cut.

## Questions for Cliff
Decisions only he can make.
```

Be concrete and specific. Prefer "cut X, because Y and Z already cover it" over "consider simplifying". Verify claims against the code before writing them; if you couldn't verify, say so. Spend your effort on depth within your area, not breadth; another reviewer has every other area.
