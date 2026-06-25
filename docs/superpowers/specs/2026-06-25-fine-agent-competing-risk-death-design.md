# Fine agents face background death (competing risk) — design (2026-06-25)

**Status:** implemented. Amends the death-exclusion in
`2026-06-22-multiscale-body-weight-design.md` (which excluded `fine` agents from
*all* vital dynamics).

## Problem

`ss.Deaths.step` excluded `fine` sub-agents from the background-death draw
("fine sub-agents are not independent bodies; they don't die of background
causes"). But a fine agent resolves a *rare disease outcome* at finer scale over
a long dwell (e.g. HPV CIN→cancer, years to decades). Background death is a
**competing risk** on that outcome: a fraction of the whole bodies the cohort
sub-resolves would die of other causes *before* reaching the outcome. Excluding
fine agents from background death lets every one survive its full dwell and reach
the outcome, biasing the resolved rare outcome **high**.

Measured in hpvsim (ratio=12, multi-decade horizon): **+18% cancer / +11% CIN
prevalence**, eliminated to ~0 (z<1) by including fine agents in the death draw.
Confirmed mechanism: with background mortality off (`rel_death=0`) the over-count
vanishes (cancer rel +0.26 → +0.01); restoring it and letting fine agents face it
restores an unbiased count.

## Change

In `Deaths.step`, do **not** filter `fine` out of `death_uids`: fine agents face
the same per-agent background-death probability as whole bodies. Births
(`get_births`) and conception (`Pregnancy`) keep excluding `fine` — those are not
competing risks on an outcome; a fine sub-agent does not independently reproduce.

The death **flow** count (`n_deaths`) stays `epi_flows`-weighted (whole bodies),
so a fractional fine death is recorded as a competing-risk removal (its `scale`
leaves the alive population — correct for results) but **not** as a reported
whole-body demographic death. This keeps the reported birth/death *flows*
body-consistent while making the disease outcome unbiased.

## Scope / not addressed

- Only **background death** is a competing risk here; the validated hpvsim result
  (cancer z=−0.2, n_cin z=+1.0) shows it is the dominant one at the tested
  configs, so age-migration/emigration exclusion is left unchanged.
- The reported `new_deaths` flow does not include fractional fine background
  deaths (by design, body-weighted). The `scale`-weighted alive population and all
  disease results are correct; only the demographic death *flow* total omits the
  fractional fine removals. Revisit if a scale-weighted death flow is ever needed.

## Validation

starsim `tests/test_demographics.py` + `tests/test_multiscale.py`: 55 passed.
hpvsim: ratio=1 bit-identity intact (7.0 / 36.0); the previously-failing
long-horizon unbiasedness gates pass (incl. the 40000-agent, 5%-tolerance
`test_multiscale_matches_single_scale_mean`).
