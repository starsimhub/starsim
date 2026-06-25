# Fine agents face background death (competing risk) — design (2026-06-25)

**Status:** implemented (starsim `51908273`, branch `feat/spawn-fine`). Amends the
death-exclusion in `2026-06-22-multiscale-body-weight-design.md` (which excluded
`fine` agents from *all* vital dynamics).

## Principle

A `fine` sub-agent (`epi_weight=0`) is a **fractional real person**, and a
fractional person is subject to background mortality like anyone. The two-axis
model excludes fine agents from **births** and **transmission** (a fraction does
not independently reproduce or form network edges) — but **death is different**:
it is a **competing risk** on the rare outcome the fine agent resolves at finer
scale (e.g. HPV CIN→cancer over a years-to-decades dwell).

## Problem

`ss.Deaths.step` excluded fine agents from the background-death draw. So every
fine cancer agent survived its full dwell and reached the outcome, whereas the
whole bodies it sub-resolves would lose a fraction to background death first —
biasing the resolved rare outcome **high**.

Measured (hpvsim, ratio=12, n=4000, 1990–2055, 8 seeds): **+19% cancer / +11% CIN
prevalence** (z>4). Proof of mechanism: with background mortality off
(`rel_death=0`) the over-count vanishes (cancer rel +0.26 → +0.01). Fine agents
inherit the parent's age at split, so they draw the correct age-specific
`p_death`.

## Change

`Deaths.step` no longer filters `fine` out of the death draw: fine agents face the
same per-agent background-death probability as whole bodies. `Births`
(`get_births`) and `Pregnancy` keep excluding `fine` (not competing risks; a
fraction does not independently reproduce).

The death **flow** count (`n_deaths`) is **scale-weighted** (people-space) — see
the Death-flow counting section below.

## Mechanism, quantified (A/B, same-seed toggle, n=4000 stop=2055, 4 seeds)

Over a run, fine agents shed **~101 sim-scale** to background death. Of that,
**~33 sim-scale die BEFORE their cancer onset** — these are the averted,
previously-over-counted cancers (and `NOFIX−FIX cancers = 33`, matching the
+19%→0 correction). The remaining ~68 die *after* onset (already counted as
cancer — no change). So the cancer reduction is exactly the pre-onset fraction of
total fine background deaths; the mechanism is confirmed, not coincidental.

## Death-flow counting: scale-weighted (decided + implemented)

`n_deaths = scale_flows(death_uids)` — the death flow counts the people-weight
(`scale`) removed, so a fine agent's death records its 1/ratio of a person and a
shrunk split-parent its `<1` scale. This makes `new_deaths` match the scale that
leaves `n_alive`, so the population books balance by construction.

Why the body-weighted (`epi_flows`) alternative was rejected: it left a
compensating-error structure under multiscale (measured n=4000, stop=2055, 4
seeds): fine deaths **under**-reported `new_deaths` by ~101 sim-scale
(`epi_weight=0`) while shrunk cancer-parent deaths **over**-reported by ~97; the
NET drift was small (~0.03% of population) but only because the two errors nearly
cancelled, and `new_deaths` was not a clean people-space count.

**Deliberate asymmetry — deaths scale-weighted, births/pregnancies NOT.** This
change is **deaths-only**. Birth and reproduction flows (`Births.n_births`,
`Pregnancy` births/pregnancies/conceptions, the CBR denominator) stay
`epi_weight`-weighted, because a birth is a **whole body reproducing** — one body
makes one baby regardless of its result-scale, and fine agents never reproduce.
`starsim/tests/test_multiscale.py::test_pregnancy_counts_are_scale_weighted`
enforces this (reproduction is body-conserved across the split; scale-weighting it
incorrectly halved the count). A death, by contrast, removes a (fractional)
person from the result population. So the framework keeps **body-weighted
reproduction flows** and a **people-weighted death flow** — different because the
events are different, not an oversight.

Mortality-calibration note: at `ratio=1` (`scale==epi_weight`) `new_deaths` is
unchanged, so existing single-scale calibration is unaffected; under multiscale it
is now a people-space death count (consistent with the scale-weighted population).
All disease results (cancer, n_cin, scale-weighted `n_alive`) are unaffected by
this choice.

## Scope / not addressed

- Only **background death** is treated as a competing risk; emigration (hpvsim
  `AgeMigration`) still excludes fine agents. This is a **config-dependent
  assumption**: at the tested nigeria configs the death fix alone gave z<1, so
  emigration's competing-risk effect is negligible there — but a high-migration
  setting could retain a smaller analogous over-count.
- Unbiasedness of the cancer count also assumes **cohort scale conservation**; the
  split's non-cancer-parent full-restore conserves it only approximately
  (~3 pp — a separate, smaller residual, not addressed here).

## Validation

starsim `tests/test_demographics.py` + `tests/test_multiscale.py`: 55 passed;
ratio=1 bit-identity intact (7.0 / 36.0). hpvsim: cancer +0.19 → **−0.007 (z=−0.2,
8 seeds)**, n_cin +0.11 → **+0.03 (z=1.0)**; multigenotype **+0.05 (z=0.9, 12
seeds)**; the 40000-agent, 5%-tolerance `test_multiscale_matches_single_scale_mean`
passes. Two under-powered hpvsim gates (multigenotype 6-seed; cancer-conservation
single-seed) flaked on Monte-Carlo noise once the bias was removed and were
powered up (hpvsim `bf7515bd`, `ffd978ce`).
