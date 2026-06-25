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

The death **flow** count (`n_deaths`) is left `epi_flows`-weighted — see the Open
Decision below.

## Mechanism, quantified (A/B, same-seed toggle, n=4000 stop=2055, 4 seeds)

Over a run, fine agents shed **~101 sim-scale** to background death. Of that,
**~33 sim-scale die BEFORE their cancer onset** — these are the averted,
previously-over-counted cancers (and `NOFIX−FIX cancers = 33`, matching the
+19%→0 correction). The remaining ~68 die *after* onset (already counted as
cancer — no change). So the cancer reduction is exactly the pre-onset fraction of
total fine background deaths; the mechanism is confirmed, not coincidental.

## OPEN DECISION: death-flow counting under multiscale

`new_deaths` stays `epi_flows` (whole-body) weighted. Measured consequence
(n=4000, stop=2055, 4 seeds):

- Fine background deaths **under**-report `new_deaths` by **~101 sim-scale**
  (`epi_weight=0`, so a 1/ratio person's death is recorded as 0). Introduced by
  this change.
- Shrunk cancer-parents (scale `1/ratio`, `epi_weight=1`) dying of background
  **over**-report by **~97 sim-scale** (recorded as 1 whole body). **Pre-existing**
  under the split, not from this change.
- **NET population-balance drift** (`n_alive` vs `cum_births − cum_deaths`):
  **~5 sim-scale = 0.03% of the population** — negligible, because the two errors
  nearly cancel.

So the books nearly balance, but `new_deaths` is **not a clean scale-weighted
death count** under multiscale — `epi_flows` is exact only where
`scale == epi_weight`. A **scale-weighted death flow** (count each death by its
`scale`) would make `new_deaths` exact for every agent type AND balance `n_alive`
by construction — at the cost of changing the two-axis flow convention and any
mortality calibration that assumes body-weighted death counts. **Deferred — the
modeler's call.** All disease results (cancer, n_cin, scale-weighted `n_alive`)
are correct regardless of this choice.

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
