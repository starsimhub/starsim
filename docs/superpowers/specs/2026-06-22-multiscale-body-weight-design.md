# Two-axis multiscale: body weight vs result scale

**Date:** 2026-06-22
**Status:** Approved direction (Option D); spec for review
**Author:** Ryan Hull (with Claude)

## Problem

The native multiscale `split()` uses a single per-agent `scale` for everything (results *and*
demographics) and tags the **parent** agent `fine` along with the siblings. Two consequences,
discovered porting hpvsim:

1. **The transmitting/reproducing body is deleted.** `fine` gates network transmission, so the
   whole split cohort (parent + siblings) is excluded — the cancer-fated woman contributes zero
   transmission, when she should remain one whole transmitting/reproducing body until cancer. This
   diverges from hpvsim v2 (the validation anchor), where the original agent stays a full
   participant (`level0`) and only its `scale` shrinks.
2. **Reproduction happens via fractional mothers.** Demographics are scale-weighted over all
   agents (they don't check `fine`), so the cohort's birth *total* is conserved, but as "N
   fractional mothers bearing fractional newborns" — structurally wrong, diverging from v2's single
   whole body, and spawning fractional non-`fine` newborns that then mis-transmit.

The root cause: **result weight and demographic/transmission weight are conflated.** For the
*outcome-resolution* use case (HPV cancer, and its cousins: TB→death, polio, congenital outcomes,
vaccine adverse events, maternal mortality), the split represents **one body** whose rare *outcome*
is resolved at finer scale — the body must stay whole for transmission and vital dynamics, while
only the outcome result is scale-weighted. v2 expresses this with two orthogonal attributes
(`level0` = body; `scale` = result weight); native conflated them.

We must also keep serving the *population-fraction* use case (survey/sampling weights, oversampling),
where an agent genuinely represents a fraction of people and demographics *should* scale by it.

## Design: two per-agent weights

Separate the two roles into two per-agent `FloatArr`s:

- **`scale`** (existing, default `1.0`) — **result weight**: how many people this agent represents
  in *outputs*. Used by `Arr.count()`, `People.count`/`scale_flows`, every `n_<state>`/flow result,
  prevalence, and disease outcomes. **Unchanged.**
- **`epi_weight`** (new, default `1.0`) — **demographic & transmission weight**: how many whole
  people this agent acts as for *vital dynamics* (births, deaths) and *transmission participation*.

In the common (non-multiscale) case both are `1.0` and behavior is identical to today.

### `split(uids, ratio)` (revised)

- `scale[all N] = scale / ratio` — N equal result-weight sub-draws (variance reduction); the cohort
  sums to the parent's original scale. **(unchanged)**
- `epi_weight[parent]` — **kept** (the parent remains a whole body for vital dynamics + transmission).
- `epi_weight[siblings] = 0` — siblings are non-participating result-only sub-agents.
- `fine` — tagged on **siblings only** (parent is no longer `fine`). `fine` becomes equivalent to
  `epi_weight == 0`; networks continue to exclude `~fine`. The parent (not `fine`) transmits.

Net for the cohort: **1 body** for transmission/vital dynamics (the parent), **N equal sub-draws**
for results.

### Demographics use `epi_weight`, not `scale`

> **AMENDED 2026-06-25** (see `2026-06-25-fine-agent-competing-risk-death-design.md`):
> the **death** path no longer follows the rule below. `Deaths` (a) does NOT
> exclude `fine` agents from the death *draw* — a fine agent faces background
> death as a **competing risk** on the rare outcome it resolves (excluding it
> biased resolved outcomes high, e.g. +18% HPV cancer); and (b) counts the death
> *flow* by **`scale`** (people removed), not `epi_weight`. The `epi_weight`
> convention below still holds for **births / pregnancy / conception** flows (a
> birth is a whole body reproducing) and for transmission — but NOT for deaths.

`Births`, `Deaths`, and `Pregnancy` count and rate-weight by `epi_weight` instead of `scale`:

- Birth/death counts and rate denominators use scale-weighted-by-`epi_weight` sums (siblings, `epi_weight=0`,
  contribute nothing; the split parent contributes 1 whole body).
- A **newborn inherits its mother's `epi_weight`** for *both* its `scale` and `epi_weight` (a fresh whole
  agent representing `mother.epi_weight` people): `scale = weight = mother.epi_weight`. For a normal mother
  this is `1/1`; for a split parent `1/1` (a whole baby, not a cancer sub-draw); for a
  population-fraction mother `w/w`.

### Transmission

Unchanged mechanism: agents with `epi_weight == 0` (i.e. `fine` siblings) are excluded from network
edges; participants (`weight > 0`) transmit as full edge endpoints. **Transmission intensity is not
weight-scaled** — scale-aware transmission (fractional-weight infectors, needed for
near-elimination / resistance / oversampled key populations / spatial hotspots) remains a separate,
deferred workstream. This design only fixes *participation* (who is a body), not edge-intensity.

### Results

`scale`-based result counting is **unchanged** — `count()`, `scale_flows`, flows, and `n_<state>`
all use `scale`, so the cohort's outcome is resolved across N sub-draws (each `1/ratio`, summing to
the parent's original scale).

## Regimes (both supported)

| Regime | `scale` | `epi_weight` | Set by |
|---|---|---|---|
| Single-scale (default) | 1 | 1 | default |
| Outcome-resolution (HPV) | parent & siblings `1/ratio` | parent kept, siblings 0 | `split()` |
| Population-fraction (survey weights, oversampling) | `w` | `w` (equal to scale) | user, at init |

## Backward compatibility

- No multiscale: `epi_weight == scale == 1` everywhere → demographics and results bit-identical;
  `test_baselines` and the full suite unchanged.
- Existing multiscale tests (split everyone, ratio R): parents keep `weight=1`, siblings `epi_weight=0`,
  so demographic counts equal the pre-split body count (conserved), and `scale`-based result counts
  equal the represented population (conserved) — both still pass.
- Population-fraction users now set `epi_weight` alongside `scale` (a convenience to set both may be
  added). Previously demographics keyed off `scale`; this is a behavior change only for sims that
  already set non-uniform `scale` *and* relied on scale-weighted demographics — none exist in-tree.

## Testing strategy (TDD)

- **Parent stays a body:** after `split`, the parent appears in network edges and reproduces as a
  whole body; siblings appear in neither. (Fixes the transmission bug.)
- **Result vs demographic divergence:** after `split(ratio=N)`, the cohort's `scale`-weighted count
  (`people.count`) is conserved (= parent's original scale) *and* its `epi_weight`-weighted demographic
  count is 1 whole body.
- **Births are whole-body:** a split parent bears whole newborns (`newborn.scale == newborn.weight
  == mother.epi_weight`); siblings bear none; total births conserved and not fractionalized.
- **Population-fraction regime:** with `epi_weight == scale == w` (no split), demographics and results
  both weight by `w`.
- **Backward compatibility:** `test_baselines` + full suite unchanged at `epi_weight == scale == 1`.

## Non-goals

- Scale-aware transmission *intensity* (Family 3) — still deferred; this is participation only.
- Merge/coarsen (the inverse of `split`) — separate future primitive.
- DALY/value (outcome-severity) weights — a distinct third axis, intentionally not folded in.

## Risks / open questions

- **Death of a split parent vs the cohort:** if the parent dies of background causes, the cohort's
  outcome resolution is moot; whether/how siblings are culled is **disease-model territory**
  (hpvsim decides), not framework — the framework provides `split` + the two weights.
- **hpvsim migration:** this removes the need for hpvsim's `Level0Births`-style body-counting and
  parent-untagging workarounds; the port should be re-pointed at the native two-axis behavior and
  re-validated against v2 baselines.
