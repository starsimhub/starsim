---
name: starsim-rust-validate
description: >
  Use when validating that a Rust-backed Starsim module (starsim.rust / ssr)
  produces output equivalent to its pure-Python counterpart, or when building
  the equivalence harness for a newly ported module. Covers the tiered
  equivalence model (identical / allclose / discrete / mismatch), how to run
  starsim.rust.compare(), and how to interpret which tier a result should land
  in. Also use when the user asks "do the Rust and Python versions match?".
---

# Validating Rust/Python equivalence

The harness lives at `starsim/rust/validate.py` and is exposed as
`starsim.rust.compare`. It compares the flattened result arrays of a reference
sim against a test sim and reports the weakest equivalence tier found.

## Tiers (strongest to weakest)

| tier | meaning |
|------|---------|
| `identical` | byte-for-byte equal |
| `allclose` | equal within float tolerance |
| `discrete` | same integer-rounded trajectory (counts/states match) |
| `mismatch` | genuine divergence — a bug |

The boundary is set by the math, not by choice: integer / uniform / Bernoulli /
CRN paths can be `identical`; anything touching `exp`/`log`/reductions
(rate->prob, lognormal durations, float means) will be `allclose` at best. See
`starsim/rust/SUPPORTED_SUBSET.md`.

## How to validate a ported module

1. Build two sims with the **same seed**: one using the `ss` module, one using
   the `ssr` module. Keep everything else identical.
2. Run `report = ss.rust.compare(ref, test, run=True, require=<tier>)`.
   - Phase 1 (RNG stays in Python): `require='identical'` — anything weaker is a
     porting bug, because the random draws are identical by construction.
   - Phase 3+ (RNG in Rust): counts/states should be `identical` or `discrete`;
     continuous aggregates may be `allclose`. A `mismatch` is always a bug.
3. If it fails, call `report.disp(all_rows=True)` to see per-result tiers, and
   narrow down: compare a 1-timestep, small-`n_agents` run first; check the
   first result that diverges.

## Writing the dev test

Mirror `tests/devtests_milestone47/test_rust_validate.py`: a `make_sim(seed,...)`
factory, an identical-against-self test (must be `identical`), and a
perturbation test (must be `mismatch`) to prove the harness isn't rubber-stamping.

## Gotchas

- The `discrete` tier only applies to integer-valued results; fractional results
  (prevalence, rel_sus) skip straight to `mismatch` if they're not `allclose`.
- Missing result keys (one sim has a result the other lacks) force the overall
  verdict to `mismatch` — check `report.only_ref` / `report.only_test`.
- Reductions are order-sensitive; if a float mean is `allclose` but not
  `identical`, suspect summation order before suspecting a logic bug.
