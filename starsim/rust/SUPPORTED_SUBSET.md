# Supported subset for Rust porting

This document defines what a Starsim module's step methods may contain to be
*portable* to the Rust backend, and explains where the equivalence-tier
boundary (see `validate.py`) comes from. The `rust-portability-linter` agent
checks a module against this spec; the `starsim-rust-port` skill performs the
port.

## The model

A `ssr` module is an `ss` module subclass that overrides only its hot methods
(typically `step` and `step_state`) with calls into a compiled Rust kernel.
Everything not overridden runs the inherited Python. The kernel operates on the
module's existing numpy-backed state arrays via **zero-copy views** — the data
stays contiguous and Python-owned; only the per-timestep computation moves to
Rust.

In **Phase 1**, the kernel does *not* draw random numbers. Python draws them
with the existing numpy RNG (including permutations) and passes the arrays into
the kernel. This makes equivalence bit-exact by construction and isolates "is
the logic ported correctly" from "is the RNG reproduced". RNG moves into Rust
only in Phase 3, validated against the Phase-1 kernels.

## Portable constructs (kernel-eligible)

A step method is auto-portable if its hot path consists only of:

- Elementwise numpy array math on state arrays (`+ - * /`, comparisons).
- Boolean-mask logic on `BoolArr`/`BoolState` (`&`, `|`, `~`, `^`) and `.uids`.
- `uids` set operations (concatenate, unique, intersect, difference).
- In-place state writes via `arr[uids] = value`.
- Integer/float reductions that are explicitly accounted for (sum, mean, max).
- Edge-array operations on networks (`p1`, `p2`, `beta`, `dur`).
- Calls to a fixed allowlist of framework side-effects: `request_death`,
  `set_prognoses` (when itself portable), result writes.

## NOT portable (keep in Python, or refactor first)

- Arbitrary user lambdas as distribution parameters
  (`p=lambda self, sim, uids: ...`) — these are opaque Python.
- Calls into arbitrary third-party Python libraries mid-step.
- Per-agent or per-edge Python callbacks inside a loop (forces a boundary
  crossing per element — see the modularity analysis; coarse per-timestep
  crossings are fine, per-element ones are fatal).
- Dynamic `isinstance`-driven control flow over heterogeneous collections that
  cannot be resolved at init time.
- Anything mutating sim structure mid-step in a data-dependent way.

## Equivalence tiers and where the boundary falls

`validate.compare()` reports the weakest tier across all results:

| Tier | Meaning | When to expect it |
|------|---------|-------------------|
| `identical` | byte-for-byte equal | integer / uniform / Bernoulli / CRN (XOR) paths |
| `allclose` | equal within tolerance | `exp`/`log` (rate->prob, waning), lognormal durations, float means |
| `discrete` | same rounded trajectory | same agents in same states despite last-ULP float drift |
| `mismatch` | genuine divergence | a real porting bug |

The practical rule: **counts and states should always be `identical` or
`discrete`; continuous aggregates may be `allclose`.** A `mismatch` on any
result is a bug to fix before shipping the kernel.

## Worked reference

`ss.SIS` + `ss.RandomNet` is the first ported pair and serves as the canonical
example the porting skill points to. Its relevant transcendental touchpoints
(why some results land at `allclose` rather than `identical` once RNG moves to
Rust): `dur_inf = ss.lognorm_ex(...)`, `waning.to_prob()` (`1-exp(-rate*dt)`),
and the transmission `beta_per_dt` rate->prob conversion.
