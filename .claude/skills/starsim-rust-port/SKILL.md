---
name: starsim-rust-port
description: >
  Use when porting a Starsim module (a disease, network, intervention, etc.) to
  the Rust backend so it runs faster while staying usable from Python — e.g.
  "convert my module to Rust", "make a ssr version of X", "speed up my custom
  disease with Rust". Walks through profiling, the portability check, extracting
  the numeric kernel, filling the Rust + PyO3 templates, wiring the ssr wrapper
  subclass, and validating equivalence. Routes to rust-portability-linter (gate)
  and starsim-rust-validate (proof).
---

# Porting a Starsim module to Rust

Goal: produce a `ssr.<Module>` subclass that overrides only its hot methods with
a compiled Rust kernel, validated to match the Python original. The user keeps
building in Python; Rust is opt-in per module.

Read `starsim/rust/SUPPORTED_SUBSET.md` first — it defines what is portable and
why the validation tiers fall where they do. The canonical worked example is
`ss.SIS` + `ss.RandomNet` (the first ported pair); point to it when in doubt.

## Workflow

### 1. Profile — is it worth it?
Run the model and read `sim.loop.cpu_df` (per-function CPU time and percent).
Only port methods that are a meaningful share of runtime. By Amdahl's law, a
method that is 3% of runtime caps your gain at 3% — leave it in Python. Port the
top entries (usually transmission `infect`, network `step`, big state sweeps).

### 2. Gate — is it portable?
Dispatch the **rust-portability-linter** agent on the module. It returns a
GO / GO(partial) / NO-GO verdict, which methods are kernel-eligible, what must
stay in Python, and the expected validation tiers. Do not write Rust for any
method it marks BLOCKING — refactor or leave it in Python.

### 3. Extract the kernel
For each portable method, identify the pure-numeric core: the state arrays read,
the masks computed, the arrays written, and (Phase 1) the random arrays that
Python will draw and pass in. The kernel signature is "numpy views in, mutations
applied in place, nothing sampled internally". Keep RNG in Python for the first
version — it makes validation `identical` by construction.

### 4. Fill the templates
Use `templates/kernel.rs` (the Rust + PyO3 function) and `templates/wrapper.py`
(the `ssr` subclass). The wrapper subclasses the `ss` module and overrides only
the ported methods, calling the kernel on zero-copy views of the state arrays;
everything else inherits the Python implementation (this also gives `validate`
a free reference). Build with maturin/cargo.

### 5. Validate
Use the **starsim-rust-validate** skill: build matched same-seed `ss` and `ssr`
sims and run `ss.rust.compare(ref, test, run=True, require='identical')`.
Phase 1 must be `identical`; anything weaker is a porting bug. Add a dev test
mirroring `tests/devtests_milestone47/test_rust_validate.py`.

### 6. (Later) Move RNG into Rust / native loop
Only after the kernel validates with RNG-in-Python. Reproducing numpy's PCG64,
permutation, and samplers in Rust is the hard part and re-introduces ULP-level
differences on transcendental paths — re-validate against the trusted Phase-1
kernel, relaxing `require` to `discrete` for counts and `allclose` for
continuous aggregates as documented in the subset spec.

## Distribution note
Library `ssr` modules ship as precompiled abi3 wheels (maturin) — users need no
toolchain. A user porting their *own* module needs a local Rust toolchain for
the build step; check for `cargo` up front and say so if it's missing.

## Hard rules
- Never sample randoms inside the kernel in Phase 1.
- Never port a BLOCKING method; mixing Python and Rust modules is cheap and
  expected (per-timestep boundary, zero-copy shared state).
- A kernel isn't done until `compare(..., require='identical')` passes.
