# Conformance

What an implementation must do to claim it implements this specification, and what must be tested for each claim in it.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md) §Test the conversion.

## The principle

> **A conversion that is not tested is a conversion that is not claimed.**

Every capability in [12-backends.md](12-backends.md)'s matrix is a claim. The matrix is machine-readable, so the test suite is generated from it: for every (element × method) cell marked exact, there is a test that it is exact; for every cell marked approximate, there is a test that the approximation is within its stated tolerance and that the approximation is *reported*; for every cell marked refused, there is a test that the specific error is raised.

## Levels

An implementation states which levels it conforms to. Each level includes the ones above it.

| Level | Requires |
|---|---|
| **L0 Core** | The object model ([01](01-objects.md)), types ([02](02-types.md)), states and transitions ([03](03-states-and-transitions.md)), parameters ([04](04-parameters.md)), the IR ([14](14-ir.md)), the error catalogue ([15](15-errors.md)), and the agent-based backend |
| **L1 Compartmental** | `ode`; exact `Homogeneous` and `MixingPool` routes; Erlang expansion; `equations()` |
| **L2 Stochastic** | `ctmc`, `tau`, `binomial`, `sde`; automatic method selection with printed reasons; extinction reporting |
| **L3 Structured** | dimensions ([05](05-dimensions.md)), effects ([07](07-effects.md)), interventions ([08](08-interventions.md)), the observation model ([09](09-results.md)) |
| **L4 Spatial** | spatial dimensions, mobility models, commuting sub-steps, geography providers |
| **L5 Inference** | calibration and sensitivity ([13](13-inference.md)) with at least optimization, MCMC, and a likelihood-free method |
| **L6 Interoperable** | import and export ([16](16-interop.md)) for at least SBML and one framework, with lossiness reporting |

An implementation MUST NOT claim a level it does not fully meet, and MUST expose `ss.conformance()` returning its levels and its per-cell capability table.

## Required test suites

### 1. Form equivalence

For every model in the corpus, the dict form, the constructor form, and the class form MUST produce byte-identical normalized IR and identical model hashes. ([01-objects.md](01-objects.md) §The equivalence rule)

### 2. Normalization

`normalize(normalize(x)) == normalize(x)` for every corpus model and every randomly permuted declaration order. Declaration order, module order, and variable names MUST NOT appear in any hash.

### 3. Round-trip

`load(save(m))` produces a model whose normalized IR equals `m`'s, for every corpus model, for the IR format. For every lossy format, the reported loss list MUST exactly match the set of declarations that do not survive.

### 4. Cross-paradigm agreement

The central empirical claim. For every corpus model and every pair of methods the capability matrix marks as mutually exact:

| Comparison | Tolerance | Test |
|---|---|---|
| `ode` vs `ctmc` | mean of 10³ CTMC replicates within 2 Monte Carlo standard errors of the ODE, at every output time, for `n ≥ 10⁵` | trajectory-wise |
| `ode` vs `abm` with `Homogeneous` | as above, agents redrawn each step | trajectory-wise |
| `ctmc` vs `tau` | distributions of peak size and peak time agree by two-sample Kolmogorov–Smirnov at α = 0.01 | distributional |
| `binomial` vs `ctmc` | agreement improves monotonically as `dt → 0`, and `check_dt()` reports convergence | convergence |
| Erlang dwell time, `ode` vs `abm` | dwell-time distributions agree by KS at α = 0.01 | distributional |
| A dimension in `ode` vs the same as an agent property in `abm` | as the `ode`/`abm` row | trajectory-wise |

Where a method is marked approximate, the test asserts that the discrepancy is within the tolerance the implementation *reported*, not within a fixed constant. An implementation that reports a large discrepancy honestly passes; one that reports a small discrepancy and delivers a large one fails.

### 5. Refusals

For every cell marked refused, the specific `E8xx` error is raised, at build time rather than at run time, with a message meeting the shape requirements of [15-errors.md](15-errors.md). A generic exception fails the test even if the run correctly does not proceed.

### 6. The paradigm-switch boundary

Take the shipped feature-model corpus of the one framework that already implements a one-word paradigm switch, push every model across `compartment`, `IBM`, and `hybrid`, and record where the switch stops working and why. **That boundary is the actual scope of this language's paradigm-independence claim**, and someone has already built the apparatus for measuring it. The result is a published table, not a pass/fail.

### 7. Closure error

For network-to-compartmental conversion, parameterize each available closure from an actual generated network and measure the closure error against the exact simulation on that same network. The closure ladder is a decade of published mathematics; the test is whether the implementation's named closures reproduce the published error behavior.

### 8. Determinism

For every corpus model: identical results across runs, machines, thread counts, and process counts, given the same IR, configuration, and seed. Parallel and serial runs bit-identical. ([10-execution.md](10-execution.md) §Determinism)

### 9. CRN

- Paired scenarios differing only in an intervention: the variance of the paired difference is lower than the variance of the unpaired difference by the factor the implementation claims.
- Every draw the summary marks CRN-safe is verified to be CRN-safe by re-running with a perturbation elsewhere in the model and asserting the draw is unchanged.
- Every draw marked unsafe is verified to be unsafe, so the report is not merely conservative.

### 10. Error coverage

Every code in [15-errors.md](15-errors.md) has a minimal failing example that raises exactly it, and a documentation page reachable offline via `ss.explain(code)`. A code with no test is removed from the catalogue.

### 11. Vignette fidelity

For every model in [vignettes](../vignettes), the translation runs, and its results match the original framework's implementation to a stated tolerance. Where fidelity is not achieved, the vignette records which construct did not survive and why. ([16-interop.md](16-interop.md) §The vignette obligation)

### 12. Documentation currency

The documentation MUST ship with the version, MUST be readable offline, and MUST be doc-tested: every code example in this folder and in the user documentation is executed in CI and its printed output compared. An agent that reads stale documentation writes broken models, and this is the only mechanism that prevents it.

## The corpus

The conformance corpus MUST include at least:

1. **SIR**, in all three forms, on every method.
2. **SEIR** with `source='I'`, testing the one place the language asks for a word.
3. **SIR with an Erlang dwell time**, testing the sub-compartment expansion.
4. **An age-stratified model with a real contact matrix**, testing the stratification operator and the mixing restriction.
5. **A metapopulation with gravity mobility**, testing the spatial dimension and commuting.
6. **A vaccination model as a dimension plus a transfer**, testing that the disease model never mentions vaccination.
7. **A two-disease model with an effect between them**, testing composition and the combiner rules.
8. **A model with a full observation cascade and a likelihood**, testing calibration end to end.
9. **A sexual-network model**, testing that compartmental methods refuse rather than approximate.
10. **A model with an `ss.process` escape hatch**, testing that the capability loss is reported and enforced.
11. **An environmental-transmission model** with `ss.Continuous`, testing the one place the per-contact factoring is extended.
12. **A model with a threshold trigger**, testing hysteresis and the absence of flapping.

Each corpus model MUST have a published expected `summary()` output, and a change to that output is a change that requires review. The summary is where every guessed default becomes visible, so a diff in the summary is a diff in the semantics.

## What conformance does not certify

- That the model is a good model. Nothing here checks epidemiology.
- That a fit converged. That is `fit.summary()`'s job, per run.
- That an approximation is appropriate for the question being asked. The language reports the approximation; the choice remains the modeler's.
