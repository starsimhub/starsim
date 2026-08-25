# Vignettes

Translations of the model examples in `examples` into the lingua franca.

## Purpose

Vignettes are where the design meets reality. Each one takes a model from `examples` and expresses it in the lingua franca, demonstrating that the specification is sufficient — and surfacing the places where it is not.

## Contents

One vignette per example, containing:

- The lingua franca representation of the model
- A side-by-side comparison with the original, highlighting what changed and why
- Verification that the translation reproduces the original's results, where reference outputs exist
- Notes on anything awkward, ambiguous, or inexpressible, as feedback to `design`

Where a model can be run under more than one paradigm, the vignette should show the interconversion and discuss what the change of paradigm does and does not preserve.

## Relationship to other folders

- Sources its models from `examples`.
- Tests, and generates feedback for, the specifications in `design`.
