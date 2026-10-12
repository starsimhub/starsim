# Vignettes

Translations of the model examples in `examples` into the lingua franca.

## Purpose

Vignettes show how the lingua franca would be used in practice. Each one takes a model from `examples` and expresses it in the lingua franca, demonstrating that the specification is sufficient.

They are also where the project's test is applied: **a model should never be easier to write without the lingua franca than with it.** Each vignette is held to two numbers from its original: no more lines of model code, and no more compute time. A vignette that misses either is not a failed vignette; it is a finding for `design`.

## Contents

One vignette per example, containing:

- The lingua franca representation of the model
- A comparison with the original: lines of model code (counted the same way as in `examples`), wall-clock time on the same machine, and agreement of the results within a stated tolerance
- Which parts were written as declared transitions and which needed a function or a `step()` method
- Notes on anything awkward, ambiguous, or inexpressible, as feedback to `design`

Where a model can be run under more than one paradigm, the vignette should show the interconversion and discuss what the change of paradigm does and does not preserve.

## Relationship to other folders

- Sources its models from `examples`.
- Tests, and generates feedback for, the specifications in `design`.
- Provides the source for the `webapp` comparisons folder.
