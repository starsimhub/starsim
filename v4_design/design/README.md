# Design

The specification of the epi modeling lingua franca, derived from the practices argued for in [best-practices](../best-practices).

## Purpose

This is the core of the project. Where `approaches` describes what frameworks do and `best-practices` takes a position on what they *should* do, `design` says exactly what this language *is*: the objects, the semantics, the compiled representation, the errors, and the tests that make its claims falsifiable.

## How to read these

Every document is normative. Requirement keywords are used in the usual sense:

- **MUST** / **MUST NOT** — a conformance requirement. An implementation that violates one does not implement this specification.
- **SHOULD** — a strong recommendation with acknowledged exceptions.
- **MAY** — genuinely optional.

Each document opens with a pointer to the `best-practices` document that justifies it, and each significant decision either cites that justification or is recorded in [19-open-questions.md](19-open-questions.md) as a choice made here. Code is written as `starsim` (`import starsim as ss`), because [the base for the lingua franca is Starsim](../README.md); where the specification departs from Starsim's current behavior, it says so.

## Contents

### Start here

- [00-overview.md](00-overview.md) — the whole language on one page, the object model, and what the compiler does.

### The language

- [01-objects.md](01-objects.md) — the class hierarchy, the three construction forms, the equivalence rule, the module contract, and `summary()`.
- [02-types.md](02-types.md) — durations, rates, probabilities, frequencies, counts; the arithmetic lattice; bare numbers; calendar time; per-module timesteps.
- [03-states-and-transitions.md](03-states-and-transitions.md) — **the core.** States, the arrow grammar, the six transition kinds, dwell times, branching, `ss.Continuous`, and every inference rule.
- [04-parameters.md](04-parameters.md) — `ss.Par`, the parameter ladder, shapes, requirements, and named data with provenance.
- [05-dimensions.md](05-dimensions.md) — stratification as an operator; dimension kinds; transfers between coordinates; the enforced restrictions.
- [06-routes.md](06-routes.md) — the force of infection, stated as an equation; route classes; layers; mobility; geography.
- [07-effects.md](07-effects.md) — `ss.Effect`, the closed target list, combiner resolution, and the `ss.Expr` predicate algebra.
- [08-interventions.md](08-interventions.md) — the four-part decomposition, products, delivery, cascades, triggers, and scenarios.
- [09-results.md](09-results.md) — auto-generated results, characteristics, the outcome cascade, the likelihood, and transmission provenance.

### Semantics

- [10-execution.md](10-execution.md) — phases, queued updates, competing hazards, interleaved timesteps, sub-steps, and determinism.
- [11-random.md](11-random.md) — per-distribution streams, common random numbers and their checked contract, five-level run identity, and convergence checks.

### Execution and inference

- [12-backends.md](12-backends.md) — the paradigm lattice, the seven methods, the backend contract, the capability matrix, and the three classes of conversion.
- [13-inference.md](13-inference.md) — calibration with nothing extra declared, automatic method selection, staged fitting, and sensitivity.

### Machinery

- [14-ir.md](14-ir.md) — the intermediate representation: schema, expression encoding, normalization, hashing, and versioning.
- [15-errors.md](15-errors.md) — the required message shape and the full error catalogue.
- [16-interop.md](16-interop.md) — import from and export to other frameworks, with fidelity stated per source.

### Discipline

- [17-vocabulary.md](17-vocabulary.md) — every public name in the language, counted, with the budget rules and the slot-type table.
- [18-conformance.md](18-conformance.md) — conformance levels, the twelve required test suites, and the model corpus.
- [19-open-questions.md](19-open-questions.md) — what was settled here, what is still open and blocking, and the risks to the design as a whole.

## The claims this specification makes falsifiable

1. **The model is data.** Everything downstream is a function of [14-ir.md](14-ir.md).
2. **The paradigm is a run-time choice.** [12-backends.md](12-backends.md).
3. **Nothing is silently approximated.** Every conversion is exact, approximate with a named closure, or refused by name.
4. **Defaults are guessed and printed.** [01-objects.md](01-objects.md) §`summary()`.
5. **The vocabulary is small enough to hold**, and the count is published rather than asserted. [17-vocabulary.md](17-vocabulary.md).

Each has a test suite in [18-conformance.md](18-conformance.md). A claim without a test is not a claim.

## Relationship to other folders

- Justified by [best-practices](../best-practices); every decision here cites the practice it follows.
- Informed by the reviews in [approaches](../approaches), which supply the evidence for those practices.
- Exercised by [vignettes](../vignettes), which translate the [examples](../examples) corpus into this specification. Each translation is a conformance test, and each failure is a documented limit rather than a defect.
