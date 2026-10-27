# Errors, warnings, and notes

What the library says when something is wrong, doubtful, or guessed, and the few errors and warnings that are classes of their own.

Justified by [best-practices/principles.md](../best-practices/principles.md).

## The policy

**Prefer guessing a default and printing it to raising an error.** Most things a user leaves out have a sensible default, and stopping the run to ask for it helps nobody. An error is for the cases where there is no sensible guess: a name that does not exist, lengths that do not match, a request the solver cannot meet.

A message is read once, by an epidemiologist or by an AI writing a model for one. Either way it should say what is wrong in the user's own terms, name the thing it is about and what exists instead, and say what to do.

## Message shape

Every error SHOULD have:

1. **The right built-in type**: a `KeyError` (raised as `sc.KeyNotFoundError`, so the message is not quoted) for a name that is not found, a `TypeError` for a value of the wrong kind, an `AttributeError` for something that cannot be read, and a `ValueError` for the rest. A class of its own only where code needs to catch that kind of error apart from others (see [Errors and warnings that are classes](#errors-and-warnings-that-are-classes)).
2. **One line saying what is wrong**, using the names the user gave things.
3. **What to do about it**, as code the user could paste where possible, with the closest matching names for anything misspelled (and none when nothing is close).

The message is what is searched for, documented, and tested (`pytest.raises(KeyError, match='has no parameter called')`), so it is kept word for word across versions where it can be.

```text
TypeError: the rate of the transition "recovery" is a plain number (a rate times a duration), not a rate:
       rate = peryear(0.1) * days(3) = 0.00082158
     A rate times a duration is a probability over that duration, not a rate; give the rate itself, e.g. rate=ss.peryear(0.1).
```

Avoid a bare `ValueError: shape mismatch` or `KeyError: 'beta'`, and a traceback through library internals as the main message. Notes are one line each and need only say what was assumed or done.

## Levels

| Level | Means | What happens |
|---|---|---|
| **Error** (a built-in exception, or one of the classes below) | There is no sensible guess, or the request cannot be met | Raises an exception |
| **Warning** (a `RuntimeWarning`, or one of the classes below) | The run can proceed, but something probably needs attention | Issued through Starsim's `ss.warn`, so Python's `warnings` filters and `ss.options.warnings='error'` apply; the run continues |
| **Note** (`ss.note()`) | Something was assumed, approximated, or is worth knowing | Kept in `sim.notes` whatever the verbosity; printed as one line when `verbose` is 1 or more |

`verbose` is one integer for every solver: 0 prints nothing, 1 (the default) the solver chosen, the notes, and the progress of a run, and 2 everything, including each step.

Most guessed defaults are not named at all: `describe()` marks them `[guessed]` ([01-objects.md](01-objects.md#describe)). The few that also print a line when the sim is built (the unit, a state treated as a count) are notes. Approximations made when a solver is requested explicitly are unnamed one-line notes ([12-backends.md](12-backends.md#approximations)); `approx=True` keeps them without printing them, and they turn into `CapabilityError` under `approx=False`. The one approximation `solver='auto'` makes without being asked, Erlang stages for a dwell time, is a note too.

## Errors and warnings that are classes

Each is `ss.errors.<Name>`, and subclasses its built-in directly. An error is a class when code catches it by kind, or when it owns its message (builds it from what it is given, raised from more than one place, or part of an API contract); a warning is a class when it is worth filtering by kind (silenced in sweeps, or escalated to an error in a pipeline).

| Name | Kind | Why it is a class |
|---|---|---|
| `CapabilityError` (also `ss.CapabilityError`) | error (a `ValueError`) | The requested solver cannot run part of the model. Caught by `solver='auto'` to run as agents instead, and by users trying another solver. Names the part, what it asked for, and what would fix it |
| `AlreadyRunError` (also `ss.AlreadyRunError`) | error (a `RuntimeError`) | A sim or module that has already run (or been finalized) is stepped, finalized, or added to again. Adds "To run it again from the start, call sim.run()" to every message, from the sim and from a module |
| `DistNotInitializedError` | error (a `RuntimeError`) | A distribution made with `strict=True` is sampled before it is initialized. Builds its message from the distribution; the error of common random numbers that a model author meets |
| `DistSeedRepeatError` | error (a `RuntimeError`) | Two distributions have the same seed, or a distribution is jumped back to a state it has passed. Builds its message from the two distributions |
| `BooleanOperationError` | error (a `NotImplementedError`) | A logical operation on an array of people's values that is not Boolean (`ppl.age & ppl.female`). Builds its message from the array's type, for the four operators |
| `AtBound` | warning | A best-fit value is at or near one of its bounds. Silenced by the fit itself while a profile is computed, and by users in sweeps |
| `NotConverged` | warning | The chains of a posterior disagree, or draws diverged. A user may escalate it to an error in a pipeline |
| `LossyExport` | warning | An export to another format, or to code, left something out or changed it. The same: an export that silently loses things matters |
| `ImportIncomplete` | warning | An import could not translate something. The same, for imports |

Every other error is a built-in (see [Message shape](#message-shape)), every other warning a `RuntimeWarning`, and every note a line from `ss.note()`.

## Ordinary usage errors

Some mistakes are ordinary Python usage errors, and raise a `TypeError` or `ValueError` with a message in the same shape, as every other error does:

- giving both `rate=` and `dur=` to a transition, or more than one of `multiply=`, `add=`, `set=` to `ss.modify`;
- a state's name where the state is meant (`ss.Transition(I, 'R', dur=10)`), which says to declare the states;
- a set of sources without parentheses (`S | I | R >> D @ mu`);
- using a lazy condition where Python needs a single truth value (`if`, `and`, `or`, `not`); the message says to use `&`, `|`, `~`;
- reading a result from a module that was given to a sim (`sir.results.I` after `ss.Sim(sir).run()`): the sim ran its own copy, so the `AttributeError` (or `KeyError`) says where the result is (`sim.results.sir.I`).

## Removed since the first draft

The first draft had about 75 codes and the second 54. Most of the ones that went were errors for mechanisms that no longer exist (queued updates, the closure ladder, declared reads and writes, the dict shorthand, predicate rank) or refusals that are now a guessed default or a printed approximation. In particular: a missing `n_agents` is guessed; a spatial dimension without mobility is an ordinary model; coverage and capacity on a stratified model are behaviors the library is required to get right ([05](05-dimensions.md#coverage-and-counts-on-a-stratified-model)), not errors; and running a network model as an ODE prints a note.

Since the second draft: `E901 ResultExists` went with result caching, which is deferred. `W101` was `UnitAssumed` ("no unit given, assuming days"); a model with no unit now runs in timesteps and the code reports an inferred unit instead. An unmatched data column was a warning (`W705`) and is an error (`E705`), now that `ss.simdata(aliases=, ignore_extra=)` gives a way to deal with it. Seven codes were added, for states that are not people, sim-level stratification, units, and `solver='auto'`: `W103`, `E210`, `E211`, `W212`, `E308`, `W404`, and `W1205`.

Since the review of October 2026: the numbers were dropped, and each error, warning, and note is a class named as before (`E804 CapabilityError` is `ss.CapabilityError`), and the catalog was made from the code. The codes that nothing raised went: `W103 UnitsDisagree`, `W207 UnreachableState`, `W309 LargeStateSpace`, `W1002 CoarseFlow`, `W1203 LeapConditionMarginal`, `W1204 DiscretizationCoarse`, `E1003 RunAfterFinish` (a finished sim raises `ss.AlreadyRunError`), and `E1401 SchemaTooNew`. Three that were raised but not listed were added: `UnknownEffect`, `UnknownDisease`, and `UnknownLocation`. `OverlappingSet` is a note, as the code has it, not a warning. Every note is kept in `sim.notes`, including under `verbose=0` and `approx=True`, which only stop it being printed.

Since the review of over-engineering (27 October 2026): the named classes went, with `ss.CodedError`, `ss.StarsimWarning`, `ss.Note`, the `category=` of `ss.note()`, and the catalog. Each error is a built-in (a lookup a `KeyError`, typed arithmetic a `TypeError`, an unkept time an `AttributeError`, the rest a `ValueError`), with its message word for word; three warnings are plain `RuntimeWarning`s; and the ten notes are lines. The five errors and four warnings that code catches by kind are kept (above).

## Documentation and tests

Every error, warning, and note SHOULD have a test that triggers it, keyed by a distinctive part of its message (`tests/devtests/test_errors.py`), and the tests that check for an error match its message ([18-conformance.md](18-conformance.md)).
