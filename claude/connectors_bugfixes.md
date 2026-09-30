# `connectors.py` bug audit

Audit of `starsim/connectors.py` (all 107 lines: `Connector`, `seasonality`) for real, unambiguous defects only: wrong results on in-contract input, documented arguments that crash or do nothing, silent data corruption, and logic/unit errors. Style, docstrings, performance, and contrived corner cases are out of scope. Audited against the working tree on branch `rc3.6.2` at commit `3d8dc9d5` (Starsim 3.6.1 editable install, numpy 2.4.6); repro scripts are in the session scratchpad under `connectors/`.

**Nothing in this document has been applied.**

## Summary

No bugs found.

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| – | – | – | No bugs found | – |

## Verified clean

The following hypotheses were tested and found correct:

- **Docstring example**: runs as written (SIS, weekly `dt`, `scale=0.5, shift=0.2`).
- **Timing and `shift`**: one factor is recorded per connector step (157 for 157 weekly steps; 37 when the connector has `dt=ss.month`). With `shift=0.2`, peak `rel_beta` falls at fraction-of-year 0.203, matching the documented meaning of `shift`.
- **Applying `rel_trans`**: `rel_trans` is set on all active agents, and is non-negative because of the clamp.
- **Newborns**: they are covered, because connectors run after demographics and before disease transmission in the loop.
- **Default `diseases`**: `diseases=None` resolves to all of the sim's diseases.
- **`plot()`**: runs. It emits a harmless "no artists with labels" legend warning, which is cosmetic and out of scope.
