# Merging v4-rs-port into v4-rs-ck: checklist

Started 2026-10-20. Cliff's instruction: merge `origin/v4-rs-port` into `v4-rs-ck`; adopt the port where it is clearly better, write tradeoffs to [merge_tradeoffs.md](merge_tradeoffs.md); for the delivery layer, keep v4's cascade design (peers, `who=`/`what=`, events, memory) and rebuild the port's features on it, in the spelling of the care cascade's third draft ([care_cascade.md](../oct17/designs/care_cascade.md)) where it has one. The merge is left uncommitted.

## Conflicts outside the delivery layer

- [x] `solvers/agents.py`, `solvers/compartmental.py`, `solvers/engine.py`, `solvers/choose.py`
- [x] `transitions.py`, `conditions.py` (the port's events get their own leaf, `op='move'`), `dimensions.py` (v4's split rule), `graph.py` (v4's error for an immediate split across destinations)
- [x] `connectors.py`, `diseases.py`, `expressions.py`, `presentation.py`, `results.py`, `sim.py`, `__init__.py`
- [x] Tests: `test_transitions.py`, `test_dimensions.py`, `test_demographics.py`, `test_networks.py`, `test_results.py`, `test_spec.py`; `baseline.yaml` (v4's for now; regenerate at the end)
- [x] Spec 05 (v4's split rule)
- [x] The suite runs, and every failure outside the delivery layer is fixed

## The delivery layer, rebuilt on v4's cascade

`interventions.py` and `products.py` start from v4's version; each of the port's features is added back in v4's spelling, with the port's tests rewritten to it.

- [x] Typed probability as coverage (`ss.probperyear(0.5)`, over the step or the round)
- [x] `(sim)` against `(sim, uids)` by required arguments, at every call site
- [x] A delivery's `move=` with an arrow (the source level was its first letter)
- [x] `ss.Treatment(moves=)` over several transitions and diseases, results by edge; `on_fail=` becomes a transition with `on=treatment.failed`
- [x] Held products over several diseases (efficacy keyed by disease); E306 for an efficacy by a strain susceptibles do not carry
- [x] Tests over several diseases (`combine=`), `ss.any()` in `detects=` and `where=`
- [x] Tests with several results: `ss.Diagnostic(results=...)`, events per result, a delay per result
- [x] `priority=` (conditions in tiers, scores highest first, functions of `(sim, uids)`)
- [x] `queue=` (`never_ran`, `waiting`, `waited`, `left`)
- [x] A shared capacity: `ss.Resource` and `uses=` (the port's `ss.Capacity`)
- [x] `target=` (people, a time series, a function, `ss.share()`, `ss.by()`), `above=` ('keep', 'stop', a rate, or a transition)
- [x] Giving a product again: routine delivery once per spell of eligibility (`ss.Delivery.spells`), `doses=` on a product
- [x] Numbers are people (`ss.results.people_now`) in deliveries
- [x] `on=` on a transition, with a delivery's outcome as the event (moved as it happens, under every solver)
- [x] The port's delivery tests (`test_programs.py`, `test_treatment.py`, `test_products.py`, `test_triggers.py`, and its additions to `test_deliveries.py`) in v4's spelling

## Finishing

- [x] Whole suite green: 480 passed, 11 skipped, 10 xfailed (v4 alone: 420 passed, 26 skipped)
- [x] `baseline.yaml` regenerated
- [x] CHANGELOG lines for what the port adds
- [x] `merge_tradeoffs.md` complete
- [x] Resolved files marked resolved (`git add`), merge left uncommitted (`MERGE_HEAD` is the port's tip, so its 53 commits come in with the merge commit)
