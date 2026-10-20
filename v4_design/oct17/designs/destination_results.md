# Results by destination of one transition (HP20)

## Summary

**Revision, 2026-10-19.** Decision 3 is (b), per Cliff: `select(to=)` is kept as a second spelling. All four decisions are settled, and the code is under way.

**The problem.** A transition with several destinations records only its total flow. In HPV, `cin = C >> [X, R]` (CIN to cancer or clearance) must be one transition, since it is the only exact form when the outcome depends on the dwell (HP1, HP18: the separate-transitions form puts cancer 22 years early, median time to cancer 3.0 years against 24.0). So new cancers, HPVsim's main output and calibration target, are not a result: `res.cin.select(to='X')` raises `ValueError: ... not stratified` (or `KeyError: ... no dimension "to"` when stratified; `probe_small.py` 2). `admin/deferred.md:36` set this aside "because separate transitions with `p=` already give a flow per branch", which does not hold for HPV.

**The proposal.** Each destination of a transition with several has its own flow result, named `'<transition>.<state>'`, the name HP6 already gives it as a target (`ss.modify('hpv.cin.X', ...)`) and the name the care cascade gives it as an event (`on=hpv.cin.X`). The transition's own result stays the total. No `to` axis, no new public name.

```python
S, P, C, X, R = ss.states('S P C X R')
hpv = ss.Disease(S, P, C, X, R,
    infection = (S | R) >> P @ beta*(P + C),
    precin    = ss.progression('P -> C | R', dur=dur_precin, p=[p_cin, None]),
    cin       = ss.progression('C -> X | R', dur=dur_cin, p=[p_cancer, None]),
    cases     = ss.outcome(0.8*cin.X, delay=ss.years(2)),     # a destination flow in a model expression
)
hpv.stratify(age=ss.years([0, 15, 25, 35, 45, 55, 65]))
sim = ss.Sim(hpv, ..., data='cancers.csv').run()                # a data column "cin.X" (with an "age" column) is matched
res = sim.results.hpv
res.cin.X                                  # new cancers: a flow, by age like any other
res['cin.X'].select(age='45-55').resample('year')
res.cin.X + res.cin.R == res.cin           # the total is the sum of its destinations
res.cin.X.cum                              # cumulative cancers
```

**Decisions for Cliff.**

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | How a destination's flow is named and selected | (a) a result per destination named `cin.X`, read `res.cin.X` or `res['cin.X']`; (b) a `to` axis on the total's strata, `res.cin.select(to='X')`, `res.cin.by('to')`; (c) both | (a). It is the name the destination already has as a modifier target (HP6) and as an event (care cascade: "every outcome is an event and a result of the same name"), so data columns, `ss.outcome()`, `on=`, and `ss.modify()` all use one spelling. A destination is not a property people hold, so as a dimension it would be the only one that does not apply to stocks, cannot be `stratify()`d, and has no level people are in [CK: ok] |
| 2 | Always kept, or only when asked | (a) always, for every transition with several destinations; (b) only when named in `results=` or used by an outcome or data | (a), as every transition has a flow now. It is one flow per destination (two or three per HPV genotype); the by-group memory rule (09, "every few steps") already bounds the cost. `results=` can still leave them out [CK: ok] |
| 3 | What happens to `select(to=)` | (a) raise, saying what to write (`res['cin.X']`); (b) keep as a second spelling | **Decided: (b).** `res.cin.select(to='X')` returns `res.cin.X`, and `to=` combines with other selections (`res.cin.select(to='X', age='45-55')`). `select()` is otherwise for dimensions; `to` is accepted only on a flow with several destinations, and an unknown destination raises listing them [CK: is there anything .select() is valid for? if so, any reason not to support it? so i favor b] |
| 4 | The total: stored or derived | (a) the destinations stored and the total their sum (worked out once at the end of the run); (b) all stored | (a). Nothing is counted twice, and the total cannot drift from its parts. User-visible behavior is the same [CK: ok] |

## Use cases

**HP20 (HPVsim): new cancers, by age and genotype.** HPVsim needs cancer incidence per 100,000 women by age, the genotype mix in cancers, and the age-standardized rate (with D15 and HP23). In Starsim 3, HPVsim keeps its own arrays and results (`hpv.py:342` to `429`: `new_cancers`, `new_undetected_cancers`, `cum_cancers`, filled in `step_state()` by counting `ti_cancerous == ti`, then `np.cumsum`), about 40 lines plus the by-age analyzer. In v4 today there is no way to write it declaratively: the only options are an analyzer that counts entries into `X` (which also counts any other route into `X`) or the separate-transitions form, which is wrong for HPV (HP18). Under the proposal it is `res.cin.X`, by age already, and across a family `sim.results.hpv16.cin.X` per genotype (the union result of `disease_family.md` sums them). The port deletes its cancer counters and the analyzer. Simpler than Starsim 3: yes.

**The same need elsewhere.** Any transition written `I -> [R, D]` (the spec's own example of a branch with one dwell) has deaths as `res.course.D`; STIsim's syphilis `[latent, tertiary]` and TB's `[cure, death]` read the same way. None of these needs a `to` dimension; each needs one series it can name in data.

## Semantics

**Definition.** `res.<tr>.<state>[ti]` is the number of people who made transition `<tr>` and arrived in `<state>` during the step that ends at time point `ti`, by group, under every solver; the same convention as every flow (09, "Results"). The group is the one they arrived at (as `finish_moves()` records it today), so `select(age=...)` and data by age mean the same as for the total. `sum` over destinations is the total, exactly, under every solver.

| Solver | How it is counted |
|---|---|
| agents | In `finish_moves()`, the `dsts` list already holds the UIDs per destination; each adds its weighted count (and its `bincount` by group) to its own flow. Exact |
| binomial | The split of whole people between the targets of a flow is already one multinomial draw, as a chain of binomials (`kernels.py:210` and `232`); each destination's part is recorded. Exact integers that sum to the total |
| ode | The flow into each destination's compartments (`update_targets()` already splits each flow by destination share): each share times the flow is recorded |
| gillespie | Each event already has one destination; it is counted there |
| crossing (diseases tracked jointly) | As the solver it runs on, per member |
| JAX gradients | As the ODE; the share is differentiable, so `cin.X` can be a likelihood target |

No solver is excluded. A transition with several sources (`(S | I | R) >> D`) is out of scope (see below); a transition with one destination has no destination results (`res.infection.S` raises, saying `infection` has one destination).

**`describe()`** lists them under the transition: `cin   C -> [X, R]   flows: cin (total), cin.X, cin.R`. **`to_spec()`**: nothing changes. Flow results are derived from the transitions, not written, so the model hash does not change. A data series named `cin.X` is written as data is now. **`to_code()`**: nothing changes, except that `ss.outcome(0.8*cin.X)` is written as such.

## Fit with what exists

- **Names.** The `'<transition>.<state>'` spelling exists already (`graph.destination()`, `graph.destinations()`, HP6). In model expressions the destination is an attribute of the transition object, `cin.X`, as a state's count is `I.n`; in strings it is `'cin.X'`; in results it is `res.cin.X`. A destination named like an attribute a result or transition already has (`cum`, `values`, `low`, `n`) raises when the disease is made, saying to rename the state. `res['cin.X']` always works.
- **`select(to=)`.** The second spelling (decision 3): on the total, `to=` picks one destination's result before any other selection; `by('to')` is not added, since a destination is not a dimension.
- **Results with dimensions.** A destination flow is an ordinary `ss.Result` of kind `flow`, with the sim's dimensions, `select()`, `by()`, `resample()`, `.cum`, `to_df()` (column `hpv_cin.X`, the key kept whole), MultiSim runs and quantiles. Nothing in `results.py`'s selection changes.
- **Data and likelihoods.** A data column `cin.X` (in long form, with `age`) is matched to `res.cin.X` by the existing name match (09, "Data"). HPVsim's `genotype` column maps to members as in `disease_family.md`.
- **Outcomes.** `ss.outcome()`'s source is a weighted sum of flows; `cin.X` is one, so detected cancers are `ss.outcome(0.8*cin.X, delay=...)` (HPVsim's `new_cancers` versus `new_undetected_cancers`) or, with a `detected` dimension on `X`, `res.cin.X.select(detected=True)`.
- **Care cascade.** Its rule that every outcome is an event and a result of the same name holds for destinations: the event `hpv.cin.X` (h1's "one destination of one", which `behavior_vocabulary.md` also assumes) and the result `res.hpv.cin.X`. `hpv.cin.X.ever` and `.since` (cancer ever, time since cancer) then come from the cascade's remembered results, not from here.

**New public names:** none. **Removed:** nothing; `.select(to=)` is kept as a second spelling (decision 3). **Spec changes:** 09-results.md, "Results": add "and `<transition>.<state>` for each destination of a transition with several, whose sum is the transition's flow"; 03-states-and-transitions.md, the branching section: one sentence pointing to it, and `cin.X` among the places a destination is named (with HP6's modifier target); 14-model-spec.md: a data series may be named `tr.state`; 17-vocabulary.md: the slot row for flow names; 18-conformance.md: the contract below. `admin/deferred.md:36`: the destination half is no longer deferred (the source half stays).

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 1 | Results: make a flow result per destination at init (`init_results()` of the engine, from `graph.destinations()`), the total as their sum at `finalize()`, the attribute on the total, the name clash check | S | `solvers/engine.py`, `results.py`, `graph.py` |
| 2 | Agents: count per destination in `finish_moves()` (`flow_groups` keyed by `'cin.X'`) | S | `solvers/agents.py` |
| 3 | Compartmental and Gillespie: record each destination's share of each flow in the program's targets; the Numba kernels write one more row per destination | S to M (the kernels' flow layout) | `solvers/compartmental.py`, `kernels.py`, `solvers/crossing.py` |
| 4 | JAX: the same flows in the gradient path | S | `gradients.py` |
| 5 | Expressions and data: `cin.X` as an attribute of a transition in a model, accepted by `ss.outcome()`; data columns `cin.X` matched; `select(to=)` on the total returns the destination's result; `describe()` line | S | `transitions.py`, `expressions.py`, `calibration.py`, `presentation.py` |
| 6 | Docs: 09's results list, the HP6 section of the interventions guide, a migration note (HPVsim's `new_cancers` is `cin.X`) | S | `docs/user_guide/basics_results.qmd`, `docs/migration/v3_v4.md`, `CHANGELOG.md` |

About two days in all. **Tests** (in `tests/test_results.py` and `tests/test_contracts.py`): a cohort `P -> [C, R]` with `p=[0.3, None]`, 10,000 people: `res.course.C.sum()` within 3 standard errors of 3,000 as agents and binomial, 3,000 to within 1% as ODE, `C + R == total` exactly under every solver, by age after `stratify`; a contract across ode, binomial, gillespie (20 runs) and agents (20 runs) on cumulative `cin.X` by age, within 1% (ODE, binomial) and 3 standard errors (stochastic); a data column `cin.X` fitted with one free parameter recovers it; `res.course.select(to='C')` equals `res.course.C`, and `select(to='Q')` raises listing `C` and `R`.

**What the ports delete.** HPVsim: the counters and results for new cancers in its declared port (`v4port/framework/declared_results.py`'s workaround) and the by-age cancer analyzer; with the cascade, `new_cancers` versus `new_undetected_cancers` becomes `cin.X` and an outcome. STIsim: the per-outcome counters of syphilis's branching stage, if it moves to one transition.

## Out of scope and dependencies

- **Flows by source** (`(S | I | R) >> D`, deaths by the state they died from). Same mechanism, but sources are not named as targets yet; it stays in `deferred.md:36`. If wanted later, the same rule would give it a name of its own, not an axis.
- **Separate transitions with `p=`** already have one flow each; nothing changes for them.
- **HP18** (`given_dwell.md`) is why HPV must use one transition; this document does not depend on it, but HPV needs both.
- **Care cascade**: I assume its event for a destination is `tr.state` (as `behavior_vocabulary.md` also assumes); if it chose another spelling, this design should follow it, since the event and the result share one name.
- **Family of diseases**: the union result over members (`disease_family.md`) applies to `cin.X` as to any flow.
