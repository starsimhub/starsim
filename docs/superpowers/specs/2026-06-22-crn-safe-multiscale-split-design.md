# CRN-safe multiscale agent splitting in Starsim

**Date:** 2026-06-22
**Status:** Draft for review
**Author:** Ryan Hull (with Claude)

## Problem

Disease models (hpvsim) want **fine-scale agents** to reduce Monte Carlo variance around rare
events (e.g. the CIN→cancer transition): replace one coarse agent of weight `W` with `ratio`
fine agents of weight `W/ratio`, each resolving the rare pathway independently, so the *expected*
count is unchanged but its variance drops ~`1/ratio`.

hpvsim implemented this at the disease layer and hit two problems:

1. **Broad, repeated scale-weighting corrections.** Every count in every module (`len(uids)` /
   `count_nonzero`) is wrong once agents have non-uniform weight, requiring hand-patched
   `scale.sum()` overrides scattered across hpv.py, interventions.py, demographics.py, network.py,
   cross_genotype.py. This is generic, not disease-specific.
2. **CRN-safe split was never achieved** (the load-bearing blocker). Splitting mints new agents
   mid-run; under Starsim's slot-based Common Random Numbers, a new agent's random draws are a pure
   function of its **slot**. hpvsim grew agents with **default slots (= sequential new UIDs)**,
   which are order-dependent: the slot an agent receives depends on how many agents were created
   before it, which differs across scenarios → fine-agent draws were not reproducible → CRN broken.

Both problems are properties of "agents can represent different numbers of people" — a
framework-level concept (Covasim shipped multiscale as first-class) — not of cervical cancer.

## Key findings from the current code

- **The scale primitive already exists but is a stub.** `People.scale` (`people.py:69`, default
  1.0) and `People.scale_flows(inds) -> self.scale[inds].sum()` (`people.py:398`, documented as
  "replacement for `len(inds)`") exist. But result counting does *not* use them: counts are raw
  `count_nonzero` (`people.py:491`, `diseases.py:605`) and the only scaling is a single global
  `× pars.pop_scale` at finalize (`sim.py:560`, `modules.py:729`, gated by `Result.scale`).
- **There is no central counting choke-point today.** Each `update_results` hand-fills results.
  And `auto_state_list` auto-*creates* `n_<state>` results for every `BoolState` (`people.py:265`)
  but `update_results` only *fills* `n_alive` — creation and filling are already out of sync.
- **CRN-safe agent creation is already solved for births.** `Pregnancy` reserves a slot range
  above the base population (`choose_slots = ss.randint(low=n_agents+1, high=slot_scale*n_agents)`,
  `demographics.py:603`) and draws the newborn slot **keyed by the mother** (`choose_slots.rvs(
  conceive_uids)`, `demographics.py:861`; multiples get `slot, slot+1, …`). This is exactly the
  reserved-range + parent-keyed-draw scheme needed for split. **Split is birth wearing a
  different hat** — and hpvsim's failure was using the default-slot `grow()` path instead of this
  one (confirmed by the author of the hpvsim attempt).

## Goal / non-goals

**Goal:** Make multiscale a first-class, CRN-safe Starsim capability via four generic primitives,
so a disease model expresses only the disease-specific decision (*where* to split). Strictly
backward compatible: when every agent has `scale == 1`, behavior is bit-identical to today.

**Non-goals:** Porting hpvsim's disease logic; deciding hpvsim's split *policy* (when/where to
split, the ratio, merge/cull policy) — those stay in the disease model. No change to default
single-scale sims' numerical output.

## Design

### Primitive 1 — `People.split(uids, ratio)` (the hard one)

Mirror `Pregnancy._make_newborn_uids` / `_set_embryo_states`:

1. For each coarse uid, draw `ratio-1` sibling slots from a reserved-range `randint` **keyed by the
   coarse uid** (so deterministic in parent identity, independent of split timing/order/count),
   using `slot, slot+1, …` for the siblings of one parent, as the multi-embryo path does.
2. `people.grow(n_new, new_slots)` to allocate UIDs.
3. **Copy all module states** from each parent to its siblings (deep per-state copy across
   `people.states`), so siblings start as identical replicas at the split point.
4. Set `scale` on parent and all siblings to `parent.scale / ratio` (conservation:
   total represented population invariant).
5. Tag siblings via a framework `BoolState` (e.g. `fine` / `subagent`) and a
   non-transmitting flag.
6. Keep the original agent as one of the `ratio` resolved agents (same slot ⇒ its trajectory is
   bit-identical to the no-split case ⇒ CRN holds for it; siblings are the new independent draws).

The reserved slot range must be sized for split volume, not just births — split can create far
more agents than pregnancy. Reserved-range sizing and collision rate is a primary test target
(see Failure mode C).

### Primitive 2 — scale-aware result counting (the broad one)

Introduce a single counting helper and route auto-generated results through it:
`count(bools_or_uids) -> people.scale[selected].sum()`. When all `scale==1` this equals
`count_nonzero` / `len`. Apply to: the auto `n_<state>` fills, the standard People flows
(`new_deaths`, `new_emigrants`, `cum_deaths`), and provide it as the idiom modules use for their
own flows (replacing manual `count_nonzero`). **Rates need matching numerator and denominator** —
`prevalence = scale-weighted affected / scale-weighted alive` — fixed in lockstep.

### `pop_scale` unification (double-counting hazard)

Today `Result.scale=True` ⇒ `× pop_scale` at finalize. If counting also becomes per-agent
scale-weighted, results double-scale. Resolution (to be confirmed in plan): initialize per-agent
`scale` from `pop_scale` (so the per-agent weight subsumes the global factor) and stop the global
multiply for scale-weighted results — OR keep them orthogonal with a clear contract that
per-agent `scale` is *relative* and `pop_scale` is the global factor applied once. Exactly one
multiply must reach each result. This is the single most likely silent corruption during
transition and gets a dedicated invariant test.

### Primitives 3 & 4 — scale-aware demographics, non-transmitting flag

- Births/deaths/migration tally proportional to scale-weighted population (falls out of Primitive
  2 once their counts route through the helper); removes hpvsim's `Level0*` wrappers.
- Networks respect the non-transmitting flag set by `split`; removes hpvsim's network exclusion.

## Testing strategy — the three CRN failure modes (mandatory, TDD)

These are written **first**, must fail before implementation, and gate the work. All run with
`ss.options.single_rng` off (multi-RNG / CRN mode).

**Failure mode A — determinism across scenarios (CRN reproducibility).**
Run a sim twice with identical seed where splits occur; assert fine-agent slots and resulting
trajectories/results are bit-identical. Then run baseline vs a perturbed scenario (e.g. an
intervention that prevents *some* splits): assert that for any agent that splits in *both*, its
fine agents are identical, and that agents which never reach the split are bit-identical to a
no-multiscale run. Failure here ⇒ slot allocation is order/timing-dependent.

**Failure mode B — non-perturbation of other agents.**
Compare a run with splitting enabled vs disabled on a population where only a subset ever splits.
Every agent that never splits must have an identical trajectory in both runs (same slot ⇒ same
draws). Failure ⇒ split is mutating global RNG state / shared slot space.

**Failure mode C — independence / actual variance reduction.**
The point of multiscale. (1) Statistical: over many seeds, the scale-weighted estimator of the
rare-event count must be ~unbiased and its variance must fall ~`1/ratio` vs no-split. (2)
Mechanism: assert fine agents of the same parent get *distinct* slots and *decorrelated*
rare-event outcomes; explicitly measure slot-collision rate vs reserved-range size and assert it
stays below a threshold for representative split volumes.

**Backward-compatibility invariant.** A suite of existing example sims with `scale==1` everywhere
produces bit-identical results before and after the change (counting helper ≡ raw count; no
double-scaling). This is the guardrail for the framework-wide counting refactor.

**Conservation invariant.** Total scale-weighted population is invariant across a split
(`Σ scale` unchanged); a fine agent's death decrements scale-weighted counts by `W/ratio`, not 1.

## Risks / open questions

- **Reserved-slot exhaustion / collisions at high split volume** — birth's `slot_scale=5`
  headroom may be insufficient; Failure mode C quantifies and the design may need dynamic
  range growth or per-parent sub-ranges.
- **`pop_scale` unification** — resolve the one-multiply contract before touching counting.
- **Recursive split / split-then-give-birth** — a fine agent later conceiving or re-splitting must
  still derive slots safely; covered by an extension test.
- **Perf** — `size = slots.max()+1` per draw means large reserved slot values inflate draw-vector
  size; measure.
- **Upstream/org cost** — this is a change to Starsim (separate package); version-dependency and
  broader test surface. Out of scope technically but real.