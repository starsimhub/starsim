# rc3.6.0 performance plan

Implementation + testing plan for the performance work targeting **rc3.6.0** (branch `rc3.5.2-performance`). Consolidates and **corrects** the recommendations in `starsim-ngm/starsim_recommendations.md` and GitHub issue #1404, after verifying every load-bearing claim against the *original* benchmark source (`starsim-ngm/malaria_benchmarks/run_benchmarks.py`), not synthetic sims.

## Status

Plan only — no `starsim/` source modified. **Updated per the Codex review** (`rc3.6.0-plan-codex-review.md`), whose independent benchmarks are folded in below. Awaiting approval before implementation.

## Method & benchmarking rigor

- All magnitude numbers are direct min-of-N wall-clock (`time.perf_counter`) with warmup, on the *isolated* operation. cProfile is used **only** for call-attribution, never for magnitudes — its per-call overhead inflates unevenly and misled an early draft.
- Every magnitude claim is cross-checked with a second, independent method before being asserted.
- Two reference regimes are used, because which cost dominates flips between them (see the cost model below):
  - **module-heavy**: benchmark scenario a (`human-ode + mos-eir`, 3-node, 60 compartment-agents, 5 yr, daily, `crn=False`) — the recommendations' "decisive measurement".
  - **bare framework**: `ss.Sim(start=0, stop=365*10, dt=ss.days(1))` — issue #1404's repro; no diseases/networks/demographics.

## Verified findings (what's true, what isn't)

| Claim in `starsim_recommendations.md` / #1404 | Verdict |
|---|---|
| The loop-plan sort is pure waste (date/dur object keys) | ✅ **True** — object sort 116 ms → `np.lexsort` 0.34 ms (**341×**); rebuilt plan is bit-identical (verified, relative *and* calendar time) |
| All module time-vectors are identical in the common case → skip-sort viable | ✅ **True** (value-identical) |
| `loop.run()` ≈ 0.44 s (scenario a) | ✅ measured **419 ms** |
| ODE/compartmental model still forces a full `People` (60 dummy agents) | ✅ **True** — 60 agents, 17 state arrays |
| `Arr[uids]` ~2.75× slower than a raw gather | ⚠️ **Already fixed on this branch** — `arr[uids] ≈ arr.raw[uids]`; residual 1.2–1.5× only on tiny (<2k) arrays; gappy `.values` already uses the fast numba gather |
| Births/demographics expensive per step | ✅ **True** — births ~**double** runtime; ~50/50 Bernoulli-scan vs 13-array per-step `grow` |
| `People.update_results` cum_deaths recompute | ✅ **True + more** — `np.sum(new_deaths[:ti])` is O(n²); **and it is not bit-identical to fix** — it is silently **one-step-lagged** (verified: `cum_deaths[0]=0` with 7 deaths at t0; final = 769 vs true 770). Fixing it is a **correctness change**, not a no-op. |
| **"Sim.init() is *mostly* `make_plan`" / "more than half the run is `make_plan`"** | ❌ **False** — `make_plan` is ~26% of init for scenario a, and only **~7%** for a bare sim (construction dominates there) |
| **"Not construction deepcopy … not a bottleneck"** | ❌ **False** — for module-heavy sims the timeline deep copy is the single largest init phase (~36%); for bare sims, timeline *construction* is ~94% |

Absolute counts have grown since the doc was written (model now 34 funcs/tick, 62k plan entries vs 22/40k), so absolute times differ; the structural conclusions hold.

## The init/run cost model (unifying picture)

Both reviews converge on decomposing total time into separable terms:

```text
total ≈ canonical timeline construction        (fixed per run)
      + modules × timeline-copy                (per module whose dt matches the sim)
      + timesteps × functions × plan/dispatch   (make_plan build+sort, per-entry dispatch)
      + timesteps × result-collection           (People + module update_results every tick)
      + model-specific computation
```

**Which term dominates depends on the model** — this is why the emphasis differs from the original recommendations:

| regime | timeline construction | module copy (deepcopy) | make_plan/sort | notes |
|---|---|---|---|---|
| bare sim, 10 yr daily (init 550 ms) | **~520 ms (~94%)** | ~0 (no extra modules) | ~40 ms (~7%) | construction is everything |
| scenario a, 5 yr (init ~395 ms) | ~138 ms (~35%) | **~141 ms (~36%)** | ~101 ms (~26%) | deepcopy + construction ≈ 71% |

Corroborating evidence from the review:
- **Input representation matters**: same 3,651 steps/plan, `start=0` → 571 ms vs `start=2000` → 355 ms init — the canonical time representation is itself a first-class target.
- **Per-module slope**: adding empty `ss.Module`s costs ~80–90 ms each (timeline copy + expanded plan): 0→577, 1→659, 2→747, 5→994, 10→1626 ms.
- **Bare run (3,651 steps, ~282 ms)**: skip `to_df()` → 263; **skip `People.update_results()` → 180** (the largest single removable run cost, ~100 ms); skip results+instrumentation+DataFrame → ~161–167.

**Implication:** the shallow-copy fix attacks the *per-module* term (big for module-heavy sims, **does nothing for a bare sim**); timeline **construction** attacks the *fixed* term (helps everything, dominant for bare sims). Hence construction is promoted to right after the shallow-copy/sort patch, and mandatory `People.update_results` opt-out is promoted into the lightweight-execution work.

### The deepcopy, isolated and settled (per-module term)

`timeline.py:417 Timeline.init()` has a share-from-sim fast-path (lines 429–437) that, when a module's `(start, stop, dt)` match the sim's, copies the sim's already-built vectors — but via `sc.dcp`, which deep-copies ~1826 date/dur/datedur objects per vector. Direct head-to-head on the real vectors:

| copy strategy | all 6 vectors, per module | vs current |
|---|---|---|
| `sc.dcp` (current) | **47.4 ms** | — |
| shallow `.copy()` | **0.022 ms** | **2144×** |
| reference share | 0.25 µs | — |

Per-vector cost is entirely the object arrays (`relvec` 26.5, `datevec` 6.7, `timevec` 6.2, `tvec` 6.0 ms; numeric `tivec`/`yearvec` ~0.001 ms). Vectors are read-only after init (only scalar `self.ti` mutates), so an independent-array shallow copy is safe.

## Implementation sequence (PRs)

Ordered per the review: fixed-cost + low-risk first, then the dominant construction term, then the per-tick term, then structural work.

### PR 1 — low-risk initialization wins

1. **Timeline vector copy: `sc.dcp` → shallow copy** — `timeline.py:434`. Independent shallow array copy (shared immutable date elements). Bit-identical. Big win for module-heavy sims (~36% of scenario-a init); **no effect on a bare sim** (no module timelines to copy). Test: vector *equality* + container *independence*; do not imply that mutating shared date elements is supported (they're immutable).
2. **Plan sort: skip-if-uniform + numeric `lexsort` fallback** — `loop.py:make_plan` (~219). When the **canonical numeric absolute-time vectors** (with unit/type check — *not* object identity) are uniform across modules, generate entries directly in `for tick: for function:` order (`ti = tick`), no sort. Else `np.lexsort((func_orders, times))`; never sort date/dur objects. Bit-identical (verified). Closes #1404. *Why the current sort is slow:* the key is a `(dur,int)` tuple; CPython tuple ordering calls both `dur.__eq__` (228 ns) and `dur.__lt__` (223 ns) per compare, each running interpreted `_compare_args`→`.years`; a `(float,int)` tuple is 35 ns. Object comparison can't be made fast (dunder-frame floor ≫ 35 ns), and Numba can't JIT it — so numeric keys are the fix.
3. **`@dataclass(slots=True)` on `LoopEntry`** (`loop.py:14`) — primarily a **memory** reduction across 10⁴–10⁵ instances; treat CPU benefit as unproven unless a direct benchmark shows it.
4. **Tests:** `test_loop.py::test_plan_identity` capturing the legacy plan and asserting equality; explicit **heterogeneous-fallback** matrix — coincident times, different `dt`, relative *and* calendar timelines, sparse schedules, `loop.insert()` insertions, near-equal floating-point times. `test_timeline.py` container-independence. `baseline.yaml` unchanged.

### PR 2 — canonical timeline construction (promoted; the dominant bare-sim cost)

5. **Cheaper/lazy `sim.t` construction** — `timeline.py:443–492`. cProfile attributes the cost to `date.arange` (repeated date/dur construction + rounding) and `date.from_array` (rebuilding dates from the numeric year vector). Spike: determine which of `datevec`/`timevec`/`relvec` must exist at init vs can be **lazy cached properties** derived from `yearvec`/`tvec`; in the duration-based path, avoid eagerly building a calendar date sequence *and then* reconstructing dates when only one representation is used by the loop. These vectors are public and may be read by modules during init → preserve the API via lazy properties. Targets up to ~94% of bare-sim init / ~35% of scenario-a init. Guarded by `baseline.yaml` + tests across all `start/stop/dt` type combinations and every public timeline attribute.

### PR 3 — lightweight per-tick execution overhead

6. **Gate loop instrumentation; lazy DataFrames** — `loop.py:store_time`/`run`/`to_df`. Record per-entry `perf_counter` only under an explicit `profile=True`; make `to_df()` build the DataFrame lazily. Make the tradeoff explicit in the API: a later `to_df()` **cannot** reconstruct per-entry `cpu_time` if timestamps weren't recorded (column absent/`NaN`); plan metadata stays available regardless. ~15–25 ms of the bare run. Result-identical.
7. **People results opt-out / cadence** (promoted from "as-measured") — the **largest removable bare-run cost** (~100 ms). Add `people_results=False`, a configurable result cadence, and skipping auto-state counts not requested. May recover much of the null-People benefit with far less structural change; useful beyond compartmental models. Opt-in/behavior-affecting → off by default preserves `baseline.yaml`.
8. **Skip no-op lifecycle phases** — `loop.py:collect_funcs`; `modules.py`. Don't add plan entries for phases a module genuinely inherits as no-ops. **Must** distinguish a truly inherited base method from an override that calls `super()` or gains behavior via a mixin. Reduces plan size (per the empty-modules slope) and per-tick dispatch. Result-identical.
9. **`cum_deaths` → O(n) running sum *and* correctness fix** — `people.py:505`. Implement the established convention `cum_deaths == np.cumsum(new_deaths)` with an explicit first-step branch:
   ```python
   res.cum_deaths[ti] = res.new_deaths[ti] if ti == 0 else res.cum_deaths[ti-1] + res.new_deaths[ti]
   ```
   This is **not** bit-identical: it fixes the current one-timestep lag (verified). Record it as both an O(n²)→O(n) perf improvement **and** a behavior correction aligning with `cum_infections`. **Will change `baseline.yaml`** when deaths occur at `ti==0` or the final step. Test the full `cumsum` invariant including constructed cases with deaths at `ti==0` and the final step.

### Later PRs

10. **Demographics** — `demographics.py:get_births`, `people.py:grow`, `arrays.py:grow`. Count-based birth draw (default under `crn=False`; one-time `baseline.yaml` refresh) instead of the per-agent Bernoulli scan; vectorized bulk `grow`; optional pre-scheduled constant-rate mode. Up to ~halve runtime for models with births. Tests: distributional match to the per-agent path over N seeds; `crn=True` stays bit-identical.
11. **Distribution bulk path** — `distributions.py`: CRN-correct `filter`/`rvs` over a `uids` array (the real residual of rec #3). Must stay CRN-identical.
12. **Null-People design spike** — for compartmental models. Premise verified (60 dummy agents), but much of its benefit may already be captured by PR 3's People-results opt-out; scope after PR 3 to see what's left.
13. **Non-expanded plan** (stretch) — only if the uniform fast path leaves meaningful init/memory overhead: iterate `funcs × tvec` in `run()` without materializing 10⁴–10⁵ `LoopEntry`; build `self.plan` lazily for `to_df`/`plot`/`insert`.

## Testing strategy

- **Hard gate for result-preserving changes:** `cd tests && ./run_tests`, with `test_baselines.py` (exact `baseline.yaml` match) as the stop condition — covers PR 1, PR 2, and the result-preserving parts of PR 3 (instrumentation gating, no-op filtering).
- **Explicitly baseline-changing:** `cum_deaths` (PR 3.9) and count-based births (PR 10) refresh `baseline.yaml` via `update_baseline.py` with a changelog note; each ships with a dedicated correctness test (cumsum invariant; distributional match).
- **New correctness tests:** plan-identity + heterogeneous-fallback matrix (`test_loop.py`), timeline container-independence and all-type-combination construction (`test_timeline.py`/`test_time.py`), People-results-off equivalence, distributional CRN tests (`test_demographics.py`/`test_randomness.py`).

### Performance regression guard (CPU-normalized, bounded, minimal)

The repo already has the normalization machinery but not the guard: `test_baselines.py::test_benchmark()` computes `ratio = mean(sc.benchmark(which='numpy')) / ref` (`ref = 270` MOPS on a reference i7-12700H), scales `min(t_init)`/`min(t_run)` by it, and stores them in `benchmark.yaml` — but only *prints* old-vs-new; **nothing asserts a bound**.

1. **Assert a major-regression bound.** After each speedup, re-record the (now faster) normalized baselines with `./update_benchmarks.py`, then **fail** when a normalized time exceeds `recorded × REGRESSION_FACTOR` (default **1.5×**). Loose by design — catches major regressions (reinstating the deepcopy/object-sort), not drift; baselines track the improved state.
2. **Core cases** (kept minimal per prior guidance; each sized to ~1 s, never > ~5 s; store separate normalized numbers — do *not* collapse into one score, since they guard different terms):
   - **bare `init()`**, 3,651 daily timesteps — guards timeline construction + `make_plan` (PR 1/2).
   - **bare `run()`**, same timeline — guards per-tick dispatch + result collection (PR 3).
   - **births** `run()`, small sim — guards demographics (PR 10).
3. **Recommended additions** (cheap, guard the newly-promoted work): **People-results on/off** pair (guards PR 3.7, the largest removable run cost) and **one heterogeneous-`dt`** sim (guards the PR 1 sort fallback). Optional/available from the review: an empty-modules-scaling point (guards the per-module slope) and a realistic lightweight ODE model.
4. **xdist is fine** — `sc.benchmark()` measures the machine at that moment, so parallel contention slows both the benchmark calls and the sim proportionally and largely cancels in the `ratio`; with `min`-of-repeats and the 1.5× bound, false failures stay rare. For sub-second cases, size the timeline (or repeat the isolated op in-process) so timer/scheduler noise is small. Isolate to a serial step only if it proves flaky.
5. **Existing large benchmark** (`benchmark_large.py`, 100k agents/100 yr; one recent run measured 1.14× vs the stored 3.5.0 normalized baseline — a single run, not a confirmed regression) **stays as a tracker but is not the primary gate**: it is intentionally agent/network-dominated and does not guard the framework-overhead regime this work targets.

Reusable before/after probes for all of the above are staged in the session scratchpad.

## Expected outcome

- **PR 1** substantially improves *module-heavy* lightweight init (shallow copy) and any module count (numeric plan), but by design leaves most *bare-sim* init untouched — that cost is canonical `sim.t` construction.
- **PR 1 + PR 2** attack the fixed + per-module init terms together (up to ~94% of bare-sim init).
- **PR 3** removes the persistent per-tick tax (People results ~100 ms + instrumentation ~15–25 ms of the bare run) and corrects `cum_deaths`.
- The combination most likely to close the motivating ODE-regime gap: shallow copies + numeric/uniform plan + lazy/cheaper timelines + optional People results + gated profiling — attacking fixed-per-run, per-module, and per-timestep overhead separately. This decomposition also explains why a residual gap can persist in ~30 s runs: fixed init alone can't account for it, but mandatory results, lifecycle dispatch, People machinery, and array/distribution costs accrue every timestep.

## Open decisions

- Confirm the PR sequence (PR 1 → PR 2 → PR 3 → later), matching the review. (Recommended.)
- Timeline copy fix: shallow copy (minimal semantic change) vs reference share (fastest)? (Recommended: shallow copy.)
- `cum_deaths` correctness fix: land in PR 3 with a `baseline.yaml` refresh now, or split into its own clearly-labeled behavior-change PR? (Recommended: own small PR, so the perf PRs stay bit-identical.)
