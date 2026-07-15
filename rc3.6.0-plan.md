# rc3.5.2 performance plan

Implementation + testing plan for the performance work on branch `rc3.5.2-performance`. Consolidates and **corrects** the recommendations in `starsim-ngm/starsim_recommendations.md` and GitHub issue #1404, after verifying every load-bearing claim against the *original* benchmark source (`starsim-ngm/malaria_benchmarks/run_benchmarks.py`), not synthetic sims.

## Status

Plan only — no `starsim/` source has been modified. Awaiting approval before implementation.

## Method & benchmarking rigor

- All magnitude numbers are direct min-of-N wall-clock (`time.perf_counter`) with warmup, on the *isolated* operation. cProfile is used **only** for call-attribution (which function calls what), never for magnitudes — its per-call overhead inflates unevenly and misled an early draft of this analysis.
- Every magnitude claim is cross-checked with a second, independent method before being asserted (e.g. the deepcopy cost below was confirmed by both a caller-attributing monkeypatch and a direct head-to-head copy benchmark, which agree within noise).
- Reference model for all numbers here: benchmark **scenario a** (`human-ode + mos-eir`, 3-node grid, ~10k humans configured but only 60 compartment-agents, 5 simulated years, daily `dt`, `crn=False`) — the recommendations' own "decisive measurement". `malariasim` is importable locally; `laser-ngm` is not, so only the Starsim side is measured (which is all these recommendations concern).

## Verified findings (what's true, what isn't)

| Claim in `starsim_recommendations.md` / #1404 | Verdict against original source |
|---|---|
| The loop-plan sort is pure waste (date/dur object keys) | ✅ **True** — object sort 116 ms → `np.lexsort` 0.34 ms (**341×**), and the result is a **bit-identical plan** (verified on the real 62k-entry plan, relative *and* absolute time) |
| All module time-vectors are identical in the common case → skip-sort viable | ✅ **True** (all value-identical) |
| `loop.run()` ≈ 0.44 s | ✅ measured **419 ms** |
| ODE/compartmental model still forces a full `People` (`n_nodes×n_bins`=60 dummy agents) | ✅ **True** — exactly 60 agents, 17 state arrays |
| `Arr[uids]` is ~2.75× slower than a raw gather | ⚠️ **Already fixed on this branch** — `arr[uids] ≈ arr.raw[uids]`; residual only 1.2–1.5× on tiny (<2k) arrays; gappy `.values` already uses the fast numba gather |
| Births/demographics are expensive per step | ✅ **True** — births ~**double** runtime; ~50/50 between the per-agent Bernoulli scan and 13-array per-step `grow` |
| `People.update_results` recomputes a cumulative sum every step | ✅ **True** — `cum_deaths[ti] = np.sum(new_deaths[:ti])` is O(n²), ~33× fixable, result-identical |
| **"Sim.init() is *mostly* `make_plan`" / "more than half the run is `make_plan`"** | ❌ **False** — `make_plan` is ~26% of init; the sort within it ~13% of init (~8% of the ~1 s run) |
| **"Not construction deepcopy … not a bottleneck"** | ❌ **False** — a deep copy of the timeline vectors is the *single largest* init cost (~36% of init) |

Note: absolute counts have grown since the doc was written (the model is now 34 funcs/tick and 62k plan entries, vs the doc's 22 and 40k), so absolute times differ, but the structural conclusions above are what matter.

## The corrected init story (settled)

Direct, non-cProfile phase timing of `Sim.init()` on scenario a (≈395 ms warm):

| init phase | time | % of init | root cause |
|---|---|---|---|
| module timeline copy (`init_modules_pre`) | ~141 ms | **~36%** | `Timeline.init()` deep-copies the 6 time-vectors from `sim.t` for each matching module |
| `sim.t` construction (`init_time`) | ~138 ms | **~35%** | building the object-dtype date/dur arrays once via `date.arange`/`dur.arange` |
| `loop.init` / `make_plan` | ~101 ms | **~26%** | of which the object-key sort is ~50–116 ms |
| everything else | ~8 ms | ~2% | — |

So **~71% of init is timeline machinery** (copy + construction), not `make_plan`. The two biggest levers are timeline-related and were *not* in the original recommendations.

### The deepcopy, isolated and settled

`timeline.py:417 Timeline.init()` has an existing fast-path (lines 429–437): when a module's `(start, stop, dt)` match the sim's, it copies the sim's already-built time vectors instead of rebuilding them. But it copies with `sc.dcp`:

```python
for attr in self._time_vecs:              # ['tvec','tivec','timevec','yearvec','datevec','relvec']
    new = sc.dcp(getattr(sim.t, attr))    # deep-copies ~1826 date/dur/datedur objects per vector
    setattr(self, attr, new)
```

Direct head-to-head on the real vectors (min-of-N, warmup, no cProfile):

| copy strategy | cost for all 6 vectors, per module | vs current |
|---|---|---|
| `sc.dcp` (current) | **47.4 ms** | — |
| shallow `.copy()` | **0.022 ms** | **2144× faster** |
| reference share | 0.25 µs | — |

Per-vector `sc.dcp` cost is entirely the object arrays: `relvec` 26.5 ms, `datevec` 6.7 ms, `timevec` 6.2 ms, `tvec` 6.0 ms; numeric `tivec`/`yearvec` ~0.001 ms. The vectors are **read-only after init** (only the scalar `self.ti` mutates), so an independent-array shallow copy (shared immutable date elements) is safe and preserves current semantics.

## Prioritized plan

Two tracks: **init** (construction cost — dominates cheap/short/compartmental runs) and **run** (per-tick cost — dominates large/long runs). Init Phase 1 is the highest value-per-risk: two bit-identical changes remove ~50% of init.

### Phase 1 — init, bit-identical, low risk (one PR)

1. **Timeline vector copy: `sc.dcp` → shallow copy** — `timeline.py:434`.
   - Replace the deep copy in the share-from-sim fast-path with an independent shallow array copy (or reference share).
   - Impact: −~141 ms on scenario a (**~36% of init**); 2144× on the operation. Bit-identical (same date objects).
   - Test: `baseline.yaml` unchanged; new `test_timeline.py` assertion that a module timeline's vectors equal the sim's and are independent objects (mutating `self.ti` on one doesn't affect the other).

2. **Plan sort: skip-if-uniform + numeric `lexsort` fallback** — `loop.py:make_plan` (line ~219); `@dataclass(slots=True)` on `LoopEntry` (line 14).
   - When all `abs_tvecs` are identical (the common case, verified), build the plan pre-ordered (`funcs × ticks`, `ti = tick`) with no sort. Otherwise sort numeric keys: `np.lexsort((func_orders, tvec.years))`. Never put `date`/`dur` objects in a sort key.
   - Impact: sort ~116 ms → ~0.34 ms (**341×**), ~13% of init; also removes ~530k `dur._compare_args` calls/init. Directly resolves #1404.
   - Correctness (verified): the rebuilt plan is **bit-identical** in (func identity, `func_order`, `ti`, label). Why the current sort is slow (answering #1404's "see why the comparisons are so slow"): the key is a `(dur, int)` tuple, and CPython tuple ordering invokes *both* `dur.__eq__` (228 ns) and `dur.__lt__` (223 ns) per comparison — each running interpreted `_compare_args` → `.years`; a `(float, int)` tuple compares at 35 ns. You cannot make object comparison fast (dunder-frame floor ≫ 35 ns), so the fix is numeric keys, not faster comparisons. Numba is irrelevant (can't JIT object compare; `np.lexsort` is already C).
   - Test: new `test_loop.py::test_plan_identity` — capture the legacy plan and assert equality across {relative, absolute, with/without demographics, **heterogeneous dt**}; `baseline.yaml` unchanged.

3. **`People.update_results` cum_deaths O(n²) → running sum** — `people.py:505`.
   - `res.cum_deaths[ti] = res.cum_deaths[ti-1] + res.new_deaths[ti]`. Result-identical; ~33× on that line.
   - Test: `baseline.yaml` unchanged.

4. **Gate loop instrumentation** — `loop.py:store_time`/`run`/`to_df`.
   - Record per-entry `perf_counter` only when profiling is requested; make `to_df()` lazy. Removes per-tick instrumentation from the default path. Result-identical.

### Phase 2 — init construction (medium effort)

5. **Cheaper `sim.t` construction** — `timeline.py:443–492`.
   - The ~138 ms builds object-dtype date/dur arrays. Options: prioritize the float `yearvec` and derive `datevec`/`timevec`/`relvec` lazily (only when accessed for plotting/output); vectorize date construction; avoid building `relvec`/`datevec` eagerly when unused. Needs a spike to confirm which vectors are actually needed at init vs on-demand.
   - Impact: targets up to ~35% of init; medium risk (touches the canonical time representation) → guarded by `baseline.yaml` + `test_timeline.py`/`test_time.py`.

### Phase 3 — run/per-tick (separate PRs)

6. **Demographics** — `demographics.py:get_births` (121), `people.py:grow` (354), `arrays.py:grow` (603).
   - Count-based birth draw (default under `crn=False`, since you're OK with a one-time `baseline.yaml` refresh) instead of the per-agent Bernoulli scan; vectorized bulk `grow` (grow all 13 state arrays in one pass); optional pre-scheduled constant-rate mode. Impact: up to ~halve runtime for models with births.
   - Test: distributional match to the per-agent path over N seeds; one-time `baseline.yaml` update via `update_baseline.py`; the `crn=True` path stays bit-identical.

7. **Skip no-op lifecycle phases** — `loop.py:collect_funcs`; `modules.py`.
   - Don't add plan entries for phases a module doesn't override (`update_results` with no auto-states and no override; base `step`). Small direct win; shrinks the plan. Result-identical → `baseline.yaml`.

### Phase 4 — structural / as-measured

8. **People-optional / null-People** for compartmental models — `people.py`, `sim.py`, `loop.py`. Premise verified (60 dummy agents), but for the ODE model the payoff is removing People's plan phases/results/aging, not the (cheap) 60 agents — so modest here; larger for bigger compartment counts. Design spike first (collect_funcs hardcodes People phases; modules register states on People).
9. **Distribution bulk path** — `distributions.py`: CRN-correct `filter`/`rvs` over a `uids` array (the real residual of rec #3, since `Arr` indexing is already fast). Must stay CRN-identical.
10. **Non-expanded plan** (stretch) — iterate `funcs × tvec` in `run()` without materializing 62k `LoopEntry`; build `self.plan` lazily for `to_df`/`plot`/`insert`.
11. **Result cadence** (opt-in) — record every k steps / opt out of built-in People results.

## Testing strategy

- **Hard gate for every result-preserving change:** `cd tests && ./run_tests`, with `test_baselines.py` (exact `baseline.yaml` match) as the stop condition. This covers Phase 1 (all bit-identical), Phase 2, and the result-preserving parts of Phase 3–4.
- **New correctness tests:** plan-identity (`test_loop.py`), timeline-vector independence (`test_timeline.py`), distributional CRN tests for behavior-changing modes (`test_demographics.py`, `test_randomness.py`), compartmental reference for null-People (`test_diseases.py`/new).
- **Behavior-changing changes** (count-based births default under `crn=False`): explicit one-time `baseline.yaml` refresh + a note in the changelog; `crn=True` remains bit-identical.

### Performance regression guard (CPU-normalized, bounded, minimal)

The repo already has the normalization machinery but not the guard: `test_baselines.py::test_benchmark()` computes a CPU-speed factor `ratio = mean(sc.benchmark(which='numpy')) / ref` (`ref = 270` MOPS on a reference i7-12700H), scales `min(t_init)`/`min(t_run)` by it, and stores them in `benchmark.yaml`. But it only *prints* previous-vs-new — **nothing asserts a bound**, so a regression passes silently. Keep the addition minimal:

1. **Assert a major-regression bound.** After each optimization lands, re-record the (now faster) normalized baselines with `./update_benchmarks.py`, then **fail** when a normalized time exceeds `recorded × REGRESSION_FACTOR` (default **1.5×**). Deliberately loose — catches *major* regressions (e.g. reinstating the deepcopy or the object-key sort), not small drift. Baselines refresh *after* each speedup, so the guard tracks the improved state.
2. **Three minimal cases** (the framework-overhead focus of this work), each sized so a single timed run is **~1 s and never more than ~5 s**, each stored in the YAML with its own CPU-normalized number and bounded:
   - **bare-sim run, many timesteps** — e.g. `ss.Sim(start=0, stop=365*10, dt=ss.days(1))` run with no diseases/networks: pure per-tick framework/loop overhead, which is the main target.
   - **`sim.init()` alone** (no run) on that same many-timestep config — isolates the init cost (timeline deepcopy + construction + `make_plan`), the Phase 1/2 target and the #1404 case.
   - **births** — a small sim with demographics, timing `run()`, to guard the Phase 3 demographics work.
3. **xdist is fine.** `sc.benchmark()` measures the machine at that moment, so parallel contention slows both the benchmark calls and the sim proportionally and largely cancels in the `ratio`; with `min`-of-repeats and the loose 1.5× bound, false failures stay rare. Only isolate to a serial step if it proves flaky in practice.

Reusable before/after probes for all of the above are staged in the session scratchpad.

## Expected outcome

- Phase 1 alone: ~50% of init removed on scenario a (deepcopy ~36% + sort ~13%), zero result change, low risk.
- Phase 1 + 2: attacks ~85% of init construction cost.
- Phase 3: up to ~halve per-tick runtime for models with vital dynamics.

## Open decisions

- Order: Phase 1 first (bit-identical init wins, closes #1404), or lead with a Phase 2 construction spike? (Recommended: Phase 1 first.)
- For the Timeline fix: shallow copy (minimal semantic change) vs reference share (fastest)? (Recommended: shallow copy.)
