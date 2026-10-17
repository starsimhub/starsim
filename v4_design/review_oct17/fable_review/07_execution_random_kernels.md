# Execution order, randomness, and the compiled kernels

I reviewed spec sections `10-execution.md`, `11-random.md`, and the kernel/JAX parts of `12-backends.md` (plus the relevant rows of `17-vocabulary.md`, `18-conformance.md`, and `19-open-questions.md`), and the code in `starsim/kernels.py` (1372 lines), `starsim/gradients.py` (1121), `starsim/loop.py` (815; the v4 diff against `main` is 88 lines), the kernel-facing parts of `starsim/solvers.py` (`kernel_only`, `run_args`, `tables`, `run_all`, `run_many`, the per-step `step_*` methods, `init_program`), `sim.py` (`run`, `kernel_only`, `kernel_reason`, `gradient`, `restart`, `describe`), and `distributions.py` as far as CRN (`hash_uniforms`, slots). I read `tests/test_kernels.py` and `tests/test_gradients.py`, and the kernel entries in `v4_design/admin/implementation.md`. I ran eight scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/kernels/` (`s1_paths.py` through `s8_streams.py`): a declared SIR under the Python engine, the stepped kernels, the whole-run kernel, and JAX; steady-state timings on three model sizes; a fresh-cache compile; `describe()` with a delivery under three solvers; reproducibility of stepped and whole-run stochastic runs; CRN pairing as agents and as compartments; and stream independence between two diseases. Quoted numbers are from those runs.

## Summary

1. **Stepped stochastic kernels share one global Numba random stream, so adding an unrelated disease changes another disease's draws** (verified: `s8_streams.py`). This breaks the first rule of `11-random.md` for any stochastic compartmental run that is not taken whole (a `when=` modifier, an analyzer, `verbose=1`). Give each engine its own stream in the kernels (a seed argument to `one_step()`, or a counter-based hash like the one the agents already use). Not breaking.
2. **`verbose=1`, `profile=True`, `until=`, and `loop.insert()` all silently switch a compartmental sim from the whole-run kernel to the Python loop, which is 30 to 100 times slower** (`s2_timing.py`: 1 ms against 28 ms for an SIR, 11 ms against 1,091 ms for ten years). Results are identical, but a print flag should not be a performance cliff; at least print why, and let verbose output print from the kernel path.
3. **The first use of the kernels compiles for 29 s on this machine (the known-issues note says 12 s) with no message at all.** One compile covers every solver and every model shape (verified), which is good; print one line while it happens, and consider shipping a compiled cache or an ahead-of-time build.
4. **`integrator='jax'` is not an integrator and JAX is not a faster way to run a sim** (150 ms floor per run against 1 ms for the kernels; identical results to 7e-13). JAX earns its place for `sim.gradient()`, NUTS, and `ss.parallel(pars=...)` batches, where it is picked automatically anyway. I agree with the solvers reviewer that `integrator=` should go; from the kernel side, JAX should be an internal backend chosen when gradients or batches are wanted, never a user-facing run mode. Breaking (removes a keyword that few will have used).
5. **The spec's reproducibility story is honest about agents and silent about compartments in the code.** `describe()` prints nothing about CRN (the spec says it MUST list paired and unpaired draws and say which form is in use), `ss.options.crn` has no effect on a compartmental run, and run-level pairing works but is unreported. Run-level pairing is worth having (it cut the standard deviation of an intervention effect from 52 to 10 infections in `s7_crn_abm.py`); say so in one line of `describe()`. Not breaking.
6. **The `describe()` schedule is not the loop order and lists modules that have no step.** It printed `every day: vax, vaccine, sir` where the loop runs `sir.step_state` before `vax.step`, and `vaccine` is a product with only `start_step`/`finish_step`. For a compartmental run the real order (deliveries, then all flows) is not the loop's at all. Derive the schedule from `loop.funcs`, and say when the kernels take the run.
7. **I agree with deleting the Python stepping path (`use_kernels=False`) from the kernel side**, with one condition: the stochastic kernels are currently checked only against that path (statistically, `test_stochastic`), so the replacement test has to be against the ODE's expected values, as `test_stages` already does for the chain-binomial solver.

## Simpler for the user

### 1. One flag that reads like a verbosity setting changes the execution path

**Issue.** `sim.kernel_only()` returns False when `self.verbose >= 1` (`sim.py:1888`), when `profile=True` or `until=` is given to `run()` (`sim.py:1837`), and when anything was inserted into the loop. A user who types `ss.Sim(sir, ..., verbose=1).run()` to see the solver line gets the stepped loop. Verified (`s1_paths.py`): `verbose=0: kernel_only=True`; `verbose=1: kernel_only=False, reason='it has inserted functions, diagnostics, or verbose output'`. Timings (`s2_timing.py`, best of 5): SIR 100 days, ODE: whole-run kernel 1.0 ms, stepped kernel 15 ms, Python 28 ms; SIR 10 years daily: 11 ms, 578 ms, 1,091 ms; SEIR with 6 age groups, 365 days: 5.7 ms, 124 ms, 467 ms. Results are bit-identical between paths for the ODE (0.0 difference) and, since `init_program()` seeds the kernel stream with the same number the whole run draws (`solvers.py:2277`), identical for stochastic runs too (`s5_repro.py`: "whole vs verbose=1 identical: True"). So this is purely a hidden cost.

**Why it matters.** The whole-run kernel is the thing that makes `sim.calibrate()`, `sim.run(n_runs=)`, and `restart()` fast. A user debugging with `verbose=1` sees a different speed and, worse, `sim.restart()` raises for the same sim (`sim.py:1945`) with a reason that mentions "verbose output".

**Proposal.** Decouple verbosity from the path: the kernel path can print the solver line and the per-run summary itself (the per-step progress banner is the only thing it cannot do, and that is `verbose>=2` territory). `profile=True` genuinely needs the loop, and `until=` and insertions do too; for those, print the one-line reason (`kernel_reason()`) when `verbose` is on, so the slowdown is explained. Also fix `kernel_reason()` returning `'unknown'` when `kernel_only()` is True (`sim.py:1884`), which my scripts hit every time.

**Cost.** None to the user; a few lines moved. Confidence: high.

### 2. `integrator=` mixes a scheme, a library, and a backend, and `'jax'` is the odd one out

**Issue.** `integrator=` accepts `'rk4'`, `'euler'`, any `scipy.integrate.solve_ivp` method name, and `'jax'` (`parameters.py:325`, `sim.py:371-374`). The spec (`12-backends.md` §How a model is built) says it accepts `'scipy'` or `'jax'`; `'scipy'` raises (`s5_repro.py`: "ValueError `method` must be one of {'RK23', 'RK45', ...}"). `'jax'` is rewritten to `'rk4'` with `sim.jax = True` (`sim.py:372-374`), so it is not a scheme but a backend. It is also slower for a single run: 156 ms for the 100-day SIR against 1.0 ms in the kernels, and 112 ms against 5.7 ms for the stratified SEIR (`s2_timing.py`), because of JAX dispatch and the fixed cost of `jit`. It wins only in batches: `implementation.md` reports 1,000 runs in 0.5 s against 1.2 s. And when it cannot run a sim it falls back with a note (`s5_repro.py`: `integrator="jax" cannot run this sim, since m has a step of its own ... so it is run by the RK4 kernels`), which is a run mode that may or may not apply.

**Proposal.** Agree with the solvers reviewer: drop `integrator=`. From the kernel side, the right shape is: RK4 in the kernels is the ODE; JAX is used, without being asked, by `sim.gradient()`, by NUTS and L-BFGS-B fits, and by `ss.parallel(sim, pars=...)` when `ss.gradients.reason(sim)` is None (`run.py:1137` already does this only when `sim.jax` is set; make it unconditional, or gate it on batch size). If a switch is wanted for debugging, `ss.options.jax = False` is the place, next to `ss.options.jax_cache`. The SciPy methods were useful while the RK4 kernel was being validated; with `test_scipy` and the 1e-15 agreement recorded, they can go too, or stay as `substeps=`'s sibling for people who want adaptive steps.

**Cost.** Breaking for anyone who wrote `integrator='jax'` or `integrator='LSODA'`; the spec section "How a model is built" has to be rewritten (it is already wrong in three of five bullets, see Spec and implementation). Confidence: high.

### 3. Compile time is not hidden; it is merely silent

**Issue.** With an empty Numba cache (`NUMBA_CACHE_DIR` pointed at a scratch folder, `s3_compile.py`), the first `sim.run()` took 29.4 s on this machine; the known-issues note says about 12 s. Nothing is printed during it, at any verbosity. With a warm cache the first run is 0.08 s. The good news, verified in `s6_compile2.py`: because `step()` is one function over a `Program` whose arrays always have the same dtypes and ranks, the one compile covers every solver (gillespie, tau, sde first runs: 0.04, 0.01, 0.01 s afterwards) and every model shape (stages, ages, places, movers, the TB model: all 0.00 s first run). There is no second cliff.

**Proposal.** Print one line (`compiling the solvers for first use (about 30 s; this happens once per Starsim install)`) when the cache is cold; Numba exposes this via the dispatcher's `stats.cache_misses` or simply by timing the first call. Better, build the cache at install or first import in the background. The 12 s figure in `implementation.md` should be replaced by a range, since it is machine-dependent.

**Cost.** None. Confidence: high.

### 4. The reproducibility story is told for agents and not for compartments

**Issue.** `11-random.md` says `describe()` lists which draws are paired (`ok infection, recovery ...`, `note network.rewire samples without replacement`) and, under a compartmental solver, "says which form is in use". None of this exists: `s4_describe.py` found no line in `describe()` mentioning random, CRN, or seed, and `sim.crn_report()` (a MAY) is absent. `ss.options.crn` has no effect on a compartmental run (`s1_paths.py`: identical `n_I` with `crn=True` and `False`), which is correct but unsaid. Run-level pairing does work: with the same seed, a baseline and a scenario that diverges at day 50 first differ at step 51 under `binomial` and as agents; over five seeds the paired effect has sd 10 against 52 unpaired under `binomial`, and 20 against 28 as agents (`s7_crn_abm.py`).

**Proposal.** Two lines in `describe()`, which the code can produce from what it already knows: `random: common random numbers on (per agent, by slot)` for agents, `random: paired by run (one stream per disease, seeded from rand_seed)` for the stochastic compartmental solvers, `random: none (deterministic)` for the ODE. Drop the per-distribution `ok`/`note` table from the spec; it needs the distributions to report what they saw during a run, which is machinery for a note, and the note's content is already deducible from the model (only networks and sampling without replacement are unpaired). Keep `crn_report()` out of the vocabulary until someone asks.

**Cost.** Spec shrinks; no code removed. Confidence: medium (the per-distribution table may be wanted by the networks reviewer).

### 5. The printed schedule is not the loop

**Issue.** `sim.py:1155-1161` builds the schedule from `mods` grouped by `dt`, in module-list order, and includes every module. For an SIR with `ss.vaccinate(...)` it printed `schedule every day: vax, vaccine, sir` under all three solvers (`s4_describe.py`), while the loop at `ti=2` runs `sir.step_state, vax.step, sir.step`; the spec says a module with two rows is listed at its first, which puts `sir` before `vax`. `vaccine` is a product module with no `step` in the loop. And under `ode`/`binomial` with `kernel_only=True`, the loop does not run at all: the kernel applies the delivery at the start of the step and then all the flows (`kernels.py:1146-1149`), which is the known issue about delivery order between solvers.

**Proposal.** Build the schedule from `sim.loop.funcs` (the `(module, func_name)` list, which exists before the plan is made and is cheap), keep the first appearance of each module, skip modules whose only entries are `start_step`/`finish_step`/`update_results`, and append `(taken in one call by the kernels)` when `kernel_only()` is True. That also makes "a module moved with `sim.loop.insert()` is shown at its new place" true, which it is not now.

**Cost.** Small. Confidence: high.

### 6. The delivery-order difference between solvers could be removed rather than documented

**Issue.** As agents, a delivery runs at row 8 (after progressions, before transmission); in the kernels and the Python compartmental engine it runs before all flows. `implementation.md` records a 7% difference in people reached on the first step of TPT. The spec's row 5/row 9 split exists so that interventions can act between state updates and transmission; a declared compartmental model has no such split (its `step_state()` is empty, `solvers.py:2857`).

**Proposal.** Either move the agent-based declared engine's spontaneous transitions into `step()` as well (so declared models have one row under every solver; open question B1 "Where transitions run in the loop" lists exactly this alternative), or accept the O(dt) difference and stop listing it as an issue. The first makes the solvers agree and removes a rule the user has to learn ("not contact-mediated means row 5"); it costs within-step chains for declared models only (a declared progression and a delivery in one step), which the spec's own argument says is O(dt).

**Cost.** Changes agent-based results of declared models by O(dt); the spec table in `10-execution.md` §Where declared transitions run shrinks to one bullet. Confidence: medium.

## Simpler codebase

### 1. The step math exists four times, and the kernels contain it twice themselves

**Issue.** The same flow arithmetic (rates from a stack program, exposures through mixing terms, amounts to targets and credits, movers) is written in: the Python engine (`solvers.py:2086-2117` rates, `2365-2400` `derivs`, `2900-3130` the five `step_*` methods and `draw_targets`); the Numba `step()` (`kernels.py:705-974`); `place_rates()` (`kernels.py:351-423`), a second copy of the stack interpreter inside the kernels for the per-place Gillespie path; and the JAX `Run.rates()`/`derivs()` (`gradients.py:155-259`). What keeps them in agreement is `test_ode` (Python against stepped and whole-run kernels, 1e-9, six models), `test_jax_run` (JAX against kernels, 1e-9), `test_scipy`, and `test_stochastic` (Python against kernels, statistically, 25 runs). That is a real safety net, but every new opcode (`kernels.py:21-23` lists 21) or program field (51 fields in `Program`, `kernels.py:25-81`) is added in four places.

**Proposal.** (a) Delete the Python stepping path, as the solvers reviewer proposes; I agree from this side. Its only remaining jobs are a readable reference and the fallback when `program is False` (a rate that is a Python function of the sim, `kernel_reason`: "a rate in X is a function of the sim"). For the second, the kernel can take the function's output as a `fixed` rate per group per stage (it already takes tabulated time-varying values this way), which removes the last need for a Python stepper. (b) Fold `place_rates()` into `step()` by giving `step()` a group range; the per-place Gillespie loop then calls `step`'s rate section rather than a copy. (c) Keep JAX as the one deliberate duplicate, since it is the differentiable one, and keep `test_jax_run` as its contract. The stochastic kernels then need a test against the ODE's expected values (`test_stages` is the model), not against Python.

**Cost.** About 300 lines of `solvers.py` and 70 of `kernels.py` go; a function-of-the-sim rate becomes a tabulated rate, which is a small behavior change (it is read at the stage's time rather than at every substep). Confidence: high for (a) and (c), medium for (b).

### 2. The kernels' random numbers come from one global, process-wide stream

**Issue.** Every stochastic kernel draws from Numba's `np.random` state, which is global per thread (`kernels.py:145, 322, 324, 448, 605, 653, 1029`), seeded by `np.random.seed()` inside `run()` (`kernels.py:1136`) and `run_many()` (`:1187`), and by `kn.seed()` at the end of each engine's `init_program()` (`solvers.py:2277`). The whole-run path is sound: each engine's `run()` reseeds with its own number. The stepped path is not: all engines interleave draws from one stream, and the last engine to build its program sets the seed for everyone. Verified in `s8_streams.py` with two independent SIRs under `binomial`: whole-run, disease `a` is identical with and without disease `b`; stepped (a `when=` modifier present), it is not (`a[10:14]` alone `[502, 573, 648, 738]`, with `b` `[505, 574, 653, 744]`). As agents the same sim is identical in both cases. The spec's first consequence of per-distribution streams, "adding a module does not change any other module's draws", is violated, and so is the design-level claim that stepped and whole runs are the same computation.

**Proposal.** Pass a stream into the kernels rather than using the global one. The cheapest fix is an explicit seed argument to `one_step()` so each engine reseeds before its step from its own generator (one `np.random.seed` per step costs nothing against a 10 µs step). The better fix is counter-based: the agents' CRN already uses a splitmix64 hash (`distributions.py:40-107`); a kernel `uniform(seed, step, flow, group, k)` from the same hash gives each flow its own stream, makes stepped and whole runs identical by construction, lets `run_many()` run groups in parallel (deferred.md says this is what blocks the Ghana measles target), and removes the dependence on Numba's own Mersenne Twister and its binomial implementation, which is what makes `use_kernels=False` disagree today. Binomial and Poisson from uniforms need an inversion routine, which is a few dozen lines.

**Cost.** Stochastic results change once (any change to the stream does). Confidence: high for the bug, medium for the hashed stream (it is more code than the one-line fix).

### 3. Two paths for "which kernel takes the run", with the decision spread across four methods

**Issue.** Whether a sim is run whole is decided by `Sim.kernel_only()` (`sim.py:1886`), `Compartmental.kernel_only()` (`solvers.py:2468`), `Crossing.can_run_whole()` (`:434`), `Delivery.tabulated` (`interventions.py:993`), and `gradients.reason()` (`gradients.py:59`), and explained by a separate `Sim.kernel_reason()` (`sim.py:1866`) that re-derives the conditions in prose and can disagree with them (it returns `'unknown'` when the answer is yes). `Compartmental.kernel_only()` also caches part of its answer in `self.whole` and not the rest.

**Proposal.** One function that returns the reason or None (as `gradients.reason()` already does), with `kernel_only()` defined as `reason() is None`. The reason strings are then the same ones printed everywhere (`restart()`, `integrator='jax'` fallback, `calibrate()`), and there is no `'unknown'`.

**Cost.** None visible. Confidence: high.

### 4. `gradient()` differentiates the run in JAX and the two ends by finite differences

**Issue.** `gradients.gradient()` (`gradients.py:641-692`) takes the JVP of the run and then central differences for the parameter-to-program map and the run-to-results map, costing two restarts and two result passes per parameter, and `Density` (`:797-1121`) re-derives the parameter-to-program map a second way (linear slopes plus exact re-evaluation of the rate expressions, `:914-958`) so that NUTS has a single JAX function. So there are two gradient mechanisms with different accuracy claims (1e-10 and "linear, checked in `ss.fit()`"). This is a documented departure and it works (`test_gradient`: 1e-5 against finite differences; `test_posterior`), so I only note it: if `ss.transitions.evaluate()` can already run an expression under a JAX context (it does, `gradients.py:955`), the `gradient()` path could use the same `Density.numbers()` map and drop the central differences, leaving one mechanism.

**Cost.** Moderate work; parameters that enter the program nonlinearly outside rate expressions (initial counts via `init=`, stages) are what the central differences currently cover. Confidence: low (not verified that `numbers()` covers every case `gradient()` does).

### 5. `jax()` changes process-wide environment on first import

**Issue.** `gradients.jax()` (`gradients.py:40-41`) sets `XLA_FLAGS=--xla_force_host_platform_device_count=<cores>` the first time Starsim imports JAX. That is for NUTS chains on separate devices, and it is harmless for Starsim alone, but it is a global side effect on any other JAX user in the process, and it is what makes forked workers deadlock afterwards (known issue). Consider setting it only inside `nuts_jax()`, or documenting it with `ss.options.jax_cache`.

**Cost.** None. Confidence: medium.

### 6. The loop's v4 changes are small and reasonable; two are subtle

The v4 diff to `loop.py` is 88 lines: a lazy `plan` (so whole-run sims never build it), `_before_first_step()` (skipping step functions at `ti=0` so results index time points), `_at_start_funcs()` (wrapping declared steps in `ss.solvers.interval()`), and `insert()` accepting a module. The first-step filter (`loop.py:383-395`) encodes the time-point convention by function name (`start_step`, `update_results`, analyzers, and `finish_step` except for sim and people) and is the kind of rule that a custom module with an unusually named method will trip over silently; a comment in `Loop`'s docstring states the convention, which is good. The module-moving branch of `_insert_into_plan()` (`loop.py:496-516`) matches plan entries by `repr(entry.time)`, which works but is a string comparison in a place that elsewhere avoids comparing time objects for speed; `ti` would do.

## Spec and implementation

- **`12-backends.md` §How a model is built is wrong in three of five bullets.** It says small models use SciPy, large models and fitted models use JAX, and user functions that JAX cannot trace fall back to SciPy. The code: fixed-step RK4 in Numba for everything, JAX only on request or for gradients, no SciPy unless an `integrator=` method name is given, and `integrator='scipy'` is not accepted. `implementation.md` lists the first as a departure; the section should be rewritten rather than annotated, and open question B3's "where is the line between SciPy and JAX" is answered: there is no line, Numba is the default and JAX is the gradient backend.
- **`13-inference.md:191`** says the ODE solver "runs through JAX" and gradients come "in the same pass that runs the model". Neither is true: the run that produces results is the kernels', and a gradient is a separate JVP batch plus 2k result passes (`gradients.py:675-688`). `implementation.md` records that the gradient costs about 8 runs.
- **`11-random.md` §What CRN covers** (the `ok`/`note` listing in `describe()`), "`describe()` says which form is in use", and `sim.crn_report()` are not implemented. See Simpler for the user §4.
- **`10-execution.md` §The schedule in `describe()`** says the schedule MUST be in loop order, list a two-row module at its first row, and show moved modules at their new place. The implementation groups by `dt` in module order (`sim.py:1155-1161`). See §5.
- **`10-execution.md` §Determinism, rule 1** ("the same seed MUST produce identical results") holds for whole runs and, in practice, for stepped runs of one disease; **`11-random.md` §Streams** ("adding a module does not change any other module's draws") is violated by stepped stochastic compartmental runs (Simpler codebase §2).
- **`10-execution.md` §Order under each solver** says that under `'ode'`, `'sde'`, `'gillespie'`, and `'tau'` interventions are "called at loop times, in loop order". In a whole-run kernel the deliveries are applied by `apply_shifts()` before the step's flows (`kernels.py:1146-1149`), and the loop is never built. This agrees with the spec in effect (interventions before flows) but the spec's wording presumes the loop runs.
- **Spec says `solver=` may be given to `sim.run()`; `integrator=` may not** (`s5_repro.py`: `run(solver=)` works, `run(integrator=)` is a `TypeError`). Moot if `integrator=` goes.
- **Open questions answered by the implementation.** B3 (SciPy versus JAX): neither by default. B10 "what does CRN mean across solvers": per agent by slot as agents, per run by seed under the stochastic compartmental solvers, nothing under the ODE; unreported. B1 "where transitions run in the loop": kept as rows 5 and 9 for agents, but a declared compartmental engine has an empty `step_state()`, so compartmental runs already use the alternative ("one slot for all of a disease's transitions"); this is the source of the delivery-order known issue.
- **`19-open-questions.md` risk 8** (module order is part of the model) is reduced in practice by the whole-run kernels: for a compartmental sim with tabulated deliveries, no module order exists at run time. Worth saying in `10-execution.md`.
- **`implementation.md` known issue on compile time** says about 12 s; I measured 29 s cold on this machine, 0.08 s warm.

## Bugs

### 1. Stepped stochastic compartmental runs share one random stream across diseases

Repro: `s8_streams.py`. Two independent declared SIRs `a` and `b`, `solver='binomial'`, `rand_seed=3`, `total_pop=10_000`, with `ss.modify('a.pars.beta', multiply=1.0, when=lambda sim: True)` present to force the stepped path (`kernel_only()` False). Output:

```
whole-run: disease a identical with and without disease b: True; a[10:14] alone=[502. 573. 648. 738.] with b=[502. 573. 648. 738.]
stepped  : disease a identical with and without disease b: False; a[10:14] alone=[502. 573. 648. 738.] with b=[505. 574. 653. 744.]
abm: disease a identical with and without disease b: True
```

Cause: `one_step()` draws from Numba's global state (`kernels.py:1284-1289` into `step()` and `draw()`), seeded once per engine at `init_program()` (`solvers.py:2277`); engines in one sim share and reseed it. Fix: see Simpler codebase §2.

### 2. `kernel_reason()` returns `'unknown'` when the kernels do take the run

`sim.py:1884`. `s1_paths.py` printed `kernel_only=True reason='unknown'` for every whole-run sim. Cosmetic, but it is the string `restart()` would print if its check and `kernel_only()` ever disagreed.

## Keep

- **One `step()` function for every solver, written as scalar loops over a flat `Program`** (`kernels.py:705-974`). It is the reason a three-compartment ODE step costs 1.4 µs, the reason one compile covers everything (verified), and the reason stepped and whole runs agree to the bit. Do not split it per solver for readability; the docstring explains why it is one function.
- **The whole-run entry points** `run()`, `run_many()`, `advance()` (`kernels.py:1113-1255`) and the `kernel_only()` idea: the model being data is what lets a run be a single call, and `restart()`, `run(n_runs=)`, the particle filter, and gradient fits all rest on it. The measured speedups (28 to 100 times over the Python engine for the ODE, 20 to 50 times for `binomial`, in `s2_timing.py`; `implementation.md` reports 4.2 s to 0.16 s for Gillespie and 93 s to 6.7 s for tau on the LASER example) justify the kernels' 1,372 lines on their own; the Python path is what does not justify its 300.
- **Tabulated time-varying values and deliveries as shifts** (`solvers.py:tables()`, `init_shifts()`, `kernels.py:apply_shifts()`, `interp()`): the design that keeps modifiers, seasonality, and routine vaccination inside the whole run instead of forcing the loop. Reading smooth values at the RK4 stage times from three rows per step is exactly right and the kernels and JAX share it.
- **JAX from the same `Program`** (`gradients.py:Run`): the JAX run is built from the kernels' flattened program, not from a second description of the model, so the two cannot drift in structure, only in arithmetic, and `test_jax_run` checks the arithmetic at 1e-9 on six models. Keep JAX import lazy (`gradients.py:32-50`) and keep the compile cache option.
- **Seeding the whole run from the engine's own generator** (`solvers.py:2627`, `run_many()`'s per-run seeds at `:2699-2702`) so that `run(n_runs=)` gives exactly the single-run results for each seed (`test_restart` asserts this). Extend this to the stepped path rather than replacing it.
- **The loop as Starsim 3 left it**, with the v4 additions kept to the lazy plan, the first-step convention, and module moving. `loop.insert()` accepting a module is the right size of feature for "put screening after transmission".
- **Hashed per-slot uniforms for agents** (`distributions.py:hash_uniforms`), and the fact that a declared model as agents gets CRN for free (verified: identical draws until divergence, sd 20 against 28). This is the pattern the kernels should adopt.

## Outside my area

- `run.py:1137`: `ss.parallel(sim, pars=...)` takes all runs in one JAX call only when `sim.jax` is set; if `integrator=` goes, decide the rule (always when `gradients.reason()` is None and the batch is large, or never).
- `sim.py:248` seeds NumPy's global `np.random` at init, and `debugtools.py:266` reseeds it per step in diagnostics; neither touches Numba's stream, which is what the kernels use. Worth a comment so the next person does not assume it does.
- The solvers reviewer's proposal to drop the Python path should also cover `Compartmental.derivs()` (`solvers.py:2365`) only if the SciPy integrators go with it; `kernels.derivs()` already provides the right-hand side for `solve_ivp`, so the Python `derivs()` is used only when `use_kernels=False`.
- `interventions.py:993` `Delivery.tabulated` requires `len(sim.declared) == 1`, so any two-disease compartmental model with a delivery runs stepped and therefore hits Bug 1 under a stochastic solver.
