# Solvers and engines

I reviewed `v4_design/design/12-backends.md` in full, the solver parts of `00-overview.md` and `10-execution.md`, open questions B3 to B5 in `19-open-questions.md`, and the solver departures in `v4_design/admin/implementation.md`. Code: `starsim/solvers.py` (4136 lines: `Mixing` 46-311, `CountState`, `Crossing` 351-534, `Engine` 556-1522, `Compartmental` 1567-3137, `Agents` 3141-4136), `Sim.init_solver()`, `agent_reasons()`, `try_functions()`, `init_crossings()` and `solvers()` in `starsim/sim.py`, and the dispatch surfaces of `kernels.py` and `gradients.py` (the kernels themselves are another reviewer's). I ran the same SIR under all nine solver names (`all_solvers.py`), the approximation notes and `approx=` under `auto`/`ode`/`abm` (`exp_notes.py`), the chain-binomial stages fix and `substeps=` (`exp_chain.py`), `use_kernels` on and off for all five compartmental solvers, a two-disease `Crossing` under every solver (`exp_cross.py`), and the triggers that send `auto` to agents plus the `'stochastic'` heuristic (`exp_auto.py`). Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/solvers/`.

## Summary

1. **Cut the solver menu from six names plus `'stochastic'` to four: `ode`, `gillespie`, `binomial`, `abm`.** Drop `sde` (one test, zero vignettes, worst agreement with the ODE in my runs, and the spec itself rejects it as a default) and fold `tau` into `gillespie` as the automatic large-population fallback, which is what `'stochastic'` already does. Breaking (two names removed), but nobody outside the test suite uses them. (Finding 1.)
2. **Make `'stochastic'` and the tau fallback estimate events honestly.** The current heuristic (`total_pop × n_transitions`) ignores rates and run length and undercounted a 50k endemic SIS by 24x; a user asking for `'stochastic'` on a 1-year daily SIS gets a 4 s exact run where tau takes 0.00 s, and for 40 years it would be minutes. Not breaking. (Finding 2.)
3. **Honor the spec's rule for distributions on non-dwell quantities, or change the spec.** `beta=ss.lognorm_ex(...)` under `auto` silently runs as an ODE at the mean with no note and no `approx=False` error (`solvers.py:840`), where the spec promises agents under `auto` and a note under `ode`. One of the two is wrong; I argue for the spec. (Finding 7.)
4. **Collapse the three implementations of the compartmental step into two by deleting the Python solvers** (`step_binomial_once`, `step_tau`, `step_gillespie`, `step_sde_once`, `derivs`: `solvers.py:2365-3137`, about 300 lines). They exist only for user Python functions in rates and for tests; the user-function case can be served by evaluating the function once per step into the program's tables, as `tables()` already does for time-varying values. Not breaking for users. (Finding 9.)
5. **Make a `Crossing` one engine over the product graph instead of N engines with a copy-ownership protocol.** `Crossing` is the right idea (the joint state is the correct math) but `owner`/`version`/`seen`/`claim()`/`refresh()`/`changed()` is a cache-coherence protocol between engines that each hold a full copy of the same array; 28 `self.cross` checks are spread through `Compartmental`. Not breaking. (Finding 10.)
6. **Fix the agent-only-attribute error path.** A function that reads `sim.people.n_partners` under `solver='ode'` passes `init()` and then dies at step 1 with a raw `AttributeError`; under `auto` the reason printed is `does not work on groups (people)`. The spec promises an `E804` naming the attribute and offering `stratify`. Not breaking. (Finding 4.)
7. **Keep `substeps=` but drop `integrator=` from the public surface** (leave it as a developer option). It only selects between RK4 kernels, Euler kernels, JAX, and SciPy's methods; `'jax'` is really a gradients backend, not an integrator, and the spec's own `SciPy, 3 compartments` line is never printed. Breaking only for `integrator='euler'` users (tests only). (Finding 3.)

## Simpler for the user

### 1. Six solver names is two too many: drop `sde`, fold `tau` into `gillespie`

**Issue.** The user can write `solver=` with `'auto'`, `'stochastic'`, `'ode'`, `'gillespie'`, `'ctmc'`, `'tau'`, `'binomial'`, `'sde'`, or `'abm'`: nine spellings for what are, from the user's point of view, four models of randomness (none, discrete-time, continuous-time, individuals). Usage across the repo and the 23 vignettes (`grep "solver="`): `ode` 33, `binomial` 21, `abm` 9, `gillespie` 5, `tau` 3 (two tests, one README benchmark line), `stochastic` 2 (one test), `sde` 1 (one calibration test), `ctmc` 1 (the alias test). No vignette uses `sde`, `tau`, `stochastic`, or `ctmc`.

Running the same SIR (10k people, 100 days, seed 1) under every name gives, for final `n_R` against the ODE's 9393: gillespie +0.4%, tau 0.0%, binomial +1.0%, abm -0.3%, **sde +1.4%**; on an Erlang SEIR at 100k people (`exp_chain.py`), sde at dt=1 was +3.0% with the peak 18% too high (46,800 vs 39,606), the worst of all solvers, and it needed `substeps=4` to come within 1%. The diffusion approximation is a large-population tool, and at large populations the ODE is what people want; the spec's own "Rejected" list says it "fails at small counts, which is where stochasticity matters". `tau` is a performance workaround for `gillespie`, not a different model a user would ask for, and `'stochastic'` already exists to pick between them automatically.

**Why it matters.** Each name is something the user has to understand to choose between, and the spec needs a row in three tables for each. The validity-condition warnings the spec promises per solver (`W1203 LeapConditionMarginal`, `W1204 DiscretizationCoarse`) were never implemented (listed as a departure), so the extra solvers come without the guard rails that justified having them.

**Proposal.** Public solvers: `auto`, `ode`, `gillespie`, `binomial`, `abm`. `gillespie` falls back to tau-leaping above an event budget (the kernel for tau stays; it becomes an implementation detail of `gillespie` with a one-line note, exactly as `'stochastic'` does today), with `exact=True` or `solver='gillespie', approx=False` to forbid the fallback. Then `'stochastic'` becomes a plain alias of `'gillespie'` or goes too. Remove `sde` and its kernel branch (`kernels.py` `solver == 5` paths, `solvers.py:3106-3137`). Keep `'ctmc'` as an alias if it costs nothing, but stop documenting it.

**Cost.** Breaking for `solver='sde'` (one test) and `solver='tau'` (two tests, one vignette README). The spec loses two rows and a paragraph and gains one sentence about the fallback. The exactness guarantee changes shape: today "`gillespie` is never swapped for tau" is absolute; it would become "unless you did not say `approx=False`", which is consistent with how every other approximation in 12-backends is treated. Someone fitting an SDE likelihood would lose a tool; the calibration test that uses `sde` would move to `binomial`.

**Confidence.** High for `sde`; medium for folding `tau` (the counter-argument is that `tau` is a well-known name and a user coming from PyRoss or StochPy will look for it; an alias that prints "tau-leaping is the large-population form of gillespie" would cover them).

### 2. The `'stochastic'` heuristic estimates events without looking at rates or time

**Issue.** `sim.py:347-352` chooses gillespie when `total_pop × n_transitions ≤ 1e6`, else tau. The true event count is (roughly) population × per-person rate × duration. For an endemic SIS, 50,000 people, one year in days:

```text
solver: gillespie (exact; about 100,000 events at this population size)
  chosen=gillespie; actual events=2,400,213; time=3.98s
  tau: time=0.00s
```

The estimate is 24x low, and the printed line is confidently wrong. At 40 years the exact run would take minutes, which is the case the heuristic exists to avoid; conversely a 1e7-person SIR with `stop=10` days would get tau when exact would have been fine.

**Proposal.** Estimate events from the model: sum over transitions of (initial source count × rate × run length) is available at `init()` from `init_fractions()` and `rates()`, and for contact-mediated flows `total_pop` is an upper bound. Even `total_pop × n_transitions × (run length / shortest mean dwell)` would be within a factor of a few. Print the estimate as "about", as now, but make it an estimate.

**Cost.** None for users. A few lines in `init_solver()`.

**Confidence.** High.

### 3. `integrator=` is a developer knob wearing a user-facing name

**Issue.** The spec says `integrator='scipy' | 'jax'` and promises the line `solver: ode (SciPy, 3 compartments)`. The code (`parameters.py:325`) accepts `'rk4'` (default), `'euler'`, `'jax'`, or any `scipy.integrate.solve_ivp` method name; nothing about the integrator is ever printed (`solver: ode (requested)` is the whole line); `'jax'` is not an integrator at all but "the same RK4 steps, run in JAX" (`sim.py:372-374`), and it silently falls back to the kernels with a note when the model has anything JAX cannot take (`sim.py:1840-1843`). `'euler'` exists for conformance against frameworks that take Euler steps (`test_transitions.py:418` is its only use) and comes with its own scaled-outflow rule (`kernels.py:904`, `solvers.py:2381-2386`). Across tests and vignettes, `integrator=` appears 7 times, 3 of them `'jax'`.

**Why it matters.** A user reading the `ss.Sim` signature sees `solver`, `integrator`, `substeps`, `approx`, and has to learn that `integrator` only applies to one of the six solvers, that two of its values are schemes, one is a library, and one is a differentiation backend.

**Proposal.** Remove `integrator=` from `ss.Sim`'s documented surface. Keep `'rk4'` as the only ODE scheme. JAX is selected by what the user asks for (`sim.gradient()`, `calibrate(method='nuts')`, as it already is: `calibration.py:2082`), not by an argument. Keep Euler and the SciPy path reachable via `ss.options` or an engine attribute for conformance tests. `substeps=` stays, since it is meaningful for `binomial` and `abm` too and is the knob users actually reach for when `W102` fires.

**Cost.** Breaking for `integrator='euler'`/`'jax'`/`'LSODA'` callers (tests only). The spec's open question B3 ("where is the line between SciPy and JAX") is answered by "there is no line to choose; JAX is used when gradients are asked for".

**Confidence.** Medium. The counter-argument is that an adaptive integrator (`'LSODA'`) is a genuinely useful escape hatch for a stiff model; I would keep that reachable, just not advertised beside `solver=`.

### 4. The "try it on groups" rule is sound, but what it says when it fails is not what the spec promises

**Issue.** `try_functions()` (`sim.py:995-1023`) builds a probe sim with `solver='ode'`, calls each user function once, and sends the model to agents if anything raises. The mechanism is good and cheap (init 0.013 s vs 0.018 s with a function). But:

- Under `auto`, a function that reads `sim.people.age` in an unstratified model prints `solver: abm (the function "age_rate" does not work on groups (people))`. The word `people` is what survives of the error message after `str(E).split(':', 1)[-1].split('.')[0]` at `sim.py:1012`. Stratifying by age makes the same function run as an ODE with no change, as the spec promises, but the user is not told that.
- A function that reads `sim.people.n_partners` under `auto` prints `(AttributeError: People object has no attribute "n_partners")`, which is the raw exception.
- Under an explicit `solver='ode'`, the same function passes `init()` (functions are not tried when the solver is named: `sim.py:339`) and the run dies at the first step with a bare `AttributeError: People object has no attribute "n_partners"`. The spec (12 §Functions) promises `E804 ... asked for people.n_partners, which is a per-agent attribute and not a stratification ... Options: stratify by it (hiv.stratify(n_partners=[0, 1, 5])), or use solver='abm'`.

**Proposal.** Try functions under an explicit compartmental solver too (it is the same probe), and have the compartmental `People` view raise `E804` with the attribute name and the `stratify` hint when an attribute that is not a dimension is read; `try_functions()` then has a coded error to quote instead of a split string.

**Cost.** None to users; a small amount of code in whatever provides `sim.people` on groups (outside this file).

**Confidence.** High.

### 5. The approximation notes are good and mostly true to the spec; two are missing and one is misleading

What prints today (all verified in `exp_notes.py`):

- Erlang stages under `auto` and `ode`: `recovery has a weibull dwell time, approximated by 2 Erlang stages (mean 7.29139 days kept; std 5.16 instead of 5.28 days)`. Matches the spec's example to the number; `approx=True` hides it, `approx=False` sends `auto` to agents with the transition named, and makes `ode` an `E804` that suggests `stages=`. All as specified. One cosmetic: `mean 7.29139 days` uses `:n` where `:.3g` would match the rest of the line.
- `RandomNet` under `ode`: `RandomNet approximated by homogeneous mixing with 10 contacts per day`, printed *before* the `solver:` line because `Mixing.init()` runs inside `init_solver()` before the print at `sim.py:389`. Reorder so the solver line comes first.
- `W102` (rate × dt) under `binomial` and `abm` only, with the fraction leaving per step and a pointer to `sim.check_dt()`. Good; the ODE and the continuous-time solvers correctly stay quiet.
- `W1201` under `ode` when the model starts with fewer than 10 infected, pointing at `solver='stochastic'` and `ss.parallel`. Good.

Missing or off:

- A distribution on `beta` is silently replaced by its mean under every compartmental solver with no note (`solvers.py:840-841`; see finding 7).
- `MFNet` and other structured networks under `ode` raise `E804 ... has no compartmental approximation yet` rather than the partner-change-rate approximation that is the headline example of 12 §Approximations. This is not in the departures list.
- A lognormal with std 3 and mean 10 becomes **11 stages** and the note says so, but not that 11 stages multiply every stratum: with `age × sex × 2 diseases` a user can reach hundreds of compartments without noticing. `describe()` reports the total; the note could add "(11 compartments per group)".

**Confidence.** High (all observed).

### 6. The chain-binomial "one stage per step" fix works, but its default is documented two different ways

**Issue.** `Compartmental.substeps()` (`solvers.py:2837-2848`) says the chain-binomial solver "takes whole steps, as most chain-binomial models do: it moves people through as many stages of a dwell time as they pass in a step" and returns 1. `parameters.py:326` says the default for `binomial` is "one for each stage of the dwell time with the most stages". The code is right and the docstring is stale.

Measured on the Erlang(4) SEIR at 100k (`exp_chain.py`): `binomial dt=1` is +6.8% in cumulative infections versus the ODE (with `W102` firing twice), `substeps=4` and `dt=0.25` both give +1.9% and agree with each other to the person, which shows the substep machinery is consistent. The residual 6.8% at dt=1 is the ordinary discretization error of a chain binomial, not a stages artifact, so the fix does what it claims.

**Proposal.** Fix the `parameters.py` docstring. Consider making `W102` suggest `substeps=` as well as `check_dt()`, since that is the fix the user will apply.

**Confidence.** High.

### 7. A distribution on `beta` is meaned silently; the spec says agents under `auto`

**Issue.** Spec 12 §What makes a model agent-based: "a distribution on a per-person quantity other than a dwell time (`beta`, `rel_sus = ss.lognorm_ex(...)`) ... [makes the model agent-based under `auto`]"; under an explicit solver, "its mean ... Reported: the mean used". Code (`exp_auto.py`):

```text
--- beta ~ lognormal, auto ---
solver: ode (every part of the model has a compartmental form)   notes=[]   cum_inf=9308.7
--- beta=0.3 fixed, ode ---
solver: ode (requested)                                           notes=[]   cum_inf=9308.7
--- beta ~ lognormal, ode, approx=False ---
solver: ode (requested)   (no error)
```

`Engine.value()` at `solvers.py:840` takes the mean of any `ss.Dist` that is not a dwell time, for every solver, and `agent_reasons()` never looks for one. Under `abm`, `value()` is also used (`Agents.value`, `solvers.py:3438`), so I believe the distribution is meaned there as well rather than sampled per agent (unverified for `abm`: I did not check whether `Agents.eval_for` samples it).

**Why it matters.** Heterogeneity in susceptibility is the single most common reason a modeler reaches for agents; silently discarding it is the kind of approximation the whole of 12-backends is written to prevent.

**Proposal.** Either implement the spec (an `ss.Dist` on a non-dwell parameter is an agent reason under `auto`, a `W`-note "mean used" under explicit compartmental solvers, an error under `approx=False`), or change the spec to say a distribution on a parameter means its mean everywhere and is not a way to express heterogeneity. The first is more useful; the second is simpler and one line in `describe()`.

**Cost.** The spec route is breaking for anyone relying on `beta=ss.lognorm_ex(...)` running as an ODE (no test does).

**Confidence.** High on the discrepancy; medium on which way to resolve it.

### 8. Starsim 3 demographics send a declared model to agents

**Issue.** `ss.Sim(sir, demographics=ss.Births(...))` prints `solver: abm (Births only works on agents)`; likewise `ss.Deaths`, `ss.Pregnancy`, and any function analyzer. The declared equivalents (`ss.birth`, `ss.death` arrows) have a compartmental form. The printed reason is correct and the behavior is what the spec says, but a user adding vital dynamics the Starsim 3 way to a two-line SIR gets a 10,000-agent run instead of three equations and only a one-line note to explain the 20x slowdown.

**Proposal.** Have `Births`/`Deaths` (the crude-rate forms) declare `compartmental = True` and expose themselves as inflow/outflow rates on groups, or have the note say `use ss.birth()/ss.death() for a compartmental form`. The second is a one-line change.

**Confidence.** Medium (depends on how far the demographics reviewer wants to take `compartmental = True`).

## Simpler codebase

### 9. The compartmental step is written three times: Python, Numba, JAX

**Issue.** The same mathematics (rates from the flow list, exposure through the mixing terms, Erlang chains, movers, draws per solver) exists in:

- Python, in `Compartmental`: `derivs()` `solvers.py:2365-2399`, `changes()` 2401-2418, `step_movers()` 2898-2910, the non-kernel branches of `step_ode()` 2920-2949, `draw_targets()` 2954-2973, `step_binomial_once()` 2985-3035, `step_tau()` 3043-3062, `step_gillespie()` 3070-3095, `step_sde_once()` 3116-3136: about 300 lines.
- Numba, in `kernels.py`: `step()` 705-978 (one function with `if solver == 0..5` branches), plus `draw`, `event`, `diffuse`, `draw_chain`, `move`: about 270 lines for the step alone.
- JAX, in `gradients.py`: `Run` 78-375 reimplements the RK4 right-hand side for `ode` only, about 300 lines.

The Python path is taken when `init_program()` cannot build a program: a Python function in any rate (`check()` raises at `solvers.py:2140`), a `dict` rate, or a modifier of `n_contacts` with `where=`. It is then taken for the *whole engine*, every step, every solver. Measured (`exp_pyfallback.py`): a rate given as `lambda sim: 0.1` makes `gillespie` 26x slower (0.87 s vs 0.033 s), and the stochastic Python path draws different numbers from the kernel path (documented as a known issue), so a user who adds a trivial function to a rate gets a different random trajectory *and* a different speed, with no note saying why.

**Proposal.** Delete the Python solvers and make the program always buildable. The step-by-step kernel path already calls `refresh_program()` once per step (`solvers.py:2316-2332`) and `step_tables()` interpolates time-varying values within the step; a user function of the sim that returns a number is exactly a "value that holds for this step" and can go into `p.fixed`/`p.consts` the same way (`value()` already calls it: `solvers.py:842-847`). A function that returns an `ss.Expr` (one that reads `I.n`) is flattened like any other expression. That leaves nothing the Python solvers do that the kernels cannot, and `use_kernels` disappears. The tests that compare kernel to Python (`test_kernels.py`) would compare kernel step-by-step to kernel whole-run, which is the comparison that actually matters.

**Cost.** About 300 lines removed and one attribute (`use_kernels`) gone; `test_kernels.py` loses its reference implementation, so a bug in the kernels would be caught by conformance against the ODE/analytic results rather than by a second implementation. A function that reads compartments mid-substep (within RK4 stages or between Gillespie events) would be evaluated once per step instead; this is a change in semantics for a case the spec does not promise anything about. The JAX copy stays because autodiff needs it, but it should be confined to `ode`.

**Confidence.** Medium-high. I did not try to build it; the risk is a user function that depends on something changed by an intervention mid-step, which `Compartmental.step()` already handles by clearing the cache at `solvers.py:2855`.

### 10. `Crossing` is the right model with the wrong plumbing

**Issue.** When a condition in one disease refers to a state of another (`ss.modify(hiv.rel_trans, multiply=3, where=syph.I)`), `init_crossings()` (`sim.py:733-788`) makes each disease a dimension of the sim, and each engine then holds the full joint array `(own comps) × (every group, including the other diseases' compartments as groups)`. Verified (`exp_cross.py`): HIV×syphilis gives `n_groups=9`, with a 4-stage Erlang 18, with age×sex 108 groups × 6 compartments; results agree with `abm` within 1-4% and `kernel_only=True` holds for `ode`.

Because N engines hold N copies of one state, `Crossing` carries a coherence protocol: `owner`, `version`, `engine.seen`, `claim()`, `refresh()`, `changed()`, `put()`, `joint()` (`solvers.py:351-534`), and `Compartmental` checks `self.cross is not None` in 28 places (`count()`, `set_count()`, `shift()`, `force()`, `state_counts()`, `step()`, `update_results()`, `run_all()`, `init_dims()`, `shared`...). `Crossing.run()` (439-479) then re-implements the step loop so the engines can alternate within the kernels, and `gradients.run_crossing()`/`JointRun` does it a third time for JAX. The only engine that is allowed to age, migrate, or take register events is `cross.names[0]` (`shared`, `solvers.py:1665-1668`), another rule a reader has to learn.

**Proposal.** A crossing is one `Compartmental` engine over the product graph: states are tuples of the member diseases' compartments, transitions are each member's transitions applied along its own axis. The flow list and the `dim_map` machinery already express "move along one dimension and keep the others" (that is what `dim_map` does for strains and `move=`). Each member disease keeps its results by marginalizing the shared `y`. This removes the protocol, the 28 checks, `Crossing.run()`, `JointRun`, and the `shared` rule, and makes `n_comps` and `describe()` honest (today `hiv.n_comps=3` while the engine actually holds 27 numbers).

**Cost.** A real refactor (the member diseases' `engine` attributes would point at one shared object, and `sir.I.n` would read a marginal). The spec gains one paragraph under 12 §How a model is built. Alternatively, if the refactor is too large, at least hoist the shared `y` into the `Crossing` object so engines hold views rather than copies; that alone removes `owner`/`version`/`seen`/`refresh`/`claim`/`changed`.

**Confidence.** Medium on the full proposal (I have not sized the `dim_map` generalization); high that the copy protocol is removable.

### 11. `Engine` is a 1000-line base class that is really a third engine

**Issue.** `Engine` (`solvers.py:556-1522`) holds dimensions, dwell times, modifiers, immunity, strain weights, initial conditions, results layout, outcomes with delay kernels, movement-register sets, and the W-notes. `Compartmental` is 1574 lines, `Agents` 996. Methods defined in two or three of the classes with the same name: `count`, `set_count`, `stock`, `exposure`, `frac`, `local_count`, `n_people`, `force`, `value`, `reduce`, `adjusted_rel`, `protection`, `by_group`, `call_uids`, `record_stocks`, `update_results`, `step`, `step_state`, `init_pre`, `init_post`, `define`. That is the right shape for a template-method base class, but the base also contains concrete machinery that belongs elsewhere: `init_outcomes()`/`step_outcomes()` (1275-1396) are a results-postprocessing module that does not depend on how the model is run, and are duplicated again for whole runs in `derived_outcomes()` (2785-2833); `init_results()` (1206-1273) is results layout; `register_sets()`/`check_short()` are movement-register plumbing.

**Proposal.** Move outcomes to `results.py` (or `transitions.py` beside `ss.outcome`) as a function of (flows by step, by group) that serves both step-by-step and whole-run paths, which removes `step_outcomes()`/`derived_outcomes()` duplication (about 110 lines). Move `init_results()` next to it. `Engine` then shrinks to what both engines genuinely share: dimensions, dwell times, modifiers, initial fractions.

**Cost.** Mechanical; no user-visible change.

**Confidence.** Medium-high.

### 12. `Mixing` carries two representations of the same contact matrix

**Issue.** `Mixing.terms` is used by the compartmental engines and `Mixing.pool_terms` by the agent engine "in a model with places" (`init_pools()`, `solvers.py:286-310`), built by the same `age_matrix()`/`dim_matrices()`/`within_places()` calls. `Agents.infect()` (3943-4057, 115 lines) then has three branches for routes: networks, pools that are in `pool_terms`, and pools that are not, plus a fourth for "any other route". `init_space()` is called up to three times (`__init__`, `init()`, `init_pools()`) because a contact matrix can add an age stratification after the fact (`solvers.py:195-199`).

**Proposal.** One `terms` list built once after stratifications are final (make the "contact matrix adds age groups" step part of `init_strats()` rather than a side effect inside `Mixing`), used by both engines. `infect()` then has two branches: per-edge (networks) and per-group hazard (everything else).

**Cost.** Moderate; touches `sim.init_strats()` ordering.

**Confidence.** Medium.

### 13. Smaller things

- `make_engine()` (`solvers.py:39-43`) is three lines and the only caller is `Disease.init_pre()`; `Compartmental if sim.solver != 'abm' else Agents` inline would do. Low value either way.
- `Engine.integer` (`solvers.py:1867-1869`) and `step_outcomes()`'s own `integer = self.sim.solver in ['abm', 'gillespie', 'tau', 'binomial']` (1346) are two definitions of "keeps whole numbers"; the second should use the first plus `isinstance(self, Agents)`.
- `Compartmental.step()` dispatches on the solver string (`solvers.py:2864-2868`) and then every branch but the Python fallback calls the same `ss.kernels.one_step(..., ss.kernels.solvers[name], ...)`; with finding 9 the dispatch collapses to one call.
- `Bunch = SimpleNamespace` is imported with a comment that `objdict` is too slow inside the RHS (`solvers.py:19`); with finding 9 the RHS is never in Python and the comment goes away.

## Spec and implementation

Beyond the departures `implementation.md` already lists (RK4 kernels not SciPy as default; `W1203`/`W1204` unused; quarter-step holding instead of thinning; `compartmental = True` on code modules):

- **Structured networks under `ode`.** Spec: `MFNet approximated by partner-change rates between sex/risk groups (4 groups)`, with a whole table row and a paragraph on how the rates are obtained. Code: `E804 ... MFNet has no compartmental approximation yet` (`solvers.py:152-153`); only `RandomExactNet`, `StaticNet`, `PartnerNet` with `n_contacts` get the mean-contacts approximation. Not in the departures list. Open question B5 is thus answered by "not built"; the spec should say so or the row should go.
- **Distribution on `beta`/`rel_sus`.** Spec: agents under `auto`, mean-with-note under explicit solvers, error under `approx=False`. Code: mean, silently, everywhere, including under `abm` (`Agents.value` → `Engine.value`, `solvers.py:3438-3441` → 840). See finding 7.
- **The printed solver line.** Spec: `solver: ode (SciPy, 3 compartments)`. Code: `solver: ode (requested)` or `(every part of the model has a compartmental form)`; neither the integrator nor the compartment count is printed (`describe()` has both).
- **`hiv.solvers()` on a disease.** Spec promises it "for one disease"; only `sim.solvers()` exists (`sim.py:1165`). The table's `? (tried when run)` for functions is as specified.
- **`solver=` on `sim.run()`.** Spec: "the second wins". Code: accepted only before `init()`, with a clear error pointing to `sim.new(solver=)` (`sim.py:1817-1821`). The code's rule is the better one; update the spec.
- **`W1201 SmallCounts` for `sde`.** Spec table: `sde` "loses accuracy when any compartment is small", best-practices says warn below ~20. Code: only `n_start < 10` and "infected falls below 1 then rises" (`note_chance()`, `solvers.py:1492-1509`), shared with `ode`. Fine if `sde` goes (finding 1).
- **`substeps` default for `binomial`.** `parameters.py:326` and `solvers.py:2841-2843` disagree (finding 6); the code is right.
- **Agent-only functions under an explicit compartmental solver** are not tried at `init()` and fail with a raw `AttributeError` at step 1 (finding 4), where the spec promises `E804` with a `stratify` hint "when it is first called".
- **Open question B3 ("should `auto` ever choose a stochastic solver?")** is answered no, with `W1201` as the "printed hint" the question suggested. Good answer; the spec can close it.
- **Open question B3 ("line between SciPy and JAX")** is answered: there is no automatic line; JAX only when gradients are asked for or `integrator='jax'`. Close it (and see finding 3).
- **Under-specified.** 12 says nothing about which solvers run step by step in the loop and which take the whole run in one kernel call (`kernel_only()`, `sim.py:1836-1845`), yet this decides whether interventions written as code see the model between steps, and the `tabulated`/`try_groups` protocol that deliveries and modifiers implement to stay eligible is a real interface that users writing an intervention class will hit. It needs a paragraph in 12 §How a model is built or in 10.
- **Internally consistent otherwise.** The approximation policy (note by default, `approx=True` hides, `approx=False` errors, `auto` approximates only dwell times) is implemented exactly as written for dwell times and random networks, and the printed lines match the spec's examples closely.

## Bugs

### B1. Agent-only attribute under a requested compartmental solver gives a raw `AttributeError` at step 1

```python
def partners(sim): return 0.1*sim.people.n_partners
ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=partners), init_prev=0.01),
       solver='ode', total_pop=10_000, unit='day', stop=100).run()
```

```text
ERROR at run: AttributeError: People object has no attribute "n_partners"
```

`init()` succeeds (`solver: ode (requested)`). Expected `E804` at `init()` or first call (finding 4).

### B2. `'stochastic'` event estimate off by 24x

See finding 2; `exp_auto.py` section 3.

### B3. `parameters.py:326` documents a `binomial` substeps default the code does not implement

See finding 6.

## Keep

- **`solver='auto'` and the printed one-line reason.** Every choice I provoked was explained in one line that named the ingredient (`RandomNet is an individual-level network`, `n_agents was given`, `"recovery" has a lognorm_ex dwell time, and approx=False`, `the function "uids_fn" takes uids`). This is the single best user-facing feature in the area; do not let it grow into a table.
- **One flat flow list stepped by every solver** (`build_layout()`, `solvers.py:1733-1816`). The expansion of states into (track, stage) compartments and transitions into flows with `targets`/`credits`/`record` is clear, declaration-order independent, and the reason five solvers can share one `Program`. Any refactor from findings 9-11 should keep this as the center.
- **Erlang stages as the one approximation `auto` makes, with the std printed next to the one asked for.** The note is exactly informative enough, and `stages=` gives the user control. The `approx=` triad behaves as specified.
- **`W102` with the fraction leaving per step and `sim.check_dt()`.** Concrete, actionable, quiet under the solvers where it does not apply.
- **The chain-binomial multi-stage draw** (`chains`, `draw_chain`): `binomial dt=1 substeps=4` and `binomial dt=0.25` agree to the person, so the substep machinery is trustworthy.
- **`try_functions()` by probing rather than analyzing.** Cheap (5 ms), honest, and the spec's "Rejected: analyzing user functions in advance" is vindicated. Only its error text needs work.
- **States that are not people as plain counts under every solver**, including RK4 inside an agent step (`step_counts()`, `solvers.py:3745-3796`). Simple and matches the spec table exactly.
- **Agreement across solvers.** A plain SIR agrees within 1.5% under all nine names, and a 100k Erlang SEIR within 2% for everything but `sde` and whole-step `binomial`. The conformance goal in 12 §Agreement is met.

## Outside my area

- `ss.modify(..., where='syph.I')` with a *string* condition fails with `ValueError: could not convert string to float: 'syph.I'` (compartmental) or `The condition syph.I returned an array of shape (); expected a boolean per agent` (agents). Either strings should be parsed as state references, as `'hiv.rel_trans'` is for the target, or the error should say "give the state object (`syph.I`), not its name". (`conditions.py`.)
- `ss.Births(birth_rate=20)` in a sim in days warns that the number is read in years; the warning is good, but a declared model's bare numbers are in the sim's unit, so two unit conventions coexist in one sim (listed as a departure; worth the demographics reviewer's attention).
- The approximation note for `RandomNet` prints before the `solver:` line (`Mixing.init()` runs inside `init_solver()` before the print at `sim.py:389`); the fix is ordering in `sim.py`.
- `kernels.py` `step()` is one 270-line function with `solver == 0..5` branches and `stages = 4 if solver == 0 else 1`; whoever reviews the kernels should weigh finding 1 (fewer solvers) against the cost of each branch there.
