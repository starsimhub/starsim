# Solvers, execution, and paradigm conversion

Reviewer area: `starsim/solvers.py`, `starsim/kernels.py`, `starsim/loop.py`, the v4 parts of `starsim/sim.py` (`init_solver`, `agent_reasons`, `try_functions`, `solvers()`, `run`, `kernel_only`, `restart`), against `design/10-execution.md`, `design/12-backends.md`, and the best-practices pages on compartmental, stochastic compartmental, agent-based, and paradigm conversion. Every bug below was reproduced with `/software/conda/envs/v4/bin/python`; the scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/solvers/` (`s1.py` to `s26.py`).

## Summary

The core is good. The compiled `Program` (a flat list of flows plus a small stack machine for rates) is the right intermediate form: one Numba kernel runs every model, a whole run is taken in one call when nothing in Python needs to see the steps, and the stepped and whole-run paths give identical results (checked for all five compartmental solvers, with and without a seasonal rate, `s3.py`, `s23.py`). On a plain SIR the six solvers agree as they should (peaks 3,036 to 3,212 of 10,000, `s1.py`), agents and chain-binomial agree with each other to within noise at every `dt` (`s4.py`), a `RandomNet` approximated as an ODE agrees with agents (`s8.py`), and two coupled diseases agree across ODE, binomial, and agents within 2% (`s14.py`). The printed reason for the solver is a real usability win.

The problems are of three kinds. First, real bugs in the less-travelled solvers and in the selection logic: the SDE solver's result depends on the order transitions are declared (100% of people leave by the first-declared exit when the step is coarse), `solver='stochastic'` underestimates the event count by 50x and picks a solver that is 2,000x slower, and `solvers()` contradicts what `'auto'` actually chose. Second, `'auto'` sends ordinary models to agents for reasons the spec says it should not (a class with a `step()` that calls `super()`, `ss.Births()`, a function analyzer), while silently collapsing a per-person distribution to its mean, which the spec says should send it to agents. Third, the architecture is larger than it needs to be: each compartmental solver exists twice (Python and Numba, plus RK4 a third time in JAX), there are three random number systems, compartmental multi-disease runs use one engine per disease plus a `Crossing` to synchronize them, and the v3 loop and the v4 engines disagree about what `sim.ti` means and about where in the step declared transitions run. The biggest simplification available is to make the compartmental engine one per sim, running one `Program`, with no Python fallback.

## Findings

### 1. The SDE solver resolves competing exits in declaration order

- **Type:** bug
- **Evidence:** `kernels.diffuse()` (`kernels.py:636`) and the Python fallback `Compartmental.step_sde_once()` (`solvers.py:3116`) draw each flow independently and then clip it to what is left in the source, flow by flow: `amount = min(amount, new[s, g]); new[s, g] -= amount`. With the default `substeps=1` for `'sde'` (`Compartmental.substeps()`, `solvers.py:2837`), any state whose total exit rate times `dt` approaches 1 is emptied by whichever transition was declared first. In 1,000,000 people with three exits from `I`, each at the same rate (`s20.py`):

```text
rate 0.5/day x3 exits ode       [0.3333 0.3333 0.3333]
rate 0.5/day x3 exits sde       [0.5004 0.4996 0.    ]
rate 1.0/day x3 exits sde       [1. 0. 0.]
rate 1.0/day x3 exits tau       [0.3338 0.3337 0.3324]
rate 1.0/day x3 exits binomial  [0.3333 0.3329 0.3338]
```

  No note is printed (`s21.py` prints only `solver: sde (requested)`), although `W102 LargeStep` is documented as covering the solvers that take whole steps. `10-execution.md` §Competing hazards says this MUST be one draw per source state.
- **Proposal:** in `diffuse()`, work per source compartment as `draw()` does for the binomial solver: total the exits, draw (or clip) the total once, and split it in proportion to the hazards. Also give `'sde'` the same automatic substeps as the ODE (no compartment turns over by more than a quarter in one substep), since it is an Euler–Maruyama scheme and has the same stability limit as Euler. Add the three-exit test above to `test_transitions.py` for every solver.
- **Cost:** a few lines in each of two places (one if finding 11 is taken). No user-visible change except correct results.

### 2. `solver='stochastic'` estimates the number of events from the population only, and can pick a solver 2,000 times slower

- **Type:** bug / design
- **Evidence:** `sim.py:346`: `n_events = total_pop*sum(len(mod.graph.transitions) ...)`, Gillespie if that is at most 1e6. It ignores the length of the run and every recurrent flow. An SIS in 400,000 people for two years (`s10.py`):

```text
solver: gillespie (exact; about 800,000 events at this population size)
stochastic gillespie 74.5s actual events 38064378
tau tau 0.0s actual events 38018817
```

  The printed estimate is off by a factor of 48 and the run takes 74 s instead of 0.03 s. The heuristic also never considers small counts, which is what `stochastic-compartmental.md` says it should lean on ("bias toward exactness when compartment counts are small").
- **Proposal:** estimate the events by running the deterministic ODE first, which the kernels do in milliseconds, and summing its flows: that is the expected number of Gillespie events, exact for the mean, and it includes run length, recurrence, births, and stages. Choose Gillespie below a budget (say 1e6 events, or an estimated run time of a few seconds), tau above it, and print the real estimate. Optionally print the minimum count of the infected compartments along the ODE trajectory, since that is the quantity that decides whether tau is safe.
- **Cost:** one extra ODE run at `init()` for `'stochastic'` only (milliseconds; reuses the program).

### 3. Modules in the same step see different values of `sim.ti`

- **Type:** design / bug-prone
- **Evidence:** `ss.solvers.interval()` (`solvers.py:537`) decrements `sim.t.ti` while a declared disease or an `ss.Delivery` steps, and `Loop._at_start_funcs()` (`loop.py:360`) wraps only those. Everything else in the same step (connectors, networks, v3 interventions, function interventions, analyzers) sees the undecremented index. Recording `sim.t.ti` from each kind of code in a 5-day run (`s25.py`):

```text
rate [0, 1, 2, 3, 4]       # a function in a declared rate
fn   [0, 1, 2, 3, 4]       # a function passed to ss.Disease(...)
intv [1, 2, 3, 4, 5]       # a function intervention
ana  [0, 1, 2, 3, 4, 5]    # an analyzer (also called at the start)
```

  So `def intv(sim): if sim.ti == 10: ...` and `ss.vaccinate(..., start=ss.days(10))` happen one step apart, and a user who writes a v3-style intervention to compare with a v4 delivery gets an off-by-one that no message explains. This is the clearest place where the v3 loop and the v4 engine are two systems bolted together: the v4 convention ("the step at time point ti goes from ti-1 to ti") is applied by swapping the clock around some calls rather than by the loop.
- **Proposal:** make the convention the loop's, for every module: during the step from time point k-1 to k, `sim.ti == k-1` and `sim.now` is its start for every function in the step; results are written at k by `update_results()`. Then `interval()` and `_at_start_funcs()` go away. Document it once in `10-execution.md` §The order of calls.
- **Cost:** breaking for v3 interventions and connectors that test `sim.ti` (they would fire one step earlier). The v3 test suite would need its expected values shifted where it depends on that.

### 4. Declared transitions run at different points of the step under agents and compartments

- **Type:** design
- **Evidence:** `Agents.step_state()` (`solvers.py:3620`) runs every transition that is not contact-mediated at row 5 and `Agents.step()` runs transmission at row 9; `Compartmental.step_state()` is empty (`solvers.py:2850`) and all flows run at row 9. So a delivery (row 8) sees a population before progression under compartments and after it under agents: known issue in `implementation.md` ("TPT on the first step reached 7% fewer people"). The spec itself concedes an O(dt) row-5/row-9 effect for a state with both a spontaneous and a contact-mediated exit (`10-execution.md` §Competing hazards, second bullet), which is exactly what a joint draw would remove.
- **Proposal:** run all of a declared disease's transitions together in `step()` (row 9) under both engines, from the state at that moment, with one competing-hazards draw per state that includes the force of infection. Then agents and every compartmental solver agree on what an intervention sees, the row-5/row-9 caveat disappears from the spec, and "competing hazards are resolved jointly" becomes true without exceptions. `step_state()` remains for v3 subclasses.
- **Cost:** births in a declared model still need to happen before the new agents are initialized (they can stay at row 4). A v3-style `step_state()` override on a declared subclass would now run before all declared transitions rather than in place of the progressions; that is simpler to explain. Results of agent runs change by O(dt).

### 5. `'auto'` sends ordinary models to agents for reasons the spec says it should not

- **Type:** design / doc-code mismatch
- **Evidence:** `Sim.agent_reasons()` (`sim.py:963`); `s6.py`, `s7.py`:

```text
subclass with step()                          -> abm | "mysir" defines its own step()
subclass with step() compartmental=True       -> abm | "mysir2" defines its own step()
declared + births                             -> abm | Births only works on agents
declared + v3 Deaths                          -> abm | Deaths only works on agents
declared + analyzer fn (lambda sim: None)     -> abm | Analyzer only works on agents
declared + intervention fn (lambda sim: None) -> abm | Intervention only works on agents
```

  The class form in `principles.md` (`class SIR(ss.Disease): ... def step(self): super().step(); self.log_something_bespoke()`) is the form the spec uses to show that subclassing "adds behavior, not vocabulary", and adding that method silently changes the paradigm. The same class runs fine as an ODE when `solver='ode'` is given (`s12.py`: identical peak to the plain declared model, `step()` called 100 times). The `compartmental = True` attribute described in the departures is checked only for undeclared modules (`sim.py:979`), so it cannot rescue a declared subclass. `12-backends.md` §What makes a model agent-based says custom code goes to agents only if it "does not work on groups (see Functions)", which implies trying it, as is done for rate functions; it is not tried. `ss.Births()`/`ss.Deaths()` are the most common v3 demographics, and `solver='ode'` with either is `E804`.
- **Proposal:** (a) honor `compartmental = True` on declared subclasses (a one-line fix). (b) Extend `try_functions()` to step overrides, function interventions, and function analyzers: run one step of the probe compartmental sim it already builds, and go to agents only if that raises. (c) Give `ss.Births` and `ss.Deaths` a compartmental form (an inflow at the birth rate, an outflow from every people state at the death rate), or have them translate themselves to `ss.birth`/`ss.death` on the declared diseases.
- **Cost:** (b) runs user code once more at `init()` (the probe already exists, so the cost is one step). A step override that silently does nothing useful on groups (writes to an attribute nobody reads) would then run as an ODE; that is the "try it" rule as written.

### 6. A per-person distribution on `beta` is silently replaced by its mean, even with `approx=False`

- **Type:** doc-code mismatch / bug
- **Evidence:** `beta=ss.lognorm_ex(0.3, 0.1)` (`s7.py`): `'auto'` chooses ODE with the reason "every part of the model has a compartmental form", prints no note, and `approx=False` still runs it as an ODE. `12-backends.md` lists "a distribution on a per-person quantity other than a dwell time (`beta`, `rel_sus = ...`)" among the things that make `'auto'` choose agents, and §The standard approximations says an explicit compartmental solver uses the mean *with a note*. Principle 8 ("say what was approximated") is violated in the quietest possible way.
- **Proposal:** add the check to `agent_reasons()` (a `Dist` on any `p.slot` other than a dwell time), and a note plus an `approx=False` error in the compartmental path. Alternatively, and I think better for simplicity, decide that a `Dist` in a rate slot means "this varies between people" and so is agents, full stop, and say so in the error for an explicit compartmental solver ("bin it with `stratify(beta=...)` to run it as compartments").
- **Cost:** none to speak of; a model that ran as a fast ODE now runs as agents, which is what the spec says.

### 7. `sim.solvers()` contradicts what the sim does

- **Type:** bug
- **Evidence:** `Sim.solvers()` (`sim.py:1165`) re-derives its answers instead of reading the decisions `init_solver()` made, and gets three of them wrong (`s11.py`, `s12.py`):
    - With `beta=lambda sim: 0.3*(sim.people.age > 50)` and no age stratification, the sim chose `abm` ("does not work on groups"), and `solvers()` then prints `auto would choose: ode (every part of the model has a compartmental form)`, because it calls `agent_reasons()` but not `try_functions()`, and still marks the function `? (tried when run)` although it was tried at `init()`.
    - With `ss.MFNet()` it prints `approx (contact rates)` under `ode` and `stochastic`; running either raises `E804 ... MFNet has no compartmental approximation yet`.
    - With a declared subclass that overrides `step()`, the table says `(every part of the model runs under every solver)` and the last line says `auto would choose: abm`.
- **Proposal:** have `init_solver()` produce one list of capability records (ingredient, per-solver verdict, reason) and make `agent_reasons()`, `solvers()`, `describe()`, and the `E804` messages all read from it. Then they cannot disagree, and the per-solver verdicts are computed once.
- **Cost:** a modest refactor of three functions; it removes code.

### 8. An explicit solver is silently ignored for a model written as code

- **Type:** bug
- **Evidence:** `init_solver()` returns early with `self.solver = 'abm'` when there are no declared modules (`sim.py:329`). `ss.Sim(diseases=ss.SIR(), networks='random', solver='gillespie').init()` prints only `Initializing sim with 10000 agents` and runs agents (`s26.py`). The user asked for something specific and got something else without a word, which is the one thing principle 3 forbids.
- **Proposal:** raise `E804` ("`ss.SIR` is written as code and runs only as agents; use a declared disease, e.g. `ss.Disease(infection=..., recovery=...)`, to choose a solver") when `pars.solver` is neither `'auto'` nor `'abm'` and nothing is declared.
- **Cost:** none.

### 9. `MFNet` and other structured networks have no compartmental form, but the spec's headline example uses one

- **Type:** doc-code mismatch
- **Evidence:** `12-backends.md` §Approximations shows `ss.Sim(hiv, networks=ss.MFNet(), solver='ode')` approximated "by partner-change rates between sex/risk groups (4 groups)", and `paradigm-conversion.md` §Sexually transmitted infections as an ODE argues for it. The code raises `E804 CapabilityError: cannot run with solver="ode": MFNet has no compartmental approximation yet` (`s7.py`). `implementation.md` §06 marks "Networks, as agents and (approximated, with a note) as compartments" as done, which is true only for `RandomNet`/`StaticNet`-style networks.
- **Proposal:** either implement the partner-change approximation (sample the network once at `init()` for a population, count partnerships formed per unit time between sex and risk groups, as the spec describes) or change the spec example and the checklist to say that structured networks are agents-only for now. Given principle 9 and the vocabulary budget, I would do the latter until a vignette needs it, and fix `solvers()` (finding 7) to say `no` rather than `approx`.
- **Cost:** doc edit now; the approximation is a few hundred lines when needed.

### 10. Three random number systems, one of them process-global

- **Type:** bug / design
- **Evidence:** agents draw from `ss.Dist` objects (CRN); the compartmental Python path draws from a per-engine `np.random.Generator` (`Engine.make_rng()`, `solvers.py:585`); the Numba kernels draw from Numba's process-global `np.random`, seeded once in `init_program()` (`solvers.py:2277`) and in `kernels.run()`. Because the kernel state is global, two stochastic compartmental sims stepped in the same process change each other's results (`s9.py`):

```text
stepped alone == whole run: True
alone == interleaved: False
```

  (`a.run(until=...); b.run(until=...); a.run()` gives `a` a different trajectory from running it alone.) The same applies to anything that steps two sims side by side in one process (a particle filter over sims, a notebook comparing scenarios step by step, and, within one sim, two declared diseases, which each reseed the same global state when their program is built and then share one stream). The known issue "a stochastic compartmental run gives different numbers with `engine.use_kernels = False`" is a direct consequence of having two generators.
- **Proposal:** pass the engine's `np.random.Generator` into the kernels; Numba has supported `Generator` arguments (`binomial`, `poisson`, `standard_normal`, `exponential`, `random`) since 0.56, and `deferred.md` already measured that a Numba `Generator` binomial costs the same 47 ns. Then each engine owns one stream, interleaving cannot matter, and (with finding 11) there is one compartmental generator, not two.
- **Cost:** the kernel signatures gain an argument; the stream of numbers changes once, so stochastic regression baselines move.

### 11. Every compartmental solver is written twice, and RK4 three times

- **Type:** code simplicity / performance
- **Evidence:** `Compartmental.step_ode/step_binomial_once/step_tau/step_gillespie/step_sde_once/derivs/draw_targets/step_movers` (`solvers.py:2365-2420`, `2898-3138`, about 350 lines) re-implement in NumPy what `kernels.step()` does, and are used only when `program is False`, i.e. when some rate is "a function of the sim". `gradients.py` implements RK4 over the same `Program` a third time in JAX. The two compartmental paths already disagree in two ways: different random numbers (finding 10), and the Python fallback is much slower (Gillespie 0.76 s against 0.04 s for 10,000 people when `beta` is merely `def beta(sim): return 0.3`, `s5.py`). Any bug has to be fixed in both (finding 1 is in both).
- **Proposal:** delete the Python solvers. A leaf of a rate that is a Python function of the sim becomes a per-step value in the program's `consts` table, evaluated by Python once at the start of each step (and, for RK4, at the start, middle, and end, as `ss.seasonal` already is). The kernel then always runs. A function that reads counts (`sim.diseases.sir.I.n`) would be held for the step instead of re-evaluated at each RK4 stage; that is O(dt), the same treatment `ss.change()` already gets, and could be noted once (`W1205`-style). The declared `np.minimum`/`np.exp`-of-counts forms, which already compile, remain the exact way to write count-dependent rates.
- **Cost:** a count-dependent Python function in a rate becomes first-order in time within a step instead of fourth-order. Removes about 350 lines, one RNG, the `use_kernels` switch, and the slow path.

### 12. One compartmental engine per disease, plus a `Crossing` to keep them in step, where one engine per sim would do

- **Type:** design / code simplicity
- **Evidence:** `make_engine()` makes one engine per declared module, which is natural for agents and unnatural for an ODE, which is one system. Coupled diseases therefore need `Crossing` (`solvers.py:351-535`): "each engine holds everyone, and takes the people over from the engine that changed them last" (`claim()`, `changed()`, `put()`, `joint()`), extra `record_stocks()` calls in `update_results()`, a separate `Crossing.run()` for whole runs that is "for now a loop in Python over the steps, with one kernel call a disease a step" (`implementation.md` line 209), and a separate JAX path (`gradients.joint_runner`, `run_crossing`). `kernel_reason()` reports it as a reason the sim cannot be run in one call (`s14.py`). The open items in `implementation.md` §06 ("an expression in one disease's rates that counts another's states", "the kernels taking the whole run in one call") are both consequences of the split.
- **Proposal:** make the compartmental engine one per sim. It builds one `Program` whose compartments are the union over diseases (the product only where diseases are crossed, a block-diagonal layout where they are not), and each `ss.Disease` keeps a view onto its slice for results and for `sir.I.n`. Then a multi-disease ODE is one kernel call per run, cross-disease counts in rates are just `COUNT` ops on another slice, and `Crossing`'s synchronization disappears. Agents keep one engine per disease.
- **Cost:** a substantial refactor of `Compartmental` init (layout, results by disease) and of `gradients.py`; the payoff is the largest single simplification available in this area, and it makes the multi-disease models (TB with HIV) restartable and fittable by NUTS without the Python loop.

### 13. `solvers.py` should be split along the seams it already has

- **Type:** code simplicity
- **Evidence:** 4,136 lines in one file: `Mixing` (lines 46-310), `CountState` and `Crossing` (313-535), `Engine` (556-1520), `StockCtx` (1524-1565), `Compartmental` (1567-3138), `Agents` (3141-4136). The classes are already well separated; only the file is not.
- **Proposal:** a `starsim/solvers/` package: `mixing.py`, `engine.py` (base and `StockCtx`), `compartmental.py`, `agents.py`, `crossing.py` (or nothing, after finding 12), and `__init__.py` with `make_engine`, `solver_names`, and `interval`. `ss.solvers.X` keeps working. Keep `kernels.py` beside `compartmental.py`.
- **Cost:** mechanical; the only risk is `ss.solvers.<private>` references in other modules and tests, which a grep finds.

### 14. The solver surface: six names, an alias, a meta-solver, and an `integrator=` that also selects a backend

- **Type:** API
- **Evidence:** `solver_names = ['auto', 'stochastic', 'abm', 'ode', 'gillespie', 'tau', 'binomial', 'sde']` plus `ctmc` (`solvers.py:22`); `integrator=` takes `'rk4'`, `'euler'`, any `solve_ivp` method, or `'jax'`, which is not an integrator ("`integrator='jax'` takes the same RK4 steps", `implementation.md` departures) and sets `sim.jax`; `substeps=`; `approx=`. `integrator=` is silently ignored for `abm` and the stochastic solvers, and an unknown one (`integrator="bogus"`) passes `init()` and fails at the first step with SciPy's message rather than a coded error (`s22.py`).
- **Proposal:** keep `solver=` as is for experts but teach four values (`'auto'`, `'ode'`, `'stochastic'`, `'abm'`), with `'gillespie'`, `'tau'`, `'binomial'`, `'sde'` documented as the methods `'stochastic'` chooses between. Drop the `'ctmc'` alias (principle 9: an alias is vocabulary the reader must hold). Make JAX a backend detail chosen automatically where it pays (gradients, NUTS, parameter sweeps), not an `integrator=`; if a switch is needed, `ss.options.backend`. Validate `integrator=` at `init()` and reject it (or note it) with a non-ODE solver.

```python
# Before
ss.Sim(sir, solver='ode', integrator='jax')       # "jax" is a backend, not a scheme
ss.Sim(sir, solver='ctmc')
# After
ss.Sim(sir, solver='ode')                         # JAX used automatically by sim.gradient() and calibrate()
ss.Sim(sir, solver='gillespie')
```
- **Cost:** removes one alias and one integrator value; `ss.parallel(..., integrator='jax')` sweeps would switch automatically or via the option.

### 15. The sim's results differ by solver

- **Type:** API / design
- **Evidence:** the same SIR (`s13.py`) has sim-level results `['timevec', 'sir', 'n_alive', 'n_female', 'new_deaths', 'new_emigrants', 'cum_deaths']` as agents and `['timevec', 'sir']` as an ODE (`init_solver()` sets `pars.people_results = False`). `len(sim.people)` is 1 for an unstratified ODE (one "agent" per group, `pars.n_agents = self.n_groups`), and `people.age` and `people.female` are deleted when they are not stratifications (`sim.py:281`). So code that reads `sim.results.n_alive`, or plots it, breaks when the solver changes, which undercuts "the solver is a free run-time choice".
- **Proposal:** compute `n_alive`, `new_deaths`, `cum_deaths` (and `n_female` when sex is a dimension) from the compartments under every compartmental solver, so the sim-level result schema is the same under every solver. In a compartmental run, make `sim.people` a small group-view object whose `len()` is the population and whose missing attributes raise a coded error naming the stratification to add, rather than a one-row `People` with attributes deleted.
- **Cost:** small for results; the `People` view is a moderate change but makes the "one row per group" trick explicit instead of leaky.

### 16. What is printed at `run()`

- **Type:** API / doc-code mismatch (minor, but this is the user's main window into the solver)
- **Evidence:**
    - `sim.run(verbose=0)` does not hide the `solver:` line or the notes, because they are printed in `init()` from `pars.verbose` before `run()` applies its argument (`s10.py`, `s24.py`). The spec says `verbose=0` hides them.
    - An approximation note can be printed before the line it qualifies: with `networks=ss.RandomNet(n_contacts=10), solver='ode'` the output is `note: RandomNet approximated by ...` and then `solver: ode (requested)` (`s2.py`), because `Mixing.init()` notes before the solver line is printed.
    - Numbers are not rounded: `mean 7.29139 days kept; std 5.16 instead of 5.28 days`.
    - As agents with `total_pop=1e6`, the scale factor of 100 is not reported (`s24.py`); the spec's table says it is. The spec shows `n_agents: 10000 (default)`; the code prints v3's `Initializing sim with 10000 agents`.
    - As agents, the v3 progress bar prints by default while compartmental runs print nothing, so the same model is quiet or noisy depending on the solver.
- **Proposal:** collect the solver line and its notes into `sim.notes` during `init()` and print them together at the start of `run()` (honoring `run(verbose=)`), in a fixed order (solver, unit, then notes), with numbers through `sc.sigfig(..., 3)`. Print `n_agents: 10000 (default; each stands for 100 people)` for agents and drop the v3 line and default progress bar for declared models.
- **Cost:** none.

### 17. `kernel_reason()` is wrong when the sim is kernel-only and after any run

- **Type:** bug (minor)
- **Evidence:** `kernel_reason()` (`sim.py:1866`) returns `'unknown'` for a sim that is kernel-only (it should return `None`), and after a stepped run it returns `'it has inserted functions, diagnostics, or verbose output'` because `self.loop.index` is nonzero, whatever the real reason was (`s5.py`, `s14.py`: a sim whose rate is a Python function reports "inserted functions" after the run and "a rate in sir is a function of the sim" only before it). `restart()` uses it in its error message.
- **Proposal:** return `None` when `kernel_only()` is true, and check `loop.index` only in `restart()`, not as a reason. Better, merge `kernel_reason()` and `kernel_only()` into one function that returns the reason or `None` (they duplicate the same conditions in the same order, `sim.py:1866-1895`).
- **Cost:** none; removes a duplicated function.

### 18. Tau-leaping's leap condition is per flow, and its draws are clipped in declaration order

- **Type:** bug (minor) / doc-code mismatch
- **Evidence:** the leap is `0.03/max(fastest, ...)` where `fastest` is the largest single flow rate (`kernels.py`, `step()`: "The fastest rate at which any compartment is being left" is computed per flow), so a compartment with `k` exits at that rate loses up to `0.03k` per leap, not 3% as `implementation.md` says ("no compartment loses more than 3% of its people in one"). Within a leap the per-flow Poisson draws are clipped sequentially (`draw()`, the `tau` branch), which gives the first-declared exit an edge when counts are small. Over 40,000 runs of three people with five equal exits (`s19.py`): tau `[0.2023 0.2016 0.2019 0.1964 0.1978]` (standard error 0.0012), against binomial `[0.1993 0.1995 0.2009 0.2003 0.2001]`. The bias is small, but it is the order dependence the spec says MUST NOT exist.
- **Proposal:** compute `fastest` as the largest total exit rate per compartment, and within a leap draw the total leaving each compartment, clip that once, and split it multinomially, as `draw()` already does for the binomial solver. The two code paths in `draw()` then become one.
- **Cost:** a few lines; slightly shorter leaps for states with many exits.

### 19. Small things in execution that change behavior or cost

- **Type:** performance / API (minor)
- **Evidence and proposals:**
    - `kernel_only()` returns `False` when `verbose >= 1` (`sim.py:1888`), so turning on printing switches from one kernel call to the Python loop (same results, verified in `s3.py`, but a different cost profile). Printing progress should not choose the execution path; print a single line after a whole run instead.
    - `integrator='euler'` uses RK4's "a quarter of a compartment per substep" rule for its default substeps, giving a 4% higher peak on a plain SIR (`s22.py`: 3,159 against 3,036). Either document that Euler is deliberately coarse (it is there to match Atomica-style models) or give it a tighter default.
    - Building the same states into a second `ss.Disease` in a loop raises `E206 ... used by transitions in two modules, "m" and "m"` (`s18.py`, first version); the message names the same module twice and does not say "build the states again for each disease".

## What is good and should be kept

- **The `Program` intermediate form and the single kernel.** One compiled function runs every model, without code generation per model, and the stack machine for rates keeps the expression language small. It is also what makes the JAX gradient path possible from the same data. Keep it as the one representation of a compartmental model (finding 11 makes it the only one).
- **Whole runs taken automatically, with identical results to stepped runs.** `sim.run()` choosing one kernel call when it can, and `restart()` for fitting, are invisible to the user and give the speed that makes calibration practical. Checked identical across all five compartmental solvers with a seasonal rate.
- **The chain-binomial solver.** One multinomial draw per source compartment, `draw_chain()` for many stages in one step, and agreement with agents at every `dt` (`s4.py`). It is the right model of "discrete-time stochastic" and should be the template for tau and SDE (findings 1 and 18).
- **`'auto'` printing its reason, and `approx=False` turning notes into errors.** The mechanism is right; findings 5 to 8 are about making its decisions consistent, not about changing it.
- **The Erlang expansion with a cap.** A constant or very narrow dwell time is capped at 50 stages with a note that prints the standard deviation achieved (`s17.py`), which answers the open question in `paradigm-conversion.md` sensibly.
- **The loop as a printable, insertable plan.** `sim.loop.insert()` of a module and the per-module timelines are simple and work across solvers (`s16.py`).
