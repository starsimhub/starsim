# Execution: merged findings

This merges every finding about execution from the four reviews (F = `fable_review/` 06, 07, 11; F2 = `fable2_review/` 05, 06; O = `opus_review/notes/solvers-and-execution.md` plus the execution items in its other notes; C = `codex_review/REVIEW.txt` D2, D6, D7, F05, F10, F17, §4, and its execution evidence): the solver menu and its knobs, automatic solver choice, the compartmental and agent engines, the Numba kernels, the Python solver copies, JAX, `Crossing`, random streams, the loop and the time index, `ss.Sim`/`ss.People`, and performance. About 120 source findings collapse to 30 here. The reviews agree closely on the facts: every factual claim I re-ran reproduced, including the four that only one review raised (O's SDE and tau exit-order bias, O's two-sims-in-one-process stream leak, F2's 17% chain-binomial bias), with two corrections (cold compile is 34 s here, not the 12 s the notes and F2 repeat; and F's guess that `beta=ss.lognorm_ex(...)` is meaned under agents too is now confirmed, which makes it the worst silent answer in the area). They differ on remedies: how many solver names to keep, whether the Python solvers are a needed oracle, which time-index convention to standardize on, and whether `auto` should probe more or ask for declared capabilities. The verdict: the execution core (one flat `Program`, one Numba `step()` for every solver, whole runs in one call, the printed solver reason) is the best-engineered part of v4 and every review says to keep it. What is wrong is around it: a handful of silent answers (a distribution discarded, a requested solver ignored, shared random streams), `auto` sending ordinary models to 10,000 agents, two time-index conventions in one step, and a Python copy of every solver that doubles the surface for bugs without catching them. My checks are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/exec/` (`v1_exit_order.py` to `v14_abm_gill.py`), run with `/software/conda/envs/v4/bin/python` at `a20c4f7e`.

## Findings

### X1. A distribution on `beta` (or any non-dwell quantity) is replaced by its mean under every solver, agents included, with no note, even with `approx=False`

**Raised by:** F 06§7 and spec bullet (abm case unverified there), O solvers 6, F syn §2 and §7 table ("spec"), O syn table; C D2 in general terms ("loss of heterogeneity" must be explicit).

**Kind:** silent wrong answer.

**Status:** reproduced by F and O under `auto`/`ode`; I re-ran it under all four cases (`v6_auto.py`). `beta=ss.lognorm_ex(0.3, 0.3)` against `beta=0.3`, 10,000 people, seed 1: cumulative infections 9308.69 against 9308.69 under `auto` (printed `solver: ode (every part of the model has a compartmental form)`), `ode`, and `ode, approx=False` (no error); and **9293 against 9293 under `solver='abm'`**, so agents get the same draws as a fixed `beta`: the heterogeneity is discarded, not sampled. Cause: `Engine.value()` takes the mean of any `ss.Dist` that is not a dwell time (`solvers.py:840`), and `Agents.value` goes through it (`solvers.py:3438`); `agent_reasons()` never looks for one.

**Merged proposal:** implement the spec's rule, which matches how dwell times are already treated: under `abm`, sample per agent (this part is a plain bug); under `auto`, a `Dist` on a non-dwell slot is an agent reason (`solver: abm (beta is a distribution across people)`); under an explicit compartmental solver, use the mean with a note, and raise `E804` under `approx=False` with the `stratify` hint O suggests. O's stricter alternative (a `Dist` in a rate slot means agents, full stop, and is an error under an explicit compartmental solver) is simpler to state but breaks the "approximate with a note" rule every other ingredient follows, so I prefer the spec.

```text
# Before                                             # After
solver: ode (every part ... compartmental form)      solver: abm (beta is a distribution across people)
```

**Cost:** a check in `agent_reasons()`, a note in `Engine.value()`, per-agent sampling in `Agents`. **Breaks:** models that relied on `beta=ss.Dist` running as an ODE (none in tests or vignettes, per F).

### X2. A solver asked for on a model written as code is silently ignored

**Raised by:** O solvers 8, F syn §1 (checked), O syn table.

**Kind:** silent wrong answer.

**Status:** reproduced by O and F; re-ran (`v6_auto.py`): `ss.Sim(diseases=ss.SIR(), networks='random', solver='gillespie')`, and the same with `solver='ode'` and `solver='ode', approx=False`, all run as `'abm'` and print only `Initializing sim with 2000 agents`. Cause: `init_solver()` returns early with `self.solver = 'abm'` when nothing is declared (`sim.py:329`).

**Merged proposal:** raise `E804` when `solver` is neither `'auto'` nor `'abm'` and no module is declared: `cannot run with solver="gillespie": ss.SIR is written as code and runs only as agents; use a declared disease (ss.Disease(...)) to choose a solver`. X11 removes the commonest case. **Cost:** a few lines. **Breaks:** nothing that worked.

### X3. Random streams are not independent: stepped stochastic runs share Numba's global generator, and engine seeds collide by name

**Raised by:** F 07§1 and codebase §2, F2 05 f1, O solvers 10, C F05 (all four); anagram seeds: C F05, F2 05 other; `beta=0.3` versus `beta=lambda sim: 0.3`: F 06§9, F2 05 f2; outcomes drawn for all runs at once in `run(n_runs=)`: F2 05 other (a known issue); ranked top-tier in every synthesis.

**Kind:** silent wrong answer (reproducibility and paired comparisons).

**Status:** reproduced by all four; I re-ran each form (`v4_streams.py`, `v5_interleave.py`). Two independent SIRs, 10,000 people, seed 3, forced onto the stepped path by `run(profile=True)`: disease `a`'s `n_I` changes by up to 93 (`binomial`) and 119 (`gillespie`) when `b` is added; the whole-run path gives 0 and one disease alone gives stepped = whole exactly. O's cross-sim case, which only O raised: `a.run(until=50); b.run(until=50); a.run()` with seeds 1 and 2 changes `a` by up to 49 (it does not show if both sims use the same seed, which is how my first attempt missed it). Modules named `ab` and `ba` get identical trajectories in a whole run (`Engine.make_rng` seeds from `sum(ord(c))`, `solvers.py:586`). `beta=lambda sim: 0.3` gives different draws from `beta=0.3` (max difference 184 and 182) because it moves the engine onto the Python path, which uses the engine's NumPy generator.

**Merged proposal:** one generator per engine, passed into the kernels, seeded from a real hash of `(rand_seed, full module path)`. The four reviews propose three mechanisms: reseed the global state per step (F's cheap fix; F2's `H(engine_seed, ti)`), pass an `np.random.Generator` into the kernels (O; Numba has supported `Generator` arguments since 0.56, and `deferred.md` measured its binomial at the same 47 ns), or a counter-based hash per flow and step (F's better fix; C's "explicit per-stream random state in every backend"). I recommend O's `Generator` now: it is the smallest change that fixes all three leaks (within a sim, across sims, by name), and once the Python solvers go (X24) it is the one compartmental generator, which also lets outcomes be drawn inside `run()` per step so `run(n_runs=)` matches single runs (F2). Keep the counter-based hash for when parallel groups are wanted (it is what `deferred.md` says the Ghana measles gap needs). Add C's metamorphic tests: alone versus together versus reordered, and whole versus stepped versus profiled. **Cost:** a kernel argument; stochastic baselines move once. **Breaks:** stored stochastic results.

### X4. A daily-step chain-binomial run (and an agent run) is 15 to 17% off the ODE on an ordinary seasonal SIRS, and the note understates it

**Raised by:** F2 05 f7b and question 22 (the number), F 06§6 (calls the residual on an Erlang SEIR "ordinary discretization error"; fix works), O solvers keep ("agents and chain-binomial agree with each other at every `dt`"), C §5 (O(dt) claims need qualification). F2 alone measured the size.

**Kind:** silent wrong answer (a note is printed, but it does not convey the size).

**Status:** re-ran F2's model (`v3_binomial_bias.py`, `v14_abm_gill.py`): seasonal SIRS (mean `beta` 0.3, amplitude 0.5, `dur` 8, waning 60 days), 100,000 people, one year, `n_I` against the daily ODE at 30-day points:

| Solver | Mean difference | Max | Infected-days |
|---|---|---|---|
| `binomial`, `dt=1` (default) | +17.1% | 38.5% | +8.6% (five seeds: +7.4 to +9.2%) |
| `binomial`, `dt=1`, `substeps=4` | +3.7% | 9.0% | +2.1% |
| `binomial`, `dt=1`, `substeps=10` | +0.6% | 7.7% | +0.5% |
| `binomial`, `dt=0.25` | +3.7% | 9.0% | +2.0% |
| `abm`, `dt=1` | +15.2% | 32.5% | +10.8% |
| `abm`, `dt=0.25` | +2.4% | 6.8% | +2.6% |
| `tau`, `dt=1` | +0.5% | 4.9% | −0.4% |
| `gillespie`, `dt=1` | −1.5% | 12.2% | −0.7% |

The only message is `note: recovery has rate × dt = 0.12 ... so 0.12 of people leave in a step; sim.check_dt() shows how much a smaller step changes the results`. The discrete-time pair (binomial, agents) agree with each other and converge to the ODE; the continuous-time solvers do not have the bias.

**Options (Cliff's call, per his standing rule):** (a) keep whole steps as the default (the field's chain-binomial convention, and what makes `binomial` agree with `abm` at the same `dt`), and make `W102` state the effect and name a solver without it, as the spec asks: `note: recovery has rate × dt = 0.12; at this step discrete-time solvers (binomial, abm) overstate prevalence; use substeps=, a smaller dt, or solver='tau'`; (b) default `binomial` to the ODE's substep rule (no compartment turns over by more than a quarter in a substep), with the importers that need whole steps setting `substeps=1` as `epydemix.py:250` and `atomica.py:283` already do; then `binomial` agrees with the ODE and disagrees with `abm` at `dt=1`. F and O lean to (a); F2 asks. My lean: (a), because the agent engine has the same O(dt) and cannot take the substep fix as cheaply, so (b) trades one disagreement for another; but the warning must say how big it can be. Also fix the stale `parameters.py:326` docstring (F, F2), which describes a substep-per-stage default the code dropped.

### X5. The SDE solver resolves competing exits in declaration order at coarse steps, silently

**Raised by:** O solvers 1 only (O syn ranks it a silent wrong answer).

**Kind:** silent wrong answer.

**Status:** re-ran (`v1_exit_order.py`, `v2_exit_realistic.py`, `v9_oracle.py`). Three equal exits from `I`, 1,000,000 people, `dt=1`: at 0.5/day each, `sde` sends `[0.5004, 0.4996, 0]` of leavers by exits A, B, C (ODE, tau, binomial, gillespie all about a third each); at 1.0/day, `[1, 0, 0]`. Only `solver: sde (requested)` is printed (no `W102`). `substeps=4` gives `[0.3333, 0.3331, 0.3336]`. In an ordinary branch (`I -> R` at 0.4/day and `I -> D` at 0.1/day) the share dying is 0.1999 or 0.2004 by order, so the bug needs a total exit rate times `dt` near 1. The Python copy (`use_kernels=False`) gives the same `[1, 0, 0]`. Cause: `kernels.diffuse()` (`kernels.py:636`) and `step_sde_once()` clip each flow in turn to what is left.

**Merged proposal:** cut `sde` (X17). If it stays: draw the total leaving each source once and split it in proportion to the hazards (as `draw()` does for binomial), give it the ODE's automatic substeps, and add the three-exit permutation test for every solver (O). **Cost:** a few lines, or deletion.

### X6. Tau-leaping clips its draws in declaration order, and its leap condition is per flow rather than per compartment

**Raised by:** O solvers 18 only.

**Kind:** silent wrong answer (small).

**Status:** re-ran O's case (`v2_exit_realistic.py`): three people, five equal exits at 1/day, 40,000 runs: tau `[0.2023, 0.2016, 0.2019, 0.1964, 0.1978]` (standard error 0.0012, so the last two exits are 2 to 3 standard errors low) against binomial `[0.1993, 0.1995, 0.2009, 0.2003, 0.2001]`. The same 40,000 runs took 76 s under tau against 4.8 s under binomial, because the leap is `0.03/fastest rate` regardless of counts.

**Merged proposal (O):** compute `fastest` as the largest total exit rate per compartment; within a leap draw the total leaving each compartment, clip once, and split multinomially. Then `draw()` has one code path. **Cost:** a few lines. If tau becomes Gillespie's internal fallback (X17), the fallback only applies to large event counts, where this bias vanishes, which is a second reason for that design.

### X7. Agents realize an exponential `dur=` differently from `rate=`, and seeded people leave a step late

**Raised by:** O core 4, F2 05 other; merged in full as language L14, which leaves the engine change to this area.

**Kind:** silent wrong answer (small).

**Status:** re-ran (`v6_auto.py`): seeds with a constant 2-day dwell give `n_I = [1000, 1000, 1000, 0, ...]` (they leave at day 3); an exponential `dur=2` lets 0.223 of seeds leave in the first step against 0.393 exact, while `rate=0.5/day` gives 0.393. In `v11_erlang_tail.py` the agent run of a lognormal dwell sits above the exact survival at every day (0.927 against 0.891 at day 5), for the same reason.

**Merged proposal (engine side):** schedule initial occupants as if they entered at `t = 0`; realize an exponential dwell as a per-step hazard, as `rate=` is, and sample only non-exponential dwells (O, F2). This also lets `rescale_exits()` and the `mult_<state>` arrays apply only to non-exponential dwells (F2), and fixes the known issue that an `adjust` on a duration does not follow a person who later changes level, for the commonest case (O). **Cost:** moderate; agent results move slightly toward the ODE.

### X8. The first run compiles the kernels for about 35 s and prints nothing

**Raised by:** F 07§3 (29 s), F 13 (12 s; ship or warm the cache), F2 05 correctness (12 s, cached; agrees with the interpreter trade); `implementation.md` known issues says about 12 s.

**Kind:** first-week friction (performance).

**Status:** re-ran with an empty `NUMBA_CACHE_DIR` (`v7_compile.py`, 20 cores, load about 7): import 1.7 s; **first ODE run 33.7 s**; then `binomial` 0.01 s, `gillespie` 0.05 s, `tau` 0.01 s, `sde` 0.01 s, and the first agent run 1.8 s (the Starsim 3 Numba functions). Warm: first ODE run 0.1 s. So F's "one compile covers every solver and model shape" holds, and the 12 s figure is wrong on this machine by about 3x. F 13's unverified worry that a read-only site-packages recompiles every process is probably not true (Numba falls back to a user-wide cache directory); I did not test it.

**Merged proposal:** print one line when the cache is cold (`building the solver kernels for first use (about 30 s; once per install)`), state the time as a range in the docs, and warm the cache in the background at first import or at install (F, F 13). **Cost:** small. Shipping prebuilt caches is fragile (keyed to the exact Numba/LLVM), so warm rather than ship.

### X9. `auto` sends ordinary models to agents for reasons the spec says it should not, and `solvers()` disagrees with what `auto` did

**Raised by:** step override: F 11§3, O solvers 5, F syn §7 table ("spec"); `compartmental = True` ignored on declared subclasses: F 11§3, O 5; function interventions and analyzers: F2 05 other, O effects 10, O solvers 5; `ss.Births`/`ss.Deaths`: F 06§8, O 5, F 11§6; agent-only attribute under an explicit solver: F 06§4 and B1, F syn table; per-step functions that pass the probe and fail later: F2 (via language L38); `solvers()` contradictions: O solvers 7; capability declarations instead of probing: C D7, C F18 (one capability registry).

**Kind:** first-week friction (a two-line policy function turns an ODE into 10,000 agents).

**Status:** reproduced by F, F2, O; re-ran (`v8_misc.py`): a declared subclass whose `step()` only calls `super().step()` gives `solver: abm ("mysir" defines its own step())`; `interventions=intv` (a plain function) under `solver='binomial'` raises `E804 ... the module "intv" (Intervention) only works on agents`; a rate reading `sim.people.n_partners` under `solver='ode'` passes `init()` and dies with a raw `AttributeError: People object has no attribute "n_partners"`; a rate reading `sim.people.age` makes `auto` choose `abm` (`does not work on groups (people)`, the word `people` being what survives of the error after `split(':')`), and `sim.solvers()` then ends `auto would choose: ode`.

**Merged proposal:** build one capability record per ingredient at `init()` (ingredient, verdict per solver, reason) that `auto`, `solvers()`, `describe()`, and `E804` all read (O 7, C F18). Fill it by: (1) probing pure functions as now, and extending the probe to function interventions, function analyzers, and `step()`/`step_state()` overrides, run once on the throwaway probe sim `try_functions()` already builds (F 11, O, F2); (2) honoring `compartmental = True/False` on any module, declared or not, as the explicit override both ways (this is C's declaration, as an escape hatch rather than a requirement); (3) making the compartmental people view raise `E804` naming the attribute and the `stratify` alternative, both at the probe and on a later call, so a time-conditional branch that fails mid-run gets a coded message (F 06§4, which also answers C's "a probe proves little"); (4) for `ss.Births`/`ss.Deaths`, a reason that says `use ss.birth()/ss.death() for a compartmental form` now (F's one line), and a compartmental form later if wanted. Name the function in the reason (`knock`, not `Intervention`).

```python
def knock(sim):
    if sim.ti == 20: sim.diseases.sir.pars.beta = 0.1
ss.Sim(sir, interventions=knock).run()
# Before: solver: abm (Intervention only works on agents)        -- 10,000 agents
# After:  solver: ode (every part of the model has a compartmental form; knock was tried on groups)
```

**Cost:** moderate; the probe exists. **Breaks:** models that ran as agents only because of a step override will run as an ODE (the spec's intent). C's concern about side effects in probed code (file writes) stays; document that `init()` calls each function once.

### X10. Leaving the whole-run kernel is silent, and it is triggered by things that look harmless

**Raised by:** `verbose=1`: F 07§1, F2 05 other, O 19; a function in a rate (Python path): F 06§9, F2 05 f2, O 11; `when=` or function modifiers: O effects 21, the flepiMoP vignette; deliveries in a multi-disease model and crossings: F 07 outside, `deferred.md`; `kernel_reason()` wrong: F 07 bug 2, O 17, F2 06 other; one reason function: F 07 codebase §3, O 17.

**Kind:** performance (first-week friction).

**Status:** re-ran (`v8_misc.py`, `v4_streams.py`): a 10-year daily SIR runs in 27 ms at `verbose=0` (`kernel_only=True`, `kernel_reason()='unknown'`) and 978 ms at `verbose=1` (36x; same results). `beta=lambda sim: 0.3` makes `gillespie` 19x slower (0.048 s against 0.903 s). O measured `ss.modify(when=...)` at about 11x on a 1,000-step ODE.

**Merged proposal:** (a) verbosity never chooses the path: the kernel path prints the solver line and a summary; only the per-step progress bar needs the loop (F, F2, O). (b) One function returns the reason the run cannot be taken whole, or `None`; `kernel_only()` is `reason() is None`; `restart()`, `calibrate()`, and the JAX fallback quote it (F, O). (c) When `verbose` is on and the run is stepped, print the reason once. (d) X24 removes the function-in-a-rate cliff entirely. **Cost:** small. **Breaks:** nothing.

### X11. The library `ss.SIR`, `ss.SIS`, `ss.SEIR` are Starsim 3 code models, and `ss.Disease` is two classes chosen by sniffing its arguments

**Raised by:** F 11§1 and codebase §4, F syn §1, F2 06 f2 and syn 3, O core 5, O surface 6, O syn; see also language L31 (the `sir.S` type change and the argument sniff).

**Kind:** first-week friction.

**Status:** reproduced by F, F2, O (`ss.SIR().equations()` raises; `ss.Sim(ss.SIR())` with no network infects nobody; 15 `graph is None` branches in `diseases.py`, which I counted). X2 is the solver half of it, re-run above.

**Merged proposal:** rewrite the three in the class form with their Starsim 3 keywords (`beta`, `dur_inf`, `init_prev`, `p_death`) and, for one release, `infected`/`susceptible` as aliases of `I`/`S`; `ss.Disease` becomes declared-only and the code path lives only in `ss.Infection` (kept for subclassers and the library diseases). F2 06 gives the twelve-line version:

```python
class SIR(ss.Disease):
    beta     = ss.par(ss.peryear(0.1))
    dur_inf  = ss.par(ss.lognorm_ex(mean=ss.years(6)))
    p_death  = ss.par(0.01)
    S, I, R, D = ss.states(4)
    infection = ss.transmission(S, I, beta=beta)
    recovery  = ss.progression(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.death(I, dur=dur_inf, p=p_death)
```

**Cost:** about 300 lines of Starsim 3 code become about 30; the 15 branches go. **Breaks:** subclasses that override `set_prognoses`, read `ti_recovered`, or rely on `n_infected`; result names change unless states are named `susceptible` and so on. Overlaps the one-system area, which owns the deletion list.

### X12. In a compartmental run `sim.people` is a one-row stand-in, so the sim misreports itself and sim-level results depend on the solver

**Raised by:** F 11§4, §5 and bugs 1-2 and codebase §2, F2 06 f7, O solvers 15, O effects 15 (`ss.People(1)` placeholder).

**Kind:** first-week friction (plus one crash).

**Status:** reproduced by F, F2, O; re-ran (`v12_people.py`): `repr(sim)` is `Sim(n=1; 0—10; diseases=sir)` for a 10,000-person ODE; `len(sim.people) == 1`, `pars.n_agents == 1`; `repr(sim.people)` raises `E804 ... people.age was asked for`; sim-level results are `['timevec']` under ODE against `['timevec', 'n_alive', 'n_female', 'new_deaths', 'new_emigrants', 'cum_deaths']` as agents; with age bins 0/15/65, `where=ppl.age < 15` runs and `where=ppl.age < 10` crashes with `ValueError: operands could not be broadcast together ... (10000,) and requested shape (3,)`.

**Merged proposal:** in a compartmental run `sim.people` is `None` and reading it raises `E804` with today's text; the per-group values (age midpoints, sex) live on `sim.groups` (X28); `repr(sim)` prints `Sim(10,000 people; ode; ...)`; `n_alive`, `new_deaths`, `cum_deaths` (and `n_female` when sex is a dimension) are sim-level results under every solver, computed from the compartments (F 11, O 15). For lazy conditions, a class-level reference needs no instance and no new name: `where=ss.People.age < 15` (F 11); F2 proposes `ss.age`, O a placeholder object. A threshold that is not a bin edge raises `E804` naming the edges (or, per C D7, says how it is represented). **Cost:** moderate (the engines read group values from `sim.people` today). **Breaks:** code reading `len(sim.people)` in an ODE run, which is wrong now.

### X13. Code in the same step sees different values of `sim.ti`

**Raised by:** O solvers 3, F2 06 f4, O effects 17 (Starsim 3 `sim.ti == 0` never fires), F 07 codebase §6 (first-step filter by function name), O syn ("one time convention, set by the loop").

**Kind:** first-week friction (an off-by-one between a v4 delivery and a hand-written intervention).

**Status:** reproduced by O and F2; re-ran (`v8_misc.py`): over a 4-day agent run, a function in a declared rate sees `ti` 0 to 3 and a function intervention 1 to 4. Cause: `ss.solvers.interval()` decrements `sim.t.ti` around declared steps and deliveries only (`solvers.py:537`, wrapped by `loop.py:360`), and `_before_first_step()` drops `ti=0` entries by function name (`loop.py:383`).

**Merged proposal:** one convention, set by the loop for every module: during the step from time point k−1 to k, `sim.ti == k−1` and `sim.now` is the step's start; `update_results()` writes index k. Then `interval()`, `_at_start_funcs()`, and the name filter go. This is O's note (O's synthesis says `ti == k`, which contradicts its own note; I take the note), and it is compatible with F2's "results at index 0, steps 1..T, one `ti` for everyone". It has a bonus: the first step has `ti == 0` again, so Starsim 3 interventions written `if sim.ti == 0:` fire as they did in Starsim 3 (O effects 17's problem goes away). **Cost:** loop change plus shifted expected values in v3-style tests. **Breaks:** v4-era code that tests `sim.ti == k` in an intervention fires one step earlier.

### X14. `solver='stochastic'` estimates events from the population only, and picks Gillespie for runs that take minutes

**Raised by:** F 06§2 (24x low), O solvers 2 (48x low, 74.5 s against 0.03 s), F2 05 f7d ("serviceable").

**Kind:** performance.

**Status:** reproduced by F and O; re-ran (`v10_stochastic.py`): an endemic SIS in 50,000 people for a year prints `solver: gillespie (exact; about 100,000 events at this population size)`; the run has 2,329,944 events and takes 8.35 s (tau: 0.02 s). O's proposed estimate, the sum of the ODE's flows, gives 2,326,176 (0.2% off), and for 10 million people over 10 days 1,184,378 against 1,183,879 actual; the ODE run costs milliseconds.

**Merged proposal:** O's: run the ODE once at `init()` for `'stochastic'` (or for `gillespie`'s fallback, X17), choose exact below a budget (about 1e6 events or a few seconds), and print the real estimate. F's alternative (source counts times rates times run length) is cruder and no cheaper. **Cost:** a few lines.

### X15. `check_dt()` crashes when a module has its own timestep

**Raised by:** C F10 only.

**Kind:** crash.

**Status:** re-ran (`v8_misc.py`): `ss.Sim(Disease(..., dt=ss.days(2)), solver='ode', unit='day', dt=1, stop=8).check_dt(output=True)` raises `ValueError: cannot reshape array of size 4 into shape (8,2)`. C also notes a zero-safe scale is missing and stochastic runs need replicates.

**Merged proposal:** align by each result's own time coordinates (C); label a stochastic `check_dt()` as one realization. **Cost:** small.

### X16. Performance at scale trails the specialist frameworks a switching user would come from

**Raised by:** the vignettes (LFD), F2 02 f1, O surface 12, C D1 (restate the time test); `deferred.md` and `implementation.md` for causes.

**Kind:** performance.

**Status:** from the vignette READMEs, not re-run: LASER England and Wales as agents, 131 s against 3.0 s (LASER on 20 cores, Starsim on one; about 20 to 110 ns per agent-step against about 1 ns); SimInf herds, Gillespie, 3.0 s against 0.08 s, and SISe3 12 s against 0.10 s; odin's 100 stochastic copies about 20x slower; Ghana measles chain-binomial 1.04 to 1.4 s against 0.84 s; summer2 between 2.5x slower and 4x faster per ODE run once built. My Gillespie run above is about 3.6 µs per event (2.33 M events in 8.35 s, X14). Whole-run ODEs are competitive; exact stochastic and large agent runs are not.

**Merged proposal:** two levers are already identified in `deferred.md`: parallel groups (which needs X3's counter-based streams so runs stay reproducible) and a model-specialized kernel as an opt-in for long runs (F2 agrees with deferring it, since a Numba kernel generated per model would take 1 to 3 s each to JIT-compile). For agents, `implementation.md` rules out a whole-run kernel (at most 4x on one thread). State the time test as a factor against compiled originals for the same paradigm, excluding first-use compile (F2, O, C), and publish the table with its failing rows. For a LASER or SimInf user this is the decisive gap and it is partly an unavoidable tradeoff: a general interpreter over a flat program will not match model-specific compiled code on one thread.

### X17. The solver menu: six names, `'stochastic'`, `'ctmc'`, `integrator=` that mixes schemes with a backend, `substeps=` with three meanings, and `use_kernels`

**Raised by:** F 06§1, §3 and §13, F 07§2, F2 05 f6 and f7(a, c, e), O solvers 14 and 19, F syn §5, O syn table, F2 syn question 9 (`sde`?).

**Kind:** surface.

**Status:** usage counts by F (vignettes use `ode` 33, `binomial` 21, `abm` 9, `gillespie` 5, `tau` 3 (two tests and one README line), `sde` 0, `stochastic` 0, `ctmc` 0); `sde` is in `pmcmc`'s list (`calibration.py:1530`); `integrator='scipy'` dies in SciPy (F2), `integrator='bogus'` passes `init()` (O), `integrator='jax'` is rewritten to `'rk4'` (F2) and is 150x slower than the kernels for one run (F 07). The atomica importer writes `integrator='euler', substeps=1` into generated code (`interop/atomica.py:283`).

**Merged proposal:** public solvers `auto`, `ode`, `binomial`, `gillespie`, `abm`. `gillespie` switches to tau-leaping above an event budget estimated by X14, with a note, and `approx=False` forbids the switch; `'tau'` and `'stochastic'` go (or stay one release as aliases that print the new spelling); `'ctmc'` goes; `sde` goes (X5, worst agreement with the ODE in F's runs, no vignette, and its niche is tau's). `integrator=` stays but only as a scheme, `'rk4'` (default) or `'euler'` (for importers matching Euler-stepped frameworks), validated at `init()`; JAX is chosen by `sim.gradient()`, gradient fits, and NUTS, never by `integrator=`; the SciPy methods move to `ss.options` for stiff-model debugging or go. `substeps=` stays, with its per-solver meaning in one docstring line. `use_kernels` goes with X24. See "Where the reviews disagree" for the positions.

```python
# Before
ss.Sim(sir, solver='stochastic')             # gillespie or tau by a population heuristic
ss.Sim(sir, solver='ode', integrator='jax')  # a backend, not a scheme
# After
ss.Sim(sir, solver='gillespie')              # exact, or tau-leaping with a note when the run would be long
ss.Sim(sir, solver='ode')                    # JAX used by sim.gradient() and calibrate() when it pays
```

**Cost:** deletes `kernels.diffuse()` and the `solver == 5` branches (about 70 lines), two `step_sde*` methods, and spec rows. **Breaks:** `solver='sde'`/`'tau'`/`'stochastic'`/`'ctmc'` and `integrator='jax'`/SciPy names (tests, one README line, `pmcmc`'s list).

### X18. The printed solver line and notes are out of order, ignore `run(verbose=0)`, and omit what the spec says they show

**Raised by:** F 06§5 and spec bullets, O solvers 16, O surface 20, F 07§5 (schedule), F 07§4 and F2 06 other and O results 18 (no CRN line in `describe()`).

**Kind:** surface.

**Status:** re-ran the order (`v13_notes.py`): `note: RandomNet approximated by ...` prints before `solver: ode (requested)`. The notes themselves are recorded in `sim.notes` at any verbosity (so C D2's worry that printing off loses the approximation record does not apply to these notes). Others reproduced by their reviewers: `run(verbose=0)` still prints init's lines; `mean 7.29139 days`; no scale factor as agents; the `describe()` schedule is module order, not loop order, and lists products with no step.

**Merged proposal:** collect the solver line and notes during `init()` and print them together at the start of `run()`, honoring `run(verbose=)`, solver line first, numbers to three significant figures (O). Add one randomness line to `describe()` (`random: paired by run (one stream per disease)` / `common random numbers by agent slot` / `none (deterministic)`, F 07§4) and drop the spec's per-distribution `ok`/`note` table. Build the schedule from `sim.loop.funcs` and append `(taken in one call by the kernels)` when it is (F 07§5).

### X19. Common random numbers work but are unreported and oversold, and two stream seeds can coincide

**Raised by:** O results 18, F 07§4 (run-level pairing measured), C F04 (`options.crn` absent from the hash; the spec area owns hashing).

**Kind:** surface.

**Status:** reproduced by O and F: run-level pairing works (F: standard deviation of an intervention effect 10 against 52 unpaired under `binomial`); with homogeneous mixing CRN on and off pair about equally; `ss.options.crn` has no effect on a compartmental run (F). Not re-run.

**Merged proposal:** keep the mechanism; say what it does in `describe()` (X18) and in the docs, without overselling it for large perturbations (O); hash `(name, rand_seed)` together instead of `str2int(name) + rand_seed` (`distributions.py:764`) so seed s of one stream cannot equal seed s+1 of another (O).

### X20. Erlang stages under `auto` are an approximation the user did not ask for (C), against "the one approximation `auto` should make" (F, F2, O)

**Raised by:** C D2; F 06 keep, O solvers keep, F2 05 good.

**Kind:** surface (default behavior).

**Status:** re-ran (`v11_erlang_tail.py`): a lognormal dwell with mean 10 and std 5 days becomes 4 Erlang stages (note: `mean 10 days kept; std 5 instead of 5 days`). Survival in the state at days 10, 20, 25, 30: lognormal 0.407, 0.044, 0.015, 0.0052; Erlang ODE 0.434, 0.042, 0.010, 0.0023. The body matches within a few points; the far tail is about half.

**Merged proposal:** keep Erlang stages as `auto`'s one approximation (three reviews), keep the note, and add to it when the tail is materially thinner (`the chance of staying past 30 days is 0.002 instead of 0.005`), which is the inspectable comparison C asks for without making users opt in. A user who needs the tail has `approx=False` (agents under `auto`) or `stages=`.

### X21. `gillespie` and `tau` hold time-varying rates for a quarter step while being described as exact, and `approx=False` does not refuse it

**Raised by:** C F17, F syn §7 table ("strike the MUST"), F2 05 decisions (agrees with the quarter-step departure), `deferred.md` (thinning).

**Kind:** spec drift.

**Status:** re-ran (`v14_abm_gill.py`): `solver='gillespie', approx=False` with a seasonal `beta` initializes with only a note (`holds values that vary in time for a quarter of a step at a time`). F2 measured the holding as accurate (seasonal SIRS at 30-day steps: tau 1.4%, gillespie 4.4% from the daily ODE).

**Merged proposal:** keep the quarter-step holding (all four agree it is adequate), describe Gillespie as "exact for rates constant within a step", strike the thinning MUST, and make `approx=False` raise for it, so the triad means the same thing everywhere (C). **Cost:** a line and a spec edit.

### X22. Delivery order differs by engine: after progressions as agents, before all flows as compartments

**Raised by:** F 07§6, O solvers 4, F syn §7 table ("state it as the rule"), `implementation.md` known issue (TPT reached 7% fewer people on the first step as agents); C §5 (ordering is model semantics, not always O(dt)).

**Kind:** spec drift (a solver disagreement).

**Status:** known issue; not re-run.

**Options (Cliff's call):** (a) run all of a declared disease's transitions together at row 9 under both engines, with one competing-hazards draw per state including the force of infection (F 07, O): deliveries see the same people under every solver, the row 5/row 9 caveat leaves the spec, and agent results change by O(dt); (b) keep the split and state it as the rule (F's synthesis). My lean: (a), since it removes a rule and makes "competing hazards are resolved jointly" true without exception; C's caution argues for testing the ordering directly rather than relying on the O(dt) argument.

### X23. Spec drift in this area not covered above

**Raised by:** F 06 spec, F 07 spec, F2 05, F 11 spec, O 9, O 16, C F18.

**Kind:** spec drift.

**Merged proposal (each: strike or fix, one line):** `12-backends.md` §How a model is built is wrong in three of five bullets (SciPy for small, JAX for large and fitted): rewrite to "RK4 in Numba; JAX for gradients" (F 07, F2); `13-inference.md:191` says the ODE "runs through JAX" and gradients come in the same pass (F 07); `MFNet` under `ode` raises `E804` though it is the spec's headline example, and `solvers()` reports `approx (contact rates)` for it (F, F2, O 9, C F18; the routes area owns the approximation, X9's capability record fixes `solvers()`); `W1203`/`W1204` never raised (F); `hiv.solvers()` on a disease absent (F, language L32); `run(solver=)` only before `init()`, which is the better rule (F, F 11); the printed `solver: ode (SciPy, 3 compartments)` (F); `ss.Sim(results=[...])` not accepted (F 11, F2 06, O results 10); `E1003 RunAfterFinish` registered but `AlreadyRunError` raised (F 11); the lifecycle table in `01-objects.md` lists five hooks where the loop calls nine (F 11); `init()` computing the spec and hash eagerly (the code's lazy form is better, F 11); `10-execution.md`'s "interventions called at loop times, in loop order" under whole runs (F 07); open questions B3 and B10 answered by the code and to be closed (F 06, F 07); the `sde` small-count warning (moot with X17).

### X24. Every compartmental solver is written twice (Python and Numba), and the Python copy shares the kernels' bugs

**Raised by:** F 06§9, F 07 codebase §1 and summary 7, F2 05 f2 and question 21, O solvers 11, F 13, F syn §5, F2 syn 8, O syn table; C D6 argues the opposite for a reference backend.

**Kind:** codebase (with user-visible effects).

**Status:** reproduced by all; re-ran: the Python path is reached by `beta=lambda sim: 0.3` and is 19x slower under Gillespie with different draws (X3, X10); and the Python `sde` gives the same `[1, 0, 0]` as the kernel (`v9_oracle.py`), so as an oracle it did not catch X5 (nor X6 or X3, which all tests passed with). The kernel-versus-Python tests are `test_kernels.py:85-171` and five places in `test_transitions.py`.

**Merged proposal:** delete `derivs`, `step_binomial_once`, `step_tau`, `step_gillespie`, `step_sde_once`, `draw_targets`, `piece` (about 330 lines, `solvers.py:2365-3137`) and `use_kernels`. A rate that is a Python function of the sim is evaluated once per step (at the start, middle, and end for RK4, as `step_tables()` already does) into the program's tables; a `dict` beta folds into a mixing term's contacts and a `where=` on `n_contacts` into per-group contacts (F2). Replace the oracle with C §5's layered contracts, which would have caught all three bugs: exact identities (constant-rate decay, fixed branch shares), permutation of competing exits, alone/together/reordered stream invariance, stepped/whole/profiled/restarted equality, and stochastic means against the ODE at large N as `test_stages` does (F 07's condition). **Cost:** a count-dependent Python function in a rate becomes first-order within a step (O); it can be noted. **Breaks:** `use_kernels=False` (tests only).

### X25. Multi-disease compartmental runs use one engine per disease plus a cache-coherence protocol (`Crossing`)

**Raised by:** F 06§10, F2 05 f10 and question 23, O solvers 12, F syn §8, F2 syn second tier, O syn table.

**Kind:** codebase (blocks whole runs, the particle filter, and part of JAX for multi-disease models).

**Status:** reproduced by F (agrees with `abm` within 1 to 4%), O (within 2%), F2 (28 `self.cross` checks, which I counted). Not re-run.

**Merged proposal:** one compartmental engine per sim, running one `Program` whose compartments are the product where diseases are crossed and block-diagonal where they are not (O); each disease keeps a view of its slice for results and `sir.I.n`. This subsumes F's "one engine over the product graph" and F2's "one joint program" and removes `owner`/`version`/`claim()`/`refresh()`, the 28 checks, `Crossing.run()`'s Python loop, `JointRun`, and the `shared` rule; cross-disease counts in rates become ordinary `COUNT` ops; X3's per-engine generator becomes per-sim. Agents keep one engine per disease. The O(dt) change (flows evaluated together rather than in turn) is the ODE's own rule (F2). If the full refactor waits, hoist the shared array into `Crossing` so engines hold views (F's fallback). **Cost:** large (layout and results by disease, `gradients.py`).

### X26. JAX is a gradients backend, and `sim.gradient()` still differentiates its two ends by finite differences

**Raised by:** F 07§2 and codebase §4-5, F2 05 f4, O solvers 14, F 06§3.

**Kind:** codebase (and surface via X17).

**Status:** reproduced by F (`integrator='jax'` 156 ms against 1.0 ms for one run; identical to 7e-13) and F2 (import 1.7 s, compile about 1 s a shape). Not re-run.

**Merged proposal:** keep both compiled paths (F2 shows neither can be the only one: Numba for the exact solvers and fast single runs, JAX for exact gradients, NUTS, and batched sweeps); JAX is chosen automatically by `sim.gradient()`, gradient fits, NUTS, and `ss.parallel(pars=...)` when `gradients.reason(sim)` is `None` (`run.py:1137` decides this today by `sim.jax`, set only by `integrator='jax'`); make `gradient()` a `jacfwd` of a `Density`-style results function and drop the central-difference tails (`gradients.py:641-699`, about 80 lines) (F2; F at low confidence, since parameters entering through `init=` and `stages=` are what the differences cover today); set `XLA_FLAGS` only inside `nuts_jax()` (F).

### X27. The compartmental engine repeats the same machinery several times

**Raised by:** F 06§11-13, F 07 codebase §1, F2 05 f3, f8 and other, O solvers 11, F syn §8.

**Kind:** codebase.

**Status:** by reading; line counts from F2's map. Not re-run.

**Merged proposal (each loses nothing a user sees):** `Agents.step_counts()` calls `kernels.settle()` instead of a fifth RK4 (F2); `place_rates()` becomes `step()`'s rate loop over a group range instead of a second stack interpreter (F 07, F2); `step_outcomes()` is `derived_outcomes()` for one step, and the four results-bookkeeping copies become one function of stock and flow arrays (F 06§11, F2); movers, deliveries as shifts, and register events become one `Move` record and kernel (F2 f8, about 350 lines out and 120 in, and the natural home for "treatments that cure"); `Mixing` builds one `terms` list after stratifications are final instead of `terms` and `pool_terms`, and no longer stratifies the sim from inside `age_matrix()` (F 06§12, F2 other; the routes area owns the semantics); small items: `make_engine()` inline, one definition of "keeps whole numbers", `eps=0.03` once, keep the kernel workspace on the engine, key the value cache on the parameter object instead of `id()` with a pin (F 06§13, F2 other).

### X28. `Sim` carries the group geometry and the run-resolution logic, and `solvers.py` is six modules in one file

**Raised by:** F 11 codebase §1 and Simpler §7, F2 06 f1, O solvers 13, F syn §8, F 13, C §4 and D6 ("extract responsibilities rather than just splitting files").

**Kind:** codebase.

**Status:** reproduced by F and F2 (76 public methods on `ss.Sim`; 59 instance attributes against Starsim 3's 21; `init_strats()` and `Mixing(self)` run up to three times per build, `sim.py:327, 378, 787`). File sizes now: `sim.py` 2,692, `solvers.py` 4,136, `kernels.py` 1,372, `gradients.py` 1,121.

**Merged proposal:** in this order: (1) the deletions (X17, X24, X25), which take about 1,000 lines out of `solvers.py` (F2's map); (2) extract the two responsibilities C names: `sim.groups` (an `ss.Groups` object owning `strats`, `group_shape`, `fold*`, `group_raw` and the other fourteen attributes) and run resolution (`agent_reasons`, `try_functions`, the capability record of X9, `init_solver`, the unit rules onto `sim.t`), with `init_*` phases made private; (3) only then split `solvers.py` into a package along its existing seams (`mixing`, `engine`, `compartmental`, `agents`), with `ss.solvers.X` still working (O). `Sim` keeps the spec's method table plus Starsim 3's. **Cost:** mechanical once the deletions are done.

### X29. One normalized model and a reference backend (C) against "the `Program` already is that" (F, F2, O)

**Raised by:** C D6 and §4-5; the keep lists of F 07, F2 05, O solvers.

**Kind:** codebase (architecture).

**Status:** design judgment. The `Program` is already the one flat form of a model for the compartmental solvers, shared by Numba and JAX (`test_jax_run` at 1e-9); what C's proposal adds that is real here is (a) parameters are evaluated outside it by `Engine.value()` (the root of X1 and of C F01's ignored modifiers on callables, both at `solvers.py:834-863`), and (b) the agent engine does not consume it.

**Merged proposal:** take C's normalization boundary, not its second interpreter: one parameter-evaluation path (raw value, unit conversion, modifiers, distribution handling, in that order, for every solver) that feeds both the `Program` and the agent engine; serialize the `Program`'s nodes rather than printed expressions (C; the spec area owns the format). Do not add a reference backend: X24 shows a second implementation shares the first's bugs; C's own §5 contracts are the oracle. **Cost:** moderate; mostly moving `Engine.value()`.

### X30. Small `Sim` and engine items

**Raised by:** F 11§9 and bugs 3-4 and codebase §5, F2 06 other, O solvers 19, F 06§13, F2 05 other.

**Kind:** first-week friction (small) and codebase.

**Merged proposal:** keep `inputs` for every sim so `new()`, `check_dt()`, `R0()`, `restart()` work for coded sims (F 11, F2: `copy_inputs=True` already copies once); `ss.Sim(sir, n_agents=None)` fails with a message naming `popdict` (F 11); `sir.S.n` works after an ODE run and raises after an agent run (F 11; language L31 owns the fix); `integrator='euler'` uses RK4's substep rule and peaks 4% high (O 19; document as deliberate for importers or tighten); `E206` names the same module twice when states are reused (O 19); delete `Filter` (`people.py:774-941`) and `People.scale_flows()` (no callers) (F 11); a declared state as agents allocates `ti_exit_*`/`exit_*` even with no dwell (F 11 outside); `interval()` makes a function of the sim see `ti=0` for five calls per RK4 step (F2; goes with X13).

## Where the reviews disagree

**Which solver names to keep.** F: `ode`, `gillespie`, `binomial`, `abm` plus `auto`, with tau as Gillespie's automatic fallback and `'stochastic'` an alias or gone (F 06§1). F2: drop `sde` only; six names are fine; keep `tau` because its cost is bounded by the rates (F2 05 f7). O: keep all for experts but teach four, `auto`, `ode`, `stochastic`, `abm`, with the methods documented under `'stochastic'`; fix `sde` rather than cut it; drop `'ctmc'` (O 14, O 1). C: no position on names; wants deterministic versus stochastic intent visible (D2). Recommendation: F's set (X17). Reasons: `sde` has a verified order bug (X5), the worst ODE agreement, and no user; `tau` is a speed device for `gillespie` and is biased at the small counts where a user would pick it by hand (X6), so making it Gillespie's noted fallback chosen by the ODE estimate (X14) keeps its benefit and removes a choice; `'stochastic'` as a teaching name is misleading because `binomial` and `abm` are stochastic too. Check run: X5, X6, X14.

**Whether the Python solvers are a needed oracle.** F, F2, O, F 13: delete; F 07 conditions it on testing the stochastic kernels against the ODE first; F2 asks Cliff whether the debug oracle is wanted. C D6: build a small reference backend as an independent oracle. Recommendation: delete, and adopt C's §5 contracts as the oracle. Check run (`v9_oracle.py`): the Python `sde` has the same order bug as the kernel, and the stream and tau bugs passed the kernel-versus-Python tests, so a second copy of the same algorithm caught none of the three semantic bugs this review found in the stochastic solvers.

**Compile time.** `implementation.md` and F2: about 12 s, cached. F 07: 29 s cold, silent. Check run (`v7_compile.py`): 33.7 s for the first ODE run plus 1.8 s for the first agent run with an empty cache; 0.1 s warm; one compile covers every solver. Recommendation: X8.

**Chain-binomial default.** F: the fix works and the residual is ordinary discretization (6.8% on an Erlang SEIR at `dt=1`). F2: the bias is large (17% on a seasonal SIRS) and the default should perhaps follow the ODE's substep rule. Check run: F2's number reproduces (+17.1%), and agents show the same (+15.2%), so both are right: it is ordinary O(dt) error and it is large at a daily step. Options and lean in X4; Cliff picks.

**`integrator=`.** F and F 07: drop it from the public surface; keep Euler and SciPy reachable via `ss.options`. F2: keep it, validate it, accept `'scipy'` as an alias, and keep the user's word. O: drop the `'jax'` value and validate the rest. Recommendation: keep it as a scheme only (`'rk4'`, `'euler'`), because the atomica importer writes `integrator='euler'` into the code it generates; JAX and SciPy leave it (X17).

**Which `ti` convention.** O's note: everyone sees k−1 during the step from k−1 to k. O's synthesis: everyone sees k. F2: results at index 0, steps 1..T, one meaning for everyone (unspecified which). Recommendation: k−1, which restores Starsim 3's `ti == 0` on the first step and matches what rates already see (X13).

**How `auto` learns what code can run on groups.** F, F2, O: probe more (step overrides, function interventions and analyzers), as `12-backends.md` says. C D7: probing is only a hint and can run side effects; prefer a documented callback contract with an optional capability declaration. O effects 10: either implement probing on a throwaway copy or document `compartmental = True` honestly. Recommendation: probe on the throwaway sim, honor `compartmental` as an explicit override both ways, and give later failures a coded `E804` (X9). Probing keeps the zero-declaration path for the two-line policy function, which is the user's common case; the declaration covers C's cases.

**A distribution on `beta`.** F and the spec: agents under `auto`, mean with a note under an explicit compartmental solver. O: agents always; an error under an explicit compartmental solver. Recommendation: the spec's rule (X1); it is the same policy as dwell times.

**Delivery order.** F 07 and O: one slot for all declared transitions under both engines. F's synthesis: keep and state the rule. C: ordering is semantics. Options in X22; Cliff picks.

**Crossings.** F: one engine over the product graph (or views as a fallback). F2: one joint program. O: one engine per sim, product where crossed and block-diagonal elsewhere. Recommendation: O's, the superset (X25).

**Erlang stages under `auto`.** C: preserve the declared distribution by default, or require explicit consent. F, F2, O: keep as `auto`'s one approximation. Recommendation: keep, and make the note show the tail when it differs (X20). Check run in X20.

**Gillespie's exactness.** C: integrate time-varying hazards or call it approximate and make `approx=False` coherent. F's synthesis: the code is right, strike the spec's MUST. F2: agrees with the quarter-step departure. Recommendation: both halves: keep the holding, strike the MUST, and have `approx=False` refuse it (X21). Check run: `approx=False` does not refuse today.

**`'stochastic'` estimate.** F: from source counts, rates, and run length. O: run the ODE and sum its flows. Recommendation: O's; it was within 0.2% in both cases I ran (X14).

## Keep

- One flat `Program` and one Numba `step()` for every solver (all four): 1.4 µs for a three-compartment ODE step, one compile for every model shape (verified), and the reason stepped and whole runs agree to the bit.
- Whole runs in one kernel call, chosen without being asked, plus `restart()` and `run(n_runs=)` with per-run seeds (F, F2, O).
- Tabulated time-varying values and deliveries as shifts, so modifiers, seasonality, and scheduled vaccination stay inside the whole run (F 07, F2).
- `solver='auto'` printing a one-line reason that names the ingredient, and the `approx=` triad (F, F2, O, F 11); fix what it decides (X1, X9), not the mechanism.
- Erlang stages with the achieved std printed, `stages=`, and the 50-stage cap (F, O; X20 for C's caveat).
- `W102` with the fraction leaving per step and `sim.check_dt()` (F), with X4's wording.
- The chain-binomial `draw_chain()` and the joint competing-hazards draw in binomial and agents, verified order-independent (F, F2, O; my runs in X5, X6).
- States that are not people as real-valued counts under every solver, and `settle()` (F, F2).
- Hashed per-slot CRN in the agent engine, and run-level pairing in compartmental runs (F 07, O results 18, F2 06).
- Both compiled paths, Numba and JAX from the same `Program` (F2 05 f4, F 07).
- The Starsim 3 module lifecycle and the ordered, printable loop with `insert()`; v4 changed `modules.py` by 17 lines and `loop.py` by 93 (F 11, F2 06, O).
- `try_functions()` probing pure functions on a throwaway sim (F 06, cheap at about 5 ms).
- The `total_pop`/`n_agents` defaults, correct in every combination F 11 tried.
- Interpreting the `Program` rather than generating code per model (F2), with a specialized kernel only as a later opt-in.

## Decisions for Cliff

1. **Solver names:** keep `auto`, `ode`, `binomial`, `gillespie`, `abm`; fold `tau` into `gillespie` as its noted fallback; drop `sde`, `'stochastic'`, `'ctmc'`? (F yes; F2 drop `sde` only; O teach `stochastic`, fix `sde`. Recommend F's.)
2. **Chain-binomial at daily steps (+17% against the ODE, agents +15%):** keep whole steps and make `W102` say so, or default to the ODE's substep rule? (F, O keep; F2 asks. Lean: keep and fix the warning.)
3. **Delete the Python solvers and `use_kernels`, with contract tests as the oracle?** (F, F2, O yes; C wants a reference backend. Recommend delete.)
4. **`ti` during a step:** the step's start (k−1) for every module, breaking v4-era interventions that test `sim.ti == k`? (O and F2 want one convention; O's note says k−1, its synthesis k. Recommend k−1.)
5. **Delivery order:** one slot for all declared transitions under both engines, or keep the split and document it? (F 07, O unify; F's synthesis document. Lean unify.)
6. **`auto` and custom code:** probe step overrides and function interventions on groups, with `compartmental` as an explicit override? (F, F2, O yes; C prefers declarations. Recommend probe plus override.)
7. **`beta=ss.Dist`:** agents under `auto`, mean with a note under an explicit compartmental solver, sampled under `abm`? (F and spec; O stricter. Recommend the spec.)
8. **One compartmental engine per sim in place of `Crossing`?** (F, F2, O agree on direction; O's scope recommended. Large.)
9. **Redeclare `ss.SIR`/`SIS`/`SEIR` and make `ss.Disease` declared-only?** (F, F2, O yes.)
10. **`sim.people` in a compartmental run:** `None` with `E804`, with `ss.People.age < 15` for lazy conditions? (F 11 and F2 for `None`; O a group view.)
11. **`integrator=`:** a scheme only (`'rk4'`, `'euler'`), JAX automatic, SciPy to `ss.options` or gone? (F drop; F2 keep and validate; O drop `'jax'`. Recommend scheme-only.)
12. **Random streams:** a `Generator` per engine now, counter-based streams when parallel groups are wanted? (O now; F, C counter-based; F2 per-step seeds. Recommend O now.)
13. **The time test:** restate as a factor against compiled originals of the same paradigm, excluding first-use compile, and publish the failing rows? (F2, O, C yes.)
