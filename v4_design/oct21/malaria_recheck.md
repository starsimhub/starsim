# Malaria: the port rechecked against v4 as of October 21

Written 2026-10-21, at commit `ee5d1fb0` on the `v4` branch of `ssv4` (run on a clean detached worktree of it, since the working tree was being edited; its uncommitted changes, mostly sexual networks and mixing, touch none of the findings below). This rechecks [oct17/malaria_review.md](../oct17/malaria_review.md), written at `1a5b7a77`, 220 commits and the simplification ago. malariasim is unchanged (`starsim-ngm`, branch `v4`, `a1351fc`). Every script behind the Oct 17 review (98 of them) was rerun, plus the malariasim suite three ways, the comparison of unchanged malariasim with Starsim 3, the migration script on a copy of malariasim, new probes, and an old-against-new speed benchmark. Scripts, logs and the patches needed to rerun the old scripts are in the `starsim-ngm` repo under `v4port/oct21/`, with malariasim's side of things in `v4_port_oct21.md` at its root.

Everything ran on the `v4` conda env (Sciris 3.4.0, Numba 0.68). New findings continue the numbering at MA34.

## In brief

**Much of the Oct 17 review is fixed.** Of its 33 findings, 13 are fixed, 5 partly, 3 superseded by a design change, 11 open, and 1 decided against (MA21).
- **A vector-borne infection is now an infection** (MA1, MA2). `rel_sus`, a scalar vaccine efficacy and the spec's bednet idiom take the spec's Ross-Macdonald model from 7,424 human infections a year to 0 (they did nothing). `ss.treat` with no `where=` reaches 2,145 infected people (it reached nobody).
- **Mosquito counts as agents are substepped** (the agents half of MA6, and MA8, MA9). They hold 50,000 at a monthly step, where on Oct 17 they died out. Modifiers on their transitions apply. People's rates see the advanced count, so the first step has 96 infections as agents against 97.5 as an ODE, where on Oct 17 it had 0.
- **A fixed incubation is exact as an ODE and chain-binomial** (MA7): 5,020 infectious mosquitoes, the exact value, against 5,092 and 1.8e17 on Oct 17.
- **Newborns from `ss.Births` keep the node split** (MA18): the population test that failed with node shares of 0.555/0.226/0.218 passes.
- **Silent level-0 fallbacks are refused** (MA10), with `MissingDimension` or `InvalidAdjustment`.
- **`ss.SIS` subclasses with coded hooks are refused at class creation** (MA23), naming `ssl.AgentSIS`.
- **Typed values are converted once** (MA27): the declared human ODE with 3 patches, 20 age groups and a seasonal EIR per patch runs 2.3 times faster than on Oct 17.
- **Fixed smaller items:** `ss.diff` on coded modules (MA25), `.n` and `.count()` under both solvers (MA13), `.by()` plots on a `MultiSim` (MA31), `people_results=False` with demography (MA32), `Disease.move_to()` for coded code (ST11), `resample('year')` (FP3), `ss.peryear(1) * ts` and `* fn` (FP31), one seed for a deterministic fit (MA26). There's also a warning when a coded `set_prognoses()` skips `super()` and so leaves `new_infections` at 0.

**What's new and wrong, most important first:**
1. **MA34 (silent): in `init=`, a value of 1.0 for a state of people means one person.** `init=dict(S=1.0)`, which reads as "everyone susceptible" (and is how `init_prev=1.0` is read), starts 1 person in `S` and puts the rest in the other state, with only a note. `init=dict(S=1.0, I=0.0)` sums to 1 and is still read as a count. Six of the Oct 17 port scripts used this spelling: the Ross-Macdonald interventions baseline now starts with 3,999 of 4,000 people infected and moves 2.5% at day 364.
2. **MA35 (silent): a forgotten `people=False` makes the mosquitoes agents.** The spec now says (rightly) that every state is people unless declared a count. It also says a forgotten `people=False` "is caught when the module is built", but here it isn't: with `init=dict(Sv=3000)`, 3,000 of the 5,000 agents are mosquitoes. With no `init`, emergence adds mosquitoes as agents, giving 75,221 "people alive" after 100 days. Every Oct 17 vector script that relied on the old guess either errors with "initial fractions sum to 10.01" (which doesn't mention `people=False`) or runs silently with mosquito agents.
3. **MA36 (solvers disagree, silently under the ODE):** `ss.change(mal.emergence, by=0.5)` now halves the mosquitoes as agents (2,013) but is ignored as an ODE (4,000). A product efficacy keyed by a mosquito transition is refused as agents (good), but ignored as an ODE, with no message.
4. **MA37 (silent): `R0()` of Ross-Macdonald is off by √2.** It was 0.0 and is now 8.78, against 12.42 analytically. The next-generation matrix uses half the disease-free number of mosquitoes (63 mosquito infections per infectious human, where 126 is right), whatever the initial count.
5. **The chain-binomial half of MA6 is still open:** at a 30-day step the mosquitoes go 10,000 → 5,000 → 2,500 … → 2 (50,000 is right).
6. **MA38 (capability, a decision):** waning by a function now has no compartmental form. So malariasim's box-then-decay vaccine and bednet can't run as an ODE: the solver is refused before the run. On Oct 19 this had been made accurate to 0.5% (MA16).
7. **MA39 (speed):** declared agents with mosquito counts are 1.5–1.9 times slower than on Oct 17, and unchanged malariasim (coded) about 10–15% slower; the declared ODEs are as fast or faster.
8. **MA40, MA41 (migration):** `pyproject.toml` allows Sciris 3.2.8, but `ss.parallel` now needs 3.4. On malariasim, the migration script plus the hand edits it lists leaves 85 of 284 tests failing: 72 from its own `ss.Disease` → `ss.Infection` rewrite (MA22 again), 9 from a rename it made in non-Starsim code, and 4 it doesn't flag. With 16 more lines by hand, all 284 pass with no shim.

## The port today

| Task | Oct 17 (`1a5b7a77`) | Oct 21 (`ee5d1fb0`) |
|---|---|---|
| malariasim unchanged | 123 of 284 tests fail | Refused at import (`MalariaHumanBase defines set_prognoses ... subclasses ssl.AgentSIS instead`): no test runs, but the message names the fix |
| With the smallest shim | 27 lines; 273 pass, 11 fail | Rewritten, 24 lines (the bare coded base is now built on `DiseaseBase`); 273 pass, the same 11 fail |
| With the full shim | 284 pass | 284 pass |
| `python -m starsim.interop.migrate` plus its listed hand edits | n/a | 199 pass, 85 fail (MA41); with 16 more lines by hand, all 284 pass with no shim |
| Unchanged malariasim against Starsim 3, 9 human × mosquito combinations × 5 seeds | Within 1% | Within 0.7% and 1.5 SE in all 9; ODE humans bit-identical to Oct 17, agent humans on a new random path |
| A vaccine or net's protection on an EIR or mosquito-count infection | No effect | Applied |
| Mass drug administration | Reaches nobody | Reaches the infected |
| Mosquito counts at a monthly step | Extinct as agents and chain-binomial | Right as agents; still extinct as chain-binomial |
| A fixed extrinsic incubation period | 7,576 / 5,092 / 1.8e17 (agents / ODE / binomial) | 7,576 / 5,020 / 5,020, with a note that agents take it as exponential |
| `ss.Births` with nodes | Every newborn in node 0 | Newborns at the node split |
| Box-then-decay vaccine or bednet as an ODE | Runs, within 0.5% of agents (after MA16) | Refused (MA38) |
| ITN redistribution, boosters | Round 2 reaches 600 of 3,237 | Unchanged: 603 (MA15) |
| A net's killing in a rate | Agents only, through a function | Unchanged (MA14) |
| Biting-weighted count, several species | No spelling; silently level 0 | No spelling; level 0 now refused |
| `R0()` | 0.0 | 8.78, against 12.42 (MA37) |

## New findings

### MA34 (silent): `init=` reads 1.0 as one person

`Engine.init_fractions()` (`solvers/engine.py:1066`) reads an initial value of a state of people as a number of people when it is `>= 1`, and a share below that. The same function reads `init_prev` the other way: "more than 1 is a number of people, and up to 1 a share (1.0 is everyone)". So, on a plain SIR with 10,000 agents (`v4port/oct21/probes/init_rule.py`):

| `init=` | Start (S / I / R) | Message |
|---|---|---|
| `dict(S=1.0)` | Error: "sum to 0.0001, not 1" | It says to give the rest a state, so the user writes the next line |
| `dict(S=1.0, I=0.0)` | 1 / 0 / 9,999 | A note: "the rest (1) start in R" |
| `dict(S=1, R=0)` | 1 / 9,999 / 0 | A note |
| `dict(Sh=1.0, Sv=5000)` (vector) | 1 / 9,999 in `Ih` | A note |

The second row sums to exactly 1, the documented rule for shares, and is still read as counts. The note's "(1)" reads as "the whole of the rest" rather than "one person". On Oct 17 `S=1.0` meant everyone. Six of the port scripts wrote it (`declared_seis.py`, `malaria_rm_v4.py`, `repro_count_order.py`, `repro_count_dwell.py`, `repro_count_rk4.py`, `repro_mosquitoes_become_people.py`), and in `malaria_rm_v4.py` it was the workaround for MA3's `IndexError`. They now run with almost everyone infected at the start: the coupling-order probe shows 0 infections in every step, and the interventions baseline moves from 2,522 to 2,584 infected at day 364.

Fix: use `init_prev`'s rule (above 1 is a count, up to 1 a share), or read a dict whose people values sum to 1 as shares; and in the note, say "the other 9,999 people". A few lines. The spec (`03-states-and-transitions.md`, item 3) should state the rule with the 1.0 case.

### MA35 (silent): a forgotten `people=False` isn't caught

The spec now says every state is people unless declared a count, and that "a forgotten `people=False` is caught when the module is built" (`03-states-and-transitions.md`, "Saying which they are"), since a transition between people and a count raises `PeopleAndCounts` and a bare count raises `ExposureToCount`. Ross-Macdonald written the natural way trips neither: the mosquito states connect only to each other and are read through `.n`. On the spec's own model without `people=False` (`interventions/repro_mosquitoes_become_people.py`, 5,000 agents):
- `init=None`: humans and mosquitoes are all people; emergence adds mosquito agents, so after 100 days there are 75,221 "people alive" and `n_alive` counts the mosquitoes.
- `init=dict(Sv=3000)`: 3,000 of the 5,000 agents start as mosquitoes and 2,000 as humans, and the run is silent.
- `init=dict(Sv=50000)` or `dict(Ih=0.01, Sv=50000)`: `ValueError: The initial fractions of "malaria" sum to 10.01, which is more than 1`, which doesn't say why.

With `ss.states(3, people=False)` every case is right, and the infections match Oct 17 (7,424 as agents, 7,431 as an ODE). The rule is the right one (it ends MA3 and MA5's guessing); what's missing is the check the spec promises. Fix: refuse, at build, a group of people states that no transition connects to the disease's first state and that only an inflow from `ss.null` fills (a population of something else), naming `people=False` and, for intended people, `people=True`. And say "if these are not people, declare them `ss.State(people=False)`" in the "sum to more than 1" error. About 15 lines. This supersedes MA3 and MA5.

### MA36 (silent under the ODE): the solvers disagree on acting on counts

`interventions/probe_count_targets.py`, mosquitoes at day 100 (4,000 untouched, about 2,000 if halved):

| Intervention | Agents | ODE |
|---|---|---|
| `ss.change(mal.emergence, at=..., by=0.5)` | 2,013 | 4,000 |
| `ss.change('mal.pars.em', by=0.5)` | 2,013 | 2,013 |
| `ss.Vaccine(efficacy=dict(mortality=-1))` (a mosquito transition) | `CapabilityError`: a modifier of a count transition can't take a condition on people | 4,000, no message |

On Oct 17 the first and third were ignored under both solvers. Now agents do the right thing and the ODE silently doesn't. Fix: apply `ss.change` to an inflow in the compartmental engine, and raise the agents' `CapabilityError` there too. About 15 lines.

### MA37 (silent): `R0()` is off by √2 for Ross-Macdonald

`malaria.R0()` on the spec's Ross-Macdonald model (`vector/repro_R0_vector.py`, with `people=False`) is 8.78; the next-generation value is √154.35 = 12.42. The matrix (`R0(matrix=True)`) has 1.225 human infections per infectious mosquito (right: a c σ/(σ+μ)/μ) but 63 mosquito infections per infectious human, where a b M/(N r) = 126 with M = 70,000 mosquitoes at the disease-free equilibrium. It is 63 whatever `init=` gives `Sv` (0, 10,000, 70,000 or 140,000), so `next_generation()` takes half the equilibrium count. A 29% understatement, silently, on the spec's own example. Fix: the equilibrium of a count filled by an inflow is the inflow over its total exit rate; add a conformance test against the analytic value. Small.

### MA38 (capability, a decision): waning by a function has no compartmental form

The simplification (S3) kept waning by a function for agents only. malariasim's products all wane that way: a box of full protection, then exponential decay (vaccine, the four bednet components, drug prophylaxis). So the declared malaria model with any of them can't run as an ODE: `CapabilityError: the protection of "vaccine" fades by a function of the time since delivery, which only agents keep` (`interventions/run_v4.py`, `run_v4_rm.py ode`, `probe_ode_waning_stages.py`). On Oct 19, MA16 had made it accurate to within 0.5% of agents (10 to 40 levels of time since delivery, chosen from the waning). malariasim has both an agent and an ODE human, and runs every intervention under both, so this is a loss of something the downstream model does today. Two options:
- keep it agents only, and say in `starsim3.md` that malariasim's ODE interventions need waning as a duration (an exponential), or stay coded;
- restore the compartmental form for the shapes that are a box followed by a duration (a delay and then an exponential), which are a conveyor and an exit rate, both already in the kernels.

The second keeps malaria's ODE interventions declared and is about 40 lines. This needs a decision.

### MA39 (speed): declared agents with counts are slower

Old v4 (`1a5b7a77`) against new (`ee5d1fb0`), each in a fresh process after a warm-up run, interleaved, best of three rounds (`v4port/oct21/bench/`). The machine was shared (load 18 to 31 on 20 cores), so take differences under about 30% as noise:

| Case | Oct 17 | Oct 21 | |
|---|---|---|---|
| Declared SEIS humans as agents + mosquito counts, 10k, 2 years | 2.19 s | 4.19 s | 1.9× slower (1.5× in a quieter first try: 2.16 against 3.30 s) |
| The same as an ODE, 20 runs with varied `b` (as a fit) | 0.086 s a run | 0.093 s | Same |
| Declared human SIS, 3 patches × 20 age groups, seasonal EIR per patch, births and deaths by age, ODE, 5 years | 1.53 s | 0.67 s (0.54 s compiled first) | 2.3× faster (MA27) |
| The same as agents, 5,000 | 2.16 s | 1.62 s | 1.3× faster |
| Unchanged malariasim, SEIS + ODE mosquitoes, 10k, 2 years | 0.95 s | 1.09 s | About 15% slower; profiled, 2.24 against 2.50 s |

- For declared agents with counts, `Agents.step_counts()` is 41% of the run (7.5 of 18.3 s profiled): four RK4 derivative calls a step in Python, each walking the rate expressions, with heavy `sc.odict` attribute access (433,000 calls). With a time series in a count rate it's worse: `parameters.py: series()` converts every stored value at every derivative call (5.4 million generator steps in a 5-year run, 20.5 of 32.7 s profiled; `vector/prof_hybrid_v4_abm.py`). That's MA28, which grew when counts gained substeps. MA28's fix stands: step counts through the compartmental kernels under every solver, and tabulate a series once per step.
- For coded models, the clearest new per-call cost is `Arr.__setitem__`: 22,322 calls take 0.29 s against 0.16 s, from the copy-on-write check (`if self.shared is not None: self.own()`) added with shared arrays. malariasim writes per-agent arrays from Python in every step, as most coded models do.

Neither was measured in S0.3's benchmark set, which has no counts and no coded model. Suggest adding a declared model with counts and malariasim's `fseis10` scenario to it.

**Startup.** Numba's cache is keyed by file path, so a fresh install or worktree compiles on first use. With `verbose=0`, as scripts and fits use, nothing is printed: the human-ODE fit took 65–80 s cold and 7.1 s warm, with no message. The plan's S5.11 says "the sim says so"; it should say so at `verbose=0` too, since a minute of silence looks like a hang.

**Calibration.** The fit now runs three searches from separate scan points to check convergence: on the one-parameter deterministic human ODE, 89 evaluations against 35 on Oct 17, 7.1 s against 3.2 s, for the same answer (0.4004). On malariasim's coded ODE, 89 runs against 23. That's a deliberate cost for a real check, but for one parameter and a deterministic model a bracketing root find (or one search) would do; consider skipping the extra searches when there is one parameter and `n_runs=1`.

### MA40 (crash): the Sciris floor is too low

`ss.parallel()` passes `start_method=` to `sc.parallelize()` (`run.py:616`), which Sciris 3.2.9 hands on to the run function: `TypeError: single_run() got an unexpected keyword argument 'start_method'`, for any sim. `pyproject.toml` asks for `sciris>=3.2.8`; it works on 3.4.0. Raise the floor to 3.4.0. One line.

### MA41 (migration): the migration script on malariasim

`python -m starsim.interop.migrate malariasim tests` made 21 changes in 5 files and listed 8 hand edits. That covers most of what the shim does: `ssl.AgentSIS`, results at `self.ti + 1`, the analyzer's first point, and `.value` for `float()`. After those, 85 of 284 tests fail (`v4port/oct21/migrate/`):
- **72 from its own rewrite of `ss.Disease` to `ss.Infection`** (MA22): `ss.Infection` defines `susceptible` and `infected`, which malariasim's ODE defines too, so `define_states()` raises `Another result named "n_susceptible" already exists`. There's still no bare coded base (the shim builds one on `ss.diseases.DiseaseBase`, which isn't public). Fix: a public bare coded base (or `ss.Infection(states=False)`), used by the script and named in the error that `ss.Disease` raises. About 15 lines.
- **9 from renaming code that isn't Starsim's.** `eligible=` → `where=` (a Starsim 4 prerelease keyword) was applied to a helper of malariasim's own tests, `_make_for_step(..., eligible=None)`, and broke it (`UnboundLocalError`). Restrict the rule to calls of Starsim deliveries.
- **4 it doesn't flag:** a coded `set_prognoses()` that doesn't call `super()` (v4 now warns at run time, but the script could list it). For malariasim the fix is `ss.Infection.set_prognoses(self, uids, sources)` rather than `super()`, since its SEIS override must not go through the SIS one.
- **1 it half-does:** it moves results written in `step()` to `self.ti + 1`, but then nothing writes index 0, the start, so the ODE's prevalence there is 0/0. The row should say to write the initial point (at `self.ti == 0` in `step()`, or in `init_post()`).
- Four of its eight hand-edit items are false positives (`float()` of NumPy arrays in tests). That's harmless, but it's noise.
- `ss.prob.array_to_prob()` is gone with no migration row; malariasim's `FastBirths` (branch `Interventions2-ck`) calls it, copying Starsim 3's `ss.Births`. `ss.prob(rate, ss.years(1)).to_prob(dt)` replaces it.

Checked: the script, its listed hand edits, reverting the test rename, those two lines for `new_infections`, three lines for the initial point, and a 10-line bare base in place of `ss.Infection` take malariasim to all 284 tests with no shim. That's 37 lines added and 18 removed in 7 files (`v4port/oct21/migrate/migrated_full.diff`). With the fixes above, the script would do all of it but the bare base's one line, which is the goal the downstream reviews set: a migration that is a script, and code that reads as v4.

### MA42 (messages, small)

- `DuplicateDeclaration` for states reused in a second disease says "Declare this disease first, and then ... with a rate that reads its state", which misleads when the cause is reusing `S, I` objects.
- `MissingDimension` on `ss.by(biting=w) * Ih.n` (MA10's refusal) says "Stratify Sv by biting too". For biting weights that's the wrong advice: mosquitoes have no biting level, and what's wanted is the weighted count (MA11). Name MA11's spelling there once it exists.
- The ITN refusal names it "vaccine" (`the protection of "vaccine" fades...`), since a net has to be an `ss.Vaccine` (MA33).
- `describe()` still says "10,000 people [guessed]" for a given `People` (FP38's rest).

## Status of the Oct 17 findings

| ID | Finding (Oct 17) | Status Oct 21 | Evidence |
|---|---|---|---|
| MA1 | `rel_sus`, scalar efficacy do nothing on a vector infection | Fixed | 7,424 → 0 (agents), 7,431 → 0 (ODE) for `modify(rel_sus)`, the vaccine and `modify(infection)`; `human/repro_relsus_vector.py`, `human/probe_relsus.py` |
| MA2 | `ss.treat` reaches nobody | Fixed | Default `ss.treat`: 2,145 delivered of 2,373 eligible, n_I 2,373 → 278; `interventions/probe_mda.py`, `probe_deliveries.py` |
| MA3 | `init=` decides who the agents are | Superseded | Every state is people unless `people=False`; but see MA35 |
| MA4 | `R0()` 0.0 | Partly | 8.78 against 12.42 (MA37) |
| MA5 | `people=True` mosquitoes in `People` | Superseded | Now the default unless `people=False` (MA35) |
| MA6 | Counts extinct at long steps | Partly | Agents: 50,000 at dt 21/28/30/30.42 days. Chain-binomial at dt 30: 10,000 → 5,000 → … → 2 (`vector/repro_count_binomial.py`) |
| MA7 | Fixed incubation: three answers | Partly | ODE and binomial 5,020 (exact); agents 7,576 (exponential), now with a note. The malariasim model with a fixed EIP: prevalence 0.246 as ODE, 0.476 as agents (`vector/ms_mosquito_v4.py`) |
| MA8 | Modifiers on counts ignored as agents | Fixed (agents) | Doubling mortality: 2,000 under both. The ODE now ignores `ss.change` on an inflow (MA36) |
| MA9 | Count-only infections before counts move | Fixed | First step 96 (agents) against 97.5 (ODE); at dt = 7 agents −1.6% against the ODE (was −3.8%) |
| MA10 | Value by a missing dimension takes level 0 | Fixed | Refused (`MissingDimension`, `InvalidAdjustment`) |
| MA11 | No weighted count | Open (deferred) | `deferred.md`: "waits for the malaria port" |
| MA12 | Coded per-node EIR for both solvers | Open (deferred) | `human/probe_external_eir.py` unchanged |
| MA13 | `.n` vs `.count()`; N = 0 in `init_post()` | Fixed | Both work under both solvers |
| MA14 | Coverage not a quantity | Open (deferred) | Unchanged |
| MA15 | No re-dosing with function waning | Open (deferred, needs a spelling) | Round 2 delivers 603 against 3,237 (`interventions/repro_no_redose.py`) |
| MA16 | ODE waning in 10 levels | Superseded | Fixed on Oct 19, then waning by a function lost its compartmental form (MA38) |
| MA17 | Function parameter evaluated ~3× per use | Partly | 2.0 calls a step for 3 rate terms (was 9.0) |
| MA18 | Newborns all in the first patch | Fixed | Node shares hold (`population/test_v4_population.py`) |
| MA19 | `ss.Births` reads a rate as a probability | Open | 16,875 ± 22 against 16,686 ± 33 (`ss.birth`) |
| MA20 | Death-rate ages tied to the age dimension | Partly | A death rate as a function of age works; 100 values on 20 groups is still `ShapeError` |
| MA21 | ODE aging at 1/width | Won't do | `deferred.md` (summer2's convention) |
| MA22 | No bare coded base | Open | Error still suggests `ss.Infection`, which clashes; the migration script rewrites to it (MA41) |
| MA23 | `ss.SIS` subclass fails late | Fixed | Refused at class creation, naming `ssl.AgentSIS` |
| MA24 | Analyzers run before the first step | Open | `framework/diag_order.py` identical; 6 suite failures without the full shim |
| MA25 | `ss.diff` crashes on coded modules | Fixed | `framework/repro_diff_coded.py` |
| MA26 | 4 seeds for a deterministic coded fit | Fixed | 1 seed; but 89 runs from the three searches (MA39) |
| MA27 | Typed values converted per group per step | Fixed | `TimePar.to` 0% of the profile; the population ODE 2.3× faster |
| MA28 | Counts stepped in Python | Open, worse | 41% of a declared agents run; `series()` per derivative call (MA39) |
| MA29 | `describe()` crashes on ODE with `People(age_data=)` | Open | `AttributeError: 'NoneType' object has no attribute 'place'` (`sim.py:886`) |
| MA30 | Strata lost through affine results | Open (deferred) | `human/probe_observables.py`: same failures; the weighted case now `setting an array element with a sequence` |
| MA31 | `.by()` on a `MultiSim` can't plot | Fixed | `results.Levels.plot()` |
| MA32 | `people_results=False` crashes demography | Fixed | Runs with `ss.Births` and `ss.Deaths` |
| MA33 | No neutral held product | Open (deferred) | Bednet still an `ss.Vaccine` |
| ST11 | No public way to end an infection | Fixed | `Disease.move_to()` |
| FP1 (last point) | Coded results at `ti` leave the last point 0 | Open | `framework/repro_last_point.py` identical, no warning; the migration script now rewrites it |
| FP1 (via v3) | `new_infections` 0 without `super()` | Improved | v4 warns, naming `super().set_prognoses(uids, sources)` |
| R8 | `sim.hash()` misses `NodeGrid` | Open | EIR 10 and 100 hash the same |
| N20 | Array parameter can't be fitted | Open | Same `TypeError` and `NothingToFit` |
| D2 | Bare times in `ss.timeseries` | Open | Flat at `start=2000` with times in days |
| L1 | `unit='day'` with `start=2000, stop=2002` | Open | Identical |
| FP31 | `ss.peryear(1) * ts`, `* fn` | Fixed | Both are expressions |
| FP38 | `describe()` drops units | Partly | Units and `ss.seasonal` printed; "[guessed]" for a given `People` remains |
| FP3 | `resample('year')` gives 4 years | Fixed | 3 bins |
| X10 | Leaving the whole-run kernel is silent | Fixed | `describe()`'s `plan` line says why |

## Keep

- **The rule that the transitions out of the first state are infection**, under every solver: it fixed MA1, MA2 and MA4's zero in one place, and `describe()` now lists the infected states.
- **Every state is people unless declared a count.** It's simpler and more honest than guessing, once MA35's check is in.
- **The class-creation refusal for coded hooks on declared classes**: unchanged malariasim fails at import with the fix named, instead of 123 tests failing in five different ways.
- **The warning for a coded `set_prognoses()` that skips `super()`**, which caught malariasim's Starsim 3.6 regression.
- **The migration script**, which already covers most of the shim; with MA41's three fixes it would cover all of it.
- **Counts as agents with substeps, and conveyors for fixed delays in the compartmental solvers**: the mosquito dynamics no longer depend on the step.
- **The declared ODE's speed with places, ages and seasonal series** (2.3× faster).

## Suggested order

1. **MA34 and MA35**, both silent and both hit by every port script: a few lines each, plus the "sum to more than 1" message naming `people=False`.
2. **MA36 and MA37**: the ODE ignoring `ss.change` on an inflow and a product on a count transition, and the √2 in `R0()`. Small, with conformance tests on the spec's Ross-Macdonald model.
3. **MA40 and MA41**: the Sciris floor (one line), a public bare coded base used by the migration script (closes MA22), and the two rule fixes, then malariasim's suite with the migrated code as a downstream check.
4. **MA38**: decide whether box-then-delay waning keeps a compartmental form.
5. **MA39 and MA28**: counts through the kernels under every solver, and the copy-on-write check out of `Arr.__setitem__`'s hot path; add a counts model and malariasim's `fseis10` to the benchmark set; print the first-use compile at `verbose=0`.
6. The rest of the chain-binomial MA6, then the deferred vector vocabulary (MA11, MA12, MA14, MA15, MA33), which the malaria port still needs for a fully declared model.
