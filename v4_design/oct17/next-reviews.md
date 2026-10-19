# Next reviews

The four code-reading reviews mostly re-found one another's problems. The switching seats instead did a real user's work at that user's scale, and found 29 problems none of the reviews had, including most of the top of the ranking in [README.md](README.md#every-finding-ranked). This file proposes five more reviews of that kind. Each one targets a class of problem that reading the code cannot see, and each leaves tests or docs in the repo rather than only a report, so it keeps paying after it is done.

| # | Review | Finds | Leaves behind | When |
|---|---|---|---|---|
| 1 | [Invariance fuzzing](#1-invariance-fuzzing) | Silent wrong answers that break a symmetry the user assumes | A model generator, a shrinker, and regression tests | Now |
| 2 | [A statistical and scaling audit](#2-a-statistical-and-scaling-audit) | Biased inference, mismatched stochastic solvers, and superlinear costs | A coverage table, a scaling table, and CI budgets | Now |
| 3 | [The textbook canon](#3-the-textbook-canon) | What v4 cannot express or gets wrong across standard epidemiology | A gallery of the canon, as both docs and corpus | Now |
| 4 | [The naive user](#4-the-naive-user) | Where a newcomer, or an assistant writing for one, goes wrong | Measured learnability, and fixes to docs and messages | After the two short doc pages (phase 6) |
| 5 | [Real models at full complexity](#5-real-models-at-full-complexity) | The long tail of real studies, and what the Starsim 3 bridge needs | Replicated studies and one ported package | After one language (phase 5) |

Rules shared by all five:
- Every failure reduces to a minimal repro, gets a severity using the tiers in [README.md](README.md#every-finding-ranked), and is checked against [findings/](findings/) and [new-findings.md](new-findings.md) so known problems are not counted as new.
- Every confirmed failure becomes a test (an `xfail` until it is fixed).
- The work runs with `/software/conda/envs/v4/bin/python`, and any timing is a repeated warm run with the kernel build reported separately.
- Read-only on `starsim/`. Findings go to the review's own folder.

## 1. Invariance fuzzing

**The idea.** Generate many small declared models, change each one in a way that should not change the answer (or should change it in a known way), and check that the answer holds. This is a property test of the guarantees a user takes for granted.

**Why it is first.** At least a dozen of the known silent wrong answers break an invariance of exactly this kind, and they were found by luck or by an expert's hunch. All of these would fall to a generator:

| Invariance | Found by hand |
|---|---|
| Renaming a dimension changes nothing | N4 (`patch` to `zone`) |
| The step size changes results only by the discretization error | L1 (`dt` sets the unit) |
| Declaration order changes nothing | L4 (who is people), X5, X6 (exit order) |
| An unrelated module leaves the draws unchanged | X3 |
| `init()` before running changes nothing | N5 |
| `sim.new()` reproduces the sim | N18 |
| `to_code()`, `save()`, then `load()` reproduces the model | R10, R15 |
| `1` and `1.0` mean the same | L17 |
| Two transitions from one partial are two objects | L7 |
| Stratifying with one level changes nothing | D11 |

**Method.**
1. **A generator** (`tests/fuzz/generate.py`). It draws models from a small grammar:
    - 2 to 6 states, with transmission, progression, branching, waning, and outflow;
    - rates as constants, parameters, products, `ss.seasonal`, `ss.timeseries`, and functions;
    - 0 to 2 dimensions (age, place, risk), with or without contact matrices and mobility;
    - 0 to 2 deliveries and modifiers;
    - optional data and outcomes.

  The generator is seeded, and it writes each model as code, so a failure is a script.
2. **The transformations,** each with its expected effect:
    - renames of states, modules, and dimensions;
    - permutations of declarations;
    - the same values in other units;
    - `dt` halved (results converge, within a Richardson-style bound);
    - an unrelated analyzer or disease added;
    - a one-level stratification;
    - `init()` before `run()` or `ss.parallel`;
    - round trips through `sim.new()`, `to_code()` and exec, and `save()`/`load()`;
    - integer against float literals;
    - equivalent spellings (string, operator, and class forms; `dur=d` against `rate=1/d` under the ODE);
    - the same model under `ode`, `binomial`, `gillespie`, and `abm` at a fine step.
3. **A comparison.** Deterministic paths compare to 1e-10. Stochastic paths compare draws exactly where common random numbers promise it, and distributions otherwise.
4. **A shrinker.** It deletes states, transitions, dimensions, and modules while the failure persists, and writes the smallest model that still fails.
5. **Triage.** Group the failures by root cause, match them against the registers, and file the new ones.

**Scale.** Start with a few thousand models per transformation. The generator, the transformations, and the shrinker are a few hundred lines. Two or three agents: one builds the harness, and the others triage failures by area.

**Leaves behind.** `tests/fuzz/` with a fixed-seed smoke set in CI, a longer run under a marker, and one regression test per root cause in `tests/corpus.py`. This builds most of phase 2 of [implementation.md](implementation.md#phase-2-contract-tests) and finds bugs with it at the same time.

**Risks.** Tolerances for stochastic and `dt` comparisons need care, or the review drowns in false alarms. Use a fine step and the dt-bias rule from the plan, and report a mismatch only when it survives a doubled sample.

## 2. A statistical and scaling audit

**The idea.** Measure, systematically, the two things users trust without checking: that inference and the stochastic solvers are statistically right, and that cost grows the way it should.

**Why.** N1 and N2 (the particle filter and the JAX density) were found only because two seats happened to write a seasonal times time-varying `beta` with a fitted multiplier. The filter's own test is skipped, and no review ran simulation-based calibration. N23, N24, and N28, which are the cheapest scale bugs to fix, were also found by accident, each by one seat at one size.

**Method, accuracy.**
- **Simulation-based calibration.** For each method (Nelder-Mead, L-BFGS-B, Optuna, NUTS, PMCMC, and ABC-SMC and IF2 once built), on each small corpus model with data:
    - draw parameters from the prior, simulate data, and fit;
    - posterior methods: check rank uniformity and interval coverage;
    - optimizers: check recovery, bias, and how often `converged` lies.
- **Likelihoods against the run.**
    - The JAX log-density against `sim.loglik()` at random points.
    - The particle filter's estimate against the exact likelihood in the ODE limit, and its variance against particle count.
    - Every rate form in the corpus: constant, seasonal, time series, fitted multiplier, function, and modifier.
- **Stochastic solvers against exact Gillespie.** Compare the whole distribution (final size, peak, peak time, and extinction probability), by KS or Anderson-Darling tests at a fine step. Then measure the bias at coarse steps, so the notes can state its size (X4).
- **Convergence in `dt`.** For each solver, the observed order of convergence against the stated one.

**Method, cost.**
- Scaling curves along one axis at a time: agents, groups, places, ages × places, diseases, steps, runs, particles, data length, and register events.
- Measure time, peak memory, and build time, at five or more sizes per axis, for each solver that applies.
- Fit the exponent. Flag anything superlinear where it should be linear, and anything whose build time exceeds its run time.
- Compare against the seats' specialist numbers where they exist (LASER, SimInf, MEmilio, odin, camdl).

**Scale.** Two agents (accuracy and cost), mostly compute time. The SBC grid is the long pole; run it in a background workflow.

**Leaves behind.**
- A coverage table and a scaling table for the docs. These are the honest version of the north-star time claim (P8).
- Fast SBC and likelihood-equivalence tests in CI.
- A budget per axis in `tests/perf_guard.yaml`, so a superlinear regression fails a test.

**Risks.** SBC on stochastic agent models is expensive; restrict it to small populations and the compartmental solvers, with one agent-based case under a marker.

## 3. The textbook canon

**The idea.** Reproduce the standard teaching corpus of infectious disease modeling in v4, from the books' descriptions, with the original programs' outputs as the reference.

**Why.** The vignettes test one showcase model per framework, chosen to show what that framework is for. The canon tests breadth across what most modelers are taught, and it includes models no vignette has:
- vectors and multiple hosts;
- stochastic extinction and critical community size;
- seasonally forced chaos;
- metapopulations with explicit movement;
- partial cross-immunity, including the exact 2^k multistrain form that D13 lacks;
- age structure with maternal immunity;
- sexual-network models with core groups.

A failure here is a gap in standard epidemiology, which is the strongest argument there is for fixing it.

**Sources.**
- The companion programs to Keeling and Rohani, *Modeling Infectious Diseases in Humans and Animals* (2008), published online chapter by chapter.
- The model files for Vynnycky and White, *An Introduction to Infectious Disease Modelling* (2010).

**Method.**
1. Inventory every program, with its chapter, what it shows, and its outputs (run the original where it runs, otherwise digitize the book's figure).
2. Write each in v4 the way the book describes it, not by translating the code line by line.
3. Score each one:
    - lines against the original;
    - agreement (to 1e-6 for deterministic models, and distributional for stochastic ones);
    - the solver `auto` chose;
    - whether `describe()` reads like the book's description;
    - what v4 could not express, and the workaround.
4. Then run each model under the other solvers, to show the one-model-many-solvers claim across the canon.

**Scale.** Roughly 40 to 60 programs. Three to five agents, split by chapter group.

**Leaves behind.**
- A gallery in `docs/` with each model next to its book reference. This is the best possible "v4 in practice" page, and it serves teachers.
- The models in `tests/corpus.py`, where they also feed reviews 1 and 2.
- Gaps filed against [implementation.md](implementation.md#phase-7-what-the-seats-need).

**Risks.** Copyright: link to the books and their programs rather than copying text or figures. The book's numbers come from the original programs or are cited, not reproduced verbatim.

## 4. The naive user

**The idea.** Agents with no access to the source code or the spec, only what a real newcomer has, do modeling tasks stated in epidemiological terms, with reference answers hidden from them. Measure how often they get the right number, and what goes wrong on the way.

**Why.** Every reviewer and every seat had expert or insider knowledge. That knowledge hides problems. Fable 2 reported `sir.age < 15` as working because it knew what to expect (N3). The front door failed (P1, L5) without any reviewer meeting it as a newcomer would. Many v4 users will write models through an AI assistant, so `llms.txt`, the docstrings, and the error messages are part of the product, and nothing tests them.

**Personas.** Each one sees only `docs/`, `llms.txt`, docstrings, and error messages:
- **The careful novice.** A public-health analyst who reads the docs first.
- **The habitual user.** Someone who tries the first plausible spelling, and the spelling from their old framework.
- **The assistant.** An LLM coding agent writing code for a user from `llms.txt` alone, as Claude Code or Copilot would.
- **The Starsim 3 user.** Only the migration page, and their own v3 habits.

**Tasks.** Twenty to thirty, each with a hidden reference answer computed by an expert solution:
- "Estimate R0 from these weekly cases."
- "What does closing schools in March buy, with 90% intervals?"
- "Vaccinate children under 5 at 80% coverage starting in June."
- "Fit this model to hospitalizations by age."
- "Make it run on 10 million people."

**Measures.**
- Correct, wrong, or gave up, per task and persona. A **wrong answer with no warning is the main measure.**
- Time and tool calls to a correct answer.
- Every error met, with whether its message led to the fix.
- The first spelling tried for each concept, which is evidence for the naming decisions.
- For the assistant persona, how many of its mistakes `llms.txt` caused or could have prevented.

**Scale.** Four personas by twenty-five tasks, run in parallel. Scoring is automatic against the hidden answers, with a human-readable log of every wrong turn.

**When.** Once the two short doc pages exist (phase 6). It is the test of whether they work, and it should be rerun after each docs phase as a learnability benchmark.

**Leaves behind.**
- The tasks as a standing benchmark with a score.
- A ranked list of message and doc fixes.
- Evidence for the open naming decisions, from the spellings people reached for first.

## 5. Real models at full complexity

**The idea.** Take real, published analyses to the end in v4: the whole model, the whole calibration, and the published figures. Separately, port one downstream package that real users run.

**Why.** The seats did a second week of routine work. Neither they nor the reviews met the long tail that real studies have:
- continuous per-agent state (the CD4 example in [switching/starsim3.md](switching/starsim3.md));
- hundreds of parameters, and calibration to many targets at once;
- custom analyzers and results pipelines;
- runs that take hours.

This is where decision 2 (one language, with coded modules first-class and the v4 verbs bridged) gets settled by evidence rather than argument. tbsim's port is the precedent.

**Method.**
- **Replication.** Choose 5 to 8 published analyses with public code and data: two or three from Starsim 3's own studies (HPVsim, STIsim, FPsim, Covasim), and three to five from the comparison frameworks' papers (an EpiModel HIV prevention study, a SimInf livestock study, a flepiMoP or MEmilio scenario study). For each:
    - rebuild it in v4 from the methods section, using the original code only to resolve ambiguity;
    - reproduce the headline figures and the calibration;
    - record lines, run time, agreement, and every place where v4 needed a coded module.
- **Ecosystem port.** Port one Starsim 3 package, most likely STIsim or FPsim, with tbsim's rule: change the package freely, and change v4 where that makes a feature fit the framework rather than adding a package-only workaround. Record each change to v4 and the reason for it.

**Scale.** One agent per study, and one longer effort for the port. This is the largest of the five.

**When.** After phase 5 (one language). Today it would mostly rediscover the two-system problem.

**Leaves behind.**
- Replication scripts with reference figures. The small ones go in `tests/`, and the rest in LFD next to the vignettes.
- A ported package on a branch.
- A list of v4 changes that real models forced, which is the best evidence for what the core needs.

## How they fit the plan

| Review | Feeds | Depends on |
|---|---|---|
| 1. Invariance fuzzing | Phase 2 (contract tests) and phase 3 (silent wrong answers) | Nothing |
| 2. Statistical and scaling audit | Phase 2, phase 3 (inference), and phase 8 (performance) | Nothing; ABC-SMC and IF2 join when built |
| 3. The textbook canon | Phase 6 (docs gallery) and phase 7 (features) | Nothing |
| 4. The naive user | Phases 4, 6, and 9 (guessing rules, docs, surface) | The two short doc pages |
| 5. Real models at full complexity | Phase 5 (one language) and phase 7 | Phase 5 |

Reviews 1 and 2 are the most valuable to run first: they need no decisions, they go straight at the trust problem (74 silent wrong answers), and what they build is the test suite the rest of the plan needs.
