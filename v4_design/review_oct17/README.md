# Starsim v4: the four reviews merged, and would anyone switch?

Written 2026-10-17, at commit `a20c4f7e` on the `v4` branch of `ssv4` (unchanged since all four reviews were written). Two pieces of work are here:

- **The synthesis.** The four reviews written today ([codex_review/](codex_review/) "C", [fable_review/](fable_review/) "F", [fable2_review/](fable2_review/) "F2", [opus_review/](opus_review/) "O") are merged area by area into [findings/](findings/). Several hundred source findings become 162 (L1–L38, D1–D45, X1–X30, R1–R34, P1–P15). Each was re-checked where only one review raised it or where the reviews disagreed. Every silent wrong answer reproduced. The corrections are small and are listed in each file.
- **The switching test.** Thirteen assessments in [switching/](switching/) take the seat of an experienced user of each framework in the comparison set, plus Starsim 3's own users. Each one reads the vignette, does that user's routine second-week tasks in v4 at their real scale, and judges whether they would switch, what stops them, and whether each gap is fixable or a real tradeoff. These seats found about thirty bugs and gaps that no review had. They are in [new-findings.md](new-findings.md), and I re-ran the eight most serious myself.

[implementation.md](implementation.md) turns both into a plan: what to change, where, in what order, and how each change is checked, with the thirteen switching demos as its acceptance test.

Nothing in the repo was edited.

## In one page

**The idea is the right one.** No other framework in the comparison set does what v4 does: one declared model runs as an ODE, a stochastic compartmental model, or agents, is fitted to data in one call, and explains itself with `describe()`. Every review says so, and so does every seat. On the first day the vignettes show it. Lines pass in 20 of 23 vignettes, often by a factor of 3 to 6: EpiModel's HIV model is 405 lines against 62, EMULSION's is 280 against 75, and odin's 51 against 10. Agreement holds wherever the models are the same, to 1e-12 against Atomica and to 6e-6 against summer2. Inference is a step change for most of these users. NUTS fit flepiMoP's benchmark in 70 s against emcee's 283 s on 10 cores. Particle MCMC in one call matched camdl's PMMH and covered the truth where SimInf's single chain missed. `calibrate()` got a 400 times smaller error than Atomica's own calibrator on its demo. One SIR runs warm under `ode`, `gillespie`, `binomial`, and `abm` in 0.01 to 0.11 s each, with peaks within 2% of each other (I checked).

**But today none of the thirteen would switch.** v4 wins the first day and loses the second week, in four ways:

1. **A silent wrong answer on the user's standard pattern.** Ten of the thirteen seats met one doing ordinary work. Among them:
   - **epydemix:** the particle filter is wrong for a seasonal times time-varying `beta`, and `sir.age < 5` compares the group index.
   - **summer2:** NUTS breaks on a fitted parameter times a time series.
   - **camdl:** naming a dimension `patch` throws away the contact matrix the user gave.
   - **Starsim 3:** a v3 `beta` dict ported verbatim is 12 times too small.
   - **Ghana measles:** "routine at 9 months" vaccinates everyone on day one.
   - **odin:** `ss.parallel` after `init()` gives identical runs.
   - **flepiMoP:** a Poisson likelihood on non-integer data is `-inf`, with no warning.

   This is the one that ends an evaluation, because once a user finds one, they stop trusting the rest.
2. **Speed and memory on the specialist's home ground:**
   - **LASER:** 3.3 million agents take 67.8 s against 2.75 s, at 11 times the memory per agent.
   - **SimInf:** 16,000 herds take 52.5 s against 1.9 s.
   - **MEmilio:** 400 counties × 6 ages take 25 s against 1.5 s, from a dense Kronecker matrix.
   - **odin's and camdl's particle filters:** 8 to 9 times slower, and serial.

   Most of this is fixable implementation, not inherent cost.
3. **The one feature the field cannot work without:**
   - **EpiModel:** degree by group.
   - **flepiMoP:** a fitted value per location.
   - **epydemix:** ABC-SMC and projections from the posterior.
   - **Atomica:** programs and budget optimization.
   - **EMULSION:** minimum stays and offspring.
   - **camdl:** one-off moves and IF2.
   - **LASER:** life tables and seeding.
   - **Starsim 3:** v4 verbs that reach their code.

   Each fits a mechanism v4 already has, and each is days to a couple of weeks.
4. **Reach:**
   - Every page in `docs/` is still Starsim 3.
   - No v4 model can be built from R at all (a frame lookup fails under reticulate), which shuts out the EpiModel, SimInf, and odin users at the door.
   - The importers, the obvious on-ramp, run silently wrong or crash on these frameworks' real models.

**The four reviews fix what is there; the seats show what is missing.** The reviews' fixes would remove most of failure 1, and their one-system and docs work would fix most of failure 4. None of them touches failure 2 or failure 3. After their fixes, three seats get close (Ghana measles, summer2, Starsim 3 with a bridge) and the rest still say no. After their fixes plus the gaps in [new-findings.md](new-findings.md), about ten of the thirteen would have a real reason to switch. The exceptions are LASER at full national scale, SimInf on a national register, and Atomica's non-programming analyst. Each of those needs either performance work that is larger but still within the design, or a product that v4 should not be.

## Would they switch? The scorecard

| Seat | What wins them on day one | What stops them in week two | Today | After the reviews' fixes | A clear win needs |
|---|---|---|---|---|---|
| [EpiModel](switching/epimodel.md) | HIV model 62 lines vs 405, 10–30x faster per run, no `netest` step, calibration in one call | No degree by group (`nodefactor`): risk structure silently vanishes (31%/31% vs 21%/77%); nothing builds from R | No | No for HIV network modelers | Per-group degree, several `match=` properties, `acts=` on `PartnerNet`; R construction |
| [SimInf](switching/siminf.md) | The register as a named table; no C; PMCMC with diagnostics; the same model as animals | 28x slower at 16,000 herds; no threads; R broken | No | No for register studies | Within about 3x of SimInf, with threads; ABC-SMC; long per-herd output; reactive movement bans |
| [odin](switching/odin.md) | 51 lines become 10; agents for free; observation and fit declared; honest diagnostics | Nothing builds from R; the particle filter is 8x slower and serial; identical runs after `init()` | No | No | R that works; a parallel filter within about 3x of dust; an importer for array models |
| [camdl](switching/camdl.md) | A third of the lines; same posterior with nothing written; agents with the same file | `patch=` discards the contact matrix; no unit checking; no one-off moves for SIAs; no IF2 | No | Closer, still no | A unit check at `init()`, `move=` deliveries, IF2, a filter within about 2x |
| [LASER](switching/laser.md) | One text as agents or compartments; half the code; interventions by keyword | 25x slower than LASER threaded at 3.3M agents; about 330 bytes per agent vs 30 | No | No | A fused, threaded agent step within 2–3x at about 50 bytes per agent; life tables and seeding |
| [Ghana measles](switching/ghana_measles.md) | One model for fitting and campaigns; a targeted campaign in three lines, checked | About 5 GB of results per run; "routine at 9 months" wrong; no all-or-nothing vaccine; `away=` cannot be fitted | Not yet | Close (with four small fixes) | Results at a reporting interval; the two vaccine fixes; maps by district |
| [MEmilio](switching/memilio.md) | The model is in the script, not C++; exact agreement; inference they lack | Dense (places × ages)² mixing: 16–21x slower at 400 × 6; per-place NPIs need a class; graphs don't import | No | No | Factored mixing; forward prior draws; a per-place threshold modifier with a hold; a graph importer |
| [flepiMoP](switching/flepimop.md) | Their config imports and agrees at 3,000 places, 5 s vs 34–58 s; NUTS 70 s vs emcee 283 s | No fitted value per location; Poisson `-inf` on non-integer data; every run kept in memory; no hub output | Yes as a fast runner, no for the pipeline | Closer | Vector parameters fitted per place, `priors=True`, results streamed per run, `to_hub()` |
| [summer2](switching/summer2.md) | A TB model in 25 lines matching to 6e-6; no per-model JIT wait; calibration in one call | NUTS breaks on a fitted rate times a time series; the summer2 importer fails on real models | No (maybe for a new project) | Closer, not clear | JAX NUTS for time-varying fitted rates; a live-model importer; the log-density as a JAX function |
| [Atomica](switching/atomica.md) | Exact agreement from their workbooks; a far better calibrator; a better optimum once one exists | No programs, no budget optimization, no spreadsheet path; v4 says it is not for them | No | No | Declared programs, `ss.optimize`, a generated databook (500–800 lines); not a GUI |
| [EMULSION](switching/emulsion.md) | 210x faster on their own metapopulation model; the trade register as data; calibration they lack | Minimum stays, offspring, and births between strata are workarounds or fail silently; their richer YAML imports silently wrong | No for published models; yes for new herd work | No | `after=`/`before=`, `offspring=`, inflows between strata, importer fixes, a readable export |
| [epydemix](switching/epydemix.md) | Products without copying compartments; 37 lines vs 161; solvers they lack; bundled data | The particle filter is wrong on their flagship model; `sir.age < 5` silently wrong; no ABC-SMC, no projections, no US states | No | No | ABC-SMC, `fit.project()`, US states, the two silent bugs fixed |
| [Starsim 3](switching/starsim3.md) | Their model runs unchanged and about 2x faster; one declared disease unlocks sweeps and `diff()` | 11 of 13 v4 features fail on their model; a verbatim beta port is 12x too small; results shift one step; docs all v3 | They upgrade, don't switch | They start porting, if there's a bridge | v4 verbs that reach coded modules; one-disease-at-a-time migration documented; declared agents as fast as code |

## What would make v4 a library people switch to

### 1. One demo, thirteen communities

Asked for the script that would make their user say "this is so cool", all thirteen seats wrote the same five steps in their own field's words:

```python
sim = ss.load('their_model.<ext>')        # bring the model you already have
sim.describe()                            # see exactly what it means, every guess marked
fit = sim.calibrate(data=...)             # fit it, with the method chosen and explained
ss.parallel(baseline=fit.sim, policy=..., n_runs=200).diff('baseline').plot()   # ask the policy question, paired by CRN
fit.sim.new(solver='abm', ...)            # and the same model as agents, or as an ODE, or at another scale
```

No other framework can write that script. **None of the thirteen versions runs today**, though for most of them every line but one or two already works on a hand-written model. The reasons it fails are the four in the summary, and the fixes are the rest of this page.

The proposal is to make these scripts the product's acceptance test. Write one switching demo per community, sized to that user's real problem, kept in the repo and run in CI, with the original framework's numbers alongside. The vignettes test the first day; the demos test the second week. The demos are also the marketing. A page of "coming from EpiModel", "coming from odin", and the rest, each with its numbers, would do more for adoption than any API change. It is also where the north-star test (P8, P9) should live.

### 2. Trust: no silent wrong answers on a user's standard pattern

Across the merged registers there are 58 silent wrong answers, and the seats added 16 more. They come from three causes, and each cause has a structural fix as well as the individual fixes:

- **A guess that reinterprets a value the user gave.** Examples: the unit of bare numbers set by `dt` (L1), a keyword that declares a new parameter (L3), the creation order deciding which states are people (L4), the dimension's name deciding the mixing (D4, N4), `1` against `1.0` (L17), a bare time meaning a year in a dated sim (D2), an age comparison on the group index (N3). The fix is the principle all four reviews converge on, in O's wording: *a guess may fill in what is missing, and must say so; it may never change the meaning of what was given, and never depend on something the user does not think of as part of the model*. Applying that principle is a set of small changes, listed in L1, L3, L4, L17, D2, D3, D4, and D11.
- **Two implementations of one thing that drift apart.** Examples:
  - modifiers skipped on callable parameters, because the callable returns early (D1);
  - a result's kind read from four places (R1, which compares a flow column as a stock and is 4.2 times low);
  - the stepped kernels sharing Numba's global stream (X3);
  - the JAX log-density and the particle filter each disagreeing with the run they stand for (N1, N2);
  - Python copies of every solver that share the kernels' bugs (X24);
  - `from_spec` and `to_code` covering different models (R10).

  The fix is one path per concept (one parameter evaluation, one `Result` with a kind, one engine per sim, one stream system). Where two paths must exist, as with JAX against Numba, add a differential test that compares them on every model in the corpus, including a fitted parameter times a time series.
- **A feature that is only half wired up.** Examples: `ss.change` ignoring `start=` (D8), `set=` on per-person quantities ignoring the base (D7), a waning vaccine re-delivered every step (D6), the SDE's exit order (X5), `resample()` keeping stale strata (R3). The fix is C's layered contracts (P10): conservation, exact identities, convergence in `dt`, backend equivalence, and metamorphic checks (an unrelated module leaves the draws unchanged, a rename leaves the results unchanged). Add simulation-based calibration for every fitting method: simulate from known values, fit, and check coverage. Run every code block in the spec as a test. Those three would have caught most of this list, and the inference checks would have caught N1 and N2 before any user did.

### 3. Inference is the headline: make it reliable and complete

This is where v4 is furthest ahead of the field. Most of the frameworks in the set have no fitting at all (MEmilio, EMULSION, LASER) or a hand-built loop (EpiModel, Ghana measles). The ones built for inference (odin, camdl, summer2, epydemix) need five objects wired by hand where v4 needs `bounds=`, `data=`, and `calibrate()`. The method rule, with its printed reason, is one of the best things in v4 (R25).

It fails in two ways. It is wrong on standard patterns (N1, N2, N6, N16, R1, R4), and it lacks the pieces each inference-heavy community treats as basic:

| Piece | Who needs it | Today |
|---|---|---|
| A fitted value per place or group, with a shared prior | flepiMoP, MEmilio, Ghana, Atomica | Every spelling raises `TypeError` |
| ABC-SMC for "stochastic, no likelihood" | epydemix, SimInf, EpiModel | Specified in `13-inference.md`, not built; Optuna with one run per evaluation is used instead and keeps the luckiest draw |
| IF2 for maximum likelihood of a stochastic model | camdl | Absent |
| Forward draws and projections (`priors=True`, `fit.project()`) with quantile tables | flepiMoP, MEmilio, epydemix, Ghana, summer2 | Absent |
| A parallel particle filter | odin, camdl, SimInf | Serial, 8–9x slower |
| The log-density and the run as JAX functions, with reverse-mode gradients | summer2, odin power users | Differentiates the run only, with finite differences either side |
| Binomial and Bernoulli observations | camdl, EpiModel | Absent |
| `converged` on every fit, and the score that was optimized | everyone | Dropped (R4, R5) |

This list sometimes runs against the reviews. F would strike staged fitting and `sim.sensitivity()`, and I agree. But ABC-SMC is not orchestration: it is the method for a whole class of models that today has only a point estimate. Build it. IF2 is a short loop over the existing `ParticleFilter`. Exposing the log-density as a JAX function is how v4 serves power users without adding a composable-sampler API of its own.

### 4. Be fast where speed is the product, and within a stated factor elsewhere

v4 is already at or ahead of parity for most of the set:
- **flepiMoP:** 5 s against 34–58 s at 3,000 places.
- **summer2:** first run 0.2 s against 3.6 s.
- **EMULSION:** 210 times faster.
- **EpiModel:** 10–30 times faster per run.
- **epydemix:** 100 runs in 0.07–0.23 s against 0.56 s.

It is behind in four places, and only one of them is close to an inherent tradeoff:

| Where | Gap | Inherent? | Better point |
|---|---|---|---|
| Metapopulation mixing with commuting (MEmilio) | 16–21x, and memory growing with the square of the number of groups | No: a bug | Apply the age matrix within places and the time share between them as two small products (N23) |
| Results memory (Ghana measles, flepiMoP, LASER) | About 5 GB per run | No: a bug | Keep groups at a reporting interval; build `results=[...]` as specified (N24) |
| Particle filters (odin, camdl) | 8–9x per particle-step | No | Particles in parallel in the kernel, a stream per particle (N27) |
| Register-scale Gillespie (SimInf) | 28x | Mostly no | Step only the herds an event touched; threads across herds (N26) |
| National-scale agents (LASER) | 25x on threads, 11x memory | Partly: a general agent engine will not match a hand-specialized one | One fused pass per step instead of 10–15, then threads, at about 50 bytes per agent (N25). Within 2–3x of LASER is the right target; parity is not worth what it would cost in generality |

The north star should say exactly this, as P8 proposes. Lines stay the hard target. Time becomes two reported numbers, time to first result (with the Numba build shown separately) and repeated-run time. A row fails only when it is more than 2x and more than 1 s slower than the original. The failing rows are published, each with its reason. Also print a line while the kernels are first built, which takes about 34 s, once per install (X8).

### 5. Meet users where they are

- **R.** Three of the twelve frameworks are R-first, and the `starsim` README advertises R-Starsim. Under v4 nothing can be built from R, because `sys._getframe(1)` fails when there is no Python caller (N17; I checked). This takes about ten lines plus an R-Starsim test in CI. The deeper lesson supports C's D5 and the merged L20: name capture from caller frames should never carry meaning. The arrow-string form with `name=` crosses any language boundary, so make it the documented, tested way in from R. A thin R layer (`transition("S -> E", rate = ~ beta * I)`, results as data frames) would then be days of work, not a port.
- **Docs.** Every page in `docs/` is Starsim 3 (P2). The order the merged P-file recommends is right: two short pages now ("v4 in ten minutes", and a v3-to-v4 migration page in the shape the Starsim 3 seat outlines), then the full tutorial port once the one-system work lands, then the "coming from X" pages in item 1. The migration page must carry the three traps the Starsim 3 seat found: the beta dict (N7), "routine" (N9), and the one-step shift (N10).
- **Importers.** The merged R22 moves the eleven importers to a companion package, and I agree, but promote it as the front door rather than tuck it away: "bring your model" is item 1's first line. The seats found that the importers' honesty, every untranslated item listed in `W1601`, is not yet true on real models (N12, N13). An import with untranslated parts should refuse to run unless the user asks it to.

### 6. One language

All four reviews reach the same structural conclusion, and every seat that touched it confirms it. v4 is today two languages in one namespace. The declared one is reached by `ss.Disease`, the v4 verbs, and `calibrate()`. The coded one is reached by `ss.SIR`, the library diseases, the v3 interventions, and `ss.Calibration`. Neither's tools work on the other's objects.

The merged recommendation:
- Rewrite `ss.SIR`, `SIS`, and `SEIR` as declared diseases (X11, L31).
- Put the duplicated v3 vocabulary in a hidden `ss.v3` namespace that warns and names the v4 spelling, rather than deleting it. P3 found that HPVsim subclasses four `ss.Base*` delivery classes and that four packages use `ss.Calibration`.
- Give results one `ss.Result` with a kind and a run axis (R18).

The Starsim 3 seat adds the piece the reviews missed: two different things are called "v3". The duplicated vocabulary should go to `ss.v3`. The coded-module API (`ss.Module`, `ss.Infection`, `step()`, `ss.FloatArr`) is not legacy. It is the escape hatch, the only home for continuous per-agent state such as CD4 counts, and what STIsim, HPVsim, and FPsim are made of. So it stays first-class, and the v4 verbs that act on arrays should reach it: `ss.modify` and `ss.vaccinate` on `rel_sus`/`rel_trans`, conditions on coded arrays, `calibrate(pars=...)`, and `sim.new()` for any unrun sim. The verbs that need structure (`equations()`, compartmental solvers) should say "this needs a declared disease". Then migration can go one disease at a time, which already works in a mixed sim and only needs to be documented.

### 7. The features each field cannot work without

From [new-findings.md](new-findings.md), sorted by how many seats need each one:
1. A fitted parameter per place, used by 4 seats.
2. Forward draws, projections, and quantile tables, used by 5.
3. One-off moves, all-or-nothing vaccines, and `at_age=`, used by 4.
4. ABC-SMC, IF2, and more observation families, used by 4.
5. Parallel particles and threads within a run, used by 4.
6. A per-place condition with a hold, used by 3.
7. Per-group degree on `PartnerNet`, used by EpiModel.
8. Programs and `ss.optimize`, used by Atomica.
9. Minimum stays and offspring, used by EMULSION.
10. Life tables and seeding, used by LASER and Ghana.
11. A dimension check through expressions, used by camdl and summer2 (L19 says build `E102`; F 09 proposed dropping it, and the camdl seat is the evidence against that).

None of these is a new concept. Each one attaches to something v4 already has: `ss.par`, `Delivery`, `PartnerNet`, the `Program`, `ss.modify`, or the typed values.

On Atomica, the seat is clear that v4 should not chase the ministry analyst with a GUI or spreadsheet-authored models. It should, though, have the program-and-budget layer, which is perhaps 500 to 800 lines on top of `cost=` and `capacity=`. That layer is what makes any model useful to the people who fund health programs, well beyond Atomica's users.

## The tradeoffs, and where v4 should sit on each

| Tradeoff | Where v4 sits | Better point |
|---|---|---|
| Brevity against stable meaning (every review) | Guesses that reinterpret given values | Guess only to fill what is missing; print every guess; error when no guess is safe (P9, principle 3 restated) |
| One general engine against specialist kernels (LASER, SimInf, odin) | 8–28x behind on their home ground | Within 2–3x, by fused passes, threads, parallel particles, and event-driven stepping; not parity |
| Declared structure against arbitrary per-agent state (Starsim 3, odin) | Coded modules lose every v4 tool | Coded modules first-class with the array verbs bridged; a declared auxiliary variable in the kernel for odin-style free equations |
| Python authoring against data formats (EMULSION YAML, Atomica spreadsheets, flepiMoP config) | Python only, with importers | Keep Python as the source (principle: "a data format is the right thing to export to and the wrong thing to write in"), but give a readable export (`to_code()`, a resolved-model YAML for review, a generated databook for analysts) |
| One-call inference against composable control (summer2, odin) | A fixed menu | Keep the menu; expose `sim.loglik(pars)` and the JAX log-density so a power user can bring their own sampler |
| Simple networks against statistical network models (EpiModel) | Degree-seeking with one `match=` | Per-group degree, several `match=` properties, a mixing matrix, and `acts=` within the same degree-seeking method; leave triangles (`gwesp`) to ERGM, and say in `06-routes.md` which terms map |
| Compartment count against features (Ghana measles, F2 P9) | Each product or delivery doubles the groups | An all-or-nothing product as a move between states (no dimension); a state-space budget past which `auto` picks agents |
| Complete results against memory (Ghana measles, flepiMoP) | Everything, every step | A reporting interval by default for stratified results, and `results=[...]` |
| Automatic solver choice against predictability (C D2, D7) | `auto` probes, and falls back silently to 10,000 agents | Probe, but make a failure on groups an error rather than a change of paradigm (X9, N22); `approx=` two-valued |

## What the four reviews found, merged

The detail is in five files. Each holds merged findings with sources, re-check status, proposals, a "where the reviews disagree" section with a recommendation, a keep list, and the decisions in its area.

| File | Findings | The headline |
|---|---|---|
| [findings/language.md](findings/language.md) | L1–L38 (18 silent wrong answers) | The unit of bare numbers comes from how the run is written (L1, `dt` alone cuts final R from 9,342 to 173); bare factors in a rate are each typed as rates (L2); a keyword number declares a parameter (L3); creation order decides who is people (L4); the two-line SIR seeds nobody and has R0 = 0.5 (L5) |
| [findings/dimensions-and-effects.md](findings/dimensions-and-effects.md) | D1–D45 | `ss.modify` does nothing on a callable, `timeseries`, or `seasonal` parameter (D1, C only, reproduced: 9,309 infections against 0); bare times change meaning with dates (D2); transfer-fed dimensions start half full (D3); the dimension's name decides mixing (D4) |
| [findings/execution.md](findings/execution.md) | X1–X30 | A distribution on `beta` is replaced by its mean under every solver, agents included (X1); a solver asked of a coded model is ignored (X2); random streams are not independent (X3); chain-binomial at a daily step is 17% off the ODE (X4, Cliff's call); the SDE resolves exits in declaration order (X5) |
| [findings/results-and-inference.md](findings/results-and-inference.md) | R1–R34 | A flow data column is compared as a stock, 4.2x low, and fitted that way (R1); the many-run object reports one run as ten and sums ratios (R2); `resample()` (R3); a stochastic fit reports a score it did not optimize (R4); the hash misses things that change results (R8) |
| [findings/docs-process-and-structure.md](findings/docs-process-and-structure.md) | P1–P15 | Both headline examples fail and nothing runs the spec's code (P1); `docs/` has no v4 (P2); 253 public objects against 94 counted (P3, settled); 59 error codes in all, 12 never emitted (P6, settled); the north star fails on time in four rows by a user-noticeable margin (P8) |

**How far the reviews agree.** They agree closely on the facts. The re-checks corrected only small things: a warning that does not print, two hashes rather than three, 84 weeks rather than 12, namespace counts that differed only in what was counted, and `PartnerNet` not being refused under an ODE as F said. Only C raised D1 (modifiers ignored on callables), and it reproduced; it is among the most serious bugs found. The reviews also agree on direction: one system, a stricter rule for guesses, fewer spellings, delete the Python solvers, keep the `Program` and the kernels, and keep `describe()`. They disagree on remedies, and each file's "where the reviews disagree" section settles them with a recommendation. The largest disagreements:
- **The v3 classes:** delete them now (F), or move them to a namespace (F2, O, C). The recommendation is a hidden `ss.v3`, because downstream packages use them.
- **The saved form:** `to_code()` against a complete JSON spec. The recommendation is `.py` plus a pickle, with `from_spec` deleted unless a non-Python consumer exists.
- **The solver names:** four, six, or "teach four". The recommendation is `auto`, `ode`, `binomial`, `gillespie`, and `abm`, with tau as Gillespie's noted fallback and `sde` cut.
- **The north star:** replace it (C), or restate it and publish the failures (F2, O). The recommendation is to restate it and publish.
- **The unit rule:** which inputs may set the model's unit. The recommendation is O's and C's: never `dt`, never year-like numbers.

**Keep.** Every file ends with a keep list, and they agree:
- the operator form and one transition class;
- the bare state against `.n`;
- states that are not people;
- `describe()` with its `[guessed]` marks;
- the flat `Program` and one Numba step for every solver, with whole runs in one call;
- `ss.modify` with a kept base;
- the product/delivery split;
- data attached by name;
- `calibrate()` from bounds, with its printed method;
- NUTS through BlackJAX and the particle filter;
- `to_code()` and `ss.diff`;
- `interop/core.py`;
- coded errors that name the fix;
- the "Rejected" tables in best-practices.

## Every finding, ranked

This section lists every merged finding (L, D, X, R, P) and every seat finding (N) in six tiers, most important first. Within the first tier the order is a ranking; below it, the order is rough. Importance means how likely a user doing ordinary work is to hit the problem, and how badly it misleads them when they do. A silent wrong answer on a common pattern ranks above a crash, a crash above friction, and friction above tidiness. The exception is structural problems that cause many others, which rank with what they cause. The details of each are in [findings/](findings/) and [new-findings.md](new-findings.md).

"Found by" names the reviews that raised each finding:

- **C:** `codex_review`
- **F:** `fable_review`
- **F2:** `fable2_review`
- **O:** `opus_review`
- **S:** a switching seat, used for the N findings, which no review had.

The attribution is read from each merged finding's "Raised by" line, so a review that made the point only in passing is counted.

### Who found what

| Review | Merged findings it raised (of 162) | Raised by it alone | What only it found |
|---|---|---|---|
| C | 53 | 4 | D1, modifiers ignored on time-varying parameters (one of the most serious bugs anywhere in the review); D9, L11, X15 |
| F | 128 | 9 | Import typing (R11, R12), `to_code()` after an import (R24), and six smaller items |
| F2 | 116 | 2 | Two delivery schedule bugs (D19, D20) |
| O | 132 | 16 | Several silent effect and dimension bugs: D6, D7, D13, D15, D16. Also `save()` crashing (R15), and L7, L12 to L14, X5, X6, and others |
| S (the seats) | 29 more (N1 to N29) | all 29 | Most of the top of the ranking: N1 to N8 on standard patterns, R from R (N17), and every measurement at a real user's scale (N23 to N29) |

Of the 162 merged findings, 36 were raised by all four reviews, 64 by three, 31 by two, and 31 by one. C raised fewer because it is the shortest of the four reviews (22 findings and 8 design notes), not because it missed things: three of its four solo findings are silent wrong answers.

### Tier 1: wrong answers on a standard pattern

These are silent, and a user doing ordinary work in at least one community will hit them.

| # | Finding | Found by |
|---|---|---|
| 1 | N1. The particle filter's likelihood is wrong for a fitted `beta` × `ss.seasonal` × `ss.timeseries`; PMCMC drives epydemix's COVID fit to its bounds | S |
| 2 | D1. `ss.modify` and `ss.change` do nothing on a callable, `timeseries`, or `seasonal` parameter (9,309 infections against 0) | C |
| 3 | L1. The unit of bare numbers is set by `dt` or the form of `start`/`stop`; changing `dt` alone takes final R from 9,342 to 173 | all four |
| 4 | N2. The JAX log-density is wrong for a fitted par × a time series; NUTS falls back to Python, and every draw is divergent | S |
| 5 | R1. A `new_infections` data column is compared as a stock, 4.2 times low, and fitted that way | all four |
| 6 | D4, N4. A dimension's name decides whether places mix; `patch=` discards an explicit contact matrix (82% fewer at the peak) | D4: F, F2, O; N4: S |
| 7 | N3. `sir.age < 5` compares the age group's index, not the age (52.2M against 6.9M) | S |
| 8 | L5. The spec's two-line SIR seeds nobody and has R0 = 0.5 | all four |
| 9 | N7. A Starsim 3 `beta` dict ported verbatim is 12 to 52 times too small | S |
| 10 | N5. `ss.parallel` after `init()` gives identical runs | S |
| 11 | L3. A keyword number declares a new parameter: `SIR(beta=0.3)` is ignored, and typos pass | F, F2, O |
| 12 | X1. A distribution on `beta` is replaced by its mean under every solver, agents included | C, F, O |
| 13 | R2. The many-run object reports one run as ten, and sums ratios across runs | F, F2, O |
| 14 | X3. Random streams are not independent: the stepped kernels share Numba's global generator, and engine seeds collide | all four |
| 15 | N6. A Poisson or negative binomial likelihood on non-integer data is `-inf`, with no warning | S |
| 16 | N8. The spec's "routine at 9 months" vaccinates 8,757 of 10,000 people on the first day | S |
| 17 | R4, N16. A stochastic fit reports a different score from the one it optimized, and keeps the luckiest draw | R4: C, O; N16: S |
| 18 | L2. Each bare factor in `R0 * gamma * I` is typed as a rate, so the product changes with the unit | F, F2 |
| 19 | D11. The default population changes when you stratify, and named data does not bring its population | F2, O |
| 20 | D2. A bare time changes meaning when the sim has dates, and an out-of-range time is clamped silently | F2, O |
| 21 | X4. Chain-binomial at a daily step is 15 to 17% off the ODE, and the note understates it | all four |
| 22 | R5. Fits never say whether they converged | C, F, O |
| 23 | N9, N10. Starsim 3's "routine" means something else in v4, and unchanged Starsim 3 code gives results one step later | S |
| 24 | N12, N13. The importers run silently wrong on these frameworks' real models | S |
| 25 | D10. Networks under a compartmental solver: one ignores the graph, one ignores duration | all four |
| 26 | L17. `init=1` is one person and `init=1.0` is everyone | all four |
| 27 | N15. `PartnerNet` with `match=` alone silently loses risk structure (31%/31% against EpiModel's 21%/77%) | S |

### Tier 2: other silent wrong answers

| Finding | Found by |
|---|---|
| D13. Partial cross-immunity runs under `auto` as an ODE that is off by up to 2.9 times and reverses which strain dominates | O |
| D14. A contact matrix is aggregated by band width, not population, and is not validated | C, O |
| R3. `resample()` keeps stale strata, averages stocks and ratios, and fails without dates | all four |
| L4. The first state created decides who is people and where the rest of the population starts | C, F |
| D3. A dimension that only a transfer fills starts half full | F, F2, O |
| D6. A waning product at full coverage is re-delivered to everyone every step | O |
| D7. `set=` and `add=` on `rel_sus` or a rate act on a multiplier of 1, not on the base | O |
| D8. `ss.change` ignores `start=`, `stop=`, and `when=` | F2, O |
| X2. A solver asked of a model written as code is ignored | F, O |
| R6. A misspelled `gof`, priors under an optimizer, and a zero-mean series silently change what is fitted | F, O |
| R8. The hash misses things that change results | all four |
| R10. `to_code()` writes a different model, or broken code, without saying so | all four |
| L10. Bounds are not enforced, a parameter with no value runs as NaN, and negative initial values are accepted | C, F2, O |
| D12. Aging is on with no deaths, and the compartmental aging rate assumes a flat pyramid | F, O |
| D17. The written-out `/N` form silently ignores the contact matrix | F, F2, O |
| D18. A held treatment goes to everyone by default | F, F2, O |
| D21. An unknown per-route `beta` key is silent, and routes carry a second `beta` | F, F2, O |
| D15. As agents, a `sex` dimension has nothing to do with the agents' sex | O |
| D16. The spec's gravity example sends everyone away all the time | O |
| L9. A bare rate after `@` is per step, and under the ODE nothing says so | F, F2, O |
| L8. The string form guesses the source of infection, wrongly for an SEIR | F, F2 |
| L16. `pars=` on `ss.Sim` is read as sim settings or model values by sniffing the keys | F, F2 |
| R7. An `every=` outcome is a comb of zeros, compared wrongly with daily data | F, O |
| R11. An imported `beta * S * I / N`, with `N` a parameter, is read as density-dependent | F |
| R12. Every import is right only at the unit it was imported at | F |
| N11. `ss.by(stage=...)` on an inflow gives 0 births | S |
| N14. `stratify(immunity=...)` makes no dimension, and an explicit `split=` is ignored | S |
| D5. The sim-level `stratify` is a second parser that skips the checks | F, O |
| D9. A test repeated more often than its delay never returns a result | C |
| D19. A follow-up delivery ignores its own schedule | F2 |
| D20. `capacity=` is per round in a campaign but per unit time in routine delivery | F2 |
| D22. An intervention's `step()` first runs at `ti == 1`, so Starsim 3 code that sets up at `ti == 0` never fires | O |
| L6. `@` accepts a second driver, drops arithmetic, and cannot see inside `ss.by` | F |
| L7. `>>` and `@` mutate in place, so two transitions built from one partial are one object | O |
| L11. `ss.by` arithmetic silently drops levels | C |
| L12. `ss.transmission` with no `beta` gets `beta = 1`, unmarked | O |
| L13. `ss.Sim(a, b)` makes the second disease the label, and drops it | O |
| L14. As agents, a seeded cohort stays a step too long, and `stages=` is ignored with `rate=` | O |
| L15. A typo in an arrow string makes a new reservoir state | F2, O |
| X5. The SDE resolves competing exits in declaration order | O |
| X6. Tau-leaping clips its draws in declaration order | O |
| X7. Agents realize `dur=` differently from `rate=` | F2, O |
| R9. `ss.compare_models` ranks models on unlike measures | O |

### Tier 3: what blocks a switch, apart from wrong answers

These are crashes, dead ends, and performance at a real user's scale.

| Finding | Found by |
|---|---|
| N17. No v4 model can be built from R | S |
| N25. Agents are 25 times slower than LASER threaded at 3.3M agents, at 11 times the memory | S |
| N23, D44. Mixing and movement between groups are dense groups-by-groups matrices, 16 to 21 times slower than MEmilio | N23: S; D44: F2, O |
| N24. Every group is kept at every step, about 5 GB of results per run | S |
| N26. Gillespie by place visits every herd every day, 28 times slower than SimInf | S |
| N27. The particle filter is serial, 8 to 9 times slower than odin and camdl | S |
| X16. Performance at scale trails the specialists | C, F2, O |
| N20. Parameters cannot be fitted where users put them (a value per place, a modifier's amount, inside `ss.seasonal`) | S |
| R14. Several natural spellings of fitted values fail | all four |
| N28. A 774-patch sim takes 100 s to build for a 5 s run | S |
| N18. `sim.stratify()` is lost by `sim.new()` | S |
| R15. `sim.save()` crashes for every declared model after a run | O |
| N19. Conditions on a coded module's arrays crash | S |
| X9, N22. `auto` sends ordinary models to agents, at worst 10,000 agents for 83 million people | X9: all four; N22: S |
| X8. The first run builds the kernels for about 35 s and prints nothing | F, F2 |
| X10. Leaving the whole-run kernel is silent, and harmless-looking things trigger it | F, F2, O |
| X14. `solver='stochastic'` picks Gillespie for runs that take minutes | F, F2, O |
| N29. Declared diseases as agents are 1.4 to 1.6 times slower than the same model as Starsim 3 code | S |
| X15. `check_dt()` crashes when a module has its own step | C |
| L21. A string state name resolves to whatever caller variable has that name | F |
| L22. `sir.pars.beta = x` replaces the `ss.par` with a float, and a later `ss.modify` crashes | F, F2 |
| L24. Rates and conditions are two expression trees that crash where they meet | C, F2 |
| N21. Small dead ends with unhelpful messages | S |

### Tier 4: two systems, and the surface built on them

These cause many of the findings above, and they block the docs.

| Finding | Found by |
|---|---|
| X11. The library `ss.SIR`, `SIS`, and `SEIR` are Starsim 3 code, and `ss.Disease` is two classes chosen by sniffing | F, F2, O |
| D23. The v4 effects and deliveries do not reach any Starsim 3 disease | F, F2, O |
| R13. There are two calibration systems, and `sim.calibrate()` refuses every class-based sim | F, F2, O |
| L31. `ss.Disease` is two classes behind one name, and `sir.S` changes type at `init()` | F, F2, O |
| R18. Three results models instead of one `ss.Result` with a kind and named axes | all four |
| P3, L29. The namespace shows 253 objects where the spec counts 94 | all four |
| X24. Every compartmental solver is written twice, and the Python copy shares the kernels' bugs | all four |
| X25. Multi-disease runs use one engine per disease plus a cache-coherence protocol | F, F2, O |
| L20. Names are taken from caller frames, which fails on the first refactor (and from R) | all four |
| X17. The solver menu: six names, `'stochastic'`, `'ctmc'`, and `integrator=` mixing schemes with a backend | F, F2, O |
| D31. The Starsim 3 intervention and product vocabulary is still at `ss.` | F, F2, O |
| R21. The model as a file has ten entry points for two jobs | all four |
| R22. Eleven framework importers live in the core package | F, F2, O |
| R20. Six ways to run many sims | F, F2, O |
| X29. One normalized model, or "the `Program` already is that" | all four |
| X12. In a compartmental run, `sim.people` is a one-row stand-in | F, F2, O |
| X13. Code in the same step sees different values of `sim.ti` | F, F2, O |
| D26. A plain function or custom intervention makes the whole sim agent-based | C, F, O |
| D28. Four ways to add births and deaths, three of which turn an ODE into agents | F, F2 |
| L23. What counts as a parameter, and its name, depends on the rest of the model | F, F2, O |
| D32. `ss.contacts` as the one way to give a contact matrix | F, F2, O |
| D33. Six spellings of movement between levels | F, F2, O |
| D34. One schedule keyword and one "who" keyword | F, F2, O |
| D35. Products are modifiers in fact, not by duck typing | F, F2, O |
| L28. 27 exported time names for three types | F, F2, O |
| R19. Four names for each flow | F, F2 |
| X26. JAX is a gradients backend, and `sim.gradient()` still uses finite differences at both ends | F, F2, O |

### Tier 5: docs, tests, and process

| Finding | Found by |
|---|---|
| P1. The spec's two headline examples fail, and nothing runs the spec's code | all four |
| P2. The user docs teach only Starsim 3 | F, F2, O |
| P10. The conformance suite is a list in prose, and the tests have hand-picked tolerances | all four |
| P8. The north-star test fails on time in a third of the vignettes | all four |
| P9. Principles 3, 8, and 9 need amending | C, F2, O |
| P4. The checks that matter most never fire | F, F2, O |
| P6. The error catalog has drifted from the code | all four |
| P7. The spec, the departures list, and the code disagree | all four |
| R25. The fitting method rule: keep it, fix the spec, and reconsider two defaults | all four |
| P5. Messages go through three channels and depend on `verbose` | F, F2, O |
| P11. The design folder is book-length and repeats each big rule | all four |
| P12. The test suite has a race and machine-specific paths | F, F2, O |
| P15. Two copies of the design | all four |

### Tier 6: friction, spec drift, small surface, and codebase

| Kind | Finding | Found by |
|---|---|---|
| Friction | L26. The first wrong guesses get Python's errors | F, F2, O |
| Friction | L27. `describe()` misreports some guesses | F, F2, O |
| Friction | L38. Three surprises in the spec's examples | F, F2 |
| Friction | D24. An error in a `when=` function is misreported | F, O |
| Friction | D25. Lazy conditions need a throwaway `ss.People(1)` | all four |
| Friction | D27. Follow-up deliveries lose the user's names | F, O |
| Friction | D29. Realized coverage averages to near zero | O |
| Friction | D30. A network's parameter cannot be named in `ss.modify` | F, F2, O |
| Friction | R16. Attaching data: wrong advice, and misalignment found late | F, F2, O |
| Friction | R24. `to_code()` after an import is unusable for data-heavy models | F |
| Friction | R28. `duration=` on an outcome is exponential, but `delay=` is exact | F |
| Friction | X18. The solver line and notes are out of order and ignore `verbose=0` | F, F2, O |
| Friction | X19. Common random numbers are unreported and oversold | C, F, O |
| Friction | X20. Erlang stages as `auto`'s one approximation | all four |
| Friction | X30. Small `Sim` and engine items | F, F2, O |
| Spec drift | L18. A sim with no unit runs in days underneath | F, F2 |
| Spec drift | L19. Typed arithmetic does not match the spec, and `E102` is not there | F, F2, O |
| Spec drift | L32. Spec promises the code does not keep | F, F2, O |
| Spec drift | D36. Base and effective values | all four |
| Spec drift | D42. `dependence='density'` and `n_ref` do not exist | all four |
| Spec drift | D43. Smaller drift and footguns in dimensions and effects | F, F2, O |
| Spec drift | X21. Gillespie holds time-varying rates while described as exact | C, F, F2 |
| Spec drift | X22. Delivery order differs by engine | C, F, O |
| Spec drift | X23. Other solver drift | all four |
| Spec drift | R17. Specified and absent, including an infection log that returns an empty table | all four |
| Spec drift | R29. The model spec's content | all four |
| Spec drift | R30. Inference and results drift | F, F2, O |
| Spec drift | R31. Interop drift | F |
| Surface | L25. `0.5 * infection` is both half the rate and half the flow | F, F2, O |
| Surface | L30. Small core spellings | F, F2, O |
| Surface | D37. Modifiers in `connectors=` and `interventions=` behave alike | F, F2, O |
| Surface | D38. Four spellings of `stratify` | F, F2, O |
| Surface | D39. Prune the network catalogue | F, F2, O |
| Surface | D40. No all-or-nothing vaccine | O |
| Surface | R23. Two Mermaid writers, and lossy SBML and odin exports | F, F2, O |
| Surface | R26. The `Fit` object and comparison names | F, O |
| Surface | R27. `ss.Samples` and `ss.Dataset` are disconnected Starsim 3 code | F, F2, O |
| Codebase | X28. `Sim` carries the group geometry, and `solvers.py` is six modules | all four |
| Codebase | P13. v4 sits beside v3, and its four largest files each hold several modules | all four |
| Codebase | L33. `transitions.py` is four spec chapters in 2,972 lines | F, F2, O |
| Codebase | L34. Pending flags, sentinels, and three printers | F, F2, O |
| Codebase | L35. Two parameter paths kept in step by hand | F, F2, O |
| Codebase | L36. Name lookup in six places | F |
| Codebase | L37. `time.py` is 2,612 lines for four types | F, F2 |
| Codebase | D41. A delay, fading efficacy, or partial coverage each add a dimension | F, F2 |
| Codebase | D45. Smaller codebase items | F, O |
| Codebase | X27. The compartmental engine repeats its machinery | F, F2, O |
| Codebase | R32. `fit()` is one 300-line function | F, F2, O |
| Codebase | R33. Results code over six files | F, F2 |
| Codebase | R34. Interop internals duplicated | F |
| Codebase | P14. The dependency list does not match the package | F, O |

## Decisions for Cliff

The five merged files list about 100 decisions, each with the reviewers' recommendations. These are the ones that unblock the most; the first three shape everything after them.

1. **Principle 3 restated:** a guess fills in what is missing and never reinterprets what was given. If yes, the unit rule (L1), keyword numbers (L3), the people anchor (L4), initial values by size (L17), bare times (D2), and dimension kind from the value (D4, N4) all follow, each a small change. (O, C, F2's guess table; F partly.) [CK: agree]
2. **One language:** declared `ss.SIR`/`SIS`/`SEIR`; `ss.Disease` declared-only; the duplicated v3 vocabulary to a hidden `ss.v3` (deleted in v5); the coded-module API kept first-class, with the array verbs bridged to it (the Starsim 3 seat). (All four reviews on the first two; P3 and the Starsim 3 seat on the rest.)  [CK: remove rather than duplicate v3 vocabulary, and make sure there is a crystal-clear+robust merging/migrating skills/guide.]
3. **Who v4 is for first.** My recommendation is compartmental and metapopulation modelers, moderate-scale agent modelers, and the Starsim community, where v4 is already at or ahead of parity, then national-scale agents and register-scale herds once the performance items land. Take on the program-and-budget layer, but not the ministry analyst's spreadsheet authoring. This decides how much of section 4 to fund and when. [CK: agree]
4. **Inference scope:** build ABC-SMC, IF2, vector parameters, forward draws, and the JAX log-density handle; strike staged fitting and `sim.sensitivity()`. (The seats; F and C on the striking.) [CK: agree on the build, disagree on the striking -- staged fitting and sensitivity both seem like very useful features]
5. **The north star:** lines stay the hard target; time becomes two reported numbers that fail at more than 2x and more than 1 s; failing rows are published; the switching demos join the vignettes as the acceptance test. (P8; F2, O, C.) [CK: note also the big gap between 1st vs subsequent runs; in both v4 and the source language, we should benchmark say best of 5 runs, not the first run.]
6. **The importers:** a companion package, promoted as the front door, that refuses to run an incomplete import unless asked; SBML in and Mermaid out stay in core. (R22, plus N12 and N13.) [CK: not "refuses", but "passes to an LLM with an appropriate loaded skill to complete it"]
7. **R:** support it officially through tested arrow strings and `name=` (N17, L20)? (The R-first seats: yes.) [CK: agree; see /home/cliffk/idm/rstarsim which we can delete or merge into this as neded]
8. **The solver menu:** `auto`, `ode`, `binomial`, `gillespie`, `abm`; `integrator=` as a scheme only; the Python solvers deleted, with contract tests as the oracle. (X17, X24.) [CK: don't entirely understand, but ok]
9. **Chain-binomial at a daily step, +17% against the ODE:** keep whole steps and fix the warning, or default to the ODE's substep rule? The options are in X4; your rule is that you pick solver fixes. [CK: i don't have a strong preference, you decide]
10. **The saved form:** `to_code()` `.py` plus a working pickle, with the JSON spec in memory for `hash()` and `diff()`. Is there a non-Python consumer of the spec in mind? (R21.) [CK: no, no non-python consumer. not sure how to_code() is different than ... the code, but i could be convinced.]
11. **Results:** one `ss.Result` with a kind and a run axis; a flow named after its transition; `sim.run(n_runs=)` dropped in favor of `ss.parallel`. (R18, R19, R20.) [CK: i think i disagree with this one; sim.run(n_runs=100) is, yes, a shortcut to ss.parallel(), but a useful one; ss.parallel() would mostly be for running lots of different sims, not the same sim with different seeds.]
12. **The time index during a step:** `k−1` for every module, and declared transitions in one slot under every engine. (X13, X22.) [CK: ok]
13. **Errors:** exception classes named as today's codes, with the numbers dropped or frozen; the catalog and vocabulary generated from the code. (P6, P3.) [CK: agree, i never liked the numbers]

## Suggested order of work

1. **A day of unblocking.** The frame-lookup fallback, so R works (N17). A line while the kernels build. The two overview examples run (P1). `ss.parallel` after `init()` gives distinct runs (N5). `sim.stratify()` survives `sim.new()` (N18).
2. **The silent wrong answers, with a regression test each.** These are the merged registers' silent items plus N1 to N16. Add the contract tests at the same time: a differential test of the JAX density and the particle filter against the run, simulation-based calibration for each method, stream invariance under an unrelated module, and the spec's code blocks as tests. These keep the rest of the work honest.
3. **The guessing rules** (decision 1). Small changes once the rule is chosen.
4. **One language** (decision 2), with one `Result` (decision 11). This is the large breaking step, and it is what makes the docs writable.
5. **Docs:** the two short pages, then the tutorial port, then the "coming from X" pages, with the switching demos running in CI.
6. **The shared features**, in the order of section 7, which serves the most seats first.
7. **Performance:** factored mixing and the results interval first, because they are bugs and cheap; then build time at many places, parallel particles, event-driven herds, and the fused, threaded agent step.
8. **Surface cuts, the spec rewrite, and code structure**, as the merged files lay out, mostly easier once step 4 has deleted the second system.

Steps 1 and 2 need no design decisions. Steps 5 to 7 can overlap with step 4.

## What I checked myself

These were run with `/software/conda/envs/v4/bin/python` at `a20c4f7e`. The scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/me/` unless another seat's folder is named.

- **Front door.** `import starsim` takes 1.0 s. With the kernel cache warm, one SIR of 10,000 people runs under `ode`, `gillespie`, `binomial`, and `abm` in 0.06, 0.04, 0.01, and 0.11 s, with peak `n_I` of 3,036, 3,064, 3,109, and 3,100 (`first.py`). Building a second disease from the same three states raises `E206` (L20).
- **N1:** the particle filter peaks at the wrong `b` for seasonal × time series (`../epydemix/t2_pf_repro.py`).
- **N2:** the JAX density is −967,465 against the sim's −146.7 when the multiplier of a time series is fitted. It agrees when the multiplier is not fitted (`../summer2/nuts_bisect.py`).
- **N3:** `sir.age < 5` gives 52.2M and `sir.age >= 50` gives 0, against 6.9M for `sir.age == '0-4'` (`../epydemix/t4_cond.py`).
- **N4:** `patch=` with a contact matrix gives [9,396, 0], against [9,882, 9,882] for the same model with the dimension named `zone` (`patchmat.py`).
- **N5:** after `init()`, `ss.parallel(sim, n_runs=4)` gives [782, 782, 782, 782] (`identical.py`).
- **N6:** a Poisson log-likelihood on non-integer data is `-inf` (`newbugs.py`).
- **N17:** a call with no Python caller frame raises `ValueError: call stack is not deep enough` at `diseases.py:74` (`noframe.py`).
- **N18:** after `sim.stratify(age=...)`, `sim.new()` gives a sim whose results are not stratified (`strat_new.py`).

Everything else is either a merge agent's re-check, recorded in each `findings/` file with its script, or a seat's own finding, recorded in each `switching/` file with its script.

## Files

- [findings/](findings/): the four reviews merged by area. [language.md](findings/language.md), [dimensions-and-effects.md](findings/dimensions-and-effects.md), [execution.md](findings/execution.md), [results-and-inference.md](findings/results-and-inference.md), and [docs-process-and-structure.md](findings/docs-process-and-structure.md).
- [switching/](switching/): one assessment per seat. [epimodel](switching/epimodel.md), [siminf](switching/siminf.md), [odin](switching/odin.md), [camdl](switching/camdl.md), [laser](switching/laser.md), [ghana_measles](switching/ghana_measles.md), [memilio](switching/memilio.md), [flepimop](switching/flepimop.md), [summer2](switching/summer2.md), [atomica](switching/atomica.md), [emulsion](switching/emulsion.md), [epydemix](switching/epydemix.md), and [starsim3](switching/starsim3.md).
- [implementation.md](implementation.md): the plan of work, in nine phases with a checkbox per item, and the acceptance table for the thirteen switching demos.
- [next-reviews.md](next-reviews.md): five more reviews of the kind that found the most here (invariance fuzzing, a statistical and scaling audit, the textbook canon, the naive user, and real models at full complexity).
- [new-findings.md](new-findings.md): what the seats found that no review did, in four groups. N1–N16 are silent wrong answers, N17–N22 are crashes and dead ends, N23–N29 are performance, and the last table lists the missing features by how many seats need each one.
