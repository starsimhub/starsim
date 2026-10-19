# Starsim v4 review: best practices, spec, and implementation

Reviewed 2026-10-17 on the `v4` branch of `ssv4` at `a20c4f7e`, covering `v4_design/best-practices/`, `v4_design/design/`, `v4_design/admin/`, and the code in `starsim/`. Six reviewers each took one area, read its docs and code, and ran repros for every bug they claimed. Their detailed notes (about 120 findings) are in [notes/](notes/), and this page is the synthesis. Repro scripts are in the session scratchpad and are referenced by name in the notes.

## The verdict

The language at the center is good, and better than anything in the frameworks it surveys. `S >> I @ beta * I`, a transition whose kind comes from its rate, states that are not people inferred from connectivity, one `stratify()` for every kind of heterogeneity, a compiled flow program that runs one model as an ODE, a stochastic compartmental model, or agents, `describe()`, the model hash, and data attached to results by name all work as designed. Where two solvers should agree, they do.

The problems are not in that core. They are in three places around it:

1. **v4 is a second system beside v3, not a replacement for it.** Almost every user-visible failure traces back to the seam.
2. **The principle "guess the default and print it" is applied to things that are not defaults.** Several guesses reinterpret a value the user gave, and they do it silently, against principle 3's own test that an error is for when a wrong guess would be silently harmful.
3. **There are several ways to do most things.** The counted vocabulary is 133 names, but the real surface is about twice that, and the spec, checklist, and code have each drifted from the others.

The single most valuable change is to finish the migration: make every built-in disease, network, demographic, and intervention a declared v4 object, and move the v3 machinery that this makes redundant out of the main namespace. Most of the remaining simplifications either follow from that or become cheap once it is done.

## 1. Make v4 one system

Five of the six reviews arrived here independently. Today a model's capabilities depend on how its disease was written, and the user cannot tell which kind they have.

| What a user tries | With a declared disease | With `ss.SIR`, `ss.SIS`, `ss.SEIR`, or any `starsim.library` disease |
|---|---|---|
| `ss.modify`, `ss.change`, `ss.vaccinate` | works | fails, with an error that blames the target ([effects 1](notes/effects-and-interventions.md)) |
| `sim.calibrate()` | works | "sim.new() is only available for sims with declared states" ([results 1](notes/results-and-calibration.md)) |
| `describe()`, `equations()`, `to_code()`, `ss.load` | works | raises `NotImplementedError` or prints internal arrays ([surface 6](notes/surface-and-machinery.md), [core 5](notes/core-language.md)) |
| `solver='gillespie'` | runs Gillespie | silently runs agents ([solvers 8](notes/solvers-and-execution.md)) |
| `infection_log` | records 0 rows | works ([results 7](notes/results-and-calibration.md)) |
| `sim.save()` | crashes | works ([surface 7](notes/surface-and-machinery.md)) |

Beneath that, the two systems disagree on basic conventions. In the same step, declared rates and deliveries see `sim.ti == k-1` while function interventions see `k` ([solvers 3](notes/solvers-and-execution.md)). As agents, declared progressions run before interventions and transmission after them; in a compartmental run every flow comes after them ([solvers 4](notes/solvers-and-execution.md)). The latter is the cause of the known issue that "TPT on the first step reached 7% fewer people" as agents. There are two calibration systems (v3 `ss.Calibration` with `CalibComponent`, about 1,150 lines, which is still what the user docs teach) and two intervention stacks (v3 `routine_vx`, `campaign_screening`, `treat_num`, and the rest, about 530 lines). `ss.Disease` itself is two unrelated classes chosen by inspecting the arguments ([core 5](notes/core-language.md)). None of the pages in `docs/` uses v4 syntax.

**Proposal.**

- `ss.Disease` is always a declared model. Rewrite `ss.SIR`, `ss.SIS`, and `ss.SEIR` in the language: each is a few lines, and the library's own models then become the best examples of it. A disease that is pure code subclasses `ss.Module` (or keeps `ss.Infection`) and declares `compartmental = True` if it works on groups.
- Rewrite the `starsim.library` diseases as declared models where they fit, starting with HIV, since it is the one most likely to need the v4 interventions.
- One intervention vocabulary: products plus deliveries. Make target lookup and the selector accept any module's states, so they work on code-defined diseases too. Move the v3 delivery classes and `ss.Calibration`/`CalibComponent` into a compatibility module (`ss.v3`, say) that the namespace does not export, or delete them, since v4 is a breaking release anyway.
- One time convention, set by the loop: every module in step `k` sees `ti == k`. Run all declared transitions of a step at one point in the step, the same under every engine, so deliveries see the same people under agents and compartments.
- Port the tutorials and user guide in `docs/` once the above lands. Until then, a new user meets only v3.

**Cost.** This is the biggest item here: several days of porting and a breaking change for v3 users of the library classes. It also removes code. The two calibration systems become one, the two intervention stacks become one, and the namespace loses most of the v3 names, which make up much of the gap between the 285 real and 94 counted names ([surface 5](notes/surface-and-machinery.md)).

## 2. A guess may fill in what is missing, but never reinterpret what was given

Principle 3 says to guess when the right answer is obvious 99% of the time. Most of the silent wrong answers found in this review are guesses that change the meaning of something the user did write. Each one below was reproduced.

| What the user wrote | What happens | Notes |
|---|---|---|
| `dt=ss.weeks(1)` instead of days, with a bare `beta` | `beta` becomes per week, and the epidemic vanishes: final R goes from 9,412 to 173 | [core 1](notes/core-language.md); `stop=ss.weeks(12)` or `start=2020` do the same |
| `SIR(beta=0.3)` | declares a new parameter that nothing uses; the model keeps 0.05 | [core 2](notes/core-language.md); `SIR(bta=0.3)` is accepted too; with a class-level `beta = ss.par()` it raises E206 instead |
| `ss.Disease(..., init_prv=0.01)` | becomes a parameter; no epidemic | [surface 2](notes/surface-and-machinery.md) |
| `ss.Sim(hiv, syph)` | `syph` becomes the sim's label | [surface 3](notes/surface-and-machinery.md) |
| `'l -> R'` (a typo for `I`) | `l` and `R` become reservoirs | [surface 9](notes/surface-and-machinery.md) |
| `ss.vaccinate(start=30)` in a dated sim | 30 is read as the year 30; nothing is delivered | [effects 2](notes/effects-and-interventions.md) |
| `stratify(region=...)` against `stratify(location=...)` | the dimension's name decides whether places mix | [dims 3](notes/dimensions-and-mixing.md) |
| `stratify(sex=...)` as agents | a random property unrelated to `people.female` | [dims 2](notes/dimensions-and-mixing.md) |
| `init=1` against `init=1.0` | one person against everyone | [core 11](notes/core-language.md) |
| The two-line SIR from the overview | nobody is seeded and R0 is 0.5, so it plots a flat line | [core 3](notes/core-language.md), confirmed |

**Proposal.**

- **Restate principle 3**: a guess may supply a value the user left out, and must say so. It may never change the meaning of a value the user gave, and it may never depend on something the user does not think of as part of the model (the step size, how the dates are written, a dimension's name).
- **Units**: a bare number is in the model's unit, which comes from `unit=` or the typed values in the model, never from `dt`, `start`, or `stop`. `dt` is only the step. This is what `sim.check_dt()` already assumes.
- **Keywords**: a keyword that names an existing parameter sets it. Any other keyword raises E602 with the closest names. New parameters are declared with `ss.par` or a variable name, never by a keyword. `ss.Sim` takes modules positionally and everything else by keyword only.
- **Inferred states**: when one name in an arrow string is used nowhere else and is one edit away from a used name, raise rather than make it a reservoir.
- **Times**: a bare number is a time in the sim's unit everywhere; a year is written as a date (`'2030'` or `ss.date(2030)`).
- **Dimension kind** comes from the value, not the name: `ss.places(...)` or `mixing=` says that places mix, or that they don't.
- **One line at `init()`** that states R0 and how many people start infected. This catches most of the cases above in the line that already announces the solver ([core 14](notes/core-language.md)). Make the two-line SIR seed by default (`init_prev=0.01`) and use a `beta` that gives an epidemic.

**Cost.** Each item is small. The unit and keyword rules are breaking for code that relied on the current behavior, which is mostly code that was silently wrong.

## 3. Correctness bugs that give wrong numbers silently

These are the bugs where nothing crashes and the answer is wrong. Crashes are listed in the notes and are less urgent, because the user knows.

| Bug | Size | Where |
|---|---|---|
| The SDE solver resolves competing exits in declaration order; with three equal exits, everyone leaves by the first | 100% versus 33% | [solvers 1](notes/solvers-and-execution.md) |
| `set=` and `add=` on `rel_sus`, `rel_trans`, or a transition act on a multiplier of 1, not the base | `set=1.0` on 0.5 leaves 0.5 | [effects 8](notes/effects-and-interventions.md) |
| A routine delivery of a waning product at full coverage re-vaccinates everyone every step | 104k doses against 9.9k at coverage 0.99 | [effects 3](notes/effects-and-interventions.md) |
| `ss.change` silently ignores `start=`, `stop=`, and `when=` | always on | [effects 4](notes/effects-and-interventions.md) |
| A stratified ratio across runs is summed as a count | 22.0 instead of 0.08 | [results 3](notes/results-and-calibration.md) |
| A v3 weekly flow is compared to data as a one-day value | 345 against 1,038 | [results 5](notes/results-and-calibration.md) |
| `resample()` averages stocks and ratios instead of taking the end value | | [results 6](notes/results-and-calibration.md) |
| Partial cross-immunity between strains runs under `auto` as an ODE about 2× off | 2× | [dims 11](notes/dimensions-and-mixing.md) |
| A per-person distribution on `beta` is replaced by its mean, even with `approx=False` | | [solvers 6](notes/solvers-and-execution.md) |
| The kernels use Numba's global generator, so two sims stepped in one process change each other's results | breaks CRN | [solvers 10](notes/solvers-and-execution.md) |
| `dur=` and `rate=`, declared equivalent, differ as agents, in branching, and with `stages=` | | [core 4](notes/core-language.md) |
| People who start in a state stay one step too long as agents | one step | [core 4](notes/core-language.md) |
| `@` and `>>` mutate the transition, so two transitions built from one partial become one object | | [core 7](notes/core-language.md) |
| The spec's vaccination-as-a-stratification example starts half the population vaccinated | 776 against 3,285 infections | [effects 6](notes/effects-and-interventions.md) |
| Default aging with no deaths inflates the 65+ group by 42% a year in the ODE against 10% as agents | | [dims 6](notes/dimensions-and-mixing.md) |
| The default stratified population is uniform from 0 to 100 years, even with `ss.contacts('Kenya')` | 35% over 65 | [dims 5](notes/dimensions-and-mixing.md) |
| `solver='stochastic'` estimates events from the population only and picks Gillespie | 74.5 s against 0.03 s for tau | [solvers 2](notes/solvers-and-execution.md) |

Twelve of the 54 catalogued error codes are never raised, including E102, E604, E606, and W207 ([surface 9](notes/surface-and-machinery.md), [core 8](notes/core-language.md)). E606 matters most: a parameter outside its bounds runs, and under `solver='ode'` gives all NaNs with no message.

## 4. Fewer ways to do each thing

Each row is a place where the user has to choose between spellings that mean the same thing, or learn a distinction that does not change anything. The proposal is the one I would pick; the notes give alternatives.

| Today | Proposal | Notes |
|---|---|---|
| `to_spec`, `to_json` (spec on a disease, v3 results on a sim), `to_code`, `ss.to_code`, `ss.from_spec`, `ss.save(format=)`, `ss.load(format=)`, `ss.interop.export`, `sim.save`, `sim.to_yaml` | `ss.save(path, obj)` and `ss.load(path)`, with the format taken from the extension (`.json` spec, `.py` code, `.sbml`, `.R`, `.mmd`, `.sim` pickle); `to_code()` kept as the one string form | [surface 8](notes/surface-and-machinery.md) |
| Six solver names, `'stochastic'`, `'auto'`, `integrator=` that also picks a backend, `approx=` | `solver=` takes `'auto'`, `'ode'`, `'stochastic'`, `'abm'`, or a named stochastic method; JAX is chosen automatically when gradients are asked for; `integrator=` goes | [solvers 14](notes/solvers-and-execution.md) |
| `ss.modify` and `ss.change` | `ss.change` is `ss.modify` with a step function as its amount, or drop it | [effects 9](notes/effects-and-interventions.md) |
| A modifier under `connectors=` or `interventions=` | the same either way already; accept modifiers in one place, `modifiers=` or any module list | [effects 11](notes/effects-and-interventions.md) |
| `sim.stratify()` and `disease.stratify()`, with two parsers that disagree | one parser; `stratify()` on either object means the same, since dimensions are global anyway | [dims 4](notes/dimensions-and-mixing.md) |
| `ss.Mobility` (commuting and migration), `ss.transfer`, aging, movement registers: six spellings of movement between levels | split "who meets whom" (a route) from "who moves" (a transfer) | [dims 10](notes/dimensions-and-mixing.md) |
| `S \| I`, `S + I`, and the condition `S \| I` | `S + I` for a count, and `\|` only for conditions | [core 15](notes/core-language.md) |
| `ss.parallel(sim, n_runs=)`, `ss.MultiSim(sim, n_runs=).run()`, `sim.run(n_runs=)`, which give different shapes | one MultiSim whose results always have a run dimension; drop `sim.run(n_runs=)` | [results 2](notes/results-and-calibration.md) |
| A result's kind spread over `summarize_by`, a name heuristic, `scale`, and `denom` | `Result(kind='flow' \| 'stock' \| 'ratio')`, which summarizing, resampling, data alignment, and reduction all read | [results 4](notes/results-and-calibration.md) |
| `ss.Treatment` switching behavior on the type of `efficacy`; `ss.Vx`/`ss.Dx`/`ss.Tx` changing class with their arguments | one behavior per class | [effects 13, 14](notes/effects-and-interventions.md) |
| `ss.People(1)` as a placeholder for lazy conditions, which gives a one-agent sim if passed in | an `ss.people` placeholder object | [effects 15](notes/effects-and-interventions.md) |
| `dependence='density'`, specified and counted but never built | delete it from the spec | [dims 13](notes/dimensions-and-mixing.md) |
| `ss.freq` in a declared model, where it means nothing | drop from the counted vocabulary | [core 16](notes/core-language.md) |

**Error codes.** The surface reviewer proposes replacing the numeric codes with exception classes. I would keep both: make each code an exception subclass, so that `except ss.UnknownParameter` works, and keep the code in the message for searching. The table in `15-errors.md` should then be generated from the classes, so a code cannot be listed without being raised, which is how twelve of them drifted ([surface 10](notes/surface-and-machinery.md)).

**Interop.** The eleven importers are about 5,500 lines in the core package, with tests that skip anywhere but Cliff's machine because of hard-coded paths ([surface 13](notes/surface-and-machinery.md)). They are also the evidence for the north-star claim. Moving them to a `starsim_interop` package would keep the core small and let them depend on the other frameworks freely. The trade-off is one more package to release. I would move them, keep SBML and Mermaid in the core since they need no other framework, and fix the hard-coded paths either way.

## 5. A smaller codebase along the seams it already has

| Today | Proposal | Notes |
|---|---|---|
| Every compartmental solver is written in Python and in Numba, and RK4 a third time in JAX | delete the Python copies (about 350 lines); make the kernels the only path, with JAX generated from the same `Program` | [solvers 11](notes/solvers-and-execution.md) |
| One compartmental engine per disease, plus a `Crossing` to keep them in step | one engine per sim running one `Program` | [solvers 12](notes/solvers-and-execution.md) |
| Three random number systems, one of them process-global | the kernels take their generator state from the sim's streams | [solvers 10](notes/solvers-and-execution.md) |
| `solvers.py` at 4,100 lines and `transitions.py` at 3,000, each holding several modules | split along the existing class seams: engines, compartmental, agents, selection; language, stratification, geography, parameter forms | [solvers 13](notes/solvers-and-execution.md), [core 19](notes/core-language.md) |
| Movement between groups as dense groups-by-groups matrices | sparse moves; default aging on 16 ages by 290 places goes from 16 s and 344 MB to about 0.16 s | [dims 7](notes/dimensions-and-mixing.md) |
| Mixing pools in three implementations, with `ss.contacts` as agents building 256 sub-modules | the counts-based path under both solvers, which already exists and is 3× faster with the same answer | [dims 8](notes/dimensions-and-mixing.md) |
| `fit()` as one 300-line function, with likelihoods, priors, and outcomes written twice | split by method, and one definition of each | [results 12](notes/results-and-calibration.md) |
| `ss.Samples` and `ss.Dataset` | v3 machinery disconnected from v4 results; move with the rest of v3 | [results 17](notes/results-and-calibration.md) |
| A state belongs to one module forever, and names come from the caller's frame | the Graph copies what it is given instead of claiming it; frame-based naming keeps working, with `name=` as the explicit form, and a helper function that builds a model no longer breaks it | [core 6, 9](notes/core-language.md), [surface 14](notes/surface-and-machinery.md) |

## 6. The docs and the spec

- **The headline examples fail.** The two-line SIR plots zeros. The full overview example stops at `ss.parallel(baseline=fit.sim)` with `AlreadyRunError`, then at `msim.plot()` with "No legend handles found". The `13-inference.md` example and the layered `ss.contacts` example in `06-routes.md` fail too ([surface 4](notes/surface-and-machinery.md), [dims 1](notes/dimensions-and-mixing.md), [results 8](notes/results-and-calibration.md)). **Proposal:** run every code block in `design/` and `best-practices/` as a test. This is the one change that would have caught most of section 3.
- **The spec no longer describes the implementation.** Twenty-six departures are recorded only in the checklist, among them the expression format, the integrator, the read rule for modified quantities, and the density rule. The spec never mentions `to_spec` or `to_code`. The code's choice is usually the better one: for example, reading a modified quantity gives the base value, which is simpler than the spec's rule ([effects 5](notes/effects-and-interventions.md)). **Proposal:** fold each departure into the spec where the code is right, and fix the code where the spec is right. The checklist then becomes a checklist again.
- **Some spec features are silently missing.** Staged fitting (`method=[...]` crashes), `sim.sensitivity()`, sim-level outcomes, `ss.Sim(results=[...])`, the CRN section of `describe()`, and a converged flag on fits are specified, counted, and not in the checklist as open ([results 10](notes/results-and-calibration.md)). Either build them or remove them from the spec and the count.
- **Each topic appears in three places**: a best-practices page, a spec page, and the checklist, about 9,400 lines in all ([surface 18](notes/surface-and-machinery.md)). Best-practices is the argument, and it is good reading. The spec should be the shorter reference, and it repeats the argument. **Proposal:** cut the "why" from the spec pages and link to best-practices instead. Aim for a spec a newcomer can read in an afternoon.
- **The vocabulary count is not enforced.** `ss.` has 285 public names against 94 counted. `ss.func` is counted and missing, while `ss.Expr`, `ss.Route`, and `ss.rate` are listed as removed and still exported ([surface 5](notes/surface-and-machinery.md)). **Proposal:** a test that compares `dir(ss)` with the table in `17-vocabulary.md`, with v3 names in an explicit allowlist that shrinks over time.
- **The north-star test is measured in prose in another repository.** Several rows fail while the checklist ticks them: LASER 131 s against 3.0 s, SimInf 9.5 s against 0.08 s, and epydemix 10 lines against 7 ([surface 12](notes/surface-and-machinery.md)). The project's standing rule is that a miss is a finding, not a failure, so the fix is to report the misses honestly in the checklist rather than to force them. A script that measures every row would keep them honest.
- **The two copies of the design docs.** They differ in one place: the vocabulary count says 93 in `ssv4` and 94 in `lingua_franca_design`, whose table total is 94. This is a small drift, but it shows that keeping two copies in sync by hand will fail. Once the docs are tested in place, I would make `ssv4` the only live copy and freeze the other as history.

## What to keep

Every reviewer was asked what is good, and their lists overlap:

- The operator form and transitions whose kind follows from the rate.
- The bare state meaning exposure through the routes and `.n` meaning a count, with the linearity check that lets one model run under any mixing.
- States that are not people, inferred, with no extra class.
- The compiled `Program` and its one Numba kernel. The stepped and whole-run paths agree exactly for all five compartmental solvers.
- `describe()`, the model hash, `to_code()`'s output, and a spec loader that never runs code.
- `ss.modify` with a stated combination order and a kept base; lazy conditions against real attributes; "what plus delivery" with one delivery class driven by keywords.
- Data matched to results by name, with automatic alignment; the NUTS path through JAX; CRN streams that agree across all three spellings of a model.
- The printed solver reason, and errors that state the fix (E204, E205, E209 to E211, E602).

## Suggested order

1. **The silent wrong answers in section 3, and the headline examples.** Each is a small fix, and none needs a design decision except the SDE and tau exit order.
2. **The guessing rules in section 2.** Units, keywords, times, dimension kind, and the R0 line at `init()`. These need Cliff's decisions on the exact rules, after which each is small.
3. **Code blocks in the docs run as tests, and the vocabulary test.** Cheap, and it holds every later step to the spec.
4. **One system (section 1).** The large piece; do it after steps 1 to 3 so the tests catch regressions in the port.
5. **The merges in section 4 and the code splits in section 5.** Most are easier after step 4, since half of them exist because of v3.
6. **The spec rewrite in section 6**, folding the departures in and cutting the repeated rationale.

## Notes

- [core-language.md](notes/core-language.md): states, transitions, parameters, units (21 findings).
- [dimensions-and-mixing.md](notes/dimensions-and-mixing.md): stratification, mixing, networks, space (18 findings).
- [effects-and-interventions.md](notes/effects-and-interventions.md): modifiers, conditions, products, deliveries (22 findings).
- [results-and-calibration.md](notes/results-and-calibration.md): results, randomness, calibration, many runs (19 findings).
- [solvers-and-execution.md](notes/solvers-and-execution.md): solvers, the loop, kernels (19 findings).
- [surface-and-machinery.md](notes/surface-and-machinery.md): principles, vocabulary, spec, errors, interop, docs (20 findings).
