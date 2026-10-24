# Starsim v4: design and implementation review

A review of `v4_design/best-practices/`, `v4_design/design/`, and their implementation in `starsim/` on the `v4` branch, as of 2026-10-17. The brief was to look for improvements with nothing sacrosanct, aiming first at simplicity for the user and second at simplicity of the codebase. Nine reviewers each took one area and worked independently (their brief is in `BRIEF.md`); every claim marked "verified" in their files was checked by running the v4 Python. This page is the synthesis: the verdict, the changes I would make first, the themes, the bugs, what not to cut, and the decisions only Cliff can make. The detail, with `file:line` references, repros, and before/after snippets, is in the numbered files:

| File | Area |
|---|---|
| `01-design-spec.md` | The 20 spec files: vocabulary, coherence, open questions, structure |
| `02-best-practices.md` | The 17 best-practices files and whether the design follows them |
| `03-user-api.md` | The API as a new user meets it, by writing and running the spec's own examples |
| `04-transitions-conditions.md` | `transitions.py`, `conditions.py`: the declarative core |
| `05-solvers-kernels.md` | `solvers.py`, `kernels.py`, `gradients.py`: the six solvers and three execution paths |
| `06-core.md` | `sim.py`, `time.py`, `distributions.py`, `modules.py`, `loop.py`, the v3 base |
| `07-interventions-networks.md` | Products, deliveries, modifiers, networks, places, demographics |
| `08-results-spec-inference-interop.md` | Results, the model spec, calibration, many runs, the importers |
| `09-tests-docs-errors.md` | Tests, docs, error codes, notes, `implementation.md` |

## Verdict

The core idea is right and it works. One `ss.Transition` whose kind follows from its rate, states inferred from arrows, `S >> I @ beta * I`, `ss.null`, states that hold counts, `describe()` with every guess marked, data attached by result name, `calibrate()` from bounds, and one declared model running under six solvers: every reviewer who tested these found them to behave as specified, the four spellings hash identically, the timing convention holds under every solver, and the whole-run kernels step a compartment model in 2 µs. None of that should be touched, and the "things that are good" section of every file says so in detail.

The costs are of three kinds, and they are the same three in every area:

1. **Two systems in one namespace.** v4 was built in place beside Starsim 3 and the seam is visible everywhere a user looks. `ss.` exports 258 names against the spec's 94. `ss.SIR` is a v3 class that `describe()`, `equations()`, `ss.modify`, `ss.vaccinate`, and `calibrate()` cannot see. A sim with a v3 module runs in years and one with a declared disease runs in days. `pars=` means sim settings or model values by heuristic. Calibration, results, demographics, interventions, and time each have a v3 vocabulary and a v4 one side by side. The spec's claim that the class form is "the same vocabulary at a longer length" is true of `class SIR(ss.Disease)` and false of everything v3 ships.
2. **One concept, several spellings.** "When" is spelled seven ways across four objects. "Who" is `eligible=`, `where=`, `when=`, and `on=`. Initial conditions have three spellings and three hashes. Every flow result has two names. `ss.change` is a special case of `ss.modify`. `ss.transfer` is `Transition(move=)` under another name and not a `Transition`. Products name their targets `acquisition`/`transmission` where `ss.modify` says `rel_sus`/`rel_trans`. Most of these are listed as open in `19-open-questions.md` and the vignettes that were to settle them have been written; it is time to close them.
3. **Guesses that are wrong often enough to be errors, and errors that do not fire.** The spec's two-line SIR, run as printed, simulates nothing and says nothing. An SEIR without `source=` makes `E` infectious with only a `describe()` mark. A misspelled keyword to `ss.Disease` becomes a parameter. `init=dict(I=1)` seeds one person and `init=dict(I=1.0)` seeds everyone. `E102 WrongDimension`, the spec's own worked example of an error, is never raised. The principle "an error is a claim that no guess is safe" is right and is not being applied strictly enough in about eight places, each listed below.

On the maintainer's side the pattern is duplication rather than size: five RK4 implementations, Python copies of every stochastic solver, three movement mechanisms, two expression languages that crash when mixed, a hand-written second `ss.modify` inside products, three results models, two calibration systems, and 5,700 lines of interop in the core package. About a third of `solvers.py` and half of `calibration.py` and `transitions.py` could go without the user noticing.

## The changes I would make first

Ranked by value to the user per unit of work, across all nine files. Each names the file with the details.

1. **Make the spec's first example work.** Default `init_prev` to a small seed when a disease has a transmission and nothing is given, marked `[guessed]`, or print a one-line note that nothing starts infected; change the canonical `beta=0.05, dur=10` to something supercritical; and let `ss.parallel()` and `run(n_runs=)` take a sim that has already run (call `new()` inside). Today the overview's one-page example fails at the scenarios line and the two-line SIR runs flat zero in silence. (`03` findings 1 and 2, `06` finding 10.)
2. **Split the namespace.** Export the spec's vocabulary plus the handful of v3 names the class form needs (`Module`, `Intervention`, `Connector`, `Analyzer`, `uids`, `FloatArr`, `BoolState`, `options`, the networks and demographics modules) at `ss.`, and move everything else to submodules or a `starsim.legacy` namespace that re-exports today's surface so a v3 script changes one import line. The v3 delivery classes (`routine_vx`, `campaign_vx`, `BaseTriage`, `treat_num`, ...), `simple_vx`, `seasonality`, the deprecated time aliases, the `ss.day`/`ss.week` instances, solver internals, and calibration internals are the bulk of the 164 extra names. (`01` finding 2, `03` finding 9 and the compatibility section, `07` finding 7.)
3. **Redeclare the library diseases.** `ss.SIR`, `ss.SIS`, `ss.SEIR` become `class SIR(ss.Disease)` with the same keywords. Then they run under every solver and every v4 tool works on them, and the class form stops being a second system. `06` finding 2 gives the twelve-line version. The v3 `ss.Infection` base stays for subclassers; it is the instances that move. tbsim's TB has already been redeclared, so this is known to be feasible.
4. **One schedule and one selector vocabulary.** `at=` for rounds everywhere (replacing `years=`/`dates=`, which already means steps in a sim without dates), `interval=`, `start=`/`stop=`; `where=` for who, on transitions, modifiers, and deliveries alike (replacing `eligible=`; `when=` is a `where=` that returns one boolean); drop `on=`, which is verified to be exactly `eligible=` plus a round every step. (`01` findings 5 and 6, `07` finding 1.)
5. **Close the open questions the vignettes have settled.** One flow result name (I would keep the transition's name, `infection`, since it matches the model text, `ss.modify` targets, and data columns; `02` prefers `new_infections` for v3 continuity, which the library classes can keep); `init` as one spec field in the `pars` hash with no int/float rule; `stratify()` idempotent on a repeated call instead of `E305`; `approx=` two-valued; unit-ladder rule 4 an error when typed values disagree; `ss.func`, `dependence='density'`, `n_ref`, `sim.sensitivity()`, `ss.Requires`, and the strain-name list cut from the spec. (`01` findings 4, 7, 8, 9 and its vocabulary section.)
6. **Make the right things errors.** A bare-number keyword to `ss.Disease` that no transition uses (`init_prv=`) is an error naming the nearest match; a `pars=` key that matches a sim setting on a sim with declared modules is `E602`, not a silent setting of the run length; an SEIR-shaped model without `source=` requires it; a bare rate after `@` of one or more per step gets a note asking whether `dur=` was meant; a delivery that finds nobody eligible on every step gets a note; `E102` is raised for a typed product in a rate slot and a rate in a `dur=` slot. (`03` findings 3, 4, 6; `02` finding 5; `06` findings 5 and other findings.)
7. **Fix the stepped-path RNG bug** before anything else in the solvers: on the stepped path, adding an unrelated disease changes another disease's stochastic draws, because the kernels share Numba's global generator and each engine reseeds it. Whole runs are fine. Per-engine, per-step seeds inside the kernels fix it and also make stepped and whole runs agree. (`05` finding 1.)
8. **Drop the Python solvers and step with the kernels always.** They are 330 lines reached only by a function-of-the-sim rate, a dict beta, or a test flag, they are no faster than the stepped kernels, and they are why the same model written two ways (`beta=0.3` versus `beta=lambda sim: 0.3`) gives different random numbers. (`05` finding 2.)
9. **One expression tree.** Merge `Cond` into `Expr` so `where=Q.n < 20` and `rate * (ppl.age > 5)` stop crashing; make arithmetic on a finished transition always a flow weight; make `ss.transfer` a `Transition(move=)`; delete `StateSet`. (`04` findings 1, 2, 4, 6.)
10. **Products are modifiers.** Replace `products.Effect` with real `ss.modify` objects, use `rel_sus`/`rel_trans` as product keys, split `ss.Treatment` by keyword (`cure=` versus `efficacy=`) rather than by the Python type of one argument, and let `ss.vaccinate(efficacy=0.85, coverage=0.7, at=[...])` build its own product. (`07` findings 2 to 5.)

Second tier, each worth doing and each a larger job: move the ten framework importers, dump scripts, and skills to a `starsim_interop` package and keep the core, SBML, and Mermaid (`08` finding 9); cut the v3 `Calibration`/`CalibComponent`/likelihood classes, 44% of `calibration.py` (`08` finding 6); one `Result` with named axes instead of three results models and two resamplers (`08` findings 3 and 4); one `Move` kernel for movers, deliveries, and the register (`05` finding 8); a joint program instead of `Crossing` (`05` finding 10); `sim.groups` and `sim.t.unit` to take 650 lines of bookkeeping off `Sim` (`06` finding 1); `ss.contacts()` as a small matrix route instead of 256 agent-level pools (`07` finding 6); split `transitions.py` along the spec's chapter lines (`04` finding 9).

## Themes in more detail

### Two systems

The v3 compatibility was a deliberate choice (build in place, keep the v3 tests green) and it worked for that purpose. What the reviews show is where it now costs the v4 user:

- `dir(ss)` is 258 names; 74 are used in no test, doc, design page, or library module. (`03` finding 9 has the classification.)
- v4 interventions, modifiers, and calibration do not reach v3 modules (`E504`, `E506`), while v3 modules run beside v4 ones as agents. Coexistence is two disjoint languages. (`03` finding 10.)
- Two time systems: `ss.Sim(ss.SIR(), stop=10)` is ten years, `ss.Sim(ss.Disease(...), stop=10)` is ten steps that are secretly days (`yearvec` and `datevec` are garbage for the unitless sim), and 50 lines exist only to warn about bare-number collisions between the two. One bare-number rule, the spec's, applied to both, with a real step unit. (`06` finding 3.)
- Two demographic systems: `ss.Births`/`ss.Deaths` are agent-only; declared models put births and deaths inside a disease. Accept `ss.birth`/`ss.death` in `demographics=`. (`07` other findings.)
- Two calibration systems, two results models, two product vocabularies, two notions of `requires`, four ways to set a parameter (one of which prints a spurious warning and breaks `ss.modify` afterwards; `06` finding 9), four records of the sim's inputs.
- The docs are entirely v3: no tutorial or user-guide page uses a single v4 form, and the only intervention page documents the classes this review proposes to remove. (`09` finding 4.)

### Several spellings

Beyond the schedule and selector keywords above: `ss.change` is verified identical to `ss.modify(multiply=ss.timeseries(..., mode='constant'))` (two reviewers would cut it, one would keep it as sugar; I would keep it only if `at=` becomes the shared keyword, so that it is visibly the short form); `ss.erlang(k=, mean=)` duplicates `stages=`; `|` and `+` on states are synonyms; `ss.immigration`/`ss.emigration` are `ss.birth`/`ss.death` with a flag; `sensitivity=dict(I=...)` and `detects=` on a test; `ss.Vaccine`/`ss.Vx`; `sim.to_spec`, `disease.to_json`, and `sim.to_json` are three names for two files; six ways to run many sims; `MFNet`'s targeted mode duplicates `PartnerNet`.

### Guesses and errors

The principle in `principles.md` is right: guess when the right answer is obvious, print the guess, and error only when a wrong guess would be silently harmful. The places where the implementation guesses when it should stop, each verified:

| Case | What happens now | Proposal |
|---|---|---|
| No `init_prev` on a transmission model | Flat zero, no note | Seed and mark, or note |
| SEIR without `source=` | `E` infects, `describe()` mark only | Require `source=` when the shape is ambiguous |
| `ss.Disease(init_prv=0.01)` | Becomes a parameter | Error, nearest match |
| `pars=dict(dur=5)` with no model `dur` | Sets the run length | `E602` |
| `init=dict(I=1)` vs `I=1.0` | 1 person vs everyone | No type rule |
| `I >> R @ 10` | 10 per step, no note | Note asking about `dur=` |
| `stop=ss.weeks(12)` with per-day rates | Runs in weeks | Error asking for `unit=` |
| `rate=ss.peryear(0.1) * ss.days(3)` | Silently a number | `E102`, as specified |
| `beta=0.3` plus `n_contacts=10` | Epidemic ten times fiercer, no note | Print the resolved hazard; warn on a bare beta with contacts |
| `ss.transfer(..., eligible=ppl.age > 65)` on an unstratified population | Delivers nothing, ages are 0 to 60 | Note when nobody is eligible on every step; one default age distribution |
| Misspelled destination with inferred states (`'R -> J'`) | A new state, nothing printed | Print the inferred state list |
| `I.n` with a space dimension | Per place in a rate, total in a result | A separate spelling for the local count |

And the error machinery itself: note codes are discarded (`ss.note(code=...)` never uses `code`), warnings paste codes into strings by hand, `verbose=0` suppresses the storing of notes that `describe()` and the conformance suite say are always stored, 59 codes are specified and 42 exist, and the numbering-by-section scheme is ceremony next to one exception class per error name. (`09` findings 1, 2, 5, 6.)

### Spec drift

Things the spec says that the code does not do, not in the departures list (the reviewers found about thirty; the ones a user would hit): `sir.infection.beta` is `None`; `ss.Sim(results=[...])` is not implemented; `integrator='scipy'` dies in SciPy; `ss.MFNet` has no compartmental approximation though it is the spec's example; `sir.solvers()` and `sir.to_json()`-as-spec do not exist; `07-effects.md` is written for `.base` where the code has `.effective`; `13-inference.md` promises ABC-SMC; `per × dur` is a probability in the code and a count in the spec; the CRN section of `describe()` is not there; `ss.transfer` lacks `p=`; `ss.Transition` accepts `beta=`/`source=` though the spec lists six keywords; `ss.contacts('Kenya')` cannot be rebuilt from the spec; `by(...)` inside a rate string does not round-trip and `to_code()` emits a `NameError`. Either side can move; the point is that the three documents (spec, departures, code) should agree, and `15-errors.md` and `17-vocabulary.md` should be generated from the code rather than maintained by hand.

The spec itself restates its three biggest rules in up to eight files each, and the reading list for writing a model is 1,828 lines where 700 would do; `01` proposes a 7-file structure. The best-practices folder has become a stale second copy of the design and should be cut to its "Why", "Trade-offs", "Rejected", and "Open questions" sections and merged to about eight files; its "Rejected" tables are the most valuable institutional memory in the repo and should be kept whole. (`02` finding 2.)

### The north-star tests

`principles.md` sets two tests and `18-conformance.md` demands a table measuring the second. The table does not exist in this repo; lines are recorded for one framework and times for five, of which four fail (SimInf 3.0 s against 0.08 s; LASER 131 s against 3.0 s). Principle 9 (no single-framework features) was set aside in wave 3 to make held-out models agree (`kind='force'`, `to_states`, `match=`, `truncnorm`, `mode='cosine'`, `'A+B'` strains, `immunity=`, ragged `where=`, `ss.Crossing`), and `17-vocabulary.md` does not count any of them. Publish the table with its failing rows, restate the time test as a factor against compiled originals, and decide which of principle 9 and test 2 wins. (`02` finding 1.)

### Codebase

What a maintainer would cut, with lines, in `05` (map of `solvers.py`, a 2,000-line version), `04` (a 1,550-line `transitions.py`), `06` (`Sim` from 59 attributes back toward 21), `08` (`calibration.py` map; interop package), and `07` (`Effect`, the v3 classes, `MFNet` targeted mode). The common pattern: each feature was added as a special case layered on the general path (places in 129 lines across two files, crossings in 28 hooks, three move mechanisms, products as a hand-rolled modifier), and in most cases a general mechanism that already exists (`dim_map`, `Move`, `ss.modify`, `Transition(move=)`, `Program`) would subsume it.

## Bugs found along the way

Not the aim of the review, but each has a repro in its file.

| Bug | File |
|---|---|
| Stepped stochastic path: an unrelated disease changes another's draws (shared Numba RNG) | `05` 1 |
| `beta=lambda sim: 0.3` gives different draws from `beta=0.3` | `05` 2 |
| `ss.parallel(baseline=fit.sim)` and `sim.run(n_runs=)` on a run sim raise `AlreadyRunError` | `03` 2 |
| `sim.run(n_runs=20).plot()` draws one line; `msim.plot(lines=True)` raises | `08` 2 |
| A `new_infections` data column is aligned as a stock (interpolated) not a flow (summed) | `08` 1 |
| `infection_log` records nothing for a declared disease | `03` other |
| A follow-up delivery silently ignores its own schedule | `07` other |
| `ss.Vaccine()` with no `efficacy` crashes at run time | `07` 2 |
| `sir.pars.beta = 0.6` prints a spurious warning and breaks later `ss.modify` | `06` 9 |
| `verbose=0` prevents unit notes from being stored; `describe()` loses the unit line | `09` 2 |
| `verbose=1` forces the stepped path and so changes random numbers | `05` other |
| `Engine.make_rng()` seeds from `sum(ord(c))`, so anagram module names share a stream | `05` other |
| `sim.hash()` and `to_spec()` initialize the sim as a side effect | `08` 10 |
| `Mixing.age_matrix()` stratifies the sim as a side effect | `05` other |
| `resample()` fails on undated sims and takes the mean of a stock | `08` 4 |
| `to_code()` emits bare `by(` for a rate with `ss.by` inside | `08` 5 |
| `where=Q.n < 20` crashes with a `TypeError` about `~` | `04` 1 |
| `msim.plot()` legend crash on the overview's stochastic Kenya sim (repro, not diagnosed) | `03` other |

## What not to cut

Every file has a list; the union: one `ss.Transition` and the bare-state rule; `ss.null`; states that are counts, inferred; `>>` and `@` with name capture (verified in sixty probes; the sharp edges are two and local); the `Expr` tree with one `evaluate()` and pluggable contexts; `describe()` with `[guessed]` marks, unshortened; `sim.solvers()`, `equations()`, `R0()`, `what_modifies()`; `ss.modify` with a base that is never overwritten; `stratify` in place; `approx=` as a switch (two-valued); the ordered loop and `insert()`; the `Delivery`/`Selector` mechanism that runs one declaration on agents and groups; the product/delivery split for reuse; `stratify(patch=geo, mobility=ss.gravity())`; the named data (`ss.contacts('Kenya')`, `ss.people('Kenya')`, `ss.geo`); the flat `Program` shared by Numba and JAX; whole-run kernels; both compiled paths (neither can be the only one; `05` finding 4 says why); the time-point convention; `ss.data` by name with `E705`; `compare()`/`lineup()`; the five-way method choice in `calibrate()` with its printed reason; NUTS via BlackJAX; the particle filter; `interop/core.py`; `ss.diff`; the "Rejected" tables in the best practices; the five-question checklist in `17-vocabulary.md`; hidden CRN in the agents engine; the lazy jump and static cache in `Dist`.

## Decisions for Cliff

Collected from the nine files, deduplicated, and grouped. The reviewers' recommendations are in parentheses where they had one.

**Scope and namespace**
1. Is the v3 vocabulary first-tier in v4, or importable only? A `starsim.legacy` namespace, or leave the names under `ss.` and stop documenting them? (All three reviewers who touched it: not first-tier.)
2. Rewrite `ss.SIR`/`SIS`/`SEIR` declaratively, breaking `sir.infected` and `set_prognoses` overrides? (Yes, with aliases for a release.)
3. Move the framework importers to `starsim_interop`? (Yes; keep core, SBML, Mermaid.)
4. Drop the v3 `Calibration`/`CalibComponent` and likelihood classes, or namespace them? (Drop; add `ss.binomial` to the v4 likelihoods.)
5. Drop `ss.Samples` (432 lines, unused)?

**Vocabulary**
6. `at=` as the one schedule keyword, replacing `years=`/`dates=`? Drop `on=`? Rename `eligible=` to `where=`? Fold `when=` into `where=`?
7. One flow result name: `infection` or `new_infections`? (Two reviewers: the transition's name; one: `new_`.)
8. Drop `ss.change`, or keep it as documented sugar? (Split; my call: keep only with `at=` shared.)
9. Drop `ss.erlang`, `ss.freq` and its four constructors, the deprecated time aliases, `ss.day`-style instances, `|` on states, `ss.immigration`/`ss.emigration` as names, `ss.Requires`, `sde`?
10. Products: `rel_sus`/`rel_trans` keys; `cure=` versus `efficacy=` on `ss.Treatment`; product-less `ss.vaccinate(efficacy=...)`?
11. Should `ss.transfer` be a `Transition(move=)`, and should `ss.transfer(coverage=...)` still return a `Delivery`?
12. Is `sir.pars` meant to hold inline slot values (`dur_recovery`) at all, or only `ss.par`-named ones, with `sir.recovery.dur` as the address?
13. Positional values after an arrow (`ss.progression('I -> R', 10)`): accept and specify, or reject?
14. `pars=` on `ss.Sim`: model values only, with v3's sim dict as the first positional?

**Guesses**
15. Default seed for a transmission model with nothing infected, or an error?
16. Require `source=` for SEIR-shaped string models?
17. Drop the `int` 1 versus `float` 1.0 rule; put `init` in the `pars` hash?
18. Unit-ladder rule 4 and typed `stop=`: an error when they disagree with the model's typed values?
19. A state-space budget past which `'auto'` picks agents rather than building test delays, waning levels, and crossings as compartments?
20. A separate spelling for the local count (`W.here`) when a space dimension exists?

**Solvers and kernels**
21. Drop the Python solvers (losing the pure-Python debug oracle)?
22. `binomial`: default to the ODE's substep rule, or keep whole steps and make `W102` name the solvers without the problem?
23. Replace `Crossing` with one joint program (an O(dt) change in how two diseases' flows are ordered)?
24. Rewrite `sim.gradient()` as `jacfwd` of a `Density`-style function (exact rather than 1e-10)?

**Documents and process**
25. Is `best-practices/` a document people read, or the design's justification appendix? (Appendix; merge to about eight files.)
26. When principle 9 and the line-count test conflict, which wins? Restate the time test, or publish SimInf and LASER as failing rows?
27. Should the spec chapters become the user guide, with `implementation.md`'s departures folded into the spec, its known issues into a limitations page, and its checklist into the changelog?
28. Generate `15-errors.md` and `17-vocabulary.md` from the code (and count the namespace, not the language)?
29. Default goodness of fit: keep normalized absolute error, or choose by result type?
30. Is `to_code()` the primary import artifact, or should `from_spec()` become complete?
