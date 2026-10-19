# Documents, process, and codebase structure: merged findings

This merges every finding the four reviews (C = `codex_review/`, F = `fable_review/`, F2 = `fable2_review/`, O = `opus_review/`) made about the principles and the north-star test, the shape of `v4_design/`, the vocabulary count, the v3/v4 namespace, error codes and notes, the user docs, the headline examples, the test strategy, and the codebase's size and dependencies. The reviews agree on almost every fact and on the direction: the spec and the code have drifted, the user docs contain no v4, the namespace shows three times what the spec counts, the error catalog lists codes that never fire, and nothing runs the spec's own examples. Their numbers differed only because they counted different things; I recounted each one (scripts below) and they are settled here. Two of my checks change a proposal: the v3 delivery and calibration classes that F proposes to delete have real downstream users (hpvsim subclasses four of the `Base*` classes, and four packages use `ss.Calibration`), so a hidden compatibility namespace beats deletion; and the test suite is not slow serially (the 55 core v4 tests take about 110 s on a warm Numba cache, against F's 870 s), so cutting agent counts is the wrong fix. Verdict: the cheapest high-value work in this area is not design but upkeep: make the overview run and test it, write two v4 doc pages, tier the namespace, and generate the vocabulary and error catalog from the code so that they cannot drift again.

All checks were run at `a20c4f7e` with `/software/conda/envs/v4/bin/python`. Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/`.

## Findings

### P1. The spec's two headline examples fail, and nothing runs the spec's code

**The two-line SIR simulates nothing, the one-page overview stops at step 8 and again at step 9, and no test or CI job runs either.**

Raised by: F 12§1, §2, spec-impl bullets 1 and 4; O surface 1, 4; F2 03 f1, f2 (F2 README change 1); F README §3, §4; O README §6; C D8 ("execute the design examples in CI"), C F19.

Kind: docs | test

Status: all four reviews reproduced it; I reran both blocks of `00-overview.md` verbatim (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/overview.py`, output in `overview.out`, with a synthetic `weekly_cases.csv`):

- The two-line SIR prints `solver: ode (...)` and nothing else; `max n_I = 0.0`, `final n_R = 0.0`. With `init_prev=0.01` added, `sim.R0()` is 0.50 and `n_I` never rises above its initial 100 (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/notes.py`). So the example is subcritical even when seeded, which O noticed and F did not.
- The overview: steps 1 to 7 (including `sim.calibrate()`, 0.8 s) run. `ss.parallel(baseline=fit.sim, ...)` raises `AlreadyRunError: Simulation is already complete`; `baseline=fit.sim.new()` works. `msim = ss.parallel(ss.Sim(sir, solver='stochastic', **kw), n_runs=100)` runs, and `msim.plot()` raises `ValueError: No legend handles found`. JAX's "os.fork() ... will likely lead to a deadlock" warning is printed between the calibration and the scenarios.
- `18-conformance.md:168` says "every code example in this folder and in the user documentation is run in continuous integration". CI renders `docs/` (`.github/workflows/test_docs.yml:41`) and never touches `v4_design/`.

Merged proposal: (a) Change the doc example to something supercritical and seeded (`beta=0.3` with `init_prev=0.01`, which gives R0 = 3); whether the library also seeds by default is the defaults agent's call. (b) Make `ss.parallel`/`MultiSim` accept a run sim by calling `sim.new()`, make `AlreadyRunError` name `sim.new()`, and fix `msim.plot()` for outcomes with `every=` (O's reduced repro); those fixes are in the results/runs area. (c) Add one test that extracts every `python` block from `design/00`–`09` and runs it, asserting no exception and, for disease examples, that something happens (`n_I.max() > n_I[0]`). Blocks that are fragments get a marker to skip. This is the single change that would have caught this finding and several bugs the other areas report.

Cost: an afternoon for (a) and (c). Breaks: nothing.

### P2. The user docs teach only Starsim 3, and v4 has no documentation except a changelog entry

**None of the 35 tutorial and user-guide pages uses any v4 construct; the only v4 text a user can find on the docs site is the 3,400-word 4.0.0 changelog entry.**

Raised by: F 12§7 (page-by-page table), §9; F 13 user §3; F2 09 f4, f6; O surface 3, 6, 18; F README §4; F2 README themes; O README §1, §6.

Kind: docs

Status: all four reviews; I recounted (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/docs_v4.sh`, output `docs_v4.out`). Across `docs/tutorials/*.qmd` and `docs/user_guide/*.qmd` (35 pages): `ss.Disease(`, `ss.transmission(`, `ss.progression(`, `>> X`, `solver=`, `describe(`, `ss.modify(`, `ss.vaccinate(` all have 0 hits. The only two pages matching any v4 pattern match `calib.calibrate()`, which is v3's `ss.Calibration`. Against that, `ss.SIR(` appears 30 times on 15 pages, `def step` 28 times on 13, `routine_vx` 12 times on 5. `docs/whatsnew.qmd` includes `CHANGELOG.md`, whose 4.0.0 entry (lines 5 to 58) is 3,421 words, not the 2,000 F estimated. `docs/migration/` stops at `v3.0_v3.1.md`. `git diff --stat main...v4 -- docs` is one file, 10 lines (`docs/publish`); F 13's "14,709 insertions in docs" is wrong. F 12 also found that 34 of the 35 pages still execute under v4 (the failure is a notebook-only `display()`), so the docs are not broken, only from the previous language.

Merged proposal, in order: (a) now, one "v4 in ten minutes" page built from the overview as it actually runs, and one v3-to-v4 migration page (F 12 §9's table of jobs and spellings is the content). Both go under `docs/`, so the existing `test_docs.yml` render tests them for free. (b) After the namespace decision (P3), lift spec chapters 00–09 into the user guide with the MUST register, "Rejected" sections, and best-practices pointers removed (F2 09 f6), make the tutorials the nine build-set models of `v4_design/examples/README.md` (already encoded as tests, F2 09), and move the v3 tutorials under "writing a module in code". (c) `describe()` output is the spine of every page (all four reviews).

Cost: two pages now (a day); the full port is a week or more and should wait for P3. Breaks: nothing.

### P3. The namespace shows 253 objects (285 names with submodules) where the spec counts 94, with no tiering and no test

**v4 added 71 names to `ss.` and removed none; internals, v3 duplicates, and deprecated v2 shims all sit beside the language at the top level.**

Raised by: F 01§1, codebase §3, spec-impl; F 13 user §1, §2; F2 01 f2; F2 03 f9; F2 02 f1; O surface 5, 6; C F20; F README §6; F2 README change 2, decision 1; O README §1, §6.

Kind: surface

Status: all four reviews; the counts are settled in the disagreements section below (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/namespace.py` and `newnames.py`). Main's `__init__.py` imports 187 names, v4's 258; the 71 new ones include solver internals (`Engine`, `Compartmental`, `Agents`, `CountState`, `Crossing`, `Mixing`, `make_engine`, `solver_names`), graph internals (`Graph`, `Strat`, `Transfer`, `Cond`, `Expr`, `CodedError`, `error`, `note`), and wave-3 additions (`Mobility`, `PartnerNet`, `truncnorm`). `ss.Expr`, `ss.Route`, `ss.rate`, `ss.time_prob`, `ss.rate_prob` are listed as removed or deprecated and still exported; `ss.func` is counted and absent; `ss.day`, `ss.week`, `ss.month`, `ss.year` are instances one letter from the classes (`ss.day(1)` raises `TypeError: 'days' object is not callable`). `tests/test_api.py:21` asserts only that the API index has more than 150 entries. Downstream (my check, not raised by any review): the Starsim-based `hpvsim` (branch `main`, 2026-09-28) subclasses `ss.BaseScreening`, `ss.BaseVaccination`, `ss.BaseTriage`, `ss.BaseTreatment` and uses `ss.treat_num`, `ss.routine_vx`, `ss.Dx`/`Tx`/`Vx`; `stisim`, `fpsim`, `rotasim`, and `hpvsim` use `ss.Calibration`; `stisim`, `hpvsim`, `tbsim` use `ss.Births` and `ss.Infection`.

Merged proposal: three tiers, enforced by a test.

1. `ss.` exports the counted vocabulary plus the module-author set (`Module`, `Intervention`, `Analyzer`, `Connector`, `Network`, `uids`, the array and state classes, `Result(s)`, `options`, ...), about 150 names.
2. Internals stay reachable in their submodules (`ss.solvers.Engine`, `ss.transitions.Graph`, `ss.debugtools.mock_sim`).
3. Superseded v3 names move to `starsim/v3.py` and are resolved lazily from `ss` with one deprecation warning naming the v4 spelling, so that `hpvsim` keeps working unedited while tab completion and `dir(ss)` show only tiers 1 and 2:

```python
# starsim/__init__.py
def __getattr__(name):                      # PEP 562: only called for names not found normally
    from . import v3
    if name in v3.moved:
        warnings.warn(f'ss.{name} is a Starsim 3 name; {v3.moved[name]}', FutureWarning, stacklevel=2)
        return getattr(v3, name)
    raise AttributeError(f"module 'starsim' has no attribute '{name}'")
def __dir__():
    return public                           # tiers 1 and 2 only
```

Delete outright what has no users anywhere: the v2 shims `rate`/`time_prob`/`rate_prob` and `warn_deprecation` (`time.py:2384-2416`), the four unit instances, `approx_compare` (`time.py:43`, referenced nowhere). Generate the counted list in `17-vocabulary.md` from one Python list, and add a test that `set(dir(ss)) == counted | module_author` (O 5, F 01 §1, F2 09). Rewriting `ss.SIR`/`SIS`/`SEIR` as declared diseases, which removes most of the double vocabulary from the docs, belongs to the "one language" agent but should land before P2(b).

Cost: a day for the tiering and the test; the v3 module is mechanical. Breaks: code that reached internals through `ss.` (none in the vignettes, library, or downstream packages I checked); v3 names keep working with a warning.

### P4. The checks that matter most never fire, and the commonest first mistakes get Python's errors

**`E604` (a parameter with no value), `E606` (a value outside its bounds), and `E102` (the spec's own worked example) are catalogued and never raised; a v3 user's first guesses (`0.1 * ss.null`, `ss.death(p=)`, `ss.Vx(pars=)`, `sim.init(force=True)`) get raw `TypeError`/`AttributeError`.**

Raised by: F 09§5 and bug 7; F 12§6; F 01 spec-impl (`E101`, `E102`); F2 01 f3; F2 09 other ("plain raises outnumber coded ones": 46 to 30 in `transitions.py`); O surface 9; O README §3 (E606 gives all NaNs under ODE); F README §2 (E102 for products of rates).

Kind: surface

Status: verified by F, F2, O; I reran (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/errors.py`): `rate=ss.peryear(0.1) * ss.days(3)` runs with no error; `ss.par(0.9, bounds=[0.01, 0.5])` in a rate runs with no error; `I >> (0.9*R + 0.1*ss.null)` gives `TypeError: unsupported operand type(s) for *: 'float' and 'Null'`; `ss.death('I ->', p=0.1)` gives `TypeError: death() got an unexpected keyword argument 'p'`; `ss.Vx(pars=dict(efficacy=0.8))` gives `TypeError: Timeline.__init__() got an unexpected keyword argument 'pars'`; `sim.run().run()` gives v3's uncoded `AlreadyRunError`, not `E1003`.

Merged proposal: implement `E604` and `E606` at `sim.init()` (both are the "silently harmful" case principle 3 reserves errors for); implement `E102` for the one case 02 describes (a typed product landing in a `rate=` or `dur=` slot), which also serves F README §2's bare-number bug; make `AlreadyRunError` say "use `sim.new()`" and drop `E1003`; wrap the four or five entry points F 12 §6 lists so the common v3-habit mistake gets a named error that states the v4 spelling. Adopt F2's rule for which errors get a name: "a user can make this mistake in a model", not "it was in the first draft's table".

Cost: two checks and about six messages; a day. Breaks: models that currently run with an unvalued or out-of-bounds parameter, which are wrong.

### P5. What the library says outside exceptions goes through three channels, loses its codes, and depends on `verbose`

**`ss.note(code=...)` never uses `code`; three notes are not stored when `verbose=0`; the solver line is a bare `print`; coded warnings paste their codes into strings by hand; the init notes print twice; and `verbose` means a fraction of steps for agents and a level for compartments.**

Raised by: F 09§3, §4, bugs 4 and 5; F 12§8 and bug 5; F 01§9 (`describe(verbose=True)` promised three times, absent); F2 09 f1, f2, f8; F2 01 f4 (`approx=True` as silence); O surface 10, 20; F README §6 table; F2 README guesses section.

Kind: surface

Status: verified by F, F2, O; I reran (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/notes.py`): `ss.note('something was assumed', code='W101', verbose=1)` prints `'  note: something was assumed'`; the same dated sim has `sim.notes == []` at `verbose=0` and `["unit='day' (from start and stop dates)"]` at `verbose=1` (cause: `sim.py:388-399` wraps the calls in `if pars.verbose:`). `inspect.signature(ss.Sim.describe)` is `(self, output=False)`. `15-errors.md` says notes use Python's `warnings` machinery; `transitions.py:58-71` prints.

Merged proposal: one function, `ss.note(name, msg, sim=)`, that always appends a `(name, msg)` record to `sim.notes`, prints once per sim under `verbose`, and issues a `warnings.warn` of an `ss.Note` category so filters and `ss.options.warnings='error'` work; the solver line becomes a note; coded warnings (`W402`, `W1301`, `W1302`, `W1401`, `W1601`) go through it. `verbose` becomes one integer for every solver (0 nothing, 1 the solver line and notes, 2 the step log), and `approx=` decides only note versus error (two values). Add `describe(verbose=True)` with the per-step conversions, which the spec promises in 01, 02, and 10. Tests then assert on note names instead of substrings (`any('Erlang' in note ...)`, `test_transitions.py:277`).

Cost: about 25 call sites and the test edits; a day. Breaks: printed lines change slightly; `approx=True` users switch to `verbose=0`.

### P6. The error catalog has drifted three ways from the code, and the numbering is tied to chapter numbers

**`15-errors.md` lists 57 codes (its header says 54), the `codes` dict holds 48, the source emits 47; 12 are never emitted, 2 emitted ones are uncatalogued, 6 bypass the dict (so `E605` prints as `E605 :`), and 28 of the 59 have any test.**

Raised by: F 09§5, spec-impl; F2 09 f5, f7, other ("`codes` dict missing names"); F2 01 error-code scheme; F2 02 other ("59 specified, 42 exist"); O surface 9, 10; O README §3, §4; C D8 ("generate vocabulary and error catalogs").

Kind: spec drift | surface

Status: the reviews' counts disagreed; settled below (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/codes.py` and `codes_raised.py`). The message register itself is good (`E204`, `E205`, `E209`, `E305`, `E504`, `E602`, `E705`), with two defects F2 found: `closest()` (`transitions.py:66`) falls back to all options, so "Declared states are S, I, R; did you mean S, I, R?", and `E205` leaks `p=par(0.3)`. None quotes the declaration with a caret as 15's message shape asks.

Merged proposal: one registry in code, the catalog generated from it. Each error is a class named as today (`class UnknownState(ss.CodedError, KeyError)`), so `except ss.UnknownState` and `pytest.raises(ss.UnknownState)` work and no dict can disagree with a call site; the 15 page is generated from the class docstrings. Add `E505`, `E506`, `E605`; drop `E101`, `E1003`, `E1401`, `W207`, `W309`, `W1002`, `W1203`, `W1204`, `W1303` unless implemented (P4 implements `E102`, `E604`, `E606`). One parametrized test triggers every class and asserts the class, that the message contains the user's identifier, and that "did you mean" never repeats the full list. Trim 15's message shape to what is delivered (name the thing, list what exists, state the fix) and drop the caret clause. Numbers: see the disagreement below.

Cost: about 35 three-line classes, a generator, one test; two days. Breaks: `match='E204'` in 28 test lines (mechanical).

### P7. The spec, the departures list, and the code say different things, and the settled open questions are still open

**About forty spec statements are not true of the code and are not in the departures list; the departures that are recorded live only in a 63 KB checklist; and roughly twenty rows of `19-open-questions.md` are already answered by the implementation or the vignettes.**

Raised by: F2 01 coherence, over-specification, open questions (18 of about 45 rows settled); F 01§10, spec-impl; F 09 spec-impl; F 12 spec-impl; F 13 spec-impl; O surface 11; F2 09 f6, f9; F2 02 f2, other; C D8 (MFNet, density, modifier reads); F README §7; F2 README spec drift; O README §6.

Kind: spec drift

Status: verified piecemeal by all four. My spot checks (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/specdocs.sh`, `vocab.py`): no design or best-practices file mentions `to_spec`, 7 mention `to_json`; `ss.func`, `disease.solvers`, `sim.sensitivity`, `sim.crn_report`, and `describe(verbose=)` do not exist; `.prev` is in 06, 14, 17 and two best-practices files while the code has only `.frac`; ABC-SMC is promised in `13-inference.md` and `calibration.md` and absent; `implementation.md:31` still says the SimInf herds run takes 9.5 s where its vignette README now says 3.0 s, and its "no named geographies" departure is contradicted by its own §05. `implementation.md` is 316 lines and 63,222 bytes and holds five things (checklist, performance log, departures, known issues, validation), per F2 09 f9.

Merged proposal: one pass that (a) folds each departure into the chapter it amends, choosing per row as F README §7 does (the code's choice is usually the better one: `.effective`, the five-line fit rule, strings for rates, `per × dur` as a probability); (b) strikes promised-but-absent features or moves them to `deferred.md`, keeping a short build list (`describe(verbose=True)`, `E804` for agent-only attributes, `E403`); (c) closes the settled rows of 19 and gives it a status column; (d) splits `implementation.md`: departures into the spec, known issues into a "Limitations" doc page or issues, the performance log into `tests/benchmark*.yaml`, validation into the generated vignette table (P8), the checklist into the changelog; (e) marks each normative statement as proposed, implemented, or validated, with a link to its test (C D8).

Cost: two to three days of editing. Breaks: nothing in code.

### P8. The north-star test is not measured where the spec says, and it fails on time in a third of the vignettes

**`18-conformance.md` §13 asks for a published table of lines and time per vignette; none exists in this repo. Tallied from the 23 vignette READMEs in the other repo, lines fail in 3 and time fails by more than 1 s of wall clock in 4, while the checklist ticks them.**

Raised by: F2 02 f1; F2 01 coherence (last bullet); O surface 12; C D1, F19; F2 README north-star section; O README §6; F 13 keep (vignettes as acceptance test).

Kind: process

Status: verified by F2 and O from `implementation.md` (five timings) and the vignette READMEs; I tallied all 23 (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/northstar.py`, output `northstar.out`):

| Measure | Pass | Fail | Failing rows |
|---|---|---|---|
| Lines (no more than the original) | 20 | 3 | epydemix `sir_kenya` 7 → 10; SimInf `sir_herds` 12 → 23; SimInf `sise3` 27 → 33 (the SimInf extra lines turn the package's event tables into a register) |
| Time, as reported (first run, including building the sim) | 9 | 14 | mostly sub-0.1 s originals against 0.02–0.04 s of sim building |
| Time, best reported Starsim number (repeated or restarted run) | 14 | 9 | adds `atomica/hiv_dyn` 33x, `odin/stochastic_sir_arrays` 20x, `summer2/sir-age-location` 2.7x, Ghana 1.6x, LASER hub-and-spoke 1.7x with a tenth of the agents |
| Time, slower by more than 1 s of wall clock | 19 | 4 | LASER England and Wales 3.0 → 131 s; SimInf `sise3` 0.10 → 12 s; SimInf `sir_herds` 0.08 → 3.0 s; odin `seir_array` 1.16 → 2.2 s |

Lines are met by wide margins where they are met (EpiModel HIV 405 → 62, EMULSION 280 → 75). Several "time" rows are lost to the cost of building the sim (Atomica SIR: 0.033 s, "nearly all of it building"), which the execution agent may want. O's "SimInf 9.5 s" is the stale checklist figure.

Merged proposal: a script in the repo that reads the vignettes and writes the table, failing rows included (O 12, F2 02 f1), run on demand; the vignette fixtures or a frozen subset under `tests/vignettes/` with `STARSIM_LFD` replacing the six hard-coded `/home/cliffk` paths (`tests/test_interop_*.py`). Restate the time clause (see disagreements). Then fix `implementation.md`'s ticks to match.

Cost: a day for the script. Breaks: a headline claim, honestly.

### P9. Three principles need amending: principle 9 lost to the north star without anyone saying so, principle 3 is applied to values the user gave, and principle 8 has no budget

**Wave 3 added single-framework features to the core to make held-out models agree, against principle 9; several defaults reinterpret what the user wrote, against principle 3; and "approximate rather than refuse" now means two realizations of most features.**

Raised by: F2 02 f1 (principle 9), f9 (principle 8), other (the AI-native section has no status); O README §2 (principle 3 restated); C D1, D3, D4 (stable meaning over brevity); F2 01 vocabulary (`kind='force'` as a principle 9 case).

Kind: process

Status: verified by F2 (feature list) and O (the reinterpretation table). I confirmed the features exist (`Mobility` `kind='force'`, `timeseries(mode='cosine')`, `PartnerNet(match=)`, `ss.truncnorm`) and that the namespace grew by 71 names where 17 counts 34 new (`newnames.out`).

Merged proposal: (a) principle 9 wins for the core namespace; test 2 becomes a measurement, not a requirement that drives features (this is where F2 and C meet). Run 17's five-question checklist ("Adding to the vocabulary") over each wave-3 addition and move the ones that only one framework needs to `starsim.library` or the interop package. (b) Restate principle 3 as O does: a guess may supply a value the user left out, and must say so; it may never change the meaning of a value the user gave, and never depend on something the user does not think of as the model (the step size, how dates are written, a dimension's name). (c) Give principle 8 a budget: past some multiplier of the state space, `'auto'` picks agents and `approx=False` refuses (F2 02 f9; `W309` already exists for the total, and is never raised). (d) Track or drop the AI-native items in `principles.md` (offline doc-tested docs; a CLI or `AGENT.md` contract).

Cost: a paragraph each, then per-feature moves. Breaks: (a) moves a handful of keywords; (b) changes defaults the other agents list.

### P10. The conformance suite is a list in prose; the tests are long scripts with hand-picked tolerances, and some of 18's statistical criteria are wrong

**`18-conformance.md` promises 14 suites over 18 corpus models with golden `describe()` output and recorded tables; what exists is about 60 v4 feature tests, the SIR defined six times, three hand-rolled solver loops, `rtol` from 0.07 to 0.2, and no golden outputs.**

Raised by: F2 09 f3, other (tolerances, giant tests, suite 10 untested, vocabulary not cross-checked); C F19 and §5 (layered contracts; the ODE-versus-stochastic criterion is too strong; qualify the O(dt) ordering claim); O surface 5 (`test_api.py`), 12; F 12§7.

Kind: test

Status: verified by F2 (file and line references) and C (CI inspection); I counted 62 `def test_` in the six v4 files (`test_transitions` 17, `test_deliveries` 21, `test_dimensions` 7, `test_gradients` 8, `test_kernels` 6, `test_named` 3), plus the interop files.

Merged proposal: (a) P1's doc-block test first. (b) `tests/corpus.py` with the corpus models as named constructors, imported everywhere, and `@pytest.mark.parametrize('solver', ...)` instead of per-file loops. (c) Golden `describe()` text per corpus model, diffed and updated through the existing `baseline.yaml`/`update_baseline.py` pattern (which already does "a change in the table is reviewed"). (d) Replace 18's "mean of 1,000 runs within 2 Monte Carlo standard errors at every time point" and its KS test with C's layered contracts: conservation and nonnegativity, exact identities (constant-rate decay, linear birth-death), convergence in `dt` with the observed order, stochastic agreement against known laws with finite-population bias allowed for and a correction for repeated comparisons, backend equivalence (stepped, whole-run, restart, serial, parallel), and metamorphic checks (an unrelated module does not change another's draws, which would have caught the shared-stream bug). (e) The vocabulary and error tests of P3 and P6. Keep every existing assertion when restructuring.

Cost: a week, spread out. Breaks: nothing.

### P11. The design folder is book-length and says each big rule up to eight times; best-practices has become a staler second copy of it

**`design/` is 21 files, 5,457 lines, 69,000 words; `best-practices/` 17 files, 3,938 lines, 47,000 words; `approaches/` another 43,000. The reading list for writing a model is 1,828 lines, 15 of the 17 best-practices files have a "The proposal" section restating a design file, and where they have drifted, the practice is wrong.**

Raised by: F2 01 f10 and structure; F2 02 f2, other (redundancy within the folder; missing practices); O surface 18; C D8; F 01 spec-impl (17's tables duplicate 01–09); F2 README spec drift; O README §6.

Kind: docs

Status: verified by F2 and O; my counts (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/specdocs.sh`): the sizes above; the two-line SIR appears in 14 files; "Rejected" headings in 9 design and 15 best-practices files; "Open questions" headings in 15 best-practices files besides 19; two `[CK:` markers remain. Drift examples F2 verified: `composition-and-effects.md` says reads give the effective value (the code gives the base); `paradigm-conversion.md` says "no declared rank" (code modules declare `compartmental = True`).

Merged proposal: best-practices keeps only "Recommendation", "Why", "Trade-offs", "Rejected", and "Open questions" (every "The proposal" section and lingua-franca code block goes; the design owns them), and its open questions move to 19. Design states each rule once, in one owner file (12 owns approximation, 03 states and the bare-state rule, 05 stratification of counts), with links elsewhere, which F2 estimates removes about 600 lines; implementation detail that constrains nothing observable (the modifier cache protocol, thinning, stream formulas) is cut to its one-line observable. File merging (F2's 7 design files and about 8 best-practices files) is optional and comes last; see the disagreement on how far to go. Keep every "Rejected" bullet.

Cost: two to four days. Breaks: links into the folder.

### P12. The test suite is fast serially; its problems are a race, machine-specific paths, and something about the parallel run

**F reports 952 s of summed test time, 870 s of it v4, and proposes cutting agent counts tenfold; serially on a warm Numba cache the same v4 tests take about 110 s, so the agent counts are not the cost.**

Raised by: F 13§10, bug (`test_samples` race); F README Part 3; O surface 12 (hard-coded paths); F2 09 other (interop tests depend on files outside the repo).

Kind: test

Status: F ran the full suite once under `pytest -n auto` (88.9 s wall, 282 passed, 1 failed). I did not run the whole suite (brief); I timed subsets serially (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/tests_count.out`): `test_deliveries.py`, all 22 tests, 35 s (F: 278 s); `test_deliveries.py::test_change` 1.5 s (F: 46 s); `test_dimensions.py::test_carriage` 3.6 s (F: 51 s); `test_transitions.py` + `test_dimensions.py` + `test_kernels.py` + `test_named.py`, 33 tests, 74 s, the slowest being `test_kernels.py::test_stochastic` at 24 s. F's per-test durations were 10 to 30 times longer inside the xdist run; the likely cause is each worker building the Numba kernels or contending for them, but I did not verify the cause. The race is real: `test_samples.py:32` and `:40` both write `tempdir/'samples_results'`. Six interop test files hard-code `/home/cliffk/...`.

Merged proposal: find why per-test time inflates under xdist (warm the kernel cache once per session in a `conftest.py` fixture, or run kernel-heavy files in one worker group with `--dist loadfile`) before touching agent counts; give `test_samples` `tmp_path` (or delete it with `ss.Samples`, P3); replace the paths with `STARSIM_LFD` and skip with a message naming it.

Cost: hours. Breaks: nothing.

### P13. v4 was added beside v3, and its four largest files each hold several modules

**`git diff main...v4` is 43,466 insertions and 151 deletions, no file, class, or exported name was removed, and `sim.py` (1,396 → 2,692 lines), `solvers.py` (4,136), `transitions.py` (2,972), and `calibration.py` (1,167 → 2,587) each mix concerns along seams that are already visible in the code.**

Raised by: F 13 codebase §1–§8; F 01 codebase §1, §3; C §4 ("avoid having Sim perform every role"; "make validation and serialization usable without `init()`"); F README §8; F2 README codebase and second tier; O README §5.

Kind: codebase

Status: verified by F and C; I reconfirmed the diff (`git diff --shortstat main...v4`: 164 files, +43,466/−151; `starsim/` and `tests/`: 70 files, +27,416/−150), the sizes (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/codebase.out`: `starsim/*.py` 38,278 lines; all of `starsim/` 45,845 including 5,704 in `interop/` and 2,035 in `library/`; 41 test files, 11,934 lines, 281 tests), and that `approx_compare` is referenced nowhere. Strictly dead code is tiny (F: one function and about 140 lines of methods); the gain is in the duplicate v3 families (about 2,300 lines, F 13 §2) and the Python solver path (other agents).

Merged proposal, after the deletions: a `sim.groups` (or `dims`) object in a `dimensions.py` that takes the stratification geometry off `Sim` (about 420 lines) together with `Strat`, `Transfer`, `Mobility`, `geo`, `gravity` from `transitions.py`; solver choice as `choose_solver(sim)` in the solvers package; `solvers.py` as a package (`engine`, `mixing`, `compartmental`, `agents`); `transitions.py` split into expressions, transitions, the graph (kept whole), and dimensions; the matplotlib converters out of `time.py`. Make the module boundary visible with a leading underscore or a per-module `__all__`, which is also what lets P3's test be generated rather than maintained.

Cost: a week, invisible to users. Breaks: internal imports only, if `ss.` exports stay.

### P14. The dependency list does not match what the package uses

**`optuna` and `seaborn` are hard dependencies though the core imports them lazily in one file; `starsim_data`, which the overview needs for `ss.contacts('Kenya')`, is not declared at all; and the interop dump scripts ship inconsistently.**

Raised by: F 13§9, §11; O surface 13 (dump scripts in the wheel).

Kind: codebase

Status: verified by F; I reconfirmed `pyproject.toml` (`dependencies` lists `seaborn` and `optuna`; `starsim_data` appears nowhere; `package-data` is `*.ipynb`, `*.rst`, `*.csv`, so `dump_epimodel.R` is not shipped while the four `dump_*.py` are). Two nuances F missed: `named.py:24` already says `pip install starsim_data` when it is missing; and Optuna is not a side branch, since `default_method` returns `'optuna'` for every stochastic sim (`calibration.py:2085`), so `sim.calibrate()` on an agent-based model needs it. `seaborn` is used only by the v3 `Calibration` plots (`calibration.py:18`). `starsim/__pycache__` holds 537 Numba cache index files.

Merged proposal: drop `seaborn` with the v3 calibration classes; keep `optuna` as long as it is the stochastic default (or replace the 10-line Optuna loop with a SciPy or Sciris optimizer and drop it; the calibration agent's call); declare `starsim_data` (as a hard dependency if its wheel is small, since the overview uses it; otherwise a `[data]` extra); `openpyxl` under an `[interop]` extra; move `dump_*` out of the package (they run in other frameworks' environments). Consider setting `NUMBA_CACHE_DIR` to a user cache when site-packages is read-only (unverified risk, per F).

Cost: an hour, plus the Optuna decision. Breaks: installs that relied on seaborn arriving with Starsim.

### P15. Two copies of the design, and the ssv4 copy points to a corpus it does not contain

**`v4_design/` in ssv4 and the same folders in lingua_franca_design are kept in sync by hand and already differ; `v4_design/examples/` and `v4_design/vignettes/` hold only READMEs; and the design README still marks examples and vignettes "[not done]".**

Raised by: O surface 11, 18; O README §6; F 12 spec-impl (`examples/README.md` describes nine models not in the repo); C D8 ("the vignette corpus currently contains only `v4_design/vignettes/README.md`"); F2 02 f1.

Kind: process

Status: verified by O, F, C; I diffed the two trees: `design/` and `best-practices/` are identical except one line of `17-vocabulary.md:150` ("60 of the 93" in ssv4, "60 of the 94" in LFD, whose table total is 94); `admin/implementation.md` is identical; `admin/` otherwise differs (`deferred.md` only in ssv4; three notes only in LFD). `v4_design/README.md:30-31` says examples and vignettes are "[not done]"; 23 vignettes exist in LFD.

Merged proposal: ssv4 is the one live copy (it ships with the code and its tests); LFD's `design/` and `best-practices/` become a pointer or a frozen snapshot; the design README's status list is updated; the build-set models live in `tests/corpus.py` (P10) and the doc pages (P2), so a reader of ssv4 can find "the SIR in all spellings".

Cost: an hour. Breaks: LFD's Quarto site if it renders those folders (it has `index.qmd` files there), which then needs a link instead.

## Where the reviews disagree

### Namespace size: 253, 258, 285, or 330

Positions: C F20 and F2 03 f9 say 258; F 01§1 says 253; O surface 5 and F 13§2 say 285; F 12§9 says "about 330".

Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/namespace.py`, output `namespace.out`): `[n for n in dir(ss) if not n.startswith('_')]` has 285 names: 32 submodules, 179 classes, 65 functions, 9 other objects. Excluding submodules, 253 objects. An AST pass over `from ... import` in `starsim/__init__.py` gives 259 names, one of which is `pytime` (debug mode only), so 258; those 258 include 6 submodules imported by `from . import x` (`conditions`, `kernels`, `named`, `interop`, `gradients`, `library`) and exclude `root`, set elsewhere. So 258 − 6 + 1 = 253. The numbers agree; they count different things. F 12's 330 matches nothing and is an overestimate.

Settled: **285 public names, of which 253 are objects a user can call or use (the comparable figure to the spec's 94).** v4 added 71 explicit imports (66 objects and 5 submodules) and removed none.

### Error codes: how many exist and how many never fire

Positions: F 09§5 says 57 in the catalog (header 54), 48 in the dict, 7 in the dict never raised, 5 catalogued and absent, 2 uncatalogued; F2 02 says 59 specified and 42 exist; F2 09 f5 says 28 of "the 50 codes in the dict" are tested; O surface 9, 10 say 54 catalogued, 47 in the registry, 12 never raised, 22 catalogued codes untested.

Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/docs/codes.py` and `codes_raised.py`): parsed the first column of every table row in the catalog section of `15-errors.md`; imported `ss.transitions.codes`; grepped quoted codes in `starsim/**/*.py` outside the dict and checked that each sits within three lines of an `error(`, `note(`, `warn(`, `raise`, or `code=`; grepped `tests/` for each code or its name.

Settled:

| Set | Count | Codes |
|---|---|---|
| Catalog tables in `15-errors.md` | 57 (35 E, 22 W; the header says 54: 34, 7, 13) | |
| `codes` dict (`transitions.py:31`) | 48 (35 E, 13 W) | |
| Emitted at a call site | 47 | |
| All distinct codes anywhere | 59 | |
| Never emitted | 12 | in the dict: `E101`, `E102`, `E604`, `E606`, `E1003`, `W207`, `W309`; catalog only: `E1401`, `W1002`, `W1203`, `W1204`, `W1303` |
| Emitted, not in the catalog | 2 | `E505`, `E506` |
| Emitted, not in the dict | 6 | `E605` (prints `E605 :`), `W402`, `W1301`, `W1302`, `W1401`, `W1601` |
| Mentioned in any test, by code or name | 28 of 59 | only two W codes (`W1401`, `W1601`) |

F's inventory is exact. O's "12 never raised" is right, against a wrong base (the header's 54) and a registry off by one (48, not 47). F2's "59 specified" is the union of all three sets, and "42 exist" matches no count I can reproduce.

### The north-star test: replace it, restate it, or publish its failures

Positions: C D1 replaces the universal "no more lines, no more time than every original" with a usability contract: report setup effort, model statements, assumptions a reader must know, time to first result including build and Numba compilation, repeated-run time, peak memory, and agreement separately; treat regressions against Starsim's own baseline as actionable; accept documented trade-offs against specialized software; do not make every longer or slower vignette a defect. F2 02 f1 keeps the test, publishes the table with its failing rows, and restates time as "within a stated factor of an original written in C or C++, and no slower than any Python original", and asks which of principle 9 and test 2 wins. F2 01 offers "lines only, with time reported". O surface 12 keeps lines as the hard rule, restates time as "the same order of magnitude as the original, for the same paradigm, excluding first-use compilation", and generates the table by script. O README: a miss is a finding, so report misses honestly rather than force them.

Recommendation: keep test 1 ("never easier to write without it") as the principle; keep lines as the one hard target, since it is the claim users feel and 20 of 23 meet it by wide margins; replace the time clause with two reported numbers per vignette, time to first result (build plus first run, Numba compilation shown separately) and repeated-run time, judged against what a user notices: a row fails if it is more than 2x slower and more than 1 s slower. Under that rule four rows fail today (LASER England and Wales, both SimInf models, and, narrowly, odin `seir_array`), each with a stated reason (one thread against 20 cores; Gillespie over many herds in a C original). C's extra dimensions (assumptions a reader must know, memory) are worth a column in the generated table but not a pass/fail rule, because no script can count assumptions. Test 2 then stops driving features into the core, which resolves the conflict with principle 9 in principle 9's favor (P9). This takes C's point that benchmark-matching rewards hidden work, keeps F2's and O's point that failing rows must be visible, and keeps the user's measure (seconds) rather than ratios on millisecond runs.

### Legacy namespace or deletion for the v3 families

Positions: F 13§2 (and F README §1) deletes the delivery lattice, `Dx`/`Tx`, `seasonality`, `Samples`/`Dataset`, and `Calibration` with its five likelihoods, about 2,300 lines and 40 names, keeping `Infection`, `Births`/`Deaths`/`Pregnancy`, and `MixingPool(s)` as library modules; F 01§1 softens it with an on-demand alias module `starsim/legacy.py`. F2 README change 2 and F2 01 f2 move them to `starsim.legacy`, re-exporting today's surface so that a v3 script changes one import line; F2 08 f6 offers "legacy or drop" for calibration. O surface 5, 6 put v3-only classes behind deprecation aliases or `ss.v3` for one release; O README: a compatibility module the namespace does not export, or delete. C F20: quarantine compatibility aliases; no immediate mass rename is necessary.

Check: F's case for deletion rests on "zero library, vignette, or v4-test use" (`usage.txt`), which did not look outside the repo. In the sibling repos, `hpvsim` (Starsim-based, `main`, 2026-09-28) subclasses `ss.BaseScreening` (5 uses), `ss.BaseVaccination` (4), `ss.BaseTriage` (2), `ss.BaseTreatment` (1) and uses `ss.treat_num` (6), `ss.routine_vx`, `ss.Dx`, `ss.Tx`, `ss.Vx`; `ss.Calibration` is used in `stisim`, `fpsim`, `rotasim`, and `hpvsim`. `ss.Samples` and `ss.seasonality` are used in none of them.

Recommendation: a hidden `ss.v3` module resolved through `ss.__getattr__` with one `FutureWarning` naming the v4 spelling (P3's snippet). It is simpler for the user than F2's `starsim.legacy` (no import line to change, and `ssl` is already taken by `import starsim.library as ssl`), it keeps `dir(ss)` and tab completion at tiers 1 and 2, which is the point of the cut, and it keeps four downstream packages running. Delete now only what has no users anywhere: the v2 shims, the unit instances, `approx_compare`, `seasonality` (`ss.seasonal` covers it under every solver), `Samples`/`Dataset` (or move them to a separate package), `simple_vx`. Delete the rest of `ss.v3` in v5, once `hpvsim` and the calibration users have ported, with the migration page (P2) as the guide.

### Error codes, exception classes, or both

Positions: F 09§5 keeps the codes as a stable developer handle, makes the dict the catalog, and generates 15 from it; F2 09 f7 makes one exception class per E code, keeps the number only as an optional class attribute, and drops codes from notes; F2 01 renumbers densely before 1.0 or drops numbers; O surface 10 replaces the numeric registry with classes, drops codes from notes, keeps the message shape; O README keeps both, a class per code with the code still printed; C D8 generates the catalog from the code.

Recommendation: classes named as today's code names, which every review agrees on; the catalog generated from them (P6). On the numbers, follow F2 and O: drop them. Users and AI agents search the name (`UnknownState`) as easily as `E204`; the numbers encode chapter numbers (1xx for `02-types.md`, 14xx for `14-model-spec.md`) and will be wrong as soon as P11 merges files; and there are already 22 gaps from earlier trims. If Cliff values the camdl-style catalog, keep the number as a frozen class attribute that is never renumbered, and print it.

### Codes on notes

Positions: F 09§4 routes every coded note through `ss.note(code, msg)` and prints the code (`W1205 StagesUnderAuto: ...`); F2 09 f1 and f7 and O surface 10 drop W codes from notes and keep names or the plain sentence.

Recommendation: a name, not a number (`ss.note('StagesUnderAuto', msg)`), stored with the message. The name gives tests a stable handle and gives filters something to match; a user reads the sentence and never looks a note up, so a printed W number adds nothing to the line.

### `approx=`: three values or two

Positions: F2 01 f4 makes it two-valued (approximate and say so, or error) and lets `verbose=0` silence notes; F2 README agrees ("`approx=` as a switch (two-valued)"); F2 02's "things that are good" lists "`approx=` as a tri-state". The F2 review contradicts itself.

Recommendation: two values. The third value only silences one note, which `verbose` already does, and F2 09 f2 found it also drops the note from `sim.notes`, which is a bug the two-valued form removes.

### How far to merge the design and best-practices, and whether the spec becomes the user guide

Positions: F2 01 merges `design/` from 20 files to 7 (the language; structure; interventions and results; running; fitting; files and other frameworks; discipline), moves "Rejected" sections into 19, and makes 18 `tests/README.md`; F2 02 f2 strips best-practices to "Why", "Trade-offs", "Rejected", and "Open questions" and merges 17 files to about 8; F2 09 f6 turns design chapters 00–09 into the user guide. O surface 18 makes best-practices argument-only, design the spec with one example per concept, open questions only in 19, design at half its length, plus a separate two-page "v4 for Starsim users". C D8: rationale in best-practices, one normative rule per topic in design, a short support matrix linked to tests, proposals, implementation, and validation marked separately. F2's two files disagree on where "Rejected" lives (19 as one table, or kept in best-practices whole).

Recommendation: the user reads `docs/`, never `v4_design/`, so the user-facing win is P2, and the spec's chapters 00–09 are the fastest route to it (F2 09). For `v4_design/`, do the mechanical parts now (strip "The proposal" sections and code from best-practices; fold departures; one owner per rule; open questions only in 19; keep "Rejected" in best-practices, where its arguments are), because they stop drift. Leave renumbering and merging files until the code decisions settle, since every merge breaks links and error-code ranges; F2 01 asks the same question (decision 8).

### When to write the v4 docs

Positions: F 12§7 says before any further API work; F README orders them after "one language" (step 4); O README ports the tutorials after the one-system work.

Recommendation: both. The two short pages (P2a) now, because they force the naming and form decisions and are cheap to rewrite; the full tutorial port after P3 and the library-disease rewrite, so the docs are written once.

### Test runtime

Positions: F 13§10 cuts `total_pop` in `test_deliveries.py` and `test_dimensions.py` tenfold to make them ten times faster. No other review measured runtime; F2 09 reports "67 v4 tests in about 95 s".

Check: serial warm-cache timings in P12 (`test_deliveries.py` 35 s for 22 tests against F's 278 s). Recommendation: the agent counts are not the cost; find the xdist inflation first (P12). F2's figure is consistent with mine.

### Optional `optuna`

Positions: F 13§9 moves `optuna` and `seaborn` to extras because the core "never imports" them.

Check: `seaborn` is v3-only, but `optuna` is v4's default fitting method for every stochastic sim (`calibration.py:2085`). Recommendation: drop `seaborn`; keep `optuna` until the calibration agent decides whether the stochastic default changes, so that `sim.calibrate()` keeps working after `pip install starsim`.

## Keep

- `describe()` with its `[guessed: ...]` marks: the center of the docs and the spine of the golden tests (F 12 keep, F2 09, O surface keep, all READMEs).
- The coded-error mechanism (`CodedError` with `.code`, raised through one `ss.error()`) and the message register of `E204`, `E205`, `E209`, `E305`, `E504`, `E602`, `E705`: name the thing, list what exists, state the fix (F 09, F 12, F2 09, O surface).
- The policy in `15-errors.md` and principle 3: guess and print; error only when no guess is safe (F 09, F2 02).
- Every "Rejected" bullet in best-practices; it is the record that stops removed ideas coming back (F2 02, F2 README, O surface 18).
- `17-vocabulary.md`'s five-question checklist for adding a name (F2 02, F2 README); apply it rather than replace it.
- `implementation.md`'s honesty about departures and known issues; move it, do not lose it (O surface keep, F 13, F2 01).
- The vignettes as the acceptance test for the vocabulary, with "no more lines" as the target (F 13 keep, O surface 12).
- `02-types.md` as the right length and shape for a spec chapter, and 18's suites 1, 3, 9, 13 with the corpus table (F2 01).
- The regression discipline: v3 tests stay green until their classes go (F 13 keep).
- `tests/baseline.yaml` and `test_baselines.py` as the mechanism for recorded, reviewed tables (F2 09).
- Tests that assert real numbers (final size against the analytic SIR, conservation, delivered coverage); keep every assertion when restructuring (F2 09).
- The build set in `v4_design/examples/README.md` as the tutorial sequence (F2 09).
- The 4.0.0 changelog entry's one-paragraph-per-feature shape (F2 09).
- The `__init__.py` import-timing scaffold (F 13, F 01), lazy imports of JAX, BlackJAX, `starsim_data`, NetworkX (F 13), and `named.py` at its size (F 01).
- The capitalization rule (classes capitalized, constructors lowercase), applied consistently in every new name (F 01).
- `kernels.py` as the one numerical core and `Graph` kept whole when `transitions.py` is split (F 13).

## Decisions for Cliff

1. **The north-star time clause.** Replace "no more time" with two reported numbers and a user-noticeable failure rule (more than 2x and more than 1 s), and keep lines as the hard target? (C would go further and drop the universal test; F2 and O restate it and publish failures. My recommendation: restate, publish, keep lines.)
2. **Principle 9 against test 2.** Which wins when they conflict? (F2 asks; C's D1 implies principle 9; I recommend principle 9, with wave-3 single-framework features reviewed against 17's checklist.)
3. **Principle 3 restated** as "a guess may fill in what is missing but never reinterpret what was given"? (O; consistent with C D3/D4 and F2's guess table.)
4. **The v3 names.** Hidden `ss.v3` through module `__getattr__` with a warning, deleted in v5; or delete now? (F: delete; F2, O: a compatibility namespace; C: quarantine. Recommendation: hidden `ss.v3`, because `hpvsim` and four packages using `ss.Calibration` depend on them.)
5. **Error identity.** Exception classes named as today's codes (all reviews), and drop the numbers? (F2 and O: drop or optional; F and O README: keep. Recommendation: drop, or freeze as an attribute if you value the catalog.)
6. **Notes** by name only, stored always, printed under `verbose`? (F2, O: yes; F: keep W codes and print them.)
7. **`approx=`** two-valued? (F2 01 and F2 README yes, F2 02 no; recommendation: two.)
8. **`verbose`** as one integer level for every solver, replacing the v3 fraction-of-steps meaning? (F2 09, F 12; yes.)
9. **The spec as the user guide.** Lift design 00–09 into `docs/` (F2 09), with two short v4 pages written now (F 12, O)? (Recommendation: yes to both.)
10. **Fold `implementation.md`'s departures into the spec now**, and split the rest of it into limitations, benchmarks, the vignette table, and the changelog? (F2 09, O; yes.)
11. **Generate `15-errors.md` and the counts in `17-vocabulary.md` from the code**, with tests that diff them? (All four; yes.)
12. **Best-practices as the design's justification appendix**, stripped of "The proposal" sections now, with file merges later? (F2 02 and F2 README: appendix, merge to about 8; O: argument-only; recommendation: strip now, merge later.)
13. **Promised but absent**: build `describe(verbose=True)`; strike or defer `ss.func`, `sir.solvers()`, `sim.sensitivity()`, `crn_report`, ABC-SMC, `dependence='density'`? (F README §7, F2 01, O 19; recommendation: build the first, strike the rest.)
14. **Which known issues are "won't fix"** (a Limitations page) and which are bugs (issues)? (F2 09.)
15. **The held-out originals**: commit a frozen subset of small fixtures under `tests/`, with `STARSIM_LFD` for the rest? (F2 09, O; yes.)
16. **One live copy of the design** in ssv4, LFD frozen or pointing to it? (O; yes.)
17. **Dependencies**: declare `starsim_data` (hard or `[data]`), drop `seaborn`, keep `optuna` while it is the stochastic default? (F 13, with my nuance on Optuna.)
