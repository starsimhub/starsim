# Surface and machinery: principles, vocabulary, spec, errors, interop, conformance, and the doc set

Reviewer scope: `best-practices/principles.md` and `README.md`; design `README.md`, `00`, `14`–`19`; `starsim/__init__.py`, `spec.py`, `interop/`, the error/note machinery in `transitions.py`, `sim.py`'s `describe`/`hash`/`to_spec`/`to_code`/`save`; `tests/test_interop*.py`, `test_api.py`. All snippets were run with `/software/conda/envs/v4/bin/python` on branch `v4`; scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/surface/`.

## Summary

The core idea holds up well hands-on. `describe()` is excellent, the best error messages (E204, E205, E602) do what the spec asks, the hash is stable (the same across runs, `ss.days(7)` and `ss.weeks(1)` hash the same, building the model twice gives the same hash), `to_code()` writes readable code that rebuilds the same hash for declared models, and the spec loader does not run code from the file. The problems are mostly at the edges, where v4 meets v3 and where the docs meet the code:

1. **Neither flagship example on the overview page runs correctly as written.** The two-line SIR runs and plots all zeros (nothing seeds the infection, and R0 is 0.5). The full overview example stops at `ss.parallel(baseline=fit.sim, ...)` with `AlreadyRunError`. With that worked around, it stops at `msim.plot()` with `ValueError: No legend handles found`.
2. **Typos get through silently, in ways that matter.** `ss.Disease(..., init_prv=0.01)` becomes a parameter, and the run is an epidemic of zero. `ss.Sim(hiv, syph)` makes `syph` the sim's *label*. A typo in an inferred state name (`'l -> R'`) turns `l` and `R` into reservoirs. Several documented checks (E102, E606, W207) are registered but never raised.
3. **The surface is about twice what is counted, and it is two systems.** `ss.` has 285 public names (32 of them submodules) against the 94 counted, and `Sim` has 145 public attributes. Half of the extras are v3 classes that the v4 machinery cannot serialize, describe, or export (`ss.SIR`, `ss.Births`, `ss.RandomNet`, and `ss.MixingPool` all raise `NotImplementedError` in `ss.load`). `sim.save()`, the v3 way to save, crashes for every v4 sim. None of the tutorials or user-guide pages in `docs/` uses v4 syntax.
4. **There are too many ways to serialize, and they overlap inconsistently.** `disease.to_json()` returns the model spec, while `sim.to_json()` returns v3 results. On top of those there are `sim.to_spec()`, `sim.to_code()`, `ss.to_code()`, `ss.from_spec()`, `ss.load(format=)`, `ss.save(format=)`, `ss.interop.export()`, `sim.save()`, and `sim.to_yaml()`.
5. **The design docs have drifted from the code, and the north star is not actually tested.** Two copies of the docs exist and have already diverged. The spec never mentions `to_spec`/`to_code`. Its expression format, its integrator, and its density rule differ from what the code does, and the departures are recorded only in a 61 KB checklist. The line and time comparison lives in prose in another repository, and several of its rows fail (LASER 131 s against 3.0 s; SimInf 9.5 s against 0.08 s; epydemix 10 lines against 7) while the checklist ticks them.

## Findings, ranked

### 1. The two-line SIR, the language's front door, produces an empty epidemic

- **Type:** design / doc-code mismatch
- **Evidence:** `00-overview.md` "The short form...", and `principles.md` §One vocabulary. Run exactly as written:

  ```python
  sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
  ss.Sim(sir, stop=100).run().plot()
  ```

  `describe()` says `init: S = 1.0 [guessed: nothing given]`, and the results are `max n_I = 0.0, final n_R = 0.0`. Even with `init_prev=0.01`, beta=0.05 and dur=10 give R0 = 0.5, so the plot is a decay from 100. The guess "everyone in the first state" (principles §Guessing defaults) is exactly the "wrong guess that is silently harmful" that principle 3 says should not be guessed.
- **Proposal:** When nothing seeds an infected state (no `init`, no `init_prev`, and no inflow into a state downstream of a contact-mediated transition), guess `init_prev=0.01` and mark it `[guessed: nothing seeds the infection]`, as `n_agents` is guessed. Change the doc's example to `beta=0.3` so that it shows an epidemic. Add a conformance test that runs every code block in `00-overview.md` and checks that there is an epidemic.

  ```python
  # before: flat line, no message
  # after: describe() shows  init: init_prev = 0.01 [guessed: nothing seeds the infection]
  ```
- **Trade-off:** One more guess. A user who wants zero seeding can write `init_prev=0`, which is explicit and rare.

### 2. Unknown keywords to `ss.Disease` silently become parameters

- **Type:** bug (API)
- **Evidence:**

  ```python
  sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prv=0.01)
  ss.Sim(sir, stop=50).run().results.sir.n_I.max()   # 0.0
  sir.pars.keys()                                      # ['init_prv', 'beta', 'dur']
  ss.Disease(..., init_prev=0.01, durr=5, bta=0.4).pars.keys()   # ['durr', 'bta', 'beta', 'dur']
  ```

  Together with finding 1, a typo in `init_prev` gives a valid-looking flat run. `E602 UnknownParameter` exists, but only for `pars=` on the sim.
- **Proposal:** "One keyword per member" should mean transitions, states, outcomes, and `ss.par`s. A bare number passed as a keyword that no transition, outcome, or init expression refers to should raise `E602` with the closest names (`did you mean init_prev?`). A parameter that is used belongs in a rate anyway, so this costs real models nothing.
- **Trade-off:** It forbids declaring an unused bare-number parameter, for later use in `step()` code, by keyword. That case can use `ss.par(...)`, which can stay allowed.

### 3. `ss.Sim(a, b)` makes the second disease the label

- **Type:** bug (API)
- **Evidence:** `starsim/sim.py:65`, `def __init__(self, model=None, label=None, ...)`.

  ```python
  sim = ss.Sim(a, b, stop=50); sim.run()
  type(sim.label)        # starsim.diseases.Disease
  sim.results.keys()     # ['timevec', 'a']   -- b is gone, no message
  ```

  Two diseases is the headline use case of Starsim (co-transmission), and `ss.Sim(sir, ...)` taking a model positionally teaches users to write it this way.
- **Proposal:** `def __init__(self, *models, label=None, ...)`, with everything else keyword-only. A list or a single model still works.

  ```python
  ss.Sim(hiv, syph, networks=...)   # after: both run
  ```
- **Trade-off:** It breaks `ss.Sim(pars_dict, 'label')`. Grepping v3 code shows that label is almost always passed by keyword.

### 4. The overview example breaks at `fit.sim` and at `msim.plot()`

- **Type:** bug / doc-code mismatch
- **Evidence:** Running `00-overview.md` verbatim, with a `weekly_cases.csv` that has a `cases` column:
  - `ss.parallel(baseline=fit.sim, masks=...)` raises `AlreadyRunError: Simulation is already complete`, because `Fit.sim` is "already run" by design (`calibration.py:1928`).
  - With the baseline rebuilt, `msim = ss.parallel(ss.Sim(sir, solver='stochastic', **kw), n_runs=100); msim.plot()` raises `ValueError: No legend handles found` (`run.py:659`). Reduced repro: any disease with `ss.outcome(..., every=ss.weeks(1))`, `solver='stochastic'`, and `n_runs` > 1 fails; without the outcome it works. This is the known issue "with `n_runs=` in one kernel call only states and transitions are kept" showing up as a crash in plotting.
  - Along the way, the calibration uses JAX and then `ss.parallel` forks, which prints JAX's "will likely lead to a deadlock" warning. That is the known issue, on the flagship page.
- **Proposal:** (a) `ss.parallel` and `ss.MultiSim` should accept sims that have already run and keep them as they are, which is simpler for the user than any rule about copying. (b) Skip results that are missing from a run in `msim.plot()`, or better, keep outcomes in the batched kernel run. (c) Run compartmental sims in-process in `ss.parallel`. They take milliseconds, and forking makes them slower: two ODE scenarios take 97 ms in parallel against 21 ms serially. That also removes the JAX deadlock for every ODE scenario run. (d) Execute the overview's code blocks in CI.
- **Trade-off:** None for (a) and (c). (b) costs some kernel work.

### 5. The `ss.` namespace is 285 names, and the counted vocabulary is not enforced

- **Type:** API / design
- **Evidence:** `len([k for k in dir(ss) if not k.startswith('_')])` gives 285: 32 submodules (`ss.time`, `ss.kernels`, `ss.spec`, ...) and 253 objects. Of the 94 counted `ss.` names, `ss.func` does not exist. Names listed as removed are still there: `ss.Expr`, `ss.Route`, `ss.rate`, and `ss.time` (now a module). The 192 uncounted names include internals (`Engine`, `Compartmental`, `Agents`, `CountState`, `Crossing`, `make_engine`, `Graph`, `Strat`, `Transfer`, `Cond`, `CodedError`, `error`, `note`, `mock_sim`, `standardize_netkey`, `module_map`, ...) and the whole v3 delivery zoo (`routine_vx`, `campaign_vx`, `BaseVaccination`, `routine_screening`, `campaign_triage`, `treat_num`, `simple_vx`, `Dx`, `Tx`, `Vx`). Several counted methods are missing: `disease.solvers`, `sim.sensitivity`, `sim.crn_report`, `msim.to_df`. `Sim` has 145 public attributes and `Disease` has 71. `test_api.py` checks only that the index has more than 150 entries, so it would not catch any of this.
- **Proposal:** Give `starsim/__init__.py` an explicit tiered export:
  - The counted names and the library models at `ss.`.
  - Engine and graph internals only in their submodules (`ss.solvers.Engine`).
  - v3-only classes behind deprecation aliases, or under `ss.v3`.
  - A test that asserts `public(ss) == counted ∪ library ∪ allowed_legacy`, with the counted list read from a single Python list that 17-vocabulary is generated from.

  This turns "the vocabulary is small and counted" from a doc claim into an invariant.
- **Trade-off:** It breaks `ss.Engine`-style access in user code (rare), and it needs a deprecation period for the v3 delivery names.

### 6. v3 and v4 coexist as two systems, not one

- **Type:** design / bug
- **Evidence:**
  - `ss.SIR` is the v3 class with `step()`, not a declared disease. `ss.Sim(diseases=ss.SIR(), networks='random')` runs in years from 2000 to 2050 (v3 time defaults), while the declared equivalent runs in timesteps 0–50.
  - For a v3 sim, `describe()` prints only `Sim: 2 modules ... schedule` with nothing about the disease. `to_code()` raises `NotImplementedError: ... cannot yet read a module of type SIR`.
  - `ss.load` of a spec raises `NotImplementedError` for `RandomNet`, `MixingPool`, `Births`, and every intervention (`tocode2.py`).
  - 17-vocabulary says `ss.Vx/Dx/Tx` "remain as aliases" of `ss.Vaccine/Test/Treatment`, but `ss.Vx is ss.Vaccine` is `False`: they are different classes.
  - In `docs/tutorials` and `docs/user_guide`, `ss.Disease(` appears 0 times, `ss.transmission(` 0 times, and `>> I @` 0 times, while `ss.SIR(` appears 15 times and `def step` 13 times. There is no v3→v4 migration guide (`docs/migration/` stops at v3.1).
  - The bare-number unit rule differs between the two systems (known issue: `ss.SIS(dur_inf=10)` keeps the default's unit, while a declared model uses the sim's).
- **Proposal:** Make the library models declared: `ss.SIR`, `ss.SIS`, and `ss.SEIR` are about five lines each as `ss.Disease` subclasses in class form, so the library's own showcase is model-as-data. Give every module a generic spec, `{"type": cls name, "pars": ...}`, which `from_spec` rebuilds with `getattr(ss, type)(**pars)`. That closes most of the `NotImplementedError` gaps for networks and demographics cheaply. Write the v4 tutorial before release.
- **Trade-off:** v3 users who subclass `ss.SIR` and override `step()` will notice. Keep the old class as `ss.v3.SIR` for one release.

### 7. `sim.save()` crashes for every v4 sim

- **Type:** bug
- **Evidence:** `save.py`:

  ```text
  v3        save/load ok
  v4_ode    ERR TypeError 'Shrunk' object is not iterable          (loop.py:308 via sim.shrink)
  v4_abm    ERR RuntimeError Module disease did not successfully shrink: 1717.81 KB > 51 KB
  v4_net    ERR RuntimeError Module disease did not successfully shrink: 542.144 KB > 51 KB
  ```

  `shrink=False` works. The default path (`shrink=True` after a run) fails on every declared disease.
- **Proposal:** Fix `Loop.shrink` (it shrinks `funcs`, which `plan` then iterates), and give `Disease`/`Engine` a `shrink()` that drops the engine's arrays. Add `sim.save()` for each solver to the tests.
- **Trade-off:** None.

### 8. Too many ways to serialize, and `to_json` means two different things

- **Type:** API / design
- **Evidence:** `disease.to_json()` returns the model spec (`diseases.py:240`), while `sim.to_json()` returns v3 `pars`/`summary`/`results` (`sim.py:2275`). `sim.to_spec()` returns the spec (`sim.py:1290`), and `disease` has no `to_spec`. `sim.to_spec(filename)` returns the filename, not the spec. The other entry points are `sim.to_code()`, `ss.to_code(spec)`, `ss.from_spec()`, `ss.load()` (which sniffs eleven formats), `ss.save(format=)`, `ss.interop.export()` (text), `sim.save()` (pickle), `sim.to_yaml()`, and `sim.to_df()`. 14-model-spec and 16-interop still document `sim.to_json('sim.json')` as the spec, and 17-vocabulary says `ss.export` was removed because "a model is a Python script", yet `ss.interop.export` exists.
- **Proposal:** Use one pair of verbs, with the format taken from the extension:

  ```python
  # before
  sim.to_spec('m.json'); sir.to_json('sir.json'); sim.to_code('m.py'); ss.save('m.xml', sim, format='sbml'); ss.interop.export(sim, 'odin'); sim.save('m.sim')
  # after
  ss.save('m.json', sim)    # model spec
  ss.save('m.py',   sim)    # to_code
  ss.save('m.xml',  sim)    # SBML (format= only to override)
  ss.save('m.sim',  sim)    # pickle, as now
  spec = sim.to_spec()      # and sir.to_spec(): the in-memory dict, for diff/hash/tools
  ```

  `ss.load` mirrors this. Remove `ss.to_code`, `ss.from_spec`, and `ss.interop.export` from the top level, and leave `sim.to_json` as the v3 results export (or deprecate it in favour of `to_df`). That is two counted names (`ss.save`, `ss.load`) plus one method (`to_spec`), against eight or so today.
- **Trade-off:** `.json` becomes ambiguous for a non-spec JSON (it already is in `ss.load`), and `.R` is ambiguous between odin and other R. `format=` stays as the override.

### 9. Typos in inferred state names become reservoirs, and documented checks are missing

- **Type:** bug / design
- **Evidence:**

  ```python
  sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('l -> R', dur=10), init_prev=0.01)
  # note: state 'l' treated as a count, not people (no transition connects l to S)
  # note: state 'R' treated as a count, not people (no transition connects R to S)
  # final n_I = 9999.99, n_R = 0
  ```

  `l` has no inflow and starts empty, which is exactly `W207 UnreachableState`, but W207 is registered and never raised. Other codes are registered but never raised (`transitions.py:33-40` and grep): E101, E102, E604, E606, E1003, W207, W309. More are documented but absent from the code: E1401, W1002, W1203, W1204, W1303. That is 12 of the 54 catalogued codes. Two checks show it:
  - The E102 example in 15-errors (`ss.peryear(0.1) * ss.days(3)` as a rate) runs without complaint.
  - `ss.par(0.9, bounds=[0.01, 0.5])` runs with 0.9, although 00-overview's build-time checks list "values outside their bounds".

  `run()` twice raises an uncoded `AlreadyRunError`, not `E1003`. A duration used as a rate (`S >> I @ ss.days(0.3) * I`) raises "has no contact-mediated transition to seed init_prev through", which is true but points away from the mistake.
- **Proposal:** In inferred mode, a state that is the source of a transition but has no inflow, no `init`, and is not the first state should be an error, with a close-name suggestion (`'l' is never entered; did you mean 'I'?`). That suggestion is what E204 already does for declared states. Implement E102 and E606, or delete them from the catalog, as 15-errors itself requires ("a code with no test is removed").
- **Trade-off:** A deliberately empty source state (rare) would need `ss.State(init=0)`.

### 10. Error codes cost more than they return; use exception classes instead

- **Type:** design / code simplicity
- **Evidence:** There are 54 catalogued codes and the registry has 47 (`transitions.py:31-41`). The registry has E505 and E506, which the doc does not. Of the catalogued codes, 22 have no test that mentions them by code or name (E101, E102, E206, E604, E606, E703, E704, E1003, E1301, W101, W102, W103, W1201, W1202, W1205, W1304, W207, W212, W309, W404, W502, W503). Notes carry codes that are dropped: `note(msg, code=None, ...)` never uses `code` (`transitions.py:58`), so the user sees `note: unit='day' (from start and stop dates)` with no `W101`. That defeats the stated purpose of note codes ("so the line can be looked up"). 15-errors says notes go through Python's `warnings` machinery, but they are `print`s, while W1601 and W1401 do go through `ss.warn`. So there are three channels. The numbering follows the design doc's file numbers (1xx = 02-types, 14xx = 14-model-spec), so it will rot as the docs are reorganized.
- **Proposal:** Replace the numeric registry with a few exception classes named as the codes are (`ss.errors.UnknownState(CodedError)`, `UnknownParameter`, `BadBranchSplit`, `CapabilityError`, ...). They are searchable, `except`-able, and testable (`pytest.raises(ss.errors.UnknownState)`), and need no catalog to keep in sync. Drop codes from notes entirely: a note is one line that says what was assumed, and `describe()` lists them. Keep the four-part message shape, which is the part that delivers the value. Shrink 15-errors to the message shape plus the list of classes, generated from the code.
- **Trade-off:** It loses the compact "E204" handle that an AI can grep for. The class name is just as greppable and carries meaning. Cliff may value the camdl-style catalog as a design statement; if so, at least make `note()` print the code and generate the catalog from the registry.

### 11. Two copies of the design, and the spec no longer describes the implementation

- **Type:** docs
- **Evidence:**
  - `~/idm/lingua_franca_design/{design,best-practices,admin}` and `ssv4/v4_design/` are copies. They have already diverged: `17-vocabulary.md` line 150 says "60 of the 94" in one and "60 of the 93" in the other.
  - Drift between design and code:
    - The design docs never mention `to_spec` or `to_code` (0 hits). The spec documents `to_json` in 6 places.
    - 14-model-spec says expressions are nested lists and "nothing in the spec is a string of code", with a `[CK: don't particularly like this]` note. The code writes strings (`"rate": "beta * I"`, `"source": "0.5 * infection"`) but still writes some nested lists (`"dur": {"dist": "expon", "mean": ["par", "dur"]}`), so the actual format is a mix of the two.
    - 12-backends describes SciPy and JAX integration, while the code uses Numba RK4 kernels.
    - 16-interop rule 3 (density dependence with `n_ref`) is replaced in the code by counts.
  - All of this is recorded only in `admin/implementation.md` (316 lines, 61 KB), a checklist that has become the real spec.
  - 19-open-questions is stale: it asks where the SciPy/JAX line falls, and whether a modifier's condition is evaluated on read or in the loop, though the departures section settles it.
- **Proposal:** Keep one copy (the one in this repo, which ships with the code) and delete or symlink the other. Fold every "Departure from the spec" into the design doc it departs from, and keep `implementation.md` as status only. Choose one expression format for the spec (strings, which is what Cliff preferred and the code mostly does) and update 14 to match. Remove the `[CK: ...]` markers once they are settled.
- **Trade-off:** About a day of editing. Without it the spec cannot be used to check the code, which is the purpose of having a spec.

### 12. The north-star test is not measured as specified, and where it is measured, it fails

- **Type:** design / docs
- **Evidence:**
  - 18-conformance §13 asks for a published table with a median of 5 runs after a warm-up, and lines over 100 characters counted twice. 00-overview, design/README claim 6, and principles call this "the test".
  - In practice the numbers are prose tables in `~/idm/lingua_franca_design/vignettes/*/README.md`, in another repository. They are not a pytest, so nothing catches a regression.
  - The rows that fail are ticked `[x]` in implementation.md:
    - Run time: LASER England and Wales takes 131 s against 3.0 s. SimInf takes 9.5 s against 0.08 s for the herds SIR, and 12 s against 0.1 s for SISe3.
    - Lines: epydemix is 10 lines against 7. SimInf is 23 against 12 and 33 against 27.
  - The interop tests that compare with the originals' results hardcode `/home/cliffk/idm/lingua_franca_design` and `/home/cliffk/idm/camdl` (`test_interop_flepimop.py:16`, `test_interop_camdl.py:15-17`, `test_interop_epydemix.py:12`, `test_interop_summer2_atomica.py:13`, `test_interop_memilio.py:12`), so they skip on any other machine.
- **Proposal:**
  - Keep "no more lines" as the hard rule. It is well posed and it is mostly met, by large margins (EpiModel 405 against 62, Atomica 217 against 67).
  - Restate the time rule as "the same order of magnitude as the original, for the same paradigm, excluding first-use compilation". Matching compiled C++ Gillespie from Python is not a design question, and treating it as one invites special cases in the core.
  - Move the vignettes, or a frozen subset of their models with reference results, into `tests/vignettes/`, with a script that writes the table from the code, so the table cannot be stale and the rows that fail are visible.
  - Replace the absolute paths with an environment variable (`STARSIM_LFD`).
- **Trade-off:** It relaxes a headline claim. But the honest version is the one that can be tested.

### 13. Interop: eleven importers in the core package

- **Type:** design / code simplicity
- **Evidence:** `starsim/interop/` is 5,600 lines across 18 files, about 13% of the 43.8k-line package. It includes five scripts that must run inside other frameworks' environments (`dump_epimodel.R`, `dump_memilio.py`, `dump_summer2.py`, `dump_epydemix.py`, `dump_pycamdl.py`), and these ship in the wheel. `ss.load` sniffs every path against eleven formats (`interop/__init__.py:39`). The importers' real value so far has been validating the design against the vignettes, and the comparisons that show that value skip off Cliff's machine (finding 12). Import cost is not an issue (2.7 ms), and all 37 interop and API tests pass in 24 s. The shared core (`Source` → `build()`, with the rate-classification rules printed) is a good design and worth keeping.
- **Proposal:** Split the package:
  - Into `starsim-interop`, a separate package with its own CI and the vignette fixtures: the eleven importers and the `dump_*` scripts.
  - In the core: the model spec, `to_code`, `ss.diff`, and the Mermaid exporter (it is what `diagram()` should use). Possibly also the SBML exporter and importer, as the one neutral standard.
  - Keep `ss.load(file, format='odin')` working through a lazy import that says `pip install starsim-interop` when the package is missing.
- **Trade-off:** A second package to release. Version skew is possible, which the schema version in the spec is meant to handle anyway. Users of `ss.load(format=)` have one more install step.

### 14. Module names from caller frames are fragile, and states cannot be reused

- **Type:** API / design
- **Evidence:**
  - `def make(): flu = ss.Disease(...); return flu` is named `disease`, not `flu`.
  - `make()` assigned to `flu` at the top level, or put in a dict, is also named `disease`, so `sim.results.disease`.
  - At the top level, `a = make(); ss.Sim(a)` is named `a`.
  - The same model therefore has results under `sir`, `disease`, or `a`, depending on where it was built, and the hash changes with the name.
  - Building two diseases from the same `S, I, R` objects raises `E206 DuplicateDeclaration: state "S" is used by transitions in two modules, "sir" and "disease"`. This blocks the natural "build variants in a loop" pattern and needs a factory function, which then hits the naming problem above.
  - Inline operator transitions are named `S_to_I`. The same model in arrow form is named `infection`, so `sim.hash().model` differs, while `ss.diff` reports every part "the same". The diff and the hash disagree, with no explanation.
- **Proposal:** Default a disease's name from its structure, which is deterministic, readable, and needs no frame inspection: the state names lowercased (`sir`, `seir`, `sis`), with `_1` on a clash as now. Use the variable name only when the caller passes `name=`, or the class name in class form. Let a state be shared across modules by copying it on construction: the second `ss.Disease` takes a copy rather than raising. Have `ss.diff` say when the hashes differ only by names.
- **Trade-off:** Results keys change for top-level scripts that rely on `sir = ...` giving `sir`. In the common case the structural name is the same (`sir`).

### 15. A bare number means a duration in one spelling and a rate in another

- **Type:** API
- **Evidence:** `ss.progression('I -> R', 10)` gives `dur = 10`, while `I >> R @ 10` gives rate 10 per step: the same "one vocabulary" with opposite readings of the same number. 15-errors says a positional value after an arrow string is a `TypeError`. In fact `ss.transmission('S -> I', 0.05)` is accepted as `beta`, which is the better behaviour, but the doc is wrong. A user who writes `I >> R @ 10` meaning "ten days" gets 10 recoveries per person per step, with no message (final `n_R` = 103 against the intended thousands).
- **Proposal:** Keep the positional forms (and fix 15-errors). After `@`, print a note when a bare number is ≥ 1 per step (`I >> R @ 10: a rate of 10 per step (mean stay 0.1 steps); for a mean stay of 10 write @ ss.days(10)`). That case is almost always a mistake, and W102 does not cover the ODE solver.
- **Trade-off:** One more note.

### 16. `to_code()` silently writes a different model

- **Type:** bug (minor)
- **Evidence:** For a sim with `ss.RandomNet()`, `ss.MixingPool()`, or `ss.vaccinate(...)`, `to_code()` succeeds and the code it writes builds a sim with a different hash (`model: networks: 1 -> 0`; `scenario: vaccinate -> none`). What was dropped is mentioned only as a comment in the generated file (`# Not written as code (add by hand): ...`). Nothing is printed when `to_code()` is called.
- **Proposal:** Emit `W1401 LossyExport` (or its class equivalent) from `to_code()`, listing what was dropped, as the exporters already do. With finding 6's generic module spec, most of these would not be dropped at all.
- **Trade-off:** None.

### 17. The spec for a named contact matrix is 36 KB of internals

- **Type:** design (spec)
- **Evidence:** `sim.to_spec()['networks']` for `ss.contacts('Kenya')` is 36,270 characters, including `"python_class": "<class 'starsim.networks.AgeGroup'>"`, `"do_cache": true`, and `"ti_cache": -1` for each of 16 age groups. That is `ss.Module.to_json` applied generically. 14-model-spec promises `{"provider": "contacts", "place": "Kenya", "source": ..., "version": ...}`. Because these fields feed `sim.hash().model`, run-state internals are part of the model's identity. They happen to be stable today (the hash is the same before and after a run), but that is fragile.
- **Proposal:** Write named data as its reference, as `out['data'] = mod.ref` already does, and stop there. Write other networks as `{type, pars}` (finding 6). Hash only those fields.
- **Trade-off:** None.

### 18. The design docs are book-length, and every topic appears in three places

- **Type:** docs
- **Evidence:** design/ is about 69k words and best-practices/ about 47k (plus 43k in approaches/), roughly 400 pages. The two-line SIR appears in 14 documents. Every best-practices document has Recommendation, Proposal, Trade-offs, Rejected, and Open questions sections, and so does almost every design document, and 19-open-questions collects them again. So an open question can live in three places, and they disagree (finding 11). principles.md's three-forms example is repeated nearly verbatim in 00-overview, model-structure, and 01-objects. A newcomer gets no route from the root README to a running model: it says vignettes are "[not done]", but they are done, in another repository.
- **Proposal:**
  - Make best-practices argument-only: the "Why" and "Rejected" tables with framework citations, and no code. Code belongs in design, once.
  - Make design the spec: one example per concept, MUST/SHOULD rules, and a link to the test that enforces each rule.
  - Keep open questions only in 19, with a status column.
  - Update the root README's project-status list.
  - Add a two-page "v4 for Starsim users" page in `docs/`, which today has none.

  Target design/ at half its length. Most documents can lose their prose recaps of other documents.
- **Trade-off:** The docs lose some of their argued feel. The argument stays in best-practices.

### 19. Smaller vocabulary cuts and merges worth making

- **Type:** API
- **Evidence and proposal:**
  - `ss.change`: merge into `ss.modify(target, set=..., at=...)`. The docs already call it "shorthand for a modifier".
  - `ss.MFNet` vs `ss.PartnerNet`: both take `n_contacts`, `concurrency`, and `max_contacts` (17-vocabulary §Mixing). That is two names for "partnerships that form and dissolve". Keep `PartnerNet` and make `MFNet` a preset of it.
  - Comparison: `ss.diff`, `ss.diff_sims`, `msim.diff`, `ss.compare`, and `ss.compare_models` are five names for "compare". Fold `diff_sims` into `diff` (structural plus results), and make `ss.compare` (data against results) a method, `sim.compare_data()`, or just what `plot` does.
  - `ss.Requires` and `sim.requires()`: optional, never enforced, and its one payoff (the generated form) is deferred. Cut them until a vignette needs them.
  - `ss.func`: in the count, not in the code, and the doc says "rarely needed". Cut it from the count.
  - Methods `check_dt`, `what_modifies`, `crn_report`, and `sensitivity`: the last two do not exist, and 19 B1 already suggests dropping `what_modifies` "if the vignettes never use it". Grep the vignettes and cut whatever is unused.
- **Trade-off:** Each cut is small. Together they are about 10 names, which is the difference between a vocabulary of "about a hundred" and one of "about ninety".

### 20. Notes are printed twice, and before the line the docs say prints them

- **Type:** API (minor)
- **Evidence:** In the overview, `sim.init()` prints `note: contact matrix ... aggregated` and `note: unit='day' (from start and stop dates)`, then `describe()` prints the same two notes again under `note`. Calling `sim.run()` afterwards prints nothing more, although 00-overview's comment says `sim.run()` "prints the unit ... and solver". Each `ss.parallel` worker reprints all of them.
- **Proposal:** Print notes once per sim, at the first of `describe()` or `run()`, and never from workers. In `describe()`, show them inline where they apply rather than as a trailing list.
- **Trade-off:** None.

## What is good and should be kept

- `describe()`: the right content, marking every guess with its reason, short. It is the best single piece of the v4 surface.
- The good error messages: E204 (`no state named "J" was declared in "disease". Declared states are S, I, R`), E205 (`branch probabilities of "prog" sum to 0.9, not 1`), E602 with the closest names, and E211's explanation of `I` against `I.n`. The four-part message shape is worth more than the codes.
- The hash: four parts, normalized for units, the same before and after a run, the same when built twice.
- `to_code()` output: idiomatic operator-form code with parameters first. For declared models it rebuilds an identical hash, and `ss.load` of the JSON spec gives identical results (difference 0.0 in `n_I`).
- The spec parser accepts only arithmetic, declared names, typed quantities, and whitelisted functions (`spec.py:62-102`), so loading a model never runs code.
- The interop core: every importer builds a `Source`, one `build()` applies the rate rules and prints how each rate was read, and what was not translated is listed in `sim.imported`. It is a clean, testable seam, wherever it ends up living.
- `implementation.md`'s honesty about departures and known issues. The fix is to move that honesty into the spec, not to remove it.
