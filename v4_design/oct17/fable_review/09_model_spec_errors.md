# The model spec and the error system

I reviewed `v4_design/design/14-model-spec.md` (334 lines) and `15-errors.md` (184 lines), and the code that implements them: `starsim/spec.py` (782 lines: `from_spec`, `to_code`, `diff`, the rate-string `Parser`), the spec writers in `starsim/transitions.py` (`Graph.to_json` at 1764, `outcome.to_json` at 975, `quantity_json` at 1971, `format_expr` at 1927, `Strat.spec` at 2866, plus `by`/`timeseries`/`seasonal.to_json`), `Sim.hash`/`to_spec`/`to_code` in `starsim/sim.py` (1272 to 1350), `ss.load`/`ss.save`/`ss.warn`/`short_hash`/`split_numbers` in `starsim/utils.py`, the `to_json` methods of `ss.modify` and `ss.Delivery`, and `CodedError`/`error`/`note`/`codes` in `transitions.py` (31 to 70). I collected every code in the package with `grep -o "[EW][0-9]\{3,4\}"` and compared it with the catalog in 15. I ran nine scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/spec/` (`rt_overview.py`, `rt2.py` to `rt9.py`): the overview model of `00-overview.md` and two vignettes (summer2 `seir-age-clinical`, MEmilio `secir_ageres`) through `to_spec` → `from_spec` → `to_spec`, `to_code()` with its output run and compared, `diff()` on five pairs of sims, hash invariance checks, and a battery of triggers for the catalog's codes.

## Summary

1. **Make `to_code()` the saved form of a model and demote the JSON spec to a hash input.** Nobody writes the spec, no vignette, test, or doc reads it, `from_spec` cannot read interventions, networks other than a named contact matrix, functions, custom modules, or Starsim 3 modules, and `to_code()` already covers more of the model than `from_spec` does. Keep `to_spec()` as the thing `hash()` and `diff()` consume; drop `from_spec`, `ss.load('sim.json')`, the `Parser`, `joint_split`, and the schema/version story (about 270 lines and a security surface). Breaking, but of a feature nothing uses.
2. **Fix the spec's network entry.** `ss.contacts('Kenya')` is written as a raw `Module.to_json()` dump with `python_class` strings and `AgeGroup` internals (`do_cache`, `ti_cache`, `uids`): 36 KB of a 38 KB document, and all of it goes into the `model` hash. Write what 14 says (`type`, `dependence`, `data`) and nothing else. Not breaking for users; hashes change.
3. **Collapse the output channels to two: exceptions and `ss.note`.** The library currently speaks through `print()` (the `solver:` line), `ss.note` (a print, not a warning, with its `code=` argument silently dropped), `ss.warn` (Python warnings, with codes pasted into the string by hand), and `describe()`. Notes are not recorded when `verbose=0` for three of the codes, so `meta.notes` depends on verbosity. One `ss.note(code, msg)` that always records to `sim.notes`, prints under `verbose`, and is filterable would replace the lot. Not breaking.
4. **Shrink the catalog to the codes that exist and are tested, and give codes to notes or drop them.** Of 57 codes in 15 (the header says 54), 48 are in the `codes` dict, 7 of those are never raised (`E101`, `E102`, `E604`, `E606`, `E1003`, `W207`, `W309`), 5 in the catalog are nowhere in the code (`E1401`, `W1002`, `W1203`, `W1204`, `W1303`), 2 in the code are not in the catalog (`E505`, `E506`), `E605` is raised but missing from the dict so prints as `E605 :`, and no note code has a test. The catalog's own rule ("a code with no test is removed") would cut it to about 28.
5. **Decide what `hash()` and `diff()` mean by "the same model", and make them agree.** `diff()` matches transitions by endpoints and substitutes parameter values, so a renamed parameter or module reports "same" on all four lines; `hash()` hashes names, so it differs. The spec says a spurious hash difference is the safe failure, which is fine, but then `diff()` should say "renamed", not "same".
6. **Write literals in rate strings with full precision.** `format_expr` writes floats with `:n` (six significant figures): `np.pi` becomes `3.14159`, `beta * (1 + 1e-9) * I` becomes `beta * 1 * I`, and the round trip changes the hash, which 14 says must not happen.
7. **Fix `to_code()` so it never emits code that builds a different model.** A function in a rate comes out as `rate = ss.par({'fn': 'clearance'})`, and `ss.seasonal`/`ss.timeseries` inside a rate come out without `ss.` (`NameError` on the MEmilio vignette). Both should go in the "not written as code" footer, which is the right mechanism and already exists.

## Simpler for the user

### 1. The spec is a format no user reads or writes; `to_code()` is the form they want

**Issue.** Section 14 opens with "Nobody is expected to write a model spec by hand" and then spends 334 lines specifying a document format with a versioning scheme, a normalization procedure, an expression grammar, and a round-trip guarantee. In the implementation, the user-facing surface is `sim.to_spec('sim.json')` and `ss.load('sim.json')`. I looked for anyone using it: no vignette in `/home/cliffk/idm/lingua_franca_design/vignettes/` calls `to_spec`, `hash`, `to_code`, or `diff`; no tutorial or user-guide page in `docs/` mentions any of them; the only callers are tests (`test_transitions.py:483`, `test_named.py:36`) and the interop importers, which produce a `Sim` and use `to_code()` to show the user the Starsim equivalent of what they imported (`sim.py:1332`).

What a user gets back from `ss.load` of their own saved spec is partial. From `rt_overview.py`, the overview model of `00-overview.md` itself:

```text
from_spec FAILED: NotImplementedError ss.load() cannot yet read interventions from a model spec
```

Without the intervention it round-trips (`hash equal: True`). From `rt4_vignettes.py` and `rt9.py`:

```text
[memilio]  from_spec FAILED: NotImplementedError ss.load() cannot yet read the network "contacts" from a model spec
[ss.SIR]   ss.load() cannot yet read a module of type SIR from a model spec
[abm]      ss.load() cannot yet read the network "randomnet" from a model spec
[function] ss.load() cannot yet read the value {'fn': 'clearance'} from a model spec
```

`from_spec` reads: declared diseases, their stratifications, a named contact matrix, named people, and `config`. It does not read: interventions (even `ss.modify`, which `to_code` writes), modifiers, any network given as a matrix or any Starsim network class, movement, functions, priors, likelihoods, or any module that is not an `ss.Disease` built from transitions. `deferred.md` lists the matrix case explicitly. The `to_code()` path handles strictly more (modifiers, interventions of type `modify`, named networks, `ss.Mobility` matrices) and lists the rest in a footer instead of raising.

Meanwhile the thing a user actually wants to archive with a paper or send to a colleague is a Python file, and `to_code()` produces one that builds an identical model: on the summer2 vignette the generated code gives the same hash and results agree to 3e-14 (`rt5_runcode.py`).

**Why it matters.** The user has three things that look like "save the model" (`sim.save()` pickle, `sim.to_spec()` JSON, `sim.to_code()` Python) plus `sim.to_json()`, which on a `Sim` is Starsim 3's results export but on a `Disease` is the spec (`rt7.py`: `Disease.to_json` keys are the spec's; `Sim.to_json()` raises `Please run the sim before exporting the results`). Section 14 and 16 say `sim.to_json('sim.json')` is the spec. Three names for two different things, and a loader that works on a fraction of models, is more to learn than it buys.

**Proposal.** Keep `to_spec()` as an internal normalized form that `hash()` and `diff()` read (and importers produce), documented in a page a quarter the length of 14: "this is what is hashed and compared". Make `to_code()` the documented save format: it is readable, diffable in git, complete for anything the spec can hold, and honest about what it cannot hold. Remove `from_spec`, the `Parser`, `joint_split`, the `lingua-franca/MAJOR.MINOR` versioning, `E1401`, and the `ss.load('sim.json')` branch of `ss.load`. Rename `Disease.to_json` to `to_spec` so the two `to_json`s stop meaning different things. If a JSON form is ever needed for a tool that is not Python, the exporters in `interop/` are the place, as 16 already says.

**Cost.** Breaking for anyone who used `ss.load('sim.json')`: nobody in the corpus. The "any implementation can run a spec with no functions" claim in 14 goes; it was not true in practice, since there is no other implementation, and the SBML/odin exporters are the portable path. The conformance item "ss.load(sim.to_json()) gives a sim with an identical spec, for every corpus model" (18, line 46) would be replaced by "to_code() gives a sim with the same hash, for every corpus model", which is already the stronger test and already partly in `test_named.py`.

**Confidence.** High that the loader is unused and partial; medium on removing it outright rather than leaving it undocumented, since Cliff may have a tool consumer in mind that the design folder does not mention.

### 2. The spec's rate strings versus the spec document's nested lists

**Issue.** Section 14 specifies rates as nested lists (`["*", ["par", "beta"], ["state", "I"]]`) and says "Nothing in the spec is a string of code to be parsed", with Cliff's own note that the list form is "extremely hard to read". The implementation writes strings (`"beta * I"`, `rt2.py`) and parses them back with a whitelist `Parser` (`spec.py:62`); `implementation.md` records this as a departure. The departure is the right call for readability, but it creates the one thing `principles.md` line 22 and line 142 argue against (expressions in strings, checked by a parser that knows nothing about epidemiology), and the whitelist is incomplete: `seasonal(...)` and `timeseries(...)` appear in the MEmilio rate string (`rt4_vignettes.py`) but are in neither `time_names` nor `func_names`, so the `Parser` would reject them and `code_expr` does not prefix them. Branch probabilities are written as strings even when they are numbers (`"p": ["0.25", "0.75"]`).

**Proposal.** With finding 1, the strings become display-only (they are what `diff()` prints and what `hash()` hashes) and the `Parser` goes. Update 14 to say rates are written as the text `format_expr` produces, and drop the nested-list grammar table. If the loader stays, add `seasonal`, `timeseries`, and `by` to the whitelist, and write `p` as numbers where they are numbers.

**Cost.** None for the user. The spec document loses its claim that the format is code-free; it was already not.

**Confidence.** High.

### 3. Dates and the `start`/`stop` encoding are inconsistent within one document

**Issue.** `config.start` is `str(ss.date)`, which is `"2020.01.01"` (`sim.py:1309`), while an intervention's `start` is `"2020-03-15"` and the document in 14 shows ISO throughout. `to_code()` therefore emits `start='2020.01.01'`, which Sciris happens to parse, but it is not what anyone writes. For a sim without dates, `start`/`stop` are floats and `unit` says what they mean; `from_spec` and `to_code` both contain a special case to turn `start=0.0, stop=50.0, unit='day'` into `stop=ss.days(50)` (`spec.py:255`, `spec.py:584`).

**Proposal.** Write dates as ISO strings in `config` (`str(date.date())` or `date.isoformat()`), and write `start`/`stop` as typed quantities like every other value (`{"type": "dur", "value": 50, "unit": "day"}`) so the special case disappears from both readers.

**Cost.** Hash of `config` changes once. Not breaking.

**Confidence.** High.

### 4. One channel for what the library says

**Issue.** Counting the ways the library talks to a user outside exceptions and `describe()`:

- `print(f'solver: {solver} ({reason})')` at `sim.py:389`: a bare print, not a note, so it is not in `sim.notes`; the same information is in `meta.solver_chosen`.
- `ss.note(msg, code=, sim=)` at `transitions.py:58`: prints `  note: msg`. The `code` argument is accepted and never used, so the printed line never shows `W101` or `W1205`, and a user cannot look a note up (`rt8.py`: `ss.note('something was assumed', code='W101')` prints `note: something was assumed`). 15 says "Notes keep a W code so that they can be looked up" and "Warnings and notes use Python's warnings machinery"; notes use neither (`rt6.py`: `ss.note` goes through `warnings`: False).
- `ss.warn(...)` at `utils.py:209`: Python warnings, with codes pasted into the string by hand in eight places (`calibration.py:1971`, `1976`; `networks.py:987`; `interop/core.py:463`; `interop/export.py:405`; `siminf.py:726`) and none of those codes in the `codes` dict. `ss.options.warnings = 'error'` turns these into exceptions but cannot touch a note.
- `sim.notes` is meant to be the record that survives `verbose=0` ("Notes are stored in sim.notes, so they are kept even if printing is turned off", `transitions.py:63`), but `W101`, `W103`, and `W212` are emitted inside `if pars.verbose:` (`sim.py:388` to `399`), so with `verbose=0` they are neither printed nor recorded. From `rt9.py`:

```text
verbose=0 notes: ['I_to_R has a lognorm_ex dwell time, approximated by 11 Erlang stages ...']
verbose=1 notes: ["unit='day' (from start and stop dates)", 'I_to_R has a lognorm_ex dwell time, ...']
meta.notes equal: False
```

So `sim.to_spec()['meta']['notes']` depends on the verbosity the sim was built with.

**Proposal.** One function, `ss.note(code, msg, sim=)`, that always appends `f'{code} {name}: {msg}'` to `sim.notes`, prints under `verbose`, and issues a `warnings.warn` of a `ss.Note` category so `warnings.filterwarnings` and `ss.options.warnings` work on it. The `solver:` line becomes a note too. `ss.warn` stays for Starsim 3's uncoded runtime warnings; coded warnings (`W1301`, `W1401`, `W1601`, `W402`) go through `ss.note` with their code so the dict is the single list. Move the three notes out of the `if pars.verbose:` block.

**Cost.** None to the user beyond slightly different printed lines. About 20 call sites.

**Confidence.** High.

### 5. The error catalog: what exists, what is tested, and whether codes help

**Issue.** The catalog in 15 lists 57 codes (I counted the tables; the header says "54 codes, of which 34 are errors, 7 warnings, and 13 notes", and the tables hold 35 errors and 22 warnings/notes). The `codes` dict at `transitions.py:31` holds 48. The sets disagree in four ways:

| Status | Codes |
|---|---|
| In the catalog, in the dict, never raised (one grep hit each: the dict entry) | `E101`, `E102`, `E604`, `E606`, `E1003`, `W207`, `W309` |
| In the catalog, nowhere in the code | `E1401`, `W1002`, `W1203`, `W1204`, `W1303` (`implementation.md` notes the two tau codes) |
| In the code, not in the catalog | `E505 UnknownEffect`, `E506 UnknownDisease` |
| In the catalog and raised, but not in the dict, so the message has no name | `E605` (`rt8.py`: `E605 : the place "Narnia" was not found`); `W402`, `W1301`, `W1302`, `W1401`, `W1601` carry their names by hand in the string |

The conditions behind the never-raised codes are real and currently pass silently or raise something else (`rt8.py`, `rt9.py`):

```text
E606 OutOfBounds:    ss.par(0.05, bounds=[0, 0.01]) in a rate      -> NO ERROR, sim initializes
E604 UnvaluedParameter: ss.par(None, name='beta') in a rate         -> NO ERROR, sim initializes and runs
E102 WrongDimension: rate=ss.peryear(0.1) * ss.days(3)              -> NO ERROR
E101 IncompatibleTypes: ss.perday(0.1) + ss.probperday(0.1)         -> TypeError (Starsim 3's message, no code)
E1003 RunAfterFinish: sim.run().run()                               -> AlreadyRunError (Starsim 3's class, no code)
```

`E604` and `E606` are the two that matter: a parameter with no value running to completion is exactly the "silently harmful" case `principles.md` line 18 reserves errors for. 15's own rule at line 184 ("Every code MUST have a test that triggers exactly it; a code with no test is removed from this list") would also remove every note: tests assert on 26 error codes and two warnings (`W1401`, `W1601`), and no test asserts that any note code fires.

Do the codes help? The messages themselves are good: `E602` and `E504` name the thing the user wrote, list what exists, and suggest the nearest names (`rt6.py`). None of them has the quoted-declaration-with-caret shape 15 asks for, and nothing in `docs/` documents any code (grep for `E204|E804|E602` finds nothing), so the lookup value of a code is currently nil. What a code does give is a stable handle for tests (`pytest.raises(ss.CodedError, match='E204')`) and a `.code` attribute to branch on. That is worth keeping, but it is a developer benefit, and the catalog should be written as such.

**Proposal.** (a) Make the dict the catalog: generate the table in 15 from `ss.transitions.codes` (or the reverse), so they cannot drift. (b) Add `E505`, `E506`, `E605`, `W402`, `W1301`, `W1302`, `W1401`, `W1601` to the dict and route them through `ss.error`/`ss.note`. (c) Implement `E604` and `E606` at `sim.init()`, or delete them from the catalog and say so; the first is better. (d) Drop `E101`, `E102` (Starsim 3's `TypeError`s already cover `E101`, and `E102` needs dimensional analysis nobody has written), `E1003` (`AlreadyRunError` is fine), `W207`, `W309`, `W1002`, `W1203`, `W1204`, `W1303`, `E1401`. (e) Add one test per remaining note that asserts the code is in `sim.notes`. (f) Fix the header count.

**Cost.** Editing a table and about 15 call sites; writing two checks. Nothing a user loses.

**Confidence.** High on the inventory (it is a grep); medium on which codes to drop.

### 6. `hash()` and `diff()` disagree about what "the same model" is

**Issue.** `diff()` matches transitions by name and then by endpoints, substitutes parameter values into rates (`spec.py:638`), and so reports two models that differ only in a parameter's name, or only in the module's name, as identical on all four lines, while `hash()` hashes names and differs (`rt6.py`, `rt9.py`):

```text
--- diff: same model, parameter renamed
model:    same
pars:     same
config:   same
scenario: same
hash equal: False
```

The spec says the same four lines of `diff()` "are the four hashed parts of `sim.hash()`", so a user who sees four `same`s and two different hashes has nothing to go on. Separately, the endpoint change is printed as raw tuples: `transition 'infection' (('S',), ('I',)) -> (('S',), ('E',))`.

**Proposal.** Either hash with names normalized the way `diff()` normalizes (position of a parameter in its sorted usage, not its name), or have `diff()` report `beta renamed transmission_rate` and `module 'sir' renamed 'flu'` under `model`. The second is cheaper and keeps the spec's "spurious difference is the safe direction". Print endpoints as `S -> I` to `S -> E`.

**Cost.** Small. Not breaking.

**Confidence.** High.

### 7. `to_code()` writes guessed values as if they were given

**Issue.** The overview model stratifies the sim by age without a split; the spec records the resolved split and `to_code()` writes `sim.stratify(age=[0.0, 15.0, 65.0], split=[0.15, 0.5, 0.35], kind='age')`. The regenerated model now has a hard-coded split and a `kind=` the user never typed; if the named population changes, the two models diverge. The `stratify` entries of the spec have no `guessed` list (only modules do), so `to_code` cannot tell. Integer levels come back as floats (`[0.0, 15.0, 65.0]`, `ss.days(10.0)`).

**Proposal.** Give `Strat.spec` a `guessed` list like a module's (it already keeps `self.guessed`, `transitions.py:2766`, `2794`), and have `to_code` omit guessed `split`, `kind`, and `people=`. Write integral floats as ints, as `to_code` already does for `total_pop`.

**Cost.** Small; `kind` changes the hash of nothing since `guessed` is skipped.

**Confidence.** Medium: hard-coding the guess is arguably a feature (the file is self-contained), but then it should be marked in a comment.

## Simpler codebase

### 8. The network entry of the spec is a raw object dump

**Issue.** `Sim.to_spec` (`sim.py:1297`) calls `ss.Module.to_json(mod)` for anything without a `graph`, which is `type`, `name`, `label`, and `sc.jsonify(self.pars)`. For `ss.contacts('Kenya')` that is every `AgeGroup` object in `src` and `dst` serialized by Sciris with `python_class`, `do_cache`, `uids`, `ti_cache` (`rt2.py`):

```text
networks entry: chars 36270   total spec chars 38496   python_class occurrences: 32
```

For `RandomNet` it is `{"n_contacts": {"type": "Dist", "classname": "constant", "tracestr": "pars_networks_randomnet_n_contacts", ...}, "dur": {"python_class": "<class 'starsim.time.years'>", ...}}` (`rt9.py`). All of this is hashed into `model` (`sim.py:1285`). The hash was stable across a run in my tests, but it depends on which internals Sciris happens to find on the object, so a refactor of `AgeGroup` changes every hash. Section 14 specifies `{"name", "type", "layer", "dependence", "n_ref", "data"}`.

**Proposal.** Give networks a `to_spec()` that writes the parameters as the user gave them (`ss.contacts(age=matrix)` as `{"by": ["age", "age"], "array": ...}` or a file reference, as 14's "Values" section already describes; a named one as `data`; `RandomNet` as `n_contacts` and `dur` as typed quantities). This also removes the deferred item about matrices not round-tripping.

**Cost.** A `to_spec` on each network class, about 40 lines. Every existing hash changes once.

**Confidence.** High.

### 9. The `functions` field is dead and function slots are written without source

**Issue.** 14 says a function in a slot is recorded under `functions` with its source text and referred to as `["fn", name]`. In the code, `Graph.funcs` is only filled by custom per-step functions (`transitions.py:1192`), so a function in a rate slot produces `"functions": []` and a parameter whose value is `{"fn": "clearance"}` (`rt7.py`). No source text, so the hash does not change when the function body changes, contrary to 14's "the hash changes when it changes". `to_code()` then emits `rate = ss.par({'fn': 'clearance'})`, which is syntactically valid Python that builds a wrong model, with nothing in the footer.

**Proposal.** Record `func_source(f)` in the `fn` leaf (or in `functions`, and reference it), and have `to_code` write the source text if available and otherwise list the function in the footer. Spec 14's "spec with functions is not portable" paragraph then matches the code.

**Cost.** About 15 lines.

**Confidence.** High.

### 10. Size and layering of the spec machinery

The spec is produced by about 1,100 lines spread over seven files: `spec.py` (782), `Graph.to_json` and `outcome.to_json` (about 90), `quantity_json`/`format_expr`/`value_str`/`join_binop` (about 120), `Strat.spec` (10), `by`/`timeseries`/`seasonal.to_json` (25), `modify`/`change`/`Delivery.to_json` (35), `Sim.hash`/`to_spec`/`to_code` (80), `split_numbers`/`short_hash` (35). Of `spec.py`, roughly 270 lines are the reader (`quantity`, `Parser`, `disease`, `strat_kwargs`, `joint_split`, `from_spec`), 340 are `to_code`, and 170 are `diff`. The reader and `to_code` each invert the writers, so every value kind exists three times (`quantity_json` writes, `quantity` reads, `code_value` writes Python). Finding 1 removes one of the three. If the writers emitted Python-ish text directly (`format_expr` already does for rates; `code_value` is 40 lines of doing it for everything else), `to_code` would shrink further, but that is a second step.

`shown()` (`spec.py:618`) and `quantity_str()` (`transitions.py:1955`) both render a typed quantity in a unit for display; `resolved()` (`spec.py:638`) re-implements parameter substitution that `format_expr(expr, values=True)` already does on the live object. `diff()` could work on the live sims rather than on specs for the `model` and `pars` lines and drop `resolved`/`refs`/`shown`.

**Confidence.** Medium; these are consolidations, not removals.

## Spec and implementation

Beyond the departures `implementation.md` lists (string rates, `to_spec` instead of `to_json`, `W1203`/`W1204` unused):

- **Round trip precision.** 14: "Floating-point values are written with full round-trip precision" and `ss.load` "MUST give a model whose own spec is identical". Literals inside rate strings go through `value_str` (`transitions.py:245`, `f'{val:n}'`, six significant figures): `np.pi` is written `3.14159` and `beta * (1 + 1e-9) * I` is written `beta * 1 * I`; the round trip of the latter gives a different hash (`rt8.py`: `hash equal after round trip with 1+1e-9 literal: False`). Named parameter values are written exactly (`0.123456789` survives). Use `repr` for literals in `format_expr` when writing the spec.
- **`E1401 SchemaTooNew`** is specified and not implemented: `from_spec` checks only that `schema` starts with `lingua-franca/` (`spec.py:241`); a `2.0` document loads (`rt7.py`).
- **The three spellings produce different documents.** 14: "the three spellings of the same model MUST produce the same document, once names agree." They produce the same hash, because `guessed` is skipped, but the documents differ in `guessed` (`['states', 'transitions.infection.source', 'transitions.recovery.dwell']` for arrow strings against `['transitions.recovery.dwell']` for explicit `ss.Transition`, `rt7.py`). Either the spec should say "the same document apart from `guessed`" or `guessed` should move to `meta`, where 14 says things that do not change the model belong.
- **Notes are not warnings.** 15: "Warnings and notes use Python's `warnings` machinery." Notes are `print` (finding 4).
- **The catalog header count** is wrong (finding 5), and the catalog says `E605` exists while the dict does not.
- **`sim.to_json`** in 14 and 16 is `sim.to_spec` in the code, and `Disease.to_json` returns the spec while `Sim.to_json` returns Starsim 3's results export. `17-vocabulary.md` lists `to_json` for both and knows nothing of `to_spec`, `to_code`, or `from_spec`.
- **Over-specified.** The `Normalization` section's step 4 (sorting the arguments of `+`, `*`, `&`, `|`) is not implemented (`format_expr` writes operands in the order given) and need not be; the `Values` section's `{"by": ..., "file": ..., "sha256": ...}` form for large arrays is not implemented (arrays are written inline); the `fit` top-level part is not written at all. Open question B11's "convert `ss.prob` to a per-step value" has been answered by the code: the period is kept.
- **Under-specified.** 14 says nothing about how a `Strat` with `where=`, a `geo`, a `split_from`, or a `mobility` register is written; the code writes `geo`, `split_from`, and a `mobility` dict with a hash of the register (`transitions.py:2866`). 14 says nothing about `config.people`, `config.substeps`, `config.use_aging`, or `transitions[].migration`, all of which the code writes conditionally "so that the hash of every other sim is as it was", which is a sign that the hash is being protected from the format rather than the other way around.
- **Open questions in 19 B11 answered by the code.** Period of `ss.prob`: kept. Expanded strata in the spec: not written. Sim-level versus disease-level stratification normalizing to one document: they do not (`stratify` and `modules[].stratifications` are separate). Functions on load: never rebuilt; the loader refuses them.

## Bugs

1. **`to_code()` emits `seasonal(...)` and `timeseries(...)` without `ss.`.** `rt4_vignettes.py` on the MEmilio `secir_ageres` vignette writes `infection = S >> E @ (seasonal(mean=0.05, amplitude=0.2, peak=-0.164...) * timeseries({0: 1.0, 29: 1.0, 30: 0.3}, mode='cosine') * ...)`; running it (`rt5_runcode.py`) gives `NameError: name 'seasonal' is not defined`. Cause: `code_expr` (`spec.py:319`) only prefixes `func_names` and `time_names`.
2. **`to_code()` writes a function-valued parameter as `ss.par({'fn': 'clearance'})`.** `rt7.py`. Cause: `code_value` falls through to the plain-dict branch (`spec.py:313`); `disease_code` only checks `spec['functions']`, which is empty (finding 9).
3. **Literal precision in rate strings** (above): `value_str` uses `:n`. `rt8.py`: `0.1 * cos(3.14159 * I.frac) * I`.
4. **`ss.note(code=...)` ignores `code`.** `transitions.py:58`. Every note in the package passes a code that is never shown or stored.
5. **Three notes are not recorded under `verbose=0`.** `sim.py:388` to `399`; `rt9.py` output above. `meta.notes` therefore varies with verbosity.
6. **`E605` has no name.** `named.py:32` and `182` raise it; it is not in `codes`, so the message reads `E605 : the place "Narnia" was not found` (`rt8.py`).
7. **`E604` and `E606` never fire.** `ss.par(None, name='beta') * I` initializes and runs to completion with no error; `ss.par(0.05, bounds=[0, 0.01])` initializes with no error (`rt8.py`, `rt9.py`).
8. **`from_spec` cannot read the overview model** because it has an `ss.modify` intervention, which `to_code` handles (`rt_overview.py`). Asymmetric for no reason: `modify_code` has the information; a `modify_from_spec` of the same five lines is missing.

## Keep

- **`sim.hash()` as an objdict of four short hashes plus the seed.** It is cheap, `pars` changes when a value changes and `model` does not (`test_transitions.py:88`, confirmed in `rt6.py`), `ss.days(7)` and `ss.weeks(1)` hash the same, labels and `doc` do not count (`rt2.py`). The split into structure and values via `split_numbers` is a small, clear mechanism. Protect it from the network dump (finding 8) and it is right.
- **`to_code()`.** It is the only thing here a user will want to read, it reproduces a vignette exactly (same hash, results to 3e-14), and its footer ("Not written as code (add by hand): the prior of beta (...)") is the right way to be honest about a lossy export. Its use by the importers, with the header "Imported from X by ss.load(); how each rate was read", is the best thing in the interop story.
- **Coded errors as a class with `.code` and `.codename`, raised through one `ss.error()`.** The messages name the user's own identifiers, list what exists, and suggest the nearest match. Keep the mechanism, shrink the catalog.
- **The policy in 15**: guess and print rather than refuse; errors only when no guess is safe. The implementation follows it.
- **`describe()` marking guesses and `guessed` as a list of paths.** Whether `guessed` lives in the module or in `meta` is a detail; the record itself is what makes "defaults are guessed and printed" true.
- **`diff()`'s four-line shape**, once it reports renames.

## Outside my area

- `ss.Sim(..., where=lambda ppl: ...)` on `stratify` is rejected (`where= takes a dict of a dimension and its level`, `rt9.py`), while 07/14 describe `where` as a lazy condition. Whichever is intended, the spec document's `"where": null` field is only ever a dict.
- The bare number after `@` is a rate, so `I >> R @ 10` is 10 per step and `ss.progression('I -> R', dur=10)` is a 10-step dwell time (`rt2.py`, `rt7.py`: the operator form produced a parameter named `rate` with value 10). The overview comment "a duration after @ is a mean dwell time" is true only for typed durations. A reviewer of 03 should look at whether the operator and constructor spellings really are "the same model" for bare numbers.
- `format_expr` constant-folds the user's arithmetic (`(0.91/2*INS.n - 35)/(5*35 - 35)` becomes `(0.455 * INS.n - 35) / 140`), which is fine for hashing but loses the user's derivation in `to_code()` and `equations()`.
