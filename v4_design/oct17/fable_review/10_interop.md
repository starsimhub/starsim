# Interop: importing other frameworks' models and exporting

I reviewed `v4_design/design/16-interop.md` (and the interop rows of `18-conformance.md`, `15-errors.md`, `17-vocabulary.md`, `19-open-questions.md`, plus the interop entries of `admin/implementation.md` and `admin/deferred.md`), and the code in `starsim/interop/` (6,007 lines: `core.py` 464, `export.py` 411, `__init__.py` 107, eleven importers totaling 4,371 lines (`siminf.py` 727, `odin.py` 527, `camdl.py` 386, `flepimop.py` 363, `summer2.py` 319, `atomica.py` 301, `emulsion.py` 276, `epimodel.py` 268, `epydemix.py` 253, `sbml.py` 251, `memilio.py` 225), and five `dump_*` scripts (654 lines, one in R)), together with `ss.load`/`ss.save` in `starsim/utils.py:651-712`, `to_code`/`diff` in `starsim/spec.py:540-770`, `sim.to_code` in `starsim/sim.py:1332`, the ten `tests/test_interop*.py` files (1,471 lines), and the `skills/import-model` and `skills/validate-import` folders (1,281 lines). I ran `ss.load()` on five originals from `/home/cliffk/idm/lingua_franca_design/examples` (EMULSION `compart_SIR.yaml`, odin `seir_array_odin.R`, flepiMoP `config_sample_2pop.yml`, Atomica `sir_framework.xlsx` with its databook, SimInf `events_SIR.R` with `u0.csv` and the event register), printed and ran each `to_code()`, compared them with the hand-written vignettes, ran `describe()` and `ss.diff()` on the EMULSION import, exported a stratified SEIRD with a contact matrix and a lognormal dwell time to Mermaid, SBML, and odin and re-imported it, and ran `tests/test_interop.py`, `test_interop_export.py`, and `test_interop_odin_sbml.py` (12 passed). Scripts and logs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/interop/`.

## Summary

1. **Keep `core.py` (the `Source` → `build()` pipeline, the rate rules, `to_code`, `ss.diff`) in `starsim`; move the eleven per-framework importers and the `dump_*` scripts to a separate package (`starsim-interop` or the design repo).** They are 5,000 lines that track eleven moving targets, five of them need the other framework (or its compiler, or R) installed anyway, their tests run only on one machine, and their value to a user is one `to_code()` file they then edit. Breaking only for `ss.load(file, format=)`, which would say to install the companion package. (Finding 1, Simpler codebase 1.)
2. **Fix the bare-number unit rule before anything else.** Every importer writes dimensionless parameters as bare numbers, and the core types every bare number in a rate as a rate per unit: a hand-written `S >> E @ R0 * gamma * I` with `R0 = 2.5`, `gamma = 0.2` gives a peak of 1,278 infected at `unit='day'` and 83 at `unit='week'`. Imports are right only at the unit they were imported at, and `to_code()` writes `Ro = ss.par(ss.perday(2.5))`. Core bug, surfaced by interop. (Bugs 1.)
3. **Make `to_code()` the product and judge it as code.** It is already close to hand-written for simple models (EMULSION: 37 lines against the vignette's 11, the difference almost all comments and float noise), but it writes a 48,000-character line for SimInf's 1,600 nodes, drops the movement register, and prints splits to 17 digits. Give large arrays a companion data file and round numbers to what the source had. Not breaking. (Simpler for the user 2.)
4. **Drop the lossy SBML and odin exporters, keep Mermaid, and give it to `sir.diagram('mermaid')`, which is a second, poorer Mermaid writer today.** The lossy exports are honest about what they lose but the result is not the model: an age-stratified SIR with a contact matrix exported to odin and re-imported differs from the original by up to 9,211 infected on a peak of 9,387. `W1401 LossyExport` then has nothing to attach to. Breaking for `ss.save(..., format='sbml'|'odin')`. (Simpler for the user 4, Simpler codebase 3.)
5. **Rule 2 of 16 should accept a parameter equal to the population as "the total".** odin's `beta * S[i] * I_tot / N` with `N <- 1e7` is read as "mass action, density-dependent" and imported as a spontaneous `S -> E @ I.n / N` that no network, `rel_sus`, or `rel_trans` will touch, and the message says "not divided by the total" about an expression visibly divided by `N`. Not breaking. (Simpler for the user 3.)
6. **Specify the surface that exists.** `sim.to_code()`, `ss.to_code()`, `sim.imported`, `ss.interop.Source`, `ss.interop.build()`, `ss.interop.sniff()`, format-specific keywords (`u0=`, `events=`, `databook=`, `scenario=`, `run=`, `rscript=`, `camdlc=`, `pars=`, `machines=`), and `ss.interop.summer2.outputs()` appear in no spec section and `17-vocabulary.md` lists `ss.export` as gone. Either spec them or make them private. (Spec and implementation 1.)
7. **Of the fourteen "gaps found while writing the importers" in `deferred.md`, four point at real holes in the language** (a transition's own flow usable in a rate; a one-off scheduled move between states; `substeps=` forced by one Erlang stage; the contact matrix not in the spec); the rest are fine as "not translated". (Spec and implementation 3.)

## Simpler for the user

### 1. Eleven importers inside `starsim` buy the user little and cost the maintainer a lot

**Issue.** `import starsim` loads `starsim.interop` eagerly (`starsim/__init__.py:155`) with eleven importers for eleven frameworks. The import-time cost is negligible (3.5 ms by `-X importtime`), so the question is purely maintenance and promise. What a user does with an import is, per the `import-model` skill and the vignettes: `ss.load()` once, `to_code('model.py')`, then edit `model.py` by hand for the listed `W1601` items. The importer is a one-shot scaffold, not a library call that stays in anyone's model.

What the importers depend on, verified against the v4 environment (none of summer2, epydemix, memilio, pycamdl, emulsion, libsbml is installed there):

| Importer | Needs the other framework? | How |
|---|---|---|
| emulsion, odin, sbml, siminf, flepimop, atomica | No | Parses the file (YAML, the odin DSL, XML, a hand-written R subset `siminf.py:36-343`, YAML+CSV, xlsx) |
| summer2, epydemix | Yes, or a JSON from `dump_*.py` run in that environment | Live object or dump |
| memilio | Yes: `dump_memilio.py` run in MEmilio's environment (the models are C++) | Dump only |
| epimodel | Yes: `Rscript` with EpiModel to run `dump_epimodel.R` (or its JSON) | Dump only |
| camdl | Yes: the `camdlc` compiler binary (or pycamdl, or its IR JSON) | Compiler |

So for five of eleven the user installs the other framework anyway; the "it just reads the file" promise holds for six. Tests of all but the inline fixtures point at `/home/cliffk/idm/lingua_franca_design` (8 of 10 test files, `tests/test_interop_camdl.py:15-17` also `/home/cliffk/idm/camdl/pycamdl` and `_envs/camdl/bin/camdlc`, `tests/test_interop_memilio.py:13` `_envs/memilio/bin/python`), so the against-the-original checks run on one machine only.

**Why it matters.** The other frameworks change (camdl's IR, epydemix's JSON, summer2's internals, MEmilio's bindings); every breakage lands in `starsim`'s test suite and release cycle. A user who never imports anything still ships 5,000 lines they do not need, and the docs (none of `docs/tutorials` or `docs/user_guide` mentions interop) do not teach them.

**Proposal.** Keep in `starsim`: `core.py` (`Source`, `build`, the rate rules), `ss.load(spec.json)`, `to_code`, `ss.diff`, `sniff` reduced to the formats shipped. Move the eleven importers, the five dumps, the per-framework tests, and the `import-model` reference guides to a companion package (`starsim_interop`, or the design repo, where the originals already live), registered through a small hook: `ss.load(..., format=)` looks up `ss.interop.formats`, which the companion fills on import, and otherwise says `pip install starsim-interop`. `Source`/`build` stays public since the `import-model` skill tells people to fill a `Source` for a framework with no importer.

**Cost.** Breaking for `ss.load('model.yaml', format='emulsion')` without the companion package (one clear error). Two repos to keep in step, but the companion can version itself against each framework rather than against Starsim. Spec `16-interop.md` needs a paragraph saying importers are a companion package; the table of sources stays. **Confidence: high** on the split; **medium** on exactly which six file-parsing importers one might still keep in core (EMULSION and SBML are the spec's own "worth doing first" and "at least SBML and one framework" conformance level).

### 2. `to_code()` output: close to hand-written for small models, unusable for spatial ones

**Issue.** Judged on five imports (`scratchpad/interop/code_*.py`):

- EMULSION `compart_SIR` (37 lines): body is `I_to_R = I >> R @ recovery`, `force_of_infection = S >> I @ transmission_I * I`, parameters as `ss.par(ss.perday(0.1), doc=...)`, states with labels. The vignette's hand version is 11 lines using `ss.transmission('S -> I', beta=ss.perday(0.5))`. The 26 extra lines are the docstring ("How each rate was read", repeating the printed import log), one line per state and parameter, and `init = dict(S=99, I=1)` where `init=dict(I=1)` suffices. Good, and `ss.diff(import, hand)` says `the same` except two extra results.
- odin `seir_array` (60 lines): `split=[0.0196078431372549, 0.07262164124909222, ...]`, `stratify(age=[0.0, 1.0, 5.0, 15.0, 30.0], ..., kind='age')` (redundant `kind`), `Births = ss.par(ss.perday(547.945205479452))` where the source has `b * N`, transitions named `S_to_out`, `in_to_S`, `E_to_out` where the hand version has `births`/`deaths`. Readable but the numbers are noise.
- flepiMoP `seir_2pop` (36 lines): good, except `start='2020.02.01'` (the spec writes dates with dots) and `Ro = ss.par(ss.perday(2.5))` (Bugs 1).
- Atomica `sir` (61 lines): `init = dict(inf=285.71429, rec=14.285710000000714)`, `total_pop=1000.0000000000005`, `recrate = ss.par(5.0, doc='Average duration of infections (years)')` with `rate=1 / recrate` (the importer knows the format is `duration` at `atomica.py:260` and could write `ss.years(5)` and `dur=`), and a force of infection containing `(contacts - contacts)` because `FloorOfName` (`atomica.py:287-301`) rewrote `floor(contacts)` as `contacts` inside `floor(contacts) - contacts` and left the dead term.
- SimInf `events_SIR` (35 lines, 56 KB): line 23 is 8,032 characters (`init=dict(I=ss.by(node=[1.0, 0.0, 0.0, ... ×1600`), line 26 is 48,463 characters (`stratify(node=['1', ..., '1600'], split=[...])`), and the one thing that makes the model a model, the event register, is a comment: `# the movement between places of "node": give it with mobility=ss.Mobility(...)`.

**Why it matters.** `to_code()` is the deliverable of every import; the brief's own test is "does it look like what a Starsim user would write by hand". For a simple model the answer is nearly yes; for anything with data it is no.

**Proposal.** (a) When an array (a split, an `ss.by`, a register, a contact matrix) is longer than about ten numbers, write it to a sidecar (`model_data.npz` or CSV next to the `.py`) and reference it, or reference the source file (`ss.Mobility(data='events.csv.gz')`). (b) Round to the source's precision: carry the source text of numbers in `Source.add_par`/`add_state` or round splits to 6 significant figures (they are shares). (c) Name inflows and outflows `birth_<state>`/`death_<state>` instead of `in_to_S`/`S_to_out` (`core.py:230`). (d) Drop the `kind='age'` repetition when the dimension is called `age`. (e) Let `Source.add_par` take a typed value from importers that know the type (Atomica durations, EpiModel already does this with `ss.probperday`). **Cost.** None breaking; `to_code` round-trip tests (`test_interop.py:test_to_code`) must still pass with a sidecar, so the sidecar has to be read by the generated code. **Confidence: high.**

### 3. Rule 2 misses the commonest way models write the total

**Issue.** `core.Rates.is_total` (`core.py:188-193`) accepts the `__total__` marker (a name the importer put in `Source.totals`) or the literal sum of every people state. odin's `seir_array_odin.R` has `N <- 1e7` and `new_inf[] <- beta * S[i] * I_tot / N`. The odin importer does not put `N` in `totals` (it is an assignment, so it becomes a parameter), and the import says:

```
S -> E   new_inf : S taken out of the flow; mass action with the number in I (not divided by the total: density-dependent, kept as counts) => beta * I.n / N
```

`describe()` then shows `new_inf  S -> E  spontaneous  rate = 1 × I.n / 1e+07`, so the transition is not contact-mediated: no network, `rel_sus`, `rel_trans`, or `I infectious` guess applies, and the hand vignette (`S >> E @ ss.perday(1) * I`) is a different object. SBML models (where `N` is routinely a parameter) and hand-written odin files will hit this constantly. The message is also wrong on its face ("not divided by the total" for a rate divided by `N`).

**Proposal.** In `build()`/`Rates`, treat a parameter whose value equals the sum of initial people (within rounding) as a total and say so (`[guessed: N = 1e7 is the population]`); and when a state-bearing factor is divided by a parameter that is not the total, say "divided by the parameter N, not by the population" rather than "not divided by the total". **Cost.** A guess that is marked; none breaking. **Confidence: high.**

### 4. Lossy export: Mermaid earns its place, SBML and odin do not

**Issue.** Three exporters. Mermaid gives something nothing else does (a diagram with rates). SBML and odin write "the compartmental form" and list what is lost; the lists are complete and honest. But exported from a stratified SIR with a contact matrix (`scratchpad/interop/export.py`), the odin file re-imported gives `n_I` peak 175 against the original's 9,387 (max difference 9,211), because `changed: "sir" is stratified (...), written as one group` and `not exported: network "contacts" (contacts written as homogeneous mixing)` together mean a different epidemic. The exact round trip works only for a model with no stratification, network, outcome, intervention, modifier, or function, which is also the model nobody needs to export. `deferred.md` already records "Export beyond Mermaid, SBML, and odin: won't do" and "SBML and odin are kept only where they are little work, since their importers check the round trip": that is a reason to keep them in tests, not in the API. Who would use them: someone wanting to run a Starsim ODE in a stiff solver or in COPASI (SBML) or in odin's C backend. For those users `sir.equations()` already prints the ODE; the SBML writer is 90 lines of MathML plumbing (`export.py:206-283`).

**Proposal.** Keep `ss.save(filename, sim, format='mermaid')` (or fold it into `sir.diagram('mermaid')`, Simpler codebase 3) and remove `sbml` and `odin` exports plus `W1401 LossyExport` from the public surface. If Cliff wants SBML as a conformance artifact, keep `to_sbml` in the tests folder. **Cost.** Breaking for two `format=` values; the spec's "Export" section shrinks to Mermaid and the model spec; one error code goes. **Confidence: medium** (the honest lossy list is well made; this is a judgment that no one will use the result).

### 5. Format-specific keywords are hidden behind `ss.load(..., **kwargs)`

**Issue.** `ss.load(source, format, **kwargs)` passes kwargs to the importer; each importer has its own: `load_siminf(u0=, events=)`, `load_atomica(databook=, progbook=, start=, stop=, dt=0.25)`, `load_camdl(pars=, scenario=, solver=, camdlc=)`, `load_epimodel(run=, unit=, rscript=)`, `load_emulsion(machines=)`, `load_odin(unit=, stop=, pars=)`, `load_sbml(unit=, stop=, solver=)`, `load_flepimop(scenario=)`, `load_epydemix(start=, stop=)`. Some kwargs go to the importer, others fall through to `ss.Sim()` (`siminf.py:508` picks a fixed list), so `ss.load('x.R', format='siminf', foo=1)` becomes a SimInf model argument while `ss.load('x.R', format='odin', foo=1)` becomes an `ss.Sim` argument. Nothing in the spec or the vocabulary lists any of these.

**Proposal.** Document each `load_<format>` signature as the public entry (`ss.interop.load_siminf(...)`), and make `ss.load`'s docstring point at them; or, if the importers move out (Finding 1), this resolves itself. **Cost.** None. **Confidence: medium.**

### 6. The import log is printed, kept, and written into the code docstring; `describe()` already marks the guesses

**Issue.** The same "how each rate was read" lines appear (a) printed by `build()` when verbose, (b) in `sim.imported.choices`, (c) in the `to_code()` docstring, and (d) as `[guessed]` lines in `describe()` (`transitions.py:1572-1573`). The docstring of the generated file is a third of its length for EMULSION (12 of 37 lines) and odin (20 of 60). `sim.imported` is also lost by `sim.new()` (verified: `hasattr(sim.new(solver='ode'), 'imported')` is `False`), so a user who changes the solver and then calls `to_code()` gets a file with no provenance.

**Proposal.** Keep one source of truth: the per-transition `guessed['import']` on the model, which `describe()` shows; have `to_code()` write the provenance as one header line (`# Imported from EMULSION (compart_SIR.yaml); run sim.describe() for how each rate was read`) and keep the `W1601` list at the end, which is the part the user acts on. Carry `imported` through `sim.new()`. **Cost.** None breaking. **Confidence: medium.**

## Simpler codebase

### 1. What is shared and what is duplicated

**Shared well.** `core.py` is the right shape: `Source` is a neutral description (`core.py:18-99`), `Rates.classify` implements rules 1-5 of the spec in 50 lines (`core.py:255-305`) and the test `test_interop.py:test_rates` pins each rule, `disease()`/`build()` make the sim, and every importer except SimInf ends with `return build(src, ...)`. The rate classification is tested once and used eleven times. This is the part to keep.

**Duplicated.**

- `siminf.py:706-727` reimplements `core.build()` (its own verbose print, its own `W1601` warning text, its own `sim.imported`) because it needs to set `init` by node after the disease is built. `core.build()` could take a post-build hook or `Source` could carry `ss.by` inits (which flepiMoP and epydemix already do through `add_state(init=ss.by(...))`, `flepimop.py:206`, `epydemix.py:164`), so the duplicate can go.
- Seven hand-rolled AST visitors (`def visit(node)` with `ast.iter_fields` loops) across `core.py` (four: `parse`, `counts_of`, `counts_in_calls`, `bare_of`), `emulsion.py:emulsion_expr`, `odin.py:strip_index`, `siminf.py:c_expr`, plus `ast.NodeTransformer` subclasses in `export.py:rename`, `sbml.py:mathml`, `atomica.py:FloorOfName`. One `transform(node, fn)` helper in `core.py` would replace the six copies of the eight-line field loop.
- `{n.id for n in ast.walk(...) if isinstance(n, ast.Name)}` appears nine times in five files; `core.names()` exists (`core.py:118`) and is used in one of them.
- `getattr(ss, f'{unit}s')(x)` appears eight times (`emulsion.py`, `sbml.py`, `odin.py`, `camdl.py` twice, `epimodel.py`, `flepimop.py`); `ss.dur(x, unit=...)` exists (`time.py:1055`).
- The unit tables `units = dict(days='day', day='day', ...)` are identical in `camdl.py:35` and `emulsion.py:20`; `epimodel.py:25-26` has its own `probs`/`rates` tables.
- `flepimop.py:24` imports `number` from `emulsion.py`: an importer depending on another importer for a generic "evaluate an expression of numbers" helper that belongs in `core.py`.

**Proposal.** Move `number()`, a `transform()` helper, `names()` use, the unit table, and a `dur_in(x, unit)` helper into `core.py`; delete `siminf.build`. About 150 lines saved and one fewer place for `W1601` text to drift. **Confidence: high.**

### 2. Half of two importers is a parser for another language

`siminf.py:36-343` (about 310 lines, 43% of the file) is a tokenizer, recursive-descent parser, and evaluator for a subset of R (`c()`, `seq()`, `rep()`, `data.frame()`, `read.csv()`, `$`, indexing, assignment to `u0$I[1]`), so that an R script that calls `mparse()` can be read without R. `odin.py:33-170` is a smaller DSL reader. Neither is wrong, and both are tested, but they are the parts most likely to break on a real-world script (the SimInf one already punts on `function` bodies, `siminf.py:129-137`, and anything it cannot evaluate becomes `Unknown` and a `skip`). The alternative design, which EpiModel already uses, is a 40-line `dump_siminf.R` that runs in R and writes the `mparse()` arguments as JSON; `load_siminf` already accepts exactly that dict (`siminf.py:499-501`). **Proposal.** Replace the R parser with a dump script; keep the dict path. Saves about 300 lines of the hardest-to-maintain code in the package. **Cost.** SimInf import then needs R, like EpiModel does; given the user has a SimInf model they have R. **Confidence: medium.**

### 3. Two Mermaid writers

`Graph.diagram(format='mermaid')` (`transitions.py:1623-1639`, exposed as `ss.Disease.diagram`, `diseases.py:180`) writes `flowchart LR` with edge labels that are transition names only. `interop.export.to_mermaid` (`export.py:143-186`) writes the same chart with rates on the edges, a title giving the stratifications, people/count shapes, and a subgraph per disease. The spec says "The diagram is produced by `sir.diagram()`" (`16-interop.md:98`) and separately lists `ss.save('model.md', sim, format='mermaid')`. **Proposal.** One writer: `to_mermaid` moves next to `Graph.diagram` and `diagram('mermaid')` calls it; `ss.save(..., format='mermaid')` is a two-line wrapper or goes. **Confidence: high.**

### 4. `export.flatten()` is a second "compartmental form" next to `Graph.equations()`

`export.py:70-141` walks `graph.flow_terms(engine)`, renames stages, sums stratifications, rewrites typed values to numbers in the unit (`rename`, `export.py:44-67`), and compiles the list of losses. `Graph.equations()` (`diseases.py:178`) prints the same ODE as text. If SBML and odin exports go (Simpler for the user 4), `flatten` and `rename` (about 100 lines) go with them; if they stay, `equations()` should be built on `flatten()` so there is one derivation. **Confidence: medium.**

### 5. `__init__.sniff()` grows with every format

`sniff` (`__init__.py:42-92`) is 50 lines of regexes and extension checks that know the top-level keys of EMULSION and flepiMoP YAML, which R functions mean EpiModel versus odin versus SimInf, and the `"format"` field of five dump files. Each new importer adds a branch; a wrong guess silently runs the wrong importer (`.R` with both `deriv(` and `mparse(` falls to SimInf). **Proposal.** Each importer declares `sniff(path)` for itself (a `formats` registry of `(loader, sniffer)` pairs), so the knowledge lives with the format, which is also what a companion package would register. **Confidence: medium.**

## Spec and implementation

### 1. Surface that exists but is not specified

`16-interop.md` specifies `ss.load(file, format=)`, `ss.load(live_object)`, `ss.save(..., format=)`, `ss.interop.export(sim, format)`, `ss.diff`, `W1601`, and `W1401`. The code also has, with no spec entry: `sim.to_code()`/`ss.to_code()` (`spec.py:540`, `sim.py:1332`), `sim.imported` (with `format`, `file`, `choices`, `incomplete`, `code`, and per-format extras `outputs`, `scenarios`, `modules`, `functions`, `pars`), `ss.interop.Source`/`build`/`sniff`/`formats`/`exporters`, `ss.interop.summer2.outputs(sim)`, the per-format keywords of Finding 5, `Source.declared`, and `tr.guessed['import']`. `17-vocabulary.md:242` lists `ss.export` as removed ("the script is already the thing to share"), which is exactly the argument for `to_code()`, yet `to_code()` is not in the vocabulary. `implementation.md:242` describes `to_code` as built. **Proposal.** Add `to_code`, `sim.imported`, and `Source`/`build` to 16 and 17 (they are the user-facing part), and make the rest private.

### 2. Spec and code disagree beyond the listed departures

- `16-interop.md` rule 3 says density-dependent rates get `dependence='density'` and an `n_ref`; the code keeps counts (`beta * I.n`). Listed in `implementation.md:285` as a departure, and the right call: counts are exact and need no new concept. The spec should be changed to match.
- `16-interop.md:60-73` says the two importers "worth doing first" are Starsim v3 and EMULSION, calls epipack "nearly complete", and lists EMOD, EpiHiper, MetaCast, epymorph. There is no v3 or epipack importer (v3 is a skill guide, `skills/import-model/references/starsim3.md`), and camdl, MEmilio, Atomica, EpiModel, epydemix, flepiMoP, SimInf, odin, SBML, summer2 were built. The table should describe what exists.
- `16-interop.md:14-20` draws "a data-entry form" as an exporter; `16:98` defers it. Drop it from the diagram.
- `18-conformance.md` §2 says "For every other export format, the printed list of what was left out matches what is actually missing from the file." `tests/test_interop_export.py:70-85` checks that `W1401` is raised and that the round trip agrees for an unstratified model; it does not compare the list against the file. Either write that test or soften the claim.
- The spec's importer rule "Mark what it guessed" is implemented only for rules 2, 3, and 5 (`core.py:258`); a guessed name (`built from its endpoints`) is marked but a guessed unit (odin, SBML: "the numbers are taken to be in `unit`") is not.
- `19-open-questions.md` has no interop row, so nothing to settle there; the implementation has effectively answered one unasked question: "is the import a sim or a disease?" The spec's examples return `sir` (a disease: `sir = ss.load('model.xml', format='sbml')`, `16:26-30`); the code returns an `ss.Sim` (`__init__.py:94-107`) because the source carries run settings (`stop`, `dt`, `solver`, `total_pop`, networks). The code is right; the spec examples should say `sim = ss.load(...)`.

### 3. The "gaps found while writing the importers" list, triaged

From `deferred.md:20-33`. Worth filling, because a hand-written model wants them too:

- **A transition's current flow usable in a rate** (MEmilio's test-and-trace load): incidence-dependent rates are common (reporting, hospital capacity). `ss.outcome` already derives from flows; letting a rate refer to `symptoms.flow` closes it.
- **A one-off move between states at given times** (camdl `transfer`, flepiMoP seeding, SimInf events): three importers hit it and worked around it three different ways (`ss.Mobility(data=...)` for SimInf, a skip for camdl, a skip for flepiMoP). One `ss.transfer(S >> V, n=1000, at='2020-03-01')` would serve all three and interventions in general.
- **`substeps=1` needed to match a discrete-time source**: that the binomial solver substeps the whole model for one Erlang stage is a solver behavior users will trip on without importing anything.
- **A contact matrix is not in the model spec** (written as the mixing pools it makes): this breaks `ss.load(spec.json)` and `to_code()` for any stratified model with contacts, imported or not; importers paper over it with `Source.code['networks']` (`summer2.py:243`, `epydemix.py:246`, `memilio.py:117`). A spec entry for `ss.contacts(matrix)` is a core fix.

Not worth filling (framework-specific, or correctly a "not translated"): `PartnerNet` `acts=` and other ERGM terms (EpiModel), EpiModel's recover-in-the-step-infected ordering, LASER's gravity cap, deaths from a life table as a declared form (data, not language), MEmilio's per-group weighted count and per-infectious-state `rel_trans` (write the rate with `.n` per state), the `< 1` initial value read as a fraction (a documented convention), priors and likelihoods in the spec (belongs to 13, not 16), sub-step evaluation of time-varying values (use a smaller `dt`).

### 4. What interop teaches about the core: features that exist for one importer

Grepping each feature across `starsim/*.py` (outside `interop/`), `v4_design/examples`, `tests`, and the vignettes:

- `tr.guessed['import']` and the `imported as ...` line in `describe()` (`transitions.py:1572-1573`): used only by `core.disease()`. Fine as a hook, but it is the only `guessed` key that is prose rather than a value.
- `Source.declared` (`core.py:46`) and `ss.State(people=True)` as an explicit declaration: set by SimInf, MEmilio, camdl because a flattened stratified model has states no transition connects. Nobody writes `people=True` by hand (0 uses outside interop and its tests).
- `stratify(kind='space')` on a disease (not the sim): 3 importers, 0 core uses, 7 in examples/vignettes, so it has users beyond import.
- `ss.Mobility(kind='visit')`, `ss.timeseries(mode='cosine')`, `ss.timeseries(..., period=ss.years(1))`, `integrator='euler'`, `substeps=1`: each was reached for by exactly one importer (flepiMoP, MEmilio, SimInf's `phi` seasons, Atomica, epydemix/Atomica) and each appears in the matching vignette and nowhere in `starsim/*.py` outside its definition. They are all legitimate knobs, but the pattern is clear: the importer vignettes are what justify them, so if a vignette goes, so should its knob, and the spec should say which vignette each is for.
- `ss.State(people=False)` with flows against `ss.null` (SimInf's `phi`, SBML rate rules, camdl ODE compartments): used by three importers and three examples; this one earned its place.

## Bugs

### 1. A bare dimensionless parameter in a rate is typed as a rate per unit (core; every importer depends on the rule)

```python
S, E, I, R = ss.states(4)
R0 = ss.par(2.5, name='R0'); gamma = ss.par(0.2, name='gamma')
seir = ss.Disease(infection=S >> E @ R0*gamma*I, onset=E >> I @ 0.25, recovery=I >> R @ gamma, init=dict(E=5), name='seir')
ss.Sim(seir, total_pop=10000, unit=unit, start='2020-02-01', stop='2020-08-31', solver='ode').run()
```

Output (`scratchpad/interop/bare.py`):

```
typed=False: peak n_I day=1277.9 week=82.7
   R0 in spec: [{'type': 'per', 'value': 2.5, 'unit': 'day'}]
typed=True: peak n_I day=1277.9 week=1277.9
```

Both factors of `R0*gamma` are typed `per day`, so at `unit='week'` the product is scaled by 49 instead of 7 (and `ss.perday(2.5)*ss.perday(0.2)` itself raises "Cannot multiply perday by perday"). The flepiMoP import shows the same in `to_spec()` (`Ro: {'type': 'per', 'value': 2.5, 'unit': 'day'}`) and `to_code()` writes `Ro = ss.par(ss.perday(2.5))`; re-running the import with `unit='week'` gives peak 81 against 1,187. Every importer writes `Ro`, `transpercontact`, `contacts`, `RiskOfInfectionFromSymptomatic` as bare numbers and relies on the import unit, so imports are correct only at that unit. The rule should type the product of a rate's constant factors, not each factor, or `Source.add_par` should let importers mark a parameter dimensionless. Outside my area in origin (03/04), in my area in effect.

### 2. `sim.imported` does not survive `sim.new()`

`ss.load(...).new(solver='ode').to_code()` writes no provenance and no `W1601` list; `sc.dcp` keeps it. `sim.new()` should copy `imported` (and anything else `ss.load` hangs on the sim).

### 3. Atomica `floor()` rewrite leaves a dead term

`atomica.py:287-301` rewrites `floor(contacts)` as `contacts` and the imported force of infection reads `... * (1 - inf.n / (...) * transpercontact * (contacts - contacts))) * (1 - susdeath)`; harmless numerically, wrong as code. Rewrite `floor(x) - x` to `0` or leave `floor()` in place (the parser allows `floor`).

### 4. Dates written with dots

`to_code()` of the flepiMoP import gives `start='2020.02.01'` for a YAML `start_date: 2020-02-01`; the spec stores `ss.date` as `str(date)`. It round-trips, but no user writes dates that way.

## Keep

- **`core.py`'s shape**: a neutral `Source`, one `Rates.classify` for the five spec rules, one `build()`. New importers are mapping code only; `test_rates` pins the rules. The `import-model` skill's advice to fill a `Source` for an unsupported framework is the right escape hatch.
- **Parsing, never executing, the source's expressions** (`ss.spec.Parser`, `core.py:216`): `ss.load()` of a YAML or R file runs no code from it. Keep this even for the dump JSONs.
- **"Keep going and list it"**: every importer skips with a reason that says what the Starsim form would be (`emulsion.py:174-178`, `sbml.py:219`, `camdl.py:371`). The `W1601` text is the best onboarding prose in the package; it tells the user the idiom to reach for.
- **`ss.diff` in the model's own terms** (`spec.py:674`): `transition 'I_to_R' (called 'recovery'): the same` is exactly how to check a translation.
- **`describe()` marking imports as guesses** on the transition itself, not in a side log.
- **The dump-script pattern** (`dump_epimodel.R`, `dump_memilio.py`, `dump_summer2.py`): small, dependency-free, run in the other environment, writing a JSON the importer reads. This is the one design that scales to frameworks whose models are code, and `siminf` should adopt it (Simpler codebase 2).
- **The exact round trip through odin/SBML for simple models** (diff 0.0) and the against-the-original checks in the tests, as a conformance artifact, even if the exporters leave the public API.

## Outside my area

- The bare-number unit rule (Bugs 1) is a `03-states-and-transitions.md`/`04-parameters.md` question: what is the unit of a product of bare numbers in a rate? The answer "the product is per unit, the factors are dimensionless" is what every imported model assumes.
- `ss.contacts(matrix)` not being in the model spec (`deferred.md:26`) breaks `to_code()` and `ss.load(spec.json)` for any stratified model with a contact matrix, imported or not (`14-model-spec.md`).
- `Graph.diagram()` and `Graph.equations()` (`transitions.py:1623`, `diseases.py:178`) and `export.flatten()` are three readers of the same engine; whoever reviews 12/14 should decide which one is canonical.
- `tests/temp/` has 167 leftover `interop_*.R`/`.xml` files from the odin/SBML tests (gitignored, but the tests should clean up or use `tmp_path`, which `test_interop_export.py` already does).
