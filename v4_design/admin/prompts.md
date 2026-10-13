# Prompts

This is a record of the prompts used to create and revise the contents of this repository, with a summary of what each one produced. All of the work was done with Claude Code: the original contents on 25 August 2026, in three sessions, and the revisions that followed human review from 12 October 2026. The prompts are reproduced exactly as typed, including typos. Times are UTC. Housekeeping prompts (asking where the review had got to, setting up this file and its skill) are not recorded.

The root `README.md` was written by hand before the first session, so it has no prompt. The full session transcripts are archived in [`transcripts/`](transcripts/); the file for each session is named below.

## Summary

| # | Time | Prompt | Result | Commit |
|---|------|--------|--------|--------|
| 1 | 14:21 | Make the five folders | `approaches/`, `best-practices/`, `design/`, `examples/`, `vignettes/`, each with a README | `de16ec4` |
| 2 | 14:34 | Add a webapp folder | `webapp/README.md` | `abc7509` |
| 3 | 15:23 | Start on `approaches` | Framework selection, plus sample reviews of EpiModel and Starsim | `5997bdb` |
| 4 | 15:47 | Apply feedback and finish the reviews | 22 framework reviews, `notes.md`, and an index (3,542 lines) | `5997bdb` |
| 5 | 17:18 | Fill out `best-practices` | 16 documents plus an index (2,847 lines) | `c24ff3d` |
| 6 | 19:40 | Build out `design` | 20 specification documents plus an index (3,658 lines) | `f6ff541` |
| 7 | 12 Oct, 07:29 | Recapitulate the prompts and apply the first `CK:` comments | `best-practices/` and `design/` revised throughout; `questions-oct11.md` with 63 guesses to confirm | `e00c694`, `0b934ce`, `3bca852` |
| 8 | 12 Oct, 16:16 | Review the work to date | A report on `best-practices`, six decisions to make, and five `design` files updated | `a4e06c5` |
| 9 | 12 Oct, 18:51 | Apply the review | The listed fixes to `best-practices/` and the decisions carried into `design/` | `a4e06c5` |
| 10 | 13 Oct, 03:19 | Address the remaining `CK:` comments and answers, then review `design` | 39 files revised: all of `best-practices/` and `design/` | not yet committed |

## Session 1: folder structure

Transcript: `a6f792a5-3d8e-4dbd-98a4-5ec539bc48d6.jsonl`

### Prompt 1 (14:21)

```text
make each of the five folders mentioned in the readme:

approaches
best
design
examples
vignettes

and put a readme in each of them
```

**Response.** Created the five folders, each with a README stating the folder's purpose, what belongs in it, and how it connects to the others:

- `approaches/`: a survey of existing frameworks and how each represents core modeling concepts.
- `best-practices/`: recommendations per paradigm and cross-cutting concern, derived from the survey.
- `design/`: the normative specification of the language itself.
- `examples/`: the corpus of original models in their source frameworks, with provenance and reference outputs.
- `vignettes/`: translations of those examples into the lingua franca, with verification and feedback to `design`.

The folder listed as `best` in the prompt (and `best-practice` in the README) was created as `best-practices`, and the root README's structure list was updated to match.

### Prompt 2 (14:34)

```text
also add a webapp folder, which will be a tool showing side-by-side comparisons of `examples` models with `vignettes` translations, with markup/highlighting like git diff
```

**Response.** Added `webapp/README.md` and a `webapp` line in the root README's structure list. The README covers:

- **Intended behavior**: a model picker, side-by-side original and translation panes with syntax highlighting for both languages, diff-style markup for what corresponds, what was added, what was dropped, and what is inexpressible, aligned scrolling, inline vignette annotations, and a results comparison where reference outputs exist.
- **A design note**: because the two sides are in different languages, a textual diff will not align meaningfully. The correspondence has to come from each vignette declaring which of its blocks map to which regions of the original, and that format is worth settling early because it changes how vignettes are written.
- **Status**: not implemented, stack undecided. No app code was scaffolded.

## Session 2: `approaches`

Transcript: `96d4b52a-9a4b-4e2c-811a-039a6fa64849.jsonl`

### Prompt 3 (15:23)

```text
i'd like you to start working on the `approaches` folder. you can use this for the source:

/home/cliffk/idm/disease_modeling_landscape

(which maps to the link in the folder readme)

you can use your judgement about which to include and exclude, but you definitely don't need to include all 100+ tools! focus primarily on disease modeling *frameworks*.

also be sure to include camdl: /home/cliffk/idm/camdl

what i'd like you to do at this stage is:
1. go through the list of tools and decide which ones to include
2. write a sample entries for  epimodel and starsim

i'll then review, give feedback, and you can proceed with the others.
```

**Response.** Produced three files: `approaches/README.md`, `approaches/epimodel.md`, and `approaches/starsim.md`.

- **Selection.** The filter applied to the 137 tools in the landscape database was "does the user specify a model in it?" That excluded four groups, each named in the README so the decisions are reviewable: inference and estimation tools, genomics and phylodynamics, data and surveillance utilities, and single-disease applications. The proposal was 18 full reviews and 11 brief notes.
- **Additions beyond the database.** Mesa, ModelingToolkit.jl/Catalyst.jl, SBML, and ASKEM AMR/MIRA were added because they bear on model representation even though the database excludes them. LASER was flagged as a decision rather than added silently.
- **Sample entries.** Both follow a fixed template: frontmatter table, what it is for, how a model is specified, core abstractions, time, stochasticity, populations and mixing, interventions, results, a minimal model, a worked example, strengths, limitations, and implications. Both were written from source (EpiModel 2.6.2 and Starsim 3.6.0) rather than from recall.
- **Main observations.** EpiModel shows why time needs types, since most of its sharp edges follow from `numeric` being its only type. Starsim and camdl converged independently on the same dimensional-typing primitive. The gap both frameworks share is that transitions are code, not data. The Starsim limitations section was written deliberately hard, since it is the design predecessor.

### Prompt 4 (15:47)

```text
ok, i updated the readme. make the changes ('[CK:]') i suggested, then proceed with implementing the remaining reviews/summaries.
```

**Response.** Applied the `[CK:]` comments from the README and completed the folder: 22 full reviews, a brief-notes file (`approaches/notes.md`), and an updated index, all written from source.

- **Changes applied.** NDlib was dropped from the full reviews and recorded in `notes.md` with the reason. Vivarium, epidemics, and LASER were promoted to full reviews. EpiHiper was added. FRED stayed excluded because it is distributed under a University of Pittsburgh EULA, not an open-source licence.
- **Full reviews.** Starsim, EpiModel, camdl, odin, summer2, MEmilio, EMULSION, epymorph, MetaCast, Epydemix, epipack, EMOD, EpiHiper, epiworld, individual, SimInf, Atomica, EoN, flepiMoP, Vivarium, LASER, and epidemics.
- **Cross-cutting findings**, added as a section of the README:
  - EMULSION already switches one model between compartmental, individual-based, and hybrid execution by changing one word.
  - Catalyst.jl declares the paradigm per reaction, not per model, which is what a hybrid model needs.
  - Structural rigour and intervention expressiveness have never appeared together in one readable framework.
  - Dimensional typing has five independent arrivals (Starsim, camdl, Atomica, EMULSION, Catalyst.jl).
  - Declarative eligibility has working versions in EMOD and EpiHiper.
  - Dwell-time distribution is the best early prototype target, since one declaration should compile to Erlang sub-compartments, a scheduled event, or a memory kernel.
- **Caveat raised.** The review is heavy on what to take and light on cost. Several recommendations are individually cheap and collectively a large language, which `best-practices` has to adjudicate.

## Session 3: `best-practices` and `design`

Transcript: `3315bd22-8804-4d57-82f0-58b2e3d2ee45.jsonl`

### Prompt 5 (17:18)

```text
based on the the findings in ./approaches, fill out the ./best-practices folder. a couple notes:
1. be skeptical, and aim for simplicity. the target users are not programmers. they are epidemiologists. most things should be stupdendously, gobsmackingly simple. throwing an error when a default could be guessed 99% of the time will discourage users. a feature that seems good in isolation could add to package bloat and be more trouble than it's worth. the library code can be complex if need be, but the code the user writes should be simple, beautiful, and elegant, and look as much as possible as the pseudocode/algorithms/equations they would be thinking about in their heads. keep boilerplate as minimal as possible.
2. the target language should be python. use all the syntatic sugar you want, but it should be python -- not ocaml, not a DSL, not yaml. multiple representations are ok, eg a dict for simple inputs or a class for more complex ones. don't throw away a simple representation to support one feature that needs a more complex representation; support both.
3. the base for the lingua franca will be starsim (yes i'm biased). if a package does something better than starsim, be honest -- but in a tiebreaker, and starsim already handles it, use starsim's version.
4. when making a proposal for a feature, cite how it works in the original package's implementation, and then show (either after or next to it) how it would be implemented in the lingua franca.
```

**Response.** Wrote 16 new documents and replaced the index. Every document follows the same shape: recommendation, evidence from the landscape, the original framework's code next to the lingua franca's, trade-offs, rejected alternatives, and open questions. Every proposal is spelled in Python as `import starsim as ss`.

- **Doctrine**: `principles.md`. The audience is the binding constraint and total vocabulary size is a design cost. The concrete rules are to guess the default and print it, to have one language at several verbosities, and to price every escape hatch.
- **Cross-cutting**: `model-structure.md` (the central document), `time-and-units.md`, `parameters-and-distributions.md`, `stratification.md`, `population-and-mixing.md`, `composition-and-effects.md`, `interventions.md`, `results-and-observation.md`, `stochasticity-and-reproducibility.md`, and `calibration.md`.
- **By paradigm**: `compartmental.md`, `stochastic-compartmental.md`, `metapopulation.md`, `agent-based.md`, and `paradigm-conversion.md`.

The main positions taken:

- **SIR is two lines.** `ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)))`, with states inferred and results named from the keywords.
- **Route unification.** β is per contact and belongs to the pathogen, and the route supplies the contacts. Mass-action, contact-matrix, and edge-based transmission then become one declaration. The cost is that only the product β·c is identifiable from data.
- **Eligibility as Python operators.** `(ss.age > 65) & ~sir.vaccinated` returns an expression object, replacing EMOD's targeting logic and EpiHiper's JSON.
- **`ss.Effect` with declared combiners** replaces Starsim's last-write-wins connectors.
- **Disagreements with Starsim**, flagged as such: connectors, opaque eligibility, the approximate compartmental limit, the two-language split, and common random numbers keyed on slot only. Typed time, per-module timesteps, per-distribution random streams, `Module`, `Route`, `Loop`, and declared results were recommended essentially unchanged.
- **Rejected as bloat**: a DSL, YAML/JSON/Excel authoring, mandatory ontology identifiers, symbolic analysis as a headline feature, summer2's `Overwrite`, additive composition, epiworld's tools as a separate object, Starsim's intervention class lattice, and budget optimization in the first version.

Two weak points were flagged: whether the β-per-contact factoring survives environmental and vector-borne transmission, and whether EoN's closure ladder should ship in the first version at all.

### Prompt 6 (19:40)

This followed a `/compact` in the same session.

```text
you're doing a great job! take everything you compiled in `./best-practices` and use it to build out `./design`
```

**Response.** Wrote 20 numbered specification documents and replaced the index. Where `best-practices` argues, `design` decides: every document is normative (MUST/SHOULD/MAY), opens by citing the best practice that justifies it, and states exact signatures, defaults, semantics, and error codes.

- **The language**: `00-overview.md`, `01-objects.md`, `02-types.md`, `03-states-and-transitions.md`, `04-parameters.md`, `05-dimensions.md`, `06-routes.md`, `07-effects.md`, `08-interventions.md`, and `09-results.md`.
- **Semantics**: `10-execution.md` (eight named phases, queued updates, competing hazards) and `11-random.md` (random streams, common random numbers, five-level run identity).
- **Execution and inference**: `12-backends.md` (seven methods, the capability matrix, three conversion classes) and `13-inference.md`.
- **Machinery**: `14-ir.md` (the JSON intermediate representation, with normalization and hashing rules), `15-errors.md` (87 error codes with a required message shape), and `16-interop.md` (import fidelity per framework).
- **Discipline**: `17-vocabulary.md` (all 134 public names, counted), `18-conformance.md` (twelve required test suites), and `19-open-questions.md`.

Writing the specification forced several decisions that the best practices had left loose:

- The force of infection became a formal equation, with density dependence defined against an explicit reference population.
- `n` (population size) and `n_agents` (resolution) were separated.
- The ambiguity of `p=` between timing and branch split was resolved by destination count.
- `sim.results.infection[age=0]` is not valid Python, so it became `.sel(age=0)`.
- Competing transitions must be a single multinomial draw, or declaration order leaks into results.
- Effect combiners got a fixed resolution order: replace, scale, shift, clamp.
- Starsim's `per` type was renamed `rate`, flagged as the only rename.

Two items were raised for attention. The largest unresolved question (B1 in `19-open-questions.md`) is that vector-borne transmission does not fit the force-of-infection form without a second host population. And `ss.Effect` carries a great deal of the design (interventions, connectors, waning, triggers, seasonality, and product efficacy all compile to it), which is an argument for refusing extensions to it.

## Session 4: first revision after review

Transcript: `0d67c1c6-dc58-4468-94d2-eceb363c910a.jsonl`

By this point `agent-based.md`, `calibration.md`, `compartmental.md`, `composition-and-effects.md`, and `interventions.md` in `best-practices/` carried inline `[CK: ...]` review comments, and `ck-feedback1.md` held two general ones.

### Prompt 7 (12 October 2026, 07:29)

```text
i was working with an earlier version of opus for this project. i'd like you to check the work and look for improvements. specifically:
1. read README.md, prompts.md, notes.md, and all the other core files
2. i'd like you to recapitulate the prompts, independently or semi-independently of the original agent who completed them, and come up with your own ideas
3. based on what you learn, and also integrating my comments (marked with "CK:"), update the .md files in the subfolders
4. you are on a new branch so go ahead and modify the files directly, but don't modify any files outside of the root folder.
5. don't ask me any questions, make your best guess and keep going until you're done. you can collect questions into questions-oct11.md.
6. you will likely run out of tokens partway through ... resume when your token quota refreshes.

thanks!
```

**Response.** Revised `best-practices/` and `design/` throughout, applied the comments, and wrote `questions-oct11.md`: 63 questions it would have asked, each with the guess it went with and what to change if the guess was wrong. `approaches/` got an American-spelling pass, and `starsim.md` and `camdl.md` were fact-checked against the current source.

The main changes from the first draft:

- **One transition class.** `S >> I @ beta * I` is contact-mediated because the rate mentions a state, and `I >> R @ gamma` is spontaneous because it does not. `ss.transmission` and `ss.progression` remain as shorthand, and a bare number after `@` is a rate.
- **`ss.modify` replaces `ss.Effect`**, with any named quantity as a target and plain `step()` code, presented as a normal way to work, for anything more complex.
- **The approximation policy.** `method='auto'` never approximates; asking for a method the model does not fit uses the standard approximation and prints one line saying so.
- **`ss.Compartments`** for vectors and environmental reservoirs, replacing `ss.Continuous`, which the first draft had left as its largest open problem.
- **Starsim's loop stays.** Queued updates, named phases, and derived ordering are gone, as are required `writes`/`requires` declarations.
- **Calibration without a likelihood**, by normalized RMSE, and `ss.compare_models`.
- **A softer voice in `design/`.** "Guess and print" replaces most hard errors; the error catalog went from 87 codes to 48, and `14-ir.md` became `14-model-spec.md`.

The vocabulary did not get smaller: 133 public names against the first draft's 134.

## Session 5: review of the revision

Transcript: `80513970-5406-458d-bed8-0e66586c8987.jsonl`

### Prompt 8 (16:16)

```text
i'd like you to do a review of the work done on this project to date. specifically:
1. read the readmes
2. you can skim or skip ./approaches, but i want you to go into detail on best-practices
3. after you're done with ./best-practices, you can update ./design
4. watch your token usage! make sure you don't run out of tokens partway through with a lot of unwritten content
```

**Response.** Found the revision coherent and `design/` consistent with `questions-oct11.md`. The problems were three `[CK:]` notes in `model-structure.md` that had not been acted on, examples in `best-practices/` that predated the October decisions, and six decisions for a human:

- how to avoid typing `S I R` twice in `ss.states('S I R')`;
- the name `ss.Compartments`, and whether a per-module run method would remove the class;
- the default time unit (days, or Starsim's years);
- whether `method=` belongs on `ss.Sim` only or also on `sim.run()`;
- `new_infections[-7:]` in triggers, which is wrong while results are preallocated;
- candidates to trim from the vocabulary (`ss.Needs`, `ss.export`, `check_reps`, three mobility models).

It listed fixes needed in eight `best-practices` files without applying them, because those files were mid-review, and updated five `design` files: forgiving arrow strings, `ss.State()` in a class body, the capitalization rule for products, `ss.Sim([...])`, and new entries in `19-open-questions.md`.

### Prompt 9 (18:51)

```text
sounds good, pleaes apply these. for the decisions, all sound good. method= should not be sim only.
```

**Response.** Applied the listed fixes to `best-practices/` and carried the decisions into `design/`, with `method=` accepted by both `ss.Sim` and `sim.run()`. The vocabulary trim deferred `ss.Needs`, `ss.export`, `check_reps`, and the mobility models beyond `ss.gravity`.

## Session 6: second revision after review

Transcript: `7a6450dc-1675-40d1-8577-516da018b17d.jsonl`

Before this session the remaining nine `best-practices` files had been reviewed, with `[CK: ...]` comments, and every question in `questions-oct11.md` had been answered inline.

### Prompt 10 (13 October 2026, 03:19)

```text
address my comments ("CK:") in the remaining files in ./best-practices, and review my answers in questions-oct11.md. then, when you're done, review all of ./design against the documents in ./best-practices and update those files as needed.
```

**Response.** Addressed and removed every comment in the nine files, applied the answers, and wrote the result down as a list of 43 decisions. Five parallel agents then brought the other eight `best-practices` files and all of `design/` into line with that list, and the results were cross-checked. 39 files changed, including the status lines in the root `README.md`.

Changes that came directly from the comments and answers:

- **`solver=` and `sim.solvers()`** replace `method=` and `methods()`. The exact stochastic solver is `'gillespie'`, with `'ctmc'` as an alias, and the ODE library is chosen with `integrator=`.
- **Time is in timesteps when nothing has a unit.** Otherwise the unit comes from `unit=`, `dt`, the dates, or the typed values, and is printed.
- **`stratify` works in place by default**, with `inplace=False` for a copy.
- **A sim is one trajectory.** Many runs are a MultiSim (`ss.parallel(sim, n_runs=100)`).
- **`solver='auto'` may use Erlang stages** for a non-exponential dwell time, and says so; `approx=True` silences an approximation note.
- **Smaller ones:** `S, I, R = ss.states(3)`, `ss.null` as the source and sink, `.select()` on results, `ss.data(..., aliases=)`, `requires`/`ss.Requires`, `ss.people('Kenya')`, an opt-in transmission log, and `track_writes` off by default.

Decisions the response made and flagged, all then accepted:

- **No `ss.Compartments`.** A reservoir or a vector population is ordinary states in the same disease, treated as a count because no transition connects it to the states people move through; `ss.State(people=False)` says so outright.
- **`S >> I @ beta` is a constant hazard**, not transmission; the word `transmission` in the short form stands for `* I`.
- **The default goodness of fit is the normalized absolute error**, as in Covasim, in place of normalized RMSE.
- **`ss.outcome` takes a weighted sum of flows**, with `every=` for reporting periods and no `p=`. When testing is modeled, reported cases come from the interventions' own results.
- **People properties are stratified on the sim** (`sim.stratify(age=...)`), and a disease's own dimensions on the disease.
- **Reproducibility was cut back.** Common random numbers are keyed on Starsim's slots only, result caching is deferred, `ss.export` is dropped, and `sim.hash()` is described as a supplement to the git tag.
- **Triggers.** `ss.modify(when=)` is a function with no memory, and `min_duration=` is gone.
- **Gradients for fitting** come from automatic differentiation of ODE models through JAX; symbolic Jacobians stay out.

`design/17-vocabulary.md` now counts 132 public names and `design/15-errors.md` 54 codes.
