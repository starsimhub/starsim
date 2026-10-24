# Docs comparison: tutorials and user guides

How the eleven frameworks in the lingua franca review lay out their tutorials and user guides (API reference left out), what they have in common, where Starsim's current tutorials and user guide fall short, and what to do about it. Written 24 October 2026; the Starsim docs were read as they stand in the working tree of `v4-oct23`, and nothing in `docs/` was changed.

## Summary

- **Recommendation: one core tutorial track, then two short "going further" tracks.** Don't write two parallel tutorial series. Six core tutorials follow one running model, and the solver is a switch inside them, not a fork. They are followed by an agents track (networks, agent demographics, modules written as code, co-infection) and a populations track (stratification in depth, time-varying parameters, places and movement, imported models and gradients), three or four tutorials each. That gives the compartmental audience its own home without duplicating the material both audiences need (results, interventions, scenarios, data, calibration).
- **No reviewed framework has separate deterministic and stochastic (or compartmental and agent) tutorial tracks**, except MEmilio, whose split by solver family is where its docs are weakest. The best docs (the odin & monty book, EMULSION's paradigm triplets, camdl's backends page) keep one model and show what changes when the solver changes. EpiModel shows the risk of parallel tracks: it has three model classes with one API, but only the network class got narrative investment, and the DCM and ICM pages have decayed until no live tutorial teaches them.
- **The user guide needs about ten new pages** for v4 features that are documented only in the architecture or "coming from" pages, or nowhere: solvers and when they disagree, stratification, time-varying parameters, places and movement, data by place, outcomes and observation, reading and checking a model (`describe`, `diagram`, `equations`), the model spec and importers, uncertainty and many runs, and debugging. Most of the existing pages also still teach v3 idioms first.
- **Things to copy:** epydemix's landing-page quickstart with its plot; the odin book's running SIR that turns stochastic, then structured, then fitted; summer2's step-by-step stratification; MEmilio's same template for every solver; EpiModel's gallery template; camdl's runbook, concepts and diagnosing-fits pages, plus its "pitfalls that have actually bitten us"; LASER's notebooks that reproduce analytic results.
- **Things to avoid,** each seen at least twice: stub and TODO pages in the navigation, notebooks whose outputs are stored rather than executed at build, tutorials in a separate repo, tutorials that duplicate the guide page for page, a first model that uses every feature, and a first plot that comes late or never.

## How the other frameworks lay out their docs

| Framework | Top-level layout | First model and first plot | Solvers and model types in the docs | Strongest topics | Notable weaknesses |
|---|---|---|---|---|---|
| summer2 | Install and quickstart, numbered example notebooks (01–11), one reference page, rationale, API | SIR on the install page, plotted at once | ODE only in the docs (a stochastic solver exists but no page covers it) | Stratification taught step by step; age; mixing matrices; flows between strata | Gaps in numbering (02, 05 missing but referred to); unexecuted notebooks; no calibration |
| Atomica | General content, examples, tutorials T1–T7, model library, API | `at.demo('sir')` on the landing page; T1 takes about 2,500 words of Excel before its first plot | Deterministic Euler; one short stochasticity page; an honest "comparison with ODEs" page | Policy workflow (calibrate, scenarios, programs, budget optimization); exercises; model library | Tutorials buried third; a 13,000-word plotting page; Excel-first onboarding |
| epydemix | User guide (7 task pages), Learn (13 notebooks in three tiers), reference | Ten-line SIR with an uncertainty band, image on the landing page, Colab link | Chain-binomial only, said up front; uncertainty bands are the default output | Contact matrices and population data; ABC calibration; short pages with boxed gotchas | Tutorials and guide cover the same topics twice; no explanation layer |
| EpiModel | Four sites: hub, package site (4 articles), a 61-page course book, a 15-example gallery | Network estimation first; the first plot is a network diagnostic; ERGM theory before the first epidemic | DCM, ICM, and network classes with one API, but only networks are taught | Network estimation from data; interventions gallery; gallery template (diagram, parameter table, modules, headline result, next steps) | Fragmented across sites; DCM and ICM orphaned; no calibration to epidemic data |
| odin (odin2, dust2, monty) | The odin & monty book (odin, monty, both together) plus thin package sites | SIR ODE plotted in about four code blocks | One track with branches: chapter 2 is continuous vs discrete time, chapter 3 turns the same SIR stochastic | A running SIR that becomes stochastic, age-structured, then fitted; order-of-events page; debugging and parse-error pages | Four package sites; sketch chapters; the odin 1 site doesn't point to odin2; no interventions |
| SimInf | Home, three vignettes, a long paper (PDF) | SIR on 1,000 nodes plotted in about five lines | Gillespie only | Scheduled events (movement, births, culling) and the order of simultaneous events | Calibration, built-in models, and uncertainty have no narrative pages |
| camdl | A flat folder of Markdown pages, served offline by `camdl docs <topic>` | SIR in the README, then a check, simulate, fit; no plot anywhere | One model, the backend a flag; one page, "When ODEs and SSA disagree" | Inference (runbook, concepts, diagnosing fits); debugging; minimal reproducible examples; dated language changes; cheat sheet with real pitfalls | No navigation; very long spec pages next to user pages; no figures |
| flepiMoP | GitBook: maths, config reference, inference, how to run (HPC), internal pages | A sample config run from the command line; plotting needs knitting R Markdown | One key: `rk4`, `euler`, or `stochastic`; the maths given in ODE and binomial form side by side | Spatial and mobility; time-varying modifiers; a config series that grows one feature at a time | The config series is never walked through in prose; placeholders and "(OLD)" pages in the navigation; more words on AWS than on modeling |
| LASER | Overview, gotchas, get started, tutorials (numbered notebooks), API | A 10-step single-patch SIR | Agents only | Notebooks that reproduce theory (outbreak size, critical community size, periodicity); spatial; a candid optimization guide | About 15 TODOs in the pages; near-empty transmission page; stored outputs, not executed |
| MEmilio | About, C++ interface, Python interface, API; tutorials in a separate repo | ODE-SECIR in the separate tutorials; a quick tour with one figure per tutorial | Separate pages per solver family (ODE, LCT, IDE, SDE, metapopulation, ABM), all with the same template | Same page template for every solver; contact matrices; interventions (dampings); data download | Tutorials outside the docs; Python and C++ double the navigation; LCT, IDE, and ABM tutorials only in C++; stubs |
| EMULSION | Getting started, modeling principles, language (basic, advanced), 20 feature examples | One CLI command runs and plots the quickstart, but the quickstart has three state machines and about 280 lines | One keyword, `compartment`, `IBM`, or `hybrid`; examples ship as triplets that differ by about 15 lines | Transparency (diagrams, `describe PARAM`); the triplets; reference per section of the language | Overwhelming first model; docs on an unreliable server and not in the package |

## What they have in common

1. **A first plot within a page.** summer2, epydemix, SimInf, odin, and EMULSION all plot an epidemic within five lines or a page of installing. The ones that don't (EpiModel, Atomica's T1, flepiMoP, camdl) are worse to start with, and the reviews of each said so.
2. **SIR first, every time,** and usually the same SIR reused (odin, MEmilio, EMULSION, epydemix's guide). The best docs extend one model rather than starting a new one on every page.
3. **Docs follow how a framework is used, not the class hierarchy.** The order is usually: model structure, then time and parameters, then stratification and mixing, then interventions and scenarios, then data and fitting. Nobody organizes tutorials by module type (diseases, networks, interventions), as Starsim's T3–T6 do.
4. **The solver is a switch, and one page explains what it changes.** camdl ("When ODEs and SSA disagree"), Atomica ("Comparison with ODEs"), odin (continuous vs discrete time), flepiMoP (dt advice for each method), EMULSION (choosing compartments, IBM, or hybrid). None has a "which solver should I use" page that covers ODE, stochastic compartments, and agents together: Starsim can be the first.
5. **Stratification is taught on its own, step by step,** not as a corner of another topic: summer2 (06–11), odin (arrays), MEmilio (age groups and contacts), EMULSION (age groups, then explicit age, then structured population).
6. **Explanation is weakest everywhere.** Most frameworks bury the "why" in a paper (SimInf, EpiModel) or leave it out (epydemix). Starsim's architecture section is already ahead of every framework reviewed.
7. **Calibration is either the centerpiece or missing.** odin and camdl teach inference in depth; summer2, SimInf, EpiModel (for epidemic data), and EMULSION barely cover it; MEmilio hands it to other packages.
8. **Rot is the most common defect:** stubs and TODOs (LASER, MEmilio, flepiMoP, odin), stale calls (summer2), stored outputs (LASER, epydemix, summer2), missing pages referred to (summer2), dead links (flepiMoP, EpiModel). Starsim already renders its docs in CI, which prevents most of this; keep it that way.

## Where Starsim's tutorials and user guide fall short

### Tutorials

| Page | What it covers | Gaps and problems |
|---|---|---|
| T1 Getting started | Declared SIR in two spellings; the same model as agents | The agent version switches to v3's `pars` dict with `type='sir'`, a second language on the first page; time silently changes from 100 steps to 50 years [CK: agree we can drop the dict representation. i think it's always 50 steps, no? if not then we should be consistent with that. let's do 50 not 100 for both compartmental and ABM] |
| T2 Building your model | `pars` dict, components, declared disease, solvers, results, time, saving | Starts again from the v3 `pars` dict, so a reader meets three ways to build a sim before the solver section; topics are a grab bag (results, time, and saving in one page) [CK: i like the 'grab bag', it's like getting started again, but in more depth. but agree we can drop the dict] |
| T3 Demographics | Declared birth and death; v3 `ss.Births`/`ss.Deaths`; Nigeria data | Comes before diseases and before stratification; the per-1,000 vs per-person warning text is heavy for a tutorial; the Nigeria example plots with raw Matplotlib [CK: all fair] |
| T4 Diseases | Library parameters, HIV and SIS with modifiers, SIRS and SEIR as classes | Co-infection (HIV with CD4 counts) comes before a reader has written a disease; branching, outcomes, and Erlang stages are absent [CK: fair] |
| T5 Networks | Random, sexual, maternal networks; a custom network stub; mixing pools; contact matrix | Long prose on network internals (`networks.py`, edge lists) for a tutorial; `age_mf.add_pairs` is an empty stub that an exercise asks the reader to modify; the contact matrix, the main compartmental topic, is tucked in at the end [CK: fair] |
| T6 Interventions | Products and deliveries; screening and treatment; `ss.modify`/`ss.change`; a function intervention | Good v4 content, but every sim is built from v3 `pars` dicts; the custom intervention edits `rel_sus` directly [CK: fair] |

Across the series:

- **No running example.** Each page starts a new SIR (or SIS, or HIV), so nothing builds up. [CK: i'm intrigued, but not necessarily convinced ... i do like the idea of building up, but am not quite sure what it would mean in practice. and i like showing a diversity of examples -- but agree that should be as well as, not instead of, a gradually building example.]
- **The solver appears once (T2) and then disappears.** Most later examples are agents on a random network, so the tutorials still read as an ABM course with a declared model bolted on. [CK: agree, hence my original question about different tracks.]
- **Missing from the tutorials altogether:** data and calibration, uncertainty across runs, stratification as a topic, time-varying parameters and seasonality, places and movement, reading a model back (`describe()`, `diagram()`), and outcomes (cases reported from infections). [CK: all good points]
- **Mixed result names** (`res.I` in T2, `n_S`/`n_I` in T4, T5 and ten-minutes) while item 5 of `oct24/plan.md` is in progress; mark these as stale rather than fixing them one by one. [CK: ok]
- **"Starsim 4 in ten minutes" is in effect the real tutorial.** It is the best page on the site, and it already follows the running-model pattern (SIR, then solvers, then age and contacts, then interventions, then data and calibration, then scenarios and runs). The core track below is mostly that page, slowed down and given exercises. [CK: LOL! it is NOT the best page! it's just the one you wrote. i don't particularly like it. its problems: "a model in two lines" sounds braggy, and is not even true. it states the obvious ("the states are declared first") without properly explaining what's happening or how it links to standard epidemiological concepts. it does a deep dive into "init" without explaining how init_prev, init_n,and init are related. it's obsessed with solvers which i think most users don't care too much about. it references starsim v3, which violates the constitution's "No baggage" rule. "What was guessed" is so claude-y, and too short compared to the other sections, and isn't even accurate (describe() describes everything, not just the guesses). diving into leaky vaccines is too much for 'ten minutes'. "vaccine = ss.vaccinate(ss.Vaccine())" is really ugly and confusing. `dict(none=None, vaccine=vaccine, masks=masks)` should be `scenarios = {"No intervention": None, "Vaccine": vaccine, "Masks": masks}`. things like `,.0f` are going to confuse and lose our non-expert readers. `np.random.default_rng(1).poisson(weekly.values)` cannot be the simplest way to add data to a time series, and you should use a different random seed for the 'truth' vs the sim. `fit = sim.calibrate()` should be `fit = sim.fit()`. "Scenarios and many runs" is too brief; you introduce `sim.new()` magically without saying what it does. "Each result of the runs has a run axis first" does not make sense. median and quantiles should be built into the results, the users shouldn't need to type "np.quantile". ]

### User guide

The guide has 27 pages, organized by module type (basics, modules, workflows, advanced). Some pages are updated for v4: diseases, parameters, results, sim, networks, calibration, time. The others still teach v3 idioms first: `def step` appears on 13 pages, `ss.SIR`/`ss.SIS` as v3-style classes on about 15, and `basics_people`, `advanced_indexing`, `advanced_random`, `modules_analyzers`, and `workflows_run` have no declared models at all. `intro_models` is a list of v3 downstream packages. [CK: agree we'll need to update these]

These v4 features have no user guide page; in brackets is where each is described today, if anywhere:

| Feature | Where it is now |
|---|---|
| Solvers: which to use, and when ODE, binomial, Gillespie, and agents disagree (extinction, small populations, step size) | `basics_sim` (a section), `architecture/solvers` |
| Stratification and dimensions: `stratify()`, several dimensions, transfers between strata, `select()`/`by()` | ten-minutes, end of T5, `architecture/model` |
| Time-varying parameters: `ss.timeseries`, `ss.seasonal`, `ss.by` | two passing mentions in the guide; "coming from" pages |
| Places and movement: `ss.geo.geography`, `Mobility`, `gravity`, `radiation` | only the "coming from" pages |
| Data by place: `ss.people('Kenya')`, `ss.contacts('Kenya')`, `ss.data` | one line of T3 |
| Outcomes and observation: `ss.outcome`, reporting delays, data attached by name | ten-minutes, part of `basics_results` |
| Reading a model back: `describe()`, `diagram()`, `equations()`, what a guess is | scattered; `architecture/model` |
| The model spec, `ss.to_code()`, `ss.diff()`, importers and exporters (`ss.interop.load`) | `architecture/spec_and_interop`, the "coming from" pages |
| Gradients and posteriors: `sim.gradient()`, `sim.log_density()`, profile likelihoods | a section of `workflows_calibration` |
| Uncertainty and many runs: `n_runs`, intervals, the run axis, seeds | ten-minutes, part of `workflows_run` |
| Several diseases tracked jointly (crossing) | one mention |
| Debugging a model and diagnosing a fit; minimal reproducible examples; common pitfalls | `advanced_profiling` (two sections) |

## Should there be a separate compartmental track?

Two parallel series (T1–T6 for agents, C1–C6 for compartments) would be the easiest thing to explain, but it goes against what v4 is: one declared model, run by whichever solver suits it. In practice:

- Most of each series would be the same pages (results, interventions, scenarios, data, calibration, plotting, many runs) with a different solver keyword. Either the pages are duplicated and drift (epydemix's guide and tutorials, EpiModel's DCM and ICM), or one series refers to the other until it is no longer a series.
- It would make readers choose a paradigm before they know the difference, which is MEmilio's weakest point (pick a language, then a solver family, before a model).
- It would hide the feature that sets Starsim apart: the same model as an ODE and as agents. No framework reviewed can show that, and a split puts it at the edge of both series instead of at the center.

What the compartmental audience does need, and doesn't get today, is (a) tutorials that don't default to agents on a random network, and (b) a home for the topics that are theirs: stratification in depth, time-varying parameters, metapopulations, contact data, gradients, and importing their existing models. Likewise the agent audience needs networks, agent demographics, and modules written as code, which the compartmental audience can skip. A shared core plus two short tracks gives both.

## Recommendation

### Tutorials: a shared core and two tracks

**Core, one running model** (a respiratory SIR that becomes an age-structured SEIR, as the odin book and ten-minutes do). Each tutorial is runnable in about ten minutes, plots in its first block, ends with exercises, and says which solver ran and why.

| # | Tutorial | Covers | Drawn from |
|---|---|---|---|
| 1 | Getting started | Install; SIR in a few lines; run, plot, `describe()`; the same model as agents with one keyword (`n_agents=`), with no v3 `pars` dict | T1, ten-minutes |
| 2 | Writing a model | States and transitions in the operator and keyword spellings; frequency and density rates; durations and Erlang stages; SEIR; branching; the class form; `ss.par` with bounds; `diagram()` | T2, T4 step 3, `modules_diseases` |
| 3 | Solvers and uncertainty | ODE, binomial, Gillespie, agents on the same model; when they disagree (small populations, extinction, step size); `n_runs`, intervals, seeds | T2's solver section; new |
| 4 | Populations | `stratify()` by age, contact matrices, declared births and deaths, real data (`ss.people('Kenya')`, `ss.contacts('Kenya')`), results by group | T3, end of T5, ten-minutes |
| 5 | Interventions and scenarios | Products and deliveries, `ss.modify`/`ss.change`, comparing scenarios with `ss.parallel` | T6 (rebuilt without `pars` dicts) |
| 6 | Data and calibration | Outcomes; data attached by name; `sim.calibrate()`; checking the fit; projecting from the fit | ten-minutes, `workflows_calibration` |

**Going further: agents** (for people modeling individuals, networks, or behavior)

| # | Tutorial | Covers |
|---|---|---|
| A1 | Networks | Random, sexual, and maternal networks; mixing pools; when a model needs agents (T5, shortened) |
| A2 | Agent demographics | Pregnancy, age- and sex-specific death data, scaling to whole populations (rest of T3) |
| A3 | Writing modules as code | `step()`, custom states, interventions and analyzers written as code, alongside a declared disease |
| A4 | Several diseases | Co-infection with `ss.modify` (the HIV and SIS example from T4) and diseases tracked jointly |

**Going further: populations** (for people coming from ODE and compartmental frameworks) [CK: call it compartments rather than populations, but ok]

| # | Tutorial | Covers |
|---|---|---|
| P1 | Stratification in depth | Several dimensions, partial stratification, transfers between strata (vaccination, migration), following summer2's sequence |
| P2 | Parameters over time | `ss.timeseries`, `ss.seasonal`, `ss.by`, and school holidays as in odin's flu case study |
| P3 | Places and movement | `ss.geo.geography`, gravity and radiation mobility, a two-patch then many-patch model (LASER's and flepiMoP's progression) [CK: these should work in agent-based as well, no? they could be T7] |
| P4 | Bringing a model in, and gradients | Importing an odin or summer2 model, `ss.to_code()`, `ss.diff()`; `sim.gradient()`, posteriors, profile likelihoods [CK: gradients and likelihoods should go in calibration] |

Exercises with solutions stay as now.

### User guide: by task, one template per page

Reorganize the guide from module types to tasks, keep the existing v4 pages, and rewrite v3-first pages so the declared form comes first and code second. Proposed sections, with new pages marked **(new)**:

- **Writing models:** Diseases (states, transitions, rates) · Parameters · Stratification and dimensions **(new)** · Parameters over time **(new)** [CK: does this really need a separate page, not part of parameters?]· Outcomes and observation **(new)** · Reading a model back: `describe`, `diagram`, `equations` **(new)** [CK: i don't think this needs its own page]· Writing modules as code (from `modules_adding`)
- **Populations and contact:** People · Demographics · Networks and mixing pools · Contact matrices and data by place **(new, partly from `modules_networks`)** · Places and movement **(new)**
- **Running:** The Sim · Solvers and when they disagree **(new; the page no framework has)** · Time · Uncertainty, seeds, and many runs **(new, partly from `workflows_run` and `advanced_random`)** · Scenarios and sweeps · Population rescaling · Deployment
- **Interventions:** Products and deliveries · Changing parameters · Interactions between diseases (connectors) [CK: connectors are not just for interactions between diseases, although that's a common use case]
- **Data and fitting:** Calibration [CK: we need to decide if we're going to use "fitting" or "calibration" ... i think 'calibration' is a bit more common, but let's be clear not to mix our terms, like `fit = sim.calibrate()`] · Gradients and posteriors **(new, from `workflows_calibration`)** · Diagnosing a fit **(new; camdl's page as a model)**
- **Sharing and interop:** [CK: call it "Interoperability"] The model spec, `to_code`, `diff` **(new)** · Importing and exporting models **(new; the user-facing half of `architecture/spec_and_interop` and the "coming from" pages)** · Library modules
- **When things go wrong:** [CK: maybe call it "Troubleshooting"] Debugging and profiling · Common pitfalls **(new; collect the real ones, as camdl does)** · Making a minimal reproducible example **(new, short)** · Array indexing and random numbers (for code modules)

Each topic page should follow one template, as MEmilio's solver pages and EpiModel's gallery pages do: what it is for, the smallest example (with a plot or printed output), the options, how it behaves under each solver (one line per solver, including "agents only" or "not under ODE"), pitfalls, and links to the tutorial that uses it and to the architecture page that explains it. The "how it behaves under each solver" line is what lets one guide serve both audiences. [CK:, but you've tended to over-emphasize solver choice in the docs pages you've written.]

### Examples gallery

`docs/examples.qmd` could become a gallery with EpiModel's template (description, a diagram from `diagram()`, a parameter table from `describe()`, the code, a headline result, next steps), seeded from:

- the nine build-set models in `v4_design/examples/README.md`, which are already tests;
- two or three LASER-style "reproduce the theory" checks (final size against Kermack–McKendrick, critical community size), which double as validation;
- for each framework, its translated showcase model, pointing to the matching "coming from" page. [CK: i like all this]

### Process

- **Write the tutorials around ten-minutes.** It already has the right arc; the core track is that page, slowed down. Keep ten-minutes as the landing quickstart, and put its first plot on the home page, as epydemix does. [CK: ok, though i don't love it nearly as much as you do!]
- **Never show the v3 `pars` dict in the core track.** Teach it once, in the user guide's Sim page, as a shorthand. [CK: ok]
- **Keep tutorials and guide distinct.** A tutorial is one path, in order; a guide page is a lookup by topic. Where they meet, the tutorial links to the guide page instead of covering it again (epydemix's duplication is the thing to avoid). [CK: agree]
- **Keep executing everything at build** (as `test_docs.yml` does), with no stub pages in the navigation: a page that isn't written yet isn't listed. [CK: ok]
- **Timing.** Starsim's design record says to rewrite the v4 docs once the implementation settles. The structure here can be set now (navigation, page list, the running model), and pages filled in as their features settle, starting with the core track, whose features (declared models, solvers, stratification, interventions, calibration) are the most stable. [CK: ok]

## Decisions for Cliff

1. **Shared core plus two tracks, or two full parallel series?** I recommend the former, for the reasons above. If you'd rather have two series, the cost is roughly doubling the tutorials (about 900 lines today), with most changes then made twice. [CK: yes, shared core]
2. **Running model.** A generic respiratory SIR that becomes an age-structured SEIR (as above), or a named disease (measles, which matches the Ghana measles work and LASER's England and Wales example)? A named disease is more engaging but brings more parameters to explain. [CK: let's do measles, it's barely more complicated]
3. **The user guide: reorganize by task (above), or keep the current sections and add the new pages into them?** Reorganizing moves most existing pages; adding is cheaper but keeps the module-type ordering that no other framework uses. [CK: we can reorganize]
4. **Gallery: build it now from the build-set models, or later?** It is mostly assembly from existing tests and vignettes. [CK: you can build it now, but call it gallery.qmd not examples.qmd]
