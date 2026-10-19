# What the switching seats found that the reviews did not

The thirteen switching assessments in [switching/](switching/) each worked through a competing framework's routine tasks in v4, and in doing so found bugs and gaps that none of the four reviews raised. They are collected here so they can join the merged registers in [findings/](findings/). "Checked" means I re-ran the agent's repro myself (scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/me/`); everything else is the agent's finding, with its repro in that agent's scratch folder and the details in its report.

## Silent wrong answers

| # | Finding | Who hits it | Status |
|---|---|---|---|
| N1 | **The particle filter's likelihood is wrong when `beta` is a fitted parameter times `ss.seasonal` times `ss.timeseries`.** With either factor alone the filter peaks at the true `b = 0.5`; with both it peaks at 0.4 (−11,349 against −17,016 at the truth), while direct runs peak correctly at 0.5 (−375). On epydemix's COVID tutorial, particle MCMC drives `R0` and `Delta` to their bounds and the log-likelihood gets worse after fitting (103 before, 3,249 after). The filter's test is skipped. | epydemix, any seasonal model fitted by PMCMC | Checked (`epydemix/t2_pf_repro.py`) |
| N2 | **The JAX log-density is wrong when a fitted `ss.par` multiplies an `ss.timeseries`**: −967,465 against the sim's −146.7, so NUTS falls back to the Python-callback path: 927 s, all 2,000 draws divergent, R-hat 5.6e15. The same model with the time series but the multiplier not fitted agrees. This is how summer2 users write case detection, and how many write any scale-up. Possibly the same root as N1; not established. | summer2, odin, anyone with a fitted scale-up | Checked (`summer2/nuts_bisect.py`, variant `fitdm_ts`) |
| N3 | **`sir.age < 5` and `sir.age >= 50` compare the age group's index, not the age.** In an age-stratified model, `eligible=sir.age < 5` vaccinates 52.2 million (everyone in Kenya), `sir.age >= 50` vaccinates nobody, while `sir.age == '0-4'` and `ss.People(1).age < 5` give the right 6.9 million. F2 07 listed the comparison as working. | epydemix, Ghana measles, anyone targeting by age | Checked (`epydemix/t4_cond.py`) |
| N4 | **Naming a dimension `patch` (or `region`, or giving `kind='space'`) silently discards an explicit contact matrix across it.** With `ss.contacts(patch=[[1, .5], [.5, 1]])`, the second place gets 0 infections; the same model with the dimension called `zone` gives 9,882 in each. 82% fewer at the peak in camdl's polio model. This is the merged D4 mechanism, but in its worst form: the user gave the matrix. | camdl, MEmilio, any spatial model with a mixing matrix | Checked (`me/patchmat.py`) |
| N5 | **After `sim.init()`, `ss.parallel(sim, n_runs=4)` returns four identical runs** (782, 782, 782, 782), silently, and 5x slower. Without `init()` the same call returns one sim that holds all runs (merged R2). | odin, anyone who inspects a sim before running it | Checked (`me/identical.py`) |
| N6 | **A Poisson or negative binomial likelihood on data that are not whole numbers is `-inf`**, with no message; `calibrate()` then returns a garbage posterior (R-hat 5.6e15) without `W1304`. Weekly averages, rescaled counts, and imputed data are all non-integer. | flepiMoP, summer2 | Checked (`me/newbugs.py`) |
| N7 | **A Starsim 3 `beta` dict of bare numbers is a per-step probability; in v4 it is a per-unit-time rate.** A verbatim port drops transmission 12x at monthly steps and 52x at weekly ones, with no warning. The import skill (`skills/import-model/references/starsim3.md:17`) calls this mapping "Exact". | every Starsim 3 user | Agent (starsim3) |
| N8 | **The spec's "routine at 9 months" vaccinates everyone already older at the start**: 8,757 of 10,000 people on the first day. | Ghana measles, any routine immunization model | Agent (ghana_measles) |
| N9 | **"Routine" means different things in v3 and v4.** v3 `routine_vx(prob=0.05)` reaches 72.7% by 2030; v4's routine `ss.vaccinate` reaches 5.5%. The v3 behavior is `interval=ss.years(1)` in v4. | every Starsim 3 user | Agent (starsim3) |
| N10 | **Unchanged Starsim 3 code gives results one step later and loses the last step** (the new time-point convention), while `CHANGELOG.md:7` says such models "work as before" and the import skill (line 79) says results line up. Published numbers do not reproduce exactly. | every Starsim 3 user | Agent (starsim3) |
| N11 | **`ss.by(stage=...)` on an inflow gives 0 births, silently**, and a birth has no `where=`; births from one stratum into another cannot be declared. | EMULSION (farrowing), any stage-structured population | Agent (emulsion) |
| N12 | **The EMULSION importer divides `proba` by `delta_t`** (`interop/emulsion.py:225`; 47.7 against EMULSION's 89.6), drops the metapopulation level and `productions` without a `W1601` entry, and a sim with 16 untranslated parts runs as if complete (15 animals alive at the end against 66). | EMULSION | Agent (emulsion) |
| N13 | **Importers run silently wrong on real models**: the odin importer turns a contact-matrix model into one with no transmission and crashes on `lambda` and `if`; the camdl importer drops every intervention that is off by default (`with_sia` imports as `{}`, unlisted); 6 of 16 Atomica library workbooks fail with tracebacks and `sir_vaccine` is 35% off; the summer2 dumper fails on any named `Parameter`; the flepiMoP importer ignores `subpop_setup.selected` and ends a day early. | odin, camdl, Atomica, summer2, flepiMoP | Agents |
| N14 | **`stratify(immunity=...)` silently makes no dimension**, and an explicit `split=` on a disease-level stratification is ignored when the sim has the same groups. | summer2 | Agent (summer2) |
| N15 | **`PartnerNet` with `match=` alone silently loses risk structure**: prevalence 31%/31% by risk group against EpiModel's 21%/77% (a 35-line subclass with per-group degree gets 23%/80%). Not a bug in what `match=` does, but nothing tells the user their model has no degree heterogeneity. | EpiModel | Agent (epimodel) |
| N16 | **A stochastic agent fit defaults to one run per evaluation and keeps the luckiest draw** (fitted 0.236, true about 0.20). Merged R4 covers the reported score; the bias in the estimate is new. | EpiModel, any stochastic fit | Agent (epimodel) |

## Crashes and dead ends

| # | Finding | Who hits it | Status |
|---|---|---|---|
| N17 | **No v4 model can be built from R.** `sys._getframe(1)` at `diseases.py:74` and `sim.py:73` raises `ValueError: call stack is not deep enough` when there is no Python caller frame, which is the case under reticulate. R-Starsim's own README example breaks; it is a regression from `main`. The fix is a fallback when there is no frame (about ten lines) and an R-Starsim test in CI. The agent checked that with the constructors wrapped, the intro model runs from R and `to_df()` returns a data frame. | EpiModel, SimInf, odin, and every R user | Checked (`me/noframe.py`: a call with no Python caller frame) |
| N18 | **`sim.stratify()` is lost by `sim.new()`**, so `ss.parallel(sim, ...)` and `calibrate()` fail on the spelling the best practices use (`sim.py:535` against `:86`/`:431`). | MEmilio, anyone stratifying at the sim | Checked (`me/strat_new.py`) |
| N19 | **Conditions on a coded module's arrays crash**: `where=hiv.cd4 < 200` raises `AttributeError` naming the wrong module (`conditions.py:295-315`); a lambda works. | every Starsim 3 user | Agent (starsim3) |
| N20 | **Parameters that cannot be fitted where users put them**: a value per place (`ss.par(ss.by(patch=...), bounds=)`, an `ss.par` as a modifier's amount, `ss.by` of `ss.par`s all raise `TypeError`); `away=` on mobility; a par inside `ss.seasonal(mean=...)` (`E206`); `ss.perday(par/8)`; typed bounds (`bounds=[ss.days(3), ss.days(30)]`); `capacity=` as an expression of a par; a par used only by an intervention cannot be set by `pars=` (`E602`). | flepiMoP, MEmilio, Ghana, odin, Atomica | Agents (the typed-bounds crash from the results merge) |
| N21 | **Small dead ends with unhelpful messages**: `n_contacts=ss.by(...)` (`'by' object has no attribute 'pars'`); `ss.perday(ss.by(...))` and `ss.days(ss.by(...))` (bare `TypeError`); `calibrate(gof=fn)` still requires `data=`; three trial-count spellings, one of which crashes inside Optuna; TSV data gives a misleading `E705`; `E705` says "matches no result" for an ambiguous column. | several | Agents |
| N22 | **A module that fails when tried on groups makes `auto` choose agents**: 10,000 agents for 60 to 83 million people, which then crashes or misleads. The merged X9 covers `auto`'s reasons; this is its worst case. | MEmilio, flepiMoP | Agents; in F 06 §4 |

## Performance at the scale these users work at

None of the reviews measured v4 at a switching user's real scale. The seats did:

| # | Finding | Numbers | Kind |
|---|---|---|---|
| N23 | **Mixing within places plus commuting builds dense (places × ages)² Kronecker matrices** (`solvers.py:181-182`) | 400 counties × 6 ages: 25 s against MEmilio's 1.5 s; 2 × 2.8 GB at 3,000 places | A bug: a factored product costs O(places × ages² + edges × ages) |
| N24 | **Results are kept for every group at every step**, cumulative ones too | Ghana with ages and a campaign: about 5 GB per run, 9.9 GB for two in `ss.parallel`; flepiMoP: 0.6 GB per run at 3,000 places; LASER: 0.28 GB per simulated year at 954 places | A bug: keep groups at a reporting interval, or `results=[...]` as specified |
| N25 | **Agents make 10 to 15 single-threaded passes over the population per step**, at about 330 bytes per agent | 3.3M agents across 149 places: 67.8 s against LASER's 2.75 s on 20 threads (8.2 s on one); LASER uses 30 bytes per agent | Mostly fixable: one fused per-step pass, then threads; int32 UIDs and int16 timers |
| N26 | **Gillespie by place visits every herd every day** and applies the register to all | 16,000 herds and 4.7M events: 52.5 s against SimInf's 1.9 s, plus 13 s to build against 2.1 s; twice the memory | Mostly fixable: step only the herds an event touched; threads across herds |
| N27 | **The particle filter is serial** | 11 µs against odin's 1.35 µs per particle-step on one thread (8x); about 9x slower than camdl's PMMH | Fixable: particles in parallel in the kernel, a stream per particle |
| N28 | **Building a sim with many places is slow** (deep copies at `ss.Sim()`) | 774 patches: 100 s to build for a 5 s run | A bug |
| N29 | **Declared diseases as agents are slower than the same model as Starsim 3 code** | 1.4 to 1.6x | Should be no slower |

## Missing features that one community cannot work without

Each is small to medium, and none needs a new design: each fits a mechanism v4 already has. The table counts how many of the thirteen seats asked for it, which is a reasonable order of work.

| Feature | Seats that need it | Size (agents' estimates) |
|---|---|---|
| A fitted parameter with a value per place or group (vector `ss.par` with a shared prior), and a par as a modifier's amount | flepiMoP, MEmilio, Ghana, Atomica | Medium |
| Forward draws from priors or a posterior for projections (`ss.parallel(..., priors=True)`, `fit.project(scenarios, n=)`), and quantile tables (`to_df(quantiles=)`, `to_hub()`) | flepiMoP, MEmilio, epydemix, Ghana, summer2 | Small each |
| A one-off move between states as a delivery (`move='S -> V'`), and an all-or-nothing vaccine; `at_age=` for routine doses | camdl, Ghana, LASER, EMULSION | Hours to medium |
| ABC-SMC (the spec's method for "stochastic, no likelihood"), IF2 for maximum likelihood, binomial and Bernoulli observations | epydemix, SimInf, camdl, EpiModel | Medium |
| Parallel particles in the filter, and threads within one large run | odin, camdl, SimInf, LASER | Medium |
| The JAX log-density (and run) exposed as a function, with reverse-mode gradients through the whole fit | summer2, odin | Medium |
| A per-place condition on a result with a hold (`when=`/`where=` on 7-day incidence by place, `hold=`) | MEmilio, flepiMoP, SimInf (reactive movement bans) | Small to medium |
| Per-group degree, several `match=` properties, a mixing matrix, and `acts=` on `ss.PartnerNet`; duration in `net.check()` | EpiModel | About 300 lines |
| Declared programs (spending to coverage to effect), `ss.optimize` over spending, a databook generated from the model and read back | Atomica | 500 to 800 lines |
| Minimum stays (`after=`/`before=` on a transition), `offspring=` on a transition, inflows from one stratum into another | EMULSION | Small to medium each |
| Declared life-table deaths, fixed-count seeding, gravity scaled to a mean share with a cap | LASER, Ghana | Days each |
| An exact fixed delay (a conveyor, not 200 stiff Erlang stages), and a declared auxiliary variable updated in the kernel | odin | Medium |
| Dimensional checking through expressions at `init()` (four of five unit slips run silently today); the merged L19 proposes building `E102` | camdl, summer2 | Medium |
| v4 verbs that reach coded modules (`ss.modify`/`ss.vaccinate` on `rel_sus`/`rel_trans`, conditions on coded arrays, `calibrate(pars=...)` on any module, `sim.new()` for any unrun sim) | Starsim 3 | 1 to 2 weeks |
| Data: US states (Mistry 2021), German level 2 with Destatis populations and commuters | epydemix, MEmilio, flepiMoP | Medium |
