# camdl: would its users switch?

## Who this user is

camdl is a modeling language for compartmental infectious disease models, with its own toolchain: a model checker written in OCaml (`camdlc`) that expands a `.camdl` file into a flat JSON IR, and a Rust runtime (`camdl`) that simulates the IR (ODE, chain binomial, Gillespie) and fits it (bootstrap particle filter, IF2, PMMH, PGAS with NUTS, profile likelihood). It is developed at IDM, as Starsim is. In the checkout at `_envs/camdl/src`, 2,709 of the 2,710 commits are by one author (Vince Buffalo), and the project calls itself alpha (`README.md`, `VERSIONING.md`). Its docs point to the users: a 238- to 774-LGA Nigerian polio model with SIAs (`docs/intro.md:464`, `ocaml/golden/polio_spatial_5.camdl`, described as "a development sandbox for the 774-LGA Nigeria model"), typhoid and cholera models, and measles fits compared with pomp (`docs/user-features.md`). So the user is a statistically minded modeler in the pomp tradition. They fit stochastic compartmental models to messy surveillance data, and they want maximum likelihood by iterated filtering, then a Bayesian posterior. They value three things above all: the observation model is part of the model file; the compiler rejects a wrong model (rates and durations are typed, so a dropped `/N` is `E300` at check time); and every run and fit is content-addressed, with provenance, so it can be cached and reproduced. They often work through a coding agent (`AGENTS.md`, `camdl docs`). They picked camdl because a `.camdl` file reads like the whiteboard and the CLI does the rest.

Lines are counted against the `.camdl` file throughout, since that is what this user writes. pycamdl, the Python port used for the stratified example, is marked "not production code" (`/home/cliffk/idm/camdl/pycamdl/README.md`) and is untracked in that checkout.

## The first day

| Vignette | Lines (camdl → v4) | Time | Agreement |
|---|---|---|---|
| `sir` (closed SIR, weekly cases, negative binomial) | 30 → 13 | ODE run 0.009 s as a command, against 0.022 s to build and run (0.6 ms to rerun a built sim) | ODE within 0.02%; 200 chain-binomial runs within 1% for `I` and 1.6% for weekly cases |
| `stratified` (SEIR by 3 ages × 2 patches, contact matrix, aging, movement, vaccination at day 20) | 41 → 15 (against my `.camdl` version; the vignette's 35 counts the pycamdl script) | ODE 0.10 s (pycamdl) against 0.023 s | ODE within 0.24% in every cell; stochastic peaks agree, with a 1-day shift at `dt = 1` |

These numbers are from `vignettes/camdl/*/README.md`, except the 41. The vignette counted pycamdl lines, so I wrote the stratified model as a `.camdl` file (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/camdl/stratified.camdl`): 41 lines that are not blank or comments, none over 100 characters. `camdl check` passes it. Its ODE runs, without and with vaccination, agree with the pycamdl reference within 0.21% and 0.28% of the peak (`cmp_strat.py`).

This user would recognize both translations. They would like `cases = ss.outcome(rho * infection, every=ss.weeks(1), likelihood=ss.nbinom(n=k))`: the observation model stays in the model, as in camdl, in one line instead of a six-line `observations {}` block. They would also notice two substitutions in the stratified vignette. The vaccination is not a move from `S` to `R`, since v4 has no one-off transfer between states. It is a fully effective vaccine, so the `S` and `R` counts differ from the original. And `ss.People(1).age < 5` stands in for "children". The camdl NUTS comparison in the vignette is fair and favorable: the same posterior, 2.5 s of sampling against 2.2 s, with more effective draws per second, but 5.5 s from the command line against 2.3 s, because of JAX compilation and imports.

`docs/` in ssv4 is all Starsim 3. This user's first question is "how do I fit with a particle filter?", and nothing in `docs/` answers it. `docs/user_guide/workflows_calibration.qmd` teaches the v3 `ss.Calibration`, which has none of the likelihood machinery. camdl ships version-matched docs in the binary (`camdl docs`); v4's are in the design folder.

## The second week

Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/camdl/`.

### 1. Fit the SIR to weekly cases, as I would with `camdl fit run`

I drew 12 weeks of cases with `camdl simulate sir.camdl --obs` at `beta = 0.4`, `gamma = 0.15`, `rho = 0.6`, and `k = 10` (seed 7: 73, 267, 1060, 1409, ... cases a week). Then I fitted `beta` and `gamma` three ways, with the same priors:

| Fit | Wall time | `beta` | `gamma` | Diagnostics |
|---|---|---|---|---|
| camdl IF2 (`fit_if2.toml`: 4 chains, 1,000 particles, 50 iterations) | 11.9 s | 0.480 (MLE) | 0.230 | Â 1.03 and 1.05 flagged ✗; 1 of 4 chains diverged; dt check passed |
| camdl PMMH (`fit_pmmh.toml`: 4 × 2,000 iterations, 1,000 burn-in, 200 particles) | 21.1 s | 0.430 (0.348 to 0.523) | 0.159 (0.062 to 0.303) | R-hat 0.995, bulk ESS 377 and 357, acceptance 32 to 37% |
| v4 `sim.calibrate()` (`fit_v4.py`: chose PMCMC itself, 4 × (1,000 + 1,000), 50 particles chosen in warm-up) | 188 s | 0.427 (0.348 to 0.528) | 0.152 (0.061 to 0.333) | Acceptance 12 to 18% |

The posteriors agree to the second decimal, which is the strongest thing in this report. v4's particle MCMC is correct, and the user wrote nothing but `data=` and `sim.calibrate()`. It is about 9× slower than camdl's, with a quarter of the particles: about 75 ms per likelihood against camdl's 10.4 ms. Part of the 188 s overlapped another fit on the same machine, so the ratio is approximate. v4 has no IF2, so there is no maximum-likelihood estimate for a stochastic model; for that, v4 falls back to Optuna, which gives a point estimate without a likelihood.

Two frictions on the way. camdl's data and outputs are TSV. `ss.Sim(..., data='syn_obs.tsv')` reads it as a CSV and fails with `E705 UnknownDataColumn: the data column "time	cases" matches no result`, then suggests renaming a column called `"time\tcases"` with `aliases=`. The fix, `pd.read_csv(path, sep='\t')`, is one line, but the message points the wrong way. The other friction is camdl's, for balance: `camdl fit summary` on the IF2 fit printed "no completed methods found" because of a missing provenance sidecar, and PMMH refused to run until I set `burn_in` below its default of 5,000.

### 2. Bring my polio model in

`ss.load('polio_spatial_5.camdl', format='camdl')` (`imp_polio.py`; it needs `camdlc` on the `PATH`) took 0.02 s and reported nothing incomplete. But this is a polio model, and its question is what the SIAs do. The five SIA interventions are "off unless enabled in camdl, so it is left out", a line visible only in `sim.imported.choices`. The `with_sia` scenario comes in as `{}`. So the import cannot answer the model's question, and `W1601` does not say so. The model also arrives flat: 25 states (`S_north`, ...) and 20 importation transitions written out with counts, not `stratify(patch=...)`. That is honest (`starsim/interop/camdl.py:12-15` says so), but it is not the model a v4 user would write. My `stratified.camdl` imports with the vaccination listed as not translated (a one-off move between states has no declared form, `v4_design/admin/deferred.md`). `seir_observations.camdl` imports with its Bernoulli detection stream dropped to a plain result. Separately, the importer's docstring gives camdl's home as `github.com/epiforecasts/camdl`; the environment builds it from `github.com/vsbuffalo/camdl`.

### 3. Write the polio model by hand, with its SIAs

`polio_v4.py` is the 5-patch SEIR with gravity importation and three SIA rounds per patch. It is about 23 lines (counting long lines twice), against 80 for `polio_spatial_5.camdl`, of which 22 are the scenario block. The importation, `kappa * W[p,q] * S[p] * I[q] / N[q]`, is a contact matrix by patch: `np.diag(R0*gamma) + kappa*W` with `ss.contacts(patch=...)`.

**It silently gave the wrong answer.** With the dimension named `patch`, which is camdl's own word and the one in v4's spec, total `I` peaked at 23,045 on day 133, and only the north patch was ever infected. Against camdl and an independent SciPy ODE (`polio_ref.py`), the right peak is 130,643 on day 145, so v4 counted 82% fewer at the peak. `kind='space'` and `region=` behave the same way. With `place=` or `location=`, v4 matches camdl to the person: total infections are 950,610 without SIAs and 863,920 with them, the same as camdl's ODE (`polio_dbg3.py` variants). The reviews have the cause, a dimension's name deciding whether places mix (opus [dims 3](../opus_review/notes/dimensions-and-mixing.md), fable [03 U1](../fable_review/03_dimensions_routes.md)). This case is worse than the reviews show, though. Here the user gave the between-patch contact matrix explicitly, and it was dropped without a word, which breaks opus's proposed principle that a guess never reinterprets what was given.

The SIAs needed a workaround. With no `transfer(S -> V)`, they are `ss.vaccinate(ss.Vaccine(efficacy=1, waning=ss.days(1/0.003)), years=[...], coverage=ss.by(place=...))`, one per patch. That is exact here only because camdl's `V -> S` waning is exponential at a constant rate. Once those were in, the stochastic answer agreed too. Over 200 chain-binomial runs, the median total infections without SIAs is 967,084 (camdl 967,042) and with them 838,948 (camdl 832,692). The mean averted by SIAs is 141,816 (camdl 145,429). v4 took 1.9 s and 3.1 s for the 200 runs; camdl took 4.4 to 4.6 s as a command. (camdl had one run in 200 that died out and v4 had none, which explains camdl's larger spread; at about 1% per run, that difference is chance.) One loss remains: in this form `R0` by patch is folded into a matrix, so it cannot be a fitted parameter, as `R0[patch] : positive in [0.5, 10]` is in camdl. `ss.par(ss.by(place=[...]), bounds=...)` exists and builds (`parby.py`), but I did not find a form that keeps a fitted `R0` by patch and importation that is not scaled by it.

### 4. Make the mistakes camdl's checker catches

camdl's headline is "dimensions are types". `units.py` tries the slips it is meant to catch:

| What I wrote | v4 | camdl's equivalent |
|---|---|---|
| `S >> I @ ss.perday(0.4) * I.n` (a count, not a share) | Runs; final `R` is NaN, with no message | `E300` wrong dimension |
| `S >> I @ ss.days(0.4) * I` (a duration as a rate) | Runs, and everyone is infected | Rejected |
| `I >> R @ ss.perday(0.1) * ss.days(5)` (a dimensionless 0.5 as a rate) | Runs, final `R` 351 | `E300` |
| `I >> R @ ss.perday(0.1) + ss.days(5)` | `TypeError: Only rates can be added to rates` | `E302` |
| `I >> R @ 10` (meaning 10 days) | Runs as 10 a day, final `R` 10 | A bare number is also accepted |

v4 has typed quantities but does not check dimensions through an expression. `E102` is in the error catalog and never raised; fable [09](../fable_review/09_model_spec_errors.md) proposes dropping it because "it needs dimensional analysis nobody has written". To this user, that would remove the one guarantee they came for. In fairness, the operator form makes camdl's most famous slip, the dropped `/N`, impossible: a bare state is already a share. And camdl's checker is not airtight either. It accepted `c[a, p] / stay[a]` with `stay` a unitless table (`stratified.camdl`, 0 warnings), as its own golden models do.

### 5. Scale to the Nigeria model

`lga/make.py` writes a 774-patch SEIR with six-neighbor gravity importation (4,644 links, 190 million people), as `lga2.camdl` in the pattern of `docs/intro.md:497-503` and as `lga_v4.py`. In v4, the ODE ran a year in 5.0 s, but building it took 44 s and `sim.new()` another 58 s (`t_build.py`). The profile puts that in `sc.dcp` deep-copying arrays at `ss.Sim()` (`starsim/sim.py:65`, through `parameters.py:79`). That is a bug, and the same family as opus's dense group matrices ([dims 7](../opus_review/notes/dimensions-and-mixing.md)). camdl did not get to a first step. `camdl check lga2.camdl` and `camdl simulate lga2.camdl --backend ode` were each still in `camdlc` when my 15-minute limit stopped them. My first version, with a dense `sum(q in patch, ...)` in the rate, had run for 10 minutes before I stopped it. The camdl docs list 774 all-pairs coupling as open work (`docs/compartmental-ir-spec.md:1607`, "O(N²) transitions"), with sparse coupling as the mitigation. I may not have used whatever pattern the real Nigeria model uses, so take this as "the documented pattern does not scale to 774" rather than "camdl cannot". At this size, a contact matrix (v4) is the better representation than one transition per pair (camdl).

## What would win them over

- **The same posterior with almost nothing written.** v4's particle MCMC matched camdl's PMMH on `beta` and `gamma`, and the user wrote no `fit.toml`, no `[estimate]`/`[fixed]` split, and no method block: bounds on `ss.par`, a likelihood on the outcome, `data=`, then `sim.calibrate()`, which picks the method and says why.
- **A third of the lines.** 30 → 13, 41 → 15, and 80 → about 23 for the polio model with its SIAs, with `R0`, populations, and SIA days as ordinary Python data rather than a scenario block.
- **Agents with the same file.** camdl is compartmental only. `solver='abm'` with a network runs the same declared model as agents, which a polio modeler eventually wants for clustering of zero-dose children or for environmental surveillance catchments.
- **Size.** A 774-patch contact matrix runs a year in 5 s as an ODE, where camdl's per-pair transitions are its own declared open problem. The build overhead in front of it is a fixable bug.
- **Scenarios are Python.** `ss.parallel(baseline=..., with_sia=...)` takes any two sims, instead of a `scenarios {}` block whose entries do not inherit each other's `set` (`with_sia` needed `--params` in camdl). Both pair runs by seed; in this model neither pairing narrowed the spread of the difference much (paired SD 56,704 in v4, 66,877 in camdl), since the SIAs change the later waves.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff? | Already in a review? |
|---|---|---|---|---|
| `patch=` (or `region=`, or `kind='space'`) silently discards an explicit between-patch contact matrix: 82% fewer at the peak in the polio model | Bug (a guess reinterpreting what was given) | Yes, small: an explicit `ss.contacts(<dim>=matrix)` defines the mixing across that dimension whatever its kind; then kind from the value | No: given a matrix, use it | Yes, the mechanism (opus dims 3, fable 03 U1); not this case |
| No dimensional checking through expressions: four of five unit slips run silently, one to NaN | Missing feature | Yes, medium: propagate `[T]`, `[1/T]`, `[P]`, `[1]` through `Expr` at `init()`; the types already exist | No. The typed values are there, and checking them is what makes them pay. Don't drop E102; build it | Partly: fable 09 (proposes dropping E102) |
| No one-off move between states (`transfer(S -> V) at [...]`), so SIAs are products with waning; exact only when `V -> S` is exponential | Missing feature | Yes, medium: a delivery with `move='S -> V'` (the kernels already apply shifts, `ss.Delivery.tabulated`) | No. "Product plus delivery" is right, but a move is the most common campaign in polio and measles models | `v4_design/admin/deferred.md` lists it; no review |
| The camdl importer leaves out every intervention that is off by default, so `with_sia` imports as `{}` and nothing is listed | Bug | Yes, small: list them in `W1601`, and import them as disabled deliveries once moves exist | Not applicable | No |
| Particle MCMC about 9× slower than camdl's PMMH (75 ms against 10.4 ms per likelihood, with 50 particles against 200) | Missing feature (performance) | Yes, medium: parallel particles in the kernel | No: a general kernel can get much closer | Partly: fable 08 (test skipped for speed) |
| No IF2 or profile likelihood: no maximum-likelihood estimate for a stochastic model | Missing feature | Yes, medium: IF2 is a short loop over `ParticleFilter` and `kernels.advance()` | No. The design's "wrap existing samplers" fits, but no Python IF2 exists to wrap, so this one is worth writing | No |
| Fewer observation families: v4 has Poisson, negative binomial, normal, and lognormal; camdl adds Bernoulli, binomial, zero-inflated, diagnostic tests, and prevalence streams | Missing feature | Yes, small each | Yes, once binomial (seroprevalence) and Bernoulli (detection) are in | No |
| TSV data fails with a misleading E705 | Bug (UX) | Yes, small: sniff the separator | Not applicable | No |
| No content-addressed store; `sim.hash()` misses options such as CRN | Inherent tradeoff (a library against a CLI with a store) | Partly | Mostly yes. A library should not own a results store, but the hash should be complete (codex F04) so a user can build one | Yes: codex F04 (hash omits CRN and modifiers) |
| A 774-patch model takes 100 s to build for a 5 s run (deep copies at `ss.Sim()`) | Bug (performance) | Yes, small to medium | Not applicable | Related: opus dims 7 |
| Docs are v3; no offline, version-matched docs (`camdl docs`) for agents | Docs | Yes | Not applicable | Yes: opus §6 |
| Python, not a checked language; a model is code that runs, not a file a checker proves | Inherent tradeoff | No | Yes, if the checks run at `init()` (above). The Python form is what buys agents, networks, and arbitrary interventions | No |

## Verdict

- **Today:** no. The first realistic polio model gave a silently wrong answer from the dimension name. SIAs need a workaround. Unit slips go through. There is no IF2.
- **After the four reviews' fixes:** closer, but still no. The reviews fix the dimension-name trap and some guessing, but not dimension checking (one proposes dropping it), one-off moves, IF2, or filter speed.
- **What would make it a clear win:** a dimension check at `init()`, a `move=` delivery for campaigns, IF2 next to PMCMC, and a filter within about 2× of camdl. Then v4 is camdl's model and fit in a third of the lines, plus agents and networks, which camdl does not plan to have. Since both are IDM projects, the cheapest path might be to borrow camdl's checker rules and IF2 design outright.

## The demo that would sell it

The polio question, end to end, in one script:

```python
import starsim as ss
sim = ss.load('polio_nigeria.camdl', format='camdl')      # 774 LGAs -> stratify(patch=...), SIAs as moves
sim.describe()                                           # every SIA listed, with its rounds
sim.check()                                              # E300-style unit errors before anything runs
fit = sim.new(solver='binomial', data=ss.data('afp_cases.tsv')).calibrate(method=['if2', 'pmcmc'])
sims = ss.parallel(baseline=fit.sim, with_sia=fit.sim.new(enable='sia'), n_runs=200)  # paired by CRN
sims.diff('baseline').plot()                             # cases averted by LGA, with intervals
agents = fit.sim.new(solver='abm', n_agents=2e6)         # the same model as agents, for zero-dose clustering
```

What v4 needs: the importer to keep strata as `stratify()` and SIAs as moves, a unit check, IF2 and staged fitting (`method=[...]` crashes today, opus [results 10](../opus_review/notes/results-and-calibration.md)), TSV data, a fast build at 774 patches, and `enable=` for deliveries. It does not run today. The import drops the SIAs, there is no IF2, and staged fitting is not built.
