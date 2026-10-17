# flepiMoP: would its users switch?

## Who this user is

A modeler in the Johns Hopkins ID Dynamics group or a Scenario Modeling Hub team that uses its code. They write YAML, not Python models: a config with `compartments` (often infection stage × age × vaccination × variant), `subpop_setup` (a geodata CSV of the 51 states or 3,142 counties and a commuting CSV), seeding from a file, `seir_modifiers` and `outcome_modifiers` (NPIs and vaccination as parameter changes over periods, often for some states only), `outcomes` chained with delays and durations (cases, hospitalizations, current hospitalizations, deaths), and `inference`: fitting each state's modifier values ("perturbations") to weekly cases, hospitalizations, and deaths per location with MCMC, over many chains run as SLURM or AWS Batch jobs. Their standard output is a hub submission: for each scenario, location, target (`inc hosp`, `inc death`, `cum death`), and week, 23 quantiles from 100 to 300 runs. They chose flepiMoP because a new round of scenarios is a config edit, because it scales to every US county, and because it is the code behind years of SMH and FluSight submissions. Post-processing is in R. Python is gempyor, under the hood; most users never touch it.

## The first day

The vignettes (`lingua_franca_design/vignettes/flepimop/seir_2pop`, `vaccine_scenarios`) are models this user recognizes immediately: the transitions read like their `transitions:` block, the stacked modifiers are a list of `ss.modify`, and the outcomes chain as in `outcomes:` (`deaths = ss.outcome(0.2*hosp, delay=ss.days(14))`). The vaccine scenarios are 36 lines against 141 of YAML plus 10 of CSV, and 30 runs (3 scenarios × 10 draws) take 6.5 s against 96 s; with the original's draws every series agrees by province within 1e-9 of its largest value. Most of flepiMoP's time is Numba compiling the integrator on every run (about 7 s), which this user has learned to live with and would be glad to lose.

What they would notice: two of the 36 lines are a workaround for drawing `Ro` once per run (the vignette's open finding 1), and the commuting is `kind='visit'`, because flepiMoP's travellers are infected where they go but infect nobody there. That the translation found and named this asymmetry is the kind of thing that earns trust.

Then the front door. `docs/` is all Starsim 3, so this user meets v4 through `v4_design/design/` and `skills/import-model/references/flepimop.md`, which is the right page for them (a mapping table, "what needs care", how to run the original) and is not reachable from the docs. The overview's example fails at the scenarios line (Fable 2 `03` finding 1).

## The second week

Scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/flepimop/`.

### 1. Their own config, at county scale

`gen.py` writes a flepiMoP project with 3,000 subpopulations (populations summing to 330 million), commuting to each one's 10 nearest neighbours (29,958 rows), 20 seeded places, an SEIR with `Ro` drawn from a truncated normal, incident cases (probability drawn per run), hospitalizations with a 10-day stay (`currHosp`), deaths chained from hospitalizations, and two scenarios (baseline, and `Ro` × 0.6 for two months). flepiMoP: `flepimop simulate config.yml --jobs 1`. v4 (`v4_import.py`):

```python
sim = ss.load('config.yml', format='flepimop')
run = sim.new(pars=dict(Ro=2.188386, p_incidCase=0.262786))   # the values slot 1 drew (spar/, hpar/)
run.run()
```

| | flepiMoP | v4 |
|---|---|---|
| 300 places, both scenarios | 36 s wall | 0.1 s a run (plus 1.9 s of imports) |
| 3,000 places: one SEIR run | 57.5 s (first, with compiling), 34.4 s (second); outcomes 6 to 9 s | 5.0 s, outcomes included; reading the config 1.7 s, building 0.2 s |
| 3,000 places: whole command, two scenarios | 111 s, 803 MB | not timed as one command (about 14 s, estimated from the parts above); 925 MB peak for the comparison script, which also reads flepiMoP's 108 MB of CSV output |

Agreement at 3,000 places, every place: incident cases, hospitalizations, deaths, and current hospitalizations within 0.007% of each place's peak on every day (exact at 300). One difference in conventions: flepiMoP's output includes the incidence during its last day (31 July), and the import's run ends at 31 July, so that day is missing (1.5% of the national total here, near the peak); setting the import's `stop=` to the end date plus one day would match. The importer listed exactly what it did not translate (the two drawn distributions, at their means). Fifty per-place modifiers (`ss.modify(..., where=dis.subpop == p, start=, stop=)`) keep the run in one kernel call (build 0.3 s, run 5.0 s). This is the strongest result in this report: a user's config comes in, runs seven times faster per run, and agrees to five digits.

### 2. A hub submission: scenarios × draws, quantiles by location and week

`smh.py` draws `Ro` and the case probability by hand (`scipy.stats.truncnorm`, since a prior is not drawn in forward runs), makes one sim per scenario and draw, runs them with `ss.parallel`, and builds the table with about ten lines of pandas: sum each run's daily `incidHosp` and `incidDeath` by place into weeks ending Saturday, take 23 quantiles over runs, and write long-form rows (`quantile`, `target_end_date`, `location`, `value`, `scenario_id`, `target`).

- 300 places, 2 scenarios × 20 draws = 40 runs: 4.3 s; the table (745,200 rows) in 0.2 s; peak memory 1.2 GB.
- 3,000 places, 2 × 5 = 10 runs: 54 s; the table (7.45 million rows) in 0.6 s; peak memory 6.0 GB, about 0.6 GB for each run kept.

So a real round (4 to 6 scenarios × 100 to 300 runs at county level) would need 250 GB to 1 TB held in memory. flepiMoP writes each slot to parquet and post-processes from disk. The faster path, `ss.parallel(sim, pars=[...])` with one sim built and restarted, used 2.0 GB for 20 runs (`smh2.py`), but on that path outcomes are not kept by group (`msim.results[dis].incidHosp.strata` is `None`; a known issue in `implementation.md`), and outcomes by location are the whole point of a hub submission. `ss.Sim(results=[...])`, which would let the user keep only `incidHosp` and `incidDeath`, is in the spec and not implemented (Fable `05` finding 7, Fable 2 `08`). There is no hub writer.

### 3. Fitting each state to its own data

flepiMoP's own benchmark (`flepimop/src/examples/simple_usa_statelevel`): an SEIR in California and New York with `Ro` scaled by a value per state in two periods ("season" and "sunny"), fitted to weekly sums of cases, hospitalizations, and deaths per state with Poisson likelihoods, ground truth in `parameter_ground_truth.npy`. Its notebook ran emcee with 15 walkers × 100 iterations in 4 min 43 s on 10 cores (from the notebook's saved output; I did not rerun it).

In v4 (`usa/v4_fit.py`), the natural spellings of "a value per state" all fail:

- `ss.modify(Ro, multiply=ss.by(patch=[season_CA, season_NY]), ...)`: `TypeError: float() argument must be ... not 'par'` (`transitions.py:2185`).
- `ss.modify(Ro, multiply=season_CA, where=seir.patch == '06000', ...)`: the same `TypeError` while trying the model on groups (`conditions.py:438`), so `solver='auto'` prints `solver: abm ("Ro_multiply" does not work on groups (TypeError: ...))`, builds 10,000 agents for 60 million people, and crashes in the run.
- `season = ss.par(ss.by(patch=[1.3, 1.3]), bounds=[0.2, 3])`, one fitted value per place: `TypeError: float() argument must be ... not 'by'` at `calibration.py:2172`.

What works is writing the modifier into the rate (4 lines):

```python
isCA, isNY = ss.by(patch=[1, 0]), ss.by(patch=[0, 1])
w_season = ss.timeseries({'2023-08-01': 0, '2023-11-01': 1, '2024-02-02': 0}, mode='constant')
w_sunny  = ss.timeseries({'2023-08-01': 0, '2024-02-01': 1, '2024-04-02': 0}, mode='constant')
factor = (1 + (season_CA*isCA + season_NY*isNY - 1)*w_season) * (1 + (sunny_CA*isCA + sunny_NY*isNY - 1)*w_sunny)
infection = S >> E @ Ro*gamma*factor*I
```

With the ground-truth values, the model reproduces flepiMoP's data: New York's hospitalizations total 882,825 against 882,816, California's 160,607 against 163,969 (2%). Data by state go in as long form with a `patch` column, and are lined up by name, which is as good as the spec promises.

The first fit failed silently. flepiMoP's data are non-integer (its own deterministic output), the Poisson log-likelihood of a non-integer is `-inf`, and NUTS ran anyway: 13 s, R-hat 5.6e15, a bulk ESS of 4, every estimate at its starting value, and a note that two parameters "are almost perfectly correlated in the posterior". No warning: `W1304 NotConverged` is raised only inside `Fit.summarize()` (`calibration.py:1955-1972`), which `calibrate()` does not call. flepiMoP casts the data and the model's values to integers for its Poisson and negative binomial likelihoods (`gempyor/statistics.py`, `astype(int)`), and with `zero_to_one` replaces zeros by ones, so the same data fit there. After rounding the data:

| Parameter | Truth | v4 NUTS (median) | 95% interval | R-hat | ESS |
|---|---|---|---|---|---|
| season, CA | 1.0418 | 1.0409 | 1.0405 to 1.0413 | 1.000 | 973 |
| season, NY | 1.5745 | 1.5801 | 1.5799 to 1.5803 | 1.001 | 2,974 |
| sunny, CA | 1.0808 | 1.0966 | 1.0957 to 1.0974 | 1.002 | 970 |
| sunny, NY | 3.2310 | 3.2795 | 3.2779 to 3.2815 | 1.002 | 1,295 |

4 chains × (500 + 500) draws in 70 s (6.5 s of JAX compiling, 61 s sampling) on 4 cores, against emcee's 283 s on 10 cores for 1,500 evaluations (15 walkers × 100 iterations, a short run for emcee). The estimates are within 1.5% of the truth; the intervals are too narrow to contain it, which is what a Poisson likelihood of counts in the hundreds of thousands does when the model differs from the data-generating one by 2% (California). For 51 states with 2 to 10 modifiers each, the user would have to write 100 to 500 `ss.par` lines and an indicator for each state; I did not test NUTS at that size.

### 4. Seeding

flepiMoP seeds from a file: on a date, move `amount` people in a place from `S` to `E`. The import guide says a movement register does this "(rows with no `from` add people)", which adds people rather than moving them. A register row with `from` = `to` and `states='S'`, `to_states='E'` does move them, and works (`ss.Mobility(data=seed)`). But a place dimension has one `mobility=`, and a list of a commuting matrix and a register raises `TypeError: float() argument must be a string or a real number, not 'Mobility'`. So the standard flepiMoP setup, commuting plus seeding, has no declared form; `deferred.md` lists "seeding a fixed number of people on a schedule" as a gap.

## What would win them over

- **Their config runs, unchanged, and agrees.** `ss.load('config.yml', format='flepimop')` on a 3,000-place project agreed with flepiMoP within 0.007% of every place's peak, with a list of what was not translated. No other framework in the review offers this user a door like that.
- **It is much faster per run.** 5 s against 34 to 58 s at 3,000 places, and 0.1 s against tens of seconds at 300. No seven-second compile on every run. Fifty per-state NPIs cost nothing.
- **Fitting is a different league.** NUTS with gradients through the ODE: 4 parameters to 6 weekly series in 70 s, R-hat 1.00, ESS about 1,000 to 3,000, against about 5 minutes on 10 cores for flepiMoP's short emcee run. Data per location go in as a long table with a `patch` column.
- **The model is readable.** A transition is `S >> E @ Ro*gamma*I` instead of `source`, `destination`, `rate`, `proportional_to`, and `proportion_exponent` in five parallel lists, and a modifier is one line. The import keeps the parameter names they know (`Ro`, `p_incidCase`).
- **Scenarios are first-class.** `sim.imported.scenarios` gives each scenario's modifiers, and `msim.results...select(patch=...).quantiles(...)` gives quantiles by place in one line.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Best point of the tradeoff? | Already in a review? |
|---|---|---|---|---|
| A parameter cannot have a fitted value per place: `ss.par(ss.by(patch=...), bounds=)`, an `ss.par` as a modifier's amount, and `ss.by` of `ss.par`s all raise `TypeError` (`calibration.py:2172`, `conditions.py:438`, `transitions.py:2185`) | Missing feature | Yes, medium: a vector parameter is one fitted value per level (with a shared prior), and a modifier's amount may be a parameter | No: it is the core of flepiMoP inference and the workaround (indicators and windows in the rate) does not scale to 51 states | No (Fable `08` covers products and distribution parameters, not vectors) |
| A Poisson likelihood of non-integer data is `-inf` without a word, and `calibrate()` returns a garbage posterior without `W1304` | Bug | Yes, small: check the data against the likelihood's support at `init_data()` (round with a note, or error), stop on a `-inf` start, and warn from `fit()` itself | n/a | No |
| Every run is kept in memory with all its results by place (0.6 GB a run at 3,000 places); on the restart path outcomes are not kept by place | Missing feature | Yes, medium: `results=[...]` as specified, outcomes by group on the restart path, and writing each run to disk (parquet) as it finishes | No: a hub round is 100 to 300 runs × 4 to 6 scenarios | `results=[...]`: Fable `05` finding 7, Fable 2 `08`; restart-path outcomes: `implementation.md` known issue |
| No draw of uncertain parameters once per run; the importer reads them at their means | Missing feature | Yes, small: `priors=True` on `ss.parallel`; `ss.truncnorm` now exists, so the importer could carry flepiMoP's `distribution: truncnorm` as `prior=` | No | flepiMoP vignette finding 1; Fable `12` |
| No hub output (23 quantiles by scenario, location, target, week) | Missing feature | Yes, small: a `to_hub()` on MultiSim results with a weekly flow sum | n/a | No |
| Seeding plus commuting: a dimension takes one `mobility=`, a list raises a `TypeError`; the guide's "rows with no `from`" adds people instead of moving them | Missing feature, docs | Yes, small: several mobilities per dimension, or a declared seeding event; fix the guide | n/a | `deferred.md`; not in the reviews |
| Multi-dimension compartments (age × vaccination × variant) import as a flat product of states; age contact structure is lost into `proportional_to` sums | Inherent tradeoff (config to model) | The guide says how to rewrite as stratifications by hand | Yes for import; a rewrite is better v4 | No |
| `subpop_setup.selected` is ignored (listed), the imported run is a day short of flepiMoP's output, and modifier `perturbation:` (what inference fits) is not mentioned | Bug (importer) | Yes, small | n/a | No |
| A failure while trying a modifier on groups makes `auto` run 10,000 agents for 60 million people | Bug / design | Yes, small: make it an error | No | Fable `06` finding 4, `11` finding 3 |
| No cluster batch submission, no R post-processing, no SMH conventions built in | Ecosystem | Partly (a batch runner is ordinary code); the R side is a choice | v4 in one process on many cores covers most of what the batch jobs do, given the speed | No |
| YAML is the interface; v4 is Python | Inherent tradeoff | The importer is the bridge; a round-trip writer is ruled out (`deferred.md`, "Won't do") | Yes: Python is more expressive and the importer makes the switch incremental | No |
| `docs/` is all Starsim 3 | Docs | Yes | n/a | Fable 2 `09` finding 4; Fable `12` |

## Verdict

Today: not for the submission pipeline (no fitted value per location, no hub output, memory that grows with every run), but yes as a fast way to run and check their existing configs, and as a place to prototype a new scenario round.

After the four reviews' fixes: closer (`results=[...]`, a clearer MultiSim, a silent fall-back made an error), but the two things this user needs most, a fitted parameter per location and runs that stream to disk, are in no review.

A clear win: vector parameters fitted per place by NUTS, `priors=True` for forward runs, outcomes by place on the fast path with results written per run, and `to_hub()`. Then their own config would go from YAML to fitted, projected, and submitted in one script, several times faster than today.

## The demo that would sell it

`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/flepimop/demo.py`:

```python
import starsim as ss
sim = ss.load('config.yml', format='flepimop')                       # 3,000 counties: places, commuting, outcomes, scenarios
sim.data = ss.data('model_input/hosp_deaths_by_county.csv')          # long form: date, subpop, incidHosp, incidDeath
sim.pars.Ro = ss.par(ss.by(subpop=2.2), bounds=[1, 4])               # one Ro per county, fitted
fit = sim.calibrate()                                                # NUTS, with gradients through the whole run
runs = {name: fit.sim.new(interventions=mods) for name, mods in sim.imported.scenarios.items()}
msim = ss.parallel(**runs, n_runs=200, priors=True)                  # each run a draw from the posterior and the priors
msim.to_hub('2026-10-17-starsim.csv', targets=dict(incidHosp='inc hosp', incidDeath='inc death'), by='subpop', every='week')
```

Eight lines from the config they already have to a hub file, in minutes on a laptop rather than hours on a cluster. It does not run today. What does: the import of the config (exact at 3,000 places), data by place in long form, NUTS on fixed parameters, scenarios from `sim.imported.scenarios`, and quantiles by place. What does not: a fitted value per county (`TypeError`), `priors=` (`KeyNotFoundError: Could not set key priors`), `to_hub` (no such method), and the memory for 200 runs × the scenarios at county scale.
