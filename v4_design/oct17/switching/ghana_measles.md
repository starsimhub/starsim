# Ghana measles: would its users switch?

## Who this user is

A country-team modeler working on measles in Ghana's 260 districts with the Ghana Health Service, WHO, or IDM, in Python. They started from a laser-measles notebook (`ghana-measles/wisdom/LASER-Measles_RDT.ipynb`) and keep a laser-measles agent model of the districts (`ghana-measles/laser-init/GHA_laser_measles/sir_laser_measles_gha.py`, 707 lines with plotting and CLI: `VitalDynamicsProcess` with MCV1 given 274 days after birth, `SIACalendarProcess` with national rounds every three years, `GravityMixing`, per-district MCV1 from an IHME raster, a `StateTracker` by district, choropleth maps), and they use a NumPy rewrite for fitting (`ghana-measles/gm-port`: an 89-line chain-binomial `Sim`, and `fit.py` with a five-term loss against monthly confirmed cases by district, a grid sweep, and Optuna; the calibration notebook takes about 4 minutes). Their weekly work is: refit when a new month of surveillance data arrives; run campaign scenarios (when, which regions, which ages, what coverage) and routine-coverage changes by district; and produce cases averted with a range, by region, as tables and maps for a ministry meeting. They are not framework developers. They picked LASER because IDM supports it and it does spatial agent models, then wrote plain NumPy because it was easier to get right and fast to iterate. They would switch for a tool that lets one model serve both the fitting and the campaign questions.

## The first day

| Measure | Original (`gm-port`) | v4 (`vignettes/ghana_measles/seir_districts/model.py`) |
|---|---|---|
| Lines of model code | 89 | 20 |
| One run, 2010 to 2025 | 0.84 s (README); about 2 s on today's shared machine (`orig_loss.py`: 5 runs in 9.8 s) | 1.37 s (README); 1.9 to 2.4 s today (`base.py`) |
| Peak memory, one run | 0.29 GB | 0.79 GB |
| Agreement | | 3,046,095 infections against 3,041,691; 15,190 reported cases against 15,197; monthly means correlate at 0.999 |

This is the vignette with the best fit of any in the corpus: nothing was missing, and the README says so. The user would recognize every line: seven transitions, an outcome at onset, the districts as `ss.geo(..., lat=, lon=)`, and their own mixing as `ss.Mobility(ss.gravity(c=1.5), kind='visit', away=0.01)`. Two lines would make them pause. Importation is `ss.peryear(0.01/1000) * (S + E + I + R).n / S.n`, a rate per susceptible written to mean a rate per person; and `describe()` (`base.py`) prints all 260 district names on one line of about 7,500 characters, before the disease is described at all.

The docs would not help them. `docs/` is Starsim 3 and has no `ss.geo`, `ss.Mobility`, `solver=`, or `ss.Disease(` (Fable `README.md:49`). The page they need is `skills/import-model/references/laser.md`, which maps the Ghana model line by line; nothing in `docs/` points to it.

## The second week

Scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/ghana_measles/`. The machine was shared with other review agents (load average 7 to 21), so times are slower than the READMEs' for both frameworks.

### 1. Fit to the district case data with their own loss

They would keep their loss, which is their science, and replace the Optuna plumbing (`fit.py`). Getting there took six attempts.

```python
beta_mean = ss.par(ss.perday(14/8), bounds=[1.0, 2.5])
report    = ss.par(0.005, bounds=[0.001, 0.02])
infection = S >> E @ ss.seasonal(mean=beta_mean, amplitude=0.15, peak=59/365) * I
cases     = ss.outcome(report*onset)
...
def gof(sim):
    return gm.losses(data.cases, monthly_by_district(sim), data.districts['population'])['loss']
fit = sim.calibrate(gof=gof, method='optuna', trials=10)
```

The errors, in the order met:

1. `ss.perday(r0/8)` with `r0 = ss.par(14, bounds=[8, 20])` raises `TypeError: Value must be a scalar number or array, not <class 'starsim.transitions.BinOp'>` (`time.py:1671`). Fitting R0 rather than beta is how this user thinks; the workaround is to fit beta. Fable `08_inference.md` finding 5 notes the same error for a parameter inside a distribution.
2. With the parameter in a variable called `beta`, inside `ss.seasonal(mean=beta, ...)`, building the disease raises `E206 DuplicateDeclaration: two members of "disease" are both named "beta"`, with or without `name=`. The same parameter outside `ss.seasonal`, or under another variable name, is fine (`repro_e206j.py`). The message gives no hint; I found the cause by bisecting. Not in the reviews.
3. `ss.Mobility(..., away=ss.par(0.01, bounds=[0.001, 0.1]))` fails at `init()` with a NumPy `TypeError` from `Mobility.time_share()` (`transitions.py:2698`). The share of contacts outside the district (`gravity_k`) is one of the project's main fitted parameters (`fit.py` searches it on a log scale), so this is a real gap. Not in the reviews.
4. `sim.calibrate(gof=fn)` raises `E1302 NoData` unless the sim has data attached (`calibration.py:2138`), though the spec says `gof=fn` is "fn(sim) returns a number to minimize" (`13-inference.md`, "Goodness of fit"). The workaround is to attach the national monthly totals as `data=`, which are then unused.

Then it worked: one run and the loss took 1.9 s, and 10 Optuna trials took 34 s, moving the loss from 0.539 to 0.475 (beta 1.91 per day, so R0 15.3; reporting 0.0082). The v4 loss at the defaults (0.539, one run) is at the low edge of the original's range over five seeds (0.542 to 0.597, mean 0.576), so the translation fits the data about as the original does. The trials ran one at a time, at 3.4 s each; the original's `gm.sweep()` uses `sc.parallelize`. The gain over `fit.py` is real but modest: the bounds live on the parameters and the build function disappears, while the loss and the monthly-by-district reshaping (8 lines) stay.

### 2. A campaign for children in the north, with routine coverage by district

The question a ministry asks: what does a follow-up campaign for children 9 months to 5 years in the five northern regions in November 2026 buy, when routine MCV1 is lower there? (`campaign.py`; the coverage values are made up for the test.)

```python
births     = ss.birth('-> S', rate=ss.by(district=[ss.peryear(0.027*(1 - x)) for x in immune]))
vaccinated = ss.birth('-> R', rate=ss.by(district=[ss.peryear(0.027*x) for x in immune]))
...
kw  = dict(..., solver='binomial', stratify=dict(age=[0, 0.75, 5, 15]))
ppl = ss.People(1)
sia = ss.vaccinate(ss.Vaccine(efficacy=0.85), dates=['2026-11-01'], coverage=0.8,
                   eligible=(ppl.age >= 0.75) & (ppl.age < 5) & measles.S & measles.district.isin(northern))
```

The errors: `ss.peryear(0.027*ss.by(...))` and `ss.peryear(0.027)*ss.by(...)` both raise `TypeError` (`time.py:1671`, `1892`); the spelling that works, a list of typed rates inside `ss.by`, is not in the spec (`04-parameters.md` shows only bare numbers in `ss.by`; `spell.py`). `ppl.district` raises `AttributeError: People object has no attribute "district"`; the condition is on the disease, `measles.district.isin(...)`, which `05-dimensions.md:118` implies only by example. The `ss.People(1)` stand-in for a compartmental condition is the camdl vignette's known wart.

Then it was right: the campaign reached 46,454 children, all in the 0.75-4 age group and all in the five northern regions (`check_sia.py`, results selected by age and by district), and cases in 2027 and 2028 fell from 1,202 to 864 (one run each). For this user that is the headline: ages, district targeting, and a campaign date were three lines added to the vignette, where the NumPy model would need a rewrite with ages and laser-measles a polars schedule with hierarchical IDs.

The cost is in memory. The age groups make 1,040 groups, and the vaccine adds a dimension (held or not), so 2,080. Every result is then kept for every group at every daily step, including the cumulative ones (23 results of 2,080 values a step: 0.38 MB a step; `campaign_mem.py`), and a run to 2028 grows at about 0.69 MB a step, to about 4.9 GB (measured at 2.35 GB for 8 years). Two scenarios in `ss.parallel` peaked at 9.9 GB, because the parent process receives both sims; that run exceeded this review's memory limit by accident. A run took about a minute, against 2 s without ages (a 2-year run took 5.9 s; `prof_sia.py`). This user's laptop would have swapped. Two other things a measles modeler would catch: `ss.Vaccine` is leaky only (its efficacy multiplies `rel_sus`; `products.py:229`), while measles vaccine is usually modeled as all-or-nothing, as laser-measles does with `mcv1_efficacy` (the workaround is `efficacy=1` and coverage times 0.85); and `eligible` had to include `measles.S` to stop the campaign spending doses on the immune, which a real campaign does.

### 3. Twenty runs per scenario, and cases averted with a range

Without ages, the campaign as a share of susceptible people in the northern districts (`sweep.py`), 20 runs each through `sim.new(...).run(n_runs=20)`, one kernel call per scenario:

| | 20 runs | Peak memory | Cases 2027 to 2028, median (5% to 95%) |
|---|---|---|---|
| Baseline | 34 to 63 s | 1.3 GB | 2,060 (1,311 to 2,466) |
| Campaign | 41 to 49 s | 2.5 GB | 1,774 (1,066 to 2,134) |
| Averted (paired by run) | | | 276 (133 to 483) |

This is the output the meeting needs, and it is short to write. Two traps cost time: `msim.sims` holds one sim, not 20, so the first version silently gave a range of zero (the runs are only in `msim.results`, with a run axis; Fable 2 `08` finding 2); and `msim.results` has no `timevec`, so the dates come from `msim.base_sim`. The original's `sc.parallelize` over seeds is about as fast.

### 4. The laser-measles model: MCV1 at 9 months and a campaign, as agents

The model they also keep in laser-measles, written once and run both ways (`ri.py`): ages from a pyramid, births into `S`, `ss.vaccinate(mcv, coverage=0.833, start='2010-01-01', eligible=(ppl.age >= 0.75) & (ppl.age < 1))` for routine doses, and a national campaign for ages 0.75 to 5 in June 2012.

- **As compartments** (`solver='binomial'`, ages `[0, 0.75, 1, 5, 15]`): 5,200 groups (district × age × `has_mcv` × `offered_routine`, since routine delivery adds a dimension of its own), 48 ms a step, 70 s and 3.2 GB for 4 years.
- **As agents** (2.5M agents, 1 in 10): 375 ms a step, 150 ns per agent-step, 1.5 GB, for 90 days. Sixteen years would take about 37 minutes, and the full 25M about 6 hours and 8 to 10 GB (estimated, not run). The interventions are most of the extra cost over a plain model (see the LASER report, section 4). laser-measles is not installed here, so I could not time the original; laser-generic ran 30M agents for a year in 11 s.

One trap is in the spec itself. `08-interventions.md:91` gives `ss.vaccinate(bcg, coverage=0.9, start=2021, eligible=ppl.age >= 0.75)` as "routine, at 9 months". Run as agents (`routine_check.py`), it vaccinates 8,757 of 10,000 people on the first day: routine delivery offers the product to everyone the first time they are eligible, and everyone over 9 months is eligible at the start. For a measles model that is a silent catch-up campaign of the whole population, which no one would spot from the results. laser-measles gives the dose to newborns after a delay (`routine_immunization_delay=274`), which is what this user means. The upper age bound is the workaround; a delivery keyword such as `at_age=ss.months(9)`, or routine delivery that skips people already past the window at the start, is the fix. Not in the reviews (Fable 2 `07` finding 9 notes the cost of the `offered_` dimension, not this).

## What would win them over

- **One model for both questions.** The 20-line vignette plus three lines gave ages, routine coverage by district, and a targeted campaign, checked to reach exactly the right children; the same text runs as agents; and `ss.geo(data.districts)` takes their dataframe as it is. Today this user keeps a NumPy model for fitting and a 707-line laser-measles model for campaigns, and the two do not agree by construction.
- **Fitting with their own loss and no plumbing.** `ss.par(..., bounds=...)` on the parameters and `sim.calibrate(gof=gof, method='optuna')` replaced the parameter-passing half of `fit.py`; the fit is consistent with the original's loss at the defaults.
- **Scenarios with uncertainty in a few lines.** `run(n_runs=20)` for each scenario gave cases averted with a 5% to 95% range in under a minute each.
- **Speed for the compartmental model is fine.** 2 s a run for 16 years over 260 districts is as fast as their NumPy loop, and fast enough for Optuna.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff, and if not, where | Already in a review? |
|---|---|---|---|---|
| Results kept for every group at every step (and cumulative ones too): about 5 GB for one age-structured run with a campaign, 9.9 GB for two in `ss.parallel`; 0.79 GB for the plain model against 0.29 GB | Performance | Yes: keep groups at a reporting interval (monthly, as the data are), drop `cum_*` by group, return only what the parent needs from `ss.parallel` (days) | No: every daily value of every group is more than anyone asked for | No |
| Each product and each routine delivery doubles the groups (260 → 1,040 → 2,080 → 5,200) | Inherent tradeoff (compartments) | Partly: a campaign that moves `S` to `R` needs no dimension | Not quite: for an all-or-nothing measles vaccine a move between states is exact and costs nothing; the dimension is needed only for leaky or waning products | Fable 2 `07` finding 9 (cost of `offered_` and `pending_` dimensions) |
| The spec's "routine, at 9 months" vaccinates everyone already older at the start | Bug (spec and behavior) | Yes: `at_age=` or skip people past the window at the start (hours) | No | No |
| No all-or-nothing vaccine (`ss.Vaccine` is leaky) | Missing feature | Yes (hours) | No: measles needs both | No |
| `away=` cannot be a fitted `ss.par`; `ss.perday(par/8)` fails; a `ss.par` in `ss.seasonal` under the name `beta` raises `E206` | Bugs | Yes (hours each) | | Partly: Fable `08` finding 5 (parameter inside a distribution) |
| `calibrate(gof=fn)` still requires `data=` | Bug | Yes (minutes) | | No |
| Optuna trials run one at a time | Missing feature | Yes: `n_jobs` or `ss.parallel` inside (a day) | No: the original sweeps in parallel | No |
| As agents at full size, about 6 hours and 8 to 10 GB for 16 years | Performance | Mostly (see the LASER report) | No | See the LASER report |
| Rates by district are spelled `ss.by(district=[ss.peryear(x) for x in ...])`; conditions on a district are on the disease, not the people | Docs and API | Yes | `ss.peryear(array_by_district)` with `ss.by` would be the obvious spelling | No |
| `docs/` is Starsim 3; `describe()` prints 260 names on one line | Docs | Yes | | Fable `README.md:49` for the docs |
| No maps, no raster-to-district coverage, no `laser-init` data pipeline | Ecosystem | Partly: a `plot_map(shapes)` for results by place (days) | | No |

## Verdict

- **Today:** not yet. The model reads better and the campaign is three lines, but the first fit took four workarounds, the age-structured campaign needs about 5 GB a run, and the routine dose given as the spec shows is wrong for measles.
- **After the four reviews' fixes:** close. The reviews' docs and API fixes help, but the memory, the routine-at-start trap, the all-or-nothing vaccine, and fitting `away=` are not in them; with those four (each days or less) this user would switch for the compartmental work.
- **What would make it a clear win:** fitting, scenarios, and maps by district from one model at under 1 GB a run, and the same text as agents at full size in minutes, so the laser-measles model can be retired.

## The demo that would sell it

```python
import starsim as ss
import ghana_measles as gm                                          # their data loaders and loss
data = gm.load_data()
S, E, I, R = ss.states(4)
beta = ss.par(ss.perday(1.75), bounds=[1, 2.5])
away = ss.par(0.01, bounds=[0.001, 0.1])                            # fails today
measles = ss.Disease(
    infection   = S >> E @ ss.seasonal(mean=beta, amplitude=0.15, peak=59/365) * I,
    importation = S >> E @ ss.peryear(1e-5),                        # per person: needs an inflow form
    onset       = E >> I @ ss.days(6),
    recovery    = I >> R @ ss.days(8),
    births      = ss.birth('-> S', rate=ss.peryear(0.027)),
    deaths      = ss.death(rate=ss.peryear(0.007)),
    cases       = ss.outcome(ss.par(0.005, bounds=[0.001, 0.02])*onset),
    init        = dict(R=1 - 1/(8*beta)),
)
measles.stratify(district=ss.geo(data.districts), mobility=ss.Mobility(ss.gravity(c=1.5), kind='visit', away=away))
mcv  = ss.Vaccine(efficacy=0.85, leaky=False)                       # all-or-nothing: missing today
ri   = ss.vaccinate(mcv, coverage=ss.by(district=data.mcv1), at_age=ss.months(9))   # at_age missing today
sim  = ss.Sim(measles, interventions=ri, stratify=dict(age=[0, 0.75, 5, 15]), start=2010, stop=2028, solver='binomial')

fit = sim.calibrate(gof=lambda s: gm.losses(data.cases, s.results.measles.cases.monthly('district'), data.districts.population).loss, method='optuna', trials=200)
northern = data.districts.region.isin(['northern', 'north east', 'savannah', 'upper east', 'upper west'])
sia  = ss.vaccinate(mcv, dates=['2026-11-01'], coverage=0.8, eligible=(sim.people.age >= 0.75) & (sim.people.age < 5) & measles.district.isin(northern))
runs = ss.parallel(base=fit.sim, sia=fit.sim.new(interventions=[ri, sia]), n_runs=50)
runs.averted('cases', start=2027).plot_map(gm.load_shapes())        # cases averted by district, with a range
```

What it needs: fitting `away`, `init` and `seasonal` from one parameter, `leaky=False` on `ss.Vaccine`, `at_age=`, monthly results by district, results kept at a reporting interval so that 100 age-structured runs fit in memory, a paired "averted" between scenarios, and a map of results by place. The pieces that run today are in `fit.py`, `campaign.py`, and `sweep.py`: the fit with two parameters, the targeted campaign, and the 20-run comparison, each with the workarounds described above. The whole script does not run today.
