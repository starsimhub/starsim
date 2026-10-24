# epydemix: would its users switch?

## Who this user is

A modeler at the ISI Foundation, Northeastern's MOBS lab, or a public health group in their orbit (the US and European Scenario Modeling Hubs, the Epistorm network) who writes Python notebooks. Their model is an age-structured compartmental SEIR-type model, written transition by transition (`add_transition(source, target, params, kind)`), with transmission scaled by a contact reduction from mobility data and by season, and run as a chain binomial in daily steps. What they use every week: `load_epydemix_population("United_States__Massachusetts")`, which gives the age structure and the four-setting contact matrices for about 400 places, US states included, in one call; `add_intervention(layer_name="school", ...)` on one setting; ABC (rejection or SMC) with `wmape` as the distance, which needs no likelihood; projections from the posterior (`run_projections`) and quantile DataFrames (`get_quantiles_compartments()`) for hub submissions. They chose epydemix (GPL-3.0) because it is quick and accessible: a predefined SIR in Kenya is seven lines, the data come with it, a posterior comes without writing a likelihood, and their students can use it.

## The first day

| Vignette | Lines (epydemix, Starsim) | Time (epydemix, Starsim) | Agreement |
|---|---|---|---|
| `sir_kenya` (predefined SIR, Kenya's population) | 7, against 10 (data loading not counted in either) | One run 0.006 s, against 0.035 s (0.0008 s restarted); 100 runs 0.56 s, against 0.07 s | Peaks within 0.3%, a day apart; final size 99.63% both |
| `covid19` (SEIR with an Erlang delay to death, mobility, season, ABC-SMC, three reopening scenarios) | 161, against 37 | One run 0.037 s, against 0.060 s (0.0021 s restarted); calibration 87 s (ABC-SMC, 200 particles × 5 generations), against 25 s (1,000 Optuna trials, a point estimate) | At fixed values, weekly deaths within 0.9% of the peak and 10,534 deaths by 2 August against 10,525, but only with `substeps=1`; without it, 6,039 |

This user would recognize both models, and the COVID one would impress them: `ss.Transition(I, [D, R], rate=mu, p=ss.by(age=IFR))` and `stages=4` say what their nine compartments and eight `add_transition` calls mean. Three things would worry them. First, the 6,039 against 10,525: a binomial solver that silently substeps the whole model for one Erlang delay changes their model (now a `W102` note, and `substeps=1` restores it; vignette finding 1). Second, the vignette reads Massachusetts from a JSON file that epydemix wrote, because v4 has no US states. Third, the calibration is a point estimate, and its projections differ from the original's posterior mean by a factor of 2.5 (897 against 2,217 weekly deaths on 2 August in the high scenario), which the vignette rightly calls exactly what a posterior is for.

Two of the vignette READMEs' findings are stale, in the user's favor: `ss.contacts('Kenya')` no longer raises `E605` (it is bundled), and `msim.results.sir.n_I.select(age='0-4').quantiles()` now works on many runs. The `docs/` folder is still all Starsim 3, which tells this user that Starsim is an agent-based framework with `ss.RandomNet` and `ss.MixingPools`; nothing public mentions `ss.contacts`, `ss.people`, `solver='binomial'`, or `sim.calibrate()`. Opus already says the docs must be ported (`opus_review/README.md:38`). For this user it is decisive: epydemix's selling point is its tutorials, and v4 has none they can find.

## The second week

All scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/epydemix/`.

### 1. Kenya again, from v4's own data

The tutorial without epydemix's JSON (`t1_kenya_named.py`, `t1_kenya_compare.py`):

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.04),
    recovery  = ss.progression('I -> R', rate=0.1),
    init      = dict(I=10),
)
sir.stratify(age=[0, 5, 20, 50, 65])
sim = ss.Sim(sir, networks=ss.contacts('Kenya'), people=ss.people('Kenya', year=2020), start='2019-12-01', stop='2020-04-01', solver='binomial', use_aging=False)
msim = sim.run(n_runs=100)
```

Eight lines, one over 100 characters, against epydemix's seven: the north-star test is now met to within a line (Opus surface 12 counted 10 against 7 with the JSON). 100 runs take 0.23 s; the mean peaks at 54.48% of the population on day 39, against epydemix's 54.46% on day 40; final size 99.59% against 99.63%. The population is the UN's (52.2 million in 2020) where epydemix's is 55.9 million; with epydemix's population and v4's matrix the peak is 54.32%, so the bundled Prem 2021 matrix and epydemix's agree for this model.

Three things this user would trip on:

- **Without `people=`, the ages are a silent guess.** `ss.contacts('Kenya')` with `total_pop=55884494` and no `ss.people` gives "ages uniform from 0 to 100" (35% over 65, against 3% in Kenya) and aging on; the peak falls to 49.3%, a 9.5% error, marked only as `[guessed]` in `describe()`. epydemix's one call gives both. Opus found this (README:92, dims 5); from this seat the fix is that `ss.contacts(country)` should bring that country's age split unless one is given, as Opus proposes (dims, line 116).
- **`use_aging=False` has to be given**, or a four-month epidemic ages people between groups. epydemix never ages people.
- **No subnational data.** `ss.contacts('Massachusetts')` and `ss.people('Massachusetts')` raise `E605 ... did you mean: Kazakhstan, Maldives, Malaysia?`. `starsim_data` has 177 countries for `prem2021`, 152 for `prem2017`, and 36 for `mistry2021`, and no US states, Canadian provinces, or other subnational units, which epydemix-data has (Mistry 2021 is built for US states). For a Scenario Modeling Hub team, US states are the whole job. The error messages for a source that lacks a country are good (`"mistry2021" has no contact matrix for Kenya; it is in prem2021, prem2017`).

The importer is the other way in, and it works well. `dump_epydemix.py sir_kenya.py kenya.json`, run in epydemix's own environment on the tutorial as it is, takes 4.2 s (it downloads the population); `ss.load('kenya.json', format='epydemix')` and 100 runs take 0.26 s and give a peak of 54.41% on day 40 and a final size of 99.63%, the same as epydemix. `sim.to_code()` writes 33 lines a person could keep (`import/load.py`).

### 2. A posterior and projections from it

The thing they do weekly: fit `R0`, `Delta`, `I0`, and the seasonal minimum to weekly deaths, then project the reopening scenarios from the posterior. v4 has no ABC (`implementation.md:231`, `:283`); with a likelihood on every series, a stochastic compartmental sim gets particle MCMC. So this user adds a likelihood (`t2_pmcmc.py`):

```python
fit = sim.new(stop='2020-05-31').calibrate(likelihood=dict(deaths=ss.nbinom(n=10)), draws=100, warmup=100)
```

- **On the vignette's model, the posterior is wrong, silently.** 4 chains of 100 + 100 draws in 49 s go to `R0` 1.54 (1.50 to 1.62, the lower bound) and `Delta` 8.8 (at its lower bound of 7), with R-hat 1.8 to 2.8; the negative log-likelihood of the "after" sim is 3,249 against 103 before. The warnings fire (`W1304 NotConverged`, `W1301 AtBound`), but they point at warm-up and identifiability, not at the cause.
- **The cause is a bug in the particle filter: it does not run the same model as the sim when the transmission rate is a seasonal factor times a time series.** The filter's log-likelihood at the starting values is −182.8 where single runs of the sim give −103 to −108, and at the chain's endpoint −111.9 where the sim gives −2,546 to −3,563 (`t2_pf_check.py`). Remove either factor and the two agree: without the mobility series `r`, −174.3 against −174 to −179; without the season, −106.5 against −108 to −113 (`t2_pf_check2.py`). A 25-line repro on a plain SIR with `beta = b*season*r` and data drawn at `b = 0.5` puts the filter's best at `b = 0.4` (−11,349, against −17,016 at 0.5), while runs of the sim put it at 0.5 (−375); with `b*season` or `b*r` alone, the filter peaks at 0.5 (`t2_pf_repro.py`). `sim.restart(pars=...).run()` is correct, so the fault is in what `ParticleFilter.filter()` hands `ss.kernels.advance()` (`calibration.py:1564`), not in restarting. Fable 08 notes that the particle MCMC test is skipped (`tests/test_calibration.py:75`), so the default method for this user's class of model is not tested at all.
- **Without the season, it works, more slowly than ABC-SMC.** On the same model with `beta = R0*mu/rho(C)*r` (`covid_model_noseason.py`), 4 chains of 400 + 400 draws take 265 s and give `R0` 2.44 (2.24 to 2.72), `Delta` 25.2 days (19.8 to 26.9), `I0` 1,504 (672 to 2,720), R-hat 1.05 to 1.07 and a bulk ESS of about 70 (not yet converged by the 1.01 rule); the log-likelihood improves from −108 to −84. The tutorial's ABC-SMC with 200 particles takes 87 s; the default here (1,000 + 1,000 draws) would take about 11 minutes. This is not the tutorial's model, so the numbers cannot be compared with epydemix's posterior.
- **Projections from the posterior are a loop the user writes.** There is no `fit.project()` (epydemix's `run_projections`): 100 posterior draws × 3 scenarios as `sim.new(pars=row, interventions=iv)` take 61 s (`t2_project.py`). Each sim is built again, and `ss.modify` takes runs out of the kernel (vignette finding 8).
- **`method='abc'`** reaches SciPy and fails with `ValueError: Unknown solver abc`, uncoded; it should say that there is no ABC, and name `pmcmc` (with a likelihood) and `optuna`.

The decision to drop ABC-SMC is already questioned by Fable2 08 ("departure I would revisit") and Fable 08 and Codex (`evidence/interfaces.txt:60`) ask that the spec's method table match the code. From this seat the argument is sharper: this user's whole workflow is "a distance, no likelihood, a posterior", and v4 offers either a point estimate or a likelihood they did not want to write, by a method that is broken on their flagship model.

### 3. Close schools

epydemix's `add_intervention(layer_name="school", start_date=..., end_date=..., reduction_factor=0.2)` (`t3_school2.py`): four layers as four `ss.contacts('Kenya', layer=...)`, and a modifier.

- The spec's spelling, `ss.modify(school.pars.n_contacts, multiply=0.2, start=..., stop=...)` (`06-routes.md:270`, `08-interventions.md:28`), raises `E504 UnmodifiableTarget ... not [[ss.constant(v=0.4197...` followed by the 16 × 16 matrix printed in full. Fable 03 and Fable2 07 found this.
- The string path, `ss.modify('school.pars.n_contacts', ...)`, works: closing schools from 20 December to 1 February moves the peak from 28.6 million on day 39 to 27.5 million on day 41. The four layers and the closure are two lines (a list of `ss.contacts` and the modifier), against epydemix's one, since its population brings the layers.

### 4. Vaccinate the over-50s

What epydemix makes them write as extra compartments (`t4_vaccine.py`, `t4_cond.py`):

```python
ppl = ss.People(1)
vax = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=200_000, start='2019-12-10', eligible=ppl.age >= 50)
```

This works as compartments in 0.33 s: infections among the over-50s fall from 5.04 million to 1.72 million. It is the clearest single gain over epydemix in this review: the disease model does not change.

But the spelling a user would reach for next, `eligible=sir.age >= 50` (the dimension on the disease), **silently vaccinates nobody**, and `eligible=sir.age < 5` vaccinates all 52.2 million people instead of the 6.9 million under 5. The comparison is made against the age group's index (0 to 4), not the age: `sir.age < 3` selects the first three groups (47.1 million). The same happens as agents (`t4_cond_abm.py`). `sir.age == '0-4'` is correct. Fable2 07 says "`sir.age < 15` already works once `age` is a stratification (verified)", which is this bug looking like success: with groups `[0, 15, 65]`, every index is below 15. The fix is small: compare a binned dimension numerically against its bin edges, and raise when a threshold is not an edge (as the known issue about functions of ages already proposes for the lazy form). Fable2's README (change 6) asks for a note when a delivery finds nobody eligible on every step; that note would have caught the `>= 50` case, but not the `< 5` one.

### 5. Hub output

`msim.results.sir.n_I.select(age='0-4').quantiles((0.05, 0.5, 0.95))` gives a 3 × 123 array. There is no DataFrame of quantiles over runs (epydemix's `get_quantiles_compartments()`, the form a hub submission is made from): `MultiSim` has no `to_df()`, and `msim.results.to_df()` on a kernel `n_runs` result fails with `AttributeError: 'NoneType' object has no attribute 'shape'`. Small to add; this user would write it on day one.

## What would win them over

- **Vaccination, testing, and treatment without touching the disease model**, as compartments (task 4). In epydemix every product is a copy of the compartments written by hand.
- **The model reads like the equations and is a third of the length** (37 lines against 161 for the COVID tutorial), with Erlang delays (`stages=4`), branches with age-specific probabilities (`p=ss.by(age=IFR)`), seasonality (`ss.seasonal`), and time series (`ss.timeseries`) as words rather than loops.
- **Solvers they do not have**: the same model as an ODE, tau-leaping, or Gillespie, and as agents, where epydemix is chain-binomial only. For small populations or late-epidemic extinction, Gillespie matters.
- **Speed for many runs**: 100 Kenya runs in 0.07 to 0.23 s against 0.56 s; a restarted run in 0.8 ms.
- **The importer**: their own scripts dump in seconds and run identically (task 1), with `to_code()` as a starting point.
- **Posteriors with a likelihood**, by particle MCMC (when it works) and by NUTS for an ODE, with R-hat, ESS, and identifiability notes (`W1302`), which epydemix's ABC does not give.
- **Bundled data with no download** (`ss.contacts`, `ss.people`, `ss.geo` for 184 countries, recorded in the model spec and hash). epydemix fetches from GitHub on every run and its tutorial's own population name broke between releases (`examples/epydemix/covid19/README.md`, "Changes needed to run it").

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff, and if not, where | Already in a review? |
|---|---|---|---|---|
| Particle filter wrong when `beta` is a seasonal factor times a time series; pMCMC goes to the bounds with no hint of why | Bug | Yes, small to locate (the repro isolates it), and add a test that compares the filter with runs of the sim | | No; Fable 08 notes the test is skipped |
| No ABC; without a likelihood, a point estimate | Missing feature (a deliberate departure) | Yes, medium: ABC-SMC over the kernels' `n_runs` is a few hundred lines, with `wmape` already the default distance | No. A point estimate for a stochastic model without a likelihood is the wrong default for this user; the better point is ABC-SMC as the method for "stochastic, no likelihood" (as `13-inference.md:158` still says), with pMCMC when a likelihood is given | Fable2 08, Fable 08, Codex interfaces |
| No projections from a posterior (`run_projections`) and no quantile DataFrame | Missing feature | Yes, small: `fit.project(scenarios, n=)` over `fit.samples`, and `msim.to_df(quantiles=)` | | No |
| No subnational populations or contact matrices (US states) | Missing data | Yes, medium: adapt epydemix-data's Mistry 2021 US states once into `starsim_data`, with provenance (`04-parameters.md:199` already plans this) | | No |
| `ss.contacts(country)` without `ss.people` gives ages uniform from 0 to 100 (peak 9.5% low in Kenya) | Bug (a bad guess) | Yes, small | No: the named matrix should bring its country's ages | Opus README:92, dims 5 |
| `sir.age >= 50` and `sir.age < 5` compare the group index, silently | Bug | Yes, small | | No; Fable2 07 reports it as working |
| `ss.modify(school.pars.n_contacts, ...)` raises `E504`; only the string path works | Bug | Yes, small | | Fable 03, Fable2 07 |
| `substeps=` silently changes a chain-binomial model with an Erlang delay | Inherent tradeoff (ODE-faithful against discrete-time-faithful) | Already noted (`W102`) | Not quite: substepping only the staged transition, or whole steps by default for `binomial`, would leave a discrete-time model as written | Fable2 05 (asks the same question) |
| Matrix aggregation weighted by band width, not population | Bug | Yes, small | | Codex F03, Opus dims 14 |
| `import starsim` takes 2.1 s against 0.41 s; kernels build in about 12 s the first time | Inherent, partly | Partly | Acceptable for scripts; noticeable in a classroom notebook | No |
| `docs/` is Starsim 3; no tutorial uses `ss.contacts` or `solver='binomial'` | Docs | Yes, large | | Opus README:38 |
| Ecosystem: the Scenario Modeling Hub pipelines, the epistorm community, MOBS's GLEAM lineage, the CLI and `AGENT.md` | Ecosystem | Slowly | | No |

## Verdict

- **Today:** no. The model language and interventions are clearly better, but this user's week is "data by place name, fit without a likelihood, project from the posterior", and v4 has no US states, no ABC, a broken posterior on their flagship model, and no projection helper.
- **After the four reviews' fixes:** closer but still no; the reviews fix the age guess, the school closure, and the matrix weighting, and question the ABC departure, but none covers the particle filter bug, the `sir.age` comparison, projections, quantile DataFrames, or subnational data.
- **A clear win needs:** ABC-SMC (or a working pMCMC with a default count likelihood they need not think about), `fit.project()`, US states in `starsim_data`, and the two silent bugs fixed; with those, v4 does everything epydemix does in fewer lines, adds products and solvers it lacks, and runs faster.

## The demo that would sell it

Massachusetts by name, by age, with layers, a school closure and a vaccine, fitted with uncertainty and projected under scenarios, in under 20 lines:

```python
import starsim as ss
R0 = ss.par(2.2, bounds=[1.5, 6]); I0 = ss.par(4000, bounds=[10, 9999])
covid = ss.Disease(
    infection = ss.transmission('S -> E', R0=R0, source='I'),
    onset     = ss.progression('E -> I', dur=ss.days(4)),
    outcome   = ss.Transition('I', ['D', 'R'], rate=ss.perday(0.4), p=ss.by(age=IFR)),
    deaths    = ss.Transition('D', 'Dead', dur=ss.days(20), stages=4),
    init      = dict(E=I0/2, I=I0/2),
)
layers = ss.contacts('United States:Massachusetts', layers=True)       # Population and matrices by name
sim = ss.Sim(covid, networks=layers, data='MA_deaths.csv', solver='binomial', start='2020-02-23', stop='2020-08-02')
fit = sim.new(stop='2020-05-31').calibrate(method='abc-smc')           # A posterior, no likelihood needed
scenarios = dict(
    persistence = None,
    schools     = ss.modify(layers.school, multiply=0.2, start='2020-06-01'),
    vaccine     = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=20_000, start='2020-06-01', eligible=covid.age >= 60),
)
proj = fit.project(scenarios, n=200)                                    # One run per posterior draw
proj.plot(); proj.to_df(quantiles=[0.05, 0.5, 0.95]).to_csv('hub.csv')
```

It does not run today: Massachusetts is not in `starsim_data`, `transmission(R0=)` and `layers=True` do not exist (the vignette writes `R0*mu/max(eigvals(C))` and four `ss.contacts` calls), there is no `method='abc-smc'`, `fit.project`, or `to_df(quantiles=)`, `covid.age >= 60` vaccinates nobody, and `ss.modify(layers.school, ...)` is `E504`. A version with Kenya, synthetic weekly infections, a likelihood, particle MCMC (100 + 100 draws), the string-path school closure, and a hand-written projection loop does run, in 38 s, and recovers the true `beta` (0.0202, 95% from 0.0197 to 0.0210, against 0.02), with closed schools halving infections (median 0.91 million against 1.99 million) (`demo_today.py`).
