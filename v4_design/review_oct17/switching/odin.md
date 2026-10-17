# odin: would its users switch?

## Who this user is

An odin user works in R, usually at or trained by the MRC Centre for Global Infectious Disease Analysis at Imperial (the odin, dust, mcstate, and monty packages come from Rich FitzJohn's team there), or at UKHSA and the groups that took up its tools. They write compartmental models as equations in odin's DSL: `update()` for a discrete-time stochastic model with `rbinom()` draws, `deriv()` for an ODE, arrays with `dim()` for age, space, and vaccine strata, `interpolate()` for time-varying inputs, and in odin2 a `data()` block with `cases ~ NegativeBinomial(...)` right in the model. The model is generated as C++, run as hundreds or thousands of particles on many threads by dust/dust2, and fitted with a bootstrap particle filter inside particle MCMC (mcstate, now monty), with adjoint gradients for deterministic fits in odin2. The canonical use is real-time analysis: COVID-19 in England (sircovid/spimalot, fitted weekly for SPI-M), squire/nimue for COVID in low- and middle-income countries, and later outbreak work. They picked odin for three things: they can write any difference equation they like, the generated code is fast enough to run a particle filter with 1,000+ particles over a year of daily data in minutes, and it all lives in R, next to their data cleaning, ggplot, and reports. The LFD environment has the classic odin 1.2.7 (`_envs/odin/README.md`), not odin2, dust2, or monty, so everything I say below about those is from my own knowledge and was not run.

## The first day

| Vignette | Lines (odin → v4) | Time | Agreement |
|---|---|---|---|
| `seir_array` (5 age groups, births, deaths, aging, waning, a 14-day `delay()`) | 53 → 17 | 1.16 s as a command, of which 2 ms is the run, against 2.2 s, of which 1.9 s is the run | Totals within 1.6% of peak, with a hand-written workaround for the delay's history; the youngest group differs by up to 15% in `E` |
| `stochastic_sir_arrays` (100 replicates in an array) | 27 → 7 | 0.002 s for 100 replicates (plus 0.7 s to compile), against 0.04 s | Mean of 2,000 runs within 4.6 people of 1,001 (I reran `compare.py`: 4.6, 1.0, and 4.2 for `S`, `I`, `R`; 45% of runs die out) |

These are numbers from `vignettes/odin/*/README.md`, and the stochastic one I reran. An odin user would recognize both translations. `S >> E @ ss.perday(1) * I` plus `seir.stratify(age=[0, 1, 5, 15, 30], split=split)` is a fair reading of eight `deriv()` equations with hand-written aging terms. The cuts in line count are real and come from things odin makes you write out: `dim()` for every array, the aging terms in every equation, and the binomial draws. The user would also see the two places where the translation strains. A fixed 14-day latent period becomes 200 Erlang stages, which are stiff: the run is 1,000 times slower than odin's `dede` and still 1.6% off (finding 2 of the vignette). The delay's history, where odin backfills the first 14 days, needs an extra transition (finding 1). Neither is a deal-breaker, but an odin user who uses `delay()` (for reporting lags, not just latency) would want the conveyor the vignette proposes.

The first thing they would meet beyond the vignettes is the docs. `docs/` in ssv4 is all Starsim 3: no page uses `ss.states`, `>>`, or `ss.outcome`, and `docs/user_guide/workflows_calibration.qmd` teaches the v3 `ss.Calibration` with `build_fn`. Particle MCMC is not mentioned anywhere in `docs/`. An odin user whose whole reason for being there is "fit a stochastic model with a particle filter" would conclude from the docs that Starsim cannot do it, and would only find `method='pmcmc'` in the docstring of `ss.fit` (`starsim/calibration.py:2109`).

## The second week

All scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/odin/`. The model for the tasks is what an odin user writes every week, and is more than either vignette: a discrete-time stochastic SEIR in four age groups with a contact matrix, seasonal forcing, a quarter-day step, and weekly incidence (`age_seir.R`, 46 lines). I ran it in odin 1.2.7 with `run_age.R` as the reference.

### 1. Bring my existing odin model in

The first thing this user tries is `ss.load('age_seir.R', format='odin')` (`imp2.py`), since the importer is advertised. It failed three times before giving up:

- `lambda[] <- beta * seas * sum(s_ij[i, ])`, the most common name for a force of infection, crashes the importer with a Python `SyntaxError` (`starsim/interop/odin.py:72`, which parses R expressions with `ast.parse`, and `lambda` is a Python keyword). Renamed to `foi`:
- `update(cases) <- if (step %% steps_per_week == 0) sum(n_SE) else cases + sum(n_SE)`, the standard odin idiom for weekly incidence, crashes the same way (`odin.py:284`). Removed:
- The import then "succeeds" with `W1601 ImportIncomplete` listing 5 items: the arrays "with no flows between levels" (the contact-matrix coupling), the `time` variable, and `S -> E` "uses s_ij__index". The sim that comes back runs: `time` has become a state holding a million people, and there is no transmission at all (`n_E` is 0 throughout).

The two vignette files import cleanly in under 0.01 s (`imp1.py`), so the importer works for the shapes it was built from, but not yet for the model on this user's desk. The first two crashes are small fixes (rename Python keywords; translate R's `if` expression to `a if c else b`). The contact matrix is the real gap: "mixing within a level" is listed as not translated in `v4_design/admin/implementation.md:244`, and it is in nearly every odin model.

### 2. Write the same model by hand in v4

From the overview page and the camdl vignette, `age_v4.py` took about ten minutes and is 10 lines of model against odin's 46 (plus 5 driver lines in R for `S0`, `I0`, and `m`):

```python
S, E, I, R = ss.states(4)
infection = S >> E @ ss.seasonal(mean=ss.perday(0.012), amplitude=0.3, peak=0) * I
onset     = E >> I @ ss.perday(0.2)
recovery  = I >> R @ ss.perday(0.1)
cases     = ss.outcome(infection, every=ss.weeks(1))
seir = ss.Disease(infection, onset, recovery, cases, init=dict(I=ss.by(age=[10, 0, 0, 0])))
seir.stratify(age=[0, 5, 18, 65], split=pop/pop.sum())
sim = ss.Sim(seir, networks=ss.contacts(m, age=[0, 5, 18, 65]), total_pop=pop.sum(), unit='day', dt=0.25,
             stop=365, solver='binomial', use_aging=False)
```

Two things a user has to know. First, the keyword form `ss.Disease(infection=..., cases=ss.outcome(infection, ...))` raises `NameError`, since a keyword cannot refer to its neighbor, so a model with an outcome has to be written in the variable form. Second, `use_aging=False` is needed: an age stratification makes people age by default, and odin's arrays do not. This matches opus [dims 6](../opus_review/notes/dimensions-and-mixing.md).

The answer is right. Over 1,000 runs, the mean of `I` peaks at 35,534 on day 166.75 (odin: 35,777 on day 166), and the mean final `R` is 533,016 (odin: 532,176), so the two agree within 0.7% and 0.2%.

Speed is where it falls behind. odin 1.2.7 runs the 1,000 replicates (1,460 steps each) in 1.98 s on one thread, plus 0.85 s to compile. v4 takes 11.2 s (`chk2.py`), or 1.6 s for 100. dust2 would run the same 1,000 particles on many threads; I could not measure that, but it can only widen the gap.

**A silent wrong answer on the way.** My first script called `sim.init()` to time the build, then `sim.run(n_runs=1000)`. That took 79 s, and all 1,000 runs were identical. A plain SIR reproduces it (`same_runs.py`): `run(n_runs=20)` gives 14 distinct final sizes, and `init(); run(n_runs=20)` gives 1, and the same goes for `ss.parallel`. The cause is `starsim/run.py:1216-1218`: an initialized sim is copied and its `rand_seed` set after initialization, when the seed no longer has any effect, and the warning at `run.py:822` is skipped because `reseed` was just set to False. It also falls off the fast restart path (`run.py:1122`). An odin user calls `gen$new()` and then `$run()` and would naturally do the same thing here. The fix is small: rebuild with `sim.new()` whenever the sim has inputs, or raise. None of the four reviews has this; the nearest are the MultiSim shape findings (opus [results 2](../opus_review/notes/results-and-calibration.md), fable [05 §2](../fable_review/05_results_runs.md)).

### 3. Fit it with a particle filter, as I would with mcstate or monty

I drew 40 weeks of negative-binomial cases from one odin run (`gen_data.R`, `cases.csv`), and fitted `beta`, `gamma`, and `rho` by `sim.calibrate()` (`fit_v4.py`). For a stochastic sim with a likelihood on every series, v4 chooses particle MCMC by itself and prints why (as it did on camdl's SIR), which is the right behavior. Here it did not finish: 4 chains of 200 warm-up and 200 draws were killed at my 20-minute limit, with nothing printed. Timing one likelihood at fixed values (`pf_time.py`):

| Particles | One likelihood | Log-likelihoods from three seeds |
|---|---|---|
| 100 | 1.5 s | −423.8, −415.4, −422.0 |
| 400 | 5.6 s | −418.8, −413.5, −421.3 |
| 1,000 | 12.3 s | −416.5, −418.2, −413.4 |

The spread of 2 to 4 nats means this model needs 1,000 particles or more. At 12 s a likelihood, that is about 33 hours per chain for 10,000 iterations. Per particle-step, the filter takes 11 µs, against 1.35 µs for odin 1.2.7's single-threaded forward run. The filter itself is correct: on camdl's SIR, its posterior matched camdl's PMMH to two decimals (see the [camdl report](camdl.md)). Speed is the gap. The fable review notes the PMCMC test is skipped "requires performance enhancement" ([08_inference.md:160](../fable_review/08_inference.md)), but no review measures it against a specialist.

The deterministic route works and is a draw for this user. With `solver='ode'` (`fit_v4_ode.py`), v4 chose NUTS with JAX gradients: 7.1 s to compile, then 980 s for 4 chains of 500 + 500 draws. I had left the seasonality out of the fitting model by mistake, and v4 said so in its own way: R-hat 1.5, 336 divergences, `W1301 AtBound` for `gamma`, and a note that `beta` and `gamma` are correlated at r = 0.994. Diagnostics that refuse to bless a misspecified fit are what this user wants. Putting a fitted parameter inside the seasonal curve, `ss.seasonal(mean=beta_par, ...)`, raises `E206 DuplicateDeclaration` (`seas_par.py`). Writing `beta * ss.seasonal(mean=1, amplitude=amp)` works, including fitting `amp` (`seas_par3.py`).

### 4. Do it from R

This user will not leave R for a library, so I tried v4 through reticulate (`rwrap2.R`, pointing reticulate at `/software/conda/envs/v4/bin/python`, which is what rstarsim does underneath; I did not install rstarsim). **No v4 sim can be built from R today.** `ss$Sim(...)` raises `ValueError: call stack is not deep enough`. So do `ss$Disease(...)` and even the v3-style `ss$Sim(diseases='sis', networks='random')` that the rstarsim README shows. The cause is v4's name capture: `sys._getframe(1)` at `starsim/sim.py:73` (new in v4; `main` has none in `sim.py`) and `starsim/diseases.py:74`, and reticulate calls these with no Python frame above. When I patched `sys._getframe` in the session to fall back to the outermost frame (`rwrap3.R`, standing in for a few lines of guard), these worked:

- The arrow-string form ran: `ss$Disease(infection=ss$transmission('S -> I', beta=0.2), ..., name='sir')`, 100 binomial runs in 0.33 s, mean final `R` 346.9.
- `ss$load('seir_array_odin.R', format='odin')` ran.

These did not:

- `ss$states(3L)` returns states with no names, since there are no variable names to capture.
- `>>` and `@` do not exist in R. `S$`__rshift__`(I)$`__matmul__`(I * 0.2)` builds a transition, but it and the explicit `ss$Transition(S, I, rate=I * 0.2, name='infection')` both fail at the sim with `E204 UnknownState: init= names a state "I" that "sir3" does not have. States are S, I, R`. That message contradicts itself. The same code in Python works (`e204.py`), and I did not find the cause.

So the R user gets the arrow-string short form and nothing else, once the frame bug is fixed. None of the four reviews mentions R or reticulate.

## What would win them over

- **The model is a fifth of the length, and it is the same model.** 46 + 5 lines become 10, and 1,000 stochastic runs agree with odin within 0.7%. No `dim()`, no aging terms written out, no `1 - exp(-r*dt)`, and no index bookkeeping for the contact matrix.
- **One model, many solvers.** The same 10 lines run as an ODE, a chain binomial, Gillespie, or agents (`solver='abm'`) with a network. odin cannot do agents at all; an odin user who needs households or contact tracing has to move to another package (`individual`) and rewrite everything.
- **Observation and fitting are declared, not written.** `ss.outcome(rho * infection, every=ss.weeks(1), likelihood=ss.nbinom(n=k))` plus `data=` gives automatic alignment of flows to weekly data and a method chosen with a printed reason: NUTS for the ODE, particle MCMC for the stochastic model. In monty this is a comparison function, a filter object, a packer, a prior model, and a sampler, each set up by hand.
- **The fits are honest.** The PMCMC posterior matched camdl's PMMH. The NUTS run flagged non-convergence, a parameter at its bound, and a ridge between two parameters, without being asked.
- **`describe()`, the printed solver choice, and the hash.** Nothing in odin tells you what the model guessed, because odin guesses nothing. v4 does guess, and it says so.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff? | Already in a review? |
|---|---|---|---|---|
| No v4 sim can be built from R (`sys._getframe(1)` raises under reticulate, `sim.py:73`, `diseases.py:74`), so rstarsim is broken on v4 | Bug | Yes: guard the frame lookups, about 10 lines, plus an rstarsim test in CI | Not applicable | No |
| Even when fixed, R gets only arrow strings: no `>>`/`@`, no name capture, and the explicit form fails with a contradictory E204 | Inherent tradeoff (a Python-shaped language) plus a bug | Partly: make `name=` and arrow strings a documented, tested R path; R-side helpers (`transition("S -> E", rate = ~ beta * I)`) in rstarsim; a few days | No. The best point keeps the operators for Python and gives R a first-class string or formula form, rather than leaving R with whatever survives reticulate | No |
| The particle filter is about 8× slower per particle-step than odin 1.2.7 on one thread (11 µs against 1.35 µs), and dust2 threads widen it; a 1,000-particle fit of a 4-group model would take a day or more | Missing feature (performance) | Yes, medium to large: particles in parallel across threads in the Numba kernel, with a stream per particle; a kernel specialized to the model (`v4_design/admin/deferred.md` "a kernel specialized to the model") | No. One general kernel can be within 2 to 3× of generated C++ if it is parallel; today it is serial | Partly: fable 08 notes the PMCMC test is skipped for speed |
| `sim.init()` then `run(n_runs=)` or `ss.parallel(sim, n_runs=)` gives identical runs, silently, and 5× slower | Bug | Yes, small (`run.py:1216`) | Not applicable | No |
| The odin importer crashes on `lambda` and on R's `if`, and turns a contact-matrix model into one with no transmission | Bug and missing feature | Keywords and `if`: small. Arrays coupled within a level (matrix sums) as a stratification plus `ss.contacts`: medium | Not applicable | Partly: fable 10 covers other importer issues |
| A fixed delay is many stiff Erlang stages (200 stages: 1.9 s and still 1.6% off, against 2 ms) | Missing feature | Yes, medium: the conveyor proposed in the vignette | No. Stages are the right default for a distribution, but a constant delay should be exact | Vignette finding 2 |
| odin lets you write any difference equation (a running maximum, a state that is not a flow, a forcing that depends on the state); v4 has transitions, and anything else is Python in `step()`, which leaves the fast kernels | Inherent tradeoff (declared flows against free equations) | Partly: a declared auxiliary variable updated in the kernel | Close. Flows cover most epidemic models, and they are what buy multi-solver and agents. But one declared escape hatch inside the kernels would move v4 to the better point | No |
| Gradients only for ODEs; no adjoint for a discrete-time deterministic model; a fixed menu of fitting methods against monty's composable samplers and priors | Inherent tradeoff (simplicity against flexibility) | Partly | Yes for most users; a power user would want `sim.loglik(pars)` and `sim.gradient()` exposed so they can bring their own sampler | No |
| A fitted parameter inside `ss.seasonal(mean=...)` raises E206 | Bug | Yes, small | Not applicable | No |
| Docs are all v3; nothing on PMCMC, `ss.outcome`, or likelihoods | Docs | Yes, after the port | Not applicable | Yes: opus §1 and §6 |
| CRAN, a decade of real-time use, and SPI-M papers, against an unreleased v4 | Ecosystem | Only with time and a published validation | Not applicable | No |

## Verdict

- **Today:** no. They cannot build a v4 model from R at all, and the particle filter they would come for is too slow for their usual model.
- **After the four reviews' fixes:** still no. None of the reviews touches R, filter speed, or the identical-runs bug. The reviews would make v4 cleaner for Python users, not reachable for this one.
- **What would make it a clear win:** v4 working from R (frame-safe, with a tested string or formula form), a parallel particle filter within about 3× of dust2, and an odin importer that reads a typical array model with a contact matrix. With those, "your model in a fifth of the lines, the same fit, plus agents for free" is a real reason to switch.

## The demo that would sell it

From R, take the model they already have, fit it, then run the same model as agents on a household network:

```r
library(starsim); load_starsim()
sim <- ss$load("age_seir.R", format = "odin", pars = list(S0 = S0, I0 = I0, m = m),
               data = "cases.csv", stop = 280L)   # contact matrix -> ss.contacts, lambda ok
sim$describe()                                     # what was read, and how
fit <- sim$calibrate()                             # "pmcmc: 4 chains, 1000 particles" in minutes
fit$plot()
agents <- fit$sim$new(solver = "abm", n_agents = 100000L,
                      networks = list(ss$RandomNet(), ss$contacts("GBR")))
ss$parallel(compartments = fit$sim, agents = agents)$plot()
```

What v4 needs: R calls that do not crash (`sim.py:73`), an importer that reads `lambda`, `if`, and matrix sums, a filter fast enough for 1,000 particles over 1,120 steps, and `data=` accepted by `ss.load`. It does not run today. It fails at the first `ss$load` from R, and in Python the import produces a model with no transmission.
