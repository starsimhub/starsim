# SimInf: would its users switch?

## Who this user is

A veterinary epidemiologist, typically at a national veterinary institute or an agricultural university (SimInf comes from Uppsala and the Swedish National Veterinary Institute), writing R with some C. Their models are compartmental in each of thousands to tens of thousands of nodes (herds, farms, pens), stochastic in continuous time with Gillespie's direct method in every node, and linked only by *data*: the national livestock register's births, deaths, and movements between holdings, millions of events over years, fed to SimInf as an events table with `select` and `shift` matrices. They write models with `mparse()` (transition strings compiled to C), give node-specific parameters in `ldata` and continuous node states in `v0` updated by a C post-time-step function, run with OpenMP across nodes, pull results with `trajectory()` and `prevalence(result, I ~ ., level = 2)` (the share of herds infected), and fit with `pfilter()`, `pmcmc()`, and `abc()`. They picked SimInf because it is fast (C, a fraction of a second for 1,600 herds over four years), because register data goes in as it is, because it is on CRAN with a JSS paper, and because their colleagues use it. They would switch for things SimInf cannot do: animals with individual histories, interventions that react to the epidemic, and less C. They will not accept losing an order of magnitude in speed on the national register.

## The first day

| Vignette | Lines (original, Starsim) | One run (original, Starsim) | Agreement |
|---|---|---|---|
| `sir_herds` (`events_SIR()`: 1,600 herds, 466,692 events) | 12, 23 (7 convert SimInf's events to a register, 4 read data) | 0.08 s, 3.0 s (+0.9 s build, 1.9 s reading) | Agrees: within about two standard errors on every day checked |
| `sise3` (SISe3: three ages, environmental pressure `phi`, seasonal decay, 783,773 events) | 27, 33 (13 read and convert events) | 0.10 s, 12 s (+4 s build) | Agrees within about one standard error, within and between herds, after eight findings were fixed |

Source: `lingua_franca_design/vignettes/siminf/*/README.md`. This user would recognize both models, and the SISe3 translation would impress them: SimInf's model is 27 lines of R *plus* C in `src/models/SISe3.c` and `SimInf_forward_euler_linear_decay.c` that they never see; the v4 version states all of it, with `phi` as `ss.State(people=False)`, shedding and decay as two transitions, and the seasonal decay as one `ss.timeseries(..., period=ss.years(1))` (`vignettes/siminf/sise3/model.py:33-46`). The register is the other thing they would notice first: a table with columns `date`, `from`, `to`, `n`, `states`, `to_states` in place of an event code, a column index into `select`, and a row offset into `shift`. I wrote the `select` matrix wrong on my first `mparse` attempt below (`E` columns transposed), and SimInf's message was `Unable to sample individuals for event`; with named states that mistake cannot be made.

They would also see the speed: 40 and 120 times slower per run. The vignette READMEs say so plainly, and the reviews flag it as a failing north-star row (fable2 02 finding 1; opus §6).

Where they would look: `00-overview.md` has nothing on metapopulations or events; `06-routes.md:291` documents `ss.Mobility(data='movements.csv')` in one line with its columns, and `16-interop.md:63` says how SimInf maps. `docs/` is entirely Starsim 3 (fable 12 §7) and has no register, no `ss.Mobility`, and no stochastic compartmental solver, so the documentation this user would actually find describes an agent-based framework that cannot run their model at all.

## The second week

Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/siminf/`. SimInf 11.1.0 was run from `_envs/siminf/`. The machine was shared with other jobs (load average 10 to 20 on 20 cores), so times are noisy; the steady-state v4 time for `sir_herds` was 4.2 to 4.9 s here against 3.0 s in the README.

### 1. A user-written model with a rate for each herd (`mparse` + `ldata`)

The everyday SimInf task: their own transitions, with a parameter that varies by node. SimInf (`s1_mparse.R`):

```r
model <- mparse(transitions = c("S -> beta*S*I/(S+I+R) -> I", "I -> gamma*I -> R"),
                compartments = c("S", "I", "R"), ldata = data.frame(beta = b), gdata = c(gamma = 0.01),
                u0 = u0, tspan = 1:1460, events = events_SIR(), E = E)
```

v4 (`t1_ldata.py`) is the vignette with one change. The natural spelling fails:

```python
beta = ss.perday(ss.by(herd=b))            # TypeError: Value must be a scalar number or array, not <class 'starsim.transitions.by'>
beta = ss.by(herd=[ss.perday(x) for x in b])   # works
beta = ss.by(herd=b)                       # works (bare numbers in the sim's unit, days)
```

The results agree (herd-specific `beta` drawn uniform on 0.08 to 0.24):

| | Infected, day 365 | Infected, day 1,460 | Herds infected, day 1,460 | Time per run |
|---|---|---|---|---|
| SimInf, 50 runs | 6,895 ± 670 | 7,613 ± 363 | 1,178 ± 56 | 0.25 s (including `trajectory()`) |
| v4, 20 runs | 6,198 ± 923 | 7,267 ± 688 | 1,111 ± 105 | 9.5 s |

A per-herd rate doubled v4's run time against a single rate (4.2 to 4.9 s here); I did not find why. SimInf compiles the `mparse` model with the system C compiler, which on Windows means Rtools; v4 needs no compiler, only Numba's first-use compile (below).

### 2. Scale to a national register

SimInf's herds and events replicated ten times, each copy seeded in its first herd (`s3_scale.R`, `t3_scale.py`): 16,000 herds and 4,666,920 events over four years, about the size of a national cattle register. One run, one thread:

| | SimInf | v4 |
|---|---|---|
| 1,600 herds: build, run | 0.14 s, 0.20 s | 1.1 s, 4.2 to 4.9 s (plus 2 s reading and converting the events) |
| 16,000 herds: build, run | 2.1 s, 1.9 s (2.0 s on 8 threads) | 13.2 s, 52.5 s (plus 20 s reading and converting) |
| Peak memory at 16,000 herds | 2.0 GB | 4.2 GB |
| Herds infected on day 1,460 (one run) | 13,344 | 13,404 |

v4 is about 28 times slower per run at this size, and the whole job (read, build, run) is 86 s against 4 s. A study that runs 1,000 replicates of each of five scenarios on this register would be 2.6 hours of SimInf on one thread and, by the same arithmetic, about three days of v4 on one core, or a few hours spread over 20. One v4 run at 1,600 herds also took 36 s once instead of 4.5 s, when the Numba kernel cache had to be rebuilt (fable 07 §3 measured the first-use compile at 29 s); SimInf's built-in models need no compile, and an `mparse` model compiled in about two seconds in task 1. SimInf's OpenMP threads help little here because the run is dominated by events; v4 has no threads within a run at all.

### 3. Fit by particle MCMC

SimInf's `?pmcmc` example in miniature (`s2_pmcmc.R`, `t2_pmcmc.py`): one herd of 1,000, SIR with `beta = 0.16`, `gamma = 0.077`, five infected, and Poisson observations of `I` every five days from one SimInf realization (`s2_data.csv`). v4 is short and needs no likelihood code:

```python
beta  = ss.par(ss.perday(0.2), bounds=[0.01, 1])
gamma = ss.par(ss.perday(0.1), bounds=[0.01, 1])
sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', rate=gamma), init=dict(I=5))
sim = ss.Sim(sir, total_pop=1000, solver='gillespie', unit='day', start=1, stop=100, data=data)   # data: columns t, n_I
fit = sim.calibrate(likelihood=dict(n_I=ss.poisson()))
```

It chose particle MCMC itself and said why, ran four chains of 1,000 warm-up and 1,000 draws in parallel (adapting to 50 particles), warned `W1304 NotConverged` on an R-hat of 1.06, and gave **beta 0.154 [0.123, 0.186], gamma 0.078 [0.066, 0.089]**, both intervals covering the truth. It took 1,072 s on the loaded machine.

SimInf's formula form, `Iobs ~ poisson(I)`, stopped at once with `Invalid observation process`: SimInf requires every particle's weight to be finite (`pfilter.R:438-441`), and one particle with `I = 0` against a positive count breaks it. With the observation process as an R function (`dpois(Iobs, pmax(I, 0.1))`), one chain of 1,500 iterations with 100 particles took 86 s and gave beta 0.128 [0.104, 0.153], gamma 0.064 [0.056, 0.073], missing the truth (probably too short a chain from that start; I did not run it longer because the R-function form took 340 s for 4,000 iterations). So v4 gave the better-calibrated answer with less code and better diagnostics, at roughly twenty times the cost per particle run (10.7 ms against 0.57 ms on this loaded machine, SimInf's including its slow R observation function).

One reporting problem: the summary printed `Negative log-likelihood: 413.2 before, 4.218e+04 after`, because "after" is the likelihood of one fresh unconditioned trajectory at the best point (`calibration.py:2390-2395`), so a good fit reads as a hundred times worse. This is the PMCMC face of codex F21. SimInf's `abc()` (ABC-SMC) has no v4 counterpart; the spec promises ABC-SMC and the code does not have it (fable §7, fable2 02 finding 2).

### 4. The outputs they make every week

SimInf's `trajectory(result, index = 1:3)` is a long data frame (`node`, `time`, `S`, `I`, `R`) and `prevalence(result, I ~ ., level = 2)` is the share of herds infected. In v4 (`t4_output.py`): `sim.to_df()` has totals only; `sim.results.sir.n_I.by('herd').to_df()` is a wide frame with one column per herd (1,600 columns for the vignette); the share of herds infected is one line of NumPy on `np.asarray(r.strata) > 0`. All of it works, but in a sim with `unit='day', start=0` (how both vignettes are written) `Result.to_df()` labels the rows `datedur(0)`, `0001.01.01`, `0002.01.01`, ..., reading days as calendar years, after a `RuntimeWarning: Dates with years < 1 are not valid`; `sim.to_df()` gives plain numbers. This is the numeric-time guessing that codex F09 and opus §2 describe, seen from a herd model.

## What would win them over

- **The register as a named table.** `date, from, to, n, states, to_states` replaces event codes and `select`/`shift` matrices, and the same table moves, adds, removes, and ages animals (vignettes; task 1). This is a clear improvement in the part of SimInf its users find hardest.
- **Models without C.** SISe3's environmental pressure, its seasonal decay, and its daily update are declared in v4 (`sise3/model.py`); in SimInf they are C files the user cannot see or must write. A new model with a continuous node state is a few transitions, not a post-time-step function.
- **Particle MCMC in one call, with multiple chains, R-hat, and an interval that covered the truth** where SimInf's single chain did not (task 3), and a likelihood that does not crash the filter on a zero.
- **The same model as animals.** `sim.new(solver='abm', n_agents=196_169)` runs the herds model with one agent per animal (108 s, `sir_herds` README). Animal-level test histories, ages in days, and within-herd contact structure are out of reach in SimInf.
- **Interventions and scenarios as declarations** (`ss.vaccinate`, `ss.modify`, `sim.new(pars=...)`) instead of extra event rows. I did not test a herd-level intervention that reacts to detected infection; see the demo.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff? | Already in a review? |
|---|---|---|---|---|
| 20 to 40 times slower per run for the SIR (0.20 vs 4.2 s at 1,600 herds here, 0.08 vs 3.0 s in the README; 1.9 vs 52.5 s at 16,000) and 120 times for SISe3 (0.10 vs 12 s), and 13 s to build the 16,000-herd sim against 2.1 s | Partly inherent (one general engine against a C specialist), mostly fixable | Largely: both run Gillespie by place in machine code (C and Numba); the vignette already took 9.5 s to 3.0 s by running each herd on its own between events. What remains is visiting every herd every day and applying the register, both of which can be done only for herds an event touched (medium) | No. A Numba engine should land within a small factor of SimInf's C, not 28 times; the README's own analysis names where the time goes | fable2 02 finding 1 and opus §6 (as a failing north-star row); the scaling and build numbers are new |
| No parallelism within a run | Missing feature | Yes, medium: herds are independent between a day's events, so they can be stepped in parallel | | No |
| Twice SimInf's memory at 16,000 herds (4.2 vs 2.0 GB) | Implementation | Probably (register held in pandas and as arrays) | | No |
| First-use kernel compile of about 30 s, silent | Implementation | Yes: ship a cache, or say what is happening | | fable 07 §3 |
| No ABC-SMC | Missing feature | Yes, medium | The spec's method table is right; build it or strike it | fable §7, fable2 02 |
| Fit summary reports an unconditioned single-run likelihood as "after" | Bug | Yes, small | | codex F21 (Optuna side) |
| `ss.perday(ss.by(herd=...))` raises a bare `TypeError` | Bug | Yes, small: accept it, or say to put units inside `ss.by` | | No |
| Results by herd are wide, not long; day times printed as years with `start=0` | Bug and missing convenience | Yes, small: a long `to_df(by='herd')`; never read a unitless or day-based time as a year | | codex F09, opus §2 (time guessing) |
| The register can choose only states, not a level of another dimension, so age must be six states with `people=True` | Missing feature | Yes, medium | | sise3 README finding 5 |
| The SimInf importer drops the register | Missing feature | Yes, small to medium | | fable 10 |
| R: every v4 model fails when built from R (`sys._getframe(1)` at `diseases.py:74`, `sim.py:73`; see the EpiModel report, task 5), so R-Starsim cannot reach v4 | Bug, then ecosystem | The crash: a few lines. A usable R front end: medium; this user needs `data.frame` in (`u0`, events) and out (a long trajectory), which reticulate mostly handles once construction works | Arrow strings cross the language boundary; frame-based naming does not | No |
| Ecosystem: CRAN, a JSS paper, use at a national institute, colleagues who read `mparse` | Ecosystem | Slowly: a published comparison on a real register | | No |

## Verdict

- **Today:** no. The register is better and the models are clearer, but 28 times slower on a national register, R construction broken, and v3-only docs outweigh them for work SimInf already does.
- **After the four reviews' fixes:** still no for production register studies, since no review proposes closing the speed gap (they propose restating the test); yes for a project that needs what SimInf lacks (animal-level agents, PMCMC with diagnostics, models with continuous node states and no C).
- **What would make it a clear win:** within about three times SimInf's run time on the 16,000-herd register, with threads within a run; ABC-SMC; long per-herd outputs; and working construction from R. Then v4 would be SimInf's speed class with a better register, no C, and agents when needed.

## The demo that would sell it

The national-register model with what SimInf cannot schedule: surveillance that detects infected herds, a movement ban on detected herds that removes their outgoing register rows while it lasts, and a switch to animals for the herds under investigation.

```python
import pandas as pd, starsim as ss
u0, register = pd.read_csv('herds.csv'), pd.read_csv('movements.csv')        # date, from, to, n, states
beta = ss.par(ss.perday(0.16), bounds=[0.05, 0.5])
infection = ss.transmission('S -> I', beta=ss.by(herd=u0.beta_scale) * beta)
sir = ss.Disease(
    infection = infection,
    recovery  = ss.progression('I -> R', rate=ss.perday(0.01)),
    init      = dict(I=ss.by(herd=u0.I)),
    detected  = ss.outcome(0.02 * infection, by='herd', likelihood=ss.poisson()),   # 2% of infections found by herd testing
)
sir.stratify(herd=u0.herd, split=u0.N/u0.N.sum(), mobility=ss.Mobility(data=register))
ban = ss.restrict(sir.mobility, where=sir.detected.by('herd').cum > 0, dur=ss.days(60))   # no sales from detected herds for 60 days
sim = ss.Sim(sir, solver='gillespie', unit='day', start='2020-01-01', stop='2024-01-01', total_pop=u0.N.sum(), data='herd_detections.csv')
fit = sim.calibrate()                                      # particle MCMC on herd-level detections
runs = ss.parallel(baseline=fit.sim, ban=fit.sim.new(interventions=ban), n_runs=500)
runs.plot('herds_infected')                                # 16,000 herds x 500 runs x 2 scenarios: minutes, not hours
```

It does not run today. What is missing: a delivery that can switch register rows off by a condition on a herd (`ss.restrict` here is a placeholder name; nothing in v4 filters the register at run time, since it is converted once into the kernel's arrays, `solvers.py:1670-1719`), an outcome stratified by place that a condition can read, `ss.parallel` accepting a run sim (fable §4), and the speed to make the last comment true (task 2). The disease, the herd stratification, the register, `ss.by` rates, Gillespie by place, and particle MCMC on a likelihood run now (vignettes; tasks 1 and 3). The ban is the point: SimInf's events are a fixed schedule, so a reactive movement restriction there means custom C or re-running with an edited table, and it is the question veterinary authorities actually ask.
