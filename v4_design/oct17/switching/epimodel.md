# EpiModel: would its users switch?

## Who this user is

An HIV/STI modeler, usually in a US school of public health (the Emory/statnet orbit), writing R. Their models are agent-based over partnership networks that are *estimated*: `netest()` fits a temporal ERGM to target statistics, usually taken from egocentric survey data (ARTnet for MSM), with terms such as `edges`, `concurrent`, `degrange`, `nodefactor("race")`, `nodematch("age.grp")`, `absdiff("sqrt.age")`, and degree in one layer conditioning formation in another; `netdx()` is the mandatory check that the simulated network recovers its targets, including partnership duration. They run `netsim()` with their own modules (`infection.FUN`, `progress.FUN`, PrEP and ART cascades) written as R functions over the `dat` object, in the EpiModelHIV template, often at 10,000 to 100,000 nodes with `tergmLite = TRUE`, and they calibrate on an HPC cluster with EpiModelHPC/`swfcalib` or EasyABC, since EpiModel itself has no calibration. They chose EpiModel because the network is a statistical model with a citation behind it, because the HIV models of their field are built on it, and because it is R. What they would switch for: speed, less code, and fitting without a cluster. What they would not give up: the network's statistical meaning.

## The first day

Both translations are faithful, and this user would recognize them.

| Vignette | Lines (original, Starsim) | One run (original, Starsim) | Agreement |
|---|---|---|---|
| `sis-network` (EpiModel's intro) | 13, 8 | 3.9 s (+0.66 s `netest`), 0.45 s | Agrees once the order within a step is allowed for (237 vs 227.5 infected, which the README explains as EpiModel counting after same-step recovery) |
| `hiv` (Gallery wave 3: two networks, cascade, PrEP, four scenarios) | 405, 62 | 54 s (+1.2 s), 2.5 s | Within noise for cascade, PrEP, and both; no-intervention prevalence 16.4% vs 14.4% (about two standard errors), plausibly because `ss.PartnerNet` hits the casual-network targets better than EpiModel's own fit (0.30 and 10.2% vs 0.28 and 8.9%) |

Source: `lingua_franca_design/vignettes/epimodel/*/README.md`. The HIV translation is the strongest single exhibit in the corpus for this user: six R modules (305 lines) become one `ss.Disease` with a care stratification and seven `ss.transfer`s, and the cascade's effects on infectiousness and progression are one `adjust=` dict (`vignettes/epimodel/hiv/model.py:38-43`). An EpiModelHIV author spends weeks keeping `dat$attr` vectors in step on arrivals and departures; here there is nothing to keep in step. They would also notice what was bent: PrEP is a `ss.Vaccine` with `waning=-1/np.log(1 - 0.015)` (README finding 6), transmission is `1-(1-p)^acts` multiplied by relative infectiousness rather than applied per act (finding 5, 0.0374 vs 0.0370), PrEP eligibility is a 10-line function reading edge lists (finding 3), and a transition named `stop` crashes the sim (finding 9).

The network is the part they will read most skeptically. `ss.PartnerNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)` is not an ERGM: people seek partners with a probability that depends on their current degree, chosen so the target degree distribution is stationary (`starsim/networks.py:909-915`), and the spec says so plainly (`06-routes.md:231`). For the intro model it gives 173 edges and 109 with concurrency against EpiModel's 177 and 111. A statnet person will accept "matches the degree distribution and duration" for these two models, and will immediately ask what happens with the terms their real models use.

Where they would look things up: `v4_design/design/00-overview.md`, then `06-routes.md` §Sexual networks. The overview is cholera and SIR; nothing on the front page is about networks. `docs/` in ssv4 is still entirely Starsim 3 (no page uses `ss.Disease(`, `>>`, `ss.PartnerNet`, or `net.check()`; fable 12 §7, fable2 09 finding 4), so the docs site they would actually find teaches `ss.SIR` + `ss.RandomNet`, the v3 network classes, and v3 `ss.Calibration`. An EpiModel user who lands there sees a v3 random-network ABM with no estimation story and leaves.

## The second week

Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/epimodel/`. EpiModel 2.6.2 was run from `_envs/epimodel/` on the same machine, which was shared with other jobs (load average 17 to 20 on 20 cores), so absolute times are inflated for both.

### 1. Heterogeneous degree by risk group (`nodefactor` + `nodematch`)

The most common thing an HIV network model does beyond the intro: some people have more partners than others, and people partner mostly within their group. The EpiModel version (`e1_risk.R`) is three lines different from the intro:

```r
nw <- set_vertex_attribute(nw, "risk", rep(c(1, 2), c(800, 200)))
formation <- ~edges + nodefactor("risk") + nodematch("risk")
target.stats <- c(320, 320, 256)   # mean degree 0.4 low, 1.6 high; 80% of edges within group
```

It fits in 0.2 s; 10 runs of 300 steps take 61 s; the simulated network sits on its targets (317.7 edges, 320.5 high-risk endpoints, 255.4 within-group), and equilibrium prevalence is **20.7% low risk, 77.4% high risk**.

In v4 the natural spelling fails with a message that does not say what is wrong:

```python
ss.PartnerNet(n_contacts=ss.by(risk=[0.4, 1.6]), match=dict(risk=0.8), dur=50)
# AttributeError: 'by' object has no attribute 'pars'
```

`degree_dist()` accepts a number or a distribution only (`networks.py:874-887`), and `match=` takes exactly one property (`networks.py:1059`). The supported model, one degree distribution for everyone plus `match=` (`t1e_matchonly.py`), runs in 0.36 s and gives **31.2% low, 30.5% high**: the risk structure is gone, and nothing warns. To get the EpiModel model I subclassed `ss.PartnerNet` (`t1d_nodefactor.py`, 35 lines that override `init_pre`, `init_post`, and `add_pairs` and read internals such as `p_seek`, `match_pairs`, and `fisher_yates_shuffle`; my first attempt also hit `AttributeError: Cannot modify attribute "dists"`, a locked name, which at least said so). It gives **22.8% low, 79.5% high** in 3.8 s for 10 runs (16 times faster than EpiModel), with the network slightly under target (310 edges, 305 high-risk endpoints, 236 within-group, a match share of 0.755 against 0.80), because `match_pairs` computes its within-group share for equal degrees. Close, but it needed reading `networks.py` and overriding private methods; an EpiModel user would not get there.

This gap is recorded in `v4_design/admin/deferred.md:28` (no `nodefactor`, `nodematch(diff=TRUE)`, or triangles). What only this seat shows is the size of the consequence: without `nodefactor` the HIV model of this user's field cannot be written at all, and the one it can write has a qualitatively different epidemic.

### 2. Scale to a national-size population

The SIS intro model with every target scaled (`e2_scale.R`, `t2_scale.py`), 200 steps, one run, EpiModel with `tergmLite = TRUE` and `resimulate.network = TRUE`:

| Nodes | EpiModel `netest` | EpiModel `netsim` | v4 `run()` | Final prevalence (EpiModel, v4) |
|---|---|---|---|---|
| 500 | 1.0 s | 6.9 s | 0.2 s | 47.4%, 46.4% |
| 5,000 | 3.8 s | 6.7 s | 0.3 s | 46.0%, 45.6% |
| 20,000 | 12.8 s | 14.6 s | 0.7 s | 44.7%, 46.6% |
| 100,000 | 233 s | 28.2 s | 2.9 s | 43.3%, 47.2% |

v4 is 10 to 30 times faster per run and has no estimation step, which at 100,000 nodes is four minutes of MCMC that EpiModel repeats whenever a target changes. This is a real gain: an EpiModelHIV run of 10 years at 10,000 nodes is minutes in EpiModel and would be seconds here, which changes what fits on a laptop. One caveat: `PartnerNet.pair()` checks for existing partnerships with a Python set of tuples every step (`networks.py:1129`), so the network, not the disease, is where the time goes as it grows; it was not a problem at 100,000.

### 3. Fit the transmission probability to observed prevalence

EpiModel has no calibration; this user would write an EasyABC rejection loop or a `swfcalib` job. In v4 it is three declarations and one call (`t3_calib.py`):

```python
beta = ss.par(ss.prob(0.64), bounds=[0.05, 0.95])
sis = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=I >> S @ ss.prob(0.05), prevalence=I.frac, init=dict(I=10))
sim = ss.Sim(sis, networks=net, n_agents=500, stop=500, data=data)   # data: prevalence 0.30 at steps 200..500
fit = sim.calibrate()
```

Two spellings failed first: `ss.prob(ss.par(...))` raises `TypeError: Value must be a scalar number or array, not par`, and `ss.transmission(S >> I, ...)` raises a raw `re` `TypeError` (it wants a string). Once written, it chose Optuna ("a stochastic sim"), ran 101 evaluations in 72 s, and returned `beta = 0.236`. A grid of 12 runs per value (`t3c_grid.py`) shows the right answer is about 0.20 (30.4% ± 1.0%); at 0.236 the model gives 34.5% ± 0.8%. The default `n_runs=1` (`calibration.py:2178`) re-seeds every evaluation, so the optimizer keeps the luckiest draw and the reported "0.158 after" is that draw's score, not the expected one (codex F21 covers the reporting; the default of one run is the cause here). No interval is given, as the spec says for Optuna. Still, for this user a one-call fit on a laptop is new, and with `n_runs=10` it would likely be good enough; the spec's ABC-SMC for cheap agent models would be the right default and is not built (fable §7, fable2 02 finding 2).

### 4. Diagnose the network (`netdx`)

`net.check(n_runs=5)` (`t4_check.py`) prints target, realized, and spread for mean degree (0.70, 0.698), concurrency (0.22, 0.221), and maximum degree (3, 3) in 5.7 s, and warns `W402` on a miss. It is the right idea in one call. Compared with `netdx`, it lacks the duration and dissolution diagnostics (the one EpiModel users look at most, since the edges-dissolution approximation is what goes wrong), plots, and a network-only mode: it reruns the whole epidemic, and it refuses for v3-style sims (`networks.py:963-965`).

### 5. Use it from R

The Starsim README advertises R-Starsim (`devtools::install_github("starsimhub/rstarsim")`). The package (`/home/cliffk/idm/rstarsim`, 178 lines in `R/starsim.R`) creates a conda environment, then `load_starsim()` does `reticulate::import('starsim')` and assigns `ss`, `Sim`, `Disease`, and a few others to the global environment; there is no R-side model syntax and no result conversion. I did not install it, but reticulate 1.46 is on this machine, so I ran what `load_starsim()` does against the v4 environment (`r1_reticulate.R`, `r2.R`, `r3.R`). **Every v4 model fails at construction from R**:

```r
sis <- ss$Disease(infection = ss$transmission("S -> I", beta = ss$prob(0.64)), ...)
# ValueError: call stack is not deep enough   (diseases.py:74, sys._getframe(1))
ss$Sim(diseases = "sis", networks = "random")      # R-Starsim's own documented example
# ValueError: call stack is not deep enough   (sim.py:73, sys._getframe(1); new in v4, absent on main)
```

v4 names objects by inspecting the Python caller's frame, and a call from reticulate has no Python caller. Wrapping the two constructors in a Python lambda (`py_eval("lambda *a, **k: ss.Disease(*a, **k)")`) makes the intro model run from R in 3.0 s, `sim$to_df()` comes back as an R `data.frame` (501 by 18, columns `sis_n_i`, `partnernet_mean_degree`, ...), and the mean infected over the last 200 steps is 231, as in Python. So the engine is reachable; the front door is broken, and no review mentions it. Beyond the crash, the v4 language does not translate to R: `>>` and `@` have no R spelling, and `ss.states(3)` naming states from variables cannot work across the language boundary, so an R user is limited to arrow strings and `ss$Transition(...)`.

## What would win them over

- **The HIV model in 62 lines instead of 405, and 2.5 s instead of 54 s, with the same answers.** The cascade as a stratification with `adjust=` is the single most persuasive thing in v4 for this user: it is the part of EpiModelHIV they spend the most time maintaining (vignette `hiv` README).
- **10 to 30 times faster per run, and no estimation step.** 2.9 s for 100,000 nodes against 233 s of `netest` plus 28 s of `netsim` (task 2). Targets can be changed and rerun instantly, which an ERGM user never gets.
- **Calibration in one call, without an HPC cluster** (task 3), and the same model can be fitted with `bounds=` and `data=` rather than a hand-written ABC loop. It needs `n_runs` set sensibly to be trusted.
- **Interventions as products plus deliveries, and scenarios as `sim.new(pars=...)`.** The four HIV scenarios are one line; in EpiModel they are `create_scenario_list` plus a loop.
- **`net.check()` and the degree results recorded on every run** cover the `netdx` habit in one call (task 4), and `describe()` shows every guessed default.
- **The same disease runs as an ODE**, which EpiModel splits into `dcm` with a separate model. This matters less to this user than to others, since their networks are the point; and the `PartnerNet` ODE approximation ignores duration (opus dims 12).

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff? | Already in a review? |
|---|---|---|---|---|
| No degree by attribute (`nodefactor`); `match=` on one property only; no mixing matrix (`nodemix`), age-difference term (`absdiff`), degree conditional on another layer, or duration by group. Without them, risk structure silently vanishes (31%/31% vs 21%/77%) | Missing feature | Yes, within the degree-seeking design: per-group degree targets and seeking tables, several `match=` properties, a `mix=` matrix, `dur=ss.by(...)`, and degree in another network as a group (about 300 lines; my 35-line subclass is the core of the first) | No. Simplicity against flexibility, but the current point is too far toward simplicity: these terms are what ARTnet-based models target, and they fit the same stationary-seeking method. Triangles (`gwesp`) genuinely need an ERGM and can stay out | Recorded in `deferred.md:28`; fable2 07 (`between='sex'`, `acts=`); the consequence is new here |
| `n_contacts=ss.by(...)` gives `AttributeError: 'by' object has no attribute 'pars'` | Bug (error message) | Yes, a few lines: say that degree by group is not supported | | No |
| Not a statistical network model: no estimation from egocentric data, no standard errors, no `ergm.ego` | Inherent tradeoff (one general engine vs a statistical specialist) | Partly: an importer that reads a fitted `netest` object's targets would bridge it; full ERGM is out of scope | Yes, for the simulation. The better point adds an honest statement in 06 of which ERGM terms map and which do not, so a reviewer can tell | Spec says it (06:231); no review weighs it |
| `net.check()` lacks duration and dissolution, plots, and a network-only run | Missing feature | Yes, small (record partnership ages; run the network alone) | Close | fable2 07 keeps it as is; the duration gap is new |
| No acts per partnership; per-act infectiousness | Missing feature | Yes, small (`acts=` on `PartnerNet`) | | Vignette finding 5; fable2 07; `deferred.md:21` |
| No network conditions (`degree`, partner of someone infected) for eligibility | Missing feature | Yes, medium | | Vignette finding 3; `deferred.md:21` |
| Calibration of a stochastic ABM defaults to one run per evaluation and keeps the luckiest draw (fitted 0.236, true about 0.20) | Bug in a default | Yes, small: default `n_runs` from the run time, or ABC-SMC as specified | No: a stochastic fit should never default to one replicate | Codex F21 (score reporting); fable §7 and fable2 02 (no ABC-SMC); the bias is new here |
| R: every v4 model fails from reticulate (`sys._getframe(1)` at `diseases.py:74`, `sim.py:73`), breaking the advertised R-Starsim; no R spelling for `>>` and `@` | Bug, then ecosystem | The crash: yes, a few lines (fall back to a default name when there is no Python frame; test from reticulate). A usable R front end: medium (R functions for `transmission`, `progression`, `PartnerNet`; results as data frames, which reticulate already gives through `to_df()`) | Arrow strings already work across the boundary, so R needs no operator syntax; the frame-based naming is the one design choice that cannot cross it | No |
| Docs are all v3 | Docs | Yes, large | | fable 12 §7, fable2 09, opus §6 |
| Ecosystem: EpiModelHIV, ARTnet, a decade of papers, the NME workshop, reviewers who know statnet | Ecosystem | Slowly: a published validation against EpiModel (the HIV vignette is most of one) and an EpiModelHIV-style template | | No |

## Verdict

- **Today:** no. The model of their field (degree by race and age, mixing by both) cannot be declared, the docs teach v3, and from R nothing runs.
- **After the four reviews' fixes:** still no for the core HIV user, yes for some: one language and v4 docs remove the confusion, but none of the reviews adds the network terms or repairs the R path. A modeler with a simple network and Python skills would switch for the speed and the calibration.
- **What would make it a clear win:** per-group degree targets, several `match=` properties, and `acts=` on `ss.PartnerNet`; duration in `net.check()`; and R construction that works. Then the EpiModelHIV template is a hundred lines of v4, runs in seconds, and fits in one call.

## The demo that would sell it

An ARTnet-style MSM model with race and age structure, a cascade, and PrEP, fitted to prevalence by race, in one screen:

```python
import starsim as ss
S = ss.State(); A = ss.State(rel_trans=5); C = ss.State(); D = ss.State(rel_trans=2)
beta = ss.par(0.004, bounds=[0.001, 0.02])                    # per act
hiv = ss.Disease(
    infection = ss.transmission('S -> A', beta=beta, source=['A', 'C', 'D']),
    chronic   = A >> C @ ss.weeks(12), aids = C >> D @ ss.years(10), aids_death = D >> ss.null @ ss.years(2),
    test      = ss.transfer(care='undx -> dx', rate=ss.by(race=[ss.peryear(0.5), ss.peryear(0.8)])),
    treat     = ss.transfer(care='dx -> art', rate=ss.peryear(2)),
    prevalence = (A + C + D).frac,
)
hiv.stratify(care=['undx', 'dx', 'art'], states=[A, C, D], split=[1, 0, 0], adjust=dict(rel_trans=[1, 1, 0.03]))
main = ss.PartnerNet(n_contacts=ss.by(race=[0.45, 0.55]), match=dict(race=0.8, age=0.6), dur=ss.years(3), acts=1.5)
casual = ss.PartnerNet(n_contacts=ss.by(race=[0.9, 0.6]), match=dict(race=0.7), dur=ss.years(1), acts=1, max_contacts=5)
prep = ss.vaccinate(ss.Treatment(efficacy=dict(acquisition=0.95), stop=ss.prob(0.015)), where=hiv.S & (main.degree + casual.degree >= 2), coverage=0.15)
sim = ss.Sim(hiv, networks=dict(main=main, casual=casual), n_agents=20_000, start='2010', stop='2030', data='prev_by_race.csv')
sim.stratify(race=['B', 'W'], split=[0.3, 0.7]); sim.stratify(age=[18, 25, 35, 45, 65])
casual.check()                                                 # degree, concurrency, match, duration: target vs realized
fit = sim.calibrate(n_runs=8)                                  # minutes on a laptop, not a cluster job
ss.parallel(baseline=fit.sim, prep=fit.sim.new(interventions=prep)).plot()
```

It does not run today. What is missing, in order: `n_contacts=ss.by(...)` and several properties in `match=` on `ss.PartnerNet` (task 1), `acts=` (vignette finding 5), network conditions such as `main.degree` in `where=` (vignette finding 3), `ss.Treatment(stop=...)` (vignette finding 6), duration in `check()` (task 4), and `ss.parallel` accepting a run sim (fable §4). The rest (the disease, the care stratification with `adjust=`, transfers, `calibrate()` from bounds and data, scenarios with `sim.new()`) is what the HIV vignette and task 3 already run; I did not run this exact script, and transfer rates by race and prevalence data by race are the parts I would check first. The same script, called from R through arrow strings and an R `calibrate()` that returns a data frame, would be the version that reaches this user where they work.
