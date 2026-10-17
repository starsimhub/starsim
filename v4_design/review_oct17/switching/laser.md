# LASER: would its users switch?

## Who this user is

An IDM modeler (or a collaborator trained by IDM) who builds national agent-based models with `laser-core` and `laser-generic` or `laser-measles`: Python with Numba, a `GeoDataFrame` of a few hundred to a thousand places, a `PropertySet` of parameters, and a list of components (`Susceptible`, `Exposed`, `Infectious`, `Transmission`, births by crude rate, deaths from a life-table estimator, importation, gravity coupling of the force of infection, routine immunization with a delay after birth, SIA calendars). Their typical run is tens of millions of agents for decades on a workstation's cores, and their typical paper question is spatial: critical community size, fade-out and reintroduction, traveling waves, campaign timing by district. They chose LASER because it runs one person per agent at national scale in about a gigabyte and a couple of minutes, which no general framework they had tried could do, and they accept writing more code for it. They fit with their own Optuna or grid loops (`laser-core` 1.0.5 and `laser-generic` 1.1.0 have no fitting module), and they get places and populations from `laser-init` (shapefiles, rasters, and demographic tables per country). Speed and memory at national scale are why they are here; everything else is secondary.

## The first day

| Vignette | Lines (LASER, v4) | One run (LASER, v4) | Agreement |
|---|---|---|---|
| `vignettes/laser/hub_spoke_seir` | 50, 16 | 0.78 s at 1.75M agents; 1.3 s at 175,000 scaled agents | Agree (final recovered 1,533,300 against 1,532,298) |
| `vignettes/laser/england_wales` (53 places, 668,000 agents, 10 years) | 126, 62 | 3.0 s; 131 s | Agree (104,092 infections against 106,639) |

They would recognize both translations at once. The hub-and-spoke model is shorter and clearer than the original: the places, the gravity coupling (`ss.Mobility(ss.gravity(c=2.5), kind='force', away=3/32)`, LASER's own form), and initial infections in Town 1 are each one line. The England and Wales model is a faithful port but shows the seams this user cares about: the life-table deaths and the fixed-count importation are Python functions that write `sim.people.ti_dead` and the disease's state arrays directly, and the gravity matrix with LASER's mean-share scaling and cap is built in NumPy (vignette README findings 3, 4, and 6; `deferred.md:24`).

The speed row has improved since it was written. At HEAD, the 10-year England and Wales run takes 57 s (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/laser/ew_10y_53.txt`), not 131 s, after the agent-pass work in `implementation.md:214-216`. The vignette README and `implementation.md:53` should say so. The hub-and-spoke row compares 175,000 agents with LASER's 1,750,000, which flatters v4; at equal size the gap is in the next section (Cliff found the same in session 13, `v4_design/admin/prompts.md`, prompt 49).

The front door works against them. `docs/` is Starsim 3, and the one LASER page a user would find there (`docs/examples/laser/laser_starsim_full.py`, "Requires starsim>=3.0") ports a LASER model as a hand-written `ss.Module` with `ss.Arr` timers: the opposite of v4's declared model. The best page for this user is `skills/import-model/references/laser.md`, which maps every LASER component honestly and says in its "Speed" bullet that v4 is "much slower per agent than LASER (about 20 to 110 ns per agent-step against 1 ns)". A LASER user reads that sentence and stops.

## The second week

All scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/laser/`. The machine (20 cores, 62 GB) was shared with other review agents throughout, with a load average of 7 to 21, so LASER's threaded times in particular are noisier and slower than its reference (England and Wales took 4.7 s against the reference 3.0 s). Single-threaded comparisons are the more reliable ones.

### 1. Scale up: 3.3 million agents in 149 places

The England and Wales model as written, with every place within 150 km of Norwich instead of 100 km (149 places, 3,307,123 people, one agent each), for 2 years. LASER is the notebook's own code through `examples/laser/england_wales/simulate.py 2 Norwich 150 1 2`; v4 is the vignette's `model.py`, copied with the radius and length as parameters (`ew_model.py`, `ew_run.py`).

| | Run (s) | ns per agent-step | Peak memory |
|---|---|---|---|
| LASER, 20 threads | 2.75 | 1.1 | 466 MB |
| LASER, 1 thread (`NUMBA_NUM_THREADS=1`) | 8.24 | 3.4 | 458 MB |
| v4, agents | 67.8 (plus 2.7 s `init`) | 28 | 1,436 MB |

The same ratios hold at the vignette's size (668,000 agents, 10 years): LASER 4.7 s on 20 threads and 7.6 s on one; v4 57 s. So v4 is 7 to 8 times slower than LASER on one core, and 12 to 25 times slower than LASER on its threads. On the hub and spoke at LASER's own 1.75M agents: LASER 1.67 s (20 threads) and 5.38 s (1 thread), v4 16.4 s; 10 times and 3 times. I did not run enough to test agreement at this size: v4's one run gave 125,817 infections in two years, 2% above the highest of LASER's four (115,921 to 122,857); at the vignette's size, v4's 10-year run gave 101,442, just below LASER's 5% quantile of 103,139, as one of the vignette's five runs was.

Memory is the sharper problem. v4's peak is about 330 bytes per agent (566 MB at 668,000 agents, 1,436 MB at 3.3M, with 312 MB for the import), and the peak is reached at `init()` or the first step, not during the run (1,428 MB after one step). LASER's is about 30 bytes per agent. The 106 bytes of each agent's own arrays (`mem_arrays.py`) are: `uid`, `slot`, and `parent` as int64 (24 bytes); `age`, `ti_dead`, `ti_removed`, `scale`, and `patch` as float32 (20 bytes, the place index among them); `rel_sus` and `rel_trans`; and for each of the four states a boolean plus three float32 arrays (`ti_X`, `ti_exit_X`, `exit_X`), 52 bytes. The rest of the peak is temporaries. At LASER's full England and Wales size (30.2M agents) v4 would need about 10 GB; LASER ran it in 1.18 GB (`laser_ew_full_1y.err`). I did not run v4 at that size, since it would exceed the memory limit of this review.

### 2. Where the time goes, and whether it is inherent

A profile of the 3.3M-agent model for one year (`ew_prof.py`, `ew_prof.txt`: 30.4 s, 83 ms a step) shows no single hot spot. Each line is a separate pass over all agents:

| Function | Time | Per call |
|---|---|---|
| `count_states` (`solvers.py:3342`), twice a step: once for the results, once for the population by place in the force of infection | 7.2 s | 10 ms |
| `BoolState.true()` (`arrays.py:854`) and `nonzero` | 6.0 s | 3 ms |
| `infect_groups` (`solvers.py:4067`) and the infection-probability kernel | 6.4 s | 18 ms |
| `exposure` (`solvers.py:3387`), infectiousness by place | 3.2 s | 9 ms |
| Indexing and setting arrays (`arrays.py:420`, `434`, `442`) | 4.8 s | |
| `People.update_post`, births, the life-table function | 6.6 s | |

LASER makes about three passes over agents a step, in Numba with `prange`; v4 makes ten to fifteen, single-threaded (there is no `prange` or `parallel=True` anywhere in `starsim/kernels.py`). Each v4 pass is cheap, about 1 to 3 ns per agent, which is memory bandwidth, not Python: the per-step Python overhead is small at this size. So the gap is almost all fixable, and it has two parts.

- **Passes.** Counting by state and place, the infectious weight by place, the infection draw, and the timers could be one Numba loop per step over the agents for the common case (one disease, homogeneous mixing within places, a mobility matrix between them). That keeps Python between steps, so interventions, modifiers, and user functions still work; it is not the "whole agent-based run in one kernel" that `implementation.md:217` rules out. Cliff's prototype there (one Numba pass over 175,000 agents, 1.1 ms a step against 4.4) suggests about 4 times; the profile above suggests more at 3.3M agents, where `count_states` alone is 20 ms of the 83.
- **Threads.** `implementation.md:217` keeps the cores for running sims side by side. That is the right default for calibration, and the wrong one for this user, whose unit of work is one large run. A session-13 experiment that threaded two kernels cut 12.0 s to 8.8 s with identical results (`prompts.md`, prompt 49). `prange` in the fused pass when `n_agents` is large and the sim is not inside `ss.parallel` would recover LASER's 2 to 3 times.

What is inherent is smaller: a general engine carries per-state timers, CRN slots, and Python between steps that a hand-written component list does not. A realistic target after both fixes is 2 to 3 times LASER's time at about 50 bytes per agent (int32 UIDs, int16 timers in days, `ti_exit_X`/`exit_X` only for states that need them, and no init temporaries the size of the population). That is a tradeoff point a LASER user could accept in exchange for the rest of v4; today's is not.

### 3. The national model as compartments

v4's distinctive offer is that the same text runs as compartments. LASER's full England and Wales model (954 places, 30.2M people) cannot be written faithfully as compartments, because the life table needs ages, so `ew_comp.py` replaces the two workarounds with a constant death rate and importation as a rate per susceptible, and keeps everything else, including the 40-stage latent period.

| Solver | One simulated year | Peak memory |
|---|---|---|
| `binomial` (one kernel call for the whole run) | 12.0 s (10 years: 81 s) | 0.71 GB (10 years: 3.2 GB) |
| `binomial` with a 4-stage latent period | 4.3 s | |
| `tau` with a 4-stage latent period | 133 s | |
| `ode` | 173 s | 0.71 GB |
| LASER, agents, 20 threads (for reference) | 11.3 s for the loop, plus about 50 s to build 30M agents | 1.18 GB |

Three findings. First, the chain-binomial solver now takes whole days with the 40 stages (`implementation.md:218`): the vignette's 48 s per simulated year is now 12 s. That is a real result: a LASER user gets the national model, with fade-outs in integer counts, at the speed of LASER's agents and with no agents to build. Second, it is not as fast as compartments should be: 24 ms a step for 954 × 44 compartments is about 570 ns per compartment-step, inside the single kernel call (`solvers.py:2613`, `run_all`). I did not profile inside the kernel. Third, the ODE is 14 times slower than the binomial, because the automatic substep rule asks that no compartment turn over by more than a quarter in a substep, including the mobility rates (`kernels.py:858-863`); for RK4, whose stability limit is near 2.8, that is conservative, and each substep applies the 954 × 954 coupling four times. A stability-based step and a sparse coupling matrix are both small fixes.

Memory grows with length: about 0.28 GB per simulated year at 954 places. The whole-run kernel returns every compartment, including each of the 40 latent sub-stages, for every place at every step, and `run_all` keeps them all; the 500 MB cap (`solvers.py:1576`) applies only to `run_many()` (`solvers.py:2706`). The notebook's 40-year run would need about 12 GB. Keeping states summed over their sub-stages, and results by place at a reporting interval (weekly, as the notebook analyzes them) rather than every step, would cut this by about two orders of magnitude.

### 4. Routine immunization and a campaign, as agents

A laser-measles user's standard model: the Ghana vignette's 260 districts with ages, births into `S`, MCV1 at 9 months, and a national campaign for children 9 months to 5 years (`../ghana_measles/ri.py`, shared with the Ghana report). With 2.5M agents (1 in 10), 90 days took 375 ms a step, 150 ns per agent-step, at 1.52 GB. That is five times the per-agent cost of England and Wales: the profile (`../ghana_measles/prof_ri.py`) shows the interventions' eligibility conditions (`conditions.py:265` calling `nan_to_num` on the age array each step), their tallies and shifts, and the importation rate `(S + E + I + R).n / S.n` counted per place each step, each one more pass. Sixteen years would take about 37 minutes at 1 in 10, and about 6 hours and 8 to 10 GB at full size (estimated, not run). LASER-measles is not installed here, so I could not run the original; laser-generic ran 30M agents for a year in 11 s.

## What would win them over

- **One text, agents or compartments.** The same declared England and Wales model ran as a chain-binomial at full national size in 12 s a simulated year, which is LASER's agent speed with no agents, and the Ghana model's 260 districts for 16 years in about 2 s. Nothing in LASER lets this user check an agent model against its compartmental twin, or prototype at national scale in seconds.
- **Half the code, all of it declarative.** 62 lines against 126 for England and Wales, 16 against 50 for the hub. Gravity coupling in LASER's own form, a repeating seasonal table (`ss.timeseries(..., period=ss.days(365))`), initial conditions by place (`ss.by(patch=...)`), and an Erlang latent period are each one line, and `describe()` prints every guess.
- **Interventions by keyword.** A campaign for children 9 months to 5 years in five northern regions was one `ss.vaccinate(...)` with `eligible=(ppl.age >= 0.75) & (ppl.age < 5) & measles.district.isin(names)`, and it reached exactly those children (46,454 delivered, all in the 0.75-4 group and the five regions; `../ghana_measles/check_sia.py`). In LASER this is an `SIACalendarProcess` with a polars schedule and a hierarchical ID convention.
- **Inference in the box.** `sim.calibrate(gof=fn, method='optuna')` fitted two parameters to Ghana's district case data with the project's own loss (see the Ghana report), and `sim.run(n_runs=20)` gives a distribution in one call. LASER users write these loops themselves.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff, and if not, where | Already in a review? |
|---|---|---|---|---|
| Agents 7 to 8 times slower than LASER on one core, 12 to 25 times slower than LASER threaded (3.3M agents: 67.8 s against 2.75 s) | Performance | Mostly: one fused Numba pass per step for the common metapopulation case (1 to 2 weeks), keeping Python between steps | No. `implementation.md:217` weighs only a whole-run kernel; a fused per-step pass is the better point, and covers the models this user writes | The failing row is noted (Fable 2 `README.md:92`, Opus `README.md:142`); no review measured at scale or profiled |
| About 330 bytes per agent at peak against LASER's 30; a 30M-agent run would need about 10 GB | Performance | Yes: int32 UIDs, int16 day timers, exit arrays only where needed, no population-sized init temporaries; to about 50 bytes (days) | No: CRN needs a UID per agent, not 106 bytes of arrays | No |
| No threads inside a run (`kernels.py` has no `prange`) | Design choice | Yes, for the large-run case (days, after the fused pass) | No for this user: thread within a run when there is one large run, across runs otherwise | No |
| Life-table deaths, fixed-count seeding, and gravity scaled to a mean share with a cap cannot be declared | Missing features | Yes: `ss.death(age=ss.lifetable(...))`, `ss.seed(n=3, interval=...)`, `ss.Mobility(..., away=ss.mean(k), cap=0.2)` (days each) | Yes once added; the vignette's workarounds write internal arrays | Vignette README findings 3, 4, 6; `deferred.md:24` |
| Compartmental ODE at 954 places 14 times slower than the binomial; 570 ns per compartment-step in the binomial kernel | Performance | Yes: a stability-based RK4 step and sparse coupling (days); the kernel itself unprofiled | No | No |
| A whole-run compartmental result keeps every sub-stage in every place at every step, uncapped (0.28 GB per simulated year at 954 places) | Performance | Yes: sum sub-stages, keep places at a reporting interval (days) | No | No |
| Each intervention as agents is one or more passes over everyone each step (150 ns per agent-step with RI and a campaign) | Performance | Yes, with the fused pass and conditions evaluated on the agents who can change | No | No |
| `docs/` is Starsim 3, and its LASER example is a v3 hand-written port | Docs | Yes | | Fable `README.md:49` (docs have no v4 terms) |
| No `laser-init` equivalent (shapefiles, rasters, country demography), no measles component library, a smaller IDM user base for this kind of model | Ecosystem | Partly: an importer from a `laser-init` scenario folder would be days | | No |
| Scaled agents (`n_agents` below the population) as the answer to speed | Inherent tradeoff | | v4 is at a sensible point by offering both, but this user cannot use it: places of 777 people need whole people for fade-out (example README, "Scaling down") | No |

## Verdict

- **Today:** no. For this user's defining task, a national agent model, v4 is about 25 times slower than LASER on its threads and needs about 10 times the memory per agent, so the full model does not fit on their machine.
- **After the four reviews' fixes:** still no. The reviews fix the API, docs, and solver bugs, not agent throughput or memory; this user might use v4 alongside LASER for compartmental prototyping and calibration.
- **What would make it a clear win:** a fused, threaded agent step within 2 to 3 times LASER's speed at about 50 bytes per agent, plus declared life tables and seeding. Then v4 would offer something LASER does not: one model that runs as agents or compartments, with inference and interventions built in, at national scale.

## The demo that would sell it

```python
import starsim as ss
from ew_data import names, pop, distances, ages, lifetable, season   # what laser-init or the notebook gives

S, E, I, R = ss.states(4)
measles = ss.Disease(
    infection = ss.transmission('S -> E', beta=ss.par(3.9, bounds=[3, 4.5])*season, source='I'),
    onset     = ss.progression('E -> I', dur=ss.erlang(k=40, mean=ss.days(10))),
    recovery  = ss.progression('I -> R', dur=ss.normal(7, 2, unit='days')),
    births    = ss.birth('-> S', rate=ss.peryear(0.017)),
    deaths    = ss.death(age=ss.lifetable(ages, lifetable)),                     # not declarable today
    seeding   = ss.seed('S -> I', n=3, interval=ss.days(30), stop=ss.years(10)), # not declarable today
    init      = dict(I=ss.by(patch=[3]*len(pop)), R=0.95),
)
towns = ss.geo(names=names, pop=pop, distances=distances)
measles.stratify(patch=towns, mobility=ss.Mobility(ss.gravity(b=0.25, c=1.56), kind='force', away=ss.mean(0.045), cap=0.2))
sim = ss.Sim(measles, people=ss.People(pop.sum(), age_data=ages), unit='day', stop=40*365, data='weekly_cases.csv')

fast = sim.new(solver='binomial')           # seconds per decade: fit the 954 places here
fit  = fast.calibrate(gof=ccs_and_phase)    # the notebook's critical-community-size and wave-phase score
full = fit.sim.new(solver='abm').run()      # then 30 million agents, one per person, in a few minutes and about 1.5 GB
full.plot()
```

What it needs: the binomial line runs today (12 s a simulated year, with the two declarations replaced by a constant death rate and a rate of importation; `ew_comp.py`). The `ss.lifetable`, `ss.seed`, and `away=ss.mean(...)`/`cap=` declarations do not exist. The last line does not run today in any reasonable sense: by the measurements above, 30M agents for 40 years would take about 3 to 4 hours and about 10 GB. Fitting on the compartments and confirming on the agents, from one text, at LASER's speed, is the thing no LASER user can do now.
