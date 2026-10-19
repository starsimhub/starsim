# summer2: would its users switch?

## Who this user is

The summer2 user is a modeler in or near Monash's AuTuMN group, or someone who has borrowed its tools. They build TB and COVID country models for health ministries and WHO offices in the Asia-Pacific, write Python every day, and are comfortable with JAX. Their models are compartmental and heavily stratified: a base SEIR or TB structure (susceptible, early and late latent, active, on treatment, recovered) is written once and then stratified by age (with ageing), clinical status, immunity or vaccination, strain, and sometimes place. Each stratification carries flow adjustments, infectiousness adjustments, and a mixing matrix, and the result is 100 to 1,000 compartments. Every week they rely on parameters that vary in time (sigmoidal scale-up of case detection, mobility curves), derived outputs (notifications, incidence by age, hospital occupancy), and calibration of 10 to 30 parameters to notification data. Calibration goes through estival, which wraps the JAX runner for numpyro and pymc samplers and optimizers (from my knowledge of estival; it is not installed in `_envs/summer2`, so I did not run it). They picked summer2 because one model text serves many structures, because the JAX build makes runs fast and differentiable, and because it is their group's own tool, with the papers to show for it.

## The first day

| Vignette | summer2 | v4 | Agreement |
|---|---|---|---|
| `sir-age-location` (two stratifications, two mixing matrices) | 35 lines; 0.0003 s a run, 1.4 s the first time (build and compile) | 12 lines; 0.038 s first run, 0.0008 s repeated | Every compartment within 0.08 people of 1,000 |
| `seir-age-clinical` (clinical status entered by age, seven derived-output requests) | 134 lines (55 for the model); 0.0008 s, 3.5 s the first time | 15 lines; 0.024 s, 0.0002 s for a built sim | Every compartment within 0.0006 people, every derived output within 0.005% |

These numbers come from `vignettes/summer2/*/README.md`. This user would recognize the translation at once. `seir.stratify(clinical=[...], states=['I'], split=ss.by([[...], [...]], dims=['age', 'clinical']), adjust={'infection_death': [0, 1, 1, 2], 'rel_trans': [...]})` is `Stratification` plus `set_flow_adjustments(..., source_strata=...)` plus `add_infectiousness_adjustments` in one call, and `Overwrite(0)` is the multiplier 0. The big gain in lines is in outputs. summer2 builds the model seven times to request outputs one group at a time; in v4 every flow and state is a result already, and `res.incidence.select(age='old', clinical='hospital')` replaces `request_output_for_flow(..., dest_strata=...)`. One thing in the vignette is still open: a weight by level inside an outcome raises a NumPy broadcast error (finding 11, also F 12).

`docs/` is entirely Starsim 3 (F README:49, F2 README:57, O README:38). A summer2 user who goes looking for "stratify" will find `ss.SIR(beta=...)` and the old module classes, not the language above. They would have to learn from the design pages and the vignettes, which as a researcher they can, but it is a real barrier to a group adopting it.

## The second week

Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/summer2/`.

### 1. Bring my TB model across

I wrote the kind of model this user maintains (`tb_summer2.py`): S, E, L, I, T, R from 1950 to 2025 in a million people, with births, background and TB deaths, reinfection of the latent and recovered at a reduced rate, six age groups with ageing, a 6 × 6 mixing matrix, early activation and infectiousness adjusted by age, and case detection scaled up over time by summer2's sigmoidal interpolation times a `Parameter('detect_max')`. There are 36 compartments. summer2 builds and runs it in 6.4 s the first time and 0.036 s after that.

**The importer could not bring it across.** `starsim/interop/dump_summer2.py` failed three ways:

- `AttributeError: 'Parameter' object has no attribute 'name'` (`dump_summer2.py:36`, and the same at `:61-62`). summer2 1.3.6 calls it `key`. Every AuTuMN model uses named `Parameter`s, so this blocks all of them. It also leaves an empty JSON file, and `ss.load` then fails with `JSONDecodeError`.
- With that patched in a scratch copy, it fails with `GraphRunError: ('parameters.contact_rate', KeyError)`, because summer2 needs the parameter values to work out the initial population, and the dumper has no way to be given them.
- With every parameter written as a literal number, the import runs but lists two items as not translated (`W1601`). The first is the detection flow itself ("its parameter is a function ... could be an `ss.timeseries`", `summer2.py:94`), and the second is the mixing matrix, because it was a plain NumPy array rather than summer2's `Data(...)`. So the flow that makes notifications, and the mixing, are both dropped. `ss.interop.summer2.outputs(sim)['notifications']` then comes back as a 0-d array.

**Writing it by hand took about 25 lines and worked on the first run** (`tb_v4.py`):

```python
S, E, L, I, T, R = ss.states('S E L I T R')
tb = ss.Disease(
    infection        = S >> E @ beta * I,
    reinfection_L    = L >> E @ beta * rr_reinf * I,
    ...
    detection        = I >> T @ detect_max * scaleup,      # scaleup: summer2's sigmoid, sampled into ss.timeseries(mode='linear')
    tb_death         = ss.death('I ->', rate=0.2),
    death            = ss.death(rate=1/70),
    birth            = ss.birth('-> S', rate=0.02),
    init             = dict(I=1000),
)
tb.stratify(age=[0, 5, 15, 35, 50, 70], split=[1/6]*6, adjust={'early_activation': [3, 1.5, 1, 1, 1, 1], 'rel_trans': [0.1, 0.2, 1, 1, 1, 1]})
sim = ss.Sim(tb, networks=ss.contacts(age=mixing), total_pop=1e6, start=1950, stop=2025, unit='year')
```

Every compartment agrees with summer2 to within 8 people in 1.24 million (6e-6). Notifications in 1990, 2010, and 2024 are 2,341, 2,260, and 2,195, against summer2's 2,341, 2,263, and 2,195. The run takes 0.115 s the first time and 0.019 s for a new sim, with no per-model JIT wait (the Numba kernels are compiled once, when Starsim is installed, and cached). Births go into the youngest group and ageing is generated from the group widths, both without being asked. One gap remains: there is no sigmoidal scale-up. `mode='cosine'` is S-shaped but is a different curve, so I sampled summer2's curve every 0.1 years. Two small things also failed: `1.5 * ss.timeseries(...)` raises `TypeError` (there is no `__rmul__`), and an inline `ss.par(1.5) * ...` raises "Could not find a name for the parameter".

### 2. Calibrate to notifications

The target was notifications for 2000 to 2024 from the summer2 reference run (truth: `contact_rate=10`, `detect_max=1.5`). Both fits started from 8 and 1.0.

| | summer2 (`get_runner` + JAX + SciPy L-BFGS-B) | v4 `sim.calibrate()` |
|---|---|---|
| What I had to write | A jitted runner, a loss, `jax.jacfwd`, and a SciPy call (`tb_summer2_grad.py`, 25 lines) | `bounds=` on two `ss.par`s, `data=`, and `sim.calibrate()` |
| What went wrong | Reverse mode fails on the sigmoid ("Reverse-mode differentiation does not work for lax.while_loop"), and the default `odeint` cannot do forward mode ("can't apply forward-mode autodiff (jvp) to a custom_vjp function"). It worked only with `solver=RUNGE_KUTTA`, whose step is the output step of a year | Nothing; it picked L-BFGS-B with JAX gradients and printed why |
| Time | 3.7 s, about 2.5 s of it compiling; 4.8 ms a run, 13 ms a gradient | 10.5 s with L-BFGS-B (69 evaluations); 2.9 s with `method='nelder-mead'` |
| Result | Recovered its own truth, but on data from its own coarse yearly RK4, which differs from its accurate run | 9.98 and 1.495, from data made by summer2's accurate run (the residual 0.05% is the difference between summer2's averaged rate and v4's flow over the year) |

The point-estimate story is good. v4's gradient agrees with finite differences (`grad_check.py`: d notifications/d `detect_max` is -700.25 both ways). The calibration code is three keywords instead of a hand-written JAX pipeline. With two parameters a simplex is faster, as `implementation.md:221` says for three.

**The posterior did not work.** Asking for NUTS with `likelihood=dict(detection=ss.nbinom(n=50))` printed "The log-density in JAX ... differs from the one from the sim ..., so the posterior is sampled again with Python in the loop" (`calibration.py:2338`). It then took 927 s, every one of its 2,000 draws diverged, the R-hat was 5.6e15, and the posterior sat at the starting point (`tb_nuts.out`). Part of that run was my fault, because the data were not whole numbers, so the sim's negative-binomial log-density was `-inf`. But nothing said so, and with whole numbers the same fallback happened and the run had not finished after 25 minutes (`tb_nuts_int.out`). I narrowed it down (`nuts_bisect.py`). The JAX log-density agrees for the model with births, deaths, age, mixing, and a time series, so long as only `beta` is fitted. **It disagrees as soon as a fitted `ss.par` multiplies an `ss.timeseries`** (`plain_fitdm` agrees; `ts_fitdm` gives -967,465 against -146.7). A time-varying rate scaled by a fitted parameter is exactly how this user writes case detection, so their standard calibration falls back to the slow path every time. This is a bug, not a design limit: the JAX run and `sim.gradient()` are both right for the same model (`tb_jax_check.py`, `grad_check.py`).

### 3. Scale up to an AuTuMN-sized COVID model

The next model is an SEIR in 16 five-year age groups with the Prem 2021 matrix for the Philippines, `I` split into five clinical levels (death and infectiousness adjusted by level), and three levels of protection on every state (susceptibility adjusted). That makes 384 compartments, over a year in days (`covid_v4.py`, `covid_summer2.py`).

| | summer2 | v4 |
|---|---|---|
| Contacts and population | Fetched and passed by hand (AuTuMN keeps its own inputs database) | `ss.contacts('Philippines')` (bundled, 177 countries) and `people=ss.people('Philippines', year=2021)` (UN WPP), one line each |
| First run | 3.6 to 4.3 s | 0.2 to 0.56 s |
| Repeated run | 0.29 to 0.41 s (SciPy); 0.23 to 0.38 s with the jitted runner | 0.08 to 0.5 s |
| 50 values of `beta` | 11 to 19 s (runner, in a loop) | 10.6 s (`ss.parallel`) |
| Agreement (no ageing, equal split) | | Susceptibles at the end are identical to the person (64,329,070); deaths 280,765 against 280,835 |

(Timings are ranges because a NUTS job was running on the same machine for part of the time.) The engine holds up at this size and starts much faster. Getting there, though, I hit two silent traps and one surprising default:

- **`stratify(immunity=[...])` silently does nothing.** `immunity=` is a keyword of `stratify()` (`transitions.py:1818`, for immunity between strains). The levels are taken as that option, no dimension is made, and the `split=` and `adjust={'rel_sus': ...}` go nowhere. There is no error. The model had 128 compartments rather than 384, and deaths were 470,798 rather than 280,765, until I renamed the dimension (`relsus_min4.py`: `immunity` has no effect, `imm` and `protection` work). "Immunity" is the name an AuTuMN COVID model gives this stratification.
- **An explicit age `split=` on the disease was ignored**, because its levels matched the sim's age groups from `ss.contacts(...)` (`covid_eq_noage.py`: the initial shares stayed at 0.05 a group and 0.25 for 75+). The spec says a disease stratified by an existing people property "does not split anything again" (`05-dimensions.md:97`), but dropping a split the user gave, without a note, is a bug.
- **The default age split is uniform from 0 to 100**, so a 75+ group holds 25% of people. summer2's default is equal shares, and AuTuMN always passes census numbers. `describe()` does mark it ("[guessed: ages uniform from 0 to 100]", `describe_age.py`), which is the design working as intended, and `ss.people(country)` fixes it. So this is a default to know about, not a bug.

To be fair to v4, summer2 has the matching trap. `AgeStratification` "assumes that each of the model's time steps represents a year" (its docstring), so in a model in days it ages everyone by a group every five days. I fell into it. v4's units made the same model correct.

### 4. Run the same model as a stochastic model or as agents

summer2 has no answer to this. In v4, the TB model above runs under other solvers with one keyword (`tb_solvers_dt.py`). The notifications in 1990, 2010, and 2024:

| Step | ODE | `binomial` | `tau` | Agents (50,000) |
|---|---|---|---|---|
| 1 year | 2,341 / 2,260 / 2,195 (0.18 s) | 4,491 / 5,593 / 5,359 (0.07 s) | 2,302 / 2,399 / 2,288 | 4,040 / 4,380 / 4,340 (4 s) |
| 0.1 year | the same | 2,537 / 2,532 / 2,431 | 2,291 / 2,276 / 2,222 | 2,280 / 2,120 / 3,040 (43 s) |
| 0.02 year | the same | 2,436 / 2,243 / 2,207 | 2,297 / 2,194 / 2,337 | 2,460 / 2,140 / 2,000 (275 s) |

This is a real capability for a TB modeler working on a small Pacific island population, where chance matters. Active case finding is one more line: `ss.treat(ss.Treatment(efficacy=0.7, transition='detection', cost=5), eligible=I, coverage=0.5, years=[2026, 2027])` cut TB cases in 2028 to 2035 by 3.4%, and 100 chain-binomial runs of each scenario took 22.5 s (`tb_acf.py`). But at the yearly step this user always uses, the binomial solver and the agents give about twice the ODE's notifications. All v4 says is "stabilisation has rate × dt = 2 ..., so 0.86 of people leave in a step; sim.check_dt() shows how much a smaller step changes the results". That is accurate, but it is only a note, for a result that is wrong by a factor of two.

## What would win them over

- **The same model in a fifth of the code, with outputs for free.** The TB model is 25 lines and agrees to 6e-6. In the vignette, seven rounds of output requests become `select()`.
- **No per-model JIT wait, at their scale.** At 384 compartments, the first v4 run took 0.2 to 0.56 s against summer2's 3.6 to 4.3 s, and repeated runs and sweeps were about the same speed.
- **Calibration as one call.** No hand-built JAX pipeline, and no finding out that reverse mode breaks on a sigmoid or that `odeint` refuses forward mode. The gradient is checked against finite differences, and the method choice is printed.
- **Units that cannot be mixed up.** `ss.days`, `ss.peryear`, and ageing from group widths in years remove the `AgeStratification`-in-days trap, which I fell into in summer2 myself.
- **Prem matrices and UN populations built in**, which AuTuMN maintains as its own inputs database.
- **One model, several solvers**, plus interventions as deliveries with costs. Stochastic runs for small populations and agents for contact tracing are things summer2 cannot do at all.

## What would stop them

| Gap | Kind | Fixable? (size) | Best point on the tradeoff? | In a review? |
|---|---|---|---|---|
| NUTS falls back to Python in the loop whenever a fitted `ss.par` multiplies an `ss.timeseries` (927 s and a failed posterior; the run with whole-number data had not finished after 25 minutes, when the timeout stopped it) | Bug | Yes, small (`ss.gradients.Density` builds that product differently from the run) | n/a | No |
| A likelihood for counts with data that are not whole numbers gives `-inf` in the sim with no error | Bug (message) | Yes, small: `E7xx` at `calibrate()` | n/a | No |
| JAX differentiates only the run; the steps either side are central differences, at two restarts and two passes over the results for each parameter (`implementation.md:263`), so a gradient's cost grows with the number of parameters, where reverse mode's does not. For this user's 20 to 30 parameters that matters, and the 2-parameter fit already took 69 evaluations | Missing feature | Yes, medium (one JAX function from parameters to log-density, then `jax.grad`) | No. The better point is a log-density the user can also hand to numpyro or estival themselves | F2 Q24 (`jacfwd` of a `Density`-style function) |
| `dump_summer2.py` fails on any named `Parameter` (`.name` against `.key`), cannot be given parameter values, and drops time functions and plain-array mixing matrices | Bug / missing | Yes, small to medium (fix the attribute, take `pars=`, map `get_sigmoidal_interpolation_function` and `get_linear_interpolation_function` to `ss.timeseries`, accept arrays) | n/a | F 10 covers the importers generally, not these failures |
| `stratify(immunity=...)` silently makes no dimension | Bug | Yes, small (an error for a list given to `immunity=`, or rename the option to `cross_immunity=`) | n/a | No |
| An explicit `split=` on a disease-level age stratification is ignored when the sim has the same groups | Bug | Yes, small (use it, or raise) | n/a | No |
| No sigmoidal scale-up; `number * ss.timeseries` fails | Missing feature | Yes, small (`mode='sigmoid'` with summer2's curvature) | n/a | No |
| A weight by level in an outcome fails (vignette finding 11) | Bug | Yes, small | n/a | F 12 |
| The stochastic solvers double TB notifications at a yearly step, with only a note | Inherent (step size), partly fixable | Make the note a warning, with the size of the change, when `check_dt()` would move the results by more than about 10% | Nearly: the notes are accurate. The better point is to say how big the change is | F2 05 (the binomial step question) |
| Docs are all Starsim 3 | Docs | Yes, large | n/a | All four |
| The pipeline is not JAX end to end: kernels in Numba, JAX only for gradients; users cannot `jit` or `vmap` their own loss over the model | Inherent tradeoff (one general engine against a JAX-native graph) | Partly: expose the log-density and the run as JAX functions | Close. Numba kernels are the right default (no per-model JIT wait), but this user wants the JAX function as an escape hatch | F README:55 (drop `integrator=`, pick JAX internally) |
| Ecosystem: estival, the AuTuMN inputs database and country pipelines, their own papers and code | Ecosystem | Not by the library | Inherent | No |

## Verdict

- **Today:** no for an AuTuMN group with live country models, and maybe for a new project. The port is easy and exact and the outputs are far simpler, but the Bayesian calibration this user depends on breaks on their standard pattern, there is no automatic import of a real model, and the docs are for Starsim 3.
- **After the four reviews' fixes:** closer, but still not clear. The docs and the gradient rewrite (F2 Q24) are covered. The NUTS-with-time-series bug, the `immunity=` trap, the ignored split, and the summer2 dumper are not.
- **A clear win would need:** NUTS that works through JAX for time-varying fitted rates, at reverse-mode cost; an importer that takes a live AuTuMN model with its parameter dict and its time functions; and docs. Then v4 does everything summer2 does with less code and no per-model JIT wait, and adds stochastic and agent-based runs and costed interventions, which summer2 cannot do.

## The demo that would sell it

```python
import starsim as ss
from my_autumn_model import build_model, params        # an existing summer2 TB model and its parameter dict

sim = ss.load(build_model(), pars=params, data='notifications.csv')   # Parameters, sigmoid scale-ups, and mixing all carried over
sim.describe()                                          # what was read, and every guess marked
fit = sim.calibrate(likelihood=dict(notifications=ss.nbinom(n=ss.par(20, bounds=[1, 200]))))   # NUTS through JAX, in seconds
acf = ss.treat(ss.Treatment(efficacy=0.7, transition='detection', cost=5), eligible=sim.diseases.tb.I, coverage=0.5, years=[2026, 2027])
runs = ss.parallel(baseline=fit.sim, acf=fit.sim.new(interventions=acf), solver='binomial', n_runs=200)
runs.plot()                                             # the calibrated model, its posterior, and ACF with chance, on an island of 60,000
```

What v4 needs for this: `ss.load` of a live summer2 model that takes its parameters (the dumper fixes above), with time functions carried over; NUTS whose JAX density handles a fitted parameter times a time series; and posterior draws passed on to the scenarios. **It does not run today.** The import fails on the `Parameter` attribute and NUTS falls back to Python. The last three lines work now on the hand-written model (`tb_acf.py`).
