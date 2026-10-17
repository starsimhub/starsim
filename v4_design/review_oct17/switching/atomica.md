# Atomica: would its users switch?

## Who this user is

Atomica has two users, and they need different things. The first is the **country analyst**: someone in a ministry of health, a national AIDS or TB program, or a consultancy working for the World Bank or the Global Fund. They do not write Python. They fill in a databook (an Excel workbook of population sizes, prevalence, cascade counts, and rates by population and year) and a program book (each program's unit cost, spending, who it reaches, and what full coverage does to each parameter). Then they run calibration, scenarios, and budget optimization, usually through a web front end built on Atomica (Optima TB and the Cascade Analysis Tool, from my knowledge; I did not check them here). What they get out is an optimized allocation table and a care-cascade chart for a report. The second is the **Burnet Institute modeler**, who writes the framework workbook (compartments, the transition matrix, parameter formulas, characteristics) and Atomica's Python glue, and supports the country teams. Both chose Atomica for the same reason: a model whose structure and data live in spreadsheets that a non-programmer can edit, with **program costing and allocative-efficiency optimization** built in. The rest of Atomica (a fixed-step difference equation, populations without networks) is plumbing. Atomica 1.32.4 loads and runs each of its demos in 0.04 to 0.5 s, its own optimizer reallocates the HIV demo's budget in 1.3 s, and the Optima models built on it have years of published country analyses behind them.

**Plainly: v4 is not aimed at this user, and it says so.** `17-vocabulary.md:271` lists "Models written in YAML, JSON, or spreadsheets" as not in the language ("Python; the model spec is produced from it"), and `:284` lists "Budget optimization" (with "`cost=` now; optimization later"). `08-interventions.md:65` puts budget optimization out of scope for v1, and the generated databook (`sim.requires(to='form.csv')`) is deferred (`19-open-questions.md:197`, `16-interop.md:98`). v4 is aimed at the Burnet modeler, and at them only as a Python programmer. Whether it *should* aim at this user is answered under the verdict. In short: not at the ministry analyst directly, but yes at the program-and-budget layer, which is small given what v4 already has.

## The first day

| Vignette | Original | v4 | Agreement |
|---|---|---|---|
| `sir` | 62 lines (59 spreadsheet rows, 3 Python); 0.0024 s a run | 17 lines; 0.033 s first run, 0.0005 s repeated | Identical to rounding |
| `hiv_dyn` (cascade by sex, 7 programs, 3 budgets) | 217 lines (188 spreadsheet rows, 29 Python); 0.0054 s | 67 lines (12 over 100 characters); 0.18 s | Identical, largest difference 2e-14, in all four runs |

Both numbers come from `vignettes/atomica/*/README.md`. The agreement is real and impressive. In particular, matching the HIV demo needed v4 to adopt Atomica's rule of scaling outflows down together when they would take more than everyone (finding 1 there, now fixed).

The analyst would not count these lines the way the table does. 188 spreadsheet rows that a program officer fills in are not comparable with 67 lines of Python such as this one (`vignettes/atomica/hiv_dyn/model.py:34`):

```python
outcome = lambda base, **effects: sum(cov[k]*sex(*np.subtract(v, base)) for k,v in effects.items()) + sex(*base)
```

The Burnet modeler would recognize the model: the transitions are the rows of the transition matrix, one per parameter, and the cascade is three sums of states. But they would see that the programs (the part they built Atomica for) are lambdas, a dict of tuples, and a function of the sim (`everyone()`, finding 2, still open). That function also takes the run out of the kernels, so a repeated run cannot be restarted. The SIR vignette needs `integrator='euler', substeps=1` to match, and the constant `0.008` is written twice because a typed rate cannot be subtracted from 1.

`docs/` is entirely Starsim 3 (all four reviews say so: F README:49, F2 README:57, O README:38). The analyst would find nothing at all for them. The modeler would find `ss.SIR(beta=...)` tutorials that teach none of what the vignettes use. The only accurate guide to bringing an Atomica model across is an AI skill, `skills/import-model/references/atomica.md`, which is good (it states the Euler step, the `dt`, and the flow-shift convention).

## The second week

Scripts and outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/atomica/`.

### 1. Bring my existing workbooks across

The modeler's first real task is to load the frameworks and databooks they already have. I ran all 16 of Atomica's shipped library demos with `ss.load(framework, format='atomica', databook=..., dt=...)` and compared every compartment with Atomica's own baseline run (`dump_demos.py` in Atomica's environment, `import_demos.py` in v4's).

| Result | Demos |
|---|---|
| Imports, runs, agrees to 1e-12 or better | `sir` (7e-16), `tb_simple`, `tb_simple_dyn`, `usdt`, `hiv`, `hiv_dyn` (1e-14), `hypertension`, `hypertension_dyn` (1e-12), `cervicalcancer` (2e-13) |
| Imports and runs, but wrong | `sir_vaccine`: `vac` off by 35%. The databook does not fix every initial compartment; v4 solves by nonnegative least squares where Atomica uses its setup weights. This is listed as `W1601`, so it is not silent |
| Fails with a raw Python exception | `udt`, `udt_dyn`, `diabetes`, `dt`: `ValueError: The model spec uses "num_diag", which is not a state, parameter, or transition`. Every one of their parameters is skipped as having "no value in the databook and no default" (even `diabetes`'s `birth_rate`, whose framework default is 15,828), so the databook reader is missing a layout these use. `combined`: `KeyError: 'undx'` (`atomica.py:136`). `tb`, Atomica's flagship: `SyntaxError` on `life_expectancy*:ddis` (`atomica.py:209`), Atomica's syntax for referring to a flow, and the framework also uses `SRC_POP_AVG(...)` |

So 9 of 16 import exactly and load in 0.03 to 0.13 s, which is a strong base. But the TB demo, which is the one a TB program would bring, does not load. The failures are tracebacks, not coded errors, and nothing imports the program book: `ss.load(..., progbook=...)` lists "programs ... are not translated" (`atomica.py:163`).

### 2. Calibrate to the databook's data

Atomica's routine is `P.calibrate(adjustables=[...], measurables=['ch_prev'])`, which scales parameters by "y-factors". On the SIR demo, fitting `transpercontact` and `recrate` to the databook's 19-year prevalence series:

| | Atomica `P.calibrate` (ASD) | v4 `sim.calibrate()` (L-BFGS-B with JAX gradients) |
|---|---|---|
| Time | 0.6 s | 1.9 s (hand model), 2.3 s (imported model) |
| Fit (mean absolute error / mean of data) | 4.5% | 0.011% |
| Values | 0.00298, 4.48 | 0.00142, 6.16 |

The v4 code is what the spec promises: bounds on the two `ss.par`s, a dataframe with a `prevalence` column, and `sim.calibrate()`. It picked its method and printed it ("Fitting by l-bfgs-b (gradients by automatic differentiation)"). This is a genuine gain: a much better fit in about two seconds, and NUTS is one keyword away. (Atomica's ASD stopped at its own convergence criterion in 0.6 s of an allowed 30 s, so this is its default behavior, not a truncated run.)

Starting from the workbooks is harder. `ss.load` returns a sim that is already initialized, so setting `sim.pars.diseases.pars.transpercontact.bounds = [...]` afterwards is silently ignored, and `calibrate()` raises `E1301 NothingToFit` (`v4_calib_import.py`, `dbg.py`). Rebuilding with `ss.Sim(imported_disease, ...)` fails with `KeyNotFoundError: Key(s) ['SIR'] not found`. What worked was `sim.to_code()`, adding `bounds=` to two lines of the generated file, and running it (`sir_imported_fit.py`). That is fine for the modeler and impossible for the analyst.

### 3. Optimize the budget

This is the task Atomica exists for: reallocate the HIV demo's current $164.2 million across its seven programs from 2020 to maximize the number virally suppressed in 2030. In Atomica (`at_optim.py`) this is about ten lines of standard API (`SpendingAdjustment` for each program, `MaximizeMeasurable('all_vs', [2030])`, `TotalSpendConstraint()`, `at.optimize`). It took **1.3 s** and raised viral suppression from 374,830 to 1,067,056, mostly by moving money into same-day initiation counseling.

v4 has no optimizer and no declared programs. I copied the vignette, replaced its single `budget` multiplier with one `ss.par` of spending per program (two changed lines, `hiv_budget_model.py:19,32`), and wrote a SciPy SLSQP loop over the spending shares with a sum-to-one constraint (`v4_optim.py`, 25 lines):

```python
def vs2030(x):
    sim = m.sim.new(pars={f'spend_{k}': float(v) for k,v in zip(names, x)}, verbose=0)
    sim.run()
    return float(sim.results.hiv.n_vs[i2030])
res = spo.minimize(lambda x: -vs2030(x*total)/1e6, alloc/total, method='SLSQP', bounds=..., constraints=[dict(type='eq', fun=lambda x: x.sum() - 1)])
```

Run with Atomica's optimal allocation, v4 gives exactly Atomica's 1,067,056, so the engines agree. The SLSQP search found **1,083,839** (1.6% more than Atomica's optimum, under the same bounds), but it took **378 runs and 51.5 s, 40 times Atomica's time**. Each run costs 0.11 s, against Atomica's 0.005 s, because every evaluation rebuilds the sim, and `sim.restart()` is blocked by `everyone()`. No analyst could write this loop. The modeler could, but they would be rebuilding what `at.optimize` already gives them: money minimization, spending bounds, time-varying allocations, and coverage interactions.

### 4. Cost and coverage with v4's own pieces

v4 does have the beginnings of programs: products carry `cost=`, and deliveries have `capacity=` and report `delivered`, `eligible`, `coverage`, and `cost` (`v4_cost.py`). A treatment of a constant 2,000 people a year at $100 ran as an ODE in 0.09 s and reported its total cost ($40,875). But the natural Atomica spelling, capacity as spending over unit cost, fails:

- `capacity=ss.peryear(budget/unit_cost)` raises `TypeError: Value must be a scalar number or array, not BinOp` when the model is written.
- `capacity=budget/unit_cost` fails at run time: `TypeError: Cannot use the expression "(200000 / 100 * 1 < 384.583)" where Python needs a single True or False`.
- A `budget` par used only by an intervention is not a parameter of the sim, so `sim.new(pars=dict(budget=...))` raises `E602 UnknownParameter ... Parameters are sir.dur, sir.rate`, which also shows odd auto-generated names.
- With the force of infection written with counts (as Atomica writes it), `ss.Treatment(efficacy=0.9)` raised `E505 ... needs a transition out of the infected states of "sir" ... and there is none`, although `recovery = inf >> rec` exists, and the sim fell back to 1,000 agents. Naming `transition='recovery'` fixed it.

## What would win them over

- **Exact agreement, from their own files.** 9 of 16 library workbooks import in under 0.15 s and match Atomica to 1e-12, and the hand-translated HIV demo matches across all three budgets to 2e-14. The modeler can trust the engine from day one.
- **Better calibration, with uncertainty available.** The same `bounds=` plus `data=` plus `sim.calibrate()` gave a 400 times smaller error than Atomica's y-factor ASD on its own demo, in two seconds. A posterior by NUTS is one keyword further, which Atomica does not offer.
- **A better optimum, once optimization exists.** SciPy's SLSQP over v4 found 1.6% more people suppressed than Atomica's ASD. A gradient through the JAX run would make this faster than finite differences.
- **The model is one readable object.** `sim.to_code()` turns a workbook into a Python file that says how each rate was read, and `ss.diff` compares two versions. For the modeler, this is a cleaner artifact than a framework workbook.
- **Delivery results by stratum, including cost.** Costs by population come with no post-processing, which is the start of cost-effectiveness.

## What would stop them

| Gap | Kind | Fixable? (size) | Best point on the tradeoff? | In a review? |
|---|---|---|---|---|
| No declared programs: spending to coverage to parameter change has no form; the progbook is not imported; the vignette needs lambdas | Missing feature | Yes, medium (a `ss.program(cost=, spend=, reaches=, effects=)` built on `Delivery(capacity=)` and `ss.modify` with an expression as its amount; plus a progbook reader); the vignette's finding 4 sketches it | No. `cost=` alone is not "the representation a health ministry needs" (`best-practices/interventions.md:282`); the better point is programs as data, with optimization optional | Vignette finding 4 only; none of the four reviews |
| No budget optimization; by hand it is 25 lines of SciPy and 40 times slower | Missing feature | Yes, medium (`ss.optimize(sim, spend=[...], total=, maximize=, at=)` over the spending pars; SLSQP worked well here) | No, for this user. It is deferred by design (`17-vocabulary.md:284`) | No |
| Spending cannot drive a delivery: `capacity=` rejects an expression of an `ss.par`, and a par used only by an intervention cannot be set by `pars=` (`E602`) | Bug / missing | Yes, small | n/a | No |
| Each run rebuilds the sim (0.11 s against Atomica's 0.005 s); a count over everyone needs a function of the sim, which blocks the kernels and `restart()` | Missing feature (`.n` within a group or over everyone) | Yes, small to medium | No; the vignette's suggestion (`sus.n` total, `sus.n.within('sex')`) is the better point | F 03 (cites `hiv_dyn` `kind='space'`) |
| No spreadsheet path: no databook out, no values read back by name, an imported sim cannot be given bounds without `to_code()` | Inherent tradeoff (Python source against spreadsheets), partly fixable | A generated databook and a reader of it, medium; bounds on an imported sim, small | No. Keeping Python as the source is right, but a form generated from `sim.requires()` and read back by name would serve the analyst without a second language | No (the deferral is in the spec) |
| 6 of 16 library workbooks fail with tracebacks, including the flagship TB; `sir_vaccine` is 35% off | Bug | Yes, small to medium (a databook layout, `:` flow references, `SRC_POP_AVG`, several frameworks in one workbook) | n/a | F 10 covers the importers generally (and the `floor()` rewrite), not these failures |
| Docs are all Starsim 3; nothing for a non-programmer | Docs | Yes, large | n/a | All four |
| No GUI, no country relationships, no track record of national allocative-efficiency studies | Ecosystem | Not by the library | Inherent; a front end is a separate product | No |
| The reviews propose dropping `integrator=` (F README:55, O README:104); `integrator='euler', substeps=1` is how v4 matches Atomica | Risk from a proposed fix | Keep a spelling for "Atomica's step" (`solver='euler'` or `dt` with `substeps=1`) | Yes, if a spelling stays | Interaction not noted |

## Verdict

- **Today:** no. The analyst cannot use v4 at all, and the modeler would gain a better calibrator but lose the progbook, the optimizer, and the spreadsheet workflow their users depend on.
- **After the four reviews' fixes:** still no. The reviews simplify the core and fix its docs, but none touches programs, optimization, or spreadsheets, and moving importers to a companion package moves the one Atomica bridge further away.
- **A clear win would need:** declared programs (spending to coverage to effect, read from a progbook), `ss.optimize` over spending, and a databook generated from the model and read back. With those, v4 is a better engine than Atomica's (better fits, posteriors, stochastic runs), and an Atomica-style front end could sit on top. Should v4 aim at this user? Not with a GUI, and not by writing models in spreadsheets. But it should aim at the program-and-budget layer: it is perhaps 500 to 800 lines on top of `cost=` and `capacity=`, and it is what makes a model useful to the people who fund health programs.

## The demo that would sell it

```python
import starsim as ss

sim = ss.load('hiv_dyn_framework.xlsx', databook='hiv_dyn_databook.xlsx', progbook='hiv_dyn_progbook.xlsx', dt=1)
fit = sim.calibrate()                                   # bounds from the databook's ranges, data from its years
best = ss.optimize(fit.sim, total=164.2e6, start=2020, maximize='all_vs', at=2030)
best.plot()                                             # the allocation, and the cascade before and after
best.to_excel('allocation.xlsx')                        # the table the ministry reads
ss.parallel(best.sim, solver='binomial', n_runs=200).plot()  # how sure we are, in a small province
```

What v4 needs for this: the progbook imported as declared programs; `calibrate()` taking bounds from the databook (or accepting them for an imported sim); `ss.optimize` over spending with a total constraint; and an Excel export of the allocation. **It does not run today.** The first line imports the baseline exactly but drops the programs, and `ss.optimize` does not exist. The last line, which is the part Atomica cannot do, already works: 10 chain-binomial runs of the HIV model with programs take 1.3 s (`hiv_binom.py`; viral suppression in 2035 is 375,922 with a standard deviation of 563).
