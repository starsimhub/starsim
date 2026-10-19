# Inference: calibration, likelihoods, gradients for fitting, model comparison

I reviewed `v4_design/design/13-inference.md` (with the inference rows of `19-open-questions.md`, the `fit` field of `14-model-spec.md`, the `ss.par` fields of `04-parameters.md`, and the recommendations of `best-practices/calibration.md`), and read `starsim/calibration.py` line by line (2587 lines: lines 26 to 1172 are Starsim 3's `Calibration`, `CalibComponent`, and its five likelihood components; lines 1174 to 2587 are v4's `data`, `compare`, `lineup`, `mismatch`, `loglik`, `ParticleFilter`, `pmmh`, `nuts`, `nuts_jax`, the R-hat/ESS code, `Fit`, `fit`, `ModelComparison`, and `compare_models`), the fitting parts of `starsim/gradients.py` (1121 lines; `reason()`, `gradient()`, and `Density`, lines 797 to 1121), `sim.calibrate()`/`sim.new()`/`sim.restart()` in `starsim/sim.py`, `ss.par` in `starsim/transitions.py:174`, and the tests (`tests/test_calibration.py`, 405 lines, plus the v4 fitting tests in `tests/test_transitions.py` and `tests/test_gradients.py`). I ran six scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/inference/`: an SIR fitted to synthetic weekly cases as an ODE (L-BFGS-B with JAX gradients, and Nelder-Mead), as `sde` and `binomial` (Optuna), as 2,000 agents (Optuna), and as an ODE with a Poisson likelihood (NUTS); the `fit.pars` round trip; the import surface; and a set of edge cases (unknown `gof` names, `method=[...]`, products in `pars=`, a parameter inside a distribution, Starsim 3 sims, `ss.Calibration` on a declared sim). Output is quoted where a claim rests on it.

## Summary

1. **Collapse the two calibration systems into one entry point.** `sim.calibrate()` works only for declared models (`sim.new()` refuses anything else), and Starsim 3's `ss.Calibration` needs a `build_fn` even for a declared model. Give `ss.fit()` two escape hatches, a `set=` function and a `gof=` function, which is all `ss.Calibration` adds, and delete `Calibration`, `CalibComponent`, and the five component classes (1,150 lines). Breaking for Starsim 3 calibration scripts, with a mechanical migration.
2. **Make `fit.pars` round-trip reliably.** `ss.Sim(sir, pars=fit.pars)` fails whenever a fitted parameter is named `dur` (or any other sim parameter), because `ss.Sim` decides what `pars=` means by sniffing its keys. Give model values their own keyword (or fold them into `fit.sim.new()`, which already works) and stop sniffing. Breaking for the spec's own one-page example.
3. **Rewrite the method table in the spec to what the code does**, which is a five-line rule: ODE without likelihood → L-BFGS-B (Nelder-Mead without JAX); ODE with likelihood → NUTS; stochastic compartmental with likelihood → particle MCMC; anything else stochastic → Optuna. The spec's ABC-SMC, "cheap enough" timing, and simplex-first rows describe a method that does not exist. Not breaking.
4. **Cut what the spec promises and the code does not deliver, or deliver it minimally.** Staged fitting (`method=[...]` raises `AttributeError`), `sim.sensitivity()` (does not exist), convergence reporting for optimizers (a SciPy `result` is discarded; `maxiter=1` passes silently), `W1303`, and the printed summary block. Either is fine; the half-state is not. Not breaking.
5. **Validate what the user passes.** `gof='foo'` is silently the default measure; `pars=[beta * contacts]` says "no parameter with bounds"; a parameter inside a distribution dies with `float() argument must be ... not 'BinOp'`; a Starsim 3 sim says "`sim.new()` is only available for sims with declared states". Each needs a one-line check. Not breaking.
6. **Split `fit()` (304 lines, seven method branches, three coordinate systems) into one small function per method**, and drop one of the two NUTS paths. The codebase finding with the most leverage; not user-visible.

## Simpler for the user

### 1. Two calibration systems that do not meet

**The issue.** A user with a declared model writes `sim.calibrate()`. A user with a Starsim 3 model (`ss.SIR()`, a class with `step()`) cannot: 

```text
=== Starsim 3 sim .calibrate() ===
error: ValueError sim.new() is only available for sims with declared states and transitions
```

They must use `ss.Calibration(sim, calib_pars=..., build_fn=..., components=[ss.Normal(...)])`, the thirty-line form the spec itself holds up as the thing to replace (`13-inference.md`, "Compared with Starsim today"). Going the other way, `ss.Calibration` on a declared model fails with no `build_fn`:

```text
=== v4 declared sim through ss.Calibration ===
error: TypeError 'NoneType' object is not callable
```

So the two systems are disjoint, not layered: the user has to know which kind of sim they have to know which API to learn, and the two have different vocabularies for the same ideas (`calib_pars` low/high/guess vs `ss.par(bounds=)`; `conform='prevalent'|'incident'` vs `summarize_by`; `components` vs `gof`/`likelihood`; `check_fit()` vs `fit.summarize()`; `calib.best_pars` vs `fit.pars`).

**Why it matters.** Starsim 3 class-based modules are, per `00-overview.md`, still first-class ("Subclassing adds behavior. It does not add vocabulary"). Every model in `starsim/library/` is one. Those users get the old API and none of the new fitting methods, printed reasons, identifiability notes, or `Fit` object.

**Proposal.** One function, `ss.fit()`, with two optional escape hatches that cover everything `ss.Calibration` does:

```python
fit = sim.calibrate(
    pars = dict(beta=[0.01, 0.3], n_contacts=[2, 10]),   # bounds by name, when the parameters are not ss.par
    set  = lambda sim, pars: ...,                        # how to put values into a sim that has no graph (today's build_fn)
    gof  = my_component,                                 # any callable sim -> number (today's eval_fn or a component)
)
```

Only `set=` is new code: `fit()` already takes `pars=`, already accepts a callable `gof=` (`calibration.py:1382`), and already builds a fresh sim per evaluation when it cannot restart (`calibration.py:2211-2217`). The only blocker is `base.new()` at `calibration.py:2130`, which should fall back to `sc.dcp(sim)` for an undeclared sim. Then `ss.Calibration`, `CalibComponent`, `BetaBinomial`, `Binomial`, `DirichletMultinomial`, `GammaPoisson`, `Normal`, `linear_interp`, `step_containing`, and `linear_accum` can go (lines 26 to 1172). If the component likelihoods are wanted, they are four `logpdf` branches away (`calibration.py:1418`): `ss.betabinom` and `ss.dirichlet_multinomial` could join `ss.poisson`, `ss.nbinom`, `ss.normal`, and `ss.lognorm_ex` as `likelihood=` values.

**Cost.** Breaking for every Starsim 3 calibration script (`docs/user_guide/workflows_calibration.qmd`, `workflows_sir_calibration.qmd`, STIsim, and the tutorials). The migration is mechanical (`calib_pars` → `pars=`, `build_fn` → `set=`, a component → `gof=`), and `17-vocabulary.md` already lists `ss.Calibration` as something "users rarely construct". Optuna's parallel workers with an SQLite study (`Calibration.run_workers`, `make_study`) would be lost unless `fit(method='optuna', n_workers=)` grows them; that is the one real feature only the old class has. **Confidence: high** that the two should merge; medium on the exact keywords.

### 2. `pars=fit.pars` does not round-trip when a parameter is called `dur`

**The issue.** The spec's one-page model ends with `ss.Sim(sir, pars=fit.pars, ...)`. `ss.Sim` decides whether `pars=` is sim settings or model values by checking whether any key is a sim parameter (`sim.py:77`: `if ... not any(key in self.pars for key in pars): self.par_values, pars = dict(pars), None`). `dur` is a sim parameter (the run length), and the spec's own examples name the dwell time `dur` (`04-parameters.md:34`, `13-inference.md`, `test_gradients.py`). So:

```text
=== fitted par named "dur" ===
fit.pars: {'beta': 0.299999719890412, 'dur': days(10)}
round trip FAILED: KeyNotFoundError Key(s) ['beta'] not found; available keys are ['label', 'verbose', 'n_agents', ...
fit.sim.new() keeps fit: True
=== fitted par named "recovery_time" ===
round trip OK; max diff n_I: 0.0
```

**Why it matters.** The failure is silent in the design and loud at run time, with an error message about sim keys that gives no hint of the cause. Any other coincidence of names (`start`, `unit`, `label`, `people`) does the same. This is exactly the kind of guess the spec elsewhere refuses to make ("Formulas in strings... needs a strong reason").

**Proposal.** Either (a) a separate keyword for model values, `ss.Sim(sir, values=fit.pars)`, so nothing is sniffed, or (b) make `fit.sim.new(...)` the documented way to build the calibrated sim with changes (it already keeps the fitted values, as shown), and drop `pars=fit.pars` from the spec. (b) removes a concept; (a) keeps the spec's picture. `fit.pars` keys could also be module-qualified (`sir.dur`) when they collide, but that moves the problem.

**Cost.** (a) is breaking for `pars=dict(beta=0.1)` and the spec's page; (b) breaks nothing and changes two lines of spec. **Confidence: high.**

### 3. The method is chosen predictably, but the spec describes a different chooser

**The issue.** The rule is `default_method()` (`calibration.py:2072-2092`), and it is five lines. Tested on the same SIR under each solver (`exp1_ode.py`, `exp2_lik.py`):

| Sim | Likelihood | Printed |
|---|---|---|
| `solver='ode'` | no | `Fitting by l-bfgs-b (gradients by automatic differentiation; override with method=)` |
| `solver='ode'` | yes | `Fitting by nuts (every data series has a likelihood, so a posterior, with gradients by automatic differentiation; ...)` |
| `solver='sde'`, `'binomial'` | no | `Fitting by optuna (a stochastic sim; override with method=)` |
| `solver='sde'` | yes | `pmcmc` ("a stochastic compartmental sim whose data series all have a likelihood, so a posterior, from particle MCMC") |
| `solver='abm'`, 2,000 agents | no | `Fitting by optuna (a stochastic sim; override with method=)` |

That is predictable and the reason is printed. But `13-inference.md` says: ABC-SMC for stochastic compartmental and cheap agent-based sims; "cheap enough is decided by timing one run against the default budget"; Nelder-Mead "when a gradient is not available". None of that is in the code (`implementation.md` records "No ABC-SMC" as a departure, but the spec text and both example transcripts still describe it). The printed block the spec shows (`model: ...`, `method: ...`, `goodness of fit: ...`, `converged: ...`) is not what is printed either; the code prints "Searching on a log scale: beta" and "Fitting by ...", and `fit.summarize()` prints the table.

**Why it matters.** The reader of the spec expects intervals from an agent-based fit (the second transcript shows `beta 0.071 [0.052, 0.094]`); the code gives a point estimate and, correctly, no interval. The spec's reason for the spread is the thing it says "MUST NOT" be presented.

**Proposal.** Replace the table and both transcripts with the five-line rule and the real output. One further simplification worth considering: for the ODE case with one or two parameters, L-BFGS-B with JAX took 1.15 s (25 evaluations, mostly compilation) against 0.15 s for Nelder-Mead (45 evaluations), which `implementation.md:221` also records (1.7 s vs 0.25 s for three parameters). The rule could say "gradients when there are four or more parameters", or the spec could simply note the compile cost. **Confidence: high** that the spec should change; low that the threshold is worth the extra rule.

### 4. The default goodness of fit (B1): a sensible answer, but names are not checked

**The issue.** The implementation answers B1 with the normalized absolute error, exactly as specified (`mismatch()`, `calibration.py:1369-1393`): mean |model − data| over the mean of |data|, falling back to the range and then to 1.0; series averaged with `weights=`. It recovered `beta=0.3` from `0.15` to 1e-11 on the ODE. This is defensible: no noise model, robust to one bad week, scale-free across series. Two caveats:

- An unknown name is silently the default: `gof='rmse'` is the only string the code compares against (`calibration.py:1389`), so `gof='foo'`, `gof='mae'`, `gof='nrmse'` all run and return NAE: `no error! mismatch 1.187e-06 equals nae? True`.
- A series whose data are all zero (common for deaths early on, or a stratum with no cases) gets scale 1.0 and so is measured in raw counts, which can swamp the other series: `all-zero data: mismatch = 273.86`.

**Proposal.** Check the name (`if gof not in ['nae', 'rmse', None]: error`), and either document the zero case in the spec's goodness-of-fit paragraph or scale a zero-mean series by the mean of the model values instead. **Confidence: high** on the check; medium on the zero case.

### 5. `bounds=` as the only way to mark a parameter fittable

**The issue.** `ss.par(value, bounds=[lo, hi])` is the sole marker. Good: it is one field, in the one place the parameter lives, and `fit()` finds every one of them, including those inside `init=`, inside `ss.seasonal`, shared across modules, and inside a likelihood (`calibration.py:2144-2161`). Four rough edges:

- `bounds` means both "valid range" and "search range" (`04-parameters.md:47` says this is deliberate). A user who wants bounds as a sanity check but does not want the parameter fitted has to pass `pars=` to exclude it, every time.
- A `prior=` on a parameter is silently ignored by every method except NUTS and PMCMC. With a likelihood and `method='l-bfgs-b'`, with and without `prior=ss.lognorm_ex(mean=0.3, std=0.2)`: `no prior: beta 0.2936, dur 11.0055 / with prior: beta 0.2936, dur 11.0055 (same => prior ignored by l-bfgs-b)`. A MAP estimate (add `-logprior` to the objective when every series has a likelihood) is three lines and makes `prior=` mean something for every method.
- A `bounds=` parameter inside a distribution (`dur=ss.gamma(a=3, scale=d)`) fails with `TypeError: float() argument must be a string or a real number, not 'BinOp'`. `implementation.md:229` says this is unsupported, but the user gets no sentence saying so.
- `pars=[beta * contacts]`, which the spec offers for fitting a product (`13-inference.md`, "Fitting a product of parameters"), fails with `E1301 NothingToFit: calibrate() found no parameter with bounds`, which is false and misleading. `implementation.md` lists it as "not yet"; the error should say that.

**Proposal.** Keep `bounds=` as the marker (it is the simplest possible rule). Add MAP when priors exist; raise clear errors for the two unsupported forms. Drop the product-fitting section from the spec until it exists, since `W1302` already tells the user which product the data determine and to "fit one and fix the other". **Confidence: high.**

### 6. Promises in the spec that the user will try and that fail

- **Staged fitting.** `sim.calibrate(method=['nelder-mead', 'optuna'])` → `AttributeError: 'list' object has no attribute 'lower'` (`calibration.py:2209`). The spec gives it a full section, a warning code (`W1303 StageStopped`), and two transcripts. `implementation.md:229` says "not yet" in a parenthesis at the end of the NUTS item.
- **`sim.sensitivity()`.** Listed in `13-inference.md`, `17-vocabulary.md:119`, and `19-open-questions.md`; `hasattr(ss.Sim, 'sensitivity')` is `False`, and I could not find it in `implementation.md` or `deferred.md`.
- **Convergence reporting.** The spec's table promises gradient norm/simplex size/evaluations for least squares and "improvement over the last quarter of trials" for Optuna. The code discards SciPy's `result` after taking `.x` (`calibration.py:2269-2272`) and stores nothing from the Optuna study but `best_params`. `calibrate(method='l-bfgs-b', maxiter=1)` returns a `Fit` with no warning; `summarize()` for Optuna prints the table and "Goodness of fit: 1.105 before, 0.8109 after" and nothing about trials or convergence. Only the posterior methods (R-hat, ESS, divergences, `W1304`) and `W1301 AtBound` are implemented.

**Proposal.** The cheapest consistent state: remove staged fitting and `sim.sensitivity()` from the spec (both can be composed by the user: run one fit, pass `pars=`/initial values to another; LHS over bounds is ten lines with `sc.parallelize`), and add `result.success`/`result.message` and `len(trials)` to `Fit` so `summarize()` can say "converged: yes/no". **Cost:** two spec sections go; nothing in code breaks. **Confidence: high.**

### 7. Error messages for the wrong kind of sim

`sim.calibrate()` on a Starsim 3 sim gives `sim.new() is only available for sims with declared states and transitions`, which names a method the user never called. If finding 1 is not adopted, `fit()` should check `hasattr(sim, 'inputs')` first and say "`calibrate()` fits declared models; for a class-based module use `ss.Calibration`" (or, with finding 1, "pass `set=`"). **Confidence: high.**

### 8. The dependency surface is hidden well

Verified: after `import starsim` (1.30 s), `jax`, `blackjax`, `optuna`, and `seaborn` are all absent from `sys.modules`; JAX is imported inside `ss.gradients.jax()` only when a gradient or NUTS is asked for, and the message when it is missing names the extra (`pip install starsim[jax]`). Without JAX the default falls back to Nelder-Mead with a printed reason; without BlackJAX, to L-BFGS-B with "for a posterior from NUTS, pip install blackjax". This is right and should be kept. One note: `optuna` and `seaborn` are hard dependencies in `pyproject.toml:40-41` although both are only used by calibration; since Optuna is the default for every stochastic sim, keeping it required is defensible, but `seaborn` is used only by the Starsim 3 component plots (`plot_facet_bootstrap`) and would leave with finding 1.

### 9. Timings, for the record

Same SIR, 10,000 people, 121 days, 17 weekly data points, one parameter: ODE L-BFGS-B 1.15 s (25 evaluations, includes JAX compile); ODE Nelder-Mead 0.148 s (45 evaluations); `sde` Optuna 30 trials 0.16 s; `binomial` 0.14 s; 2,000 agents Optuna 30 trials 3.2 s (0.14 s per run); ODE with Poisson likelihood, NUTS, 2 chains of 200 + 200: 4.9 s (2.7 s compiling, 1.1 s sampling), R-hat 1.02, ESS about 75, with a correct `W1302` note that `beta` and `dur` are correlated (r = −0.975) in the posterior. The ODE path through `sim.restart()` rather than a new sim per evaluation is what makes the stochastic compartmental fits this fast.

## Simpler codebase

### 1. Delete the Starsim 3 half of `calibration.py`

Lines 26 to 1172 (`Calibration`, `linear_interp`, `step_containing`, `linear_accum`, `CalibComponent`, `BetaBinomial`, `Binomial`, `DirichletMultinomial`, `GammaPoisson`, `Normal`) are 1,150 lines, 44% of the file, and duplicate the v4 half concept for concept: the three `conform` functions (`calibration.py:467-527`) are `compare()`'s interpolate-or-sum rule (`calibration.py:1330-1335`) written again; each component class carries `compute_nll`, `plot_facet`, and `plot_facet_bootstrap` (five classes × three methods, about 60 lines each); `Calibration.check_fit`, `plot_final`, `plot_optuna`, `to_json`, `parse_study` are `Fit.summarize/plot/to_df`. The Optuna SQLite study machinery (`make_study`, `remove_db`, `run_workers`, 70 lines) exists to run workers in processes, which the v4 path does not do. See user finding 1 for the replacement; the test file `tests/test_calibration.py` would shrink from four Starsim 3 tests to tests of `fit()`. **Confidence: high.**

### 2. `fit()` is one 304-line function with seven methods inside it

`fit()` (`calibration.py:2095-2398`) contains: parameter discovery (2144-2167), three coordinate systems (`encode`/`decode` for log-scale optimizers at 2233-2234; `to_x`/`to_z`/`share_of` logit transforms for samplers at 2276-2289; and the raw scale), a scan over Latin hypercube points (2238-2244), then branches for Optuna (2247-2256), `asd` (2257-2260), SciPy with and without gradients (2261-2272), PMCMC (2291-2304), NUTS in JAX with a fallback to NUTS in Python (2305-2342), then posterior summaries (2343-2362), three kinds of identifiability check (2366-2388), and assembly of `Fit`. The closures `run`, `objective`, `with_gradient`, `measure`, `to_pars`, `logpost`, `logpost_pf`, `logpost_jax`, `exact` all share state through the enclosing scope. Adding a method means editing the middle of this function.

**Proposal.** A `Problem` object (bounds, log flags, `objective(x)`, `gradient(x)`, `logpost(z)`) built once, and one function per method with the signature `method(problem, x0, **kwargs) -> (best, extras)`; a dict `methods = {'optuna': ..., 'nuts': ..., ...}` that `default_method` and the `method=` lookup share (which also gives the unknown-name check for free). Staged fitting, if kept, becomes a loop over that dict. **Cost:** a refactor of 300 lines with the existing tests as the net. **Confidence: high.**

### 3. Two NUTS samplers and two copies of the likelihoods and priors

- `nuts()` (`calibration.py:1709-1778`) runs BlackJAX with a `pure_callback` to a NumPy log-density and finite-difference gradients, one chain at a time; `nuts_jax()` (1781-1850) runs the same sampler on a JAX log-density, compiled and vmapped across chains. `fit()` tries the second, checks three draws against the sim (2335-2340), and falls back to the first.
- `logpdf()` and `logprior()` in `calibration.py:1418-1475` (NumPy/SciPy) are mirrored by `Density.loglik` and `Density.logprior` in `gradients.py:1049-1100` (JAX), each a chain of `isinstance` branches over the same four likelihoods and five priors. A new likelihood has to be added in both and in the `ParticleFilter` path.

**Proposal.** Drop `nuts()` and the fallback: when `Density` raises `Unsupported`, say so and suggest `method='l-bfgs-b'` (the fallback is slow enough that the user would do that anyway). Write each likelihood once as a function taking an array namespace (`np` or `jnp`), since `scipy.stats` and `jax.scipy.stats` have the same names for all four. **Cost:** models the JAX density does not cover (modifiers of fitted parameters, per `implementation.md:229`) lose NUTS; about 150 lines go. **Confidence: medium** (the fallback was clearly built for a reason; the question is whether that reason outlives the next `Density` extension).

### 4. `compare()` and `lineup()` are the same alignment written twice

`compare()` (`calibration.py:1310-1338`) computes aligned model values; `lineup()` (1341-1366) computes the matrix `W` that does the same alignment, for `ParticleFilter` and `Density`. Both re-derive `eps`, `inside`, `spacing`, `starts`. `compare()` could be `W @ values` from `lineup()`, one rule in one place. Twenty lines. **Confidence: high.**

### 5. Hand-rolled MCMC diagnostics

`rank_normalize`, `split_chains`, `basic_rhat`, `basic_ess`, `diagnose` (`calibration.py:1853-1920`, 70 lines) reimplement ArviZ's rank-normalized split R-hat and Geyer ESS to avoid a dependency. That is a reasonable trade and the code cites its sources; I would keep it but note it is untested on its own (no test compares it with ArviZ or with a known chain).

### 6. Naming: a lowercase class `data` that shadows its own argument

`class data` (`calibration.py:1174`) is a class spelled like a function, exported as `ss.data`, and `fit()` has an argument also called `data`, so inside `fit()` the class is reached as `ss.data` with a comment explaining why (`calibration.py:2361`: "ss.data, since data is an argument here"). It also has six static methods (`widen`, `parse`, `levels`, `match`, `match_name`, `check`) that take `sim` as the first argument and never use `self`, i.e. it is a module namespace wearing a class. Either `ss.Data` (a class, as `ss.Sim`), or plain functions `ss.match_data(sim, col)` etc. and keep `ss.data(...)` as the user-facing constructor only. **Confidence: medium** (the spec chose `ss.data`; the cost is a rename in the spec's one line).

### 7. Where the tests are

`tests/test_calibration.py` tests the Starsim 3 class (four tests) and particle MCMC, which is skipped (`@pytest.mark.skip(reason="Test requires performance enhancement")`, `test_calibration.py:75`). So the default method for a stochastic compartmental sim with a likelihood is not exercised in CI. The `fit()` tests that do run are in `test_transitions.py:672-790` and `test_gradients.py:70-190`. After finding 1, `test_calibration.py` is the natural home for all of them, and the PMCMC test could run with a 20-second budget (2 chains, 50 draws) rather than be skipped.

## Spec and implementation

- **The method table** (`13-inference.md`, "Choosing the method") and both transcripts describe ABC-SMC and a timing-based "cheap enough" decision; the code has neither (`default_method`, `calibration.py:2072`). `implementation.md:283` records the departure but the spec was not updated. The second transcript shows intervals from an agent-based fit, which the spec's own rule forbids presenting.
- **The printed output.** Spec: a four-line block plus the table. Code: "Searching on a log scale: ..." and "Fitting by ... (reason; override with method=)", then `summarize()` on request.
- **`fit.pars`.** Spec: "a table: parameter, initial value, best fit, interval if the method gives one, bounds". Code: a dict of best values (that is what `ss.Sim(pars=)` needs); the table is `fit.table`. The code's split is the right one; the spec should say `fit.pars` (dict) and `fit.table` (table).
- **`17-vocabulary.md:110`** says `sim.calibrate()` "returns the calibration object that this specification calls `fit`" and lists `ss.Calibration` as that object. The code returns `ss.Fit`, and `ss.Calibration` is a different, unrelated class. One of the two names should go (finding 1).
- **Staged fitting, `W1303`, `sim.sensitivity()`, convergence reporting, fitting a product**: specified, not implemented (user finding 6 and 5). `sim.sensitivity()` is not recorded as deferred anywhere I could find.
- **Priors and likelihoods in the spec.** `deferred.md:31` says "Priors and likelihoods are not in the model spec, so `to_code()` lists them for adding by hand". `sim.to_spec()` does include them, as strings: `"prior": "ss.lognorm_ex(mean=0.3, std=0.1)"` and `"likelihood": "ss.nbinom(n=par(10, bounds=[0.1, 100]), p=0.5)"` (verified, `exp5_spec.py`). What is missing is the reverse direction (`ss.load()`/`to_code()` do not read them back), and the `likelihood` string loses that `n` was a named `ss.par`. The deferred note should say that. The hole is real but small: a likelihood is model, so it belongs in the spec and should round-trip; a prior is arguably a fitting setting and `13-inference.md` says fitting settings are not part of the hash, yet the spec stores `prior` on the parameter and hashes it. The spec should pick: either `prior` is model (stays, round-trips) or fitting (moves to `calibrate(priors=)`). I lean to keeping it on the parameter (it is where a reader looks) and excluding it from `sim.hash()`.
- **The `fit` field.** `14-model-spec.md:111` has a top-level `fit` field for "calibration settings (data, weights, method, trials), or null". `to_spec()` has no such field (`fit field in spec: ABSENT`), and `13-inference.md` says fitting settings are not part of the spec's hash. Drop the row from `14`.
- **Time columns.** Spec (`09-results.md:164`, `13-inference.md`): `date`, `year`, `t`, `time`. Code (`utils.py:592`, `calibration.py:1209`): also `timevec`, `tvec`, `day`. Harmless, but the spec says four and the code says seven; the code's list is the one to document, or trim.
- **B1 answers given by the implementation.** Default goodness of fit: NAE as proposed, with the range-then-1.0 fallback for zero-mean series (defensible; see user finding 4). Log-scale search: positive bounds spanning a factor of ten or more, printed (`calibration.py:2174-2176`), as proposed. Held-out data in `compare_models`: the last 25% per series, with `folds=` as the alternative, as proposed. All three guesses stand and work; the spec can mark them settled.
- **`n_runs`.** Spec: "for a stochastic sim the model value is the mean over `n_runs` runs". Code: the goodness of fit is the mean over runs of the goodness of fit (`calibration.py:2221-2222`), not the goodness of fit of the mean trajectory. For NAE these differ (the former penalizes variance). The code's choice is arguably better for a point estimate; the spec sentence should match it.
- **Common random numbers.** The spec says CRN pairs runs across parameter sets. Verified in the Optuna path: the enqueued initial trial and `mismatch_before` are identical (`1.104916` twice), so every evaluation reuses the sim's seed. Good, and unstated in the code except by omission of `rand_seed`; one comment would do.

## Bugs

1. **`ss.Sim(sir, pars=fit.pars)` fails if a fitted parameter is named `dur`** (or any sim parameter). Repro `exp3_roundtrip.py`; output under user finding 2. Cause: key sniffing at `sim.py:77`.
2. **Unknown `gof` names run silently as NAE.** `calibrate(gof='foo', method='nelder-mead')` → `no error! mismatch 1.187e-06 equals nae? True`. Cause: `calibration.py:1389` tests only `== 'rmse'`.
3. **`method=[...]` raises `AttributeError: 'list' object has no attribute 'lower'`** (`calibration.py:2209`) instead of "staged fitting is not implemented".
4. **`pars=[beta * contacts]` raises `E1301 NothingToFit`**, a false statement; the parameters have bounds. Cause: the membership test at `calibration.py:2152` (`name in pars or f'{mod.name}.{name}' in pars`) never matches an `Expr`, so nothing is selected and the "no bounds" branch fires.
5. **A bounded parameter inside a distribution** (`I >> R @ ss.gamma(a=3, scale=ss.par(10, bounds=[3, 30]))`) fails with `TypeError: float() argument must be a string or a real number, not 'BinOp'` rather than a coded error.
6. **`ss.Calibration(sim, calib_pars=...)` without `build_fn`** fails inside Optuna with `TypeError: 'NoneType' object is not callable`, once per trial, then again at the end; `__init__` should refuse a missing `build_fn` (or default to setting `sim.pars.diseases.pars[k]`).
7. **Optimizer convergence is never checked**: `calibrate(method='l-bfgs-b', maxiter=1)` returns a `Fit` whose `summarize()` says nothing (output under user finding 6). Not a crash, but a result the spec says must be reported.

## Keep

- **Bounds on the parameter, data on the sim, one call.** `sim.calibrate()` with nothing else found and fitted a parameter inside `init=`, inside a seasonal expression, inside a likelihood, and shared across modules in the tests, and recovered `beta` to 1e-11 on the ODE here. This is the whole point of the design and it works.
- **Alignment by result kind** (`compare()`: a flow is summed over the interval ending at each data time, a stock is interpolated) removes `conform=` from the user's vocabulary. Keep it as the single rule.
- **`default_method()` is five lines and prints its reason.** Do not grow it into the spec's table; shrink the spec to it.
- **`sim.restart()` for compartmental sims** makes each evaluation cheap enough that Optuna on a stochastic compartmental fit is 5 ms per trial. This is why the stochastic compartmental fits above took 0.15 s.
- **Common random numbers by default** in every evaluation (the seed is not changed between trials), so Optuna compares parameter sets on paired runs.
- **The `Fit` object**: `fit.sim` (run, with the best values), `fit.before`, `fit.pars`, `fit.table`, `fit.trials`, `fit.samples`, `fit.band`, `fit.plot()` before/after with a posterior band. The spec's rejection of `fit.apply(sim)` in favor of `fit.sim` is right, and `fit.sim.new()` keeps the fitted values.
- **Identifiability notes (`W1302`)** from the model (products and ratios in the expression tree, `product_pairs`) and from the fit (posterior correlation, gradient collinearity). The posterior note fired correctly on `beta` and `dur` here.
- **`ss.compare_models` by held-out prediction**, with `folds=` and the "no clear difference" check against the spread. Small (100 lines) and does what the spec says.
- **Lazy imports and graceful fallbacks** for JAX, BlackJAX, and Optuna, with the install hint in the message.
- **`ParticleFilter`** is a compact, well-documented bootstrap filter that reuses the kernels (`ss.kernels.advance`) rather than stepping sims; keep it, but un-skip its test.

## Outside my area

- **`ss.Sim(pars=)` key sniffing** (`sim.py:77-78`) is the root of bug 1 and is a general hazard, not an inference one: any model parameter that shares a name with a sim parameter is misrouted. Belongs to the objects/Sim review.
- **`sim.new()` only exists for declared sims** (`sim.py:429-431`), which is why every v4 workflow that rebuilds a sim (`fit`, `compare_models`, `sim.gradient`) excludes Starsim 3 modules. A `new()` that falls back to `sc.dcp(self)` plus `set_par_values()` would widen all of them at once.
- `rel_sus`/`rel_trans` on a state are plain numbers, not `ss.par`, so they cannot be fitted (`implementation.md:265`); a user following the one-page model will try.
