# odin

## What the source is

An odin model is R code in odin's DSL (one `.R` file, or the body of `odin::odin({...})`): `deriv(X) <-` for an ODE or `update(X) <-` for discrete time, `initial(X) <-`, `user()` (or `parameter()`) for inputs with defaults, `rbinom()`/`rpois()` draws, arrays (`X[]`, `dim(X) <-`), `sum()`, `interpolate()`, `delay()`, and `output()`. odin has no time unit and is told when to stop at run time.

## Mapping

| odin | Starsim | Mechanical |
|---|---|---|
| `deriv(X)` equations | Flows: each equation multiplied out into signed terms, and a term negative in one state and positive in another paired into a flow (a total, then classified) | yes |
| `update(X)` with `n <- rbinom(X, 1 - exp(-R * dt))` | A rate `R` under `solver='binomial'` (another `p` becomes `-log(1 - p)` per step) | yes |
| `rpois(lambda)` | A total per step | yes |
| `user(x, default)` | A parameter (override with `ss.interop.load(..., pars=dict(x=...))`) | yes |
| Arrays of alike, independent copies (replicates) | One model, run `n_runs=` times | yes |
| Arrays with flows from each level to the next (aging) | An age stratification, with bands from `1/rate` | yes |
| `sum(I)` | `I.n` (the total over every group) | yes |
| `delay(new_inf * surv, tau)` into a state | A fixed dwell time (`ss.constant`), with its factor checked against the survival through the other outflows | yes |
| The history of a `delay()` | An inflow for the first `tau` (below) | no |
| A chained draw (`n_ID <- rbinom(n_I, p_death)`) | A branch: `ss.Transition(I, [R, D], rate=gamma, p=[None, p_death])` | no |
| `rmultinom` | Branches with `p=` | no |
| A contact matrix: `s_ij[, ] <- m[i, j] * I[j] / N[j]` and `lambda[] <- beta * sum(s_ij[i, ])` | A stratification by the arrays' dimension (named from `dim(S) <- n_age`, levels `"1"` to `"n"`, initial numbers by level) and `networks=ss.contacts(m)`; the force is `beta * I` | yes |
| `1 + A * cos(2 * pi * step * dt / 365)` (or `t`, `time`; or `sin`) | `ss.seasonal(mean=1, amplitude=A, peak=...)` as a parameter | yes |
| `update(time) <- (step + 1) * dt` | Nothing: the sim keeps the time | yes |
| `update(cases) <- if (step %% k == 0) sum(n_SE) else cases + sum(n_SE)` | `ss.outcome(n_SE, every=ss.days(k * dt))` | yes |
| Names Python reserves (`lambda`, `is`, ...) | Renamed with a trailing underscore (`lambda_`) | yes |
| Another `if (c) a else b` in a rate | `ss.timeseries` (a condition on the time), `where=`, or a comparison of counts (`(Q.n < K)`) | no |
| A state at its own level in a stratified model with no matrix (`beta * S[i] * I[i] / N[i]`) | A contact matrix by the dimension, `networks=ss.contacts(np.eye(n))` | no |
| Arrays over levels with no flows between them (patches) | `stratify(patch=[...], kind='space')`, with `ss.by(patch=[...])` for parameters that differ | no |
| `interpolate()`, `t`/`time`/`step` | `ss.timeseries({t0: v0, ...}, mode='linear')` as a parameter's value | no |
| `output(x)` | A result on the disease, e.g. `prev = I / (S + I + R)` | no |

## What needs care

- Give `unit=` (default `'day'`) and `stop=`: odin knows neither.
- A `delay()` returns the delayed expression's value at time 0 for the first `tau`, as if the flow had run before the start. Without that history the epidemic in `seir_array` peaks 14 days late. Add it as an inflow: `history = ss.null >> I @ ss.timeseries({0: ss.perday(surv * new_inf_0), tau: 0})`.
- A parameter equal to the population (`N <- 1e7` used as a divisor) is kept as a number (`beta * I.n / N`), which is exact while the population is constant; with births and deaths, the frequency-dependent form is `beta * I`.
- odin's latent people do not age in `seir_array`, so the youngest group differs from Starsim's (a finding of the vignette).

## Worked example

`lingua_franca_design/vignettes/odin/seir_array/model.py` (SEIR in 5 age groups with births, deaths, aging, waning, and a 14-day `delay()`), and `vignettes/odin/stochastic_sir_arrays/model.py` (replicates as `n_runs`). `ss.interop.load('seir_array_odin.R', stop=...)` gives the aging as a stratification and the delay as a fixed dwell; the one line it lists, the delay's history, is:

```python
history = ss.null >> E @ ss.timeseries({0: ss.perday(surv), tau: 0}) # The first tau of the delayed flow, which odin takes as the value at time 0
```

With it, the import agrees with odin within 5.6, 8.4, 6.3, and 3.6% in S, E, I, and R (as the vignette's 50-stage version does). The stochastic SIR arrays agree within 4.7 of 1,001 people in the mean over 2,000 runs.

## Running the original

odin's environment is `lingua_franca_design/_envs/odin` (`bin/Rscript` puts odin in R's library path). As in `examples/odin/*/run.py`: write an R driver that does `gen <- odin::odin_(path); x <- gen$new(<user values>); set.seed(seed); out <- x$run(0:<stop>)` and `write.csv(out, ...)`, and run it with `tools.run(['Rscript', driver, ...], 'odin')`. (`odin::odin()` quotes its argument, so a path in a variable, such as `commandArgs()`, needs `odin::odin_()`.) The columns are `t` (or `step`) and one per variable (`S[1]`, ... for arrays). A stock at time `k` is the number at time `k` in both, so no shift is needed. Deterministic models compare exactly; for discrete stochastic ones, many runs, compared statistically (many die out early).
