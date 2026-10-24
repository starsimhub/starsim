# Spec gaps: camdl's models in Starsim v4

Whether every model bundled with camdl can be written easily in Starsim v4: its syntax, keywords, and concepts, not a run-by-run comparison of results. Written 24 October 2026, against the `ssv4` working tree on `v4-oct24b` and camdl at `eb205657`. Nothing in `starsim/` or the spec was changed.

## Summary

Mostly yes. About 45 of the 55 models in camdl's golden library (`ocaml/golden/`) translate into v4 nearly line for line, and so do most of the other bundled models (about 45 more: test fixtures, external cases, the pycamdl examples, and the Rust fixtures). The real gaps are a few camdl features for stochastic runs (noise on a transition, deterministic flows, algebraic balance), a true bimolecular reaction, and some likelihoods. They pile up in one important model, He et al. 2010, which is pomp's classic benchmark and camdl's main external case.

## What was checked

- **The models.** All 55 golden models, and every other `.camdl` file in the camdl repo except the deliberately broken ones (`ocaml/test/lints/`, `ocaml/test/errors/`, `ocaml/golden/errors/`): `pycamdl/examples/`, `tests/fixtures/` (corner cases, reactive interventions, quantities, contrasts, gradients, regressions), `tests/external/` (He 2010, boarding school, the ODE oracles), `tests/recovery/`, the Rust fixtures, and the Garki proposal fixtures in `docs/dev/proposals/`.
- **The importer.** v4's camdl importer (`starsim/interop/camdl.py`) was run on the IR of all 55 golden models: all 55 build and initialize, though it leaves out what it does not translate (scheduled interventions, functions of time, ODE compartments, and some likelihoods), so this shows only that nothing in them breaks the importer.
- **By hand.** Every model's source was read, and each feature whose v4 form was in doubt was written as a small v4 model and run. "Verified" below means that; "spec" means it is read from the spec and not run.

## Written almost as in camdl

| camdl | v4 | Checked |
|---|---|---|
| Rates written as the whole flow, `S --> I @ beta * S * I / N` | Read as written: `S >> I @ beta * S * I / N` ([03](../design/03-states-and-transitions.md#the-whole-flow)) | verified |
| Inflows and outflows (`--> S`, `S -->`) | `ss.null >> S`, `S >> ss.null`, or `ss.birth()` and `ss.death()` | verified |
| Branching, `S --> { I_symp : p, I_asym : 1 - p }` | `ss.Transition(S, [I_symp, I_asym], p=[p, None], rate=...)` | spec |
| `via hyper_erlang(branch(...), branch(...))` | Two transitions out of the state, each with `p=` and `dur=ss.erlang(k=..., mean=...)` | verified |
| `via erlang(stages=3, rate=sigma)`; stages by `stratify(only=[E])` with per-stage rates | `dur=ss.erlang(k=3, mean=...)` or `stages=3`; per-stage rates by `stratify(..., states=[E])` and `ss.by` | spec |
| Real-valued ODE compartments (`W : real` with `ode { W = shed * I - decay * W }`) | `W = ss.State(people=False)`, `ss.null >> W @ shed * I.n`, `W >> ss.null @ decay` | verified |
| Catalytic reactions, `S_h + I_v --> I_h + I_v` | `Sh >> Ih @ a * b_h * Iv.n / H`, with the vector states `people=False` | spec |
| Power mixing, `(I + iota)^alpha * S / pop(t)` | The same with counts: `S * (I.n + iota)**alpha / pop` | verified |
| `R^alpha` added to exposure, `(cap - I)` in a recovery rate | Counts, `(I.n + R.n**alpha) / N.n`; `gamma * (cap - I.n) / N.n` | verified |
| Guards, `if N > 0 then I / N else 0` | `(N.n > 0) * I.n / np.maximum(N.n, 1e-300)` | verified |
| Dimensions with named levels and a contact matrix, `sum(b in age, C[a, b] * I[b] / N[b])` | `stratify(group=['child', 'adult'])` and `ss.contacts(group=C)` | verified |
| Aging between consecutive age groups, with a rate per group | `stratify(age=[0, 5, 15, ...])`: groups age at `1/width` when people are born or die, or with `use_aging=True` | verified |
| Per-stratum parameters (`R0[patch]`, `beta[region, age]`) and init tables (`S[p] = pop[p]`) | `ss.by(patch=[...])`, `ss.by(arr, dims=['region', 'age'])`, `init_n=dict(I=ss.by(...))` | verified |
| Tables and dimension levels read from a TSV | pandas, then `ss.by()` or a NumPy matrix | spec |
| Patches coupled by a matrix, `kappa * W[p, q] * S[p] * I[q] / N[q] where p != q` | `stratify(patch=..., kind='space')` and `ss.contacts(patch=np.eye(n) + kappa*W/beta[:, None])` | verified |
| `where dist[p, q] < 50`, gravity kernels of tables | The matrix worked out in NumPy | spec |
| Movement between patches, `I[p] --> I[q] @ m * I[p]` | `ss.transfer(patch=('a', 'b'), rate=m, states=['I'])` | spec |
| `interpolated` forcing from a TSV | `ss.timeseries(df, mode='linear')` | verified |
| `periodic` on/off forcing (school terms) | `ss.timeseries({0: 1, 182: 0}, period=ss.days(365.25))` | verified |
| `sinusoidal` forcing with a yearly period, `lag` | `ss.seasonal(mean=, amplitude=amp/baseline, peak=)`, the lag folded into `peak` | spec |
| `transfer(fraction = f, from = S, to = V) at [...]`, also per stratum and from a table of dates by region and round | `ss.Delivery(move=S >> V, at=[...], coverage=f, where=d.patch == p)`, one per row in a loop | verified |
| `origin = date(...)`, `add_calendar_months`, `date_range` | `start=ss.date(...)`, `ss.date + ss.datedur(months=6)` | verified |
| Init depending on parameters and on other states (`E = epi * I`, `R = frac_immune * N0`) | `init_n=dict(I=I0, E=epi*I0, R=frac_immune*N0)`, with `ss.par` | verified |
| `incidence(flow)` over weeks with a negative binomial or Poisson likelihood, `rho * projected` | `ss.outcome(rho * infection, likelihood=ss.nbinom(n=k))`; data by week is compared with weekly totals | verified |
| Observations by stratum, `cases[p in patch]`; sums over strata | The outcome's results by group (`res.cases.by('patch')`); its total | verified (results only; fitting by stratum not run) |
| `prevalence(I)`, `R / N`, `I_total` | Results: `seroprev=R.frac` on the disease, with a likelihood at `calibrate()` | verified |
| Binomial and Bernoulli likelihoods | `ss.binom(n=...)`, `ss.bernoulli()` | spec |
| `diagnostic_test(base = binomial(...), sens, spec)` | `ss.binom` on a result `sens * prev + (1 - spec) * (1 - prev)` | spec |
| Priors: log-normal, half-normal, beta, gamma, exponential, truncated normal | `ss.par(prior=ss.lognorm_im(...))`, `ss.truncnorm(low=0)`, `ss.beta_dist`, `ss.gamma`, `ss.expon`, `ss.truncnorm` | spec (as the importer maps them) |
| Scenarios: `set`, `scale`, `extends`, `enable`, `simulate { to }` | `sim.new(pars=...)`, interventions given or not | verified |
| `dt = 0.5`, `time_unit = 'weeks` | `dt=0.5`, `unit='week'` | verified |
| ODE, chain binomial, and Gillespie backends | `solver='ode'`, `'binomial'`, `'gillespie'` (and `'tau'`) | verified |

## Possible, with some friction

- **A fitted coupling inside a contact matrix.** `ss.contacts(patch=np.eye(n) + kappa*W)` with `kappa` an `ss.par` fails (`float()` of an expression), so a coupling to be fitted cannot sit in the matrix. `ss.geo.Mobility(matrix=W, kind='force', away=ss.par(...))` can be fitted, but it is a different form: each patch keeps `1 − Σ M` of its own force of infection, where camdl's adds the imported force on top. A beta that differs by patch also has to be divided into the matrix by hand, since the matrix sits under the transition's beta. Affected: `polio_spatial_5`, `seir_spatial_5_inference`, `seir_defines_adj`, `polio_afp_es_2patch`, `pathogen_migration`, `licm_ab` (whose `theta` and `gamma_k` are fitted inside the kernel), `sparse_coupling_ab`.
- **Adding people from outside at given times**, `add(I, n)`. `ss.Delivery(move=ss.null >> I, ...)` is refused. The choices are `move=S >> I, capacity=n` (from `S`, so the population does not grow, which is what most importation means) or an inflow with a one-step `ss.timeseries` pulse. Affected: `seir_seasonal_importation`, the corner cases `all_lifecycle`, `gh70_absorbing_importation`, `event_intervention_agree`, and `multi_effect_same_time`, and He 2010's cohort entry, whose size also changes over time.
- **Random initial conditions** (`init_laws`: `I ~ poisson(rate = I0)`, and negative binomial, binomial, and normal draws). `init_n=dict(I=ss.poisson(lam=50))` fails. The numbers can be drawn in Python before the sim is built, but they are then the same in every run.
- **Time in a rate** (`seed_timing`: `lambda / (1 + exp(-(t - tau) / w))` with `tau` fitted). There is no time variable in expressions. A function of the sim works, and so does a linear `ss.timeseries` ramp standing in for `t`, which keeps `tau` an `ss.par`.
- **A sinusoid with a period other than a year**, or with an absolute amplitude. `ss.seasonal` is yearly with a relative amplitude; anything else is an `ss.timeseries` or a function of the sim.
- **Reactive interventions** (`reactive_interventions { when sum_observed(...) >= x { after, cooldown, once } }`). `when=` on a delivery, plus a small module with a `step()` for the cooldown and once-only memory, as [08](../design/08-interventions.md) says. camdl triggers on simulated observations; v4 on the model's results.
- **`quantities` and `contrasts`** (`final`, `max`, `time_of_max`, `first_above`, `integral`, differences between scenarios). Ordinary Python on the results; nothing declared.
- **Likelihoods the core lacks.** Beta (mean and concentration), beta-binomial, zero-inflated negative binomial, and a normal whose standard deviation depends on the model's value (`sd = sigma_rel * projected`, and He 2010's `sqrt(rho * projected * (1 - rho + psi^2 * rho * projected))`). The core has Poisson, negative binomial, binomial, Bernoulli, normal with a fixed scale, and lognormal ([13](../design/13-inference.md#likelihoods-when-wanted)). Affected: `death_fraction_window`, `surveillance_likelihoods`, `zinb_vector_catch`, and He 2010.
- **A log-uniform prior** (`sir_priors`). None in the core.
- **RK45.** v4's integrators are `rk4` and `euler`; only some camdl lint fixtures use the adaptive one.
- **Hierarchical priors** (the Garki proposal fixture: `r1[age] ~ log_normal(mu = mu_r1, sigma = sigma_r1) | age`). An `ss.par` whose prior has another `ss.par` as its mean builds and runs, but whether `calibrate()` treats it as a hierarchy was not checked.

## Not expressible

1. **A true bimolecular reaction, `A + B --> C`** (`bimolecular`), in which one event takes one from each source. In v4, `A + B >> C` is a transition from either state, and the rate `k * A * B / N` raises `NonlinearExposure`. It is a solver-correctness fixture rather than an epidemiological model, and catalytic forms (the source stays, as in Ross-Macdonald) are fine.
2. **`overdispersed(rate, sigma_se)`**, gamma white noise on a transition's rate (extra-demographic noise). Affected: `sir_overdispersion`, `sir_two_overdispersed`, `seir_spatial_5_inference`, and the three He 2010 cases.
3. **`deterministic(...)`**, a flow taken as its expected value inside a stochastic run. Affected: `phenom_mixing_unchecked` and He 2010's births. Items 2 and 3 are already deferred in [19-open-questions.md](../design/19-open-questions.md#open-questions-from-best-practices); under `solver='ode'` both make no difference, so those models run there as written.
4. **`balance { R = pop(t) - S - E - I }`**, a compartment given by an algebraic constraint rather than by flows. Affected: `seir_pop_balance`, `all_lifecycle`, and He 2010.

## He et al. 2010

The one real model that is not easy. It has every hard piece at once: noise on infection (`overdispersed`), births as a deterministic flow, `balance` to tie `R` to the census population, a yearly cohort entry whose size follows the birth rate and population on the day, and the normal likelihood whose standard deviation depends on the model's value. The rest of it (power mixing over `pop(t)`, school-term forcing, interpolated population and birth rate from a TSV) is written easily. As the classic benchmark of pomp and of camdl's particle filter, it is worth deciding whether v4 means to cover it.

## By model

The golden models that need something from the two lists above; every other golden model is written as in the table.

| Model | Needs |
|---|---|
| `bimolecular` | Bimolecular reaction (not expressible) |
| `sir_overdispersion`, `sir_two_overdispersed` | `overdispersed` (not expressible) |
| `seir_spatial_5_inference` | `overdispersed` (not expressible); fitted coupling (friction) |
| `phenom_mixing_unchecked` | `deterministic` (not expressible as stochastic; fine as ODE) |
| `seir_pop_balance` | `balance` (not expressible) |
| `death_fraction_window` | Beta likelihood |
| `surveillance_likelihoods` | Normal with sd from the model, beta-binomial, and beta likelihoods |
| `zinb_vector_catch` | Zero-inflated negative binomial likelihood |
| `sir_priors` | Log-uniform prior |
| `init_laws` | Random initial conditions (friction) |
| `seir_seasonal_importation` | Adding people from outside (friction) |
| `polio_spatial_5`, `seir_defines_adj` | Fitted coupling in the matrix (friction, only if fitted) |
| `ross_macdonald` | `diagnostic_test` as a derived result (friction) |
