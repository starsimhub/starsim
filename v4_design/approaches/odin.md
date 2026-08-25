# odin / odin2 (with dust and monty)

|  |  |
|---|---|
| **Language** | R DSL, compiled to C++ (and optionally JavaScript) |
| **Paradigms** | Continuous-time ODE; discrete-time stochastic (difference equations with distributional draws) |
| **Specification style** | Embedded DSL — R syntax, non-R semantics — compiled |
| **Version reviewed** | odin2 0.3.46 |
| **Licence** | MIT |
| **Code** | <https://github.com/mrc-ide/odin2>, <https://github.com/mrc-ide/dust2>, <https://github.com/mrc-ide/monty> |
| **Docs** | <https://mrc-ide.github.io/odin2>, the [odin & monty book](https://mrc-ide.github.io/odin-monty/) |
| **Paper** | dust: [10.12688/wellcomeopenres.16466.2](https://doi.org/10.12688/wellcomeopenres.16466.2) |
| **Ecosystem** | odin (DSL) → dust (parallel simulation engine) → monty (Monte Carlo inference); ~5 current packages |

## What it's for

odin lets a modeller write a system of differential or difference equations more or less as they would write it on paper, and get a compiled C++ implementation with a particle filter and an MCMC layer attached. It is the workhorse behind a large fraction of MRC Centre / Imperial modelling output, and the closest thing in R to what camdl is in its own language.

The three-package split is the design: **odin** is a language, **dust** is an execution engine, **monty** is an inference layer, and they meet at defined interfaces rather than being one monolith. A model written in odin compiles to the dust interface; monty runs particle filters and MCMC against anything that implements that interface, odin-generated or not.

## How a model is specified

The DSL is syntactically R — it parses with R's parser — but is not R. Every line is either an **assignment** (`a <- expr`) or a **relationship** (`b ~ Distribution(...)`). There is no control flow: `if` exists only as an inline expression, loops exist only implicitly through array indexing.

A discrete-time stochastic SIR:

```r
gen <- odin2::odin({
  p_IR <- 1 - exp(-gamma * dt)
  N <- parameter(1000)

  p_SI <- 1 - exp(-(beta * I / N * dt))
  n_SI <- Binomial(S, p_SI)
  n_IR <- Binomial(I, p_IR)

  update(S) <- S - n_SI
  update(I) <- I + n_SI - n_IR
  update(R) <- R + n_IR

  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0

  beta <- parameter(0.2)
  gamma <- parameter(0.1)
  I0 <- parameter(10)
})
```

The continuous-time version replaces `update()` with `deriv()`. **The variables of the system are determined by the presence of `initial()`**, which is common to both, so the same detection logic serves both paradigms.

Crucially, **statement order does not matter**. odin builds a dependency graph and topologically sorts. The model is a set of equations, not a sequence of operations, and that is what makes it a specification rather than a program.

Running is a separate concern, handled by dust:

```r
pars <- list(beta = 0.2, gamma = 0.1, I0 = 10, N = 1000)
sys  <- dust2::dust_system_create(gen(), pars, n_particles = 10)
dust2::dust_system_set_state_initial(sys)
y <- dust2::dust_system_simulate(sys, 0:100)
```

## Core abstractions

| Concept | odin name | Notes |
|---|---|---|
| State variable | anything with an `initial()` | `deriv()` for ODEs, `update()` for discrete time |
| Derived quantity | ordinary assignment | Order-independent; topologically sorted at compile time |
| Parameter | `parameter(default, constant=, differentiate=, type=, rank=)` | Typed (`real`/`integer`/`logical`), with defaults and a differentiability flag |
| Array | `x[]`, with a mandatory `dim(x) <- ...` | Up to 8 dimensions; implicit loops driven by the *left*-hand side |
| Output | `output(y) <- expr` | ODE-only: quantities reported as state without having derivatives |
| Data | `d <- data()` | A time series to compare against |
| Likelihood | `cases ~ Poisson(incidence)` | Comparison to data written in the model file |
| Accumulator | `initial(incidence, zero_every = 1) <- 0` | Auto-resetting accumulator for incidence-style quantities |
| Interpolation | `interpolate(time, value, mode)` | `constant`, `linear`, or `spline` external time series |
| Delay | `delay(...)` | Delay differential equations, with explicit history state |
| Time step | `dt` | Available as a variable in discrete-time models |

## Time

Time is `time`, the step is `dt`, and both are available inside the model. That last point matters more than it sounds: because `dt` is in scope, the modeller writes `p_SI <- 1 - exp(-beta * I / N * dt)` explicitly, so the rate-to-probability conversion is *visible in the model text* rather than hidden in the engine.

That is a real design position, and a defensible one — it is the opposite of Starsim's (where `ss.peryear(0.1).to_prob(dt)` makes the conversion the type's job). odin's version is more transparent and less safe: nothing stops a modeller writing `p_SI <- beta * I / N * dt`, which is correct only for small `dt`, and nothing marks `beta` as a rate rather than a probability. **There is no unit or dimension system.** Parameters are `real`, `integer`, or `logical`.

For discrete-time models the order of events within a step is documented precisely: reset `zero_every` accumulators, read variables, look up interpolations at `t0`, evaluate all assignments (with `time = t0` and all variables at start-of-step values), write new state, advance time — then, separately, compare to data at `t0 + dt`. Having this written down at all puts odin ahead of most of the review.

## Stochasticity and reproducibility

Discrete-time models draw from distributions inline: `n_SI <- Binomial(S, p_SI)`. The semantics of a draw are specified carefully — `x[] <- Normal(0, 1)` gives every element its own draw, while `a <- Normal(0, 1); x[] <- a` gives every element the same draw. That distinction is exactly the kind of thing that is ambiguous in most frameworks and silently wrong in some.

Continuous-time models cannot sample: stochastic functions are rejected in ODE mode. This is a hard boundary rather than a silent approximation, which is the right choice.

dust runs `n_particles` in parallel with per-particle RNG streams. There is no common-random-numbers facility across scenarios.

## Comparison to data

The likelihood lives in the model file, in two extra lines:

```r
cases <- data()
cases ~ Poisson(incidence)
```

`data()` declares that a variable comes from the data time series; `~` adds a log-likelihood term. dust then provides the machinery — `dust_filter_create()` builds a particle filter, `dust_likelihood_run()` returns a marginal likelihood estimate, and monty runs MCMC over it.

The `zero_every` accumulator deserves special mention: `initial(incidence, zero_every = 1) <- 0` declares a quantity that accumulates within a period and resets at the period boundary. Incidence-per-unit-time is the single most commonly needed derived output in the field, and this is the cleanest declaration of it anywhere in the review.

## Arrays and stratification

There is no stratification *operator*. Age structure, spatial patches, and strata are written as array dimensions:

```r
update(S[]) <- S[i] - n_SI[i]
dim(S) <- n_age
lambda[] <- sum(m_beta[i, ])
ax_tmp[, ] <- a[i, j] * x[j]
ax[] <- sum(ax_tmp[, i])
```

Loop extents come from the left-hand side; `i`, `j`, `k`, `l` are the index variables; `dim()` is mandatory and checked at parse time; `sum()` supports partial sums over slices, which is how contact-matrix products are written.

This is powerful and correct, and it is a markedly lower-level treatment than `summer2`'s `stratify_with()` or camdl's `stratify(by = age)`. Adding an age dimension to an odin model means rewriting every equation with indices; adding one in summer2 means one extra call. The trade is that odin's version has no restrictions — any indexed structure you can write down works — while the stratification-operator approach only supports the transformations the operator implements.

## Strengths

- **The model is a set of equations, not a program.** Order-independent, dependency-sorted, no control flow. This is the property that makes it a specification.
- **Same DSL, two paradigms.** ODE via `deriv()`, discrete-time stochastic via `update()`, with `initial()` common to both and the variable set inferred from it.
- **Compiled.** C++ generation gives serious performance from a language that reads like maths; a JavaScript target exists and CUDA is signposted.
- **Clean separation of language, engine, and inference.** odin → dust → monty is a genuinely modular toolchain, and monty works against any dust model.
- **The likelihood is in the model.** Two lines, and the particle-filter machinery follows.
- **`zero_every` accumulators** — the best answer in the review to "how do I declare incidence".
- **Documented order of events** for discrete-time models, including where data comparison sits.
- **Careful RNG semantics**, including the scalar-versus-array draw distinction.
- **Differentiability as a parameter attribute** (`differentiate = TRUE`), with the coherence rules between `differentiate`, `constant`, and `type` checked at compile time.
- **Restricted names are checked** against C, C++, and JavaScript keywords, so generated code cannot collide.

## Limitations

- **No dimensional types.** Rates, probabilities, counts, and durations are all `real`. The `1 - exp(-r*dt)` conversion is written by hand every time, and writing `r*dt` instead is not an error.
- **Compartmental only.** No agents, no networks, no individual heterogeneity. Stratification-by-array is the only structural axis.
- **Stratification is manual.** Adding a dimension means editing every equation.
- **No intervention vocabulary at all.** Time-varying parameters go through `interpolate()`; anything else is an equation the modeller writes. There is no notion of a scheduled action, a coverage level, or a targeted subgroup.
- **No common random numbers**, so paired scenario comparison is unavailable.
- **R-embedded.** The DSL is written inside an R call, parsed by R's parser, and compiled by R's toolchain. Portability outside R is limited to the JavaScript target.
- **Delays carry implicit state**, and the docs acknowledge that continuing a delayed system is not the same as starting one from a point in state space. This is honest and unresolved.
- **The `parameter(rank = 2)` idiom** for data-determined array dimensions is acknowledged in the docs as an interface the authors expect to change.

## Implications for the lingua franca

1. **Order-independence is a requirement, not a nicety.** odin's dependency-sorted equations are what let a model be read as a specification rather than executed as a script. Any lingua franca whose statements have to be read in order has failed the "no hidden behavior" goal, because execution order *is* hidden behavior.
2. **One `initial()` for both paradigms is the right factoring.** odin identifies the state of the system by what has an initial condition, then lets `deriv()` or `update()` select the paradigm. That is a small, elegant version of the paradigm-independence this project wants: the state space is shared, the dynamics operator differs.
3. **Take `zero_every`.** Declarative auto-resetting accumulators solve the incidence problem generically. Combined with camdl's `incidence(transition)`, this is most of a complete answer to "how do outputs get declared".
4. **Take the RNG semantics rule** — an assignment to an array draws once per element, an assignment to a scalar then broadcast draws once — and state it in the spec. It is the kind of ambiguity that produces silently wrong models.
5. **Document the order of events per paradigm, in the spec.** odin does this for discrete-time models in a dozen lines. A multi-paradigm language needs it for each paradigm and for the interleaving between them.
6. **Do not copy the untyped rate handling.** Writing `1 - exp(-r*dt)` by hand is transparent, but it is exactly the arithmetic that dimensional types exist to check. camdl and Starsim both do better; the lingua franca should follow them, and can keep odin's transparency by making the expansion of a typed rate visible rather than by making the user write it.
7. **Weigh stratification-by-array against stratification-as-operator.** odin's arrays are maximally general and maximally laborious; summer2's `stratify_with()` and camdl's `stratify(by = ...)` are the opposite. The lingua franca probably needs both — an operator for the common cases, explicit indexing underneath it — with the operator documented as sugar that expands to the indexed form, as camdl already does.
8. **Separate language from engine from inference.** The odin/dust/monty split is a good architectural precedent: it means an engine can be replaced, and it means the inference layer is reusable by models that were never written in the DSL.
