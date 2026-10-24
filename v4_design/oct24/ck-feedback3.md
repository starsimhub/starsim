# CK feedback 3

## Terminal interface

Let's have a SUPER nice terminal experience. `sim.describe()` should have formatting, sections, etc. (but also a fallback so it renders nicely in e.g. notebook output which might not support terminal features). Running a sim should also be as elegant and informative as possible. Can think about other improvements too.

## Stratification

Not sure if it's already supported, but in addition to this:

```py
sir.stratify(age=[0, 15, 65], split=[0.3, 0.6, 0.1])
```

let's support

```py
sir.stratify(age={0: 0.3, 15: 0.6, 65: 0.1})
```

Also does stratify work as a kwarg as well? (`sir = ss.Disease(..., stratify={'age': {0: 0.3, 15: 0.6, 65: 0.1}})`)

I don't like "bands" in contacts. Change this:

```py
contacts = ss.contacts([[10, 4, 1], [4, 8, 2], [1, 2, 3]], bands=[0, 15, 65])
kw = dict(networks=contacts, total_pop=1e6, start='2025-01-01', stop='2025-07-01')
sim = ss.Sim(sir, **kw).run()
```

to this:

```py
contacts = ss.contacts(ages==[0, 15, 65], matrix=[[10, 4, 1], [4, 8, 2], [1, 2, 3]])
pars = dict(networks=contacts, total_pop=1e6, start='2025-01-01', stop='2025-07-01')
sim = ss.Sim(sir, **pars).run()
```

## R

We can work on getting R to work later, but basically, the R example in `ten_minutes.qmd` sucks. Instead of this:

```r
library(reticulate)
ss <- import("starsim")

st <- ss$states("S I R")
S <- st[[1]]; I <- st[[2]]; R <- st[[3]]
sir <- ss$Disease(
    infection = ss$Transition(S, I, rate = 0.3 * I),
    recovery  = ss$Transition(I, R, dur = 10),
    init_prev = 0.01,
    name = "sir"
)
sim <- ss$Sim(sir, stop = 100)
sim$run()
n_I <- py_to_r(sim$results$sir$I$values)
cat("Peak infected:", max(n_I), "\n")

msim <- sim$new(solver = "binomial")$run(n_runs = 20L) # A count is an integer: 20L, not 20
```

It should look something like this:

```r
library(rstarsim)
load_starsim()

S, I, R <- states("S I R")
sir <- Disease(
    infection = Transition(S, I, rate = 0.3 * I),
    recovery  = Transition(I, R, dur = 10),
    init_prev = 0.01,
)
sim <- Sim(sir, stop = 100)
sim$run()
sim$plot()

msim <- sim$new(solver = "binomial")$run(n_runs = 20)
msim$plot()
```

Also, the user shouldn't have to call `sim.new()` themselves. it should call itself automatically instead of failing with an error in most cases.

## "In ten minutes" doc

You said this was the best page in the docs. I strongly disagree. Here are the problems with it, also mentioned in /home/cliffk/idm/ssv4/v4_design/oct24/docs_comparison.md. Fix these issues (including the underlying code if necessary) -- but be expansive, don't fix these narrowly, but instead for each one consider the *principle* being applied, and fix that throughout the docs and codebase. You don't need to edit "my" docs (e.g. the Starsim v3 docs which are more or less unchanged).

"a model in two lines" sounds braggy, and is not even true. it states the obvious ("the states are declared first") without properly explaining what's happening or how it links to standard epidemiological concepts. it does a deep dive into "init" without explaining how init_prev, init_n,and init are related. it's obsessed with solvers which i think most users don't care too much about. it references starsim v3, which violates the constitution's "No baggage" rule. "What was guessed" is so claude-y, and too short compared to the other sections, and isn't even accurate (describe() describes everything, not just the guesses). diving into leaky vaccines is too much for 'ten minutes'. "vaccine = ss.vaccinate(ss.Vaccine())" is really ugly and confusing. `dict(none=None, vaccine=vaccine, masks=masks)` should be `scenarios = {"No intervention": None, "Vaccine": vaccine, "Masks": masks}`. things like `,.0f` are going to confuse and lose our non-expert readers. `np.random.default_rng(1).poisson(weekly.values)` cannot be the simplest way to add data to a time series, and you should use a different random seed for the 'truth' vs the sim. `fit = sim.calibrate()` should be `fit = sim.fit()`. "Scenarios and many runs" is too brief; you introduce `sim.new()` magically without saying what it does. "Each result of the runs has a run axis first" does not make sense. median and quantiles should be built into the results, the users shouldn't need to type "np.quantile".


