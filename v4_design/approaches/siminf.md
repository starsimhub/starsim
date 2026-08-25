# SimInf

|  |  |
|---|---|
| **Language** | R, with generated C compiled at model-definition time |
| **Paradigms** | Stochastic compartmental (continuous-time Markov chain, Gillespie) on a network of nodes, with scheduled events |
| **Specification style** | Transitions as strings — `"S -> propensity -> I"` — parsed and compiled to C |
| **Version reviewed** | 10.1.0 |
| **Licence** | GPL-3 |
| **Code** | <https://github.com/stewid/SimInf> |
| **Docs** | <https://CRAN.R-project.org/package=SimInf> |
| **Paper** | [10.18637/jss.v091.i12](https://doi.org/10.18637/jss.v091.i12) |

## What it's for

SimInf simulates disease spread over networks of subpopulations, using Gillespie's algorithm for within-node dynamics and **scheduled events** for between-node movement. It was built for livestock disease modelling, where the data is a national animal-movement register: real, dated, recorded transfers of animals between holdings.

That origin produced a design worth studying — a clean separation between **continuous stochastic dynamics** (the transitions) and **discrete scheduled interventions on the population** (the events) — that generalises well beyond veterinary epidemiology.

## How a model is specified

```r
model <- mparse(
  transitions = c(
    "S -> beta * S * I / N -> I",
    "I -> gamma * I -> R",
    "N <- S + I + R"
  ),
  compartments = c("S", "I", "R"),
  gdata = c(beta = 0.16, gamma = 0.077),
  u0 = data.frame(S = 100, I = 1, R = 0),
  tspan = 1:100
)

result <- run(model, seed = 22)
plot(result)
```

`mparse` parses the transition strings, builds the state-change matrix `S` and the dependency graph `G`, generates C code, compiles it, and returns a model object. Syntax:

- `"X -> rate_expr -> Y"` — move individuals from `X` to `Y` at propensity `rate_expr`
- `"@"` is the empty set: `"I -> mu*I -> @"` is death, `"@ -> lambda -> S"` is birth
- `"N <- S + I + R"` defines an intermediate variable, usable in any propensity
- `"(int)N <- S+I+R"` forces integer type

**Order does not matter** — the parser resolves the dependency graph. Intermediate variables are the same idea as odin's derived quantities and camdl's `let` bindings.

## Data layers and the local/global split

SimInf's parameter model is explicit about scope in a way most frameworks are not:

| Layer | Meaning | Form |
|---|---|---|
| `u0` | Initial discrete state per node | data.frame, one row per node |
| `gdata` | **Global** data — parameters common to all nodes | named numeric vector |
| `ldata` | **Local** data — node-specific parameters | data.frame or matrix, one column per node |
| `v0` | Initial **continuous** state per node | Continuous variables alongside the discrete compartments |

The `u`/`v` split is unusual and useful: `u` holds integer compartment counts governed by the Markov chain, `v` holds continuous node-level variables (environmental contamination, accumulated pathogen load) updated by a post-time-step function. The `SISe` family of built-in models uses exactly this — a SIS disease coupled to a continuous environmental compartment.

`gdata` versus `ldata` is the same distinction MetaCast draws as `universal_params` versus `subpop_params`, made here as a first-class part of the model's data layout.

## Scheduled events

The second half of the design. Events are a data frame of dated, typed population movements:

- **Exit** — individuals leave a node
- **Enter** — individuals arrive
- **Internal transfer** — individuals move between compartments within a node
- **External transfer** — individuals move between nodes

Two matrices control them. The **select matrix `E`** says which compartments an event can draw from and with what sampling weights; the **shift matrix `N`** says how individuals are moved between compartments during an event.

The consequence is that a real animal-movement register — 200,000 dated transfers between holdings — is loaded directly as the events data frame, and the model runs against actual movement data rather than a mobility model. That is a genuinely different way to drive a metapopulation, and it is the one the data often supports.

## Core abstractions

| Concept | SimInf name | Notes |
|---|---|---|
| Compartment | `compartments` | Discrete, per node |
| Transition | `"S -> propensity -> I"` | Parsed to a state-change matrix and a dependency graph |
| Intermediate | `"N <- S + I + R"` | Order-independent, typed (`double` or `(int)`) |
| Global parameter | `gdata` | Same in every node |
| Local parameter | `ldata` | Per node |
| Continuous state | `v0` + `pts_fun` | Per-node continuous variables updated post-step |
| Event | events data.frame + `E` (select) + `N` (shift) | Exit / enter / internal / external transfer |
| Escape hatch | `pts_fun`, `pre_code` | C code inserted into the generated model |
| Inference | `abc`, `pfilter`, `pmcmc` | ABC, particle filter, particle MCMC, built in |
| Packaging | `package_skeleton()` | Turns a model into an installable R package |

## Time

`tspan` is the vector of times at which output is recorded; the underlying process is continuous-time Gillespie, so there is no `dt` and no rate-to-probability conversion anywhere. Events are dated by time index.

Being continuous-time is a real advantage for correctness — the exponential-conversion errors that afflict discrete-time frameworks simply do not arise — and a real limitation for scale, since Gillespie cost grows with event count.

No dimensional types; propensities are C expressions and their units are the modeller's responsibility.

## Strengths

- **Transitions as parsed strings that read like the reaction.** `"S -> beta * S * I / N -> I"` is close to the notation a modeller would write, is data before it is code, and compiles to C.
- **Order-independent intermediate variables**, with an optional integer type annotation.
- **Explicit global/local parameter split** as part of the model's data layout.
- **Discrete and continuous state side by side** (`u` and `v`), which is the natural formulation for environmentally-transmitted diseases.
- **Scheduled events as a first-class, data-driven mechanism**, with select and shift matrices controlling which compartments are affected and how.
- **Real movement data as the metapopulation driver.** No mobility model needed when a register exists.
- **Continuous-time Gillespie**, so no timestep-conversion errors.
- **Inference built in**: ABC, particle filter, particle MCMC.
- **`package_skeleton()`** turns a model into a distributable R package — a genuinely good idea for reproducibility that nothing else here offers.
- **OpenMP parallelism** across replicates and nodes.

## Limitations

- **Transitions are strings.** The syntax is readable but unchecked until parse, and the propensity expression is C, so an error is a C compilation error surfacing through R.
- **The `E` and `N` matrices are opaque.** Understanding what an event does requires reading two matrices alongside the event data frame. This is the least legible part of an otherwise clear design, and it is the part encoding the most important semantics.
- **Gillespie only.** No ODE limit, no tau-leaping, no agent-based mode. Large populations are expensive.
- **No agents.** Individuals within a node are indistinguishable; individual heterogeneity is not representable except by adding compartments.
- **No dimensional types**, no calendar dates in the core (times are indices).
- **No intervention vocabulary** beyond events; interventions are parameter changes or scheduled transfers.
- **Compilation required at model definition**, so a working C toolchain is a hard dependency.
- **No observation model**; inference compares to compartment counts through user-supplied functions.

## Implications for the lingua franca

1. **The transition string form is a good readability target.** `"S -> beta * S * I / N -> I"` and camdl's `infection : S --> I @ beta * S * (I / N)` are the same shape; camdl's adds a name and a type system. That convergence suggests the notation is close to right, and that the improvements needed are names, types, and a real parser rather than a different form.
2. **Take the events mechanism, and fix its legibility.** "At time t, move n individuals of these compartments from node a to node b" is a primitive the lingua franca needs, because it is how real intervention and movement data arrive. SimInf's version is data-driven and correct; its select/shift matrix encoding is the wrong surface. MetaCast's transfer dictionaries (`{from_coordinates, to_coordinates, states, parameter}`) are the readable version of the same semantics, and the two together suggest what the primitive should look like.
3. **The `u`/`v` split — discrete stochastic state alongside continuous state — is worth adopting.** Environmental reservoirs, accumulated exposure, and vector abundance are all naturally continuous and coupled to a discrete disease process. Frameworks without this force everything into compartments. Starsim's `FloatArr` states play a similar role at the agent level; SimInf's is at the node level.
4. **Global versus local parameter scope should be declared.** SimInf's `gdata`/`ldata` and MetaCast's `universal_params`/`subpop_params` are the same idea arrived at twice. In a language with dimensions, this becomes "a parameter has a shape" — which is epymorph's answer (`Shapes.Scalar` versus `Shapes.N` versus `Shapes.TxN`) and is the more general form.
5. **Continuous-time execution should be a backend, not a paradigm.** SimInf shows the correctness benefit of never converting a rate to a per-step probability. camdl already offers Gillespie alongside chain-binomial and ODE from one model. The lingua franca should treat discrete-time versus continuous-time as a backend choice with a declared cost, not as a property of the model.
6. **`package_skeleton()` is an idea worth stealing outright.** A command that turns a model into a distributable, installable, citable artefact directly serves reproducibility, and it is trivial once the model is data.
