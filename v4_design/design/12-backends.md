# Backends: paradigms, methods, and conversion

The paradigm is a run-time choice. This document specifies the methods, what each requires of a model, and how the compiler decides between running, approximating, and refusing.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md), [best-practices/compartmental.md](../best-practices/compartmental.md), [best-practices/stochastic-compartmental.md](../best-practices/stochastic-compartmental.md), [best-practices/agent-based.md](../best-practices/agent-based.md), and [best-practices/metapopulation.md](../best-practices/metapopulation.md).

## Think in axes

Not four paradigms in a list, but a lattice with four independent axes. A model selects a point; a conversion moves along one axis.

| Axis | Values | Declared by |
|---|---|---|
| Aggregation | compartment ↔ agent | `method` |
| Stochasticity | deterministic ↔ diffusion ↔ jump ↔ individual | `method` |
| Dwell time | exponential ↔ Erlang ↔ arbitrary | the transition's `dur` |
| Space | single ↔ patch ↔ explicit mobility | a spatial dimension |

This framing is what makes "which conversions are meaningful" answerable per axis rather than per pair of paradigms.

## The methods

```python
sim.run(method='ode')        # deterministic compartmental
sim.run(method='ctmc')       # exact continuous-time Markov chain (Gillespie)
sim.run(method='tau')        # tau-leaping
sim.run(method='binomial')   # chain-binomial, discrete time
sim.run(method='sde')        # diffusion approximation
sim.run(method='ide')        # integro-differential, for arbitrary dwell times
sim.run(method='abm')        # agent-based (the default)
sim.run(method='stochastic') # choose among ctmc/tau/binomial and say which
```

Same model object. `method=` may be given to `Sim` or to `run`; `run` wins.

| Method | Time | State | Exact for | Fails when |
|---|---|---|---|---|
| `ode` | continuous | real-valued | large populations, no extinction question | counts are small or extinction matters |
| `ctmc` | continuous | integer | always (it is exact) | event count is intractable |
| `tau` | continuous | integer | propensities roughly constant over τ | fast transitions relative to τ |
| `binomial` | discrete | integer | small `rate × dt` | `rate × dt` is large |
| `sde` | continuous | real-valued | counts large in every compartment | any compartment falls below ~20 |
| `ide` | continuous | real-valued | arbitrary dwell times | large state spaces (memory cost) |
| `abm` | discrete | individual | everything the model can declare | population size × step count |

Validity conditions are **checked**, not documented: `W1201`–`W1206` name the condition, the offending quantity, and the method that would not have the problem.

### Automatic selection

```text
>>> ss.Sim(sir, n=1000).run(method='stochastic')
n = 1000, 3 states, 2 transitions
method: ctmc (exact) — event count tractable at this population size
```

```text
>>> ss.Sim(sir, n=10_000_000).run(method='stochastic')
method: tau — exact CTMC would require ~1e9 events
  ! tau-leaping is an approximation; tau chosen adaptively, leap condition eps=0.03
```

The selection MUST be printed with its reason and MUST be overridable by one keyword. The heuristic MUST bias toward exactness when compartment counts are small, because that is exactly when extinction questions arise and when an approximation is most misleading.

**Naming the approximation is the point.** A trajectory from tau-leaping is not a trajectory from the CTMC, and a framework that silently substitutes one for the other has produced a result whose provenance is lost.

## The backend contract

A backend consumes the IR ([14-ir.md](14-ir.md)) and MUST implement:

```
capabilities(ir)  -> per-element {exact | approximate(closure) | refused(reason)}
lower(ir, config) -> an executable
step(state, dt)   -> state                # or solve(state, tspan) for continuous methods
results(state)    -> the declared result set
```

Requirements on any backend:

1. It MUST NOT read anything outside the IR and its configuration.
2. It MUST report reduced capability rather than silently approximating. Every approximation it makes MUST be named in `capabilities()` before the run and in `summary()` after it.
3. It MUST honor the RNG contract in [11-random.md](11-random.md), including determinism under parallelism.
4. It MUST produce the same declared result names as every other backend, so that results are comparable across methods without translation.
5. Where the model declares something it cannot represent, it MUST raise the specific `E8xx` error, never a generic failure.

## The capability matrix

```python
sir.capabilities()
```

```text
                        ode    ctmc   tau    sde    abm
mixing=Homogeneous       ok     ok     ok     ok     ok
mixing=MixingPool        ok     ok     ok     ok     ok
mixing=RandomNet         ~      ~      ~      ~      ok    ~ mean-field limit, exact as n->inf
mixing=SexualNet         X      X      X      X      ok    X concurrent partnerships: no closed form
dur=lognorm              ~      ~      ~      ~      ok    ~ nearest Erlang; discrepancy reported
dur=erlang(k=3)          ok     ok     ok     ok     ok
eligible=ss.age>65       ok     ok     ok     ok     ok      (age is a dimension: rank=group)
eligible=lambda ...      X      X      X      X      ok    X opaque predicate: rank=agent
ss.process(...)          X      X      X      X      ok    X arbitrary Python
sim.transmissions        X      X      X      X      ok    X requires individual provenance
```

This is the honest form of the project's central claim. The claim is not "any model runs any way"; it is **"you will be told, by name, what does not"**.

Granularity is per (element × method), where an element is a route, a transition, a dimension, an effect, or an escape hatch. Per-route granularity would be cheaper and would not localize the refusal to the declaration that caused it.

## Three classes of conversion

### Exact

No approximation; the same mathematics, differently executed.

| Conversion | Condition |
|---|---|
| Homogeneous or contact-matrix mixing → `ode`, `ctmc`, `tau`, `binomial`, `abm` | always |
| Exponential dwell time → a rate, or a sampled clock | always |
| Erlang(k) dwell time → k sub-compartments, or a sampled time | always |
| A dimension → a compartment axis, or an agent property | always |
| A group-rank predicate → a coordinate selection | always |

### Approximate, with the approximation named

| Conversion | Named as | Reported |
|---|---|---|
| Non-Erlang dwell time → compartmental | nearest Erlang or hyper-Erlang fit | the fitted parameters and the distributional discrepancy |
| Network → compartmental | **a closure**, chosen explicitly | `to_ode(closure='heterogeneous_pairwise')` |
| Large populations → `tau` or `sde` | leap condition or count threshold | the chosen τ, or the minimum compartment count |
| Continuous agent heterogeneity → a dimension | a discretization | the bin edges |
| `n_agents < n` → scaled results | a resolution choice | the scale factor and the inflated variance |

Network-to-compartmental conversion MUST NOT be presented as a single operation. It is a choice of closure — individual-based, pair-based, homogeneous or heterogeneous pairwise, compact pairwise, effective degree, mean field, edge-based compartmental — and the choice MUST be named in the output. The default, if any, is `mean_field`, and it MUST be labelled as the crudest rung.

### Refused, by name

- Persisting concurrent partnerships → any compartmental form. There is no closure; concurrency is the mechanism being modeled.
- An `ss.process`, or an agent-rank predicate → any method but `abm`.
- Individual histories, contact tracing, transmission trees → any method but `abm`.
- A closure that has not been derived for the model class in question.

The last is the important one. Analytic closures exist for SIS and SIR, not for arbitrary declared processes; each is a mathematical derivation, not a compiler pass. Where the compiler cannot derive the conversion, it MUST refuse.

```text
E804 CapabilityError: cannot run 'hiv' with method='ode'.
     route SexualNet has concurrent, persisting partnerships.
     Concurrency has no mean-field closure: a compartmental rendering would discard
     the mechanism the model exists to represent.
     Nearest available: mixing=ss.MixingPool(matrix=...) for an age-mixing
     approximation, which is a different model.
     See design/12-backends.md and best-practices/paradigm-conversion.md.
```

## Per-transition scale

A transition may opt out of the run's method:

```python
polio = ss.Disease(
    importation = ss.transmission('S -> I', beta=1e-6, scale='jump'),   # rare: exact events
    infection   = ss.transmission('S -> I', beta=0.05),                 # bulk: follows the method
    recovery    = ss.progression('I -> R', dur=ss.days(20)),
)
```

`scale=` takes `'jump'` (simulate as discrete events regardless of method), `'continuous'` (treat as a flow), or `None` (follow the run method, the default).

This is the elimination and eradication case, where the deterministic model says the epidemic is over and the real question is the probability of reintroduction. It is one keyword, it defaults to doing nothing, and it is the sharpest available version of paradigm independence: not "this model is stochastic" but "this transition is a jump process and that one is a flow", in one system.

Hybrid coupling at the boundary — a jump transition feeding a continuous compartment — MUST conserve counts exactly, and the backend MUST round in a way that does not systematically bias the flow.

## Compilation

Every backend compiles; none interprets the model text.

```
IR --normalize--> flat transition list + dependency DAG --lower--> executable
```

The dependency graph is topologically sorted; declaration order is irrelevant by construction. The compartmental backends target a JIT-and-autodiff array library, because the same compilation that makes them fast makes **gradients of the whole model with respect to its parameters** available, which turns gradient-based calibration and sensitivity analysis from a project into a keyword ([13-inference.md](13-inference.md)). A small model with no gradient required may use a plain ODE solver; the choice is an implementation detail and is printed.

## Compartmental specifics

```python
sim.equations()     # the generated ODE system
sir.R0()            # from the next-generation matrix, symbolically where possible
sir.jacobian()
sir.fixed_points()
```

```text
>>> ss.Sim(sir, mixing=ss.Homogeneous(n_contacts=10), n=1000).equations()
dS/dt = -10 * 0.05 * S * I/N
dI/dt =  10 * 0.05 * S * I/N - I/10
dR/dt =  I/10
```

Nothing in the disease declaration is ODE-specific; the `10` comes from the route and the `1/10` from the dwell time.

The symbolic outputs are **opt-in and must say when they give up**. Symbolic analysis is tractable for a few compartments and intractable for a stratified model — the very case where the derivation is hardest by hand. The required behavior is: attempt symbolically, fall back to numerical, and report which one was obtained. `R0()` is the number the audience actually asks for, and it is computable rather than derived by hand once the model is a graph with a declared route.

Erlang expansion multiplies compartments, and stacked with stratification it multiplies fast. The resolved compartment count MUST be reported, with the thresholds in [05-dimensions.md](05-dimensions.md) §Size.

## Agent-based specifics

The agent-based backend is Starsim's, and its architecture does not need replacing: typed per-agent state arrays with declared dtypes and defaults, persistent UIDs with a live view so births and deaths do not invalidate references, vectorized operations, and one module abstraction.

| Declaration | Realization |
|---|---|
| `ss.State('S')` | a boolean per-agent array |
| `ss.transmission(...)` | enumerate route contacts, compute per-contact probability, draw |
| `ss.progression(dur=...)` | sample a time, schedule the event |
| `sir.stratify(risk=[...])` | a per-agent property |
| `ss.Effect(cond, x, multiply=)` | a registered modifier, resolved in phase 3 |
| `ss.Outcome(...)` | a delayed, thinned draw from the transition's event stream |

**The select-thin-act primitive.** Nearly every agent-based process reduces to *for the agents matching this predicate, with this probability, do this*:

```python
ss.act(who=(sir.I & ~ss.scheduled('recovery')), p=0.1, do=...)
```

`ss.act` is the shape of the escape hatch, not something users write for an SIR — the declared transitions *are* this operation, compiled. Bitsets are the natural implementation of the predicate side.

Preallocation, dead-flagging, and columnar storage are implementation, not semantics. A declarative model states its maximum population, its node set, and its output shape, so a backend can preallocate without the modeler thinking about it, and `calc_capacity`-style arithmetic MUST NOT be exposed.

## Stochastic-compartmental specifics

Time-varying rates MUST work under every method or be refused by name; under an exact continuous-time method this requires the inhomogeneous-Poisson algorithm, which most frameworks skip or approximate.

Continuous-time backends have no `dt` and therefore no rate-to-probability conversion at all, which is a real correctness advantage and the reason they are preferred where tractable. "Results at time t" then becomes a resampling question, and the backend MUST resample onto the requested output grid by default rather than leaving it to the user.

Extinction reporting ([09-results.md](09-results.md)) is required for every method with integer state, because it is the actual output of the paradigm.

## Testing the claim

The paradigm-independence claim is empirical and MUST be tested, not asserted. [18-conformance.md](18-conformance.md) specifies the suite: for every model in the corpus, run every backend the capability matrix permits and assert agreement within a stated tolerance. **A conversion that is not tested is a conversion that is not claimed.**

## Rejected

- **A separate model class per paradigm.** Twenty-plus separately implemented model families is the cost estimate for not having this language.
- **Stochasticity as a model property.** It is a run method.
- **Silent approximation of any kind.** The category error this document exists to prevent.
- **Automatic conversion to whatever is fastest.** The paradigm changes what questions the model can answer; it is the modeler's choice, informed by the matrix.
- **Compartmental as the default.** The default stays agent-based; ODE is one keyword away.
- **Making the user write the derivative function.** Positional, unchecked, unstratifiable.
