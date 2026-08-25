# epipack

|  |  |
|---|---|
| **Language** | Python (with SymPy for the symbolic layer) |
| **Paradigms** | Compartmental — deterministic ODE, stochastic well-mixed (Gillespie), and stochastic on networks; all from one process specification |
| **Specification style** | Reaction-style tuples — the model is a list of processes |
| **Version reviewed** | 0.1.x |
| **Licence** | MIT |
| **Code** | <https://github.com/benmaier/epipack> |
| **Docs** | <http://epipack.benmaier.org> |
| **Paper** | [10.21105/joss.03097](https://doi.org/10.21105/joss.03097) |

## What it's for

epipack exists to make the *analysis* steps of compartmental modelling cheap enough that changing the model stays cheap. Its stated motivation is that adding or removing a compartment with pen and paper is easy, while redoing the ODE derivation, the numerical integration, the stochastic simulation, and the stability analysis is not.

For this project it is the single most direct piece of evidence that **one process specification can drive multiple execution paradigms**, because that is literally its API surface: the same list of tuples goes to `EpiModel` (numeric ODE + Gillespie), `SymbolicEpiModel` (symbolic ODEs, Jacobian, fixed points), or `StochasticEpiModel` (network simulation).

## How a model is specified

The model is a list of reaction tuples. A 3-tuple is a spontaneous transition, a 5-tuple is a transmission:

```python
from epipack import EpiModel

S, I, R = list("SIR")
N = 1000

SIRS = EpiModel([S, I, R], N)\
    .set_processes([
        (S, I, 2.5, I, I),   # S + I --(2.5/day)--> I + I    transmission
        (I, 1, R),           # I     --(1/day)-->   R         transition
        (R, 1/14, S),        # R     --(1/14/day)-> S         waning
    ])\
    .set_initial_conditions({S: N-10, I: 10})

result_int = SIRS.integrate(np.linspace(0, 40, 1000))   # deterministic
t_sim, result_sim = SIRS.simulate(40)                    # stochastic, same model
```

Swap the class and the same tuples become symbolic:

```python
import sympy as sy
from epipack import SymbolicEpiModel

S, I, R, eta, rho, omega = sy.symbols("S I R eta rho omega")

SIRS = SymbolicEpiModel([S, I, R]).set_processes([
    (S, I, eta, I, I),
    (I, rho, R),
    (R, omega, S),
])

SIRS.ODEs_jupyter()                             # renders the ODE system
SIRS.jacobian()                                 # symbolic Jacobian
SIRS.find_fixed_points()                        # symbolic fixed points
SIRS.get_eigenvalues_at_disease_free_state()    # {-omega: 1, eta - rho: 1, 0: 1}
```

Or the network version, where transmission is edge-mediated:

```python
model = StochasticEpiModel([S, I, R], N, links)\
    .set_link_transmission_processes([(I, S, 1.0, I, I)])\
    .set_node_transition_processes([(I, 1.0, R)])\
    .set_random_initial_conditions({S: N-5, I: 5})
```

Note that this last one is a *different* process vocabulary — link transmission and node transition are separated — rather than the same tuples reinterpreted. The unification is not quite complete, which is itself a useful finding.

## Core abstractions

| Concept | epipack form | Notes |
|---|---|---|
| Compartment | a string, or a SymPy symbol | Symbols in the symbolic model, strings elsewhere |
| Transition | `(A, rate, B)` | Spontaneous |
| Transmission | `(A, B, rate, C, D)` | `A + B → C + D`; general enough to express fission, fusion, birth, death |
| Rate | number, callable `f(t, y)`, or SymPy expression | Time- and state-dependent rates in all three modes |
| Network | edge list of `(u, v, weight)` | `StochasticEpiModel` only |
| Conditional transmission | `set_conditional_link_transmission_processes` | Cascading events on a link — e.g. contact tracing |
| Model variants | `EpiModel`, `SymbolicEpiModel`, `StochasticEpiModel`, `MatrixEpiModel`, `SymbolicMatrixEpiModel` | Five classes over one conceptual model |

## Time and rates

Rates are plain numbers, callables of `(t, y, *args)`, or SymPy expressions in `t`. Time-varying transmission is written directly:

```python
def infection_rate(t, y, *args, **kwargs):
    return 3 + np.sin(2*np.pi*t/100)
```

or symbolically, in which case `SIRS.ODEs_jupyter()` renders the forced system and the Gillespie simulation uses the algorithm for inhomogeneous Poisson processes. Getting a *correct* stochastic simulation of a time-varying-rate system for free from the same declaration is a real capability that most frameworks skip or approximate.

There are no units and no dimensional types.

## Analysis capabilities

This is where epipack is unlike anything else in the review. From the symbolic model:

- the ODE system, rendered in LaTeX
- the Jacobian, symbolically
- fixed points, symbolically
- eigenvalues at the disease-free state — that is, **the epidemic threshold, derived**

`SIRS.get_eigenvalues_at_disease_free_state()` returning `{-omega: 1, eta - rho: 1, 0: 1}` means R₀ > 1 ⟺ η > ρ, obtained from the model declaration with no derivation by the modeller.

This is the payoff of transitions-as-data taken further than anyone else takes it: because the model is a symbolic object, standard dynamical-systems analysis is a method call.

## Strengths

- **One process list, three execution modes** — deterministic ODE, well-mixed stochastic, and network stochastic — which is the paradigm-independence goal, demonstrated.
- **Symbolic analysis for free**: ODEs, Jacobian, fixed points, eigenvalues, epidemic threshold.
- **A genuinely general reaction form.** `(A, B, rate, C, D)` covers transmission, fission, fusion, birth, and death in one primitive rather than a vocabulary of flow kinds.
- **Time- and state-dependent rates in every mode**, including a correct inhomogeneous-Poisson Gillespie implementation.
- **Conditional link transmission** — cascading events on an edge — which is the natural way to express contact tracing and is rare elsewhere.
- **Tiny specification.** An SIRS model is five lines and reads like the reaction scheme.
- **Interactive exploration** (`InteractiveIntegrator`) and animated network visualisation.

## Limitations

- **Effectively unmaintained.** Dependency pins (`sympy==1.6`, `pyglet<1.6`, Python 3.6–3.8) and the changelog put it well behind current Python. The ideas are current; the package is not.
- **The unification is incomplete.** `StochasticEpiModel` uses a different process vocabulary (`set_link_transmission_processes` / `set_node_transition_processes`) from `EpiModel` (`set_processes`). The network paradigm is adjacent to the others, not the same specification reinterpreted.
- **No stratification operator.** Age structure means writing out `S_child`, `S_adult` and every process between them by hand — the opposite end of the spectrum from summer2.
- **No dimensional types, no units.**
- **No intervention vocabulary.** Interventions are time-varying rates.
- **No observation model or inference layer.** Simulation and analysis only.
- **Network support depends on a manually installed C++ package** (`SamplableSet`) for acceptable performance.
- **Fixed points and eigenvalues do not scale.** Symbolic analysis is tractable for a few compartments and intractable for a stratified model — the very cases where the derivation is hardest to do by hand.

## Implications for the lingua franca

1. **The strongest existing proof of the central premise.** One specification, three paradigms, in a working package. Whatever else is uncertain about this project, "the same model text can be deterministic, stochastic, and network-simulated" is not.
2. **The reaction form `A + B → C + D @ rate` is a candidate primitive.** It is more general than summer2's named flow kinds and more structured than an arbitrary rate expression. The trade-off is exactly the one summer2 illustrates from the other side: a general reaction is easy to write and hard to stratify automatically, because the framework does not know what it means. The lingua franca probably needs the general form as the primitive and named kinds as annotated sugar over it — which is also how camdl handles the sugar/primitive relationship.
3. **Symbolic analysis is an argument for the IR, not a feature request.** epipack gets the Jacobian and the epidemic threshold because the model is symbolic data. If the lingua franca's IR is symbolic in the same sense, R₀ derivation, sensitivity analysis, and structural identifiability checks come along with it. camdl already exploits this for gradients; epipack shows the wider payoff.
4. **Watch where the unification breaks.** epipack's network mode needed a separate vocabulary because edge-mediated transmission genuinely is different from mass-action — the rate is per-edge rather than per-population-pair. Starsim's `Route` abstraction is the other attempt at this seam. **How transmission is expressed such that mass-action, contact-matrix, and edge-based forms are the same declaration is the hardest unsolved problem in this review**, and it is where the lingua franca has to do original work.
5. **Time-varying rates should be uniform across paradigms.** epipack accepts a callable or a symbolic expression and does the right thing in ODE, Gillespie, and inhomogeneous-Poisson modes. This should be the standard: a rate is an expression in time and state, and each backend is responsible for handling that correctly or refusing.
6. **Symbolic methods do not scale, and the language should say so.** For a stratified model, `find_fixed_points()` is intractable. The capability-check discipline camdl uses — name the limitation rather than fail slowly — applies here too.
