# EoN (Epidemics on Networks)

|  |  |
|---|---|
| **Language** | Python, over NetworkX |
| **Paradigms** | Stochastic simulation on networks, **plus** the full hierarchy of analytic ODE approximations to those simulations |
| **Specification style** | Function calls for the standard models; a pair of directed graphs for custom processes |
| **Version reviewed** | 1.92 |
| **Licence** | MIT |
| **Code** | <https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks> |
| **Docs** | <http://epidemicsonnetworks.readthedocs.io/> |
| **Paper** | [10.21105/joss.01731](https://doi.org/10.21105/joss.01731); accompanies Kiss, Miller & Simon, *Mathematics of Epidemics on Networks* (Springer, 2017) |

## What it's for

EoN is the software companion to a textbook. It does two things: it simulates SIS and SIR (and more general) processes on networks, and it implements **the analytic approximations to those same processes** — the mean-field, pairwise, and effective-degree closures that the book derives.

That second half is why it is in this review. EoN is the only package here that ships, side by side, a stochastic network simulation *and* a graded family of ODE approximations to it. The paradigm-conversion question this project has to answer — when is a compartmental model a valid summary of an agent-based one? — is a question EoN's API answers empirically, by letting you run both.

## How a model is specified

**Standard models** are function calls over a NetworkX graph:

```python
import EoN, networkx as nx
G = nx.configuration_model([...])
t, S, I, R = EoN.fast_SIR(G, tau=0.5, gamma=1.0, rho=0.01)
```

**Custom processes** are specified as **two directed graphs**:

```python
import networkx as nx

# Spontaneous transitions: nodes are statuses, edges are transitions
H = nx.DiGraph()
H.add_edge('E', 'I', rate=0.6)      # E -> I
H.add_edge('I', 'R', rate=0.1)      # I -> R

# Neighbour-induced transitions: nodes are (source_status, target_status) pairs
J = nx.DiGraph()
J.add_edge(('I', 'S'), ('I', 'E'), rate=0.1)   # an I-S edge makes the S become E

IC = {node: 'S' for node in G}
t, D = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses=('S','E','I','R'), tmax=100)
```

`H` is the spontaneous-transition graph; `J` is the neighbour-induced-transition graph, whose nodes are *pairs* of statuses so that `('I','S') → ('I','I')` reads as "an infectious–susceptible edge turns the susceptible one infectious".

**The model is literally a graph.** Not a graph metaphor — a `networkx.DiGraph` you can traverse, draw, and edit programmatically. This is transitions-as-data taken about as far as it goes, and it is worth noting that the representation the textbook uses for exposition (Fig 4.3 of the book, cited in the docstring) is the same object the code executes.

Rates can be modified per node or per edge, in two ways: a `weight_label` naming an attribute in `G` that multiplies the rate, or a `rate_function(G, u)` / `rate_function(G, source, target)` computing a scale factor from the network. The docstring is explicit that the rate function may not depend on node *statuses*, because it must be computable once at the start — a stated, checkable restriction that keeps the Gillespie implementation correct.

## The analytic hierarchy

The `EoN.analytic` module implements, for both SIS and SIR:

| Approximation | What it tracks |
|---|---|
| `*_individual_based` | One equation per node — exact-ish, N equations |
| `*_pair_based` | Node and edge pair probabilities |
| `*_homogeneous_pairwise` | Pair counts, homogeneous degree |
| `*_heterogeneous_pairwise` | Pair counts by degree class |
| `*_compact_pairwise` | Reduced pairwise |
| `*_super_compact_pairwise` | Further reduced |
| `*_effective_degree` | Susceptible-neighbour count as a state |
| `*_compact_effective_degree` | Reduced effective degree |
| `*_homogeneous_meanfield` | Standard mass-action ODEs |
| `*_heterogeneous_meanfield` | Degree-block ODEs |
| plus `EBCM` | Edge-based compartmental model |

Each also has a `*_from_graph` variant that extracts the required moments (degree distribution, `Nk`, `NkNl`) from an actual network. So the workflow is: build a network, run the exact simulation, and run any approximation *parameterised from that same network*, then compare.

This is a **conversion ladder made executable**. Every step down it is an explicit closure assumption, and the package lets you measure what each assumption costs on your network.

Supporting machinery: `get_Pk`, `get_PGF`, `get_PGFPrime`, `get_PGFDPrime` (degree distribution and its probability generating function), and `estimate_R0(G, tau, gamma)`.

## Time

Continuous time throughout — Gillespie and the fast event-driven algorithms for simulation, `scipy.integrate.odeint` for the analytics. No timestep, so no rate-to-probability conversion. Rates are per-unit-time floats; no units.

## Strengths

- **The model is a pair of directed graphs.** Fully introspectable, drawable, and programmatically constructible — and it is the same representation the textbook uses to explain the model.
- **The spontaneous / neighbour-induced split.** Two transition kinds, cleanly separated, covering SIS, SIR, SEIR, SIRS, competing and cooperating pathogens, and complex multi-status processes. Exactly epipack's and EpiHiper's split, arrived at independently.
- **The analytic hierarchy, executable and parameterisable from a real network.** Nothing else in the review lets you quantify the cost of a closure assumption.
- **Conditional / cascading transmission** — `Gillespie_complex_contagion` and the conditional link transmission facility, which is how contact tracing gets expressed.
- **Per-node and per-edge rate heterogeneity**, by attribute weight or by a rate function, with a stated restriction (no status dependence) that preserves algorithmic correctness.
- **Fast event-driven algorithms** (`fast_SIR`, `fast_SIS`) that are orders of magnitude quicker than the general Gillespie implementation, with the docstrings saying so plainly.
- **`estimate_R0` and PGF machinery** for network-theoretic quantities.
- **Textbook provenance.** The algorithms are the published ones, with errata tracked.

## Limitations

- **No stratification, no demographics, no interventions.** Rates can vary by node attribute; there is no vocabulary for age structure, births, deaths, vaccination, or treatment.
- **Networks are inputs.** EoN consumes a NetworkX graph; it has no network-specification vocabulary and no dynamic-network model beyond the temporal-network simulation entry points.
- **The two-graph specification is idiosyncratic.** Encoding "an I–S edge infects the S" as an edge from the tuple-node `('I','S')` to `('I','I')` is precise and takes real explaining, and the docstring runs to hundreds of lines because of it.
- **Only the second node in a pair may change status**, which is stated but limits the process class.
- **No dimensional types, no dates, no observation model, no inference.**
- **The analytic and simulation halves have different interfaces.** `Gillespie_simple_contagion(G, H, J, ...)` versus `SIR_heterogeneous_pairwise(Sk0, Ik0, ...)`. The approximations are only implemented for SIS and SIR, not for an arbitrary `H`/`J` process — so the conversion ladder exists for the standard models only.
- **Maintenance is intermittent** — a five-year gap between 1.1 and 1.2 — though 1.92 (2026) is recent.

## Implications for the lingua franca

1. **EoN is the reference for what paradigm conversion actually costs.** The project's interconversion goal implicitly claims that a compartmental rendering of an agent-based model is meaningful. EoN's analytic hierarchy is a decade of mathematics saying *when*: mean-field ignores correlations, pairwise closes at pairs, effective-degree tracks susceptible-neighbour counts, and each is right under different network conditions. **The lingua franca should not present network → compartmental conversion as a single operation.** It is a choice of closure, and the choice should be named in the output the way `SIR_heterogeneous_pairwise` names it.
2. **The `H` / `J` split is the third independent arrival at spontaneous-versus-induced transitions.** EoN's two graphs, epipack's 3-tuples versus 5-tuples, and EpiHiper's `transitions` versus `transmissions` are the same distinction. Together with the fact that each framework needed it, this looks like a genuine primitive: **a transition is either autonomous or contact-mediated, and the two have different mathematics.** The lingua franca should make the distinction structural rather than leaving it implicit in whether a rate expression happens to reference another compartment.
3. **Take the rate-modifier restriction seriously.** EoN allows per-node and per-edge rate scaling but forbids it depending on node status, because that would break the event-driven algorithm's precomputation. This is a good example of a **backend capability constraint surfacing in the specification**: some rate expressions are simulable by some algorithms and not others. camdl's capability-check discipline is the general form; EoN shows a concrete instance.
4. **Deriving R₀ from the model plus the network should be a language feature.** `estimate_R0(G, tau, gamma)` and epipack's `get_eigenvalues_at_disease_free_state()` are two routes to the same quantity from two paradigms. If the IR is symbolic and the contact structure is declared, the lingua franca can offer this generally — and R₀ is the number the audience actually asks for.
5. **The graph representation is the natural IR for the disease process.** EoN executes a `DiGraph`; MIRA's `TemplateModel` is a graph; Atomica's transition matrix is an adjacency matrix; EpiHiper's states-and-transitions JSON is an edge list. Whatever the surface syntax, **the disease model is a labelled directed graph over states**, and the design should say so explicitly.
6. **Note that the ladder only exists for standard models.** EoN's approximations cover SIS and SIR, not arbitrary `H`/`J` processes, because each closure has to be derived. That is a real warning about the cost of general paradigm conversion: it is a mathematical derivation per model class, not a compiler pass. Where the lingua franca cannot derive the conversion, it should refuse rather than approximate silently.
