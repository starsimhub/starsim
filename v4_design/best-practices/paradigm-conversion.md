# Paradigm conversion

The project's reason for existing, and the place where honesty matters most.

## Recommendation

1. **Paradigm is a run-time choice, not a property of the model.** One model text; `run(method=...)` selects the backend.
2. **Think in axes, not in a list of paradigms.** Aggregation, stochasticity, dwell time, and space are four independent axes; a model picks a point and a conversion moves along one axis.
3. **Classify every conversion as exact, approximate-with-a-named-closure, or refused.** Never silently approximate.
4. **Where a conversion needs a mathematical derivation rather than a compiler pass, refuse.** EoN's analytic hierarchy is a decade of mathematics saying when this is true.
5. **Allow per-transition paradigm.** Catalyst.jl's `PhysicalScale` is the sharpest version of this idea anywhere, and hybrid epidemiological models need it.
6. **Report the capability matrix.** A model should be able to say which backends it can run on and why not, before it runs.

## Why

### The existence proofs

[EMULSION](../approaches/emulsion.md) switches a whole model between stochastic compartmental and individual-based with **one word** — `aggregation_type: 'compartment'` versus `'IBM'` — with the state machine, transitions, parameters, initial conditions, and outputs identical between the two files. Published in 2019, Apache-2.0. Its review: "This is the paradigm-independence goal, shipped."

[epipack](../approaches/epipack.md) runs one list of process tuples deterministically, stochastically well-mixed, and stochastically on a network. [camdl](../approaches/camdl.md) runs one model text on ODE, chain-binomial, and Gillespie. [odin](../approaches/odin.md) shares `initial()` between `deriv()` and `update()`.

### The cost of not having it

[MEmilio](../approaches/memilio.md): twenty-plus model families — `ode_*`, `ide_*`, `lct_*`, `glct_*`, `sde_*`, metapop, `abm`, `d_abm`, `graph_abm`, `smm`, `hybrid` — each a separate C++ implementation, most with their own paper. Its review: "MEmilio is the cost estimate for *not* having a lingua franca… the paradigm coverage was bought by writing each one separately."

[EpiModel](../approaches/epimodel.md) gets three paradigms with one input vocabulary and three separately written models — "the honest baseline this project is trying to beat: same vocabulary *and* same model text."

### The axes

From the [MEmilio review](../approaches/memilio.md), and it is the most useful reframing in the whole landscape:

> Not "compartmental / metapopulation / SDE / ABM" as four options, but a lattice with axes: aggregation (compartment ↔ agent), stochasticity (deterministic ↔ diffusion ↔ jump), dwell time (exponential ↔ Erlang ↔ arbitrary), and space (single ↔ patch ↔ explicit mobility). A model specification selects a point; a conversion moves along an axis. This is a more useful framing than a list of paradigms, and it makes the "which conversions are meaningful" question answerable per axis.

### The warning

[EoN](../approaches/eon.md) implements, for SIS and SIR, a full ladder of analytic approximations to network simulation: individual-based, pair-based, homogeneous pairwise, heterogeneous pairwise, compact pairwise, super-compact pairwise, effective degree, compact effective degree, homogeneous mean-field, heterogeneous mean-field, and the edge-based compartmental model. Each `*_from_graph` variant extracts the required moments from an actual network, so you can run the exact simulation and each approximation on the same graph and measure what the closure costs.

Its review states the consequence directly:

> **The lingua franca should not present network → compartmental conversion as a single operation.** It is a choice of closure, and the choice should be named in the output the way `SIR_heterogeneous_pairwise` names it.

and the harder warning:

> EoN's approximations cover SIS and SIR, not arbitrary `H`/`J` processes, because each closure has to be derived. That is a real warning about the cost of general paradigm conversion: it is a mathematical derivation per model class, not a compiler pass. Where the lingua franca cannot derive the conversion, it should refuse rather than approximate silently.

## The proposal

### One model, several runs

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

ss.Sim(sir, mixing=ss.Homogeneous(n_contacts=10)).run(method='ode')
ss.Sim(sir, mixing=ss.Homogeneous(n_contacts=10)).run(method='ctmc')
ss.Sim(sir, mixing=ss.RandomNet(n_contacts=10),  n_agents=50_000).run()
```

The route changes; the disease does not. That is the [population-and-mixing](population-and-mixing.md) factoring doing the work, and it is why the paradigm switch is a keyword rather than a rewrite.

### The capability matrix

Before running anything:

```python
sir.capabilities()
```

```text
                    ode    ctmc   tau    sde    abm
mixing=Homogeneous   ✓      ✓      ✓      ✓      ✓
mixing=MixingPool    ✓      ✓      ✓      ✓      ✓
mixing=RandomNet     ~      ~      ~      ~      ✓     ~ mean-field limit, exact as n→∞
mixing=SexualNet     ✗      ✗      ✗      ✗      ✓     ✗ concurrent partnerships: no closed form
dur=lognorm          ~      ~      ~      ~      ✓     ~ nearest Erlang; discrepancy reported
eligible=ss.age>65   ✓      ✓      ✓      ✓      ✓       (age is a dimension)
eligible=lambda ...  ✗      ✗      ✗      ✗      ✓     ✗ opaque predicate
```

This is [camdl](../approaches/camdl.md)'s discipline — "every backend × method combination either works and is tested, or fails through a capability check that names the limitation — no silent gaps" — generalized to a much larger matrix, and it is the honest form of the project's central claim. The claim is not "any model runs any way"; it is "you will be told, by name, what does not."

### Three classes of conversion

**Exact.** No approximation; the same mathematics, differently executed.

- Homogeneous or contact-matrix mixing → ODE, CTMC, chain-binomial, ABM. Exact in the limit and in distribution.
- Exponential dwell time → rate, or a sampled clock.
- Erlang dwell time → sub-compartments, or a sampled time.
- A stratification dimension → compartment axis, or an agent property.

**Approximate, with the approximation named.**

- Non-Erlang dwell time → ODE: nearest Erlang or hyper-Erlang fit, with the discrepancy in the distribution reported, or `method='ide'` for the exact memory kernel.
- Network → compartmental: **a named closure**, following EoN. Not `to_ode()` but `to_ode(closure='heterogeneous_pairwise')`, with `closure='mean_field'` available and clearly labelled as the crudest rung.
- Large populations → tau-leaping or diffusion, with the leap condition or the count threshold reported ([stochastic-compartmental.md](stochastic-compartmental.md)).
- Continuous agent heterogeneity → a discretizing dimension, with the bins shown.

**Refused, by name.**

- Persisting concurrent partnerships → any compartmental form. There is no closure; concurrency is the thing being modelled.
- An opaque `ss.process` or lambda predicate → anything but ABM.
- Individual histories, contact tracing, transmission trees → anything but ABM.
- A closure that has not been derived for the model class in question. **This is the important one**: EoN's ladder exists for SIS and SIR, not for arbitrary processes, and pretending otherwise is the failure mode this whole document exists to prevent.

```text
CapabilityError: cannot run 'hiv' as method='ode'.
  route SexualNet has concurrent, persisting partnerships.
  Concurrency has no mean-field closure — a compartmental rendering would discard
  the mechanism the model exists to represent.
  Nearest available: mixing=ss.MixingPool(matrix=...) for an age-mixing approximation,
  which is a different model. See best-practices/paradigm-conversion.md.
```

### Per-transition paradigm

[Catalyst.jl](../approaches/notes.md#modelingtoolkitjl--catalystjl) v16 adds per-reaction `PhysicalScale` metadata "so that a single model can have some reactions treated as ODEs and others as jumps, in one simulation". From that note:

> **`PhysicalScale` per reaction is the sharpest version of the paradigm-independence idea in this entire review.** Not "this model is stochastic" but "this *transition* is a jump process and that one is an ODE", declared per transition, in one model. That is exactly what a hybrid epidemiological model needs — rare importation events as jumps, bulk within-population transmission as an ODE.

[EMULSION](../approaches/emulsion.md) reaches the same place with `aggregation_type: 'hybrid'` per level; [MEmilio](../approaches/memilio.md) has a hybrid agent–metapopulation family with a paper on the coupling.

```python
polio = ss.Disease(
    importation = ss.transmission('S -> I', beta=1e-6, scale='jump'),   # rare: exact
    infection   = ss.transmission('S -> I', beta=0.05),                 # bulk: follows the run method
    recovery    = ss.progression('I -> R', dur=ss.days(20)),
)
```

This is the elimination and eradication case, where the deterministic model says the epidemic is over and the real question is the probability of reintroduction. It is worth the one keyword.

EMULSION's per-*level* version is the other half — individuals aggregated inside herds that are explicit, or the reverse — and it generalizes to `sir.stratify(patch=..., scale='agent')`: agents within patches, patches explicit. Whether both forms are needed is an open question below.

### Test the conversion

The claim is empirical, so it should be a test suite, and the landscape hands us the instruments:

- **EMULSION's shipped feature models.** Its review proposes exactly this: "take the shipped feature models, push each one across `compartment` / `IBM` / `hybrid`, and record where the switch stops working and why. That boundary is the actual scope of the lingua franca's paradigm-independence claim, and someone has already built the apparatus for measuring it."
- **EoN's `*_from_graph` approximations**, which parameterize an analytic approximation from a real network and let you measure the closure error directly.
- **epipack's three modes** on the same tuples, as a cheap regression baseline.
- **camdl's Richardson dt-convergence audit**, to distinguish "it ran" from "it converged".

Concretely: for every model in [vignettes](../vignettes), run every backend the capability matrix permits and assert agreement to within a stated tolerance. A conversion that is not tested is a conversion that is not claimed.

### Interconversion with other frameworks

Distinct from paradigm conversion and worth separating. [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) is the closest prior art — a `TemplateModel` with a closed transition taxonomy, ontology-identified Concepts, and importers/exporters for SBML, Petri-net AMR, RegNet AMR, StockFlow AMR, ACSets, bilayer, Vensim/Stella, and PySB. [MEmilio](../approaches/memilio.md) has an SBML importer, "a working instance of the exact artefact this project needs".

Two lessons. From MIRA: "Nobody except MIRA can tell whether two models' compartments mean the same thing" — which is why optional ontology annotation earns its place ([model-structure.md](model-structure.md)) even though requiring it does not. From [SBML](../approaches/notes.md#sbml-systems-biology-markup-language): "an interchange format that nobody authors in becomes a lossy export target, and partial conformance is the normal state."

The consequence for us is the [principles.md](principles.md) rule restated: this has to be a genuinely pleasant authoring language first. An interchange format that is not authored in loses.

## Trade-offs

- **The capability matrix is a large testing obligation.** Every cell is a claim. Mitigation: most cells are structural (they follow from route × method), and the genuinely model-specific ones are few.
- **Refusing is unfriendly.** A user who wants an ODE of their sexual-network model will be annoyed. They should be: the alternative is a number that looks fine and is wrong.
- **Named closures are a research surface.** Implementing heterogeneous pairwise for arbitrary declared processes is not a compiler pass. The honest v1 scope is: exact conversions everywhere they exist, mean-field with a loud label, and refusal otherwise — with the ladder as a later addition per model class.
- **Per-transition `scale=` adds vocabulary** for a case (elimination) that most users never hit. One keyword, defaulting to "follow the run method", is the smallest form that works.

## Rejected

- **Presenting network → compartmental as one operation.** EoN's decade of mathematics says it is a choice of closure.
- **Silent approximation of any kind.** The category error this document exists to prevent.
- **A separate model per paradigm** (MEmilio, EpiModel). The cost is quantified.
- **Paradigm as a model property** (EMULSION's per-level `aggregation_type` as the *only* mechanism). It is a run-time choice, with per-transition and per-level override.
- **Automatic conversion to whatever is fastest.** The paradigm changes what questions the model can answer; it is the modeler's choice, informed by the capability matrix.
- **An interchange format as the primary artifact.** SBML's history. The authoring language is the artifact; the serialization is derived.

## Open questions

- **Are per-transition and per-level scale the same mechanism?** Catalyst does the first, EMULSION the second, MEmilio's hybrid family the second at the coupling boundary. Probably one mechanism with two spellings, but the coupling semantics at the boundary — MEmilio has a paper on exactly this — are not trivial.
- **What is the exact statement of the ABM ↔ ODE agreement?** "Agrees in distribution as n → ∞ under Homogeneous mixing" is provable for the simple case, and the conditions for stratified, multi-route, non-exponential-dwell-time models need writing down rather than assuming.
- **Should the closure ladder ship at all in v1?** It is the most intellectually interesting part and the least commonly needed. Refusal plus a documented pointer to EoN may be the right v1 answer.
- **What is the granularity of the capability matrix?** Per (route × method) is coarse and cheap; per (transition × method) is precise and large. The escape hatches force at least some per-element resolution.
