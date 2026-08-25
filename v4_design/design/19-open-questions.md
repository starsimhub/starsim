# Open questions

The register of what this specification settled, what it deliberately deferred, and what remains genuinely unresolved. Nothing here is hidden in a footnote elsewhere.

Sources are the open-questions sections of every document in [best-practices](../best-practices), plus questions raised while writing this specification.

## A. Settled here

Recorded so that they are not reopened without reason.

| Question | Resolution | Where |
|---|---|---|
| Where does the dict shorthand stop being useful? | Two coercion rules, one context-dependent; anything else raises `E202` | [01](01-objects.md#form-1-the-dict) |
| What happens when a `freq` and a `rate` are added? | `E101`. The full type lattice is tabulated, not discovered per operator | [02](02-types.md#arithmetic) |
| Sub-step scheduling versus per-module timesteps: one mechanism or two? | One, with two spellings; both resolve to absolute times in the loop | [02](02-types.md), [10](10-execution.md#sub-steps) |
| `n` versus `n_agents` | `n` is semantic population size; `n_agents` is resolution, ABM only, and scaling is reported | [01](01-objects.md#n-versus-n_agents) |
| The `p=` ambiguity between timing and branch split | Resolved by destination count, normatively | [03](03-states-and-transitions.md#branching) |
| Can transmission fire from something that is not a disease state? | Yes: `ss.Continuous`, with a dose-response force of infection | [03](03-states-and-transitions.md#sscontinuous), [06](06-routes.md#environmental) |
| How does `ss.Continuous` stratify? | It declares which dimensions apply, defaulting to the spatial one | [03](03-states-and-transitions.md#sscontinuous) |
| Should outcomes feed back into the process? | No. `E701`. If it feeds back it is a state | [09](09-results.md#the-observation-model) |
| Under-reporting, reporting delay, reporting schedule | Three distinct keywords, kept distinct | [09](09-results.md) |
| Stock constraints: coverage as fraction or number? | Both; `coverage=ss.count(50_000)` is the supply-limited form | [08](08-interventions.md#delivery) |
| The CRN guarantee when populations diverge | Stated: identically-keyed agents retain draws until the first differing draw index; `crn_report()` measures it | [11](11-random.md#the-guarantee-under-divergent-populations) |
| Deterministic flows inside a stochastic model | Per-transition `scale='jump'` / `'continuous'`, defaulting to follow the run method | [12](12-backends.md#per-transition-scale) |
| Capability-matrix granularity | Per (element × method), where an element is a route, transition, dimension, effect, or escape hatch | [12](12-backends.md#the-capability-matrix) |
| Spelling for fitting an unidentifiable product | `calibrate(pars=[sir.beta * mixing.n_contacts])`, with `W1302` when the check fails | [13](13-inference.md#fitting-a-product-of-parameters) |
| Is strain a dimension or a second disease? | A dimension, with a declared cross-immunity matrix; the force-of-infection restructuring is derived, not special-cased | [05](05-dimensions.md#strain) |
| Execution order of effects versus transitions | Derived, not declared: effects resolve in phase 3, transitions read in phase 4 | [10](10-execution.md#phases) |
| Multi-stream likelihood default | Sum of log-likelihoods assuming independence, printed as an assumption | [13](13-inference.md#multiple-streams) |

## B. Open, and blocking something

These have to be answered before the corresponding capability can be claimed. Each names what is blocked.

### B1. Does the per-contact factoring survive environmental and vector-borne transmission?

**Blocks:** cholera, typhoid, malaria, dengue, schistosomiasis — a large fraction of the global disease burden.

`ss.Continuous` with a dose-response λ ([06](06-routes.md#environmental)) is specified as the answer for reservoir transmission, and it is the most likely place the unification breaks, because there is no contact for β to be *per*. **Vector-borne transmission is worse**: it is two-host, and λᵢ = Σⱼ cᵢⱼ β Iⱼ/Nⱼ does not describe it without a second population with its own demography and its own force of infection.

The three candidate answers are a fourth route kind, a second `People` with a cross-population route, and a second `Disease` with a coupling effect. This specification does not choose, and **this is the largest unresolved item in the design.** It should be tested early, against a real vector-borne model, because the answer may change [06-routes.md](06-routes.md).

### B2. What is the exact mean-field limit of a random network?

**Blocks:** the `~` cells in the capability matrix for `RandomNet`, and the tolerance in the conformance suite.

For a Poisson-degree network regenerated every step, the limit is `Homogeneous` and the agreement is exact as `n → ∞`. For a static network with degree heterogeneity it is not, and the published closure ladder names the corrections. The compiler must know which case it is in, and currently the specification asserts the distinction without giving the discriminating rule.

### B3. Should the closure ladder ship at all in v1?

**Blocks:** nothing, but it determines whether `E805 UnderivedClosure` is the common case or the rare one.

It is the most intellectually interesting part of the design and the least commonly needed. Implementing pair approximations for *arbitrary declared processes* is a research program, not a compiler pass. The honest v1 scope is probably: exact conversions everywhere they exist, mean-field with a loud label, refusal otherwise, and a documented pointer to the published ladder.

### B4. How much of the parameter ladder does the compartmental backend accept?

**Blocks:** the per-rung rows of the capability matrix.

An agent-level `ss.lognorm` *dwell time* has an Erlang expansion. An agent-level `ss.lognorm` on *`beta`* has no compartmental analogue without either adding a discretizing dimension or accepting a mean-field approximation that is wrong in a known direction. The rule needs stating per rung, and it is currently stated only for dwell times.

### B5. Where does per-agent continuous heterogeneity live?

**Blocks:** B4, and every model with individual-level susceptibility variation.

`rel_sus ~ lognorm(1, 0.3)` is not a state and not a dimension. In an agent-based backend it is ordinary; in a compartmental backend it requires a discretization whose bin count changes the answer. This needs a named answer — probably an automatic discretizing dimension with the bins printed — rather than silence.

### B6. Hybrid coupling semantics

**Blocks:** `scale='jump'` in models where a jump transition feeds a continuous compartment.

Count conservation and unbiased rounding at the boundary are stated as requirements in [12](12-backends.md#per-transition-scale) without an algorithm. There is published work on exactly this coupling in the agent–metapopulation case, and it should be followed rather than invented.

## C. Open, and deferrable

Real questions with no v1 dependency.

| Question | Note |
|---|---|
| **Cross-immunity as an effect between coordinates of one dimension** | [05](05-dimensions.md) specifies an `immunity=` matrix, which works for strains but is a shape `ss.Effect` does not have generally. If a second case appears, the mechanism should be generalized rather than duplicated. |
| **Should waning be an effect-level modifier rather than a product field?** | `ss.Effect(vaccinated, sir.rel_sus, multiply=ss.exp_decay(0.1, halflife=ss.days(180)))` would collapse waning, product efficacy, and time-varying effects into one thing. Appealing, and possibly doing too much: it makes `ss.Effect` time-aware, and risk 2 below argues against extending it. Waning currently lives on the product. |
| **Should `bounds=` and `prior=` be one field?** | One framework has a single combined form. They serve validation and inference respectively; collapsing them may be right and is not urgent. |
| **Partial stratification of transitions**, not just states | `only=[R]` in one framework. It may be a sign the model wanted two transitions. |
| **Does eligibility need edge predicates?** | Typed set algebra over edges makes "close school contacts" expressible directly rather than as a layer. Layers are probably enough. |
| **Cascades deeper than three steps** | `on_positive=` chains arbitrarily; beyond depth three a cascade probably wants to be a module. |
| **Agent-level outputs** — line lists, age-at-infection distributions, partnership histories | Not time series. An imperative analyzer is honest and unstructured, and may be correct. |
| **Overdispersion** — process or observation? | Superspreading is a process property; reporting overdispersion is an observation property. They are currently spelled the same way and should not be. |
| **Count versus proportion formulations** | One framework names `counts_to_dimensionless` as an operation and nobody else does. Probably an output option, not a model property. |
| **Adaptive mobility** — people stop travelling during an epidemic | An effect on the mobility matrix. The mechanism supports it; nothing in the landscape does it declaratively, so there is no prior art to follow. |
| **Within-host state** — viral load, CD4, immune waning | Continuous per-agent trajectories that are not transitions and not dimensions. Currently the escape hatch, which may be correct. |
| **Contact tracing** | Requires contact history, which not every route retains. Network-only in every existing treatment. |
| **What does a patch mean in an agent-based backend?** | An agent property, per the unification. Agents in patches with aggregate coupling is a third thing, probably `ss.MixingPool` over a spatial dimension; it should be confirmed rather than assumed. |
| **Should space get special treatment after all?** | One framework puts the node index in its core for eradication-scale performance, and is not obviously wrong. "Space is a dimension carrying movement" may be too uniform for a subnational planning audience whose every question is spatial. |
| **Travellers in transit** | Neither `commute` nor `migrate`. Declared a refusal in v1. |
| **`prob` period in the IR** | Keep the period, or normalize to per-step at compile time. Faithfulness versus IR size. |
| **Structural calibration** | "Does this model need an exposed compartment?" is mechanizable once models are data. Out of scope for v1; do not foreclose. |
| **Joint likelihood with declared correlation between streams** | The independence default is usually wrong in detail and always visible. |

## D. Risks to the design as a whole

Not questions with answers; things that would invalidate parts of this specification if they turn out badly.

1. **The closed transition vocabulary may be too small for veterinary, vector-borne, and within-host models.** The escape hatch keeps them expressible and unstratifiable. The test is what fraction of the [vignettes](../vignettes) corpus needs `ss.process`; above roughly a fifth, the vocabulary is wrong.
2. **`ss.Effect` is doing a great deal of work.** Interventions, connectors, waning, triggers, seasonality, and product efficacy all compile to it. If its semantics are wrong, much is wrong. That is an argument for it staying small — a condition, a target, a combiner — and against every proposed extension to it.
3. **`summary()` is load-bearing for correctness.** Guessing defaults is safe only if people read the printout. Mitigations are that it is short, that it is printed by default, and that it shows only what was inferred. If usability testing shows it is not read, the default-guessing policy has to be revisited, not the summary.
4. **The capability matrix is a large testing obligation**, and an untested cell is a false claim. Most cells are structural and follow from (route × method), but the escape hatches force per-element resolution, and the matrix grows with the vocabulary.
5. **Python's dynamism means some checks land at build time rather than compile time.** That is later than a compiled DSL manages and earlier than any Python framework in the landscape manages today. It is accepted, and it is why [10-execution.md](10-execution.md) specifies that every error is raised at the earliest initialization step that can detect it.
