# Brief notes

Short entries for frameworks that are worth placing and learning from without a full review. Each records what it is, the one or two ideas worth carrying into the design, and why it is not reviewed at length.

Full reviews are listed in [README.md](README.md).

---

## Covasim / HPVsim / STIsim / FPsim

**Python · agent-based · Starsim's predecessors and descendants · [covasim.org](https://covasim.org), [hpvsim.org](https://hpvsim.org), [stisim.org](https://stisim.org), [fpsim.org](https://fpsim.org)**

Covered as a section within [starsim.md](starsim.md), because the design lineage is the point: three disease-specific frameworks were generalised into one modular framework, and the cost of *not* having done that earlier (three near-identical codebases) is what motivated `ss.Module`.

Worth extracting individually:

- **Covasim's intervention form.** `change_beta(days=[30, 60], changes=[0.5, 1.0])` — a list of days and a list of changes — is a smaller, more legible time-varying-parameter grammar than the class lattice that replaced it. flepiMoP's modifiers and EpiModel's scenario tables are the same shape. Simplicity here was arguably lost in the generalisation.
- **HPVsim's genotype-as-a-state-dimension.** Multi-strain modelling by adding an axis to the state array rather than by duplicating the disease module, plus the screening → triage → treatment product cascade that Starsim's `BaseTest`/`BaseTreatment` classes generalise.
- **FPsim** demonstrating that the same machinery models a non-infectious process (contraceptive use, pregnancy, birth outcomes), which is why `ss.Disease` sits under a general `ss.Module` rather than at the root.

---

## Kendrick

**Pharo / Smalltalk · compartmental, with separation of concerns as the organising principle · [doi.org/10.1016/j.procs.2015.05.126](https://doi.org/10.1016/j.procs.2015.05.126)**

A domain-specific modelling language built around the claim that an epidemiological model is a *composition of independent concerns* — the disease process, the demographic process, the spatial process, the intervention process — and that these should be written separately and combined mechanically rather than being interleaved in one system of equations.

Conceptually important and practically almost unused: it is implemented in Pharo Smalltalk, has essentially no adoption outside its originating group, and is not in the landscape database.

**What to take:** the framing. Kendrick's separation-of-concerns argument is the same one epymorph makes with IPM/MM/GEO, MetaCast makes with dimensions, and this project makes with paradigm-independence. Kendrick states it earliest and most explicitly, and the paper is worth reading for the argument rather than the tool. Its concern-composition operator — combining a disease process with an age process to get an age-structured disease process — is what `stratify_with()` and `stratify(by = age)` implement in more limited forms.

**Why not a full review:** the implementation is not reachable for most of this project's purposes, and its ideas are better evidenced by the frameworks that later implemented them.

---

## NDlib

**Python · diffusion processes on complex networks · [ndlib.readthedocs.io](https://ndlib.readthedocs.io)**

*Excluded from the review at CK's direction.* Recorded here for completeness: a library of epidemic and opinion-diffusion models on NetworkX graphs, with a compartment-composition API (`NodeStochastic`, `NodeThreshold`, `EdgeStochastic` composed into a rule) and a web front end. Strong on general process composition, thin on epidemiological semantics; EoN covers the network-simulation ground more rigorously and with the analytic approximations attached.

---

## PyRoss

**Python · age-structured stochastic compartmental, with inference and optimal control · [10.5281/zenodo.4290294](https://doi.org/10.5281/zenodo.4290294)**

Age-structured compartmental models with several stochastic formulations (Gillespie, tau-leaping, Gaussian/van Kampen approximations), Bayesian inference over them, and — unusually — **optimal control**: computing the intervention trajectory that optimises an objective subject to constraints.

**What to take:** the optimal-control layer. Almost every framework in this review can answer "what happens if we do X?"; PyRoss can answer "what X is best?". Atomica's budget optimisation is the other instance. If interventions are declared objects with declared costs and effects (as EMOD's campaigns and Atomica's programmes are), then optimising over them is a natural capability of the language rather than a bespoke analysis, and the design should not foreclose it.

Also worth noting: PyRoss implements multiple *stochastic approximations* of the same model (exact Gillespie, tau-leaping, diffusion) side by side, which is MEmilio's paradigm-lattice point on the stochasticity axis specifically.

**Why not a full review:** the model-specification surface is a set of Python classes with fixed structures parameterised by age-contact matrices, which adds little beyond what summer2, epipack, and MEmilio already contribute.

---

## Mesa

**Python · general-purpose agent-based modelling · [github.com/projectmesa/mesa](https://github.com/projectmesa/mesa) · v4 in development**

The default general-purpose ABM framework in Python — the Python answer to NetLogo, Repast, and MASON. Not epidemiological, and excluded from the landscape database on exactly that basis, but it is what a substantial number of one-off epidemic ABMs are actually built on.

**What to take:**

- **Agent activation semantics as an explicit, named choice.** Mesa's scheduler question — simultaneous activation, random activation, staged activation, activation by type — is the general form of the ordering problem that Starsim's `Loop`, EpiModel's `module.order`, Vivarium's lifecycle priorities, and `individual`'s queued updates each answer differently. Mesa is the framework that treats it as a *first-class modelling decision the user selects*, and names the options. That is the right treatment: activation order changes results, so it should be declared.
- **`AgentSet`** (Mesa 3+) as a queryable, composable collection of agents with `select`, `shuffle`, `do`, and `agg` — the same select-then-act primitive identified in the `individual` and LASER entries, in a general-purpose framework.
- **Space as a separate, swappable component** (`discrete_space`, continuous space, networks) rather than baked into the model.

**Why not a full review:** it contributes no epidemiological vocabulary, and its structural ideas are better evidenced in this domain by the frameworks above.

---

## ModelingToolkit.jl / Catalyst.jl

**Julia · symbolic reaction networks with automatic multi-paradigm code generation · [docs.sciml.ai/Catalyst](https://docs.sciml.ai/Catalyst/stable/) · [10.1371/journal.pcbi.1011530](https://doi.org/10.1371/journal.pcbi.1011530)**

**The strongest existing demonstration anywhere of "one specification, many paradigms", and it is not in epidemiology.**

Catalyst is a symbolic modelling package for chemical reaction networks built on ModelingToolkit/Symbolics. A model is written in a reaction DSL:

```julia
model = @reaction_network begin
    kB, S + E --> SE
    kD, SE --> S + E
    kP, SE --> P + E
end
```

and the *same* `ReactionSystem` then generates an ODE system, an SDE system, a jump (Gillespie) system, a steady-state system, or a hybrid — via `ode_model`, `sde_model`, `jump_model`, `ss_ode_model`. Version 16 adds **per-reaction `PhysicalScale` metadata**, so a single model can have some reactions treated as ODEs and others as jumps, in one simulation.

**What to take:**

1. **`PhysicalScale` per reaction is the sharpest version of the paradigm-independence idea in this entire review.** Not "this model is stochastic" but "this *transition* is a jump process and that one is an ODE", declared per transition, in one model. That is exactly what a hybrid epidemiological model needs — rare importation events as jumps, bulk within-population transmission as an ODE — and it is what EMULSION's `hybrid` mode and MEmilio's hybrid models achieve with much more machinery.
2. **Symbolic core with generated code.** Catalyst exploits the symbolic representation for sparsity, Jacobians, dependency-graph analysis, and parallelism — the same argument as epipack's, at production quality.
3. **Unit validation** (`@unit_checks`, `validate_units`) with non-SI units via DynamicQuantities.jl. A fourth independent arrival at dimensional checking, in a general-purpose modelling ecosystem.
4. **Compositional modelling** — `@network_component`, `compose`, `extend` — building models hierarchically from named sub-networks. This is the composition mechanism the epidemiological frameworks mostly lack.
5. **Coupled models**: reactions plus differential equations plus discrete events plus Brownian noise (`@brownians`) plus Poisson jumps (`@poissonians`) in one system. Environmental forcing and stochastic parameters become part of the model rather than external drivers.
6. **Network analysis from the structure** — linkage classes, deficiency, reversibility — a family of structural results computable because the model is a graph.

**Why not a full review:** it is not an epidemiological tool and has no epidemiological vocabulary (no interventions, no contact structure, no observation model). But **the design team should read its documentation before finalising the paradigm-conversion semantics**, because the SciML ecosystem has been solving this specific problem, at scale, for longer than anyone in this review.

---

## SBML (Systems Biology Markup Language)

**XML standard · model interchange · [sbml.org](https://sbml.org) · [10.1093/bioinformatics/btg015](https://doi.org/10.1093/bioinformatics/btg015)**

Twenty-five years of experience with a machine-readable model interchange format for reaction-network models, supported by hundreds of tools, with a curated model repository (BioModels) and a formal specification with versioned Levels.

Directly relevant as **precedent**, including the parts that went badly.

**What to take:**

- **Levels and Versions as a compatibility contract.** SBML L2V4 versus L3V2 is an explicit statement about what a consumer must support. camdl's IR version envelope is the same idea; a lingua franca will need it.
- **Annotation as a separate, structured layer.** SBML separates the mathematics from `<annotation>` blocks carrying MIRIAM-standard identifiers, so a species can be linked to a database entry without the semantics depending on it. EpiHiper's `ann:` prefix is a lightweight version of the same discipline.
- **Units are in the standard** (`unitDefinition`), and are widely ignored in practice — a cautionary data point about optional typing.
- **What went wrong:** XML verbosity meant nobody writes SBML by hand, so authoring moved to tool-specific UIs and the interchange format became an *export* format; "supports SBML" varies enormously in practice because the standard is large and conformance is partial; and semantic gaps (events, delays, algebraic rules) are where round-tripping breaks.

**The lesson for this project:** an interchange format that nobody authors in becomes a lossy export target, and partial conformance is the normal state. If the lingua franca is to be *both* an authoring language and an interchange format, the authoring surface has to be genuinely pleasant, and the conformance story has to be all-or-nothing per feature — which is camdl's capability-check discipline again.

---

## ASKEM model representation (AMR) / MIRA

**JSON schemas and a Python meta-model · [github.com/gyorilab/mira](https://github.com/gyorilab/mira) · DARPA ASKEM**

MIRA is a **meta-model registry and model-transformation toolkit** for epidemiological and biological models: the most recent, and most directly comparable, attempt at exactly what this project is attempting.

Its core is a `TemplateModel` built from typed **Templates** — a closed vocabulary of transition archetypes:

`NaturalConversion` · `ControlledConversion` · `GroupedControlledConversion` · `NaturalProduction` · `ControlledProduction` · `GroupedControlledProduction` · `NaturalDegradation` · `ControlledDegradation` · `GroupedControlledDegradation` · `MultiConversion` · `ReversibleFlux` · `NaturalReplication` · `ControlledReplication` · `StaticConcept`

Each references **Concepts**: a name, a display name, `identifiers` (a mapping of namespace to ontology identifier), `context` (a mapping of context keys to values), and **`units`** (a SymPy expression over a unit vocabulary).

**What to take — this is the most idea-dense entry in these notes:**

1. **A closed, named transition vocabulary with semantic types.** `ControlledConversion` (S → I controlled by I) versus `NaturalConversion` (I → R) is the spontaneous/contact-mediated distinction — the same one EoN, epipack, and EpiHiper each arrived at — formalised into a taxonomy. The `Grouped*` variants handle multiple controllers. This is the most thought-through version of "what kinds of transitions are there" in this review, and it exists precisely because MIRA needs to *translate between formats*, which forces the question.
2. **Concepts carry ontology identifiers.** A compartment is not the string `"I"`; it is a Concept with `identifiers={"ido": "0000511"}` (IDO's "infected population") and `context={"age": "youth", "location": "Boston"}`. This makes two models' compartments **comparable** — which is what `mira.metamodel.comparison` and `search.py` exploit, and what makes automated model comparison and merging possible at all. **No other framework in this review can tell whether two models' compartments mean the same thing.**
3. **Context as a generic stratification mechanism.** A Concept's `context` dict is the stratum coordinate, and `mira.metamodel.ops.stratify()` operates on it generically — the same dimension algebra as MetaCast and camdl, but attached to a semantically identified base concept.
4. **Units as symbolic expressions**, per Concept. A fifth independent arrival at dimensional typing.
5. **Model operations as a library**: `stratify`, `rewrite_rate_law`, `simplify_rate_laws`, `aggregate_parameters`, `counts_to_dimensionless`, `deactivate_templates`, `add_observable_pattern`, `get_observable_for_concepts`. Model *transformation* as a first-class API — the thing that becomes possible once a model is data, and the thing this project will need.
6. **Many source and target formats**: SBML, Petri-net AMR, RegNet AMR, StockFlow AMR, ACSets, bilayer, system dynamics (Vensim/Stella), PySB, and `sympy_ode` (extracting ODE systems from papers). The AMR JSON schemas are an explicit interchange target with a `semantics` / `metadata` split.
7. **`counts_to_dimensionless`** as a named operation — converting a model between count and proportion formulations — is exactly the kind of paradigm-adjacent transformation the lingua franca will need, and it is one nobody else names.

**Why not a full review:** MIRA is a translation and transformation layer rather than a modelling framework — you do not author models in it, and it has no simulation engine, no intervention vocabulary, and no observation model. But **it is the closest prior art to this project's interconversion goal**, its Template taxonomy is directly reusable, and the ontology-identified Concept is an idea this project does not currently have and probably needs.

---

## Frameworks considered and not written up

Recorded so the decisions are reviewable.

- **FRED** (University of Pittsburgh) — synthetic-population ABM with household/school/workplace networks. **Excluded on licence**: distributed under a University of Pittsburgh EULA, not an open-source licence, per the landscape database's licence audit. (CK's instruction was to include it if open source; it is not.)
- **GLEAM / GLEAMviz** — the landmark global metapopulation model integrating worldwide census and air-travel data. Closed source; referenced from [epymorph.md](epymorph.md) as the reference point for the metapopulation paradigm rather than reviewed.
- **MetaWards** — ward-level metapopulation for England and Wales. Real, but effectively one application with a configuration surface; epymorph and flepiMoP cover the metapopulation specification ground better.
- **GEMS** (Julia) — nationwide microsimulation over synthetic German populations. Would have been the review's only Julia agent-based entry, but its specification style adds nothing beyond `individual` and Starsim; Catalyst.jl covers Julia's contribution to this project far better.
- **Episimmer** — institutional-reopening simulator. Excluded on licence (Commons Clause is not open source).
- **statnet / ergm** — the network substrate under EpiModel; discussed inside [epimodel.md](epimodel.md).
- **dust / monty** — the engine and inference layers under odin; discussed inside [odin.md](odin.md).
- **Stan, PyMC, NumPyro** — inference substrates rather than model-structure languages. They belong to the calibration discussion in `best-practices`.
