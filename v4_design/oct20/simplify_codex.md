# Simplifying Starsim v4

Surveyed October 20, 2026, at commit `79592ae6`. This is a design review, not an implementation proposal that preserves every existing behavior. No implementation changes were made. Source locations below name the relevant functions as well as files, since line numbers will move.

## Recommendation

**Make ordinary Python components the foundation, and make the declarative model language an optional way of building components. Limit automatic compartmental conversion to a substantially smaller set of models.**

The main source of complexity is the promise that a compact declaration can acquire individual history, pregnancy, products, queues, several diseases, geography, stochastic execution, compilation, inference, and serialization while retaining broadly the same interface. Each feature looks reasonable by itself. Their combinations require additional representations of the same model and substantial reconciliation machinery.

There is useful complexity here. Competing hazards, common random numbers, correct time conversion, observation alignment, and composable modifiers solve real problems. I would protect those. I would be much less protective of automatic solver probing, inferred model structure, individual-history emulation in compartments, multiple intervention systems, automatic closure repair, and general reconstruction of Python models from a saved description.

My first five decisions would be:

1. Stop expanding compartmental support for individual histories and parent–child relationships. Use ABM for those declarations; allow experts to write their own aggregate approximation as a Python component.
2. Establish a small scheduler/component interface that works without `People`, a declared graph, or Starsim arrays. Prove it with a NumPy ODE and a NumPy ABM before adding capabilities.
3. Separate the generic model language and execution machinery from disease-specific features and services. Move importers and specialized reference models out of the base distribution; put advanced inference behind an optional boundary.
4. Retire the parallel v3-style intervention/product implementation through a compatibility package, rather than teaching two intervention systems indefinitely.
5. Reduce inference and convenience in construction: explicit ownership, explicit parameter updates, one canonical declaration path, and predictable solver semantics.

**About the line target:** 40k package lines is a plausible stretch target if functionality moves out *and* some promises are removed. Merely rearranging modules will not get there. Roughly 43k is an illustrative central planning case below. I do not see an evidence-based route to 30k that preserves most of today's scope and scientific guarantees. Nor would moving 20k lines into equally entangled companion packages count as success.

## 1. What was measured and reviewed

I inventoried all 83 Python files under `starsim/`, inspected the architecture and source of each major subsystem, traced solver selection and execution boundaries, examined representative API, contract, inference, intervention, pregnancy, and importer tests, and ran small composition probes. This is a comprehensive architectural survey, not a line-by-line correctness audit. I did not run the full test suite or benchmark suite, and did not independently validate the algorithms statistically.

The measurements include `__init__.py`, examples shipped in `starsim.library`, and import/export helpers shipped in `starsim.interop`. They exclude tests, docs, design experiments, notebooks, and other repositories. The user's v3 baseline was not recounted with this method, so the growth comparison is approximate.

| Package area | Python files | Physical lines |
|---|---:|---:|
| Files directly under `starsim/` | 41 | 48,136 |
| `starsim/solvers/` | 7 | 9,495 |
| `starsim/interop/` | 18 | 6,348 |
| `starsim/library/` | 17 | 2,727 |
| **Total package** | **83** | **66,706** |
| Tests, separately | 82 | 23,839 |

An exclusive physical-line classification gives 13,324 docstring lines, 6,045 blank lines outside docstrings, 1,198 comment-only lines outside docstrings, and 46,139 remaining lines. The last number is not a logical statement count. This matters: a 40k *physical*-line target should not be met by deleting useful documentation or compressing readable code.

The same package, regrouped by responsibility, is:

| Responsibility | Included files | Lines |
|---|---|---:|
| Execution and backend support | All solvers, `kernels`, `gradients` | 14,187 |
| Declarative language and its presentation | `expressions`, `conditions`, `transitions`, `graph`, `dimensions`, `presentation`, `spec` | 7,330 |
| Population, arrays, networks, reproductive demography, strains | `people`, `arrays`, `networks`, `demographics`, `pregnancy`, `strains` | 7,851 |
| Results, ensembles, calibration, sample storage | `results`, `run`, `calibration`, `samples` | 6,648 |
| Import/export adapters | `interop/` | 6,348 |
| Orchestration and parameter handling | `sim`, `modules`, `loop`, `parameters` | 5,949 |
| Interventions and products | `interventions`, `products` | 4,495 |
| Time and calendar handling | `time`, `dates`, `timeline`, `timeplot` | 3,363 |
| Disease base classes and basic models | `diseases` | 1,525 |
| Random distributions | `distributions` | 2,349 |
| Reference model library | `library/` | 2,727 |
| Other support | Remaining root files | 3,934 |

These categories are disjoint. They are an accounting device, not my proposed package boundaries.

I also counted branches and loops in function ASTs as a rough navigation aid, **not** a formal complexity score. The large concentrations include `choose_solver()` (244 lines), `init_capabilities()` (187), `Compartmental.build_layout()` (171), `Compartmental.init_program()` (207), `Groups.init()` (185), `Crossing.build()` (259), `kernels.step()` (343), and several importers with 200–380-line translation functions. This supports the architectural diagnosis: the hard parts reconcile representations, features, and inferred behavior.

## 2. The compartmental boundary should become much firmer

### 2.1 Keep a useful, bounded common subset

I would keep automatic execution of declared models for:

- Ordinary state-to-state hazards, inflows/outflows, and contact-mediated transmission.
- Explicit categorical stratification and explicit contact matrices.
- A bounded, documented set of dwell representations: exponential hazards, explicit Erlang stages, and, if justified by representative models, one fixed-delay mechanism.
- Parameters depending on time or explicit groups, and basic scheduled parameter changes.
- Basic aggregate observations and results.

This is already a substantial compartmental framework. It covers useful models without requiring it to simulate arbitrary individual memory through hidden dimensions.

The policy should be: `auto` uses compartments only for this known subset, otherwise ABM. An explicitly requested compartmental solver refuses unsupported semantics and explains the working alternative. It should not silently drop an event, assume independence between correlated histories, or infer a biologically meaningful approximation from an individual's code.

This does not mean every unsupported *Python component* must become ABM. A user-owned NumPy ODE should continue to run as the ODE it is. The boundary concerns automatic lowering of *Starsim declarations*; ordinary components retain their own algorithms.

### 2.2 Fetal compartment expansion: highest-priority candidate to stop

Evidence: [pregnancy.py](../../starsim/pregnancy.py), especially `fetal_specs()`, `route()`, `add_fetal_dims()` at approximately lines 221–397 and `unmother()` near 797; [choose.py](../../starsim/solvers/choose.py), `choose_solver()`; [compartmental.py](../../starsim/solvers/compartmental.py), fetal birth and routing logic; [current implementation plan](implementation.md), B7.

An unborn individual's states become private dimensions of pregnant compartments. The implementation recognizes fetal transitions, substitutes maternal expressions, merges fetal hazards into pregnancy loss rates, restores original rates on rebuilding, and routes children at delivery. Solver selection knows about `Pregnancy` specifically. This is a domain feature reaching deeply into generic execution.

The B7 notes are especially informative: adding breastfeeding mass action exposed a need to mix by disease dimensions within a crossing or retain mother–child links. That is a strong indication that the compartmental abstraction has reached its practical boundary.

**Recommendation:** keep pregnancy and vertical transmission as composable ABM modules; stop extending automatic fetal compartmental conversion. Offer a separate explicitly aggregate maternal/infant model if users need one. Its compartments and transmission assumptions should be written by the model author rather than derived from relationship expressions.

The tradeoff is real: small, smooth maternal/infant models can be much cheaper to fit as ODEs. Preserve that option through user-authored aggregate components, not a universal relationship compiler. Existing passing fetal ODE tests demonstrate functionality, not that its maintenance cost is justified.

### 2.3 Arbitrary attributes, remembered events, and clocks should usually mean agents

Evidence: [expressions.py](../../starsim/expressions.py), `attr`, `Clock`, `Since`; [choose.py](../../starsim/solvers/choose.py), `attr_form()`, `sample_attr()`, `add_attr_strats()` around lines 343–412; [interventions.py](../../starsim/interventions.py), `memory_needs()`, `memory_layout()`, `remember_levels()`, `Delivery.memory_strats()`; [products.py](../../starsim/products.py), `EventSource`.

The new attribute conversion draws 100,000 samples to construct equal-probability representative bins, adds an unset level, and builds a transition matrix for `last`, `first`, `max`, or `min`. These hidden dimensions apply across the module's people states. This is a substantial modeling approximation, not simply another storage layout.

An illustrative state-space calculation: 5 disease states × 5 antibody levels × 4 dose levels × 12 time-since-event levels × 10 age groups gives 12,000 cells before another disease or geography. Actual layouts can be state-local or pruned; the multiplication is the concern, not a measured layout for a particular model.

**Recommendation:** preserve per-agent attributes and event memory, but remove their automatic compartmental expansion from the default engine. Experts who want antibody categories or dose strata can declare those strata explicitly. Start by making `auto` choose ABM for these features and stop developing the remaining B6 conversions.

For clocks, retain one simple dwell mechanism if it serves common natural histories. Do not grow it into a general compiler for set clocks, arbitrary progress, memories read from another module, and clocks carried across crossings. The current distinction between `X.time`, `X.dwell`, `(A | B).time`, event `.since`, `progress`, Erlang stages, and conveyor slots is already a large conceptual burden. Users should primarily learn “time since entry” and “time since event”; backend-specific approximations belong in the advanced compartmental documentation.

Workaround: use the existing event callback/NumPy array path for a running maximum, a frozen CD4 value, or a last-visit time. This is often clearer than a new declaration variant. Keep the declarative agent attribute helper where it substantially improves model readability; do not require it for custom code.

### 2.4 Care queues and remembered eligibility have an especially expensive aggregate form

Evidence: [interventions.py](../../starsim/interventions.py), `Delivery` beginning near 954, constructor near 1038, `dims()`, `memory_strats()`, `serve_queue()` / `serve_queue_groups()`, `carry_events()`, `cure_groups()` / `cure_crossed()` / `force_part()`; [products.py](../../starsim/products.py), `Held`.

`Delivery` combines product creation, scheduling, eligibility, triggering, coverage, capacity, shared resources, priority, waiting, stock targets, age thresholds, delivery delay, diagnostic delay, persistent effects, event tracking, movement between states, movement between strata, counters, and multi-disease treatment. It has over thirty named constructor parameters before product-specific keyword forwarding. The difficulty is the number of interacting policies, not the long signature alone.

The compartmental versions must preserve the identity of a selected fraction as people move; distinguish previously offered from eligible people; move queued mass among groups; retain levels for elapsed time; and distribute multi-disease outcomes through crossings. These are exactly the tasks agent identity makes straightforward.

**Recommendation:** keep simple vaccination, testing, and treatment easy to write. Make complex care pathways an optional `programs` layer, with ABM as the standard execution for queues, priority, remembered selection, and individual appointment histories. If a project needs a health-service ODE, expose an explicit aggregate program model rather than translating an individual queue.

Capacity limits alone are common enough to retain in a simple form. A universal priority/queue/target/resource system in the mandatory delivery path is not necessary for a basic vaccination campaign. Factor policies behind ordinary Python objects or functions and compose them only when used; avoid replacing one large constructor with a mandatory collection of builder classes.

### 2.5 Automatic joint disease crossings should be bounded or explicit

Evidence: [choose.py](../../starsim/solvers/choose.py), `crossing_groups()` / `init_crossings()` around 415–499; [crossing.py](../../starsim/solvers/crossing.py), 777 lines; [sim.py](../../starsim/sim.py), `_init_flows()`.

Crossings correctly address an important issue: knowing HIV prevalence and syphilis prevalence does not give their joint distribution. They create product compartments, transform each disease's view, merge compiled programs, synchronize events, and reconcile births and deaths. Population-wide births/deaths can link *all* declared modules into joint tracking. Adding demography can therefore change the representation of a seemingly unrelated multi-disease model.

**Recommendation:** retain a basic joint-state mechanism only if canonical multi-disease compartmental models justify it. Make its cost explicit at build time, estimate expanded cells before constructing them, and use a documented resource budget for `auto`. Choose ABM when the budget or supported semantics is exceeded; explicit ODE should explain the issue rather than quietly change solver.

A stronger simplification is to require a deliberately authored joint compartmental model and remove automatic crossings entirely. That is a second-stage decision, not my first cut: co-infection models are central to Starsim, and an explicit aggregate model can be verbose. Either way, stop extending crossings to every history and relationship feature.

### 2.6 Do not turn individual networks into specialized ODE models automatically

Evidence: [mixing.py](../../starsim/solvers/mixing.py), `init()`, `partnerships()` near 220; [compartmental.py](../../starsim/solvers/compartmental.py), `partnership_chance()` and related contact/exit calculations; [networks.py](../../starsim/networks.py), `PartnerNet`, `MFNet`, maternal routes.

Homogeneous mixing and explicitly supplied matrices are generic. Inferring a partnership approximation from lasting individual partnerships adds assumptions about formation, duration, infectious partners, and transmission over a relationship. Prenatal and breastfeeding conversion add other assumptions. These are model choices.

**Recommendation:** individual networks select ABM; contact matrices select aggregate mixing. If a user wants to replace an individual network with an approximation, require an explicit matrix or a separate approximation function in an extension package. A simple mean-degree conversion may remain as an explicit utility, not an automatic scientific promise.

### 2.7 Stop choosing exact versus approximate stochastic algorithms from a trial run

Evidence: [choose.py](../../starsim/solvers/choose.py), `max_seconds = 1.0`, `exact_seconds()`, `count_events()`, and the Gillespie branch of `choose_solver()`.

Asking for Gillespie can initialize and run an ODE to predict event count, estimate runtime with hardware/model heuristics, and change the actual algorithm to tau-leaping. `approx=False` prevents that switch, but the default solver name no longer completely describes the algorithm. This adds startup work and a surprising conceptual exception.

**Recommendation:** an explicit exact solver stays exact. If tau-leaping is needed, make it an explicit algorithm or an explicitly requested acceleration policy. `auto` may recommend an algorithm from cheap structural estimates; a full preliminary ODE run should not be required for a routine solver choice. Removing the internal tau implementation is a separate benchmark-dependent decision.

## 3. Package boundaries: move capabilities without leaving hooks everywhere

Usage claims below are judgments about likely audience, not telemetry. Before removing compatibility for existing downstream projects, check actual callers. Specialized does not mean unimportant.

| Capability | Current location and size context | Suggested home | Keep in core / practical workaround |
|---|---|---|---|
| Foreign-framework parsers and exporters | `interop/`, 6,348 lines | Optional `starsim-interop` distribution; per-format adapters inside it [CK: i think keep] | Small explicit model construction interface and an optional adapter entry point. Users can also port or wrap their source model manually. |
| Reference disease/network/MNCH models | `library/`, 2,727 lines | Separate reference-model distribution, with a few small teaching models retained if useful [CK: no, keep and expand, this is useful] | Basic SIR/SEIR/SIS examples and stable extension contracts. Model-specific policy remains downstream. |
| Advanced inference | `calibration.py`, 3,428 lines; `gradients.py`, 1,242 | Optional inference/JAX packages or independently imported subpackages first [CK: think this could be a submodule, e.g. ss.fit.[...]] | Black-box run/read objective, restart/factory API, result alignment. A simple fitting convenience can delegate. |
| Pregnancy, lactation, fetal health, relationship-specific policy | `pregnancy.py`, 928; maternal network classes; reproductive defaults | Reproductive-health extension, potentially shared by FPsim/STIsim/MNCH projects [CK: hard since pregnancy is used by multiple downstream models ... inclined to keep] | Generic birth/death, population growth, parent IDs/relationships if shared enough, events. No `Pregnancy` checks in solver selection. |
| Sexual partnership policy and per-act conventions | `SexualNetwork`, `MFNet`, some maternal/sexual routes | STIsim or a shared networks extension [CK: these should be relatively minimal with PartnerNet, so would be inclined to keep ... but agree some simplification is probably fine] | Generic edges, optional edge metadata/directions, generic transmission hooks. `PartnerNet` should be assessed separately: persistent partnerships are not exclusively sexual. |
| Multi-strain immunity policy | `strains.py`, 756; strain-specific logic in engines | `ss.strains` initially; domain extension where policy is specific [CK: hm, i do feel multiple strains should be a first-class feature] | Generic dimensions, modifiers, and agent history primitives. Do not generalize the backend for every immunity rule. |
| Full care-program resource/priority/queue policy | Parts of `interventions.py` and `products.py` | Optional `ss.programs`, eventually separately distributed if useful [CK: potentially] | Simple product delivery and generic events. Handwritten module handles an unusual pathway. |
| Geography lookup, gravity, mobility data import | Parts of `networks.py`, `named.py` | `ss.spatial` and the existing external data provider [CK: hm, probably] | Explicit mixing matrices, categorical groups, and transfers. Plain NumPy/SciPy arrays remain valid inputs. |
| Forecast-hub/sample archive conventions | `samples.py`, 432; `run.py` hub quantiles | `ss.io` / downstream forecast tooling [CK: don't love `ss.io` for this, but generally agree] | Results as arrays/dataframes; ZIP/CSV/Parquet handled externally. Preserve current consumers during migration. |
| Migration tooling | `migrate.py`, 377 | CLI/dev extra or compatibility package [CK: oh gosh yes ... this should go in ss.interop] | Short migration guidance, no perpetual runtime interception for every removed keyword. |
| Debug UI and plotting helpers | `debugtools.py`, 496; plotting portions throughout | `ss.debug`, `ss.plotting` [CK: debug yes, plotting no, it's a first class feature] | Clear exceptions, structured diagnostics, and result export. Lazily import presentation dependencies. |

### Importers are the cleanest first extraction

The project currently understands eleven named formats in `interop.__init__`, with format sniffing and export support. This is an entire translation product, including external expression syntaxes and source-framework conventions. Several of its loaders are among the package's most branch-heavy functions.

Its `Source` intermediate representation is already a useful seam. Move the adapters and, if appropriate, `Source` plus its build logic together; do not force every importer convention into the core graph. Keep a small constructor-facing interface. The generic simulation package should not need to recognize Atomica workbook layouts or R model syntax. [CK: disagree, i think it will really help uptake if someone can just type `ss.interop.import('my-atomica-workbook.xlsx')`]

Do not break `ss.load()` for ordinary Starsim objects while doing this. Separate loading saved simulations from translating foreign models in the documentation and dispatch layer. [CK: agree that `ss.load()` should be reserved for actual starsim models; `ss.interop.import()` for everything else]

### Inference should consume simulations, not define their minimum interface

The current file contains the older `Calibration` / `CalibComponent` system as well as the newer `Problem` / `Fit` system, several optimizers, ABC-SMC, iterated filtering, particle MCMC, two NUTS execution routes, posterior diagnostics, policy optimization, sensitivity, and model comparison. [CK: yes, this is bad, Calibration/CalibComponent needs to go, though not sure i love the name `Problem`]

`default_method()` around 2611 can choose among fundamentally different inferential methods based on solver, likelihood coverage, gradient support, parameter count, and installed libraries. A fit can therefore change character when BlackJAX is installed. Convenience here carries scientific meaning.

**Recommendation:** retain one fitting entry point with an explicit method for posterior inference; let a conservative point-estimation default remain if desired. Consolidate common data alignment and objective construction. Keep algorithms in an optional inference layer. Export a black-box `objective(parameters) -> scalar` path for *every* Python component, with optional derivatives for those that provide them.

`gradients.py` currently re-executes the compiled program in JAX, while differentiating Python setup/output along directions with central differences. That is clever and valuable for the supported subset. It should be an optional accelerator with an explicit eligibility contract, not an expectation every extension must eventually satisfy. Do not add a third backend implementation of advanced care-program policy simply for parity. [CK: agree]

### Extraction must remove reverse dependencies

There are 70 package files containing `import starsim as ss`. Many internal base classes and dispatches resolve through the fully populated root package. Core selection currently recognizes `Pregnancy`, `Strains`, `Delivery`, and particular network subclasses. Moving their definitions without removing those checks merely spreads the tangle across distributions.

Use direct relative imports inside genuinely low-level modules, and small generic extension contracts where needed. A domain component supplies its implementation and supported execution modes; core should not import a catalog of downstream component types. Avoid replacing the current checks with a large plugin framework. A few explicit adapter functions and component methods should be enough. [CK: not sure i agree with this. i hate big import blocks at the top of files, so i wouldn't want one `import starsim as ss` line to replaced by 20  `from . import sim` lines]

## 4. Duplication and convenience: which pieces are actually worth cutting?

### 4.1 `mother()` is exactly the kind of tiny duplicate worth removing, but not a major LOC source

[expressions.py](../../starsim/expressions.py) defines `Mother` near 513 and `mother()` near 536. The latter only returns `Mother(x, born=born)` and repeats a substantial amount of documentation. Currently only lowercase `mother` is imported at the `ss` root; `ss.Mother` is **not** another root export. The implementation class remains accessible through the expressions submodule.

Choose one public constructor spelling and put its documentation there. Either make the implementation class lowercase, use a direct alias if compatibility requires it, or expose only the class. Do the same for `Child` / `child()` in [conditions.py](../../starsim/conditions.py). If maternal functionality moves out, both move with it.

Do not infer that every lowercase/uppercase pair is redundant:

- `People` and `people('Kenya')` have different behavior; the latter looks up population data and constructs configured people. Put the lookup under a named-data namespace rather than maintaining a confusing distinction by case alone.
- `Vaccine` and `vaccinate` describe a product and its delivery. That distinction is useful. The convenience that `vaccinate(efficacy=...)` constructs a product internally is the duplicate *construction path* to reconsider.
- `State` and `states()` distinguish one object from a batch factory; this is inexpensive and useful.
- `prob`, hazard rates, and event frequencies carry scientifically different meanings. Do not collapse them into an untyped number to remove classes.

### 4.2 The larger duplication is old and new intervention/product systems

[interventions.py](../../starsim/interventions.py), roughly lines 15–459, contains `RoutineDelivery`, `CampaignDelivery`, `BaseTest`, `BaseScreening`, `BaseTriage`, `BaseTreatment`, `treat_num`, and `BaseVaccination`, before the newer delivery/event/program system. [products.py](../../starsim/products.py), roughly 194–335, contains `Dx`, `Tx`, and `Vx`, alongside `Diagnostic`, `Treatment`, and `Vaccine`. [CK: yes these old classes can be removed, by and large or entirely]

These have real differences and downstream subclass users; they are not safe mechanical aliases. Nevertheless, two systems mean two eligibility vocabularies, product protocols, scheduling patterns, result conventions, and extension guides.

**Recommendation:** move the older family into an explicitly supported v3 compatibility package or namespace with a defined migration path. Preserve handwritten-product extensibility in the new small interface. Remove old/new special handling from core once consumers migrate. The directly isolated blocks are only about 600 physical lines; a 700–1,200-line total saving including associated glue is a planning estimate, not a measured deletion. [CK: no, just remove the old names, with a clear migration script/skill. but ensure that the new classes fully support the functionality of the old classes, which is not to say that every bell and whistle must be exactly replicated in the new classes.]

### 4.3 Canonicalize construction instead of supporting every form deep into the runtime

The declarative front end accepts arrow strings, operator expressions, class members, positional objects, keywords, grouped declarations, inferred states, and reused objects. `Disease.reuse()` handles identity, cross-module references, and copying; `init_graph()` copies class members and rebinds references; `Graph.collect()` and `resolve_states()` resolve ambiguity and naming.

Keep a friendly short syntax, but normalize it immediately into one explicit model description. Solvers should never care whether a state came from a string, class, operator, or positional argument. A helpful syntax is justified when its entire implementation is a thin front end; it is expensive when its ambiguities survive into ownership, serialization, and inference.

Specific candidates:

- Stop inspecting calling frames to discover variable names. Keyword/class attribute names and explicit `name=` are sufficient; unnamed inline transitions can receive deterministic generated names. Keep `__set_name__` for class declarations. [CK: ok]
- Avoid inferring whether arbitrary stocks are people or reservoirs from graph connectivity. An explicit `people=False` is a small price for removing a consequential guess. A basic disease helper can supply sensible defaults. [CK: ok]
- Keep initial conditions explicit in scientific model definitions. The automatic 1% infection guess is acceptable for `demo()`, less defensible as a generic graph default. [CK: agree, but again one shouldn't ever have to write S=0.99, I=0.01. just one or the other, although supplying both works too, unless they don't add up to 1!]
- Limit full model-construction syntax to one recommended function form and one class extension form. Other convenient forms can remain if they normalize cheaply; they should not each acquire unique features. [CK: ok]
- Prefer `Sim(modules=[...])` or positional modules as the documented composition path. Keep category keywords as conveniences, but avoid a separate registration/discovery story for each category. [CK: hm ... not sure i agree with this. it's a big departure from v3, and there is a lot of value in having a consistent integration loop, at least for agents. true that for ODEs it doesn't matter so much.]

### 4.4 Simplify updates and ownership; stop repairing arbitrary Python

[parameters.py](../../starsim/parameters.py), `Pars` update methods, dispatches on the *old* and *new* types: a number can mutate a distribution parameter rather than replace it; a list can mean positional distribution arguments; a dict can mean values or a new distribution type. Separately, `par` has a `.value`, `.raw`, implicit float conversion, bounds, modifiers, and module ownership. `float(par)` can average a modified value across groups.

**Recommendation:** assignment replaces the value; changing a distribution uses that distribution's explicit update method. Keep parameter metadata in a stable wrapper, but provide one explicit way to obtain an effective value in a context. Do not let scalar coercion silently average heterogeneous values. This removes a considerable number of rules users and agents must remember. [CK: ok]

[sim.py](../../starsim/sim.py), `remap_closures()` around 2424, inspects bytecode for globals, reconstructs closure cells and function objects, and recursively walks copied objects. This exists because simulations copy modules but functions can keep references to the originals.

**Recommendation:** document a simple ownership model: a built simulation owns its mutable components; independent runs come from a model factory or an explicit clone operation. Keep `copy_inputs` compatibility initially, but stop promising repair of arbitrary callbacks. A callback receives its context or an explicitly bound component. Prefer factories for ensembles and fitting so copying is a deliberate operation at one boundary. [CK: ok]

This is a behavioral change, not an optimization. It needs migration examples and checks for accidental component reuse. Do not simply disable copying and allow two simulations to mutate the same objects. [CK: not sure what 'migration examples' means here since the v4 code has never been released; no one is using this currently]

### 4.5 Consolidate expression machinery without building a bigger language

`Expr` and `Cond` both implement arithmetic, comparisons, NumPy dispatch, composition, and walkers. The language is then evaluated or rewritten in Python execution contexts, compartment lowering, JAX contexts, display, source generation, and importers. There are repeated visitors such as `clocks_in()`, `mothers_in()`, `state_refs()`, parameter scans, and serializers.

Keep numerical expressions and selections semantically distinct, but use one small immutable internal tree representation or common traversal utility where it demonstrably removes repetition. Distinguish “count of I”, “exposure to I”, “membership in I”, and “event affecting I” explicitly at normalization. Today a state changes meaning by context, and bare `I` versus `I.n` is a particularly consequential difference for transmission. [CK: ok]

Do not add a universal expression language, schema engine, or visitor hierarchy solely to save repeated `isinstance` checks. First remove maternal/history/aggregate conversions and importer-specific needs; then consolidate the much smaller surviving language. [CK: not sure exactly what this means, but ... ok]

## 5. Reduce logical concepts and organize the public API

I counted 328 non-underscore names in `dir(starsim)` in the probe environment. That includes submodules and support names and is not a count of independent user concepts. `errors.__all__` alone exports 65 names through the root wildcard. Root imports and the library flattening mechanism make API boundaries harder to infer.

The root already keeps solvers and reference library models out of its direct exports. Build on that work rather than starting another namespace redesign.

| Namespace | Intended audience / contents |
|---|---|
| `ss` | Small recommended modeling vocabulary: simulation/component, basic disease declarations, common time quantities, common distributions, basic interventions and results. Exact membership should follow tutorials and real usage. |
| `ss.errors` | Specific errors, warnings, and notes; optionally keep one base error at root for convenience. |
| `ss.library` | Optional examples/reference models; retain nested disease/network/MNCH modules, stop automatically flattening every subpackage object into `ssl`. |
| `ss.solvers` | Advanced solver configuration and implementations; current direction is good. |
| `ss.programs` | Care pathways, priority, resources, queues, cascade analysis. Optional. [CK: worried about the boundary of what's a 'common' program in `ss.x`, and what's an 'advanced' one in `ss.programs.x`. but we could probably define a reasonable boundary...] |
| `ss.strains` | Strain families and immunity policies. [CK: if it's just `ss.strains()`, i think it's fine ... if there's a big vocabulary (?) then agree maybe a submodule would be good] |
| `ss.spatial` | Geography, mobility, gravity, specialized spatial construction. [CK: ok, or `ss.geo`] |
| `ss.inference` or `ss.calibration` | Fitting methods, posterior diagnostics, model comparison, policy optimization, optional derivatives. Choose one canonical namespace rather than keeping both. [CK: ok, or `ss.fit`] |
| `ss.data` / `ss.named` | Named population/contact/geography lookups and provenance. Resolve the existing root `data` observation object before assigning this name. [CK: there's already `starsim_data` as a separate repo, but yes] |
| `ss.debug`, `ss.plotting`, `ss.io` | Development, visualization, and persistence tooling. [CK: debug yes, io yes (but maybe a better name for a probably rarely used feature), plotting no -- except maybe for very advanced plotting tools, and then maybe `ss.plt` in parallel with `import matplotlib.pyplot as plt`] |

These are directions, not a proposal to create ten new mandatory packages. Start by exposing the *existing* submodules consistently and hiding specialized root exports. Create a distribution only when the dependency boundary is real.

Examples of names to move off root early: specific exceptions; `mock_*`, `Profile`, `Debugger`; specialized sexual/maternal network classes; `PregnancyPars`; old intervention/product bases; posterior likelihood component classes; `Dataset` / `Samples`; low-level plotting helpers. Common distributions and unit constructors are cheap vocabulary and need not all become `ss.distributions.*` / `ss.time.*` to reach a symbol quota. [CK: agree, and a lot of these are historical, so maybe could even be removed entirely]

The larger conceptual reductions should be:

1. **One component extension story.** Users start with ordinary state and an update function. Specialized base classes supply useful defaults; they are not prerequisites for scheduling. [CK: love it]
2. **One default execution story.** A Python schedule calls components; a declared disease optionally compiles. Compilation changes execution speed, not the scientific interpretation of a custom component. [CK: sounds good]
3. **One clear selection story.** Current membership, an event, and historical memory are related but distinct. Prefer explicit event filters to accepting an overloaded `who` expression and splitting it into event and eligibility parts with hidden inference. Keep the concise common cases. [CK: ok]
4. **One parameter-value story.** A parameter definition, its assigned value, and its effective value under modifiers should be discoverable and explicit. Avoid scalar conversions that choose a reduction. [CK: hm, i think this is ok]
5. **One intervention story.** A product describes an effect; delivery says when/to whom. Advanced resource policies are optional. A modeler should not learn `BaseScreening` versus `screen` to pick an extension route. [CK: ok]
6. **One scientific model description, with a limited serialization promise.** Results export, a model summary, a checkpoint, and generated executable source are different operations. `sim.to_json()` and disease `to_json()` currently serve different purposes; document and separate those meanings. [CK: ok]

Avoid long-lived root aliases for every move. A short compatibility layer with an expiry is preferable to maintaining two supported spellings forever. Namespace changes mostly improve discoverability; do not count them as thousands of lines saved. [CK: we don't need any compatibility layer in the context of AI migrations; we should just build the best library we can]

## 6. Make incremental adoption an executable architectural requirement

### 6.1 What works today

I ran a tiny `ss.Module` containing only a NumPy array and a `step()` that multiplies it by 0.9. In `Sim(..., n_agents=1, start=0, stop=3, dt=1)` it took three steps at indices `[0, 1, 2]` and returned `[0.7290000000000001]`. It used no Starsim state arrays or disease declarations.

The simulation nevertheless selected `abm` and created a `People` object. Asking for `solver='ode'` with only this handwritten module raised `CapabilityError`: the model had no declared states/transitions and so ran as agents. A standalone `Timeline(start=0, stop=3, dt=1)` worked and returned four time points.

I also ran the same NumPy update alongside `ss.SIR()` with `compartmental=True` on the custom module, under both `solver='abm'` and `solver='ode'`. Both returned `[0.7290000000000001]`; both used the Python schedule rather than whole-run kernel execution. This is a useful existing seam, but “works on groups” and “is a user-owned numerical solver” are different properties. Do not require a generic ODE component to masquerade as population code. [CK: agree]

### 6.2 The dependency direction should be reversed

Current [Loop.collect_funcs()](../../starsim/loop.py) constructs a 17-phase schedule by reaching into `Sim`, categories of modules, people lifecycle, results, analyzers, and module timelines. `Module.init_pre()` links distributions, time parameters, `sim.pars`, results, and `sim.people`. `Sim.init()` always enters solver/population setup. Therefore exporting `Loop` alone does not yet provide an independently reusable scheduler.

The small foundation should instead own:

- A numeric schedule and deterministic ordering of ordinary callbacks.
- Optional component initialization/finalization and recording hooks.
- An optional context carrying time, named component references, and an RNG supplied by the caller.

Population storage, disease phases, calendars, results, and the declarative engine should *build a schedule using that foundation*. They should not be required to instantiate it. Keep today's epidemiological phase ordering as a useful preset; do not impose all its phases on a NumPy ODE. [CK: sounds good]

### 6.3 A minimal contract, with explicit limits

Use ordinary Python callables plus a thin optional adapter, conceptually `step(t, dt)` and `record(t)`; a richer component can expose initialization/finalization as needed. Exact signatures are design work, not new API promised by this report. State stays with the component. A context is passed explicitly where required rather than found by object-graph inspection.

For example, the following is **proposed usage**, not code that works against today's API:

```python
y = np.array([990.0, 10.0, 0.0])

def rhs(t, y):
    S, I, R = y
    infection = 0.3 * S * I / y.sum()
    recovery = 0.1 * I
    return np.array([-infection, infection - recovery, recovery])

def step(t, dt):
    y[:] = y + dt * rhs(t, y)  # Or call the user's preferred integrator

# First adoption step: callbacks and numeric times only.
schedule = ss.Loop.from_steps(times=np.arange(0, 101), steps=[step])
schedule.run()

# Later: wrap the same step in a component and put it beside an ABM disease.
component = ss.Component(name='numpy_sir', step=step)
sim = ss.Sim(modules=[component, agent_disease])
```

The adapter does not inspect or translate `rhs`. Euler above is merely a short illustration. An ODE adapter can provide RK4 or a SciPy integration call without changing the user's function. A NumPy ABM component is another callback with its own arrays and RNG. A user replacing only time handling should be able to use `Timeline` around their existing `for` loop, as they already partly can. [CK: yes]

For coupling, use explicit reads/writes or callbacks: an ABM may publish prevalence; a NumPy reservoir component may read it and publish an exposure rate. Components need agreement on units, when values are sampled, and who owns shared state. Do **not** automatically interpret independent aggregate counts as individual membership. That conversion needs a user-supplied coupling model. [CK: ok]

Start with one shared clock. Retain multi-rate scheduling where existing scientific models require it, but make it an advanced scheduler option with explicit sample/hold semantics. A component may integrate internally with smaller steps between communication times. General adaptive, bidirectional co-simulation is outside the minimum contract. [CK: hm ... i do like the loop flexibility, and don't like the idea of having a default system and a totally separate system for slightly more complex things. but equally see the point that >90% of sims all use the same timesteps, so that should be as simple and fast as possible.]

Optional capabilities belong to the component or adapter: result export, restart/snapshot, gradients, compiled execution. Unsupported gradients should explain the limitation; they should not prevent an ordinary run. A black-box fit can rebuild through a factory. Common-random-number guarantees apply to components using the CRN service, not arbitrary `np.random` calls. These limits should be explicit and local. [CK: yes]

### 6.4 Preserve a simple reference execution path

The Python schedule should define behavior for custom components. A declared graph can have optimized NumPy/Numba/JAX implementations for a defined subset. When callbacks or other unsupported features are present, it can run step by step without the whole-run optimization.

There are already useful `kernel_reason()` checks in [sim.py](../../starsim/sim.py) and [compartmental.py](../../starsim/solvers/compartmental.py). Turn them into a small inspectable execution plan: “this graph compiles; this component runs in Python.” Avoid discovering support by running arbitrary user code on a temporary simulation, as `try_capabilities()` currently does. Side effects and one successful probe do not establish semantic compatibility. [CK: i think i agree, unless it needlessly rejects large blocks of code. maybe instead of try_capabilities() it could be up to the user to assert whether a module has a certain capability, and if they're wrong it errors.]

Do not immediately build arbitrary mixtures of agent and compartmental disease *declarations* sharing a population. Today the solver is chosen for the simulation. The first modularity milestone is independent components with explicit coupling, which meets the NumPy adoption goal without a new universal hybrid population engine. [CK: ok]

## 7. Other opportunities and things I would protect

### Numerical kernels: reduce variants only after measuring their value

[kernels.py](../../starsim/kernels.py), 3,450 lines, includes ordinary and sparse place-event algorithms, threaded drivers, group-level Gillespie using tree structures, conveyor handling, whole-run and many-run paths, movement/shifts, and ABM array helpers. The complexity is not all caused by obscure model features; a substantial amount buys speed.

Inventory each specialized execution variant against an explicit benchmark and a user-relevant crossover. Retire variants with no substantial benefit on supported representative workloads. Extract spatial stochastic acceleration into an optional backend if only large spatial runs need it. Share stable low-level operations, but avoid an abstraction that sacrifices Numba performance merely to eliminate repeated loops. [CK: agree]

Do not assume the 3,450 lines are deletable. Nor assume a generic SciPy call preserves stochastic semantics, event accounting, or performance. The existing optimized whole-run path is valuable for ensemble runs. Keep it as an optimization for its subset, rather than adding queues and fetal policy to it for feature parity.

### Time handling: simplify the boundary, retain the distinctions

`time`, `dates`, `timeline`, and `timeplot` total 3,363 lines. They support numeric relative time, calendar time, scalar/array arithmetic, calendar-aware durations, and plotting conversions. This is a broad time library inside a modeling library.

Normalize numeric time to one internal unit at component boundaries. Put calendar parsing and display on an optional adapter. Retain explicit duration versus hazard versus probability versus frequency semantics: confusing these produces wrong models. A `peract` probability is another distinct quantity; its specialized directional conventions can live with network extensions.

Review whether all custom date arithmetic and `datedur` operations need to be public. Use pandas/NumPy/calendar facilities behind a small adapter where they match the intended calendar semantics. Do not mechanically replace a calendar month with a fixed number of days or a probability with a linearized hazard. [CK: would be interested to see the implications of this .. we built the time library because it was needed, which is what v3 got right and v2 got wrong. but it is a lot of code, and willing to consider alternatives. pandas/numpy/datetime/calendar handlings of time all suck, though, so i think we want at least some customization. if they didn't suck there wouldn't already be four of them.]

### Arrays, people, and random distributions: foundational, but make them opt-in

`arrays.py` (1,692), `people.py` (1,178), and `distributions.py` (2,349) provide active-agent indexing, growth/removal, array-like operations, typed states, per-agent draws, and CRN. Those are a large fraction of the original ABM value proposition. Replacing them wholesale with raw NumPy would move bugs and bookkeeping into models.

Keep these services for ABM users. Remove their requirement for numerical components that do not need them. Concentrate possible trimming on attribute proxies, implicit lazy symbolic conversion, and specialized scalar/person convenience APIs, after checking callers. Keep UID/index semantics and CRN guarantees precise. [CK: ok]

`ppl`, `people()`, `People`, `Person`, filters, declared `State`, `BoolState`, and runtime `CountState` currently present many related concepts. Teach two clear paths: ordinary arrays for custom components; the agent population/state service for shared ABMs. Advanced declaration objects should not appear to be interchangeable with arrays. [CK: actually, i was thinking how nice it would be if declared models were fully intercompatible with fully coded ones, at least in the ABM solver. so unclear on this point.]

### Serialization: distinguish recording a model from reconstructing arbitrary Python

`spec.py` is 941 lines; `presentation.py` is 631, and serialization branches occur throughout diseases, interventions, products, and sim. They support a readable model description, generated Python, imported formulas, functions/source capture, and model differences. Some are among the branchiest non-numerical functions.

Keep structured provenance and a readable description of declared models. Limit executable source generation to the supported declarative subset and move it into optional tooling. For handwritten components, save the factory identifier/configuration where available or say that executable reconstruction is unavailable. Do not add reconstruction code for every callback or niche program policy. [CK: ok]

A model-definition fingerprint and a simulation checkpoint are different things. Arbitrary source capture should not become a prerequisite for hashing or fitting a NumPy component. This boundary can simplify code and reduce misleading reproducibility claims. [CK: agree]

### Results and ensembles: keep the data model small and explicit

`Result` alone spans roughly a thousand lines of `results.py`, including array behavior, metadata, group/run axes, aggregation, and plotting. `MultiSim` also handles scenarios, sweeps, reductions, restarted simulations, and optimized backends. Some of this is central and worth keeping.

Preserve labeled axes, stock/flow distinction, observation windows, and numeric export. Move forecast-specific quantile conventions and specialized plotting/storage out. Avoid retaining a second older ensemble/result representation after a newer one is established. Prefer explicit run × time × group axes and views over special rules for whether a simulation was restarted or retained. [CK: i think plotting and storage are critical; result.plot() and multisim.plot() should both just work, with good options for customization. but agree about not preserving an older superseded representation.]

Keep stock versus flow alignment particularly carefully: `results.compare()` and calibration's older conform functions both address this territory. Consolidate the numeric alignment implementation; preserve the scientific distinction between interpolation of a stock and integration of incidence over an interval.

### Import side effects and dependencies

Root import currently imports plotting, the reference library, and the interop package; the latter imports all its adapters. It registers plotting date types and can install/load fonts. `optuna` is a mandatory dependency in `pyproject.toml`, even though calibration imports it lazily. JAX is already optional, which is the right direction. [CK: ok]

Move plotting and font setup to first plotting use or an explicit style initializer. Make inference and specialized networking dependencies optional where feasible. Use a small static public export list; do not replace eager import with a complicated dynamic export system. This improves independent adoption and startup behavior more than raw LOC. [CK: all sounds fine, although i do think plotting will be done in a very large number of cases, so not sure it's worth the complexity of lazily importing. and with plotting comes fonts, so maybe worthwhile. but agree should be opt-out.]

### Diagnostics and defensive machinery

`@required`, method-call tracking, attribute locks, time-index checking, and old-name interception protect users from lifecycle mistakes. Some reflect genuine hazards; some compensate for a demanding subclass contract. Keep clear errors for invalid scientific states. As the component contract shrinks, remove checks whose only purpose was enforcing the old lifecycle. [CK: agree, but things like @required have been super helpful]

`__init__._removed` and constructor-specific removed/renamed keyword tables should have a release-limited compatibility policy. They are not the biggest bloat source, but indefinite accumulation guarantees growth. Do not delete useful exception classes or helpful messages simply to make the API appear smaller; namespace them. [CK: agree, we can kind of start afresh, especially with AI help in doing code migrations.]

## 8. A practical reduction budget

There are three different outcomes to track:

- **Extraction:** fewer lines/dependencies in the base installation, but functionality and maintenance still exist elsewhere.
- **Deletion/consolidation:** fewer ecosystem implementation lines and fewer behavior combinations.
- **Boundary improvement:** smaller mental model and fewer reverse dependencies, even if LOC initially increases.

The following is an **illustrative planning ledger**, not a patch estimate. Exact module counts are measured; partial-file savings are judgment calls. The rows allocate distinct work to avoid counting the same cuts twice. Net values assume the retained service code in calibration and domain areas is excluded from their extraction allocations.

| Change | Base-package line reduction, central case | Nature |
|---|---:|---|
| Extract all interop adapters/build tooling [CK: no, keep, but don't import automatically, require import starsim.interop] | 6,348 | Measured current subtree; tiny dispatch glue budgeted below |
| Extract reference library [CK: no, keep, but again require explicit import] | 2,727 | Measured current subtree; a few examples may stay |
| Extract advanced inference and JAX, retain small objective/alignment interface [CK: can be in ss.fit, but keep, this is a major part of the value proposition] | 4,300 | Most of current 4,670 combined lines |
| Extract reproductive/sexual/domain policy and selected strain/program policy [CK: maybe, want to keep strains] | 2,500 | Partial files; requires removing core type checks |
| Remove older intervention/product paths and their glue [CK: ok] | 750 | Separate from extracted new program policy |
| Remove advanced compartmental history/fetal/queue conversions [CK: ok] | 2,800 | Only code still in core after the extraction rows |
| Simplify probing, solver pre-runs, naming, ownership/callback repair [CK: ok] | 1,500 | Partial files; no count of algorithms extracted above |
| Limit source reconstruction and consolidate serializers [CK: ok] | 1,100 | Tooling/support still in core after importer extraction |
| Retire marginal kernel/driver variants [CK: ok] | 1,500 | Provisional; must earn this with benchmarks |
| Consolidate parameter, expression, and result conveniences [CK: ok] | 900 | Provisional; excludes old APIs already counted |
| Add small scheduler/adapters and extraction glue [CK: ok] | **−900** | New code, included rather than ignored |
| **Illustrative total reduction** | **23,525** | |
| **Illustrative remaining package** | **43,181** | From 66,706 physical lines |

The first two extraction rows alone yield **57,631** lines. The next two bring the base to about **50,831**, before genuine simplification. The uncertain deletion rows make roughly **40–47k** a more honest planning range than an exact target. Reaching the lower end requires additional cuts or more aggressive narrowing, particularly of automatic crossings and backend variants. Those cuts need design and benchmark evidence.

The 30k target would require removing another roughly 13k beyond this central case, or defining a much smaller base package with most modeling services outside it. That could be a valid product, but it is a larger decision than trimming bloat.

Track total maintained LOC across optional distributions too. Moving importers is useful because their dependencies and release concerns separate cleanly. Moving fetal conversion into an extension while core continues to know all its internals is largely accounting.

## 9. Suggested sequence and acceptance criteria

### Stage A: change the scope before writing more code

Amend the current design principles and the [October 20 implementation plan](implementation.md). The existing [principles](../best-practices/principles.md) explicitly favor short model text, inferred defaults, and approximation. The source follows those incentives. “Match every framework example in no more model lines” and “plain Python is first class” need a core-complexity constraint to coexist.

Suggested criterion: a new core feature needs repeated use across independent models, a clear alternative, a support matrix, and an explanation of which existing concept it replaces or why the added concept is essential. A few extra lines of ordinary model code are acceptable when they remove a special-purpose framework feature. [CK: agree]

Specifically, pause B1's general set-clock extension, B6's further recorded/derived attribute conversion, B7's remaining breastfeeding conversion, and kernel/JAX parity for full care cascades until the reduced compartmental boundary is chosen. Existing agent features can continue to improve. A planned compartmental xfail is not automatically a requirement to implement that capability. [CK: agree]

### Stage B: prove composition with a small vertical slice

Build the scheduler seam and demonstrate these uses with short runnable examples and focused contract tests:

1. An ordinary NumPy update using Starsim scheduling, with no `People`, distribution search, graph, or result declaration.
2. The same update with a standalone Starsim timeline and explicit unit conversion.
3. A user-owned NumPy ODE component beside a declared ABM disease, coupled through explicit aggregate values.
4. A NumPy ABM component with its own arrays and RNG beside a standard module.
5. An ordinary intervention callback changing a custom component parameter, without declaring Starsim products or modifiers.
6. A black-box fit using a factory for one of these custom components, with a clear local refusal for unsupported gradients.

The same model function must remain usable outside Starsim. No example should need subclasses of `People`, frame inspection, manual initialization of hidden graph fields, or a dummy disease to obtain a desired execution mode. [CK: all sounds good]

### Stage C: take clean extractions and consolidate duplicate APIs

Extract interop first, make `library` optional, namespace errors/debugging, consolidate intervention/product extensions, then split inference. Preserve downstream behavior through narrow adapters and migration examples. Establish direct-import dependency boundaries before moving the heavily integrated pregnancy/program code.

For each extraction, demonstrate that the base package imports and runs its simplest model without the optional dependency, and that the extension runs against the public component contract. A file move alone is not the acceptance test. [CK: didn't quite follow all of this, but ss.interop and ss.library should stay part of the library, but requiring explicit import]

### Stage D: remove aggregate emulation and simplify the compiler

Replace unsupported declaration combinations with early capability decisions. Delete hidden attribute/memory/fetal expansion, narrow network approximations, and decide the crossing boundary. Simplify expression normalization and solver planning *after* those removals, when fewer cases remain.

Protect conservation, competing-risk semantics, event accounting, CRN, time conversion, and observations with existing contract tests. Change tests for deliberately unsupported combinations into tests of clear fallback/refusal. Do not delete the scientific invariants or blanket-xfail failures to claim a successful simplification. [CK: ok]

### Stage E: prune optimizations with measurements

Run a small representative benchmark set: basic ODE, basic ABM, a multi-disease ABM, a large spatial model, and an ensemble/calibration workload. Measure cold setup/compile time as well as repeated execution. Retire costly implementation variants only where the user-facing benefit is small or an optional backend can own them independently. [CK: ok]

Track physical LOC, base dependencies, startup cost, explicit supported feature combinations, and how many subsystems change for a new ordinary component. The last two metrics are closer to the user's concern about slowed development than a symbol count. [CK: ok]

## 10. Survey checklist: disposition of the whole package

| Files/subsystems | Disposition |
|---|---|
| `sim`, `modules`, `loop` | Highest architectural priority: invert scheduling dependency; ordinary components; simpler ownership and lifecycle. |
| `expressions`, `conditions`, `transitions`, `graph` | Keep useful declarative subset; normalize once; reduce inference and repeated walkers; narrow supported semantics. |
| `dimensions` | Keep explicit categorical groups/transfers; remove hidden history expansion; bound product dimensions. |
| `solvers/choose` | Replace runtime probing and scientific approximation heuristics with structural checks and explicit plans. |
| `solvers/engine`, `agents` | Keep ABM/state service value; remove generic-layer domain policy; shrink base responsibilities after feature cuts. |
| `solvers/compartmental`, `crossing`, `mixing` | Largest deletion opportunity from a reduced support boundary. Keep common aggregate modeling; avoid automatic individual-history and relationship emulation. |
| `kernels`, `gradients` | Keep measured acceleration for a defined subset; optional JAX; benchmark specialized variants before deletion. |
| `parameters` | Explicit replacement/update semantics; separate metadata from effective contextual values. |
| `arrays`, `people`, `distributions` | Keep ABM services and CRN; make optional to nonpopulation components; trim proxies only with caller evidence. |
| `time`, `dates`, `timeline`, `timeplot` | Keep scientific quantities and standalone timeline; numeric core, optional calendar/plot adapters. [CK: maybe, be careful changing this] |
| `diseases`, `demographics` | Keep generic disease interfaces/basic models/birth-death support; distinguish generic components from population assumptions. |
| `pregnancy`, maternal network pieces | Domain extension; ABM-first relationships; avoid pregnancy-specific solver checks. [CK: leaning towards keeping pregnancy] |
| `networks`, `named` | Generic contacts/edges in core; sexual policy, spatial construction, and named-data lookup in specialized namespaces/extensions. |
| `strains` | Specialized namespace; retain useful composition, limit automatic cross-immunity approximations. [CK: not sure about specialized namespace] |
| `interventions`, `products`, `connectors` | One extension protocol; keep composable effects/simple deliveries; optional advanced programs; retire duplicate old families. |
| `results`, `run` | Keep explicit numeric/labeled output and ensembles; simplify historical representations and move specialized conventions. |
| `calibration`, `samples` | Optional inference/storage layer; consolidate fitting/alignment; explicit posterior methods. |
| `spec`, `presentation` | Keep readable descriptions/provenance; optional limited source generation, no universal Python reconstruction. |
| `errors`, `debugtools`, `analyzers` | Namespace rare tools and specific exceptions; retain clear diagnostics and ordinary analyzer hooks. |
| `utils`, `settings`, root exports | Remove import side effects and cross-package dispatch; keep support small; no new general plugin architecture. |
| `migrate`, `version` | Migration tooling outside runtime; version remains trivial. |
| `interop`, `library` | Cleanest immediate optional-distribution boundaries, with independent extension tests. [CK: keep, but functionally separate from root] |

The most valuable outcome would be a modeler saying, “I can use this loop with my arrays, then add only the services I need,” and an engineer being able to implement an ordinary component without changing solver selection, graph expansion, JAX, source generation, and the public namespace together. That is the standard by which I would judge the reduction. [CK: love this!]
