# Paradigm conversion

Running one model text as an ODE, as a stochastic compartmental model, or with agents, and what happens when the model does not fit the solver that was asked for.

## Recommendation

1. **One model text; `solver=` selects how it runs.** For a simple model this is a free run-time choice. For a complex model the paradigm is effectively part of the specification: a model with an individual-level network or per-agent logic is an agent-based model.
2. **`solver='auto'` picks the simplest, fastest representation that keeps the model's mixing and its individual-level structure as written**, and prints what it picked. A user may still ask for agents on a simple model, for example to study elimination.
3. **`auto` never approximates the mixing and never drops individual-level structure. The one thing it does unasked is represent a non-exponential dwell time with Erlang stages, and it says so. An explicitly requested solver uses the standard approximation where one exists, and says so in one line.** It does not refuse.
4. **Stop only when there is no sensible approximation**: individual contact tracing, transmission trees, custom code that only makes sense for agents.
5. **Functions work wherever the things they use exist.** A function that only uses state counts, parameters, time, and stratifications runs in an ODE as well as with agents. No static analysis: try it and report. A module written as code may say which it is (`compartmental = True` or `False`), which overrides the trial.
6. **`solvers()` shows which solvers can run a model**, as a coarse table.
7. **Test agreement between solvers on simple models.** For complex models, report what was approximated; do not promise equivalence.
8. **Think in axes, not in a list of paradigms.** Aggregation, stochasticity, dwell time, and space are four independent axes.

## Why

### It has been done

[EMULSION](../approaches/emulsion.md) switches a whole model between stochastic compartmental and individual-based with **one word**, `aggregation_type: 'compartment'` versus `'IBM'`, with the state machine, transitions, parameters, initial conditions, and outputs identical between the two files. Published in 2019, Apache-2.0. Its review: "This is the paradigm-independence goal, shipped."

[epipack](../approaches/epipack.md) runs one list of process tuples deterministically, stochastically well-mixed, and stochastically on a network. [camdl](../approaches/camdl.md) runs one model text on ODE, chain-binomial, and Gillespie. [odin](../approaches/odin.md) shares `initial()` between `deriv()` and `update()`.

### The alternative is writing each one separately

[MEmilio](../approaches/memilio.md): twenty-plus model families (`ode_*`, `ide_*`, `lct_*`, `glct_*`, `sde_*`, metapop, `abm`, `d_abm`, `graph_abm`, `smm`, `hybrid`), each a separate C++ implementation, most with their own paper. Its review: "MEmilio is the cost estimate for *not* having a lingua franca… the paradigm coverage was bought by writing each one separately."

[EpiModel](../approaches/epimodel.md) gets three paradigms with one input vocabulary and three separately written models: "the honest baseline this project is trying to beat: same vocabulary *and* same model text."

### The axes

From the [MEmilio review](../approaches/memilio.md):

> Not "compartmental / metapopulation / SDE / ABM" as four options, but a lattice with axes: aggregation (compartment ↔ agent), stochasticity (deterministic ↔ diffusion ↔ jump), dwell time (exponential ↔ Erlang ↔ arbitrary), and space (single ↔ patch ↔ explicit mobility). A model specification selects a point; a conversion moves along an axis. This is a more useful framing than a list of paradigms, and it makes the "which conversions are meaningful" question answerable per axis.

### Approximating is normal practice

Much of the published modeling of sexually transmitted infections is compartmental. [Atomica](../approaches/atomica.md) is a compartmental engine with no network in it, and its models "are used for national HIV, TB, and health-system planning". Compartmental STI models usually represent sexual mixing with partner-change rates between population groups. That is a coarser model than a partnership network, and it is a valid one that has informed a great deal of policy. A lingua franca that refused to run an STI model as an ODE would be refusing to do what the field does every day.

[camdl](../approaches/camdl.md) takes the strict line: "Every backend × method combination either works and is tested, or fails through a capability check that names the limitation — no silent gaps". The second half of that is worth keeping (the user is always told). The first half is too strict for our users, who should not have to rewrite a model by hand to get an approximation that the library could have produced.

[EoN](../approaches/eon.md) is the evidence for why the approximation has to be named. It implements about a dozen different analytic approximations to SIS and SIR on a network, which differ in accuracy, so "the ODE version of a network model" is not one thing. We do not offer that choice to users. The library uses one standard approximation per ingredient and states which. A user who needs the exact network result runs agents.

For a sexual network the standard approximation is partner-change rates between groups (by sex, risk, and age), a mixing matrix between them, and a transmission probability per partnership, all derived from the network. It leaves out that partnerships persist (someone in a long partnership with an uninfected partner is protected while it lasts), concurrency, and variation in partner numbers within a group, which matter for early growth and partner-based interventions; for projections over years it is the standard tool. The alternatives in the review are worse: **Atomica** has the user write the force of infection by hand in the framework spreadsheet (`(1 - (1-ch_prev*transpercontact)**floor(contacts)*...)`, the point "where a spreadsheet model stops being legible"), and **[EpiModel](../approaches/epimodel.md)** keeps the deterministic and network versions as separate models (`dcm()` with a hand-written derivative, `netsim()` with a fitted network). It is not built yet: a network of partnerships under an explicit compartmental solver raises `ss.CapabilityError` ([12-backends.md](../design/12-backends.md#the-standard-approximations)). A user who wants the compartmental STI model as the model writes it that way from the start, with a stratification and mixing pools between the groups, and then nothing is approximated.

### Simple models and complex models are different

A two-line SIR can be run any way, and choosing is a matter of the question being asked. A national HIV model with age-disparate partnerships, a care cascade, and contact tracing cannot: its structure decides how it runs. So `solver=` being a free choice is a property of simple models. For complex ones the useful features are that the vocabulary stays the same, that the parts which do have a compartmental form can be reused, and that the library says plainly what an ODE version would leave out.

### Agreement is an empirical question

Whether solvers agree is tested, and the landscape supplies the instruments: EMULSION's shipped feature models pushed across `compartment` and `IBM` (its review: "That boundary is the actual scope of the lingua franca's paradigm-independence claim"), epipack's three modes on the same tuples as a cheap regression baseline, and camdl's Richardson audit of convergence in `dt`, which distinguishes "it ran" from "it converged". An approximated run is tested differently: the test records how far it is from the agent-based run, without asserting that they are equal.

### Interconversion with other frameworks is a different thing

[MIRA](../approaches/notes.md#askem-model-representation-amr--mira) is the closest prior art for converting between frameworks: a closed transition taxonomy, ontology-identified concepts, and importers and exporters for many formats. Two lessons. From MIRA: "Nobody except MIRA can tell whether two models' compartments mean the same thing", so ontology annotation would help model merging, though requiring it would not (it is not in v1). From [SBML](../approaches/notes.md#sbml-systems-biology-markup-language): "an interchange format that nobody authors in becomes a lossy export target, and partial conformance is the normal state." So the language has to be pleasant to write models in first, and every other form is derived from it.

## The specification

[12-backends.md](../design/12-backends.md) specifies `solver='auto'` and what makes a model agent-based, the solvers (`ode`, `binomial`, `gillespie`, and `abm`), the standard approximations and how they are reported (`approx=`), what has no approximation, `solvers()`, the try-it rule for functions, and agreement between solvers, which [18-conformance.md](../design/18-conformance.md) tests.

## Trade-offs

- **An approximated run is a different model, and a one-line note is easy to miss.** A user may compare an ODE run of a network model with its agent-based run and be surprised that they differ. The note, `solvers()`, and `approx=False` are the protections, and the notes are kept with the sim (`sim.notes`, even at `verbose=0`) so that they survive into saved output.
- **Each standard approximation has to be chosen, implemented, and maintained.** Deriving partner-change rates and a mixing matrix from a sexual network is real work per network class. Networks without one fall into "not possible" until someone writes it, with a message that says so. The networks of partnerships (`ss.SexualNet`, `ssl.DegreeNet`) now have one, the partner-change model, which leaves out concurrency and reinfection within a partnership ([12-backends.md](../design/12-backends.md)).
- **`auto` staging a dwell time means the default run of such a model is not exactly the model as written.** The mean is kept and the variance is close, and the note gives both, but a result that depends on the tail of the distribution (a long incubation period in a few people) is not reproduced by a handful of stages. `solver='abm'` or `approx=False` runs it as written.
- **`auto` can change paradigm when the model changes slightly.** Adding a network or a per-agent function to an ODE model makes it agent-based, which changes run time and makes results stochastic. The printed reason names the ingredient that caused it.
- **The try-it rule finds problems at build time, not when the model is written.** A function that fails on groups is reported when the sim is initialized. That is later than a static check would be, and much simpler for the user: there is nothing to declare (a module written as code may still say `compartmental = True` or `False`, to skip the trial).
- **`solvers()` is a testing obligation.** Every "yes" is a claim. Most entries follow from the kind of ingredient and the solver, so the number of distinct cases is small.

## Rejected

- **Refusing to run when a standard approximation exists** (camdl's strict capability checks). There are plenty of valid ODE models of sexually transmitted infections; the library should produce one when asked.
- **Approximating the mixing or the individuals under `auto`.** The default never replaces a network with rates between groups, or people who differ with their average. That happens only when the user names a solver.
- **An `auto` that never approximates anything.** An earlier draft had this rule, and under it a single lognormal dwell time forced an agent-based run of an otherwise compartmental model: slower, noisier, and not what someone writing an SEIR expects. Erlang stages are the standard compartmental treatment of a dwell time, so `auto` uses them and says so.
- **Approximating without saying so.** The note is one line and can be silenced with `approx=True`, but it is on by default.
- **A menu of network approximations for the user to choose from.** EoN shows how many there are. One standard approximation per ingredient is enough; anyone who needs more accuracy uses agents.
- **A separate model per paradigm** (MEmilio, EpiModel).
- **Paradigm as a fixed property of every model** (EMULSION's `aggregation_type` written into the model file). For simple models it is a run-time choice.
- **Promising that any model runs any way.** For complex models the paradigm follows from the content.
- **Static analysis of user functions** to decide in advance where they can run. Try them.
- **An interchange format as the primary artifact.** SBML's history. The authoring language comes first; the model spec is derived.

## Open questions

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
