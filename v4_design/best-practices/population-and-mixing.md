# Population and mixing

Finding 4 of the [landscape review](../approaches/README.md#cross-cutting-findings) calls this "the hardest unsolved problem in this review, and where the project has to do original work." This document proposes the solution.

## Recommendation

1. **Factor transmission into two facts: how infectious the pathogen is, and how contact happens.** The disease states the first; the networks state the second. Where infectiousness itself depends on the route (HIV through sex or through shared needles, male to female or female to male), the disease gives one `beta` per network.
2. **Keep Starsim's route abstraction and its keyword.** `networks=` accepts individual-level networks and mixing pools alike.
3. **Homogeneous mixing is the default.** With no network given, `S >> I @ beta * I` is the textbook `beta * S * I / N`, and `beta` is the classical transmission rate. The user does not write `/N`.
4. **A bare state in a rate means contact with people in that state, through the model's networks.** `I.n` means the number of people in it. The first covers person-to-person transmission; the second covers everything else.
5. **Make the denominator convention visible**, because frequency- and density-dependent transmission are different models and the difference is invisible in most frameworks. A bare state is frequency-dependent (`beta * I`); density dependence is written with the count (`beta * I.n`), and `describe()` says which a rate is.
6. **Let structure be specified by its generating model, not only its realization.** "Mean degree 0.7, partnerships lasting 50 days" is what modelers have; an edge list is not.
7. **Keep partnership and act distinct.** Duration of relationship and acts per unit time have different data sources.
8. **A network model can be run as a compartmental model when the user asks for it**, using the standard approximation and saying so: a random network as homogeneous mixing with its mean number of contacts, a static network by its mean degree. A network of lasting partnerships (`ss.MFNet`, `ss.PartnerNet`) has no compartmental approximation yet, and runs as agents.

## Why: three independent arrivals at route abstraction

[Starsim](../approaches/starsim.md) puts `ss.Network` (edge-based) and `ss.MixingPool` (aggregate, contact-matrix) behind one `ss.Route`, so "the *same disease module* can be run as an individual-network ABM or as a mixing-matrix metapopulation model without editing the disease". [epiworld](../approaches/epiworld.md) has `ModelSEIR` / `ModelSEIRCONN` / `ModelSEIRMixing` over one engine. [epipack](../approaches/epipack.md) has network versus well-mixed.

Three independent arrivals is strong evidence that this is the right place to divide things. Starsim's version is the most developed and is the starting point.

But every version is partial, and the reviews say where:

- **Starsim**: "the disease module is still written as agent code, so the compartmental limit is reached by making the population homogeneous rather than by solving equations… paradigm conversion is therefore one-directional."
- **epipack**: "`StochasticEpiModel` uses a different process vocabulary from `EpiModel`. The network paradigm is adjacent to the others, not the same specification reinterpreted." Its review is explicit that this is because "edge-mediated transmission genuinely is different from mass-action — the rate is per-edge rather than per-population-pair."

### Infectiousness and contact are already separated in one framework

[EpiModel](../approaches/epimodel.md) separates infectiousness from contact, and its review flags this as a strength worth keeping: `inf.prob` is a per-act transmission probability and `act.rate` is acts per partnership per time step, "an unusually explicit separation of *relationship* from *exposure*", and "collapsing them into a single 'contact rate' is lossy for anything sexually transmitted". Number of partners, partnership duration, and acts per partnership have three different data sources.

### The denominator is the most commonly mis-written line

[MetaCast](../approaches/metacast.md) computes the force of infection for you, which is the right instinct, "implemented in the most fragile possible way, through string concatenation of parameter names (`'beta_[high,vaccinated]'`)"; its `foi_population_focus` option selects the denominator, a real modeling decision that almost nothing else in the review exposes. [EMULSION](../approaches/emulsion.md) makes the modeler write it (`'transmission_I * total_I / total_population'`), and "nothing checks it, and nothing knows that this transition is an infection rather than a progression. That is the same missing-`/N` class of bug camdl catches." [summer2](../approaches/summer2.md) makes it a choice of function name, `add_infection_frequency_flow` versus `add_infection_density_flow`, "a coarse but effective form of typing". Here the frequency-dependent form is what a bare state means (`beta * I`, with the denominator supplied by the mixing), and the density-dependent form is written with the count (`beta * I.n`), which is also what the importers write for a source's `beta * S * I`. `describe()` prints the resolved rate with its denominator, and a rate per person that grows with a count of people is noted as density-dependent, because a model that silently has the other form is wrong in a way that only shows up when the population size changes.

### Structure is described by its generating model

This is finding 11 of the [landscape review](../approaches/README.md#cross-cutting-findings). [EpiModel](../approaches/epimodel.md) fits a network from target statistics (`~edges + concurrent + degrange(from = 4)`, a partnership duration) and treats checking that the simulated network recovers them as mandatory. [LASER](../approaches/laser.md) ships `gravity`, `competing_destinations`, `stouffer`, and `radiation`, "the best spatial-interaction vocabulary in the review". EoN, epiworld, and EpiHiper take a network as an input file and have no specification vocabulary at all. From the EpiModel review: "A lingua franca that only accepts an edgelist cannot express 'a network where mean degree is 0.7 and partnerships last 50 days' — and that description, not the edgelist, is what modelers actually have." "0.7 partners" on its own is also ambiguous (one partner 70% of the time, or a Poisson number with mean 0.7), which is why the number of contacts takes a distribution.

## The specification

[06-routes.md](../design/06-routes.md) specifies the force of infection, `networks=` (networks and mixing pools), the bare state and `.n` in a rate, a `beta` per network and direction, counts used literally (environmental and vector-borne transmission), `ss.PartnerNet` and `ss.MFNet` described by targets (`n_contacts`, `concurrency`, `max_contacts`, `match=`, duration, `acts=`) with `net.check()`, contact matrices (`ss.contacts`), and layers. States that are not people are in [03-states-and-transitions.md](../design/03-states-and-transitions.md), and the compartmental forms of networks, with the notes they print, in [12-backends.md](../design/12-backends.md).

## Trade-offs

- **`beta` means "per contact" once a network is given, and the contact rate is then a second number.** Incidence data identifies only their product. This is not a new problem; it is the existing one made visible, and [calibration.md](calibration.md) covers fitting it.
- **`I` and `I.n` look similar and mean different things.** With no network they differ only by the `/N`; with one, the bare state is something only the library can compute. `describe()` prints the resolved rate for every transition, including the denominator, so the difference is visible. A longer spelling for the bare state (`I.contact`, say) would be clearer and would stop `S >> I @ beta * I` reading like the textbook equation.
- **An approximated network model is a different model.** The one-line note says what was done; it does not say how much the answer changed. Running both and comparing is the only way to know.
- **A network library is a maintenance surface.** Every named generator (gravity, radiation, contact matrices for 180 countries) is code and data we own. That is accepted: the generators live in the library and the data in [`starsim_data`](https://github.com/starsimhub/starsim_data), which bundles what is small and common (national demographics, the Prem and Mistry contact matrices) and fetches the rest on demand with a cache. We do not depend on other frameworks' packages (LASER's migration module, `epydemix-data`, `socialmixr`) for them: an outside dependency is harder to keep working than code we control, and adapting a data set once, with its provenance recorded, is cheap.
- **A contact matrix and a network are not interchangeable even when both are available.** The matrix is the mean of a distribution whose variance matters for threshold behavior.

## Rejected

- **Making the modeler write the denominator** (EMULSION, MetaCast's subpop model, [`individual`](../approaches/individual.md)'s `foi <- beta * I / N`, [odin](../approaches/odin.md)). It is the most commonly mis-written line in epidemiological modeling. Writing it out stays possible with `I.n`.
- **A separate `ss.Homogeneous` class.** It is the default, and `ss.MixingPool` already exists.
- **A new keyword for mixing.** Starsim's `networks=` takes networks and mixing pools today.
- **Special route kinds for environmental and vector-borne transmission.** Count-based rates and ordinary states that are not people cover them with the ordinary transition syntax.
- **A degree-grouped ODE approximation for networks with strong degree heterogeneity.** Homogeneous mixing with the same mean is a poor approximation for these, and the standard improvement groups people by degree. It is not worth building: the agent-based run is one keyword away and is exact.
- **Refusing to run a network model as an ODE.** The approximation is standard and widely published; it is used when asked for and reported. (For a network of lasting partnerships it is not built yet, so such a model is refused with a `CapabilityError` until it is.)
- **Frequency versus density as different transition kinds** (summer2). It is written in the rate itself (a bare state, or its count with `.n`), so no transition needs a kind for it.
- **Networks as inputs only** (EoN, epiworld, EpiHiper, LASER). Accept an edge list; do not require one.
- **A separate transmission vocabulary for networks** (epipack's `set_link_transmission_processes` versus `set_processes`).
- **Node-level force of infection as the only spatial mechanism** (LASER). Correct for eradication-scale spatial models and not general.

## Open questions

The open questions from this document are in [19-open-questions.md](../design/19-open-questions.md).
