# Paradigm conversion

Running one model text as an ODE, as a stochastic compartmental model, or with agents, and what happens when the model does not fit the method that was asked for.

## Recommendation

1. **One model text; `method=` selects how it runs.** For a simple model this is a free run-time choice. For a complex model the paradigm is effectively part of the specification: a model with an individual-level network or per-agent logic is an agent-based model.
2. **`method='auto'` picks the simplest, fastest representation that runs the model exactly**, and prints what it picked. A user may still ask for agents on a simple model, for example to study elimination.
3. **`auto` never approximates. An explicitly requested method uses the standard approximation where one exists, and says so in one line.** It does not refuse.
4. **Stop only when there is no sensible approximation**: individual contact tracing, transmission trees, custom code that only makes sense for agents.
5. **Functions work wherever the things they use exist.** A function that only uses state counts, parameters, time, and stratifications runs in an ODE as well as with agents. No static analysis: try it and report.
6. **`methods()` shows which run methods can run a model**, as a coarse table.
7. **Test agreement between methods on simple models.** For complex models, report what was approximated; do not promise equivalence.
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

### Simple models and complex models are different

A two-line SIR can be run any way, and choosing is a matter of the question being asked. A national HIV model with age-disparate partnerships, a care cascade, and contact tracing cannot: its structure decides how it runs. So `method=` being a free choice is a property of simple models. For complex ones the useful features are that the vocabulary stays the same, that the parts which do have a compartmental form can be reused, and that the library says plainly what an ODE version would leave out.

## The proposal

### One model, several runs

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.03),
    recovery  = ss.progression('I -> R', dur=10),
)
pool = ss.MixingPool(n_contacts=10)

ss.Sim(sir, networks=pool, total_pop=5000, unit='day').run()                 # auto: ODE
ss.Sim(sir, networks=pool, total_pop=5000, unit='day').run(method='ctmc')    # Gillespie
ss.Sim(sir, networks=pool, total_pop=5000, unit='day').run(method='abm')     # agents, same mixing
ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), n_agents=5000, unit='day').run()   # auto: agents
```

The disease is the same object in all four. The first three are the same model run three ways; `method='abm'` on a simple model is what you want when the question is about chance, such as whether an infection is eliminated. The fourth is a different model (contacts are individual pairs), and `auto` runs it with agents because a random network has no exact compartmental form.

Each run prints one line saying which method was used and why:

```text
method: ode (every part of this model has a compartmental form)
```

```text
method: abm (RandomNet is an individual-level network)
```

### The approximation policy

Three cases.

**`auto`: nothing is approximated.** If every piece has a compartmental form the model runs as an ODE, otherwise with agents.

**An explicit method the model does not natively fit: the standard approximation, with a one-line note.**

```python
sim = ss.Sim(hiv, networks=ss.MFNet(), unit='year')
sim.run(method='ode')
```

```text
method: ode
  note: MFNet approximated by partner-change rates between sex/risk groups
  note: 'acute' has a lognormal dwell time, approximated by 4 Erlang stages
```

**No sensible approximation: stop, and say what was asked for.**

```text
Cannot run 'hiv' with method='ode': the intervention 'partner_notification' traces each person's own contacts, and a compartmental model has groups, not individuals.
  Run with method='abm', or remove 'partner_notification' to run as an ODE.
```

`approx=False` makes any approximation an error, for users who want only exact runs. `verbose=0` hides the notes.

The standard approximations:

| Ingredient | Asked to run as | Standard approximation |
|---|---|---|
| Sexual network | ODE or stochastic compartmental | Partner-change rates between sex/risk/age groups (below) |
| Random or other contact network | ODE or stochastic compartmental | Homogeneous mixing at the network's mean contact rate, within any declared groups |
| Non-exponential, non-Erlang dwell time | ODE or stochastic compartmental | Erlang stages matching the mean and, as closely as possible, the variance; the difference is printed |
| Per-agent continuous attribute (`rel_sus` drawn from a distribution) | ODE or stochastic compartmental | Binned into groups, with the bins printed |
| Very large population | exact CTMC | Not substituted; `method='stochastic'` picks tau-leaping and says so ([stochastic-compartmental.md](stochastic-compartmental.md)) |

### Sexually transmitted infections as an ODE

This is the most common approximation in practice, so it is worth spelling out. An agent-based STI model has individual partnerships that form, last for a while, and dissolve. The compartmental version replaces them with rates:

- The population is divided into groups by sex, risk level, and (if stratified) age.
- Each group has a **partner-change rate**: new partners per person per year. The library computes it from the network (how often partnerships form and how long they last in each group).
- A **mixing matrix** says which groups those partners come from, between fully assortative (like with like) and proportionate (at random, in proportion to partnerships on offer). It is also computed from the network's pairing rules.
- The **transmission probability per partnership** is computed from the per-act probability, the acts per unit time, and the partnership duration.

The force of infection on a susceptible person in group *i* is then their partner-change rate, times the sum over partner groups *j* of the share of partners from *j*, times the prevalence in *j*, times the per-partnership transmission probability.

What this leaves out: that partnerships persist (a person in a long partnership with an uninfected partner is protected while it lasts), concurrency, and variation in partner numbers within a group. Those matter for some questions, especially early epidemic growth and partner-based interventions, and for those the model should be run with agents. For population-level projections over years, this approximation is the standard tool.

Compare **Atomica**, where the user writes the force of infection by hand as a string expression in the framework spreadsheet (`(1 - (1-ch_prev*transpercontact)**floor(contacts)*...)`, which its review calls the point "where a spreadsheet model stops being legible"), and **[EpiModel](../approaches/epimodel.md)**, where the deterministic and network versions are separate models with separate code (`dcm()` with a hand-written derivative function, `netsim()` with a fitted network). Here the network is declared once and the rates are derived from it.

A user who wants the compartmental STI model as the model, and not as an approximation of a network, writes it that way from the start with a stratification and a mixing pool between the groups ([population-and-mixing.md](population-and-mixing.md), [stratification.md](stratification.md)). Then `auto` runs it as an ODE and there is nothing to approximate.

### Functions in any method

Any slot that takes a number also takes a function of the sim. A function is tried on whatever the run method provides: agents in an agent-based run, groups (one row per stratum) in a compartmental run.

```python
def cautious(sim):
    """ People halve their contacts when many are infected """
    return 0.03 if sir.I.n < 500 else 0.015

sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=cautious),
    recovery  = ss.progression('I -> R', dur=10),
)
ss.Sim(sir, networks=ss.MixingPool(n_contacts=10), total_pop=5000, unit='day').run()   # auto: ODE
```

`cautious` uses a state count (`sir.I.n`, the number of people in `I`) and nothing else, so it works as an ODE, as a stochastic compartmental model, and with agents. The same holds for conditions: `eligible=lambda sim: sim.people.age > 65` works in a compartmental run if age is a stratification, where it selects the groups aged over 65.

A function that reaches for something only agents have stops the compartmental run with a message naming it:

```text
Cannot run 'sir' with method='ode': the function 'superspreaders' asked for sim.people.n_friends, which is a per-agent attribute and not a stratification.
  Run with method='abm', or stratify by n_friends.
```

Under `auto`, a function like that simply makes the model agent-based, with the reason printed. Fully custom code (a function passed to `ss.Disease`, or an overridden `step()`) follows the same rule.

EMULSION is the comparison: behavior its YAML cannot express goes into a separate Python add-on module, and its review notes that "any model using them loses the 'model is a document' property". Here a function is an ordinary part of the model, and the same function is tried in every method.

### `methods()`

Before running anything:

```python
hiv.methods()
```

```text
                                 ode                               stochastic                        abm
network: MFNet                   approx (partner-change rates)     approx (partner-change rates)     yes
dwell time: acute (lognormal)    approx (4 Erlang stages)          approx (4 Erlang stages)          yes
function: care_seeking           yes (uses age, sex)               yes (uses age, sex)               yes
partner_notification             no (traces individual contacts)   no (traces individual contacts)   yes

auto would choose: abm
```

One row per ingredient that matters (each network, each non-exponential dwell time, each custom function), one column per method, and entries of "yes", "approx (how)", or "no (why)". Ingredients that run everywhere are not listed. The stochastic compartmental methods share a column unless they differ. `sim.methods()` gives the same table for the whole sim. For functions the table is filled in by trying them, so it is produced at build time.

The table is coarse on purpose. It is meant to answer "can I run this as an ODE, and what would I lose?" in a few lines.

### What is exact, what is approximated, what is not possible

**Exact.** The same mathematics, run differently.

- Homogeneous or contact-matrix mixing: ODE, CTMC, chain-binomial, agents.
- Exponential dwell time: a rate, or a sampled time.
- Erlang dwell time (or any dwell time with `stages=k`): sub-compartments, or a sampled time.
- A stratification: a compartment axis, or an agent property.
- A function that only uses counts, parameters, time, and stratifications.

**Approximated, when explicitly asked for, with a note.** The table of standard approximations above.

**Not possible.** Stops with a message.

- Individual histories, contact tracing, transmission trees: agents only.
- Custom code or functions that use per-agent information that is not a stratification: agents only.

### Test the agreement

Whether methods agree is an empirical question, so it is a test suite, and the landscape supplies the instruments:

- **EMULSION's shipped feature models.** Its review proposes exactly this: "take the shipped feature models, push each one across `compartment` / `IBM` / `hybrid`, and record where the switch stops working and why. That boundary is the actual scope of the lingua franca's paradigm-independence claim, and someone has already built the apparatus for measuring it."
- **epipack's three modes** on the same tuples, as a cheap regression baseline.
- **camdl's Richardson dt-convergence audit**, to distinguish "it ran" from "it converged".

Concretely: for every simple model in [vignettes](../vignettes), run every method that `methods()` marks "yes" and assert agreement within a stated tolerance (the mean of the stochastic runs against the ODE, at a population large enough for the comparison to be fair). Runs marked "approx" are tested differently: the test records how far the approximation is from the agent-based result, so that the size of the difference is known and regressions are caught, without asserting that the two are equal.

For complex real-world models, exact equivalence between paradigms is neither provable nor the aim. What the library owes the user there is the list of what was approximated.

### Interconversion with other frameworks

This is distinct from paradigm conversion. [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) is the closest prior art: a `TemplateModel` with a closed transition taxonomy, ontology-identified Concepts, and importers/exporters for SBML, Petri-net AMR, RegNet AMR, StockFlow AMR, ACSets, bilayer, Vensim/Stella, and PySB. [MEmilio](../approaches/memilio.md) has an SBML importer, "a working instance of the exact artifact this project needs".

Two lessons. From MIRA: "Nobody except MIRA can tell whether two models' compartments mean the same thing", which is why optional ontology annotation is worth having ([model-structure.md](model-structure.md)) even though requiring it is not. From [SBML](../approaches/notes.md#sbml-systems-biology-markup-language): "an interchange format that nobody authors in becomes a lossy export target, and partial conformance is the normal state."

The consequence is the [principles.md](principles.md) rule restated: this has to be a pleasant language to write models in first. The model spec (JSON) is derived from it, not the other way around.

## Trade-offs

- **An approximated run is a different model, and a one-line note is easy to miss.** A user may compare an ODE run of a network model with its agent-based run and be surprised that they differ. The note, `methods()`, and `approx=False` are the protections; the note should also be stored with the results so that it survives into saved output.
- **Each standard approximation has to be chosen, implemented, and maintained.** Deriving partner-change rates and a mixing matrix from a sexual network is real work per network class. Networks without one fall into "not possible" until someone writes it, with a message that says so.
- **`auto` can change paradigm when the model changes slightly.** Adding a network or a per-agent function to an ODE model makes it agent-based, which changes run time and makes results stochastic. The printed reason names the ingredient that caused it.
- **The try-it rule finds problems at build time, not when the model is written.** A function that fails on groups is reported when the sim is initialized. That is later than a static check would be, and much simpler for the user: there is nothing to declare.
- **`methods()` is a testing obligation.** Every "yes" is a claim. Most entries follow from the kind of ingredient and the method, so the number of distinct cases is small.

## Rejected

- **Refusing to run when a standard approximation exists** (camdl's strict capability checks). There are plenty of valid ODE models of sexually transmitted infections; the library should produce one when asked.
- **Approximating under `auto`.** The default should never change the model. Approximation happens only when the user names a method.
- **Approximating without saying so.** The note is one line and can be hidden with `verbose=0`, but it is on by default.
- **A menu of network approximations for the user to choose from.** EoN shows how many there are. One standard approximation per ingredient is enough; anyone who needs more accuracy uses agents.
- **A separate model per paradigm** (MEmilio, EpiModel).
- **Paradigm as a fixed property of every model** (EMULSION's `aggregation_type` written into the model file). For simple models it is a run-time choice.
- **Promising that any model runs any way.** For complex models the paradigm follows from the content.
- **Static analysis of user functions** to decide in advance where they can run. Try them.
- **An interchange format as the primary artifact.** SBML's history. The authoring language comes first; the model spec is derived.

## Open questions

- **Mixing methods within one run (deferred, not in v1).** In plain words: sometimes most of a model is well described by smooth average behavior, but one part of it is a rare chance event. The standard example is a country that has nearly eliminated polio. Transmission inside a large population can be run as an ODE, but an imported case arrives perhaps a few times a year, and whether one arrives this month is a matter of chance that an ODE cannot represent (it would add a steady trickle of 0.01 cases per day). A hybrid run would treat that one transition as discrete random events and everything else as an ODE. [Catalyst.jl](../approaches/notes.md#modelingtoolkitjl--catalystjl) does this per reaction and [EMULSION](../approaches/emulsion.md) has a `'hybrid'` setting. If we add it, it would be a `method=` keyword on the transition, for example `ss.transmission('S -> I', beta=1e-6, method='ctmc')`. For now, the way to ask this kind of question is to run the whole model with a stochastic method or with agents.
- **Agents inside patches, patches as compartments, or the reverse.** A related idea from EMULSION and MEmilio: different levels of one model run in different ways (individual animals inside herds that are tracked only as counts). Also deferred. The existing way to combine the two is an `ss.Compartments` module inside an agent-based sim ([agent-based.md](agent-based.md)).
- **What tolerance counts as agreement for simple models?** It depends on population size and the number of stochastic runs. It needs to be stated per test, with the reasoning, so a failure means something.
- **Which approximation is "standard" for each network class?** Partner-change rates between groups is clear for sexual networks. For a static network with very uneven numbers of contacts, the mean contact rate may be a poor summary, and the note should probably say so.
- **Should `auto` ever pick a stochastic compartmental method?** As written it chooses between ODE and agents. A small population with a purely compartmental model is a case where the ODE is exact as mathematics and still a poor description; a printed hint may be enough.
