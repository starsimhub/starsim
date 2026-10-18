# Interventions

## Recommendation

1. **An intervention is *what* plus *delivery*.** What is a product (a vaccine, a test, a treatment) or a change to a quantity. Delivery is who gets it, when, and how many. The "what" separates cleanly; who, when, where, and how many often do not separate from each other, so delivery is either a few simple keywords or one function.
2. **Eligibility is written against the model's real attributes.** The general form is a function of the sim, as in Starsim today. The short form is the same expression written directly on the people and disease objects, which is evaluated later when the sim runs. No separate vocabulary of predicate names.
3. **Keep Starsim's product/delivery split**, and separate efficacy from waning as EMOD does.
4. **Many interventions are an `ss.transfer` between strata or an `ss.modify` on a named quantity.** Do not build a class for what is already one of those.
5. **The simple case stays a one-liner.** Covasim's `days`/`changes` form was simpler than what replaced it, and it comes back as `ss.change(..., at=, by=)`.
6. **Coverage as a fraction and capacity as a number are different things**, and both are needed: `coverage=0.7` and `capacity=50_000`.
7. **Interventions carry a cost.** Nothing else in the review except Atomica does this, and it is what the policy audience needs. With a cost, total cost is a result, and `ss.optimize()` can reallocate spending between programs (as the Atomica importer's program books do).
8. **Scenarios live in the model file**, not in a shell script, and a baseline contains whatever is really in place.

## Why

### The split that runs through the whole review

Finding 1 of the [landscape review](../approaches/README.md#cross-cutting-findings): "Structural rigor and intervention expressiveness have not yet appeared in the same framework." [camdl](../approaches/camdl.md), [odin](../approaches/odin.md), [summer2](../approaches/summer2.md), and [epipack](../approaches/epipack.md) have the best model specification and essentially no intervention vocabulary. [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [Starsim](../approaches/starsim.md), and [Atomica](../approaches/atomica.md) have the best intervention vocabulary and weak structural specification.

Closing that split is, per the review, "the most concrete contribution available to this project."

### What separates and what does not

[EMOD](../approaches/emod.md) decomposes a campaign event into four pieces:

| Part | Question | EMOD |
|---|---|---|
| Where | Spatial scope | `Nodeset_Config` |
| When, how many, how often | Schedule and coverage | `Event_Coordinator_Config` |
| To whom | Eligibility | `Targeting_Config` |
| What | The product | `Intervention_Config` + `Waning_Config` |

This is the most complete treatment in the review and it is a useful checklist. As a design it overstates how independent the pieces are. Consider a realistic policy: *distribute treatment drugs to the 100 clinics with the most positive cases, while stocks last.* Where, when, how many, and to whom are all functions of who tested positive. Only "what" (the drug) can be swapped without touching the rest.

So the lingua franca keeps two parts. Starsim already has this split (`ss.Product` and the `RoutineDelivery` / `CampaignDelivery` mixins), and its review says "the product/delivery split is the right decomposition and should survive into the design."

### Eligibility in existing frameworks

[Starsim](../approaches/starsim.md) takes eligibility as a function of the sim that returns who is eligible. That is fully general, works with any custom attribute, and is the form to keep. Its limits, per its review: "it is Python, so it cannot be serialized, diagrammed, or checked."

[EpiModel](../approaches/epimodel.md)'s tabular scenario grammar, `{scenario, param, at, value}`, covers a large fraction of real specifications without any code, and its limit is that "it can only set parameter values, so anything that targets a *subset of agents* (screen 40% of over-50s annually) has to fall back to procedural code." [flepiMoP](../approaches/flepimop.md)'s modifiers have the same shape and the same ceiling.

Two frameworks have eligibility as data. **EMOD:**

```json
"Targeting_Config": {
  "class": "TargetingLogic",
  "Logic": [ [ {"class": "HasIntervention", "Is_Equal_To": 0, "Intervention_Name": "MyVaccine"} ],
             [ {"class": "HasIP", "Is_Equal_To": 1, "IP_Key_Value": "Risk:HIGH"} ] ]
}
```

**EpiHiper** has set algebra over typed node and edge predicates, "the most expressive targeting mechanism in the review", at the cost of a separate rule language. And [`individual`](../approaches/individual.md) shows the same algebra in code, with bitsets:

```r
I <- health$get_index_of("I")
I$and(already_scheduled$not(inplace = TRUE))
```

The aim is to get eligibility that reads like the `individual` version, can be saved like the EMOD version when it is simple, and falls back to a plain function when it is not.

## The specification

[08-interventions.md](../design/08-interventions.md) specifies products (`ss.Vaccine`, `ss.Test`, `ss.Treatment`, with `efficacy=`, `waning=`, and `cost=`), the deliveries (`ss.vaccinate`, `ss.screen`, `ss.treat`) with their keywords or one `deliver=` function, follow-ups (`on_positive=`, `on_negative=`), event-triggered delivery (`on=`), `ss.change`, triggers (`when=`, and `hold=` on `ss.modify`), scenarios (`ss.parallel(baseline=..., vaccine=...)`), and how each solver realizes an intervention. Eligibility as conditions on the real attributes, and `ss.modify` for a change to a named quantity, are in [07-effects.md](../design/07-effects.md); vaccination as an `ss.transfer` between strata is in [05-dimensions.md](../design/05-dimensions.md).

## Trade-offs

- **Conditions that evaluate later are an implementation cost, and a possible source of confusion.** The same expression builds a condition before the sim is initialized and returns an array after. The rule is simple to state, but error messages have to be good, and `describe()` should show each condition as written.
- **A function cannot be saved or drawn.** Eligibility or delivery written as a function is printed by name in `describe()`, and the model spec (the normalized form behind `sim.hash()` and `ss.diff()`) records that a function was used, not what it does. This is accepted: the function is the general form and must always work.
- **A condition on a per-agent attribute has no compartmental meaning** unless the attribute is a stratification. `ppl.age > 65` works in both if the model is stratified by age with a boundary at 65; `ppl.clinic == 12` works only with agents. `solver='auto'` then chooses agents, and an explicit compartmental solver stops with `ss.CapabilityError`, naming the attribute.
- **`when=` is evaluated mid-run, on results so far.** The function reads a results array that is only filled up to the current step, so it has to index by `sim.ti` (a bare `[-7:]` would read the unfilled end of the array), and what it sees depends on where in the loop the modifier runs. It is kept because a stateless threshold is common and short. The commonest case, a threshold on recent incidence, is a condition with no indexing at all (`where=sir.infection.incidence(over=ss.days(7)) > 1e-3`), and `hold=` keeps the modifier on for a while once it holds; anything subtler belongs in a class.
- **`deliver=fn` hides structure.** A delivery function can do anything, so two interventions using one cannot be compared keyword by keyword. Use the keywords when they fit.
- **Products are a vocabulary that will grow.** Vaccines, tests, treatments, bednets, condoms, contact tracing. Most are an `ss.modify` on a named quantity with a coverage and an eligibility, and only the ones with distinct structure (a diagnostic returns a *result*) need a class.

## Rejected

- **Four independent parts** (EMOD's decomposition taken literally). Where, when, how many, and to whom depend on each other in real programs. What + delivery.
- **A vocabulary of global predicate names** (`ss.age > 65`, `ss.weekday == 'Fri'`). It reads well for the built-in attributes and has no answer for custom ones without something like Polars' `pl.col('name')`, and the list of names never stays small. The real attributes on the real objects need no list.
- **A class per combination of intervention type and delivery type.** Starsim 3's `routine_screening` / `campaign_screening` / `routine_triage` / `campaign_triage` / `routine_vx` / `campaign_vx` is two axes spelled out as classes (removed in Starsim 4). Keyword arguments, one function each.
- **Interventions as contact reductions only** (Epydemix, MEmilio). Covers COVID-era scenarios and little else.
- **Interventions as extra compartments written by hand** (Epydemix's answer for vaccination, MEmilio's `secirvvs`). This is what `ss.transfer` between strata replaces, and it is why vaccination need not touch the disease model.
- **`'no -> yes'` arrows for booleans.** `ss.transfer(vaccinated=True, ...)`.
- **`min_duration=` on `ss.modify`,** and any other trigger keyword that needs memory. State that persists between steps belongs where it can be seen and tested: in a `step()` method or an intervention class. (One such keyword was later added: `hold=`, the minimum time a modifier stays on once its condition holds, which MEmilio's dynamic NPIs need; any other memory is still a `step()`.)
- **A separate rule language** (EpiHiper's variables/sets/triggers/operations). The semantics are the best in the review; the syntax is 200 lines of JSON for one Friday.
- **Edge-level eligibility** (EpiHiper's edge predicates). Networks are enough: to change school contacts, modify the school network.
- **Interventions that are inactive unless a scenario enables them** (camdl). Baselines usually contain real interventions; each scenario lists what it contains.
- **Budget optimization in v1** (Atomica, PyRoss). Do not rule it out; do not build it yet. (Since built for programs with spending, `ss.optimize()`, to match Atomica; optimal control is still out.)

## Open questions

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
