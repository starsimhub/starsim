# Stratification

## Recommendation

1. **One concept covers age, space, risk, strain, and vaccination status.** They are all dimensions; the difference is which structure they carry, not what kind of thing they are.
2. **Every dimension is an agent property; not every agent property is a dimension.** A property becomes a dimension when you stratify by it. Continuous properties (age, CD4 count) are binned for that purpose and stay continuous in an agent-based run.
3. **Stratification is an operator applied to a finished model.** `sir.stratify(age=[0, 15, 65])` stratifies `sir`; `inplace=False` returns a stratified copy and leaves the original untouched, so the two can be compared. A property of people (age, sex, place) is usually stratified once for the whole sim, `sim.stratify(age=[0, 15, 65])`; a dimension that belongs to one disease (strain, resistance) on that disease.
4. **The base model does not mention strata; the stratification supplies multiplicative adjustments.**
5. **Some dimensions carry structure.** Age implies aging (where people are born and die, or with `use_aging=True`). Space may have movement between patches, or none. Strain changes who can infect whom.
6. **Strain can be a dimension or a second disease.** Support both, and say when each fits.
7. **Check the restrictions that follow from the structure**, with errors that say what to do.

## Why

### Four frameworks, one operation

[MetaCast](../approaches/metacast.md) states the general case most plainly: stratification, metapopulation structure, and any other partition are the same operation — a dimension the model is replicated over. [summer2](../approaches/summer2.md) implements it as a transformation (`stratify_with`). [camdl](../approaches/camdl.md) has `dimensions { age = [child, adult] }` plus `stratify(by = age)`. [flepiMoP](../approaches/flepimop.md) declares compartments as a Cartesian product of named axes. [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) attaches a `context` dict to each Concept and operates on it with a generic `stratify()`.

And [Kendrick](../approaches/notes.md#kendrick), in 2015, made the same argument as the organizing principle for the whole field: an epidemiological model is a composition of independent concerns, combined mechanically rather than interleaved.

The other end of the spectrum is [odin](../approaches/odin.md), where stratification means array indices and "adding an age dimension to an odin model means rewriting every equation with indices". Its own review: "The lingua franca probably needs both — an operator for the common cases, explicit indexing underneath it — with the operator documented as sugar that expands to the indexed form."

### Why the operator needs a closed transition vocabulary

From the [summer2 review](../approaches/summer2.md):

> **A closed vocabulary of named flow kinds.** … An open "rate expression" cannot be stratified automatically; a named flow kind can.

This is the connection between this document and [model-structure.md](model-structure.md). The library can split a transition correctly only if it knows what the transition is: where it starts, where it ends, and whether its rate involves an infectious state. An `ss.Transition` carries exactly that, whichever way it was written (`S >> I @ beta * I`, or in full as `ss.Transition(S, I, rate=beta * I)`). Custom code in a function or a `step()` method is not split automatically: in a stratified compartmental run it is called on the groups, and works if it only uses things that groups have.

### Agent properties and compartment axes are the same declaration

[EMOD](../approaches/emod.md)'s Individual Properties are user-defined categorical agent attributes (`Risk: HIGH/MEDIUM/LOW`) declared in demographics and targetable by interventions. [EpiHiper](../approaches/epihiper.md)'s traits are the same. [HPVsim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) added genotype "as a state dimension rather than by duplicating the disease module". [MetaCast](../approaches/metacast.md)'s coordinates and [camdl](../approaches/camdl.md)'s strata are the compartmental version.

From the [EMOD implications](../approaches/emod.md): "This is the agent-based counterpart to a stratification dimension, and the lingua franca should treat them as the same concept expressed at different aggregation levels."

Nobody currently does.

The relationship runs one way. Agents have many properties (a unique ID, an exact age, a household, a CD4 count), and most are never dimensions. A property is a dimension only when the model is stratified by it, and for a continuous property that means choosing bins.

### Some dimensions carry structure

summer2 special-cases `AgeStratification` (which generates aging flows) and `StrainStratification` (which restructures the force of infection). camdl distinguishes population strata from *residence structure* (`via erlang`). [LASER](../approaches/laser.md) puts `node_id` in the core rather than treating space as one dimension among many. All of them recognize the same thing: **dimensions are not interchangeable.** An age dimension implies aging between its groups, a place implies separate populations with movement between them only if a mobility model is given, and a strain means that strain A infects with strain A. An age dimension that generated no aging flows where people are born and die would be wrong, and asking the user to write them by hand invites an error the library can prevent. Which structure a dimension carries is taken from its value (`ss.geo` or `mobility=` makes a place, `immunity=` a strain) or given with `kind=`, and from its name only for `age`; a guess never reinterprets what was given ([principles.md](principles.md#guessing-defaults)).

### A dimension or a second disease

A strain is a dimension when the strains share one natural history and differ in a few numbers, as drug-resistant TB does (the same states and progression, different treatment outcomes, and a flow from one strain to the other); [HPVsim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) added genotype this way, "as a state dimension rather than by duplicating the disease module", and summer2's `StrainStratification` is the compartmental version. It is a second disease when the pathogens are different enough that sharing a structure would be forced (different states, routes, or timescales), with the interaction between the two written with `ss.modify`, which is how Starsim handles co-circulating diseases. The rule of thumb: many similar strains with one structure are a dimension; two things you would draw as two diagrams are two diseases.

### Restrictions follow from the structure

summer2 enforces that a mixing matrix requires a *complete* stratification, and that age and strain stratifications are applied once each; its review is right that "a mixing matrix requires a complete stratification is a correctness property, not an implementation limit". camdl's errors are the model for how to deliver these: E238 rejects a `count` on a bare stratified transfer "because it would multiply the intended total by the number of cells", and E239 rejects a bare endpoint inside an indexed family "because realized coverage would be `1 − (1 − f)^P` rather than `f`". From its review: "Each of these is a real modeling bug that other frameworks silently commit." Since every state is stratified by default, most such restrictions can only be broken when the user narrows a stratification, and then the error says what to do.

### Adjustments are multiplicative

summer2 has `Multiply` and `Overwrite`. Its own implications section asks "consider whether `Overwrite` should exist at all, given how it interacts with multiple stratifications", and the answer is no. Multiplicative adjustments commute, so the result does not depend on the order of the stratifications; overwrites do not, so it does. Anyone who needs an overwrite is describing a different transition, and should write one.

## The specification

[05-dimensions.md](../design/05-dimensions.md) specifies `stratify` on a model or on the sim (in place by default, `inplace=False` for a copy), its keywords (`split=`, `adjust=`, `states=`, `where=`, `kind=`, `mobility=`, `immunity=`), the kinds of dimension, `ss.transfer`, strains carried together, the agent-based and compartmental readings of one declaration, and reading stratified results (`select()`, `by()`). Values by stratum are `ss.by` ([04-parameters.md](../design/04-parameters.md)), and places and mobility are in [06-routes.md](../design/06-routes.md).

## Trade-offs

- **Stratification is order-dependent even with multiplicative adjustments**, because the *set of transitions* being adjusted changes as strata are added. summer2 enforces ordering constraints "but they are not derivable from the model text — you have to know them". `describe()` prints the resolved stratified model, and combinations that are wrong are errors.
- **Stratifying in place loses the unstratified model** unless a copy was asked for with `inplace=False`. In place is the default because it is the usual case, and a line like `sir.stratify(age=...)` that silently did nothing to `sir` would be the more common mistake.
- **Two ways to do strains is one more decision for the user.** The rule of thumb above covers most cases; the two give the same answer where they overlap.
- **The operator only supports what it implements.** odin's arrays are maximally general and maximally laborious. Anything it does not cover can be written as explicit per-group transitions; `describe()` shows a hand-written transition alongside the generated ones.
- **Strata multiply.** Age (16) × risk (2) × strain (3) × patch (47) is 4,512 compartments, and someone will do it by accident. `describe()` lists every dimension with its levels, and an agent-based run does not grow this way, which is a good reason to switch.
- **Two places to stratify is one more decision.** The sim for people properties, the disease for its own dimensions. Stratifying a disease by age in a one-disease sim does the same thing as stratifying the sim, so the choice only matters when there are several modules.
- **A continuous property in an agent-based run and its binned version in a compartmental run are not the same model.** Binning is an approximation ([paradigm-conversion.md](paradigm-conversion.md)).

## Rejected

- **Manual array indexing as the only mechanism** (odin). General, laborious, and it makes the model unstratifiable by any tool.
- **`Overwrite` adjustments** (summer2), for the reason above.
- **Stratification by writing out the compartments** ([epipack](../approaches/epipack.md), [Epydemix](../approaches/epydemix.md)). "Age structure means writing out `S_child`, `S_adult` and every process between them by hand."
- **A separate metapopulation concept.** Space is a dimension, with or without movement, not a different kind of model. See [metapopulation.md](metapopulation.md).
- **Always returning a new model.** An earlier draft did, following NumPy. Pandas' `inplace` keyword covers both uses, and in place is the common one.
- **Every agent property as a dimension.** Dimensions multiply compartments; properties do not have to.
- **`'no -> yes'` arrows for boolean dimensions.** `vaccinated=True` says the same thing.
- **Ontology-identified strata as a requirement** (MIRA's `context` keyed to ontologies). Optional.

## Open questions

The open questions from this document are in [19-open-questions.md](../design/19-open-questions.md).
