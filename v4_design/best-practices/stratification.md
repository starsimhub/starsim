# Stratification

## Recommendation

1. **One concept covers age, space, risk, strain, and vaccination status.** They are all dimensions; the difference is which structure they carry, not what kind of thing they are.
2. **Every dimension is an agent property; not every agent property is a dimension.** A property becomes a dimension when you stratify by it. Continuous properties (age, CD4 count) are binned for that purpose and stay continuous in an agent-based run.
3. **Stratification is an operator applied to a finished model.** `sir.stratify(age=[0, 15, 65])` stratifies `sir`; `inplace=False` returns a stratified copy and leaves the original untouched, so the two can be compared. A property of people (age, sex, place) is usually stratified once for the whole sim, `sim.stratify(age=[0, 15, 65])`; a dimension that belongs to one disease (strain, resistance) on that disease.
4. **The base model does not mention strata; the stratification supplies multiplicative adjustments.**
5. **Some dimensions carry structure.** Age implies aging. Space may have movement between patches, or none. Strain changes who can infect whom.
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

This is the connection between this document and [model-structure.md](model-structure.md). The library can split a transition correctly only if it knows what the transition is: where it starts, where it ends, and whether its rate involves an infectious state. An `ss.Transition` carries exactly that, whichever way it was written (`ss.transmission('S -> I', ...)`, `S >> I @ beta * I`, or in full). Custom code in a function or a `step()` method is not split automatically: in a stratified compartmental run it is called on the groups, and works if it only uses things that groups have.

### Agent properties and compartment axes are the same declaration

[EMOD](../approaches/emod.md)'s Individual Properties are user-defined categorical agent attributes (`Risk: HIGH/MEDIUM/LOW`) declared in demographics and targetable by interventions. [EpiHiper](../approaches/epihiper.md)'s traits are the same. [HPVsim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) added genotype "as a state dimension rather than by duplicating the disease module". [MetaCast](../approaches/metacast.md)'s coordinates and [camdl](../approaches/camdl.md)'s strata are the compartmental version.

From the [EMOD implications](../approaches/emod.md): "This is the agent-based counterpart to a stratification dimension, and the lingua franca should treat them as the same concept expressed at different aggregation levels."

Nobody currently does.

The relationship runs one way. Agents have many properties (a unique ID, an exact age, a household, a CD4 count), and most are never dimensions. A property is a dimension only when the model is stratified by it, and for a continuous property that means choosing bins.

## The proposal

### The operator

**summer2** ([approaches/summer2.md](../approaches/summer2.md)):

```python
age_strat = Stratification("age", agegroup_keys, ["S", "I", "R"])
age_strat.set_population_split({...})
age_strat.set_flow_adjustments("recovery", rec_adj)
age_strat.add_infectiousness_adjustments("I", {...})
age_strat.set_mixing_matrix(mm)
m.stratify_with(age_strat)
```

**camdl:**

```camdl
dimensions { age = [child, adult] }
stratify(by = age)
```

**Lingua franca**: one call, one keyword per adjustment:

```python
sir.stratify(
    age    = [0, 15, 65],                       # bin edges, or a list of labels
    split  = [0.3, 0.55, 0.15],                 # population split; defaults to the demography
    adjust = {'recovery': [1.5, 1.0, 0.5]},     # per-group multipliers on the hazard of named transitions (1.5 = faster recovery, even if given as dur=)
)
sim = ss.Sim(sir, networks=ss.contacts('Kenya'), unit='day').run()   # who meets whom, by age
```

`stratify` changes the model it is called on, which is what is wanted nearly every time, and returns it, so the call can also go straight into `ss.Sim(sir.stratify(age=[0, 15, 65]))`. Stratifying twice by the same dimension is an error, which is what a re-run notebook cell will hit. To keep the unstratified model for a side-by-side comparison, ask for a copy:

```python
sir_age = sir.stratify(age=[0, 15, 65], inplace=False)     # sir is unchanged
ss.parallel(plain=ss.Sim(sir, unit='day'), by_age=ss.Sim(sir_age, networks=ss.contacts('Kenya'), unit='day'))
```

`ss.stratify(sir, age=[0, 15, 65])` is the function form of the same call.

A stratified result is still one result, with a dimension for each stratification. It is the total until a group is selected, so nobody sums `I_0_female_low` through `I_55_male_high` by hand ([results-and-observation.md](results-and-observation.md)):

```python
res = sim.results.sir
res.n_I                                # everyone in I
res.n_I.select(age=0)                  # the youngest group
res.n_I.select(sex='female', risk='high')
res.n_I.by('age')                      # one series per age group
res.infection.select(age=0)            # "incidence in children"
```

The base model says "recovery takes `dur_inf`"; the stratification says "×1.5 for children". Neither knows about the other. That factoring is summer2's, and it is the reason one model text serves many structures.

Stratifications compose, each applied to the result of the last:

```python
sir.stratify(age=[0, 15, 65])
sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
```

### Stratifying the sim or the disease

Age, sex, and place are properties of people, and every disease in a sim has to agree on them: a contact matrix by age makes no sense if two diseases use different age groups. So a people property is normally stratified once, on the sim, and every module gets it:

```python
sim = ss.Sim([flu, rsv], networks=ss.contacts('Kenya'), unit='day')
sim.stratify(age=[0, 5, 18, 65])       # or ss.Sim(..., stratify=dict(age=[0, 5, 18, 65]))
```

Strain, drug resistance, and vaccination status against one pathogen belong to one disease, and are stratified there, as are per-disease adjustments (`flu.stratify(age=..., adjust={...})` on a sim already stratified by the same age groups just adds the adjustments). Two diseases in one sim stratified by the same people property with different groups is an error that suggests stratifying the sim.

A contact matrix whose age groups are not the model's is handled by the library: aggregated to the model's groups in a compartmental run, used as it is by agents ([population-and-mixing.md](population-and-mixing.md#layers)).

### Stratifying part of the model

By default every state is split, in every group. Two keywords narrow that:

```python
hiv.stratify(cd4=[0, 200, 350, 500], states=[I])               # only the infected have a CD4 stage
sir.stratify(sex=['female', 'male'])
sir.stratify(age=[15, 25, 35, 50], where=dict(sex='female'))   # age groups for women only
```

`states=` restricts which states are split, and `where=` restricts which existing groups are. The second gives a ragged model (four age groups of women, one group of men), and selecting an age group then selects women only. In an agent-based run neither keyword changes much, since every agent has an age and a sex regardless; they matter for the number of compartments in a compartmental run.

### Dimensions that carry structure

summer2 special-cases `AgeStratification` (generates aging flows) and `StrainStratification` (restructures the force of infection). camdl distinguishes population strata from *residence structure* (`via erlang`). [LASER](../approaches/laser.md) puts `node_id` in the core rather than treating space as one dimension among many.

Both summer2 and camdl are recognizing the same thing: **dimensions are not interchangeable.** The kind is taken from the name where possible, because `age` is always age:

| Dimension | What it implies | Declared as |
|---|---|---|
| `age` | Aging flows between adjacent groups, at the right rate | `sir.stratify(age=[0, 15, 65])` |
| `patch` (space) | Separate populations. Movement between them only if a mobility model is given | `sim.stratify(patch=['north', 'south'])`, plus `mobility=ss.gravity(...)` on the same call if people move |
| `strain` | Infection is strain-specific: strain A infects with strain A | `flu.stratify(strain=['H1N1', 'H3N2'])` |
| anything else | Nothing | `sir.stratify(risk=['low', 'high'])` |

An `age` dimension that generated no aging flows would be wrong, and asking the user to write them by hand invites an error the library can prevent.

Space is a valid dimension with or without movement. Patches that share nothing but parameters, or that are linked through a shared environment and not through travel (wastewater catchments, environmentally driven malaria), are ordinary models and produce no warning. See [metapopulation.md](metapopulation.md).

### Strain: a dimension or a second disease

Both are supported, because they fit different situations.

**A dimension**, when the strains share one natural history and differ in a few numbers. Drug-resistant TB is the standard case: the same states, the same progression, different treatment outcomes, and a flow from one strain to the other.

```python
tb.stratify(
    resistance = ['DS', 'MDR'],
    adjust     = {'treatment_success': [1.0, 0.6]},
    acquired   = ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment']),
)
```

`states=['on_treatment']` restricts the transfer to people on treatment, which is where resistance is acquired. Everyone else keeps their strain. [HPVsim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) added genotype this way, "as a state dimension rather than by duplicating the disease module", and summer2's `StrainStratification` is the compartmental version.

**A second disease**, when the pathogens are different enough that sharing a structure would be forced (different states, different routes, different timescales). Each is its own `ss.Disease`, and the interaction between them is written with `ss.modify`:

```python
sim = ss.Sim(
    diseases   = [wild, variant],
    connectors = ss.modify(variant.rel_sus, multiply=0.3, where=wild.R),   # partial cross-immunity
    unit       = 'day',
)
```

This is how Starsim handles co-circulating diseases today (see [composition-and-effects.md](composition-and-effects.md)).

The rule of thumb: many similar strains, one structure: a dimension. Two things you would draw as two diagrams: two diseases.

### Moving between coordinates

`ss.transfer` from [model-structure.md](model-structure.md) exists for this. **MetaCast** ([approaches/metacast.md](../approaches/metacast.md)):

```python
{"from_coordinates": ["high", "unvaccinated"],
 "to_coordinates":   ["high", "vaccination_lag"],
 "states":           "all",
 "parameter":        "nu_unvaccinated"}
```

**Lingua franca:**

```python
ss.transfer(risk='low -> high', rate=ss.peryear(0.1))     # between named groups: an arrow
ss.transfer(vaccinated=True, rate=ss.peryear(0.2))        # a yes/no dimension: the target value
```

A boolean dimension takes the value people move to, not an arrow between `'no'` and `'yes'`. `states=[...]` restricts which disease states move, as in the TB example above.

The individual does not change disease state; they change group, for all disease states at once. This is what vaccination, aging, and migration *are*, and it means the disease model never mentions vaccination. See [interventions.md](interventions.md).

### The agent-based and compartmental readings

The same declaration, realized two ways:

```python
sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
```

| Solver | What it does |
|---|---|
| Compartmental (`'ode'`, `'gillespie'`, ...) | Splits `S`, `I`, `R` into `S_low`, `S_high`, … and splits every transition, applying the adjustment |
| Agent-based | Adds a per-agent `risk` property with the given split, and scales `rel_sus` by the adjustment |

Nothing in the model text changes. This is why an EMOD Individual Property and a camdl stratum are the same declaration. The split compartments are internal: results are read by dimension (`res.n_I.select(risk='high')`), in the same way under both solvers, and never by a compartment's generated name.

In an agent-based run a continuous property stays continuous. The bins are used only where something needs groups: a compartmental run, a per-group parameter, or a result selected by group.

```python
hiv.stratify(cd4=[0, 200, 350, 500])   # agents keep their exact CD4 count
```

### Restrictions, checked and explained

summer2 enforces: a mixing matrix requires a *complete* stratification; `AgeStratification` may be applied once; `StrainStratification` may be applied once; strains cannot carry a mixing matrix. Its review is right that "a mixing matrix requires a complete stratification is a correctness property, not an implementation limit".

camdl's errors are the model for how to deliver these. E238 rejects a `count` on a bare stratified transfer "because it would multiply the intended total by the number of cells"; E239 rejects a bare endpoint inside an indexed family "because realized coverage would be `1 − (1 − f)^P` rather than `f`". From its review: "Each of these is a real modeling bug that other frameworks silently commit."

All states are stratified by default, so most of these cannot arise unless the user narrowed the stratification with `states=`. When they did:

```text
>>> sir.stratify(age=[0, 15, 65], states=[S, I])
StratifyError: contact matrix by 'age' given, but 'age' does not cover every infectious state.
  'Ia' is not stratified by age, so there is no age for the people in 'Ia' to mix as.
  Fix: leave out states= to stratify every state, or use a network that does not depend on age.
```

### Adjustments are multiplicative only

summer2 has `Multiply` and `Overwrite`. Its own implications section asks "consider whether `Overwrite` should exist at all, given how it interacts with multiple stratifications" — and the answer is no. Multiplicative adjustments commute, so the result does not depend on stratification order; overwrites do not, so it does. Anyone who needs an overwrite is describing a different transition, and should write one.

## Trade-offs

- **Stratification is order-dependent even with multiplicative adjustments**, because the *set of transitions* being adjusted changes as strata are added. summer2 enforces ordering constraints "but they are not derivable from the model text — you have to know them". `describe()` prints the resolved stratified model, and combinations that are wrong are errors.
- **Stratifying in place loses the unstratified model** unless a copy was asked for with `inplace=False`. In place is the default because it is the usual case, and a line like `sir.stratify(age=...)` that silently did nothing to `sir` would be the more common mistake.
- **Two ways to do strains is one more decision for the user.** The rule of thumb above covers most cases; the two give the same answer where they overlap.
- **The operator only supports what it implements.** odin's arrays are maximally general and maximally laborious. Anything it does not cover can be written as explicit per-group transitions; `describe()` shows a hand-written transition alongside the generated ones.
- **Strata multiply.** Age (16) × risk (2) × strain (3) × patch (47) is 4,512 compartments, and someone will do it by accident. Print the compartment count at build time, and note that an agent-based run does not grow this way, which is a good reason to switch.
- **Two places to stratify is one more decision.** The sim for people properties, the disease for its own dimensions. Stratifying a disease by age in a one-disease sim does the same thing as stratifying the sim, so the choice only matters when there are several modules.
- **A continuous property in an agent-based run and its binned version in a compartmental run are not the same model.** Binning is an approximation, and the run says so in one line ([paradigm-conversion.md](paradigm-conversion.md)).

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

- **Marking a dimension as strain-like.** A dimension named `strain` passes on the infector's strain. `resistance` in the TB example needs the same behavior, and the name alone does not say so. Either a short list of recognized names or a keyword on `stratify` is needed.
- **Cross-immunity between strains of one dimension.** With two diseases it is an `ss.modify` between them. Within a strain dimension it is an interaction between groups of the same dimension, which needs a spelling (probably a matrix keyword on `stratify`).
- **Where does per-agent continuous heterogeneity sit?** `rel_sus` drawn from `ss.lognorm_ex(mean=1, std=0.3)` is an agent property and not a dimension. A compartmental run can represent it only by binning it into a dimension or by using the mean. Under `solver='auto'` such a model runs as agents. An explicit `solver='ode'` uses the mean and says so; binning it into a dimension (`sir.stratify(rel_sus=[...])`) is the way to keep the heterogeneity in a compartmental run.
- **Partial stratification of transitions**, not just states: camdl's `only = [R]`. Does the operator need it, or is it a sign the model wanted two transitions?
- **Ragged stratifications.** `where=` makes groups that exist for some people and not others. Selecting, plotting, and exporting such results (is the age of a man missing, or "all ages"?) needs working through on a real model before the keyword is promised.
- **States that are not people.** A water reservoir is split by `patch` and not by `age`; mosquito states are split by species with `states=`. The rule proposed is that people dimensions skip states that are not people ([model-structure.md](model-structure.md#transmission-that-is-not-person-to-person)), and it needs checking on a malaria model.
