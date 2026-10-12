# Dimensions and stratification

Age, space, risk, strain, and vaccination status are handled by one operation: take a finished model and stratify it by a property of the people in it.

Justified by [best-practices/stratification.md](../best-practices/stratification.md).

## Dimensions are agent properties

Every stratification dimension is a property of a person (age, sex, patch, risk group, CD4 count). Not every property is a dimension: a property becomes one when a model is stratified by it.

- In an agent-based run, the property is a per-agent array on `sim.people`, exactly as in Starsim. If it already exists (`age`, `female`, a float state such as `hiv.cd4`), stratifying uses it. If it does not (`risk`), stratifying creates it and assigns values using `split`.
- In a compartmental run, the property becomes an axis: every state it applies to is split into one compartment per level.
- A continuous property is binned for the purpose of stratifying. In an agent-based run it stays continuous, and the bins are used only for adjustments, group-level conditions, and results.

A dimension name refers to one property across the whole sim, so two modules stratified by `patch` share the same patches. Each module may bin a continuous property its own way.

## `stratify`

```python
sir_age = sir.stratify(age=[0, 15, 65])        # method
sir_age = ss.stratify(sir, age=[0, 15, 65])    # function; identical
```

Both forms **return a new, stratified model and leave the original untouched**, so the stratified and unstratified versions can be run side by side. Calls chain, each one stratifying the result of the one before:

```python
sir_strat = (sir.stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
                .stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]}))
```

The full signature:

```python
model.stratify(
    age      = [0, 15, 65],  # the dimension, by name: bin edges, labels, or a geography
    split    = None,       # share of the population in each level
    adjust   = None,       # {transition_name: [multiplier per level]}
    states   = None,       # which states the dimension applies to (default: all; see Strain)
    mobility = None,       # movement between levels (space only)
    immunity = None,       # cross-protection matrix (strain only)
    kind     = None,       # override the inferred kind
    **transitions,         # extra transitions that only exist once the dimension does
)
```

Several dimensions MAY be given in one call when none of them needs options (`sir.stratify(age=[0, 15, 65], sex=['f', 'm'])`). When `split`, `adjust`, or the other options are used, give one dimension per call so it is clear which dimension they belong to. A keyword whose value is an `ss.Transition` (usually an `ss.transfer`) is added to the new model under that name; this is how transitions that move people along the new dimension are declared ([below](#moving-between-coordinates-sstransfer)).

### What it does

`stratify()` returns a model in which:

1. Every state in `states` is split into one state per level.
2. Every transition touching those states is split, keeping its name and its rate or dwell-time declaration.
3. Each `adjust` multiplier is applied to the named transition in the corresponding level.
4. The transitions implied by the dimension's kind are generated ([below](#kinds)).
5. Any extra transitions passed as keywords are added.

The base model says "recovery takes `dur_inf`"; the stratification says "1.5 times faster in children". Neither refers to the other, which is what lets one disease definition be used with many structures.

An `adjust` naming a transition that does not exist raises `E301 UnknownTransition` and lists the transitions that do. A `split` of the wrong length raises `E302 BadSplit`; a split that does not sum to 1 is normalized, and `describe()` says so. If `split` is omitted, it is taken from the population where the property already exists (age, sex, patch populations from a geography) and is otherwise equal across levels, printed as a guess.

### Levels

```python
sir.stratify(age=[0, 15, 65])                  # bin edges: 0-14, 15-64, 65+
sir.stratify(risk=['low', 'high'])             # labels
sir.stratify(vaccinated=[False, True])         # a boolean property
hiv.stratify(cd4=[0, 200, 350, 500])           # bin edges on a continuous property
sir.stratify(patch=ss.geo('Kenya', level='county'))   # a named source
```

A list of numbers is bin edges when the property is continuous and labels otherwise; `kind=` settles any ambiguity. The last bin is open-ended. Levels can be referred to by label (`'15-64'`, `'500+'`) or by position.

The CD4 example shows the general rule for continuous properties. `hiv.cd4` is a float per agent, updated by the HIV module's own code. In an agent-based run nothing about that changes: CD4 stays continuous, and the bins let you write `adjust={'death': [8, 3, 1.5, 1]}`, select `sim.results.hiv.n_infected.sel(cd4='0-200')`, and target interventions by CD4 group. In a compartmental run there are no per-agent values, so the model needs transitions that move people between bins (`decline = ss.transfer(cd4='500+ -> 350-500', rate=...)` and so on); without them the bins are static, and `describe()` says so.

## Kinds

Some dimensions carry structure. The kind is inferred from the name, printed by `describe()`, and overridden with `kind=`.

| Kind | Inferred from the name | What is generated |
|---|---|---|
| `age` | `age` | Aging between adjacent levels |
| `space` | `patch`, `node`, `region`, `district`, `county`, or an `ss.geo` value | Nothing by default; movement if `mobility=` is given |
| `strain` | `strain`, `variant`, `genotype`, `serotype`, `resistance` | A force of infection per strain; cross-immunity |
| `plain` | anything else | Nothing |

### Age

In an agent-based run people simply get older, as in Starsim. In a compartmental run, aging is generated as a transfer between each adjacent pair of levels at rate `1/width`, applied to all states. This is the usual approximation and `describe()` lists the generated transfers; they are ordinary transitions afterwards and can be adjusted or modified like any other. A model has at most one age dimension; stratifying by age twice, or by any dimension the model already has, raises `E305 DuplicateStructuralDimension`.

### Space

Space is a dimension, not a separate kind of model. It is valid with or without movement between patches:

```python
kenya = ss.geo('Kenya', level='county')

sir_kenya = sir.stratify(patch=kenya)                                # patches; nobody moves
sir_kenya = sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))    # patches coupled by travel
```

Patches without mobility are an ordinary model (wastewater catchments, malaria driven by local rainfall, the same disease with per-patch parameters) and produce no warning. `describe()` lists the mobility, or "none". Mobility itself is specified in [06-routes.md](06-routes.md#mobility).

An `ss.Compartments` module can be stratified by the same geography, which gives one count per patch (virus in each catchment's wastewater, mosquitoes in each district); see [06-routes.md](06-routes.md#counts-used-literally).

### Strain

There are two ways to model several strains, and both are supported.

**As a dimension**, when the strains are similar and share one structure:

```python
flu2 = flu.stratify(strain=['H1N1', 'H3N2'], immunity=[[1.0, 0.3], [0.3, 1.0]])
```

A strain dimension changes the force of infection: each strain has its own infectious states and its own force of infection, and a person infected by strain `k` enters strain `k`'s states. By default the dimension applies to every state that can only be reached through infection (`E`, `I`, `R`, ...) and not to the never-infected state. `immunity[i][j]` is the protection that prior infection with strain `i` gives against strain `j` (1 is complete). If `immunity=` is omitted, the base model's structure applies as written (in an SIR model, recovery from any strain protects against all of them), and `describe()` says so. `adjust` on a transmission multiplies that strain's transmissibility.

A person holds one strain coordinate at a time. A compartmental run tracks only the most recent strain a person was infected with, which is an approximation when immunity is partial and is noted when the model is built; an agent-based run keeps the full history.

Drug resistance in TB is the standard example. Resistance is a strain dimension, and acquired resistance is a transfer along it that only happens during treatment:

```python
susceptible, latent, active, on_treatment = ss.states('susceptible latent active on_treatment')

tb = ss.Disease(
    infection  = susceptible >> latent @ ss.peryear(8) * active,
    activation = latent >> active @ ss.peryear(0.05),
    diagnosis  = active >> on_treatment @ ss.peryear(0.7),
    cure       = on_treatment >> susceptible @ ss.months(6),
)

tb_dr = tb.stratify(
    resistance = ['DS', 'MDR'],
    adjust     = {'infection': [1.0, 0.7],     # MDR strains are less transmissible
                  'cure':      [1.0, 0.3]},    # and much slower to cure
    acquired   = ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment']),
)
```

**As a second disease**, when the pathogens are qualitatively different (different natural histories, or a person can carry both at once): write two `ss.Disease` modules and connect them with `ss.modify` ([07-effects.md](07-effects.md)). Co-infection needs this form, because a dimension gives each person one coordinate.

## Moving between coordinates: `ss.transfer`

`ss.transfer` moves people between levels of a dimension without changing their disease state.

```python
ss.transfer(risk='low -> high', rate=ss.peryear(0.1))       # an arrow between two levels
ss.transfer(vaccinated=True, rate=ss.peryear(0.2))          # a target value: everyone not at it moves to it
ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment'])
```

- The keyword is the dimension. Its value is an arrow string between two levels, or a single target level, meaning "from every other level to this one". A boolean dimension is written with its target value (`vaccinated=True`), not as an arrow.
- `rate=` is a per-person hazard; `p=` and `dur=` are accepted as for any transition ([03-states-and-transitions.md](03-states-and-transitions.md)).
- `states=[...]` restricts the transfer to people in the named disease states. Without it, people in every state move.
- A transfer is an `ss.Transition`. It gets a flow result like any other, and it can be the target of `ss.modify`.

A transfer can be declared in three places: in the `ss.Disease(...)` constructor with the other transitions, as a keyword to `stratify()` (as in the TB example), or, when it is delivered as an intervention, in `ss.Sim(interventions=...)` with the delivery keywords of [08-interventions.md](08-interventions.md):

```python
ss.transfer(vaccinated=True, coverage=0.7, years=[2021, 2022])
```

A transfer naming a dimension the model does not have when the sim is built raises `E1102 MissingDimension`.

### Coverage and counts on a stratified model

Two mistakes are easy to make when a delivery meets a stratified model, and the library MUST get both right without the user doing anything:

- **`coverage=` is a fraction of the eligible people, applied once.** Each eligible person is considered once, whatever stratum they are in. A person who falls under `P` generated transfers MUST NOT get `P` independent chances; that would deliver `1 − (1 − f)^P` instead of `f`.
- **`capacity=` is a total.** A capacity of 50,000 doses on a model with 16 age groups delivers 50,000 doses, shared across the eligible strata in proportion to their eligible population, not 50,000 per stratum.

Realized coverage and doses delivered are reported as results of the delivery, in total and per stratum, so either can be checked. These two rules are tested in [18-conformance.md](18-conformance.md).

## Adjustments multiply

`adjust` values are multipliers on the named transition's rate (for a transmission, on `beta`; for a transition given as a duration, the duration is divided). There is no overwrite form: multipliers commute, so the result does not depend on the order in which dimensions were added, and someone who needs a different value in one stratum can get it with a multiplier.

A negative multiplier raises `E306 InvalidAdjustment`. A multiplier that pushes a probability above 1 is clipped to 1 with a note. Adjustments combine with `ss.modify` multipliers on the same quantity by multiplication ([07-effects.md](07-effects.md)).

Parameters that differ by level can also be given directly in the base model with `ss.by(age=[...])` ([04-parameters.md](04-parameters.md)); `adjust` is for when the base model should not know about the dimension.

## Mixing over a dimension

Who mixes with whom is given to the sim, not to `stratify`:

```python
sir_age = sir.stratify(age=[0, 15, 65])
ss.Sim(sir_age, networks=ss.contacts('Kenya'))
```

A mixing matrix is matched to the dimension whose levels it describes (here `age`). If its groups differ from the model's bins it is re-binned using population weights, and `describe()` shows the result. With no mixing given, people mix homogeneously across the levels of every dimension except space, where the default is to mix within the patch ([06-routes.md](06-routes.md)).

States that a dimension does not apply to take no part in mixing over it. If such a state is the origin or the source of a transmission, there is no coordinate for those people to mix as, and the build raises `E307 PartialStratificationMixing` naming the state.

## How each method realizes a dimension

| Method | Realization |
|---|---|
| Compartmental (`ode`, `ctmc`, `tau`, `binomial`, `sde`) | Splits every state into one per level, splits every transition, applies the adjustments |
| Agent-based | Uses (or adds) a per-agent property; scales the per-agent rate by the adjustment for the agent's level; continuous properties stay continuous |

A categorical agent attribute that interventions can target (EMOD's Individual Properties, EpiHiper's traits) is the same thing as a stratification dimension, seen from the agent side.

## Size

Strata multiply: age (16) × risk (2) × strain (3) × patch (47) is 4,512 compartments per state, and it is easy to get there by accident.

- `describe()` MUST report the resolved number of compartments.
- When that number is large enough to make a compartmental run slow (by default above about 10,000), the build prints `W309 LargeStateSpace`, naming the dimensions and their sizes and noting that an agent-based run does not grow with the number of strata. It is a warning; the run proceeds.

There is no hard upper limit. If the model does not fit in memory, the error says how many compartments were requested.

## Rejected

- **Manual array indexing as the mechanism.** General, laborious, and it leaves the model impossible to restratify. Indices are an implementation detail.
- **Overwrite adjustments.** They make the result depend on the order of stratification.
- **Stratifying in place.** Returning a new model costs nothing and keeps the unstratified model available for comparison.
- **Writing out the compartments by hand.** `S_child`, `S_adult`, and every transition between them.
- **A separate metapopulation concept.** Space is a dimension that may carry movement.
- **Strain as a dimension only, or as a second disease only.** Each fits a different kind of model.
