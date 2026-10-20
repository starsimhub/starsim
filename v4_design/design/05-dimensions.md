# Dimensions and stratification

Age, space, risk, strain, and vaccination status are handled by one operation: take a finished model and stratify it by a property of the people in it.

Justified by [best-practices/stratification.md](../best-practices/stratification.md).

## Dimensions are agent properties

Every stratification dimension is a property of a person (age, sex, patch, risk group, CD4 count). Not every property is a dimension: a property becomes one when a model is stratified by it.

- In an agent-based run, the property is a per-agent array on `sim.people`, exactly as in Starsim. If it already exists (`age`, `female`, a float state such as `hiv.cd4`), stratifying uses it. If it does not (`risk`), stratifying creates it and assigns values using `split`.
- In a compartmental run, the property becomes an axis: every state it applies to is split into one compartment per level. The split compartments are internal; results are read by dimension ([below](#reading-stratified-results)), never by a compartment's generated name.
- A continuous property is binned for the purpose of stratifying. In an agent-based run it stays continuous, and the bins are used only for adjustments, group-level conditions, and results.

A dimension name refers to one property across the whole sim, so two modules stratified by `patch` share the same patches, and two diseases stratified by `age` use the same age groups ([below](#stratifying-the-sim-or-the-disease)).

## `stratify`

<!-- fragment -->
```python
sir.stratify(age=[0, 15, 65])        # a disease
sim.stratify(age=[0, 15, 65])        # every disease in the sim; the same as ss.Sim(..., stratify=dict(age=[0, 15, 65]))
```

`stratify()` **stratifies the model it is given, in place, and returns it**. In place is the default because it is what is wanted nearly every time, and a line like `sir.stratify(age=...)` that left `sir` unchanged would be the more common mistake. Because the model is returned, the call can go straight into the sim, and calls chain, each one stratifying the result of the one before:

<!-- fragment -->
```python
sim = ss.Sim(sir.stratify(age=[0, 15, 65]), networks=ss.contacts('Kenya'))

sir.stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
sir.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
```

`inplace=False` returns a stratified copy and leaves the original untouched, for running the two side by side:

<!-- fragment -->
```python
sir_age = sir.stratify(age=[0, 15, 65], inplace=False)     # sir is unchanged
ss.parallel(plain=ss.Sim(sir), by_age=ss.Sim(sir_age, networks=ss.contacts('Kenya')))
```

Stratifying a model again in the same way (the same levels, split, and adjustments) does nothing, with a note: this is what a notebook cell that is run twice does. Stratifying it by a dimension it already has in another way raises `DuplicateStructuralDimension`.

The full signature:

<!-- fragment -->
```python
model.stratify(
    age      = [0, 15, 65],  # the dimension, by name: bin edges, labels, or a geography
    split    = None,       # share of the population in each level
    adjust   = None,       # {transition_name: [multiplier per level]}; also rel_trans and rel_sus
    states   = None,       # which states the dimension applies to (default: every state that is people; see Strain)
    where    = None,       # which existing groups are split, e.g. dict(sex='female') (default: all)
    mobility = None,       # movement between levels (space only)
    immunity = None,       # cross-protection matrix (strain only)
    kind     = None,       # the kind, if not the one its value gives: 'space', 'strain', 'age', or 'plain'
    inplace  = True,       # False returns a stratified copy and leaves the model unchanged
    **transitions,         # extra transitions that only exist once the dimension does
)
```

Several dimensions MAY be given in one call when none of them needs options (`sir.stratify(age=[0, 15, 65], sex=['f', 'm'])`). When `split`, `adjust`, or the other options are used, give one dimension per call so it is clear which dimension they belong to. A keyword whose value is an `ss.Transition` (usually an `ss.transfer`) is added to the new model under that name; this is how transitions that move people along the new dimension are declared ([below](#moving-between-coordinates-sstransfer)).

### What it does

After `stratify()`, the model is one in which:

1. Every state in `states` is split into one state per level (within the groups selected by `where=`, if given).
2. Every transition touching those states is split, keeping its name and its rate or dwell-time declaration.
3. Each `adjust` multiplier is applied to the named transition in the corresponding level. `adjust={'rel_trans': [...]}` and `adjust={'rel_sus': [...]}` multiply the infectiousness and susceptibility of the people in each level (of the states the dimension applies to), as summer2's infectiousness adjustments do.
4. The transitions implied by the dimension's kind are generated ([below](#kinds)).
5. Any extra transitions passed as keywords are added.

The base model says "recovery takes `dur_inf`"; the stratification says "1.5 times faster in children". Neither refers to the other, which is what lets one disease definition be used with many structures.

An `adjust` naming a transition that does not exist raises `UnknownTransition` and lists the transitions that do. A `split` of the wrong length raises `BadSplit`; a split that does not sum to 1 is normalized, and `describe()` says so. If `split` is omitted, it is taken from the population where the property already exists (age from a named population, sex, patch populations from a geography); otherwise age groups follow the world's population by age (the UN's for 2024, also the agents' ages when nothing else gives them), a boolean dimension (`[False, True]`) or one that a transfer, a transition's `move=`, or a delivery moves people along starts with everyone at its first level, and any other is equal across levels, each printed as a guess.

### Stratifying the sim or the disease

Age, sex, and place are properties of people, and every disease in a sim has to agree on them: a contact matrix by age has no meaning if two diseases use different age groups. A people property is therefore normally stratified once, on the sim, and every module gets it:

<!-- fragment -->
```python
sim = ss.Sim([flu, rsv], networks=ss.contacts('Kenya'))
sim.stratify(age=[0, 5, 18, 65])

sim = ss.Sim([flu, rsv], networks=ss.contacts('Kenya'), stratify=dict(age=[0, 5, 18, 65]))    # the same
```

`sim.stratify()` takes the same arguments as `model.stratify()`, applies them to every module in the sim, and returns the sim. `ss.Sim(stratify=...)` takes a dict of the same keyword arguments and is the same call made when the sim is created. All three go through one parser with the same checks: an option (`split`, `adjust`, `kind`, `states`, `where`, `mobility`, `immunity`) is never taken for a dimension, a call with no dimension raises, and `adjust=` on the sim names a transition by its name if only one module has it, or as `'flu.infection'` (an unknown or ambiguous one is `UnknownTransition`). A dimension belongs to the sim either way; where it is declared decides only the default `states=` and which transitions `adjust=` names.

A dimension that belongs to one disease (strain, drug resistance, vaccination status against one pathogen) is stratified on that disease, and so are per-disease adjustments:

<!-- fragment -->
```python
sim.stratify(age=[0, 5, 18, 65])
flu.stratify(age=[0, 5, 18, 65], adjust={'infection': [1.5, 1.2, 1.0, 0.8]})    # adds the adjustments; the age groups are the sim's
```

Rules:

- Stratifying a disease by a people property the sim is already stratified by, with the same levels, adds the `adjust` and extra transitions and does not split anything again. This is the one case in which naming an existing dimension is not `DuplicateStructuralDimension`. A `split=` given on either is used; two different ones raise `InconsistentDimension`.
- Two modules in one sim stratified by the same people property with different levels raise `InconsistentDimension` when the sim is built; the message names both sets of levels and suggests stratifying the sim.
- In a sim with one disease, stratifying the disease by age and stratifying the sim by age do the same thing.

### Stratifying part of the model

By default a dimension applies to every state that is people, in every group. Two keywords narrow that:

<!-- fragment -->
```python
hiv.stratify(cd4=[0, 200, 350, 500], states=[I])               # only the infected have a CD4 stage
sir.stratify(sex=['female', 'male'])
sir.stratify(age=[15, 25, 35, 50], where=dict(sex='female'))   # age groups for women only
```

- **`states=`** restricts which states are split. It takes state objects or their names.
- **People entering a state with the dimension** from one without it (an infection entering `I`, above) take a level in proportion to `split`. A split that depends on another dimension is an `ss.by` over both, read as the share in each level for each level of the other: `sir.stratify(clinical=['mild', 'severe'], states=[I], split=ss.by([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]], dims=['age', 'clinical']))` makes the old more often severe, as summer2's `source_strata` do. Each row is normalized, so the other dimension's own split is unchanged. An `adjust` on a transition out of states that have no level of the dimension raises `InvalidAdjustment`, since there is no level to apply it in.
- **`where=`** restricts which existing groups are split, as a dict of dimension and level (or list of levels). The result is a ragged model: four age groups of women and one group of men. The dimension can be one the sim is stratified by, or another `where=` dimension (`parity` where `age` is `'25-34'` or `'35-49'`, so women of those ages only).

People outside `where=` have no level of the dimension. Everything that refers to a level follows from that:

- **Results.** Selecting a level selects the people who have one: `res.n_I.select(age='15-24')` is women aged 15 to 24, and `res.n_I.by('age')` has a column for each age group of women, summing to the women and not to everyone. Men are in the totals and in `select(sex='male')`, and in no column of a result split by age: their age is missing, not "all ages".
- **Conditions.** `sir.age == '15-24'` is false for men, so `where=sir.age == '15-24'` reaches women only, and `~(sir.age == '15-24')` includes every man.
- **Adjustments and movement.** `adjust=` applies to the people with a level; for the others the multiplier is 1. People outside `where=` do not move between its levels (by aging or by a transfer), and someone who moves into a group outside it (a woman aging out of the `where=` of `parity`) leaves its levels.
- **Values by level.** A value given for each level (`init=dict(I=ss.by(age=[...]))`, or a parameter by age) gives the people with no level the mean over the levels, weighted by the split; a number of people at each level gives them none.
- **Mixing.** A contact matrix by the dimension has no level for the others to mix as, which is `PartialStratificationMixing` (below).

In an agent-based run neither keyword changes the model, since every agent has an age and a sex regardless; they decide the number of compartments in a compartmental run. The groups there are still a grid of every combination of levels: the people outside `where=` are all counted at the first level, and the groups at the other levels stay empty.

### States that are not people

A state that is not people (bacteria in water, mosquitoes; [06-routes.md](06-routes.md#counts-used-literally)) has no age, sex, or risk group. The rule:

- **People dimensions skip it.** Stratifying by age, sex, risk, or any other property of people leaves a state that is not people unsplit, whether the call is on the disease or on the sim.
- **Space applies to it.** Stratifying by `patch` gives one count per patch: a reservoir per catchment, mosquitoes per district.
- **Its own dimensions are given with `states=`.** `malaria.stratify(species=['gambiae', 'funestus'], states=[Sv, Ev, Iv])` splits the mosquito states by species and leaves the human states alone.

`describe()` lists, for each dimension, the states it applies to.

### Reading stratified results

A stratified result is still one result, with a dimension for each stratification ([09-results.md](09-results.md)). It is the total until a group is selected, so nobody sums `I_0_female_low` through `I_55_male_high` by hand:

<!-- fragment -->
```python
res = sim.results.sir
res.n_I                                        # everyone in I
res.n_I.select(age=0)                          # the youngest group, by position or by label ('0-14')
res.n_I.select(sex='female', risk='high')      # one group of two dimensions, summed over the rest
res.n_I.by('age')                              # one series per age group
res.infection.select(age=0)                    # incidence in children
```

`.select()` and `.by()` work the same way under every solver. There are no per-stratum result names. `.by()` gives a result with one column per level, named by its label; if a label is also the name of an attribute of a result (`low`, `high`), every column is named `<dimension>_<label>` instead (`risk_low`, `risk_high`).

### Levels

<!-- fragment -->
```python
sir.stratify(age=[0, 15, 65])                  # bin edges: 0-14, 15-64, 65+
sir.stratify(risk=['low', 'high'])             # labels
sir.stratify(vaccinated=[False, True])         # a boolean property
hiv.stratify(cd4=[0, 200, 350, 500])           # bin edges on a continuous property
sir.stratify(patch=ss.geo('Kenya', level=1))          # a named source: Kenya's 47 counties
```

A list of numbers is bin edges when the property is continuous and labels otherwise; `kind=` settles any ambiguity. The last bin is open-ended. Levels can be referred to by label (`'15-64'`, `'500+'`) or by position.

The CD4 example shows the general rule for continuous properties. `hiv.cd4` is a float per agent, updated by the HIV module's own code. In an agent-based run nothing about that changes: CD4 stays continuous, and the bins let you write `adjust={'death': [8, 3, 1.5, 1]}`, select `sim.results.hiv.n_I.select(cd4='0-200')`, and target interventions by CD4 group. In a compartmental run there are no per-agent values, so the model needs transitions that move people between bins (`decline = ss.transfer(cd4='500+ -> 350-500', rate=...)` and so on); without them the bins are static, and `describe()` says so.

## Kinds

Some dimensions carry structure. The kind comes from the value, not the name: a guess fills in what is missing and never reinterprets what was given, so `region=['north', 'south']` is a plain dimension like any other. It is printed by `describe()` and given with `kind=` when the value does not say it. A name that suggests another kind (`patch`, `region`, `strain`, `resistance`, ...) gets a note saying how to make it one.

| Kind | From the value | What is generated |
|---|---|---|
| `age` | a dimension called `age`, with bin edges | Aging between adjacent levels, if people are born or die |
| `space` | an `ss.geo` value, or `mobility=` | Nothing by default; movement if `mobility=` is given |
| `strain` | `immunity=` | A force of infection per strain; cross-immunity |
| `clock` | `kind='clock'`, with levels that are times | The time since an event, which advances with time; a level `none` for people who have not had it |
| `plain` | anything else | Nothing |

### Age

People age if and only if they are born or die (demographics, or a declared `ss.birth`, `ss.immigration`, or `ss.death` from every state), as in Starsim; `use_aging=` says otherwise. A closed population does not age, since its age groups would empty out and fill up in a way no one meant, and this is noted and marked as a guess in `describe()`. In an agent-based run people simply get older. In a compartmental run, people in every state that is people move from each age group to the next at the rate the age split gives: the density of people just below the upper edge over the number in the group, from the pyramid of a named population (or of the world, when the split is a guess), and `1/width` for a split that was given. `describe()` says on the dimension's line whether people age between the groups. Aging is not one of the disease's transitions, so it has no flow result and cannot be adjusted or modified. The ages of a stratified population run from 0 to 100. A model has at most one age dimension; stratifying by age twice, or by any dimension the model already has, raises `DuplicateStructuralDimension` (with the one exception [above](#stratifying-the-sim-or-the-disease)).

### Counters and clocks

A **counter** is a dimension with whole-number levels that events move people up (or down) a number of levels, with `increment=` on a transition, a delivery, or a transfer ([03](03-states-and-transitions.md)): `increment='parity'` adds 1, and `increment=dict(parity=2)` adds 2 (twins), `dict(doses=-1)` takes one off. The levels are the counts and the last is open-ended, so a move up from `7+` stays in it (as agents, the value keeps counting); a move below the first level stays at the first. A reset is an ordinary `move=dict(parity=0)`. Nothing else marks a counter. In a compartmental run each level moves to the one `k` up, which is exact when the levels are one apart; with wider levels (`[0, 1, 2, 5]`) the people in each are taken to be at its lower edge, with a note.

<!-- fragment -->
```python
sim.stratify(parity=list(range(8)))            # 0, 1, ..., 7+
delivery = ss.Transition(P, PP, dur=ss.months(9), p=0.98, increment='parity', move=dict(since_birth=0))
twins    = ss.Transition(P, PP, dur=ss.months(9), p=0.02, increment=dict(parity=2), move=dict(since_birth=0))
campaign = ss.Delivery(increment='doses', coverage=0.6, at=[2025, 2026, 2027])
waning   = ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5))
```

A **clock** (`kind='clock'`) holds the time since an event, and advances with time as age does. Its levels are the lower edges of bins of time, as durations (`ss.months([0, 1, 2, 3, 6, 12, 24])`) or numbers in the sim's unit, followed by a level `none` for the people who have not had the event, where everyone starts; it takes no `split=`. An event sets it with `move=` (`move=dict(since_birth=0)`, a time in the clock's unit), and a value by it is an `ss.by` table whose row with key `None` is for the people at `none` ([04](04-parameters.md)). As agents, each person holds the time since it was set (NaN at `none`), advanced by a step at the end of each step. In a compartmental run the levels are whole steps from 0 to the last edge (the last open-ended, then `none`), and everyone moves up one at the start of each step, which is exact; edges that are not whole steps are rounded, with a note, and results by group are by step. Age is the clock that birth sets; `after=` and `before=` stay the time in the source state, which needs no clock.

### Space

Space is a dimension, not a separate kind of model. It is valid with or without movement between patches:

<!-- fragment -->
```python
kenya = ss.geo('Kenya', level=1)

sir.stratify(patch=kenya)                                # patches; nobody moves
sir.stratify(patch=kenya, mobility=ss.gravity(away=0.1, a=1.0, b=1.0, c=2.0))    # or: patches coupled by travel
```

Patches without mobility are an ordinary model (wastewater catchments, malaria driven by local rainfall, the same disease with per-patch parameters) and produce no warning. `describe()` lists the mobility, or "none". Mobility itself is specified in [06-routes.md](06-routes.md#mobility).

A state that is not people is split by the same geography, which gives one count per patch (virus in each catchment's wastewater, mosquitoes in each district); see [above](#states-that-are-not-people) and [06-routes.md](06-routes.md#counts-used-literally). An inflow whose rate uses a count (`ss.null >> W @ xi * I.n`, or `ss.birth`) arrives in each patch according to the count there; an inflow at a fixed rate is shared between patches by their split.

### Strain

There are three ways to model several strains, and all are supported. The choice:

| The strains | Form |
|---|---|
| One at a time, sharing one natural history (dengue serotypes, influenza strains, drug resistance) | A strain dimension of one disease: `stratify(strain=..., immunity=...)` |
| Several at once, each with its own natural history (HPV genotypes) | A family of diseases: `ss.Strains(name, members, immunity=...)` |
| Unrelated pathogens (HIV and syphilis) | Two diseases, with `ss.modify` between them |

The strain dimension and the family take the same `ss.immunity` ([07-effects.md](07-effects.md#protection-between-strains-ssimmunity)), with the same matrix layout (rows from, columns against) and the same rules for combining protection, so cross-immunity is one mechanism in both.

**As a dimension**, when the strains are similar and share one structure:

<!-- fragment -->
```python
flu.stratify(strain=['H1N1', 'H3N2'], immunity=[[1.0, 0.3], [0.3, 1.0]])    # a strain, since immunity= is given
flu.stratify(strain=['H1N1', 'H3N2'], immunity=ss.immunity(rel_sus=[[1.0, 0.3], [0.3, 1.0]], combine='multiply'))    # the same, with another rule
```

A strain dimension changes the force of infection: each strain has its own infectious states and its own force of infection, and a person infected by strain `k` enters strain `k`'s states. By default the dimension applies to every state that can only be reached through infection (`E`, `I`, `R`, ...) and not to the never-infected state. `immunity[i][j]` is the protection that prior infection with strain `i` gives against strain `j` (1 is complete). A bare matrix is `ss.immunity(rel_sus=matrix)` with the default rule, the strongest protection of the strains a person has had; `ss.immunity(rel_sus=matrix, combine=...)` gives another rule (`'sum'`, `'multiply'`, or a function, which needs agents). In a dimension a person is protected by every strain they have had, so `protected=` is not given, and the protection is against infection only (`rel_sus=`). If `immunity=` is omitted (with `kind='strain'`), the base model's structure applies as written (in an SIR model, recovery from any strain protects against all of them). `adjust` on a transmission multiplies that strain's transmissibility.

A person holds one strain coordinate at a time. A compartmental run would track only the most recent strain a person was infected with, which is wrong when immunity is partial (with no protection between two strains, people could go from A to B and back to A), so a compartmental solver raises `CapabilityError` for a strain dimension with partial immunity and people who can be infected again; an agent-based run keeps the full history, and the protection of the strains a person has had is combined by the rule. So `solver='auto'` runs a strain dimension with partial immunity as agents, and `sim.solvers()` marks the compartmental solvers as unable to run it ("would keep only the last strain"). The exact compartmental form, a history of the strains had (`2^k` levels), is deferred. As agents, `immunity=` does not yet run over networks (`CapabilityError`); it runs with mixing pools or homogeneous mixing. A new infection is recorded under the strain acquired, so `res.infection.by('strain')` is the incidence of each strain. The people who can be infected again are those in states that have the strain dimension and are neither infected nor already a source of infection (in an SIR, `R`).

Drug resistance in TB is the standard example. Resistance is a strain dimension, and acquired resistance is a transfer along it that only happens during treatment:

```python
susceptible, latent, active, on_treatment = ss.states(4)

tb = ss.Disease(
    infection  = susceptible >> latent @ ss.peryear(8) * active,
    activation = latent >> active @ ss.peryear(0.05),
    diagnosis  = active >> on_treatment @ ss.peryear(0.7),
    cure       = on_treatment >> susceptible @ ss.months(6),
)

tb.stratify(
    resistance = ['DS', 'MDR'],
    kind       = 'strain',                     # with no immunity=, the kind is given
    adjust     = {'infection': [1.0, 0.7],     # MDR strains are less transmissible
                  'cure':      [1.0, 0.3]},    # and much slower to cure
    acquired   = ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment']),
)
```

`resistance` is a dimension of this disease and not of people, so it is stratified on `tb` and not on the sim.

**Strains carried together.** A level written as other levels joined by `+` is for people who carry those strains at once: `strain=['A', 'B', 'A+B'], kind='strain'`. Each strain carried needs a level of its own. Someone who carries several strains is as infectious as their level (`adjust={'rel_trans': [...]}`), and passes each strain on in proportion to that strain's own infectiousness. Someone infected with a strain they do not carry carries it as well, at the level that adds it; if they carry it already, or there is no level for carrying it as well, they keep the strains they have. Superinfection is then a transmission that leaves people in their state, and is not counted for people it does not change:

```python
superinfection = latent >> latent @ ss.peryear(3) * active
```

A transition can change a person's level as well as their state with `move=`, an arrow between two levels (which also limits the transition to people at the first) or the level people move to (for a clock, the time it is set to), and move them up a counter with `increment=` ([counters](#counters-and-clocks)). This is how one of two strains progresses alone, or resistance is acquired on progression:

```python
ss.progression('latent -> active', rate=ss.peryear(0.1), move=dict(strain='A+B -> A'))
```

tbsim's drug resistance (`tbsim.TB(drugs=['RIF', 'INH'])`) is built this way: the levels are the sets of strains people can carry, and treatment that fails is a transition that moves people to the strains that were not cured, with any resistance they acquired.

**As a family of diseases**, when a person can carry several strains at once and each has its own natural history: HPV genotypes, where co-infection would need a level for every set of genotypes carried (15 at 4 genotypes, 511 at 9) and one clearance would clear every genotype at once. The members are ordinary declared diseases, given as one list; the family ties them together:

<!-- fragment -->
```python
hpv = ss.Strains('hpv', hpvsim.genotypes('hpv16 hpv18 hi5 ohr'),
    immunity = [
        ss.immunity(protected='nab', rel_sus=m_sus, combine='sum'),                    # rel_sus of each genotype × (1 − min(1, Σ_k m_sus[k, g]·[in k's nab]))
        ss.immunity(protected='cleared', dur=dict(In=m_sev, Ic=m_sev), combine='sum'), # shorter stays in In and Ic: severity immunity
    ],
    init_prev = hpvsim.total_prev,     # the share with any genotype: a number, an ss.by, or a function
    split     = [0.3, 0.2, 0.3, 0.2],  # each seeded person gets one genotype, in these shares
    exclusive = 'Ca',                  # cancer from one genotype moves the person to the state with no exits (X) in the others
)
sim = ss.Sim(diseases=hpv, ...)
sim.results.hpv.infected     # infected with any genotype: a union, counting the co-infected once
sim.results.hpv.cancer       # new cancers, any genotype: a sum of flows
sim.results.hpv.infected.by('hpv')   # each member's own
```

The members are in `sim.diseases` under their own names, beside the family. The family's names (`protected=`, the states of `dur=`, `exclusive=`) name states or named sums of states of each member and are resolved when the sim is built, so no member needs another to exist when it is written; a member that lacks one raises, unless it has a row of zeros in that matrix. A stock that every member has (a state, or a named sum of states such as `infected`) is a union over the members, and a flow (a transition) a sum. The family's attributes are the same unions as conditions: `where=hpv.infected` is the people infected with any member, and `hpv.infected.count() >= 2` the co-infected. With `init_prev=`, each person seeded gets one member, entering it as a new infection would (through its first transmission, in its shares), on the family's own random numbers; a member may not then have initial values of its own. The family is exact as agents, over networks too. It runs as agents for now: a compartmental solver raises `CapabilityError`, and `solver='auto'` chooses agents. Members that run side by side under the compartmental solvers, coupled through their shares, are the next step ([19-open-questions.md](19-open-questions.md)). A family named like a dimension raises.

**As a second disease**, when the pathogens are unrelated (HIV and syphilis): write two `ss.Disease` modules and connect them with `ss.modify` ([07-effects.md](07-effects.md)).

## Moving between coordinates: `ss.transfer`

`ss.transfer` moves people between levels of a dimension without changing their disease state.

```python
ss.transfer(risk='low -> high', rate=ss.peryear(0.1))       # an arrow between two levels
ss.transfer(vaccinated=True, rate=ss.peryear(0.2))          # a target value: everyone not at it moves to it
ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment'])
ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5)) # a counter, down a level at a time
```

- The keyword is the dimension. Its value is an arrow string between two levels, or a single target level, meaning "from every other level to this one". A boolean dimension is written with its target value (`vaccinated=True`), not as an arrow. For a counter, `increment=` gives the dimension and the amount instead, and everyone who can move that far does.
- `rate=` is a per-person hazard; `p=` and `dur=` are accepted as for any transition ([03-states-and-transitions.md](03-states-and-transitions.md)).
- `states=[...]` restricts the transfer to people in the named disease states. Without it, people in every state move.
- A transfer is an `ss.Transition`. It gets a flow result like any other, and it can be the target of `ss.modify`.

A transfer can be declared in two places: in the `ss.Disease(...)` constructor with the other transitions, or as a keyword to `stratify()` (as in the TB example). A transfer always happens at a rate; a move delivered to some people at some times is a delivery, with the keywords of [08-interventions.md](08-interventions.md):

```python
ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021, 2022])
```

A transfer naming a dimension the model does not have when the sim is built raises `MissingDimension`.

### Coverage and counts on a stratified model

Two mistakes are easy to make when a delivery meets a stratified model, and the library MUST get both right without the user doing anything:

- **`coverage=` is a fraction of the eligible people, applied once.** Each eligible person is considered once, whatever stratum they are in. A person who falls under `P` generated transfers MUST NOT get `P` independent chances; that would deliver `1 − (1 − f)^P` instead of `f`.
- **`capacity=` is a total.** A capacity of 50,000 doses on a model with 16 age groups delivers 50,000 doses, shared across the eligible strata in proportion to their eligible population, not 50,000 per stratum.

Realized coverage and doses delivered are reported as results of the delivery, in total and per stratum, so either can be checked. These two rules are tested in [18-conformance.md](18-conformance.md).

## Adjustments multiply

`adjust` values are multipliers on the named transition's rate (for a transmission, on `beta`; for a transition given as a duration, the duration is divided; for a branch chosen with `p=`, its probability, and the other branches share the rest, as for a modifier on it in [07-effects.md](07-effects.md)). On one transition with several destinations chosen with `p=` and a dwell time, an adjustment (or a modifier) divides the dwell time, as [03](03-states-and-transitions.md#branching) says, and a note says to name a destination (`'ga31.PT31'`) to change its probability instead. There is no overwrite form: multipliers commute, so the result does not depend on the order in which dimensions were added, and someone who needs a different value in one stratum can get it with a multiplier.

A negative multiplier raises `InvalidAdjustment`. A multiplier that makes the branch probabilities out of a state add up to more than 1 scales them to sum to 1, with a note (`ClippedToRange`). Adjustments combine with `ss.modify` multipliers on the same quantity by multiplication ([07-effects.md](07-effects.md)).

Parameters that differ by level can also be given directly in the base model with `ss.by(age=[...])` ([04-parameters.md](04-parameters.md)); `adjust` is for when the base model should not know about the dimension.

A population that is not the product of a split by each dimension (more children in one patch than another) is given as one split over several dimensions: `sir.stratify(patch=ss.geo(names=['north', 'south']), split=ss.by(pop, dims=['age', 'patch']))`, where `pop` has a row for each age group and a column for each patch. Each dimension's own split is then the total over the others. A split given to one of them as well must agree with that total, or `InconsistentDimension` is raised (rows of shares given one dimension, as in a table of P(wealth | urban), are multiplied by that dimension's split first). A split over age and other dimensions whose row for each age group sums to 1 (`ss.by([[0, 1], [1, 0], [1, 0]], dims=['age', 'wealth'])`) is instead the share within each age group: people keep their ages, and the age groups their own split (noted as guessed). People born into the model (as agents, from `ss.Births` or `ss.Pregnancy`) take the youngest age group's share of the other dimensions, and the place of their parent for a dimension of places.

## Mixing over a dimension

Who mixes with whom is given to the sim, not to `stratify`:

<!-- fragment -->
```python
sir.stratify(age=[0, 15, 65])
ss.Sim(sir, networks=ss.contacts('Kenya'))
```

`ss.contacts()` takes a contact matrix in three forms: a country's, by name (`ss.contacts('Kenya')`, by five-year age group, from `starsim_data`); a matrix with the lower edge of each of its own age groups (`ss.contacts(matrix, age=[0, 15, 65])`); or a matrix over the levels of one or more of the model's own dimensions, by name (`ss.contacts(risk=[[2, 1], [1, 8]])`, used whatever the dimension is called; with several, the contacts between two groups are the product of the entries for each); or one matrix over several dimensions together, for mixing that is not such a product (men mix with women, and by risk differently for each sex): `ss.contacts(matrix, dims=['sex', 'risk'])`, with `matrix[i, k, j, l]` the contacts per unit time of one person at levels `(i, k)` with people at `(j, l)`, or the same as a square matrix over the combinations of levels, the first dimension slowest. Its shape is checked against the dimensions' levels when the sim is built (`ShapeError`, naming each dimension's levels), and it is used as given, not balanced; a separable matrix given whole gives the same results as the product form. A matrix by age need not have the model's age groups: the Prem matrices come in five-year bands whatever the model uses.

- **In a compartmental run** the matrix is aggregated to the model's groups, weighted by population. Where a model group cuts through a matrix band, the band is split in proportion to population. A one-line note says the matrix was aggregated (`MatrixAggregated`).
- **In an agent-based run** the matrix is used with its own groups: each agent has an exact age and is counted in whichever band the matrix uses.
- **With places**, a contact matrix acts within each place, under every solver ([06-routes.md](06-routes.md#mobility)).

With no mixing given, people mix homogeneously across the levels of every dimension except space, where the default is to mix within the patch ([06-routes.md](06-routes.md)).

Every state that is people is stratified by default, so the following cannot arise unless the stratification was narrowed with `states=` or `where=`. States that a dimension does not apply to take no part in mixing over it. If such a state is the origin or the source of a transmission, there is no coordinate for those people to mix as, and the build raises `PartialStratificationMixing` naming the state. The same holds for people outside a dimension's `where=` (men, for age groups of women), so a contact matrix by a dimension with `where=` is `PartialStratificationMixing` too:

```text
>>> seir.stratify(age=[0, 15, 65], states=[S, E, I])
>>> ss.Sim(seir, networks=ss.contacts('Kenya')).run()
PartialStratificationMixing: the contacts "contacts" are given by "age", but "seir" is stratified by it in S, E, I only, so there is no level of it for the people in Ia to mix as.
     Fix: leave out states= to stratify every state, or use contacts that do not depend on "age".
```

A state that is not people is never the origin of a transmission through the routes, and it is mentioned in rates only as a count (`W.n`), so it does not raise this error.

## How each solver realizes a dimension

| Solver | Realization |
|---|---|
| Compartmental (`ode`, `binomial`, `gillespie`) | Splits every state the dimension applies to into one per level, splits every transition, applies the adjustments |
| Agent-based | Uses (or adds) a per-agent property; scales the per-agent rate by the adjustment for the agent's level; continuous properties stay continuous |

A categorical agent attribute that interventions can target (EMOD's Individual Properties, EpiHiper's traits) is the same thing as a stratification dimension, seen from the agent side.

## Size

Strata multiply: age (16) × risk (2) × strain (3) × patch (47) is 4,512 compartments per state, and it is easy to get there by accident. `describe()` lists each dimension with its levels and the states it applies to, so the size can be worked out from it. There is no upper limit. Movement between groups (aging, transfers, migration) is kept as sparse rates and a contact matrix within places as the product of its factors, so memory grows linearly with the number of places. Results by group are kept at every step only while they fit in about 100 MB a module, and at a coarser reporting interval beyond that, with a note ([09-results.md](09-results.md)).

## Rejected

- **Manual array indexing as the mechanism.** General, laborious, and it leaves the model impossible to restratify. Indices are an implementation detail.
- **Overwrite adjustments.** They make the result depend on the order of stratification.
- **Always returning a new model.** In place is the common use, and `inplace=False` covers the comparison.
- **Per-stratum result names** (`n_I_0_female_low`). A result has dimensions; `.select()` and `.by()` read them.
- **Every agent property as a dimension.** Dimensions multiply compartments; properties do not have to.
- **Writing out the compartments by hand.** `S_child`, `S_adult`, and every transition between them.
- **A separate metapopulation concept.** Space is a dimension that may carry movement.
- **Strain as a dimension only, or as a family only.** Each fits a different kind of model.
- **A replicated dimension for a family** (each person holding a state in every level). A dimension gives each person one coordinate, and every dimension feature (split, transfers, `move=`, mixing by level) would need an exception for one that does not.
