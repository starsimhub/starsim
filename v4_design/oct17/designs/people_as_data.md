# People as data: per-person tables, time already spent, conditional splits, a module of properties, and large grids of groups

A design pass from the Thanzi la Onse review ([tlo_review.md](../../oct19/tlo_review.md), group 1), for Cliff to decide before any code. Findings: TL1 (`init=` functions and `ss.People.from_df()`), TL2 (time already spent in a state at the start), TL5 (several conditional splits), TL10 (a module of properties), TL7 (the full grid of groups as agents). Related and already fixed in the working tree: TL3 (an existing property kept), TL4 (a joint split that contradicts a dimension's own), TL6 (more than 32,767 groups), TL8 (a function rate after deaths), TL9 (`ss.ppl.<property>` in a rate, as agents). Evidence: the TLO ports in `TLOmodel:v4port/`, and probes of the current tree (`77362e62` plus the working tree) in the session scratchpad (`design_probes/p1_people.py`, `p2_people.py`, `p4_groups.py`, `p5_misc.py`).

## Summary

**The problem.** TLO's people are a pandas dataframe of about 150 columns. Each module sets its columns at the start from data, reads any column in a rate, and moves people between levels. v4 has most of the pieces (`ss.ppl.<property>` in rates, `ss.by` tables, dimensions, transfers, `increment=`, clocks), but not the start. Probed today:

- `init=dict(I=fn)` raises `TypeError: '>=' not supported between instances of 'function' and 'int'` (in `Engine.init_fractions()`). A per-agent array raises `The truth value of an array ... is ambiguous`. A condition (`ss.ppl.age > 50`) raises `has no single truth value`. Only `ss.by(...)` over a property works, as agents.
- `ss.People` takes no table. `ss.People(n, extra_states=[ss.BoolState('tob', default=array)])` works as agents, but under a compartmental solver v4 draws 100,000 ages from `people.age.default` and ignores every other property (`Sim._init_named_people()`).
- There is no way to say how long a prevalent case has been infected. TLO draws time to AIDS given years since infection, so 34 of 380 should progress in the first year; the port's seeded agents gave 3. Writing the engine's three arrays by hand gave 35 (TL2). `init=dict(I=0.3, since=...)` reads `since` as a state: `UnknownState`.
- A second split that depends on other dimensions raises `Only one split= can cover several dimensions at once`. The declared Lifestyle works round it with `states=['S']` on every split. `split=ss.by(urban=[rows])` crashes (`cannot reshape array of size 6 into shape (2,)`), and a split written as an expression of `ss.ppl` crashes (`float() argument must be ... not 'Cond'`).
- A module that is only properties needs a fake disease (`stay = S -> S @ 0`), which `describe()` calls an infection. `sim.results.n_alive.by('district')` raises "not stratified".
- As agents, every group of the grid is allocated, though each agent is in one. With 20,000 agents and boolean dimensions: 16 groups take 316 MB at peak, 2^16 take 381 MB, 2^20 take 1,096 MB, and 2^22 take 3,622 MB with 3.1 s to initialize. That is about 790 bytes per group of the grid, while at most 19,021 groups are occupied. TLO's whole Lifestyle with sex, age and districts is about 1.4e9 groups.

**The idea.** People are data, and the vocabulary that reads them already exists. So the design adds a way in, and lets the existing slots take values per person:

- **A table becomes the people.** `ss.People.from_df(df)` makes each column a property (the mirror of `People.to_df()`). Under a compartmental solver, the same table gives the groups' joint split exactly.
- **`init=` takes a value per person.** An expression over people's properties is each person's chance of starting in the state; a condition is exactly the people it holds for.
- **`since=` beside `init=`.** It gives the time already spent in the state at the start. As agents, exits are drawn from the dwell given that it is longer than that time (TLO's rule); under compartments, people are placed by stage or slot.
- **Splits may depend on other properties.** `split=ss.by(urban=[rows])` is the split given urban, and a split may be an expression (a regression). Any number of them apply in turn.
- **`ss.Properties`**: a declared module of dimensions and transfers with no disease states.
- **As agents, only occupied groups are stored** (internal; nothing to write).

One new `ss.` name, `ss.Properties`, and one classmethod, `ss.People.from_df()`. One new keyword, `since=`. Everything else is a value that existing slots (`init=`, `split=`) now accept.

```python
import starsim as ss, tlo4
from scipy.special import expit
P = ss.ppl

# TL1: TLO's own initial population, one row per person; its columns become people's properties
people = ss.People.from_df(tlo4.population(2010))   # columns: age, female, district, wealth, urban, hv_inf, hv_aids, hv_years_inf, ...

# TL1, TL2: HIV starts from those columns, including how long each person has been infected
S, I, A = ss.states('S I A')
hiv = ss.Disease(S, I, A,
    infection = S >> I @ beta*(I + A),
    aids      = ss.progression('I -> A', dur=ss.weibull(c=2.2, scale=ss.years(11))),
    init      = dict(I=P.hv_inf & ~P.hv_aids, A=P.hv_aids),   # exactly the people the columns say
    since     = dict(I=ss.years(1)*P.hv_years_inf),          # time already spent in I
)
sim = ss.Sim(people=people, diseases=hiv, total_pop=14.5e6, start=2010, stop=2030)
```

```python
# TL5, TL10: starting from the census instead, Lifestyle as a module of properties
census = ss.People.from_df(tlo4.census(2010), weight='count')   # a row per district, sex, and year of age; agents drawn by count
p_tob = expit(b0 + b_female*P.female + ss.by(age_odds) + b_wealth*P.wealth)   # a regression, as in the behavior vocabulary

life = ss.Properties(
    to_urban  = ss.transfer(urban=True, rate=r_urban),
    start_tob = ss.transfer(tob=True,   rate=r_tob*rr_wealth**P.wealth),
    stop_tob  = ss.transfer(tob=False,  rate=r_quit),
    name = 'lifestyle',
)
life.stratify(urban=[False, True], split=[0.83, 0.17])
life.stratify(wealth=[1, 2, 3, 4, 5], split=ss.by(urban=[shares_rural, shares_urban]))   # wealth given urban
life.stratify(tob=[False, True], split=[1 - p_tob, p_tob])                               # tobacco given sex, age, and wealth

sim = ss.Sim(people=census, diseases=[hiv, life], n_agents=20_000, start=2010, stop=2030)
sim.results.lifestyle.n.by('tob')   # people by level, under every solver
```

The decisions for Cliff are in the [last section](#decisions-for-cliff); the main ones are the constructor's name (1), the meaning of `since=` (6), and how a split given other dimensions is written (8).

## Use cases

| ID | What the model needs | Today | Proposed | Simpler? |
|---|---|---|---|---|
| TL1 (TLO) | Start every module from TLO's exported initial population (20 lifestyle properties, HIV status, cancer stage) | Every port wrote its arrays after `sim.init()`. Then the result at index 0 is stale (`n_I` 100 where 694 were written), and the guessed `init_prev` of 1% and the unborn have to be cleared by hand | `ss.People.from_df(df)`; `init=dict(S1=P.oc_status == 1, ...)` | Yes: the post-init writes go, and the first point is right |
| TL1 (TLO) | Start from the census (district, sex, single year of age, count) | `ss.People(n, age_data=...)` for ages, then districts drawn from a split (so 4.9% of newborns are in their mother's district, MA18) | `ss.People.from_df(census, weight='count')` | Yes, and districts and ages are joint, as in the data |
| TL2 (TLO) | Prevalent HIV infections progress to AIDS given their years since infection | Fresh dwells (3 of 380 progress in the first year, against 34); writing `ti_exit_I`, `mult_I` and `exit_I` by hand (35) | `since=dict(I=ss.years(1)*P.hv_years_inf)` | Yes, and no engine internals |
| TL69 (TLO) | Women pregnant at the start, at every stage of gestation | TLO fakes 9 months of births; v4 starts everyone at conception | `since=dict(P=ss.uniform(ss.weeks(0), ss.weeks(40)))` on the pregnancy's gestation state | Yes |
| TL5 (TLO) | Each Lifestyle property's starting share depends on sex, age, wealth and urban; diarrhea's factors likewise | `states=['S']` on every split, with the shares worked out by hand in nested loops (`v4_lifestyle_declared.py`, about 45 lines of loops) | `split=ss.by(urban=[...])`, or `split=[1 - p, p]` with `p` an expression | Yes: the loops become one expression per property |
| TL10 (TLO) | Lifestyle as a module, with results by level | A fake disease `stay = S -> S @ 0`, described as an infection; `n_alive.by(...)` raises | `ss.Properties(...)`; `res.lifestyle.n.by('tob')`; `sim.results.n_alive.by('district')` | Yes |
| TL7 (TLO) | 32 districts × Lifestyle × sex × age as agents | 327,680 groups crashed before TL6's fix; now about 790 bytes per group, so 1.4e9 groups cannot be allocated | Only occupied groups (at most one per agent) | Runs; nothing to write |
| FPsim | Start from DHS women's records with survey weights: age, parity, urban, wealth, method in use, months on the method, months since the last birth | Coded `fp.People` draws each from distributions; the declared port keeps `people.parity` from data, but time on the current method starts at 0 | `from_df(dhs, weight='v005')`; `init={m: P.method == m for m in methods}`; `since={m: ss.months(1)*P.months_on_method for m in methods}`; the `since_birth` clock taken from its column (as the behavior design says) | Yes: the first year's switching is right, since durations of use are long-tailed |
| HPVsim | Women with prevalent CIN at the start, whose chance of cancer depends on the time already spent in CIN (`C.dwell`, the given-dwell design) | Seeds start with `C.dwell = 0`, so they all get a fresh CIN's (lowest) chance of cancer | `since=dict(C=ss.lognorm_ex(ss.years(3), ss.years(3)))` | Yes, and it is what makes `C.dwell` right for seeds |
| STIsim, malaria | Syphilis seeds at a stage of latency; a malaria run started from a burn-in population (LASER's `load_snapshot()` convention) | Fresh latency; a coded copy of arrays between sims | `since=`; `ss.People.from_df(burnin.people.to_df())` for people's own properties (module states are a separate matter; see [out of scope](#out-of-scope-and-dependencies)) | Partly |

**Other frameworks.** How each starts from data:

| Framework | Population from data | Time already spent |
|---|---|---|
| TLO | A dataframe; each module's `initialise_population()` writes its columns from data, often by a linear model | HIV: years since infection drawn from data, then time to AIDS redrawn until it exceeds it (`hiv.py`, about lines 1383 to 1398): the dwell given that it is longer |
| vivarium (IHME) | `BasePopulation` from age, sex, location structure data; each disease's prevalence by age and sex from data | `<state>_event_time` is NaT at the start (fresh), with an optional minimum `dwell_time` |
| EpiModel | `init.net(i.num=)` or a status vector per node; attributes set on the network | The HIV example sets `stage.time` to 0 for the initially infected (fresh) |
| LASER | `LaserFrame.add_scalar_property()` arrays, filled by user code; `save_snapshot()`/`load_snapshot()` for a burned-in population | Per-agent timers, set by user code |
| Emulsion | Initial conditions as counts per state, or as counts of prototypes (named sets of property values) | Not given |
| SimInf, odin, summer2, MEmilio, flepiMoP, epydemix | Counts per compartment and group (`u0` data frame, `initial(S[]) <- ...`, `set_initial_population(distribution=...)`, initial-conditions files) | Not applicable (memoryless compartments) |

So `from_df` follows TLO, vivarium and LASER for agents, and the compartmental frameworks' "counts per group from data" for compartments. For time already spent, only TLO conditions the remaining dwell on it; the others start seeds fresh, which is v4's default today and stays the default.

## The proposal

### 1. `ss.People.from_df()`

```python
people = ss.People.from_df(df)                    # each row is a person
people = ss.People.from_df(df, weight='count')    # each row stands for `count` people (census cells, or survey weights)
```

- **Columns by name.** `age` is people's age, in years. `female` (booleans) is people's sex; a `sex` column whose values are `female`/`male` or `F`/`M` is read as `female`, with a note; any other values raise. Every other column becomes a property of that name, read as `ss.ppl.<name>` and binned by any dimension of that name (the TL3 rule, `Groups.kept`). A boolean column is a `BoolState`, a numeric column a `FloatArr` (an integer column keeps its integers), and a column of strings is stored as its labels, as a dimension's labels are now. A column named like an existing array (`alive`, `uid`, `slot`, `ti_dead`) raises, naming it.
- **`weight=`** names the column of how many people each row stands for. Then the population size is the column's total (unless `total_pop=` is given), and the sim's `n_agents` are drawn from the rows in proportion to it, with common random numbers (a named stream, `from_df`). Without `weight=`, each row is one agent, and `n_agents` is the number of rows; an `n_agents` that differs draws rows with replacement, with a note.
- **Newborns** take each property's default: the dimension's birth split if it is a dimension, otherwise `False` or NaN, with a note listing the columns that newborns get no value for. Inheriting from the mother is TL34's `default=ss.inherit()` (see dependencies).
- **Nothing is allocated until `sim.init()`**, as for `ss.people('Kenya')`. So a table given to a compartmental run never becomes agents.
- **Round trip.** `ss.People.from_df(sim.people.to_df())` gives the same people's own properties (age, sex, and properties written into People), with new UIDs.

### 2. `init=` with a value per person

`init=` of a state (on `ss.State` or the disease's `init=dict(...)`) keeps its meaning for numbers and `ss.by`. It also takes:

- **An expression of people's properties** (`0.02*P.age`, `expit(...)`, `ss.by(table)` over a property): each person's chance of starting in the state, between 0 and 1. A value above 1 raises (a count makes no sense per person).
- **A condition** (`P.hv_inf & ~P.hv_aids`, `P.district == 'Lilongwe'`, `P.age > 50`): 1 where it holds and 0 elsewhere, so exactly those people. This is the rule the behavior design gave conditions in rates.
- **A function** `fn(sim, uids)` returning a chance or a boolean for each agent, as `init_prev=` already takes (agents only, as for any function).

Arrays are not accepted: an array has no names, so it is fragile under births and deaths (TL8). The error says to put the values in a column of `ss.People.from_df()` and write `init=dict(I=P.<column>)`. If two states' conditions both hold for someone, it raises, with the count of such people. The first state takes the rest, as now.

### 3. `since=`: the time already spent

```python
hiv = ss.Disease(S, I, A, ..., init=dict(I=P.hv_inf), since=dict(I=ss.years(1)*P.hv_years_inf))
I = ss.State('I', init=P.hv_inf, since=ss.years(1)*P.hv_years_inf)    # the same, on the state
```

`since=` mirrors `init=`: a dict by state on the disease, or a keyword on `ss.State`. Its value is a duration: a number in the sim's unit, a typed duration, a distribution (`ss.uniform(ss.weeks(0), ss.weeks(40))`), or an expression of people's properties times a unit. It applies to the people who start in that state. Not given, it is 0 (today's rule, with CR4's dating of seeds a step back).

What it means, as agents:
- The person entered the state `since` ago: `ti_<state>` is set that many steps before the start. `I.time`, `ss.clock(...)`, and `I.dwell` (the given-dwell design) all count from there.
- **A scheduled exit is drawn from the dwell given that it is longer than `since`** (TLO's rule). With the person's one uniform draw `u`, the dwell is `F⁻¹(F(s) + u(1 − F(s)))`, so common random numbers are kept. This is the draw `ss.lifetable.draw()` already makes for an age at death given survival to an age. A distribution with no inverse CDF falls back to drawing again from the same stream, as TLO does.
- **Branches decided on entry are weighted by survival.** Among people still in `P` after time `s`, the share bound for each branch is `p_j S_j(s)`, normalized, where `S_j` is that branch's dwell survival. Branches with long dwells are over-represented among prevalent cases, and this keeps them so.
- **Competing rates** are memoryless and need nothing.
- **A `since` beyond the longest possible dwell** (a fixed 40 weeks with `since` of 41) exits in the first step, with a note giving the count.

### 4. Splits given other properties (TL5)

```python
life.stratify(wealth=[1, 2, 3, 4, 5], split=ss.by(urban=[shares_rural, shares_urban]))   # a split for each level of urban
life.stratify(tob=[False, True], split=[1 - p_tob, p_tob])                               # p_tob an expression of other properties
```

- **`split=ss.by(<dim>=[split, split, ...])`** gives a split for each level of another dimension: the shares of this dimension's levels given that one. Several dimensions are `ss.by(table)` with their columns, one column per level of this dimension (or a `value` column and a column named for this dimension), normalized over this dimension, as `p=` with a `to` column is normalized over the destinations (behavior design).
- **A split may be an expression** of people's properties: a regression or a table lookup, as in any rate.
- **Any number of splits may depend on other dimensions.** They apply in order: each after the dimensions it reads. A cycle (urban given wealth, and wealth given urban) raises `InconsistentDimension`, naming both.
- **The joint form is kept as it is.** `split=ss.by(table, dims=['urban', 'wealth'])` is the share at each combination, and contradicting a split given to one of them raises (TL4, as `merge_tradeoffs.md` decided). The keyed form above is new and has one reading, so no table is ever reinterpreted.
- A property the people already have (from `from_df`) takes no `split=`, which raises `BadSplit` (TL3's rule).

### 5. `ss.Properties`: a module of properties

```python
life = ss.Properties(
    to_urban  = ss.transfer(urban=True, rate=r_urban),
    start_tob = ss.transfer(tob=True,   rate=r_tob*rr_wealth**P.wealth),
    stop_tob  = ss.transfer(tob=False,  rate=r_quit),
    name = 'lifestyle',
)
life.stratify(urban=[False, True], split=[0.83, 0.17])
```

- A declared module whose people are everyone alive, with no disease states. It takes transfers (and `increment=` counters and clocks, from the behavior design) as keywords, as `ss.Disease` takes transitions, and dimensions with `.stratify()`. Its dimensions are the sim's, as every module's are (a property has the same groups for every module).
- **Results**: `res.lifestyle.n`, the people in it, kept by group (so `.by('tob')`, `.select(urban=True)`), and a flow for each transfer (`res.lifestyle.start_tob`). They exist under every solver.
- `describe()` lists its properties, their levels, where each comes from (given by the people, or a split, and what the split reads), and the transfers. It never says "infected" (TL53's wording problem, for this module).
- Also, as agents, `sim.results.n_alive`, `new_deaths` and births are kept by group whenever the sim is stratified, within the existing budget for results by group (TL10's second half).

### 6. Only occupied groups, as agents (TL7)

Nothing to write. As agents, the engine stores the groups people are in, not the grid:

- A group's code is the mixed-radix number over all dimensions, as `group_raw()` makes now, but as a 64-bit integer whenever the grid is larger than 2^31 (today `level_dtype()` stops at 32-bit). A grid larger than 2^63 raises, naming the dimensions.
- The occupied codes are numbered as they first appear (at the start, by moves, and by births), and never removed, so a group keeps its number and its results column. Every per-group array in the agent engine (counts, flows, live totals, rates by group) has one entry per occupied group. A rate by group is evaluated on the occupied groups' levels.
- Initial levels are drawn one dimension at a time, using the conditional splits of section 4, so the grid's `group_split` is never formed. So is the birth split.
- Results by group are stored by occupied group, with the code table. `.by('district')` and `.select(...)` decode them, and show every level of the dimension, with zeros where nobody is.

The compartmental solvers keep the full grid, since every group is a row of every compartment. Above 100,000 groups, a note gives the count and says that agents store only occupied groups. TLO's Lifestyle with districts is therefore agents-only, which is fine (Cliff: "some things will be agent only and that's fine").

## Semantics under every solver

| Feature | Agents | ODE | Binomial | Gillespie |
|---|---|---|---|---|
| `from_df` people | Each row an agent, or agents drawn from rows by `weight=` | The weighted counts at each combination of the dimensions' levels are the groups' split: exact, with no assumption of independence. The population is the total weight. Columns that no dimension, `init=` or `since=` reads are listed in a note | As the ODE, with whole people by the binomial's own rounding | As the binomial |
| `init=` an expression or condition | Each person's chance, one draw from the state's stream; a condition is exact | Each group's share: the mean of the value over the table's rows in that group (exact); or, with no table, the value evaluated per group when every property in it is a dimension; otherwise `CapabilityError` naming `stratify()` (behavior decision 5) | As the ODE | As the ODE |
| `init=` a function | Per agent | `CapabilityError` | Same | Same |
| `since=` | Entry dated back; the dwell given survival; branches weighted by survival; clocks and `.dwell` include it | Exponential exits: no effect (memoryless). Erlang stages (k of them, rate kλ): the stage given time `s`, a Poisson(kλs) truncated below k, so exact for the stages. A conveyor (`after=`, `.dwell` slots, clocks, the gestation conveyor): the slot of `s` in whole steps, exact to the step. Hyperexponential phases (HP28): weights `w_i e^{-λ_i s}`, normalized, exact. A distribution of `s`, or values per row of the table, is placed by its mass in each stage or slot | As the ODE | As the ODE |
| Splits given other properties | Drawn per person, in order | Their product over the grid, per group: exact | As the ODE | As the ODE |
| `ss.Properties` | Transfers per person | Transfers between groups, as now | Same | Same |
| Occupied groups | Only occupied groups stored | Full grid (note above 100,000 groups) | Same | Same |

Agents only: `init=` functions; and rates, `init=` and `since=` reading a property that is not a dimension when there is no table to average over. Approximate: nothing new. `since=` is exact under every solver for each dwell form, to the step for conveyors; the approximation of a dwell by Erlang stages is the existing one.

**`describe()`.** A state's `init` line says where it comes from (`init: I = people.hv_inf & ~people.hv_aids (3,351 people)`), and a `since` line gives the median and range (`since: years(1)*people.hv_years_inf, median 6.2 years`). A split names what it is given (`wealth: split given urban`). For `from_df` people: `people: 20,000 agents from a table of 14,516 rows (weight count, total 14.5 million)`.

**`to_spec()`.** The table is data, so it goes in the data hash, not the model's, as `ss.people('Kenya')` does. The spec records `"people": {"from_df": {"columns": [...], "weight": "count"}}`. `init` and `since` expressions are written as text (`"people.hv_inf & ~people.hv_aids"`). A split given another dimension is `{"by": ["urban"], "values": [...]}`, as `ss.by` is now. `ss.Properties` is a module of type `properties`. **`to_code()`** writes the same spellings, with `ss.People.from_df(df)` and a note that the table must be given again.

## Fit with what exists

| Need | Carried by today | Changes |
|---|---|---|
| People from a table | `ss.People(n, age_data=, extra_states=)` (`people.py`, `People.__init__()`); `extra_states` already takes arrays as defaults; `People.to_df()` | `People.from_df()`, a classmethod holding the table until `init`; `Sim.init_people()` (`sim.py`) makes the arrays and draws rows; `Groups.kept` (the TL3 fix in `Sim.init_people()`) covers the columns |
| The table under compartments | `Sim._init_named_people()` (`sim.py`) turns given people into an age pyramid from 100,000 draws of `people.age.default`, its `given_pyramid` | The joint weighted counts of every dimension column, as the groups' split (`Groups.init()`, `dimensions.py`) |
| `init=` per person | `Engine.init_fractions()` (`solvers/engine.py`) already defers an `ss.by` over people's properties to `Agents.init_post()` (the `init.on_people(sim)` branch); the crash is the `init >= 1` test further down | The same deferral for expressions, conditions and functions; per group from the table under compartments; the overlap check |
| `since=` | `Agents.init_post()` (`solvers/agents.py`) dates seeds `ti_<state>` = −1 (CR4); `Agents.schedule()` draws a fresh dwell and sets `ti_exit = entered + best/dt - 0.5`; `Agents.sample_dwell()`; `ss.lifetable.draw()` (`transitions.py`) is the conditional draw; `Compartmental.arrive(before=True)` (`solvers/compartmental.py`) puts seeds one slot along | `entered` set back by `since`; the residual draw in `sample_dwell()`; branch shares weighted by survival in `schedule()`; `arrive()` takes a time and places by stage, slot or phase |
| Conditional splits | `Groups.init()` already normalizes a split of "a dimension of some states only, given the levels of others" (`strat.cond`); the error `Only one split= can cover several dimensions at once` is in the same function | The `strat.cond` path for any dimension, in dependency order; agents draw per dimension (`Groups.draw_joint()` and the initial split in `Agents.init_post()`); a cycle error |
| A module of properties | `ss.Disease` (`diseases.py`) with transfers via `stratify()`; `parse_stratify()` (`dimensions.py`) refuses transfers on the sim; D23's note for a stratified sim with no declared module | `ss.Properties`, a declared module whose one implicit state is everyone alive; `presentation.py` words it as properties |
| Population results by group | `Sim` results `n_alive`, `new_deaths` (agents only) | Kept by group when the sim is stratified, under the budget in `Sim.kept()` and the thinning that follows it (`sim.py`) |
| Occupied groups | `Groups.group_raw()` and `level_dtype()` (`dimensions.py`); `group_coords`, `group_split`, `birth_split` over the grid in `Groups.init()`; every `np.bincount(..., minlength=n_groups)` in `solvers/agents.py` (counts, `flow_groups`, `live_total`) | A compact index for agents, kept in `Groups`; `n_groups` for agents becomes the occupied count; results by group with a code table (`results.py`, `Result.by()`/`select()`) |

**Not added.** `init=dict(I=..., since=...)` (the review's spelling): `since` would be a state's name, as the probe shows. `ss.People(df)`: `ss.People(n)` takes a number, and two meanings for one argument hide which is meant. Arrays in `init=`. A `scale=ss.geo(pop=...)` argument (the review's sketch): the sim's existing `total_pop=` does that. A Python function in `split=` (an expression covers TLO's linear models; a function would be agents-only and unprintable).

**Spec changes.** 03-states-and-transitions.md: `init=` values per person, `since=` and its rule. 05-dimensions.md: `from_df`, splits given other dimensions and their order, `ss.Properties`, the occupied-groups rule for agents, population results by group. 09-results.md: `n_alive` by group. 12-backends.md: `since=` under compartments, by dwell form; the table as the groups' split. 14-model-spec.md: the spellings above. 17-vocabulary.md: `ss.Properties` (one new counted name), `from_df`, `since=`. 18-conformance.md: the contracts below.

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 1 | `init=` per person as agents: expressions, conditions, functions; the overlap check; the error for arrays; the first result point right (TL1's stale index 0) | S | `solvers/engine.py`, `solvers/agents.py`, `transitions.py` |
| 2 | `ss.People.from_df()` as agents: columns to arrays, `sex` and `age`, `weight=` and drawing rows, string labels, newborn defaults and their note, the round trip with `to_df()` | M | `people.py`, `sim.py` |
| 3 | `from_df` under compartments: the joint weighted split, `init=` per group from the table, the note for unused columns | M | `sim.py`, `dimensions.py`, `solvers/engine.py` |
| 4 | `since=` as agents: dated entry, the residual dwell, branches weighted by survival, the note beyond the longest dwell; clocks and `.dwell` read it | M | `solvers/agents.py`, `transitions.py`, `distributions.py` |
| 5 | `since=` under compartments: Erlang stages, conveyor slots, hyperexponential phases, from a value, a distribution, or the table | M | `solvers/compartmental.py`, `solvers/crossing.py`, `kernels.py` |
| 6 | Splits given other dimensions (TL5): the keyed `ss.by`, expressions in `split=`, any number in dependency order, the cycle error; agents draw per dimension | M | `dimensions.py`, `parameters.py`, `solvers/agents.py` |
| 7 | `ss.Properties`, its results and `describe()`; sim results by group as agents | M | `diseases.py`, `presentation.py`, `sim.py`, `results.py` |
| 8 | Occupied groups as agents (TL7): 64-bit codes, the compact index, per-occupied arrays, results by occupied group decoded by `.by()`/`.select()`, the compartmental note | L | `dimensions.py`, `solvers/agents.py`, `results.py`, `sim.py` |
| 9 | `describe()`, `to_spec()`, `to_code()`, the spec sections, the user guide page on starting from data, CHANGELOG | S to M, spread over the steps | `presentation.py`, `spec.py`, `v4_design/design/*.md`, `docs/user_guide/` |

Steps 1, 2 and 4 make TLO's start declarable as agents; steps 3 and 5 give it to the compartmental solvers; step 6 makes Lifestyle short; step 8 is independent and is the only L. About two weeks of agent work in all, step 8 being about a third of it.

**Tests**, on numbers a user sees:
- TL1: the probes in `design_probes/p1_people.py` pass. `init=dict(I=P.hv_inf)` with 694 people marked gives `n_I[0] == 694` (TL1's stale point), under agents and the ODE. Two overlapping conditions raise with the count.
- `from_df`: 20,000 rows with a `district` column of 3 levels and a `tob` column. Agents keep every value. Under the ODE, `n_S + n_I` by district and `tob` at index 0 equal the table's counts, to 1e-9. With `weight=`, agents drawn from 1,000 weighted rows reproduce the weighted shares within 3 standard errors. `from_df(sim.people.to_df())` gives the same ages, sexes and properties.
- TL2: the TLO HIV case (`infections/probes/probe11_ti_exit.py`'s setup): AIDS onsets in the first year within 3 standard errors of 34 of 380 as agents. A known answer: a Weibull dwell, everyone with `since = s`, P(exit by `s + t`) equal to `1 − S(s + t)/S(s)` within 3 standard errors (agents) and to 1% (ODE with Erlang stages fitted to the same mean and variance, against its own exact value).
- Branch weighting: `P -> C | R` with `p = [0.5, 0.5]` and dwells of mean 1 and 10 years; with `since = 5 years`, the share leaving to `C` among those still in `P` is `S_C(5)/(S_C(5) + S_R(5))`, as agents and ODE.
- TL69: a fixed 40-week gestation with `since=ss.uniform(0, 40 weeks)` gives births at a constant rate over the first 40 weeks, not a gap and then a wave.
- TL5: urban 0.83/0.17, wealth given urban, and `tob` given sex and wealth by an expression: the shares within 1% as agents, and exact under the ODE. A cycle raises.
- TL10: `ss.Properties` with two transfers: `describe()` has no "infected"; `res.lifestyle.n.by('tob')` agrees between agents and ODE within 1%. `sim.results.n_alive.by('district')` works as agents.
- TL7: 20,000 agents with 2^22 groups use under 400 MB at peak (3,622 MB today), and their results agree with today's run run for run. 32 districts with TLO's 8 declared Lifestyle properties, sex and age run as agents (`v4_lifestyle_declared.py` with districts), with prevalences within 0.5 percentage points of TLO, as without districts.

**What the ports delete.** TLO: every post-`init()` write of states and dimension levels, and the three engine arrays for time spent (`ti_exit_I`, `mult_I`, `exit_I`); the nested loops of `v4_lifestyle_declared.py` (about 45 lines) and its fake disease. FPsim: the coded draws of starting parity, method and education, if FPsim starts from DHS records. HPVsim: nothing until its declared port seeds CIN.

## Out of scope and dependencies

- **Module states in a saved population.** `from_df` restores people's own properties. A checkpoint of a whole sim (each disease's states, scheduled exits, networks) is another feature: `sim.save()`/`ss.load()` already do it.
- **Newborns inheriting from the mother** (TL34's `default=ss.inherit()`, and TL28 with MA18's rest): `from_df` gives newborns defaults until then, with a note.
- **Properties in rates that are not dimensions** (TL9): done as agents (behavior steps 3 and 4). This design adds only their use in `init=`, `since=` and `split=`.
- **Linear models** (TL11, done; TL14's odds form; TL15's mean-matched parameters): the regression in `split=` uses the behavior design's expressions.
- **A starting state that is the steady state** (time since entry drawn from the state's equilibrium, the backward recurrence time): useful for seeds whose time is unknown, but a second meaning of `since=`; see decision 7.
- **The burden of disease** ([burden_of_disease.md](burden_of_disease.md)) reads people's age, sex and cause of death, and benefits from occupied groups when results are by many dimensions.

## Decisions for Cliff

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | How a table becomes people (TL1) | (a) `ss.People.from_df(df, weight=)`; (b) `ss.People(df)`, a second meaning of the first argument; (c) `ss.people(df)`, beside `ss.people('Kenya')` | (a): it mirrors `People.to_df()` and pandas' `from_dict`, and one argument keeps one meaning. (c) is close, but `ss.people()` is for named data |
| 2 | Columns and weights | (a) `age`, `female` (and `sex` read as `female` with a note) are people's own; every other column is a property; `weight=` names a column, and agents are drawn from rows by it; (b) a fixed column `scale` for weights; (c) every column a property, with ages and sex mapped by the user | (a): no magic column names besides the two People already has, and the weight is named where it is used |
| 3 | A table under a compartmental solver | (a) the joint weighted counts are the groups' split, `init=` and `since=` are averaged per group from the rows, and unused columns are noted; (b) only ages, as today; (c) `CapabilityError` | (a): an initial condition is only counts, so a table gives it exactly, and this is the compartmental frameworks' own convention (SimInf's `u0`, odin's `initial()`) |
| 4 | `init=` per person | (a) an expression or condition of people's properties is each person's chance (a condition is exact), and a function is per agent; arrays raise and point to `from_df`; (b) also accept arrays of length `n_agents` | (a): arrays have no names and break under births and deaths (TL8); a column is the same data with a name |
| 5 | The spelling of time already spent (TL2) | (a) `since=` beside `init=` (a dict by state on the disease, a keyword on `ss.State`); (b) inside `init=` (`init=dict(I=..., since=...)`, the review's); (c) a pair, `init=dict(I=(share, since))`; (d) an initial value for `I.time` | (a): it mirrors `init=` exactly; (b) collides with state names; (c) is opaque; (d) is the same thing, but there is no slot for initial values of expressions |
| 6 | What `since=` means for a scheduled exit | (a) the dwell given that it is longer than `since` (TLO's rule), with branches weighted by survival; (b) a fresh dwell counted from the past entry, so those drawn shorter leave at once; (c) a fresh dwell from now (today) | (a): it is the only one that keeps the dwell distribution of the state; (b) sends everyone with a short draw out in the first step; (c) ignores `since` except in clocks |
| 7 | When `since=` is not given | (a) 0, as today (and as EpiModel and vivarium start seeds); (b) the steady state of the state (the backward recurrence time), which changes results; (c) (a), plus a named value for the steady state later | (c): keep today's results, and add the steady state when a second model asks |
| 8 | Splits given other dimensions (TL5) | (a) the keyed `ss.by(urban=[...])` and expressions in `split=`, any number, applied in dependency order; the `dims=[...]` table stays a joint distribution (TL4); (b) read every table over several dimensions as the share of this one given the others (the port's rule in `merge_tradeoffs.md`); (c) keep one such split, with `states=` as the workaround | (a): the keyed form has one reading, so no table changes meaning, and expressions give TLO's linear models in one line |
| 9 | A module of properties (TL10) | (a) `ss.Properties(transfers..., name=)` with `.stratify()`; (b) `ss.Disease` with no states (`kind='condition'`, TL53); (c) transfers on `sim.stratify()`, with no module | (a): it gives the results and `describe()` a home, and says what it is. (c) has no name to put results under, and needs a declared module anyway under compartments |
| 10 | Occupied groups as agents (TL7) | (a) always, as agents, with `.by()` showing every level (zeros where empty); (b) only above some grid size; (c) not now (only the 64-bit codes) | (a): one code path, and it removes the grid's cost everywhere; (b) means two layouts of results to test |
| 11 | Population results by group as agents (TL10) | (a) `n_alive`, `new_deaths` and births by group whenever the sim is stratified, under the results budget; (b) only through `ss.Properties`' `n` | (a): cheap, and what a modeler expects of `sim.results.n_alive.by('district')` |
