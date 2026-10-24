# Behavior vocabulary, and dimensions over people's properties

Design H4 for Phase H ([implementation_oct18.md](../implementation_oct18.md), fourth and fifth bullets), for Cliff to decide before any code. Findings: FP24 to FP28, FP30, HP19 (behavior); FP19, FP20, FP39, D15 (dimensions over existing properties); FP21, FP22, FP35 (units of bare numbers). Evidence: [status_oct18/fp.md](../status_oct18/fp.md), [fpsim_review.md](../fpsim_review.md) sections 4 and 5, `fpsim:v4port/`, and probes of the current tree in `wt/scratch-h4/` (`probe.log`, `probe2.log`, `probe3.log`), all of which reproduce the status rows except FP39's boolean arrow, which now works.

**Revision, 2026-10-19** (H0 of [oct19/implementation.md](../../oct19/implementation.md), after Cliff's `[CK:]` notes, which are kept below as the record):

- **Counters**: `move=dict(parity='+1')` is withdrawn (no string expressions). Section 3 now has five worked examples, each in four spellings: `move='parity'`, `move=dict(to='parity')`, `move=dict(parity=1)` on a counter, and a separate `add=dict(parity=1)`. The recommendation is `add=`, but the choice is open (decision A at the end).
- **`ss.ppl`** replaces `ss.People` as the lazy reference to people's properties (`ss.ppl.urban`), as Cliff asked. On a `People` object the arrays are unchanged.
- **The FP26 example** is rewritten so each line holds one idea, with named terms. The FP27 transitions are built in a plain loop, not a `**{...}` comprehension.
- **Decisions A to C answered, and steps 3 to 8 done** (second revision, 2026-10-19): counters are `increment=` (section 3), `ss.People.age` as a lazy reference is removed in favor of `ss.ppl.age`, and `increment=` is for counters only. See the "Done" notes in the plan.
- **Steps 1 and 2 are done** (see the plan): FP19, FP20, FP21 (with the warning Cliff asked for: `ss.UnitInferred` when a bare `dt` takes the years of `start`/`stop`), FP22 (a note), FP35, `ss.histogram(unit=)`, and FP39's empty-level note. Moved: D15 to steps 3 and 4 (a keyed `ss.by(sex=...)` is needed), FP30's distributions by level to step 4, and FP39's counts after aging to Stage 2 of the plan.

## Summary

**The problem.** FPsim's declared contraceptive choice agrees with FPsim (mCPR within 3 to 6%, births within 0.6%) at a fifth of the lines (92 against 468), but only with a function in every slot that matters: a logistic regression of eight covariates, a 10 × 9 switching matrix per age group, and durations of use whose scale depends on age. So it runs only as agents, `describe()` prints `<lambda>` ten times per transition, its calibrated coefficients can't be `ss.par`s (and `ss.change` on them silently does nothing, FP14), and `update_methods()` becomes about 70 `ss.change` calls. Parity needs one copy of each delivery branch per level (3 × 8) or a per-step function that can't tell twins from singles; time since the last birth is read from the engine's `ti_PP`. Underneath are silent errors in how dimensions meet the properties people already have: `stratify(urban=[False, True])` sets everyone rural (share urban 0.000, People set 0.9); `wealthquintile=[1, ..., 5]` counts [800, 400, 400, 400, 0] of 2,000; a bare `dt=1/12` becomes 1/12 of a month (2,881 points instead of 241); a duration function returning 6 (months) is read as 6 years; `ss.choice` of durations and `dur=ss.by(urban=[h0, h1])` crash.

**The idea.** The spec already says a dimension is a property of people seen from the compartment side (05, "Dimensions are agent properties"). Take that literally, and nearly all of this is reading and writing properties with the vocabulary that exists:

- **Read** a property in any rate, `p=`, or distribution parameter as a per-person term (`ss.ppl.urban`, and a state's membership `~never`), as conditions already can; look values up by a property with `ss.by`, whether or not it is a dimension. A regression is then an expression, and a table is an `ss.by` with a column for the destination.
- **Write** a property on an event with `move=`, which already changes a dimension's level on a transition: a counter is a relative move, `increment='parity'` (section 3); time since an event is a dimension of kind `clock` that the event sets to 0 (`move=dict(since_birth=0)`), as age is a clock that birth sets.
- **Group** by a property that already exists without overwriting it: people keep their values and levels are bins of them.

One new `ss.` name, `ss.ppl` (revision: before, none). New: a few values and a column (`kind='clock'`, a relative move for counters, a `to` column in an `ss.by` table), `expit` among the functions a rate may use, and `ss.ppl.<any property>` as a lazy reference (today only `age`, `female`, `alive`) [CK: ok but maybe `ss.ppl` instead of `ss.People`, the latter looks like it's creating a class rather than a helper function, plus it's much harder to type]. The reviews' proposed `ss.logistic`, `ss.spline`, `ss.table`, `ss.time_since`, and `ss.value` are not added. [CK: ok]

```python
P = ss.ppl                                     # lazy references to people's properties, as in conditions today
never, none = ss.states(2)
on = {m: ss.State(m) for m in methods}         # one state per method
ever = ~never                                  # a condition in a rate is 1 where it holds, 0 elsewhere

# FP26: the probability of use is an expression, so describe() prints it and calibrate() fits its ss.pars
b = sc.objdict({k: ss.par(v, name=f'b_{k}') for k,v in coefs.items()})   # one ss.par per coefficient: intercept, ever, urban, ...

age_spline = ss.by(age_spline_table)      # the age spline, evaluated at single years of age (columns: age, value)
age_ever   = ss.by(age_ever_table)        # its interaction with ever-use, the same way
year_trend = ss.timeseries({2000: -12, 2050: 38}, mode='linear')   # years since 2012, a straight line

ever_terms   = b.ever*ever + ever*age_ever                                    # ever-use, and its effect by age
person_terms = b.urban*P.urban + b.parity*P.parity + b.wealth*P.wealthquintile
logit_use    = b.intercept + age_spline + ever_terms + person_terms + b.trend*year_trend
p_use        = expit(logit_use)

# FP27: the switching matrix is a table with a column for the destination, normalized over it when used
def mix(src):
    rows = switch[switch['from'] == src.name]     # this method's rows of the matrix
    return ss.by(rows.drop(columns='from'))       # columns: age, to, value

w = ss.by(to=[ss.par(1.0, name=f'w_{m}') for m in methods])   # calibrated weights per method

# FP30: a duration of use by age, in months
def dur_use(src):
    scale = np.exp(base[src.name] + ss.by(age_factor[src.name]))
    return ss.weibull(c=c[src.name], scale=scale, unit='month')

# Each woman re-decides when her spell ends (never-users likewise, to [never, *methods])
choices = {}
for src in [none, *on.values()]:
    p_methods = p_use*mix(src)*w                  # a row over the methods, normalized over them
    choices[f'from_{src.name}'] = ss.Transition(src, [none, *on.values()], dur=dur_use(src), p=[1 - p_use, p_methods])
contra = ss.Disease(never, none, *on.values(), **choices)

# FP24, FP25 (when a pregnancy is declared): parity as a counter, months since the last birth as a clock
sim.stratify(parity=list(range(8)))                                            # people.parity from the data is kept (FP20); 7+ is open-ended
sim.stratify(since_birth=ss.months([0, 1, 2, 3, 6, 12, 24]), kind='clock')    # plus a level "none" for women with no birth
delivery = ss.Transition(P3, PP, dur=dur_preg, p=1 - p_twins, increment='parity', move=dict(since_birth=0))   # a counter (section 3)
twins    = ss.Transition(P3, PP, dur=dur_preg, p=p_twins,     increment=dict(parity=2), move=dict(since_birth=0))
```

**Decisions for Cliff**

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | How a regression is written (FP26) | (a) an expression over people's properties, with `ss.by` tables for splines; (b) `ss.logistic(intercept=, terms=, age=ss.spline(...))` | (a): no new names, and everything an expression has (printing, `ss.par`, modifiers, group evaluation) comes free [CK: ok] |
| 2 | Tables and a row-valued `p=` (FP27) | (a) an `ss.by` table with a `to` column, normalized over the destinations it names; (b) `ss.table(..., axes=[...])` with a declared renormalization | (a) [CK: ok] |
| 3 | Counters (FP24) | (a) a dimension with a relative move, `move=dict(parity='+1')`; (b) a count on the event (`delivery.count`); (c) `ss.counter` | (a): exact under every solver, and gives results, `adjust=`, and `ss.by` by parity for nothing [CK: i don't like `parity='+1'`, i'd prefer `dict(to='parity')` or just `move='parity'` for something like this, but maybe i need to see more examples] **Revised:** examples in four spellings in section 3; open as decision A |
| 4 | Time since an event (FP25) | (a) a dimension of kind `clock` set to 0 by `move=`; (b) an expression `ss.time_since(tr)`, agents only | (a): age and a product's time since delivery are already clocks of this kind; it runs under every solver [CK: ok] |
| 5 | A value by a property that is not a dimension (D15, FP26) | (a) agents under `'auto'`, `CapabilityError` under an explicit compartmental solver, naming `sim.stratify(...)`; (b) stratify by it implicitly | (a), as a distribution on `beta` is treated today; dimensions multiply compartments, so they are never added unasked [CK: ok] |
| 6 | Numeric levels (FP19) | (a) always lower bin edges, and a person's property holds the value, never the level's index; (b) integer labels with `kind=` | (a) [CK: ok] |
| 7 | An existing property (FP20) | (a) kept and binned; `split=` for it raises; (b) overwritten from `split=` (today) | (a), which is what spec 05 already says [CK: ok] |
| 8 | Units of bare numbers (FP21, FP22, FP35) | (a) a bare `dt` has the unit of bare `start`/`stop` (years when they are years); a number from a function in a distribution takes the distribution's `unit=`; `ss.choice` and `ss.histogram` take `unit=` for their values; (b) warnings only | (a), with a note where the unit was guessed [CK: ok] |
| 9 | What FPsim declares | (a) contraceptive choice and education declared; pregnancy and conception stay code, writing the properties the declared parts read; (b) declare everything; (c) declare nothing yet | (a); the reasons are in [What FPsim should declare](#what-fpsim-should-declare-and-what-stays-code)  [CK: ok] |

## Use cases

One per finding. "Today" is FPsim's Starsim 3 code or the v4 port's workaround; "proposed" is the v4 spelling under this document.

| ID | What the package needs | Today | Proposed | Simpler? |
|---|---|---|---|---|
| FP24 | Parity: a count of births (twins +2, stillbirths +1) that conception and the regression depend on | `FPmod.parity += 1` in `progress_pregnancies()` and again for twins; the declared port's `update_parity()` (15 lines) diffs `ti_PP`/`ti_SB` stamps each step and counts twins as 1 | `stratify(parity=list(range(8)))` and `move=dict(parity='+1')` on each delivery branch (`'+2'` on twins) [CK: how about just parity=1 instead of '+1'? the string expressions freak me out] | Yes: 15 lines and two bookkeeping arrays go; twins are right |
| FP25 | Sexual activity, LAM, and birth-spacing preference by months since the last live birth, which outlives the postpartum state | Functions reading `ti_live_birth` or the engine's `ti_PP` (`tsb = d.ti - d.ti_PP[uids]`, and `np.where(has_lb, ...)` for women with none) | `stratify(since_birth=ss.months([...]), kind='clock')`, `move=dict(since_birth=0)` on delivery; lookups `ss.by(table)` with a row for `None` (no birth yet) [CK: ok] | Yes, and no engine internals |
| FP26 | The probability of use: logistic in an intercept, ever-use, urban, parity, wealth quintile, a 3-knot age spline and its interaction with ever-use, education bins, a year trend; the intercept and trend are calibrated | `StandardChoice.get_prob_use()` (35 lines), and in the port a 13-line function closed over the coefficients | An expression: `expit(b.intercept + b.ever*ever + ... + b.trend*trend)`, with the spline as `ss.by(age_table)` precomputed from the knots (FPsim's knot coefficients are data, not calibrated) [CK: ok, but please try to keep the code interpretable and not insanely dense] | Yes: about 6 lines; `b.intercept` is an `ss.par` that `calibrate()` and `ss.change` reach |
| FP27 | Method choice: a row of the switching matrix for the woman's age group and source method (and postpartum event), times calibrated weights, renormalized | `choose_method()` and `choose_method_post_birth()` (60 lines); in the port, ten lambdas per transition sharing a per-step cache (`branch()`, 12 lines), because a function returning the row crashes | `p=[1 - p_use, p_use*mix(src)*w]`: the second entry has a `to` axis over the method states, normalized over them each time it is evaluated [CK: ok] | Yes: the cache and the lambdas go; `ss.change(mix_par, to=new_table)` replaces about 70 scalar changes |
| FP28 | "Start a method from the usual mix" in an intervention; a modifier on every transition into a method | One delivery per method with thinned coverage; 20 modifiers | The first: a delivery's move to several states takes `p=` with the same `to` rule (spelling with h1). The second: a modifier over a set of transitions is h2's [CK: ok] | Partly here |
| FP30 | Education objective: a histogram of years that depends on urban; a dwell in a unit | A coded `set_objectives()` with two `ss.histogram`s; `dur=ss.by(urban=[h0, h1])` crashes; `ss.histogram(unit=)` refused | `ss.by(urban=[h0, h1])` of distributions, drawn per person from their level's; `ss.histogram(..., unit='year')`  [CK: ok] | Yes: 12 lines of code go |
| HP19 | Per-person values written on events (immunity as a running maximum on clearance, drawn; age at CIN onset frozen at entry; years of schooling accumulated) | Coded connectors and function modifiers, agents only | Not here, except the two writes that are counters and clocks. The general write (an expression, a maximum, a draw, frozen on entry) is h1's primitive, spelled as a value of `move=` if h1 agrees ([dependencies](#out-of-scope-and-dependencies)) [CK: ok, but yes in general some things will be agent only and that's fine] | Partly here |
| FP19 | `wealthquintile` levels 1 to 5, multiplied by a coefficient in the regression | `stratify(wealthquintile=[1, ..., 5])` stores indices 0 to 4 and groups by edges 1 to 5: [800, 400, 400, 400, 0] | People hold the value (1 to 5); levels are bins of it  [CK: ok] | Fix |
| FP20 | `urban` and `wealth` set from data on People, then used as dimensions | `stratify(urban=[False, True])` overwrites them from the split: share urban 0.000 | Kept and binned; `split=` with an existing property raises  [CK: ok] | Fix |
| FP39 | `stratify(age=[0, 15, 120])` makes an empty `120+`; counts by age use ages from before the step's aging; `move=dict(ever='True -> False')` | As stated (the boolean arrow now works) | A note for a level that is empty by construction (an age edge of 100 or more, since ages run to 100); counts by age after aging  [CK: ok] | Fix |
| D15 | mCPR among women 15 to 49; death rates by sex, without declaring a sex dimension | `ss.death(rate=ss.by(sex=...))` raises "The model is not stratified by sex" | A value by `sex` or `age` reads `people.female` or `people.age` as agents; a compartmental run needs the dimension (decision 5)  [CK: ok] | Fix |
| FP21 | Durations in months, a step of 1/12 year, dates in years | `dt=1/12` with typed months becomes 1/12 month: 2,881 points instead of 241 | A bare `dt` is in years when `start`/`stop` are years  [CK: ok, but print a warning] **Done:** `ss.UnitInferred` warns when a bare `dt` takes the years of `start`/`stop` | Fix |
| FP22 | Duration callbacks that return months | A bare 6 from a function is read as 6 years (no one of 1,000 leaves within 2 years at a mean meant to be 6 months), silently | The number takes the distribution's `unit=` (`ss.lognorm_ex(mean=fn, std=..., unit='month')`); with no unit anywhere, a note naming the function  [CK: ok] | Fix |
| FP35 | `ss.Pregnancy`'s own `dur_pregnancy` form: `ss.choice` of durations | Crashes (`loop of ufunc does not support ... months`); `ss.choice(unit=)` refused | `ss.choice(a=[ss.months(9), ...])` and `ss.choice(a=[9, 10], unit='month')` both work: the unit applies to the values, which is exact  [CK: ok] | Fix |

Where the proposal does not make a package simpler: the conception rate. Written with this vocabulary it is an expression of twelve factors (age fecundity, personal fecundity, nulliparity, exposure by age and parity, activity by age and months postpartum, spacing preference, LAM), about the same length as the port's 30-line function, and it still runs only as agents (personal fecundity is continuous per person; the tables are by single year of age). What it gains is `describe()` and an `exposure_factor` that `calibrate()` reaches. That is part of why pregnancy stays code (decision 9).

## The proposal

### 1. People's properties in expressions

Rates, branch probabilities, and distribution parameters are expressions (`expressions.py`); conditions (`conditions.py`) already have per-person leaves (`people.age`, a state's membership) but cannot appear in a rate (`as_expr` raises "A condition cannot be used in a rate"). The change:

- `ss.ppl.<name>` is a lazy reference for any name, not only `age`, `female`, and `alive`, checked when the sim is built (an unknown name is `UnknownState` with a suggestion). It is a small helper object, not the `People` class, so it reads as a reference and is short to type (Cliff). On a `People` object, `people.<name>` stays the array. `ss.People.age` and the other two are removed (decision B): `ss.People` is the class only.
- A property is a leaf of an expression, worth the person's value. A condition in a rate is 1 for the people it holds for and 0 for the others, so `0.3*(P.age > 65)` and `b*ever` work. The one exception keeps today's error: a condition made only of bare states joined by `|` (`0.3*(Ia | Is)`), which almost always meant a sum of states for exposure.
- `expit` (SciPy's, a NumPy ufunc) joins `exp`, `log`, and the others as a function a rate may use, since it is how a regression is written; `1/(1 + np.exp(-x))` already works and prints the same.
- A property of another module is named through the module, as its states are (`edu.attainment`, `hiv.cd4`).

A regression is then an ordinary expression: `describe()` and `equations()` print it, its coefficients are `ss.par`s with bounds that `calibrate()` fits, `ss.change` and `ss.modify` reach them during the run (which fixes FP14 for this form), and it is evaluated on groups when every property in it is a dimension. A function in a slot remains, for what is not arithmetic.

### 2. Values by a property: `ss.by` over properties

`ss.by` today needs a dimension: it gives one value per level and raises otherwise. Two changes:

- **A table over a property carries its own levels.** In `ss.by(table)`, a column with numbers holds lower bin edges (the last bin open-ended, as for dimensions), and a column of booleans or labels holds the property's values. So FPsim's tables by single year of age are `ss.by(dict(age=range(15, 50), value=vals))`, whatever the model's age groups. A row whose key is `None` gives the value for people with no value (no level), such as women with no birth yet for a clock; without one, those people get the mean over the levels weighted by the split, as spec 05 already says.
- **A value by a property that is not a dimension** reads the property per person as agents. Under `solver='auto'` it makes the model run as agents, with a note (`rate depends on people.urban, which is not a dimension; stratify by it to run as compartments`); under an explicit compartmental solver it raises `CapabilityError` with the same fix. This is the rule for a distribution on `beta` today, and it answers D15: `ss.by(sex=[...])` reads `people.female`, and `ss.by(age=[...])` (a list, so no edges) points to the table form or to `stratify(age=...)`.
- In a compartmental run, a table finer than the model's dimension (single years against five-year groups) is averaged over each group, weighted by the population in it, with a note, as contact matrices are aggregated (`MatrixAggregated`).
- `ss.by` values by two different dimensions multiply as their outer product (`mix*w`, a table by age and `to` times weights by `to`), which the keyword form already means; today `_op` returns `NotImplemented`.

### 3. Counters: `increment=`

`move=` on a transition (and on a delivery and a transfer) changes a dimension's level as well as the state, to an absolute level (`move=dict(vaccinated=True)`, `ss.transfer(art=True, ...)`) or along an arrow between two levels (`move=dict(strain='A+B -> A')`). A counter needs a relative change: up or down `k` levels from wherever each person is. That is a separate keyword, `increment=`, on `ss.Transition` (and `ss.progression`), `ss.Delivery`, and `ss.transfer`: `increment='parity'` adds 1, and is the same as `increment=dict(parity=1)`; the dict form gives other amounts (`dict(parity=2)` for twins, `dict(doses=-1)`) and several counters at once (`dict(parity=1, births=1)`). `move=` always says where people go, and `increment=` how far; a reset to a level is the existing `move=dict(parity=0)`. `increment=` is for counters only (decision C); other per-person writes (a running maximum, a value frozen at entry) are a later design.

On a counter whose levels are lower edges `0, 1, ..., 7` the levels are the counts, `7+` is open-ended, and so a move up from it stays in it (as agents, the value keeps counting: 9 is in `7+`). A move below the first level stays at the first. Several transitions may move the same counter by different amounts (singles 1, twins 2, stillbirths 1), which is what FPsim needs and what a per-step function cannot do. People start at the first level, as a dimension that people are moved along already starts them (05). A counter is any dimension with integer edges; nothing else marks it. In a compartmental run, a counter whose levels are not one apart (`[0, 1, 2, 5]`) takes the people in each level to be at its lower edge, with a note.

Other models that need it: TB's previous treatments, a vaccine's doses received, STIsim's lifetime partners, malaria's number of infections.

The dimensions in the examples: `stratify(parity=list(range(8)))`, `stratify(doses=[0, 1, 2, 3])`, `stratify(episodes=list(range(6)))`, `stratify(schooling=list(range(13)))`, and `stratify(immunity=[0, 1, 2, 3])`.

```python
# 1. Parity: singles add 1, twins 2, stillbirths 1; each birth also resets a clock
delivery   = ss.Transition(P3, PP, dur=dur_preg, p=p_single, increment='parity', move=dict(since_birth=0))
twins      = ss.Transition(P3, PP, dur=dur_preg, p=p_twins,  increment=dict(parity=2), move=dict(since_birth=0))
stillbirth = ss.Transition(P3, PP, dur=dur_preg, p=p_still,  increment='parity')

# 2. Doses received: each round of a campaign adds a dose
campaign = ss.Delivery(increment='doses', coverage=0.6, at=[2025, 2026, 2027])

# 3. Episodes of an illness: each recovery adds one (malaria's prior infections)
recovery = ss.progression('I -> S', dur=dur_inf, increment='episodes')

# 4. Years of schooling: a year added for each year in school, and dropping out
school_year = ss.transfer(increment='schooling', dur=ss.years(1), states=['enrolled'])
dropout     = ss.progression('enrolled -> left', rate=p_drop)

# 5. A decrement: immunity drops one level at a time as it wanes, and recovery sets it back to the top
waning   = ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5))
recovery = ss.progression('I -> R', dur=dur_inf, move=dict(immunity=3)) # A set, as before
```

(A transfer with `dur=` runs at the constant rate of its mean, X1; the exact yearly step is Stage 4's.)

Four spellings were considered: (a) `move='parity'`, with `move=('parity', 2)`; (b) `move=dict(to='parity', by=2)`; (c) `move=dict(parity=2)` on a dimension declared `kind='counter'`; and (d) a separate keyword, `add=dict(parity=2)`. Cliff chose (d), renamed `increment=`, with a bare name for adding 1 (decision A). In the model spec it is `"increment": {"parity": 1}` on the transition, transfer, or delivery.

### 4. Clocks: time since an event

A dimension of kind `clock` holds, for each person, the time since it was last set, and advances with time, as age does. It is declared like any dimension with duration edges, and an event sets it with `move=` (to 0, or another value):

```python
sim.stratify(since_birth=ss.months([0, 1, 2, 3, 6, 12, 24]), kind='clock')   # levels none, 0, 1, 2, 3-5, 6-11, 12-23, 24+ months
delivery = ss.Transition(P3, PP, dur=dur_preg, move=dict(since_birth=0))
active_pp = ss.by(dict(since_birth=[None, 0, 1, 2, 3, 6, 12, 24], value=[...]))   # None: no birth yet
```

- People who have never had the event have no value, and in a compartmental run are in a generated first level `none`. A clock whose property People already has (given from data) starts from it.
- Age is the clock that birth sets; a product's `has_<product>` levels of time since delivery (`products.py`, `init_stages()`) are a clock that delivery sets, and its machinery is what a clock reuses. `after=` and `before=` stay as they are: they are the time in the source state, which a clock is not needed for.
- A clock's edges need not be whole steps as agents; in a compartmental run they are rounded to steps (`ss.LargeStep` note), and people move along a conveyor of whole steps, as a fixed delay does (03), which is exact.
- Kinds are inferred from the value (05). A list of durations does not by itself say clock (it could be a binned duration property set by code), so `kind='clock'` is given. The name of the event is not part of the dimension: whatever moves it to 0 is its event, and `describe()` lists them.

### 5. Tables with a destination column, and a row-valued `p=`

On a transition with several destinations, `p=` is a list with one entry per destination (03). An entry may instead have a `to` axis (an `ss.by` table with a `to` column, or an expression containing one), whose labels are destination state names. It covers those destinations, and it is normalized over them each time it is evaluated, so weights and other parameters inside it are renormalized as FPsim does. The other entries are ordinary probabilities, and the whole must still sum to 1 (`BadBranchSplit`). So `p=[1 - p_use, p_use*mix(src)*w]` is: `none` with `1 - p_use`, and each method with `p_use` times its normalized share. A row that sums to 0 for some group raises `BadBranchSplit` naming the group. `describe()` says the entry is normalized.

A single function given as `p=` for several destinations may return one column per destination (today it crashes, FP27's S part).

With the table as one `ss.par` (`mix_par = ss.par(ss.by(switch), name='switching')`), an intervention that changes the matrix is `ss.change(mix_par, to=new_table, at=2025)`: one call, not 70.

A table with a `from` column is not proposed: each source has its own dwell (FPsim's durations of use differ by method), so each is its own transition, and the table is filtered in pandas.

### 6. Distributions by level, and distributions with units

- `ss.by(urban=[h0, h1])` of distributions (FP30) is a distribution: as agents, each person draws from their level's, using their one uniform draw from the slot's stream (so common random numbers are kept); in a compartmental run, each group gets its own Erlang stages for a dwell, with one number of stages for all groups (the largest of their matched numbers), noted.
- A distribution parameter may be an expression with properties and tables (`scale=np.exp(base + ss.by(age_factor))`), evaluated per person as agents and per group in a compartmental run; today it has to be a `fn(module, sim, uids)`.
- `ss.choice` and `ss.histogram` accept typed values (`a=[ss.months(9), ss.months(10)]`) and `unit=` (FP35, FP30). The unit applies to the values, before the draw, so it is exact; the refusal today comes from scaling after the draw, which these two cannot do.

### 7. Dimensions over properties people already have

The rules spec 05 states, applied everywhere, and two it does not yet state:

- **A property People already has when the sim is built is kept** (given in `extra_states`, by a named population, or set from data by a module before `init_vals()`). A dimension over it bins its values; `split=` for it raises (`BadSplit: "urban" is given by the people, so it has no split`); its split for a compartmental run is taken from it, as for age from a named population (`split_from: 'people'`). A property set later by a module written as code is that module's, which calls `sim.groups.groups_changed()`, as today.
- **A person's property holds a value, never a level's index.** A list of numbers is always lower bin edges (the last open-ended); a dimension that creates a property gives each person the lower edge of the level drawn. Labels that are strings are stored as their position, as now, and a boolean as a boolean. So `wealthquintile=[1, ..., 5]` stores 1 to 5 and groups them right, and `b.wealth*P.wealthquintile` multiplies the quintile, not its index.
- In an expression, a property that is a dimension is worth, in a compartmental run: 0 or 1 for a boolean; the lower edge for a counter (a counter is integer-valued, so this is exact below the top level; the top level `7+` is taken as 7, with a note); the middle of the bin for a binned continuous property, with a note. A dimension with string labels has no number, so using it in arithmetic raises, pointing to `ss.by`.
- A note when a level is empty by construction: an age edge of 100 or more (ages run to 100, 05), or a level of an existing property that no one has.
- Counts by age group are taken after the step's aging (FP39's 164 under 10 with nobody under 10).

### 8. Units of bare numbers

- **A bare `dt` has the unit of a bare `start` and `stop`** (FP21): years when they are years (`start=2000`), and otherwise the sim's unit. The model's unit (months, from the typed durations) still sets what bare numbers in the model mean; `dt` is a time, like `start` and `stop`, and is read with them. `describe()` prints `every 1 month (dt=1/12 year)`.
- **A number returned by a function in a distribution takes the distribution's unit** (FP22): `ss.lognorm_ex(mean=fn, std=1, unit='month')`, or the unit of a typed sibling parameter (02 already says other parameters of the same dimension inherit it). With no unit on the distribution, the sim's unit applies as today, and if that unit was inferred, a note names the function the first time it returns a bare number (`UnitInferred: dur_use returned bare numbers, read in years; give the distribution a unit=`).
- The same rule for every function in a time slot (`rate=fn`): a bare number is in the sim's unit, and the note above applies when that unit was guessed.

## Semantics under every solver

| Feature | Agents | ODE | Binomial | Gillespie |
|---|---|---|---|---|
| A property in an expression, the property a dimension | The person's value | The group's level value (rules in section 7): exact for booleans and counters below the top level, the bin's middle for binned continuous values, with a note | As the ODE | As the ODE |
| A property not a dimension | The person's value | `CapabilityError` naming `stratify()`; `'auto'` runs as agents | Same | Same |
| A condition in a rate (`b*ever`) | 1 or 0 per person | The fraction of each state and group selected, as for `where=` today | Same | Same |
| `ss.by` table over a property | Lookup by the person's value | Per group; a finer table is averaged with a note | Same | Same |
| Counter (`increment=`, by k) | The property increases by k; the groups of those people are worked out again | Generated moves from each level to the one k up (capped at the top); exact | Same, binomial draws on the moves | Same, as events |
| Clock | Time since the step of the last set (stored as a step number, the 16-bit timers of item 21) | A conveyor of whole steps within each level, then on to the next; `none` first; exact when the edges are whole steps | Same | Same, people moved along at step boundaries |
| `p=` entry with a `to` axis | The person's row, normalized | The group's row, normalized, as a branch share per group (FP13's fix applies: a modifier on a parameter inside `p=` makes the run stepped) | Same | Same |
| A regression expression | Exact | Exact when every covariate is a boolean, counter, or labeled dimension; binned continuous ones as above | Same | Same |
| A distribution by level, or with properties in its parameters | Each person's own | Erlang stages per group for a dwell (one number of stages, noted); elsewhere the group's mean, noted, as for any distribution on a quantity | Same | Same |

Gillespie holds a value that varies in time for a quarter of a step, as today. No feature here needs agents when its properties are dimensions; the only new `CapabilityError` is a property that is not one, under an explicit compartmental solver, or with `approx=False` where a note above says something is approximated.

**`describe()`.** Expressions are printed as written, with parameter names (`p_use = expit(b_intercept + b_ever*~never + b_urban*people.urban + ...)`); a table as `by(table of 54 rows by age, to)`; an entry with a `to` axis as `p_use × by(...) × w, normalized over pill, iud, ...`; a dimension's line says where its values come from and what moves it:

```text
dimensions:
  urban        False, True                                    people.urban, from the people given
  parity       0, 1, 2, 3, 4, 5, 6, 7+                        counter: +1 by delivery, stillbirth; +2 by twins; from the people given
  since_birth  none, 0, 1, 2, 3-5, 6-11, 12-23, 24+ months    clock: set to 0 by delivery, twins
```

**`to_spec()`.** A property in an expression is written `people.urban` (or `edu.attainment`), as conditions already write `people.age` (14, "Rate expressions"); `expit` joins the list of functions there and in `ss.spec.Parser`. A table is `{"by": [...], "table": {...}}`, as today, with a `to` column. A counter's move is written `{"increment": {"parity": 1}}` and a clock's reset as `{"since_birth": 0}`. A stratification gains `"kind": "clock"` and `"split_from": "people"` for a property the people give (its values are in the data hash, not the model's).

**`to_code()`.** Writes the same spellings: `ss.ppl.urban`, `expit(...)` (with `from scipy.special import expit`), the counter's move (`increment='parity'`, or `increment=dict(parity=2)`), `kind='clock'`, and tables as data frames written in full (as arrays are now).

## Fit with what exists

| Need | Carried by | New |
|---|---|---|
| Read a property in a rate, `p=`, or distribution | Expressions, and conditions' `people.<name>` leaves | Conditions and properties allowed in rates; `ss.ppl.<any>`; `expit` |
| Look up by a property | `ss.by` (its table form) | Numeric columns as edges, a `None` row, the not-a-dimension rule, outer products |
| A regression | An expression with `ss.par`s and `ss.timeseries` (the year trend is a linear time series) | Nothing more |
| A table, a choice among many | `ss.by` tables and `p=` lists | A `to` column, normalized |
| A counter | A dimension, `move=` | A relative move, `increment=` |
| Time since an event | A dimension, `move=`, age's aging, a product's levels of time since delivery, the fixed-delay conveyor | `kind='clock'`, a `none` level |
| Distributions by level, with units | `ss.by`, `unit=` | Distributions inside `ss.by`; `unit=` on `ss.choice` and `ss.histogram` |

Why not the reviews' names:

- **`ss.logistic` and `ss.spline`** (FP26): a logistic is `expit` of a sum, and FPsim's spline is a table by single year of age once its knots are evaluated; a function form would print and fit only what its arguments name, where an expression prints and fits everything. If a model needs a spline whose coefficients are fitted, `ss.by(dict(age=knots, value=[ss.par(...), ...]))` with linear interpolation is the extension (not proposed now: FPsim does not fit them).
- **`ss.table`** (FP27): an `ss.by` table already is one; the destination is one more column.
- **`ss.time_since`** (FP25): an expression of a per-person time cannot run as compartments without the bins a clock has, and the bins are needed for every lookup anyway.
- **`ss.value`** (HP19): h1's.
- **`ss.counter`**: a counter is a dimension with integer edges.

What it removes or replaces: in the ports, `update_parity()` and its two stamp arrays, the per-step row cache and the ten lambdas per transition, the `p_use` function, the `dur()` callbacks with `/12`, `set_objectives()`, the reads of `ti_PP` and other engine stamps for time since an event; in the spec, the sentence "A bare number returned from a function is in the sim's unit" (03, 04) gains the distribution's unit. Nothing in the vocabulary is removed. `ss.update` is unchanged (it is for counts that are not people).

Spec sections that change: 02 (bare `dt`; units of function returns; `ss.choice`, `ss.histogram`); 03 (`move=` values, `p=` entries with a `to` axis, conditions and properties in rates); 04 (`ss.by` tables over properties, `None` rows, outer products, distributions inside `ss.by`); 05 (existing properties kept, values not indices, clocks, counters, the empty-level note, the compartmental worth of a property); 07 (conditions as 0 or 1 in a rate); 12 (clocks as conveyors); 14 (the spellings above); 17 (keywords and slot rows: `kind='clock'`, the `move` values; no new counted names, so the count stays 105).

## What FPsim should declare, and what stays code

| Part | Declare? | Why |
|---|---|---|
| Contraceptive choice (`StandardChoice`, `SimpleChoice`, methods and durations) | Yes | It is what FPsim calibrates (the intercept, the trend, the method weights) and what its interventions change (the matrix, initiation, efficacy). Declared, those are `ss.par`s and `ss.change`s, `describe()` shows the whole choice model, results by method and age come with no analyzer, and the module runs as an ODE when its covariates are dimensions (0.1 s against 1.3 s measured). About 70 lines against 468 |
| Education | Yes, after h1 | Start, dropout (a table by age and parity), graduation, and the objective (a histogram by urban) are this document's vocabulary. Attainment, years in school frozen on leaving, is HP19's write, h1's. Without it, attainment stays a 2-line coded step, which is fine |
| Deaths | Already | `ss.Deaths(death_rate=table)`, identical run for run |
| Pregnancy and conception (`FPmod`, a subclass of `ss.Pregnancy`) | No, for now | (1) `ss.Pregnancy` holds mother-fetus structure shared with STIsim and HPVsim (the maternal network, prenatal transmission, `parent`), which the declared engine does not model (FP18: `offspring=` sets no parent), so a declared FPsim pregnancy would fork it. (2) Conception reads per-person processes drawn each month (sexual activity, breastfeeding and LAM, personal fecundity) whose means the declared port had to use, which changes the model. (3) It keeps per-person histories (age at each birth, short intervals) that no declared form holds, and none is proposed. (4) It runs only as agents whatever it is written in, and the declared port already agrees within 1 to 2%, so declaring it gains `describe()` and nothing in solver choice. It needs v4's modifiers to reach a coded module instead (FP23, D23) |
| Intent to use, fertility intent | Optional | The same form as the probability of use, so it can be declared alongside; low value until a scenario changes it |
| `fp.Sim`, `fp.Scenarios`, `fp.Calibration`, `Experiment` | Replaced by run tools | Phases A and B (FP7's `new()`); not this document |

So the coded pregnancy writes `people.parity` (as it does now) and the time since the last birth, and the declared choice reads them as properties. Counters and clocks are therefore not needed by FPsim until its pregnancy is declared; they are in this design because TB, STIsim, HPVsim, and malaria need the same two things, and because given dimensions they are small. They come last in the plan.

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 1 | **Done, 2026-10-19** (FP19, FP20, FP39's empty-level note; D15 moved to steps 3 and 4, FP39's counts after aging to Stage 2 of [the plan](../../oct19/implementation.md)). Dimension fixes: values not indices (FP19); existing properties kept and `split=` refused (FP20); the empty-level note and counts after aging (FP39); `ss.by(sex=...)` and other values by a property that is not a dimension as agents, with the `'auto'` rule and the `CapabilityError` (D15, decision 5) | S to M | `dimensions.py`, `parameters.py`, `solvers/choose.py`, `solvers/agents.py`, `results.py` |
| 2 | **Done, 2026-10-19** (FP21 with `ss.UnitInferred`'s warning, FP22 as a note, FP35, `ss.histogram(unit=)`; FP30's distributions by level moved to step 4). Units: bare `dt` with `start`/`stop` (FP21); function returns in the distribution's unit, and the note (FP22); `ss.choice` and `ss.histogram` with typed values and `unit=` (FP35, FP30) | S | `timeline.py`, `time.py`, `distributions.py`, `transitions.py` |
| 3 | **Done, 2026-10-19.** Properties and conditions in expressions; `ss.ppl.<any>`; `expit`; group evaluation by the rules of section 7; D15 (a value by `sex` or `age` with no such dimension, read from people as agents, with a keyed `ss.by(sex=dict(female=a, male=b))`, since a list has no level order without the dimension) | M | `expressions.py`, `conditions.py`, `people.py`, `solvers/engine.py`, `solvers/compartmental.py`, `kernels.py`, `spec.py` |
| 4 | **Done, 2026-10-19** (properties in distribution parameters, and outer products of tables, finished with steps 5 to 8). `ss.by` over properties: tables with edges and a `None` row, averaging, outer products; distributions inside `ss.by` (FP30's `dur=ss.by(urban=[h0, h1])`), and properties in distribution parameters | M | `parameters.py`, `distributions.py`, `solvers/agents.py`, `solvers/compartmental.py` |
| 5 | **Done, 2026-10-19** (`Row` and `Share` in `expressions.py`, `Graph.expand_rows()`; `ss.change` of a table with a `to` column is not yet reached). `p=` entries with a `to` axis, normalized; a single function returning a row | M | `transitions.py`, `graph.py`, `solvers/agents.py`, `solvers/compartmental.py`, `kernels.py` |
| 6 | **Done, 2026-10-19** (`increment=`). Counters: the relative move in the chosen spelling (decision A) on transitions, transfers, and deliveries | S to M | `transitions.py`, `dimensions.py`, `graph.py`, both solvers, `products.py` |
| 7 | **Done, 2026-10-19** (agents hold the time since the event, advanced each step; compartments keep the clock a step at a time, moved up by the kernels' deliveries; `none` is the last level, not the first). Clocks: `kind='clock'`, the `none` level, agents by step stamps, compartments by the conveyor (reusing `products.py`'s levels of time since delivery and the fixed delay) | M (agents S, compartments M) | `dimensions.py`, `solvers/agents.py`, `solvers/compartmental.py`, `kernels.py`, `products.py` |
| 8 | **Done, 2026-10-19, except the user guide** (which waits until the implementation is final). `describe()`, `to_spec()`, `to_code()`, the spec sections listed above, the user guide, and the vocabulary keywords | M, spread over the steps | `presentation.py`, `spec.py`, `v4_design/design/*.md`, `docs/user_guide/` |
| 9 | **In part, 2026-10-19**: FPsim is not installed in the v4 environment, so the comparison with FPsim waits; the contract test across solvers (below) is in `tests/test_contracts.py`. The FPsim contraception port rewritten in this vocabulary, as a test against FPsim's (fixed population, mCPR by year over 4 seeds) and a docs example | M | `tests/`, `docs/examples/`, `fpsim:v4port/contraception/` |

Steps 1 and 2 were fixes that needed no decision beyond 5 to 8, and are done. Steps 3 and 4 need no open decision and are being built. Steps 3 to 5 make the contraception declarable and are the core (about two weeks of agent work together). Steps 6 and 7 wait for decision A (the counter spelling) and h1's decision on the write primitive, so the spelling is shared.

Tests, on numbers a user sees:

- FP19: `wealthquintile=[1, ..., 5]` with an equal split gives 20% in each level, `people.wealthquintile` takes the values 1 to 5, and `adjust=` on level 5 reaches 20%.
- FP20: People with `urban` at 0.9 keep a share of 0.9 after `stratify(urban=[False, True])`, the results by urban say 0.9, and `split=` with it raises.
- FP21: `start=2000, stop=2020, dt=1/12` with durations in months has 241 points. FP22: a callback returning 6 with `unit='month'` gives a median stay of 6 months. FP35: `ss.choice(a=[ss.months(9), ss.months(10)])` as a dwell gives a mean of 9.5 months.
- D15: `ss.death(rate=ss.by(sex=[...]))` as agents gives the two death rates by sex within 3 standard errors; under `solver='ode'` it raises naming `stratify(sex=...)`.
- A regression: `p_use` printed by `describe()` lists every coefficient by name; `ss.change` on the intercept to −∞ after 2005 gives 0 starts after 2005 (FP14's case: today 3,531 with and without the change).
- Counter: with births at a constant rate, mean parity by age as agents equals the ODE's within 1%; twins add 2.
- Clock: activity by months since birth from a table; the share of women in each level matches between agents, binomial, and ODE within 1% at a monthly step (exact edges), and `none` holds the nulliparous.
- Contract across solvers (in `tests/test_contracts.py`): a method module with a regression over two boolean dimensions, a counter, a table with `to` by age, and a clock gives the same expected shares by method and age under ODE, binomial, Gillespie (20 runs), and agents (20 runs), within 1% (ODE and binomial) and 3 standard errors (stochastic).
- `to_code()` round trip of the contraception module, and its `hash()` stable.

What the downstream ports would delete: in `fpsim:v4port/contraception/contra_v4.py`, `p_use()`, `rows()`, `branch()` and its cache, and `dur()` (about 60 of 134 lines, with no functions left in a slot); in `pregnancy/fp_declared.py`, `update_parity()`, the `seen_pp`/`seen_sb` arrays, and the `ti_PP` reads in `conception_rate()`; in `population/edu_v4.py`, `set_objectives()` and `objective_dists()`; in FPsim itself, once the choice module is declared, `StandardChoice`, `SimpleChoice`, `get_switching_matrix()`, and the matrix-editing interventions (`update_methods`, `method_switching`) become `ss.change` on tables (most of the 1,617 lines of `methods.py`, and the matrix-editing parts of `interventions.py`). HPVsim and STIsim delete nothing until h1's write and their own declared ports.

## Out of scope and dependencies

Out of scope:

- **General per-person writes** (HP19's running maximum with a draw, a value frozen at entry, a value accumulated over several spells): h1's "remembered results per person".
- **An outcome that depends on the drawn dwell** (HP18), which the third-trimester stillbirth hazard also reads (`ti_exit_P3`); and branching at exit (HP1). HP18's design.
- **"As soon as" transitions** (FPsim's one-month and six-month postpartum decisions, written as `rate=1e6` with a `where=`): ST26 wants a move instead.
- **Typed values times series and other expression arithmetic** (FP31, FP36): fp.md group 7. The conception rate as an expression needs FP31's `ss.probperyear(expr)`.
- **Modifiers reaching coded modules** (FP23, D23), which the coded pregnancy needs for efficacy: Phase C or D work, not a design pass.
- **A `from` column** in tables, fitted spline coefficients, and list-valued per-person histories (FPsim's ages at each birth).

Dependencies and the interfaces assumed:

- **h1, the care cascade.** I assume h1 defines (1) an event as a transition, one destination of one (`'tr.state'`), or a delivery outcome (`screen.positive`); (2) a remembered result as a property of people written when an event happens, read like any property (in `where=`, rates, and `ss.by`); and (3) the general write spelled as a value in the event's `move=` (or one keyword that every event takes). My counter (a relative move, `add=` if chosen) and clock reset (`0` on a clock) are then two values of that write, and a clock's `none` level is h1's "never happened". If h1 instead gives events attributes (`ev.ever`, `ev.last`), a clock is `kind='clock'` over `ev.last`, the counter stays a move, and nothing else here changes. Either way h1 and this design must agree on one spelling for "set a property on an event", which is why steps 6 and 7 wait for it.
- **h2, a family of diseases.** The second half of FP28 (a modifier over a set of transitions) is h2's; `p=` entries with a `to` axis renormalize, and h2's combination rule for modifiers on branches should state how it composes with that (a modifier on one destination of a normalized entry scales its share before normalizing, which is FPsim's rule for method weights).
- **h3, per-act transmission.** No overlap.

## Decisions still open

For Cliff's `[CK:]`. Decisions 1, 2, and 4 to 9 above are approved and not repeated. A to C are answered (second revision).

| # | Decision | Options | Recommendation |
|---|---|---|---|
| A | How a counter is moved (decision 3 reopened; section 3 has five examples in each spelling) | (a) `move='parity'`, with `move=('parity', 2)`; (b) `move=dict(to='parity', by=2)`; (c) `move=dict(parity=2)` on a dimension declared `kind='counter'`; (d) a separate keyword, `add=dict(parity=2)`, on transitions, deliveries, and transfers | (d): one meaning per keyword (`move=` is where people go, `add=` how far), no strings, two counters on one event, a reset is the existing `move=dict(parity=0)`, and the saved form needs no context. Second choice (c) [CK: agree with d, except use `increment` instead of `add`. `increment='parity'` means the same as `increment=dict(parity=1)`, and the dict form can be used to give more complex forms (+2, -1, etc)] **Decided:** (d), named `increment=`: `increment='parity'` is `increment=dict(parity=1)`, and the dict gives other amounts (`dict(parity=2)`, `dict(doses=-1)`) and several counters at once; a reset stays `move=dict(parity=0)` |
| B | `ss.People.age`, `.female`, and `.alive`, which conditions use today, once `ss.ppl` exists | (a) keep both, documenting `ss.ppl`; (b) keep `ss.People.<x>` as a compatibility alias with a deprecation note; (c) remove it | (a) for now: they give the same lazy references, and removing a working spelling needs a reason; the docs use only `ss.ppl` [CK: ss.People was never released, so let's just remove it. well not remove it, but keep it as the class -- sim.people = ss.People(), but "ss.People.age" is not a thing. unless there's a reason why it should?] **Decided:** `ss.People` stays the class (`sim.people = ss.People(...)`); `ss.People.age` and the other lazy references are removed, so `ss.ppl.<x>` is the only spelling (the vocabulary count stays 105, since `ss.ppl` replaces them) |
| C | Whether `add=` (if chosen) is also how h1's other per-person writes are spelled (a running maximum, a value frozen at entry) | (a) `add=` for counters only, and h1 picks its own keyword for the rest; (b) one keyword for all writes, with `add=` one of its forms | (a): a counter is a move along a dimension, which works under every solver; the other writes are agent-only values (HP19) and read better under their own name. Revisit with h1's second draft [CK: ok] **Decided:** (a) |
