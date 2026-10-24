# The burden of disease: disability weights, YLD, YLL, DALYs, and person-time

A design pass from the Thanzi la Onse review ([tlo_review.md](../../oct19/tlo_review.md), group 6, with TL23 from group 4), for Cliff to decide before any code. Findings: TL50 (no burden of disease), TL51 (an outcome weighted by age crashed as agents; fixed), TL23 (one cause of death per person; its first half fixed, `net_of=` open). TL52 (every stage of a staged condition counts as infected) is fixed and does not bear on this design; TL53 (conditions described as infections) is touched only through wording. Related: R17 and HP23 (no result across modules). Evidence: the TLO burden port (`TLOmodel:v4port/mnch_burden/burden_v4.py`, `REPORT.md`, `probe_daly_vocab.py`), HPVsim's `dalys` analyzer, STIsim's syphilis DALY analyzers, MIGHTI's disability weights, vivarium_public_health, and probes of the current tree in the session scratchpad (`design_probes/p3_burden.py`, `p6_deaths.py`, `p7_integral.py`).

## Summary

**The problem.** TLO's main output is DALYs by cause, age, sex and district. Each disease declares its causes, reports a disability weight for each living person every month (`report_daly_values()`), and names the cause when it kills (`do_death(cause=)`). `HealthBurden` caps each person's total weight at 1, and counts YLL as 90 minus the age at death. v4 has no DALY, YLL, YLD or disability weight anywhere. What exists, probed today:

- A weighted sum of states is a declared result (`yld_rate = 0.187*J + 0.215*K + 0.402*JK`), but a stock: `res.yld_rate.resample('year')` gives the value at each year's end, not the year's total.
- A time integral can be had as a flow into a count, `accrue = ss.null >> Y @ ss.peryear(0.2)*I.n`. Every solver runs it: agents and binomial give the sum over steps exactly (62.344 and 58.907), the ODE and Gillespie integrate within the step (48.225 against 48.63 by the step sum; 47.036 against 47.995). But it is a trick, and it is per disease.
- YLL as an outcome weighted by age works under both solvers since TL51's fix: `ss.outcome(ss.by(age=[60, 50, 30, 10])*death)` gives 21,820 years for 470 deaths as agents and 22,202 for 482 as an ODE.
- Each death has one cause, `people.cause_of_death`, with the names in `people.causes` (`['deaths', 'mock']`); a contested death goes to one module at random (TL23, fixed).
- Nothing combines weights across diseases, caps them per person, or puts the result anywhere but under one disease. The port's coded 65-line `Burden` analyzer does, and agrees with TLO within 2 to 5% (TLO's `healthburden.py` and `causes.py` are 993 lines). It also shows that TLO's monthly snapshot misses 44% of Mockitis's YLD (18.4 against 29.8 integrated over each step).
- `ss.Deaths` is agents-only (`CapabilityError ... ss.death() in the disease is its compartmental form`), and there is no "net of modeled causes", so a disease's deaths are counted twice in WPP's all-cause rates.

**The proposal.** Three pieces, each reusing what exists:

- **`ss.integral(expr)`**: person-time as a result. It is a flow (the amount accrued in each step), so it sums when resampled, has `.cum`, matches data, and is kept by group. It is the flow-into-a-count above, without the count.
- **`disability=dict(S1=0.288, ...)`** on a declared disease, as `init=` is written. A coded module defines `disability(uids)`.
- **`ss.Burden(...)`** in `analyzers=`: YLD, YLL and DALYs by cause and by every dimension of the sim, under every solver. It combines a person's weights by `combine='multiply'|'sum'|'max'`, the rule names of `ss.immunity`, and counts YLL from a life table or an age limit.

Two new `ss.` names (`ss.Burden`, `ss.integral`), one keyword on diseases (`disability=`), one on death transitions (`cause=`), and one on `ss.Deaths` (`net_of=`). `ss.disability` (the review's) is not added.

```python
import starsim as ss, tlo4

N, LGD, HGD, S1, S2, S3, S4 = ss.states(7)
oc = ss.Disease(N, LGD, HGD, S1, S2, S3, S4,
    onset      = N >> LGD @ ss.prob(4e-6, ss.months(3)),
    progress   = LGD >> HGD @ ss.prob(0.04, ss.months(3)),   # ... and so on to S4
    death      = S4 >> ss.null @ ss.prob(0.4, ss.months(3)),
    disability = dict(S1=0.288, S2=0.288, S3=0.451, S4=ss.par(0.54, bounds=[0.38, 0.71])),   # a weight per state; any value an ss.par takes
)

burden = ss.Burden(age_limit=90, combine='sum', causes=tlo4.gbd_labels)   # TLO's conventions; ss.Burden() alone is GBD's
sim = ss.Sim(people=..., diseases=[oc, tlo4.cmd(), tlo4.hiv()], connectors=tlo4.Schisto(),
             demographics=[ss.Births(...), ss.Deaths(death_rate=wpp, net_of=tlo4.modeled_share)],
             analyzers=burden, start=2010, stop=2030)
sim.run()

res = sim.results.burden
res.dalys.by('cause')                              # YLD + YLL, one series per cause
res.yld.select(sex='female').by('age')             # by any dimension of the sim
res.yll.resample('year')                           # flows: summed over each year
ss.parallel(base=sim, scaleup=sim.new(...)).diff('base').burden.dalys   # DALYs averted
```

```python
class Schisto(ss.Module):              # a module written as code reports a weight per person
    def disability(self, uids):
        return np.where(self.heavy[uids], 0.114, 0.006)

bed_days = ss.integral(H)              # person-time on its own: bed-days, person-years on ART
```

The decisions for Cliff are in the [last section](#decisions-for-cliff); the main ones are how a person's weights combine (2), the reference for YLL (3), and whether discounting is in now (5).

## Use cases

| Model | What it needs | Today | Proposed | Simpler? |
|---|---|---|---|---|
| TLO (TL50) | DALYs by cause, age, sex, district, from about 30 conditions, declared and coded; weights by stage and by treatment; YLL to age 90; the cap at 1 per person | A coded 65-line analyzer reading a dict of weights per declared state and `daly_weights()` from coded modules; a coded `cause` dict for declared deaths, which had none (TL23, now fixed) | `disability=` on each declared disease, `disability(uids)` on coded ones, `ss.Burden(age_limit=90, combine='sum', causes=...)` | Yes: the analyzer goes, and DALYs run under the ODE too |
| TLO (TL23) | Background deaths net of the causes modeled | WPP all-cause rates plus every disease's deaths, so those are counted twice | `ss.Deaths(death_rate=wpp, net_of=share)`, TLO's `OtherDeathPoll` | Yes |
| HPVsim | Cervical cancer DALYs for cost-effectiveness: YLD from stage weights (GBD 2017: 0.288, 0.049, 0.451, 0.54 with time fractions), YLL as 84 minus the age at death | `hpv.dalys`, about 100 lines: incidence-based, every case's YLD and YLL counted at onset from its scheduled death | `disability=` on the declared cancer's stages; `ss.Burden(age_limit=84)` | Yes, and exact by stage rather than by average time fractions. Prevalence-based, so years differ from HPVsim's attribution at onset (decision 4) |
| STIsim | Syphilis DALYs: incidence-based (`syph_idalys`) and hybrid (`syph_hdalys`: YLL at death, YLD each year), with YLL discounted | Two analyzers in the Zimbabwe analysis, about 115 lines | `disability=` on syphilis's stages; `ss.Burden(discount=0.03)` for the hybrid form | Yes, for the hybrid form; the incidence form is an outcome (decision 4) |
| MIGHTI (Starsim, HIV and NCDs) | One weight per condition, HIV by CD4 stage; multimorbidity multiplicative by default, additive and maximum as sensitivities; YLL from a life table by sex and age; outcomes discounted at 3% | `disability_weights.py` with `MULTIMORBIDITY_RULES = ("multiplicative", "additive", "maximum")`; coded YLL and cost analyzers | `disability=` per disease; `ss.Burden(combine='multiply', life_table=..., discount=0.03)`; `combine='sum'` or `'max'` for the sensitivities | Yes: its three rules are exactly `combine=`'s |
| Malaria | DALYs per clinical and severe episode for cost-effectiveness, often with a fixed duration per episode | A coded analyzer | Prevalence-based with `disability=` on the clinical and severe states; or an outcome per episode (decision 4) | Yes |
| Any model | Person-time: bed-days, person-years at risk, person-years on ART | `ss.null >> Y @ rate*X.n` into a count, per disease | `ss.integral(X)` | Yes |

**Other frameworks.** Every burden model in this list counts YLD per unit time lived in a state and YLL at death; they differ in four choices:

| Framework | When YLD is counted | Several conditions in one person | YLL | Discounting |
|---|---|---|---|---|
| TLO (`healthburden.py`) | As lived, from a monthly snapshot of each person's weight, times 1/12 | Summed, then scaled down to 1 if above (`Get_Current_DALYS`) | 90 minus the age at death; "stacked" at the death, plus versions spread over the remaining sim | None |
| GBD (since 2010) | As lived (prevalence-based) | Multiplicative, `1 − Π(1 − w)`, by microsimulation assuming independence | A standard reference life table (remaining life expectancy at each age) | None |
| vivarium_public_health (IHME) | Weight × step, per person (`DisabilityObserver`) | Union, `1 − Π(1 − w)` (`all_causes.disability_weight`) | Theoretical minimum risk life expectancy minus age | None |
| MIGHTI (Starsim 3) | As lived | Multiplicative (default), additive, or maximum | Life table e(x) by sex and age, or a reference age | 3% |
| HPVsim | At onset (incidence-based), weight × time fraction × duration | One disease | 84 minus the age at death | None |
| STIsim (syphilis analyses) | At onset, or each year | One disease | Life expectancy minus age | Yes, on YLL |
| DALY (R package) | Incidence-based | Not modeled | A life table | Optional, with GBD 1990's age weighting |
| odin, summer2, Atomica, MEmilio, flepiMoP, epydemix, EpiModel, LASER, SimInf, Emulsion | No burden. Person-time is user-declared: odin `deriv(Y) <- w*I`; Atomica's `time_aggregation='integrate'` when plotting; summer2's cumulative outputs are of flows only | | | |

So the proposal's defaults follow GBD and vivarium (as lived, multiplicative, a reference life table, no discounting), and TLO, MIGHTI, HPVsim and STIsim are each one or two keywords away.

## The proposal

### 1. `ss.integral(expr)`: person-time

```python
yld_cs    = ss.integral(0.187*J + 0.215*K + 0.402*JK)        # in a disease's results=, as any result
bed_days  = ss.integral(H, unit='day')
on_art    = ss.integral(hiv.ART)                              # person-years on ART
```

- The expression is what a result may be: states (worth the people in them), sums with weights (numbers, `ss.par`, `ss.by`), and counts (`.n`) in nonlinear terms.
- The result is a **flow**: its value at time point `ti` is the person-time accrued over the step that ends there (0 at the start), as for a transition. So `resample('year')` sums, `.cum` is the running total, `sim.summarize()` gives the total, and data by period are compared with totals.
- Its unit is person-years by default (`unit='year'`); `unit=` gives another (`'day'` for bed-days). The label says which.
- It is kept by group like any result, so `res.on_art.by('age')` works.

There is no new result kind: person-time accrued in a step adds up over steps exactly as events do. Spec 09's "a flow is a number of events in each step" becomes "an amount accrued in each step: events, or person-time".

### 2. Disability weights

```python
oc = ss.Disease(..., disability=dict(S1=0.288, S2=0.288, S3=0.451, S4=0.54))
oc = ss.Disease(..., disability=dict(S1=ss.by(tx=[0.288, 0.2])))              # by level of a dimension (treated in its stage)
hiv = ss.Disease(..., disability=dict(I=ss.by(dict(cd4=[0, 200, 350], value=[0.582, 0.274, 0.078]))))   # by a property's value
```

- **`disability=`** on a declared disease is a dict by state name, as `init=` is. Each value is a weight between 0 and 1: a number, an `ss.par` (so it can be changed or fitted), an `ss.by` by a dimension or a property, or an expression. A state not named has weight 0. A weight outside 0 to 1 raises.
- **A module written as code** defines `disability(self, uids)`, returning one weight per person (TLO's `report_daly_values()`). It may return a dict `{cause: weights}` for a module with several causes.
- The **cause** of a module's disability is the module's name (`oc`), as for its deaths (TL23's fix). `ss.Burden(causes=dict(oc='Esophageal cancer', ...))` maps names to the labels reported, and several names to one label add up (TLO's `Cause(label=, gbd_causes=)`).
- Under the behavior design's rules, a weight by a property that is not a dimension runs as agents only (decision 5 there).

### 3. Causes of death

- A death's cause is its module's name, kept per person in `people.cause_of_death` (done: TL23's fix in `People.request_death()` and `People.resolve_causes()`).
- **`cause=`** on `ss.death()` and on a transition to `ss.null` names another cause, for a module with several (`ss.death('PPH ->', rate=..., cause='maternal_hemorrhage')`). Under compartments, each death transition's flow is its cause.
- **`ss.Deaths(death_rate=..., net_of=share)`**: background mortality net of the causes the model represents. `share` is the share of all-cause deaths from modeled causes, by age and sex (an `ss.by` table, or a time series of them); the background rate is multiplied by `1 − share`. This is TLO's `OtherDeathPoll` with its GBD-derived share. A share cannot make a rate negative, which subtracting cause-specific rates can when the data sources disagree (decision 7).

### 4. `ss.Burden`

```python
ss.Burden(
    combine   = 'multiply',    # how one person's weights combine: 'multiply' (1 − Π(1 − w), GBD), 'sum' (capped at 1, TLO), 'max', or a function
    life_table= None,          # remaining life expectancy at each age: an ss.lifetable, or an ss.by table by age (and sex); default the GBD reference life table
    age_limit = None,          # instead, everyone could live to this age: YLL = max(age_limit − age, 0) (TLO's 90, HPVsim's 84)
    discount  = 0,             # a yearly rate, e.g. 0.03
    causes    = None,          # a dict of cause names to the labels reported
)
```

Results, under `sim.results.burden`, all flows in years, with a `cause` axis and every dimension of the sim:
- `yld`: years lived with disability, as lived.
- `yll`: years of life lost, at the death, in the age group and place of the person who died.
- `dalys`: `yld + yll`.
- `deaths`: deaths by cause (a convenience; the same as the death flows).

`res.dalys.by('cause')` is one series per cause, as `res.hpv.infected.by('hpv')` is one per member of a family. `res.dalys.select(cause='oc')` is one. A plain `res.dalys` is the total.

**How YLD is worked out.**
- Each module gives each person a weight (from `disability=`, or `disability(uids)`), for the state they are in during the step.
- A person's weights are combined by `combine=`. `'multiply'` gives `1 − Π(1 − w_i)`. `'sum'` gives `min(1, Σ w_i)`. `'max'` gives the largest. A function takes the array of weights (causes on the first axis), as `ss.immunity`'s does.
- The combined weight is shared among the causes in proportion to their own weights. This is TLO's rescaling under `'sum'`, and GBD's apportioning under `'multiply'`.
- Each cause accrues its share × the step's length, in years.

**How YLL is worked out.** Each death accrues the remaining life expectancy at the age at death, from `life_table=` (interpolated by age, and by sex if the table has a sex column), or `max(age_limit − age, 0)`. With `discount=r`, those years are discounted from the death, `(1 − e^{−ρL})/ρ` with `ρ = ln(1 + r)` (STIsim's). The table may be an `ss.lifetable` (its remaining life expectancy is worked out from its survival, a new `lifetable.expectancy(age)`), or an `ss.by(dict(age=..., value=...))` of e(x), the form GBD and vivarium publish. Every death counts, including background ones (whose cause is `deaths`, the module's name, or a label by `causes=`).

**Discounting.** With `discount=r`, everything accrued at time `t` (YLD in its step, YLL at the death) is also multiplied by `(1 + r)^{−(t − t0)}`, where `t0` is the sim's start. That is the convention of cost-effectiveness analysis (MIGHTI's 3%).

**Default.** `ss.Burden()` with no arguments is GBD's convention: multiplicative, the GBD reference life table (bundled, about 20 numbers; life expectancy at birth about 89), no discounting. TLO writes `ss.Burden(age_limit=90, combine='sum')`.

## Semantics under every solver

| Piece | Agents | ODE | Binomial | Gillespie |
|---|---|---|---|---|
| `ss.integral(expr)` | The expression at the start of each step × the step (exact: agents change state at step boundaries) | Integrated by the solver within the step (exact for the ODE) | As agents | Between events (exact for the process) |
| `disability=` | Per person | Per group: the people in each weighted state, × weight | As the ODE | As the ODE |
| Combining weights | Per person: exact | One disease: exact (a weighted sum). Several diseases crossed (joined by `ss.modify`, which v4 crosses automatically): exact, from the joint compartments. Several not crossed: by independence within each group, which is the compartmental model's own reading and GBD's assumption. Under `'multiply'` the total is then closed-form and exact. The split among causes, and `'sum'` and `'max'`, are worked out over every combination of weighted states when that is at most 4,096 combinations per group, and otherwise from the expected weights, marked `approx` with a note | As the ODE | As the ODE |
| YLL | Per death: age at death, sex, cause | Per group: death flows × e(x) at the age group, averaged over the group if the table is finer (a note, as for `MatrixAggregated`); needs an age dimension, else `CapabilityError` naming `stratify(age=...)` | As the ODE | As the ODE |
| Discounting | Exact | Exact for YLL; YLD at the middle of the step | As agents | As the ODE |
| `net_of=` | A multiplier on `ss.Deaths`' rate | `ss.Deaths` is agents-only; `ss.death(rate=..., net_of=...)` in the disease is its compartmental form, with the same keyword | Same | Same |
| `cause=` on deaths | `people.cause_of_death` | The death flow's cause | Same | Same |

Agents only: a weight from a coded module's `disability(uids)` (coded modules are agents only), and a weight by a property that is not a dimension. Approximate: the split among causes, and `'sum'`/`'max'`, for diseases not crossed under a compartmental solver above 4,096 combinations per group, and YLL by age groups wider than the table.

**`describe()`.** A disease lists its weights (`disability: S1 0.288, S2 0.288, S3 0.451, S4 0.54`). `ss.Burden` gives one paragraph: the combination rule, the YLL reference, the discount, the causes and their labels, and under a compartmental solver whether combining is exact or approximate. **`to_spec()`.** `disability` is a dict by state, with values as parameters are written. `ss.Burden` is a module of type `burden` with its keywords; a bundled life table is named (`"life_table": "gbd2019_reference"`), a given one written as data. `ss.integral` is `{"type": "integral", "expr": "...", "unit": "year"}`. **`to_code()`** writes the same spellings.

## Fit with what exists

| Need | Carried by today | Changes |
|---|---|---|
| Person-time | A flow into a count, `ss.null >> Y @ ...`, which every solver integrates (`p7_integral.py`); `Result.kinds` (`results.py`) with `'flow'` | `ss.integral` in `transitions.py`, built as that flow without a visible count; `Result` gains a `unit` for person-time labels |
| Weights per state | Declared results as weighted sums of states (spec 09); `ss.par`, `ss.by`, expressions | `disability=` read by `ss.Disease` (`diseases.py`) and checked in `graph.py` |
| Weights from code | Nothing (the port's `daly_weights()`) | `Module.disability(uids)`, an optional method looked up by `ss.Burden` |
| Causes of death | `People.request_death(uids, cause=)`, `People.resolve_causes()`, `people.cause_of_death`, `people.causes` (`people.py`); `ss.Deaths.step()` passes `cause=self` (`demographics.py`) | `cause=` on `ss.death()` and on transitions to `ss.null` (`transitions.py`), passed to `request_death()` by `Agents` (`solvers/agents.py`) |
| YLL by age | `ss.outcome(weight*death)` with `ss.by(age=...)` weights (TL51's fix); `ss.lifetable` and `lifetable.survive()` (`transitions.py`) | `lifetable.expectancy(age)`; a bundled reference table (in `starsim/assets` or `ss.named`) |
| The analyzer | `ss.Analyzer` (`analyzers.py`), run after each step and at the first point; results by group in the engines (`flow_groups`) | `ss.Burden`, a built-in analyzer that, under compartments, adds generated `ss.integral`s and outcomes to the declared modules' results instead of reading people |
| Combining | `ss.immunity(combine=...)`, with `'max'`, `'sum'`, `'multiply'`, or a function (`strains.py`) and its rules | The same rule names and function signature, shared code |
| Net of modeled causes | `ss.Deaths(death_rate=)` (`demographics.py`, `Deaths.make_p_death()`); `ss.death(rate=)` | `net_of=` on both |
| Results across modules | None (R17, HP23) | `ss.Burden` is the first: a module whose results read other modules'. R17's general form (a declared outcome over several diseases) stays open |

**Not added.** `ss.disability({S1: ...}, cause=...)` (the review's): `disability=dict(...)` is the `init=` convention, and a module's cause is already its name. A new result kind `'integral'`: person-time is a flow. A `cap=` keyword: `combine='sum'` is capped at 1, as in `ss.immunity`. TLO's "unstacked" YLL (spread over the years after the death, to the end of the sim) and age weighting (GBD 1990): neither is used in current practice beyond TLO's secondary outputs.

**Spec changes.** 03-states-and-transitions.md: `cause=` on deaths. 07-effects.md or a new section of 09-results.md: disability weights and `ss.Burden`. 09-results.md: `ss.integral`, flows as amounts accrued, results with a `cause` axis. 12-backends.md: combining weights under compartments, and its `approx` case. 14-model-spec.md: the spellings. 17-vocabulary.md: `ss.Burden` and `ss.integral` (two new counted names), `disability=`, `cause=`, `net_of=`. 18-conformance.md: the contracts below.

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 1 | `ss.integral`: the flow without a count, `unit=`, by group, in results, describe and the spec | S | `transitions.py`, `graph.py`, `results.py`, `spec.py` |
| 2 | `disability=` on declared diseases (checks, values per state, by dimension or property); `Module.disability(uids)` | S | `diseases.py`, `graph.py`, `modules.py` |
| 3 | `cause=` on death transitions; `lifetable.expectancy()`; the bundled reference life table | S | `transitions.py`, `solvers/agents.py`, `starsim/assets/` |
| 4 | `ss.Burden` as agents: weights per person, combined and shared by cause, YLD per step, YLL per death, discounting, `causes=` labels, results by cause and group | M | `analyzers.py`, `people.py`, `results.py` |
| 5 | `ss.Burden` under compartments: generated integrals and outcomes per group; combining exact for one disease and crossed diseases, closed form for `'multiply'`, enumeration up to 4,096 combinations, `approx` beyond; YLL by age group with the averaging note | M | `analyzers.py`, `solvers/engine.py`, `solvers/compartmental.py`, `solvers/crossing.py` |
| 6 | `net_of=` on `ss.Deaths` and `ss.death()` | S | `demographics.py`, `transitions.py` |
| 7 | `describe()`, `to_spec()`, `to_code()`, the user guide page (DALYs by cause, averted), CHANGELOG | S to M | `presentation.py`, `spec.py`, `docs/user_guide/` |

About a week of agent work, steps 4 and 5 being most of it. Steps 1 to 3 need only decisions 1 and 6, and can start at once.

**Tests**, on numbers a user sees:
- `ss.integral`: an SIS at constant prevalence `p` in `N` people over `T` years gives `p N T` person-years, within 1% (ODE) and 3 standard errors (agents, binomial, Gillespie). `resample('year')` sums.
- The TLO comparison (`mnch_burden/compare_burden.py`), with `burden_v4.Burden` replaced by `ss.Burden(age_limit=90, combine='sum')`: ChronicSyndrome YLD within 2 standard errors of the port's 1,005 ± 9; Mockitis stacked DALYs within 2 standard errors of 16,060 ± 1,251 (TLO 15,281 ± 534).
- Combining: two independent diseases with weights 0.3 and 0.5 at prevalences 0.2 and 0.4, as agents (per person) and ODE (closed form) under `'multiply'`, agree within 1%; under `'sum'` the share of the co-affected capped at 1 is right. The split by cause sums to the total under every rule. Two diseases joined by `ss.modify` give the same totals as agents and as a crossed ODE.
- YLL: everyone dies at 60 with `age_limit=90`: 30 years each. With `life_table=` an `ss.lifetable` of a known survival, e(x) matches the integral of the survival to 1e-6. With `discount=0.03`, 30 years become `(1 − 1.03^{−30})/ln(1.03)` = 19.9 years each.
- HPVsim: the declared cancer with `ss.Burden(age_limit=84)` against `hpv.dalys` over 50 years: total DALYs within 5% (they differ in timing, not in total, over a long horizon).
- `net_of=`: with a share of 0.2 at all ages, background deaths fall by 20% and all-cause deaths match WPP's when the modeled disease has the deaths the share implies.

**What the ports delete.** TLO: `burden_v4.Burden` (65 lines) and the `cause` dict for declared deaths; its full port would not need `healthburden.py`'s 844 lines or the 149 of `causes.py`. HPVsim: `hpv.dalys` (about 100 lines), if the prevalence-based reading is accepted. STIsim: `syph_hdalys` and most of `syph_dalys`. MIGHTI: its multimorbidity combination and YLL analyzer (`disability_weights.py`'s combining functions, `CauseOfDeathYLLAnalyzer`).

## Out of scope and dependencies

- **Incidence-based DALYs** (HPVsim's and STIsim's `syph_idalys`): every case's YLD and YLL counted at onset. As agents it needs the scheduled time of death at onset; under compartments the expected duration. An outcome with a weight times an expected duration (`ss.outcome(w*dur*onset)`) covers the simple case; see decision 4.
- **Costs** (MIGHTI's, the cascade's): a cost per delivery or per person-year is `ss.integral` or an outcome times a price, so the vocabulary is the same; a costing design is separate.
- **A declared outcome over several diseases** (R17, HP23): `ss.Burden` is a special case; the general form stays open.
- **People as data** ([people_as_data.md](people_as_data.md)): occupied groups make results by many dimensions cheap as agents, which DALYs by cause, age, sex and district need.
- **The shared resource** (the care cascade's third draft): independent, but TLO's health-system papers report DALYs averted against a perfect health system, which `diff()` on `res.burden.dalys` gives.

## Decisions for Cliff

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | How a disease gives its weights | (a) `disability=dict(S1=0.288, ...)`, as `init=`; (b) `ss.disability({S1: 0.288, ...}, cause=...)` (the review's); (c) a declared result `ss.integral(0.288*S1 + ...)` read by the burden | (a): no new name, the `init=` convention, and the cause is the module's name already. (c) cannot cap per person |
| 2 | How one person's weights combine | (a) `combine='multiply'` by default (GBD, vivarium, MIGHTI), with `'sum'` (TLO's, capped at 1), `'max'` and a function, the names of `ss.immunity`'s rule; (b) `'sum'` by default (TLO's); (c) only one rule | (a): it is the standard, and it is the one rule with a closed form under compartments, so it is exact there for diseases that are not crossed |
| 3 | The reference for YLL | (a) `life_table=` (an `ss.lifetable` or a table of e(x)) and `age_limit=` (a number), one at a time, with the GBD reference life table bundled as the default; (b) one keyword, `yll=`, taking either; (c) no default: one must be given | (a): an age limit and a remaining life expectancy are different quantities, so they get different names; GBD's table is the usual default |
| 4 | When burden is counted | (a) as lived (prevalence-based YLD) and at death (YLL), only; (b) also incidence-based, `ss.Burden(at='onset')`, agents only | (a) for now: it is GBD's and TLO's, exact under every solver, and over a long horizon gives the same totals. The incidence form is an outcome (`w*dur*onset`) where it is needed |
| 5 | Discounting | (a) `discount=` now, default 0: YLL discounted from the death, everything from the sim's start; (b) later, with costs | (a): three downstream models (STIsim, MIGHTI, HPVsim's cost-effectiveness work) discount, and it is a few lines |
| 6 | Person-time | (a) `ss.integral(expr)`, a flow in person-years by default, `unit=` for others; (b) a new result kind `'integral'`; (c) an attribute, `X.persontime` | (a): an amount accrued per step is a flow in every way that matters (sums, `.cum`, data); (c) fits a state but not a weighted sum |
| 7 | Background deaths net of modeled causes (TL23) | (a) `net_of=` a share of all-cause deaths from modeled causes (TLO's); (b) cause-specific rates subtracted (vivarium's ACMR − CSMR); (c) the model's own deaths subtracted each step, so all-cause deaths always match the data | (a): data-based, can never make a rate negative, and the same under every solver; (b) is the same data in another form (share × all-cause rate); (c) changes the model's dynamics |
| 8 | Causes | (a) a module's name, with `cause=` on a death transition for another, and `causes=` on `ss.Burden` mapping names to labels; (b) causes declared per module, TLO's `CAUSES_OF_DEATH` | (a): name first, nothing to declare for the usual one cause per module, and TLO's labels are one dict |
| 9 | Combining under compartments for diseases not crossed | (a) exact by enumeration up to 4,096 combinations per group, then expected weights marked `approx`; (b) always `CapabilityError` unless crossed; (c) cross them for the burden | (a): totals under `'multiply'` are exact anyway, and the approximation is only in the split among causes; (c) can multiply compartments a hundredfold |
