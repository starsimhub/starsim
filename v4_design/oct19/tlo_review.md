# Thanzi La Onse: what v4 can and can't do for a whole-population health model

Written 2026-10-19, at commit `e4dbda48` on the `v4` branch of `ssv4`. Everything ran on a clean worktree of `e4dbda48`, since the `ssv4` working tree was being edited (fixes made there since, listed in [handoff.md](handoff.md), are noted where they bear on a finding). The Thanzi La Onse model (TLO, `TLOmodel`, branch `v4`, `3d8fdb7dc`) was used as a real model to test v4. It is an individual-based model of the whole health of Malawi's population: about 30 conditions in 56 modules and 73,000 lines, on a pandas dataframe of people with an event queue, a health system with staff time, consumables and beds, and DALYs. The aim was to test v4, not to build a working port of TLO, so each part was ported far enough to compare with TLO on the same people at small scale. Six agents each took one part:
- the framework, demography and lifestyle (`core.py`, the event queue, `LinearModel`, `Demography`, `SimplifiedBirths`, `Lifestyle`);
- the health system (`HealthSystem`, HSIs, consumables, beds, `DxManager`, symptoms, care-seeking, the generic first appointment, EPI);
- the infections (HIV, TB and their interaction, measles, schistosomiasis);
- the non-communicable diseases (oesophageal cancer and its family, cardiometabolic disorders, depression, epilepsy, COPD);
- child health and injuries (diarrhoea, ALRI, stunting, wasting, road traffic injuries);
- maternal and newborn health and the burden of disease (pregnancy to birth, labour, newborn outcomes, `HealthBurden`).

TLO was never run in full (it takes about 10 hours). Each comparison used the module subset TLO's own tests use, at 2,000 to 20,000 people over 1 to 8 years, with the health system disabled except in the health-system part, and the v4 port started from TLO's own initial population, exported from TLO. Every Python job ran one at a time on a shared 20-core machine (load 4 to 30, mostly under 10), so timings under about 20% apart are noise. Scripts, outputs and each part's full report (`REPORT.md`) are in `TLOmodel` under `v4port/<part>/`. I reran the main new findings myself (TL3, TL6, TL8, TL12, TL16, TL17, TL23–TL31, TL35, TL36 (the doses), TL37, TL51, TL52), and all of them reproduced.

TLO is unlike the earlier test models. Most of it isn't transmission: TB and measles are assigned from incidence data, and only HIV and schisto transmit. What TLO is built around is people with many properties, linear models of relative risk on those properties, a health system that every disease draws on, and a burden of disease that every disease reports into. So this tests v4's vocabulary away from its home ground. New findings get IDs TL1–TL72. Everything else is cross-referenced to the malaria review (MA), the FPsim review (FP), the HPVsim review (HP), the STIsim review (ST), the merged findings (L, D, X, R), and [new-findings.md](../oct17/new-findings.md) (N). Duplicates get one line each.

## In brief

**What works as well as TLO.** The natural history of almost every condition can be declared, in about a tenth of the code, and agrees with TLO on the same people:

| Part | v4 | TLO code it replaces | Agreement with TLO |
|---|---|---|---|
| Demography and births, from built-ins only | 20 lines, plus 66 of data loading | 588 | Population 22,404 against 22,411 after 5 years; births and deaths within 1 SE |
| Lifestyle (22 risk factors), coded; 8 of them declared | 256 coded; 44 declared | 1,039; 274 for the same 8 | All 22 prevalences within 0.5 percentage points after 5 years |
| A care cascade (symptom, first appointment, test with stock-outs, treatment queue in staff minutes) | 12 declared plus 152 coded for the health system | (7,458 lines of framework, not like for like) | Prevalence 0.1139 against 0.1138 (elastic capacity), 0.1145 against 0.1168 (constrained); waiting times match |
| HIV (FOI by sex, age and risk, care cascade, VMMC, MTCT), TB with HIV, measles | about 100, 129, 101 | 4,624, 3,370, 654 | Adult HIV prevalence within 0.005, ART coverage within 0.01–0.04; 38% of new TB in people with HIV against 37%; measles cases within noise |
| Schistosomiasis, coded beside declared HIV | 250 | 2,076 | Prevalence 0.309/0.396 against 0.310/0.397, with MDA too |
| Oesophageal cancer, CMD (5 conditions, 2 acute events) and depression, epilepsy, COPD | about 60, 137, 32 | about 500, 850, 550 | Every count within 1 to 2 SE of TLO once TLO's own bugs are removed |
| Diarrhoea (10 pathogens), stunting, road traffic injuries | 55, 39, 56 | 581, 245, 471 | Episodes 2,228 against 2,200, pathogen shares within 0.01 |
| Pregnancy to birth (checkpoints, abortion, stillbirth, preterm, twins, newborns) | 77 | 411 | All 11 counts within 2.3 SE |
| DALYs for a declared and a coded disease | 65 (a coded analyzer) | 605 | Within 2–5% |

The particular things that fit well:
- **Staged diseases with treatment that slows only the stage it was given in.** That is a dimension with `adjust=` and `move=` on each progression. TLO needs an extra property and a comparison every poll.
- **Several conditions in one person, as several diseases joined by `ss.modify`.** As an ODE, v4 crosses them automatically ("jointly dm × htn × ckd × clbp × cihd × stroke × mi: 128 compartments"), and agrees with agents.
- **TLO's gestational checkpoints.** These are fixed dwells that branch, and delivery creates newborns in a preterm or term state with `offspring=`.
- **Several pathogens as one disease.** Pathogen is a dimension of the episode state, split on entry, so episodes by pathogen and age come with no logging.
- **A coded health system beside declared cascades.** A booking function in `on_positive=` and `ss.treat(deliver=...)` for the effect match TLO's queue to 0.0001 in prevalence.
- **Monthly consumable availability by district.** It is coverage `ss.by(district=[ss.timeseries(...)])`, under agents and the ODE alike.

**What v4 does better than TLO:**
- **Speed: 18 to 1,000 times faster on the same people and period.** For example, 0.24 s against 112 s for oesophageal cancer, 0.7 s against 298 s for schisto, and 0.8 s against 22–34 s for the health-system slice. Most of TLO's time is framework: births at 8 ms each, a daily age update, logging events, and pandas scans. Its disease code is often too cheap to measure.
- **Common random numbers.** In TLO, the order in which modules are registered changes births from 239 to 220. 10% more fertility gives fewer births (230 against 239), and only 46 of 77 deaths are shared between the two runs. In v4 the same 62 people die in both.
- **Results by any group with no logging code.** TLO's HIV logger is about 500 lines, and its maternal outcomes are a 120-key dict incremented by hand in about 100 places.
- **Solver choice for free.** Declared HIV, TB with HIV, measles, CMD, COPD, epilepsy and the care cascade with monthly stock-outs each run as an ODE as well, and agree with agents.
- **`describe()`, many seeds for the price of one TLO run, `ss.parallel(...).diff()` for scenarios, and calibration with the bounds already on each `ss.par`.** TLO's priors are only logged.
- **The declared vocabulary rules out a class of TLO's own bugs.** The agents found about 45, mostly by reading the code. For example:
  - a typo in an exhaustive predictor means nobody with high-grade dysplasia is ever diagnosed;
  - stage progression cascades within a poll and doubles cancer deaths;
  - `df.loc[...].values[:] =` writes to a copy, so ChronicSyndrome never kills;
  - pandas index alignment silently skips 90% of CLBP removals;
  - a category name that doesn't exist ("High-infection") means HIV never sees schisto;
  - rounding means worms never die from burdens of 1 or 2.

  None of these has a counterpart in a declared transition (see [TLO's own bugs](#tlos-own-bugs)).

**What is harder in v4 than in TLO, in six groups:**

1. **People as data** (TL1–TL10). TLO's people are a dataframe: start from the census, give each person 20 lifestyle properties, and read any of them in any rate.
   - In v4 `init=` crashes on a function or array, `ss.People` takes no per-person values, and a property can't appear in a rate.
   - Stratifying by an existing property silently resets it.
   - Only one split may depend on another dimension.
   - Past 32,767 groups the agent engine crashes or wraps silently.
2. **Linear models of relative risk** (TL11–TL15). TLO multiplies an annual probability by relative risks, with logistic models where it needs odds.
   - In v4, `ss.by * ss.by` and `ss.prob * ss.by` raise outside a transition.
   - `ss.probperyear(1) * 0.8` moves nobody, and `ss.probperyear(0.8) * 0.2` scales the hazard rather than the probability.
   - A branch probability over 1 is used as 1 with no note.
3. **Relative risk on an outcome, and links between diseases** (TL16–TL22). The natural way to scale one branch (`adjust=`, or `ss.modify` of the transition) silently speeds up the dwell instead.
   - Preterm birth stays at 0.044 where 0.14 is right, and anaemic women's pregnancies run 4.6 times faster.
   - A product's efficacy against a transition written with `dur=` does nothing as agents, while the ODE applies it in full.
4. **Births, deaths and their causes** (TL23–TL34). Deaths from two modules in one step are counted twice, and no cause is kept per person.
   - `ss.Deaths` gives 2013's deaths the 2015 rate.
   - `ss.Pregnancy` gives women over 45 no births, counts the unborn as alive, reads `p_loss` per step, and ignores `rr_ptb`.
   - `ss.Births` picks parents from everyone, so 52% of "mothers" are men.
5. **The health system** (TL35–TL49). The deliveries vocabulary covers one cascade well, but has no shared resource:
   - no capacity in staff minutes shared across diseases;
   - no queue;
   - no appointment that isn't a product.

   Some of what exists is silently wrong. A closure over a coded health-system module reads a copy and treats nobody. Three doses of one vaccine are one dose with three chances. Routine treatment is offered once per lifetime.
6. **The burden of disease** (TL50–TL53). There is no DALY, YLL, YLD or disability weight anywhere in v4. A staged NCD counts only its first stage as "infected", so a test finds 649 of 1,637 cases.

**Counts:** 72 new findings (TL1–TL72), about 25 of them related to an existing finding, plus about 20 known findings that TLO hits again. 24 are silent wrong answers. Most of those are a few lines each, and most of the rest are in `ss.Pregnancy`, `ss.Births`, `ss.Deaths` and the agent engine's branches. Groups 1, 5 and 6 are design work, which this review proposes:
- `ss.People.from_df()`, people properties in rates, and `ss.Properties`;
- a shared resource with visits and a queue;
- `ss.disability` and `ss.Burden`.

With those, everything in TLO except its per-person continuous quantities can be declared. That leaves worm burdens, the list of injuries, and MUAC, which stay coded and plug in.

## What a TLO port could look like

The target, in proposed spellings (the names marked `# proposed` don't exist today). Natural history, risk factors, the care pathway, the health system's constraints and the burden are declared. Schisto's worm burdens and the injury list stay coded and plug in.

```python
import starsim as ss, tlo4

people = ss.People.from_df(tlo4.census(2010), scale=ss.geo(pop=14.5e6))            # proposed: per-person columns (TL1)
staff = ss.Resource(tlo4.minutes(by=['district', 'level', 'officer']), per=ss.days(1)) # proposed: one constraint for every disease (TL39)
opd = ss.Visit(uses={staff: dict(Clinical=27, Nursing=18, Pharmacy=9.45)}, level='1a') # proposed (TL41)

N, LGD, HGD, S1, S2, S3, S4 = ss.states(7)
oc = ss.Disease(
    onset    = N >> LGD @ ss.prob(4e-6, ss.months(3)) * ss.rr(age=ss.People.age**2.3, li_tob=2.1, li_ex_alc=1.6),  # proposed: properties in rates (TL9, TL11)
    progress = LGD >> HGD @ ss.prob(0.04, ss.months(3)),  # ... and so on to S4
    death    = S4 >> ss.null @ ss.prob(0.4, ss.months(3)),
    disability = ss.disability({S1: 0.288, S2: 0.288, S3: 0.451, S4: 0.54}),            # proposed (TL50)
).stratify(tx=[False, True], adjust={'progress': [1, 0.1]})                           # treatment slows its own stage (works today)
dysphagia = ss.Disease(...)

care = ss.screen(endoscopy, on=dysphagia.onset, visit=opd, stock=tlo4.availability('endoscope'),        # proposed: visit=, stock= (TL41)
                 on_positive=[ss.treat(surgery, where=~oc.S4, queue=ss.days(7), record=dict(stage_dx=oc.state)),  # proposed: lists, queue=, record= (TL40–TL42)
                              ss.treat(palliative, where=oc.S4)])

sim = ss.Sim(people=people, diseases=[oc, dysphagia, tlo4.cmd(), tlo4.hiv(), tlo4.tb()], connectors=tlo4.Schisto(),
             resources=staff, interventions=care,
             demographics=[ss.Births(wpp.asfr), ss.Deaths(wpp.mx, net_of=tlo4.gbd_modeled_share)],    # proposed: net_of= (TL23)
             analyzers=ss.Burden(age_limit=90, cap=1), start=2010, stop=2030, dt=ss.months(1))        # proposed (TL50)
res = ss.parallel(perfect=sim.new(resources=None), constrained=sim, n_runs=10).diff('perfect')
res.burden.dalys.by('cause').plot()
```

Today the parts of this that run need these workarounds:
- writing every state and dimension level after `sim.init()`, including three internal arrays of the agent engine for time already spent in a state (TL1, TL2);
- functions of the sim that return `.raw`, never values per living agent (TL8), for any risk factor that isn't a dimension;
- linear models built as `np.einsum` arrays (TL11);
- relative risk on a branch spelled `'disease.transition.destination'`, which is undocumented (TL16);
- `ss.modify(..., multiply=0)` in place of a product's efficacy on a `dur=` transition (TL17);
- a coded burden analyzer reading `daly_weights()` from coded modules, and no cause per person for declared deaths (TL23, TL50);
- a coded health system reached through bound methods, never closures (TL35), with a separate product for each dose (TL36);
- `ss.Births` with `parent` ignored, and districts copied from mothers by hand (TL27, MA18).

## The port today, and after the changes

| Task | Today | After the changes below |
|---|---|---|
| Start from TLO's census and lifestyle | Write arrays after `sim.init()`; the initial results point goes stale; ODE ignores everything but age (TL1, TL2) | `ss.People.from_df(df)`, under every solver |
| A risk factor that is a property, not a dimension | A function of the sim, agents only, crashes after the first death unless it returns `.raw` (TL8, TL9) | `ss.rr(li_tob=2.1)` in a rate |
| Stratify by a property people already have | Everyone silently reset to the first level (TL3) | The property kept (fixed in the working tree) |
| 32 districts × lifestyle | Crash, or silent wrap past 32,767 groups (TL6); 400 bytes per group (TL7) | Runs; only occupied groups stored |
| Relative risk of preterm birth if anaemic | Dwell 4.6 times faster, share unchanged (TL16) | The branch scaled |
| ART as a product against AIDS onset | No effect as agents, full effect as an ODE (TL17) | The same under both |
| Deaths from two modules | Counted twice; no cause per person (TL23) | One death, one cause; `net_of=` for background |
| WPP death rates for 2010–2014 | 2015's rates from mid-2012 (TL24) | Each period's rate held |
| `ss.Pregnancy` with 5-year ASFR | No births over 45; unborn counted alive; `p_loss` per step; `rr_ptb` ignored (TL25, TL26, TL29, TL30) | As given |
| Staff time shared by every disease, with a queue | A coded module (152 lines) | `ss.Resource`, `visit=`, `queue=` |
| Three doses of pentavalent | 351 / 31 / 5 delivered (TL36) | 351 / 327 / 327, with a dose count |
| Treatment of repeated episodes | Once per lifetime (TL36) | Every episode |
| DALYs by cause | No vocabulary; a 65-line analyzer; declared deaths have no cause (TL50) | `ss.disability`, `ss.Burden` |
| Screen for a staged cancer | Finds stage 1 only (TL52) | Every stage after onset |

## What should change in v4

### 1. People with many properties

TLO's people are a dataframe of about 150 columns. A disease reads any of them in a linear model, and newborns inherit some from their mothers. v4 can carry the same properties, as coded arrays or as dimensions, but the declared vocabulary reaches them only as dimensions, and the agent engine's dimensions don't scale to TLO's.

- **TL1 (dead end, crashes): a declared disease can't start from a per-person table.**
  - `init=dict(I=fn)` raises `TypeError: '>=' not supported between instances of 'function' and 'int'` in `Engine.init_fractions()`. Spec 03 says `init=` takes the parameter ladder, functions included. A per-agent array raises `The truth value of an array ... is ambiguous`.
  - `ss.People` takes no per-person values. Under a compartmental solver it draws 100,000 ages from `people.age.default` and ignores every other property (`sim.py:712`).
  - Every port wrote its states after `sim.init()`. Then the result at index 0 is stale (`n_I` 100 where 694 were written), and the guessed `init_prev` of 1% and the unborn (TL26) have to be cleared by hand.
  - Repros: `infections/probes/repro_init_function.py`, `ncds/repro_init_function.py`.
  - Fix: accept a function or per-agent array in `init=` as agents (about 20 lines). Then `ss.People.from_df(df)`: columns become properties, counted into groups and joint splits for a compartmental solver (M).
  - Related to MA24.
- **TL2 (vocabulary, with a silent trap): no spelling for time already spent in a state at the start.**
  - TLO draws each prevalent HIV infection's time to AIDS from its years since infection: 34 of 380 should progress in the first year. Seeded agents get a fresh dwell instead, and the port gave 3.
  - Writing the engine's `ti_exit_I` alone is silently rescaled at the first step (94 → 112, 8 → 4, 200 → 139 steps), because `mult_I` and `exit_I` keep their defaults. Writing all three arrays gives 35.
  - Repro: `infections/probes/probe11_ti_exit.py`.
  - Proposed: `init=dict(I=..., since=...)`. M.
  - Related to CR4/L14 and MA's mid-pregnancy case (TL69).
- **TL3 (silent): stratifying by a dimension that is already a people property resets it to the first level.**
  - With `tob` given (30% True), `stratify(tob=[False, True])` leaves 0 smokers. `describe()` says "[guessed: everyone at False]", while spec 05 says an existing property is used.
  - In the COPD port this cut progression from 1,100 to 704 until the values were written back.
  - Repro: `ncds/repro_strat_overwrites_property.py`.
  - Fixed in the working tree by "behavior steps 1 and 2" (`Groups.kept`); the repro should become a test.
- **TL4 (silent): a split over two dimensions overrides the split already given to one of them.**
  - TLO's wealth-by-urban table (rows are P(wealth | urban)) is read as a joint distribution. It gives urban 0.497 where `split=[0.83, 0.17]` was given, with no note.
  - The same table with `states=` is read as conditional.
  - Repro: `framework/repros/repro_joint_split_overrides.py`.
  - Fix: raise `InconsistentDimension`, as for two explicit splits. About 5 lines.
- **TL5 (dead end, spec drift): only one split may cover several dimensions.**
  - Each of Lifestyle's starting shares depends on other properties, and so do diarrhoea's lifestyle factors. A second such split raises `Only one split= can cover several dimensions at once`.
  - Spec 05 says a split that depends on another dimension is an `ss.by` over both.
  - Repros: `framework/repros/repro_conditional_split.py`, `child/repro_two_joint_splits.py`.
  - Fix: apply conditional splits in turn. M.
- **TL6 (crash, or silent wrap): more than 32,767 groups fail as agents.**
  - At 65,536 the error is `OverflowError: Python integer 32768 out of bounds for int16` in `Groups.group_raw()`.
  - Where group codes are updated incrementally they wrap round silently, then crash later in `Agents.wrote()` with `'list' argument must have no negative elements`. The declared lifestyle with 32 districts (327,680 groups) and ALRI's attributes (110,592) both hit this.
  - The ODE runs 65,536 groups.
  - Repros: `framework/repros/repro_many_groups.py`, `child/repro_int16_groups.py`.
  - Fix: choose the dtype from the number of groups. About 5 lines.
- **TL7 (cost): as agents, v4 allocates every group of the grid**, though each agent is in one.
  - 2^20 groups take 726 MB at init for 20,000 agents, about 400 bytes per group.
  - All of Lifestyle is about 2.2 million groups, or 1.4e9 with sex, age and districts.
  - Data: `framework/out/probe_lifestyle_groups.txt`.
  - Fix: store only occupied groups, and compute rates from per-dimension factors. L.
  - Until then, TL9's properties are the way to carry risk factors that aren't stratifications.
- **TL8 (crash): a rate given as `fn(sim)` returning a value per living agent crashes after the first death.**
  - As agents the value is indexed by UID (`agents.py:536`), giving `IndexError: index 4768 is out of bounds for axis 0 with size 4768`. The function isn't named.
  - Spec 03's own example, `0.01 + 0.1 * (sim.people.age > 65)`, hits this in any model with deaths. Returning `.raw` works, but nothing says so.
  - Repro: `ncds/repro_fn_alive_index.py`.
  - Fix: accept either length, and name the function on a mismatch. S.
  - Related to MA12.
- **TL9 (vocabulary): a people property can't be used in a rate.**
  - `0.01 * (ss.People.age > 20)` raises `A condition cannot be used in a rate: | on states is a condition`, about a `|` that wasn't written.
  - `np.maximum(age - 20, 0)**1.1` fails on the lazy object.
  - Repro: `ncds/repro_property_in_rate.py`.
  - Proposed: let `ss.by(li_tob=[1, 2])` (or `ss.rr(li_tob=2)`) name a property that isn't a dimension, and allow arithmetic on `ss.People.age`.
    - Agents read the value per person.
    - A compartmental solver asks for the stratification with `CapabilityError`, as conditions do, or uses group midpoints with a note.
  - M. This is the single change that would make TLO's linear models declarable as agents.
- **TL10 (vocabulary): a module of properties needs a fake disease, and population results are never by group.**
  - Declared lifestyle needs `stay = S -> S @ 0`. Then `describe()` calls `S` "infected … cured by ss.treat".
  - `sim.results.n_alive.by('patch')` raises "not stratified".
  - Naming the fake module `pop` makes `sim.results.pop` the dict method.
  - Data: `framework/out/probe_results_by_district.txt`.
  - Proposed: `ss.Properties(tob=[False, True], start=ss.transfer(...))`, and population results by group whenever the sim is stratified. M.
  - Related to D23.

### 2. Linear models of relative risk

Every TLO rate is a `LinearModel`: an intercept (usually an annual probability) times relative risks by property, sometimes logistic. In v4 the multiplicative form works as a number times one `ss.by` over several dimensions, but the obvious spellings fail.

- **TL11 (friction, spec drift): arithmetic on `ss.by` can't build a linear model outside a transition.**
  - `ss.by(type=...) * ss.by(dehyd=...)` raises `TypeError: unsupported operand type(s) for *: 'by' and 'by'`.
  - `ss.peryear(0.0016) * ss.by(sex=...)` and `ss.prob(...) * ss.by(...)` raise `IncompatibleTypes ... has no meaning`.
  - `ss.days(1) * ss.by(...)` and `ss.peryear(ss.by(...))` raise `TypeError`.
  - The same products after `@` work, so `A >> B @ ss.prob(p) * ss.by(...)` runs and `ss.Transition(A, B, rate=ss.prob(p) * ss.by(...))` doesn't. Spec 04 says arithmetic works on `ss.by`.
  - Diarrhoea's case fatality became a 900-number `np.einsum` array.
  - Repros: `framework/repros/repro_by_arithmetic.py`, `child/repro_by_arithmetic.py`, `ncds/repro_by_times_by.py`.
  - Fix: outer products of `ss.by` by dimension name, and typed values times `ss.by`. About 30 lines.
  - Related to FP31.
- **TL12 (silent): `ss.probperyear(1) * 0.8` moves nobody, and `ss.probperyear(0.8) * 0.2` scales the hazard.**
  - The first gives 0 of 10,000 where 8,000 are expected: an infinite hazard times 0.8 is NaN, and the only sign is a NumPy RuntimeWarning.
  - The second gives 2,789 where TLO's convention (probability × relative risk) gives 1,600. That is v4's rule (02-types), but every TLO port will write it.
  - Repro: `child/repro_probperyear_times_by.py`.
  - Fix: infinity times k stays infinite, or raise. Accept `ss.by` inside the typed constructors (`ss.probperyear(ss.by(...))`, one probability per level). Add a migration note. About 10 lines plus docs.
- **TL13 (silent): a branch probability over 1 is used as 1.** `p=ss.by(type=[0.1, 1.5])` kills 5,854 of 5,901, with no `ClippedToRange`, which the spec promises. Repro: `child/repro_by_arithmetic.py` case 4. A few lines.
- **TL14 (vocabulary): no logistic or odds form.**
  - Care-seeking, ALRI deaths and the initial prevalences multiply odds, not probabilities. Today that takes a hand-made table over five dimensions, or a function (agents only).
  - Proposed: `ss.odds(base, rr=ss.by(...))`. Small.
  - Duplicate of FP26.
- **TL15 (vocabulary): no form for a parameter defined as a population mean.**
  - TLO rescales 10 maternal linear models at initialization so that their mean over all women 15–49 equals the parameter. Without that, spontaneous abortions are 738 against TLO's 504; realized, they are 0.127 per pregnancy where the parameter implies 0.223.
  - Proposed: `ss.mean_matched(..., over=..., at='init')`. Low priority; the port computes it by hand.

### 3. Relative risk on an outcome, and links between diseases

TLO's maternal module applies about 40 concurrent conditions as relative risks on single outcomes at fixed checkpoints, and its diseases raise each other's risks constantly (HIV on TB, TB on AIDS, CMD conditions on each other). v4's modifiers between diseases work well (`ss.modify(tb.infection, multiply=5, where=hiv.I)`, with `what_modifies()` listing them). But the outcome-level ones fail silently.

- **TL16 (silent, spec drift): `stratify(adjust=)` and `ss.modify(transition)` on a branching transition with a dwell change the dwell, not the branch.**
  - Pregnancy: `ss.modify(mnh.ga31, multiply=4.58, where=anaemia.A)` on `G31 -> G35 | SB | PT31` (a 4-week dwell with `p=`) leaves the preterm share at 0.044 (0.038 without it; 0.14 expected). It also cuts the time in G31 from 4.15 to 2.51 weeks: anaemic women's pregnancies run 4.6 times faster.
  - Diarrhoea: `adjust={'death': [1, 5]}` on bloody episodes gives 1,702 deaths against 1,665 unadjusted, where about 8,500 are expected, and shortens the episodes.
  - The undocumented `'mnh.ga31.PT31'` gives 0.128, and `ss.modify(death, where=type=='bloody')` gives 7,067. Both leave the dwell alone.
  - The docstring says a modifier scales "its probability if it is a branch with `p=`". Spec 05 says adjustments divide durations, and specs 03/07 that both combine by one rule. They disagree.
  - Repros: `mnch_burden/probe_branch_rr.py`, `child/repro_adjust_branch.py`.
  - Fix: an adjustment or modifier on a transition whose exits are branches with `p=` scales the non-remainder branches (or refuses, naming the branch targets). Document `<disease>.<transition>.<destination>` as a target. About 15 lines.
  - Related to FP13.
- **TL17 (silent; solvers disagree): as agents, a product's efficacy against a transition written with `dur=` does nothing.**
  - ART as `ss.Treatment(efficacy=dict(aids=1))`, delivered to all 3,351 infected: AIDS onsets over 2011–13 are 2,252 with it and 2,252 without as agents. The ODE gives 0, and so do agents when the same exponential is written as a rate.
  - `ss.modify(hiv.aids, multiply=0)` gives 104. With a Weibull dwell: agents 3,582 against 3,582; ODE 0.
  - Likely cause: C17's rescaling of already-drawn dwells consults `sim.modifiers`, where product efficacies aren't registered.
  - Repro: `infections/probes/repro_product_efficacy_dwell.py`.
  - Fix: let it see product efficacies, plus a conformance test that a product, a modifier and an adjustment on a `dur=` transition agree under every solver. About 10 lines.
  - Related to C17 and MA8; not in the handoff.
- **TL18 (friction): a case-fatality branch can't leave the survivors as the remainder.**
  - `ss.Transition(acute, ss.null, p=0.2)` beside `ss.Transition(acute, ever)` raises `UndrivenTransition`.
  - One transition to `[ss.null, ever]` with only the death's `p=` raises `BadBranchSplit`.
  - With a per-person case fatality, the survivors need a second function, `1 - fatal(sim)`.
  - Repro: `ncds/repro_branch_to_null.py`.
  - Fix: an immediate branch with no `p=` takes the remainder. S.
  - Related to L14 (whose remainder rule covers `dur=` exits only) and HP16.
- **TL19 (dead end): a split can't depend on another disease's state.**
  - Persistent diarrhoea is 33 times likelier in severely stunted children. As a split, `stunting` gives `KeyError: 'The model is not stratified by "stunting"'`, and `ss.modify` can't target a split (`UnmodifiableTarget`). It works only when stunting is a dimension of the same sim.
  - Repro: `child/probe_interaction.py` B.
  - Proposed: `ss.modify('diarrhoea.course.persistent', multiply=33.3, where=stunting.severe)`. M.
- **TL20 (vocabulary, spec choice): a drawn dwell always follows its current age group.**
  - TLO fixes time to AIDS by age at infection. With everyone infected at 24.5 and the 25–29 group 10 times faster, v4 gives 11,052 onsets by 18 months, against about 2,350 under TLO's rule.
  - The uncommitted KI6 work extends this rescaling to every dimension people move along, which would make it the only behavior.
  - Repro: `infections/probes/probe19_dwell_by_age_at_entry.py`.
  - Proposed: `adjust=..., at='entry'`. M.
  - Related to KI6, C17 and HP18.
- **TL21 (vocabulary): an event in one disease can't move people in another.**
  - TLO starts AIDS the moment TB becomes active. v4 can only scale a rate, so the port uses `ss.modify(hiv.aids, multiply=100, where=tb.active)`, a magic number that lags a step.
  - Proposed: `aids_tb = I >> A @ ss.on(tb.infection)`. M.
  - Related to FP25 and ST13.
- **TL22 (dead end): no code at entry to a state, no per-episode attributes, and no per-person counts.**
  - Road traffic injuries draw a list of up to 8 injuries (75 codes) with severity scores when a crash happens. ALRI draws complications; wasting draws MUAC.
  - The port draws them inside the death branch's `p=` function, which v4 happens to call once per step with exactly the entering agents (365 calls, 4,583 UIDs, for 4,591 entries).
  - Stunting's `1.04 ** episodes` and wasting's "ever wasted" need per-person counts and sticky flags (`child/probe_interaction.py` C).
  - Proposed: `ss.Transition(..., on_enter=fn)`, and `ss.count(diarrhoea.onset)` as a per-person quantity. M.
  - Related to HP19 and FP30.
- **TL70 (vocabulary): counts at another level.** TLO's HIV denominator, and pooled prevalence among adults of the other sex, need a count at another level of a dimension. The port uses a contact matrix of each group's share of adults, computed by hand and fixed while the population ages. Proposed: `ss.contacts(..., proportionate=True)`, renormalized each step. S to M. Related to MA11.

### 4. Births, deaths and their causes

TLO's `do_death(cause=)` records one cause per person. Its `OtherDeathPoll` applies WPP mortality net of the causes the model represents, and every module's `on_birth(mother, child)` sets the newborn's properties from the mother's. v4's built-in demography matches TLO's population within noise, and `offspring=` with `people.parent` (FP18, fixed at `e4dbda48`) covers the simple cases. The rest is silently wrong in several places.

- **TL23 (silent): deaths from two modules in one step are both counted, and no cause is kept per person.**
  - A declared death at 1 a year beside `ss.Deaths` at 1 a year, monthly steps: 9,127 + 8,921 = 18,048 deaths by cause against 17,264 people who died, so 4.5% are counted twice. Two coded modules killing the same 1,000 people record 1,000 each.
  - Agents are removed at death with no cause, so a burden analyzer puts all 224 non-Mockitis deaths under "Other", where 164 were ChronicSyndrome's.
  - There is no "net of modeled causes", so adding a disease's deaths to WPP all-cause mortality double counts them.
  - Repros: `framework/repros/repro_cause_of_death.py`, `mnch_burden/probe_declared_death_cause.py`.
  - Fix: `request_death(uids, cause=)` resolved once per person per step into `people.cause` and `people.age_at_death`, with `ss.Deaths` writing "Other" (about 20 lines). Then `ss.Deaths(net_of=)` (M).
- **TL24 (silent): `ss.Deaths` reads rates labeled by the start of their period as the nearest year.**
  - WPP and TLO label the 2010–2014 rates "2010", so from mid-2012 people die at the 2015 rate. On TLO's data that is 790 deaths over 5 years against 861 with each period's rate (−8%).
  - In a clean case (0.04 a year, then 0.01 from 2015), deaths per year are 803, 716, 500, 168, 178, where about 784 to 668 is right.
  - `ss.death(rate=ss.timeseries(...))` holds each value and is right. `ss.Pregnancy` interpolates linearly. That is three readings of the same kind of data in v4's own modules.
  - Repro: `framework/repros/repro_period_rates.py`.
  - Fix: hold each value until the next, as `ss.timeseries` does. About 10 lines.
- **TL25 (silent): `ss.Pregnancy` gives women aged 46–49 no fertility when the data are in 5-year groups.**
  - The zero column goes at `max_age + 1`, 46. Women aged 45.5 have 1,973 pregnancies a year; women aged 47.5 have 0, where about 1,813 are expected.
  - Repro: `framework/repros/repro_asfr_last_group.py`.
  - Fix: put the zero at the last edge plus the group width. 2 lines.
- **TL26 (silent): with `ss.Pregnancy`, the unborn are counted alive and seeded with infection.**
  - Agents made at conception have `alive=True`. So `n_alive` is 3.2% high in the 5-year port (23,367 against 22,652 born), and a declared state's count includes them.
  - `I = ss.State(init=0.5)` puts 51 of 110 fetuses in `I`.
  - Repros: `framework/repros/repro_unborn_counted_alive.py`, `infections/probes/repro_mtct_unborn_and_direction.py`.
  - Fix: count and place only people who are born; report the unborn separately. About 10 lines.
- **TL27 (silent): `ss.Births` draws `people.parent` from everyone alive.**
  - Of 1,646 newborns, 51.5% have a male parent and 25.9% a parent under 15. Anything reading the parent uses a father or a child as the mother: mother-to-child transmission, a newborn's district, a coded `on_birth`.
  - Repro: `infections/probes/repro_births_parents.py`.
  - Fix: draw parents from women of reproductive age (by the birth rate's ages), or leave `parent` unset. About 5 lines.
  - Related to FP18.
- **TL28 (silent): a population split over age × another dimension resets ages and mis-assigns newborns.**
  - `split=ss.by([[0,1],[1,0],[1,0]], dims=['age','wealth'])` makes each age group a third of the population (3,335 under 1, against 164).
  - Newborns get the split's total over ages: 303 poor and 154 rich, where all should be rich. In the stunting port, 236 of 276 newborns got the starting children's breastfeeding level.
  - Repro: `child/repro_newborn_levels.py`.
  - Fix: newborns draw from their age group's row. About 15 lines.
  - Related to FP20 (the age reset may be fixed in the working tree) and MA18.
- **MA18, the rest (duplicate):** newborns' district is drawn from the split, so 4.9% are in their mother's district against TLO's 100%. The first half (all newborns in the first patch) is fixed at `e4dbda48`; this half isn't fixed in the working tree either. Copy the parent's level for space dimensions, about 5 lines (after TL27).
- **TL29 (silent): `ss.Pregnancy(p_loss=)` is a probability per time step.** `p_loss=0.01` loses 44% of pregnancies at a weekly step and 26% at a monthly one. Repro: `mnch_burden/repro_pregnancy_rates.py` (a). Fix: a rate, or `ss.prob(x, ss.weeks(1))`. About 5 lines. Related to L1.
- **TL30 (silent): `ss.Pregnancy`'s `rr_ptb` and `rr_ptb_age` can't change the preterm share.**
  - The durations drawn each step are sorted and the shortest given to the highest risk, so a relative risk only reorders who is preterm. `rr_ptb_age=3` for everyone gives the identical 0.046 and 1,252 births.
  - Repro: `mnch_burden/repro_pregnancy_rates.py` (b).
  - Fix: apply the relative risk to the preterm probability. About 20 lines.
- **TL31 (silent trap): a single `beta` on a maternal network transmits from child to mother.**
  - `beta=dict(prenatalnet=0.33, breastfeedingnet=0.12)` infects 28 of 114 uninfected mothers from their seeded fetuses and infants; `[b, 0]` pairs infect none.
  - Repro: `infections/probes/repro_mtct_unborn_and_direction.py`.
  - Fix: default the maternal networks to mother → child, or note it. About 5 lines.
- **TL32 (vocabulary): susceptibility by age applies to every route.**
  - `adjust={'rel_sus': [0, 1]}` by age, meant for sexual transmission, also halves mother-to-child transmission (0.239 → 0.109). The port moved the restriction into the contact matrix instead.
  - Repro: `infections/probes/repro_route_susceptibility.py`.
  - Proposed: values by level in a route's `beta`. M.
- **TL33 (friction): a per-birth probability on a network has to be converted by hand.**
  - `beta=0.33` on `PrenatalNet`, meant per birth, is read per month and infects 98% of children of infected mothers. The right value, `-log(1 - 0.22)/9` per month, assumes 9 months, while `ss.Pregnancy` draws 32–42 weeks.
  - Proposed: `ss.per_edge(0.22)`. S to M.
- **TL34 (friction): coded modules have no birth hook, and `offspring=` is per transition, not per branch.**
  - Inheriting from the mother means scanning for newborns each step; with a 3-month module step, newborns go up to 3 months without their properties.
  - Delivery is 4 transitions per labour state because each newborn type needs its own transition. Writing it as one immediate split crashes (HP16).
  - Proposed: `Module.on_birth(parents, children)`, `default=ss.inherit()`, and `offspring=` per branch. M.

### 5. The health system

TLO's `HealthSystem` is what makes it TLO. Every condition's care goes through it:
- appointments with footprints in staff minutes by officer type, facility level and district;
- consumables available with a probability by facility and month;
- beds and equipment;
- a queue ordered by priority;
- appointments that don't run, and the counterfactual of a perfect health system.

Some TLO facts matter here:
- Its default mode 1 is elastic, with no staff limit in practice.
- Squeeze factors are vestigial.
- The generic first appointment has used no staff time since 2023.
- At small populations, mode 2's overtime rule ignores the capability scaling altogether (TLO bug T1 in `healthsystem/REPORT.md`).

v4's deliveries vocabulary covers a single cascade well:
- `ss.screen(test, on=dis.infection, coverage=availability, on_positive=ss.treat(...))` with TLO's availability by district and month gives prevalence 0.1142 against TLO's 0.1138, and runs as an ODE too;
- spurious symptoms are a screen on another disease's onset;
- beds are a capacity function on a move into hospital;
- results come by district for free.

What it lacks is anything shared.

- **TL35 (silent): a closure over a coded module reads the original, not the sim's copy, and does nothing.**
  - `ss.Sim` copies its modules. A bound method is copied with its module and works; a closure from a factory method, or a lambda naming the module, keeps the original.
  - Treated: 995 with a bound method, 0 with `hs.serve()` or `lambda sim: hs.served_now`, with no error; infected at the end 331 against 605.
  - The factory form (`hs.book(...)`, `hs.availability(...)`) is the natural one for a health system.
  - Repro: `healthsystem/repro_closure_copy.py`.
  - Fix: remap closure cells through the deepcopy memo, or warn when a closure holds a module that isn't in the sim. About 20 lines.
  - Related to MA14, of which this is the general case.
- **TL36 (silent): a held product can't be given again.** One vaccine at three ages is one dose with three chances, and routine treatment is offered once per lifetime.
  - `ss.vaccinate(penta, at_age=w, coverage=0.9)` for w = 6, 10, 14 weeks delivers 351 / 31 / 5. Three separate products deliver 351 / 327 / 327, as TLO's EPI does, but there is no dose count (TLO's `va_dtp`).
  - Routine `ss.treat(Treatment(cure=1), coverage=1)` against repeated diarrhoea leaves 619 deaths of 1,752, against 0 with `on=onset`.
  - Repros: `healthsystem/probe_epi.py`, `child/probe_treatment_branch.py`.
  - Fix: a held product is released when its effect ends; routine treatment re-arms when the person leaves the infected states; warn when two deliveries share a held product; a doses spelling. S to M.
  - Related to MA15, still open (Stage 4 of the plan).
- **TL37 (silent, or crash): under a compartmental solver, a test delivered on one disease's event that detects another disease's state gives wrong answers.**
  - Cancer onset at 0.05 a year, a symptom at 0.2 a year, the test on symptom onset detecting cancer. Uncrossed: agents diagnose 96 (analytic 104), the ODE 219.6. Crossed: agents 173, the ODE 0.0.
  - With the 7-state cancer, the ODE crashes in `conditions.py:475` (`operands could not be broadcast together ... (2,3360) and requested shape (7,3360)`).
  - Repro: `ncds/repro_crossed_on_event.py`.
  - Fix: evaluate `detects=` on the event's flow by joint group (the crossing has it), and refuse or cross the uncrossed case. M.
  - Related to DF1 in the care-cascade design.
- **TL38 (silent, marked guessed): a state entered only by a delivery's `move=` becomes the first state.**
  - With hospital `H` filled only by `ss.Delivery(move='Sev -> H')`, all 5,000 people start in hospital ("init: H = 1.0 [guessed]"): 34,001 bed-days against 4,066.
  - The rule ("the earliest state no transition moves people into") ignores deliveries' moves, and states reached only by care are common in a health-system model.
  - Repro: `healthsystem/probe_beddays.py`.
  - Fix: count delivery destinations. About 5 lines.
  - Related to CR3/L4.
- **TL39 (vocabulary): no capacity shared between deliveries.**
  - Two diseases sharing a clinic of 10 a day: `ss.perday(10)` on each treats 16.7 a day. A fixed split of 5 each wastes the quiet disease's share. Only a capacity function reading the other delivery's `delivered[sim.ti]` gives exactly 10, with strict priority to the first. A function for `capacity=` is accepted but undocumented.
  - Repro: `healthsystem/repro_shared_capacity.py`.
  - Proposed: `ss.Resource(ss.by(district=..., officer=...), per=ss.days(1))`, drawn on by visits, with `sim.results.<resource>` giving minutes used and appointments by type, level and district. L.
  - TLO's own capacity is only binding in mode 2, but the counterfactual "what if staff were the limit" is what TLO's health-system papers are about.
- **TL40 (vocabulary): a capped delivery has no queue; those not reached are lost.**
  - TLO's per-district capacity written as 32 capped follow-ups loses 718 people: 2,825 treatment appointments against TLO's 3,559, and prevalence 0.1256 against 0.1168. It also takes 31.5 s against 0.9 s for the coded health system (192 deliveries).
  - Repro: `healthsystem/v4_toyitis_declared.py`.
  - Proposed: `queue=ss.days(7)` (TLO's `tclose`) and `priority=` on a capped delivery, with "never ran" as a result. About 40 lines.
  - Answers "When capacity runs out" in [19-open-questions.md](../design/19-open-questions.md).
- **TL41 (vocabulary, and a crash): no appointment without a product, and only one follow-up.**
  - Capacity counts product recipients. So an appointment that finds the drug out of stock can't use capacity, and appointments can't be counted by type; the workaround is a "test" with sensitivity 1.
  - `on_positive=[a, b]` (treatment for stages 1–3, palliative care for stage 4) is accepted, then fails at the first positive with `'list' object is not callable`.
  - Repro: `ncds/check_vocabulary.py` item 3.
  - Proposed: `ss.Visit(uses=..., level=...)` and `visit=`/`stock=` on any delivery; lists in `on_positive=`. M.
  - Related to decision 3 of the care-cascade design.
- **TL42 (vocabulary): a property set at an event from the person's state has no spelling.**
  - TLO records the stage at which treatment was given, for progression and for DALYs. A delivery's `move=` is the same for everyone.
  - Proposed: `record=dict(stage_dx=oc.state)`. M.
- **TL43 (dead end): a delivery can't be triggered by a transfer.**
  - `on=d.onset`, with `onset = ss.transfer(sym=True)`, raises "on= of a delivery is an event … not on=transfer(...)", though spec 05 says a transfer is an `ss.Transition`. So a symptom can't be a dimension of the disease and still lead to care; dysphagia had to be a second disease with six modifiers.
  - Repro: `ncds/repro_on_transfer.py`.
  - Fix: accept transfers. S.
- **TL44 (vocabulary): a product needs a declared disease.** `ss.vaccinate` with no declared disease raises `UnknownDisease`, while TLO's EPI records doses for nine diseases it doesn't model. Repro: `healthsystem/probe_epi.py`. S to M.
- **TL45 (vocabulary): "test not done" isn't a result.** TLO's `DxTest` returns nothing when its consumable is out, and falls back through a list of tests (RDT, then microscopy). Proposed: `on_not_done=` and a fallback list. M.
- **TL46 (vocabulary): presentation is per disease, and level isn't a property of a visit.**
  - TLO runs one generic first appointment over all of a person's symptoms, at a level drawn by care-seeking.
  - v4 gives two first appointments to someone with two diseases, and can't report appointments by level (611 / 3,132 / 923 at levels 0, 1a and 2 in TLO).
  - Design work, with TL39 and TL41.
- **TL47 (crash): `capacity=ss.by(district=...)`** raises `int() argument must be ... not 'by'`. Repro: `healthsystem/repro_capacity_by.py`. S for a message; M to support capacity per level.
- **TL48 (friction): event-triggered care acts a step late,** because interventions run before diseases. At TLO's 3-month step, diagnosis comes 3 months after the symptom, against 0–4 days in TLO. It made no difference here, but would for anything time-sensitive. Document it, or offer a same-step option. S.
- **TL49 (spec drift): the time-index warning contradicts the built-in deliveries.**
  - A coded clinic writing `results.served[self.ti]` matches `tx.delivered` exactly, but is warned "a result goes at self.ti + 1". Following the warning puts it a step away from the deliveries.
  - Repro: `healthsystem/repro_result_index.py`.
  - One convention for both. S.
  - Related to FP1 and MA24.

**Design note: a shared resource.** TL39–TL46 are faces of one gap. v4 has deliveries, each with its own coverage and capacity, but nothing they share. A small vocabulary would make TLO's health system declarable, and serve any model with a constrained service (bed capacity in COVID models, test kits in MEmilio's test-and-trace, ART slots in Atomica's programs):
- `ss.Resource` (a quantity per unit time by dimension);
- `ss.Visit` (a footprint on resources at a level);
- `visit=`, `stock=`, `queue=` and `priority=` on deliveries;
- lists of follow-ups.

As agents it is TLO's queue. In a compartmental run, a resource caps the summed flows of the deliveries that use it, shared by priority. The coded 152-line health system in `healthsystem/v4_toyitis.py` is a working prototype of the agent side.

### 6. The burden of disease

TLO's contract with a disease is three things:
- `CAUSES_OF_DEATH` and `CAUSES_OF_DISABILITY` (mapped to GBD labels);
- `report_daly_values()` each month;
- `do_death(cause=)`.

`HealthBurden` turns these into YLL (90 minus age at death), YLD (disability weights, capped at 1 per person) and DALYs by cause, age, sex and district. This is TLO's main output.

- **TL50 (vocabulary): v4 has no burden of disease.**
  - No DALY, YLL, YLD or disability weight appears anywhere in the spec or code. A declared YLD rate works as a weighted sum of states (`0.187*J + 0.215*K + 0.402*JK`, and a weight can be an `ss.par`).
  - But a time integral of a stock isn't a result kind: `resample('year')` gives the year-end value. A result across modules with a per-person cap has nowhere to live (R17, HP23). TL23 and TL51 block YLL.
  - A coded 65-line analyzer (`mnch_burden/burden_v4.py`) reading a weight per state from declared diseases and `daly_weights()` from coded ones agrees with TLO within 2–5%.
  - It also shows a TLO weakness: TLO samples disability once a month, so it misses 44% of Mockitis's YLD (18.4 against 29.8 integrated each step).
  - Proposed: `disability=ss.disability({S1: ss.par(0.288), ...}, cause=...)` on a disease, `cause=` on death transitions, `daly_weights(uids)` on coded modules, and `ss.Burden(age_limit=90, life_table=None, cap=1, by=['age', 'sex'])`, giving `sim.results.burden.yld`, `.yll` and `.dalys` by cause. About 150 lines.
- **TL51 (crash): an `ss.outcome` weighted by `ss.by(age=...)` or a function crashes as agents** with `'Agents' object has no attribute '_uids'`. As an ODE it runs (YLL 8,881 for 193 deaths). This is the natural declared YLL, and 09-results allows such weights. Repro: `mnch_burden/repro_outcome_weight_by_age.py`. About 15 lines. Same root cause as HPN2 and MA30; a new instance.

### 7. Not every condition is an infection

v4 decides which states are "infected" from the shape of the model. Most of TLO's conditions are not infections, and v4's guesses mislead or silently change answers.

- **TL52 (silent): in a staged NCD, only the first stage is "infected".**
  - A test with no `detects=` finds stage 1 only: 649 of 1,637 people with cancer at a full-coverage screen (649 / 488 / 500 by stage). `describe()` prints "infected: s1".
  - With `acquisition=False` on the onset, there are 0 positives and no message.
  - Repro: `ncds/repro_ncd_infected.py`.
  - Fix: the infected states are every state reachable from an acquisition without passing through the first state. A test or treatment on a disease with no infected states raises. S.
  - Related to MA2 and ST5. The same rule as MA's group 1 should cover both.
- **TL53 (friction): pregnancy, injury and a module of properties are described as infections.**
  - `describe()` calls conception "an acquisition [guessed]" and the first gestational week "infected … cured by ss.treat". It calls a crash an acquisition, and the fake lifestyle state "infected".
  - It also marks infected states "[counted in prevalence]", though a declared disease has no `prevalence` result.
  - Proposed: `kind='condition'` (or `ss.Condition`) with no infected states unless named. Otherwise word the guesses neutrally ("entered by", "condition states"). S.

### 8. The framework

- **TL54 (silent): coded modules initialize and step in a fixed order by kind, with `custom=` first.**
  - A coded Demography in `demographics=` that sets districts in `init_post()` runs after a coded Lifestyle in `custom=` that reads them. The reader sees 100% unset, and nothing raises.
  - TLO sorts modules by `INIT_DEPENDENCIES` and refuses a missing one.
  - Repro: `framework/repros/repro_init_order.py`.
  - Fix: run `custom=` last, and add an `after = [...]` attribute. S to M.
- **TL55 (accuracy, by design): a fixed step loses what TLO's event queue keeps.**
  - A 17-day fixed delay is recorded at day 61 at a monthly step, day 21 weekly, and day 18 daily. First-year births are 125 against TLO's 206, because results are taken at the end of each step.
  - A daily step costs 17 times as much (11.6 s against 0.68 s), still 3 times faster than TLO.
  - Also lost: ordering within a day, and per-person event chains.
  - Proposed: scheduled exits keep their fraction of a step, and `ss.schedule(uids, at=dates, fn=)` for coded modules. M to L.
- **TL56 (friction, partly silent): an `ss.par` default in a coded module's `define_pars()` can be neither used nor fitted.**
  - Used in code it is an expression (`Cannot use the expression "(... < 0.008 * 1)" where Python needs a single True or False`).
  - `calibrate()` raises `NothingToFit`, even when an `ss.par` is passed at construction. Only a plain default plus an `ss.par` at construction fits (0.0071 for 0.008).
  - Data: `framework/out/probe_pars.txt`.
  - Fix: register defaults as construction does. About 10 lines.
  - Related to ST9.
- **TL57 (friction): `load_pars()` refuses TLO's parameter files** (it wants 2 columns; TLO's have 8, with `prior_min` and `prior_max`). A 12-line reader making `ss.par(value, bounds=[prior_min, prior_max])` works, except for list values (N20). Proposed: `load_pars(path, bounds=('prior_min', 'prior_max'))`. S.

### 9. Small things

- **TL58 (silent):** `d.pars.init = dict(B=100)` after construction is ignored: `n_B` is 0 and `describe()` says "nothing given" (`ncds/repro_pars_init_ignored.py`). Keep `init` in `pars`, or raise on an unknown key. S.
- **TL59 (silent, in display):** `ss.prob(0.15, ss.months(3))` prints as "0.15 probability per month". The conversion is right; the label is 3 times off, on every TLO parameter (`ncds/repro_describe_prob_period.py`). Print the period. S. Related to FP38.
- **TL60:** `init=dict(I=1.0)` starts one person; 0.999 starts 19,980 of 20,000 (`framework/repros/repro_init_one_is_a_count.py`). This is the spec's rule, but spec 03 writes `init=` values as shares. A note when a float is exactly 1. S.
- **TL61:** `describe()` omits transfers and dimensions ("1 states, 1 transitions"). It prints `ss.constant(v=1)` for a one-week dwell, with no unit, and `p = the rest, f` for functions. It says "not tried on groups" for modifiers that ran on groups, and "[guessed: equal]" for a sex split read from `people.female`. Small each.
- **TL62:** `ss.death(rate=ss.timeseries(...))` with no declared module raises `"->" in demographics= needs a declared disease`, though no arrow was written. S.
- **TL63:** diseases with different age edges can't share a sim (`InconsistentDimension`, as specified), and the error names the module "sdr". TLO's modules each have their own age groups. Refine to the union of edges, with a note. M.
- **TL64:** `check_dt` treats a fixed delay as a rate. It says "0.18 of people leave in a step" for `ss.constant(ss.days(5))`, though nobody leaves early. S.
- **TL65:** `ss.timeseries(times, values)` raises `'list' object has no attribute 'iloc'`, because the second list is taken as `mode=` (`healthsystem/repro_timeseries_args.py`). 3 lines.
- **TL66:** `tr.p = ss.by(...)` after building raises `'by' object has no attribute 'leaves'` deep in `graph.all_pars()`, while `tr.before = ...` works (`infections/probes/repro_set_transition_attr.py`). A setter, or a message. S.
- **TL67:** `ss.transfer(antidepr='True -> False')` on a boolean dimension is accepted, then fails at the first step with `"False" is not a level`. Say at build to write `antidepr=False`. S.
- **TL68:** two unconnected states are accepted silently. Warn at build. S.
- **TL69:** there is no declared way to start mid-pregnancy (or mid-dwell generally). TLO fakes 9 months of births. Draw the time already spent uniformly, with TL2's `since=`. S after TL2.
- **TL71 (docs drift):** 07-effects.md's ANC example still uses `eligible=` on `ss.screen`, which now raises. S.
- **TL72 (not verified):** random draws inside function slots get no common random numbers; no way was found to give a declared module's function an `ss.Dist`. Needs checking.

**Known findings TLO hits again, one line each:**
- **FP26:** no logistic linear model (TL14).
- **FP30:** a dwell distribution by level crashes (diarrhoea's duration by course), while `describe()` prints it.
- **ST18:** `dur=fn` returning each agent's time is an exponential's mean; 3,656 of 10,000 leave by day 5 of 10.
- **FP19:** numeric levels are read as bin edges (`wealth=[1,2,3,4,5]` gives 4,000/2,000/2,000/2,000/0). Fixed in the working tree.
- **FP22:** a function rate returning a bare number is read per step, giving 2,888 conceptions instead of 276.
- **HP16:** an immediate split as one transition raises `Cannot evaluate None`.
- **HP19:** per-person continuous quantities (worm burdens, the injury list, MUAC) stay coded.
- **MA12:** a function of the sim is indexed by UID as agents (TL8 is the any-death case).
- **MA29:** `describe()` crashes on an ODE with `People(age_data=)` and an age dimension.
- **FP38, L27:** "[guessed]" on a given `People`.
- **ST15:** `on=a | b` isn't built.
- **FP25, HP18, ST13:** no time since an event, and no event of another module as a trigger.
- **FP27:** no tabular `p=` for TLO's contraception switching matrix.
- **ST26, HP29:** conception driven from another declared module refuses the ODE.
- **R17, HP23:** no result across modules.
- **ST9:** a coded module's own parameter can't be fitted (TL56).
- **N20:** list-valued parameters can't have bounds.
- **D23:** a stratified sim with no declared module (TL10).
- **Decision 16 (not a bug):** the declared cancer as an ODE at a 3-month step gives twice the agents' deaths, since the ODE passes through several stages in a step. It behaves like TLO with its cascade bug, and agents like TLO without it. The rate × dt notes warn.

Fixed at `e4dbda48` and checked: FP18 (newborns from `offspring=` age and have a parent), ST1 (newborns join the first state), and the first half of MA18.

## Speed

The same people and period on both sides, single-threaded. TLO runs used the module set from each module's tests (Demography, Lifestyle, the symptom manager and the health system, usually disabled), so they include framework work the v4 ports don't do. That framework is most of TLO's time.

| What | TLO | v4 | Ratio |
|---|---|---|---|
| Demography + births, 20,000, 5 years | 21.9 s | 0.39 s (monthly) | 56 |
| Demography + births + Lifestyle, 20,000, 5 years | 32.5 s | 0.60 s (coded Lifestyle); 11.6 s daily | 54 |
| Health-system slice, 5,000, 1 year, mode 1 / mode 2 | 22 s / 34 s | 0.81 s / 0.91 s | 27–37 |
| HIV, 10,000, 2010–15 | 138–173 s | 6 s agents; 0.7 s ODE | 25–200 |
| HIV + TB, 10,000, 2010–13 | 97 s | 10.6 s | 9 |
| Measles, 10,000, 5 years | 84 s | 0.56 s | 150 |
| Schistosomiasis, 10,000, 5 years | 298 s | 0.7 s (coded) | 430 |
| Oesophageal cancer, 10,000 / 20,000, 5 years | 112 s / 358 s | 0.24 s / 0.3–0.7 s | 470–1,000 |
| CMD + depression, 10,000, 5 years | 160 s | 1.0 s | 160 |
| Epilepsy + COPD, 10,000, 5 years | 49–56 s | 0.30 s | 170 |
| Diarrhoea / RTI, 20,000, 2 years | 65 s / 71–109 s | 3.5 s / 2.6 s | 18–40 |
| Pregnancy to birth, 5,000, 2 years | 104–162 s | 0.8–0.9 s | 150 (TLO also runs complications and ANC) |
| Burden with two diseases, 5,000, 2 years | 35 s | 0.40 s | 90 |
| Per-district capacity as 192 deliveries (TL40) | | 31.5 s, against 0.9 s coded | |

TLO's time goes mostly to:
- births at 8 ms each (7.5 s of 23.4 s at 20,000 people for 2 years);
- the Lifestyle logging event (6.1 s);
- the daily age update (5.3 s);
- daily whole-population scans in TB;
- monthly pandas groupbys in schisto;
- a quadratic scan of the dataframe at every ART visit in HIV.

The disease modules themselves are often too cheap to measure: TLO with and without diarrhoea took 64.6 s and 64.9 s. So the ratios grow with the number of modules, and the full model would likely see the larger ones. v4's agent engine is slower than its ODE where it matters: declared HIV is 6 s as agents against 0.7 s as an ODE. The large-group cost (TL7) is the one v4 cost that grows with TLO's design.

## Keep

- **Staged diseases as dimensions with `adjust=` and `move=`.** Treatment that slows only its own stage is two keywords, checked to 0.177 against 0.181.
- **Several diseases joined by modifiers, crossed automatically as an ODE.** CMD's 5 conditions and 2 events are 7 small diseases and 24 modifiers. That is far easier to read than 288 joint states, and `what_modifies()` lists every link by name.
- **Branches at onset for properties drawn at onset.** Depression's resolution class, diarrhoea's type and dehydration, and a case fatality decided at entry and still changed by later treatment are all exact for TLO's discrete draws.
- **Dimensions of the episode state with splits on entry,** for pathogens and courses, with results by pathogen and age for free.
- **Fixed dwells that branch,** which are exactly TLO's gestational checkpoints, and `offspring=` into a newborn state.
- **Deliveries for a single cascade:**
  - `on=` an event;
  - `coverage=` a time series by district;
  - `ss.screen` with `sensitivity=dict(...)` by state and `delay=`;
  - `on_positive=`;
  - `ss.treat(deliver=...)` as the seam to a coded health system (which answers ST11 in practice).
- **Coded modules beside declared ones:** coded schisto, the injury list and the health system plug in, `ss.modify(where=fn)` reads them, and the ODE refuses them clearly.
- **Common random numbers,** which TLO lacks entirely.
- **Results by group, and delivery results** (`delivered`, `eligible`, `diagnosed`), which replace TLO's logging and its hand-incremented counters.
- **The error messages,** praised by every agent: `RouteMismatch`, `TimeNotKept`, the `at_age=` bin edge, the `offspring=` ODE refusal, the `capacity=` rate message.

## TLO's own bugs

The agents found these while comparing; each part's `REPORT.md` has the evidence. They are listed because several show what a declared vocabulary prevents, and because the TLO team may want them. Confirmed by running:
- **Cancer progression cascades within a poll.** In oesophageal, other adult, bladder and prostate cancer, each stage's model reads the status written by the one before. So 167 of 10,000 people skip stages, and cancer deaths are 38 against 14.
- **A typo in oesophageal cancer** (`'high_grade_dysplaisa'`): people with high-grade dysplasia never get dysphagia, so they are never diagnosed at that stage.
- **CMD's removal of a condition skips everyone who has its symptom** (pandas index alignment): 35 CLBP removals against 352.
- **ChronicSyndrome never kills:** `df.loc[...].values[:] =` writes to a copy.
- **Schisto:** worm burdens of 1 or 2 never die (rounding). HIV and bladder cancer test for "High-infection", a category that doesn't exist, so `rr_schisto` never applies.
- **Stunting incidence by wealth points the wrong way:** the richest children have 5 times the incidence. Annual probabilities clipped at 1 make stunting certain at the first monthly poll.
- **Lifestyle:**
  - an operator-precedence slip means `rr_tob_agege40` never applies;
  - hand-washing's initial odds are `1/(1-p)`;
  - education is drawn twice with fresh random numbers;
  - the move to urban never triggers its reassessment.
- **At small populations, mode 2 ignores `capabilities_coefficient`:** 0.000344 and 0.0001 give bit-identical results.
- **Without care, some injured people never recover** (recovery dated after the end of the sim).
- **The yearly maternal counter isn't reset when a denominator is 0,** and conceptions are undercounted by 19% (`date_of_last_pregnancy`).

From reading the code:
- CMD's case fatality is evaluated after the event flag is set, so the heart-attack case fatality is 1.62 for men over 65.
- Post-treatment checks use months as years in three cancers.
- Treated children under 5 with TB die with probability 0.003, not 0.05 (overlapping conditions multiplied).
- HealthBurden's monthly snapshot misses short episodes.
- The cancer modules share 48–62% of their lines with no shared code, which is how several of these spread.

## Suggested order

1. **The silent wrong answers in v4's own modules and engine,** mostly a few lines each:
   - TL24–TL27, TL29, TL30 (`ss.Deaths`, `ss.Pregnancy`, `ss.Births`);
   - TL6 (int16 groups);
   - TL12, TL13 (infinite hazards, probabilities over 1);
   - TL16, TL17 (branches and products on `dur=` transitions);
   - TL35 (closures over modules);
   - TL38, TL52 (first state and infected states);
   - TL4, TL54, TL58.

   Each should get a conformance test from its repro.
2. **TL23 (one death, one cause) and TL28 with the rest of MA18 (newborns from the parent's levels),** after TL27. Then `ss.Deaths(net_of=)`.
3. **People as data:** TL1 (`init=` functions, then `ss.People.from_df`), TL2 (`since=`), TL9 (properties in rates), TL11 (`ss.by` arithmetic), TL5 (several conditional splits), TL8. With these, TLO's linear models and Lifestyle are declarable as agents.
4. **Re-dosing and repeated treatment** (TL36 with MA15), lists of follow-ups (TL41), and `on=` transfers (TL43).
5. **Design pass on the burden** (TL50, TL51): `ss.disability`, `ss.Burden`, and a time-integral result kind.
6. **Design pass on a shared resource** (TL39–TL42, TL44–TL46): `ss.Resource`, `ss.Visit`, `queue=`, `priority=`, `record=`.
7. **The rest:**
   - TL3 (as a test; fixed in the working tree);
   - TL7, TL10, TL14, TL15, TL18–TL22, TL31–TL34, TL37, TL40, TL47–TL49, TL53, TL55–TL57, TL59–TL72.

With steps 1 to 6, everything in TLO except its per-person continuous quantities can be declared, runs as agents, and much of it under every solver.

## Index of new findings

| ID | Finding | Kind | Section | Agent's ID | Cross-reference |
|---|---|---|---|---|---|
| TL1 | `init=` can't take a function or array; `ss.People` takes no per-person values | Dead end, crash | 1 | IN6, NC7, NC14 | Related to MA24 |
| TL2 | No time already spent in a state; writing the engine's arrays is rescaled | Vocabulary, silent trap | 1 | IN7 | Related to CR4/L14 |
| TL3 | Stratifying by an existing property resets it | Silent | 1 | NC2 | Fixed in the working tree |
| TL4 | A joint split overrides a dimension's own split | Silent | 1 | FW5 | New |
| TL5 | Only one split may cover several dimensions | Dead end | 1 | FW10, CH4 | New |
| TL6 | More than 32,767 groups crash or wrap as agents | Crash, silent | 1 | FW9, CH10 | New |
| TL7 | Agents allocate the full grid of groups | Cost | 1 | FW11 | New |
| TL8 | A function rate per living agent crashes after a death | Crash | 1 | NC6 | Related to MA12 |
| TL9 | A people property can't be used in a rate | Vocabulary | 1 | NC10 | New |
| TL10 | A module of properties needs a fake disease; `n_alive` never by group | Vocabulary | 1 | FW12 | Related to D23 |
| TL11 | `ss.by` arithmetic fails outside a transition | Friction, spec drift | 2 | FW13, CH6, NC11 | Related to FP31 |
| TL12 | `ss.probperyear(1) * 0.8` moves nobody; prob × rr scales the hazard | Silent | 2 | CH2 | New |
| TL13 | A branch probability over 1 used as 1 with no note | Silent | 2 | CH3 | New |
| TL14 | No logistic (odds) form | Vocabulary | 2 | HS (care-seeking), CH (ALRI) | Duplicate of FP26 |
| TL15 | No parameter defined as a population mean | Vocabulary | 2 | MB11 | New |
| TL16 | `adjust=`/`ss.modify` on a branching transition with a dwell change the dwell | Silent, spec drift | 3 | MB1, CH1 | Related to FP13 |
| TL17 | A product's efficacy on a `dur=` transition does nothing as agents | Silent | 3 | IN1 | Related to C17, MA8 |
| TL18 | A case-fatality branch can't leave the survivors as the remainder | Friction | 3 | NC12 | Related to L14, HP16 |
| TL19 | A split can't depend on another disease's state | Dead end | 3 | CH18 | New |
| TL20 | A drawn dwell follows the current age group; no at-entry rule | Vocabulary | 3 | IN5 | Related to KI6, C17, HP18 |
| TL21 | An event in one disease can't move people in another | Vocabulary | 3 | IN (AIDS-TB) | Related to FP25, ST13 |
| TL22 | No code at entry, per-episode attributes, or per-person counts | Dead end | 3 | CH13, CH19 | Related to HP19, FP30 |
| TL23 | Deaths counted twice; no cause per person; no net-of | Silent | 4 | FW4, MB5 | New |
| TL24 | `ss.Deaths` reads period rates as the nearest year | Silent | 4 | FW1 | New |
| TL25 | `ss.Pregnancy`: no fertility at 46–49 with 5-year data | Silent | 4 | FW2 | New |
| TL26 | The unborn counted alive and seeded | Silent | 4 | FW3, IN3 | Related to ST1 |
| TL27 | `ss.Births` draws parents from everyone | Silent | 4 | IN2 | Related to FP18 |
| TL28 | A split by age × another dimension resets ages and mis-assigns newborns | Silent | 4 | CH9 | Related to FP20, MA18 |
| TL29 | `ss.Pregnancy(p_loss=)` is per step | Silent | 4 | MB2 | Related to L1 |
| TL30 | `ss.Pregnancy`'s `rr_ptb` can't change the preterm share | Silent | 4 | MB3 | New |
| TL31 | A maternal network's `beta` transmits child to mother | Silent trap | 4 | IN4 | New |
| TL32 | Susceptibility by age applies to every route | Vocabulary | 4 | IN8 | New |
| TL33 | A per-birth probability must be converted by hand | Friction | 4 | IN10 | New |
| TL34 | No birth hook; `offspring=` per transition, not branch | Friction | 4 | FW17, MB (§3) | Related to HP16 |
| TL35 | A closure over a coded module reads the original copy | Silent | 5 | HS1 | Related to MA14 |
| TL36 | A held product can't be given again: one dose for three; treatment once per life | Silent | 5 | HS2, CH7 | Related to MA15 |
| TL37 | Compartmental test on one disease's event detecting another's state | Silent, crash | 5 | NC1 | Related to DF1 |
| TL38 | A state entered only by a delivery becomes the first state | Silent | 5 | HS3 | Related to CR3/L4 |
| TL39 | No capacity shared between deliveries | Vocabulary | 5 | HS5 | New |
| TL40 | A capped delivery has no queue | Vocabulary | 5 | HS6, HS9 | Open question "when capacity runs out" |
| TL41 | No appointment without a product; a list of follow-ups crashes | Vocabulary, crash | 5 | HS7, NC8 | Care-cascade decision 3 |
| TL42 | No property set at an event from the person's state | Vocabulary | 5 | NC13 | New |
| TL43 | A delivery can't be triggered by a transfer | Dead end | 5 | NC9 | New |
| TL44 | A product needs a declared disease | Vocabulary | 5 | HS11 | New |
| TL45 | "Test not done" isn't a result; no fallback tests | Vocabulary | 5 | HS12 | New |
| TL46 | Presentation per disease; no level on a visit | Vocabulary | 5 | HS13 | New |
| TL47 | `capacity=ss.by(...)` crashes | Crash | 5 | HS8 | New |
| TL48 | Event-triggered care acts a step late | Friction | 5 | NC16 | New |
| TL49 | The time-index warning contradicts the deliveries | Spec drift | 5 | HS4 | Related to FP1, MA24 |
| TL50 | No burden of disease | Vocabulary | 6 | MB6 | Related to R17, HP23 |
| TL51 | `ss.outcome` weighted by age crashes as agents | Crash | 6 | MB4 | Related to HPN2, MA30 |
| TL52 | A staged NCD counts only its first stage as infected | Silent | 7 | NC3 | Related to MA2, ST5 |
| TL53 | Pregnancy, injury and properties described as infections | Friction | 7 | MB8, CH20, FW12, HS14 | Related to MA1 |
| TL54 | Coded modules initialize by kind, `custom=` first | Silent | 8 | FW7 | New |
| TL55 | A fixed step loses event timing | Accuracy | 8 | FW18 | New |
| TL56 | An `ss.par` default in `define_pars()` can't be used or fitted | Friction | 8 | FW15 | Related to ST9 |
| TL57 | `load_pars()` refuses TLO's files | Friction | 8 | FW16 | Related to N20 |
| TL58 | `pars.init` set after construction is ignored | Silent | 9 | NC4 | New |
| TL59 | A probability per 3 months labeled per month | Silent (display) | 9 | NC5 | Related to FP38 |
| TL60 | `init=dict(I=1.0)` is one person | Friction | 9 | FW6 | New |
| TL61 | `describe()` omits transfers, units; misreports modifiers and splits | Friction | 9 | FW14, MB9, IN12 | Related to FP38 |
| TL62 | A misleading error for `ss.death` with no declared module | Friction | 9 | FW19 | New |
| TL63 | Diseases with different age edges can't share a sim | Dead end | 9 | CH11 | New |
| TL64 | `check_dt` treats a fixed delay as a rate | Friction | 9 | CH12 | New |
| TL65 | `ss.timeseries(times, values)` crashes | Friction | 9 | HS10 | New |
| TL66 | Setting `tr.p` after building crashes | Crash | 9 | IN11 | New |
| TL67 | A boolean transfer written with an arrow fails late | Friction | 9 | NC15 | New |
| TL68 | Unconnected states accepted silently | Friction | 9 | MB10 | New |
| TL69 | No start mid-pregnancy or mid-dwell | Friction | 9 | MB12 | Related to L14 |
| TL70 | Pooled prevalence within a subgroup needs a stale hand-made matrix | Vocabulary | 3 | IN9 | Related to MA11 |
| TL71 | 07-effects.md uses the removed `eligible=` | Docs drift | 9 | NC17 | New |
| TL72 | No common random numbers inside function slots (not verified) | Friction | 9 | CH14 | New |
