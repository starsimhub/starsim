
# HIV (STIsim/HIVsim) on Starsim v4: review

`port.py`: a class-form `ss.Disease` with declared stages and ART states, CD4 as custom per-agent arrays, and v4 deliveries/modifiers for testing, ART, VMMC and PrEP. Zimbabwe data, 5k agents, 1990–2020, monthly, MF + maternal + breastfeeding networks, Pregnancy + Deaths. Runs in about 2 s. It hits the ART stock target (1.18M vs 1.18M in 2020), VMMC follows the data's flows (1.79M), and there is vertical transmission (after a workaround). The `alt_*` functions at the bottom test the design questions (product vs state, capacity vs priority, a dimension vs a flag, annual coverage).

Prevalence is not calibrated: it rises to 18.6% with no 1990s peak. I did not chase this.

## Features

### 1. Stages acute → latent → falling, with rel_trans by stage
- v3: `hiv.py:28-30` (durations), `:512-517` (stage flips in `step_state`), `:662-663` (rel_trans).
- v4: `port.py:51-67`: `latency = acute >> latent @ ss.lognorm_ex(...)`, `ss.State(rel_trans=5.3)`.
- **Declarative. Clean.** Better than v3, which has to manage `ti_latent`, `ti_falling` and `ti_zero` and clear 30 arrays by hand in `step_die` (`:567-618`). v4 handles all of that for free, and lognormal dwells become Erlang stages in the ODE.
- v3 applies the "AIDS" rel_trans to CD4 < 200, not to the falling stage. That needs a per-agent modifier amount (`aids_rel_trans`, `port.py:200`). Awkward, but it works.

### 2. CD4 as a continuous per-agent trajectory
- v3: `hiv.py:308-384` (acute/falling/post-ART decline, logistic recovery), `:500-529`.
- v4: `port.py:114-166`: `define_states(ss.FloatArr('cd4'), ...)` plus a `step_state` override around `super().step_state()`.
- **Only via `step_state()`. Ugly (ordering).** Mixing declared and custom code works: `self.engine.events[...]` and the free `ti_<state>` / `ti_exit_<state>` arrays make closed-form trajectories easy. What I had to learn the hard way:
  - **There is no on-entry hook.** "Set CD4 when infected" has to read `engine.events['infection']` and must run *before* `super().step_state()`. That is because super() evaluates `cd4_death`'s hazard function, which reads `cd4`. If you write it after super(), newly infected agents have `cd4=NaN` for one step. `np.digitize(NaN)` then silently gives the lowest mortality bin; with maternal networks, a negative CD4 crashed with `IndexError: index 6 is out of bounds`. The ART entry hook, by contrast, has to run *after* super(), because forced moves only appear in `events` once super() has run. So two hooks on the same object must sit on opposite sides of super().
  - **`ti_exit_<state>` is NaN on the step of entry.** It is drawn lazily at the start of the next step (`solvers.py:3489 schedule`), so any trajectory that uses the exit time must guard against NaN (`port.py:151`).
  - `ti_exit` is fractional (`entered + dur/dt - 0.5`). Incremental per-step updates overshoot, so use closed-form interpolation instead.
- **It does force the whole disease to agents, but honestly.** `sim.solvers()` says `function: cd4_hazard  no (takes uids)`, and `auto` picks abm. `solver='ode'` does **not** give E804. It crashes in user code (`AttributeError: 'CountState' object has no attribute 'uids'`), because a subclass with custom `step_state` is not tried on groups.
- Suggestion: `on_enter=dict(acute=fn, on_art=fn)` / `on_exit=` callbacks of `(sim, uids)`, run inside `finish_moves`. Also schedule exits at entry, or document the lag.

### 3. Mortality depending on CD4, off and on ART
- v3: `hiv.py:394-455` (binned off-ART rate × sex, × ART/sex/age multipliers), `:541-564`.
- v4: `cd4_death = (acute|latent|falling|on_art|post_art) >> ss.null @ cd4_hazard`, with `cd4_hazard(sim, uids)` at `port.py:31`.
- **Partly declarative. Clean.** A rate function of `(sim, uids)` competing with the scheduled `aids_death` dwell is exactly the competing-hazards case, and v4 resolves it correctly. Sex and age multipliers go in the function (fine).
- Per-cause deaths are automatic flows (`aids_death`, `cd4_death`, `post_art_death`).
- **Deaths by CD4 are not automatic.** CD4 is not a dimension, so I needed an analyzer that reads `engine.events` and indexes CD4 at death (`port.py:181`). Awkward.

### 4. ART: state vs product (special focus 3)
- v3: `hiv.py:775-876` (`start_art`/`stop_art` nullify the natural history; post-ART decline sets `ti_zero`), `:665-680` (6-month efficacy ramp, suppressive vs not).
- **v4 as states** (main port): `start_art = (acute|latent|falling|post_art) >> on_art @ 0`, `dropout = on_art >> post_art @ ss.lognorm_ex(3y, 1.5y)`, `post_art_death = Transition(post_art, null, dur=post_art_dur, stages=100)`. The intervention is `ss.Treatment(efficacy=1.0, transition='start_art')`, which forces the move.
  - **Mostly declarative. Awkward.** Upsides: dropout is non-exponential and declared, re-initiation from `post_art` comes for free, `n_on_art` and `n_post_art` are automatic results, and leaving the stage states pauses progression with no extra code. Downsides:
    - `@ 0` is a fake rate, needed just to declare an edge that only an intervention drives.
    - "A treatment that cures" is the wrong word for starting ART.
    - The ART mortality and transmission effects have to be written by hand (in `cd4_hazard`, and `ss.modify(hiv.rel_trans, multiply=art_rel_trans, where=hiv.on_art)`).
    - A per-agent `dur=` function plus `stages=100` approximates v3's `normal(mean, 0.1·mean)`. It works, but it is a trick.
- **v4 as a product** (`alt_art_as_product`): `ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.75), waning=ss.years(3))`.
  - **Clean for the effects:** `mortality` lands on every outflow transition, and `what_modifies` lists it.
  - **Wrong for the program:**
    - Dropout can only be exponential (`waning=` duration), and there is no post-ART state, so the v3 CD4 decline and death clock after stopping cannot be expressed.
    - `mortality` hits *every* outflow, including non-HIV `other_death`.
    - Halting progression needs `ss.modify(hiv.progression, multiply=0, where=ss.Cond('dim','has_artp'))`. There is no public handle for "holds the product", so you have to spell out the dim name. This pauses the clock rather than resetting it as v3 does, and emits `RuntimeWarning: divide by zero` at `solvers.py:3570`.
    - A time ramp in efficacy (`waning=callable`) cannot be combined with a loss duration.
    - Per-agent suppressive vs non-suppressive efficacy has no slot.
- Verdict: for HIV, ART is a **state** (it changes the natural history). Products fit PrEP and VMMC.
- Suggestion: (a) `ss.Treatment(..., moves='start_art')` instead of "cures" + `transition=`, and allow undriven edges (`>> on_art` with no `@`) for intervention-only transitions. (b) Let `mortality=` take a list of transitions. (c) Give products a lazy condition, `art.held`. (d) Allow `waning=` to be a dist (non-exponential loss) and `ramp=` separately.

### 5. Dropout and re-initiation (special focus 4)
- v3: `hiv_interventions.py:410-418` (stop at `ti_stop_art`), `:523-527` (re-add from the diagnosed).
- v4: `dropout` is a declared transition, and re-initiation is the `post_art` source of `start_art`.
- **Declarative. Clean.** In the product version, re-initiation works too (2.26M starts for an 0.87M stock), because "a delivery never gives a product to someone who already holds it" stops applying once they lose it.
- **Caveat:** routine delivery (no `deliver=`/rounds/`on=`) remembers `offered_<delivery>` when it has coverage or capacity, so it would *never* re-offer. You must use `deliver=` or rounds.
- In this model, post-ART deaths are 0: under a binding stock target, dropouts are restarted the next step. v3 behaves the same way; it is not a v4 issue.

### 6. ART coverage as numbers on treatment, with priority (special focus 1)
- v3: `hiv_interventions.py:392-572` (stock target from `n_art`, priority `cd4/care_seeking`, removal of the highest CD4 when over target, stratified by age and sex), plus `utils.py:397-458` (`n / pop_scale`).
- v4: `art_deliver(sim)` at `port.py:247`: target = `np.interp(year, n_art)/sim.current_scale − on_art.count()`, top-n by priority.
- **Only via `deliver=`. Awkward, and partly impossible.**
  - **`capacity=` is a flow** ("number per unit time", `interventions.py:983-1002`). Fed the stock, it starts 1.47M by 2012 and puts 960k on ART against a target of 566k (`alt_capacity_as_flow`). Nothing in the delivery API knows "number currently holding".
  - **`capacity=` caps at random and throws away the priority order from `deliver=`** (`cap()` uses `argsort(rng)`). In `alt_priority_lost`, the same sorted list capped by `capacity=` gives a mean CD4 at start of 527, against 388 when capped inside the function. So "prioritize by CD4, up to N" must be capped by hand.
  - **Being over target cannot be expressed.** v3 removes people. v4 has nothing for this, so I called `hiv.engine.force(drop, graph.transitions['dropout'], graph.states['post_art'])`, which is private API.
  - `sim.current_scale` for people→agents is undocumented. `capacity=` does this conversion internally, but `deliver=` does not.
  - `n_on_art` is free, and the target was hit to within about 1% every year.
  - Stratified (age × sex) targets would be a loop inside `deliver=`. Not attempted.
- Suggestion: `ss.treat(art, target=ss.timeseries('art_coverage.csv'), target_of='n' | 'p', priority=lambda sim, uids: cd4/care, remove=True)`. That would cover a target on the number currently holding the product or in a state, which a modeler states as data. `priority=` should also be honored by `capacity=`. This is the central gap for HIV, and also for PrEP and VMMC targets.

### 7. Testing and diagnosis (HIVTest)
- v3: `hiv_interventions.py:41-154`; Zimbabwe `sim.py:36-70` (three testing streams with linear ramps of annual probabilities).
- v4: `ss.screen(ss.Test(), interval=ss.months(1), coverage=annual_to_step(...), eligible=..., on_positive=diagnose)` at `port.py:227`.
- **Mostly declarative. Awkward.**
  - Repeated testing needs `interval=dt`, because routine delivery is once per person.
  - **Coverage is per round**, so an annual test probability must be converted by hand. `coverage=ss.probperyear(0.5)` raises `TypeError: Cannot apply float() to probperyear(0.5)...`. It should convert, as rates do everywhere else.
  - `diagnosed` as a `BoolState` written by a `(sim, uids)` function works.
  - The declarative alternative also works: `hiv.stratify(dx=[False, True], states=[...])` plus `on_positive=ss.transfer(dx=True, coverage=1.0)`. But `ss.transfer(dx=True)` without `coverage=` fails (`ValueError: ss.transfer() takes exactly one of rate= or dur=`), because the same name is both a transition and a delivery.
  - `screen.diagnosed` gives new diagnoses per program for free.
  - v3's `dur_dx2tx` scheduling (`ti_art`) was dropped. It would be a `delay=` on the test, or a `dx >> ready` dwell.
- Suggestion: accept `ss.probperyear`/`ss.peryear` in `coverage=` and convert per round. Split `ss.transfer` (transition) from the delivery form, or make `on_positive=ss.transfer(dx=True)` mean coverage 1.

### 8. VMMC
- v3: `hiv_interventions.py:575-732`. It treats `n_vmmc` as a **stock** target ("circumcision prevalence", `:686-690`).
- v4: `ss.vaccinate(ss.Vaccine(efficacy=0.6), eligible=men 15–49, capacity=ss.timeseries(n_vmmc), start=2008)` at `port.py:268`.
- **Declarative. Clean.** `ss.timeseries` from a dict of years works, and the totals match the data (1.79M vs 1.87M; the small loss is sub-agent rounding at low counts).
- **v3 data bug:** `vmmc_coverage.csv` is clearly annual procedures (a flow: 355k in 2019, 82k in 2020 from COVID), but v3 uses it as a stock target, so it under-circumcises after year one. capacity= is the right semantics here.
- `n_circumcised` (holders) is not a result. Only `delivered`/`eligible`/`coverage` are.
- v3's traditional circumcision at age 15 (one assessment per person) would be routine delivery with coverage (once per person); not done.

### 9. PrEP
- v3: `hiv_interventions.py:735-908` (stock target among FSW, fixed 3-month courses, renewal).
- v4: `ss.vaccinate(ss.Vaccine(efficacy=0.8, waning=ss.months(3)), eligible=AGYW & ~infected, coverage=per-step, interval=dt)`.
- **Declarative. Clean, but approximate.** Courses end at an exponential time, not after a fixed 3 months, and the target is a flow rather than a stock (same gap as #6). There is no FSW without StructuredSexual. The "one PrEP product at a time" rule across products would need `eligible=... & ~ss.Cond('dim','has_other')`.

### 10. Sex- and age-specific parameters
- v3: `rel_sus_age` loop (`hiv.py:637-644`), `rel_death_f`, `rel_art_mortality_unsupp_m/f`.
- v4: `ss.modify(hiv.rel_sus, multiply=1.7, where=ppl.female & (ppl.age>=15) & (ppl.age<25))` (`port.py:291`); sex inside `cd4_hazard`.
- **Declarative. Clean.** Lazy conditions on `ss.People` work and `what_modifies` lists them. Sex-specific beta is `dict(mf=[m2f, f2m])` (clean).

### 11. Maternal and breastfeeding transmission
- v3: `beta_m2c`, `beta_breastfeed` (`hiv.py:41-42`), `set_congenital` (`:768`), and PMTCT in `ART.step` (`hiv_interventions.py:444-466`).
- v4: `beta = dict(mf=[...], maternal=0.3, breastfeeding=0.06)` with `ss.MaternalNet()` and `ss.BreastfeedingNet()`.
- **Declarative once it works. Blocked by a bug.**
  - **BUG: agents created by `ss.Pregnancy` (unborn at conception, and newborns) are put in no state of a declared disease.** Only `ss.birth()` places people. After 3 years, 305 of 3,297 agents were in no HIV state, and only 81 of 391 under-3s were in `S`. Newborns are silently immune forever and there is zero vertical transmission. My workaround puts orphans in `S` each step (`port.py:129`); with it, 51 infected children under 3 by 2020.
  - Congenital CD4 inheritance (`set_congenital`) is not done. PMTCT (maternal ART cuts infant `rel_sus`) would be `ss.modify(hiv.rel_sus, where=<infant of a mother on ART>)`, but a condition on a network partner is not expressible (questions-oct12 #17). It would need `step()`.

### 12. Initial conditions
- v3 back-dates seeded infections uniformly 5 months to 10 years before the start (`hiv.py:297-299`), so seeds are spread across stages and dwell remainders.
- v4 `init=dict(acute=, latent=, falling=)` starts everyone at the *beginning* of their dwell. Nobody progresses for years (a cohort seeded in `latent` had 0 progressions in its first 4 years), and seeded CD4 had to be faked in `init_post`.
- **Impossible to do faithfully.** Suggestion: `init=dict(latent=ss.spread(0.04))`, meaning the elapsed dwell is drawn as for people who arrived at a steady rate (questions-oct12 #8).

### 13. Results
- Automatic: `n_<state>` (including `n_on_art`, `n_post_art`), one flow per transition (deaths by cause, ART starts, dropouts), `prevalence`, and per-delivery `delivered`/`eligible`/`coverage`/`diagnosed`.
- Hand-written: `p_on_art`, mean CD4 on ART, deaths by CD4, holders of a product (circumcised, on PrEP), prevalence 15–49.
- `scale=True` results must be filled in agents and scaled at the end. I double-scaled at first; easy to get wrong.
- Products show up as empty result groups (`hiv_test`, `art`, `vmmc`, `prep`).

## Bugs and mismatches found
1. **Transmission over a sexual network scales with `dt`.** `beta` is converted as a rate to a per-step probability (`-expm1(-beta·dt)`, `solvers.py:3949`) and then compounded over acts per step (`SexualNetwork.net_beta`, `networks.py:556`). Infections over 5 years are 62 / 265 / 644 at weekly / monthly / quarterly steps (`alt_beta_dt`). That is roughly linear in dt, so a per-act beta has no stable meaning. **Severe.**
2. **`ss.Pregnancy` agents are never placed in a declared disease's states** (#11). **Severe.**
3. `capacity=` throws away the priority order that `deliver=` returns (#6).
4. `coverage=ss.probperyear(...)` raises TypeError (#7).
5. `solver='ode'` on a subclass with custom `step_state` gives an AttributeError from user code instead of E804 (#2).
6. A `dict` beta with no networks gives `TypeError: unsupported operand type(s) for *: 'dict' and 'float'` instead of a coded error.
7. A product and its delivery share one namespace: `ss.treat(art, name='art')` fails with "a module with that name already exists", because the product was named `art` from its variable.
8. `ss.modify(transition, multiply=0)` gives a divide-by-zero RuntimeWarning when it schedules new entrants (`solvers.py:3570`).
9. Bare numbers take the smallest typed unit: without `unit='year'`, `@ 0` and the beta dict were "per month" (`describe()` showed `rate = 0 per month`). `describe()` printed it, but the default bites.

## Cross-cutting answers
- **Beyond I→R:** latency and stage progression are clean. "Leave any infected state for treatment" works by giving one transition several sources and forcing it. Competing scheduled dwell vs CD4 hazard is resolved correctly. Treatment that doesn't cure (ART) is best as a state.
- **Count-based delivery:** `capacity=` works as a flow, with carry-over of fractional agents, random choice, and an implicit queue for routine delivery (unreached people stay eligible). There is no priority, no stock target, and no removal.
- **Per-agent continuous state:** it forces `step_state()` for the trajectory only. The stages, deaths, dropout and re-initiation all stay declared. The pain is ordering and lazy exit scheduling, not expressiveness.
- **Mixing declared and custom code in one class** works, with the ordering caveats in #2.
- **Line counts:** v3 HIV is `hiv.py` 939 + `hiv_interventions.py` 910 + coverage utils ~200, plus the `BaseSTI` and `STITest` base classes. v4 is `port.py:1-302`: 238 non-blank, non-comment lines for comparable function (without stratified targets, FSW, VLS correction or newborn tests).

## Ranked: top problems for HIV
1. Per-act `beta` on sexual networks scales with dt (bug #1).
2. Pregnancy-born agents are in no state, so there is no vertical transmission (bug #2).
3. No stock target or priority in delivery. "N on ART from data, lowest CD4 first, remove if over" is the core HIV workflow, and it needs `deliver=` plus private `engine.force`; `capacity=` silently randomizes priority.
4. There are no on-entry/exit hooks, and exits are scheduled lazily, so CD4 code is fragile around `super().step_state()`.
5. Seeds cannot be started mid-dwell.
6. A product cannot be a natural-history state (no post-ART state, exponential-only dropout, `mortality` hits all outflows).
7. Coverage is per round, so annual test rates are converted by hand.

## Surprisingly nice
- The natural history in about 10 declarative lines, with lognormal dwells, competing CD4 hazard, and dropout/re-initiation as transitions. `step_die` bookkeeping is gone.
- `ti_<state>` / `ti_exit_<state>` / `engine.events` for free make closed-form CD4 easy.
- `ss.modify` with a per-agent function amount and lazy `ppl` conditions replaces v3's `update_transmission` reset-and-multiply, and `what_modifies` shows both.
- VMMC as `ss.vaccinate(..., capacity=ss.timeseries(...))` is one line and matches the data. It also exposed a v3 stock/flow bug.
- `sim.solvers()` says exactly why it is agents (`cd4_hazard takes uids`).
