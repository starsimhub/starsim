# Syphilis: STIsim v3 → Starsim v4 review

`port.py` runs (`conda run -n starsim python port.py`, about 2 s per sim: 5,000 agents, 2000–2030, monthly steps). It has three scenarios: no interventions, ANC dual test + BPG + symptomatic treatment + fetal and newborn treatment + an RPR survey, and a stock-out (ANC treatment capped at 6 a year). Output:

```
No interventions      prev 0.132 at 2030; fetal infections 297 → 32 stillbirths, 10 NND, 45 CS, 201 normal; 463 reactivations
ANC + treatment       prev 0.092; ever infected 876, 201 cured and trep+, 39 sero-reverted; RPR+ in Sp 12/201, LL 191/430
                      fetal infections 152 → 6 / 3 / 14 / 123; ANC tested 2570, positive 410; fetal cures 20; RPR survey 74/1354
Stock-out (6/yr)      ANC positive 422, treated 114 (the other 308 are dropped, not queued); 22 CS births
```

The model is not calibrated. The point is what the API forced.

Line counts: v3 has about 1,700 lines for the same functionality (`syphilis.py` 861, `syphilis_interventions.py` 442, and about 400 in `base_interventions.py`: `STIDx`, `ANCTest`, `STITreatment`). The v4 port is 258 lines, about 200 of them code. It has no hand-written results, `step_die`, `set_prognoses`, or `change_states`.

## Feature by feature

### 1. Stages and progression: clean
v3 `syphilis.py:173-215` has 30 bools and `ti_*` arrays, plus `set_prognoses`/`set_secondary_prognoses`/`set_latent_prognoses` (698-799) and `step_state` (431-545). All of it is hand-scheduled `ti_x <= ti` checks.
v4:
```python
primary_v  = ss.Transition(E, Pv, p=chancre_visible, dur=ss.normal(ss.days(50), ss.days(10)))
primary_h  = ss.Transition(E, Ph, dur=ss.normal(ss.days(50), ss.days(10)))
secondary  = ss.Transition([Pv, Ph], S2, dur=ss.normal(ss.weeks(6), ss.weeks(1)), move=dict(rpr=True))
latent     = S2 >> EL @ ss.lognorm_ex(ss.months(3.6), ss.months(1.5))
```
Declarative. Exact dwell distributions as agents. Results like `n_<state>`, a flow per transition, and `cum_*` come for free. v3 needed about 150 lines of `init_results`/`update_results` for this.

### 2. Early → late latent by time since infection: clean (as a dwell time)
v3 `step_state:519-526` relabels latent agents as early or late each step by `ti - ti_latent <= dur_early`. Despite the comment at :83 ("≤2 years post-infection"), it measures from latent onset.
v4: `to_late = EL >> LL @ ss.uniform(ss.months(22), ss.months(24))`. A time in state is just a dwell time, and it competes with reactivation (earliest wins). No `step()` needed. Time since *infection* (as opposed to time in state) has no expression: there is no `ti_E`-based duration.

### 3. Reactivation latent → secondary: clean, with one approximation
v3 `set_latent_prognoses:773-778`: 35% reactivate at lognorm(1y, 1y) after latent onset. The draw survives the early→late relabel.
v4: `reactivate = ss.Transition(EL, S2, p=0.35, dur=...)` competes with `to_late`. Reactivators whose draw is over 22–24 months (about 4% of latents) go to LL instead and lose it, because nothing carries a pending exit across states. Re-entering EL after a relapse redraws everything, which matches v3.

### 4. "35% go on to X, the rest stay forever": ugly
v3: `p_tertiary`, `p_death`, `p_trep_persists`, each a bernoulli plus a scheduled time for the yes branch only.
v4: `ss.Transition(LL, T, p=0.35, dur=...)` alone raises
`E205 BadBranchSplit: the branch probabilities out of state "Sp" sum to 0.2, not 1`.
The workaround is a branch that never fires: `stay_LL = ss.Transition(LL, T, p=0.65, rate=ss.peryear(0))`. I needed three of these (LL, T, Sp). They show up in `describe()` and in results as fake transitions.
**Suggest:** `p<1` with no remainder means "the rest stay". Or `ss.Transition(LL, T, p=0.35, dur=..., else_=ss.stay)`.

Also, the spec says a remainder branch can be "one transition with no p=". With no `dur` that raises `E209 UndrivenTransition: the transition "c_norm" (C -> Cnorm) has no rate, no duration, and no p=`. An immediate remainder (`C -> Cnorm`) is impossible; I had to compute `1 - sum` in a fourth `p` function. **Spec/impl mismatch.**

### 5. Stage-dependent infectiousness: clean, except the decay is agent-only
v3: `infect()` override (219-279, 60 lines, copied from ss.Infection) so that MTC uses rel_trans=1 and sexual transmission uses stage rel_trans. `set_latent_trans` (688-696) does exponential decay from `ti_latent`.
v4:
```python
infection = (S | Sp | Sr) >> E @ beta * (Pv + Ph + S2 + EL + LL)        # E and T excluded by omission
ss.modify(syph.rel_trans, multiply=latent_decay, where=syph.EL | syph.LL)  # latent_decay(sim) reads syph.ti_EL.raw
```
- The stage weights are clean. Per-network, per-state coefficients are possible (`dict(net=b) * EL`).
- The decay as a modifier with a function amount works, and gives results identical to a `step_state` write to `rel_trans` (389 vs 389 infections; 559 with no decay). But it relies on `ti_EL`, an engine-internal array that happens to survive the move EL→LL. Agents loaded straight into LL by `init=dict(LL=...)` have `ti_EL = NaN`, so their rel_trans is NaN and they transmit nothing. That is correct by accident, through `_nb_transmit`'s NaN-compares-False. `sim.solvers()` doesn't list this modifier as agent-only.
- **Impossible:** a per-agent `rel_trans` that applies to one route only. v3 wants decay on sex but not on MTC. In v4, `rel_trans` multiplies every route, so in-utero transmission can't be a network transmission here (see 9).
- **Suggest:** a lazy `EL.time` / `ss.since(EL | LL)` (time since entering a set of states) that works in rates, `p=`, and modifier amounts. It would be agents-only, or staged as an ODE. Also allow `rel_trans` adjust per route: `ss.modify(..., route='structuredsexual')`.

### 6. Sex-specific parameters: awkward
v3 `set_prognoses:746-752`: `p_symp_primary = [f, m]`, written into a bernoulli per call.
v4: `p=chancre_visible`, where that is a `fn(sim, uids)` with `np.where(female, .3, .5)`. That makes the model agent-only. `ss.by(sex=[...])` would need a sex stratification, and stratifying the sim by sex next to `StructuredSexual` was not tried. Acceptable, but the obvious spelling `p=dict(female=0.3, male=0.5)` doesn't exist. **Suggest:** accept `ss.by(sex=...)` on agents without requiring a stratification.

### 7. Treatment from any stage: clean. Different outcomes per stage: impossible
v3 `SyphTx.change_states:68-118` clears seven flags and four dates by hand. It also special-cases early-stage RPR reversion and window-period trep.
v4:
```python
cure = (E | Pv | Ph | S2 | EL | LL | T) >> Sp @ ss.peryear(0)   # moved only by treatment
bpg  = ss.Treatment(efficacy=0.98, transition='cure')
```
The many-to-one transition plus `Treatment(transition=)` is clean, and v3's "unnecessary" treatment simply moves nobody. But a Treatment has exactly one transition with one destination (`products.py:456`). v3 sends early-treated people to "RPR reverts in 6–12 months" and late-treated people to "RPR stays". To get that I'd need two cure transitions and two `ss.treat`s, but `on_positive=` takes one delivery. I approximated: every cured person reverts RPR at 9 months mean (`ss.transfer(rpr=False, dur=ss.months(9), states=['S','Sp','Sr'])`).
v3's window-period rule ("treated before trep appears: never trep+") is approximated by the dual test's 0.2 sensitivity in E/P. Cured-from-E still goes to Sp.
**Suggest:** `ss.Treatment(transition=dict(early='cure_early', late='cure_late'))`, or transitions with a destination per source: `cure = {E | Pv | Ph | S2 | EL: Sp, (LL | T): Spl}`. Also allow `on_positive=[...]`.
Delivery results have no success/failure/unnecessary split (v3 `new_treated_success/failure/unnecessary`, by sex). You can get success from the `cure` flow, but not failure versus unnecessary.

### 8. Serology: clean, the nicest part
v3: `trep`/`nontrep` bools with `ti_trep_start`/`ti_trep_end`/`ti_nontrep_end` (syphilis.py:184-185, 205-207, 528-543, 725-735, 760-771; SyphTx 96-118). `sus_not_naive` is a property. `SyphDx` probabilities by state come from a CSV.
v4:
- "Previously treated, seropositive" is a state, `Sp` (susceptible, reinfectable), plus `Sr` for sero-reverted, entered by `trep_revert` (p=0.2).
- The dual test is `ss.Test(sensitivity=dict(Sp=0.95, E=0.2, Pv=0.2, Ph=0.2, S2=0.95, EL=0.95, LL=0.95, T=0.95), specificity=0.99)`. That is the syph_dx_zim `dual` row, verbatim.
- RPR is a dimension: `syph.stratify(rpr=[False, True], split=[1, 0], rpr_wane_late=ss.transfer(rpr=False, dur=ss.years(15), states=['LL','T']), rpr_wane_tx=ss.transfer(...))`, plus `move=dict(rpr=True)` on `secondary` and `reactivate`. `n_LL.by('rpr')` comes for free.
- `ss.Test(detects=syph.rpr, ...)` is a test for titre, not infection. This is what (5) in the brief asked for, and it is one line.
- Gotchas:
  - Naming the test product `'rpr'` raises `AttributeError: Module rpr already added`, because products share a namespace with people attributes.
  - A `move=dict(rpr=True)` on an unstratified disease raises `E1102` only at the first move during the run, not at init.
  - The v3 timing is lognormal(15, 5) years from late-latent onset. As an `ss.transfer` it is exponential from LL entry.

### 9. Maternal transmission and stage-dependent birth outcomes: ugly
v3: `set_congenital` (801-858) draws an outcome table by the mother's stage (`mat_active`/`early`/`late`), schedules `ti_<outcome>` at the mother's `ti_delivery` (fetal losses one step early, per #588), and fires it in `step_state` (486-509).
v4 needed six fetal states (`C, Cnnd, Cstill, Ccs, Cnorm, CS`) and five `fn(sim, uids)` functions that look up `people.parent`:
```python
mtct       = S >> C @ mtct_rate                                     # not a network transmission (see 5)
c_nnd      = ss.Transition(C, Cnnd, p=outcome_p(0))  # ... x4, each reading the mother's stage via people.parent
stillbirth = Cstill >> ss.null @ before_delivery                    # BIG hazard when ti >= mother's ti_delivery - 1
nnd        = ss.progression('Cnnd ->', rate=BIG, where=ppl.age >= 0)
```
- There is no "at a scheduled time" transition, so events at delivery use the idiom "hazard of 1e4 per year while a condition holds". `where=ppl.age >= 0` needs a dummy `ppl = ss.People(1)` at module level (known issue in implementation.md).
- MTC is written as a spontaneous hazard on the fetus that reads the mother's state. Network transmission over `MaternalNet` would apply the mother's per-agent latent decay, which v3 does not.
- **Critical bug: agents created by `ss.Pregnancy` or `ss.Births` are in no state of a declared disease.** Probe: after 5 years, 329 of 2,231 alive agents (all of them under 5) were in no state with `ss.Births`, and 212 of 2,150 with `ss.Pregnancy`. Unborn fetuses are also in no state, so nothing can infect them. This is silent: `n_S` just stops counting them. Only the `ss.birth()` inflow places people. The port overrides `step_state` to put stateless agents in `S` (`place_newcomers`). This is the one `step()` override left.
- Branch `p=` is evaluated at the step after entry, so the mother's stage is read a step late. A fetus infected in the step before delivery is born in `C`, branched after birth, and its stillbirth becomes a newborn death. v3 behaves similarly.
- **Suggest:**
  - Declared diseases must place newcomers from any demography module (into the first state, or a `born=` state).
  - A lazy `ss.mother(syph.EL)` / `ppl.parent` condition for `p=` and `where=`.
  - A timed transition `@ ss.at(lambda sim, uids: ...)` or `@ ss.when(cond)` for an event at a known time, instead of BIG hazards.
  - Possibly `ss.congenital(src=..., outcomes=ss.by_mother(...), at='delivery')` as a building block, since HIV, HBV, and syphilis all need it.

### 10. ANC screening: awkward
v3: `ANCTest` (base_interventions.py:658-827) and `ANCSyphTest` (syphilis_interventions.py:378-426) are two ANC paths with different semantics. The former tests `infected` with a single sensitivity, so treated-seropositive women never test positive. The latter uses `SyphDx` by state.
v4:
```python
anc = ss.screen(dual, on=lambda sim: sim.interventions.anc_visits.ti_visit == sim.ti, on_positive=ss.treat(bpg), start=2005)
```
I needed a 15-line custom `ANCVisits(ss.Intervention)` to draw one visit per pregnancy at month 1–8.
- Routine delivery remembers `offered_` for life, so a woman isn't screened again in her second pregnancy.
- Test `delay=` is the time to the *result*, not to the visit.
- There is no pregnancy event to put in `on=`, because Pregnancy isn't declared.
**Suggest:** `on=preg.conception, delay=ss.randint(...)` (a delivery delay), and per-episode routine delivery (`once_per=preg.pregnancy`).

### 11. Fetal treatment when the mother is treated: clean
v3 `SyphTx.treat_fetus:120-166` walks `maternalnet` by hand.
v4: `ss.treat(ss.Treatment(efficacy=fetal_eff, transition='fetal_cure'), on=syph.cure, trace='maternalnet')`, where `fetal_eff(sim)` is 0.75 in the last trimester and 0.98 otherwise. Tracing and event triggers compose well. It fires a step after the cure (forced moves are events from the next step).

### 12. Newborn test and treatment: partial
`ss.treat(Treatment(transition='cs_cure'), on=syph.born_cs, coverage=0.5)` works. v3 only tests newborns of ANC-positive mothers, which needs memory of the mother's result. Delivery records `result` only for delayed tests, and there is no "mother was diagnosed" condition, so this isn't reproduced.

### 13. Capacity: works, no queue
`ss.treat(bpg, capacity=ss.peryear(6))` as the ANC follow-up: 422 positive, 114 treated, and the rest are dropped, not queued. For state-eligible rounds (`symp`), the people not reached stay eligible, so a backlog is implicit. There is no priority. v3 `STITreatment` has `max_capacity` and an unused `queue`.

### 14. Using the v3 network (`sti.StructuredSexual`) under a declared disease: awkward
- Declared pars are `ss.par` objects, so `net_beta` reading `disease.pars.eff_condom` builds a `BinOp` and crashes: `TypeError: float() argument must be a string or a real number, not 'BinOp'`. Workaround: `syph.pars['eff_condom'] = 0.5` (a plain value).
- A bare beta is a hazard converted to `1-exp(-beta*dt)` before `net_beta`, which wants a per-act probability. I had to invert it: `per_act(p) = -log(1-p)/dt`.
- **Units footgun:** with `dt=ss.months(1)` and no `unit=`, the unit is inferred as *month*. Every bare rate (beta, and the numbers my functions return) was silently read per month, which is 12× too high. `describe()` does print `unit month [from dt]`. Fixed with `unit='year'`.

### 15. Results: mostly automatic, a few gaps
- Free: state counts, a flow for every transition (`stillbirth`, `nnd`, `born_cs`, `reactivate`, `tertiary`), `by('rpr')`, and delivery `delivered`/`diagnosed`.
- Sums and ratios (`infected`, `prev`) work as constructor kwargs but are **silently ignored in the class body**: `class_members()` collects only State/Transition/par/outcome. That breaks the spec's equivalence rule, and I had to pass them from `__init__`.
- The spec's `infection.select(source='Sp')` (for reinfections) isn't implemented: `"The result "infection" has no dimension "source"; its dimensions are ['rpr']"`.
- v3's prevalence among sexually active adults by age and sex (`update_results:580-671`) has no declared equivalent. It would need an analyzer.
- `describe()` prints rate transitions as `dwell ~ exponential(rate = mtct_rate)`, although they are drawn as hazards each step, which is misleading.

### Other v4 bugs and mismatches
- `ss.Transition(A | B, C, ...)` (a StateSet source in the explicit form) is rejected. The error path itself then crashes: `TypeError: argument 'a': 'StateSet' object cannot be converted to 'PyString'` in `transitions.closest()`. A list `[A, B]` works.
- An ODE run correctly refuses with E804 (Pregnancy works on agents).

### v3 bugs found while porting (stisim)
- `SyphTx.change_states` (syphilis_interventions.py:81-94) never clears `exposed`. Someone treated in incubation stays `exposed=True` and `susceptible=True` forever: they inflate `n_exposed` and `mat_active`, and are stuck, since `ti_primary` is NaN.
- `set_congenital` (syphilis.py:813) sets `susceptible=False` for *every* in-utero-infected fetus, including the "normal" outcome. `NewbornTreatment` (208-212) doesn't restore it either. These children can never acquire syphilis as adults. The port sends them to `S`.
- `treat_fetus` (syphilis_interventions.py:153-164) adds the "success" count once per outcome key outside the `if`, so for `'normal'` it re-adds the previous key's count. Successes are double-counted and "unnecessary" is under-counted.
- `immune`/`ti_immune` (syphilis.py:182, 209) are declared and never used. There is no post-treatment immunity in v3, so none is ported.
- The dual-test CSV keys on `sus_not_naive` (ever-exposed), so trep sero-reversion (`p_trep_persists`) has no effect on test results.
- The `dur_early` comment says "post-infection", but the code measures from latent onset.

## Ranked: top problems for this model
1. **Newborns and fetuses from `ss.Pregnancy`/`ss.Births` are in no state of a declared disease.** It is silent and breaks any model with demography. Needs a `step_state` override.
2. **Pregnancy-linked biology has no vocabulary.** Mother's state, an event at delivery, and per-pregnancy ANC scheduling cost six fetal states, five `fn(sim, uids)` lookups via `people.parent`, BIG-hazard "due" transitions, a dummy `ss.People(1)`, and a custom scheduling intervention.
3. **Partial branches ("p of them eventually do X, the rest stay")** are inexpressible (E205), and an immediate remainder raises E209, against the spec. Fake rate-0 branches pollute `describe()` and results.
4. **One Treatment = one transition = one destination.** Stage-specific post-treatment outcomes need several deliveries, and `on_positive=` takes one.
5. **Time since entering a state** is reachable only through the engine-internal `ti_<state>` (agent-only, NaN for initial cases). There is no route-specific `rel_trans`.
6. **Units are inferred from `dt`, and declared pars are `ss.par` objects.** Both silently break v3 networks and bare-number functions.
7. Class-body result expressions are silently ignored. `select(source=)` is missing.

## Surprisingly nice
- Stage natural history, competing dwell times (early→late vs reactivation), and automatic per-transition flows replace about 400 lines of `ti_*` bookkeeping, `step_die`, and results.
- Serology as states (`Sp`, `Sr`) plus an `rpr` dimension with `move=` and state-scoped `ss.transfer`s. A test by state dict reproduces the v3 CSV, and `detects=syph.rpr` is a titre test in one line.
- Treatment from every stage is one many-to-one transition. Fetal treatment is `on=syph.cure, trace='maternalnet'`.
- `ss.modify(rel_trans, multiply=fn, where=EL | LL)` matched the hand-written `step_state` exactly.
- v3's `sti.StructuredSexual` ran unmodified under a v4 declared disease, once the two issues in 14 were worked around.
