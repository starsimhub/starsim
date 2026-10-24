# STIsim bacterial STIs on Starsim v4: review

Scope: `stisim/diseases/sti.py` (BaseSTI, SEIS), `chlamydia.py`, `gonorrhea.py`, `trichomoniasis.py`, `interventions/base_interventions.py` (STIDx, STITest, SymptomaticTesting, SyndromicManagement, STITreatment), `gonorrhea_interventions.py`.
Port: `port.py` (runs all 8 sections in about 10 s; output in `run_output.txt`). 4,000 agents, 100k people, monthly steps, 2000–2015, `ss.MFNet(n_contacts=0.8, concurrency=0.15)`. Betas are uncalibrated.

## State graph used

```
S -> E -> P (presymptomatic) -> Y (symptomatic) -> C (seeking care: the clinic queue) -> N (gave up) -> S
          \-> A (asymptomatic) -> S | PID (women) | S after 100 y (TV women, persistent)
Y -> PID | C | S      PID -> C | S
(E|A|P|Y|C|N|PID) -> T (treated) -> S (cure, p=efficacy) | A (fail) | A + strain sens->res (acquire)
```
v3 keeps overlapping booleans (`asymptomatic`, `symptomatic`, `pid`, `seeking_care` all set at once) plus scheduled `ti_*` times. v4 needs exclusive states, so the overlaps turn into extra states (P, C, N, T).

## Feature by feature

### 1. Latent period, sex-specific symptomatic/asymptomatic split
- v3: `sti.py:627-663` (`set_exposure`, `set_symptoms`, `set_pars` turning `[f, m]` into per-agent arrays).
- v4: `ss.progression('E -> P | A', p=by_sex(0.20, 0.54), dur=ss.constant(...))`, with `by_sex = lambda f, m: lambda sim: np.where(sim.people.female, f, m)`.
- Partly declarative. **Awkward.** I'd want to write `p=ss.by(sex=[...])`, but that is broken (bug 1). The split is right: NG F 0.35 / M 0.66 (target 0.35 / 0.65), CT 0.22 / 0.57 (target 0.20 / 0.54).
- When `dur_exp=0` (NG, TV), `ss.Transition(E, [P, A], p=...)` with no `dur` crashes at run time (bug 3), so it has to be two transitions with complementary `p`. Also, "immediate" means at the next step, so E always lasts one step. That matches v3, which also waits a step.

### 2. Sex-specific dwell-time distributions (lognormal mean and std per sex)
- v3: `sti.py:690-703`. Means and stds come from `set_pars`, then `dist.set(mean=arr, std=arr)`.
- v4: there is no way to say this on one transition:
  - `dur=ss.by(sex=[lognorm_f, lognorm_m])` gives `TypeError: float() argument must be ... not 'lognorm_ex'` (`transitions.py:2161`).
  - `ss.lognorm_ex(mean=ss.by(...), std=ss.by(...))` gives `TypeError ... not 'by'` (`transitions.py:1041 dwell_stats`).
  - A callable distribution parameter fails the same way.
  - `dur=fn(sim, uids)` runs, but it sets the *mean of an exponential* (`solvers.py:3563`), not the draw.
  - `stratify(adjust=)` only scales the rate, so the coefficient of variation can't differ by sex (v3 has CV 12 for women and 4 for men).
- What I did: one transition per sex. For timed exits I used `where=ppl.female` / `where=~ppl.female`. For branches I used `p=by_sex(p_f, 0)` and `p=by_sex(0, p_m)` (helper `sexed()`). A three-way split for symptomatic women (PID / care / clear) becomes 5 transitions. I had to multiply out the joint probabilities by hand (`(1-p_pid)*p_care`).
- **Ugly.** About 20 transitions per disease, where v3 has one parameter list per quantity.
- The `where=` transitions also print `RuntimeWarning: divide by zero` (`solvers.py:3570`) on every run.
- Suggestion: let any `ss.Dist` parameter take `ss.by(...)` (and accept a Dist inside `ss.by`), sampled per agent's group. Compartmental runs can moment-match each group.

### 3. Clearance competing with progression; PID; TV persistence
- v3: `sti.py:678-703`, `trichomoniasis.py:75-90`. The fate is decided at infection and `ti_*` times are scheduled.
- v4: "split on entry, then wait" (`p=` plus `dur=`) is exactly v3's logic, so this maps well:
  - PID comes from A and Y with `p_pid` for women. CT PID shows up as 5,225 people in section 1.
  - TV persistence is a second A → S branch with `dur=ss.constant(1200)`.
- **Clean apart from the sex problem.** Two transitions with the same source and destination but different names are allowed.
- Caveats:
  - PID-destined people can't also clear before PID (same as v3).
  - v3 schedules PID from the time of becoming infectious, whatever the symptom state. v4 needs a PID branch out of every state, so the timing is approximate.

### 4. Reinfection (SIS, no immunity)
- Every clearance goes to S. **Clean.**

### 5. Treatment from any infected state, with failure (focus 1)
- v3: `STITreatment.administer/change_states` (`base_interventions.py:906-941`) moves `treatable` people to S with `treat_eff`. Failures stay put and are counted (`new_treated_failure`).
- What `ss.Treatment` does: `efficacy` is the chance of being moved along *one* transition (`products.py:441`, `interventions.py:1026-1033`, `eng.force`).
  - (a) With no `transition=` you get `E505 ... there are several: clear_a_f, clear_a_m, ...`. Every clearance counts as a "recovery transition", so any disease with more than one clearance path has to name one.
  - (b) A multi-source transition `(E|A|P|Y|C|N|PID) >> S @ ss.permonth(0)` plus `Treatment(efficacy=0.9, transition='cure_direct')` works: `force()` moves people from whichever source they're in. But failures are silent (not a result). A failure in C stays in C, so the next round treats them again, and a failed treatment becomes "retreat next month".
  - (c) What I used: `Treatment(efficacy=1, transition='treat')` into a transient state T, which branches `cure` (p=efficacy) or `fail`. Failures are now a flow (`cured 3375, failed 275`), and efficacy can vary by strain and time.
    - Cost: T lasts a whole step, so cure lags a month, and the person is infected but not infectious meanwhile.
    - "Fail back to the state they came from" is impossible; they all go to A.
- `graph.infected` (default eligibility, and what tests detect) is "transmission destinations plus infectious states". T is left out. A non-infectious sequela state would be left out too, silently.
- **Awkward.**
- Suggestions:
  - `ss.Treatment(efficacy=..., to='S', from_states='infected', on_fail=...)`, with built-in `cured`/`failed` results.
  - An explicit `infected=` set on the disease.

### 6. Symptomatic care seeking with a delay
- v3: `sti.py:665-688` (`ti_seeks_care`), with `seeking_care` reset each step (`sti.py:576-581`). Demand that isn't met that step is lost.
- v4: care seeking is a branch Y → C with a lognormal delay, and C is a state. **Clean.** It needs 3 transitions (`seek_f`, `seek_m`, `seek_pid`).

### 7. Capacity: 100 a week (focus 2)
- v3: `max_capacity` is accepted by `STITreatment` (`base_interventions.py:839,853`) and `GonorrheaTreatment`, and its docstring says there is a queue. But `step()` never uses it, so **v3 has no capacity at all**.
- v4 results (section 3, 1M people, about 700 care seekers a month):

| delivery | seeking/month | treated/month | queue n_C | gave up/month |
|---|---|---|---|---|
| (a) `eligible=C`, routine | 713 | 533 | 201 | 168 |
| (b) `eligible=C, interval=1 month` | 668 | 652 | 0 | 0 |
| (c) (b) + `capacity=ss.perweek(100)` | 1057 | 418 (cap 434) | 852 | 611 |
| (d) `on=seek_*`, cap | 1586 | 430 | 1549 | 1119 |

- Queue: it works, and nothing about it is special. Unmet demand stays in C, every round re-offers it, and `balk = C >> N @ ss.months(1)` drains it. **Clean**, once you know to use `interval=` rather than routine delivery.
- Lost demand: `on=` gives it, because only people who started seeking this step are offered.
- **Footgun:** routine delivery offers each person treatment *once in their lifetime* (`offered_<name>`), so reinfections are never treated (533 treated vs 713 seeking). For any SIS disease that is wrong by default.
- **Priority: impossible declaratively.**
  - `deliver=` returning UIDs in priority order and the same list reversed give bit-identical results. `cap()` picks at random (`interventions.py:996`), and `uids(sel)` drops the order before that.
  - Capacity belongs to one delivery, so a clinic can't share one budget between symptomatic care and screening, or between NG/CT/TV drugs.
  - Workaround (e2): the second delivery gets `capacity=fn` returning what the first left. That needed `* sim.current_scale`, because mid-run results are in agents and only scaled at the end. Without it the cap was off by 25×. It also overshoots by rounding (max 125 vs 109).
- `on=ng.seek_f | ng.seek_m` gives `TypeError: A set of sources needs parentheses: write (S | I | R) >> D @ rate`, which is misleading. It works through the internal `ss.conditions.as_cond(ng.seek_f) | ...`.
- Suggestions:
  - `capacity=` as a shared object, e.g. `clinic = ss.Capacity(ss.perweek(100))`, passed to several deliveries.
  - `priority=` (a condition list, or a key function), honored by `cap()`.
  - `queue=True/False`, with a `waiting` result, so queue vs lost doesn't depend on whether you picked `eligible=` or `on=`.
  - Make routine delivery once per *episode*, or warn when the eligible state can be re-entered.

### 8. Screening by age, sex and risk with coverage; testing to treatment with loss to follow-up
- v3: `STITest` (`base_interventions.py:58-255`) plus treatment eligibility lists passed by hand (`tx.eligibility = tx.eligibility | uids`, which is fragile; see `:971`).
- v4:
  ```python
  ss.screen(ss.Test(sensitivity=0.9, specificity=0.98, delay=ss.weeks(2)), eligible=..., coverage=0.3,
            interval=ss.years(1), on_positive=ss.treat(drug, coverage=0.8))
  ```
- **Clean**, and much better than v3:
  - The delay shifts results by a step.
  - Loss to follow-up comes out at 0.83.
  - False positives are treated and show up as overtreatment.
  - Results come by sex and risk for free.
- Risk: `ppl.risk == 'high'` gives `AttributeError: People object has no attribute "risk"`. `ng.risk` also errors. Lazy conditions only exist for a stratification declared *on the disease*, not on the sim. The fallback is a function using the level index (`sim.people.risk == 1`), not the label. **Awkward.**

### 9. Syndromic management and overtreatment (results by sex)
- v3: `SymptomaticTesting`/`SyndromicManagement` (`base_interventions.py:258-655`). One module routes each visit to several treatments and records TP/FP/FN/TN on each disease.
- v4: one `ss.treat` per disease (a delivery acts on one disease), each with `eligible=ng.C | ct.C | tv.C`. Conditions across three diseases' states that share a name work.
- **Bug-like order dependence:**
  - The NG delivery moves its own C people into T first, so the CT delivery no longer sees them: 9,250 vs 3,275 delivered for the same clinic visits. Overtreatment is badly undercounted.
  - Fix: a `Clinic` module with `step()` that flags visits before any delivery runs. That makes them equal (9,175 each), and CT overtreatment goes from 1,200 to 7,075.
- Overtreatment is not a result. I compute it as `delivered.by('sex') - treat.by('sex')`. This only works because T is entered only by treatment. There is no `.select(source=)` (not implemented).
- **Awkward.**
- Suggestion: a multi-disease delivery (`ss.treat([ng_drug, ct_drug], ...)`) that picks its recipients once, with `overtreated` and `failed` results per disease.

### 10. Gonorrhea AMR
- v3: `GonorrheaTreatment` (`gonorrhea_interventions.py:15-75`).
  - Per-agent continuous `rel_treat`, copied from the infector (`gonorrhea.py:88-93`), multiplied down on failure (×0.99) and on unnecessary treatment (×0.995).
  - `UpdateDrugs` switches regimen under a threshold. It reads `sim.people.rel_treat`, which doesn't exist, so it looks broken in v3.
- v4: resistance as a strain.
  ```python
  ng.stratify(strain=['sens', 'res'], states=[...infected...])
  ```
  - `cure`/`fail` p: `ss.by(strain=[ss.timeseries({2000: .96, 2010: .90}), 0.30])`.
  - An `acquire` branch `T -> A` with `move=dict(strain='sens -> res')`.
  - The infector's strain is passed on automatically.
- Results:
  - Resistant share of new infections goes 0.02 → 0.72.
  - Cure share: sensitive 0.94, resistant 0.24 (0.29 pooled over 4 seeds; target 0.30).
  - Overall cure share goes 0.95 → 0.40.
- **Surprisingly clean.**
- Gaps:
  - No continuous inherited per-agent value.
  - Resistance from *unnecessary* treatment can't be represented: uninfected people have no strain. That is a modeling difference, not just syntax.
  - `p=None` for "the rest" raises `E209 UndrivenTransition` (the spec says it should take the remainder). And `ss.by` holding time series has no `1 - x`, so every branch has to be spelled out.
  - A transition can't be added after `ss.Disease` is built, so I had to thread `extra=` through the factory.
  - The drug switch (memory) would be a `step()` class plus `ss.modify(when=)`. Not ported.

### 11. ChlamydiaBL bacterial load (per-agent continuous state)
- v3: `chlamydia.py:89-177`. Log-linear growth to a peak, then decay, mapped to `rel_trans`.
- v4: `ss.modify(ct.rel_trans, multiply=fn(sim))`, where `fn` works out the load from time since infection. It runs (23,225 → 6,325 CT infections).
- **Awkward.** There is no "time since infection": `ti_E` only exists for people who entered E, and seeded people start in A with `ti_E = NaN`. My first version made every seed non-infectious and CT died out. I had to write `np.where(isnan(ti_E), ti_A, ti_E)`. Agents only.
- Suggestion: a per-disease `ti_infected` that the engine maintains.

### 12. Results
- Automatic:
  - `n_<state>` and every flow, by sex, risk and strain via `.by()`.
  - Delivery `delivered`/`eligible`/`coverage`/`diagnosed` by stratum.
  - The declared ratio `prev=infected/(S+infected)`, stratified.
- That replaces about 150 lines of v3 result code (`sti.py:241-427`, `505-625`, `STITreatment.update_results`). **Very nice.**
- Missing:
  - `.select(to='P')` on a branching flow: `KeyError: The result "latency" has no dimension "to"` (spec 03 says it exists).
  - Overtreated and failed counts.
  - Age-bin results without stratifying by age (which would split every state).

### 13. Mixing declarative transitions with `step()`
- A custom `ss.Intervention` with `define_states`/`step()` sitting next to declared deliveries (the `Clinic` workaround) works. **Clean.**

## Bugs and spec mismatches (with error text)
1. **Sex stratification is not `people.female`.** `stratify(sex=['female','male'])` draws `people.sex` independently of `people.female`, which MFNet pairs on: they agree for 0.487 of agents. So `ss.by(sex=...)` and `.by('sex')` are silently random with respect to the network's sexes. Workaround `sync_sex()`: copy after `sim.init()` and call `sim.groups_changed()`. That would break with births.
2. **`unit='month'` with `start=2000, stop=2015` silently reads them as months.** The timeline starts in year 0166 with 3 steps. `dt=ss.months(1)` without `unit=` works.
3. **A branching transition with no rate or dur** (`ss.Transition(E, [P, A], p=...)`, or `ss.progression('E -> P | A', p=...)`) is classified as a hazard with rate None, then fails at run time: `TypeError: Cannot evaluate None` (`transitions.py:423`). It should be an immediate branch (spec 03) or an E209 at build. One transition per destination works.
4. `p=None` as "the remainder" gives `E209 UndrivenTransition` (spec: "one transition with no p= gets the remainder").
5. `.select(to=...)` and `.select(source=...)` are in the spec but not implemented.
6. Dists don't accept `ss.by` parameters, and `ss.by` doesn't accept Dists (TypeErrors above).
7. Deliveries and `capacity=fn` read results mid-run in agents, not people. That isn't documented.
8. Two products with the same explicit name: `ValueError: Cannot add module "drug" to products`. Unnamed products are auto-renamed with a warning instead. Products made inside lambdas can't be named from the variable.
9. Reusing state objects in a second `ss.Disease`: `E206 DuplicateDeclaration: state "S" is used by transitions in two modules, "ng" and "ng"`. Factories must make fresh states. Fine, but the message is confusing when both module names are the same.
10. The `where=` transitions emit `RuntimeWarning: divide by zero` every run.
11. `on=a | b` on transitions: misleading TypeError (see 7).
12. Lazy conditions don't exist for sim-level stratifications (`ppl.risk`).

## Line counts (non-blank, non-comment)
- v3:
  - `sti.py` 555 (BaseSTI + SEIS)
  - `gonorrhea.py` 72, `chlamydia.py` 133, `trichomoniasis.py` 63
  - `STITreatment` 154, `SymptomaticTesting` 133, `gonorrhea_interventions.py` 98
- v4: `make_sti` 46, plus about 20 lines of helpers (`by_sex`, `sexed`, `sync_sex`). That covers all three diseases (parameters are a dict), PID, persistence, care seeking, treatment states and every result.
- Interventions are 1–3 lines each; AMR is about 10.
- Overall about 5× less code. The v3 results code alone outweighs the whole v4 disease.

## Ranking

**Top problems**
1. Sex. `stratify(sex)` disagrees with `people.female` (silent and wrong). On top of that, there are no sex-specific dwell *distributions*, so the number of transitions doubles and branch probabilities have to be multiplied out by hand.
2. Capacity can't be shared across deliveries and ignores priority. Combined with one-disease-per-delivery and loop-order effects, a single syndromic clinic can't be expressed without a `step()` module.
3. Treatment semantics. "Efficacy" means one transition, and failures are invisible. Writing a treated state T to get failures costs a step of lag, and "fail back to where you were" is impossible. `graph.infected` silently leaves out non-infectious infected states.
4. Routine delivery is once per lifetime, which is a trap for reinfecting diseases.
5. Smaller sharp edges:
   - `unit='month'` misreads dates.
   - Immediate multi-destination branches crash.
   - `p=None` remainder errors.
   - `.select(to=)` is missing.
   - Mid-run results are in agent units.
   - There is no time-of-infection per agent.

**Surprisingly nice**
- A queue with balking comes for free from a persistent state, `interval=`, and `capacity=`, which v3 never actually had.
- `ss.screen(..., delay=, on_positive=ss.treat(coverage=))` covers the whole test → treat chain with loss to follow-up, false positives and by-sex results in two lines.
- AMR as a strain, with `move=` and an `ss.timeseries` inside `ss.by`, works and is short.
- Automatic results by any stratum, including deliveries and declared ratios.
- Branch-on-entry-then-wait matches v3's "decide fate at infection, schedule times" logic one-to-one.
