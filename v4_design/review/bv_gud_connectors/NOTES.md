# BV (CST model), GUD, and coinfection connectors in Starsim v4

Scope: `stisim/diseases/bv.py` (`BV`, `BVPars`; `SimpleBV` not ported), `stisim/interventions/bv_interventions.py` (`treat_BV`), `stisim/diseases/gud.py` (`GUD`), `stisim/connectors/hiv_sti.py`, `stisim/connectors/gud_syph.py`. HIV (with continuous CD4), syphilis, and CT/NG/TV are minimal stand-ins so the connectors have something to act on.

`conda run -n starsim python port.py`: 5,000 agents, 2000–2015, monthly, 7 diseases, `ss.MFNet`, births/deaths. 1.5 s per sim, 5 s for all four sims.

```
BV (CST4) prevalence among women 15+: 0.60 -> 0.48 (yr 5) -> 0.48 (yr 15); symptomatic 0.030
CST shares at end: CST1 0.08, CST3 0.35, CST4a 0.43, CST4p 0.02, CST4s 0.03, P 0.09
BV treated: 8917 (MTZ doses 124838); cured into P: 8023; recurred: 4467; settled: 3350; recurrence share 0.57
  BV prev with/without poor_hygiene: 0.58 / 0.36        (concurrency 0.59/0.48, uncircumcised partner 0.60/0.44; all ~0.27 with risk off)
Connector checks (at the end of the run):
  hiv.rel_sus composition matches product of 5 STI multipliers: True; agents by number of coinfections: {0: 3961, 1: 1199, 2: 448, 3: 160, 4: 39, 5: 1}
  syph.rel_sus = CD4<500 x CD4<200 x GUD: True
With vs without connectors (cumulative infections): hiv 795 vs 232, syph 3148 vs 1307, syph tertiary 329 vs 153
  effective syph.rel_sus among HIV+ = 1 + (500-CD4)/500 (x GUD): True
BV prevalence at end, with/without treatment: 0.484 / 0.551
```

## Feature by feature

### 1. Current CST (CST1/3/4): states
- v3: `cst` FloatArr plus `cst1`/`cst3`/`cst4` properties that each re-mask `female & age > 15` (bv.py:407, 452–480). `infected`/`susceptible`/`asymptomatic`/`symptomatic` flags are kept in parallel with `cst` by hand.
- v4: `U, CST1, CST3, CST4a, CST4p, CST4s, P = ss.states(...)`. BV is CST4, split into asymptomatic, presymptomatic, and symptomatic. `U` holds everyone without a CST (men, girls under 15). `P` is the post-treatment window.
- **Declarative. Clean.** Each agent is in exactly one state, which removes v3's need to keep four flags in sync. The cost is that men sit in `U` forever, so `n_U` is a meaningless result.

### 2. Stable CST: a stratification
- v3: `stable_cst` FloatArr from `ss.choice([1,3,4])`, `rr_stable_cst*` applied to `rel_sus`, `p_symp[stable]`, and `spontaneous_clearance[stable]` (bv.py:315–336, 563–564, 639–641, 679–693).
- v4: `bv.stratify(stable=['1','3','4'], split=[.1,.3,.6], adjust={'shift13':[0.1,1,2], 'onset':[...], 'recur':[...]})`. Symptom probability and the clearance destination are `p=ss.by(stable=[...])` on branches.
- **Declarative. Clean** for adjust and `p`. **Awkward** for `where=`. Inside the class body `bv.stable` doesn't exist yet (it's created by `stratify()`), so I had to write `where=ss.Cond('dim','stable') == '1'` by hand. `describe()` lists the dimension but not its `adjust=`.
- Suggestion: allow `stratify` in the class body (`stable = ss.dim(['1','3','4'], split=...)`), so `where=stable == '1'` works.

### 3. Who is at risk (women 15+): gate state and "as soon as" transition
- v3: every property re-ANDs `female & age > 15` (bv.py:450–480). `step()` also runs `set_cst(upper_age=dt.years)` on newborns.
- v4: `debut = ss.progression('U -> CST1 | CST3 | CST4p | CST4a', rate=ss.perday(1), where=adult_f, p=[by_stable(...)...])`, where `adult_f = ppl.female & (ppl.age >= 15)` on a throwaway `ss.People(1)`.
- **Declarative. Awkward.** "Move immediately when a condition becomes true" is a rate of 30/month, and it triggers a `rate × dt = 30` note (questions-oct12 #11). The `ss.People(1)` object exists only so that a lazy condition can be written (known issue).
- Suggestion: `ss.progression('U -> ...', when=adult_f)`, meaning an immediate move on the step the condition becomes true. Also expose a global lazy `ss.people`.

### 4. Spontaneous onset driven by risk factors (focus 1)
- v3: `step()` re-draws douching, hygiene, and concurrency for every woman every step. `set_rel_sus()` multiplies RRs into `rel_sus`. `spontaneous()` returns `p_cst_change[cst] * rel_sus` as a per-step probability. Circumcision does a network lookup (bv.py:497–566, 582–602).
- v4:
  - Base rates: `shift13 = ss.progression('CST1 -> CST3', rate=ss.probpermonth(0.10))`; `onset = ... ss.probpermonth(0.05)`.
  - Each risk factor is `ss.modify(transition, multiply=rr, where=...)` (`bv_risk()`).
  - `where=` is a per-agent BoolArr (`bv.poor_hygiene`), a dimension (`douching`, `hc` via `adjust=`), or a function of the sim that reads the network (`concurrent`: `net.degrees(auids) > 1`; `uncirc_partner`: women whose `MFNet` partner `p1` is uncircumcised).
  - Hormonal contraception (not in v3) is a dimension with two `ss.transfer`s (start and stop), so the risk factor changes over time and is fully declarative.
- **Declarative. Clean.**
  - `what_modifies('bv.onset')` lists every factor. Modifiers multiply as specified.
  - Converting to rates fixes a v3 bug: v3's `p × RR` can exceed 1.
  - A `rate=fn(sim, uids)` hazard also works (used for recurrence, below).
- **Awkward:**
  - One modifier per factor per transition: 8 modifiers for 3 factors over 3 transitions, built in a loop with hand-made names. Suggestion: allow a list target, `ss.modify([bv.shift13, bv.onset, bv.recur], multiply=1.28, where=concurrent)`.
  - Each modifier calls its `where=` function separately, so the same network scan runs once per transition each step. Nothing caches it.
  - `ss.modify(bv.rel_sus, ...)` on a disease with no transmission is silently a no-op (tested: `multiply=0` changes nothing). It should warn, since v3 users will reach for `rel_sus` (v3 BV uses it as the onset multiplier).

### 5. Symptoms, clearance, return to the stable CST
- v3: `set_symptoms`/`set_duration` compute `ti_symptomatic`/`ti_clearance` from uniform durations. `clear_infection` maps stable → destination (bv.py:633–700).
- v4: `CST4p >> CST4s @ ss.uniform(weeks 1–2)` and `clear_s = ss.progression('CST4s -> CST1 | CST3', dur=ss.uniform(weeks 5–50), p=[by_stable(1,0,0), None])`. `revert31` returns transient CST3 to CST1 in stable-CST1 women only.
- **Declarative. Clean.** Competing dwell times and hazards out of CST3 (`revert31` vs `onset`) just work.
- `describe()` problems:
  - It prints `ss.uniform(low=1, high=18)` with no unit (it is weeks).
  - It says `as 4 Erlang stages` under `abm`, where the uniform is actually sampled exactly.
  - `where=` on a `dur=` transition emits `RuntimeWarning: divide by zero` at solvers.py:3570 (duration / 0 = inf, which is harmless).

### 6. Treatment and recurrence after treatment (focus 2)
- v3 `treat_BV` (bv_interventions.py:135–215):
  - Eligibility is `ti_symptomatic == ti`, then `p_seek_care = 0.9`, then efficacy by stable CST. A cured woman goes to `cst = 1` if stable-1, else 3.
  - The optional `durable_cst1_change` sets `stable_cst = 1` for 6 months (shorter with douching or poor hygiene), then reverts it. That is the only time-since-treatment dependence in v3.
- v4:
  ```python
  tx     = (CST4a | CST4p | CST4s) >> P @ 0          # a channel for the cure; never fires on its own
  recur  = ss.progression('P -> CST4p | CST4a', rate=recur_rate, p=[by_stable(.8,.7,.1), None])
  settle = ss.progression('P -> CST1 | CST3', dur=ss.months(6), p=[by_stable(1,0,0), None])
  def recur_rate(sim, uids): return ss.permonth(recur0 * np.exp(-(sim.ti - bv.ti_P[uids]) * dt_months / tau))
  mtz = ss.Treatment(efficacy=ss.by(stable=[.95,.9,.9]), transition='tx')
  ss.treat(mtz, on=bv.symptoms, coverage=0.9, disease='bv')
  ```
  The recurrence hazard decays with time since treatment (from the engine's automatic `ti_P`) and competes with a fixed post-treatment window. Recurrence share is 0.57; treatment lowers BV prevalence from 0.55 to 0.48.
- **Mostly declarative. Awkward in three places:**
  - **A cure needs a dummy `@ 0` transition.** `ss.Treatment` cures only along a named transition out of the "infected" states, and BV has none: `graph.infected` is built from transmissions, so it's empty for a non-transmitted condition. Suggestion: `ss.Treatment(efficacy=..., to=bv.P)` or `move='CST4* -> P'`, with no fake transition.
  - **`disease='bv'` is still required** even though `on=bv.symptoms` already names a BV transition: `E506 UnknownDisease: "treat_bv" needs to know which disease it is for, and the sim has 7 declared diseases`.
  - **Time since entering a state is agent-only.** It is written through the undocumented `ti_<state>` arrays in a `(sim, uids)` function. Suggestion: an expression `P.time` (time in the current state), usable in rates, so `recur0 * np.exp(-P.time / tau)` stays declarative and could become stages in a compartmental run.
- `on=bv.symptoms` is exactly v3's "newly symptomatic" rule. MTZ doses (`tx_doses = 14`) are `delivered × 14` after the run; no per-delivery quantity exists (only `cost=`).

### 7. Not ported
- **`sexual_transmission=True`** (a BV-microbe network transmission to and from men, with male states; bv.py:614–631, 725–734). It would need a second, male set of states (a male CST or carriage) plus a transmission, which is doable but off by default in v3.
- **Pregnancy and preterm birth** (bv.py:703–723; bv_interventions.py:195–206). The v3 code is broken anyway (see v3 bugs).
- **`SimpleBV`.**
- **GUD `init_prev_data`** by age, sex, and risk group.
- **Condoms.** Not implemented in v3 either; spontaneous mode only prints a warning (bv.py:533–534).

### 8. GUD
- v3: `ss.Infection` subclass with `step_state`/`set_prognoses`, 64 lines (gud.py:97–160).
- v4: `ss.Disease(infection=ss.transmission('S -> I', beta=dict(mfnet=[...])), recovery=ss.progression('I -> S', dur=ss.lognorm_ex(mean=ss.months(3), std=ss.months(1))))`, 4 lines.
- **Declarative. Clean.**

### 9. HIV with continuous CD4 (focus 4: a multiplier that is a function of a continuous variable)
- v4: a class-form HIV (S, A, C, D) with `define_states(ss.FloatArr('cd4'))`. `step_state()` calls `super().step_state()`, then sets CD4 from `ti_A`/`ti_C`/`ti_D`.
- **Partially declarative. Clean.** Mixing declared transitions with an overridden `step_state` in one class worked first time.
- A continuous effect is `ss.modify(syph.rel_sus, multiply=cd4_mult)`, where `cd4_mult(sim)` returns one value per agent (`1 + clip((500-cd4)/500)`). Verified exactly against the effective values.

### 10. Connectors (focus 4 and 5)
- v3: one `ss.Connector` class per pair. Each does `rel_sus[...] *= ...` in `step()` and relies on every disease resetting `rel_sus[:] = 1` in `step_state` (hiv_sti.py, gud_syph.py).
- v4: a flat list of 18 `ss.modify(...)` calls (`connectors()` in port.py), for example:
  ```python
  m(hiv.rel_sus,  multiply=2.67, where=syph.P | syph.Sec)
  m(syph.rel_sus, multiply=1.5,  where=lambda sim: sim.diseases.hiv.cd4.raw < 500)
  m(syph.tertiary, multiply=3,   where=aids)     # HIV speeds syphilis natural history (not in v3)
  m(hiv.rel_sus,  multiply=2.0,  where=bv.CST4a | bv.CST4p | bv.CST4s)
  ```
- **Declarative. Clean, and the best part of the port.**
  - Composition is exact. `hiv.rel_sus.effective` equals the product of the syph × TV × NG × CT × BV multipliers for every alive agent, across 0–5 coinfections. `syph.rel_sus` correctly stacks CD4 < 500 × CD4 < 200 × GUD.
  - There's no reset bookkeeping.
  - `what_modifies('hiv.rel_sus')` prints the whole matrix.
  - Modifying a transition of another disease (`syph.tertiary`) works, including rescaling dwell times already drawn.
  - The connectors roughly triple HIV infections (795 vs 232), so they really reach transmission.
- **Ordering and references (focus 5): fine.** Declare the diseases, build the modifier list from them, then pass both to `ss.Sim`; there's no need for the sim to exist first. Conditions on states of other diseases with the same state name (`ng.I`, `tv.I`, `ct.I`) resolve to the right module.
  - A modifier whose condition names a disease not in the sim fails at the first step with a plain `KeyError: 'A condition refers to the module "ng", which is not in the sim; the modules are rel_sus_multiply, mfnet, hiv'`, not at init and not with a code.
  - Unnamed modifiers are named `rel_sus_multiply`, `rel_sus_multiply_1`, so the default name omits the module and is useless in `what_modifies()`.
  - Disease factories return modules named `'disease'`, which get renamed from variables only at `ss.Sim`, so pass `name=`.
  - v3's automatic per-pair connectors (`connector_register()`, `sti.Sim`) have no v4 equivalent; a function returning a list is the natural replacement.
- **Lazy conditions on per-agent arrays are broken (bug 3)**, so every CD4 condition had to become a lambda.

## Bugs and mismatches between the spec and the implementation (v4)

1. **People born through `ss.Births`/`ss.Pregnancy` are in no state of any declared disease.** They can never be infected and are not counted. Repro: an SIS with `ss.Births(birth_rate=30)`, 1,000 agents, 20 years gives `n_alive` 1815 but `n_S + n_I` 1000; with `ss.Pregnancy()` it's 1285 vs 854. Only a disease's own `ss.birth()` places newborns. This is silent and changes dynamics badly (797 of 2236 alive agents were in no state of any of the 7 diseases). The port works around it with `place_newborns`, a connector that puts unplaced agents into each disease's first state.
2. **Newborns get level 0 of every disease dimension** (`stable='1'`, `douching=False`), not a draw from `split`. Measured: all 797 newborns were at level 0, against 126/385/785 among adults. Also worked around in `place_newborns`, which must then call `sim.groups_changed()`.
3. **A lazy condition on a module's non-state array loses its module.** The spec (07) names `hiv.cd4` as lazy. In practice `hiv.cd4 < 500` warns `Trying to access non-initialized Arr object`, prints as `(cd4 < 500)`, and is resolved against the module being modified: `AttributeError: Syph object has no attribute "cd4"`. The same happens for `where=bv.poor_hygiene` on an HIV target (`HIV object has no attribute "poor_hygiene"`). It only works by accident when the array belongs to the target module.
4. **A ratio result whose denominator is 0 raises `ZeroDivisionError: float division by zero`** (in `update_results`, transitions.py:410). It should give `nan`.
5. **Result expressions are silently dropped in two cases.** Expressions in a class body are dropped: `class_members()` keeps only State/Transition/par/outcome. A set of states (`|`) passed as a result kwarg is also dropped, while a sum (`+`) is kept. Neither raises an error.
6. **An arrow on boolean levels in `ss.transfer(hc='False -> True')` fails at the first step:** `KeyError: '"True" is not a level of "hc"; the levels are False, True'`. `hc=True` works. This should be accepted, or rejected at init.
7. **`ss.treat(on=bv.symptoms)` still raises E506** without `disease=`.
8. **`where=` on a `dur=` transition** emits `RuntimeWarning: divide by zero` (solvers.py:3570).
9. **`describe()` problems:**
   - It drops the units of `ss.uniform(ss.weeks(...))`.
   - It reports Erlang stages under `abm`.
   - It omits stratification `adjust=`.
10. **`ss.modify(x.rel_sus)` on a disease with no contact transition is a silent no-op.**

## v3 bugs found while porting
- **`gud_syph` is dead code** (gud_syph.py:28–44):
  - It defines `update(sim)`, which the loop never calls (it needs `step()`).
  - It reads `sim.people.gud`, which doesn't exist.
  - It *assigns* `rel_sus = 2` rather than multiplying, so it would overwrite other connectors.
  - It isn't in `connector_register()`.
  - Nothing in the repo uses it.
- **`treat_BV` never clears `bv.infected` or `asymptomatic`** (bv_interventions.py:190–193). It sets `bv.ti_clearance = nan`, so treated women stay `infected=True` forever, which inflates the `BaseSTI` prevalence and infection results.
  - `tx_delay` and `tx_effect_delay` (lines 41–42) are never used.
  - The pregnancy branch calls `pregnancy.rel_sus_ptb` and `pregnancy.set_prognoses(..., bv_update=True)`, neither of which exists on `ss.Pregnancy`. It would raise `AttributeError` the first time a pregnant woman is cured. `BV.update_pregnancy` uses `rel_ptb`/`set_ptb`, which do exist.
- **`BV.step()` runs `set_cst(upper_age=self.t.dt.years)`** (bv.py:605, which has its own `# TODO this looks wrong!`). It assigns a CST to every girl under one month old and calls `set_prognoses(..., 4)` for stable-CST4 newborns, flagging infants as `infected`.
- **`p_cst_change × rel_sus` is used as a Bernoulli probability and can exceed 1.** The maximum is 0.05 × 2 × 1.21 × 4.1 × 1.28 × 4 ≈ 2.5 (bv.py:590–600).
- **Douching, hygiene, and concurrency are re-drawn every step for every woman** (bv.py:497–505, called from `step()`). They behave as monthly coin flips, not traits. Concurrency is `poisson(0.05) + 1 > 1` unless `count_partners=True`.

## Line counts (non-blank, non-comment)
| | v3 | v4 |
|---|---|---|
| BV disease + pars | ~380 (bv.py:293–770) | ~45 (class) + 8 (stratify) |
| Risk factors (incl. network/circumcision) | inside the above | ~25 |
| BV treatment | 179 | 2 (+ the `tx`/`recur`/`settle` lines in the class) |
| GUD | ~50 | 6 |
| Connectors (6 pairs) | 184 | ~45 |
| Workaround for bugs 1–2 | – | 25 |

## Ranked: top problems
1. **Bug 1, combined with bug 2:** external demographics are incompatible with declared diseases. Every STIsim model uses `ss.Pregnancy`, so this blocks any real port.
2. **Bug 3:** lazy conditions on per-agent arrays (`hiv.cd4 < 200`, `bv.poor_hygiene`) are broken across modules, which is precisely the connector use case.
3. **Gating by sex and age** needs a gate state plus a rate-1e9-style transition plus a throwaway `ss.People(1)`. STIsim gates on sex everywhere (BV, PID, pregnancy outcomes).
4. **Treatment of a condition with no transmission** needs a dummy `@ 0` transition, and `disease=` is redundant.
5. **Results:**
   - Ratios crash on a zero denominator.
   - Class-body expressions and `|` sets are dropped silently.
   - There's no prevalence among a subgroup (women 15+) without hand-built sums.
6. **One modifier per transition per risk factor**, with no list target and no shared condition evaluation.
7. **Time since entering a state** exists only as undocumented `ti_<state>` arrays in a `(sim, uids)` function.

## Surprisingly nice
- **Connectors as `ss.modify`:** 18 one-line rules replace 6 classes. They compose exactly, need no reset bookkeeping, are listed by `what_modifies()`, can target another disease's transitions, and accept continuous per-agent multipliers.
- **Risk factors as modifiers on spontaneous transitions:**
  - A network-derived risk factor is a 5-line `where=` function.
  - A time-varying behaviour (contraception) is a dimension plus two `ss.transfer`s.
  - Rates × RR replace v3's probability × RR, which could exceed 1.
- **Branches with `p=ss.by(stable=...)` and `dur=`** express "symptoms and clearance destination depend on stable CST" in one line each. Competing dwell times and hazards out of CST3 and P just work.
- **Mixing declared transitions with an overridden `step_state()`** (continuous CD4) worked first time.
- Runtime: about 1.5 s for 5,000 agents × 180 monthly steps × 7 diseases.
