# 04 — Pregnancy and births

Status: proposal, not reviewed. Answers 03 Q10 and the README's "FPsim gaps" rows 1–2. Evidence: `fpsim/NOTES.md` §1–8, §10, "What the three proposals must cover"; `syphilis/NOTES.md` §9–10; `hiv/NOTES.md` §11; `hpvsim_interventions/NOTES.md` (newborn vaccination needs placed newborns). Current code: `demographics.py:380` (`ss.Pregnancy`, v3 unchanged), `networks.py:1730-1835` (`PrenatalNet`, `PostnatalNet`, `BreastfeedingNet`), `solvers.py` `Agents.place`/`place_new` (6e10bfe1 on `v4-rs`: agents from `people.grow` now land in each module's first state).

Uses, not repeated: 01 (`X.time`, `ss.clock`, `ss.attr`, events, exits scheduled at entry), 02 (deliveries, once per eligibility spell, `.since`), 03 (`ss.on`, `ss.stay`, `ss.mother`, `before=`/`after=`, the conveyor `stages='step'`, the cascade).

## 0. Summary

| Need | Today | Proposal | Agents | Compartments |
|---|---|---|---|---|
| Pregnancy visible to the declared world | v3 class: `ti_*`, callbacks | declared `ss.Pregnancy`: `NP`, `P` | states, flows | compartments |
| Gestational age, trimesters | `gestation` float | `preg.P.time`; `preg.tri1/2/3` | `ti - ti_P` | conveyor slots of `P` (exact on the grid) |
| Conception | `make_p_conceive`, `rel_sus` | one transition `conception`, modified like any other | hazard | flow |
| Loss by GA threshold | classified after the fact | `miscarriage` `before=loss_threshold`, `stillbirth` `after=` | gated hazards | flows from slots either side of the edge |
| Abortion | FPsim: 20% at conception | immediate branch `abortion = ss.Transition(P, NP, p=pab)` | same step | pass-through |
| Twins | `embryos_per_pregnancy`, two agents | `embryos = ss.attr(..., on=conception)` | n fetus agents | 2-level dimension over `P` |
| Births from deliveries, linked to the mother | `people.grow` in `step()`; ODE: `ss.null >> ... @ k*P2.n` | `ss.birth(per=delivery, n=embryos)` | children with `parent` | inflow whose rate is the delivery flow |
| Fetus | agent from conception, always | `fetus='auto'`: an agent only if something reads it | agent, age < 0 | `fetal_<disease>` level of `P` |
| Mother → fetus, child → mother | `people.parent` lookups | `ss.mother(expr)` (03), `ss.child(event)` | parent index | same compartment / approx. hazard |
| Events | none public | `conception`, `delivery`, `birth`, `live_birth`, `loss`, `infant_death` | `engine.events` | flows |
| Time since last birth | `ti - ti_delivery`; port: 5-state chain | `preg.since_birth` (event clock on `live_birth`) | float | clock dimension |
| Parity, births, intervals | counters, 2-D arrays | `ss.attr(..., on=..., combine='sum'/'first')` | floats | `bins=` dimension, moved on the event |
| Lactation, LAM, weaning on infant death | `ti_stop_breastfeed` | sibling module `ss.Lactation`: `N`, `L` | states | ×2, synchronized with births |
| Maternal death | `ti_dead`, `p_maternal_death` | antepartum hazard + `ss.on(delivery)` branch | yes | yes |

New names: `ss.birth(per=, n=)`, `ss.child(event, under=)`, `ss.Lactation`, `fetus=`, events above. Everything else is 01–03.

## 1. Rules

1. **Pregnancy is a declared module.** Its states, flows and modifiers are the same objects as a disease's, so `ss.modify`, `stratify`, results, `describe()` and the compartmental solvers apply unchanged.
2. **The mother carries the pregnancy.** Gestation, embryos and fetal disease levels are properties of her compartment. The fetus as an agent is an agent-side convenience, created only when a model needs to address it.
3. **Unborn agents are not people for results.** `n_<state>`, denominators and `ss.people.n` exclude age < 0, so agent and compartmental results agree. `.select(unborn=True)` shows them.
4. **A child is born by an inflow driven by a flow.** `ss.birth(per=delivery, n=embryos)` is the one way newborns appear in a declared sim, as agents or compartments. `ss.birth(rate=)` (crude) and `ss.Births` stay.
5. **Every link between mother and child is an event or an expression**, never a lookup in user code: `ss.mother(expr)` (on the fetus), `ss.child(event)` (on the mother), networks for anything pairwise after birth.

## 2. The module

```python
class Pregnancy(ss.Module):                      # declared; kind 'demographics'
    NP, P = ss.states('NP P')                    # everyone not pregnant (men and children included), pregnant
    infertile = ss.attr(ss.bernoulli(p_infertile))                         # trait at creation; 2 bins in an ODE
    fecund    = ss.people.female & (ss.people.age >= min_age) & (ss.people.age < max_age) & ~infertile
    fertility = asfr(ss.t + mean_dur_pregnancy) * fecund.n.by('age') / (fecund & NP).n.by('age')   # v3 make_p_conceive

    conception  = ss.Transition(NP, P, rate=fertility*rel_fertility, where=fecund)
    embryos     = ss.attr(embryos_per_pregnancy, on=conception)            # 1 or 2
    abortion    = ss.Transition(P, NP, p=p_abortion)                       # immediate; the rest stay (default 0)
    miscarriage = ss.Transition(P, NP, rate=loss_rate, before=loss_threshold)
    stillbirth  = ss.Transition(P, NP, rate=loss_rate, after=loss_threshold)
    antepartum  = P >> ss.null @ maternal_rate                             # maternal death while pregnant
    delivery    = ss.Transition(P, NP, dur=dur_pregnancy, stages='step')   # the conveyor: P.time is gestational age
    birth       = ss.birth(per=delivery, n=embryos, sex=sex_ratio)         # children, parent = mother
    intrapartum = ss.Transition(NP, [ss.null, ss.stay], p=[p_maternal_death, None], rate=ss.on(delivery))

    live_birth   = ss.child(birth)                                         # mother-side events
    infant_death = ss.child(ss.people.died, under=ss.years(1))
    loss         = miscarriage | stillbirth | abortion | infant_death
    since_birth  = ss.clock(start=live_birth)                              # inf if never; survives later losses
    parity       = ss.attr(1, on=delivery, combine='sum', init=0)
    tri1, tri2, tri3 = P & (P.time < trimesters[0]), P & (P.time >= trimesters[0]) & (P.time < trimesters[1]), P & (P.time >= trimesters[1])
```

Constructor: the v3 `PregnancyPars` names and meanings (`fertility_rate` as a frame or rate, `rel_fertility`, `p_infertile`, `min_age`, `max_age`, `rate_units`, `dur_pregnancy`, `p_loss`→`loss_rate`, `loss_threshold`, `embryos_per_pregnancy`, `p_maternal_death`, `sex_ratio`, `trimesters`, `burnin`, `dur_breastfeed`, `p_breastfeed`) plus `p_abortion=0`, `fetus='auto'`, `lactation='auto'`. `rr_ptb`/`rel_ptb` are §9 Q6.

### 2.1 Gestation

- `delivery` with `stages='step'`: `P` is a conveyor of `ceil(42 wk/dt)` slots; slot `i`'s exit hazard is `dur_pregnancy`'s discrete hazard at `i·dt`. `P.time` is the slot. Monthly: 10 slots; weekly: 42.
- Agents: delivery time drawn at entry (01 §4), rounded to the step as now. `ti_delivery` is that scheduled exit.
- Trimesters and the loss threshold are edges on the conveyor: exact on the grid.
- Shifting delivery (FetalHealth-style PTB): `ss.modify(preg.delivery, multiply=rr, where=...)` rescales the remaining dwell (`rescale_exits`), as for any dwell. Compartments: multiplies each slot's hazard (exact for the hazard model; the dwell is `choice`, so "×rr on duration" is approximated by hazard scaling, with a note).

### 2.2 Conception and what modifies it

- One transition, `NP >> P`. Everything else is a modifier or an expression in its rate:
  - multiplicative (HIV on fertility, contraception as `1 - eff`, LAM, partner availability): `ss.modify(preg.conception, multiply=..., where=...)`;
  - non-multiplicative (FPsim's cap at 1 per year before conversion, §5.1): a subclass redeclares `conception` with its own rate expression. One transition, so 54 modifiers become one expression of `contra.eff` (an `ss.attr` by method state, 01 §3).
- **Sexual activity.** As an expected-value multiplier `p_active(age, since_birth)`: exact for conceptions, loses per-woman activity (aCPR). The STIsim form, `where=structuredsexual.has_partner`, is agent-only (a network degree).
- **Fecundity by age and parity**: `ss.by(table, dims=['age', 'parity'])`; `parity` is a dimension when `bins=` is given (§2.6). Personal fecundity is a trait `ss.attr(ss.uniform(.7, 1.1))`: mean in an ODE with a note, or `bins=3`.
- **ASFR targeting** (v3): the rate is read at the birth year and rescaled to non-pregnant fecund women. It is a count expression, so it runs per group in compartments. It hits ASFR only when losses are 0 (as v3).

### 2.3 Outcomes

| Outcome | Declared as | Mother goes to | Classified by |
|---|---|---|---|
| live birth | `delivery` + `birth` with ≥1 child | `NP` | — |
| stillbirth | `stillbirth` hazard, or fetal removal at GA ≥ threshold | `NP` | `P.time` |
| miscarriage | `miscarriage` hazard, or fetal removal at GA < threshold | `NP` | `P.time` |
| abortion | `abortion`, immediate branch on entering `P` (FPsim v3) | `NP`, same step | — |
| maternal death | `antepartum`, `intrapartum` | `ss.null` | — |

- **A fetal removal is a loss.** Agents: when the last fetus of a pregnancy is removed (a disease's `Cstill >> ss.null`, a death hazard on the fetus), the engine moves the mother along `miscarriage` or `stillbirth` by `P.time` (a forced move, so both are also events). Removed at `preg.delivery` → `stillbirth`. One twin removed → the pregnancy continues.
- **Mother dies while pregnant** (any cause): every fetus is removed and counted as a loss by GA. `p_survive_maternal_death` (v3) is dropped (§9 Q5).
- Abortion as an immediate branch needs one clarification of 03 §2.2: **an immediate `p=` branch never takes a driven sibling as its remainder**; its complement stays in `P` and faces the timed and hazard exits. With that, the README's "event with no state change" row is closed without `ss.outcome` destinations.
- Results: one flow per outcome; `res.preg.stillbirth.by('cause')` is out of scope (§9 Q3).

### 2.4 Births: `ss.birth(per=, n=)`

```python
birth = ss.birth(per=delivery, n=embryos, sex=ss.bernoulli(0.5), into=None)
```

- `per=`: a transition (or branch, `delivery.to(NP)`). `n=`: per-event count, an attr or a number. `into=`: states of declared modules for the child (default each module's first state; with fetus agents, they keep the states they have).
- **Agents, `fetus=False`**: at the delivery, `n` children are grown, `parent` set, placed through `Agents.place` (so `S.enter` fires, 01 §8), age 0. **`fetus=True`**: the fetus agents made at conception (age = −dur) are revealed: age set to 0, prenatal edges removed, `birth` fires on them. Either way the flow `birth` counts children and fires `preg.birth` (child) and `preg.live_birth` (mother).
- **Compartments**: an inflow into the newborn groups (youngest age, sex split, each module's first state or the fetal level, §3) at rate `Σ_level n · delivery flow`. Exact.
- Placement order inside the cascade (03 §3): `delivery` (mother; fetal removals now are losses) → `ss.on(preg.delivery)` → `birth` → `ss.on(preg.birth)` (removals now are neonatal deaths) → `live_birth`, `intrapartum`.
- Any declared module may use it (FPsim's port without `ss.Pregnancy`, an animal model). `ss.null >> X @ k*P2.n` is no longer needed.

### 2.5 The fetus: agent or not

`fetus='auto'` makes fetus agents at conception if any of: `ss.mother(...)` is referenced, a `PrenatalNet` is in the sim, a delivery or transition names `preg.delivery` on a disease's state, or a product traces `maternalnet`. Otherwise children are made at delivery. `describe()` prints which and why:

```text
preg   fetus: agents from conception   [auto: ss.mother in syph.mtct; maternalnet]
```

- FPsim: no fetus agents (they are 10% of agents at any time, and nothing reads them).
- `ss.mother(expr)` (03 §2.9) is 0 for a born agent. `ss.child(event, under=None)` is an event on the mother: one of her children went through `event` this step (`under=`: the child was younger than that). Agents: `people.parent` of the event's uids. Compartments: §3.

### 2.6 After the birth: clocks, counters, parity

```python
since_birth   = ss.clock(start=live_birth)                                  # FPsim: activity, spacing, LAM, pp decisions
n_births      = ss.attr(embryos, on=live_birth, combine='sum', init=0)
age_first     = ss.attr(ss.people.age, on=live_birth, combine='first')
last_interval = ss.attr(since_birth, on=live_birth)                          # read before the clock restarts (01: recorded = pre-move)
postpartum    = since_birth < dur_postpartum
preg.stratify(parity=ss.attr(1, on=delivery, combine='sum', init=0, bins=[0, 1, 2, 3, 4]))   # ODE: a dimension, top level capped
```

- A recorded attribute reads the values before the event, and clocks restart after it; that is what makes `last_interval` the birth interval. (A rule to state in 01 §3.)
- `since_birth` is an event clock, so a later miscarriage does not reset it (the port's `NP` chain did, NOTES §5).
- Parity moved on delivery is 01's `combine='sum'` with `bins=`: a transition between bins on each event. Twins: `combine='sum'` of `embryos` (2-level dimension over `P`) moves +2 from that level. Exact. This replaces 10 `deliver_*` transitions and `move=dict(parity='i -> i+k')`.

### 2.7 Lactation: `ss.Lactation`

```python
class Lactation(ss.Module):
    N, L  = ss.states('N L')
    start = ss.Transition([N, L], [L, ss.stay], p=[p_breastfeed, None], rate=ss.on(preg.live_birth))   # L -> L restarts L.time
    wean  = L >> N @ dur_breastfeed                                          # lognormal, v3
    stop  = L >> N @ ss.on(preg.infant_death)                                # v3 FPsim handle_loss (next step there; same step here)
```

- A sibling module, not a dimension, because lactation overlaps the next pregnancy and needs a non-exponential dwell. `lactation='auto'` adds it (named `lactation`) when a `BreastfeedingNet`, `lact.*` or a LAM modifier is present.
- **LAM**: `ss.modify(preg.conception, multiply=1 - lam_eff*ss.by(since_birth=p_lam), where=lact.L & (preg.since_birth < ss.months(6)))`. v3 draws LAM per woman per month; the expected value is exact for conceptions.
- Compartments: ×2 groups; `start` and `stop` are synchronized flows with the pregnancy module (03 §2.3).

### 2.8 Networks

| Network | v3 | v4 declared |
|---|---|---|
| `MaternalNet` (`PrenatalNet`) | edges added in `Pregnancy.step` | edges on `preg.conception` (fetus agents), removed on `preg.delivery`/loss. Compartments: the mother↔fetal-level route (exact) |
| `BreastfeedingNet` | reads `pregnancy.breastfeeding` | edges on `preg.birth` if the mother is in `lact.L`; ends when she leaves `L` or either dies. Compartments: §3 (approx.) |
| `PostnatalNet(dur=)` | edges on delivery | edges on `preg.birth`, fixed duration. Compartments: as breastfeeding |

Routes keep their names, so `beta=dict(maternal=..., breastfeeding=...)` and STIsim's `beta_m2c`/`beta_breastfeed` routing (`validate_beta`) are unchanged.

### 2.9 Extending: subclass or compose

- **Subclass to change the reproductive model** (FPsim). Redeclaring a name in a subclass body replaces the base member; `describe()` marks it `[overrides ss.Pregnancy]`. Adding states (e.g. `INF` for primary infertility as a state) is allowed; removing `NP`/`P`/`delivery`/`birth` is `E215`.
- **Compose to use it** (STIsim, HIVsim, HPVsim). Diseases reference `ss.mother`, `ss.on(preg.*)`; interventions use `on=preg.*` and `preg.P.time` in `eligible=`. `sti.Sim(demographics='zimbabwe')` keeps building `ss.Pregnancy(fertility_rate=asfr, pars=preg_pars)`.
- `preg` is the lazy handle to the sim's pregnancy module (like `ss.people`): `ss.pregnancy` resolves at init to the one `ss.Pregnancy`; two raise `E215`.

## 3. Compartmental realization

| Piece | Realization | Exact? |
|---|---|---|
| `P`, `P.time`, trimesters, GA thresholds | conveyor of slots (03 §2.6) | exact on the step grid |
| `delivery` dwell | per-slot discrete hazard | exact on the grid |
| `embryos` | 2-level dimension over `P`, split at conception | exact |
| twins' fetal states | one fetal level per pregnancy | **approx.**: twins concordant (both infected or neither) |
| fetus | `fetal_<disease>` level of `P` per disease that `ss.mother` or `maternal` reaches; levels = the disease states unborn people can occupy | exact given the pregnancy |
| `ss.mother(X)` | the mother's own coordinate in the same compartment | exact |
| delivery routing | delivery flow carries the fetal level; `ss.on(preg.delivery)` outcomes leave; the rest enter the newborn groups via `birth` | exact |
| `birth` | inflow = Σ levels `n · delivery flow` | exact |
| `live_birth` (`ss.child(birth)`) | delivery flow × P(≥1 live child \| fetal level) | exact |
| `infant_death` (`ss.child(died, under=1y)`) | mother-side hazard = age-0 death hazard × (1 + p_twin), on mothers with `since_birth < 1 y` | **approx.**: infant death independent of the mother's state; twins as one |
| `since_birth` | clock dimension over everyone; levels from comparisons and `ss.by(since_birth=)` | exact at one level per step; grouped levels read mid-level, with a note |
| `parity`, `n_births` | `bins=` dimension, moved on the event | exact for the bins |
| `age_first`, `last_interval` | recorded attr, bins or E804 | exact for the bins |
| `BreastfeedingNet` | mass action between `lact.L` mothers (weighted by `rel_trans`) and breastfed children (a `bf` level entered at birth with `p_breastfeed`, left at the weaning rate) | **approx.**: pairing lost (an infant's mother is a random lactating mother); weaning exponential on the child side |
| maternal death of a pregnant woman | the fetal level dies with her compartment | exact |
| `fertility` (ASFR rescale) | count expression per group | exact |

`describe()` prints every approximated row as a note, e.g.:
```text
preg  [ode] P: conveyor of 10 monthly slots; embryos 2 levels; fetal_syph 5 levels (S, Cnnd, Cstill, Ccs, Cnorm)
      note: twins share one fetal level (concordant)
      note: infant_death on the mother approximated as a hazard of 0.031/y on mothers within 1 y of a live birth
```

Cost per age×sex group: `NP + 10 slots × 2 embryos × Π fetal levels`. Syphilis: 1 + 100 = 101 mother-side groups for the pregnancy module crossed with the mother's own syphilis states. That is where W309 bites; §9 Q1.

## 4. Back-compat

- The v3 class is kept as **`ss.LegacyPregnancy`**, unchanged (FPsim v3's `FPmod` changes one import; everything else runs as now). Agents only; `E804` under a compartmental solver, as now.
- `ss.Pregnancy(...)` with v3 pars builds the declared module. It also exposes agent-side aliases read by STIsim v3 code: `pregnant` (= `P`), `ti_pregnant` (= `ti_P`), `ti_delivery` (= scheduled exit of `P` by `delivery`), `gestation` (= `P.time` in weeks), `parity`, `breastfeeding` (= `lact.L`), plus the v3 result names (`pregnancies`, `births`, `miscarriages`, `stillbirths`, `maternal_deaths`, `neonatal_deaths`, `asfr`, `tfr`). Writes to the aliases raise.
- `add_conception_callback`/`add_delivery_callback` on the declared class: `ss.func(fn, on=preg.conception)` (01 §4), with a deprecation message.
- `ss.LegacyPregnancy` fires the same events (`conception`, `delivery`, `birth`, `live_birth`, `loss`) into `engine.events` at the points it already has (`make_pregnancies`, `process_delivery`, `process_prenatal_deaths`), so `ss.mother`, `ss.on(preg.*)` and `on=preg.*` work with it as agents.
- Newborns and the unborn from either class are placed by `place_new()` (6e10bfe1). Unborn agents go in each module's first state; under rule 3 they are not counted until born.

## 5. Worked examples

### 5.1 FPsim: postpartum chain, conception routes, contraception pause/resume

Before (`fpsim/port.py:119, 145-152, 166-171, 245-259, 305-309`):

```python
NP, P1, P2, PA, PB, PC, PD, PE = ss.states('NP P1 P2 PA PB PC PD PE') # Not pregnant, trimester 1, 2-3, postpartum bands
conceive = dict(
    conceive    = NP >> P1 @ rate('NP'),
    conceive_pa = PA >> P1 @ rate('PA'),
    ...                                                       # pb, pc, pd, pe
)
    pp1  = ss.Transition(PA, PB, dur=ss.constant(ss.months(1)), **st),
    ...                                                       # pp6, pp18, pp39
    pp51 = ss.Transition(PE, NP, dur=ss.constant(ss.months(12)), **st),
    births = ss.null >> (0.95*NP + 0.05*INF) @ ss.peryear(births_per_p2)*P2.n, # v3 p_infertile = 0.05
tr['pause'] = ss.Transition([none, dec0, wait6, dec6] + meth, paused, rate=BIG, where=pregnant) # v3 make_pregnancies: method = 0
tr['resume'] = ss.Transition(paused, dec0, rate=BIG, where=fp.NP | fp.INF) # Pregnancy lost: choose again next step (v3 handle_loss)
tr['go_pp1'] = ss.Transition(paused, dec1, rate=BIG, where=fp.PA) # Seen the step of delivery; decided the step after, as v3 (ti_delivery + 1)
tr['go_pp6'] = ss.Transition(wait6, dec6, rate=BIG, where=fp.PC)
for key, src, pk, rows, dflt in [('choose', dec0, 'pp0', rows0, none), ('pp1', dec1, 'pp1', rows1, wait6), ('pp6', dec6, 'pp6', rows6, none)]:
    tr[f'{key}_none'] = ss.Transition(src, dflt, p=ss.by(1 - pu[pk], dims=['age', 'ever_used']), rate=BIG) # rate=BIG: a p= branch with no rate crashes (REVIEW §6)
    ...
for route in ['NP', 'PA', 'PB', 'PC', 'PD', 'PE']:
    for m in METHODS: # 54 modifiers: the method's effect depends on age, parity and route (see make_fertility)
        mods.append(ss.modify(getattr(fp, tname), multiply=ss.by(fp.F.mult[(route, m)], dims=['age', 'parity']), where=getattr(contra, m), ...))
```

After:

```python
B = ss.Pregnancy                                                          # a class body does not see the base's names
class FPmod(B):
    contra_eff, lact = ss.state('contra.eff'), ss.state('lactation.L')      # path references (README)
    sb   = B.since_birth                                                    # levels: monthly to 18 m, 3-monthly to 54 m
    act  = ss.where(sb < ss.months(18), ss.by(since_birth=sa_pp), ss.by(age=sa))   # activity by month postpartum, else by age
    lam  = lam_eff * ss.by(since_birth=lam_rate) * lact                     # 0 past 6 months by the table
    fec_personal = ss.attr(ss.uniform(0.7, 1.1))
    raw  = fec_age * fec_personal * ss.by(exp_parity, dims=['parity']) * (1 - lam) * (1 - contra_eff)
    conception  = ss.Transition(B.NP, B.P, rate=-np.log1p(-act*p_month(raw))/ss.months(1), where=B.fecund & ~ss.state('contra.pre'))
    miscarriage = ss.Transition(B.P, B.NP, rate=ss.by(age=h_mc), before=ss.weeks(13))   # v3: spread over trimester 1
    stillbirth  = ss.Transition(B.P, B.NP, rate=ss.by(age=h_sb), after=ss.weeks(13))
    abortion    = ss.Transition(B.P, B.NP, p=pab)                                       # same step, as v3
    # n_births, age_first, last_interval: as in §2.6
fp = FPmod(location='kenya', fetus=False)

# contra module: junction states dec0/dec1/dec6 and BIG are gone; each decision is one triggered split
pause  = ss.Transition([none, wait6] + meth, paused, rate=ss.on(preg.conception))
resume = ss.Transition(paused, [none] + meth, p=pp0_split, rate=ss.on(preg.loss))                    # loss: miscarriage | stillbirth | abortion | infant death
pp1    = ss.Transition(paused, [wait6] + meth, p=pp1_split, rate=ss.on(preg.NP & (preg.since_birth >= ss.months(1))))
pp6    = ss.Transition(wait6, [none] + meth, p=pp6_split, rate=ss.on(preg.NP & (preg.since_birth >= ss.months(6))))
eff    = ss.attr({pill: .94, iud: .99, inj: .983, ...})                                              # derived, by state; read by FPmod
```

- 9 reproductive states → 2; 6 conception routes and 54 modifiers → 1 transition; 10 `deliver_*` → `delivery` + `parity` attr; the births inflow is `ss.birth(per=delivery)` (in the base). TFR's three suspected causes in NOTES (personal fecundity averaged, band-averaged activity, a loss resetting time since birth) are gone as agents.
- `ss.where` is `np.where` in expressions (01 §6). `p_month` must stay lazy-safe (`np.minimum`, no `np.asarray`).
- Losses at 13–20 weeks: v3 calls them miscarriages by `loss_threshold`. Here the hazard switches at 13 weeks and the label at `loss_threshold` would need a third transition (`stillbirth_early`, `after=13wk, before=20wk`) counted as `miscarriage` by result alias. Or keep two hazards and set `loss_threshold=13wk`. §9 Q4.
- Contraception decisions read age at the decision (split at the moment of the trigger, 03 §1.2), and happen in the delivery step plus 1 month, not one step later.
- **Compartments**: `NP` + `P` (10 slots × 2 embryos) × `sb` levels (18 + 12 + never = 31) × parity 5 × age 11 × contra 16. ~570k groups: W309, and FPsim's ODE needs `bins=ss.months([0, 1, 6, 18, 39, 51])` on `sb` (7 levels, read mid-level: the port's band averages, now stated) → ~130k, still too many crossed with `contra`. The port's one-module rewrite (contra as a dimension, 7.9k compartments) remains the ODE form; this proposal gives it the same `delivery`/`birth`/`sb` objects. §9 Q1.

### 5.2 Syphilis: congenital outcomes on the pregnancy events

Before (`syphilis/port.py:87-96, 142-145, 157-173`):

```python
def before_delivery(sim, uids):
    """ Due one step before the mother delivers, so the fetal death is a fetal loss (v3 issue #588) ... """
    preg = sim.demographics.pregnancy
    mom = np.asarray(sim.people.parent[uids])
    ...
    due[ok] |= np.asarray(sim.ti >= ti_del - 1)
    return BIG*due
stillbirth = Cstill >> ss.null @ before_delivery
nnd        = ss.progression('Cnnd ->', rate=BIG, where=ppl.age >= 0)
born_cs    = ss.progression('Ccs -> CS', rate=BIG, where=ppl.age >= 0)
def place_newcomers(self):
    """ Agents made by ss.Pregnancy/ss.Births are in no state of a declared disease: put them in S (see NOTES) """
```

After (disease side as 03 §4.2; pregnancy side here):

```python
mtct = ss.Transition(S, [Cnnd, Cstill, Ccs, Cnorm], rate=beta_m2c * ss.mother(Pv | Ph | S2 | EL | LL),
                     p=[by_stage(.25, .05, 0), by_stage(.25, .10, .10), by_stage(.45, .40, .10), None])
stillbirth = Cstill >> ss.null @ ss.on(preg.delivery)     # fetal removal at delivery -> preg.stillbirth for the mother (§2.3)
nnd        = Cnnd >> ss.null @ ss.on(preg.birth)          # removal after birth -> neonatal death
born_cs    = Ccs >> CS @ ss.on(preg.birth)
born_ok    = Cnorm >> S @ ss.on(preg.birth)
fetal_eff  = ss.where(ss.mother(preg.P.time) > ss.weeks(27), 0.75, 0.98)   # was sim.people.age.raw > -0.25
sim = ss.Sim(syph, networks=[sti.StructuredSexual(), ss.MaternalNet()], demographics=[ss.Pregnancy(fertility_rate=...), ss.Deaths(...)])
```

- `before_delivery`, `BIG`, `ppl = ss.People(1)`, `place_newcomers` and the `step_state` override are gone. A congenital stillbirth now also counts in `res.preg.stillbirth` and moves the mother through the `stillbirth` event (FPsim, ANC and contraception see it).
- `fetus='auto'` → agents from conception (reason: `ss.mother` in `mtct`).
- Compartments: `fetal_syph` = (S, Cnnd, Cstill, Ccs, Cnorm); `mtct` is a flow between fetal levels at a rate from the mother's coordinate; `stillbirth` leaves at delivery; the others enter the newborn groups in `CS` or `S` via the cascade. Fetal treatment (`trace='maternalnet'`) is a forced flow `fetal_cure` on the fetal levels of mothers moved by `cure` (synchronized). Exact except twins.

### 5.3 HIV: vertical and breastfeeding transmission

Before (`hiv/port.py:59, 127-131, 289-294`):

```python
beta = dict(mf=[0.035, 0.0175], maternal=0.3, breastfeeding=0.06)   # per year; MF pair is [m2f, f2m]
# BUG WORKAROUND: agents added by ss.Pregnancy (newborns, and unborn agents at conception) are put in NO state
# of a declared disease -- only ss.birth() places people. Put them in the first state by hand.
orphans = (self.sim.people.alive & ~self.S & ~self.infected).uids
...
demog = [ss.Pregnancy(fertility_rate=asfr, metadata=...), ss.Deaths(...)]
if maternal:
    nets += [ss.MaternalNet(), ss.BreastfeedingNet()]
```

After:

```python
beta = dict(mf=[0.035, 0.0175], maternal=0.3, breastfeeding=0.06)   # unchanged
demog = [ss.Pregnancy(fertility_rate=asfr, metadata=...), ss.Deaths(...)]     # lactation='auto' adds ss.Lactation (BreastfeedingNet present)
nets += [ss.MaternalNet(), ss.BreastfeedingNet()]
pmtct = ss.treat(art, eligible=hiv.diagnosed & (preg.P | lact.L), name='pmtct')        # option B+: mother-side, so both routes
mom_dx = ss.attr(ss.mother(hiv.diagnosed), on=preg.birth)              # recorded just before birth, while ss.mother is defined (§9 Q2)
infant_ppx = ss.modify(hiv.rel_sus, multiply=1 - eff_nvp, where=(ss.people.age < ss.weeks(6)) & (mom_dx > 0))
```

- The orphan workaround is gone (6e10bfe1 plus rule 4). Newborns of infected mothers infected in utero are born in `acute` through the fetus agent's own states.
- PMTCT is ART for pregnant and lactating women: one delivery with a lazy eligibility. No condition on a network partner is needed (questions-oct12 #17). Infant prophylaxis needs the mother's state at birth: a recorded attr on the child (§9 Q2).
- Compartments: `fetal_hiv` = (S, acute) on `P`; `maternal` is a flow S → acute in the fetal level at `beta_m · rel_trans(mother's coordinate)`; delivery routes into newborn `S`/`acute`. Breastfeeding is mass action (§3), with a note. HIV then runs both ways once CD4 (01 §7.4) and the ART target (02) do.

### 5.4 ANC: a test at a visit some months after conception

Before (`syphilis/port.py:185-206`):

```python
class ANCVisits(ss.Intervention):
    """ One ANC visit per pregnancy, 1-8 months after conception, for visit_prob of pregnancies (v3 ANCTest scheduling) """
    def __init__(self, visit_prob=0.85, **kwargs):
        ...
        self.timing = ss.randint(1, 9)
        self.define_states(ss.FloatArr('ti_visit'))
    def step(self):
        preg = self.sim.demographics.pregnancy
        new = (preg.ti_pregnant == self.ti).uids
        go = self.visit_prob.filter(new)
        self.ti_visit[go] = self.ti + self.timing.rvs(go)
anc = ss.screen(dual, on=lambda sim: sim.interventions.anc_visits.ti_visit == sim.ti, on_positive=ss.treat(bpg, capacity=capacity), start=2005, name='anc')
```

After:

```python
anc_ga = ss.attr(ss.randint(ss.months(1), ss.months(9)), on=preg.conception)          # per pregnancy
anc = ss.screen(dual, eligible=preg.P & (preg.P.time >= anc_ga), coverage=0.85, on_positive=ss.treat(bpg, capacity=capacity), start=2005)
# first visit at conception:          ss.screen(dual, on=preg.conception, coverage=0.85, ...)
# visits at 12, 20, 28, 36 weeks:     one delivery per contact, eligible=preg.P & (preg.P.time >= ss.weeks(w)); or 02 repeat=ss.weeks(8)
```

- Routine with bare coverage is once per eligibility spell (02 decision 2), and the spell is the pregnancy from GA `anc_ga` to its end: one offer per pregnancy, re-offered next pregnancy (v4 today: once per life). Women who lose the pregnancy before `anc_ga` are never offered, as in v3.
- Agents: `anc_ga` drawn at conception; eligibility is a lazy condition. Compartments: `anc_ga` with `bins=8` is a dimension over `P` (×8), exact. Cheaper and the same in distribution: `eligible=preg.P`, `coverage=ss.by(P_time=h_visit)` with the per-slot visit hazard, once per spell.
- No `delay=` on deliveries: the delay is a clock condition (§8).

## 6. Spec and errors

- Module kind `pregnancy` (a `demographics` subtype) with the fields of a disease, plus `fetus` (`"agents"`/`"level"`/`"none"`, derived) and the `birth` inflow `{"per": "delivery", "n": "embryos"}`.
- Expression heads: `["child", event, under]`; `["mother", expr]` (03).
- Results: v3 names as aliases of flows (`pregnancies`→`conception`, `births`→`birth`, `maternal_deaths`→`antepartum + intrapartum`, `neonatal_deaths` = removals within 28 days of `birth`); `asfr`, `tfr` derived from `delivery.by('age')`.

| Code | Meaning |
|---|---|
| `E215 PregnancyStructure` | no pregnancy module but `ss.mother`/`ss.child`/`preg.*` used; two pregnancy modules; a subclass removed `NP`/`P`/`delivery`/`birth` |
| `E804 CapabilityError` | `ss.LegacyPregnancy` under a compartmental solver; `ss.child` of an event other than `birth` or `died` |
| `W309` | as now, for fetal levels × conveyor slots |

## 7. Implementation notes (pointers, read-only survey of `v4-rs`)

- **Placement.** `Agents.place` / `place_new` (6e10bfe1) already put grown agents in each module's first state. `ss.birth(per=)` calls `place` with `newborn=True` and the parent's place; fetus agents call it at conception with `newborn=False` (age < 0). Rule 3 is a mask `age >= 0` in `step_counts`.
- **Inflow per flow.** Inflows are per-capita or absolute today (`transitions.py:846-900`, `per_capita`, `migration`). `per=` adds a third kind whose amount is a flow's count this step; the compartmental builder already has flows as expressions (`ss.outcome(delay=)`'s convolution uses them).
- **Fetal level.** A dimension over `P` built from the reaching diseases' state lists; `ss.mother(X)` compiles to the mother's coordinate in the joint compartment (the same `dim_map` machinery as `stratify(states=)`; sparse maps are a prerequisite, README).
- **Forced loss.** On removal of the last fetus, `engine.force(mother, miscarriage|stillbirth)` by `ti - ti_P` against the threshold. Twins: count living fetuses per mother (`find_unborn_children`).
- **Legacy events.** `ss.LegacyPregnancy` writes `engine.events['conception'|'delivery'|...]` where it now calls callbacks (`demographics.py:1026, 808, 1092`).
- **Clocks and attrs** are 01's; `since_birth` is an event clock (`ss.clock(start=tr)`); BreastfeedingNet reads `lact.L` in `end_pairs` instead of `pregnancy.breastfeeding` (`networks.py:1832`).

## 8. Rejected

- **Postpartum as a chain of states** (the port's `PA..PE`). Six conception routes, band averages, and a loss resets time since birth. A clock is one object and survives losses.
- **Trimesters as states** (`P1`, `P2`, `P3` with fixed dwells). Three states, the delivery dwell split across them, and GA thresholds that are not trimester edges (20 weeks) need more states. The conveyor gives every edge.
- **Lactation as a dimension of `preg`.** Dimension transfers are exponential (syphilis NOTES §8), and lactation needs a lognormal and a restart on the next birth.
- **The fetus always an agent** (v3). FPsim makes ~10% extra agents that nothing reads; `fetus='auto'` costs nothing to the models that need them.
- **The fetus never an agent** (fetal levels only, also for agents). Fetal treatment by tracing, per-fetus outcomes and twins need individuals; it is also what v3 users have.
- **`delay=` on deliveries** (syphilis NOTES §10). A per-pregnancy GA attribute plus `P.time` does the same, is exact in compartments, and keeps one waiting mechanism (clocks).
- **`ss.outcome('abortion')` as a destination.** An immediate branch `P >> NP` is a real move, fires an event (contraception resumes on it), and needs no new kind of destination.
- **`ss.congenital(...)`** (syphilis NOTES §9). As 03 §7: with `ss.mother` and `ss.on(preg.*)` congenital syphilis is 6 lines.
- **Callbacks on the declared class** (`add_delivery_callback`). `ss.func(fn, on=preg.delivery)` is the same with one concept (01 §4).
- **Infant as a level of the lactating mother** (exact breastfeeding pairing in compartments). Children already exist as newborn compartments, so the infant's state would be counted twice or need a synchronized flow per breastfeeding infection. Kept as §9 Q2.

## 9. Open questions

1. **Compartmental size.** 10 conveyor slots × 2 embryos × fetal levels × the mother's own disease states is ~100 groups per stratum for syphilis, and FPsim crossed with `contra` is out of reach. Default the conveyor to Erlang stages for `P` when no GA threshold is used (`stages='auto'`)? Drop `embryos` in ODEs (twins as `n=1 + p_twin`) with a note?
2. **The mother's state after birth.** Infant prophylaxis and STIsim's "newborns of ANC-positive mothers" (syphilis NOTES §12) need it. `ss.mother(expr, while_=lact.L)`, a recorded attr on the child (`ss.attr(ss.mother(hiv.diagnosed), on=preg.birth)`), or `ss.mother` defined for all children with `under=`? The attr form needs nothing new and is binnable: preferred?
3. **Loss causes.** Should a fetal removal record the transition that removed it (`res.preg.stillbirth.by('cause')`: syphilis, maternal death, background)? Cheap as agents; a dimension of the flow in compartments.
4. **GA threshold vs hazard switch.** v3 FPsim switches hazards at 13 weeks but labels at 20. Is the third transition acceptable, or should `before=`/`after=` transitions take a `label=` per GA band?
5. **`p_survive_maternal_death`** (v3: an unborn child may survive the mother's death). Dropped here; keep as a `birth` branch on `antepartum` (a perimortem delivery)?
6. **Preterm birth** (`rr_ptb`, `rel_ptb`, preterm flags). `ss.modify(preg.delivery, multiply=rr_ptb)` plus `preterm = ss.attr(preg.P.time < ss.weeks(37), on=preg.birth)` on the child: enough, or keep v3's sort-by-risk assignment?
7. **`ss.child` scope.** Only `birth` and `died` events in compartments (E804 otherwise). Is `under=` the right qualifier, or `ss.child(event) & (child_age < 1y)` as a general child-side expression?
8. **Names.** `ss.LegacyPregnancy` vs keeping `ss.Pregnancy` for v3 through 4.0 and naming the declared one `ss.Gestation`; `since_birth` vs `since_live_birth`; `ss.child`; `ss.pregnancy` (the lazy handle) vs `preg` by convention.
9. **Burn-in.** v3 seeds pregnancies before the start (`burnin=True`). Here `init=dict(P=ss.spread(...))` (01 §5) fills the conveyor; parity and `since_birth` at the start need a split by age from data (FPsim v3 bug 6). Is `ss.spread` on the whole pregnancy module (stationary under the ASFR) the default?

## 10. Coordination

| Item | Owner | This proposal assumes |
|---|---|---|
| immediate branch never takes a driven sibling as remainder | 03 §2.2 | stated here (§2.3) |
| recorded attr reads pre-event values; clocks restart after | 01 §3 | `last_interval` |
| `ss.on` on events of another module (synchronized flows) | 03 §2.3 | `pause`, `stop`, outcomes |
| once per eligibility spell | 02 decision 2 | ANC per pregnancy |
| newborn placement | 6e10bfe1 (`v4-rs`) | done; rule 3 still to add |
| sparse `dim_map`s | README (implementation bug) | fetal levels |
