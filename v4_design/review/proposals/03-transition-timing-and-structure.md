# 03 — Transition timing and structure

Status: proposal, not reviewed. Scope: REVIEW.md §2a rows 7, 8, 10, 11, 12, §3 rows 1–4 and 10, §5 items 3, 8, 10. Builds on spec 03, 05, 07, 10, 12, 14 and questions-oct12 #7, #8, #11.

Siblings, referenced rather than repeated:
- **01-time-and-within-host-state.md**: `X.time` (time in state), `disease.time_infected`, `on_enter=`/`on_exit=`, public per-step events (`tr.uids`). This proposal uses `X.time` only at the moment someone leaves `X`.
- **02-programs-and-treatment.md**: how interventions drive undriven edges (`ss.Treatment(to=)`, outcomes per source, `on_fail=`), queues, ANC scheduling. This proposal only makes undriven edges legal.

## 0. Summary

Every workaround in the ports is fake structure telling the engine **when** someone moves or **whether** they move. The fix is a closed list of drivers, plus one rule for **when a split is decided**.

| Need | Today | Proposal | Agents | Compartments |
|---|---|---|---|---|
| Branch on the dwell just done | zero-dwell junction, +1 dt each | one transition, several destinations, `p=` reads `P.time` | split drawn at exit | conditional tracks (Erlang) or a conveyor (exact on the grid) |
| "The rest stay" | `stay = A >> B @ 0`, `p=1-p` | `p` summing below 1 means the rest stay; `ss.stay` says it outright | no exit scheduled | a track with no exit |
| Move as soon as a condition holds | `rate=1e4`, `where=` | `@ ss.on(cond)` | same step, cascaded | compartments where `src & cond` vanish; inflow rerouted |
| Move when an event happens | BIG hazard after `ti_delivery` | `@ ss.on(event)` | same step | synchronized flow in the joint space |
| Intervention-only edge | `@ 0` | `A >> B` (no driver) | moved only by deliveries/code | a flow with no base rate |
| Fixed delay | `ss.constant` as ≤50 Erlang stages | conveyor realization (`stages='step'`) | unchanged | ring buffer of cohorts, exact |
| Minimum stay / escape window | not expressible | `after=`, `before=` on a transition | defer / cancel at exit | state split at the threshold |
| Dwell distribution by sex | one transition per sex | `ss.by` inside a Dist, or a Dist inside `ss.by` | per-agent parameters | per-level stages (ragged) |
| Mother's state, delivery | 6 fetal states, 5 lookups, BIG | `ss.mother(expr)`, `preg.delivery`, `preg.birth` | read via `people.parent` | fetal level of the mother's pregnancy compartments |
| Lazy condition before a sim | `ppl = ss.People(1)` | `ss.people` | — | — |

New names: `ss.on`, `ss.stay`, `ss.mother`, `ss.people`, `after=`, `before=`, `stages='step'`, `preg.conception/delivery/birth`. No new transition class, no new keyword for split timing.

## 1. Rules

1. **A driver says when; `p=` says where.** The drivers are: a rate (`@ r`), a dwell (`@ dur`), a trigger (`@ ss.on(x)`), or none. `p=` alone (no driver) is a split that happens on entry, in the same step.
2. **When a split is decided follows from its shape**, as the implementation already does (see §6):
   - several transitions out of one state, each with `p=`: **split on entry**, then each branch waits its own dwell (branch-then-wait; v3's "decide fate at infection").
   - one transition with several destinations: **split when it is taken** (wait-then-branch). Its `p=` is evaluated at that moment and may read the source's `X.time`, which is then the dwell just completed.
   - For a constant `p` and exponential dwells the two are the same model. `describe()` prints which one each split is.
3. **Moves without a wait happen in the step that caused them**, under every solver. Immediate splits, triggered moves, and their consequences cascade within the step (§3). This also fixes "immediate branches happen a step late".
4. **Nothing is approximated silently.** Every compartmental realization below is exact, or prints a one-line note naming what was matched.

## 2. Primitives

### 2.1 Split at exit, reading the dwell

```python
p_cin = logf2(P.time.years * sev, k)                          # P.time: the dwell just completed
precin = ss.Transition(P, [C, S, L], dur=dur_precin, p=[p_cin, (1 - p_cin)*(1 - ctrl), None])
# operator form
precin = P >> (p_cin*C + (1 - p_cin)*(1 - ctrl)*S + ss.rest*L) @ dur_precin
```

- `None` in the list (or `ss.rest` in the weighted form) is the remainder. Probabilities are clipped to [0, 1]; a negative remainder is clipped with `W503`.
- `P.time` (01) in a `p=` of a transition out of `P` is the time actually spent in `P`, including any rescaling by modifiers during the dwell. Elsewhere `X.time` is 01's business.
- **Agents.** Dwell sampled on entry, as now. At exit, `p` is evaluated for the leavers and one draw picks the destination (`finish_moves`). `ti_P` is still set, so `P.time = ti - ti_P`.
- **Compartments, default: conditional tracks.** Wait-then-branch with dwell density `f` and branch `j` taken with `p_j(t)` has the same joint law of (branch, dwell) as branch-then-wait with `q_j = ∫ f p_j` and dwell density `f p_j / q_j`. So the build computes `q_j` and the conditional mean and std by quadrature (per group, if `p` depends on strata), makes one track per destination, and gives each its own Erlang stages. Competing hazards (death, cancellation) act on every track; the probability of reaching each exit is then the same as for agents. **Approximated:** each conditional dwell is matched in mean and std only, and anything in `p` other than `X.time` is read at entry. Printed:
  `note: precin splits at exit by P.time; 3 conditional tracks (C: q=0.21, 6 Erlang stages, mean 7.9 y; S: ...)`
- **Compartments, exact on the grid: `stages='step'`.** The state is a conveyor (§2.6) of time-in-state slots, the exit hazard of slot `i` is the dwell's discrete hazard, and leavers from slot `i` split with `p_j(t_i)`. This is "branch probability per stage", made exact by making a stage one step. Cost: `max_dwell/dt` compartments per group (tail past the 99.9% quantile lumped into an exponential last slot, with a note).
- **Modifiers.** `ss.modify(precin, multiply=m)` scales the rate (divides the dwell), as now. `ss.modify(precin.to(C), multiply=m)` scales one branch; the others share the rest (`W503` as for `p=` branches). `adjust=` on `stratify` the same.
- **Results.** One flow, `precin`; `.select(to='C')` per destination (not implemented yet: REVIEW §6). The public event set (01) takes the same selector: `hpv.precin.uids(to=S)`.
- A function of the sim works in `p=` as anywhere. Under a compartmental solver it cannot be decomposed, so the try-it rule stops it with `E804` unless it reads only strata.

### 2.2 The rest stay

```python
tertiary = ss.Transition(LL, T, p=0.35, dur=ss.normal(ss.years(18), ss.years(2)))   # the other 65% stay  [guessed]
tertiary = ss.Transition(LL, [T, ss.stay], p=[0.35, None], dur=...)                  # the same, said outright
```

- Branch probabilities out of a state that sum to less than 1, with no transition left to take the remainder, mean **the rest stay**. `describe()` prints `the other 65% stay [guessed]`. A sum above 1 is still `E205`.
- `ss.stay` as a destination removes the guess. In a split at exit it means the move is cancelled: no re-entry, no new dwell, `X.time` keeps running, and they never take this transition again in this stay.
- **Agents.** No exit scheduled by the branch set; competing hazards and interventions still apply.
- **Compartments.** One more track with no exit flow of its own. Exact.
- **Results.** No flow for staying. `n_LL` sums the tracks.
- The remainder rule (spec 03: "one transition with no `p=`") applies only to a sibling **with a driver** (`dur=` or `rate=`), and works when the other `p` are not constants (today it needs constants: `transitions.py:1470`). A sibling with no `p=` and no driver is undriven (§2.4), not the remainder. This is what lets `cure` leave `LL` while `tertiary` splits it.

### 2.3 Triggered moves: `ss.on`

```python
debut     = U >> (p1*CST1 + p3*CST3 + p4*CST4p + ss.rest*CST4a) @ ss.on(ss.people.female & (ss.people.age >= 15))
cancelled = (S | P | C | L) >> Z @ ss.on(other_cancer)
born_cs   = Ccs >> CS @ ss.on(preg.birth)
ss.progression('U -> CST1 | CST3', on=adult_f, p=[...])          # string form: on= is rate=ss.on(...)
```

`ss.on(x)` takes what `on=` takes on a delivery (spec 08): a condition (lazy, a boolean state, or a function of the sim) or an event (a transition of any module, `preg.delivery`, or `ss.mother(event)`).

- **Semantics.** A condition is level-triggered: anyone in the source for whom it holds moves, in the step it first holds, including on entry. An event is edge-triggered: people it happened to this step move. `where=` on the same transition is ANDed into the condition.
- **Agents.** Resolved in the cascade (§3), so in the same step as whatever made the condition true, whatever the module order.
- **Compartments.**
  - Condition on states and strata (sex, age bins, other diseases' states under joint tracking): the compartments of `src` where the condition holds **vanish**. Every flow into them (aging from 10–14 into 15+, a progression in another disease) is routed straight to the destination, through the same pass-through the engine already uses for immediate tracks (`update_targets.enter()`). Exact.
  - Condition on time or counts (`sim.t >= 2025`, `I.n > 100`): evaluated at loop times, and the mass of `src & cond` is moved as one shift, as forced moves are. Printed: `note: debut is triggered by a count; applied at loop times`.
  - Event: the event's flow in the joint space also carries this module's coordinate from `src` to `dst` (a synchronized flow). Exact. Needs the two modules tracked jointly (spec 07 §On agents and on groups), so `describe()` prints the joint size and `W309` may fire.
- **Modifiers.** A triggered move has no rate to scale. `ss.modify(debut, ...)` and `adjust={'debut': ...}` raise `E504` ("use where= or a p= branch"). Branch multipliers (`debut.to(CST1)`) work.
- **Results.** A flow like any other; `.by()` works.
- Replaces every `rate=ss.perday(1)`, `rate=BIG`, `ss.peryear(1e3)` + `where=` in the ports, and the `rate × dt = 30` note they trigger.

`ss.people` is a module-level lazy handle to the sim's people (`ss.people.female`, `.age`, `.parent`), resolved at init. It replaces `ppl = ss.People(1)` in BV, syphilis and bacterial (implementation.md, known issue).

### 2.4 Undriven edges

```python
cure = (E | Pv | Ph | S2 | EL | LL | T) >> Sp          # no @: only interventions or code move people along it
```

- A transition with no rate, dwell, trigger, or `p=` is **undriven**. It replaces `@ 0` (which stays legal and means a rate of 0 that a modifier could raise).
- `E209` becomes `W209 UndrivenTransition` at `sim.init()`, raised only if no product, delivery or `on=` names the transition. It is a warning because custom `step()` code may move people along it.
- **Agents.** Nothing scheduled. `engine.force()` (via a product) moves people; the flow counts them.
- **Compartments.** A flow with base rate 0, driven by the delivery (02).
- **Modifiers.** `ss.modify(cure, multiply=...)` raises `E504`: there is nothing to scale. Efficacy belongs to the product (02).
- `describe()`: `cure  E|Pv|Ph|S2|EL|LL|T -> Sp  undriven: moved by bpg, fetal_tx`.

### 2.5 Same-step immediacy for `p=` splits

`ss.Transition(C, [Cnnd, Cstill, Ccs, Cnorm], p=[...])` with no driver: split on entry, **same step**. Today it crashes (`Cannot evaluate None`, REVIEW §6), and the separate-transition form is a step late. Both are fixed by the cascade (§3). Compartments already pass these straight through. Ports should rarely need them once §2.1 exists, since junctions were the main use.

### 2.6 Fixed delays: the conveyor

```python
latent = E >> I @ ss.constant(ss.days(14))                      # conveyor by default in compartments
recover = I >> R @ ss.lognorm_ex(ss.days(10), ss.days(2)), stages='step'   # any dwell, exact on the grid
```

- **Agents.** Unchanged (sampled; exits rounded to the nearest step).
- **Compartments.** The state is `n = round(d/dt)` slot compartments per group. Between loop times each slot is an ordinary compartment: competing hazards, deaths, modifiers and aging act on it (aging moves a cohort to the same slot of the next age group). At each loop time every slot shifts by one, and the last one exits. This is questions-oct12 #7. It is what `ss.outcome(delay=)` (convolution, `derived_outcomes`) and test `delay=` (`pending_<delivery>` levels) already do for results and deliveries.
  - Exact on the step grid; the delay is rounded to whole steps with a note when it is not a multiple of `dt`, the same rounding agents make.
  - Gillespie/tau: a cohort's completion is a scheduled event (delay SSA). Binomial: a shift. JAX: a fixed permutation, differentiable.
  - Cost: `n` compartments per group, linear, against Erlang's quadratic run time and its 50-stage cap (`erlang_stages(max_stages=50)`, undocumented). odin's 14-day latency becomes 14 slots, exact.
- **Defaults.** `ss.constant` (or any dwell with CV < 0.1) uses the conveyor; everything else uses Erlang as now. `stages=k` (int) and `stages='step'` override. Printed: `note: latent has a fixed dwell of 14 days, realized as a conveyor of 14 steps (exact)`.
- **History before the start** (#8): `init=dict(E=ss.spread(100))` fills the slots as if arrivals had been constant; default is slot 0, as now. Agents: entry times spread over the dwell.
- The conveyor is also the compartmental form of time in state, which 01 needs for `X.time` thresholds, and of §2.1 and §2.7. One mechanism, three uses.

### 2.7 Minimum stays and escape windows: `after=`, `before=`

```python
ss.Transition(A, B, rate=r, after=ss.days(7))      # cannot happen in the first 7 days in A
ss.Transition(A, C, dur=d, before=ss.days(30))     # only within the first 30 days; later draws are cancelled
```

- `before=d` is `where=A.time < d`; `after=d` on a hazard is `where=A.time >= d`. Sugar, since 01 provides `A.time`.
- `after=d` on a dwell **defers**: the exit is at `max(T, d)`. This is EMULSION's `duration` (other exits held back until it is over), and is why it is a keyword and not a `where=`.
- **Agents.** Hazards gated per step; dwells deferred or cancelled when they fall due.
- **Compartments.** The source is split at `d` into a conveyor (if `d` is a constant) or Erlang stages (with a note), and the gated transitions leave only from the part they are allowed in. Exact with the conveyor.
- **Modifiers.** Scale the hazard or dwell; `after`/`before` are parameters, so a modifier can target them by path.

### 2.8 Distributions by group

```python
dur = ss.lognorm_ex(mean=ss.by(sex=[ss.weeks(1), ss.weeks(0.25)]), std=ss.by(sex=[ss.weeks(12), ss.weeks(1)]))
dur = ss.by(sex=[ss.lognorm_ex(ss.weeks(1), ss.weeks(12)), ss.lognorm_ex(ss.weeks(0.25), ss.weeks(1))])   # the same
p   = ss.by(sex=[0.20, 0.54])
```

- Any Dist parameter accepts `ss.by`, and `ss.by` accepts Dists (of one family or several). Both normalize to one spec entry: a Dist per level.
- `sex` is `people.female` (fixes REVIEW §1.3). As agents, `ss.by(sex=...)` needs no stratification. In a compartmental run it implies `stratify(sex=...)` on the sim, added with `[guessed]`; any other undeclared dimension is `E1102`.
- **Agents.** Parameters looked up per agent's level, one draw.
- **Compartments.** Each level gets its own Erlang `k` (ragged stages, like `where=` strata). Exact for the per-level mean; std as for any Erlang match.
- **Modifiers and `adjust=`** multiply the rate per level, as now. A CV that differs by sex is now expressible (it was not with `adjust=`).
- `ss.by` supports arithmetic with numbers and other `ss.by` (`1 - ss.by(...)`, also asked for by the AMR port).

### 2.9 Pregnancy and the mother

```python
ss.mother(Pv | S2 | EL)            # 1 for an unborn agent whose mother is in these states, else 0
ss.mother(syph.rel_trans)          # any expression, evaluated on the mother
preg.conception, preg.delivery, preg.birth                     # events of the pregnancy module
```

- `ss.mother(expr)` evaluates `expr` on the agent's mother **while the agent is unborn**, and is 0 otherwise. It is an expression, not a state, so `beta * ss.mother(X)` is a spontaneous rate (no route, no `rel_trans` unless asked for), which is what v3 does for in-utero syphilis. It works in rates, `p=`, `where=`, `ss.on()`, and modifier conditions (`where=ss.mother(anc_treated)`).
- Events: `preg.conception` (mother), `preg.delivery` (mother and fetus, **before** the child is born: a removal now is a fetal loss), `preg.birth` (child, just after birth: a removal now is a neonatal death). Usable in `ss.on()`, and in `on=` on deliveries (02: ANC on conception).
- **Agents.** `people.parent` lookup; the fetus is an agent in the disease's states from conception (needs REVIEW §1.1 fixed: fetuses and newborns placed in the first state).
- **Compartments.** Needs a declared pregnancy (states, gestation as a conveyor). Unborn people are not compartments: the disease's states that unborn people can occupy become a `fetal_<disease>` level of the mother's pregnant compartments, `ss.mother(X)` is the mother's own state, and `ss.on(preg.delivery/birth)` routes the delivery flow by fetal level (stillbirths out, live births into the newborn state). Exact given the pregnancy model. With the Starsim-3 `ss.Pregnancy` module: agents only (`E804`, as now).
- No `ss.congenital` class (see §7).

## 3. Same-step cascade

Agents, inside one step:

```
row 4/5/8/9 of any module moves people  ->  finish_moves()
  -> for every transition whose trigger reads a state, transition or event that just changed:
       resolve it for the people affected (all modules, not only this one)
  -> people who entered a state with an immediate exit (p= only, or ss.on true on entry): resolve now
  -> repeat until nothing moves; more than n_states rounds raises E212 TriggerLoop (names the cycle)
```

- What a trigger depends on is known from its expression (states, transitions, events it names), so only those are re-checked; functions of the sim are re-checked once per step at the owning module's row 5.
- Deaths are still deferred (row 10): someone requested to die can still be cancelled or moved.
- Compartments: all of this is instantaneous by construction (vanishing compartments, synchronized flows), so the cascade is what makes agents agree with the ODE.

## 4. Worked examples

### 4.1 HPV: CIN and cancer from the dwell (`hpvsim_disease/port.py:83–120`)

Before:

```python
S, P, Pe, C, Ce, X, L, Z = ss.states('S P Pe C Ce X L Z')
def pcin(sim, uids):
    hpv = sim.diseases[g]
    dur = (hpv.ti_Pe[uids] - hpv.ti_P[uids])*sim.t.dt_year
    return p_cin_of(dur, rel_sev(sim, uids), gp['k'])
def pcancer(sim, uids): ...                                    # the same, from ti_Ce - ti_C
...
    precin        = ss.Transition(P, Pe, dur=lognorm(*gp['dur_precin']), where=is_female),
    clear_male    = ss.Transition(P, S,  dur=lognorm(*sp['dur_inf_male']), where=is_male),
    to_cin        = ss.Transition(Pe, C, p=pcin),
    clear_precin  = ss.Transition(Pe, S, p=lambda sim, uids: (1 - pcin(sim, uids))*(1 - ctrl)),
    latent_precin = ss.Transition(Pe, L, p=lambda sim, uids: (1 - pcin(sim, uids))*ctrl),
    cin           = ss.Transition(C, Ce, dur=lognorm(*gp['dur_cin'])),
    to_cancer     = ss.Transition(Ce, X, p=pcancer),
    clear_cin     = ss.Transition(Ce, S, p=lambda sim, uids: (1 - pcancer(sim, uids))*(1 - ctrl)),
    latent_cin    = ss.Transition(Ce, L, p=lambda sim, uids: (1 - pcancer(sim, uids))*ctrl),
```

After:

```python
S, P, C, X, L, Z = ss.states('S P C X L Z')                 # no junctions
female = ss.people.female
sev    = cross.rel_sev * hiv_sev                             # lazy per-agent expressions (01); a fn(sim) also works
p_cin  = female * logf2(P.time.years * sev, gp['k'])         # men never reach CIN
p_canc = 1 - (1 - gp['transform_prob'])**(intlogf2(C.time.years * sev, gp['k'])**2)
c_f    = ctrl * female                                       # v4 port: men clear to S only

precin = ss.Transition(P, [C, S, L], p=[p_cin, (1 - p_cin)*(1 - c_f), None],
                       dur=ss.by(sex=[lognorm(*gp['dur_precin']), lognorm(*sp['dur_inf_male'])]))
cin    = ss.Transition(C, [X, S, L], p=[p_canc, (1 - p_canc)*(1 - ctrl), None], dur=lognorm(*gp['dur_cin']))
```

- 9 transitions and 2 fake states → 2 transitions. The +0.5 y bias on every cancer path (two junctions at dt=0.25) is gone, and so are `n_Pe`, `n_Ce`, the `ti_*` reads, and the three unchecked functions per split.
- The `where=` on dwells (and their divide-by-zero warnings) are gone: sex is in the dwell.
- Results: `to_cin` → `res.hpv16.precin.select(to='C')`; `CrossImmunity` reads `hpv.precin.uids(to=S) | hpv.cin.uids(to=S)` (01) instead of `engine.events`.
- `intlogf2` must avoid `np.asarray` to stay lazy (or be passed as `ss.par(fn)`).
- Compartments: `sev` is a per-agent draw (`rel_sev_dist`), so the model runs as agents unless `rel_sev` is binned into a dimension. If it is, each group gets 3 conditional tracks out of `P` and 3 out of `C`, or exact conveyors with `stages='step'` (dt=0.25 y, `dur_cin` tail cut at 99.9%: about 200 slots).

`describe()`:

```text
precin   P -> C | S | L   dwell ~ by sex: lognormal(mean 3, std 9 y) | lognormal(1, 1 y)
                          split at exit: C female*logf2(P.time*sev, 0.3); S (1-C)*(1-ctrl*female); L the rest
         [ode] 3 tracks by dwell (C: q=0.18, 4 stages, mean 6.9 y; S: q=0.74, ...; L: ...)
```

### 4.2 Syphilis: "the rest stay", cure, congenital outcomes (`syphilis/port.py:14, 47–97, 125–147`)

Before (stays and treatment):

```python
tertiary   = ss.Transition(LL, T, p=0.35, dur=ss.normal(ss.years(18), ss.years(2)))
stay_LL    = ss.Transition(LL, T, p=0.65, rate=ss.peryear(0)) # "The other 65% stay": p= must sum to 1 (E205)...
tert_death = ss.Transition(T, ss.null, p=0.05, dur=ss.lognorm_ex(ss.years(5), ss.years(5)))
stay_T     = ss.Transition(T, ss.null, p=0.95, rate=ss.peryear(0))
cure = (E | Pv | Ph | S2 | EL | LL | T) >> Sp @ ss.peryear(0)
trep_revert = ss.Transition(Sp, Sr, p=0.2, dur=ss.lognorm_ex(ss.years(10), ss.years(5)))
stay_Sp     = ss.Transition(Sp, Sr, p=0.8, rate=ss.peryear(0))
```

After:

```python
tertiary    = ss.Transition(LL, T, p=0.35, dur=ss.normal(ss.years(18), ss.years(2)))    # the rest stay
tert_death  = ss.Transition(T, ss.null, p=0.05, dur=ss.lognorm_ex(ss.years(5), ss.years(5)))
cure        = (E | Pv | Ph | S2 | EL | LL | T) >> Sp                                     # undriven
trep_revert = ss.Transition(Sp, Sr, p=0.2, dur=ss.lognorm_ex(ss.years(10), ss.years(5)))
```

Before (congenital, abridged; full text at the lines above):

```python
ppl = ss.People(1)
BIG = 1e4
def mother_stage(sim, uids): ...      # people.parent lookup -> 0 active, 1 early, 2 late, -1 none
def mtct_rate(sim, uids): ...         # beta_m2c if unborn and mother in Pv|Ph|S2|EL|LL
def outcome_p(i): ...                 # table[mother_stage, i], normal as remainder
def before_delivery(sim, uids): ...   # BIG once sim.ti >= mother's ti_delivery - 1
C, Cnnd, Cstill, Ccs, Cnorm, CS = ss.states(6)
mtct    = S >> C @ mtct_rate
c_nnd   = ss.Transition(C, Cnnd,   p=outcome_p(0))
c_still = ss.Transition(C, Cstill, p=outcome_p(1))
c_cs    = ss.Transition(C, Ccs,    p=outcome_p(2))
c_norm  = ss.Transition(C, Cnorm,  p=outcome_p(3))
stillbirth = Cstill >> ss.null @ before_delivery
nnd        = ss.progression('Cnnd ->', rate=BIG, where=ppl.age >= 0)
born_cs    = ss.progression('Ccs -> CS', rate=BIG, where=ppl.age >= 0)
born_ok    = ss.progression('Cnorm -> S', rate=BIG, where=ppl.age >= 0)
fetal_cure = (Cnnd | Cstill | Ccs) >> Cnorm @ ss.peryear(0)
cs_cure    = CS >> S @ ss.peryear(0)
```

After:

```python
Cnnd, Cstill, Ccs, Cnorm, CS = ss.states(5)                  # C is gone: the outcome is drawn at infection
active, late = E | Pv | Ph | S2, LL | T
def by_stage(a, e, l):                                        # v3 birth_outcomes table, by mother's stage at infection
    return a*ss.mother(active) + e*ss.mother(EL) + l*ss.mother(late)

mtct = ss.Transition(S, [Cnnd, Cstill, Ccs, Cnorm], rate=beta_m2c * ss.mother(Pv | Ph | S2 | EL | LL),
                     p=[by_stage(.25, .05, 0), by_stage(.25, .10, .10), by_stage(.45, .40, .10), None])
stillbirth = Cstill >> ss.null @ ss.on(preg.delivery)         # fetal loss, at delivery (v3 #588)
nnd        = Cnnd >> ss.null @ ss.on(preg.birth)              # born, then dies
born_cs    = Ccs >> CS @ ss.on(preg.birth)
born_ok    = Cnorm >> S @ ss.on(preg.birth)
fetal_cure = (Cnnd | Cstill | Ccs) >> Cnorm                   # undriven: fetal_tx (02)
cs_cure    = CS >> S                                          # undriven: newborn_tx (02)
```

- 3 fake `stay_*` branches, 4 `@ peryear(0)`, the `C` state, 5 lookup functions, `BIG`, and `ss.People(1)` are gone.
- The outcome uses the mother's stage at infection (v3), because `mtct` is one transition taken at infection (§1.2). For the stage at delivery instead, keep `C` and split there: `ss.Transition(C, [...], p=[...]) @ ss.on(preg.delivery)`.
- A fetus infected in the step before delivery is now born in the right outcome state, in the same step (the port notes that it was a step late).
- Compartments (with a declared pregnancy): the mother's pregnant compartments get a `fetal_syph` level (S, Cnnd, Cstill, Ccs, Cnorm); `mtct` is a flow between levels from the mother's own stage; delivery routes by level. Runs as an ODE once `StructuredSexual` is approximated or replaced.

### 4.3 BV: gating to women 15+ (`bv_gud_connectors/port.py:14–58`)

Before:

```python
ppl = ss.People(1)  # Only so that people's attributes can be used in lazy conditions before the sim exists
adult_f = ppl.female & (ppl.age >= 15)
debut = ss.progression('U -> CST1 | CST3 | CST4p | CST4a', rate=ss.perday(1), where=adult_f,
                       p=[by_stable(1, 0, 0), by_stable(0, 1, 0), by_stable(0, 0, 0.1), None])
tx = (CST4a | CST4p | CST4s) >> P @ 0
```

After:

```python
adult_f = ss.people.female & (ss.people.age >= 15)
debut = ss.progression('U -> CST1 | CST3 | CST4p | CST4a', on=adult_f,
                       p=[by_stable(1, 0, 0), by_stable(0, 1, 0), by_stable(0, 0, 0.1), None])
tx = (CST4a | CST4p | CST4s) >> P
```

- Same step as the 15th birthday; no `rate × dt = 30` note; girls already 15+ at the start (or seeded) move at init.
- `recur_rate` reading `ti_P` becomes `rate=recur0 * np.exp(-P.time.months / recur_tau)` (01).
- Compartments: with `sim.stratify(age=[0, 15, ...])`, sex (implied) and `stable`, the compartments `(U, female, 15+, *)` vanish; aging out of `(U, female, 10–14)` lands in the CST states split by `by_stable`. Exact. The `place_newborns` connector goes away with REVIEW §1.1, not with this proposal.

`describe()`:

```text
debut   U -> CST1 | CST3 | CST4p | CST4a   on female & age >= 15 (same step); split by stable
        [ode] (U, female, 15+) is empty: arrivals go straight to the CST states
```

### 4.4 CT/NG/TV: sex-specific durations (`stisim_bacterial/port.py:77–147`)

Before:

```python
def by_sex(f, m):
    return lambda sim: np.where(sim.people.female, f, m)
def sexed(src, dst, p, durs, prefix):
    return {
        f'{prefix}_f': ss.Transition(src, dst, p=by_sex(p[0], 0.0), dur=lognorm(durs[0])),
        f'{prefix}_m': ss.Transition(src, dst, p=by_sex(0.0, p[1]), dur=lognorm(durs[1])),
    }
...
    trs['latency']   = ss.Transition(E, P, p=by_sex(ps[0], ps[1]))
    trs['latency_a'] = ss.Transition(E, A, p=by_sex(1 - ps[0], 1 - ps[1]))
    'onset_f': ss.Transition(P, Y, dur=lognorm(p['dur_presymp'][0]), where=ppl.female),
    'onset_m': ss.Transition(P, Y, dur=lognorm(p['dur_presymp'][1]), where=~ppl.female),
    trs['pid_a'] = ss.Transition(A, PID, p=by_sex(pf, 0.0), dur=lognorm(p['dur_prepid']))
    trs.update(sexed(A, S, [q*(1 - p['p_persist']), 1.0], p['dur_asymp2clear'], 'clear_a'))
    trs['pid_y'] = ss.Transition(Y, PID, p=by_sex(pf, 0.0), dur=lognorm(p['dur_prepid']))
    trs.update(sexed(Y, C, [q*pc[0], pc[1]], p['dur_symp2care'], 'seek'))
    trs.update(sexed(Y, S, [q*(1 - pc[0]), 1 - pc[1]], p['dur_symp2clear'], 'clear_y'))
    trs.update(sexed(N, S, [1.0, 1.0], p['dur_symp2clear'], 'clear_n'))
    trs['treat'] = (E | A | P | Y | C | N | PID) >> T @ ss.permonth(0)
# plus sync_sex(sim) after every sim.init()
```

After:

```python
def lognorm(ms):              # [mean, std], or [[mean_f, std_f], [mean_m, std_m]]
    if np.ndim(ms) == 1:
        return ss.lognorm_ex(mean=ms[0], std=ms[1])
    return ss.lognorm_ex(mean=ss.by(sex=[ms[0][0], ms[1][0]]), std=ss.by(sex=[ms[0][1], ms[1][1]]))

pid = ss.by(sex=[pf, 0])
trs = dict(
    latency = ss.Transition(E, [P, A], p=ss.by(sex=p['p_symp']), dur=ss.constant(p['dur_exp'])),  # or no dur: same step
    onset   = P >> Y @ lognorm(p['dur_presymp']),
    pid_a   = ss.Transition(A, PID, p=pid, dur=lognorm(p['dur_prepid'])),
    persist = ss.Transition(A, S, p=ss.by(sex=[(1 - pf)*p['p_persist'], 0]), dur=ss.constant(p['dur_persist'])),
    clear_a = ss.Transition(A, S, dur=lognorm(p['dur_asymp2clear'])),                  # the remainder
    pid_y   = ss.Transition(Y, PID, p=pid, dur=lognorm(p['dur_prepid'])),
    seek    = ss.Transition(Y, C, p=ss.by(sex=[(1 - pf)*pc[0], pc[1]]), dur=lognorm(p['dur_symp2care'])),
    clear_y = ss.Transition(Y, S, dur=lognorm(p['dur_symp2clear'])),                   # the remainder
    clear_n = N >> S @ lognorm(p['dur_symp2clear']),
    treat   = (E | A | P | Y | C | N | PID) >> T,                                     # undriven; 02 removes T
)
```

- `by_sex`, `sexed`, `sync_sex` and 8 duplicated transitions are gone; one flow per event, `.by('sex')` for the split. `res.ct.latency.select(to='P')` replaces the onset-based workaround.
- `(1 - pf)` is still multiplied by hand: a flat three-way split from one state is joint probabilities, and that is the model. A nested split (`Y -> PID | (C | S)`) is not proposed.
- Compartments: sex implied (§2.8); each sex level gets its own Erlang `k` per dwell. CV 12 (women, presymptomatic) gets one stage with the note that its extra variance is lost; `stages='step'` keeps it.
- `persist` with `ss.constant(1200)` months is a 1,200-slot conveyor at monthly dt; use `rate=ss.peryear(1/100)` unless exactness matters, or the default threshold (§8.4) should not apply to dwells this long.

### 4.5 HPV: cancel the other genotypes on cancer (`hpvsim_disease/port.py:105, 123`)

Before:

```python
other_cancer = lambda sim: np.any([d.X.raw for name,d in sim.diseases.items() if name in GENOTYPES and name != g], axis=0)
cancelled = ss.Transition([S, P, C, L], Z, rate=ss.peryear(1e3), where=other_cancer) # v3 _cancel_other_genotype_progression_for
```

After:

```python
other_cancer = functools.reduce(operator.or_, [ss.state(f'{o}.X') for o in GENOTYPES if o != g])
cancelled = (S | P | C | L) >> Z @ ss.on(other_cancer)
```

- `ss.state('hpv18.X')`: a lazy reference by path, resolved at init, for factories that build one module before its siblings exist (§8.9). The lambda also still works.
- Agents: same step as the other genotype's `cin -> X`, whatever the order of the genotypes in `diseases=`. Today it is the next step, and a lagging genotype can progress for one step.
- Compartments: the 4 genotypes are tracked jointly (`(8 states)^4` per group, `W309`); joint compartments with one genotype in `X` and another in `S|P|C|L` vanish, and the `cin -> X` flow carries the others to `Z` with it. Exact, and big: this model is agents in practice.

## 5. Spec and errors

Transition object (spec 14), new fields:

| Field | Holds |
|---|---|
| `trigger` | the `ss.on()` condition or event expression, or `null` |
| `after`, `before` | durations, or `null` |
| `split` | `"entry"` or `"exit"`; derived, written for the reader, recomputed on load |
| `driven` | `false` for an undriven edge; derived |

`ss.stay` is `"dst": [..., "stay"]`; the implicit stay is listed in `guessed`. `stages` accepts `"step"`. Dists hold `by` values per parameter. `["mother", expr]` and `["time", "P"]` are new expression heads; `["event", "preg.delivery"]` in `trigger`.

| Code | Change |
|---|---|
| `E205 BadBranchSplit` | only for a sum above 1, or a length mismatch |
| `E209` → `W209 UndrivenTransition` | at `sim.init()`, only when nothing names the transition |
| `E212 TriggerLoop` | new: the cascade did not settle |
| `E504 UnmodifiableTarget` | also for a rate modifier on a triggered or undriven transition |
| `E804 CapabilityError` | `p=` at exit reads something other than `X.time` and strata; `ss.mother` without a declared pregnancy |

## 6. Implementation notes (read-only survey of `v4-rs`)

- **Exit splits already exist.** `Graph.resolve_exits` (`transitions.py:1458`) files a multi-destination transition under `timed`/`hazards`, and `Agents.finish_moves` (`solvers.py:3455`) draws its destination when people leave, with `p` evaluated then; the compartmental `update_targets` (`solvers.py:2053`) splits the exit flow. Only the spec text ("decided on entry", spec 03 §Branching) and the dwell access are missing. `ti_<src>` is still set in `finish_moves`, so `P.time` at exit is `ti - ti_P`.
- **Same step.** `step_state` (`solvers.py:3574`) schedules at the start of the step, so people who enter during `finish_moves` wait a step. The cascade is a loop around `schedule` + due exits for newly entered agents with immediate or triggered exits, plus a subscriber list for triggers that read other modules.
- **Vanishing compartments.** `update_targets.enter()` (`solvers.py:2029`) already passes arrivals through immediate tracks, with a loop guard. Triggers on strata add a per-group mask to that pass-through.
- **Stay.** A track with `tr=None` and no exit flow in `build_layout` (`solvers.py:1758`).
- **Conveyor.** A new compartment kind with a shift at loop times in `step_ode`/`step_binomial`/`step_tau`/`step_gillespie`; the kernel already handles a fixed program per step. Removes the cap in `erlang_stages(max_stages=50)` (`transitions.py:1063`) for constant dwells.
- **Remainder with non-constant `p`** (`transitions.py:1470`) and `p=None` (`E209`) are being fixed on `v4-rs` now; §2.2's "sibling with a driver" rule should be kept in that fix.
- **Dists with `ss.by`.** `dwell_stats` (`transitions.py:1009`) and `numeric_pars` need per-level values; the Erlang layout needs per-level `k`.

## 7. Rejected

- **`decide='exit'` keyword** (REVIEW §3). The two shapes already behave this way in both engines; a keyword would allow a third, contradictory spelling. `describe()` prints which one applies.
- **`p=fn(sim, uids, dwell)`.** A third argument for one slot. `X.time` (01) is the same information, lazy, and usable in rates and conditions too.
- **Junction states** (zero-dwell pseudo-compartments). Spec 03 already rejects them; the ports used them only for want of §2.1.
- **Coxian per-stage branching as the default ODE form of exit splits.** It changes the dwell distribution to fit the branch. Conditional tracks keep the dwell; the conveyor is the exact version of per-stage branching.
- **More Erlang stages for fixed delays.** Error 6% at 50 stages, quadratic run time (questions #7).
- **`ss.at(time)`** for events at a per-agent absolute time. Every case in the ports is a dwell from entry, an event (`preg.delivery`), or a condition on time (`ss.on(ss.t >= x)`). Revisit if a real model needs it.
- **`when=` on transitions** (REVIEW §3). On `ss.modify`, `when=` is a function of the sim with no per-person meaning; `ss.on` matches the delivery keyword `on=`, which already means "this event or condition, this step".
- **`ss.now`, `rate=ss.inf`, `dur=0`.** `@ 0` (never) next to `@ ss.days(0)` (at once) is a trap; a split with no driver, or `ss.on(...)`, covers it.
- **`ss.congenital(...)` building block.** With `ss.mother`, `ss.on(preg.*)` and exit splits, syphilis is 6 lines. Revisit after HIV and HBV, if they repeat those lines.
- **A disease-level `where=` (BV exists only in women 15+).** Saves `U` and the denominator sum, but adds a second gating concept beside `stratify(where=)`. A holding state plus `ss.on` is explicit and runs both ways. Promote later if STIsim's PID and pregnancy outcomes want the same.
- **`else_=ss.stay` keyword.** A destination is enough.

## 8. Open questions

1. **Implicit stay.** A sum below 1 by mistake becomes "the rest stay", printed as a guess. Is the guess enough, or should only `ss.stay` be accepted?
2. **Cross-module cascade cost.** Subscribers keep it cheap for lazy conditions; functions of the sim are checked once per step (an O(dt) lag across modules). Acceptable, or should functions be re-checked after every `finish_moves`?
3. **Level vs edge.** `ss.on(cond)` fires whenever `cond` holds for someone in the source, including after a treatment returns them to it. Is a separate edge form (`ss.on(ss.became(cond))`) needed? None of the ports needs it.
4. **Conveyor default.** Constant dwells only, or CV < 0.1? A cap on slots (the 1,200-month TV case), above which Erlang is used with a note?
5. **Conditional tracks with time-varying `p`.** When `p` depends on a time series or modifier, `q_j` and the conditional moments change; recompute per step (cost), or freeze at build with a note?
6. **Implied sex stratification** in compartmental runs from `ss.by(sex=...)`: guess and print, or `E1102`?
7. **`ss.mother` beyond pregnancy.** Breastfeeding is a network route today. Is "while unborn" the right limit, or `ss.mother(expr, during='breastfeeding')`?
8. **Names.** `ss.on` (vs `ss.when`), `ss.stay`, `ss.rest` (the weighted-form remainder), `stages='step'` (vs `stages=ss.conveyor`).
9. **`ss.state('hpv18.X')`** path references in conditions, for module factories. Or a sibling-group helper (REVIEW hpvsim §1: `ss.Disease.replicate`).
10. **Pregnancy events.** `preg.delivery` vs `preg.birth` needs `ss.Pregnancy` to expose both, and to record removals of unborn agents as fetal losses. Who owns the change, and does a declared pregnancy (for compartmental runs) belong in this proposal or in demography?
11. **Exit splits and interventions targeting "destined to progress".** Screening that detects people who will progress (HPV) needs branch-then-wait, which §2.1 gives up. Write it as entry splits when that matters, or expose the eventual branch as a hidden attribute?

## 9. Coordination

| Item | Owner | This proposal assumes |
|---|---|---|
| `X.time`, `disease.time_infected` | 01 | available lazily; at exit of `X` it is the dwell |
| `tr.uids(to=)`, `on_enter=`/`on_exit=` | 01 | public; `.select(to=)` on results here |
| conveyor as the compartmental form of time in state | 01 and 03 | one mechanism (§2.6) |
| driving undriven edges, outcomes per source, `on_fail=` | 02 | edges are legal and named (§2.4) |
| ANC on `preg.conception`, per-pregnancy delivery | 02 | event names from §2.9 |
| newborn and fetus placement (REVIEW §1.1), `sex` = `people.female` (§1.3) | bug fixes on `v4-rs` | prerequisites for §2.8, §2.9 |
