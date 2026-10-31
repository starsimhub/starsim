# Starsim v4: what happens when you port STIsim and HPVsim

Starsim `v4` branch at commit `002c4ec3` (4.0.0). Seven ports, each with a runnable `port.py` and a `NOTES.md`:

| Folder | Scope | v3 lines → v4 lines |
|---|---|---|
| `stisim_bacterial/` | CT, NG, TV, PID, care seeking, syndromic management, AMR | ~1,200 → ~70 + helpers |
| `syphilis/` | full staging, serology, congenital outcomes, ANC | ~1,700 → 258 |
| `hiv/` | stages, CD4, ART/VMMC/PrEP, Zimbabwe targets | ~2,000 → 238 |
| `bv_gud_connectors/` | BV community state types (CSTs), GUD, all coinfection connectors | ~800 → ~150 |
| `networks/` | StructuredSexual, sex work, MSM, partner notification | 2,100 → subclass + 40 |
| `hpvsim_disease/` | 4 genotypes, CIN/cancer, immunity, HIV | ~2,000 → ~300 |
| `hpvsim_interventions/` | vaccination, screen–triage–treat, capacity | ~1,300 → 356 |

All ports run in seconds on 2–10k agents. None is calibrated; the numbers only show the mechanics work. The two engine bugs in §1 are reproduced in `repros/`.

## 0. Bottom line

- **Natural history ports well.** Stages, latency, reactivation, competing dwell times and branch-then-wait need about 5–10× less code than v3. The `ti_*` bookkeeping, `step_die`, and ~150 lines of results per disease all disappear. Connectors as `ss.modify` are the clearest improvement in the whole design.
- **The SIR-shaped vocabulary breaks at four seams. Every model hit at least two of them:**
  1. **Time.** Anything that depends on *time since an event* (since infection, since entering a state, since treatment, since last screen) has no public expression.
  2. **Splits by person attribute.** Sex-specific *distributions*, and branches decided by an attribute known only later.
  3. **Programs.** Stock targets, priority, queues, shared capacity, per-person delivery memory, and multi-disease products.
  4. **Demography.** Agents created by `ss.Births`/`ss.Pregnancy` are invisible to declared diseases.
- **The workarounds share one shape.** They invent fake model structure to steer the engine: rate-0 transitions, 1e4 hazards, zero-dwell junction states, `stay` branches, and dummy `ss.People(1)` objects. They also read engine internals: `engine.events`, `ti_<state>`, `engine.force`, `sim.current_scale`. §3 catalogues them. Each is a symptom of a missing primitive, not a one-off.

## 1. Blocking bugs (fix before anyone ports anything)

1. **Newborns from `ss.Births`/`ss.Pregnancy` enter no state of any declared disease.** All 7 ports hit it. `repros/births.py`: 1,493 alive, only 825 in `S`/`I`. These agents can never be infected and silently drop out of counts. Unborn fetuses are also stateless, so there is no vertical transmission. They also get level 0 of every dimension instead of a draw from `split`. Only the `ss.birth()` inflow goes through `Agents.add_people()`.
2. **A per-act beta on sexual networks depends on dt and the unit.** It is converted as a rate (`1-exp(-beta*dt)`, `solvers.py:3949`) and then compounded over acts again in `net_beta`. `repros/beta_dt.py`: same model, 5 years, 29 / 76 / 99 infections at weekly / monthly / quarterly dt, and 76 vs 99 at monthly dt from changing `unit=` alone. This contradicts `06-routes.md`. The effect is small for HIV-sized betas and large for bacterial STIs.
3. **As agents, `stratify(sex=...)` creates a random `sex` attribute independent of `people.female`.** It agreed for 48% of agents, so `.by('sex')`, `ss.by(sex=...)` and anything sex-stratified are silently wrong relative to the network.
4. **Lazy conditions on a non-state array lose their module.** `hiv.cd4 < 500`, `bv.poor_hygiene` and `net.partners_12m >= 2` are all resolved against the module being *modified*: `AttributeError: Syph object has no attribute "cd4"`. This is the spec's own example, and it is the connector use case.

## 2. Your two questions

### 2a. Do `ss.transmission`/`ss.progression` generalize beyond I→R?

| Mechanism | Verdict | How |
|---|---|---|
| Latency / exposed period | clean | `E >> I @ dur` |
| Stage progression with branch-on-entry | clean | `ss.progression('E -> P \| A', p=..., dur=...)`; matches v3's "decide fate at infection" exactly |
| Clearance competing with progression | clean | two transitions out of one state; earliest dwell or competing hazard wins |
| Reactivation (syph latent → secondary, HPV latent → infected) | clean | `L >> P @ rate`; its own flow, so nothing is backed out of incidence. Best part of the HPV port |
| Reinfection / SIS | clean | clearance goes to `S` |
| Stage-specific infectiousness | clean | `ss.State(rel_trans=...)` or `beta * (P + S2 + 0.1*EL)` |
| Sequela with probability ("35% eventually tertiary, the rest stay") | **ugly** | E205 unless the probabilities sum to 1. Needs a fake `stay = LL >> T @ 0` branch (3 in syphilis) |
| Branch whose probability depends on the *dwell just completed* (HPV logf2 severity) | **ugly** | `p=` is decided on entry, before the dwell is drawn. Needs zero-dwell junction states (which the spec says aren't needed). Each costs a full timestep: +0.5 y on every cancer path at dt=0.25 |
| Rate depending on time in state / since infection (latent decay, BV recurrence, CT bacterial load) | **awkward, agent-only** | `fn(sim, uids)` reading undocumented `ti_<state>` arrays. NaN for seeded agents. No `ti_infected` across states |
| Sex-specific dwell *distributions* (v3: CV 12 for women, 4 for men) | **ugly** | `ss.by` can't hold a Dist and a Dist can't take `ss.by`. One transition per sex with hand-multiplied joint probabilities, about 20 transitions per disease |
| "Move as soon as condition holds" (BV at age 15, cancel other HPV genotypes on cancer) | **ugly** | `rate=ss.perday(1)` or `1e4` plus `where=`, plus a throwaway `ss.People(1)` |
| Event at a scheduled time (birth outcome at delivery) | **ugly** | a big hazard while `ti >= mother's ti_delivery`; 6 fetal states and 5 lookup functions |
| Treatment from any infected state | clean-ish | `(E \| A \| Y \| PID) >> S @ 0` plus `ss.Treatment(transition='cure')`. The `@ 0` is a fake rate to declare an intervention-only edge |
| Treatment with failure | **awkward** | failures are invisible unless you add a transient `T` state (a one-step lag). "Fail back to where you were" is impossible |
| Treatment with outcomes per stage (syph early → RPR reverts, late → stays) | impossible | one Treatment = one transition = one destination; `on_positive=` takes one delivery |
| Treatment that changes natural history without curing (ART) | must be a state | as a product: dropout only exponential, no post-ART state, `mortality=` hits *all* outflows incl. background death |
| Continuous per-agent state (CD4, immunity levels, dysplasia) | `step_state()` | works, and the stages stay declarative. But there is no on-entry hook, and exits are scheduled a step late (`ti_exit` NaN on entry). The CD4 code has to sit on *both* sides of `super().step_state()`; get it wrong and CD4 goes NaN, which silently selects the lowest mortality bin |
| Multiple strains with independent courses (HPV genotypes) | strain dimension can't | one natural-history state per person for all strains, 2ⁿ−1 carriage levels, binary `immunity=` that fails over networks. One disease per genotype (as v3) works, but every product and test is then per-disease (§2b) |

So `ss.transmission`/`ss.progression` generalize well to anything that is *state + dwell + branch decided at entry*. They break on (a) dependence on elapsed time, (b) decisions at exit, (c) moves triggered by conditions or schedules, and (d) "nothing happens" as an outcome.

### 2b. Count-based delivery ("treat 100/week")

`capacity=ss.perweek(100)` exists and runs. What it does and doesn't do:

| Need | Works? | Notes |
|---|---|---|
| Cap a flow per unit time | yes | VMMC from annual procedure counts is one line and matches data |
| Backlog / queue | **only by accident** | unmet demand carries over only if eligibility is a *persistent state* (`eligible=C`, with `interval=dt`). With `on=` (event-triggered) or `on_positive=` follow-ups, overflow is **silently dropped**: 60/yr vs ~240/yr with a hand-written FIFO queue |
| Priority (lowest CD4 first, symptomatic first) | **no** | `cap()` picks at random and discards the order `deliver=` returned. Reversing the priority order gave bit-identical results. HIV: mean CD4 at ART start was 527 with `capacity=` vs 388 when capped by hand |
| Target as a *stock* (N on ART from data) | **no** | `capacity=` is a flow. Fed the stock, it put 960k on ART vs a 566k target. Only `deliver=` can do it; removing people when over target needs the private `engine.force` |
| Shared capacity across deliveries (one clinic, several drugs or programs) | **no** | needs a `capacity=fn` that reads the other delivery's results, times `sim.current_scale` (mid-run results are in agents, which is undocumented) |
| Rate capacity on yearly rounds | **bug** | scaled by dt, not the round interval: `ss.peryear(1000)` with quarterly dt screens 250/yr |
| Delivered / eligible / coverage / cost results | yes | automatic, by stratum, `annualize()` works |

Related traps:
- **"Routine" delivery means once per person, ever.** That's wrong for any reinfecting disease (533 of 713 care seekers treated) and for rescreening. v3's `routine_vx` is an annual probability, which is v4 *rounds*.
- **Deliveries keep no per-person memory.** "Not screened in 5 years", dose 2 of the same vaccine and "treated before" each need a subclass. A second dose of the same product is silently refused.
- **Coverage is per round.** `coverage=ss.probperyear(0.5)` raises TypeError, so annual test rates are converted by hand.
- **Products act on one disease.** A syndromic clinic treating NG/CT/TV depends on loop order (9,250 vs 3,275 treated for the same visits). Multi-genotype vaccines, tests and ablation need fan-out code calling `Delivery` internals. `on_positive=[a, b]` crashes mid-run.

## 3. The ugly patterns (each one points at a missing primitive)

| Workaround seen | Ports | Missing primitive |
|---|---|---|
| `x >> y @ 0` (rate-0 edge so an intervention can move people) | all 7 | an undriven edge: `x >> y` with no `@`, or `ss.Treatment(to=y)` |
| `stay = A >> B @ 0` with `p=1-p` | syph, bact | "the rest stay": `p<1` with no remainder means stay |
| `rate=ss.perday(1)` / `1e4` with `where=` | BV, syph, HPV | `when=cond` on a transition: move on the step the condition becomes true |
| zero-dwell junction states | HPV | `decide='exit'`: a branch `p=fn(sim, uids, dwell)` evaluated at exit |
| `fn(sim, uids)` reading `ti_<state>`, `ti_E or ti_A` | syph, BV, CT, HIV | `X.time` (time in state) and `disease.time_infected`, usable in rates, `p=` and modifiers |
| `engine.events['clear_precin']` | HPV, HIV | a public "who went through transition X this step": `hpv.clear_precin.uids`, and `on_enter=`/`on_exit=` hooks |
| `engine.force(...)` | HIV | `remove=` on deliveries (take people off when over target) |
| `* sim.current_scale` | HIV, bact | deliveries and results in people, not agents, or a documented `to_agents()` |
| `ss.People(1)` at module level | BV, syph | a global lazy `ss.people` |
| one transition per sex, probabilities multiplied out | bact, HPV | Dist parameters accepting `ss.by(sex=...)`, with `sex` bound to `people.female` |
| 16–33 `ss.modify` modules, each with an empty result group | HPV, BV | a list target `ss.modify([a, b, c], ...)`, and no result group for modifiers |
| custom FIFO queue / ANC visit scheduler / clinic module | HPV-int, syph, bact | `queue=`, `priority=`, shared `ss.Capacity`, `on=preg.conception`, multi-disease delivery |
| `T` state for treatment failure | bact | `ss.Treatment(..., on_fail=...)` with `cured`/`failed` results |
| `priorpartners=0` in every disease's beta dict | networks | `Network(transmits=False)`, or skip all-zero networks |
| hand-inverted beta `-log(1-p)/dt` | networks, syph | fix bug §1.2 |

## 4. What is genuinely better than v3

- **Connectors.** 18 one-line `ss.modify` calls replace 6 classes. They compose exactly for 0–5 coinfections, need no `rel_sus[:] = 1` resets, and accept continuous per-agent multipliers (CD4). They can target another disease's *transitions*. `what_modifies()` prints the whole matrix.
- **Screening trees.** `ss.screen(test, on_positive=ss.screen(triage, coverage=.9, on_positive=ss.treat(...)))` reads like the WHO flowcharts, with loss to follow-up as `coverage=` at each node and `delay=` on tests. The v3 `.outcomes` lambda glue is gone.
- **Results.** A count per state, a flow per transition, delivery results, and `.by(age|sex|strain)` from one `stratify=`: no results code.
- **Serology.** States (`Sp`, `Sr`) plus an `rpr` dimension. `ss.Test(detects=syph.rpr)` is a titre test in one line, and a sensitivity-by-state dict reproduces the syph_dx_zim CSV row.
- **Gonorrhea AMR as a strain.** Inheritance from the infector, `move=` on treatment failure, and efficacy as an `ss.timeseries` inside `ss.by`, in about 10 lines.
- **Partner notification.** `ss.treat(drug, on=ng.cure, trace=['structuredsexual', 'priorpartners'])`.
- **Solver choice.** `sim.solvers()` says exactly why a model runs as agents (`cd4_hazard takes uids`).
- **Speed.** The declared HIV ran about 2× faster than v3 `sti.HIV` on the same network.
- **The v3 network API carried over almost unchanged.** StructuredSexual runs under a declared disease with a 40-line shim.

## 5. Suggested API changes, by impact

1. Fix §1.1–1.4.
2. **Time as a first-class expression:** `X.time`, `disease.time_infected`, `delivery.since`. Usable in rates, `p=`, conditions and modifier amounts; agent-only, or staged in an ODE.
3. **Branch timing:** `decide='exit'` with access to the dwell; `when=` for condition-triggered moves; `ss.at(fn)` for scheduled events; "the rest stay"; and immediate remainders (`p=None` currently raises E209, which contradicts the spec).
4. **Delivery semantics for programs:**
   - `target=` (stock) with `remove=`;
   - `priority=` honored by `capacity=`;
   - `queue=`/`max_wait=` with `waiting`/`dropped` results;
   - shared `ss.Capacity`;
   - per-person `ti_last`/`n_received`, with routine delivery once per *episode*;
   - `coverage=` accepting typed probabilities;
   - `on_positive=[...]`.
5. **Products over several diseases:** efficacy and sensitivity keyed by disease or `State`; one delivery giving one product across genotypes; `vx.held` as a lazy condition; `doses=`.
6. **Treatment:** `to=`, efficacy by source state, destination per source, `on_fail=`, and `cured`/`failed`/`overtreated` results. An explicit `infected=` set on the disease, since the guessed one silently drops non-infectious infected states.
7. **Hooks:** `on_enter=`/`on_exit=` callbacks run in `finish_moves`, and exits scheduled at entry. This is the minimum for continuous within-host state.
8. **Sex/age as Dist parameters:** `ss.lognorm_ex(mean=ss.by(sex=[...]), std=ss.by(sex=[...]))`.
9. **Networks:** move STIsim's condom logic into `ss.SexualNetwork` (`condoms` edge column plus disease `eff_condom`); add `match=` and per-group targets on `MFNet`, `acts=` on `PartnerNet`, and lookback windows on `trace=`. Make `describe()` say how beta is read.
10. **Pregnancy vocabulary:** a `ss.mother(cond)` condition, events on `preg.conception`/`delivery`, and possibly a `ss.congenital(...)` building block (HIV, syphilis and HBV all need it).

## 6. Other v4 bugs and spec mismatches (deduplicated)

- **Units.** `unit='month'` with `start=2000, stop=2015` reads them as months: the timeline starts in year 166. With `dt=ss.months(1)` and no `unit=`, bare rates are silently per month (12×).
- **Branches:**
  - A branch with `p=` and no `dur`/`rate` crashes at runtime (`Cannot evaluate None`).
  - "Immediate" branches happen a step late.
  - `p=None` remainder raises E209.
- **Results:**
  - `.select(to=)` and `.select(source=)` are not implemented.
  - Ratio results raise `ZeroDivisionError` on a zero denominator.
  - Result expressions in a class body are silently ignored.
  - A `|` set passed as a result is silently dropped.
  - A transition named `clear` is shadowed by `dict.clear`.
- **Error paths:**
  - `ss.Transition(A | B, ...)` crashes inside the error reporter (`closest()`).
  - `on=a | b` gives a misleading "add parentheses" error.
  - A per-network dict beta times a weighted state raises a raw `TypeError`.
  - A sensitivity dict keyed by another module's `State` gives a self-contradicting E505.
  - `solver='ode'` on a subclass with custom `step_state` crashes in user code instead of raising E804.
  - `sim.solvers()` claims MFNet has an ODE approximation; running it raises E804.
- **Silent wrong answers:**
  - `ss.Vaccine(efficacy=ss.by(genotype=...))` reads a stale level for susceptibles.
  - A 2-D `adjust=` raises a TypeError (although `split=` accepts the same form).
  - A `timeseries` inside a per-network beta dict is never evaluated.
  - `ss.modify(x.rel_sus)` on a non-transmitted disease is a silent no-op.
  - Negative efficacy is accepted.
- **Calling conventions:** a function with defaulted args is called as `(sim, uids)`. Declared pars are `ss.par` Exprs, so external code doing arithmetic on `disease.pars.x` breaks.
- **Noise and naming:**
  - `where=` on a `dur=` transition emits divide-by-zero warnings every step.
  - Products share a namespace with modules and people attributes (`'rpr'`, `'art'` collide).
  - `describe()` drops units on `ss.uniform`, reports Erlang stages under `abm`, and omits `adjust=`.

## 7. Bugs found in current STIsim/HPVsim (v3), independent of v4

- **`connectors/gud_syph.py` is dead code.** It defines `update()`, which is never called; reads `sim.people.gud`, which doesn't exist; and assigns rather than multiplies.
- **`treat_BV`:**
  - It never clears `infected`, so treated women stay infected forever.
  - The pregnancy branch calls a nonexistent `rel_sus_ptb`.
  - `tx_delay` and `tx_effect_delay` are never used.
- **BV onset:**
  - `p_cst_change × RRs` can exceed 1 (up to ~2.5).
  - Risk factors are re-drawn every step rather than kept as traits.
  - Newborns are flagged as infected (`set_cst(upper_age=dt)`).
- **`SyphTx.change_states` never clears `exposed`.** People treated in incubation are stuck there.
- **`set_congenital` makes every in-utero-infected fetus non-susceptible for life**, including the "normal" outcome.
- **Smaller syphilis issues:**
  - `treat_fetus` double-counts successes.
  - `immune` is declared but never used.
  - The `dur_early` comment says "since infection", but the code measures from latent onset.
- **The `max_capacity` parameter of `STITreatment`/`GonorrheaTreatment` is accepted but never used.** v3 has no capacity.
- **Gonorrhea `UpdateDrugs` reads `sim.people.rel_treat`, which doesn't exist.**
- **VMMC treats `vmmc_coverage.csv` (annual procedures, a flow) as a stock target**, so it under-circumcises after year one.
- **`sti.Sim(dt=...)` on starsim 4:** `dt` is routed as a disease/connector parameter and raises a ValueError.

These are worth issues in stisim regardless of v4.

## 8. Limits of this review

- The ports were written by agents in one pass against a moving branch. Some "impossible" items may have an idiom the tests don't show; where an agent was unsure, the NOTES say so.
- Nothing is calibrated. Speed comparisons are single runs.
- Not attempted: HPVsim multiscale; stratified (age × sex) ART targets; FSW entry/exit ages; BV sexual transmission; FPsim.
- The design doc is marked "not yet human-reviewed". Several mismatches above may be spec bugs rather than implementation bugs.
