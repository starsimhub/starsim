# Proposal 02: programs and treatment

Status: draft for Robyn and Cliff. Scope: REVIEW.md §2b, §3 rows 1, 7, 8, 9, 12, 13, §5 items 4–6. Read-only against `v4-rs` (`interventions.py`, `products.py`).

Hard requirement: one spec, two realizations. Every primitive below has an agent semantics and a compartmental semantics. Where they differ, the difference is stated, and printed at run time as a note under the usual `approx=` rules (12-backends).

## 0. Summary

| Need (from the ports) | Primitive | Where |
|---|---|---|
| intervention-only edges (`@ 0` in all 7 ports) | undriven transition `a >> b` (no `@`) | disease |
| treatment destination/efficacy per source, failure, multi-disease, cured/failed/overtreated | `ss.Treatment(moves={edge: efficacy}, on_fail=)` | product |
| vaccine/test over genotype diseases | efficacy and sensitivity keyed by disease or `State` | product |
| colposcopy, tx assigner | `ss.Test(outcomes=...)` + `on_result=` | product, delivery |
| annual probability, routine per episode, rescreen, doses | typed `coverage=`; routine = once per eligibility spell; `repeat=`; `dose=`; lazy `.since`, `.n`, `.held` | delivery, product |
| "N on ART", "X% circumcised", removal, age×sex | `target=` (stock), `remove=` | delivery |
| lowest CD4 first, symptomatic first | `priority=` | delivery |
| backlog, max wait, dropped | `queue=` (True or a max wait) | delivery |
| one clinic, several programs/drugs | `ss.Capacity(...)` passed to several `capacity=` | new object |
| a second product for the same visit | `then=` | delivery |
| `* sim.current_scale` | every number a user gives or reads is people | engine rule |

One new class (`ss.Capacity`), one new transition form, eight delivery keywords, four product keywords. Everything else is a change of default.

## 1. Rules that the primitives share

1. **A delivery is an impulse at a time point**, in both solvers (09-results: "a delivery at a time point happens there, between the steps"). Agents: who moves at the start of the step. Compartments: what fraction of each (state, group) cell moves. This is what `Delivery.give` already does via `eng.force` (`interventions.py:1031/1033`). Everything below keeps it; the one exception is `target=` under an ODE (§2.5), which may instead be a tracking flow.
2. **Numbers are people.** `capacity=`, `target=`, `deliver=` counts, results read mid-run, and `I.n` are all in people. Conversion to agents is internal, with the existing stochastic carry (`self.carry`, `interventions.py:991-993`). Results are scaled when written, not at the end. `sim.current_scale` becomes private. This removes `n/sim.current_scale` (hiv `art_target`) and `*sim.current_scale` (bact `leftover`).
3. **Memory is allocated only when referenced.** A per-agent array, or a compartmental dimension, for `since`/`n`/`waiting` exists only if a condition, `repeat=`, `dose=` or `queue=` uses it. `describe()` lists it the way it already lists `has_<product>` and `offered_<delivery>`.
4. **Compartmental memory is a staged dimension.** The machinery exists: `has_<product>` levels of time since delivery (`products.py:312-339`), and `pending_<delivery>` levels for test delays (`interventions.py:673-676`). `since`, `n`, `waiting` reuse it: one level per step where that is affordable (exact), otherwise grouped levels with a note.
5. **Selections are made once, then applied.** A delivery evaluates eligibility, coverage, priority and capacity for every product and disease it touches *before* moving anyone. That removes the syndromic loop-order artefact (9,250 vs 3,275).

## 2. Primitives

### 2.1 Undriven edges and `ss.Treatment(moves=)`

```python
class HIV(ss.Disease):
    start_art = (acute | latent | falling | post_art) >> on_art      # no @: only a delivery moves people along it
class Syph(ss.Disease):
    cure_early = (E | Pv | Ph | S2 | EL) >> Sp                       # destination by source: two edges
    cure_late  = (LL | T) >> Spl

bpg  = ss.Treatment(moves={syph.cure_early: 0.98, syph.cure_late: 0.95})
drug = ss.Treatment(moves={ng.cure: 0.95, ct.cure: 0.95})            # several diseases, one product
art  = ss.Treatment(moves=hiv.start_art)                              # efficacy 1
ss.Treatment(moves=ng.cure, on_fail=ss.transfer(strain='sens -> res', p=0.25))
```

- **Syntax.** `a >> b` with no `@` is legal and *undriven*: it has no rate, so E209 does not apply. `describe()` prints `driven by: deliveries (bpg)` and W207 if no delivery names it. `moves=` takes one edge, a list, or `{edge: efficacy}`. Efficacy is anything a parameter accepts (`ss.by(strain=[...])`, a timeseries, `fn(sim)`). `transition=` and scalar `efficacy=` stay as aliases for a single edge. A dict `efficacy=dict(transmission=..., mortality=...)` remains the held-product form (PrEP-like; *not* ART, see below).
- **Agents.** For each recipient: find the edges whose source holds their current state (in any disease the product names). With prob. efficacy, force along it (as `eng.force`). If the person is in a source state but does not move, that is a **failure**: `on_fail=` (an edge, a transfer, or `fn(sim, uids)`), default stay put. A recipient in no source state is **overtreated**. If a person matches two edges of one disease (an author error) it is E-coded at build.
- **Compartments.** For each edge and (state, group) cell: forced fraction = chosen × efficacy; the failed fraction = chosen × (1 − efficacy) goes to `on_fail` or stays. Exact.
- **Results** (per delivery, by stratum, and with a `disease` and `edge` dimension when the product names several): `delivered`, `cured`, `failed`, `overtreated`. The disease also gets its usual flow per edge (`res.ng.cure`), so the edge counts are the same numbers seen from the disease.
- **Modifiers.** An undriven edge is a transition, so `ss.modify(ng.cure, multiply=0.3, where=ng.strain == 'res')` multiplies the efficacy for those people, by the rule already implemented for `p=` branches (implementation.md 07: "it multiplies the branch's probability"). `ss.modify(bpg.efficacy, ...)` scales the product's own number.
- **ART is a state, not a product.** It changes the natural history (pauses progression, has a post-ART state with its own dwell). So it is an undriven edge into `on_art` plus declared dropout. The product form `efficacy=dict(...)` stays for PrEP/VMMC/vaccines.
- **Approximated:** nothing. **Removes:** the `T` state, its one-step lag and its "fail to A" (bact), `@ 0` everywhere, the fake "treatment that cures" wording for ART.

### 2.2 Products over several diseases

```python
bivalent = ss.Vaccine(efficacy={hpv16: 0.95, hpv18: 0.95, hi5: 0.20})            # acquisition, per disease
txvx     = ss.Vaccine(efficacy={hpv16: dict(dysplasia=0.5, invasion=0.5), hpv18: dict(dysplasia=0.4)})
hpv_dna  = ss.Test(sensitivity={hpv16.I: 0.9, hpv16.CIN: 0.95, hpv18.I: 0.9, hi5.I: 0.6}, specificity=0.97)
```

- Keys are diseases (or their names) for held products, and `State` objects (of any disease) for tests. A key naming another module's state no longer triggers E505.
- **Holding is per person, not per disease.** `has_bivalent` is already a property of people. The effect is one modifier per (disease, quantity), all keyed on that one property. `bivalent.held` is the public lazy condition (replaces `ss.Cond('people', 'has_bivalent')`).
- **Test across diseases.** P(positive) = 1 − ∏(1 − sens_k) over the keyed states the person is in (independent detection); `1 − spec` if in none. `combine='max'` is the alternative (v3's hierarchy). `detects=` keeps its meaning for a single condition.
- **Agents:** exact. **Compartments:** needs the diseases crossed (`ss.Crossing`, already implemented), because one selection then depends on several diseases' states. The library crosses them and prints the size (3 genotypes × 4 states = 64 joint states per group). No approximation, just size; `solvers()` reports it.
- **Delivery `disease=`** is no longer needed when the product names its diseases. Deliveries of such products have no "the disease" (`Delivery.dis`); `eligible=` defaults to "in any keyed source state".

### 2.3 Multi-outcome tests

```python
colpo = ss.Test(outcomes=dict(              # P(result | state); the first outcome takes the remainder
    normal = None,
    lsil   = {g.I: 0.15 for g in genos} | {g.CIN: 0.20 for g in genos},
    hsil   = {g.CIN: 0.70 for g in genos} | {g.CA: 0.10 for g in genos},
    cancer = {g.CA: 0.85 for g in genos},
), combine='worst')                         # several genotypes: the most severe (last-listed) outcome wins
assign = ss.Test(outcomes=dict(none=None, ablation={...}, excision={...}, radiation={...}), name='tx_assigner')
ss.screen(colpo, on_result=dict(hsil=assign_visit, cancer=refer))
```

- `sensitivity`/`specificity` are sugar for `outcomes=dict(negative=None, positive=...)`, and `on_positive=`/`on_negative=` for `on_result=dict(positive=..., negative=...)`.
- **Agents:** one categorical draw per person. With several diseases, draw the outcome per disease and take the worst (or the independent rule for two outcomes). **Compartments:** a multinomial split of each chosen cell. For `combine='worst'`, the per-cell probability is P(worst ≤ k) = ∏_d P_d(≤ k), on crossed diseases. Exact.
- **Results:** `diagnosed` (any outcome other than the first) and `outcome`, a flow with a `result` dimension: `res.colpo.outcome.select(result='hsil')`. `delay=` works per outcome as now (`pending_` levels get one block per outcome with a follow-up).

### 2.4 Offer occasions, coverage, and per-person memory

**When people are offered** (one of these, as now): rounds (`years=`, `interval=`), an event (`on=`), a follow-up, or routine. Two default changes:

1. **Typed coverage is a probability per unit time.** `coverage=ss.probperyear(0.3)` is converted with the actual time between offers: `dt` for routine, the interval for rounds. A bare number stays per offer. The same rule fixes the rate-capacity bug: `capacity=ss.peryear(1000)` uses the round interval, not `dt` (`interventions.py:987-988`).
2. **Routine with bare coverage means once per eligibility spell**, not once per life. Someone is offered on the first step of each spell of being eligible. `offered_<delivery>` resets when they stop being eligible. Then reinfections are re-treated (`eligible=ng.C`: 533 → 713 of 713) and each pregnancy is re-screened. `repeat=False` gives once ever. Routine with typed coverage is a per-step rate, with no `offered_` record.

**Memory.** Lazy, readable anywhere a condition or a parameter function is:

| Expression | Is | Agents | Compartments |
|---|---|---|---|
| `d.since`, `p.since` | time since last *received* from delivery `d` / product `p` (inf if never) | `ti_last` array | `since_<x>` dimension: levels of time, one step each up to the largest threshold used, then absorbing |
| `d.n`, `p.n` | number received (doses for a vaccine) | int array | `n_<x>` dimension, levels 0..k (k = largest compared value; last absorbing) |
| `p.held` | holds product `p` now | `has_<p>` | existing `has_<p>` dimension |

```python
ss.screen(hpv_dna, eligible=women(30, 50), coverage=ss.probperyear(0.3), repeat=ss.years(5))   # self-reference
ss.treat(art, eligible=hiv.diagnosed & (hpv_dna.since < ss.years(1)))                          # another product
ss.vaccinate(vx, dose=2, eligible=vx.since >= ss.months(6), coverage=0.9)
```

- `repeat=` is when someone may receive *this* delivery again: `True` (any time), `False` (once ever), or a duration (not within that long of the last receipt). It is the self-referential case of `d.since`.
- **Doses.** `ss.Vaccine(..., doses=[0.85, 1.0])` gives relative efficacy after 1 and 2 doses. Then `has_<vx>` has levels 0, 1, 2 (people dimension, as now). The rule "a delivery never gives a product to someone who already holds it" (implementation.md, departures) becomes: *not to someone who holds every dose*. `dose=k` adds `vx.n == k − 1` to eligibility. Results: `delivered.select(dose=2)`.
- **Approximated (compartments).** `since >= 5y` is exact when the dimension has one level per step: 20 levels at quarterly dt. Above a cap (default 50 levels), levels hold several steps and the threshold is met at a level boundary; a note says so, and `approx=False` makes it agents. Each level multiplies compartments; `describe()` reports the size. A threshold that falls inside a level raises E804, as an age condition that cuts a bin does.
- **Modifiers.** `coverage=` is modifiable (`ss.modify(screen.coverage, multiply=0.5, start=2020.25, stop=2021)`). `repeat=` is a duration and modifiable like any duration.

### 2.5 Stock targets: `target=` and `remove=`

```python
ss.treat(art, eligible=hiv.diagnosed, target=n_art_series, priority=hiv.cd4 / hiv.care_seeking, remove=hiv.dropout)
ss.vaccinate(vmmc, eligible=men15_49, target=ss.share(p_circ, among=men15_49))     # "X% circumcised"
ss.treat(art, eligible=hiv.diagnosed, target=ss.by(art_df, dims=['age', 'sex'], time='year', value='n_art'))
```

- **What is counted.** The stock is the number of people in the delivery's *end state*: the destinations of a Treatment's `moves=` (`on_art`), the holders of a held product, or the level a transfer moves people to. It is inferred and printed (`target counts: hiv.on_art`). `of=` overrides it. `ss.share(x, among=cond)` means a target of x × count(among), counting only end-state people within `among`. A bare number is people (rule 1.2).
- **Stratified.** A `target=` given as `ss.by(...)` is solved per stratum. As agents, the strata are binned on the fly (the bins of the `ss.by`, even if the sim is not stratified by them). As compartments, the bin edges must be edges of the model's stratification, else E804 (the same rule as conditions).
- **Agents** (per step, per stratum): gap = target − n_end. If gap > 0, take the eligible (plus the waiting, if `queue=`), apply `coverage=` as an acceptance probability, order by `priority=`, and deliver the first gap; capacity, if given, caps it further. If gap < 0 and `remove=` is given, force −gap end-state people along `remove=` (an edge, or `True` for a held product: they lose it), the highest priority key first. `remove=None` (the default) only stops adding. This is v3 `ART.step` (`hiv_interventions.py:392-572`) with nothing private.
- **Compartments.** Two realizations, chosen by solver, both printed:
  - stepped solvers and the default ODE: the impulse of rule 1.1, so the target holds at each time point, as with agents.
  - `target_mode='track'` (ODE): a controlled flow along the edge. Rate u(t) = clip(dT/dt + outflow(N) + (T − N)/τ, 0, u_max(E)), with τ = dt, evaluated at every RK4 stage. This is "solving for the initiation rate". It is an expression of counts and parameters, so it can run in the kernels and in JAX, unlike stepped deliveries (deferred.md). Negative u is a flow along `remove=`. The two modes differ by O(dt) between time points; the note says which is used.
  - with priority: see §2.6 (water-filling over groups).
- **Results:** `target` (the target itself, so `plot()` draws it against the count), `n_end` (a stock: on ART, circumcised; this also fixes the missing "holders" result), `delivered`, `removed`, `shortfall` (target − n_end after delivery, when the eligible pool runs out).
- **Interaction with capacity and coverage.** coverage, then priority, then the gap, then capacity: the smallest wins. Target and capacity together cover "N on ART, but at most M initiations a month".
- **Modifiers.** `target` is a modifiable quantity (`ss.change(art_prog.target, at=2020.25, by=0.9)`). A scenario that rescales targets does not have to touch the data.

### 2.6 Priority

```python
priority = hiv.cd4 / hiv.care_seeking            # a lazy expression: lowest first
priority = ss.desc(hiv.viral_load)               # highest first
priority = [ng.C, screen.positive]               # tiers: all of the first, then the second
priority = fn(sim)  /  fn(sim, uids)             # key per person (agents) or per group
```

- Used by every mechanism that has to pick some of the eligible: `capacity=` (instead of picking at random; `interventions.py:996`), `target=`, `queue=` (priority, then longest wait), and removal (reverse order, or `remove_priority=`). With no `priority=`, the order is random, as now. Ties are broken at random with the delivery's CRN stream, so scenarios stay paired.
- `deliver=` returning an ordered UID list is **not** a priority: capacity keeps treating it as a set. The docs say to use `priority=`.
- **Agents.** Stable sort by key; take the first k.
- **Compartments.** Water-filling over cells. Evaluate the key on groups (tiers: each condition evaluated on groups; an expression: on group values, e.g. CD4 bin midpoints). Fill cells in key order until the allocation is used up, pro rata within a tie. Exact for keys that are states or strata (CD4 as a stratification or as CD4 states, symptom states). A key on an agent-only attribute: under `'auto'` the model runs as agents (it already does, since the attribute is agent-only). Under an explicit compartmental solver the allocation is pro rata with a note (`priority ignored: hiv.care_seeking is per agent`); E804 with `approx=False`.
- **Modifiers.** None: an order is not a number. The key reads the effective values of modified quantities.

### 2.7 Capacity, queues, and shared capacity

```python
clinic = ss.Capacity(ss.perweek(100), name='clinic')                 # people per unit time, or a timeseries
symp   = ss.treat(kit, on=seek, capacity=clinic, queue=ss.months(1))  # wait up to a month
screen = ss.screen(naat, eligible=..., capacity=clinic, priority=2)   # 2 = served after priority-1 deliveries
abl    = ss.treat(ablation, capacity=ss.perweek(100), queue=True)    # a private capacity: a number works as now
```

- **`queue=`**: `None`/`False` drops anyone over capacity (now; reported as `dropped`), `True` waits indefinitely, a duration is the longest wait. Order within the queue: `priority=`, then longest waiting (FIFO). People who stop being eligible (cured, aged out, died) leave the queue (`left`). A queue gives event-triggered and follow-up deliveries the backlog that today only `eligible=<persistent state>` has: the 60/yr vs 240/yr gap in HPV.
- **`ss.Capacity`** is a shared amount (number per round, rate, timeseries, modifiable). Deliveries that share one are resolved **jointly**. At the loop position of the first of them, the capacity collects their demand (new arrivals plus queues), allocates it by delivery priority (the integer `priority=`, default equal, then pro rata), then within each delivery by its own key, and only then do the deliveries give. `sim.loop` prints them adjacent. A follow-up that fires later in the same step (a delayed result) draws from the remainder. `uses=2` on a delivery means each recipient takes two units (e.g. excision vs ablation; open).
- **Agents.** Per delivery, a `ti_queued` array (NaN = not waiting). Each step: drop the waiting who are past `queue=`; remove the no-longer-eligible; candidates = arrivals ∪ waiting; allocate; the unserved keep (or get) `ti_queued`.
- **Compartments.** `waiting_<delivery>` is a people dimension, with level w = steps waited (1..K, K = max wait / dt). It crosses the disease states, so waiting people keep progressing, clearing and dying. Each step: shift the cells that are no longer eligible to level 0 (`left`); serve from the highest level down (FIFO), or by the group key first; move the unserved up a level; the overflow at K is `dropped`. With `queue=True`, K is capped (default 50) and the last level absorbs, so FIFO is approximate within it (note). Exact otherwise. The cost is K levels × cells; `describe()` reports it.
- **Results** (in people, by stratum): `delivered`, `waiting` (a stock), `wait` (mean steps waited, among those served), `dropped`, `left`. On `ss.Capacity`: `used`, `available`, `unused`.
- **Modifiers.** `ss.modify(clinic.capacity, multiply=0.3, start=2020.25, stop=2020.75)` (COVID disruption); it is listed by `what_modifies`.
- **Rate in rounds.** `capacity=ss.peryear(1000)` with yearly rounds means 1000 per round (the bug in §2.4).

### 2.8 `then=`

`then=` is a follow-up for people who *received* a non-test delivery, on the same step: `ss.treat(kit, ..., then=ss.treat(metronidazole, eligible=ppl.female))`. It is the generalization of `on_positive=`, with the same registration and auto-naming. A list in any follow-up slot (`on_positive=[a, b]`, `then=[a, b]`) is accepted; the members choose their recipients from the same people before anyone moves (rule 1.5). Today a list crashes mid-run (`interventions.py:1114`).

## 3. Compartmental realization at a glance

| Primitive | Compartmental form | Exact? |
|---|---|---|
| undriven edge + `moves=` | a forced fraction of each source cell at the time point | yes |
| `on_fail=` | the complementary fraction to another edge / transfer / stays | yes |
| multi-disease product or test | modifiers per disease on one `has_` dimension; selection on crossed diseases | yes (size grows) |
| `outcomes=` | multinomial split of the chosen cells | yes |
| typed coverage | a transfer rate among eligible cells | yes |
| routine once per spell | `offered_` dimension, shifted back to 0 when a cell is not eligible | yes, to dt |
| `.since`, `repeat=` | `since_` staged dimension | exact at 1 level/step; grouped levels with a note |
| `.n`, `dose=`, `doses=` | `n_` dimension, or `has_` levels 0..k | yes |
| `target=` | impulse to the target at each time point; or (`track`) a controlled flow | yes / O(dt) |
| stratified target | solved per stratum on the model's bins | yes if edges align, else E804 |
| `priority=` | water-filling over cells by a key evaluated on groups | yes for state/strata keys; pro rata with a note for agent attributes |
| `queue=` | `waiting_` staged dimension, served top-down | yes for a finite max wait; last level absorbs otherwise (note) |
| `ss.Capacity` | joint allocation across deliveries per time point | yes |
| `then=` | same-step follow-up on the delivered cells | yes |

What stays agent-only: `deliver=fn(sim) → uids`, `trace=`, `fn(sim, uids)` in any slot, and priority on a per-agent attribute (approximated above). Everything in the worked examples except the HIV CD4 float runs both ways.

## 4. Worked examples

### 4.1 HIV: ART stock target, CD4 priority, removal, age × sex

Before (`hiv/port.py:64, 242-264`):
```python
start_art   = (acute | latent | falling | post_art) >> on_art @ 0                 # only ever forced by ART delivery
def art_target(sim):
    n = np.interp(sim.t.now('year'), art_data.year, art_data.n_art)
    return n/sim.current_scale
def art_deliver(sim):
    hiv = sim.diseases.hiv
    n_on = hiv.on_art.count()
    gap = art_target(sim) - n_on
    if gap < 0: # Over target: v3 removes the highest-CD4 / lowest-care-seeking. No delivery can do this; force the dropout transition by hand
        on = hiv.on_art.uids
        drop = on[np.argsort(-hiv.cd4[on]/hiv.care_seeking[on])[:int(-gap)]]
        hiv.engine.force(drop, hiv.graph.transitions['dropout'], hiv.graph.states['post_art'])
        return ss.uids()
    cands = (hiv.diagnosed & ~hiv.on_art).uids
    cands = cands[hiv.infected[cands]]
    order = np.argsort(hiv.cd4[cands]/hiv.care_seeking[cands], kind='stable')
    return cands[order[:int(round(gap))]]
art = ss.Treatment(efficacy=1.0, transition='start_art')   # "cures" = forces start_art; re-initiation from post_art for free
return ss.treat(art, deliver=art_deliver, name='art_prog')  # name='art' collides with the product's name
```

After:
```python
class HIV(ss.Disease):
    ...
    start_art = (acute | latent | falling | post_art) >> on_art          # undriven
    dropout   = on_art >> post_art @ ss.lognorm_ex(ss.years(3), ss.years(1.5))

hiv = HIV(...)
n_art = ss.timeseries(dict(zip(art_data.year, art_data.n_art)), mode='linear')   # people
art_prog = ss.treat(ss.Treatment(moves=hiv.start_art),
    eligible = hiv.diagnosed,
    target   = n_art,                             # counts hiv.on_art (printed)
    priority = hiv.cd4 / hiv.care_seeking,        # lowest first; removal takes the highest first
    remove   = hiv.dropout,
)
# age x sex: target=ss.by(art_df, dims=['age', 'sex'], time='year', value='n_art')
```
- Results: `art_prog.target`, `n_end` (= `hiv.n_on_art`), `delivered`, `removed`, `shortfall`.
- Compartmental: with CD4 as states or a stratification (`hiv.stratify(cd4=[0, 200, 350, 500])`), priority water-fills the lowest-CD4 cells first. `care_seeking` is per agent, so under `solver='ode'` it is dropped from the key with a note. The port's float `cd4` already makes the model agent-based; that is unchanged.
- v3 `art_initiation=0.9` becomes `coverage=0.9` (an acceptance probability per offer, applied before the gap).

### 4.2 VMMC: a flow from procedure counts, or a stock

Before (`hiv/port.py:268-273`):
```python
vmmc = ss.Vaccine(efficacy=0.6, name='vmmc')                # rel_sus x 0.4 for those who hold it
cap = ss.timeseries(dict(zip(vmmc_data.year, vmmc_data.n_vmmc)))  # people per year
men = lambda sim: sim.people.male & (sim.people.age >= 15) & (sim.people.age < 50)
return ss.vaccinate(vmmc, eligible=men, capacity=cap, start=2008, name='vmmc_prog')
```

After: unchanged for the flow, which was already right. `n_end` (circumcised men) is now a result. If the data were prevalence:
```python
men15_49 = ppl.male & (ppl.age >= 15) & (ppl.age < 50)
ss.vaccinate(vmmc, eligible=men15_49, target=ss.share(p_circ_series, among=men15_49), start=2008)
```
No `remove=`, so nobody is un-circumcised when men age out of the window.

### 4.3 HPV: screen (1,000/yr) → ablation-eligible? → treat (100/week, FIFO)

Before (`hpvsim_interventions/port.py:114-130, 156-176, 207-217`, abridged):
```python
def treat_all_genotypes(product_name, coverage=1.0):
    txs = []
    for g in G:
        tx = ss.Treatment(efficacy=tx_efficacy(product_name, g), transition='excised', name=f'{product_name}_{g}_tx', cost=None)
        txs.append(ss.treat(tx, disease=g, eligible=ss.Cond('state', g, 'CIN'), coverage=coverage, name=f'{product_name}_{g}'))
        txs[-1].triggered = True # otherwise it also runs on its own, treating everyone infected
    def followup(sim, uids):
        for d in txs:
            mine = sim.interventions[d.name]
            mine.receive(mine.selector.to_selection(uids))
    ...
class Queue(ss.Intervention):
    """ FIFO waiting list, like v3's treat_num: on_positive=queue.add, then deliver=queue.take """
    ...
def algo_a(coverage=0.3, screen_capacity=1000, tx_capacity=None, queue=None):
    ablate, excise = treat_all_genotypes('ablation'), treat_all_genotypes('excision')
    eligible_for_ablation = ss.Test(sensitivity=0.9, specificity=0.9, detects=~cancer_any(), name='abl_elig')
    tx_visit = ss.screen(eligible_for_ablation, coverage=0.8, capacity=tx_capacity, on_positive=ablate, on_negative=excise, disease='hpv16', name='tx_visit')
    if queue is not None: # waiting list instead of drop
        tx_visit = ss.screen(eligible_for_ablation, deliver=queue.take, on_positive=ablate, on_negative=excise, disease='hpv16', name='tx_visit')
    primary = ss.screen(dx('hpv'), coverage=coverage, interval=ss.years(1), start=T0, eligible=women(30, 50),
                        capacity=screen_capacity, on_positive=queue.add if queue else tx_visit, disease='hpv16', name='primary')
```

After:
```python
genos    = [make_genotype(g) for g in G]                         # each has excised = CIN >> I (undriven)
tx       = lambda name: ss.Treatment(moves={d.excised: tx_efficacy(name, d.name) for d in genos}, name=name)
hpv_dna  = ss.Test(sensitivity={d[s]: p for d in genos for s, p in dx_table('hpv', d.name).items()}, specificity=0.97)
any_ca   = ss.any(d.CA for d in genos)

tx_visit = ss.screen(ss.Test(sensitivity=0.9, specificity=0.9, detects=~any_ca),
                     coverage=0.8, capacity=ss.perweek(100), queue=ss.months(6),
                     on_positive=ss.treat(tx('ablation')), on_negative=ss.treat(tx('excision')))
primary  = ss.screen(hpv_dna, eligible=women(30, 50), coverage=ss.probperyear(0.3), start=2020,
                     capacity=ss.peryear(1000), repeat=ss.years(5), on_positive=tx_visit)
```
- The over-capacity positives wait in `tx_visit` FIFO for up to 6 months (`waiting`, `dropped`). Screens over 1,000/yr are simply not done that step (no queue); under routine with typed coverage, those women are offered again at the next step.
- Results: `primary.delivered/diagnosed`, `tx_visit.waiting/wait/dropped`, `ablation.cured.by('disease')`, and `overtreated` (HPV+ without CIN, ablated).
- Compartmental: the genotypes are crossed (64 joint states per group), `waiting_tx_visit` has 26 levels at weekly dt or 2 at quarterly dt, and `since_primary` has 20 levels at quarterly dt. All exact. `describe()` prints the size.

### 4.4 Bivalent vs nonavalent over genotype diseases, two doses

Before (`hpvsim_interventions/port.py:65-76, 195-203`):
```python
vx = ss.Vaccine(efficacy=eff(primary), name=name, cost=5.0)
holds = ss.Cond('people', f'has_{name}') # no public spelling for "holds this product"
cross = [ss.modify(f'{g}.rel_sus', multiply=1 - eff(g), where=holds, name=f'{name}_{g}') for g in G if g != primary]
...
    dose2 = ss.Vaccine(efficacy=dose2_boost, name=f'{product}_dose2', cost=5.0)
    def due(sim):
        u = sim.people.auids
        got = sim.people[f'has_{product}'][u].astype(bool)
        t1 = sim.interventions.routine_vx.product.ti_delivered[u]
        return u[got & ((sim.ti - t1)*sim.t.dt_year >= 0.5)]
    intvs.append(ss.vaccinate(dose2, eligible=due, coverage=0.9, start=start, disease='hpv16', name='dose2_vx'))
```

After:
```python
def hpv_vaccine(name):
    rel = csv.vx[csv.vx.name == name].set_index('genotype').rel_imm
    return ss.Vaccine(efficacy={g: rel[g] for g in G}, doses=[0.85, 1.0], cost=5.0, name=name)   # keyed by disease

vx = hpv_vaccine('bivalent')      # or 'nonavalent': the only change between the scenarios
progs = [
    ss.vaccinate(vx, dose=1, eligible=women(9, 15), coverage=ss.probperyear(ss.timeseries({2020: 0.2, 2025: 0.8}, mode='linear')), start=2020),
    ss.vaccinate(vx, dose=1, eligible=women(15, 26), coverage=0.5, years=[2020]),
    ss.vaccinate(vx, dose=2, eligible=vx.since >= ss.months(6), coverage=0.9),
]
```
- `has_bivalent` has levels 0/1/2. Efficacy per genotype × dose factor gives three modifiers (one per genotype's `rel_sus`), listed by `what_modifies('hi5.rel_sus')`.
- Results: `delivered.select(dose=2)`; `vx.n_end` (by dose); `cost`.
- Compartmental: the dose dimension (3 levels) and `since_bivalent` (2 levels at quarterly dt). Exact. The routine program's typed coverage is a rate among girls 9–14.

### 4.5 Syndromic management (NG/CT/TV) from one visit, shared clinic

Before (`stisim_bacterial/port.py:365-371, 384-398`):
```python
seeking = ng.C | ct.C | tv.C # Conditions on three diseases' states of the same name, each keeping its own
deliveries = []
for dis, eligible in [(ng, seeking), (ct, seeking), (tv, seeking & ppl.female)]:
    drug = ss.Treatment(efficacy=1.0, transition='treat', name=f'{dis.name}_drug')
    deliveries.append(ss.treat(drug, eligible=eligible, interval=ss.months(1), disease=dis.name, name=f'{dis.name}_tx'))
...
class Clinic(ss.Intervention):
    def step(self):
        d = self.sim.diseases
        self.visit[:] = False
        self.visit[(d.ng.C | d.ct.C | d.tv.C).uids] = True
```

After:
```python
clinic = ss.Capacity(ss.perweek(100))
discharge = ss.Treatment(moves={ng.cure: 0.95, ct.cure: 0.95}, name='ceftriaxone_doxy')
metro     = ss.Treatment(moves={tv.cure: 0.90}, name='metronidazole')
syndromic = ss.treat(discharge, eligible=ng.C | ct.C | tv.C, capacity=clinic, queue=ss.months(1), priority=1,
                     then=ss.treat(metro, eligible=ppl.female))
screening = ss.screen(naat, eligible=young_women, coverage=ss.probperyear(0.3), capacity=clinic, priority=2,
                      on_positive=ss.treat(discharge))
```
- Recipients are chosen once from the union, before anyone moves, so NG and CT see the same visits (no 9,250 vs 3,275). Routine once per spell: each care-seeking episode is offered once, and each reinfection again.
- Results: `syndromic.cured.select(disease='ct')`, `overtreated.by('sex')` (seeking for NG, treated for CT), `failed`, `waiting`, `dropped`; `clinic.used`. Symptomatic visits are served before screen positives because of the delivery priorities 1 and 2.
- `queue=` can replace the port's `C`/`N`/`balk` states. `on=ng.seek | ct.seek | tv.seek`, with a one-month max wait, is the balking. Keeping `C` is still allowed.
- Compartmental: the three diseases are crossed (as any condition across them requires). Exact.

### 4.6 STI treatment with failure (and AMR)

Before (`stisim_bacterial/port.py:146-148, 270`):
```python
trs['treat'] = (E | A | P | Y | C | N | PID) >> T @ ss.permonth(0) # Only ever moved along by ss.Treatment(transition='treat')
trs['cure'] = ss.Transition(T, S, p=eff)
trs['fail'] = ss.Transition(T, A, p=1 - eff if fail is None else fail) # A failure stays infected... p=None should be "the rest" (spec), but is E209
drug = lambda: ss.Treatment(efficacy=1.0, transition='treat', name='drug')
```

After:
```python
cure = (E | A | P | Y | C | N | PID) >> S                        # undriven; no T state
drug = ss.Treatment(moves={ng.cure: 0.9})                        # failures stay in their state, counted
# AMR: efficacy by strain and time; resistance acquired on failure
drug = ss.Treatment(moves={ng.cure: ss.by(strain=[ss.timeseries({2000: 0.96, 2010: 0.90}), 0.30])},
                    on_fail=ss.transfer(strain='sens -> res', p=0.25))
```
- Results: `tx.cured`, `tx.failed`, `tx.overtreated`, each `.by('strain')` / `.by('sex')`; `ng.cure` (the flow).
- Compartmental: a forced fraction per (state, strain, group) cell; failures with p 0.25 shift along the strain dimension. Exact.

### 4.7 Rescreen every 5 years

Before (`hpvsim_interventions/port.py:133-153, 229-235`):
```python
class screen_mem(ss.screen):
    """ ss.screen that remembers when each person was last tested (v4 deliveries keep no such record) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_states(ss.FloatArr('ti_screened', label='Time last screened'))
    def test(self, chosen):
        self.ti_screened[self.uids(chosen)] = self.sim.ti
        return super().test(chosen)
def due_for_screen(name, lo=30, hi=50, every=5):
    def eligible(sim):
        ...
        last = sim.interventions[name].ti_screened[u]
        since = (sim.ti - last)*sim.t.dt_year
        ok = p.female[u] & (p.age[u] >= lo) & (p.age[u] < hi) & (np.isnan(last) | (since >= every))
        return u[ok]
p_step = 1 - (1 - annual_prob)**0.25
primary = screen_mem(dx('hpv'), coverage=p_step, interval=ss.years(0.25), start=T0, eligible=due_for_screen('primary'), ...)
```

After:
```python
primary = ss.screen(hpv_dna, eligible=women(30, 50), coverage=ss.probperyear(0.3), repeat=ss.years(5), start=2020, on_positive=...)
# "not screened by any HPV test in 5 years":  eligible=women(30, 50) & (hpv_dna.since >= ss.years(5))
```
- Calendar rounds every 5 years stay `interval=ss.years(5)`: a different program, still available.
- Compartmental: `since_primary`, 20 levels at quarterly dt; the condition `since >= 5y` is the top level. Exact.

## 5. Rejected

- **Leave stock targets, priority and removal to `deliver=`** (the status quo). This is the core HIV/PrEP/VMMC workflow. It needs `engine.force` and `current_scale`, and it is agent-only. A keyword is shorter, and it runs compartmentally.
- **Priority by the order of `deliver=`'s return value.** It is invisible, lost by any set operation, and has no group form.
- **`capacity=` meaning a stock.** It is a flow (VMMC was right). A stock is a different keyword, so the two can be given together.
- **A queue as a disease state** (the bacterial port's `C`) as the only way. It works for one disease with persistent eligibility. It cannot be shared across diseases, puts program structure into natural history, and does nothing for follow-ups. It stays allowed.
- **A transient treated state `T` for failures.** It costs a step of lag, sends every failure to one state, and puts fake structure in the disease.
- **ART as a held product** (`efficacy=dict(transmission=, mortality=)`). It has exponential dropout only, no post-ART state, and `mortality` hits every outflow. A product is for what changes rates; a state is for what changes the natural history.
- **Genotype as a strain dimension to dodge multi-disease products.** It gets vaccine efficacy silently wrong (stale level) and loses coinfection. Keyed products work for both forms.
- **One delivery per disease, with shared eligibility.** This is the loop-order artefact itself.
- **A `Program` class with budget → coverage curves** (Atomica; questions-oct12 #10). `cost=` results plus a modifiable `coverage=`/`capacity=`/`target=` cover scenarios; budget optimization is out of scope for v1 (08).
- **Always-on per-person delivery memory.** It is cheap per agent, but as compartments every `since`/`n` is a dimension. It is allocated only when referenced (rule 1.3).
- **`once=`/`every=` as separate keywords.** `repeat=` covers both, and `every=` is already the reporting period of `ss.outcome`.
- **Making routine a rate by default.** A bare coverage stays per offer, which is what a vaccination-at-9-months user means; only typed coverage is a rate.

## 6. Open questions

1. **Target timing.** An impulse at the time point (agents and default ODE) vs `track` (ODE). Should `track` be the ODE default? It is smoother and compiles, but differs from agents by O(dt). Gradients through `clip` and water-filling are piecewise: acceptable for NUTS?
2. **Mixed n → p targets** (v3: historical `n_art`, then `p_art` from 2023). Proposal: `target=ss.switch({1990: n_art, 2023: ss.share(0.9, among=hiv.infected)})`. Or is a list enough?
3. **`coverage=` together with `target=`.** Acceptance per offer (v3 `art_initiation`, once at diagnosis) or per step among the waiting? v3 applies it once; the `repeat=`/spell rule would give the same result if coverage is per spell.
4. **Priority with `queue=`.** Priority first, then wait (proposed), or a combined key (`ss.desc(waiting.time)`)? Does starvation of low-priority waiters matter enough for `max_wait` to be required?
5. **`ss.Capacity` units.** `uses=` per delivery (excision = 2 slots), cost-denominated capacity (a budget), and stock that carries over (19-open-questions B7: batches, stock-outs). Fold these into `ss.Capacity(stock=, resupply=)`, or defer them to `starsim_econ`?
6. **Removal destination for held products.** `remove=True` loses the product. Should a held product be able to name an exit state, or is that the sign it should be a state?
7. **Memory of a result.** The syphilis newborn test needs "mother tested positive" (`screen.positive.since`?), and v3 has first-time vs repeat treatment results. `d.n` covers the second. Is `d.positive` (the last result, lazy) worth it?
8. **Names.** `moves=` (vs `cures=`/`via=`), `repeat=`, `then=`, `ss.share`, `ss.desc`, `n_end`, `target_mode='track'`. `ss.any(...)` for unions of states across diseases is also used above; is the `|` reduction enough?
9. **Crossing cost.** HPV with 4 real genotypes and 8 states is 8⁴ joint states per group before strata. Is a multi-genotype product in a compartmental run realistic, or should `solvers()` steer such models to agents with a size note?
10. **Episode definition.** "Once per eligibility spell" uses the delivery's own `eligible=`. Some programs want a different episode (`once_per=preg.pregnancy` while eligibility is `anc.visit`). Is `repeat=preg.pregnancy` (a condition as the episode) worth adding?
11. **FPsim.** Method mix is a categorical choice (the same table as `outcomes=`) plus per-method stock targets (`target=ss.by(method=...)`). Check that against `fpsim/methods.py` `ContraceptiveChoice` before adding a separate `ss.choose`.
12. **Order inside a step.** As agents, deliveries run after progressions (known issue, implementation.md); as compartments, before all flows. A target met "at the time point" therefore means slightly different stocks in the two solvers. Fix the order, or document it?
