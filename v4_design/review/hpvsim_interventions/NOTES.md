# HPVsim v3 interventions/products → Starsim v4: review

Files: `port.py` (main port: one `ss.Disease` per genotype, as in HPVsim v3), `genotype_dim.py` (sidebar: genotype as a strain dimension of one disease). Both run: `conda run -n starsim python port.py` (12 sims, 5k agents, 1990–2050, quarterly, ~15 s, ABM).

Natural history is a stand-in: per genotype `S → I → CIN → CA → death`, `I → S` clearance, `CIN → S` regression, plus `excised = CIN → I @ 0` (only treatment moves people along it: the lesion is removed, the infection is not). Genotypes `hpv16, hpv18, hi5`, so the CSV rows apply directly.

At 5k agents, cancer outcomes of screening are inside the noise (±10% between scenarios that differ only in CRN divergence). Program counts (screened, positive, treated, doses) are the things to read; they behave as intended.

## Feature by feature

### 1. Products from the CSVs
v3: `hpv.vx/dx/tx/txvx` read `products_*.csv` (products.py:41–190) and loop over genotype modules in `administer()` (products.py:246–283, 350–393, 431–477).
v4: `vaccine()`, `p_positive()`/`dx()`, `tx_efficacy()` in port.py read the same CSVs and build `ss.Vaccine`/`ss.Test`/`ss.Treatment`.
- **Partial.** The CSVs map onto v4 products only per disease. A v4 product acts on exactly one disease (`Held.init_pre`: `dis = sim.get_declared(self.disease)`), so anything multi-genotype gets split or pushed into a function.
- Rating: **awkward** overall. The individual parts are rated below.

### 2. Prophylactic vaccine, genotype-specific efficacy (focus 3)
v3: one product writes `vax_imm[g] = rel_imm[g]` to every genotype (products.py:269–283).
v4 (port.py `vaccine()`):
```python
vx = ss.Vaccine(efficacy=eff('hpv16'), name=name, cost=5.0)
cross = [ss.modify(f'{g}.rel_sus', multiply=1 - eff(g), where=ss.Cond('people', f'has_{name}')) for g in G[1:]]
```
- **Partial.** The product holds for one genotype, and every other genotype needs a connector keyed on "holds the product". This works, and `what_modifies('hi5.rel_sus')` lists it nicely. Bivalent and nonavalent differ as they should: hi5 infections −36% vs −59%.
- There is no public spelling for "holds product X". `ppl.has_bivalent` → `AttributeError: People object has no attribute "has_bivalent"`, so I had to build `ss.Cond('people','has_bivalent')` by hand.
- v3's all-or-nothing "take" (`sterilizing_p`) has no v4 equivalent. The new `ss.Vaccine` is leaky only, so I folded it into the efficacy.
- Rating: **awkward**.
- Suggestion: accept `efficacy=dict(hpv16=1.0, hpv18=1.0, hi5=0.5)` keyed by disease (a product acting on several diseases), plus `vx.held` as a condition and `take=` for all-or-nothing.
- **Strain-dimension variant (genotype_dim.py) is worse.** `ss.Vaccine(efficacy=ss.by(genotype=[1,0,0]))` is accepted silently but is wrong. Efficacy is looked up per *person* (`Effect.apply → selector.per_person`), and a susceptible's genotype level is stale (their last infection's). Result: hpv18 and hi5 infections fell ~50%, and 811 vaccinated women still got hpv16. **Bug: this should raise an error**, since `by()` names a dimension that the state the modifier acts on (`S`) does not have.
- The faithful alternative, `stratify(vx=[False, True], adjust={'infection': ss.by([[...]], dims=['vx','genotype'])})`, fails with `TypeError: float() argument must be a string or a real number, not 'by'`. A 2-D adjust is not supported, though `split=` accepts the same thing.

### 3. Routine girls 9–14, coverage scale-up; catch-up; doses
v3: `routine_vx(prob=..., years=..., age_range=(9,14))`: an annual probability among the eligible, applied every year (interventions.py:186–247, ss RoutineDelivery).
v4:
```python
ss.vaccinate(vx, eligible=women(9, 15), coverage=ss.timeseries({2020: 0.2, 2025: 0.8}, mode='linear'), interval=ss.years(1), start=2020)
ss.vaccinate(vx, eligible=women(15, 26), coverage=0.5, years=[2020])
```
- **Clean.** Two deliveries can share one product. Holders are excluded automatically, and coverage can be a time series.
- **Semantic trap.** v4 "routine" (no rounds) means *offered once per person, ever, at first eligibility*. The coverage of the step a girl turns 9 is all she ever gets, and every girl aged 9–14 at the start gets one offer. v3's routine is an annual probability, which is v4 *rounds* (`interval=ss.years(1)`). Users porting `routine_vx` will pick the wrong one.
- **Doses: ugly.** "A delivery never gives a product to someone who already holds it", so a second dose of the same product is silently refused: `eligible` shows 0 and nothing is delivered. Dose 2 has to be a second product (`ss.Vaccine(efficacy=0)` for v3 parity, where dose 2 only adds cost and counts). Its eligibility has to be a function reading `sim.interventions.routine_vx.product.ti_delivered` and `sim.people['has_bivalent']`.
- Rating: doses **ugly**. Suggestion: `ss.Vaccine(doses=2, interval=ss.months(6), efficacy=[0.8, 0.9])`, or `ss.vaccinate(..., dose=2, after=routine, wait=ss.months(6))`, plus a `doses` result.

### 4. Tests with stage- and genotype-specific sensitivity (focus 5)
v3: `hpv.dx` draws a multinomial result per (state, genotype) from products_dx.csv and takes the most severe result across genotypes (products.py:350–393).
v4, single disease: `ss.Test(sensitivity=dict(I=ss.by(genotype=[.9,.9,.5]), CIN=0.9, CA=0.9))` works and reads well (genotype_dim.py).
v4, one disease per genotype: the dict must name states of *one* disease. `sensitivity={hpv16.CIN: 0.75, hpv18.CIN: 0.75}` fails with a misleading message:
`E505 UnknownEffect: the test "via" has a sensitivity for "CIN", which is not a state of "hpv16". Its states are S, I, CIN, CA.`
The key's module is dropped, so the message lists CIN among the states it says CIN is not one of. So port.py builds `p_positive(name)`, a function of the sim returning 1 − ∏(1 − p[state, genotype]) per agent, used as `ss.Test(sensitivity=fn, detects=everyone)`.
- `detects=` with a cross-disease condition (`any CIN | any CA`) works, and is how "ablation-eligible = no cancer" is written (`detects=~cancer_any()`).
- **Multi-outcome results are impossible.** The v3 hierarchies (`colposcopy`: normal/ascus/lsil/hsil/cancer; `tx_assigner`: ablation/excision/radiation/none; `hpv_type`: 16/18 vs OHR) don't fit, because v4 tests are binary. tx_assigner becomes a chain of binary tests with conditional probabilities. That works, but each "test" shows up as `delivered`/`diagnosed` with test semantics.
- Rating: single disease **clean**; multi-genotype **awkward**; multi-outcome **impossible** (only by chaining).
- Suggestions:
  - sensitivity dicts keyed by `State` objects from several diseases, combined as independent detections;
  - `ss.Test(results={'hsil': {...}, 'lsil': {...}})` with `on_result={'hsil': ...}`.

### 5. Treatment: clears CIN, not necessarily the infection
v3: `hpv.tx` clears CIN/cancer state for each genotype and schedules clearance (products.py:431–477).
v4: `ss.Treatment(efficacy=0.936, transition='excised')`, with `excised = ss.Transition(CIN, I, rate=0)`. It clears the lesion and leaves the infection to clear or progress naturally.
- **Clean for one disease.** `force()` moves only people in the transition's source states, and re-entry into `I` gets a fresh exit time (I checked: no one stuck).
- **Limits.**
  - Only one transition, and `force` always uses `tr.dst[0]`. "Clears CIN, and with probability q also the infection" needs either two treatments (but `ss.treat` has no follow-ups) or a branching destination (ignored).
  - Efficacy by state (ablation 0.936 for CIN, 0 for cancer) is a single scalar. Here I relied on cancer not being a source state of `excised`.
  - A curing treatment keeps no per-agent record (`ti_delivered` exists only for held products), so "treated before" cannot be read.
- **Multi-genotype: ugly.** Ablation must act on every genotype. `on_positive=[tx16, tx18, tx_hi5]` fails late, during the run: `TypeError: 'list' object is not callable` (interventions.py:1114). So `treat_all_genotypes()` builds one `ss.treat` per genotype, sets `.triggered = True` by hand (otherwise each also runs on its own and treats everyone infected), and uses a follow-up function that calls `sim.interventions[name].receive(selector.to_selection(uids))`. That is internals in three places.
- A related gotcha: the function cannot close over the delivery objects, because `ss.Sim` copies its modules. The first attempt failed with `AttributeError: treat object has no attribute "selector"` on the uninitialised originals.
- Suggestions:
  - `on_positive=` accepting a list;
  - `ss.Treatment(efficacy=dict(CIN=0.936, CA=0), transition=...)` by state, with branches;
  - a treatment acting on several diseases.

### 6. Multi-step algorithms with `on_positive=` (focus 1)
v3 (examples/t05_screen_algorithms.py), algorithm 5 is four modules glued with lambdas over `.outcomes`:
```python
hpv_primary = hpv.routine_screening(product='hpv', prob=0.2, start_year=2025, name='hpv primary')
via_triage = hpv.routine_triage(product='via', prob=0.9, eligibility=lambda sim: sim.interventions['hpv primary'].outcomes['positive'], name='via triage')
tx_assigner = hpv.routine_triage(product='tx_assigner', prob=0.9, eligibility=lambda sim: sim.interventions['via triage'].outcomes['positive'], name='tx assigner')
ablation = hpv.treat_num(prob=0.9, product='ablation', eligibility=lambda sim: sim.interventions['tx assigner'].outcomes['ablation'])
```
v4 (port.py `algo_b`):
```python
via = ss.screen(dx('via'), coverage=0.9, on_positive=ablate)
primary = ss.screen(dx('hpv'), coverage=0.3, interval=ss.years(1), start=2020, eligible=women(30, 50), on_positive=via)
```
- **Much more readable.** The tree is visible, and loss to follow-up is `coverage=` at each node. Follow-ups are auto-registered and auto-named (`primary_positive`), and a follow-up shared by two parents is added once.
- Order-of-execution worries about v3's `.outcomes` (a triage reading the previous step's screen) are gone.
- This is the best part of the v4 design for HPV, and it breaks down only at the leaves (multi-genotype treatment, multi-outcome tests).
- Rating: **clean**. Single-disease and binary-result trees are entirely declarative (genotype_dim.py `t_test`).

### 7. Capacity (focus 2)
v3: `treat_num(max_capacity=N)` keeps a FIFO queue. Accepted people wait, are removed if no longer eligible, and the first N are treated each step (ss interventions.py:418–461).
v4 `capacity=`: those over capacity are chosen at random and **dropped**.
- Routine (no rounds): the dropped are not marked offered, so they are retried on later steps. That is an implicit backlog, but with random order, not FIFO.
- Rounds: the dropped wait for the next round.
- Follow-ups (`on_positive`): **dropped are lost.** There is no backlog and no warning.
  - port.py with 5 treatments/week: `a_cap` treats 60/yr of ~220 who need it. The FIFO `Queue` (an `on_positive=queue.add` function plus `ss.screen(..., deliver=queue.take)`) treats ~240/yr with a waiting list of 6–14 agents.
  - The drop is worse than the capacity suggests, because yearly rounds deliver every positive on one step, and a per-step cap (5/wk × 13 wk) applies to that step only.
- No priority keyword (v3 doesn't have one either).
- **Bug: rate capacity in rounds is scaled by dt, not by the round interval.** `cap()` multiplies a `TimePar` by `dt_sim` whether or not delivery is in rounds (interventions.py:987–988). `capacity=ss.peryear(1000)` with yearly rounds and quarterly dt screens **250/yr** (port.py `a_rate`). A plain `1000` (per round) gives 1000.
- Rating: drop semantics **awkward** (silent loss); queue **ugly** (custom module plus two functions).
- Suggestion: `capacity=..., queue=True` (or `backlog='fifo'|'random'|ss.by priority`, `max_wait=`). This keeps those over capacity eligible for the next steps, and reports `waiting` and `dropped` results. Count dropped people at minimum.

### 8. Time since last screen (focus 4)
v3: `ti_screened` per agent on every screening module (ss interventions.py:215, 277). Eligibility is a lambda reading it.
v4: deliveries record nothing per agent. Routine is "once ever", and rounds re-offer everyone each round. "Every 5 years" therefore comes in two forms:
- `interval=ss.years(5)`: calendar-synchronised mass rounds. Declarative, but not the same program.
- per-step uptake among women not screened in the last 5 years: needs `screen_mem`, a subclass overriding the internal `test()` to write `ti_screened`, plus an eligibility function returning UIDs.
- Gotcha: `def due(sim, years=5)` is called as `due(sim, uids)`. The signature inspection counts defaulted parameters as arguments (conditions.py:374), so `years` silently receives the UIDs. The result was a broadcast `ValueError`, not a clear error.
- Gotcha: arrays defined on an intervention have a `.raw` of a different length from `people.raw` (3529 vs 4500 after births), so `.raw` arithmetic across modules fails. Index by `auids` instead.
- Rating: **ugly**.
- Suggestion: every delivery keeps `ti_last` (and `n_received`) per agent, readable as a condition: `eligible=women(30,50) & (screening.since > ss.years(5))`, or `ss.screen(..., every=ss.years(5))` meaning "re-offer when last received ≥5 y ago".

### 9. Radiation
v3: adds ~18 months to `ti_dead_cancer` (products.py:566–580).
v4: `ss.Treatment(efficacy=dict(mortality=0.5))`, a held product that scales `cancer_death`.
- **Clean** as a rate approximation. A fixed delay of death isn't expressible.
- It is still per genotype: 3 deliveries, so a woman with two cancers gets two independent coverage draws.

### 10. Therapeutic vaccine
v3: `txvx` clears lesions with genotype- and state-specific efficacy (products_tx.csv) and adds severity immunity (products.py:479–547).
v4 (`txvx()`):
- `ss.Vaccine(efficacy=dict(dysplasia=..., invasion=...))` against named transitions, which is nice, plus cross-genotype `ss.modify('hpv18.dysplasia', where=holds)`;
- then, in the same step, one `ss.treat(..., eligible=holds & g.CIN, years=[2030])` per genotype.
- Genotype-specific clearance falls out naturally from one disease per genotype.
- Efficacy dicts only *reduce* rates (×(1−e)). "Boost clearance ×5" needs `efficacy=dict(regress=-4.0)`, which is accepted silently. That works, but it is a hack.
- Rating: **awkward**. It relies on delivery order in the list for "the same people".
- Suggestion: `effect=dict(regress=ss.multiply(5))` on products, and one delivery able to give two products.

### 11. Counts and costs by year
- **Clean.** Every delivery has `delivered`, `eligible`, `coverage`, `cost` (when `cost=` is given) and `diagnosed`, and `res.annualize()` gives yearly sums.
- Per-genotype treat deliveries count procedures per genotype, not women. Women treated is the `delivered` of the gate test (`tx_visit`) or of a separate count.
- v3's "first-time vs repeat" results (`new_cin_treated` vs `new_treatments`) have no equivalent, since there is no per-agent record.

## Bugs / mismatches found
1. **`ss.Births()`/`ss.Deaths()` with a declared disease: newborns are in no state.** They have `S = I = CIN = CA = False`: in partnerships but never infected. In my first port, 11,875 newborns had 0 infections, `n_S + n_I + ... = 40k` against `n_alive = 219k`, and vaccinating girls born during the sim had *exactly* zero effect. This is silent. Fix used: `births=ss.birth('-> S')` and `deaths=ss.death()` on one genotype; newborns then enter S of every genotype. That still means demography is declared on one arbitrary genotype. The v3 classes should raise an error or enter newborns into each declared disease's first state.
2. `capacity=` given as a rate in rounds is scaled by dt, not the round interval (§7).
3. `ss.by(dim)` efficacy on a strain dimension uses susceptibles' stale level, silently (§2).
4. 2-D `ss.by(dims=[...])` in `adjust=` → `TypeError` (§2).
5. `on_positive=[list]`: accepted at construction, `TypeError` during the run (§5).
6. A sensitivity dict keyed by another module's `State`: error message drops the module and contradicts itself (§4).
7. A function with defaulted parameters is treated as `(sim, uids)` (§8).
8. A negative efficacy is accepted without warning (§10).
9. An inline `ss.Treatment(...)` in a comprehension can't be named from a variable, so the products collide on name (`treatment`, `treatment_1`, … with a warning). I had to pass `name=` everywhere.

## Line counts
- v3: hpvsim `interventions.py` + `products.py` = 1,304 lines, plus the inherited Starsim 3 Routine/Campaign/Test/Treatment classes (~450 in ss interventions.py:77–530).
- v4 port.py: 356 lines including the natural history, scenarios and printing. Products ≈ 41 code lines, helpers (fan-out, memory, queue) ≈ 56, programs ≈ 72.
- The helpers are the part a modeler shouldn't have to write.

## Ranked problems for HPVsim
1. **Products and deliveries are bound to one disease.** Genotypes-as-diseases (needed for coinfection, and how v3 works) forces vaccine connectors, sensitivity functions, and fan-out treatments that call `Delivery` internals. Genotype-as-strain avoids that, but gets vaccine efficacy silently wrong and loses coinfection. Neither form is clean. Multi-disease products, or `on_positive=[...]` plus disease-keyed efficacy and sensitivity, would fix most of it.
2. **Capacity has no queue, and follow-up overflow is silently lost.** That changes results a lot (60 vs 240 treated/yr). The rate-in-rounds dt bug compounds it.
3. **No per-agent delivery memory** (last screened, doses, treated before). Rescreening intervals, multi-dose schedules and "first vs repeat" results each need a subclass or functions.
4. **Births via `ss.Births` silently produce uninfectable newborns** (cross-cutting, and serious for anything long-run).
5. **Binary tests only.** Colposcopy, cytology and tx_assigner hierarchies have to be emulated with chains.
6. **"Routine" means once-ever**, unlike v3's annual probability. That will be mis-ported.

## Surprisingly nice
- `on_positive=` trees with `coverage=` at each node: algorithms read like the WHO flowcharts. Follow-ups are auto-registered and auto-named, and can be shared between parents.
- `detects=` taking any condition, including across diseases (`~any cancer` as "ablation-eligible").
- Efficacy against named transitions (`dict(dysplasia=..., mortality=...)`), and `what_modifies()` showing every vaccine and connector on `rel_sus`.
- Treatment along a declared, rate-0 transition (`excised: CIN → I`) cleanly expresses "clears the lesion, not the infection".
- `deliver=` makes a FIFO queue possible in a few lines, without subclassing a delivery.
- Built-in `delivered`/`eligible`/`coverage`/`cost`/`diagnosed` results, with `annualize()`.
