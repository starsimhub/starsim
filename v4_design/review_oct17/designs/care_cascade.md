# The care cascade

Design pass H1 for Phase H of [implementation_oct18.md](../implementation_oct18.md) (decision D-F: one design document before code). Written 2026-10-19 against the `v4` working tree. Findings covered: ST12 to ST16, ST20, HP24, HP26, D27, D29, DF1, PL2, PL3, PL7, with ST5/HP25 (cures, done) and HP7 (one product over several diseases) for the interface. Probes are in `wt/scratch-h1/` of the session scratchpad.

## Summary

**The problem.** A cascade in v4 is a test with its follow-ups nested inside it (`ss.screen(test, on_positive=ss.treat(...))`), and nothing about it is remembered once the step is over. STIsim's four real cascades and HPVsim's seven WHO algorithms are not trees: a trigger can be the union of three diseases' events (ST15), one result feeds several actions months apart (ST16, ST14), a test has three to five results (HP24), and ART, VMMC and PrEP are kept at a stock target with a priority (ST12). Today each of these is a function, an extra class, or a chain of fake tests: HPVsim's cytology reports 35,397 tests for 20,000 done (costs 77% high), a bivalent vaccine silently protects against one genotype (HP7: still 250 delivered for the second disease, with no protection), and a screen-and-treat cascade forces the ODE to step from Python, taking 0.72 s for ten years of daily steps against 0.009 s for a vaccination the kernels take (DF1, measured here).

**The proposal, as a user writes it.** Deliveries are peers linked by events. Every delivery, every outcome of one, every product, and every transition is an event; `on=` takes one or a union (`|`); and every event remembers, per person, whether, how often, and how long ago it happened (`.ever`, `.count`, `.since`).

```python
lbc    = ss.Test(results=dict(abnormal=dict(C=0.9, X=1.0), ascus=dict(S=0.1, P=0.1, C=0.1)), default='normal', cost=10)  # three results (HP24)
hpvdna = ss.Test(sensitivity=dict(P=0.95, C=0.95, X=1.0), specificity=0.9)
ablate = ss.Treatment(cure=dict(P=0.936, C=0.936))                                    # cure by state (HP25)
women  = ppl.female & (ppl.age >= 25) & (ppl.age < 65)

cyto   = ss.screen(lbc, coverage=0.2, interval=ss.years(1), where=women & (lbc.since >= ss.years(5)))   # remembered per person (ST14)
triage = ss.screen(hpvdna, on=cyto.ascus, coverage=0.9)                               # a follow-up is a peer (ST16)
treat  = ss.treat(ablate, on=cyto.abnormal | triage.positive, delay=ss.months(3),     # a union of events (ST15), a delay (HP26)
                  capacity=ss.peryear(5_000), priority=cyto.abnormal.ever)            # limited supply, served in order
sim    = hpv.Sim(diseases=genotypes, interventions=[cyto, triage, treat])            # each test takes the worst result across the family

dx     = hiv_test.positive | anc_test.positive                                 # diagnosed by either program
art    = ss.treat(art_drug, where=dx.since >= ss.months(6), target=ss.timeseries(n_art), priority=[hiv.D, hiv.C])  # a stock target (ST12)
resist = ss.Delivery(move='I -> Ir', on=ng_tx.failed)                                # an outcome as an event: resistance at failure (ST16)
visit  = ss.Delivery(on=preg.conception, delay=ss.months(3), coverage=0.7)           # a visit: a delivery of nothing (ST13)
```

Under a compartmental solver the same cascade is built into a short program of delivery operations that the kernels run between steps, so it no longer stops a whole run (DF1).

**Decisions for Cliff.**

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | Follow-ups: nested or peers? | (a) peers only: `on=screen.positive`, and `on_positive=`/`on_negative=` removed; (b) both; (c) peers inside a new `ss.cascade(...)` container | (a). One spelling; the container adds a name and a scope that variables already give (decision 2) |
| 2 | A cascade container (`ss.cascade(visit=..., test=..., treat=...)`) | (a) none: a cascade is a list of deliveries; (b) add it now, for names, string references and a cascade table | (a), and revisit if a cascade results table ("lost at each step") is wanted; `ss.parallel` and `sim.new()` already take lists |
| 3 | Several events in `on=` | (a) `\|` only (`on=a.infection \| b.infection`); (b) a list too | (a). `\|` is already the union of conditions; a list raises, pointing to `\|` |
| 4 | Remembered results | (a) `.ever`, `.count`, `.since` on every event, kept only when something reads them; (b) fixed states (`tested`, `positive`, `ti_positive`) on `ss.screen` only | (a), as the one primitive for memory of events; h4 (counters, time since an event) uses it rather than a second one |
| 5 | Events on products as well as deliveries | (a) a product's events are the union over its deliveries (`lbc.since`: any program's cytology); (b) deliveries only | (a). It is what "screened in the last five years" means, and it lets a delivery's `where=` refer to its own product, which it cannot do to itself |
| 6 | What a stock target does above the target | (a) tops up only; (b) also stops the product for the lowest-priority holders | (a). Dropout is the product's `waning=`; STIsim's removal exists to force the stock onto data, which is calibration's job |
| 7 | Deliveries in the kernels (DF1) | (a) a program of delivery operations in the kernels, used by whole and stepped runs alike; (b) keep stepping from Python; (c) only tests and cures, without capacity or targets | (a), Numba first and JAX second. It is L, but it is what makes cascades fittable by NUTS and gradients, and it makes stepped and whole runs draw the same numbers |
| 8 | The name of a test's count of positives | (a) `positive`, the same name as the event, so `cases = a.positive + b.positive`; (b) keep `diagnosed` | (a). Every outcome is then a flow result and an event of one name, as a transition is |
| 9 | Values of a product by disease (HP7) | (a) a dict keyed by disease name (`efficacy=dict(hpv16=0.9975, hpv18=0.9975)`); (b) `ss.by(disease=...)` | (a), as `sensitivity=` is keyed by state; a key that is both a disease and an effect name raises |

## The shape of a cascade

STIsim's four cascades and HPVsim's algorithms, against what nesting can say:

| Cascade | Steps | What nesting cannot say |
|---|---|---|
| Syndromic management (STIsim) | care seeking for any of NG, CT, TV; assessment by sex and syndrome; one of four bundles, each treating several diseases | a trigger that is a union of three diseases' events; one product over several diseases; nothing is "positive" |
| Etiological testing (STIsim) | care seeking or screening; a test per disease; result after a delay; treat; notify partners; retest at three months | two triggers for one treatment; a retest triggered by a treatment long past |
| ANC (STIsim) | conception; a visit after a delay, if attended; HIV and syphilis tests; for syphilis, treat mother and fetus and notify partners; at delivery, test and treat the newborn | one visit runs several tests on the same people; a result acts months later, from memory |
| HIV care (STIsim) | three testing programs; diagnosed; linkage after a delay; ART to a stock target with a priority; dropout; re-engagement | a stock target; the steps are states people stay in |
| WHO algorithm 3 (HPVsim) | cytology with four results; HPV triage of ASCUS; colposcopy for abnormal or HPV-positive; ablation of HSIL | five-valued results; colposcopy has two parents |

So a cascade is steps that people flow through, each with its own trigger, coverage, delay and capacity (where the losses are), each producing events and memory that any later step can use. The proposal makes those steps ordinary deliveries and adds nothing else structural:

- **Peers, not nesting.** `ss.screen(test)` wraps the test it delivers, as `ss.vaccinate` wraps its vaccine. It does not wrap what comes after. A follow-up is a delivery with `on=` the event it follows.
- **Events.** An event is the set of people something happened to on this step. Transitions are events today (`on=sir.infection`). The proposal adds deliveries (the people who received it), their outcomes (a test's results by name; a cure's `cured` and `failed`), products (the union over their deliveries), `ss.Event` for modules written as code (`preg.conception`), unions (`a | b`), and an event narrowed by a condition (`preg.delivery & syph_test.positive.ever`).
- **Memory.** Every event has `.ever` (a condition), `.count` and `.since` (per-person quantities that compare with numbers and durations). They are kept only for events something reads.

**When a cascade should be states instead.** The EpiModel HIV vignette writes care as a dimension with transfers (`ss.transfer(care='dx -> art', rate=...)`). That is the right form when the steps are stages people stay in, entered and left at rates, and the data are the numbers in each stage. Deliveries are the right form when a step is an act with a product, an accuracy, a cost, a coverage, a capacity, or a target. The two combine: `ss.Delivery(move=dict(care='dx -> art'), target=..., priority=...)` is a stock target on a care dimension. `describe()` of a cascade written as deliveries says which care-like states it adds (below), so the choice is visible.

## Use cases

Each finding: what the package needs, how it does it in Starsim 3 code, and the v4 spelling under this proposal. `ppl` is `ss.People`.

**ST14, remembered test results.** STIsim's `STITest` keeps seven arrays per person (`tested`, `diagnosed`, `ti_tested`, `ti_scheduled`, `ti_positive`, `ti_negative`, `tests`), set by hand in `step()` and read by other interventions' `eligibility=` lambdas; HPVsim keeps `ti_screened` for "not screened in five years". v4's `ss.screen` holds `result` only while a delayed result is pending, so 0 people are positive at the end of a run that tested everyone (`sti/my/st_misc.py`). Proposed: nothing to declare.

```python
hiv_test = ss.screen(rdt, coverage=ss.probperyear(0.2), where=~rdt.positive.ever)   # retest only the undiagnosed
art      = ss.treat(art_drug, where=rdt.positive.ever & (rdt.positive.since >= ss.months(6)))  # linkage after six months
recent   = rdt.count >= 2                                                           # tested at least twice
```

Simpler than Starsim 3: the seven arrays, their bookkeeping, and `ti_scheduled` (a per-person schedule written by one intervention and read by another) go.

**ST15, `on=` over several events.** STIsim's syndromic management is triggered by care seeking for any of three diseases; it is one class (`SymptomaticTesting`, about 180 lines) that loops over diseases. v4 raises `TypeError: Several sources are a sum of states` for `on=a.infection | b.infection` (rechecked). Proposed: `|` of events is their union, each person once.

```python
seek = ng.careseek | ct.careseek | tv.careseek
vds  = ss.treat(bundle_ngcttv, on=seek, where=ppl.female, disease=[ng, ct, tv])     # vaginal discharge: treat all three
uds  = ss.treat(bundle_ngct,   on=seek, where=ppl.male,   disease=[ng, ct])         # urethral discharge
```

The syndrome-to-bundle map, which STIsim writes as `disease_treatment_map`, becomes one delivery per bundle with `where=`. An inexact assessment (STIsim's per-disease `sens`/`spec`) is a test over several diseases, `ss.Test(results=...)`, with each bundle `on=` one of its results.

**ST16, several follow-ups, and outcomes as events.** HPVsim's ablation is one treatment per genotype and a therapeutic vaccine is a cure plus immunity; `on_positive=[treat, vaccinate]` fails with `TypeError: 'list' object is not callable`. STIsim's drug resistance is acquired at treatment failure, which v4 cannot say. Proposed: any number of deliveries take `on=` the same event; a curing treatment's delivery has the events `cured` (the infected people it cured), `failed` (infected, not cured) and `unneeded` (not infected: overtreatment).

```python
cure   = ss.treat(txvx_cure,  on=colpo.lsil)
immune = ss.vaccinate(txvx_imm, on=colpo.lsil)
resist = ss.Delivery(move='I -> Ir', on=cure.failed, coverage=0.05)
```

**ST13, pregnancy events, and visits.** `on=preg.conception` raises `AttributeError`; STIsim's `ANCTest` (170 lines) schedules a visit at `ti_pregnant + randint(1, 9)` months, tests the attendees for HIV and syphilis, and routes positives by a map; `NewbornSyphTest` and `InfantHIVTest` are scheduled on the unborn through the maternal network. Proposed: `ss.Pregnancy` has the events `conception`, `delivery` and `loss` (`ss.Event`, below); a delivery of nothing is a visit; `delay=` on any delivery.

```python
visit = ss.Delivery(on=preg.conception, delay=ss.randint(ss.months(1), ss.months(8)), coverage=0.85)   # attendance (agents: a delay drawn per person)
hivt  = ss.screen(hiv_rdt, on=visit, disease=hiv)                                    # the same attendees take both tests
syph  = ss.screen(rpr, on=visit, disease=syph)
mom   = ss.treat(bpg, on=syph.positive, coverage=0.95)
fetus = ss.treat(bpg_fetal, on=mom, trace='prenatalnet')
baby  = ss.screen(rpr, on=preg.delivery & (syph.positive.since < ss.months(10)), trace='postnatalnet')
eid   = ss.screen(pcr, on=preg.delivery & hivt.positive.ever, delay=ss.weeks(6), trace='postnatalnet')
```

`trace=` finds the contacts when the trigger happens, then the delay applies; `where=` still applies to the contacts, which is why the condition on the mother goes on the event (`&`). About 500 lines of STIsim (`ANCTest`, `ANCSyphTest`, `NewbornSyphTest`, `InfantHIVTest`, `NewbornTreatment`, and `HIVTest`'s `newborn_test=` scheduling) become these seven lines.

**ST12, stock targets with a priority.** STIsim's `ART`, `VMMC` and `Prep` keep "N people (or a share of the eligible) on it", by age and sex, filling by CD4 over care seeking, and removing the highest-CD4 above the target; with `parse_coverage` and `resolve_coverage_targets` that is about 700 lines. In v4 it is two `deliver=` functions, agents only, whose state leaked between calibration trials. The priority matters: allocating by stage alone halved 2010 deaths. Proposed: `target=` and `priority=` on any delivery of a held product or a move (below).

```python
art  = ss.treat(art_drug, where=dx.ever, target=ss.timeseries(n_art), priority=[hiv.D, hiv.C, hiv.B])   # declared HIV: by stage, under every solver
art  = ss.treat(art_drug, where=dx.ever, target=n_art_by_age_sex, priority=lambda sim, uids: -sim.diseases.hiv.cd4[uids]/sim.diseases.hiv.care_seeking[uids])  # coded HIV: agents
vmmc = ss.Delivery(move=dict(circumcised=True), where=ppl.male, target=ss.by(age=[0, 0.1, 0.3, 0.25]))
```

Here `dx = hiv_test.positive | anc_test.positive | fsw_test.positive`. Re-engagement needs nothing: someone who drops out (`waning=` on the product) is eligible again, and the target fills the gap in priority order. With decision 6 (a), STIsim's removal above the target is not reproduced; it is the one behavior the port loses.

**HP24, tests with several results.** HPVsim's tests have three to five results, each with a probability by state, from a CSV (`products_dx.csv`), and `hierarchy` takes the most severe across genotypes (`dx.administer()`, about 80 lines, plus the hierarchy table). v4's workaround is a chain of binary tests on `on_negative=`, which reproduces the draw (abnormal 0.897 against 0.900 in `C`) but reports 35,397 tests for 20,000. Proposed: `ss.Test(results={category: probability by state}, default=...)`.

```python
lbc  = ss.Test(results=dict(abnormal=dict(C=0.9, X=1.0), ascus=dict(S=0.1, P=0.1, C=0.1)), default='normal')
lbc  = ss.Test(results=hpv.dx_table('lbc'))           # the same from HPVsim's CSV: a frame of state by result
cyto = ss.screen(lbc, disease=genotypes, coverage=0.2, interval=ss.years(1))
```

The categories are in order of severity, most severe first, and `default` is the last. Each person draws one uniform number per test and member disease; a person's result is the most severe over the family's members (HPVsim's hierarchy). The events and results are `cyto.abnormal`, `cyto.ascus`, `cyto.normal`. `sensitivity=`/`specificity=` remain the spelling of a two-valued test, with the results `positive` and `negative`.

**HP26, a delay on a treatment.** HPVsim's `treat_delay` holds people in a queue for a time; v4 has `delay=` only on `ss.Test`, and the pseudo-test workaround is reported as 1,452 tests. Proposed: `delay=` on every delivery triggered by `on=`, as on tests: the people triggered now receive it after the delay if they are still eligible then.

```python
treat = ss.treat(ablate, on=colpo.hsil, delay=ss.months(3))
```

**HP7, one product over several diseases.** HPVsim's bivalent vaccine protects against two genotypes at once. In v4 two deliveries of one `ss.Vaccine` for two diseases deliver 250 to the second and protect against the first only (rechecked), and `disease=[a, b]` raises `AttributeError`. Proposed: `disease=` takes a list or a family (h2's `ss.Diseases`); the product's values are one number for all, or a dict by disease name; holding it is one property of people (`has_bivalent`).

```python
bivalent = ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975))
ss.vaccinate(bivalent, at_age=9, where=ppl.female, coverage=0.8)                  # disease= from the product's keys
ablate   = ss.Treatment(cure=dict(P=0.936, C=0.936))                             # each member's P and C; 'hpv16.C' for one member
```

A product bound to one disease and delivered for another raises at `init()` (HP7's S part), naming `disease=[...]`.

**ST5 and HP25, cures.** Done: a cure moves people from every infected state, or raises asking for `to=`. The remaining part is a cure by state, `cure=dict(P=0.936, C=0.936)`, the same dict as `sensitivity=`, and across a family the same keys mean each member's state. A destination by state (BV to CST 1 or 3, late syphilis to serofast) is `to=dict(EL='Sx', LL='Sf')`.

**ST20, stratified standard results.** `ss.SIS` stratified by risk gives `ValueError: The result "prevalence" is not stratified` for `prevalence.select(risk='high')` (rechecked). Proposed: the standard `prevalence` of the library's diseases is built as the declared ratio of infected states over people, so it is stratified like any declared ratio. Every delivery result, including the new outcome results, is kept by stratum, as `delivered` is today.

**D27, names.** Follow-ups are named `screen_positive`, `screen_positive_positive`, and a product and its delivery cannot share a name (`ValueError: Cannot add module "vx"`, rechecked). Under peers every delivery is a variable, and is named from it; the product's module name gets a suffix only when it collides with a delivery's (`vx` the delivery, `vx_vaccine` the product), with no warning.

**D29, realized coverage.** Fixed in the current tree: a weekly screen at coverage 0.4 summarizes as 0.40 (agents 0.398), from the ratio of sums. Kept as a contract test.

**DF1, PL3, PL7, deliveries in the kernels.** Scheduled deliveries of a held product or a transfer run in the kernels (`ss.Delivery.tabulated`); tests and their follow-ups, cures, `capacity=`, deliveries in a model of several diseases, and crossed diseases step the run from Python. Ten years of daily ODE steps: no intervention 0.065 s, a vaccination 0.009 s, a treatment 0.60 s, a screen with a follow-up 0.72 s (`probe2.py`), about 170 µs a step. Stepped runs also draw deliveries differently from whole ones (Phase F's deferred part), and cannot be differentiated or run as many runs in one kernel call (KI2, KI23). PL7 (tables of time-varying values built whole on the stepped path) matters less once cascades no longer step. The proposal is in "Deliveries in the kernels" below.

**PL2, deliveries on diseases written as code.** `ss.vaccinate` and `ss.screen` on `ssl.AgentSIR` raise `UnknownDisease: ... the sim has no declared diseases`; STIsim's HIV and HPVsim's `hpv.HPV` are coded. Proposed: a module written as code is a delivery's `disease=` if it has the parts the product uses: `infected` (what a test detects by default), `rel_sus` and `rel_trans` (what a product's efficacy scales, as modifiers already reach coded diseases since D23), and `clear_infection(uids)` for a cure. A missing part raises naming it. Agents only, since a coded disease is.

## The vocabulary

### Events

An event is the set of people something happened to on the current step. These are events:

| Event | Who | Example |
|---|---|---|
| a transition | the people who went through it | `sir.infection` |
| a delivery | the people who received it | `visit`, `treat` |
| an outcome of a delivery | for a test, the people with each result, on the step it is available; for a curing treatment, `cured`, `failed`, `unneeded` | `cyto.ascus`, `tx.failed` |
| a product | the union over its deliveries, and its outcomes likewise | `lbc.since`, `rdt.positive` |
| `ss.Event` of a module written as code | whom the module says | `preg.conception` |
| a union | anyone in either | `a.positive \| b.positive` |
| an event and a condition | those in the event who meet the condition | `preg.delivery & hivt.positive.ever` |

`on=` takes any of them, or a function of the sim returning UIDs, as now. A state or condition alone still raises, pointing to `where=`. A misspelled outcome (`cyto.ascsu`) raises `AttributeError` listing the outcomes. Outcome names are taken from the product, so a test's categories cannot be `ever`, `count` or `since`.

**Order within a step.** Deliveries run in an order where each comes after the deliveries it is triggered by, so a follow-up acts on the step its trigger happens, as nesting does today. A cycle (a retest triggered by a treatment triggered by the test) must contain a `delay=`; one without raises, naming the cycle. A transition's event is the one of the last step, as now.

**`ss.Event`**, the one new class: an event that a module written as code fires. It is defined with the module's states and fired with the UIDs.

```python
class Pregnancy(ss.Demographics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_states(ss.Event('conception'), ss.Event('delivery'), ss.Event('loss'))

    def step(self):
        ...
        self.conception.fire(uids)
```

### Memory: `.ever`, `.count`, `.since`

Every event has three per-person readings, which the event's module keeps from the start of the run, but only if something in the sim reads them (found by walking every condition, rate and priority at `init()`):

| Reading | Is | Compared with |
|---|---|---|
| `ev.ever` | a condition: it has happened to them at least once | (it is a condition) |
| `ev.count` | how many times | numbers: `ev.count >= 2` |
| `ev.since` | the time since the last time (infinite if never) | durations: `ev.since < ss.years(5)` |

They are quantities like a state's attributes, so they go anywhere a condition or a per-person value goes: `where=`, `priority=`, `ss.modify(where=)`, and a transition's `where=` (`hpv.reinfection` slower for `ablate.cured.ever`). As agents they are one 16-bit step number and, if read, one 16-bit count per person (decisions item 21's timers). For **h4**: counters (`preg.delivery.count` is parity) and time since an event (`preg.delivery.since` for postpartum) should be these readings of an event, so that FPsim's behavior and a cascade's memory are one mechanism; h4's regressions and tables then take them as inputs.

### Deliveries: new keywords

| Keyword | On | Meaning |
|---|---|---|
| `on=` | every delivery | now also a delivery, an outcome, a product, an `ss.Event`, `\|`, and `& condition` |
| `delay=` | a delivery with `on=` | the time from the trigger to delivery: a duration, or a distribution of durations (agents); the people still eligible then receive it, and someone waiting is not triggered again |
| `target=` | a delivery of a held product, or a move | the number to keep holding it among the eligible: a number, a share of the eligible (below 1), a time series, or `ss.by(...)` for one target per stratum |
| `priority=` | with `target=` or `capacity=` | who is served first when there is not enough: a condition, or a list of them served in order; or a function of `(sim, uids)` returning a score, highest first (agents); ties are chosen at random, as now |
| `disease=` | every delivery | now also a list or a family |

**`target=`** is a stock, evaluated every step between `start` and `stop`: if fewer than the target hold the product among those eligible, the gap is delivered to eligible people who do not hold it, in priority order, capped by `capacity=` if given. Above the target nothing happens (decision 6). A share is a share of the eligible, holders included (STIsim's `p_art`); `ss.by(age=..., sex=...)` gives a target per stratum, filled within it. A target is not a schedule, so it does not combine with `at=`, `interval=` or `on=`, which raise. **`priority=`** is otherwise random selection; with `capacity=` it is "while stocks last, the sickest first".

### Products

| Product | New |
|---|---|
| `ss.Test` | `results={category: probability}` with `default=`, each probability a number or a dict by state (each a number or `ss.by`); or a frame of state by result; categories in order of severity |
| `ss.Treatment` | `cure=` a dict by state; `to=` a dict by source state |
| all | values keyed by disease name, and `disease=` a list or a family, from the product's keys if not given |

A test's results must add up to at most 1 for each state, else `ValueError` naming the state. Under a family with cross-immunity (h2), the most severe result over the members is exact under every solver; without a crossing it needs agents (below).

### Results

Each delivery keeps `delivered`, `eligible`, `coverage` and `cost` as now, by stratum. Each outcome adds a flow result of its name: a two-valued test `positive`, `negative`, `false_positive`, `false_negative`; a categorical test one per category; a curing treatment `cured`, `failed`, `unneeded`. This replaces `diagnosed` (decision 8) and gives STIsim's accuracy and treatment results (14 series in Starsim 3 against 3 today), by sex. A delivery with `target=` adds `holding` (the stock) and `target`. Reported cases are `a.positive + b.positive`, as the spec's `cases` example already reads with `diagnosed`.

## Semantics under every solver

| Feature | ODE | Binomial | Gillespie | Agents |
|---|---|---|---|---|
| an event | the expected number in each compartment and group | whole numbers, drawn | as binomial (deliveries happen at the time points) | UIDs |
| `\|` of events | the union in each compartment and group, for diseases tracked jointly (one disease, or a crossing); otherwise `CapabilityError` | same | same | union of UIDs |
| `.ever` | a two-level property of people, `ever_<event>` | same | same | from the step number |
| `.count >= k` | `k + 1` levels, the last kept | same | same | a 16-bit count |
| `.since` against `d` | levels of time since, as a product whose protection fades by a function (one step each up to 24 levels, then several steps each, with a note) | same | same | from the step number |
| categorical test | each compartment's shares by category | multinomial | multinomial | one uniform draw per person and member |
| worst over a family | exact in a crossing; otherwise `CapabilityError` | same | same | per person |
| `delay=` fixed | pending levels of one step each (as a test's delay today) | same | same | a per-person step |
| `delay=` a distribution | an exponential only, as a rate out of one pending level; otherwise `CapabilityError` | same | same | drawn per person |
| `target=`, `priority=` by condition | filled class by class, shared within a class in proportion to the eligible | a multivariate hypergeometric draw within a class | same | ranked |
| `priority=` a function of uids | `CapabilityError` | same | same | ranked |
| `ss.Event` of a coded module, coded `disease=`, `trace=` | `CapabilityError` (the module runs as agents) | same | same | yes |

`solver='auto'` picks agents for anything in the last rows, as now; an explicit compartmental solver stops with the reason, naming the delivery and the keyword. Every level of memory multiplies the groups of every disease, so a property is added only for readings something uses, and `describe()` prints each with its levels and a note when the groups grow more than tenfold.

**`describe()`** prints each delivery on one line, with its trigger and what it adds:

```
cyto:   screen lbc (abnormal, ascus, else normal; worst over hpv16, hpv18, hi5, ohr) in rounds every 1 year, coverage 0.2, where women & lbc.since >= 5 years
triage: screen hpvdna on cyto.ascus, coverage 0.9
treat:  treat ablate (cures P 0.936, C 0.936 to S) on cyto.abnormal | triage.positive, after 3 months; capacity 5,000 a year, served cyto.abnormal.ever first
adds as properties of people: since_lbc (11 levels: 0.5 years each to 5 years), pending_treat (7 levels)
```

**`to_spec()`**: a delivery's `on` is written as the text of the event expression (`"cyto.ascus | triage.positive"`), `delay`, `target` and `priority` as typed values and condition text, and a test's `results` as a table; the properties memory adds are derived, not stored, so the hash depends only on what was written. **`to_code()`** writes products, then deliveries in the order of their triggers, each as a variable named after the delivery, so `on=cyto.ascus` is valid code; a function (`priority=lambda ...`) is written with its source, as now.

## Deliveries in the kernels

Today the kernels take a delivery as one move between groups (`ss.kernels.Shifts`: `due`, `sel`, `cov`, `to`, `rest`). The proposal replaces that with a small program per time point, a list of operations on registers that each hold a number of people per compartment and group, run at the start of each step. The deliveries are built into it at `init()`, in their trigger order.

| Operation | Does |
|---|---|
| `SELECT` | the eligible: a register from the compartments, times a share per compartment and group (states, stratifications, memory levels) |
| `EVENT` | a transition's people from the last step's flows, in the destination's compartments (as `event_frac()` does now) |
| `THIN` | times coverage, from a table by time |
| `CAP` | to a capacity from a table, in priority classes |
| `TARGET` | holders counted, the gap filled from the register in priority classes |
| `SHIFT` | the register's people moved to other groups (holding, offered, memory, pending levels), and the register with them |
| `MOVE` | the register's people moved to another state's entry compartments, with branches and maps between dimensions, and the flow credited (a cure, `move=`) |
| `SPLIT` | a test: one register into one per result, by shares per compartment and group |
| `UNION`, `MASK` | `\|` of two registers; a register times a condition's shares |
| `TALLY` | a register's total per group into a result |

Under the ODE every operation is an expected value; under the stochastic solvers `THIN` and `SPLIT` draw binomials and multinomials, and `CAP` and `TARGET` multivariate hypergeometrics, from the kernel's stream. The stepped path calls the same kernel function at each step, so a stepped and a whole run draw the same numbers. The many-run kernel (`run_many`) and the crossing's whole run (PL3) take the same program, a crossing's compartments being joint. What stays in Python: `when=`, `deliver=`, coverage or capacity given as a function of the sim, and anything agents-only. JAX: the same operations in `gradients.py`, second (`CAP` and `TARGET` have kinks, so gradients are piecewise, which the docs say).

This is the design `deferred.md` sketched (2026-10-16), with targets and categorical results added. Expected cost: a cascade of four deliveries a few µs a step against about 170 µs now.

## Fit with what exists

**Reused.** `on=`, `where=`, `coverage=`, `capacity=`, `trace=`, `move=` and `ss.Delivery` unchanged; events are what `on=sir.infection` already is; memory as compartments uses the existing mechanism for levels of time since delivery (`Held.init_stages`) and for pending results; `target=` by stratum uses `ss.by`; `|` and `&` as for conditions; a visit is `ss.Delivery` with no product, rather than a new `ss.attend`.

**New public names, each against extending an existing one.**
- `ss.Event`: modules written as code need a way to publish an event with memory; `on=` a function of the sim already gives the people but no name, memory, or `describe()` line. Defined with `define_states()`, so no new method.
- `.ever`, `.count`, `.since` on events: the alternative, named states on `ss.screen`, covers tests only and leaves h4 to invent its own.
- Outcome attributes (`.positive`, `.cured`, a test's categories): named by the product, as transitions are named by the disease.
- Keywords `target=`, `priority=`, `delay=` on deliveries; `results=`, `default=` on `ss.Test`. `capacity=` cannot carry a stock (it is a flow), and coverage is per round.

**Removed or replaced.** `on_positive=` and `on_negative=` (replaced by `on=test.positive`; a removed name raises saying what to write, and the migration script rewrites the nested form into peers); the `diagnosed` result (now `positive`); `ss.Delivery`'s public `result` and `ti_result` states (internal pending results); automatic follow-up names (`screen_positive`); `ss.kernels.Shifts` (replaced by the delivery program). Nothing is added for compatibility.

**Spec changes.** [08-interventions.md](../../design/08-interventions.md): "Follow-ups" becomes "Cascades" (peers, events, memory, order), the delivery keywords table gains `target=`, `priority=`, `delay=` and loses `on_positive=`/`on_negative=`, "Tests" and "Treatments" gain categorical results, cure by state and values by disease, "What a delivery reports" lists outcome results, and the solver table gains the rows above. [09-results.md](../../design/09-results.md) and [19-open-questions.md](../../design/19-open-questions.md): `cases = a.positive + b.positive`. [12-backends.md](../../design/12-backends.md): the delivery program. [14-model-spec.md](../../design/14-model-spec.md): deliveries' `on`, `target`, `priority`, `delay`, and test results. [17-vocabulary.md](../../design/17-vocabulary.md): the names above. [18-conformance.md](../../design/18-conformance.md): the contracts below, and item 13 reworded. [01-objects.md](../../design/01-objects.md): `ss.Event` among module parts. [10-execution.md](../../design/10-execution.md): deliveries ordered by trigger.

## Implementation plan

| Step | Size | Files | Tests |
|---|---|---|---|
| 1. Events: deliveries, outcomes and products as events, `\|` and `& condition`, `ss.Event`, trigger order and the cycle error; `ss.Pregnancy`'s three events (ST13, ST15, ST16) | M | `interventions.py`, `conditions.py`, `products.py`, `arrays.py`, `demographics.py`, `loop.py` | a union fires once per person; a follow-up on the step of its trigger under agents and ODE; ANC: the same attendees take both tests |
| 2. Memory: `.ever`, `.count`, `.since`, found at `init()`, as agents and as levels (ST14) | M | `conditions.py`, `interventions.py`, `solvers/engine.py`, `solvers/compartmental.py` | nobody screened twice within five years; positives at the end equal to those ever positive; `since` as levels against agents at a fine step |
| 3. Peers replace nesting: remove `on_positive=`/`on_negative=`, names from variables (D27), `diagnosed` to `positive`, outcome results by stratum; the migration script's rewrite | S | `interventions.py`, `migrate.py`, `skills/migrate-starsim3`, docs | the spec's screening examples give the same numbers as before; names and results as written |
| 4. `delay=` on every delivery (HP26) | S | `interventions.py` | a treatment three months after the test, under every solver |
| 5. Categorical tests, cure by state, values by disease, `disease=` a list or family, the HP7 error (HP24, HP25, HP7) | M | `products.py`, `interventions.py` | 20,000 cytologies report 20,000 tests and cost 200,000; the shares by state match HPVsim's table within 1%; a bivalent vaccine protects against both; worst over a crossed family as ODE against agents |
| 6. `target=`, `priority=` (ST12) | M | `interventions.py` | the stock follows a time series within 1% while enough are eligible; the lowest priority is served last; by stratum |
| 7. Coded diseases as `disease=` (PL2) | S | `interventions.py`, `products.py` | screen and treat on `ssl.AgentSIR` |
| 8. Standard `prevalence` as a ratio (ST20) | S | `solvers/engine.py`, `diseases.py` | `ss.SIS(...).prevalence.select(risk='high')` |
| 9. The delivery program in the kernels (DF1), in `run`, `run_many`, the stepped path and the crossing (PL3); tables a part at a time on the stepped path (PL7) | L | `kernels.py`, `solvers/compartmental.py`, `solvers/crossing.py`, `interventions.py` | whole against stepped: identical for the ODE, the same draws for binomial; a cascade with categorical test, delay, capacity and target, ODE against the mean of 200 agent runs; the cascade ODE at under twice the time of no intervention |
| 10. The same in JAX (KI23) | M | `gradients.py` | `sim.gradient()` of a screen-and-treat model against finite differences |
| 11. Spec and docs | S | `v4_design/design/*.md`, `docs/user_guide/modules_products.qmd`, interventions tutorials | the docs' examples run |

Steps 1 to 3 come first and in that order; 4 to 8 are independent of each other; 9 needs 1 to 6. A contract test, `test_contracts.py::test_cascade`, runs one cascade (union trigger, categorical test, delay, capacity with priority, target, memory) under all four solvers and checks the agreement and the reported numbers above.

**What the ports then delete** (estimated from the class sizes): STIsim, about 2,200 of its 3,700 lines of interventions: `STITest` and its six subclasses, `ANCTest`, `STITreatment` and its two subclasses, `SymptomaticTesting` and `SyndromicManagement`, the target machinery of `ART`, `VMMC` and `Prep`, and `parse_coverage`/`resolve_coverage_targets` (data loading into `ss.timeseries` and `ss.by` stays). `PartnerNotification` (300 lines) is `trace=` once its partner rules are; that is outside this design. HPVsim: the wrapper classes in `interventions.py` (most of its 723 lines), `dx.administer()` and the hierarchy, `treat_delay`, `treat_num`'s pseudo-test, and the `linked_txvx` eligibility lambdas; the CSVs stay, read into tables. Both packages' cascade examples become one delivery per step, as in the use cases.

## Out of scope, and the other Phase H documents

**Out of scope.** A capacity queue in arrival order (D31) is `priority=ev.since` over the people still eligible, which a routine delivery with capacity already leaves unoffered, so it needs no keyword; whether that is enough is D31's question. Partner rules for notification beyond `trace=`; budget allocation across programs (`ss.optimize`); `after=preg.delivery` as a dwell (ST18); optimal control.

**h2, a family of diseases.** Assumed: `ss.Diseases(*members)` with members by name, the same state names in each, and a crossing in compartmental runs when the family has cross-immunity. This design uses `disease=family`, keys `'hpv16.C'` and `'C'` (each member), and "worst over the members", which is exact only in a crossing.

**h3, per-act transmission.** Assumed: a product's `acquisition` efficacy scales `rel_sus`, which multiplies the per-act probability, so PrEP and VMMC need nothing more from h3.

**h4, behavior and dimensions over people's properties.** h4 should use `.ever`, `.count` and `.since` of events for counters and time since an event (decision 4), and `ss.Event` for events of FPsim's coded modules. Assumed from h4: sex and age as dimensions in compartmental runs (D15), so `where=ppl.female` and `target=ss.by(age=..., sex=...)` run under every solver; until then they are agents only, as now.
