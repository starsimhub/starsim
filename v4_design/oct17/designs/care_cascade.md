# The care cascade

Design pass H1 for Phase H of [implementation_oct18.md](../implementation_oct18.md) (decision D-F: one design document before code). Written 2026-10-19 against the `v4` working tree. Findings covered: ST12 to ST16, ST20, HP24, HP26, D27, D29, DF1, PL2, PL3, PL7, with ST5/HP25 (cures, done) and HP7 (one product over several diseases) for the interface. Probes are in `wt/scratch-h1/` of the session scratchpad.

## Revision, 2026-10-19

The second draft of the user spelling, after Cliff's `[CK:]` notes on the first ([oct19/implementation.md](../../oct19/implementation.md), H0). What changed:

- **`ss.cascade()` is added** (decision 2). A step is written `who >> what`: a trigger or a condition on the left, a delivery on the right, as `S >> I @ beta*I` is "from >> to @ how fast". The cascade groups the steps, names them, and gives a table of who reaches each step and who is lost. See "The cascade, as a user writes it".
- **Products and interventions can no longer be confused.** The test product `ss.Test` becomes `ss.Diagnostic`. Products are nouns for things (`ss.Diagnostic`, `ss.Treatment`, `ss.Vaccine`); deliveries are verbs (`ss.screen`, `ss.treat`, `ss.vaccinate`, `ss.visit`). No two names differ only by case.
- **Full state names** in every example (`infected`, `cin`, `cancer`, not `P`, `C`, `X`).
- **`default='normal'` is gone.** A test's last result is written `normal=None`, "everyone else", as `None` is the rest in a branch's `p=`.
- **Resistance at drug failure is a transition of the disease** with `on=` an event (`resistance = ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05)`), not a delivery.
- **A clinic visit is `ss.visit()`**, a step with no product, not "a delivery of nothing".
- **`on=` takes `|`, `&`, and lists** with `operator='any'` (the default, the same as `|`) or `operator='all'` (the same as `&`) (decision 3).
- **`count_events=True`** keeps every event's memory and, as agents, a log of events per person, for analysis after the run (decision 4).
- **`above=` on a stock target** says what happens above it: `'keep'` (the default, as approved), `'stop'`, or a rate (decision 6).
- **A move with no wait** ("immediately") has a section of its own, since the family, the cascade, the behavior vocabulary, ST26, HP2, and HP16 all need it: `dur=0`.
- **Thanzi la Onse** is not reviewed yet. Cliff is working on TLO (`/home/cliffk/idm/git/TLOmodel`) with the lingua franca library first; this design is revisited after that (its appointments, footprints, and consumables are what a cascade's `capacity=` and `cost=` would grow into).
- The family is `ss.Strains` (from `disease_family.md`'s revision).

The decisions still open are in the last section.

## Summary

**The problem.** A cascade in v4 is a test with its follow-ups nested inside it (`ss.screen(test, on_positive=ss.treat(...))`), and nothing about it is remembered once the step is over. STIsim's four real cascades and HPVsim's seven WHO algorithms are not trees: a trigger can be the union of three diseases' events (ST15), one result feeds several actions months apart (ST16, ST14), a test has three to five results (HP24), and ART, VMMC and PrEP are kept at a stock target with a priority (ST12). Today each of these is a function, an extra class, or a chain of fake tests: HPVsim's cytology reports 35,397 tests for 20,000 done (costs 77% high), a bivalent vaccine silently protects against one genotype (HP7: still 250 delivered for the second disease, with no protection), and a screen-and-treat cascade forces the ODE to step from Python, taking 0.72 s for ten years of daily steps against 0.009 s for a vaccination the kernels take (DF1, measured here).

**The proposal, as a user writes it.** Deliveries are peers linked by events, written as steps `who >> what` and grouped by `ss.cascade()`. Every delivery, every outcome of one, every product, and every transition is an event; and every event remembers, per person, whether, how often, and how long ago it happened (`.ever`, `.count`, `.since`).

```python
# Products: the things delivered
cytology = ss.Diagnostic(results=dict(
    high_grade = dict(cin=0.9, cancer=1.0),                  # the chance of each result, by state; most severe first (HP24)
    ascus      = dict(susceptible=0.1, infected=0.1, cin=0.1),
    normal     = None,                                       # everyone else
), cost=10)
hpv_dna  = ss.Diagnostic(sensitivity=dict(infected=0.95, cin=0.95, cancer=1.0), specificity=0.9)
ablation = ss.Treatment(cure=dict(infected=0.936, cin=0.936))  # cure by state (HP25) # [CK: if same for all groups should be able to do cure=0.936, but here i guess you're saying it's only for these two]

# Who is eligible
women = ss.ppl.female & (ss.ppl.age >= 25) & (ss.ppl.age < 65)
due   = women & (cytology.since >= ss.years(5))             # remembered per person (ST14)

# The cascade: who >> what
screening = due >> ss.screen(cytology, coverage=0.2, interval=ss.years(1))
triage    = screening.ascus >> ss.screen(hpv_dna, coverage=0.9)                     # a follow-up is a peer (ST16)
treatment = (screening.high_grade | triage.positive) >> ss.treat(ablation, delay=ss.months(3), capacity=ss.peryear(5_000), priority=screening.high_grade)  # a union (ST15), a delay (HP26), limited supply
cervical  = ss.cascade(screening, triage, treatment)
# [CK: no, i don't like this. suggest this:
screening = ss.screen(who=due, what=cytology, coverage=0.2, interval=ss.years(1))
triage    = ss.screen(who=screening.ascus, what=hpv_dna, coverage=0.9)
treatment = ss.treat(who=(screening.high_grade | triage.positive), what=ablation, delay=ss.months(3), capacity=ss.peryear(5_000), priority=screening.high_grade)
cervical = ss.cascade(screening, triage, treatment)
# ]

sim = ss.Sim(diseases=hpv, interventions=cervical)          # hpv an ss.Strains: each test takes the worst result across genotypes
sim.results.cervical.to_df()                                # reached and lost at each step
# [CK: how does interventions=cervical differ from interventions=[screening, triage, treatment]? ]
```

The same for HIV and STIs:

```python
diagnosed = hiv_rdt.positive.ever | anc_rdt.positive.ever                          # by either program
testing   = ~diagnosed >> ss.screen(hiv_rdt, coverage=ss.probperyear(0.2))
anc_visit = preg.conception >> ss.visit(delay=ss.randint(ss.months(1), ss.months(8)), coverage=0.85)  # attendance (ST13)
anc_test  = anc_visit >> ss.screen(anc_rdt)
art       = diagnosed >> ss.treat(art_drug, target=ss.timeseries(n_art), priority=[hiv.aids, hiv.chronic])  # a stock target (ST12)
hiv_care  = ss.cascade(testing, anc_visit, anc_test, art)

ng = ss.Disease(susceptible, infected, resistant,
    ...,
    resistance = ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05),  # resistance at drug failure is natural history (ST16)
)
```

Under a compartmental solver the same cascade is built into a short program of delivery operations that the kernels run between steps, so it no longer stops a whole run (DF1).

**Decisions for Cliff, first round (answered).** Kept as the record; the last column says what this draft does with each.

| # | Decision | Options | Recommendation | This draft |
|---|---|---|---|---|
| 1 | Follow-ups: nested or peers? | (a) peers only: `on=screen.positive`, and `on_positive=`/`on_negative=` removed; (b) both; (c) peers inside a new `ss.cascade(...)` container | (a). One spelling; the container adds a name and a scope that variables already give (decision 2) [CK: ok] | Peers, which `ss.cascade()` groups (decision 2) |
| 2 | A cascade container (`ss.cascade(visit=..., test=..., treat=...)`) | (a) none: a cascade is a list of deliveries; (b) add it now, for names, string references and a cascade table | (a), and revisit if a cascade results table ("lost at each step") is wanted; `ss.parallel` and `sim.new()` already take lists [CK: i think we should do `ss.cascade()`, but make it compelling: make it super ultra clear what the logic is, as clear as the `S >> I @ beta*I` logic for diseases. care cascades are some of the most complex parts of disease models, so we if we can make that better, this would be a huge benefit.] | Added: `who >> what` steps, `ss.cascade(...)`, and its table |
| 3 | Several events in `on=` | (a) `\|` only (`on=a.infection \| b.infection`); (b) a list too | (a). `\|` is already the union of conditions; a list raises, pointing to `\|`  [CK: no -- \| or a list are valid and equivalent, but & is also a valid operator. `[a,b], operator='any' == a | b`, `a,b], operator='all' == a & b`.] | As Cliff says |
| 4 | Remembered results | (a) `.ever`, `.count`, `.since` on every event, kept only when something reads them; (b) fixed states (`tested`, `positive`, `ti_positive`) on `ss.screen` only | (a), as the one primitive for memory of events; h4 (counters, time since an event) uses it rather than a second one [CK: ok, but maybe have an optional count_events=True flag for when people want to save results for later analyses] | (a), with `count_events=True` |
| 5 | Events on products as well as deliveries | (a) a product's events are the union over its deliveries (`lbc.since`: any program's cytology); (b) deliveries only | (a). It is what "screened in the last five years" means, and it lets a delivery's `where=` refer to its own product, which it cannot do to itself  [CK: ok] | Unchanged |
| 6 | What a stock target does above the target | (a) tops up only; (b) also stops the product for the lowest-priority holders | (a). Dropout is the product's `waning=`; STIsim's removal exists to force the stock onto data, which is calibration's job [CK: ok, but might need some machinery to allow user choices here] | (a) by default, and `above=` for the other choices |
| 7 | Deliveries in the kernels (DF1) | (a) a program of delivery operations in the kernels, used by whole and stepped runs alike; (b) keep stepping from Python; (c) only tests and cures, without capacity or targets | (a), Numba first and JAX second. It is L, but it is what makes cascades fittable by NUTS and gradients, and it makes stepped and whole runs draw the same numbers [CK: ok] | Unchanged |
| 8 | The name of a test's count of positives | (a) `positive`, the same name as the event, so `cases = a.positive + b.positive`; (b) keep `diagnosed` | (a). Every outcome is then a flow result and an event of one name, as a transition is [CK: ok] | Unchanged |
| 9 | Values of a product by disease (HP7) | (a) a dict keyed by disease name (`efficacy=dict(hpv16=0.9975, hpv18=0.9975)`); (b) `ss.by(disease=...)` | (a), as `sensitivity=` is keyed by state; a key that is both a disease and an effect name raises [CK: ok] | Unchanged |

Cliff's note on the first draft's example, kept as the record: "[CK: i don't love this ... it's confusing that ss.Test() is the product and ss.test() is the intervention. i don't know what C, X, S, P, etc mean (precancer and cancer for P and C, maybe?). i don't know what default='normal' means. i don't think drug failure leading to resistance should be ss.Delivery, that makes no senes to me. nor should a visit be a 'delivery of nothing'. maybe we should include thanzi la onse in the review...]"

## The cascade, as a user writes it

### A step: `who >> what`

A step is one line. The left of `>>` says who; the right says what they get.

| Left of `>>` | Means | The same as |
|---|---|---|
| an event: `screening.ascus`, `preg.conception`, `hpv.infection` | the people it happened to on this step | `on=` |
| several events: `a \| b`, `[a, b]` | anyone in either | `on=[a, b]` or `on=a \| b` |
| events together: `a & b` | anyone in both on the same step | `on=[a, b], operator='all'` |
| an event and a condition: `preg.delivery & hiv_rdt.positive.ever` | those in the event who meet the condition | `on=preg.delivery, where=hiv_rdt.positive.ever` |
| a condition: `due`, `~diagnosed`, `hiv.infected` | the people who meet it, offered by the delivery's schedule (`interval=`, `at=`, routine) | `where=` |

The right of `>>` is any delivery: `ss.screen`, `ss.treat`, `ss.vaccinate`, `ss.visit`, or `ss.Delivery(move=...)`. The step is the delivery, with `on=` or `where=` set from the left, so `screening.ascus >> ss.screen(hpv_dna)` and `ss.screen(hpv_dna, on=screening.ascus)` are the same object. Both spellings stay: `>>` reads as a flowchart; the keywords are what `to_code()` can always write, and what a delivery outside a cascade uses.

**Why `>>`.** It is the arrow diseases use: `S >> I` is "people go from S to I"; `screening.ascus >> ss.screen(hpv_dna)` is "people with ASCUS go to triage". Read down the column of steps, a cascade is its flowchart.

**Parentheses.** In Python, `>>` binds tighter than `|`, `&`, and comparisons, so a trigger made of several parts is put in parentheses, as `(S | R) >> P` is for diseases: `(screening.high_grade | triage.positive) >> ss.treat(...)`. Written without them, `a | b >> ss.treat(...)` would join `a` with a delivery; a delivery refuses `|`, `&`, and comparisons, with `TypeError: put the trigger in parentheses: (a | b) >> ss.treat(...)`. The examples name longer conditions as variables (`due`, `diagnosed`), which also reads better.

### `ss.cascade(*steps, name=None)`

```python
cervical = ss.cascade(screening, triage, treatment)
```

- **Steps** are deliveries, named from their variables (D27). Their order is the order of the flowchart, used for `describe()` and the table; the order they run in within a step is by trigger (below), whatever is written.
- **What it adds.** A name for the group; the table of who reaches each step and who is lost (below); a `describe()` block that prints the flowchart; and one object to pass to `interventions=`, `sim.new()`, and scenarios (`ss.change` on `cervical.treatment.capacity`). It adds no behavior: a cascade runs exactly as its steps would if passed as a list.
- **Steps refer to each other as variables**, as transitions refer to states. A step's events (`screening.ascus`) are its own; a product's (`cytology.ascus`) are the union over every step that delivers it (decision 5). A step in another cascade can be named: cascades are not scopes.
- **Results.** Each step keeps its results under its own name, as now (`sim.results.treatment.delivered`). The cascade adds `sim.results.cervical`, whose `to_df()` is the table below and whose `plot()` is a funnel.

```text
step        trigger                                   eligible  reached  lost (coverage)  lost (capacity)  outcomes
screening   due, every 1 year                          410,200   82,040           328,160                0  high_grade 1,804; ascus 4,100; normal 76,136
triage      screening.ascus                              4,100    3,690               410                0  positive 1,107; negative 2,583
treatment   screening.high_grade | triage.positive       2,911    2,744                 0              167  cured 2,407; failed 165; unneeded 172
```

`describe()` prints it as the user wrote it:

```text
cervical (cascade of 3 steps):
  screening   due >> screen cytology (high_grade, ascus, else normal; worst over hpv16, hpv18, hi5, ohr) every 1 year, coverage 0.2
  triage      screening.ascus >> screen hpv_dna, coverage 0.9
  treatment   (screening.high_grade | triage.positive) >> treat ablation (cures infected 0.936, cin 0.936) after 3 months; capacity 5,000 a year, screening.high_grade first
  adds as properties of people: since_cytology (11 levels: 0.5 years each to 5 years), pending_treatment (3 levels)
```

### Products and steps: nouns and verbs

| Product (a thing) | Delivered by (a verb) | Outcomes (events and results) |
|---|---|---|
| `ss.Diagnostic` (was `ss.Test`) | `ss.screen` | `positive`, `negative` (two results), or one per category |
| `ss.Treatment` | `ss.treat` | `cured`, `failed`, `unneeded` for a cure; none for a held treatment (ART) |
| `ss.Vaccine` | `ss.vaccinate` | none |
| none | `ss.visit` | the attendees are the event |
| none | `ss.Delivery(move=...)` | the movers are the event |

A product's own keywords may also be given to its delivery, which then makes the product (`ss.screen(sensitivity=0.9)`, as today).

**`ss.visit(**delivery keywords, cost=None)`** is a step that people attend, with no product: its event is the attendees. ANC, a clinic visit for symptoms, a follow-up appointment. It takes every delivery keyword (`delay=`, `coverage=`, `capacity=`, `priority=`), and `cost=` per visit. Several steps take `on=` the same visit, so one visit runs several tests on the same people. It replaces the first draft's `ss.Delivery` with no product.

**Resistance and other outcomes in the disease.** Something that happens to a person's infection because of an event (resistance at treatment failure, a relapse on stopping a drug) is natural history, so it is a transition of the disease, triggered by the event: `ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05)`. A transition with `on=` happens to those people on the step of the event (with `p=` of them, and `where=` as for any transition); it has no rate or dwell. It is the transition form of "immediately" (below), with the event as its trigger. The first draft's `ss.Delivery(move='I -> Ir', on=...)` still works, but the docs use the transition.

## The shape of a cascade

STIsim's four cascades and HPVsim's algorithms, against what nesting can say:

| Cascade | Steps | What nesting cannot say |
|---|---|---|
| Syndromic management (STIsim) | care seeking for any of NG, CT, TV; assessment by sex and syndrome; one of four bundles, each treating several diseases | a trigger that is a union of three diseases' events; one product over several diseases; nothing is "positive" |
| Etiological testing (STIsim) | care seeking or screening; a test per disease; result after a delay; treat; notify partners; retest at three months | two triggers for one treatment; a retest triggered by a treatment long past |
| ANC (STIsim) | conception; a visit after a delay, if attended; HIV and syphilis tests; for syphilis, treat mother and fetus and notify partners; at delivery, test and treat the newborn | one visit runs several tests on the same people; a result acts months later, from memory |
| HIV care (STIsim) | three testing programs; diagnosed; linkage after a delay; ART to a stock target with a priority; dropout; re-engagement | a stock target; the steps are states people stay in |
| WHO algorithm 3 (HPVsim) | cytology with four results; HPV triage of ASCUS; colposcopy for abnormal or HPV-positive; ablation of HSIL | five-valued results; colposcopy has two parents |

So a cascade is steps that people flow through, each with its own trigger, coverage, delay and capacity (where the losses are), each producing events and memory that any later step can use. The proposal makes those steps ordinary deliveries, written `who >> what`, and adds nothing else structural:

- **Peers, not nesting.** `ss.screen(diagnostic)` wraps the diagnostic it delivers, as `ss.vaccinate` wraps its vaccine. It does not wrap what comes after. A follow-up is a step whose trigger is the event it follows.
- **Events.** An event is the set of people something happened to on this step. Transitions are events today (`on=sir.infection`). The proposal adds deliveries (the people who received it), their outcomes (a test's results by name; a cure's `cured` and `failed`), products (the union over their deliveries), visits, `ss.Event` for modules written as code (`preg.conception`), unions (`a | b`), intersections (`a & b`), and an event narrowed by a condition (`preg.delivery & syph_test.positive.ever`).
- **Memory.** Every event has `.ever` (a condition), `.count` and `.since` (per-person quantities that compare with numbers and durations). They are kept only for events something reads, or for all with `count_events=True`.

**When a cascade should be states instead.** The EpiModel HIV vignette writes care as a dimension with transfers (`ss.transfer(care='dx -> art', rate=...)`). That is the right form when the steps are stages people stay in, entered and left at rates, and the data are the numbers in each stage. Deliveries are the right form when a step is an act with a product, an accuracy, a cost, a coverage, a capacity, or a target. The two combine: `diagnosed >> ss.Delivery(move=dict(care='diagnosed -> on_art'), target=..., priority=...)` is a stock target on a care dimension. `describe()` of a cascade says which care-like states it adds (below), so the choice is visible.

## Use cases

Each finding: what the package needs, how it does it in Starsim 3 code, and the v4 spelling under this proposal. `ss.ppl` is the lazy reference to people's properties (`behavior_vocabulary.md`).

**ST14, remembered test results.** STIsim's `STITest` keeps seven arrays per person (`tested`, `diagnosed`, `ti_tested`, `ti_scheduled`, `ti_positive`, `ti_negative`, `tests`), set by hand in `step()` and read by other interventions' `eligibility=` lambdas; HPVsim keeps `ti_screened` for "not screened in five years". v4's `ss.screen` holds `result` only while a delayed result is pending, so 0 people are positive at the end of a run that tested everyone (`sti/my/st_misc.py`). Proposed: nothing to declare.

```python
testing  = ~hiv_rdt.positive.ever >> ss.screen(hiv_rdt, coverage=ss.probperyear(0.2))   # retest only the undiagnosed
linked   = hiv_rdt.positive.since >= ss.months(6)                                       # linkage after six months
art      = linked >> ss.treat(art_drug)
retested = hiv_rdt.count >= 2                                                           # tested at least twice
```

Simpler than Starsim 3: the seven arrays, their bookkeeping, and `ti_scheduled` (a per-person schedule written by one intervention and read by another) go.

**ST15, triggers over several events.** STIsim's syndromic management is triggered by care seeking for any of three diseases; it is one class (`SymptomaticTesting`, about 180 lines) that loops over diseases. v4 raises `TypeError: Several sources are a sum of states` for `on=a.infection | b.infection` (rechecked). Proposed: `|` of events, or a list, is their union, each person once.

```python
seeking   = [ng.careseek, ct.careseek, tv.careseek]                                     # the same as ng.careseek | ct.careseek | tv.careseek
discharge = seeking >> ss.visit(cost=5)
vaginal   = (discharge & ss.ppl.female) >> ss.treat(bundle_ngcttv, disease=[ng, ct, tv])  # treat all three
urethral  = (discharge & ss.ppl.male)   >> ss.treat(bundle_ngct,   disease=[ng, ct])
syndromic = ss.cascade(discharge, vaginal, urethral)
```

The syndrome-to-bundle map, which STIsim writes as `disease_treatment_map`, becomes one step per bundle. An inexact assessment (STIsim's per-disease `sens`/`spec`) is a diagnostic over several diseases, `ss.Diagnostic(results=...)`, with each bundle triggered by one of its results.

**ST16, several follow-ups, and outcomes as events.** HPVsim's ablation is one treatment per genotype and a therapeutic vaccine is a cure plus immunity; `on_positive=[treat, vaccinate]` fails with `TypeError: 'list' object is not callable`. STIsim's drug resistance is acquired at treatment failure, which v4 cannot say. Proposed: any number of steps take the same trigger; a curing treatment's step has the events `cured` (the infected people it cured), `failed` (infected, not cured) and `unneeded` (not infected: overtreatment); resistance is a transition of the disease on the event `failed`.

```python
txvx_cure   = colposcopy.low_grade >> ss.treat(txvx)
txvx_immune = colposcopy.low_grade >> ss.vaccinate(txvx_imm)
ng = ss.Disease(susceptible, infected, resistant, ...,
    resistance = ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05),
)
```

**ST13, pregnancy events, and visits.** `on=preg.conception` raises `AttributeError`; STIsim's `ANCTest` (170 lines) schedules a visit at `ti_pregnant + randint(1, 9)` months, tests the attendees for HIV and syphilis, and routes positives by a map; `NewbornSyphTest` and `InfantHIVTest` are scheduled on the unborn through the maternal network. Proposed: `ss.Pregnancy` has the events `conception`, `delivery` and `loss` (`ss.Event`, below); a visit is `ss.visit()`; `delay=` on any step.

```python
anc_visit    = preg.conception >> ss.visit(delay=ss.randint(ss.months(1), ss.months(8)), coverage=0.85)   # attendance (agents: a delay drawn per person)
anc_hiv      = anc_visit >> ss.screen(hiv_rdt, disease=hiv)                            # the same attendees take both tests
anc_syph     = anc_visit >> ss.screen(rpr, disease=syph)
mother_tx    = anc_syph.positive >> ss.treat(benzathine_penicillin, coverage=0.95)
fetus_tx     = mother_tx >> ss.treat(fetal_penicillin, trace='prenatalnet')
recent_syph  = anc_syph.positive.since < ss.months(10)
newborn_syph = (preg.delivery & recent_syph) >> ss.screen(rpr, trace='postnatalnet')
infant_hiv   = (preg.delivery & anc_hiv.positive.ever) >> ss.screen(pcr, delay=ss.weeks(6), trace='postnatalnet')
anc          = ss.cascade(anc_visit, anc_hiv, anc_syph, mother_tx, fetus_tx, newborn_syph, infant_hiv)
```

`trace=` finds the contacts when the trigger happens, then the delay applies; `where=` still applies to the contacts, which is why the condition on the mother goes on the trigger (`&`). About 500 lines of STIsim (`ANCTest`, `ANCSyphTest`, `NewbornSyphTest`, `InfantHIVTest`, `NewbornTreatment`, and `HIVTest`'s `newborn_test=` scheduling) become these nine lines.

**ST12, stock targets with a priority.** STIsim's `ART`, `VMMC` and `Prep` keep "N people (or a share of the eligible) on it", by age and sex, filling by CD4 over care seeking, and removing the highest-CD4 above the target; with `parse_coverage` and `resolve_coverage_targets` that is about 700 lines. In v4 it is two `deliver=` functions, agents only, whose state leaked between calibration trials. The priority matters: allocating by stage alone halved 2010 deaths. Proposed: `target=`, `priority=` and `above=` on any delivery of a held product or a move (below).

```python
art  = diagnosed >> ss.treat(art_drug, target=ss.timeseries(n_art), priority=[hiv.aids, hiv.chronic, hiv.acute])   # declared HIV: by stage, under every solver
art  = diagnosed >> ss.treat(art_drug, target=n_art_by_age_sex, above='stop', priority=lambda sim, uids: -sim.diseases.hiv.cd4[uids]/sim.diseases.hiv.care_seeking[uids])  # coded HIV, with STIsim's removal: agents
vmmc = ss.ppl.male >> ss.Delivery(move=dict(circumcised=True), target=ss.by(age=[0, 0.1, 0.3, 0.25]))
```

Here `diagnosed = hiv_rdt.positive.ever | anc_rdt.positive.ever | fsw_rdt.positive.ever`. Re-engagement needs nothing: someone who drops out (`waning=` on the product) is eligible again, and the target fills the gap in priority order. With `above='stop'`, STIsim's removal above the target is reproduced, so the port loses nothing.

**HP24, tests with several results.** HPVsim's tests have three to five results, each with a probability by state, from a CSV (`products_dx.csv`), and `hierarchy` takes the most severe across genotypes (`dx.administer()`, about 80 lines, plus the hierarchy table). v4's workaround is a chain of binary tests on `on_negative=`, which reproduces the draw (abnormal 0.897 against 0.900 in CIN) but reports 35,397 tests for 20,000. Proposed: `ss.Diagnostic(results={category: probability by state})`, with the last category `None`.

```python
cytology  = ss.Diagnostic(results=dict(high_grade=dict(cin=0.9, cancer=1.0), ascus=dict(susceptible=0.1, infected=0.1, cin=0.1), normal=None))
cytology  = ss.Diagnostic(results=hpvsim.dx_table('lbc'))     # the same from HPVsim's CSV: a frame of state by result
screening = due >> ss.screen(cytology, disease=hpv, coverage=0.2, interval=ss.years(1))
```

The categories are in order of severity, most severe first. Exactly one is `None`, the last: everyone the others do not take, as `None` is the rest in a branch's `p=`. Each person draws one uniform number per test and member disease; a person's result is the most severe over the family's members (HPVsim's hierarchy). The events and results are `screening.high_grade`, `screening.ascus`, `screening.normal`. `sensitivity=`/`specificity=` remain the spelling of a two-valued test, with the results `positive` and `negative`.

**HP26, a delay on a treatment.** HPVsim's `treat_delay` holds people in a queue for a time; v4 has `delay=` only on `ss.Test`, and the pseudo-test workaround is reported as 1,452 tests. Proposed: `delay=` on every step with a trigger: the people triggered now receive it after the delay if they are still eligible then.

```python
treatment = colposcopy.high_grade >> ss.treat(ablation, delay=ss.months(3))
```

**HP7, one product over several diseases.** HPVsim's bivalent vaccine protects against two genotypes at once. In v4 two deliveries of one `ss.Vaccine` for two diseases deliver 250 to the second and protect against the first only (rechecked), and `disease=[a, b]` raises `AttributeError`. Proposed: `disease=` takes a list or an `ss.Strains`; the product's values are one number for all, or a dict by disease name; holding it is one property of people (`has_bivalent`).

```python
bivalent    = ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975))
vaccination = ss.ppl.female >> ss.vaccinate(bivalent, at_age=9, coverage=0.8)           # disease= from the product's keys
ablation    = ss.Treatment(cure=dict(infected=0.936, cin=0.936))                        # each member's infected and cin; 'hpv16.cin' for one member
```

A product bound to one disease and delivered for another raises at `init()` (HP7's S part), naming `disease=[...]`.

**ST5 and HP25, cures.** Done: a cure moves people from every infected state, or raises asking for `to=`. The remaining part is a cure by state, `cure=dict(infected=0.936, cin=0.936)`, the same dict as `sensitivity=`, and across a family the same keys mean each member's state. A destination by state (BV to CST 1 or 3, late syphilis to serofast) is `to=dict(early_latent='susceptible', late_latent='serofast')`.

**ST20, stratified standard results.** `ss.SIS` stratified by risk gives `ValueError: The result "prevalence" is not stratified` for `prevalence.select(risk='high')` (rechecked). Proposed: the standard `prevalence` of the library's diseases is built as the declared ratio of infected states over people, so it is stratified like any declared ratio. Every delivery result, including the new outcome results, is kept by stratum, as `delivered` is today.

**D27, names.** Follow-ups are named `screen_positive`, `screen_positive_positive`, and a product and its delivery cannot share a name (`ValueError: Cannot add module "vx"`, rechecked). Under peers every step is a variable, and is named from it; the product's module name gets a suffix only when it collides with a step's (`vx` the step, `vx_vaccine` the product), with no warning.

**D29, realized coverage.** Fixed in the current tree: a weekly screen at coverage 0.4 summarizes as 0.40 (agents 0.398), from the ratio of sums. Kept as a contract test.

**DF1, PL3, PL7, deliveries in the kernels.** Scheduled deliveries of a held product or a transfer run in the kernels (`ss.Delivery.tabulated`); tests and their follow-ups, cures, `capacity=`, deliveries in a model of several diseases, and crossed diseases step the run from Python. Ten years of daily ODE steps: no intervention 0.065 s, a vaccination 0.009 s, a treatment 0.60 s, a screen with a follow-up 0.72 s (`probe2.py`), about 170 µs a step. Stepped runs also draw deliveries differently from whole ones (Phase F's deferred part), and cannot be differentiated or run as many runs in one kernel call (KI2, KI23). PL7 (tables of time-varying values built whole on the stepped path) matters less once cascades no longer step. The proposal is in "Deliveries in the kernels" below.

**PL2, deliveries on diseases written as code.** `ss.vaccinate` and `ss.screen` on `ssl.AgentSIR` raise `UnknownDisease: ... the sim has no declared diseases`; STIsim's HIV and HPVsim's `hpv.HPV` are coded. Proposed: a module written as code is a delivery's `disease=` if it has the parts the product uses: `infected` (what a test detects by default), `rel_sus` and `rel_trans` (what a product's efficacy scales, as modifiers already reach coded diseases since D23), and `clear_infection(uids)` for a cure. A missing part raises naming it. Agents only, since a coded disease is.

## The vocabulary

### Events

An event is the set of people something happened to on the current step. These are events:

| Event | Who | Example |
|---|---|---|
| a transition | the people who went through it | `sir.infection` |
| a step (a delivery or a visit) | the people who received or attended it | `anc_visit`, `treatment` |
| an outcome of a step | for a test, the people with each result, on the step it is available; for a curing treatment, `cured`, `failed`, `unneeded` | `screening.ascus`, `treatment.failed` |
| a product | the union over its steps, and its outcomes likewise | `cytology.since`, `hiv_rdt.positive` |
| `ss.Event` of a module written as code | whom the module says | `preg.conception` |
| a union | anyone in either: `a \| b`, or `[a, b]` | `[ng.careseek, ct.careseek]` |
| an intersection | anyone in both on the same step: `a & b`, or `[a, b], operator='all'` | `anc_hiv.positive & anc_syph.positive` |
| an event and a condition | those in the event who meet the condition | `preg.delivery & hiv_rdt.positive.ever` |

`on=` (or the left of `>>`) takes any of them, or a function of the sim returning UIDs, as now. `operator=` is `'any'` (the default) or `'all'`, and applies to a list only; nested lists are not taken (write `(a | b) & c`). An intersection of events is people both happened to on the same step, which is rare except for outcomes of the same visit; "both, at any time" is a condition, `a.ever & b.ever`. A state or condition alone in `on=` still raises, pointing to `where=` (on the left of `>>` it is `where=`). A misspelled outcome (`screening.ascsu`) raises `AttributeError` listing the outcomes. Outcome names are taken from the product, so a test's categories cannot be `ever`, `count` or `since`.

**Order within a step.** Steps run in an order where each comes after the steps it is triggered by, so a follow-up acts on the step its trigger happens, as nesting does today. A cycle (a retest triggered by a treatment triggered by the test) must contain a `delay=`; one without raises, naming the cycle. A transition's event is the one of the last step, as now; a transition with `on=` an event of a step runs after that step.

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

Every event has three per-person readings, which the event's module keeps from the start of the run, but only if something in the sim reads them (found by walking every condition, rate and priority at `init()`), or if `count_events=True`:

| Reading | Is | Compared with |
|---|---|---|
| `ev.ever` | a condition: it has happened to them at least once | (it is a condition) |
| `ev.count` | how many times | numbers: `ev.count >= 2` |
| `ev.since` | the time since the last time (infinite if never) | durations: `ev.since < ss.years(5)` |

They are quantities like a state's attributes, so they go anywhere a condition or a per-person value goes: `where=`, the left of `>>`, `priority=`, `ss.modify(where=)`, and a transition's `where=` (`hpv.reinfection` slower for `treatment.cured.ever`). As agents they are one 16-bit step number and, if read, one 16-bit count per person (decisions item 21's timers). For **h4**: counters (`preg.delivery.count` is parity) and time since an event (`preg.delivery.since` for postpartum) should be these readings of an event, so that FPsim's behavior and a cascade's memory are one mechanism; h4's regressions and tables then take them as inputs.

**`count_events=True`** (on `ss.cascade()`, on one step, or on the sim for every event) keeps `.ever`, `.count` and `.since` for every event in scope whether or not something reads them, and as agents a log of events: one row per person and event (`uid`, `time`, `event`, and the product's result where there is one), as `sim.results.<cascade>.events.to_df()`. Under a compartmental solver every event's flow by group is already a result, so the log has no per-person form there; `count_events=True` keeps the memory as levels only if asked by name (`count_events=['cytology.since']`), since each adds levels to every group, and otherwise says so in a note.

### Deliveries: new keywords

| Keyword | On | Meaning |
|---|---|---|
| `on=` | every step | now also a step, an outcome, a product, an `ss.Event`, `\|`, `&`, a list, and `& condition` (or the left of `>>`) |
| `operator=` | a step whose `on=` is a list | `'any'` (default) or `'all'` |
| `delay=` | a step with a trigger | the time from the trigger to delivery: a duration, or a distribution of durations (agents); the people still eligible then receive it, and someone waiting is not triggered again |
| `target=` | a delivery of a held product, or a move | the number to keep holding it among the eligible: a number, a share of the eligible (below 1), a time series, or `ss.by(...)` for one target per stratum |
| `above=` | with `target=` | what happens above the target: `'keep'` (default: nothing), `'stop'` (the product is stopped for the lowest-priority holders down to the target), or a rate (`ss.peryear(0.5)`: holders above the target stop at this rate, lowest priority first) |
| `priority=` | with `target=` or `capacity=` | who is served first when there is not enough: a condition, or a list of them served in order; or a function of `(sim, uids)` returning a score, highest first (agents); ties are chosen at random, as now |
| `disease=` | every step | now also a list or an `ss.Strains` |
| `count_events=` | every step, and `ss.cascade()` | keep memory and, as agents, a log for every event (above) |

**`target=`** is a stock, evaluated every step between `start` and `stop`: if fewer than the target hold the product among those eligible, the gap is delivered to eligible people who do not hold it, in priority order, capped by `capacity=` if given. Above the target, `above=` applies (decision 6). A share is a share of the eligible, holders included (STIsim's `p_art`); `ss.by(age=..., sex=...)` gives a target per stratum, filled within it. A target is not a schedule, so it does not combine with `at=`, `interval=` or `on=`, which raise. **`priority=`** is otherwise random selection; with `capacity=` it is "while stocks last, the sickest first".

**What a user can choose above a target** (Cliff's note on decision 6). Three cases come up: the stock should never shrink except by dropout (`'keep'`, the default: the product's `waning=` is the dropout); the stock should follow the data down as well as up, as STIsim's ART does when coverage falls (`'stop'`); the stock should drift down toward the target, as when a program is wound down over a year or two (a rate). What is stopped first is the reverse of `priority=`. Anything else (stopping by a rule of the user's own) is a function in `deliver=`, agents only, as now.

### Products

| Product | New |
|---|---|
| `ss.Diagnostic` (renamed from `ss.Test`) | `results={category: probability}`, each probability a number or a dict by state (each a number or `ss.by`), the last category `None`; or a frame of state by result; categories in order of severity |
| `ss.Treatment` | `cure=` a dict by state; `to=` a dict by source state |
| all | values keyed by disease name, and `disease=` a list or an `ss.Strains`, from the product's keys if not given |

A test's results must add up to at most 1 for each state, else `ValueError` naming the state. Under an `ss.Strains` with cross-immunity, the most severe result over the members is exact under every solver; without a crossing it needs agents (below). `ss.Test` raises, saying it is now `ss.Diagnostic` (Starsim 3's `ss.Dx` already maps there in the migration script).

### Results

Each step keeps `delivered`, `eligible`, `coverage` and `cost` as now, by stratum. Each outcome adds a flow result of its name: a two-valued test `positive`, `negative`, `false_positive`, `false_negative`; a categorical test one per category; a curing treatment `cured`, `failed`, `unneeded`; a visit `attended`. This replaces `diagnosed` (decision 8) and gives STIsim's accuracy and treatment results (14 series in Starsim 3 against 3 today), by sex. A delivery with `target=` adds `holding` (the stock), `target`, and with `above=` other than `'keep'`, `stopped`. Reported cases are `a.positive + b.positive`, as the spec's `cases` example already reads with `diagnosed`. A cascade adds its table (above), built from its steps' results, so it costs nothing during the run.

## A move with no wait ("immediately")

Several designs and findings need people to pass through a state, or to make a move, with no time spent: the family's `exclusive=` (cancer from one genotype ends the course in the others), a visit or follow-up with no delay, resistance at treatment failure, the behavior vocabulary's counters (`move=` on an event), HP2 (a junction state), HP16 (a split with no dwell), and ST26 (`I >> R @ ss.perday(1e6)`, the idiom for "at once", which takes 277 s a simulated year as an ODE because the solver follows the fastest rate with substeps). Today each has its own answer or none. One rule covers them.

**The spelling.** A transition with `dur=0` is immediate: people who are in its source (and meet its `where=`) at the end of a step's moves go through it in the same step. With several destinations, `p=` splits them, as for any branch. A transition with `on=` an event is immediate for the people of the event. Nothing else is immediate: a transition with no rate, no `dur=`, no `on=`, and no `p=` keeps raising `UndrivenTransition`, since a forgotten rate is the commoner mistake, and HP16's split with `p=` and nothing else keeps its error, which now says `dur=0` (today it says to write one transition per destination).

```python
triage = ss.Transition('presenting -> [mild, severe]', dur=0, p=[0.8, None])   # a junction: nobody stays in presenting (HP2, HP16)
clear  = ss.Transition('infected -> recovered', dur=0, where=hpv.others.cancer) # as soon as the condition holds (exclusive=, ST26)
resist = ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05)  # on an event (the cascade)
```

| Solver | How |
|---|---|
| agents | In `finish_moves()`, as the junctions of HP2 already are (`solvers/agents.py: junctions`, built for states left only by branches with no dwell): people who arrive in a state with an immediate exit are moved on in the same call, recursively, and a state with a conditional immediate exit is checked once per step, after the moves. HP2's rule becomes "a state whose exits include `dur=0`", so it also covers a single destination and `where=` |
| ode, binomial | No rate. A state whose only exits are immediate and unconditional has no compartment of its own: the flows into it are sent to its destinations by the shares of `p=` (as `update_targets()` splits a branch), and both flows are recorded. A conditional one (`where=`) is a move of the condition's share once per step, between steps, as a delivery's `MOVE` is (below). So `I >> R` with `dur=0` costs nothing, against ST26's 277 s |
| gillespie | As the ODE: moves at the step boundaries |
| JAX | As the ODE; the shares are differentiable |

A loop of immediate moves (a state whose immediate exit leads back to it with nothing in between) raises when the model is built, naming the states. `describe()` says `immediately` where it now says a dwell. The family's `exclusive=` is built on this (a `dur=0` exit with `where=` the other members' state), as is a counter's `move=` on an event in the behavior vocabulary. Spec 03 gets the rule; this is a short section of the spec, not a new name.

## Semantics under every solver

| Feature | ODE | Binomial | Gillespie | Agents |
|---|---|---|---|---|
| an event | the expected number in each compartment and group | whole numbers, drawn | as binomial (deliveries happen at the time points) | UIDs |
| `\|` or a list of events | the union in each compartment and group, for diseases tracked jointly (one disease, or a crossing); otherwise `CapabilityError` | same | same | union of UIDs |
| `&` of events | the people in both, in each compartment and group, for events of the same step (outcomes of one visit); events of different modules not tracked jointly: `CapabilityError` | same | same | intersection of UIDs |
| `.ever` | a two-level property of people, `ever_<event>` | same | same | from the step number |
| `.count >= k` | `k + 1` levels, the last kept | same | same | a 16-bit count |
| `.since` against `d` | levels of time since, as a product whose protection fades by a function (one step each up to 24 levels, then several steps each, with a note) | same | same | from the step number |
| categorical test | each compartment's shares by category | multinomial | multinomial | one uniform draw per person and member |
| worst over an `ss.Strains` | exact in a crossing; otherwise `CapabilityError` | same | same | per person |
| `ss.visit` | a register of the attendees, as for any delivery | same | same | UIDs |
| a transition with `on=` | a `MOVE` of the event's register, times `p=` | same, drawn | same | UIDs |
| `delay=` fixed | pending levels of one step each (as a test's delay today) | same | same | a per-person step |
| `delay=` a distribution | an exponential only, as a rate out of one pending level; otherwise `CapabilityError` | same | same | drawn per person |
| `target=`, `priority=` by condition | filled class by class, shared within a class in proportion to the eligible | a multivariate hypergeometric draw within a class | same | ranked |
| `above='stop'` or a rate | stopped class by class, lowest priority first | drawn | same | ranked |
| `priority=` a function of uids | `CapabilityError` | same | same | ranked |
| `count_events=True` log | none (flows by group are results already) | same | same | per person |
| `ss.Event` of a coded module, coded `disease=`, `trace=` | `CapabilityError` (the module runs as agents) | same | same | yes |

`solver='auto'` picks agents for anything in the agents-only rows, as now; an explicit compartmental solver stops with the reason, naming the step and the keyword. Every level of memory multiplies the groups of every disease, so a property is added only for readings something uses, and `describe()` prints each with its levels and a note when the groups grow more than tenfold.

**`to_spec()`**: a step's `on` is written as the text of the event expression (`"screening.ascus | triage.positive"`), `delay`, `target`, `above` and `priority` as typed values and condition text, and a test's `results` as a table; a cascade as the list of its steps' names; the properties memory adds are derived, not stored, so the hash depends only on what was written. **`to_code()`** writes products, then each step as a variable in the order of its triggers, in the `who >> what` form, then the cascade; a function (`priority=lambda ...`) is written with its source, as now.

## Deliveries in the kernels

Today the kernels take a delivery as one move between groups (`ss.kernels.Shifts`: `due`, `sel`, `cov`, `to`, `rest`). The proposal replaces that with a small program per time point, a list of operations on registers that each hold a number of people per compartment and group, run at the start of each step. The steps are built into it at `init()`, in their trigger order.

| Operation | Does |
|---|---|
| `SELECT` | the eligible: a register from the compartments, times a share per compartment and group (states, stratifications, memory levels) |
| `EVENT` | a transition's people from the last step's flows, in the destination's compartments (as `event_frac()` does now) |
| `THIN` | times coverage, from a table by time |
| `CAP` | to a capacity from a table, in priority classes |
| `TARGET` | holders counted, the gap filled from the register in priority classes; with `above=`, the excess stopped in reverse priority |
| `SHIFT` | the register's people moved to other groups (holding, offered, memory, pending levels), and the register with them |
| `MOVE` | the register's people moved to another state's entry compartments, with branches and maps between dimensions, and the flow credited (a cure, `move=`, a transition with `on=`, an immediate move with `where=`) |
| `SPLIT` | a test: one register into one per result, by shares per compartment and group |
| `UNION`, `INTERSECT`, `MASK` | `\|` and `&` of two registers; a register times a condition's shares |
| `TALLY` | a register's total per group into a result |

Under the ODE every operation is an expected value; under the stochastic solvers `THIN` and `SPLIT` draw binomials and multinomials, and `CAP` and `TARGET` multivariate hypergeometrics, from the kernel's stream. The stepped path calls the same kernel function at each step, so a stepped and a whole run draw the same numbers. The many-run kernel (`run_many`) and the crossing's whole run (PL3) take the same program, a crossing's compartments being joint. What stays in Python: `when=`, `deliver=`, coverage or capacity given as a function of the sim, and anything agents-only. JAX: the same operations in `gradients.py`, second (`CAP` and `TARGET` have kinks, so gradients are piecewise, which the docs say).

This is the design `deferred.md` sketched (2026-10-16), with targets and categorical results added. Expected cost: a cascade of four steps a few µs a step against about 170 µs now.

## Fit with what exists

**Reused.** `on=`, `where=`, `coverage=`, `capacity=`, `trace=`, `move=` and `ss.Delivery` unchanged; `>>` as diseases use it; events are what `on=sir.infection` already is; memory as compartments uses the existing mechanism for levels of time since delivery (`Held.init_stages`) and for pending results; `target=` by stratum uses `ss.by`; `|` and `&` as for conditions; the immediate move extends HP2's junctions.

**New public names, each against extending an existing one.**
- `ss.cascade()`: Cliff's decision 2. It groups, names, and tabulates; it adds no behavior, so a list of steps still works anywhere a cascade does.
- `ss.visit()`: a step with no product. The alternative, `ss.Delivery()` with no product, read as "a delivery of nothing".
- `ss.Diagnostic`: a rename of `ss.Test`, so that no product and delivery differ only by case (`ss.Test` and a test).
- `ss.Event`: modules written as code need a way to publish an event with memory; `on=` a function of the sim already gives the people but no name, memory, or `describe()` line. Defined with `define_states()`, so no new method.
- `.ever`, `.count`, `.since` on events: the alternative, named states on `ss.screen`, covers tests only and leaves h4 to invent its own.
- Outcome attributes (`.positive`, `.cured`, a test's categories): named by the product, as transitions are named by the disease.
- Keywords `target=`, `above=`, `priority=`, `delay=`, `operator=`, `count_events=` on steps; `results=` on `ss.Diagnostic`; `on=` and `dur=0` on transitions. `capacity=` cannot carry a stock (it is a flow), and coverage is per round.

**Removed or replaced.** `ss.Test` (now `ss.Diagnostic`); `on_positive=` and `on_negative=` (replaced by `test.positive >> ...`; a removed name raises saying what to write, and the migration script rewrites the nested form into steps); the `diagnosed` result (now `positive`); `ss.Delivery`'s public `result` and `ti_result` states (internal pending results); automatic follow-up names (`screen_positive`); `ss.kernels.Shifts` (replaced by the delivery program); the `perday(1e6)` idiom in `skills/import-model/references/idioms.md` (now `dur=0`). Nothing is added for compatibility.

**Spec changes.** [08-interventions.md](../../design/08-interventions.md): "Follow-ups" becomes "Cascades" (`who >> what`, `ss.cascade()`, peers, events, memory, order, visits), the delivery keywords table gains `target=`, `above=`, `priority=`, `delay=`, `operator=`, `count_events=` and loses `on_positive=`/`on_negative=`, "Tests" (now diagnostics) and "Treatments" gain categorical results, cure by state and values by disease, "What a delivery reports" lists outcome results and the cascade's table, and the solver table gains the rows above. [03-states-and-transitions.md](../../design/03-states-and-transitions.md): `dur=0` and `on=` on transitions, the immediate move. [09-results.md](../../design/09-results.md) and [19-open-questions.md](../../design/19-open-questions.md): `cases = a.positive + b.positive`. [12-backends.md](../../design/12-backends.md): the delivery program, and immediate moves without a rate. [14-model-spec.md](../../design/14-model-spec.md): steps' `on`, `target`, `above`, `priority`, `delay`, test results, and cascades. [17-vocabulary.md](../../design/17-vocabulary.md): the names above. [18-conformance.md](../../design/18-conformance.md): the contracts below, and item 13 reworded. [01-objects.md](../../design/01-objects.md): `ss.Event` among module parts. [10-execution.md](../../design/10-execution.md): steps ordered by trigger.

## Implementation plan

| Step | Size | Files | Tests |
|---|---|---|---|
| 1. Events: steps, outcomes and products as events, `\|`, `&`, lists with `operator=`, and `& condition`, `ss.Event`, trigger order and the cycle error; `ss.Pregnancy`'s three events (ST13, ST15, ST16) | M | `interventions.py`, `conditions.py`, `products.py`, `arrays.py`, `demographics.py`, `loop.py` | a union fires once per person, written as `\|` or a list; a follow-up on the step of its trigger under agents and ODE; ANC: the same attendees take both tests |
| 2. The cascade's spelling: `who >> what` (a delivery's `__rrshift__`, and the refusals of `\|`, `&`, and comparisons with a delivery that say to use parentheses), `ss.cascade()`, its table and funnel, its `describe()` block; `ss.visit()` | S to M | `interventions.py`, `conditions.py`, `transitions.py`, `results.py`, `presentation.py` | `a >> ss.screen(...)` equals `ss.screen(..., on=a)`; `a \| b >> ss.treat(...)` raises naming the parentheses; the table's reached and lost add up to each step's eligible |
| 3. Memory: `.ever`, `.count`, `.since`, found at `init()`, as agents and as levels (ST14); `count_events=` and the agents' log | M | `conditions.py`, `interventions.py`, `solvers/engine.py`, `solvers/compartmental.py` | nobody screened twice within five years; positives at the end equal to those ever positive; `since` as levels against agents at a fine step; the log has one row per event per person |
| 4. Peers replace nesting: remove `on_positive=`/`on_negative=`, names from variables (D27), `diagnosed` to `positive`, outcome results by stratum, `ss.Test` to `ss.Diagnostic`; the migration script's rewrite | S | `interventions.py`, `products.py`, `migrate.py`, `skills/migrate-starsim3`, docs | the spec's screening examples give the same numbers as before; names and results as written |
| 5. The immediate move: `dur=0` and `on=` on transitions, under every solver; HP2's junctions generalized; the compartmental form without a rate (ST26, HP16) | M | `transitions.py`, `graph.py`, `solvers/agents.py`, `solvers/compartmental.py`, `kernels.py` | a junction holds nobody under every solver; `I -> R` with `dur=0` as an ODE takes no longer than without it; resistance at failure is 5% of `failed` |
| 6. `delay=` on every step (HP26) | S | `interventions.py` | a treatment three months after the test, under every solver |
| 7. Categorical tests with a `None` last category, cure by state, values by disease, `disease=` a list or `ss.Strains`, the HP7 error (HP24, HP25, HP7) | M | `products.py`, `interventions.py` | 20,000 cytologies report 20,000 tests and cost 200,000; the shares by state match HPVsim's table within 1%; a bivalent vaccine protects against both; worst over a crossed family as ODE against agents |
| 8. `target=`, `above=`, `priority=` (ST12) | M | `interventions.py` | the stock follows a time series within 1% while enough are eligible, and with `above='stop'` also when it falls; the lowest priority is served last and stopped first; by stratum |
| 9. Coded diseases as `disease=` (PL2) | S | `interventions.py`, `products.py` | screen and treat on `ssl.AgentSIR` |
| 10. Standard `prevalence` as a ratio (ST20) | S | `solvers/engine.py`, `diseases.py` | `ss.SIS(...).prevalence.select(risk='high')` |
| 11. The delivery program in the kernels (DF1), in `run`, `run_many`, the stepped path and the crossing (PL3); tables a part at a time on the stepped path (PL7) | L | `kernels.py`, `solvers/compartmental.py`, `solvers/crossing.py`, `interventions.py` | whole against stepped: identical for the ODE, the same draws for binomial; a cascade with categorical test, delay, capacity and target, ODE against the mean of 200 agent runs; the cascade ODE at under twice the time of no intervention |
| 12. The same in JAX (KI23) | M | `gradients.py` | `sim.gradient()` of a screen-and-treat model against finite differences |
| 13. Spec and docs | S | `v4_design/design/*.md`, `docs/user_guide/modules_products.qmd`, interventions tutorials | the docs' examples run |

Steps 1 to 4 come first and in that order; 5 to 10 are independent of each other; 11 needs 1 to 8. Step 5 can go first of all, since the family and the behavior vocabulary need it too. A contract test, `test_contracts.py::test_cascade`, runs one cascade (union trigger, categorical test, delay, capacity with priority, target, memory, a visit, a resistance transition) under all four solvers and checks the agreement and the reported numbers above.

**What the ports then delete** (estimated from the class sizes): STIsim, about 2,200 of its 3,700 lines of interventions: `STITest` and its six subclasses, `ANCTest`, `STITreatment` and its two subclasses, `SymptomaticTesting` and `SyndromicManagement`, the target machinery of `ART`, `VMMC` and `Prep` (with `above='stop'`, including the removal), and `parse_coverage`/`resolve_coverage_targets` (data loading into `ss.timeseries` and `ss.by` stays). `PartnerNotification` (300 lines) is `trace=` once its partner rules are; that is outside this design. HPVsim: the wrapper classes in `interventions.py` (most of its 723 lines), `dx.administer()` and the hierarchy, `treat_delay`, `treat_num`'s pseudo-test, and the `linked_txvx` eligibility lambdas; the CSVs stay, read into tables. Both packages' cascade examples become one `ss.cascade()` of one line per step, as in the use cases.

## Out of scope, and the other Phase H documents

**Out of scope.** A capacity queue in arrival order (D31) is `priority=ev.since` over the people still eligible, which a routine delivery with capacity already leaves unoffered, so it needs no keyword; whether that is enough is D31's question. Partner rules for notification beyond `trace=`; budget allocation across programs (`ss.optimize`); `after=preg.delivery` as a dwell (ST18); optimal control.

**Thanzi la Onse.** Its health system (HSI events, appointment footprints by cadre and facility level, consumables with availability) is the most complete model of care delivery in the field, and should inform what a cascade can say: whether `ss.visit()` grows an appointment type and a footprint, whether `capacity=` becomes staff time by cadre, and whether a product's availability is a consumable's. It is cloned at `/home/cliffk/idm/git/TLOmodel`, but is not reviewed here: Cliff is working on TLO with the lingua franca library first. This design is revisited once that work is done (Stage 8 of [oct19/implementation.md](../../oct19/implementation.md)).

**h2, a family of diseases (`ss.Strains`).** Assumed: `ss.Strains(name, members, ...)` with members by name, the same state names in each, and a crossing in compartmental runs when the family has cross-immunity. This design uses `disease=family`, keys `'hpv16.cin'` and `'cin'` (each member), and "worst over the members", which is exact only in a crossing. Its `exclusive=` is built on the immediate move above.

**h3, per-act transmission.** Assumed: a product's `acquisition` efficacy scales `rel_sus`, which multiplies the per-act probability, so PrEP and VMMC need nothing more from h3. Condoms as an intervention (h3's revision) are an `ss.modify` delivered by a step, and fit `who >> what` the same way.

**h4, behavior and dimensions over people's properties.** h4 should use `.ever`, `.count` and `.since` of events for counters and time since an event (decision 4), `ss.Event` for events of FPsim's coded modules, and the immediate move for a counter's `move=` on an event. Assumed from h4: sex and age as dimensions in compartmental runs (D15), so `ss.ppl.female >> ...` and `target=ss.by(age=..., sex=...)` run under every solver; until then they are agents only, as now.

## Decisions still open

For Cliff to answer with `[CK:]`.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 10 | The step spelling | (a) `who >> what`, with `on=`/`where=` as the same thing in keywords; (b) keywords only, with `ss.cascade()` grouping them; (c) `who >> what @ coverage`, as `>> ... @ rate` for diseases | (a). It reads as a flowchart and as the disease arrow. (c) is shorter but `@` would mean coverage here and a rate in diseases, and coverage is one of several keywords a step needs (`interval=`, `delay=`, `capacity=`), so it saves little. The cost of (a) is parentheses around a trigger of several parts, which the refusals catch [CK: dislike; see my comments above; try again with the API] |
| 11 | Steps in `ss.cascade()` | (a) positional, named from their variables, as in the examples; (b) keywords, `ss.cascade(screening=..., triage=...)`; (c) both | (a). Later steps must refer to earlier ones (`screening.ascus`), which needs variables anyway; with keywords the step is written in one place and referred to as another object. (b) reads like `ss.Disease(..., infection=...)` but only for cascades with no step referring to another [CK: the cascade can be anything, not necessarily screen/triage/treat. should be agnostic to the steps.] |
| 12 | The test product's name | (a) `ss.Diagnostic`; (b) `ss.Assay`; (c) keep `ss.Test` (it does not differ from `ss.screen` only by case) | (a). It says what it is, and no verb is near it. `ss.Treatment` and `ss.Vaccine` stay: they differ from `ss.treat` and `ss.vaccinate` by more than case, and are the common words [CK: ok] |
| 13 | A step with no product | (a) `ss.visit()`; (b) `ss.attend()`; (c) `ss.appointment()`, TLO's word | (a), revisited after the TLO review, which may give a visit a type and a footprint [CK: i'm ok with visit] |
| 14 | Resistance and other outcomes of an event | (a) a transition of the disease with `on=` an event; (b) `ss.Delivery(move=..., on=...)` as in the first draft | (a), as Cliff said. (b) keeps working, since it is the general move, but the docs use (a) [CK: ok] |
| 15 | Above a stock target | (a) `above='keep'` (default), `'stop'`, or a rate; (b) `'keep'` only, as approved; (c) a function too | (a). It lets STIsim's ART keep its removal, which the first draft lost, with no function [CK: ok] |
| 16 | `count_events=True` | (a) memory for every event, and a per-person log as agents; (b) memory only; (c) the log only | (a). The log is what "save results for later analyses" asks for; memory alone is per person but only the last time [CK: yes, and other than just true/false also allow a list of states/events to track] |
| 17 | "Immediately" | (a) `dur=0` on a transition, and `on=` an event; a split with only `p=` keeps its error, which says `dur=0`; (b) any transition with `p=` and no rate or `dur=` is immediate; (c) a new `ss.now` as the driver (`I >> R @ ss.now`) | (a). It reuses `dur=` (a stay of zero), keeps the forgotten-rate error, and is one rule for the junction, the condition, and the event [CK: i'm ok with (a) unless it comes up a lot, in which case would prefer (b)] |
