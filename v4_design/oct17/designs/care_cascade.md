# The care cascade

Design pass H1 for Phase H of [implementation_oct18.md](../implementation_oct18.md) (decision D-F: one design document before code). Written 2026-10-19 against the `v4` working tree. Findings covered: ST12 to ST16, ST20, HP24, HP26, D27, D29, DF1, PL2, PL3, PL7, with ST5/HP25 (cures, done) and HP7 (one product over several diseases) for the interface; and from the Thanzi la Onse review ([oct19/tlo_review.md](../../oct19/tlo_review.md)), TL36 to TL48 of its section 5, the health system. Probes are in `wt/scratch-h1/` of the session scratchpad; TLO's repros are in `TLOmodel/v4port/`.

## Third draft, 2026-10-19

The third draft of the user spelling, after Cliff's `[CK:]` notes on the second (decisions 10 to 17, and notes in the example), and folding in the Thanzi la Onse review. What changed:

- **A step is written with keywords, `who=` and `what=`** (decision 10), in Cliff's spelling: `ss.screen(who=due, what=cytology, coverage=0.2, interval=ss.years(1))`. `what=` is the product, and still the first positional argument; `ss.visit()` has none. `who=` takes a condition, an event, or a combination with `&`, `|`, or a list with `operator=`. If it contains an event the step is triggered; otherwise it is routine. `on=` and `where=` keep working and join `who=` by `&`. The `who >> what` form, its parentheses, and the refusals that came with them are gone. See "A step: `who=` and `what=`".
- **`ss.cascade()` is agnostic about its steps** (decision 11): any deliveries, in any number and order, referring to each other; it runs them in dependency order, and a cycle raises. A new subsection says what a cascade adds over a plain list (Cliff's question), honestly: a name, a table, a check, and a `describe()` block, and no behavior.
- **`cure=` takes a scalar** for every state a treatment acts on, or a dict by state (Cliff's note in the example).
- **`count_events=`** takes `True`, `False`, or a list of states and events to track (decision 16).
- **"Immediately" stays `dur=0`** (decision 17, option a), with Cliff's fallback recorded: if it comes up a lot, any transition with `p=` and no rate or `dur=` becomes immediate (option b).
- **Thanzi la Onse is folded in.** The deferral is replaced by a section on TLO's health system: which of the review's findings the cascade's steps already close (TL36, TL37, TL38, TL41, TL43, TL44, TL45, TL46, partly TL48), and a shared resource for the rest (TL39, TL40, TL42, TL47): `ss.Resource`, and `uses=` (a footprint), `level=`, `queue=`, `priority=` across steps, `stock=` and `record=`. The review's `ss.Visit` is not added: the step stays `ss.visit()`, and its footprint is the keyword `uses=`. The plan gains steps 13 to 17.
- **Decisions 10 to 17 have Cliff's answers recorded**, and a short third round (18 to 23) is in the last section.

**Second draft (history).** After Cliff's notes on the first: `ss.cascade()` was added (decision 2), with steps written `who >> what`; the test product became `ss.Diagnostic` so that products are nouns and deliveries verbs; full state names in every example; a test's last result became `normal=None`; resistance at drug failure became a transition with `on=`; a clinic visit became `ss.visit()`; `on=` took `|`, `&` and lists with `operator=` (decision 3); `count_events=True` (decision 4); `above=` on a stock target (decision 6); a section on moves with no wait (`dur=0`); and the family became `ss.Strains`.

## Summary

**The problem.** A cascade in v4 is a test with its follow-ups nested inside it (`ss.screen(test, on_positive=ss.treat(...))`), and nothing about it is remembered once the step is over. STIsim's four real cascades and HPVsim's seven WHO algorithms are not trees: a trigger can be the union of three diseases' events (ST15), one result feeds several actions months apart (ST16, ST14), a test has three to five results (HP24), and ART, VMMC and PrEP are kept at a stock target with a priority (ST12). Today each of these is a function, an extra class, or a chain of fake tests: HPVsim's cytology reports 35,397 tests for 20,000 done (costs 77% high), a bivalent vaccine silently protects against one genotype (HP7: still 250 delivered for the second disease, with no protection), and a screen-and-treat cascade forces the ODE to step from Python, taking 0.72 s for ten years of daily steps against 0.009 s for a vaccination the kernels take (DF1, measured here). Thanzi la Onse adds what every disease's care shares: staff time, a queue, consumables and facility levels (TL39 to TL46).

**The proposal, as a user writes it.** Deliveries are peers linked by events. Each step says who gets what (`who=`, `what=`), and `ss.cascade()` groups the steps. Every delivery, every outcome of one, every product, and every transition is an event; and every event remembers, per person, whether, how often, and how long ago it happened (`.ever`, `.count`, `.since`).

```python
# Products: the things delivered
cytology = ss.Diagnostic(results=dict(
    high_grade = dict(cin=0.9, cancer=1.0),                  # the chance of each result, by state; most severe first (HP24)
    ascus      = dict(susceptible=0.1, infected=0.1, cin=0.1),
    normal     = None,                                       # everyone else
), cost=10)
hpv_dna  = ss.Diagnostic(sensitivity=dict(infected=0.95, cin=0.95, cancer=1.0), specificity=0.9)
ablation = ss.Treatment(cure=dict(infected=0.936, cin=0.936))  # cure by state (HP25); ablation does not cure cancer # [CK: if same for all groups should be able to do cure=0.936, but here i guess you're saying it's only for these two] (done: cure=0.936 now cures every state the treatment acts on; the dict is kept here because cancer is left out)

# Who is eligible
women = ss.ppl.female & (ss.ppl.age >= 25) & (ss.ppl.age < 65)
due   = women & (cytology.since >= ss.years(5))             # remembered per person (ST14)

# The cascade: who gets what
# [CK: no, i don't like this. suggest this: ...] (done: the `who >> what` lines are gone; Cliff's spelling below is used throughout)
screening = ss.screen(who=due, what=cytology, coverage=0.2, interval=ss.years(1))  # routine: who= is a condition
triage    = ss.screen(who=screening.ascus, what=hpv_dna, coverage=0.9)             # triggered: who= is an event; a follow-up is a peer (ST16)
treatment = ss.treat(who=screening.high_grade | triage.positive, what=ablation, delay=ss.months(3), capacity=ss.peryear(5_000), priority=screening.high_grade)  # a union (ST15), a delay (HP26), limited supply
cervical  = ss.cascade(screening, triage, treatment)

sim = ss.Sim(diseases=hpv, interventions=cervical)          # hpv an ss.Strains: each test takes the worst result across genotypes
sim.results.cervical.to_df()                                # reached and lost at each step
# [CK: how does interventions=cervical differ from interventions=[screening, triage, treatment]? ] (done: it runs identically; the cascade adds a name, the table of reached and lost, a check on what its steps refer to, and a describe() block; see "What a cascade adds over a list")
```

The same for HIV and STIs:

```python
diagnosed = hiv_rdt.positive.ever | anc_rdt.positive.ever                          # by either program
testing   = ss.screen(who=~diagnosed, what=hiv_rdt, coverage=ss.probperyear(0.2))
anc_visit = ss.visit(who=preg.conception, delay=ss.randint(ss.months(1), ss.months(8)), coverage=0.85)  # attendance (ST13)
anc_test  = ss.screen(who=anc_visit, what=anc_rdt)
art       = ss.treat(who=diagnosed, what=art_drug, target=ss.timeseries(n_art), priority=[hiv.aids, hiv.chronic])  # a stock target (ST12)
hiv_care  = ss.cascade(testing, anc_visit, anc_test, art)

ceftriaxone = ss.Treatment(cure=0.95)                                              # a scalar: every infected state
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
| 2 | A cascade container (`ss.cascade(visit=..., test=..., treat=...)`) | (a) none: a cascade is a list of deliveries; (b) add it now, for names, string references and a cascade table | (a), and revisit if a cascade results table ("lost at each step") is wanted; `ss.parallel` and `sim.new()` already take lists [CK: i think we should do `ss.cascade()`, but make it compelling: make it super ultra clear what the logic is, as clear as the `S >> I @ beta*I` logic for diseases. care cascades are some of the most complex parts of disease models, so we if we can make that better, this would be a huge benefit.] | Added: steps with `who=` and `what=` (decision 10), `ss.cascade(...)` over any steps (decision 11), and its table |
| 3 | Several events in `on=` | (a) `\|` only (`on=a.infection \| b.infection`); (b) a list too | (a). `\|` is already the union of conditions; a list raises, pointing to `\|`  [CK: no -- \| or a list are valid and equivalent, but & is also a valid operator. `[a,b], operator='any' == a \| b`, `a,b], operator='all' == a & b`.] | As Cliff says, in `who=` (and `on=`) |
| 4 | Remembered results | (a) `.ever`, `.count`, `.since` on every event, kept only when something reads them; (b) fixed states (`tested`, `positive`, `ti_positive`) on `ss.screen` only | (a), as the one primitive for memory of events; h4 (counters, time since an event) uses it rather than a second one [CK: ok, but maybe have an optional count_events=True flag for when people want to save results for later analyses] | (a), with `count_events=` `True` or a list (decision 16) |
| 5 | Events on products as well as deliveries | (a) a product's events are the union over its deliveries (`lbc.since`: any program's cytology); (b) deliveries only | (a). It is what "screened in the last five years" means, and it lets a delivery's `where=` refer to its own product, which it cannot do to itself  [CK: ok] | Unchanged |
| 6 | What a stock target does above the target | (a) tops up only; (b) also stops the product for the lowest-priority holders | (a). Dropout is the product's `waning=`; STIsim's removal exists to force the stock onto data, which is calibration's job [CK: ok, but might need some machinery to allow user choices here] | (a) by default, and `above=` for the other choices (decision 15) |
| 7 | Deliveries in the kernels (DF1) | (a) a program of delivery operations in the kernels, used by whole and stepped runs alike; (b) keep stepping from Python; (c) only tests and cures, without capacity or targets | (a), Numba first and JAX second. It is L, but it is what makes cascades fittable by NUTS and gradients, and it makes stepped and whole runs draw the same numbers [CK: ok] | Unchanged; resources and queues join it (step 17) |
| 8 | The name of a test's count of positives | (a) `positive`, the same name as the event, so `cases = a.positive + b.positive`; (b) keep `diagnosed` | (a). Every outcome is then a flow result and an event of one name, as a transition is [CK: ok] | Unchanged |
| 9 | Values of a product by disease (HP7) | (a) a dict keyed by disease name (`efficacy=dict(hpv16=0.9975, hpv18=0.9975)`); (b) `ss.by(disease=...)` | (a), as `sensitivity=` is keyed by state; a key that is both a disease and an effect name raises [CK: ok] | Unchanged |

Cliff's note on the first draft's example, kept as the record: "[CK: i don't love this ... it's confusing that ss.Test() is the product and ss.test() is the intervention. i don't know what C, X, S, P, etc mean (precancer and cancer for P and C, maybe?). i don't know what default='normal' means. i don't think drug failure leading to resistance should be ss.Delivery, that makes no senes to me. nor should a visit be a 'delivery of nothing'. maybe we should include thanzi la onse in the review...]" (done: `ss.Diagnostic`, full state names, `normal=None`, resistance as a transition, `ss.visit()`, and TLO in "Thanzi la Onse's health system".)

## The cascade, as a user writes it

### A step: `who=` and `what=`

A step is one call: the delivery's verb, who gets it, and what they get.

```python
triage = ss.screen(who=screening.ascus, what=hpv_dna, coverage=0.9)
```

**`what=`** is the product: a `ss.Diagnostic` for `ss.screen`, a `ss.Treatment` for `ss.treat`, a `ss.Vaccine` for `ss.vaccinate`. It is still the first positional argument, so `ss.screen(hpv_dna, who=screening.ascus)` is the same step. `ss.visit()` has no `what=` (its event is the attendees), and `ss.Delivery` has `move=` in its place.

**`who=`** says who is reached, and whether the step is triggered or routine:

| `who=` | Example | The step is | Reaches |
|---|---|---|---|
| a condition | `due`, `~diagnosed`, `hiv.infected` | routine | the people who meet it, offered by the step's schedule (`interval=`, `at=`, `at_age=`, or every step) with `coverage=` |
| conditions combined | `women & ~diagnosed`, `[a, b], operator='all'` | routine | as above |
| an event | `screening.ascus`, `preg.conception`, `hpv.infection`, a transfer | triggered | the people it happened to on this step |
| events, either | `a \| b`, or `[a, b]` (`operator='any'`, the default) | triggered | anyone in either, each once |
| events, both | `a & b`, or `[a, b], operator='all'` | triggered | anyone in both on the same step |
| an event and a condition | `preg.delivery & hiv_rdt.positive.ever` | triggered | those in the event who meet the condition |

The rule is one sentence: if `who=` contains an event, the step is triggered, and reaches those in whom the event happened this step and who meet the condition parts; otherwise it is routine, and reaches eligible people each step, as `coverage=` and `interval=` say. A condition joins an event only by `&` (and `operator='all'`); `event | condition` raises, saying to write two steps, since "anyone the event happened to, or anyone who meets the condition" is a triggered step and a routine one at once (decision 22).

**`on=` and `where=`** keep working, as the same thing: each joins `who=` by `&`. So `ss.screen(hpv_dna, on=screening.ascus)` is `ss.screen(who=screening.ascus, what=hpv_dna)`, and `ss.screen(who=due, what=cytology, where=ss.ppl.urban)` is `who=due & ss.ppl.urban`. `on=` still takes only events (a condition alone in `on=` raises, pointing to `who=`). The docs and `to_code()` use `who=` (decision 21).

**Why keywords.** It reads as "screen who, with what", in the order a reader asks; every step has one form, inside a cascade or not, and it is what `to_code()` writes. A trigger of several parts needs no parentheses (`who=screening.high_grade | triage.positive`), and the triggered/routine distinction is read off `who=` rather than from which of `on=` and `where=` was used.

### `ss.cascade(*steps, name=None, count_events=False)`

```python
cervical = ss.cascade(screening, triage, treatment)
```

- **Steps are any deliveries**: `ss.screen`, `ss.treat`, `ss.vaccinate`, `ss.visit`, `ss.Delivery(move=...)`, in any number and any order. Screen, triage and treat is one shape; ANC (a visit, two tests, three treatments), syndromic management (a visit and two bundles), a vaccination schedule of three doses, or a referral chain through facility levels are others.
- **Names** are taken from the steps' variables (D27), or from `name=` on a step; a step made inline, with no variable, is named from its verb and product (`screen_cytology`).
- **Order.** The order written is the order of `describe()` and the table. The order they run in is dependency order: each step after the steps its `who=` and `priority=` refer to, so a follow-up acts on the step its trigger happens, as nesting does today. A cycle raises, naming it. A `delay=` breaks a cycle, since the delayed step acts on a later step, so a retest three months after a treatment triggered by the test is not one.
- **Steps refer to each other as variables**, as transitions refer to states. A step's events (`screening.ascus`) are its own; a product's (`cytology.ascus`) are the union over every step that delivers it (decision 5).
- **Results.** Each step keeps its results under its own name, as now (`sim.results.treatment.delivered`). The cascade adds `sim.results.cervical`, whose `to_df()` is the table below and whose `plot()` is a funnel.

```text
step        who                                       eligible  reached  lost (coverage)  lost (capacity)  outcomes
screening   due, every 1 year                          410,200   82,040           328,160                0  high_grade 1,804; ascus 4,100; normal 76,136
triage      screening.ascus                              4,100    3,690               410                0  positive 1,107; negative 2,583
treatment   screening.high_grade | triage.positive       2,911    2,744                 0              167  cured 2,407; failed 165; unneeded 172
```

`describe()` prints it as a flowchart, in the order written:

```text
cervical (cascade of 3 steps):
  screening   routine    due                                    -> screen cytology (high_grade, ascus, else normal; worst over hpv16, hpv18, hi5, ohr) every 1 year, coverage 0.2
  triage      triggered  screening.ascus                        -> screen hpv_dna, coverage 0.9
  treatment   triggered  screening.high_grade | triage.positive -> treat ablation (cures infected 0.936, cin 0.936) after 3 months; capacity 5,000 a year, screening.high_grade first
  adds as properties of people: since_cytology (11 levels: 0.5 years each to 5 years), pending_treatment (3 levels)
```

### What a cascade adds over a list

Cliff asked how `interventions=cervical` differs from `interventions=[screening, triage, treatment]`. **Running them is identical**: the same steps, in the same dependency order, drawing the same numbers, with the same results under each step's name. The cascade adds four things, all about reading and checking the model:

1. **A name.** `sim.results.cervical`, and `cervical.treatment.capacity` for `ss.change` and scenarios; one object for `interventions=` and `sim.new()`.
2. **The table of reached and lost at each step, and its funnel.** A list has no notion of a first step or of people lost between steps. This is the usual report of a care cascade (the 90-90-90 targets are such a table), and today each package builds it by hand.
3. **A check on references.** Every step that a step's `who=` or `priority=` refers to should be in the cascade; one that is not gets a note naming it (`triage refers to screening, which is not in cascade 'cervical'`). It is a note, not an error, since a step may rightly refer to another program (ART for those diagnosed by any of three testing programs).
4. **A `describe()` block** that reads as the flowchart above.

That is modest: it adds no behavior. If it is not enough for a public name, the alternative is no container, with the table as a function of results (`sim.results.cascade(screening, triage, treatment)`), at the cost of the check, the `describe()` block, and the name for scenarios. This draft keeps `ss.cascade()`, as decision 2 asked, because the table and the check are where cascade bugs show (a step never reached, a step wired to the wrong program).

### Products and steps: nouns and verbs

| Product (a thing, `what=`) | Delivered by (a verb) | Outcomes (events and results) |
|---|---|---|
| `ss.Diagnostic` (was `ss.Test`) | `ss.screen` | `positive`, `negative` (two results), or one per category; `not_done` when out of stock |
| `ss.Treatment` | `ss.treat` | `cured`, `failed`, `unneeded` for a cure; none for a held treatment (ART) |
| `ss.Vaccine` | `ss.vaccinate` | none |
| none | `ss.visit` | the attendees are the event |
| none | `ss.Delivery(move=...)` | the movers are the event |

A product's own keywords may also be given to its step, which then makes the product (`ss.screen(sensitivity=0.9)`, as today).

**`ss.visit(who=..., **delivery keywords, cost=None)`** is a step that people attend, with no product: its event is the attendees. ANC, a clinic visit for symptoms, a follow-up appointment, TLO's generic first appointment. It takes every delivery keyword (`delay=`, `coverage=`, `capacity=`, `priority=`, and the resource keywords `uses=`, `level=`, `queue=`), and `cost=` per visit. Several steps take `who=` the same visit, so one visit runs several tests on the same people.

**Resistance and other outcomes in the disease.** Something that happens to a person's infection because of an event (resistance at treatment failure, a relapse on stopping a drug) is natural history, so it is a transition of the disease, triggered by the event: `ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05)`. A transition with `on=` happens to those people on the step of the event (with `p=` of them, and `where=` as for any transition); it has no rate or dwell. It is the transition form of "immediately" (below), with the event as its trigger. The first draft's `ss.Delivery(move='I -> Ir', on=...)` still works, but the docs use the transition.

## The shape of a cascade

STIsim's four cascades, HPVsim's algorithms, and TLO's care pathways, against what nesting can say:

| Cascade | Steps | What nesting cannot say |
|---|---|---|
| Syndromic management (STIsim) | care seeking for any of NG, CT, TV; assessment by sex and syndrome; one of four bundles, each treating several diseases | a trigger that is a union of three diseases' events; one product over several diseases; nothing is "positive" |
| Etiological testing (STIsim) | care seeking or screening; a test per disease; result after a delay; treat; notify partners; retest at three months | two triggers for one treatment; a retest triggered by a treatment long past |
| ANC (STIsim) | conception; a visit after a delay, if attended; HIV and syphilis tests; for syphilis, treat mother and fetus and notify partners; at delivery, test and treat the newborn | one visit runs several tests on the same people; a result acts months later, from memory |
| HIV care (STIsim) | three testing programs; diagnosed; linkage after a delay; ART to a stock target with a priority; dropout; re-engagement | a stock target; the steps are states people stay in |
| WHO algorithm 3 (HPVsim) | cytology with four results; HPV triage of ASCUS; colposcopy for abnormal or HPV-positive; ablation of HSIL | five-valued results; colposcopy has two parents |
| Oesophageal cancer (TLO) | a symptom; the generic first appointment over all symptoms; endoscopy if in stock, at level 2; surgery for stages 1 to 3 or palliative care for stage 4, waiting for staff | two follow-ups of one result (TL41); a test not done (TL45); staff time shared with every other disease (TL39) |

So a cascade is steps that people flow through, each with its own trigger, coverage, delay and capacity (where the losses are), each producing events and memory that any later step can use. The proposal makes those steps ordinary deliveries, each saying `who=` and `what=`, and adds nothing else structural:

- **Peers, not nesting.** `ss.screen(what=diagnostic)` wraps the diagnostic it delivers, as `ss.vaccinate` wraps its vaccine. It does not wrap what comes after. A follow-up is a step whose `who=` is the event it follows.
- **Events.** An event is the set of people something happened to on this step. Transitions are events today (`on=sir.infection`). The proposal adds transfers (TL43), deliveries (the people who received it), their outcomes (a test's results by name; a cure's `cured` and `failed`), products (the union over their deliveries), visits, `ss.Event` for modules written as code (`preg.conception`), unions (`a | b`), intersections (`a & b`), and an event narrowed by a condition (`preg.delivery & syph_test.positive.ever`).
- **Memory.** Every event has `.ever` (a condition), `.count` and `.since` (per-person quantities that compare with numbers and durations). They are kept only for events something reads, or for those `count_events=` names.

**When a cascade should be states instead.** The EpiModel HIV vignette writes care as a dimension with transfers (`ss.transfer(care='dx -> art', rate=...)`). That is the right form when the steps are stages people stay in, entered and left at rates, and the data are the numbers in each stage. Deliveries are the right form when a step is an act with a product, an accuracy, a cost, a coverage, a capacity, or a target. The two combine: `ss.Delivery(who=diagnosed, move=dict(care='diagnosed -> on_art'), target=..., priority=...)` is a stock target on a care dimension. `describe()` of a cascade says which care-like states it adds (below), so the choice is visible.

## Use cases

Each finding: what the package needs, how it does it in Starsim 3 code, and the v4 spelling under this proposal. `ss.ppl` is the lazy reference to people's properties (`behavior_vocabulary.md`).

**ST14, remembered test results.** STIsim's `STITest` keeps seven arrays per person (`tested`, `diagnosed`, `ti_tested`, `ti_scheduled`, `ti_positive`, `ti_negative`, `tests`), set by hand in `step()` and read by other interventions' `eligibility=` lambdas; HPVsim keeps `ti_screened` for "not screened in five years". v4's `ss.screen` holds `result` only while a delayed result is pending, so 0 people are positive at the end of a run that tested everyone (`sti/my/st_misc.py`). Proposed: nothing to declare.

```python
testing  = ss.screen(who=~hiv_rdt.positive.ever, what=hiv_rdt, coverage=ss.probperyear(0.2))  # retest only the undiagnosed
linked   = hiv_rdt.positive.since >= ss.months(6)                                            # linkage after six months
art      = ss.treat(who=linked, what=art_drug)
retested = hiv_rdt.count >= 2                                                                # tested at least twice
```

Simpler than Starsim 3: the seven arrays, their bookkeeping, and `ti_scheduled` (a per-person schedule written by one intervention and read by another) go.

**ST15, triggers over several events.** STIsim's syndromic management is triggered by care seeking for any of three diseases; it is one class (`SymptomaticTesting`, about 180 lines) that loops over diseases. v4 raises `TypeError: Several sources are a sum of states` for `on=a.infection | b.infection` (rechecked). Proposed: `|` of events, or a list, is their union, each person once.

```python
seeking   = [ng.careseek, ct.careseek, tv.careseek]                                     # the same as ng.careseek | ct.careseek | tv.careseek
discharge = ss.visit(who=seeking, cost=5)
vaginal   = ss.treat(who=discharge & ss.ppl.female, what=bundle_ngcttv, disease=[ng, ct, tv])  # treat all three
urethral  = ss.treat(who=discharge & ss.ppl.male,   what=bundle_ngct,   disease=[ng, ct])
syndromic = ss.cascade(discharge, vaginal, urethral)
```

The syndrome-to-bundle map, which STIsim writes as `disease_treatment_map`, becomes one step per bundle. An inexact assessment (STIsim's per-disease `sens`/`spec`) is a diagnostic over several diseases, `ss.Diagnostic(results=...)`, with each bundle triggered by one of its results.

**ST16, several follow-ups, and outcomes as events.** HPVsim's ablation is one treatment per genotype and a therapeutic vaccine is a cure plus immunity; `on_positive=[treat, vaccinate]` fails with `TypeError: 'list' object is not callable` (TLO hits the same, TL41). STIsim's drug resistance is acquired at treatment failure, which v4 cannot say. Proposed: any number of steps take the same trigger; a curing treatment's step has the events `cured` (the infected people it cured), `failed` (infected, not cured) and `unneeded` (not infected: overtreatment); resistance is a transition of the disease on the event `failed`.

```python
txvx_cure   = ss.treat(who=colposcopy.low_grade, what=txvx)
txvx_immune = ss.vaccinate(who=colposcopy.low_grade, what=txvx_imm)
ng = ss.Disease(susceptible, infected, resistant, ...,
    resistance = ss.Transition('infected -> resistant', on=ceftriaxone.failed, p=0.05),
)
```

**ST13, pregnancy events, and visits.** `on=preg.conception` raises `AttributeError`; STIsim's `ANCTest` (170 lines) schedules a visit at `ti_pregnant + randint(1, 9)` months, tests the attendees for HIV and syphilis, and routes positives by a map; `NewbornSyphTest` and `InfantHIVTest` are scheduled on the unborn through the maternal network. Proposed: `ss.LegacyPregnancy` has the events `conception`, `delivery` and `loss` (`ss.Event`, below); a visit is `ss.visit()`; `delay=` on any triggered step.

```python
anc_visit    = ss.visit(who=preg.conception, delay=ss.randint(ss.months(1), ss.months(8)), coverage=0.85)  # attendance (agents: a delay drawn per person)
anc_hiv      = ss.screen(who=anc_visit, what=hiv_rdt, disease=hiv)                    # the same attendees take both tests
anc_syph     = ss.screen(who=anc_visit, what=rpr, disease=syph)
mother_tx    = ss.treat(who=anc_syph.positive, what=benzathine_penicillin, coverage=0.95)
fetus_tx     = ss.treat(who=mother_tx, what=fetal_penicillin, trace='prenatalnet')
recent_syph  = anc_syph.positive.since < ss.months(10)
newborn_syph = ss.screen(who=preg.delivery & recent_syph, what=rpr, trace='postnatalnet')
infant_hiv   = ss.screen(who=preg.delivery & anc_hiv.positive.ever, what=pcr, delay=ss.weeks(6), trace='postnatalnet')
anc          = ss.cascade(anc_visit, anc_hiv, anc_syph, mother_tx, fetus_tx, newborn_syph, infant_hiv)
```

With `trace=`, `who=` (with any `on=` and `where=` joined to it) picks the people traced from, here the mothers; their contacts are found when the trigger happens, then the delay applies. A condition on the contacts themselves is a partner rule, outside this design. About 500 lines of STIsim (`ANCTest`, `ANCSyphTest`, `NewbornSyphTest`, `InfantHIVTest`, `NewbornTreatment`, and `HIVTest`'s `newborn_test=` scheduling) become these nine lines.

**ST12, stock targets with a priority.** STIsim's `ART`, `VMMC` and `Prep` keep "N people (or a share of the eligible) on it", by age and sex, filling by CD4 over care seeking, and removing the highest-CD4 above the target; with `parse_coverage` and `resolve_coverage_targets` that is about 700 lines. In v4 it is two `deliver=` functions, agents only, whose state leaked between calibration trials. The priority matters: allocating by stage alone halved 2010 deaths. Proposed: `target=`, `priority=` and `above=` on any routine step of a held product or a move (below).

```python
art  = ss.treat(who=diagnosed, what=art_drug, target=ss.timeseries(n_art), priority=[hiv.aids, hiv.chronic, hiv.acute])   # declared HIV: by stage, under every solver
art  = ss.treat(who=diagnosed, what=art_drug, target=n_art_by_age_sex, above='stop', priority=lambda sim, uids: -sim.diseases.hiv.cd4[uids]/sim.diseases.hiv.care_seeking[uids])  # coded HIV, with STIsim's removal: agents
vmmc = ss.Delivery(who=ss.ppl.male, move=dict(circumcised=True), target=ss.by(age=[0, 0.1, 0.3, 0.25]))
```

Here `diagnosed = hiv_rdt.positive.ever | anc_rdt.positive.ever | fsw_rdt.positive.ever`. Re-engagement needs nothing: someone who drops out (`waning=` on the product) is eligible again, and the target fills the gap in priority order. With `above='stop'`, STIsim's removal above the target is reproduced, so the port loses nothing.

**HP24, tests with several results.** HPVsim's tests have three to five results, each with a probability by state, from a CSV (`products_dx.csv`), and `hierarchy` takes the most severe across genotypes (`dx.administer()`, about 80 lines, plus the hierarchy table). v4's workaround is a chain of binary tests on `on_negative=`, which reproduces the draw (abnormal 0.897 against 0.900 in CIN) but reports 35,397 tests for 20,000. Proposed: `ss.Diagnostic(results={category: probability by state})`, with the last category `None`.

```python
cytology  = ss.Diagnostic(results=dict(high_grade=dict(cin=0.9, cancer=1.0), ascus=dict(susceptible=0.1, infected=0.1, cin=0.1), normal=None))
cytology  = ss.Diagnostic(results=hpvsim.dx_table('lbc'))     # the same from HPVsim's CSV: a frame of state by result
screening = ss.screen(who=due, what=cytology, disease=hpv, coverage=0.2, interval=ss.years(1))
```

The categories are in order of severity, most severe first. Exactly one is `None`, the last: everyone the others do not take, as `None` is the rest in a branch's `p=`. Each person draws one uniform number per test and member disease; a person's result is the most severe over the family's members (HPVsim's hierarchy). The events and results are `screening.high_grade`, `screening.ascus`, `screening.normal`. `sensitivity=`/`specificity=` remain the spelling of a two-valued test, with the results `positive` and `negative`.

**HP26, a delay on a treatment.** HPVsim's `treat_delay` holds people in a queue for a time; v4 has `delay=` only on `ss.Test`, and the pseudo-test workaround is reported as 1,452 tests. Proposed: `delay=` on every triggered step: the people triggered now receive it after the delay if they are still eligible then.

```python
treatment = ss.treat(who=colposcopy.high_grade, what=ablation, delay=ss.months(3))
```

**HP7, one product over several diseases.** HPVsim's bivalent vaccine protects against two genotypes at once. In v4 two deliveries of one `ss.Vaccine` for two diseases deliver 250 to the second and protect against the first only (rechecked), and `disease=[a, b]` raises `AttributeError`. Proposed: `disease=` takes a list or an `ss.Strains`; the product's values are one number for all, or a dict by disease name; holding it is one property of people (`has_bivalent`).

```python
bivalent    = ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975))
vaccination = ss.vaccinate(who=ss.ppl.female, what=bivalent, at_age=9, coverage=0.8)       # disease= from the product's keys
ablation    = ss.Treatment(cure=dict(infected=0.936, cin=0.936))                           # each member's infected and cin; 'hpv16.cin' for one member
```

A product bound to one disease and delivered for another raises at `init()` (HP7's S part), naming `disease=[...]`.

**ST5 and HP25, cures.** Done: a cure moves people from every infected state, or raises asking for `to=`. The remaining part is the value of `cure=`: a scalar, the chance of cure in every state the treatment acts on (its disease's infected states, `cure=0.936`), or a dict by state, the same dict as `sensitivity=`, for a cure that differs by state or leaves some out (`cure=dict(infected=0.936, cin=0.936)`: ablation does not cure cancer). Across a family the same keys mean each member's state. A destination by state (BV to CST 1 or 3, late syphilis to serofast) is `to=dict(early_latent='susceptible', late_latent='serofast')`.

**ST20, stratified standard results.** `ss.SIS` stratified by risk gives `ValueError: The result "prevalence" is not stratified` for `prevalence.select(risk='high')` (rechecked). Proposed: the standard `prevalence` of the library's diseases is built as the declared ratio of infected states over people, so it is stratified like any declared ratio. Every delivery result, including the new outcome results, is kept by stratum, as `delivered` is today.

**D27, names.** Follow-ups are named `screen_positive`, `screen_positive_positive`, and a product and its delivery cannot share a name (`ValueError: Cannot add module "vx"`, rechecked). Under peers every step is a variable, and is named from it; the product's module name gets a suffix only when it collides with a step's (`vx` the step, `vx_vaccine` the product), with no warning.

**D29, realized coverage.** Fixed in the current tree: a weekly screen at coverage 0.4 summarizes as 0.40 (agents 0.398), from the ratio of sums. Kept as a contract test.

**DF1, PL3, PL7, deliveries in the kernels.** Scheduled deliveries of a held product or a transfer run in the kernels (`ss.Delivery.tabulated`); tests and their follow-ups, cures, `capacity=`, deliveries in a model of several diseases, and crossed diseases step the run from Python. Ten years of daily ODE steps: no intervention 0.065 s, a vaccination 0.009 s, a treatment 0.60 s, a screen with a follow-up 0.72 s (`probe2.py`), about 170 µs a step. Stepped runs also draw deliveries differently from whole ones (Phase F's deferred part), and cannot be differentiated or run as many runs in one kernel call (KI2, KI23). PL7 (tables of time-varying values built whole on the stepped path) matters less once cascades no longer step. The proposal is in "Deliveries in the kernels" below.

**PL2, deliveries on diseases written as code.** `ss.vaccinate` and `ss.screen` on `ssl.AgentSIR` raise `UnknownDisease: ... the sim has no declared diseases`; STIsim's HIV and HPVsim's `hpv.HPV` are coded. Proposed: a module written as code is a delivery's `disease=` if it has the parts the product uses: `infected` (what a test detects by default), `rel_sus` and `rel_trans` (what a product's efficacy scales, as modifiers already reach coded diseases since D23), and `clear_infection(uids)` for a cure. A missing part raises naming it. Agents only, since a coded disease is. TLO's EPI (TL44) is the case with no disease at all: a product with `disease=None` is delivered and counted, and has no effect.

## The vocabulary

### Events

An event is the set of people something happened to on the current step. These are events:

| Event | Who | Example |
|---|---|---|
| a transition, including a transfer (TL43) | the people who went through it | `sir.infection`, `oc.symptom` (a transfer into a symptom level) |
| a step (a delivery or a visit) | the people who received or attended it | `anc_visit`, `treatment` |
| an outcome of a step | for a test, the people with each result, on the step it is available; for a curing treatment, `cured`, `failed`, `unneeded`; `not_done` and `never_ran` (below) | `screening.ascus`, `treatment.failed`, `endoscopy.not_done` |
| a product | the union over its steps, and its outcomes likewise | `cytology.since`, `hiv_rdt.positive` |
| `ss.Event` of a module written as code | whom the module says | `preg.conception` |
| a union | anyone in either: `a \| b`, or `[a, b]` | `[ng.careseek, ct.careseek]` |
| an intersection | anyone in both on the same step: `a & b`, or `[a, b], operator='all'` | `anc_hiv.positive & anc_syph.positive` |
| an event and a condition | those in the event who meet the condition | `preg.delivery & hiv_rdt.positive.ever` |

`who=` takes any of them (and conditions, for a routine step), or a function of the sim returning UIDs, as `on=` does now. `operator=` is `'any'` (the default) or `'all'`, and applies to a list only; nested lists are not taken (write `(a | b) & c`). An intersection of events is people both happened to on the same step, which is rare except for outcomes of the same visit; "both, at any time" is a condition, `a.ever & b.ever`. A misspelled outcome (`screening.ascsu`) raises `AttributeError` listing the outcomes. Outcome names are taken from the product, so a test's categories cannot be `ever`, `count` or `since`.

**Order within a step.** Steps run in dependency order (above). A transition's event is the one of the last step, as now; a transition with `on=` an event of a step runs after that step. A step triggered by a disease's event acts at the first step boundary after it, so one `dt` after the event, under every solver (TL48, below).

**`ss.Event`**, the one new class of event: an event that a module written as code fires. It is defined with the module's states and fired with the UIDs.

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

Every event has three per-person readings, which the event's module keeps from the start of the run, but only if something in the sim reads them (found by walking every condition, rate and priority at `init()`), or if `count_events=` asks for them:

| Reading | Is | Compared with |
|---|---|---|
| `ev.ever` | a condition: it has happened to them at least once | (it is a condition) |
| `ev.count` | how many times | numbers: `ev.count >= 2` |
| `ev.since` | the time since the last time (infinite if never) | durations: `ev.since < ss.years(5)` |

They are quantities like a state's attributes, so they go anywhere a condition or a per-person value goes: `who=`, `priority=`, `ss.modify(where=)`, and a transition's `where=` (`hpv.reinfection` slower for `treatment.cured.ever`). As agents they are one 16-bit step number and, if read, one 16-bit count per person (decisions item 21's timers). For **h4**: counters (`preg.delivery.count` is parity) and time since an event (`preg.delivery.since` for postpartum) should be these readings of an event, so that FPsim's behavior and a cascade's memory are one mechanism; h4's regressions and tables then take them as inputs. A dose count is the same reading: `penta.count` is TLO's `va_dtp` (TL36).

**`count_events=`** (on `ss.cascade()`, on one step, or on the sim) is `False` (the default), `True`, or a list of states and events. `True` keeps `.ever`, `.count` and `.since` for every event in scope whether or not something reads them, and as agents a log of events: one row per person and event (`uid`, `time`, `event`, and the product's result where there is one), as `sim.results.<cascade>.events.to_df()`. A list keeps only those named (`count_events=[cytology, treatment.failed, hpv.cin]`); a state in the list means the events of entering it. Under a compartmental solver every event's flow by group is already a result, so the log has no per-person form there; memory is kept as levels only for events named in a list, since each adds levels to every group, and `True` says so in a note.

### Deliveries: new keywords

| Keyword | On | Meaning |
|---|---|---|
| `who=` | every step | a condition (routine), an event (triggered), or `&`, `\|` and lists of them (above) |
| `what=` | every step but `ss.visit` and `ss.Delivery` | the product; the first positional argument |
| `on=`, `where=` | every step | as before; each joins `who=` by `&` |
| `operator=` | a step whose `who=` (or `on=`) is a list | `'any'` (default) or `'all'` |
| `delay=` | a triggered step | the time from the trigger to delivery: a duration, or a distribution of durations (agents); the people still eligible then receive it, and someone waiting is not triggered again |
| `target=` | a routine step of a held product, or a move | the number to keep holding it among the eligible: a number, a share of the eligible (below 1), a time series, or `ss.by(...)` for one target per stratum |
| `above=` | with `target=` | what happens above the target: `'keep'` (default: nothing), `'stop'` (the product is stopped for the lowest-priority holders down to the target), or a rate (`ss.peryear(0.5)`: holders above the target stop at this rate, lowest priority first) |
| `priority=` | with `target=`, `capacity=` or `uses=` | who is served first when there is not enough: a condition, or a list of them served in order; a number, one class for everyone the step reaches (across the steps that share a resource, 0 first); or a function of `(sim, uids)` returning a score, highest first (agents); ties are chosen at random, as now |
| `disease=` | every step | now also a list, an `ss.Strains`, or `None` (counted, no effect) |
| `count_events=` | every step, and `ss.cascade()` | `True`, or a list of states and events: memory and, as agents, a log (above) |
| `uses=`, `level=`, `queue=`, `record=` | every step | the shared resource (below) |

**`target=`** is a stock, evaluated every step between `start` and `stop`: if fewer than the target hold the product among those eligible, the gap is delivered to eligible people who do not hold it, in priority order, capped by `capacity=` if given. Above the target, `above=` applies (decision 6). A share is a share of the eligible, holders included (STIsim's `p_art`); `ss.by(age=..., sex=...)` gives a target per stratum, filled within it. A target is not a schedule, so it does not combine with `at=`, `interval=`, or an event in `who=`, which raise. **`priority=`** is otherwise random selection; with `capacity=` it is "while stocks last, the sickest first".

**What a user can choose above a target** (Cliff's note on decision 6). Three cases come up: the stock should never shrink except by dropout (`'keep'`, the default: the product's `waning=` is the dropout); the stock should follow the data down as well as up, as STIsim's ART does when coverage falls (`'stop'`); the stock should drift down toward the target, as when a program is wound down over a year or two (a rate). What is stopped first is the reverse of `priority=`. Anything else (stopping by a rule of the user's own) is a function in `deliver=`, agents only, as now.

**Giving a product again** (TL36). A triggered step gives its product each time its trigger happens, held or not, so a schedule of doses is steps triggered by the dose before (`dose2 = ss.vaccinate(who=dose1, what=penta, delay=ss.weeks(4))`), and `penta.count` is the dose count. A routine step re-arms when its product is released (its effect ended) or when the person leaves the infected states, which is MA15's fix (Stage 4), assumed here; so routine treatment of repeated episodes treats every episode.

### Products

| Product | New |
|---|---|
| `ss.Diagnostic` (renamed from `ss.Test`) | `results={category: probability}`, each probability a number or a dict by state (each a number or `ss.by`), the last category `None`; or a frame of state by result; categories in order of severity |
| `ss.Treatment` | `cure=` a scalar (every state it acts on) or a dict by state; `to=` a dict by source state |
| all | values keyed by disease name, and `disease=` a list or an `ss.Strains`, from the product's keys if not given; `stock=` (below) |

A test's results must add up to at most 1 for each state, else `ValueError` naming the state. Under an `ss.Strains` with cross-immunity, the most severe result over the members is exact under every solver; without a crossing it needs agents (below). `ss.Test` raises, saying it is now `ss.Diagnostic` (Starsim 3's `ss.Dx` already maps there in the migration script).

### Results

Each step keeps `delivered`, `eligible`, `coverage` and `cost` as now, by stratum. Each outcome adds a flow result of its name: a two-valued test `positive`, `negative`, `false_positive`, `false_negative`; a categorical test one per category; a curing treatment `cured`, `failed`, `unneeded`; a visit `attended`; with `stock=`, `not_done`; with `queue=`, `never_ran` and `waited`. This replaces `diagnosed` (decision 8) and gives STIsim's accuracy and treatment results (14 series in Starsim 3 against 3 today), by sex. A delivery with `target=` adds `holding` (the stock), `target`, and with `above=` other than `'keep'`, `stopped`. Reported cases are `a.positive + b.positive`, as the spec's `cases` example already reads with `diagnosed`. A cascade adds its table (above), built from its steps' results, so it costs nothing during the run.

## A move with no wait ("immediately")

Several designs and findings need people to pass through a state, or to make a move, with no time spent: the family's `exclusive=` (cancer from one genotype ends the course in the others), a visit or follow-up with no delay, resistance at treatment failure, the behavior vocabulary's counters (`move=` on an event), HP2 (a junction state), HP16 (a split with no dwell), and ST26 (`I >> R @ ss.perday(1e6)`, the idiom for "at once", which takes 277 s a simulated year as an ODE because the solver follows the fastest rate with substeps). Today each has its own answer or none. One rule covers them.

**The spelling** (decision 17, option a). A transition with `dur=0` is immediate: people who are in its source (and meet its `where=`) at the end of a step's moves go through it in the same step. With several destinations, `p=` splits them, as for any branch. A transition with `on=` an event is immediate for the people of the event. Nothing else is immediate: a transition with no rate, no `dur=`, no `on=`, and no `p=` keeps raising `UndrivenTransition`, since a forgotten rate is the commoner mistake, and HP16's split with `p=` and nothing else keeps its error, which now says `dur=0` (today it says to write one transition per destination).

**Cliff's fallback** (option b). If `dur=0` turns out to be written a lot, any transition with `p=` and no rate or `dur=` becomes immediate, and the error for it goes. What to count: how often the ports write `dur=0` by hand, against a split with only `p=` (HP16's case). The family's `exclusive=` and the behavior counters are built on the rule and write nothing, so they do not count. Switching later changes no model that runs today: it only turns an error into a meaning.

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
| a step on one disease's event detecting another's state (TL37) | exact on the joint groups of a crossing; uncrossed: `CapabilityError` (today 219.6 against agents' 96, or 0.0, or a crash) | same | same | UIDs |
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
| `count_events=` log | none (flows by group are results already) | same | same | per person |
| `ss.Event` of a coded module, coded `disease=`, `trace=` | `CapabilityError` (the module runs as agents) | same | same | yes |

The shared resource's rows are in its own section below. `solver='auto'` picks agents for anything in the agents-only rows, as now; an explicit compartmental solver stops with the reason, naming the step and the keyword. Every level of memory multiplies the groups of every disease, so a property is added only for readings something uses, and `describe()` prints each with its levels and a note when the groups grow more than tenfold.

**`to_spec()`**: a step's `who` is written as the text of the expression (`"screening.ascus | triage.positive"`), `delay`, `target`, `above`, `priority`, `uses`, `level` and `queue` as typed values and condition text, and a test's `results` as a table; a cascade as the list of its steps' names; a resource as its amounts and dimensions; the properties memory adds are derived, not stored, so the hash depends only on what was written. **`to_code()`** writes resources and products, then each step as a variable in dependency order, with `who=` and `what=`, then the cascade; a function (`priority=lambda ...`) is written with its source, as now.

## Deliveries in the kernels

Today the kernels take a delivery as one move between groups (`ss.kernels.Shifts`: `due`, `sel`, `cov`, `to`, `rest`). The proposal replaces that with a small program per time point, a list of operations on registers that each hold a number of people per compartment and group, run at the start of each step. The steps are built into it at `init()`, in dependency order.

| Operation | Does |
|---|---|
| `SELECT` | the eligible: a register from the compartments, times a share per compartment and group (states, stratifications, memory levels) |
| `EVENT` | a transition's people from the last step's flows, in the destination's compartments (as `event_frac()` does now) |
| `THIN` | times coverage, or a product's `stock=`, from a table by time |
| `CAP` | to a capacity from a table, in priority classes; for a resource, over the registers of every step that uses it (below) |
| `TARGET` | holders counted, the gap filled from the register in priority classes; with `above=`, the excess stopped in reverse priority |
| `SHIFT` | the register's people moved to other groups (holding, offered, memory, pending and queue levels), and the register with them |
| `MOVE` | the register's people moved to another state's entry compartments, with branches and maps between dimensions, and the flow credited (a cure, `move=`, a transition with `on=`, an immediate move with `where=`) |
| `SPLIT` | a test: one register into one per result, by shares per compartment and group |
| `UNION`, `INTERSECT`, `MASK` | `\|` and `&` of two registers; a register times a condition's shares |
| `TALLY` | a register's total per group into a result |

Under the ODE every operation is an expected value; under the stochastic solvers `THIN` and `SPLIT` draw binomials and multinomials, and `CAP` and `TARGET` multivariate hypergeometrics, from the kernel's stream. The stepped path calls the same kernel function at each step, so a stepped and a whole run draw the same numbers. The many-run kernel (`run_many`) and the crossing's whole run (PL3) take the same program, a crossing's compartments being joint. What stays in Python: `when=`, `deliver=`, coverage or capacity given as a function of the sim, and anything agents-only. JAX: the same operations in `gradients.py`, second (`CAP` and `TARGET` have kinks, so gradients are piecewise, which the docs say).

This is the design `deferred.md` sketched (2026-10-16), with targets and categorical results added. Expected cost: a cascade of four steps a few µs a step against about 170 µs now.

## Thanzi la Onse's health system

TLO's `HealthSystem` is what every one of its conditions' care goes through: appointments (HSI events) with footprints in staff minutes by officer type, facility level and district; consumables available with a probability by facility and month; beds and equipment; a queue ordered by priority, with a time after which an appointment is dropped (`tclose`); facility levels drawn by care-seeking; and the counterfactual of a perfect health system. Its default mode 1 is elastic (no staff limit in practice), and the generic first appointment has used no staff time since 2023, but the question its health-system papers ask is "what if staff were the limit". The review ported a slice of it (a symptom, the first appointment, a test with stock-outs, a treatment queue in staff minutes) as 12 declared lines plus a coded health system of 152 lines (`v4port/healthsystem/v4_toyitis.py`), and matched TLO: prevalence 0.1139 against 0.1138 elastic, 0.1145 against 0.1168 constrained, with waiting times that match. Its summary: the deliveries vocabulary covers one cascade well, but has nothing shared.

### What the cascade's steps already close

| Finding | What TLO needs | Closed by |
|---|---|---|
| TL41 | two follow-ups of one result (treatment for stages 1 to 3, palliative care for stage 4): `on_positive=[a, b]` crashes at the first positive; an appointment with no product | peers: any number of steps take `who=` the same outcome (step 1), and `on_positive=` goes (step 4); `ss.visit()` (step 2) |
| TL43 | a symptom as a level of the disease (`ss.transfer(sym=True)`) leading to care: `on=` a transfer raises | transfers are events (step 1): `who=oc.symptom`. Dysphagia no longer needs to be a second disease with six modifiers |
| TL38 | a state entered only by a delivery's `move=` (hospital) becomes the first state: everyone starts in hospital, 34,001 bed-days against 4,066 | the steps are known at `init()`, so their destinations count as entered (step 4, about 5 lines) |
| TL36 | three doses of one vaccine give 351 / 31 / 5; routine treatment once per lifetime | a triggered step gives its product each time (doses triggered by the dose before, `penta.count` the dose count), and a routine step re-arms with MA15's fix (step 13) |
| TL45 | "test not done" when out of stock, and a fallback test | `stock=` on the product and the outcome `not_done`; a fallback is a step on it, `ss.screen(who=rdt.not_done, what=microscopy)` (step 13) |
| TL46 | one first appointment over all of a person's symptoms; appointments by level | `ss.visit(who=[oc.symptom, cmd.symptom, tb.symptom])` is a union, each person once a step; the level is `level=` (below) |
| TL44 | EPI records doses for nine diseases it doesn't model | `disease=None`: delivered and counted, with no effect (step 13) |
| TL37 | a test on one disease's event detecting another's state is wrong under compartments (219.6 against 96) or crashes | the delivery program evaluates it on a crossing's joint groups, and refuses the uncrossed case (step 11) |
| TL48 | care triggered by a symptom acts one step late: 3 months at TLO's step, against 0 to 4 days | partly. Dependency order puts a triggered step at the first boundary after its event, under every solver, and `describe()` says "one step after the event"; less than one step needs a shorter `dt` (or event times within a step, TL55). The cascade states the lag rather than removing it |

Not closed by the steps alone: staff time shared across diseases (TL39), a queue (TL40), a property recorded at an event (TL42), and capacity by district (TL47). They are the shared resource, below. TL35 (closures over a coded module read the original) and TL49 (the result-index warning) are fixes outside this design.

### A shared resource: the proposal

TL39 to TL46 are faces of one gap: deliveries each have their own coverage and capacity, and nothing they share. One noun and a few keywords make TLO's health system declarable, and serve any model with a constrained service (beds in COVID models, test kits in MEmilio's test-and-trace, ART slots in Atomica's programs).

```python
staff = ss.Resource(tlo4.minutes(by=['district', 'level', 'officer']), per=ss.days(1))   # minutes a day by district, facility level and officer type (TL39)
opd   = {staff: dict(clinical=27, nursing=18, pharmacy=9.45)}                              # a footprint: what one appointment uses, reused by name

endoscope = ss.Diagnostic(sensitivity=dict(low_grade=0.8, high_grade=0.9, stage1=1.0, ...), stock=tlo4.availability('endoscope'))  # in stock by district and month

first_appt = ss.visit(who=[oc.symptom, cmd.symptom, tb.symptom], uses=opd, level=tlo4.care_level)  # one first appointment over all symptoms, at a level drawn by care-seeking (TL46)
endoscopy  = ss.screen(who=first_appt & oc.symptom.ever, what=endoscope, uses={staff: dict(clinical=60)}, level='2', queue=ss.days(14))
barium     = ss.screen(who=endoscopy.not_done, what=barium_swallow, level='2')              # the fallback when the endoscope is out (TL45)
found      = endoscopy.positive | barium.positive
surgery    = ss.treat(who=found & ~oc.stage4, what=oesophagectomy, uses={staff: dict(clinical=240, nursing=480)}, queue=ss.days(30), priority=1, record=dict(stage_dx=oc.state))  # (TL40, TL42)
palliative = ss.treat(who=found & oc.stage4, what=palliative_care, uses=opd)               # two follow-ups of one result (TL41)
oc_care    = ss.cascade(first_appt, endoscopy, barium, surgery, palliative)

sim = ss.Sim(people=people, diseases=[oc, cmd, tb], interventions=[oc_care, cmd_care, tb_care])   # every cascade draws on staff
res = ss.parallel(perfect=sim.new(resources='unlimited'), constrained=sim, n_runs=10).diff('perfect')
```

**`ss.Resource(amount, per=None)`** is a quantity available per period: a number, an `ss.by(...)` over dimensions, a table by dimension (as here), or a time series. Its dimensions are of three kinds: people's (`district`), matched to the person reached; the step's (`level`), matched to the step's `level=`; and the resource's own (`officer`), which a footprint names. The amount is scaled to the step, as rates are, and what is not used in a step is not carried over (TLO's daily capability). With `per=None` it is a supply that is used up (test kits, a stockpile), restocked only by a time series. Resources are found from the steps that use them, as products are, so nothing is passed to the sim.

**`uses=`** (the footprint) on any step: a dict of resource to amount per person reached, the amount a number or a dict over the resource's own dimension (`dict(clinical=27, nursing=18)`). A footprint is a plain dict, reused by name (`opd`); TLO's appointment types are a table of them (`tlo4.footprints()['Under5OPD']`). The step reaches someone only if the whole footprint fits in what is left of each part, at their district and the step's level (TLO's rule); otherwise they wait if the step has `queue=`, and are lost to capacity if not.

**`level=`** on any step: the facility level, a value or a distribution per person (TLO draws it from care-seeking), or a condition-keyed `ss.by`. A resource with a `level` dimension is drawn on at that level, and every step's results are by level (TLO's appointments: 611 / 3,132 / 923 at levels 0, 1a and 2).

**`priority=`** across steps: the people of every step that uses a resource are pooled and served class by class, each step's `priority=` giving its people's classes (a number, the same class for all, 0 first as TLO's priorities; or conditions, as within a step); within a class, those who have waited longest first, then at random.

**`queue=`** on a step with `uses=` or `capacity=`: how long someone not served waits, a duration (TLO's `tclose`), or `True` (until served). The default, `None`, loses them, as now. Someone waiting is served before anyone newly reached in their class, and leaves the queue if they stop meeting `who=`'s conditions (cured, dead). Leaving at the end of `queue=` is the outcome `never_ran` (TLO's name), an event, so a referral can follow it. The time waited is a result (`waited`). This answers "When capacity runs out" in spec 19, and D31 (a capacity queue in arrival order).

**`stock=`** on a product (or its step, as any product keyword): the chance it is in stock when someone is reached: a number, `ss.by(district=...)`, or a time series (TLO's consumables by facility and month). Someone reached when it is out uses the footprint (the appointment happened), is not given the product, and is in the outcome `not_done`. This differs from `coverage=`, which is people not offered at all: they use nothing and are no event. A counted supply is a resource with `per=None` in `uses=` instead.

**`record=`** on any step: properties set on the people reached, from their state at the time: `record=dict(stage_dx=oc.state)` (TL42), read like any property (`oc.stage_dx == 'stage2'` in a transition's `where=`, or a disability weight).

**Results.** `sim.results.staff` has `available`, `used` and `fraction_used`, by the resource's dimensions; each step has `appointments` (people reached, by level), `waited`, `never_ran` and `not_done`; a cascade's table gains `lost (queue)` and `not done` columns. **`sim.new(resources='unlimited')`** makes every resource unlimited and still counts what is used: TLO's perfect health system, the counterfactual of its papers (decision 23).

### A shared resource: semantics under agents and compartments

| Feature | Agents | ODE | Binomial, Gillespie |
|---|---|---|---|
| a resource | per step and group of its people's and steps' dimensions, the amount left; the steps' people pooled, ranked by class, then time waited, then at random, and served while the whole footprint fits: TLO's queue | a `CAP` over the registers of every step that uses it: in each group, served = the demand, or what fits, whichever is less, where what fits is the least over the footprint's parts (the binding officer type); shared class by class, and within a class in proportion to demand | as the ODE, the served drawn by a multivariate hypergeometric within a class |
| `level=` a value | a group of the resource | same | same |
| `level=` a distribution | drawn per person | split by shares | drawn |
| `queue=` a duration | per person, the step they joined | pending levels of one step each up to the queue's length, served first, the oldest first; the last level leaves as `never_ran` | same, drawn |
| `queue=True` | until served or ineligible | one pending level with no exit but service and the conditions of `who=` | same |
| `stock=` | one draw per person | a share, `not_done` the rest | binomial |
| a resource with `per=None` | counted down | counted down in expected numbers | drawn |
| `record=` | an array per person | a dimension with a level per recorded value, added only if read, with a note on the groups (as memory) | same |
| a people's dimension of the resource (`district`) | yes | needs it as a dimension of people (h4); otherwise `CapabilityError` | same |
| `priority=` a function | ranked | `CapabilityError` | same |

Under JAX, `CAP` over a resource has kinks (the least over parts, the demand against what fits), so gradients are piecewise, as for `capacity=`. The coded 152-line health system of the review (`v4port/healthsystem/v4_toyitis.py`) is a working prototype of the agent side and is the reference for its tests; TLO's 32 capped follow-ups (TL40: 2,825 treatment appointments against 3,559, and 31.5 s against 0.9 s) are the case the queue must fix.

## Fit with what exists

**Reused.** `on=`, `where=`, `coverage=`, `capacity=`, `trace=`, `move=` and `ss.Delivery` unchanged; events are what `on=sir.infection` already is; memory as compartments uses the existing mechanism for levels of time since delivery (`Held.init_stages`) and for pending results, and a queue the same pending levels; `target=` by stratum and a resource by dimension use `ss.by`; `|` and `&` as for conditions; the immediate move extends HP2's junctions.

**New public names, each against extending an existing one.**
- `ss.cascade()`: Cliff's decision 2. It groups, names, tabulates and checks; it adds no behavior, so a list of steps still works anywhere a cascade does.
- `ss.visit()`: a step with no product. The alternative, `ss.Delivery()` with no product, read as "a delivery of nothing". It is also TLO's appointment: the review's `ss.Visit` noun (a footprint at a level, given to deliveries as `visit=`) is not added, since a footprint is `uses=` on any step and a delivery given at a visit is a step with `who=` the visit; one name, lowercase, for the step (decision 18).
- `ss.Resource`: the one new noun of the health system. `capacity=` cannot be shared, since it is per step.
- `ss.Diagnostic`: a rename of `ss.Test`, so that no product and delivery differ only by case (`ss.Test` and a test).
- `ss.Event`: modules written as code need a way to publish an event with memory; `on=` a function of the sim already gives the people but no name, memory, or `describe()` line. Defined with `define_states()`, so no new method.
- `.ever`, `.count`, `.since` on events: the alternative, named states on `ss.screen`, covers tests only and leaves h4 to invent its own.
- Outcome attributes (`.positive`, `.cured`, `.not_done`, `.never_ran`, a test's categories): named by the product, as transitions are named by the disease.
- Keywords `who=`, `what=`, `target=`, `above=`, `priority=`, `delay=`, `operator=`, `count_events=`, `uses=`, `level=`, `queue=`, `record=` on steps; `results=` on `ss.Diagnostic`; `stock=` on products; `on=` and `dur=0` on transitions. `capacity=` cannot carry a stock (it is a flow), and coverage is per round.

**Removed or replaced.** `ss.Test` (now `ss.Diagnostic`); `on_positive=` and `on_negative=` (replaced by `ss.treat(who=test.positive, ...)`; a removed name raises saying what to write, and the migration script rewrites the nested form into steps); the `diagnosed` result (now `positive`); `ss.Delivery`'s public `result` and `ti_result` states (internal pending results); automatic follow-up names (`screen_positive`); `ss.kernels.Shifts` (replaced by the delivery program); the `perday(1e6)` idiom in `skills/import-model/references/idioms.md` (now `dur=0`). Nothing is added for compatibility.

**Spec changes.** [08-interventions.md](../../design/08-interventions.md): "Follow-ups" becomes "Cascades" (`who=` and `what=`, triggered and routine steps, `ss.cascade()`, peers, events, memory, order, visits), a new "Shared resources" (`ss.Resource`, `uses=`, `level=`, `queue=`, `priority=` across steps, `stock=`, `record=`), the delivery keywords table gains the keywords above and loses `on_positive=`/`on_negative=`, "Tests" (now diagnostics) and "Treatments" gain categorical results, cure by state or scalar, and values by disease, "What a delivery reports" lists outcome results, resource results and the cascade's table, and the solver table gains the rows above. [03-states-and-transitions.md](../../design/03-states-and-transitions.md): `dur=0` and `on=` on transitions, the immediate move, transfers as events. [09-results.md](../../design/09-results.md) and [19-open-questions.md](../../design/19-open-questions.md): `cases = a.positive + b.positive`; "When capacity runs out" answered by `queue=`. [12-backends.md](../../design/12-backends.md): the delivery program with resources and queues, and immediate moves without a rate. [14-model-spec.md](../../design/14-model-spec.md): steps' `who`, `target`, `above`, `priority`, `delay`, `uses`, `level`, `queue`, test results, resources and cascades. [17-vocabulary.md](../../design/17-vocabulary.md): the names above. [18-conformance.md](../../design/18-conformance.md): the contracts below, and item 13 reworded. [01-objects.md](../../design/01-objects.md): `ss.Event` among module parts, `ss.Resource`. [10-execution.md](../../design/10-execution.md): steps in dependency order.

## Implementation plan

| Step | Size | Files | Tests |
|---|---|---|---|
| 1. Events: steps, outcomes, products and transfers (TL43) as events, `\|`, `&`, lists with `operator=`, and `& condition`, `ss.Event`, dependency order and the cycle error; `ss.LegacyPregnancy`'s three events (ST13, ST15, ST16) | M | `interventions.py`, `conditions.py`, `products.py`, `arrays.py`, `demographics.py`, `loop.py` | a union fires once per person, written as `\|` or a list; a follow-up on the step of its trigger under agents and ODE; ANC: the same attendees take both tests; a step on a transfer |
| 2. The step's spelling: `who=` (conditions, events, `&`, `\|`, lists with `operator=`; triggered if it contains an event, routine otherwise; `event \| condition` refused; `on=` and `where=` joined by `&`), `what=` as the first positional argument; `ss.cascade()` over any steps, its table and funnel, the note on steps referred to outside it, its `describe()` block; `ss.visit()` | S to M | `interventions.py`, `conditions.py`, `results.py`, `presentation.py`, `spec.py` | `ss.screen(who=a, what=t)` equals `ss.screen(t, on=a)`, and `who=c` equals `where=c`; `who=[a, b], operator='all'` equals `who=a & b`; a step referring to one outside its cascade gives the note; the table's reached and lost add up to each step's eligible; `to_code()` round-trips with `who=` |
| 3. Memory: `.ever`, `.count`, `.since`, found at `init()`, as agents and as levels (ST14); `count_events=` `True` or a list, and the agents' log | M | `conditions.py`, `interventions.py`, `solvers/engine.py`, `solvers/compartmental.py` | nobody screened twice within five years; positives at the end equal to those ever positive; `since` as levels against agents at a fine step; the log has one row per event per person, and only the listed events with a list |
| 4. Peers replace nesting: remove `on_positive=`/`on_negative=`, names from variables (D27), `diagnosed` to `positive`, outcome results by stratum, `ss.Test` to `ss.Diagnostic`, delivery destinations count as entered (TL38); the migration script's rewrite | S | `interventions.py`, `products.py`, `graph.py`, `migrate.py`, `skills/migrate-starsim3`, docs | the spec's screening examples give the same numbers as before; names and results as written; a hospital state filled only by a move starts empty |
| 5. The immediate move: `dur=0` and `on=` on transitions, under every solver; HP2's junctions generalized; the compartmental form without a rate (ST26, HP16) | M | `transitions.py`, `graph.py`, `solvers/agents.py`, `solvers/compartmental.py`, `kernels.py` | a junction holds nobody under every solver; `I -> R` with `dur=0` as an ODE takes no longer than without it; resistance at failure is 5% of `failed` |
| 6. `delay=` on every triggered step (HP26) | S | `interventions.py` | a treatment three months after the test, under every solver |
| 7. Categorical tests with a `None` last category, cure by state or scalar, values by disease, `disease=` a list or `ss.Strains`, the HP7 error (HP24, HP25, HP7) | M | `products.py`, `interventions.py` | 20,000 cytologies report 20,000 tests and cost 200,000; the shares by state match HPVsim's table within 1%; `cure=0.936` equals the dict over every infected state; a bivalent vaccine protects against both; worst over a crossed family as ODE against agents |
| 8. `target=`, `above=`, `priority=` (ST12) | M | `interventions.py` | the stock follows a time series within 1% while enough are eligible, and with `above='stop'` also when it falls; the lowest priority is served last and stopped first; by stratum |
| 9. Coded diseases as `disease=` (PL2) | S | `interventions.py`, `products.py` | screen and treat on `ssl.AgentSIR` |
| 10. Standard `prevalence` as a ratio (ST20) | S | `solvers/engine.py`, `diseases.py` | `ss.SIS(...).prevalence.select(risk='high')` |
| 11. The delivery program in the kernels (DF1), in `run`, `run_many`, the stepped path and the crossing (PL3); a test on one disease's event detecting another's state on joint groups (TL37); tables a part at a time on the stepped path (PL7) | L | `kernels.py`, `solvers/compartmental.py`, `solvers/crossing.py`, `interventions.py` | whole against stepped: identical for the ODE, the same draws for binomial; a cascade with categorical test, delay, capacity and target, ODE against the mean of 200 agent runs; TL37's repro, crossed, against agents (96 diagnosed, analytic 104); the cascade ODE at under twice the time of no intervention |
| 12. The same in JAX (KI23) | M | `gradients.py` | `sim.gradient()` of a screen-and-treat model against finite differences |
| 13. Giving a product again (TL36, with MA15's re-arming), `stock=` and `not_done` (TL45), `disease=None` (TL44) | S to M | `products.py`, `interventions.py` | pentavalent at 6, 10 and 14 weeks as steps triggered by the dose before: 351 / 327 / 327, and `penta.count` 3 for the fully dosed; repeated diarrhoea treated every episode; at stock 0.5, half `not_done`, and the fallback reaches them |
| 14. `ss.Resource`, `uses=`, `level=`, `priority=` across steps, resource results, `sim.new(resources='unlimited')`, as agents (TL39, TL46) | M to L | `interventions.py` (or a new `resources.py`), `products.py`, `results.py`, `sim.py` | two diseases sharing a clinic of 10 a day treat 10 a day, the higher class first; the review's slice against the coded health system and TLO (prevalence 0.1145 against 0.1168, waiting times); appointments by level |
| 15. `queue=` and `never_ran` (TL40); `capacity=` and resources by `ss.by` (TL47) | M | `interventions.py` | TLO's per-district capacity as one queued step: treatment appointments within noise of TLO's 3,559, against 2,825 for capped steps, in under the coded system's 0.9 s; nobody waits past `queue=` |
| 16. `record=` (TL42) | M | `interventions.py`, `conditions.py`, `solvers/compartmental.py` | the stage at treatment recorded and read in a transition's `where=`; as an ODE against agents |
| 17. Resources, queues and stock in the delivery program, and in JAX | M to L | `kernels.py`, `solvers/compartmental.py`, `gradients.py` | a shared resource with two steps and a queue, ODE against the mean of 200 agent runs; `fraction_used` within 2% |
| 18. Spec and docs | S | `v4_design/design/*.md`, `docs/user_guide/modules_products.qmd`, interventions tutorials | the docs' examples run |

Steps 1 to 4 come first and in that order; 5 to 10 are independent of each other; 11 needs 1 to 8. Step 5 can go first of all, since the family and the behavior vocabulary need it too. Steps 13 to 17 are the TLO review's suggested items 4 and 6: 13 needs 1 and MA15 (Stage 4); 14 to 16 need 1, 2 and 4; 17 needs 11 and 14 to 16. A contract test, `test_contracts.py::test_cascade`, runs one cascade (union trigger, categorical test, delay, capacity with priority, target, memory, a visit, a resistance transition) under all four solvers and checks the agreement and the reported numbers above; `test_contracts.py::test_resource` does the same for two cascades sharing a resource with a queue, once step 17 is in.

**Implementation notes, steps 5 to 10 (2026-10-20).** Step 5: `dur=0` is `Transition.immediate`; an unconditional one is an immediate branch (no compartment, no dwell) under every solver, and `dur=0` with `where=` becomes `on=` the condition. Triggered transitions (`on=` a condition or an event of the same module) now run under the compartmental solvers, moved at the end of each step after its flows (so people put in the source by an intervention between steps move one step later than agents would); `on=` another module's event or a function stays agents only. ST26's idiom: `X >> R @ ss.perday(1e6)` 48 s for 30 days as an ODE, `dur=0` 0.08 s. Step 6: `delay=` on any triggered step, a per-person due step as agents and levels `delayed_<step>` under compartments; on `ss.screen`, `delay=` is now the step's, and the result's delay is only `ss.Diagnostic(delay=)`. A delay drawn from a distribution is agents only (the exponential-as-a-rate form is not built). A `delay=` does not yet break a cycle in the dependency order. Step 7: `disease=` a list or an `ss.Strains` keys the product's values by each disease at `init()` (`Product.key_by()`); a test or a cure of several is agents only (the crossing is chosen before the deliveries are initialized). `cure=0.936` was already the scalar form. Found: `stopped` was undercounted under compartments (tallied after the move), fixed. Step 8: the merge had the rest; added the test of `above='stop'` on a falling target. Step 9: a coded disease is a `disease=` as agents (`ss.connectors.AgentModifiers` is its engine); `ssl.AgentSIR.clear_infection()` leaves the cured in no state until their scheduled recovery, which is that class's own behavior. Step 10: the standard `prevalence` keeps its parts by group, as a declared ratio.

**What the ports then delete** (estimated from the class sizes): STIsim, about 2,200 of its 3,700 lines of interventions: `STITest` and its six subclasses, `ANCTest`, `STITreatment` and its two subclasses, `SymptomaticTesting` and `SyndromicManagement`, the target machinery of `ART`, `VMMC` and `Prep` (with `above='stop'`, including the removal), and `parse_coverage`/`resolve_coverage_targets` (data loading into `ss.timeseries` and `ss.by` stays). `PartnerNotification` (300 lines) is `trace=` once its partner rules are; that is outside this design. HPVsim: the wrapper classes in `interventions.py` (most of its 723 lines), `dx.administer()` and the hierarchy, `treat_delay`, `treat_num`'s pseudo-test, and the `linked_txvx` eligibility lambdas; the CSVs stay, read into tables. TLO's port: the coded 152-line health system, and the booking functions in `on_positive=`. All three packages' cascade examples become one `ss.cascade()` of one line per step, as in the use cases.

## Out of scope, and the other Phase H documents

**Out of scope.** Partner rules for notification beyond `trace=`; budget allocation across programs (`ss.optimize`); `after=preg.delivery` as a dwell (ST18); optimal control. From TLO: beds as anything but a resource of bed-days with `uses=` (equipment likewise); TLO's squeeze factors (vestigial) and mode 2's overtime rule; care-seeking itself, which is a disease's transition or a `level=` distribution; and the burden of disease (the review's section 6, its own design).

**h2, a family of diseases (`ss.Strains`).** Assumed: `ss.Strains(name, members, ...)` with members by name, the same state names in each, and a crossing in compartmental runs when the family has cross-immunity. This design uses `disease=family`, keys `'hpv16.cin'` and `'cin'` (each member), and "worst over the members", which is exact only in a crossing. Its `exclusive=` is built on the immediate move above.

**h3, per-act transmission.** Assumed: a product's `acquisition` efficacy scales `rel_sus`, which multiplies the per-act probability, so PrEP and VMMC need nothing more from h3. Condoms as an intervention (h3's revision) are an `ss.modify` delivered by a step, with `who=` like any other.

**h4, behavior and dimensions over people's properties.** h4 should use `.ever`, `.count` and `.since` of events for counters and time since an event (decision 4), `ss.Event` for events of FPsim's coded modules, and the immediate move for a counter's `move=` on an event. Assumed from h4: sex, age and district as dimensions in compartmental runs (D15), so `who=ss.ppl.female`, `target=ss.by(age=..., sex=...)` and a resource by district run under every solver; until then they are agents only, as now.

## Decisions

### Second round (answered)

Kept as the record, with what this draft does after each of Cliff's notes.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 10 | The step spelling | (a) `who >> what`, with `on=`/`where=` as the same thing in keywords; (b) keywords only, with `ss.cascade()` grouping them; (c) `who >> what @ coverage`, as `>> ... @ rate` for diseases | (a). It reads as a flowchart and as the disease arrow. (c) is shorter but `@` would mean coverage here and a rate in diseases, and coverage is one of several keywords a step needs (`interval=`, `delay=`, `capacity=`), so it saves little. The cost of (a) is parentheses around a trigger of several parts, which the refusals catch [CK: dislike; see my comments above; try again with the API] (done: Cliff's keywords, `ss.screen(who=..., what=..., ...)`; triggered if `who=` contains an event, routine otherwise; `on=`/`where=` join `who=` by `&`; `>>` and its refusals dropped) |
| 11 | Steps in `ss.cascade()` | (a) positional, named from their variables, as in the examples; (b) keywords, `ss.cascade(screening=..., triage=...)`; (c) both | (a). Later steps must refer to earlier ones (`screening.ascus`), which needs variables anyway; with keywords the step is written in one place and referred to as another object. (b) reads like `ss.Disease(..., infection=...)` but only for cascades with no step referring to another [CK: the cascade can be anything, not necessarily screen/triage/treat. should be agnostic to the steps.] (done: any deliveries in any number and order, positional, named from their variables where possible, run in dependency order, a cycle raises; "What a cascade adds over a list" answers Cliff's question in the example) |
| 12 | The test product's name | (a) `ss.Diagnostic`; (b) `ss.Assay`; (c) keep `ss.Test` (it does not differ from `ss.screen` only by case) | (a). It says what it is, and no verb is near it. `ss.Treatment` and `ss.Vaccine` stay: they differ from `ss.treat` and `ss.vaccinate` by more than case, and are the common words [CK: ok] (done: `ss.Diagnostic`) |
| 13 | A step with no product | (a) `ss.visit()`; (b) `ss.attend()`; (c) `ss.appointment()`, TLO's word | (a), revisited after the TLO review, which may give a visit a type and a footprint [CK: i'm ok with visit] (done: `ss.visit()`; after the TLO review its footprint is `uses=`, and the review's `ss.Visit` is not added; decision 18) |
| 14 | Resistance and other outcomes of an event | (a) a transition of the disease with `on=` an event; (b) `ss.Delivery(move=..., on=...)` as in the first draft | (a), as Cliff said. (b) keeps working, since it is the general move, but the docs use (a) [CK: ok] (done: as (a)) |
| 15 | Above a stock target | (a) `above='keep'` (default), `'stop'`, or a rate; (b) `'keep'` only, as approved; (c) a function too | (a). It lets STIsim's ART keep its removal, which the first draft lost, with no function [CK: ok] (done: as (a)) |
| 16 | `count_events=True` | (a) memory for every event, and a per-person log as agents; (b) memory only; (c) the log only | (a). The log is what "save results for later analyses" asks for; memory alone is per person but only the last time [CK: yes, and other than just true/false also allow a list of states/events to track] (done: `count_events=` `False`, `True`, or a list of states and events; a state means the events of entering it) |
| 17 | "Immediately" | (a) `dur=0` on a transition, and `on=` an event; a split with only `p=` keeps its error, which says `dur=0`; (b) any transition with `p=` and no rate or `dur=` is immediate; (c) a new `ss.now` as the driver (`I >> R @ ss.now`) | (a). It reuses `dur=` (a stay of zero), keeps the forgotten-rate error, and is one rule for the junction, the condition, and the event [CK: i'm ok with (a) unless it comes up a lot, in which case would prefer (b)] (done: (a), with (b) recorded as the fallback and what to count, in "A move with no wait") |

### Decisions still open (third round)

For Cliff to answer with `[CK:]`. Only new choices, most from folding in TLO.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 18 | A visit's footprint | (a) `uses=` on any step, a dict of resource to amounts, reused as a variable (`opd`); (b) a noun, `ss.Appointment(uses=..., level=...)`, given to steps as `appointment=`; (c) the review's `ss.Visit` noun given as `visit=` | (a). No new noun, and every step (a visit, a test, a surgery) can carry its own footprint, as TLO's HSIs do. (c) puts `ss.Visit` beside `ss.visit`, two names differing only by case, which decision 12 ruled out; (b) is that noun under another name, and adds a second way to say "a delivery given at a visit" (`appointment=` against `who=` the visit) |
| 19 | The shared noun | (a) `ss.Resource`; (b) `ss.Capacity` (but `capacity=` is per step); (c) `ss.Supply` | (a), the review's word. It covers staff minutes, beds and kits, and is not a keyword already |
| 20 | The queue's spelling | (a) `queue=` a duration, `True` (until served), or `None` (lost, the default); (b) `wait=` with the same values; (c) `max_wait=` and `queue=True` separately | (a). One keyword says both that there is a queue and how long people stay in it; the outcome is TLO's `never_ran` |
| 21 | `on=` and `where=` | (a) kept and working, documented only in the reference as joining `who=` by `&`; tutorials and `to_code()` use `who=`; (b) documented as equals of `who=`; (c) deprecated, with a note | (a). Existing v4 code and the migration script's output keep working, and readers see one spelling. (c) would churn code that is right |
| 22 | An event `\|` a condition in `who=` | (a) raises, saying to write two steps; (b) allowed: triggered for the event's people and routine for the condition's | (a). The rule "triggered if it contains an event" then has no exception, and a union of an act and a schedule is two steps in every real cascade we have |
| 23 | The perfect-health-system counterfactual | (a) `sim.new(resources='unlimited')`, use still counted; (b) `sim.new(resources=None)`, resources removed with their results | (a). The comparison TLO's papers make needs the minutes a perfect system would use, which (b) loses |
