# Remaining work, staged

Written 2026-10-19. This plan follows [implementation_oct18.md](../oct17/implementation_oct18.md) (phases A to K, now largely done), Cliff's answers in [decisions_oct19.md](../oct17/decisions_oct19.md), and his `[CK:]` notes on the six Phase H designs in [oct17/designs/](../oct17/designs/README.md). To list what is still open, four agents took the October 18 status tables ([status_oct18/](../oct17/status_oct18/)), the canon and naive reviews, the October 19 worklogs, `admin/deferred.md`, and `admin/implementation.md`. They dropped whatever the October 19 work fixed, judging from the worklogs, the CHANGELOG, and greps of the code. Nothing was re-run, so numbers are from when each item was found. Their full lists, one line per item, are in the session scratchpad and are not kept. This file carries everything they found.

Sizes: S is under a day of agent work, M a few days, L more. **SWA** marks a silent wrong answer.

## In one page

- **Cliff's October 19 answers close the agents' step question.** Decision 16 says each method keeps its simplest, most natural reading, and that the differences between them get documented rather than removed. Item 18, B′, convention C, ST4, and CR7 therefore become documentation (Stage 0).
- **About 30 silent wrong answers are still open and need no design.** Most are S. They go first (Stage 1), together with the steps of the approved designs that fix silent answers: the behavior vocabulary's steps 1 and 2. The per-act steps 1 to 4 need a quick revision first (H0) and come right after.
- **Cliff approved most of the 36 design decisions, but asked for changes to four of the six designs.** The care cascade's user-facing spelling needs a second draft (`ss.cascade()`, clearer names, and a comparison with Thanzi la Onse). The family becomes `ss.Strains` with a simpler constructor. Per-act transmission needs clearer direction keys, condoms as an intervention, and the ODE approximation built rather than deferred. The counter spelling (`parity='+1'`) is open again. So there is a short design round, H0, before the code that depends on it. The destination results design (with `select(to=)` kept) and the given-dwell design (with rates allowed, option b) can start now.
- **About 25 crashes and misleading errors** are each S (Stage 2).
- **Performance, inference, the saved form, and importers** follow, in that order. The importers come last: they matter least to current users.

## Cliff's answers, and what each changes

### decisions_oct19.md

| Item | Answer | Action |
|---|---|---|
| 1, 16 | B′ was ok, but after B′ and C were each built and reverted: "revert to the simplest, most natural interpretation in each case, and document how the different models give slightly different results" | Stage 0: documentation, and the item closed. Agents are already in the natural form: a `dur=` dwell is drawn, and a `rate=` exit is a per-step hazard. Check that no half-applied convention is left |
| 2, 5, 7, 8, 9, 12, 17 | ok | Nothing (already applied) |
| 3 | Fix MA7 | Done by PB. The `after=` conveyor may have the same half step: Stage 1 |
| 4 | Exponential `ss.SIR` etc., with `ssl.AgentSIR` as Starsim 3's | Done by PNR |
| 6, 13 | One run a point for agent fits; a fresh seed per point | **Not done**: `calibration.py:2696` still uses the same seeds at every point. Stage 1 |
| 10 | Write the HP20 and HP18 designs | Done (`destination_results.md`, `given_dwell.md`) |
| 11 | Phase H: "i'll review" | Reviewed: see the next table |
| 14 | Two-parameter NUTS coverage is low priority | Moved to `deferred.md` |
| 15 | (a): keep `ti_entry` dating in `ssl.Agent*` | Nothing |

### The designs

| Design | Approved | Changed or questioned by Cliff |
|---|---|---|
| Disagreements (README) | All three: `ss.by(<family>=...)`, members side by side under the compartmental solvers, and the cascade's `.ever`/`.count`/`.since` as the user spelling | — |
| Care cascade | 1 (peers), 4 to 9 | **The example as a whole**: `ss.Test` (product) beside `ss.test` (intervention) is confusing; the state letters (`C`, `X`, `S`, `P`) are opaque; `default='normal'` is unclear; drug failure leading to resistance should not be an `ss.Delivery`, nor should a visit be "a delivery of nothing"; and Thanzi la Onse should be included in the review. **2**: add `ss.cascade()`, and make its logic "as clear as `S >> I @ beta*I`". **3**: `|` and a list are both valid and equivalent; `&` is valid too (`[a, b], operator='any'` is `a | b`, and `operator='all'` is `a & b`). **4**: add an optional `count_events=True` that keeps results for later analysis. **6**: may need machinery for user choices above the target |
| Disease family | D2 to D5, D7 | Name it **`ss.Strains`**, not `ss.Diseases`. Put `name` first. No `*[genotype(**p) for p in ...]`: let `ss.Strains(hpvsim.genotypes('hpv16 hpv18 hi5 ohr'))` do the expansion. Rename **`of=`**, which Cliff can't read. **D2**: allow a custom function for `combine=` (not differentiable). Cliff also asked "i thought the port was faster?" (answered below) |
| Per-act transmission | D1, D2, D4, D7 | **D3**: the direction keys (`m2f`, ...) feel like hard-coded magic; say how a user knows which keys are valid, and how injecting drug use fits. **D5**: "why can't it be above one? beta=0.03 with rel_trans=10 should give 0.3" (answered below; now a decision). **D6**: a condom or a microbicide should be an intervention (`ss.modify(rel_trans=..., who=...)`), not a parameter of the disease or the network. **D8**: keep `ss.MFNet` (yes, it stays). **D9**: build the partnership approximation for the ODE and the other compartmental solvers too, since it is a common use case, even if approximate |
| Behavior vocabulary | 1, 2, 4 to 9, FP25 to FP35, HP19 ("some things will be agent only, and that's fine") | **`ss.ppl`** rather than `ss.People` for the lazy reference. **Counters**: not `parity='+1'` ("the string expressions freak me out"); maybe `dict(to='parity')`, `move='parity'`, or `parity=1`, but Cliff wants to see more examples. **FP26**: the regression example is too dense, so keep code interpretable. **FP21**: print a warning when a bare `dt` takes the unit of `start`/`stop` |
| Destination results | 1, 2, 4 | **3**: keep `select(to=)` as a second spelling (b), not an error |
| Given dwell | 1, 4 | **2**: (b), so `P.dwell` is allowed in the rate of any exit out of `P` too. **3** (the conveyor) has no answer; taken as the recommendation unless Cliff says otherwise |
| Implementation history | D-D | "SOME compatibility aliases" are allowed, each with a very good rationale |

### Answers to Cliff's questions in the designs

- **"I thought the port was faster?"** (`disease_family.md:129`). Unchanged HPVsim on v4 is faster, 1.3 to 2.4 times (one genotype: 4.0 s against 9.6 s). The fully declared port is what is slower. With per-person immunity it takes 8.9 s against 8.7 s. With history states and 88 modifiers it takes 13.5 s, because each lazy condition rebuilds a module lookup on every leaf (HP30). The family design replaces the 88 modifiers with one `ss.immunity`, and HP30's fix (Stage 2, S) removes the per-leaf cost either way.
- **`rel_trans=10` with `beta=0.03`** (per-act D5). In the proposal, `rel_trans` multiplies the per-act hazard: `1 − (1 − 0.03)^10 = 0.263`, not 0.30. The two readings are close when `beta·rel` is small: at `beta=0.003` they give 0.0296 against 0.03. The hazard form never passes 1 (`rel_trans=40` gives 0.70 where the linear form gives 1.2), and factors compose (`rel 2` then `rel 5` is the same as `rel 10`). Multiplying the probability gives Cliff's 0.3, but then needs a clip at 1. This is a decision now: see "Waiting on Cliff".
- **`ss.MFNet`** stays.

## Stage 0: close what decision 16 settled, and fix the records (S; first, one agent)

- [x] Spec 12 (line 329 still calls item 18 "an open question") and the user guide: say that the methods differ by design, and give the sizes from the canon review (KR 6.3 at a daily step: agents peak 25% low and 9 days late, binomial 23% high) and from PB's set (contact and count SIRs, an SEIR, cholera, a capped rate, a fixed dwell). Give the W102 note a size. The natural home for this is D-T3's recorded table of solver differences.
- [x] Check that the agents engine has no leftover half-convention. A `dur=` dwell is drawn at entry; a `rate=` exit is `1 − exp(−r·dt)` each step; transmission entry is dated like progression entry. Close `test_entry_kinds` as not to be added.
- [x] Turn the canon xfails for CR7 (agents and network) into documented tolerances. Close ST4 the same way: it counts one more step in results, while people act for the right dwell.
- [x] `deferred.md`: move item 18 (B′, C) and ST4 to "Won't do", with the reason. Move the two-parameter NUTS coverage there (decision 14). Move N20's `away=` and R6's all-zero series to `implementation.md` as done.
- [x] Clear the stale items in `admin/implementation.md`: line 366 (IF2, `away=`) and line 371 (N25 to N27 "being built in").
- [x] Fix the CHANGELOG where it contradicts itself. Line 100 says ten runs a point for agent fits, but line 51 says one. Line 109 says `ss.SIR` keeps lognormal durations, against line 50. Lines 23 and 30 list the removed `'tau'`/`'sde'` solvers. Line 24 says `ss.load()` reads a spec back, which was removed.
- [x] FP4: a CHANGELOG line and a migration row saying that resampled stocks are now year-end values, not annual means.

## Stage 1: silent wrong answers (S each unless marked; with Stage 0, in two or three worktrees)

Fitting and results:
- [x] **Decision 13**: a fresh seed per point for global, Optuna-style fits of stochastic models. Local methods such as Nelder-Mead keep one seed for now (Cliff, 2026-10-19). Ideally they would run several times, each with its own seed; that is in `deferred.md`. Then rerun `test_epimodel_fit` (23.4% against the 30% target at one run a point with one seed).
- [x] **FP3**: `resample('year')` on a dated sim bins by calendar year, so a 3-year sim gives 4 "years" (`results.py:618`).
- [x] **FP12**: not a bug: `msim.results.<mod>.<res>.quantiles()` pools every run, as `reduce()` did, and a scenario of one run has no spread; the `reduce` error message now says so.
- [~] **FP2**: what `finish_step()` records never reaches `update_results()` (100 against 105 births a step). The end-of-run warning should catch this case too. (Left as documented: the migration skill has the row (`SKILL.md:79`); catching it generically would mean comparing every coded result before and after `finish_step()` on every step.)
- [x] **`away=` in `to_code()`**: it writes the fitted value, not the `ss.par`, so a round trip loses the bounds.

Timing, seeding, and branches:
- [x] **CR4 / L14** (M): a seeded case with a fixed one-step stay stays two steps under every solver, so Reed-Frost and Greenwood fail (the chance of no further case is 0.013 against the exact 0.131). This is seeding, not decision 16's bias, since the ODE shows it too.
- [x] **CR3 / L4** (M): disconnected risk groups written as plain states become counts (9,000 infected against 1,003, with only a note). Anchor on the transmission's source, and warn on a disconnected population. (Done as an error: it already raised, since the people's values sum to 0.2; the error now names the states given a share but not connected to the first state, and says to declare them `ss.State(people=True)`. Making any group seeded with a fraction into people was not done: mosquitoes given a fraction would become people, MA5's problem.)
- [x] **HP2**: an immediate junction holds agents one step (the ODE holds them none).
- [x] **FP15**: a remainder next to a time-series branch becomes a competing exit (0.153 where 0.30 is expected). (Already fixed: the review's probe, with its `init=dict(A=1.0)` removed (1.0 is now one person), gives B 0 then 0.19, C 0.30, D 0.70 then 0.51.)
- [x] **L14 remainder**: `dur=10` on a branch silently takes the 0.9 remainder, while `rate=0.1` raises. (Not silent and by the rule: one timed exit with no `p=` takes the rest, marked as guessed in `describe()`; a `rate=` exit competes with the branches. The error for the rate case now says so and how to write either.)
- [x] **The `after=` conveyor in the ODE** (S to M): its last slot is exempt from competing exits. It may have MA7's half step (5,279 against 5,020), so measure it first. (Measured: not affected. A 12-day minimum stay then a rate of 0.5 a day, with a competing death rate of 0.1 a day: the share reaching I is 0.2510 under the ODE at `dt` 1 and 0.5 against the exact 0.2510; binomial 0.2521, agents 0.2488.)
- [~] **X1 transfer**: a `dur=` distribution on a transfer is replaced by its mean (0.521 against about 0.99). (A note now says so at init. Drawing each person's time per the distribution needs a per-agent time of entering the source level and a scheduled move (M): Stage 4.)
- [x] **KI6** (M): check whether C17's rescaling covers a person who moves between levels under an `adjust=` on a duration.
- [x] **A strain's initial share**: `I=ss.by(strain=[0.01, 0.01])` seeds 0.5% in each strain, but with `risk=` the share is of each group. Check it against the spec. (Not a bug: as for any dimension, a value by level is the share of that level's people, and a strain's levels split everyone evenly by default, so `[0.01, 0.01]` is 500 and 500 of 100,000, as `risk=` is; the same under ode and abm.)
- [x] **NR5**: closed by Cliff's answer: `init_prev=` seeding `E` is right; remove the strict xfail `test_nr5_seir_init_prev_seeds_or_warns`. (Its strict xfail is in the untracked `naive_review/tests/`, left for Cliff to remove.)

Networks:
- [x] **HP8**: `ss.MFNet`'s `mean_degree` and `concurrency` results are scaled by `pop_scale` (0.551 becomes 110.1).
- [x] **HP9, HP10**: `MFNet` newcomers at any age, and people born during the run, never get debut or participation (557 immigrants and 2,671 people born in the run, all at 0).
- [ ] **`ssl.MSMNet`**: the same newcomer bug as HP9 (it sets debut and participation only for people younger than one step); give it MFNet's `set_network_states()`.
- [x] **HP11**: `MFNet` misses its degree target in an open population with long partnerships (0.664 against 0.87).
- [x] **N15**: `PartnerNet(match=)` on its own drops the risk structure with no message. (Done as a note: matching says nothing about how many partners each level has.)
- [x] **NR13**: a contact matrix that is not reciprocal is accepted without a note. (Done as a note at build, from the dimension's split; it is used as given.)

Demography and units:
- [x] **FP18**: `offspring=` doesn't turn aging on, and newborns get no parent.
- [~] **MA19**: `ss.Births` reads a rate as a probability, while `ss.birth` reads it as a rate. The import reference still says "Exact". (Measured: 20 years at 3.5% a year from 20,000 agents, 4 seeds: `ss.Births` 39,629 and `ss.birth` 39,870, against 39,796 compounded each year (0.6% apart, within about 2.5 seeds' spread); both draw r·dt·N a step. Left: the import reference's "Exact".)
- [ ] **MA21**: ODE aging at `1/width` drifts a non-flat pyramid (N −1.7% after 20 years).
- [ ] **MA16**: ODE waning uses a fixed 10 levels and is 6.2% off at day 729 (0.7% with 40). Choose the number of levels from the waning curve.
- [ ] **D2, L1** (with behavior step 2, whose decision 8 sets the unit of bare numbers): a bare-number time in `ss.timeseries` is in years with `start=2000` but in days with `start=0`. `unit='day', start=2000, stop=2002` gives 3 points (a note now; make it right).
- [x] **T6a**: `ss.seasonal(peak=142)` is accepted and taken mod 1. (A number outside 0 to 1 raises, naming `ss.days(142)`, which now works.)
- [x] **N22**: `auto` silently runs 10,000 agents for 80 million people; warn about scale. (A note when `auto` chooses agents and each agent stands for more than 1,000 people.)
- [x] **NR13 timevec**: with `unit='week'`, `timevec` runs to 365 in days. (Already fixed: in weeks for `stop=52`, `ss.weeks(52)`, `ss.years(1)`, or `ss.days(365)`; only float rounding (0.99999999) is left.)

Coded modules:
- [x] **D23** (S to M): `stratify=` on `fp.Sim` silently makes nothing (0 of 72 results have dimensions). Raise, or make it work. (A note: a stratified sim with no declared module has no results by group; the levels are still on people for coded modules.)
- [~] **R8** (S to M): `sim.hash()` doesn't see a `NodeGrid` held by a coded connector (EIR 10 and 100 hash the same). (Documented in `sim.hash()`: a coded module is seen through its parameters, so a value kept elsewhere should be given with `define_pars()`. Hashing arbitrary attributes would include arrays and random states.)
- [ ] **FP14** (M): a function that reads a parameter sees its base value, so `ss.change` does nothing to it. The behavior design fixes this for regressions written as expressions; a plain function needs `ss.change` to reach the value the function reads, or an error.

The approved design steps that fix silent answers:
- [ ] **Behavior steps 1 and 2** (S to M): FP19 (levels are values, not indices), FP20 (an existing property is kept, and `split=` for it raises), FP39 (the empty level, and counts after aging), D15 (`ss.by(sex=)` as agents; `CapabilityError` under an explicit compartmental solver), FP21 (bare `dt` takes the unit of `start`/`stop`, **with a warning**, per Cliff), FP22, FP35, FP30's `unit=`.

## Stage 2: crashes and misleading errors (S each; alongside Stage 1)

- [ ] Agents: **HPN2** (`_uids` crash with an `ss.by` branch `p=` at seeding, which blocks the HPV review's `v4hpv.py`), **HP16** (a split with no dwell), **HP21** (a remainder beside another timed exit), **ST19** (no branch to "stay").
- [ ] Modifiers and parameters: **HP27** (`ss.change` on a per-network `beta` dict), **HP17** (a coded array as a modifier amount), **HP34** (a modifier named like a `People` state), **HP35** (wrong example name), **FP32** (`ss.change` can't reach a parameter inside a modifier's amount), **N20** (array parameters can't be fitted; `ss.par(<dict>, bounds=)` is accepted as one scalar), **C04** (the class form can't use a module-level `ss.par`).
- [ ] Building and printing: **FP9** (`to_spec()`/`hash()`/`diff()` on an array duration), **FP36** (`Transition(A + B, ...)` crashes in its own error), **FP37** (`disease.solvers()` after `init()`), **MA25** (`ss.diff` on coded modules), **MA29** (`describe()` with `People(age_data=)` under the ODE), **FP27** (a row-valued `p=` crashes; refuse it clearly until behavior step 5).
- [ ] Conditions: **L24 / DF10b** (`where=Q.n < 20`, and a current flow inside a rate), **DC29** (Cliff approved an error pointing to `sim.all_networks()`, which is missing).
- [ ] Results: **MA13** (`.n` against `.count()` by solver, and N is 0 in `init_post()`), **MA31** (`by('patch')` on msim results returns an `objdict`), **R2** (a sweep labels its runs as scenario `''`).
- [ ] Coded code: **ST8** (a guard or a clear error when `update_results()` runs before any `step()`), **ST11** (a public `disease.move()`/`cure()` for coded code), **HP30** (cache the module lookup in lazy conditions, about 10 lines; it is most of the declared HPV port's slowdown), **FP8** (recheck `calibrate()` on `fp.Sim` after FP7's fix).
- [ ] **`sim.people` in a compartmental run** (X12): `people.n` as each group's live size from the compartments, read when asked, and `repr()` as `People(6 groups; ...)`.
- [ ] **R27**: remove `ss.Samples` (count downstream uses first), with a `deferred.md` entry to restore it if needed.
- [ ] **HP33**: named WPP rates (`ss.Births(birth_rate='nigeria')`) are in spec 04 but not in the code. Add a loader, or strike them from the spec.
- [ ] **FP29** (M): `ss.transfer` with `p=` and `dur=` over several destinations, as the spec says. **MA20** (M): death-rate ages tied to the age dimension.

## Stage 3: Phase H

### H0: the design round Cliff asked for (S to M; starts now, beside Stages 1 and 2)

One agent revises each document in place, keeping what is approved. Each document's revision ends with the decisions that are still open.

- [ ] **Care cascade, second draft of the user spelling.**
  - Add `ss.cascade()`, written so the logic reads at a glance, as `S >> I @ beta*I` does for diseases.
  - Make products and interventions impossible to confuse: no `ss.Test` beside `ss.test`.
  - Use full state names in the examples, and replace `default='normal'`.
  - Give resistance at drug failure, and a clinic visit, their own natural spellings, not `ss.Delivery`.
  - `on=` takes `|`, `&`, and lists with `operator='any'|'all'`.
  - Add `count_events=True`, and say what choices a user has above a stock target.
  - **Thanzi la Onse's health system** (its HSI events, appointment footprints, and consumables) should inform what a cascade can say. It is cloned at `/home/cliffk/idm/git/TLOmodel`, but don't touch it yet: Cliff is working on it with the lingua franca library first. Until then the draft goes ahead without it, and is revisited once his TLO work is done.
- [ ] **Family as `ss.Strains`.**
  - Name first.
  - `ss.Strains(hpvsim.genotypes(...))` builds the members, with no list expansion in user code.
  - Give `of=` a name that says what it is (it is the state whose holders are protected, such as `nab`); propose two or three.
  - Allow a function for `combine=`.
  - Say how `ss.Strains` relates to the existing `kind='strain'` dimension, which `ss.immunity` shares, so the two names don't collide.
- [ ] **Per-act transmission.**
  - D3: the valid direction keys come from the network (sex for `MFNet`, `p1`/`p2` or modes for `between=`), and a network can say which keys it accepts. Work through injecting drug use as an example.
  - D5: wait for Cliff's choice (below).
  - D6: condoms and microbicides as interventions acting through `ss.modify` on per-act transmission, not as a parameter of the disease or the network.
  - D9: specify the partnership approximation for the compartmental solvers (step 9), marked `approx`.
- [ ] **Behavior vocabulary.** Write the counter spelling with four or five worked examples (parity with twins, doses received, episodes, years of schooling) so Cliff can choose among `move='parity'`, `dict(to='parity')`, and `parity=1`. Rename `ss.People.<x>` to `ss.ppl.<x>`. Rewrite the FP26 example so each line is readable.
- [ ] **Given dwell.** Add decision 2 (b): `P.dwell` in the rate of any exit out of `P`.
- [ ] **Destination results.** Keep `select(to=)` as the second spelling.
- [ ] **No design owns "immediately".** The family (`exclusive=`), the cascade, the behavior vocabulary, ST26 (`perday(1e6)` takes 277 s a simulated year as an ODE), HP2, and HP16 all need a move with no dwell. Write it as a short section of whichever design lands first, most likely the cascade's.

### Code that can start now (in parallel)

- [ ] **Destination results** (about 2 days): steps 1 to 6 as written, with `select(to=)` kept. This is HP20 and DF13's destination half.
- [ ] **Given dwell** (about 4 days, plus rates): steps 1 to 4, with decision 2 (b). This is HP18; it covers HP28 only for states whose `p=` reads `P.dwell`.
- [ ] **Behavior steps 3 and 4** (M): properties and conditions in expressions (as `ss.ppl`), `expit`, and `ss.by` over properties with tables. They need no open decision.

### After H0

- [ ] **Per-act steps 1 to 4** (about 2 days; after D3 and D5): ST2, ST3, HP3, N7 (all **SWA**). Then steps 5 to 7 (condoms as in the revision, ST17, ST23), step 8 (the two-mode network, ST22), and step 9 (the partnership approximation for the compartmental solvers, now in scope; also DF16/X23).
- [ ] **Strains steps 0 to 4** (agents exact: what the HPV port needs): HP15's single-disease part, members, unions, `ss.immunity` (and KI15, where the compartmental solvers keep only the last strain, **SWA**), joint seeding, `exclusive=`. Then step 5 (side by side for the compartmental solvers, M to L; HP29), step 6 (with the cascade), and step 7.
- [ ] **Care cascade steps 1 to 3** (events, memory, peers replace nesting), then **4 to 8** in parallel (`delay=`, categorical tests and cure by state, values by disease, `target=`/`priority=`, coded diseases as `disease=`, stratified `prevalence`). These close ST12 to ST16, ST20, HP7, HP24 to HP26, and D23's delivery part.
- [ ] **Care cascade steps 9 and 10** (L): the delivery program in the Numba kernels, then JAX. This closes DF1, PL2, PL3, PL7, KI23, and C12c's delivery part, and lets cascades be fitted by gradients and NUTS.
- [ ] **Behavior steps 5 to 9**: `p=` with a `to` axis, counters (in the chosen spelling), clocks, the saved form, and the FPsim contraception port as the test. This closes FP24 to FP28 and FP38's regression part.

## Stage 4: other missing capabilities (M each; after the Phase H pieces they lean on)

- [ ] **X1 transfer, exact** (M): draw a transfer's `dur=` distribution per person on entering its source level, and move them when it is up, as dwells are; a compartmental form as a conveyor or stages.
- [ ] **ST24**: several non-exponential exits per state when the exits are disjoint (by sex), and `ss.lognorm_ex(mean=ss.by(...))`.
- [ ] **ST18**: an exact dwell until an external event (`after=preg.delivery`), and a warning that `dur=fn` is exponential.
- [ ] **HP28**: hyperexponential dwells (CV above 1) under the compartmental solvers, beyond what given dwell covers. With D-A2 (conveyors for lognormal or Weibull dwells), and Cliff's unanswered question "would it add a lot of code?" (DF35), which this item should answer.
- [ ] **HP14**: a strain dimension with `immunity=` over a network as agents.
- [ ] **HP19**: a general per-person write on events (a running maximum, frozen at entry), agents only where needed (Cliff: "some things will be agent only and that's fine").
- [ ] **Vector models**: standard `prevalence`/`new_infections` from the acquisitions, not only from a transmission (about 8 sites in `compartmental.py`). MA11 (weighted counts), MA12 (per-node EIR from coded modules under both solvers), MA14 (a product's coverage in a rate), MA33 (a neutral held product, and several wanings).
- [ ] **MA15** (moved from Stage 1; needs a spelling, e.g. `redose=`, with the products work): a product whose waning is a function is held for good, so a second round reaches nobody who had the first (600 against 3,237).
- [ ] **Demography**: FP33 (life tables by year and sex; `ss.Deaths(lifetable=)`), N11/SW3 (births from one stratum into another), SW2 (`offspring=` by prototype).
- [ ] **Mixing**: rebuild the mixing inside `restart()`, so PMCMC, IF2, `run(n_runs=)`, and gradients work with a fitted `away=` (they raise today). CR14: commuting with finite leave and return rates (`ss.Mobility(kind='commute', back=...)`). PL4: `PartnerNet` with several `match=` properties.
- [ ] **NR2**: `msim.outbreaks(threshold=)`, about 15 lines, or close the xfail.

## Stage 5: performance and scale (M to L)

- [ ] **MA28 and MA7's agents part**: counts through the compartmental kernels under `abm`, so a fixed EIP is exact as agents (7,576 against 5,020 today) and counts stop costing Python time.
- [ ] **N29**: the lean path for small declared models (0.95 to 1.2 times the time of Starsim 3 code). **N25 / SW7**: agent speed at LASER's scale (3.3 million agents in 7 to 8.5 s against 2.75 s). **D-P4**: the rest of an agent step.
- [ ] Re-measure, then fix where still slow: **FP34** (declared deaths by age and year, 30× slower than `fp.Deaths` before MA17), **ST27** (`calibrate()` at 4.5 s an evaluation on a 0.6 s ODE), **HP28**'s 480-group SEIS.
- [ ] **D-P5**: the particle filter on outcomes, now unblocked by N27. **MA26**: a deterministic coded ODE fitted with a deterministic optimizer.
- [ ] **SC6 residual** (exact Gillespie with every place reaching every other, n^1.5), **W16** (crossings whose places mix within a step), **DF7** (thinning for time-varying rates under Gillespie), **DF2**, **DF3**, **D-P6**, **W17** (`forkserver` everywhere to silence the fork warning).
- [x] **CR9**: Gillespie's automatic tau leaps are biased at 10 million people (peak +0.74%, about 4 sd). Cliff: whatever is fastest on average, which is today's rule; kept, with the note.
- [ ] Internals: **X27 / PL12** (the four skipped compartmental-engine refactors), **PL11** (results creation out of `solvers/`).

## Stage 6: inference, the saved form, and random streams (S to M)

- [ ] **X26** (`sim.gradient()` as `jacfwd`), **DF15** (fitting a product of parameters), **DF26** (IF2 for observation and initial-value parameters), **R16** (date columns and data outside the run), **KI21** (fittable `rel_sus`/`rel_trans` on a state), **N20 remainder** (`capacity=` as an expression, `ss.par` inside `ss.seasonal`, data as averages).
- [ ] **DF22 / R29** (priors and likelihoods in the spec, and the grammar rewrite), **DF30** (operand order for hashes; it changes every hash once, so do it with R29), **R23** (a stratification exported as species), **DF38** (exact delay exports), **PL10** (fold `ss.diff_sims` into `ss.diff`).
- [ ] **C12c remainder** (outcomes drawn after a whole run differ from a stepped run; the delivery part goes with the cascade's step 9), **X19** (`sim.crn_report()`).
- [ ] **MA30** (M; moved from Stage 1, since it is not silent: the totals are right and `.by()` raises): results that are an affine function of a ratio (`98.6 + 4*I/(S + I)`), or `I.frac`, are not kept by group; `ss.by(patch=...)*I/(S + I)` as a result crashes (agents: `_uids`; ODE: a broadcast error). Keep a ratio's parts by group and apply the outer arithmetic after selecting.
- [ ] Results and printing: **KI14** (arithmetic on stratified results), **DF14 / R17** (a sim-level outcome across modules), **DF12** (the size of a stratified model, and equations by group), **FP38**'s rest and **D-F4** (`describe(verbose=True)`), **X10** (say at run time when a run leaves the whole-run kernel).

## Stage 7: importers and interop (lowest priority; L in all)

- [ ] **DF10f** (moved from Stage 1): an initial number below 1 is a fraction by the spec's rule, so the camdl importer should write its counts below one person as counts (or the spec needs a way to say so).
- [ ] **SW1 / N13** (Atomica TB: 81 parts untranslated), **SW10 / N12** (EMULSION: 13 parts), **PL5**, **PL9**, **R34** (duplicated interop internals), **DF10a** (a declared network condition), **DF10d** (ERGM terms), **DF10h/i/j** (MEmilio's damping, weighted counts, and per-state group parameters), **DF10e** (EpiModel's within-step order), **DF36**, **DF37**, **SW8** (a seed below one agent).
- [ ] **R22 / PL14**: the importers as a companion package in this repo (approved), with the `dump_*` scripts moved out of the core. **SW11 / PL8** (US states and German level 2 in `starsim_data`), and **P14** (`starsim_data` on PyPI, then a dependency).
- [ ] **DF8** (triangles as an ERGM target) and **DF9** (WorldMove, licensing) wait for a real need.

## Stage 8: reviews, tests, and records (any time; some gate the end)

- [ ] **Thanzi la Onse**: a review for the cascade, after Cliff's own work on TLO with the lingua franca library (`/home/cliffk/idm/git/TLOmodel`; leave it alone until then).
- [ ] **Canon**: the 38 short canon models into a docs gallery and `tests/corpus.py`. Move `canon_review/tests/` into `tests/` as each item is fixed. Add a docs line that the book's k-regular network is `ss.StaticNet` over networkx, not `ss.RandomNet`.
- [ ] **Naive review**: move the remaining strict xfails into `tests/` once fixed. Rerun the benchmark after the docs are rewritten, with the harder tasks listed. Fix `check_notebooks.py` so it can run `coming_from/` (it runs `eval: false` cells and writes `numpy.py` into the folder).
- [ ] **Real models at full complexity** ([next-reviews.md](../oct17/next-reviews.md) item 5): after Phase H, since it would mostly rediscover the gaps Phase H fills.
- [ ] **D-T1** (a test for each named error), **D-T2** (golden `describe()` per corpus model, after `describe()` settles), **D-T3** (the recorded table of solver differences, from Stage 0).
- [ ] **P7, D45, X30**: check the spec, the departures, and the code item by item, at the end.
- [ ] **The downstream CI gate** (`.github/workflows/test_downstream_v4.yml`) is written but not pushed. Pushing it needs Cliff. STIsim and HPVsim still fail 13 and 12 of their migrated tests, the same as before this round. Look at those failures once the per-act and cascade steps land.
- [ ] Docs (tutorials, user guide, migration guide, release changelog) wait until the implementation is final, as before.

## Waiting on Cliff

Answered on 2026-10-19 (his `[CK:]` notes, folded in):
- **Per-act D5**: the hazard (`rel_trans` multiplies the per-act hazard). H0 and per-act step 2 use it.
- **`sim.people` in a compartmental run**: the recommendation in [people_compartmental.md](../oct17/people_compartmental.md), B plus two changes: `people.n` gives each group's live size from the compartments when read, and `repr()` says `People(6 groups; ...)`. Now a Stage 2 item.
- **Given dwell decision 3**: the whole-step conveyor.
- **NR5**: Cliff asked what is wrong with `init_prev=` seeding `E`. Nothing is wrong in the model: `E` is infected, so seeding it is the literal reading of "the share infected". The naive user expected infectious people (`I`), so the first SEIR peaked 4 days later than they expected (83 against 79), and nothing told them which state was seeded unless `verbose=1`. With the note kept and the docs showing `init=dict(I=...)`, this is closed: no warning, and the strict xfail goes.
- **CR9**: whatever is fastest on average, which is today's rule, so it stays as is, with the note.
- **R27**: drop `ss.Samples` for now (restore it later if needed): Stage 2, with a `deferred.md` entry. `ss.Dataset` was not mentioned, so it stays until asked.
- **The downstream CI gate**: waits.

Still open:
1. **The counter spelling**, after H0's examples.
2. **NR13**: no progress bar for very short runs, and naming a clashing product after its delivery (this changes result keys).
3. Older ones: **L20 / L23** (names without frames; `<transition>.<slot>` names), **DF20** (contact matrices as release assets), **SW8** (round a sub-agent seed stochastically, or raise).

## Acceptance

- No silent wrong answer left open from Stage 1, and the per-act and strains steps that fix ST2, ST3, HP3, and KI15 are merged.
- The six designs are implemented as revised, each with its contract test across solvers. HPVsim, STIsim, and FPsim delete what each design says they would.
- The four downstream packages, migrated, pass their suites in CI on `v4`.
- The full suite, the switching demos (41 of 51 or better), and the corpus hashes are unchanged except where a step changes results on purpose, which its CHANGELOG line says.
- At the end, the seven status checks behind [status_oct18/](../oct17/status_oct18/) are re-run against the tree.
