# The October 21 downstream reviews: synthesis and recommendations

Written 2026-10-21. A synthesis of the six reviews in this folder: [tbsim_recheck.md](tbsim_recheck.md) [CK: why no TB, there were lots of findings in that report], [malaria_recheck.md](malaria_recheck.md) (MA), [hpvsim_recheck.md](hpvsim_recheck.md) (HP), [fpsim_recheck.md](fpsim_recheck.md) (FP), [covasim_review.md](covasim_review.md) (CV) and [stisim_port.md](stisim_port.md) (ST). All were run at `ee5d1fb0` (STIsim's also at `88208f57`). I checked that the headline findings still hold at HEAD (`88208f57`, with `starsim/` unchanged since then): `use_aging` is still overwritten (`dimensions.py:719`), `init=` still reads 1.0 as one person (`solvers/engine.py:1049, 1066`), `sim.current_scale` is still `_current_scale`, and `ss.Calibration` and `ss.SexualNetwork` are still gone. Nothing in this file was rerun; the numbers are the reviews' own.

The reviews' IDs are kept, so each point can be traced to its evidence. The [constitution](../constitution.md) is the yardstick.

## In brief

**The core of v4 holds up.** Every downstream model that was run reproduces Starsim 3 within noise once it gets past import: Covasim's default (21,131 ± 67 infections against 21,108 ± 50), `hpv.demo()` (cancers 0.994 of v3), STIsim's five diseases and Zimbabwe (once ST28 is worked around), and malariasim (within 0.7% in all 9 combinations). v4 is also faster: Covasim runs in 0.69 of the time, STIsim in 1/1.5 to 1/1.7, HPVsim in 1/1.3, and FPsim in about 0.75 to 0.92. The declared versions are much smaller: Covasim's natural history is about 60 lines against about 1,000; HPVsim's genotype machinery is 20 lines against 402, at HPVsim's speed; FPsim's pregnancy and contraception have no per-step functions left. Most of the October findings are fixed (MA 13 of 33 fixed, 5 partly; HP 24 of 36 fixed, 9 partly; FP 22 of 39 fixed, 11 partly; ST 19 of 27 fixed, 3 closed by decision).

**But nothing downstream imports unchanged [CK: this is fine, we wouldn't expect it to], and the reviews found about 45 new silent wrong answers.** Many are the same few causes seen from different models. The main ones:

1. **Things that are accepted and then do nothing.** This is the most common kind of silent wrong answer (theme 1).
2. **`init=` reads 1.0 as one person.** Found independently by four reviews (MA34, HP48, FP69, ST31).
3. **The index conventions for results and times in modules written as code.** Six findings across four reviews, and the end-of-run warning that caught them has been removed (theme 3).
4. **No public base class for "any disease".** Four reviews hit it (theme 4). [CK: this is very bad!]
5. **`ss.by.lookup()` rebuilds its table on every call.** This makes declared models 3 to 107 times slower (FP57, CV11). [CK: definitely should be fixed]
6. **Compartmental forms removed in S3.** One of these now gives a silent wrong answer under `solver='auto'` (FP53).

Most fixes are a few lines. Six questions need Cliff's decision (see [Decisions](#decisions-for-cliff)).

**What this does to the plan.** The plan says that a silent wrong answer goes into the plan, not deferred.md. Taken literally, about 45 small items are added, and the 91% estimate falls to roughly 70–75% (see [The plan](#what-this-does-to-the-plan)). [CK: we're going to make a new plan]

## What to keep

The reviews agree on these, and nothing below should undo them:
- **One transmission path for declared diseases and diseases written as code (S5.5).** Covasim's variants, NAbs and severity reproduce v3 exactly through it.
- **`new()` rebuilding a subclass from its call.** Every run tool now works on `fp.Sim`, `hpv.Sim`, `sti.Sim` and `cv.Sim` with no change to those packages.
- **The behavior vocabulary.** This covers counters, clocks, `expit` over `ss.ppl`, `p=` tables with a `to` axis, `dur=0`, branches decided at exit from the time spent, and `ss.attr` on events.
- **The delivery vocabulary.** This covers outcomes as events, `.since`/`.ever`, `delay=`, `capacity=`, `queue=`, `trace=`, `doses=` and `ss.Diagnostic(results=...)`. Covasim's cascade ran on the first try, and HPVsim's `treat_num`/`treat_delay` are one keyword each.
- **`ss.Strains` with `ss.immunity(combine='sum')`.** It replaced 402 lines of HPVsim code and gets the combination rule right.
- **Refusals that name the fix.** These include the class-creation refusal of coded hooks on declared classes, `TimeOutsideSim` at `stop`, the NaN check on modifiers (which found a real tbsim bug), and the warning for a coded `set_prognoses()` that skips `super()` (which found real regressions in malariasim and HPVsim's HIV).
- **The declared ODE's speed with places, ages and series.** It is 2.3 times faster than on October 17 (MA27).

## Cross-cutting themes

Each theme lists its evidence across the reviews, then a recommendation. Sizes are S (under a day of agent work), M (a few days) and L (more).

### 1. Accepted, then does nothing

This is the largest class of silent wrong answer:
- `ss.change`/`ss.modify` on a `p=` table (FP52);
- a modifier on a declared module's plain parameter (FP54);
- `ss.change` on an inflow under the ODE (MA36);
- products and `ss.modify` on a disease written as code that overrides `infect()` (CV1);
- `move=` on a transition triggered by `on=drug.failed` (ST30);
- a bare key on a multi-destination exit (HP6);
- overrides of removed `ss.Pregnancy` methods (FP41);
- strain `immunity=` beside an explicit reinfection (HP46);
- `stratify=` on a coded module (D23);
- a bounded parameter dropped by `Pars.update()` (ST37).

Fixing each one separately is a few lines, but the pattern will keep coming back.

**Recommendation:** give every modifier, change and product a single check at `init()` that it resolved to something that reads it, and raise if not (*"`ss.change` on `contra.switch`'s `p=` table is not applied by any solver"*). Then fix the individual cases on top of that check: either implement them (CV1 in `infect_route()`, ST30, MA36) or refuse them (FP52 until it is implemented, FP54, HP6). This is the constitution's "think big picture" principle applied to correctness. Size: S for the check, plus S for each case. [CK: this sounds good]

### 2. `init=` reads 1.0 as one person

Four reviews found this independently: MA34, HP48, FP69 and ST31. For the declared states, `dict(S=1.0)` means one person, while `init_prev=1.0` means everyone. Every October cohort script used `1.0`. ST31 is the worst case: BV's `ss.by(stable=[1.0, 0, 0])` silently puts women in the wrong state.

**Recommendation:** use `init_prev`'s rule everywhere (above 1 is a count, up to 1 is a share). Change the note to say "the other 9,999 people", and accept functions in `init=` as `init_prev=` does (HP48). A few lines, plus a spec line. [CK: ok]

**Tradeoff:** this gives "exactly one person" no spelling as a float (`1` vs `1.0`). Write it as `init=dict(I=1)` with an int, or as `ss.count(1)` if a typed spelling is wanted. I recommend the int, since it reads naturally.  [CK: that's ok, exactly one person is an int, not a float. but does I=1 mean 1 person or 100%? i think users will typically mean 100% even if an int is given, as in init=[1,0,0]. so i think it should be a fraction (but it should print a warning telling them to use a float instead). but then how can users express 'exactly one person'? i don't mind `ss.count()` if it already exists, but if this is the only use case for it, maybe we could ask them to use np.int64(1)? but that seems super clunky. not sure what the best solution is...]

### 3. Index and time conventions for modules written as code

This is the most widespread trap in coded code:
- **CV2:** `self.ti` in `update_results()` is the end of the step, so a property shared with `step()` is off by one. Covasim's `cum_quarantined` is always 0, and `n_infectious` is about 10% high.
- **FP1/FP42:** the end-of-run warning for results written at v3's indices was removed in `7c0c8c96`, while the skill still promises it. The migration script misses writes inside helpers.
- **ST35:** a delivery's results and a transition's flow land one index apart.
- **ST39:** a stored `ti` makes every partnership duration one step long.
- **ST8:** interventions' `update_results()` runs before the first step. It is marked done, but it isn't.
- **CV6:** `run(until=)` takes one step fewer.
- **MA41:** nothing writes the initial point.

**Recommendation:** keep v4's convention (N+1 points, with index 0 the start), since it is right for every solver. But take the index arithmetic out of user code. Give modules one way to write "this step's value" (for example `self.results.x.now = v` or `self.record(x=v)`) and one name for the step just taken (`self.t.ti_step`, which X13 also asked for). Have the script rewrite `results.x[self.ti] = v` to that form. Restore the end-of-run warning, and use one index convention for deliveries (ST35). S to M.

**Decision:** whether to add the accessor (a new name) or to rely on documentation and the warning (no new API). I recommend the accessor: the same off-by-one has now been found in Covasim, FPsim, STIsim and malariasim, which is more than documentation will fix. [CK: i don't love this ...`x[self.ti]` should work in most cases, and if it doesn't, then something has gone wrong with how the loop is being run. it's a very obvious thing 'index this array, at the current point in time'. i don't mind having a .now as well to support corner cases, but nothing could be simpler than 'we're in a loop, here's the value at the current loop index', so if that's broken, it seems that something more systematic has gone awry (see the constitution).]

### 4. The disease base class

Four reviews hit this:
- **MA22/MA41:** there is no bare coded base. The script rewrites `ss.Disease` to `ss.Infection`, which defines `susceptible`/`infected` and clashes with malariasim's own (72 test failures).
- **ST41:** `isinstance(x, ss.Disease)` is false for `ss.Infection` [CK: this is a problem! let's fix that], so `sti.Sim` refuses every coded disease (66 failures). STIsim's `SimpleBV` hits the same clash of states.
- **HP53:** HPVsim fails at import through STIsim's `class SimpleBV(ss.Disease)`.
- **tbsim §4.6 and FP71:** malnutrition and contraception have to be "infections", and `describe()` calls a behavior model a "DiseaseBase" whose states are "infected ... cured by `ss.treat`".

Today `DiseaseBase` (unexported) is the common base, `ss.Disease` is the declared disease, and `ss.Infection` is the coded one with transmission. [CK: whoa, this is an architectural mess. the whole point is that declared and coded parts should be able to live peacefully together, not be quarantined into separate classes. `ss.Disease` should be the common base as it currently is, supporting both declared and coded parts. `ss.Infection` should be, well, an infection, that contains additional machinery, as it currently does.]

**Decision (options):**
- **(a)** Export one public name for "any disease". `isinstance(x, ss.DiseaseBase)` would work for both, and a bare coded class would subclass it. This is the smallest change (about 15 lines, MA41's estimate), but it adds a third name users must learn, and "DiseaseBase" will show up in their code.
- **(b)** Make `ss.Disease` the base of both. A subclass that declares no states and overrides `step()` is a coded disease with no transmission, and `ss.Infection(ss.Disease)` adds the network machinery. `isinstance(x, ss.Disease)` then means what users expect, v3's `class X(ss.Disease)` keeps working, and malnutrition is no longer an infection. The cost is that `ss.Disease` decides at class creation whether it is declared or coded. The existing `__init_subclass__` and the "declares no states" check already do most of this. M.
- **(c)** Keep the classes as they are, and have the script and the error point at `ss.Infection(states=False)`. This is the least work, but it keeps both name problems.

**I recommend (b).** It is the fewest concepts for modelers (constitution, primary 1 and secondary 1), and it removes a needless incompatibility (development principle 4) rather than adding a name. [CK: yes, i like B, except i'm not sure about 'deciding whether it's declared or coded'. what about a mix of both?]

### 5. Speed of `ss.by` lookups and other hot spots

- **FP57 and CV11:** `ss.by.lookup()` rebuilds its table from pandas rows on every call [CK: ohmygod no. do NOT use pandas in any hot path!]. This makes the declared deaths 107 times slower than `fp.Deaths`, the declared contraception 9.4 times slower than FPsim's code, and Covasim's declared prognoses 3 times slower. Caching `keys()` alone (6 lines) brings the pregnancy level with `FPmod`. Building the grid once and indexing it each step (about 15 lines) should fix the rest. **Do this first: it decides whether the declared forms are usable at all.** S. [CK: agree]
- **MA39/MA28:** counts are stepped in Python as agents (41% of the run), and `series()` is re-tabulated at every derivative call. `Arr.__setitem__`'s copy-on-write check slows every coded model about 10–15%. M. [CK: ok]
- **FP58:** declared rate exits under a changing modifier are redrawn (2.1 times slower). This is a measured cost of the C17 correctness fix, so it goes in deferred.md. [CK: ok]
- **FP59:** `ss.years(array)` scans every element in Python. 1 line. [CK: ok]
- **The benchmark set (S0.3) has no counts, no coded model and no `ss.by` table**, which is why none of this was caught. Add malariasim's `fseis10`, a declared model with counts, and a model with an `ss.by` age table before S8.1 measures against it. [CK: ok]

### 6. The solvers disagree, or don't conserve

- **FP51:** the ODE loses 9–11% of people when two modules are crossed and one has a fixed dwell.
- **FP53:** a Weibull dwell with shape below 1 becomes one exponential stage, and `solver='auto'` picks the ODE (mCPR 0.09% against 0.37% as agents).
- **ST32:** `ss.constant(0)` waits a step under the compartmental solvers.
- **MA36:** see theme 1.
- **MA37:** `R0()` is off by √2 on the spec's own Ross-Macdonald model.
- **MA6:** chain-binomial mosquitoes still go extinct at a 30-day step.

**Recommendation:** two small contract tests catch this whole class:

- conservation of people in every crossing;
- agents against ODE on the spec's own models, plus the analytic `R0()`.

**FP53 is the one that `auto` causes**, which makes it worse than an approximation the user chose. Have `auto` run agents when a dwell's coefficient of variation is above about 3, and warn under an explicit ODE. S each. [CK: ok]

### 7. Compartmental forms removed in S3

The S3 cuts cost these downstream models:
- **MA38:** box-then-decay waning (malariasim's vaccine, bednets and prophylaxis) now has no ODE form. malariasim runs every intervention under both solvers today, so this is a loss of something it already does. [CK: we can't anticipate every possible downstream form, but what should be possible is for things like box-then-decay to be written in *malariasim*, and then used in starsim. is this not possible? if not why not? seems like if i write something in a reasonable-ish ODE-ish format, it should be possible to pick it up in the ODE solvers.]
- **HP18:** an outcome that depends on the drawn dwell is agents only, so there is no declared HPV with a fittable `cin_fn` as an ODE. [CK: again, would love for this to be expressible in HPVsim code]
- **HP28/FP53:** dwells with a CV above 1 become one Erlang stage.

**Decision:** for MA38, restore only the box-then-exponential shape. It is a conveyor plus an exit rate, both already in the kernels (about 40 lines). It keeps a real downstream feature, and it stays inside the support matrix's style ("explicitly supported forms"). Leave HP18 and HP28 agents only, as S3 decided, with FP53's guard.

**Tradeoff:** MA38's 40 lines against the "firmer boundary". I recommend restoring it, since the shape is common (every product that protects fully and then wanes) and adds no new mechanism. [CK: we can do that if we must, but i would prefer for there to be an easy way to write it, and the myriad of related-but-not-implemented forms (e.g. dcay at one rate, then another), to be implementable outside starsim. this is part of the 'flexibility' requirement of the constitution: we can't hard-code every conceivable immunity waning form! if the user can write it to look like an ODE, we should be able to consume it as an ODE.]

### 8. Births, newborns and placement

These are mostly from FPsim, plus HPVsim and STIsim, and all are silent:
- `embryos_per_pregnancy=1` makes no children (FP43);
- `ss.birth(per=)` makes no children from a transition to `ss.null` (FP44), and loses about 2.5% when the parent's transitions have dwells (FP46, cause unknown);
- `ss.mother(ss.ppl.age)` reads 0 for newborns (FP45);
- newborns wait a step before a `dur=0` split (FP47);
- `ss.histogram` draws uniform(0, 1) in the engine (FP48);
- traits are not drawn for people born to a coded module (FP49);
- `move_to()` in `init_post()` backfills a false history (FP50);
- placement doesn't fire enter events (HP37);
- a neonatal death `on=preg.delivery` counts as a stillbirth (ST34).

**Recommendation:** fix these as one batch, each with a regression test, plus one contract test that the engine's draws match `dist.rvs()` (FP48). All are S except FP46 (cause not found) and FP50 (which needs a choice: don't backfill people placed by `move_to()`; I recommend that). [CK: ok, although can probably skip the regression tests; the test suite should not become a creaky history of every random bug that happened to be encountered.]

### 9. What stops downstream packages importing

This is S8.2 in the plan, which waits for the design. But some of it reverses earlier decisions, and those are design questions to settle now.

**`ss.Calibration` (CV7, ST40, HP53).** It blocks `import` of Covasim, STIsim and HPVsim. It was kept for 4.0 by `oct17/decisions_oct18.md` item 3 and then removed in S2.1 without recording the reversal. `sim.calibrate()` can now replace each package's subclass (CV10: Covasim's becomes about 10 lines once two gaps are closed).

**Decision:** I recommend no compatibility class (no baggage). Instead:
- record the reversal;
- have the migration script comment out the subclass and point at the mapping;
- close the gaps that make the replacement real (theme 10). [CK: yes, this all sounds good]

**`sim.current_scale` (CV7).** It was promised as an alias (`admin/implementation.md:204`), is still documented in the user guide, and v4 itself reads `_current_scale`. It is the first error in 55 of Covasim's 118 failing tests. This isn't baggage: it is a live concept that was renamed by accident. Restore it as a public property. 3 lines. [CK: ok]

**Starsim 3's coded `ss.Pregnancy` (FP40, ST42).** FPsim's `FPmod` overrides 9 methods that are gone, and STIsim reads `pregnant`, `ti_delivery` and `tri1_uids`. The migration guide says coded modules "run unchanged" and never mentions pregnancy.

**Decision:** I agree with FPsim's review: treat the rewrite onto the declared `ss.Pregnancy` as the port, and don't vendor 850 lines of v3 pregnancy into `ssl`. The seams exist (`conception_rate()`, `ss.modify` of its transitions, `since_birth`, callbacks, `ss.Lactation`). Add FP41's `__init_subclass__` refusal, ST42's script renames, and a guide row. Infant mortality has no slot yet, so it needs a declared newborn transition. [CK: ok]

**`verbose=0.1` (CV9).** This is Covasim's default, and it makes `cv.Sim().run()` fail out of the box. Read a fraction as "progress every 1/verbose of the run" instead of refusing it. It is a reasonable meaning, and refusing it costs 24 tests. About 5 lines. [CK: this is starsim's default too -- so how]

**Script rules.** These are S8.2 work, but they are cheap and found three times: [CK: ok]

- rewrite keywords only in calls to Starsim's deliveries, never in a `def` (tbsim, MA41);
- don't delete `on_positive=`/`on_negative=` (tbsim);
- rename `ss.SexualNetwork` (HP53, ST53) and `copy_inputs=` (HP53).

### 10. Calibration

Fitting is where downstream packages lean on v4 hardest, and the reviews found these problems:
- three silent wrong answers: annual data compared a year off (ST36), bounds lost through `Pars.update()` (ST37), and "Converged" reported at the start (ST38);
- two crashes: a data column that matches no result crashes even when `gof=` ignores it (ST44, FP8, CV10), and `fit.compare()` on a module with its own timeline (ST51);
- three gaps: no worker processes or `storage=` resume (HP54, ST40, CV10), no fitting of a coded intervention's settings (CV10), and no mapping from a data column to a family's members (HP54);
- one cost: three searches for a one-parameter deterministic fit (MA39).

**Recommendation:**
- Fix the silent and crash items. All are S.
- Add `n_workers=`/`storage=`, which three reviews ask for (M).
- Read an integer `year` column as a calendar period. [CK: all ok]

### 11. Startup and compiling

- **ST50:** `compile='auto'` ignores substeps, so stiff STI ODEs run as plain Python by default (syphilis 266 s against 1.72 s compiled).
- **MA39:** the first-use compile is silent at `verbose=0`, so a fit sits silent for 65–80 s.

The constitution ranks performance above startup time and says compiling is opt-in. These two make the opt-in invisible, which defeats it.

**Recommendation:** include substeps in the estimate (they are known at `init()`), and always print one line when a run will compile, or when a plain run passes about 20 s ("this would take about 2 s compiled; `ss.compile()` once"). S. [CK: if compiling would take less time than the run, do it automatically with a note; if it would take more time, print a note estimating the pre/post compiled times]

### 12. Strains (S7.4)

The evidence for S7.4 converges:
- **HPVsim needs the family; tbsim needs co-carriage in a dimension.** Both support [strains.md](strains.md)'s option A, and both confirm A.1: infer `kind='strain'` from `'+'` levels. Without it, tbsim's resistance is silently 100% off its reference ODE (tbsim §1, HP46).
- **The HPV port adds four things to the family:** seeding through each person's own transmission (HP41, silent), a `cap=` for a summed severity rule (HP56), a trait shared across members (HP43), and a raise for HP46.
- **Covasim's variants fit a strain dimension**, but need cross-immunity over networks (HP14), protection weighted by a waning attribute, and protection against branches (CV12).
- **ST30**, `move=` on `on=drug.failed`, is how the plan spells resistance at treatment failure, and it does nothing.

**Recommendation:** choose option A, with A.1, HP41, HP46's raise and ST30 now. Add HP56 and HP43 if they stay small. Put Covasim's dimension needs (HP14, CV12, CV13's time since an event) in deferred.md until a Covasim port is actually planned. This closes S7.4 at about M. [CK: see strain_plan.md for more details on decisions]

### 13. Crashes in the expression and results algebra

These come from several reviews:
- `1e5*cancers/N` (HP49, the first HPV output anyone writes);
- a member's flow over the family's flow (HP50);
- `ss.ppl` expressions in `multiply=` (HP51);
- `ss.by + ss.by` and `np.minimum(ss.by, 1)` (FP60);
- a modifier amount over another module's states (FP61);
- `ss.attr(..., on=X.exit)` (FP62);
- dwell expressions over `ss.ppl` once people are born (FP63);
- a factor times `ss.peract` (HP47);
- `ss.by` over `ss.contacts()` as agents (ST47);
- per-route weights (ST49);
- `ss.peract` in a dict under the ODE (ST46).

**Recommendation:** make `ss.by` an expression node (FP60), and add one small table-driven test: each node type × each operator × each solver, run on a tiny model. That is a few dozen cases and fast, not a fuzz test. It should catch most of these, and the next ones. S to M. [CK: ok]

### 14. Networks after decision 18

- **HP45:** `ss.SexualNet` pairs on behalf of subclasses that pair their own. This is rule 2 of [sexual_network.md](sexual_network.md), which wasn't implemented. 3 lines and a test. [CK: ok]
- **ST45:** MSM subclasses inherit the man–woman `sides`. `sides='mm'` is now the fix; the error should suggest it. [CK: ok]
- **ST22 and ST52:** `between=` (sex workers and clients) and risk matching are gone, which is consistent with decision 18 ("for more complex things people can go to STIsim"). [CK: not sure i understand the difference between 'sides' and 'between'. i actually like 'between' better. so would this fix both cases?]
- **`test_parameter_docs_coverage`:** subclasses now carry base parameters they don't use, a known cost of one configurable base. [CK: ok]

**Recommendation:** HP45 and ST45 now, and nothing more. [CK: no, do all four]

### 15. Units and time on step boundaries

**Unit traps:**
- a rate function's bare numbers are read in the inferred unit (ST33);
- a factor times a per-act beta becomes a bare beta with a unit (HP47);
- the S5.9 message suggests `.set(20.27, 4.27)`, which reads months as years (FP68);
- one `ss.perday` rate makes a 10-year sim step daily (tbsim §5);
- `dt=ss.months(1)` without `unit=` (L1, still open).

**Step-boundary conventions:**
- a declared infection is dated at the end of its step, so Covasim's peak moves 11 days (CV5);
- the extra step on entry costs trichomoniasis 14% of its infections (ST4, decided);
- a dwell under half a step ends in its first step (FP55).

**Recommendation:**
- Refuse bare numbers returned by a rate function when the model has typed values, rather than guess a unit. It is one rule, and simpler than inference. [CK: ok]
- Give a scheduled dwell a minimum of one step (FP55). [CK: hmm ... not sure i understand why. seems like zero-time dwells are reasonable?]
- Add a short "timing contract" section to spec 02, with one `describe()` line each for CV5 and ST4, so the conventions are visible where modelers look. [CK: ok]

## Decisions for Cliff

In order of how much depends on them:

1. **The disease base class (theme 4).** (a) export a "DiseaseBase"; (b) `ss.Disease` as the base of both, with `ss.Infection` beneath it; or (c) keep and point. **Recommend (b).** [CK: B, with caveats, see above]
2. **Results in modules written as code (theme 3).** Add a "this step's value" accessor and `ti_step`, or rely on documentation and the restored warning. **Recommend the accessor**, plus the warning. [CK: i think indexing by self.ti should work, and am worried that it doesn't]
3. **Starsim 3's coded pregnancy (theme 9).** Rewrite onto the declared one, or keep a coded base in `ssl`. **Recommend the rewrite** (FPsim's review agrees). [CK: ok]
4. **`ss.Calibration` (theme 9).** Record the reversal of decisions_oct18 item 3 and map the old calls onto `sim.calibrate()` in the script, or keep a thin class. **Recommend recording the reversal**, plus `n_workers=`/`storage=`.  [CK: ok]
5. **Box-then-exponential waning as an ODE (MA38, theme 7).** About 40 lines on existing kernels. **Recommend restoring it**, with HP18/HP28 staying agents only.  [CK: i'm worried that external libraries can't implement this themselves ...that seems to fail the flexibility criterion]
6. **S7.4 (theme 12).** Option A, plus A.1, HP41, HP46 and ST30, with Covasim's dimension needs deferred. **Recommend this.** [CK: don't remember what these refer to, but see answers above]

Smaller items where I'd go ahead unless you say otherwise: `init=` 1.0 as a share (theme 2); restoring `current_scale` as a public property; `verbose` fractions as progress; `auto` avoiding the ODE for CV > 3; one minimum step for scheduled dwells. [CK: yes go ahead]

## What this does to the plan

The [definition of done](plan.md#definition-of-done) says new items go to deferred.md "unless it is a silent wrong answer or blocks an item above". By that rule, the reviews add the following to the plan:
- **Silent wrong answers, about 45.** Most are S, and many collapse into themes 1, 2, 3 and 8.
  - tbsim: 1.
  - malaria: MA34–MA37.
  - HPV: HP37–HP42, HP44, HP46.
  - FPsim: FP41–FP56.
  - Covasim: CV1–CV5.
  - STIsim: ST28–ST39.
- **Blocks S8.1:** the benchmark set has no counts, coded models or `ss.by` tables (theme 5). FP57 should be fixed before S8.1 measures anything.
- **Blocks S7.4:** theme 12.

**My suggestion is a new stage, "S9 Review fixes", with three parts:**
- **S9.1:** the cross-cutting fixes (themes 1, 2, 3, 6 and 8, and ST28), each with a test.
- **S9.2:** the speed fixes (FP57, MA28, the copy-on-write check, ST50), and the benchmark additions.
- **S9.3:** the calibration fixes (theme 10, without the new features).

Themes 9 and 13 (beyond HP49–HP51), the script rules, describe() and the messages go to deferred.md or wait for S8.2. Counted at the plan's weights, this is about 15–20 units against the 62 the plan now counts, so **the estimate drops from 91% to about 72%**. The plan's update line should say so and why: the downstream rechecks of October 21 added the silent wrong answers they found, which the plan's rule puts in scope.

**The alternative** is to keep the plan's scope and move all of this to deferred.md, except ST28 (an unchanged downstream package silently wrong) and FP53 (`auto` choosing a wrong answer). That keeps the plan near 90%, but it ships v4's design with known silent wrong answers, which the plan's own rule forbids. I don't recommend it.

## Suggested order

1. **ST28** (nobody ages in any `sti.Sim`; a few lines). It is the only finding that changes an unchanged downstream package's results, and it hides behind a plausible epidemic.
2. **FP57** (`ss.by` caching), then add the missing benchmarks. Without it, every declared port looks unusably slow.
3. **Decisions 1–6**, since several of the fixes below depend on them.
4. **Theme 2 (`init=`), theme 1's check plus its cases, and theme 6's two contract tests with FP53.**
5. **Theme 8 (births), theme 3 (once decided), HP37–HP42 and HP44.**
6. **S7.4 per theme 12.**
7. **Theme 10's silent items and crashes, theme 11, and HP49–HP51.**
8. **The rest** to deferred.md or S8.2, each with a line saying why.

## Not covered

- [numba_vs_jax.md](numba_vs_jax.md), [partnerships_lit.md](partnerships_lit.md), [sexual_network.md](sexual_network.md) and [strains.md](strains.md) are studies rather than reviews. They are used here only where a review cites them.
- No finding was rerun for this synthesis.
- Speed numbers are the reviews' own, on a heavily loaded machine. Treat differences under about 30% as noise.
