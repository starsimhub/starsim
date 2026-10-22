# HPVsim: the port rechecked against v4 as of October 21

Written 2026-10-21. This rechecks [oct17/hpvsim_review.md](../oct17/hpvsim_review.md) (HP1–HP36, written at `03741ba2`) against `ssv4` at `ee5d1fb0` plus the uncommitted working tree (which holds `ss.SexualNet`, decision 18), frozen as a snapshot and run in the `v4` env, with Starsim 3.7.2 for comparison. HPVsim is unchanged (3.2.1, `9037e7b2`). As on October 18, five agents each took one part of HPVsim (natural history, genotypes, interventions, population, framework), ported every probe they reran to current spelling, and wrote their findings to the `hpvsim` repo under `v4port/<part>/oct21/FINDINGS.md`, with scripts and logs beside them. HPVsim's side (what HPVsim must change) is in `v4_port_oct21.md` at the root of the `hpvsim` repo. I reran the new silent wrong answers myself (HP37, HP38, HP39, HP40, HP41, HP42, HP44) and the result crashes (HP49, HP50); all reproduce. The machine was heavily loaded throughout (load 10 to 50), so timings are best runs.

New findings continue the numbering at HP37. Cross-references: [malaria_recheck.md](malaria_recheck.md) (MA), [tbsim_recheck.md](tbsim_recheck.md), and the October reviews (ST, D, N, L, X, R).

## In brief

**Most of the October 18 review is fixed.** Of HP1–HP36, 24 are fixed, 9 partly fixed or changed (mostly to a clear error, or to agents only), and 3 are still open (HP14, HP28, HP36). The STIsim set that HPV hit hardest is fixed too: ST1 (newcomers in no state), ST3 (one transmission per sex), and ST2 (per-act betas, now `ss.peract(m2f=, f2m=)` with directions resolved from sex, so HP3 is gone). ST4 is documented rather than changed.

**The declared HPV is now small, exact and fully declared:**

| Part | Oct 21 | Oct 18 | HPVsim | Agreement now |
|---|---|---|---|---|
| hpv16 natural history (exact form) | about 56 lines | about 102 | 446 | Women infected at 25: median time to cancer 23.1 against 23.4 years, age at cancer 53.8 against 54.0, P(cancer) 0.0275 against 0.0273 |
| Genotype machinery (cross-immunity, own immunity, seeding, one cancer per person, any-genotype results) | 20 | 95 | 402 | Cross-immunity effect 0.654 against 0.659 at 2 genotypes, 0.369 against 0.428 at 4; at HPVsim's speed (10.8 against 11.0 s CPU) |
| Products from HPVsim's CSVs | 22 | 29 (12 of them workarounds) | 330, plus 317 in Starsim 3 | Yields by state within noise (cytology abnormal on CIN 0.897 against 0.900); 20,000 tests counted for 20,000 (was 35,397) |
| WHO algorithms 1 and 3, routine vaccination | 16 | 17 | 55 | |

Every workaround the October 18 port needed in these parts is gone: the exit-time helper functions, the coded immunity connector (draws outside common random numbers), the age reconstruction, the newborns function, per-sex betas worked out by hand, 88 cross-immunity modifiers, the seeding function, the 41-line any-genotype analyzer, chains of binary tests, function-valued efficacies with rate-0 cure transitions, and the pseudo-test for a delay. What replaced them is what the October 18 review asked for: `p=` as an expression of `P.time` (HP18), `ss.attr(..., on=event, combine='max')` (HP19), `ss.Strains` with `ss.immunity(combine='sum')`, joint seeding and `exclusive=` (HP15, HP22, HP23), `ss.Diagnostic(results=...)` across genotypes (HP24), `cure=dict(P=, C=)` (HP25), `delay=`, `capacity=` and `queue=` (HP26, D31), and `disease=[...]` products (HP7). The run tools work on `hpv.Sim` (HP12, HP13), results are weighted by `people.scale` (HP4, for declared modules), and v3 seeds no longer lose a step (HP5: `hpv.demo()` over 8 seeds now matches v3, cancers 0.994, z −0.3).

**What's new and wrong, most important first:**
1. **Seven new silent wrong answers, each small to fix** (HP37–HP42, HP44):
   - placement doesn't fire `P.enter`, so `ss.attr(..., on=P.enter)` is NaN for everyone placed in a model with no transmission (HP37);
   - a product shared by a single-disease delivery and a `disease=[a, b]` delivery protects against `b` only if the list delivery is initialized first (decision 7; 0.000 against 0.996) (HP38);
   - `ss.treat(..., disease=[a, b])` with no `who=`/`where=` treats only people infected with `a`, and counts its outcomes twice (HP39);
   - `queue=ss.years(1)` drops the `never_ran` result, because `ss.years(1) == True` (HP40);
   - seeding goes only through a disease's first transmission, so with HPV written as one transmission per sex, every seed lands in the first sex's state (HP41);
   - `ss.by(age={0: 0, 15: 0.3, ...})` reads numeric keys as exact labels and gives everyone the mean (HP42);
   - coded `ss.Infection` results are not weighted by `people.scale`, so a multiscale sim reports a weighted `n_alive` beside an unweighted `n_infected` (HP44).
2. **HPVsim no longer imports** (HP53). It needs 17 names v4 removed, including `ss.SexualNetwork` (decision 18) and the v3 delivery, product and calibration classes the October 18 review asked to keep until v4 could replace them. v4 now can replace them (section 3 of the review is essentially done), but the migration script handles none of the three blockers (`ss.SexualNetwork`, `copy_inputs=`, the v3 classes), and HPVsim also imports STIsim, which fails on `class SimpleBV(ss.Disease)`. With a 60-line shim of v3's own classes, 338 of 373 tests pass and none fails in the engine.
3. **`ss.SexualNet` pairs on behalf of subclasses that pair their own** (HP45). `hpv.SexualNetwork` rebased on `ss.SexualNet` crashes with a `KeyError` about `layer_id`; a subclass with no extra edge keys would silently get Starsim 3 pairing at the start. Rule 2 of [sexual_network.md](sexual_network.md) wasn't implemented. With `super().init_post(add_pairs=False)`, HPVsim's network is 2 changed lines and matches v3 (age gap 3.70 ± 6.31 against 3.67 ± 6.13; marital degree 0.867 against 0.871).
4. **Two declared-results crashes** (HP49, HP50): `1e5*cancers/N` raises `TypeError: 'FlowSum' / float`, `cancers/N*1e5` silently loses its strata, and a member's flow over the family's flow fails when stratified. Cancers per 100,000 women by age is HPVsim's main output, so this is hit at once.
5. **Two regressions against the October 19 work, both from S3** (decided, but worth knowing what they cost HPV): the given-dwell natural history (HP18) runs only as agents (its compartmental form was built on October 19 and removed with clocks), and HP28 (dwells with a CV above 1) is deferred. So no declared HPV with a fittable `cin_fn` runs as an ODE.
6. **`init=dict(P=1.0)` is one person** (HP48, the same as MA34). All five parts tripped on it; every October 18 cohort script used it.

**S7.4 (strains):** the HPV evidence supports option A of [strains.md](strains.md), plus four additions to the family that the HPV port hit (section 4).

## The port today

| Task | Oct 18 (`03741ba2`) | Oct 21 (`ee5d1fb0` + working tree) |
|---|---|---|
| HPVsim unchanged | Runs; 359 of 368 tests pass, none an engine error; 1.3–2.4× faster | Doesn't import (HP53) |
| With a shim of v3's own classes (about 60 lines) | n/a | 338 of 373 pass, none an engine error: 20 removed names, 8 `copy_inputs`, 5 documented time conventions, 3 fragile by seed |
| `python -m starsim.interop.migrate` on HPVsim | n/a | 30 edits in 14 files, 78 items to do by hand; still fails at import (HP53) |
| HPVsim against Starsim 3 (`hpv.demo()`, 8 seeds) | Cancers 0.89 of v3 (HP5) | Within noise on every output (cancers 0.994, ASR incidence 43.9 against 43.7) |
| Speed, unchanged HPVsim (with the shim) | 4.0 s against 9.6 s (one genotype); 29.9 against 38.2 s (demo) | 2.9 s against 3.6 s; 11.2 against 15.1 s. v4 got faster; the 2.4× of Oct 18 was probably a v3 number inflated by load |
| Natural history, exact | Relied on an undocumented exit-time draw (HP1) | One transition, `p=` of `P.time`, documented; agents only |
| Per-person immunity, age at CIN | A coded connector, outside CRN | `ss.attr` on events; agents only |
| Four genotypes | 88 modifiers with the wrong rule, a seeding function, a 41-line analyzer | One `ss.Strains`; 20 lines; exact combination rule |
| HPV on HPVsim's network | Prevalence 1.6–1.8× too high, men a third, betas 4.3× too small, newcomers in no state | Right with HP45's one-line workaround |
| Bivalent vaccine | Silently hpv16 only (HP7) | `disease=[...]` or efficacy keyed by genotype; HP38 if mixed with a single-disease delivery |
| Cytology, colposcopy, treatment assignment | Chains of binary tests | `ss.Diagnostic(results=...)` from the CSV, worst result across genotypes |
| `treat_num`, `treat_delay` | No queue; a pseudo-test | `capacity=`, `queue=`, `delay=` |
| Multiscale | No result weighted | Declared results and `n_alive` weighted; coded ones not (HP44) |
| Cancer incidence per 100,000 women, age-standardized | No form | Per 100,000 by age works with a workaround (HP49); no standardized rates (HP23) |

## New findings

| ID | Finding | Kind | Evidence (`hpvsim:v4port/...`) | Fix | Size |
|---|---|---|---|---|---|
| HP37 | **Placement doesn't fire enter events for `ss.attr`.** With `init=dict(P=n)` and no transmission, `ss.attr(ss.ppl.age, on=P.enter)` is NaN for everyone placed; with a transmission it is set. Spec 03 says entering includes placement. The workaround (a rate-0 transmission) stamps placed people `ti_P = -1`, which moves time to CIN about 0.4 years earlier | Silent wrong answer | `natural_history/oct21/probe_attr_on_enter.py` (reran): 100 of 100 and 1,000 of 1,000 NaN without a transmission, 0 with | Fire the enter events for every placed person in the agents' `init_post` | 5–10 lines |
| HP38 | **Decision 7 depends on delivery order.** If a single-disease delivery initializes a shared product first, its modifiers are built for that disease only; a later `disease=[hpv16, hpv18]` delivery rekeys too late, and the one-product-one-disease check misses it because the product's disease equals the list's first | Silent wrong answer | `interventions/oct21/probe_hp7_multigenotype.py` 1d2 (reran): protection against hpv18 0.000; 0.996 in the other order | Key every product by the diseases of all its deliveries before any product's `init_pre()`, or raise when an initialized product is rekeyed | About 10 lines |
| HP39 | **`ss.treat(..., disease=[a, b])` with no `who=`/`where=` treats only people infected with `a`** (`default_eligible()` uses `self.dis`). Outcomes are counted per disease, so `cured + failed + unneeded` is twice `delivered`, and one person can be both cured and unneeded | Silent wrong answer | `interventions/oct21/probe_hp25_treat_family.py`, `probe_hp7_multigenotype.py` 3 (reran): hpv18 cured shares 0.459 and 0.472 against 0.936; 10,000 delivered, 13,973 cured, 986 failed, 5,041 unneeded | Default eligibility over every listed disease (`selector.select(self.edge_sources())`); outcomes per person | 3 lines, plus about 15 for outcomes |
| HP40 | **`queue=ss.years(1)` drops the `never_ran` result and event**: `queue not in [None, True]` is False because `ss.years(1) == True`. The limit is enforced, but 619 people leave the queue unrecorded; `ss.months(12)` works | Silent wrong answer (a missing result) | `interventions/oct21/repro_queue_one_year.py` (reran) | `self.queue is not None and self.queue is not True` (`interventions.py:513`). Worth a search for other `in [None, True]` tests on typed values | 1 line |
| HP41 | **Seeding goes only through a disease's first transmission** (`Agents.seed()` takes `graph.transmissions[0]`). HPV written as one transmission per sex (the natural form, and the October 18 port's) seeds every person into the first transmission's destination whatever their sex: 754 women in the men's state. The workaround, one transmission with a branch by sex, seeds right through a family but raises for a member's own `init_prev` | Silent wrong answer | `genotypes/oct21/probe_seed_sex.py` (reran) | Seed each person through the transmission whose `where=`/direction or branch admits them; raise if none or several do; allow a per-person `ss.by` branch when seeding as agents | 20–30 lines |
| HP42 | **`ss.by(age={0: 0, 15: 0.3, 25: 0.1, 50: 0})` reads numeric keys as exact labels**, so continuous ages never match and everyone, under-15s included, gets the weighted mean (0.10). The table form `ss.by(dict(age=[...], value=[...]))` is right | Silent wrong answer | `genotypes/oct21/probe_init_fn.py` a (reran) | Numeric keys on a numeric property are lower bin edges, as in the table form; or raise when no agent matches a key | About 5 lines |
| HP43 | **No per-person trait shared across a family.** One `ss.attr` object given to two members draws independently (correlation −0.024). HPVsim's `rel_sev` is one draw per woman, shared by every genotype | Silent (if shared), and missing | `genotypes/oct21/probe_attr.py` c | `ss.Strains(..., attrs=dict(rel_sev=ss.attr(...)))` read by members by name; meanwhile raise on a shared object | Small to medium |
| HP44 | **Coded `ss.Infection` results aren't weighted by `people.scale`**, while declared results and `n_alive` now are (HP4). In one sim: `n_alive` 5,050 (weighted) beside a coded `n_infected` of 7,105 (weighted 3,585); `people.n_alive` is 10,000 while `sim.results.n_alive` is 5,050. HPVsim's own genotypes weight by hand with `scale_flows()`; STIsim's HIV doesn't | Silent inconsistency (multiscale) | `population/oct21/repro_people_scale.py` (reran), `framework/oct21/repro_people_scale.py` | Weight `ss.Infection`'s default results (and `Module.update_results`' `state.sum()`); make `people.n_alive` weighted or rename it | About 15 lines |
| HP45 | **`ss.SexualNet.init_post()` pairs for subclasses that pair their own** (rule 2 of the design study, not implemented). Unchanged `hpv.SexualNetwork` on `ss.SexualNet` crashes: `KeyError: 'Cannot append edges since required key "layer_id" is missing'`; a subclass with no extra edge keys gets Starsim 3 pairing (15-year durations) silently | Crash with a misleading message; silent for other subclasses | `population/oct21/netstats.py hpv` | Skip the base pairing when a subclass overrides `step()` (or a class flag), with the study's `Couples` test | About 3 lines plus a test |
| HP46 | **A strain dimension's `immunity=` does nothing when the model has its own `R >> I` reinfection**: the immune states stay empty, with only a note that `verbose=0` hides (n_I 1,110 against 45). Also confirmed: `'+'` levels without `kind='strain'` give 0 co-carriers against 453, as tbsim found | Silent trap | `genotypes/oct21/probe_strain.py` 0, 3 | Raise, or make the reinfection's source an immune state; strains.md's A.1 for the second | About 5 lines |
| HP47 | **A dimensionless factor times a per-act beta becomes a bare-number beta with a unit.** `rate=0.275*({net: ss.peract(...)}*I)` (HPVsim's protection of seropositive women) makes `beta_reinfect = 0.275 per step`, and a sim mixing STIsim's months with HPV's years fails at `init()` with `IncompatibleTypes ... the unit of the bare numbers ... is not known`. `0.275*{net: ss.peract(...)}` is a `TypeError` | Crash in a mixed-unit sim; possibly wrong in one unit | `population/oct21/hiv_try.py` (c) | Treat a bare number multiplying a typed per-act beta as dimensionless, and print it as a factor | Small to medium |
| HP48 | **`init=dict(P=1.0)` is one person** (MA34), and `init=dict(P=fn)` raises a bare `TypeError: '>=' not supported between 'function' and 'int'` while `init_prev=fn` works. Found independently by four of the five agents | Silent change; unhelpful crash | `natural_history/oct21/probe_small.py`, `population/oct21/repro_hp36.py`, `genotypes/oct21/probe_init_fn.py` | MA34's fix; accept functions in `init=` as in `init_prev=`, or name `init_prev=` in the error | Small |
| HP49 | **`1e5*cancers/N` and `(1e5*flow)/N` crash** at the first `update_results` (`TypeError: unsupported operand type(s) for /: 'FlowSum' and 'float'`, `expressions.py:844`); `cancers/N*1e5` runs but silently drops the strata, so `.select(sex=...)` then says "not stratified" | Crash; silent loss of strata | `framework/oct21/repro_flow_ratio.py` (reran), `repro_result_strata.py` | Handle `FlowSum` in the evaluator; keep strata through scalar arithmetic | Small |
| HP50 | **A member's flow over the family's flow fails when stratified**: `hpv16.cancer / hpv.cancer` raises `AttributeError: 'numpy.ndarray' object has no attribute 'strata'` (`arrays.py:91`); unstratified, or `hpv16/(hpv16 + hpv18)`, works. The genotype share of cancers is an HPVsim calibration target | Crash | `framework/oct21/repro_family_ratio.py` (reran) | Keep a family's result a `Result` with strata | Small |
| HP51 | **`ss.ppl` expressions in `multiply=` crash**: `multiply=ss.ppl.age/50` or `1.0*ss.ppl.x` raises `TypeError: float() argument must be ... not 'Cond'` (`conditions.py:1071`, `per_person`), though expressions of `ss.attr` work there and the spec allows `ss.ppl` expressions in rates | Crash | `genotypes/oct21/probe_attr.py` c2 | Evaluate `ss.ppl` expressions per person in `per_person` | About 5 lines |
| HP52 | **`disease=[a, b]` over coded diseases fails**: "...which are not declared diseases in the sim; these are ." (an empty list). The spec says coded diseases count as `disease=`, and HPVsim's genotypes are coded modules, so this is exactly its bivalent vaccine if it keeps them coded | Dead end, unclear error | `interventions/oct21/probe_d23_coded.py` 8 | Accept coded names in the list (they have `rel_sus`, `infected`, `clear_infection()`), or at least name them | About 15 lines (2 for the message) |
| HP53 | **Migration: HPVsim neither imports nor migrates.** 17 removed names (`ss.SexualNetwork`, `ss.Calibration`, `ss.standardize_netkey`, the v3 delivery, product and test classes, `ss.treat_num`). `migrate.py` doesn't rename or flag `ss.SexualNetwork` (only `MFNet`/`MSMNet`), doesn't know `copy_inputs=` (removed in S2.6; `hpv.Sim` always passes it, giving a bare `KeyNotFoundError`), and only flags the v3 classes. Eight HPVsim tests rely on `copy_inputs=False` keeping a module reference live. HPVsim also imports STIsim even without HIV, and STIsim fails at import on `class SimpleBV(ss.Disease)` (`CapabilityError: ... which a declared disease never calls`) | Dead end (S8.2) | `framework/oct21/migrate_run.log`, `migrate.diff`; `population/oct21/v4shim.py` | Rules for `ss.SexualNetwork` → `ss.SexualNet` (with `add_pairs=False` until HP45), `copy_inputs=` (remove, and read `sim.diseases.x`), `standardize_netkey`; the `CapabilityError` names `ss.Infection` (MA41's bare coded base would serve STIsim's `SimpleBV` too) | Small each; part of S8.2 |
| HP54 | **Calibration infrastructure for HPVsim's fits.** Evaluations are serial (only Optuna's `n_jobs`, threads, is passed through); `storage=` is refused (`TypeError`), so no resume; long data with a `genotype` column fails (`UnknownDataColumn: "hpv"`); a genotype share as a data column fails, so it needs `gof=fn(sim)`. Saved results (`sc.save(fit)`) and top-N ensembles (2 lines) work | Missing | `framework/oct21/declared_calibrate.py` d | Process workers, `storage=`, and a mapping from a data column's values to a family's members | Medium |
| HP55 | **Cures in HPV's shape.** HPV reaches R only through branches, so a cure needs `to='R'`, and the error suggests `to="S"`; cured women are counted in men's clearance (`clear_m`, `where=` male), which `describe()` says. A scalar `cure=` also cures cancer (X counts as infected; X 0.948) | Friction; misleading counts | `interventions/oct21/probe_hp25_cure_by_state.py`, `probe_hp6_txvx.py` | Treat branch destinations of exits from infected states as recoveries; never count a cure in a transition whose `where=` excludes the people cured; document the cancer case | About 15 lines |
| HP56 | **HPVsim's severity rule can't be `combine='sum'` on `dur=`**: the family refuses any matrix whose sum can reach 1 (`OutOfBounds`), though HPVsim clips. It needs `combine=lambda p: np.minimum(p.sum(0), 0.999)` | Expressiveness | `genotypes/oct21/genotypes_v4_perperson.py` | A `cap=`, or clip at 1 − ε, rather than refusing | About 5 lines |
| HP57 | **`describe()`**: a transmission on a sexual network reads `× I.n / N [homogeneous mixing]`; `move=` is no longer shown (it was HP36's only visible trace); a declared disease prints as "DiseaseBase"; the male-clearance modifier reads `where=~~people.female`; an expression `p=` prints as several hundred characters of inlined functions; a family prints one line per `dur=` state | Misleading output | `population/oct21/describe_net.py`, `natural_history/oct21/describe_exact.log` | Name the network route and "per act, per partnership"; print `move=`; the class name; simplify `~~` | Small |
| HP58 | **Three spurious `RuntimeWarning: Dates with years < 1 are not valid` per sim** with a round at a date and steps under a year (`connectors.reached()` subtracts two dates). Replaces HP31's warning, which is gone | Friction | `interventions/oct21/repro_hp31_where_dur_warning.py` | Compare years as floats | 1–2 lines |
| HP59 | **`X.time[uids]` and `X.time` in an expression differ in type**: indexed, it is a plain number in the sim's unit, so `P.time[uids]/ss.years(1)` is a `peryear` and fails on `>`; in an expression `P.time/ss.years(1)` is years | Inconsistency | `natural_history/oct21/probe_branch_at_exit.py` | Return durations from the indexed read, or document it | Small |
| HP60 | **The uncoupled family under the ODE is slow**: four 21-compartment genotypes side by side take 14.6 s against 0.1 s for one. Not needed by HPV; bears on strains.md option B | Performance | `genotypes/oct21/probe_ode.py c_uncoupled` | Profile; likely a Python path per step | Unknown |
| HP61 | **Messages** that should name the fix: weighted destinations (removed in S6.2) give a bare `TypeError: unsupported operand type(s) for @: 'BinOp' and 'peract'`; `ss.by(age=[...], value=[...])` in a cure gives `cannot unpack non-iterable NoneType`; `on_positive=` gives a generic "unexpected keyword (see the migrate-starsim3 skill)" rather than the two steps to write, as the spec promises; `ss.Disease('hpv', ...)` rejects a leading name that `ss.Strains` accepts; `ss.attr('age', ...)` says "'age' is not one of its attributes" | Friction | Various | One line each | Small |

## Status of the October 18 findings

| ID | Finding (Oct 18) | Status Oct 21 | Evidence and note |
|---|---|---|---|
| HP1 | Two spellings of a branch decide at different times | Fixed (documented) | Spec 03 states both rules; `P.time` in the entry spelling raises a `CapabilityError` naming the one-transition form. A function or age expression there still runs silently, per the rule; `describe()` says when a branch is decided at exit only for `P.time` |
| HP2 | An immediate junction holds people a step | Fixed | `dur=0`; `n_J` = 0 under abm, ode and binomial |
| HP3 | A `beta` pair is p1 → p2 | Fixed | `ss.peract(m2f=, f2m=)` resolves directions from sex; a bare pair raises `RouteMismatch` naming `peract` |
| HP4 | Results ignore `people.scale` | Partly fixed | Declared results and `n_alive` weighted (HPVsim's Nigeria `n_alive` 206.7M, WPP 211.7M, v3 328.8M); coded ones not (HP44) |
| HP5 | Unchanged v3 seeds transmit one step fewer | Fixed | `hpv.demo()`, 8 seeds: infections 0.989, cancers 0.994, ASR incidence 43.9 against 43.7 |
| HP6 | An efficacy on a branching exit scales the exit's rate | Half fixed | `'precin.C'` is right (CIN share 0.102, dwell unchanged); the bare key `'precin'` is still silently the rate (0.179, 2.88 against 1.67 years in P), and `describe()` doesn't say so. Refuse a bare key on a multi-destination exit, listing `'precin.C'`/`'precin.R'` (about 10 lines) |
| HP7 | One vaccine for two diseases protects against the first | Fixed, with HP38 | Two single-disease deliveries now raise `UnknownDisease` with the fix; `disease=[...]` and efficacy keyed by disease are right (0.998/0.996; 0.998/0.468) |
| HP8 | `MFNet` degree results × `pop_scale` | Fixed (`ss.SexualNet`) | 0.497 with and without `total_pop` |
| HP9 | `MFNet` newcomers at any age never join | Fixed | Immigrants 25+: participant 1.000 |
| HP10 | Participation by age read at age 0 | Half fixed | Now drawn at debut (0.822 for p = 0.9 from 15). HPVsim's `layer_probs` are an eligibility each step; mapped literally at debut they halve the married share (0.42 against 0.87), and nobody leaves the network with age |
| HP11 | `MFNet` degree target misses in an open population | Fixed | 80-year marriages: 0.862 against 0.87 (was 0.654) |
| HP12 | `hpv.Sim` loses every run tool | Fixed | `new()` returns an `hpv.Sim`; `ss.parallel`, sweeps, `diff()`, `calibrate()` all work |
| HP13 | `diff()` refuses coded-only sims | Fixed | |
| HP14 | Strain `immunity=` refuses networks | Still open | The message suggests compartments, which also refuse (KI15). Not needed by HPV |
| HP15 | Initial conditions can't be functions; nothing joint | Fixed | Function `init_prev` in all spellings; the family seeds jointly and exclusively. Gaps: HP41, HP42, HP48 |
| HP16 | A split with no dwell crashes | Clear error | `UndrivenTransition ... give it dur=0`. Spec 03 contradicts itself (line 347 "happens immediately" against line 355) |
| HP17 | A coded array as a modifier amount crashes | Fixed | `ss.modify(hpv.rel_sus, multiply=hiv.hiv_rel_sus)` |
| HP18 | No outcome that depends on the drawn dwell | Exact as agents; nothing as compartments | `p=` of `P.time` on one transition: within 1% of v3. ODE and binomial raise `CapabilityError: P.time is a clock`; the Oct 19 compartmental form went with S3 |
| HP19 | No per-person value on events, or frozen at entry | Fixed as agents | `ss.attr(..., on=event, combine='max', init=0)`, gated by an earlier attribute, is exactly HPVsim's running maximum (0 decreases in 44,838 updates); `ss.attr(ss.ppl.age, on=P.enter)`. Gaps: HP37, HP43 |
| HP20 | A branching flow has no destination axis | Fixed | `res.course.C`, `select(to='C')` |
| HP21 | Remainder beside another timed exit raises E205 | Fixed | |
| HP22 | 88 modifiers and the wrong combination rule | Fixed through `ss.Strains` | `ss.immunity(combine='sum')` gives 0.3500 exactly; plain modifiers still multiply (cross-immunity effect 0.583 against 0.432 and HPVsim's 0.428) |
| HP23 | No union across diseases; no standardized rates | Partly fixed | Unions within a family (`res.hpv.infected` is the OR); outside one, `ss.outcome(A.x + B.x)` raises a cryptic `KeyNotFoundError`; no standardized rates |
| HP24 | Tests have two results and act on one disease | Fixed | `ss.Diagnostic(results=...)`, `'disease.state'` keys, worst result across genotypes. A result can't be named `positive`/`negative`, so HPVsim's CSV needs two renames |
| HP25 | No treatment efficacy by state | Fixed for one disease | `cure=dict(P=0.936, C=0.936)` gives 0.928/0.939; over a list see HP39; shape issues in HP55 |
| HP26 | No `delay=` on `ss.treat` | Fixed (agents) | |
| HP27 | `ss.change` on a `beta` pair crashes | Fixed | |
| HP28 | CV > 1 dwells become one Erlang stage | Deferred (superseded by S3) | ODE numbers identical to Oct 18 (mean time to cancer 33.4 against 27.5 years as agents) |
| HP29 | Cross-disease conditions make compartments grow as 20^k | Partly fixed | Counted against `max_joint` with a clear reason (3 genotypes 9,261, 34 s; 4 over budget). The "as soon as" cancel still hangs the ODE at 441 compartments (ST26); the family's `exclusive=` avoids it as agents |
| HP30 | Lazy conditions cost 0.2 ms per modifier per step | Partly fixed | About 0.13 ms remains (88 modifiers: 1.50 s against 0.59 s per 80 steps); the same `where=` is evaluated 88 times. Evaluate each distinct condition once per step and fold `multiply=` modifiers on one target (about 15 lines). The family removes the need for HPV |
| HP31 | Divide-by-zero warning on every `where=` dwell | Fixed | See HP58 |
| HP32 | `describe()` says steps for a sim in years | Fixed | |
| HP33 | Named birth and death rates missing | Mostly fixed | `ss.Births(birth_rate='nigeria')`, `ss.Deaths(death_rate='nigeria')` work; `ss.Pregnancy(fertility_rate='nigeria')` still raises `TypeError` |
| HP34 | A modifier named like a People state | Fixed | Clear error suggesting a name |
| HP35 | A branch `ss.par` gets a wrong example name | Fixed | |
| HP36 | A `move=` branch that doesn't apply is dropped | Still open, and less visible | The docstring now says a `move=` applies only to people at the first level, not that other branches take its probability; `describe()` no longer shows it (HP57) |

Of the known items HPV hit again: ST1, ST2, ST3, L4, L26, ST14, ST16, D23 (on `ss.Infection`) and D31 are fixed; ST4 is documented ("Won't do"); D40 (all-or-nothing) exists, but HPVsim's vaccine is a mixture (a 95% take with per-genotype protection, the rest leaky) that `leaky=False` can't say, since it takes only a number; D27 (a product and its delivery can't share a name) is a clear error; `stratify=` on a coded module is still silently ignored.

## What a joyful HPVsim looks like now

Mostly real, in current spelling (the names marked `# proposed` don't exist):

```python
import starsim as ss, hpvsim as hpv

def genotype(g):
    gp = hpv.get_genotype_pars(g)
    S, P, C, X, R, L = ss.states('S P C X R L')
    p_cin = hpv.logf2(sev*P.time/yrs, **gp.cin_fn)                       # decided at exit, from the time spent (HP18)
    return ss.Disease(S, P, C, X, R, L, name=g,
        infection = ss.Transition(S + R, P, rate=dict(sexualnet=ss.peract(m2f=gp.beta*gp.transm2f, f2m=gp.beta))*(P + C)),
        precin    = ss.Transition(P, [C, R], dur=gp.dur_precin, p=[p_cin, None]),
        cin       = ss.Transition(C, [X, R], dur=gp.dur_cin, p=[p_cancer, None]),
        age_cin   = ss.attr(ss.ppl.age, on=C.enter),                         # frozen at entry (HP19; HP37 for cohorts)
        nab       = ss.attr(ss.uniform(0.5, 0.95)*sero, on=R.enter, combine='max', init=0))

genotypes = ss.Strains('hpv', [genotype(g) for g in ['hpv16', 'hpv18', 'hi5', 'ohr']],
    immunity  = [ss.immunity(protected='nab', rel_sus=m_sus, combine='sum'),
                 ss.immunity(protected='cell', dur=dict(P=m_sev), combine='sum', cap=0.999)],   # proposed: cap= (HP56)
    attrs     = dict(rel_sev=ss.attr(ss.normal(1, 0.2))),                                       # proposed: shared trait (HP43)
    init_prev = hpv.init_prev_total, exclusive='X')
genotypes.stratify(age=hpv.WHO_EDGES)

cytology = ss.screen(dx('lbc'), who=women & (lbc.since >= ss.years(3)), coverage=ss.probperyear(0.2), disease=genotypes)
colpo    = ss.screen(dx('colposcopy'), who=cytology.abnormal | hpv_triage.positive, disease=genotypes)
ablate   = ss.treat(tx('ablation'), who=colpo.hsil, capacity=ss.peryear(5000), queue=ss.months(12), delay=ss.months(3), disease=genotypes)
bivalent = ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975))
sim = hpv.Sim(location='nigeria', diseases=genotypes, networks=hpv.SexualNetwork(), interventions=[ss.vaccinate(bivalent, at_age=9), cytology, colpo, ablate], data='nigeria_cancers.csv')
fit = sim.calibrate(n_runs=4)
ss.parallel(baseline=fit.sim, screening=fit.sim.new(interventions=...), n_runs=50).diff('baseline').plot()
fit.sim.results.hpv.cancer.rate(per=1e5, of='women', standard='who2000')                       # proposed (HP23, HP49)
```

Still needing a workaround today: a rate-0 transmission in a cohort (HP37); one transmission with a branch by sex, so seeding works (HP41); `rel_sev` folded into each genotype's branch probabilities (HP43); a function `combine=` (HP56); `to='R'` and `where=~X` on every treatment (HP55); `'precin.C'`, never `'precin'` (HP6); declaring rates as `x/N` and scaling after selecting (HP49); `gof=fn(sim)` for the genotype share (HP54); the network rebased with `add_pairs=False` (HP45); and no standardized rates (HP23).

## S7.4: what HPV says about strains

[strains.md](strains.md) asks for a comparison of strain families and a strain dimension on HPVsim's and tbsim's needs before any representation rewrite. HPV's answer, with evidence from `genotypes/oct21/`:

- **HPV needs the family, not a dimension.** Co-carriage as levels needs 15 levels at 4 genotypes and 511 at 9, and one clearance clears them all. A strain dimension with `immunity=` refuses networks (HP14) and compartments (KI15), and has the HP46 trap.
- **The family, as built, works for HPV.** With per-person attributes it reproduces HPVsim's cross-immunity (0.654 against 0.659 at 2 genotypes; 0.369 against 0.428 at 4) in 20 lines at HPVsim's speed. The combination rule matters: a product, all that plain modifiers can do, moves the 4-genotype effect to 0.58.
- **HPV is agents only** (the network and `ss.attr` need agents), so option B (making the family agents only) costs HPV nothing, and C and D gain it nothing (D would break the separate courses per genotype).

**Recommendation: option A**, which tbsim also supports (A.1, inferring `kind='strain'` from `'+'` levels, is confirmed by both), plus four things the HPV port hit in the family: seeding through each person's own transmission (HP41), a cap for summed severity (HP56), a trait shared across members (HP43), and, on the results side rather than strains, unions over diseases outside a family and standardized rates (HP23). HP14's message should stop pointing to compartments that also refuse.

## Speed

| What | Oct 21 | Oct 18 | Verdict |
|---|---|---|---|
| Unchanged HPVsim (with the shim), one genotype, 10,000 agents | 2.9 s against v3's 3.6 s | 4.0 against 9.6 s | v4 itself got faster; occasional slow outliers (17–24 s), probably a mid-run compile, not investigated |
| 4-genotype demo with multiscale | 11.2 against 15.1 s | 29.9 against 38.2 s | 1.3× |
| `hpv.SexualNetwork` on `ss.SexualNet` | Whole runs, 7.1–9.8 s against v3's 10.9–11.7 s (3 seeds) | Its `step()` 2.26 against 5.12 s | Same network, faster |
| Declared 4 genotypes, per-person family | 10.8 s CPU against HPVsim's 11.0 | 8.9 against 8.7 | Par |
| Declared natural history (exact form), 20,000-woman cohort | 3.6–9.2 s against v3's 4.7–5.2 s, under load 40 | About 2× faster | Within noise at this load; remeasure on a quiet machine |
| Interventions, 20,000 agents, 1980–2060 | Baseline 5.2 s; algorithm 3 10.4 s | | |

## Keep

- **`ss.Strains` with `ss.attr`.** It turned 402 lines of HPVsim connectors and analyzers into 20 and gets the combination rule right.
- **Branches decided at exit from the time spent**, which made HPV's natural history exact in one transition.
- **`ss.Diagnostic(results=...)` from a table, across genotypes**, and outcomes as named events (`colpo.hsil`, `cytology.abnormal | hpv_triage.positive`), ordered by trigger rather than by listing.
- **`capacity=`, `queue=`, `delay=`, `.since`**: `treat_num` and `treat_delay` are a keyword each.
- **`ss.SexualNet`'s fixes** (HP8, HP9, HP11) and the run tools on subclassed sims (HP12).
- **The new refusals**: a product bound to another disease, unknown keywords, by-state values on a coded disease, an ODE asked of a coded model. And the warning when a coded `set_prognoses()` skips `super()`, which found that HPVsim's HIV records no new infections on v4.

## Suggested order

1. **The silent wrong answers, about a day together:** HP40 (1 line), HP42, HP37, HP38, HP39, HP44, HP41, and HP6's refusal of a bare key; each with a regression test. By the plan's rule these belong in the plan rather than deferred.md.
2. **The result crashes:** HP49, HP50, HP51; small, and hit by the first HPV output anyone writes.
3. **HP45** (3 lines and a test), so that downstream networks rebase on `ss.SexualNet` without surprises.
4. **S7.4:** option A with HP41, HP56 and HP43; HP46's raise.
5. **When S8.2 starts:** HP53's migration rules (`ss.SexualNetwork`, `copy_inputs=`, the `ss.Infection` hint), then HP54 for HPVsim's calibrations, and HP52 if HPVsim keeps coded genotypes.
6. **For Cliff:** whether any compartmental form of HP18 should come back (removed with clocks in S3; it cost HPV its ODE), and whether HP10's per-step participation belongs in `ss.SexualNet` or stays with HPVsim's own network (N15 says the latter).

## Index of new findings

| ID | Finding | Kind |
|---|---|---|
| HP37 | Placement doesn't fire enter events for `ss.attr` | Silent wrong answer |
| HP38 | A shared product's diseases depend on delivery order | Silent wrong answer |
| HP39 | `ss.treat` over a list treats only the first disease's infected | Silent wrong answer |
| HP40 | `queue=ss.years(1)` drops `never_ran` | Silent wrong answer |
| HP41 | Seeding only through the first transmission | Silent wrong answer |
| HP42 | `ss.by` dict keys on age are exact labels | Silent wrong answer |
| HP43 | No trait shared across a family | Missing |
| HP44 | Coded results unweighted by `people.scale` | Silent inconsistency |
| HP45 | `ss.SexualNet` pairs for subclasses | Crash; silent |
| HP46 | Strain `immunity=` inert beside an explicit reinfection | Silent trap |
| HP47 | A factor times a per-act beta becomes a bare beta | Crash |
| HP48 | `init=dict(P=1.0)` is one person; no function in `init=` | Silent change; crash |
| HP49 | `1e5*flow/N` crashes; scalar arithmetic drops strata | Crash; silent |
| HP50 | Member over family flow fails when stratified | Crash |
| HP51 | `ss.ppl` expressions in `multiply=` crash | Crash |
| HP52 | `disease=[...]` over coded diseases fails | Dead end |
| HP53 | HPVsim neither imports nor migrates | Dead end (S8.2) |
| HP54 | Calibration: workers, resume, genotype columns | Missing |
| HP55 | Cures in HPV's shape | Friction |
| HP56 | No cap on a summed severity rule | Expressiveness |
| HP57 | `describe()` on networks, `move=`, class names | Misleading output |
| HP58 | Spurious date warnings | Friction |
| HP59 | `X.time[uids]` and `X.time` differ in type | Inconsistency |
| HP60 | Uncoupled family slow under the ODE | Performance |
| HP61 | Messages that should name the fix | Friction |
