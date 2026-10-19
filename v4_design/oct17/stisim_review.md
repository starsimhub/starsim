# STIsim: what v4 should change to make the port a joy

Written 2026-10-18, at commit `03741ba2` on the `v4` branch of `ssv4`. The `ssv4` working tree was being edited during the work, so every finding was rerun against the committed `03741ba2`. STIsim 1.7.1 (`f9af9e8`, the `v4` branch of `stisim`) was used as a real model to test v4: five agents each took one part of STIsim (curable STIs and BV; HIV; syphilis; networks; the framework of sim, test and treat, calibration and analyzers), ran it unchanged under v4, prototyped a declared version, and compared the two over several seeds. Scripts and outputs are in the `stisim` repo under `v4port/<part>/`. That repo's `stisim_port_review.md` is the companion written for STIsim's side (effort per part, and STIsim's own bugs).

This file is about v4: what has to change, or should, so that porting STIsim is simpler and faster with no loss of what STIsim does, and with v4's added functionality on top. New findings get IDs ST1–ST27 so they can join the registers. The rest are cross-referenced to the merged findings (L, D, X, R, P) and [new-findings.md](new-findings.md) (N). The in-progress `fuzz_review/` and `stats_review/` have none of the ST findings.

## In brief

**v4's promise holds for STIsim:**
- **Unchanged STIsim runs about 1.6 times faster**, with the same statistics: 192 of 198 tests pass, and the Zimbabwe HIV example takes 18.8 s against 29.7 s on the same interpreter.
- **The declared versions are a tenth to a fifth of the size and agree with v3 where the model is the same:**

  | Part | Declared | v3 |
  |---|---|---|
  | Syphilis with its care cascade | about 160 lines | 1,460 |
  | Gonorrhea, chlamydia and trichomoniasis | about 85 | 823 |
  | BV | about 50 | 400 |
  | HIV natural history | 53 | 939 |
  | Connectors | about 15 | 220 |

- **They are more correct than the v3 code.** Two of v3's hand-built syphilis results are wrong, the GUD connector overwrites stage infectiousness, a treated BV case stays "infected", and BV's rates depend on `dt`.
- **They gain what STIsim never had**:
  - an ODE of the same disease;
  - `describe()`;
  - results by stratum for free;
  - `diff()` scenarios;
  - `calibrate()` without a `build_fn`.

**What stops the port being a joy, in four groups:**

1. **Seven new silent wrong answers in the agent engine for declared diseases** (ST1–ST7). They hit any STI model, and none raises anything:
   - children born during a run are in no state of the disease;
   - a per-act probability has no spelling;
   - per-direction `beta` is applied after the acts;
   - newly infected people stay a step too long;
   - a cure misses most infected states;
   - two branch forms never fire or don't take the remainder.

   Each is 10 to 50 lines to fix.
2. **The seam between code and declarations.** STIsim will keep coded modules for a long time, because its sexual network and CD4 have no declared form. The seam failed everywhere it was tried: four failures are new (ST8 to ST11), and the rest are the planned bridge (D23, R13, N19, N20).
3. **The STI vocabulary is missing**:
   - sex as a real dimension (D15);
   - per-act transmission;
   - condoms, partnership types, age preference and sex work on networks;
   - stock coverage targets (ART, VMMC, PrEP);
   - pregnancy events for ANC;
   - remembered test results;
   - cures with several destinations;
   - accuracy and treatment-outcome results.
4. **Compartmental runs refuse STI shapes or are slow on them.** Two non-exponential exits from one state are refused even when disjoint by sex. Narrow dwell times become hundreds of Erlang stages and hundreds of substeps: the syphilis ODE takes 5 s, BV's takes 10 s per simulated year, and `calibrate()` took 1,063 s on a 0.6 s ODE.

**Counts:** 50 adverse findings: 27 new, 7 related to an existing finding, 16 duplicates. Groups 1 and 2 are mostly small fixes, about 2 to 3 weeks of v4 work besides the bridge for coded modules already planned (D23, R13, N19, N20). Group 3 is about 4 to 6 weeks if the network work is included. Group 4 is medium.

## What a joyful STIsim looks like

The target, in proposed spellings (the names marked `# proposed` don't exist today). The network and HIV's per-person CD4 stay coded; everything else is declared, and every v4 verb reaches every module.

```python
import starsim as ss, stisim as sti

S, E, P, Sec, EL, LL, T, Sx = ss.states('S E P Sec EL LL T Sx')
syph = ss.Disease(
    infection = ss.transmission('S -> E', beta=ss.peract(m2f=0.3, f2m=0.15),          # proposed: per act, per direction (ST2, ST3)
                                source=dict(P=1, Sec=1, EL=0.55, LL=0.05),
                                maternal=ss.permonth(0.8)),                           # proposed: route-specific weights (ST17)
    primary   = E >> P @ ss.lognorm_ex(ss.weeks(6), ss.weeks(1)),
    ...                                                                               # about 60 lines for the natural history
    cure      = ss.cure(to=Sx),                                                       # proposed: a cure from every infected state (ST5)
)
syph.stratify(sex=..., age=[15, 20, 25, 35, 50])                                      # sex is people's sex as agents (D15)

net  = sti.StructuredSexual(condom_eff=ss.par(0.9, bounds=[0.6, 1]))                  # coded, but its pars are fittable (bridge, ST9)
preg = ss.Pregnancy(fertility_rate='zimbabwe')                                        # newborns join every declared disease (ST1)
rpr  = ss.Test(sensitivity=dict(P=0.9, Sec=0.95, EL=0.85))
anc  = ss.cascade(                                                                    # proposed: a cascade of peer steps, linked by name (section 3)
    visit = ss.attend(on=preg.conception, delay=ss.months(3), coverage=0.7),          # proposed: pregnancy events (ST13)
    test  = ss.screen(rpr, on='visit'),                                               # the result is kept per person (ST14)
    treat = ss.treat(bpg, on='test.positive', coverage=0.95),
    fetus = ss.treat(bpg_fetal, on='treat', trace='maternalnet'),
)
art  = ss.treat(art_product, disease='hiv', target=ss.timeseries(n_art), priority=hiv.cd4)                     # proposed: stock target (ST12)
conn = ss.modify(hiv.rel_sus, multiply=2.67, where=syph.P | syph.Sec)                 # coded HIV, declared syphilis (bridge, N19)

sim = sti.Sim(diseases=[hiv, syph], networks=net, demographics=preg, interventions=[anc, art], connectors=conn, data='zimbabwe.csv')
sim.describe()
fit = sim.calibrate(workers=8, resume=True)                                           # proposed: STIsim's calibration features (section 5)
ss.parallel(baseline=fit.sim, scale_up=fit.sim.new(interventions=[anc.new(visit=dict(coverage=0.95)), art]), n_runs=50).diff('baseline').plot()
```

Today, the pieces of this that run do so only with workarounds: a per-step function putting newborns in `S`, `ss.prob(b, dt)` for per-act betas, a dedicated rate-0 `cure` transition from every state, a `net_beta` override reading `.value`, lambdas for conditions on HIV, `deliver=` functions for ART, functions for ANC timing, and `sti.Sim` rebuilt to pass its modules to `ss.Sim`.

## The port today, and after the changes

| Task | Today | After the changes below |
|---|---|---|
| A declared STI next to STIsim's demography | The epidemic silently dies out as the population turns over (ST1) | Works |
| STIsim's betas | A verbatim port is 12x too small (N7); no spelling is right at another step (ST2); unequal directions wrong (ST3) | `ss.peract(m2f=, f2m=)` |
| Treat an STI with separate asymptomatic and symptomatic exits | Silently cures 47% instead of 88% (ST5) | `ss.cure()` or `ss.Treatment` with no `transition=` cures everyone infected |
| Declared disease over the coded network | Crashes on `eff_condom` (ST9) | Works |
| Connectors | STI → coded HIV is `E504` (D23); `where=hiv.cd4 < 500` crashes (N19) | One `ss.modify` line each |
| Scenarios, sweeps, `calibrate()` on `sti.Sim` | Refused, even with a declared disease (ST10, R13) | Work |
| ART, VMMC, PrEP to coverage targets | `deliver=` functions, agents only, state leaking across copies (ST12) | `target=` and `priority=` |
| ANC and newborn testing | Functions reading `ss.Pregnancy` arrays; no remembered result (ST13, ST14) | A cascade of peer steps from `preg.conception`, with results kept per person |
| Results by sex | A random half (D15) | Right under every solver |
| An ODE version | Refused for sex-specific durations (ST24); BV 10 s per simulated year (ST26) | One text for agents and ODE; fast |
| Declared network | 33.7% prevalence against 8.8% (N15) | Not needed if the coded network's parameters are first-class; a declared `SexualNet` later |

## What should change in v4

### 1. Fix the silent wrong answers in the agent engine

These were each found while matching a v3 module and confirmed at `03741ba2`. Every one is silent.

| ID | Finding | STIsim evidence | Fix | Size |
|---|---|---|---|---|
| ST1 | **People added by `people.grow()` (`ss.Births`, `ss.Pregnancy`, STIsim's `Migration`) are in no state of a declared disease.** They are never susceptible, fetuses can't be infected, and they are missing from the state counts. Only `ss.birth()` inside the disease places people (`Agents.step_births`); every state is a `BoolState` defaulting to False. | `stisim:v4port/hiv/repro_pregnancy_newborns_no_state.py` (I reran it): all 797 children born during a 10-year run in no state with `ss.Births`, 738 of 738 with `ss.Pregnancy`, 0 with a coded `ss.SIS`; `n_S + n_I` 1,834 against 2,701 alive. Declared HIV prevalence 0.034 against 0.078 in 2024; congenital syphilis 180 to 1.8 infections a run | Place new agents in each declared disease's first people state (at the split of its dimensions), as `ss.birth()` does, with `ti_` set; a test that `n_` states sum to `n_alive` with every demography module | About 20 lines. Not in the review: D28 compares only population size |
| ST2 | **A per-act probability has no spelling.** `06-routes.md` says `beta` is per act on a network with acts, but `Agents.infect()` converts it as a rate to a per-step probability (`-expm1(-rate*dt)`) and then the network raises it to the number of acts. `ss.probpermonth(b)` is right only at monthly steps. Every STIsim beta is per act | `stisim:v4port/networks/beta_per_act_dt.py`: per-act 0.01 on `ss.MFNet`; coded convention 37.5% monthly, 28.9% weekly; `ss.prob(b, dt)` 38.6%, 30.5%; `ss.probpermonth(b)` 38.6%, 11.6% | A typed `ss.peract(p)` that reaches `net_beta()` unconverted, or read a bare `beta` on a network with acts as per act (as the spec says); `describe()` says which | About 20 lines. Related to N7 |
| ST3 | **A per-direction `beta` pair is applied after the acts, not per act.** v3 computes `1 - (1 - beta_d)^acts` for each direction. v4 uses the larger of the two for both and multiplies the per-partnership probability by their ratio (`Agents.infect()`). | `stisim:v4port/syphilis/probe_direction_beta.py`: female to male per partnership-month 0.278/0.232, 0.478/0.346, 0.728/0.429 (v3/v4) at 2, 4, 8 acts (StructuredSexual's median is 6) | Work out the per-act probability per direction | About 10 lines; changes results for unequal pairs |
| ST4 | **As agents, everyone infected through a declared transmission spends one extra step in the state they enter.** States entered by progression are exact. | `stisim:v4port/syphilis/probe_entry_dwell.py` (reran): `E >> I @ ss.constant(ss.days(3))` lasts 4 steps for people infected during the run, `I >> R @ 5 days` lasts 5. Syphilis incubation 2.64 monthly steps against 1.66 in v3; the epidemic grows about 15% slower in year 1 | Count the dwell from the step of infection, as for progressions | Small. Related to L14 and X7, which cover seeds only |
| ST5 | **A curing `ss.Treatment` moves only people in the source states of one transition.** When several exits qualify it raises `E505`, and naming one (the natural fix) silently leaves everyone in the other infected states uncured. A cure whose destination depends on the state (BV to CST 1 or 3; syphilis serofast after late treatment) needs two products and two deliveries. | `stisim:v4port/framework/test_and_treat.py`: chlamydia 47% of treated positives cured against 88% with a rate-0 `cure` transition from every infected state; gonorrhea 74% against 90% | A cure with no `transition=` moves every infected state to the recovery destination, or `to=`; `transition=[...]` for several; warn when the named transition's sources are a strict subset of the infected states | About 30 lines. Related to D18 |
| ST6 | **`where=` on a transition with `p=` never fires.** The branch exit is scheduled once, as never, and not reconsidered when the condition becomes true. | `stisim:v4port/syphilis/probe_dur_function.py` (reran): 0 of 2,000 leave with `p=1, where=due`; 2,000 without `p=`. 188 children stayed fetal up to age 14 | Re-evaluate in `rescale_exits()`, or refuse `where=` with `p=` | Small |
| ST7 | **A function-valued branch `p=` beside a timed exit without `p=` competes with it instead of leaving it the remainder.** (`transitions.py:1472` takes the remainder only when every `p` is a constant.) | `stisim:v4port/seis_bv/branch_remainder.py`: 0.440 to the branch against 0.9 | Take the remainder at run time, or raise | Small. Related to L14 |

Three more silent problems are in the review already, and STIsim makes each worse:

- **X7, dwells not following a change of level:** with `adjust=0` ("ART halts progression", the standard use of a care dimension), anyone who enters a stage on ART never progresses again after stopping. Before the workaround, 100% of the infected had CD4 ≥ 350 in 2024, against 93% in v3.
- **X1, a distribution replaced by its mean (related, a new instance):** the same happens to a transfer's `dur=`. `ss.lognorm_ex(15 y, 5 y)` behaves as an exponential, and syphilis RPR reversion is 41 points off at 8 years.
- **D11 and D12, ages redrawn when stratifying:** adding STIsim's age bins for results redraws ages from 0 to 100 and switches on aging, which halves infections.

### 2. Make the seam between code and declarations seamless

STIsim is the strongest evidence for the Starsim 3 seat's argument that coded modules must stay first-class. Its network is 2,100 lines that `PartnerNet` can't replace (N15). Its HIV carries per-person CD4 with a nadir effect that stages lose (25 to 50% fewer deaths in the ART scale-up years). Its syndromic algorithm and partner notification are decisions in code. A real port is therefore always mixed, and the seam fails at every point it was tried.

| ID | Finding | Evidence | Fix | Size |
|---|---|---|---|---|
| ST8 | **v4 calls every module's `update_results()` at `ti=0`, before any `step()`, so coded modules that create state in `step()` crash.** All of STIsim's treatments do (`STITreatment`, `GonorrheaTreatment`, `SyphTx`, `NewbornTreatment`, `SymptomaticTesting`). No STIsim test uses them, so its suite (192 of 198 pass) doesn't show it. | `stisim:v4port/framework/first_step_update_results.py`: `AttributeError: STITreatment object has no attribute "outcomes"` (`base_interventions.py:1002`); all fine on 3.7.x | The STIsim fix is three lines, and code is expected to change for v4; what matters is that the cause is findable. Either record index 0 from `init_results()` (X13's single convention), or keep the call and put "`update_results()` now runs at `ti=0`, before the first `step()`" in the migration guide with the error it gives | About 5 lines in `loop.py` (`_before_first_step`), or a line of docs. Related to D22, X13 |
| ST9 | **A coded module that reads a declared module's parameter gets an `ss.par` expression, not a number.** STIsim's `net_beta()` reads `disease.pars.eff_condom`, so a declared disease on STIsim's network crashes. A bare number given as a keyword becomes an `ss.par` too. Found independently by four of the five agents. | `stisim:v4port/networks/eff_condom_par.py`: `TypeError: float() argument must be a string or a real number, not 'BinOp'` (in `Agents.infect()`) | `ss.par.__float__` (L30's proposal), or hand coded modules resolved values at run time | Small. Related to L30, L22 |
| ST10 | **A `ss.Sim` subclass gets no `sim.new()`, and `new()` drops the subclass.** `sti.Sim` (and `hivsim.Sim`) builds its modules in `init()` and calls `ss.Sim.__init__(pars=None)`, so there are no inputs to rebuild from, even with a declared disease, and `sim.given` is empty (`describe()` says the given population is "[guessed]"). `Sim.new()` rebuilds with `Sim(inputs)`, so a subclass loses its class, `plot()` and extra pars on every `new()`, `ss.parallel()` copy and `fit.sim`. | `stisim:v4port/framework/sti_sim_v4.py`: `sti.Sim` refuses `new()`; a 25-line version that builds modules in its constructor works, and then sweeps, `diff()` and `calibrate()` work on coded HIV plus a declared STI; `type(sim.new())` is `starsim.sim.Sim` | `new()` builds `type(self)`; record inputs at `init()` as well; fall back to a copy of the uninitialized sim (R13's proposal) | About 10 lines. Related to R13 |
| ST11 | **Coded code can't move people in a declared disease.** There is no public method; only `ss.treat` can, through the engine's internal `force`. STIsim's syndromic chooser and partner notification have to hand UIDs to a delivery through `deliver=`. | `stisim:v4port/framework/syndromic_bridge.py` (works that way: deliveries equal assignments, 835/835) | `disease.move(uids, to)` and `disease.cure(uids)` on declared diseases as agents | About 20 lines. Related to D23 (the other direction) |

The rest of the seam is the planned bridge, and STIsim confirms each piece:

| Item | STIsim evidence | Cross-reference |
|---|---|---|
| v4 verbs do nothing on coded modules (`ss.modify`, `ss.screen`/`ss.treat`, `calibrate()`, `new()`, sweeps, `diff()`, `what_modifies()`, `to_code()`; `describe()` two lines; `solver='ode'` ignored). `ss.modify('hiv.pars.beta_m2f', ...)` by name crashes with `ValueError: min() iterable argument is empty` | `stisim:v4port/framework/v4_features.py`, `stisim:v4port/hiv/v4_verbs_on_coded_hiv.py` | Duplicate of D23, R13, X2; the crash is related to N21 |
| `sti.HIV(beta_m2f=ss.par(...))` crashes in `validate_beta` (`Cannot use the expression "par(...)" where Python needs a single True or False`) | `stisim:v4port/framework/calibrate_compare.py` | Duplicate of R13 |
| `where=hiv.cd4 < 500`, `where=hiv.infected`, `where=gud.symptomatic` raise `AttributeError` naming the wrong module | `stisim:v4port/hiv/connectors_mixed_sim.py`, `stisim:v4port/syphilis/compare_gud.py` | Duplicate of N19 (two more instances) |
| `ss.probpermonth(ss.par(...))` raises `TypeError`; an `ss.par` inside a per-network `beta` dict gives `E1301`. STIsim's networks need the dict (the maternal network has its own beta), so its main calibration target can't be fitted | `stisim:v4port/hiv/calibrate_zimbabwe.py` | Duplicate of N20 |
| Connectors holding the module objects they were given go stale when `ss.Sim` copies its inputs: silently a no-op in 3.7.2 (a STIsim bug), and in v4 `TypeError: object of type 'Cond' has no len()` from an uninitialized `BoolArr`'s lazy `|` | `stisim:v4port/lead/connector_refs.py` | Related to L24 (the message); `ss.modify` can't go stale, a point for it |

One more seam is new and design-level: **continuous per-person state** (CD4, its nadir, time on ART) has no declared form. As stages, CD4 on ART can't depend on CD4 at the start of treatment, which gives 25 to 50% fewer HIV deaths in the scale-up years and a third as many people with CD4 < 50 (`stisim:v4port/hiv/compare_declared_vs_v3.py`). A nadir dimension recovers most of it, at the cost of doubling the treated states. Keeping HIV coded is the right answer only if the bridge exists. Otherwise the alternative is a declared per-person variable with an update rule (agents only), which odin's seat also asked for (related to the Starsim 3 seat, `switching/starsim3.md`, no ID).

### 3. Add the STI vocabulary

Each item fits a mechanism v4 already has.

**Sex as a real dimension (D15, duplicate, and the top item for STIs).** As agents, a `sex` dimension is drawn independently of `people.female`: agreement is 0.496, and results by sex reported 40 infected women against 86. So every sex-specific duration has to be a pair of `where=ppl.female`/`ppl.male` transitions, which compartmental runs then refuse (ST24). Fixing D15 and allowing per-group distribution parameters (`dur=ss.lognorm_ex(mean=ss.by(sex=[...]), std=ss.by(sex=[...]))`) gives one text for every solver and right results by sex.

**Transmission on sexual networks.**
- **Per-act `beta`, per direction (ST2, ST3, above).**
- **ST17. Route-specific infectiousness.** Syphilis infectiousness is weighted by stage for sex but not for mother-to-child. `rel_trans` applies to every route, and a per-route `beta` dict can't multiply a weighted sum of states (`TypeError: unsupported operand type(s) for *: 'dict' and 'float'`, `stisim:v4port/syphilis/probe_small_errors.py`). The workaround is one dict per weight, which `describe()` prints as three dicts of internal rates. Proposal: source weights on the transmission (`source=dict(P=1, EL=0.55)`) that routes can override, about 50 lines with D21's fix (related to D21).
- **ST21. Condoms.** No v4 network has them. STIsim uses condom use by partnership type and year with an efficacy that is a disease parameter. The only v4 lever is `ss.modify` of `n_contacts`, one factor per network. Proposal: a `condoms` edge attribute (by layer or by a property of both partners, a time series) used in `SexualNetwork.net_beta()`, with the efficacy on the disease. About 40 lines.
- **N15 (duplicate), STIsim's numbers.** `PartnerNet`/`MFNet` can't express StructuredSexual. A best-effort port gives 33.7% (one `MFNet`) or 14.7% (two layers) prevalence against 8.8%, with no risk gradient, an age gap of 0.1 ± 22 years against 7.6 ± 3.2, and concurrency 8%/8% against 1.8%/13.9% (`stisim:v4port/networks/partnernet_port.py`). Beyond the EpiModel seat's list it needs:
  - degree by sex as well as by group;
  - an age-preference term (preferred partner age by age and risk group);
  - durations by partnership type under one concurrency cap per person;
  - a participant subset (MSM);
  - ST22, a two-mode network between pre-selected groups (sex work).

  About 600 lines in all. The joyful path doesn't need this first: with ST9 and the bridge, the coded network's parameters are fittable and modifiable, and it is 1.5 to 1.9 times faster under v4 already. A declared `ss.SexualNet` (StructuredSexual promoted into Starsim) can come later.
- **Network conditions** ("is a sex worker", "has debuted", "is the mother of a fetus on ART") have no declared form, so FSW testing, traditional circumcision before debut, PMTCT and PrEP eligibility are functions (related to `deferred.md`, network conditions).

**The care cascade.**

*The shape of a care cascade.* The spec's form nests the next step inside the last, as in `ss.screen(test, on_positive=ss.treat(drug))`. Nothing about a cascade says why the screen should wrap the treatment rather than the other way around. And STIsim's four real cascades are not trees:

| Cascade | Steps | What nesting can't say |
|---|---|---|
| Syndromic management (`SymptomaticTesting`, `SyndromicManagement`) | Care seeking for any of NG, CT, TV → assessment by sex and syndrome → one of four treatment bundles, each treating several diseases, with no test | The trigger is the union of three diseases' events (ST15); one decision feeds several treatments; nothing is "positive" |
| Etiological testing | Care seeking or screening → a test per disease, sometimes with a confirmatory test → result after a delay, with loss before it → treat positives → notify partners → retest at 3 months | Two paths (syndromic and etiological) lead to the same treatment; the retest is triggered by a treatment long past |
| ANC (`ANCTest`, `SyphTest`, `NewbornTreatment`) | Conception → a first visit after a delay, if attended → a dual HIV and syphilis test → for syphilis: treat the mother, treat the fetus, notify partners, and test and treat the newborn at delivery; for HIV: ART and infant testing at 6 weeks | One visit runs several tests; one result triggers four actions at different times, the last one months later at delivery, from a remembered result (ST14) |
| HIV (`HIVTest`, `ART`) | Three testing programs → diagnosed → linkage after a delay → ART to a stock target → dropout → re-engagement | A stock target with a priority (ST12); the steps are states people stay in, which the EpiModel vignette already writes as a care dimension with transfers |

So a cascade is steps that people flow through, each with its own trigger, coverage, delay and capacity (where the losses are), and each producing events and remembered states that any later step can use. Three things follow:

- **The steps should be peers linked by name, not nested.** `ss.screen(test)` is "deliver a test" in the same way that `ss.vaccinate(vaccine)` is "deliver a vaccine", so it is right for it to wrap the test. It should not wrap what comes after.
- **A step's results should be events and states** (`test.positive`, `treat`, `visit`) that later steps take as `on=` or `eligible=`, by name or as objects. Several triggers, several follow-ups, and long gaps then need no extra mechanism.
- **A grouping container** (`ss.cascade(visit=..., test=..., treat=...)`, along the lines of an `ss.intervention(test=, screen=, treat=)`) would give the steps their names, a scope for referring to each other by string, and one set of cascade results (reached, tested, positive, treated, lost at each step, by stratum). The steps would also work on their own.

That is the form in the sketch above. It would replace `on_positive=`/`on_negative=` and absorb ST14 to ST16 and D27's naming problem. This needs a design pass against these four cascades before anything is built. The HIV care dimension shows that some cascades are better as states than as deliveries, and the design should say when to use which.
- **ST12. A stock coverage target.** STIsim's ART, VMMC and PrEP all take data as "keep N people (or X% of the eligible) on it". `coverage=` is per round and `capacity=` is a flow, so a target with a priority order is two `deliver=` functions. That makes the model agents-only, and the functions' state leaked from one calibration trial into the next (`ValueError: operands could not be broadcast together`). The priority rule matters: allocating by stage alone halved 2010 deaths. Proposal: `target=` (a number, a fraction of the eligible, a time series, or a table by age and sex) with `priority=` (an expression to rank by) on `ss.treat`/`ss.vaccinate`/`ss.transfer`; compartmental when the priority is by group. 100 to 150 lines, replacing about 500 lines of STIsim's coverage machinery. Atomica's programs are the nearest item in the review.
- **ST13. Pregnancy events.** The spec's ANC example uses `on=anc.visit`, but no module provides it, and `ss.Pregnancy` exposes arrays, not events. ANC timing, "at delivery" and "newborns of positive mothers" are functions. Proposal: `on=preg.conception`, `on=preg.delivery`, and `delay=` after an event. About 80 lines.
- **ST14. A remembered test result.** `Delivery.result` holds a result only while a delayed one is pending. In a 10-year run with ANC testing, `anc.result` was false for everyone at every step (`stisim:v4port/syphilis/probe_newborn_link.py`). Proposal: per-person `tested`, `positive` and `ti_positive` on `ss.screen`, usable by later steps as states and events (see "The shape of a care cascade"). About 30 lines.
- **ST15. `on=` takes one transition.** `ng.careseek | ct.careseek` raises "A set of sources needs parentheses", which doesn't apply. Care seeking by sex, or for either of two diseases (syndromic management), needs one delivery each. Proposal: `on=[...]`, or `|` of transitions as the union of their events. About 10 lines.
- **ST16. A delivery outcome as an event.** Resistance acquired at treatment failure can't be declared without `on=tx.failed`. Small to medium.
- **Delivery results (related to D29).** STIsim reports true and false positives and negatives per disease and treatment success, failure and overtreatment, all by sex. Deliveries report `delivered`, `eligible`, `coverage`, `cost` and `diagnosed` (`stisim:v4port/framework/test_and_treat.py`: 14 v3 series against 3). Proposal: accuracy results on `ss.screen` and cured, not cured and not infected on a curing `ss.treat`, by stratum. About 40 lines.
- **ST5, cures with more than one source or destination (above).**

**Natural history.**
- **ST18. A dwell until an external event.** "Until delivery" can't be written as a dwell: `dur=fn` is the mean of an exponential (per-person targets came out with SD 4 days), and `ss.constant(fn)` is refused. The workaround, `rate=ss.perday(1e6), where=...`, is exact as agents but disastrous as an ODE (ST26). Proposal: an exact per-person exit time from a function, or `after=preg.delivery`; the EMULSION seat's `after=`/`before=` is the same mechanism. About 50 lines.
- **ST19. A branch to "stay".** "5% go on to fatal tertiary after a delay, the rest stay" raises `E205 ... sum to 0.05, not 1`, so the state has to be split. A generated split would do it. Minor.
- **ST20. The automatic `prevalence` isn't stratified.** Its `.select()` raises `ValueError: The result "prevalence" is not stratified`, though `09-results.md` shows `res.prevalence.select(risk='high')`. Small.
- **Per-route flows (related to R17).** Sexual against vertical infections, with `infection_log` empty for declared agents.

### 4. Make compartmental runs fit STI models, and fast

A declared STI should run as an ODE for fitting and fast sweeps, with agents for the network. Today:

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| ST24 | **A state may have only one non-exponential exit in a compartmental run**, even when the exits are disjoint by `where=` (durations by sex) or compete as STIsim's clocks do (care seeking against clearance; BV returning to CST 1 or 3). So a sex-specific model needs one text as agents and another as an ODE. | `stisim:v4port/seis_bv/seis_ode.py`: `E804 ... two non-exponential dwell times out of "A" has no compartmental form yet` | Stage disjoint exits separately (each group has one); per-group distribution parameters with D15. Medium, around `init_dwells` in `solvers.py` |
| ST25 | `ss.constant(0)` (zero latency, as v3's gonorrhea and trichomoniasis) crashes compartmental runs with a bare `ZeroDivisionError` in `flow_base`; agents are fine | `stisim:v4port/seis_bv/zero_dwell.py` | An immediate move, or a clear error. A few lines |
| ST26 | **Substeps follow the fastest rate, silently.** BV's 1-to-2-week uniform dwell becomes 27 Erlang stages and about 300 substeps per monthly step: 10 s per simulated year, against 0.12 s as agents. With `ss.perday(1e6)` for "as soon as" (recommended in `skills/import-model/references/idioms.md`), one 15-year run didn't finish in 32 minutes. | `stisim:v4port/seis_bv/bv_ode_speed.py`, `bv_ode_profile.py` (kernel 1.60 of 1.82 s for 3 steps) | Print the substep count when it exceeds, say, 10 per step, with the transition responsible; cap Erlang stages by the step (a stage shorter than a substep buys nothing); an "immediately" transition that is a move, not a rate; fix the idiom page. Related to X20 |
| ST27 | **`calibrate()` on a 0.6 s ODE took 1,063 s for one parameter** (L-BFGS-B with JAX gradients, 126 compartments). Not diagnosed: the JAX compile, the scan, or the substeps. | `stisim:v4port/seis_bv/seis_calibrate.py` | Profile; probably the same substep count inside the JAX density (`stats_review/accuracy` A1 is the related clamp). Related to X26 |

Also:

- **Erlang stage counts (related to X20).** The syphilis ODE is 273 compartments and takes 5 s for 30 monthly years, against 0.09 s with exponential dwells and 7.2 s for 20,000 agents (`stisim:v4port/syphilis/probe_ode_speed.py`). Narrow dwells (6 ± 1 weeks; uniform 22 to 24 months) become 25 to 50 stages each. Capping stages at what the step can resolve would recover most of the 50 times.
- **Function-valued branch probabilities (related to X10)** slow the ODE 1.7 times and the binomial 3.4 times compared with `ss.by(sex=...)`.
- **Sexual networks under an ODE (D10, duplicate).** `E804` for StructuredSexual and `MFNet`. Built by hand (`stisim:v4port/networks/riskgroup_ode.py`), the partner-change form that `06-routes.md` specifies (new partnerships times the chance of transmission over the whole partnership) gets 8.2% against agents' 10.4%, with the right risk ordering. The "mean current contacts" form gives 54% with the gradient reversed, because most low-risk partnerships are long and monogamous. This is evidence that D10's proposal is the right one, and that what `PartnerNet` does under an ODE today would be badly wrong for STIs.
- **ST23. A contact matrix over two dimensions.** `ss.contacts(sex=..., risk=...)` multiplies one matrix per dimension (`np.kron`), which can't say "men mix with women, and risk mixing differs by sex"; `ss.MixingPools` by group is `E804` under an ODE. Proposal: `ss.contacts(matrix, dims=['sex', 'risk'])`, applied by both engines, with `sex` mapped to `people.female`. About 60 lines (related to D32, D15).
- **L1 (duplicate):** `dt=ss.months(1)` without `unit=` read the hand-built matrix per month (prevalence 1.000 against 0.149).

### 5. Before removing the v3 vocabulary (decision 2)

The decision is to remove, not duplicate, the v3 vocabulary. These are STIsim's dependencies on it, and what v4 needs first so that STIsim loses nothing:

| v3 class STIsim uses | Where | What v4 needs first |
|---|---|---|
| `ss.Calibration` (`sti.Calibration`, 800 lines) | Every STIsim calibration | `calibrate()` with: multi-worker Optuna and resume from a database (`continue_db`); a `check_fn` that rejects runs (die-outs) as infinite; keeping each trial's results (`save_results`); top-N parameter sets as an ensemble (`get_pars(n)`, `make_calib_sims()`); STIsim's goodness-of-fit options (normalized, fractional, squared, summed or mean). Plus R13 and N20, so coded parameters and parameters inside `beta` dicts can be fitted. The Optuna path is one process today (`calibration.py`), and 20 trials of a 2,000-agent STIsim model took 549 s |
| `ss.Product` (`STIDx`, `HIVDx`, `ProductMix`) | Test products with positive probability by state, and a mix of products by year | `ss.Test(sensitivity=dict(...))` covers `STIDx` and `HIVDx` (prototyped, by sex through a function). `ProductMix` (which product, by year) has no form beyond several `ss.screen`s with time-series coverage |
| `ss.Births`, `ss.Deaths`, `ss.Pregnancy`, `ss.MaternalNet`, `ss.BreastfeedingNet`, `ss.PrenatalNet`, `ss.PostnatalNet` | All demography | Keep them, with ST1 fixed and ST13 added |
| `ss.Infection`, `ss.Intervention`, `ss.Connector`, `ss.Analyzer`, `ss.SexualNetwork`, `ss.DynamicNetwork`, `ss.FloatArr`, `ss.BoolState`, `ss.Result` (189 uses) | The coded-module API | Keep first-class (the Starsim 3 seat's line), with the seam in section 2 |

Also: STIsim's `ss.ANCTest` reference (in its AI skills) and its `ss.Route` check (`syphilis.py:261`) should be re-checked when the vocabulary moves.

### 6. Small things

| Finding | Evidence | Cross-reference |
|---|---|---|
| Unchanged code: results move one step; four HIV tests and the baseline fail; `RelationshipDurations.mean_duration[0]` and `partner_age_diff.age_diff_mean[0]` are NaN, so their sums and means are NaN; the network has no edges at `ti=0` | `stisim:v4port/lead/hiv_test_failures.py`, `stisim:v4port/framework/analyzers_check_v4.log` | Duplicate of N10, D22 |
| An analyzer filtering `ti_infected < self.ti` lets just-infected agents through; deliveries and declared transitions see `ti` one lower than coded modules (`ti_Sx` against `ti_pregnant` needs `- 1`) | `stisim:v4port/syphilis/compare_care.py` | Duplicate of X13 |
| A callable `capacity=` returning the stock left is read per unit time, so a stock-limited treatment is rationed instead of stocking out | `stisim:v4port/framework/stockout.py` | Duplicate of D20 |
| Products defined inline become modules `test`, `test_1`, `treatment_1` with warnings; a lambda sensitivity prints as `<function <lambda>>` | `stisim:v4port/framework/test_and_treat.py` | Duplicate of D27 |
| `ss.modify` with an amount function takes `fn(sim)` only; `fn(sim, uids)` raises | `stisim:v4port/syphilis/probe_small_errors.py` | Related to N21 |
| A state not connected to the first one becomes a count, and its branch `p=` is ignored | `stisim:v4port/seis_bv/branch_remainder.py` | Duplicate of L4 |
| A fit on agents landed on its lower bound without saying so | `stisim:v4port/hiv/calib_check.py` | Duplicate of R5 |

## Speed and inefficiencies

| What | Measured | Verdict |
|---|---|---|
| Unchanged STIsim | 1.6x faster (Zimbabwe 18.8 s against 29.7 s, same interpreter); five diseases 14 s against 22 s | Good. From v4's core: Starsim 3's per-call `Dist.jump()` (2.3 s self time at 20,000 agents) is gone, and array indexing is twice as fast |
| StructuredSexual's step | 1.5 to 1.9x faster; most of its time is per-step overhead (about 30 distribution and boolean calls), not size (5.6 s at 5,000 agents, 7.5 s at 20,000 under v3) | Good |
| Declared diseases as agents against v3 code | SEIS 3 to 13% slower; HIV 9 to 13% faster; syphilis natural history 13 to 31% faster (fewer results); BV twice as fast; the syphilis care model level (its deliveries step from Python) | Doesn't reproduce N29's 1.4 to 1.6x penalty; the penalty seems specific to small library models |
| ODE with narrow dwells | Syphilis 5 s (0.09 s with exponential dwells); BV 10 s per simulated year (ST26); `perday(1e6)` hangs | Fix (ST26, X20) |
| `calibrate()` | 1,063 s on a 0.6 s ODE (ST27); Optuna single-process, 549 s for 20 trials at 2,000 agents | Fix ST27; add workers (section 5) |

## Keep

STIsim confirms the review's keep list:
- the product/delivery split, with `trace=` (partner notification, and fetal treatment along `maternalnet`), `capacity=` and `cost=` (but not `on_positive=` nesting; see "The shape of a care cascade");
- `ss.modify` with a kept base (it fixes three v3 overwrite bugs);
- strain dimensions (gonorrhea resistance);
- serology as dimensions moved by transfers;
- every transition as a flow;
- `describe()`;
- named data (`ss.people('Zimbabwe')` without a UN token);
- the unit warning on bare numbers (it caught STIsim's `beta_m2c` in a yearly sim).

One more for the list: **a mixed sim** (coded HIV beside one declared STI) **unlocks `ss.parallel(...).diff()`, sweeps and `calibrate()` for the whole sim**. `calibrate()` recovered a declared STI's duration, 1.04 against 1.0, in 20 trials with no `build_fn`. That is the incremental path a package like STIsim will take, and ST10 is the only thing in its way.

## Suggested order

1. **A day:** ST9 and ST10, plus ST8 (the fix or the migration note). A mixed STIsim sim then gets the run tools as soon as one disease is declared.
2. **About a week:** ST1 to ST7, each with a regression test. Add to P10's contract tests: "states sum to `n_alive` under every demography module", "a dwell entered by transmission equals one entered by progression", and "a per-act probability is invariant to `dt`".
3. **The planned bridge** (D23, R13, N19, N20), with ST11.
4. **D15 and per-group distribution parameters**, then ST24, so a sex-specific STI is one text under every solver.
5. **The care-cascade vocabulary:** first a design pass on the cascade's shape, against STIsim's four cascades; then ST12 to ST16, ST5's destinations, and the delivery results within it.
6. **`calibrate()` parity with `ss.Calibration`** (section 5), before the v3 vocabulary goes.
7. **Compartmental speed:** ST26, ST27, stage caps (X20).
8. **Networks:** condoms (ST21), the partner-change ODE (D10, ST23), then the `PartnerNet`/`SexualNet` additions (N15, ST22).

## Index of new findings

| ID | Finding | Kind |
|---|---|---|
| ST1 | Births, pregnancy and migration add people who are in no state of a declared disease | Silent wrong answer |
| ST2 | No spelling for a per-act probability | Silent wrong answer |
| ST3 | Per-direction `beta` applied after the acts | Silent wrong answer |
| ST4 | States entered by transmission last one step too long | Silent wrong answer |
| ST5 | A cure moves only the sources of one transition, to one destination | Silent wrong answer |
| ST6 | `where=` with `p=` never fires | Silent wrong answer |
| ST7 | A function-valued `p=` doesn't leave the remainder | Silent wrong answer |
| ST8 | `update_results()` before the first `step()` crashes coded modules | Crash |
| ST9 | Coded modules get `ss.par` expressions from declared modules' pars | Crash |
| ST10 | `ss.Sim` subclasses get no `new()`, and `new()` drops the subclass | Dead end |
| ST11 | Coded code can't move people in a declared disease | Missing |
| ST12 | No stock coverage target with a priority | Missing |
| ST13 | No pregnancy events | Missing |
| ST14 | Test results aren't remembered per person | Missing |
| ST15 | `on=` takes one transition; `\|` of transitions gives a misleading error | Missing |
| ST16 | A delivery outcome isn't an event | Missing |
| ST17 | No route-specific infectiousness; a `beta` dict can't multiply a weighted sum | Missing |
| ST18 | No exact dwell until an external event | Missing |
| ST19 | No branch to "stay" | Minor |
| ST20 | Automatic `prevalence` isn't stratified | Spec drift |
| ST21 | No condoms on networks | Missing |
| ST22 | No two-mode network | Missing |
| ST23 | No contact matrix over two dimensions | Missing |
| ST24 | One non-exponential exit per state in compartmental runs, even when disjoint | Missing |
| ST25 | `ss.constant(0)` crashes compartmental runs | Crash |
| ST26 | Substeps follow the fastest rate silently; the `perday(1e6)` idiom hangs an ODE | Performance |
| ST27 | `calibrate()` takes 1,063 s on a 0.6 s ODE | Performance |
