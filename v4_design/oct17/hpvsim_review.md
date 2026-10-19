# HPVsim: what v4 should change to make the port a joy

Written 2026-10-18, at commit `03741ba2` on the `v4` branch of `ssv4`, the same commit as [stisim_review.md](stisim_review.md). The `ssv4` working tree was being edited during the work, so everything ran on a clean worktree of `03741ba2`. HPVsim 3.2.1 (`9037e7b2`, the `v4` branch of `hpvsim`) was used as a real model to test v4. Five agents each took one part of HPVsim:
- the natural history of one genotype;
- several genotypes and their immunity;
- interventions and products;
- population (network, demography, multiscale agents, HIV);
- the framework (sim, results, analyzers, calibration, the unchanged test suite and speed).

Each agent ran its part unchanged under v4, prototyped a declared version, and compared the two over several seeds against Starsim 3.7.2 on the same interpreter. Scripts and outputs are in the `hpvsim` repo under `v4port/<part>/`, with HPVsim's own bugs in `v4port/README.md`. I reran the main new findings myself.

This file follows the STIsim review's shape, and is about v4 rather than HPVsim. New findings get IDs HP1–HP36. Everything else is cross-referenced to the STIsim review (ST), the merged findings (L, D, X, R, P), and [new-findings.md](new-findings.md) (N). Duplicates get one line each.

## In brief

**What works:**
- **Unchanged HPVsim runs, and is faster.** 359 of 368 tests pass, and none of the 9 failures is an engine error. It runs 1.3 to 2.4 times faster:
  - 4.0 s against 9.6 s for one genotype at 10,000 agents;
  - 29.9 s against 38.2 s for the four-genotype demo, best runs on a loaded machine.
- **Distributions sample bit for bit the same**, and so do the network's statistics.
- **One timing change moves the results** (HP5): seeded infections transmit for one step fewer. That makes cancers 11% lower over 1990–2020. Shifting the seeds by one step brings every output back within noise.
- **The declared versions are 3 to 35 times smaller and agree with v3 where they model the same thing:**

  | Part | Declared | v3 | Agreement |
  |---|---|---|---|
  | hpv16 natural history | about 105 lines | 446 | Cohort: P(CIN) 0.259 against 0.256; median time to cancer 23.3 against 23.4 years; age at cancer 54.0 against 54.0 |
  | Genotype machinery (cross-immunity, own immunity, seeding, cancer per person, any-genotype results) | 95 | 402 | Any-HPV prevalence 0.106 against 0.106 at 4 genotypes; cross-immunity effect 0.42 against 0.43 |
  | Products | 29 (one genotype) | 720, plus 317 in Starsim 3 | Yields from the product tables to 1%: 0.138 against 0.139 ablated per screened |
  | WHO screening algorithms 1 and 3, and vaccination | 17 | 55 | Same ranking of strategies |
  | The `by_age` analyzer | `stratify=` | 133 | Free for declared modules |

- **They gain what HPVsim never had:**
  - an ODE and chain-binomial of the natural history (0.43 s against 0.88 s as agents);
  - `describe()` and `what_modifies()`, which list every cross-immunity term with its condition;
  - flows for every transition;
  - `diff()` scenarios;
  - `calibrate()` to cancers by age with no analyzer setup or `build_fn`.
- **Two HPVsim bugs can't happen in v4,** since follow-ups are structural and modifiers keep their base. HPVsim keeps re-treating the last screening round after screening stops (811 extra ablations). Its cross-immunity depends on module order, through a reset-then-multiply protocol on `rel_sus`.

**What stops the port being a joy, in five groups:**

1. **HPV's natural history has a shape the spec doesn't anticipate** (HP18). HPVsim draws a duration, then an outcome whose probability depends on it, twice in a chain. The spec's branching decides first and then waits, and runs silently with cancer 22 years too early. The exact form works only because one spelling happens to decide at exit (HP1), which contradicts the spec. HPV also needs per-person values that are updated on events and frozen at entry (HP19): immunity as a running maximum, `rel_sev`, and age at CIN onset.
2. **Seven new silent wrong answers in the agent engine and effects** (HP1–HP7):
   - two spellings of a branch decide at different times;
   - a junction holds people for a step;
   - a beta pair runs reversed on a network with women in `p1`;
   - `people.scale` is ignored by every result;
   - unchanged v3 seeds transmit one step fewer;
   - an efficacy on a branching exit lengthens infection instead of cutting progression;
   - a bivalent vaccine protects against one genotype only.

   Each is 5 to 30 lines to fix. The STIsim set (ST1–ST4) all apply, with larger effects, because HPV infections are short and HPVsim makes about 14 acts per partnership per quarter. ST4 alone raises prevalence 1.6 to 1.8 times.
3. **Genotypes are a family of parallel diseases, and v4 has no vocabulary for a family.** A strain dimension can't hold co-infection (each genotype progresses on its own), so each genotype is its own disease. Three things then have no declared form:
   - immunity across the family: 88 modifiers at 4 genotypes, with the wrong combination rule (HP22);
   - joint seeding (HP15);
   - "infected with any genotype" (HP23).

   Products have the same gap: every HPVsim product acts on all genotypes at once (HP7, HP24).
4. **The screening vocabulary is two-valued.** HPVsim's tests have 3 to 5 results by state with a hierarchy across genotypes (HP24). Its treatments cure by state (HP25). It needs `delay=` (HP26) and a queue (D31). The STIsim cascade findings all apply (ST14, ST16, the cascade shape).
5. **Population and multiscale.** Results ignore agent weights (HP4), so a declared HPV can't use multiscale. `ss.MFNet` has four silent problems in an open population (HP8–HP11), and HPVsim's network has no declared form (N15). Once one module is declared, `hpv.Sim` loses every run tool (HP12).

**Counts:** 36 new findings (HP1–HP36), 25 of them related to an existing finding, plus about 25 known findings that HPVsim hits again, most with larger effects than before. Groups 2 and 5 are mostly small fixes: about 2 weeks of v4 work besides the planned bridge and the STIsim fixes, which HPV shares. Groups 1 and 3 are design work: a week or two each once the design is settled. Group 4 belongs in the cascade design pass the STIsim review already asks for.

## What a joyful HPVsim looks like

The target, in proposed spellings (the names marked `# proposed` don't exist today). The network stays coded. Multiscale stays coded, or becomes a resolution option on one transition. Everything else is declared.

```python
import starsim as ss, hpvsim as hpv

S, P, C, X, R, L = ss.states('S P C X R L')
def hpv_genotype(g):
    gp = hpv.get_genotype_pars(g)
    return ss.Disease(
        infection = ss.transmission('S -> P', beta=ss.peract(m2f=gp.beta*gp.transm2f, f2m=gp.beta)),  # proposed: per act, named directions (ST2, ST3, HP3)
        precin    = ss.progression('P -> [C, R]', dur=gp.dur_precin,
                                   p=ss.given_dwell(lambda d: hpv.compute_severity(d, pars=gp.cin_fn))),  # proposed: outcome given the dwell (HP18)
        cin       = ss.progression('C -> [X, R]', dur=gp.dur_cin, p=ss.given_dwell(...)),
        name      = g,
    )

genotypes = ss.Diseases(*[hpv_genotype(g) for g in ['hpv16', 'hpv18', 'hi5', 'ohr']],          # proposed: a family of parallel diseases (section 3)
                        immunity = ss.cross_immunity(sus=matrix_sus, sev=matrix_sev,          # proposed: one declaration, not 88 modifiers (HP22)
                                                     value=ss.value(on='clear', update='max', draw=ss.uniform(0.5, 0.95))),  # proposed: per-person value on events (HP19)
                        cancer   = 'once',                                                     # one cancer per person across the family
                        seed     = ss.by(age=..., sex=...))                                    # joint, exclusive seeding (HP15)
genotypes.stratify(age=hpv.WHO_EDGES)

cyto  = ss.Test(results=dict(abnormal=dict(C=0.9, X=1), ascus=dict(S=0.1, P=0.1, C=0.1)), default='normal')  # proposed: categorical results (HP24)
ablate = ss.Treatment(cure=dict(P=0.936, C=0.936))                                          # proposed: cure by state, every genotype (HP25)
vx    = ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975))                               # proposed: one product, several diseases (HP7)
care  = ss.cascade(
    screen = ss.screen(cyto, sex='f', age=[25, 50], interval=ss.years(3)),                     # proposed: eligibility by keyword
    triage = ss.screen(hpv_test, on='screen.ascus'),
    treat  = ss.treat(ablate, on=['screen.abnormal', 'triage.positive'], delay=ss.months(3), capacity=..., queue=True),  # proposed (HP26, D31)
)
sim = hpv.Sim(location='nigeria', diseases=genotypes, interventions=[ss.vaccinate(vx, at_age=9), care], data='nigeria_cancers.csv')
sim.describe()
fit = sim.calibrate(workers=8)
ss.parallel(baseline=fit.sim, screening=fit.sim.new(interventions=care), n_runs=50).diff('baseline').plot()
fit.sim.results.cancers.rate(per=1e5, of='women', standard='who2000')                      # proposed: standardized rates (HP23, D15)
```

Today the pieces of this that run need workarounds:
- a newborns function (ST1);
- per-act betas worked out by hand for one `dt` (ST2);
- one transmission per sex (ST3);
- a step taken off every dwell entered by transmission (ST4);
- a coded connector for per-person immunity;
- 88 modifiers with lazy conditions for cross-immunity;
- a seeding function run after `init()`;
- a 41-line analyzer for any-genotype results;
- chains of binary tests for categorical results, each counted and costed as a test;
- a pseudo-test for a treatment delay;
- function-valued efficacies;
- `hpv.Sim` rebuilt so that `new()` replays its arguments.

## The port today, and after the changes

| Task | Today | After the changes below |
|---|---|---|
| HPVsim unchanged | Runs 1.3 to 2.4 times faster; cancers 11% lower, from seeding (HP5) | Same answers once HP5 is documented or matched |
| Natural history | The spec's branching puts cancer 22 years early (HP18); the exact form relies on an undocumented exit-time draw (HP1) | `p=ss.given_dwell(fn)`, exact as agents and staged under compartments |
| Per-person immunity | A coded connector and function modifiers, agents only, or 15 history states with mean values (HP19) | A declared per-person value with an update rule |
| Four genotypes | One disease each, which is right; 88 modifiers with the wrong rule for vaccine against natural immunity (HP22); a seeding function (HP15); a 41-line any-genotype analyzer (HP23) | One `ss.Diseases` family with cross-immunity, joint seeding and union results |
| HPV on HPVsim's network | Prevalence 1.6 to 1.8 times too high (ST4), men's prevalence a third (ST3), betas 4.3 times too small (ST2), newcomers in no state (ST1) | Right |
| Bivalent vaccine | Silently protects against hpv16 only (HP7) | One product over several diseases |
| Therapeutic vaccine ("halve progression to CIN") | Lengthens infection instead (HP6), and needs a list of follow-ups (ST16) | Efficacy on a named branch |
| Cytology, colposcopy, treatment assignment | Chains of binary tests, each reported and costed (35,397 tests for 20,000 cytologies) (HP24) | Categorical results with a hierarchy across genotypes |
| Ablation, excision | A function-valued efficacy and a rate-0 cure transition per state (HP25) | `cure=dict(P=..., C=...)` |
| `treat_delay`, `treat_num` | A pseudo-test (HP26); no queue (D31) | `delay=` and `queue=True` on every delivery |
| Multiscale | Every v4 result ignores `people.scale` (HP4) | Weighted results, and `scale_flows()` kept |
| A declared disease in `hpv.Sim` | Every run tool fails in `AgeMigration` (HP12) | `new()` replays the subclass's call |
| Cancer incidence per 100,000 women, age-standardized | No denominator by sex (D15), no flow by destination (HP20), no standard weights (HP23) | One result |

## What should change in v4

### 1. The natural-history shape: an outcome that depends on the time spent

HPVsim's progression is "draw how long the infection lasts, then decide from that duration whether it became CIN; draw how long CIN lasts, then decide from that whether it became cancer". Long infections progress; short ones clear. The joint distribution of (duration, outcome) is the model. The duration depends on per-person severity (`rel_sev`), on prior immunity (`sev_imm`), and on age at CIN (`age_risk`).

| ID | Finding | HPVsim evidence | Fix | Size |
|---|---|---|---|---|
| HP18 | **v4 cannot declare an outcome that depends on the drawn dwell, and the spec's own form is silently wrong for it.** The spec's branching decides the destination on entry with the marginal probability, then gives every branch the same dwell, so the people who go on to CIN are no longer the long infections. | `hpvsim:v4port/natural_history/compare_cohort.py`, 20,000 women, 5 seeds. Spec form: median time to cancer 3.25 years against 23.4, mean age at cancer 32.0 against 54.0 for women infected at 25. P(cancer) is right (0.0275 against 0.0273), so a fit to totals would not notice. A "split by outcome" form (one branch per outcome, each with the conditional dwell) runs under every solver and is within 5 to 10%. But its probabilities are worked out outside the model, so `cin_fn` and `cancer_fn` can't be fitted | `p=ss.given_dwell(fn)` on a multi-destination transition: exact as agents, decided at exit from the dwell drawn. Under compartments, stage the dwell (phase-type) with a branch share per stage, which generalizes the split form, so one text runs under every solver and the same staging also fixes HP28 | About 100 lines, plus a spec section |
| HP1 | **Two spellings of a branch decide at different times, silently.** `ss.Transition(P, [A, B], dur=, p=[fn, None])` draws the destination at exit (`Agents.finish_moves`); separate transitions with `p=` draw on entry (`Agents.schedule`). The spec says `p` is decided on entry and treats the spellings as one. HPVsim's exact form only works because of the exit-time draw. | `hpvsim:v4port/natural_history/probe_branch_at_exit.py` (reran): women aged 28 with a 4-year dwell and p = 0.9 from age 30 go to A with share 0.905 in one spelling and 0.000 in the other | Pick a rule and document it. For HPV, make "a multi-destination transition with a dwell decides at exit" official, give `fn` the time in state (HP18), and have `describe()` say which | Small |
| HP19 | **Per-person values that are updated on events, or frozen at entry, have no declared form.** HPVsim needs: `rel_sev` (drawn once, shared across genotypes); `nab_imm` and `cell_imm` (drawn on clearance, kept as running maxima, gated by `sero_prob`); and the age at CIN onset, frozen when the dwell is drawn. As history states, immunity takes 15 states per genotype instead of 7 and holds means: own-immunity prevalence 0.133 against 0.101. The per-person form needs a coded connector and function-valued modifiers, which make the model agents-only and draw from numpy outside the common random numbers. Rebuilding the age at entry as `age − (ti − ti_P)·dt` is silently wrong without `use_aging`: cancer at 45 was a third low until it was caught | `hpvsim:v4port/genotypes/genotypes_v4.py` against `genotypes_v4_perperson.py`, which matches HPVsim (any-HPV 0.106 against 0.106); `hpvsim:v4port/natural_history/hpv_declared.py` | A declared per-person value with an update rule on an event (`ss.value(0, on=clear, update='max', draw=ss.uniform(0.5, 0.95), p=sero_prob)`), usable as a modifier amount; `at='entry'` on `ss.modify` to freeze a value when the dwell is drawn. Agents only, or binned under compartments | About 80 lines. Related to the STIsim review's continuous per-person state note (CD4) and odin's auxiliary variable; this is now the third real model to need it |
| HP2 | **An immediate junction (a state left only by `p=` branches with no dwell) holds people for a whole step as agents.** The ODE passes them through at once, so the two differ by one step per junction (0.25 years each in HPVsim) | `hpvsim:v4port/natural_history/probe_junction_lag.py` (reran): entered J at step 4 and A at step 5 as agents; the ODE leaves `n_J` at 0 | In `finish_moves`, resolve the zero-dwell branches of the states people land in, in the same step | About 20 lines. Related to ST4, L14 |
| HP16 | `ss.Transition(J, [A, B], p=[0.5, 0.5])` with no dwell crashes as agents: `TypeError: Cannot evaluate None` | `hpvsim:v4port/natural_history/probe_small.py` 1 | An immediate split | A few lines |
| HP21 | A remainder branch (`p=None`) beside another timed exit from the same state (men's clearance, with `where=`) raises `E205 ... sum to 0.25, not 1` | `probe_small.py` 3 | Take the remainder among the branches, whatever other exits there are (`transitions.py:1476`) | Small. Related to ST7, ST19 |
| HP20 | **A branching flow has no destination axis**, though `03-states-and-transitions.md:309` and `09-results.md` show `.select(to=)`. With CIN → [cancer, clearance] as one transition, new cancers (HPVsim's main output and calibration target) are not a result. `ss.outcome(cin)` is the total too. Found by two agents | `hpvsim:v4port/framework/declared_results.py`: `KeyError: 'The result "cin" has no dimension "to"'` | Keep a destination axis on a branching flow's strata | About 30 lines. Related to R17, R19 |

Also, already known:
- **ST4 and L14:** a seeded cohort's mean dwell is 1.37 years for a 1.125-year lognormal at quarterly steps.
- **L4:** with transmission off (a cohort), every state but `S` silently becomes a count, and `init=dict(P=1.0)` puts 1.00 in P instead of 1,000 people.

### 2. Fix the silent wrong answers in the agent engine and effects

| ID | Finding | HPVsim evidence | Fix | Size |
|---|---|---|---|---|
| HP3 | **A `beta` pair means [p1 → p2, p2 → p1] of the network's edges, not "male to female, then female to male" as the spec says.** `hpv.SexualNetwork` puts women in `p1`, so a pair written to the spec runs reversed on it, and correctly on `ss.MFNet` | `hpvsim:v4port/natural_history/probe_beta_pair_order.py` (reran): `beta=[b, 0]` infects 0 women and 172 men on `hpv.SexualNetwork`, and 92 women and 0 men on `ss.MFNet` | Named directions, resolved from the sexes of `p1` and `p2` (`ss.peract(m2f=, f2m=)`, which ST2 and ST3 already need), with `describe()` printing which is which | About 20 lines. Related to ST3, D15 |
| HP4 | **Every v4 result ignores `people.scale`**, documented as "the scale factor for the agents (multiplied for making results)" (`people.py:74`). That covers `n_alive`, a declared disease's counts and flows, and the library diseases. HPVsim's multiscale (`ms_agent_ratio=100` in its demo) gives fine cancer agents a weight of 1/100. Its coded modules recompute every result by hand with 11 calls to `people.scale_flows()`, but a declared disease has no hook. `findings/execution.md` proposes deleting `People.scale_flows()` as having no callers. Starsim 3 is the same, so this is not a regression; it is what blocks a declared HPV from using multiscale | `hpvsim:v4port/framework/repro_people_scale.py` (reran): half the agents at scale 0.01, `n_I` 4,999 against a weighted 2,510, `n_alive` 10,000 against 5,050. HPVsim's own `n_alive` is 318M for Nigeria in 2020 against 211M weighted (WPP: 212M) | Weight counts and flows by `people.scale` when it isn't uniform (`np.bincount(..., weights=)`), in the agent engine and `Module.update_results()`, stratified results included; keep `scale_flows()` | About 30 lines. New, and reverses part of the `execution.md` proposal |
| HP5 | **Under v4, infections seeded at `init()` by unchanged Starsim 3 code transmit for one step fewer.** v4's first step is at `ti=1`, and clearance at `ti_x <= ti` runs before transmission. So a seed with a dwell of d steps transmits d − 1 times, where v3 gave it d. This makes seeds consistent with later infections, which is defensible. But it is not just the one-step shift of N10: it changes the epidemic. 19% of HPVsim's seeds clear by `ti=1` | `hpvsim:v4port/framework/repro_seed_lost_step.py` (reran): `ss.SIR` with a one-step dwell infects 203 people from its seeds under v3 and 0 under v4. `hpv.demo()` over 8 seeds: infections 0.93 of v3, cancers 0.89, age-standardized incidence 40.6 against 43.7 (z −4 to −6). Seeding one step later (`compare_demo.py shift`) brings every output within noise | Say it in the migration guide and the import skill beside N10 ("models calibrated on v3 move; refit `init_prev`"), or give coded modules' seeds `ti_infected = −1`. Note that a declared seeded cohort stays a step too long (L14), the opposite way | Docs, or about 5 lines. Related to N10, D22 |
| HP6 | **An efficacy on a transition written as one branching exit (`P -> [C, R]` with `p=[...]`) scales the rate of the whole exit, not the chance of the branch.** HPVsim's therapeutic vaccine ("halve progression to CIN") written this way makes infections last longer and barely changes CIN. With separate `P -> C` (`p=`) and `P -> R` transitions, the same key is right; `modifier_kind()` decides by the form | `hpvsim:v4port/interventions/probe_txvx.py` (reran): CIN share 0.179 against a target of 0.10, years in P 3.07 against 1.94; split form 0.095, dwell unchanged | Accept a key naming one destination (`'precin->C'`, or `branch=`) that acts on its probability; on a multi-destination exit, refuse a bare key and name both forms | About 30 lines. Related to the `modifier_kind` rule in `07-effects.md` |
| HP7 | **One `ss.Vaccine` delivered for two diseases protects against only the first.** The product's `disease` is set by its first delivery, and the second delivery's `disease=` is ignored. The two deliveries also share `has_<product>`, so the second skips the first's recipients. This is the natural way to write a bivalent vaccine with genotypes as diseases. `disease=[hpv16, hpv18]` raises `AttributeError: 'list' object has no attribute 'name'`, and `efficacy=dict(hpv16=, hpv18=)` raises `E506` | `hpvsim:v4port/interventions/probe_multigenotype.py` (reran) 1a: 10,110 and 4,903 delivered; the only modifier is `('hpv16', 'rel_sus')`. Nobody is protected against hpv18 | Now: an error when a product bound to one disease is delivered for another (about 5 lines). Then: products over several diseases, with `disease=` taking a list and parameters keyed by disease (section 4) | About 5 lines now, then part of section 4 |

A known silent problem with a new mechanism (related to ST5 and D18): **with no cure transition, a curing `ss.Treatment` guesses a transition limited by `where=` to the other sex.** The only infected-to-cleared exit in HPV that isn't a branch is men's clearance. So ablation cures women's precin through it and cures no CIN (cured P 0.936, C 0.000, against HPVsim's 0.936 and 0.936). `describe()` doesn't name the guess, though `08-interventions.md` says guesses are printed (`hpvsim:v4port/interventions/probe_treatment_by_state.py`). ST5's fix absorbs it; also never guess a `where=`-limited transition.

**The STIsim set applies, with larger effects for HPV:**
- **ST1:** 3,191 of 3,191 people added by `hpv.Births` and `hpv.AgeMigration` are in no state. Without a workaround, 75% of the living are in no state after 30 years, and prevalence falls from 23.6% to 4.5%.
- **ST2:** HPVsim's per-act 0.5 written as a bare `0.5` arrives 4.3 times too small.
- **ST3:** one per-direction pair gives 3.4% prevalence in women against 23.6% with one transmission per sex. HPVsim makes about 14 acts per partnership per quarter, so v3's per-partnership female-to-male probability is 0.985 and v4's 0.41.
- **ST4:** an extra quarter on infections that last about a year raises prevalence 1.6 to 1.8 times. HPVsim given the same extra step goes from 0.101 to 0.152.

ST4 is the largest single gap between a declared HPV and HPVsim.

### 3. A family of parallel diseases: genotypes

A woman can carry hpv16 in CIN and have just acquired hpv18. A strain dimension holds one strain per person and gives the strains one natural history, so it can't say that. Writing co-infection as levels takes 15 levels for 4 genotypes, and one clearance clears all of them. `05-dimensions.md` already says different natural histories mean a second disease. So each genotype is its own declared disease, and that part works: the four genotypes are one loop of 48 lines. What is missing is the vocabulary that ties the family together, which HPVsim built as 400 lines of connectors.

| ID | Finding | Evidence | Fix | Size |
|---|---|---|---|---|
| HP22 | **Cross-immunity on severity takes one modifier per source, target and transition (72 at 4 genotypes, 88 with susceptibility), and v4's combination rule can't express HPVsim's.** HPVsim sums natural immunity across genotypes, clips it, then combines it with the vaccine as an independent path: `(1 − clip(Σ)) × (1 − vax)`. v4's `base × Πmultiply + Σadd` gives 0.2 where HPVsim gives 0.35. Silent: a different model, not an error | `hpvsim:v4port/genotypes/probe_vocab.py` (e); `genotypes_v4.py` (`cross_modifiers`) | A modifier on several transitions at once (all of a state's exits), and `group=` on modifiers, so additive terms within a group are summed and clipped, then multiplied with other groups. Better: one `ss.cross_immunity(sus=, sev=, combine='add')` on a family | About 40 lines. Related to D7 |
| HP15 | **Initial conditions can't be functions, and nothing is joint across diseases.** HPVsim's own idiom `init_prev=ss.bernoulli(p=fn)` (a function of age and sex) crashes with a bare `TypeError: unsupported operand type(s) for *: 'function' and 'float'`, and `init=dict(P=fn)` likewise. `init=ss.by(age=...)` makes age a dimension, so D11 and D12 apply, and it can't depend on sex as agents (D15). HPVsim seeds exclusively across genotypes. Found by three agents | `hpvsim:v4port/genotypes/probe_init_fn.py` (reran), `natural_history/probe_small.py` 6, `population/smoke.py` | Evaluate a function-valued `init_prev`/`init=` per agent (agents), or `E804` naming it; a joint initializer on a family, or a documented `init_post` hook | About 20 lines. Related to N21 |
| HP23 | **"Infected with any genotype", a union of states across diseases, has no form.** A sum double-counts the co-infected (4 to 7% of women). The spec's sim-level `ss.outcome` raises (R17). Age-standardized rates and rates per 100,000 women have no form either. The numerator by age is free, but there is no denominator of women by age (D15) and no standard weights | `hpvsim:v4port/genotypes/probe_vocab.py` (b); the workaround is a 41-line analyzer (`AnyHPV`), and in `framework/declared_results.py` about 10 lines | `ss.any(hpv16.infected, hpv18.infected, ...)` as a result (OR as agents; the crossed model under compartments), and `res.x.rate(per=1e5, of='women', standard='who2000')` | About 60 lines. Related to R17 |
| HP14 | **A strain dimension with `immunity=` refuses to run over a network as agents** (`E804 ... cannot run over networks yet`), and no network runs under an ODE (D10). So the spec's strain-with-cross-immunity form can't be used for any STI. It is listed in `implementation.md` as not yet done, but not in the registers. It matters for single-history strains such as gonorrhea resistance, not for HPV | `hpvsim:v4port/genotypes/probe_strain.py` (reran) | In `Agents.infect()`, thin each new case by `protection()` for the infecting source's strain; the pairs are already known | About 15 lines. Related to D13 |

**The design proposal: a declared family.** `ss.Diseases(hpv16, hpv18, ..., immunity=ss.cross_immunity(...), seed=..., cancer='once')` would hold the three shared pieces:
- cross-disease immunity, with a combination rule;
- joint initial conditions;
- union results ("any member").

It would also hold "one outcome per person across the family". That works today as an "as soon as" transition with a function condition, which hangs an ODE (ST26). Products and tests should take the family too (section 4). Without the family, the declared port works, but each piece is a small block of code. The per-person immunity (HP19) makes it agents-only.

### 4. Screening, triage and treatment

HPVsim's interventions are the WHO screen-triage-treat algorithms, vaccination by age and sex, and therapeutic vaccines. Algorithms 1 and 3 and routine vaccination are 17 lines in v4 against 55 in HPVsim's example. HPVsim's products take 29 lines against 720 plus 317 in the Starsim 3 classes they subclass. The yields from the product tables agree to 1%.

| ID | Finding | Evidence | Fix | Size |
|---|---|---|---|---|
| HP24 | **A test has two results; HPVsim's have 3 to 5 by state, and combine across genotypes.** Cytology's results are abnormal, ASCUS, inadequate and normal. Colposcopy's are cancer, HSIL, LSIL, ASCUS and normal. The treatment assigner chooses radiation, excision, ablation or none. HPVsim's `hierarchy` then takes the most severe result across genotypes. The workaround, a chain of binary tests on `on_negative=`, reproduces the draw exactly but reports and costs every link: 35,397 tests for 20,000 cytologies, and costs 77% high. A test acts on one disease: `detects=hpv16.C \| hpv18.C` works for "any", but `sensitivity={'hpv16.C': 0.9}` gives `E505` | `hpvsim:v4port/interventions/probe_multicategory.py`, `probe_multigenotype.py` (reran) 2a, 2b | `ss.Test(results={'abnormal': dict(C=0.9, X=1), ...}, default='normal')`, with follow-ups per category (`on='cytology.abnormal'` in the peer cascade); `'disease.state'` keys and `hierarchy=` across a family | About 80 lines. Related to D31, which lists `Dx(df=)` hierarchies as having no v4 form |
| HP25 | **A treatment's efficacy can't vary by state**, though a test's sensitivity can (`sensitivity=dict(C=0.9)`). `efficacy=dict(P=, C=)` is read as rate effects (`E505`), and `ss.by(state=...)` raises a `KeyError`. The workaround is a function returning a value per agent plus a rate-0 cure transition from each state, agents only; `describe()` prints the function's repr | `hpvsim:v4port/interventions/probe_treatment_by_state.py` 2, 2b, 3 | `ss.Treatment(cure=dict(P=0.936, C=0.936))`, the same dict form as `sensitivity=`, each state curing to its recovery destination; this absorbs ST5 | About 40 lines. Related to ST5 |
| HP26 | **No `delay=` on `ss.treat`** (HPVsim's `treat_delay`): `TypeError: Timeline.__init__() got an unexpected keyword argument 'delay'`. The workaround is a pseudo-test with sensitivity 1, which is reported and costed as a test | `hpvsim:v4port/interventions/probe_v4_capacity_delay.py` 3a, 3b | `delay=` on every delivery, as on `ss.Test`, with D9's fix | About 20 lines. Related to ST13, D9 |
| HP27 | `ss.change` on a per-network `beta` pair (the form HPVsim's network needs, and its `dynamic_pars`) crashes with `TypeError: '<' not supported between instances of 'dict' and 'int'`. `ss.modify(target, multiply=ss.timeseries(...))` and `ss.change` on a scalar work, and replace `dynamic_pars` | `hpvsim:v4port/interventions/probe_dynamic_pars.py` 5 | A clear error, or apply the factor to each entry | Small. Related to N20, D21 |

**Products over several diseases** is the design point behind HP7, HP24 and the cross-genotype half of HP25. Every HPVsim product acts on all genotypes at once:
- the vaccine, with protection per genotype;
- the tests, which take the most severe result across genotypes;
- ablation, which clears every genotype's lesions.

v4 binds a product and its delivery to one disease, and is silently wrong when that is violated (HP7). Proposal:
- `disease=` takes a list or a family;
- product parameters may be keyed by disease (`ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975, hi5=0.5))`);
- test results and cures may be keyed by `'disease.state'`.

This has the same shape as one ANC visit testing for HIV and syphilis in STIsim, so it belongs in the cascade design pass, not in HPV code.

**Known items that HPV confirms:**
- **ST16 and the cascade shape:** one result must trigger several deliveries. Ablation is one treatment per genotype, and a therapeutic vaccine is a cure plus immunity. `on_positive=[a, b]` fails at run time with `TypeError: 'list' object is not callable`. Algorithm 3 is not a tree: colposcopy has two parents. That works as two follow-ups, but it is the peer cascade, not nesting.
- **D31:** no capacity queue (`treat_num`). Those over capacity are lost, so yearly rounds at half-year steps lose half the capacity: 400 of 1,467 referred are treated.
- **D40:** no all-or-nothing vaccine (`sterilizing_p`). The leaky mean, 0.9975, was used instead.
- **ST14:** no remembered screening date for "not screened in the last 5 years".
- **D23:** `ss.vaccinate`, `ss.screen` and `ss.treat` raise `E506` on the coded `hpv.HPV`. `ss.modify('hpv16.rel_sus', ...)` raises `ValueError: min() iterable argument is empty` (related to N21).
- **D22:** six of HPVsim's tests fail because the first `run_one_step()` calls no intervention.
- **X13:** an analyzer must test `ti_X == sim.ti − 1` for states entered by progression and `== sim.ti` for states entered by transmission.
- **D27:** a product and its delivery can't share a name.
- **L26:** unknown keywords on products reach `Timeline.__init__()`.

ST12 (stock targets) is not needed: HPV programs set coverage.

### 5. Population: networks, multiscale and the subclassed sim

| ID | Finding | Evidence | Fix | Size |
|---|---|---|---|---|
| HP8 | **`ss.MFNet`'s `mean_degree` and `concurrency` results are multiplied by `pop_scale` whenever `total_pop` is given**, which every HPVsim and STIsim sim does. So `net.check()` compares 176 with a target of 0.5 and raises a false `W402` | `hpvsim:v4port/population/repro_degree_results_scaled.py` (reran): 0.521 without `total_pop`, 104.1 with `total_pop=1e6`; the actual degree is 0.521 both times | `scale=False` on the two results in `DegreeTargets.define_degree_results()` (and on PartnerNet's `match`) | 2 lines |
| HP9 | **`ss.MFNet` sets debut and participation only for people younger than one step**, so anyone added at another age by `people.grow()` never joins the network. That covers HPVsim's `AgeMigration` and STIsim's `Migration`. The same is true in 3.7.2 | `hpvsim:v4port/population/probe_mfnet_immigrants.py`: 614 immigrants aged 25 and over, participant 0.000 under v4 and v3, against 1.000 for the original agents | Set network states for any alive agent whose states are unset, as `hpv.SexualNetwork.set_network_states()` does, and add a contract test with a demography module that adds people at any age | About 5 lines. Related to ST1: the same `people.grow()` blind spot, for networks |
| HP10 | **A participation probability that depends on age is read at age 0**, because `MFNet.set_participation()` runs once, at birth. So HPVsim's participation by age and layer can't be ported: written as a function of age, nobody born during the run ever takes part. The same is true in 3.7.2 | `hpvsim:v4port/population/repro_participation_age.py` (reran on both): 2,632 people born during the run and now 15 or older, participant 0.000 (v3: 2,825, 0.000) | Participation as an eligibility evaluated each step (HPVsim's form), or drawn at debut | Small to medium |
| HP11 | **`MFNet`'s degree target misses when partnerships are long relative to people's time in the population.** `p_seek` assumes partnerships end only by duration; PartnerNet has a departures correction and MFNet doesn't. With HPVsim's marriages (mean 80 years) and Nigeria's demography, the realized degree is 0.51 against a target of 0.87 | `repro_degree_results_scaled.py` (reran): with births and deaths, durations of 80, 20 and 5 years give 0.654, 0.863 and 0.998 against 0.87 | PartnerNet's `departures` correction in MFNet; warn at `init()` when `p_seek` saturates | About 30 lines. Related to N15 |
| HP12 | **Once one module of an `hpv.Sim` is declared, every run tool breaks, including `ss.parallel(sim, n_runs=)`, which worked on the coded sim.** `hpv.Sim` builds its modules in its constructor, so unlike `sti.Sim` (ST10) it has `inputs`. But `new()` rebuilds a plain `ss.Sim(inputs)`. That drops the class, `sim.location` (which `AgeMigration.init_pre` reads), and parameters routed after `super().__init__`. Every run tool goes through `new()` as soon as one module is declared | `hpvsim:v4port/framework/hpvsim_mixed_new_v4.log`: `AttributeError: AgeMigration could not find a location on the sim` from every verb. `hpv_sim_v4.py`: 10 lines that record the constructor's arguments and replay them give identical results. With them, `new()`, `ss.parallel`, sweeps, `diff()` and `calibrate()` all work on coded HPV plus a declared disease, and `fit.sim` is an `hpv.Sim` | `ss.Sim.__init__` records `type(self)` and its call's arguments, and `new()` replays them with overrides. That covers both shapes, `sti.Sim` and `hpv.Sim`, better than ST10's "`type(self)(inputs)`" | About 10 lines. Related to ST10, R13, N18 |
| HP13 | `msim.diff()` refuses a coded-only sim ("only available for sims with declared states and transitions"), because `ss.parallel` builds `MultiResults` only when something is declared. Built by hand (`MultiResults.from_sims`), `diff()` is right for coded HPVsim: vaccination against baseline gives −21.5M, −12.2M and −16.9M infections | `hpvsim:v4port/framework/hpv_sim_v4.py` | Build `MultiResults` for every sim | 1 line. Related to R13, R20 |
| HP17 | A coded module's array as a modifier's amount (`ss.modify(hpv.rel_sus, multiply=hiv.hiv_rel_sus)`, the natural spelling of HPVsim's HIV factor) raises `ValueError: Cannot add state <FloatArr "factor"...> since already added`, a message unrelated to the cause. A lambda works | `hpvsim:v4port/population/repro_modify_codedarr.py` | Accept an `Arr` as an amount, or say a coded state can't be used here yet | Small. Related to N19, D23 |

**Multiscale.** HPVsim reduces the variance of rare cancers by cloning, at CIN, extra agents with weight 1/ratio (`ms_agent_ratio=100` in its demo; `_grow_fine_agents` is its single biggest cost). Unchanged, it runs and agrees under v4 (cancers at ratio 10: 723k ± 49k against 806k ± 71k, with HP5 in the way). For a declared HPV, two things are missing:
- results weighted by agent weight (HP4);
- a way to clone. Coded cloning copies a declared disease's arrays, including the scheduled exits (`ti_exit_*`), so a clone leaves its state on its source's step, and coded code has no public way to give it a fresh trajectory (related to ST11).

Under compartmental solvers multiscale is unnecessary, but HPV needs the network, so it is agents for the foreseeable future. The cheaper path is to keep multiscale coded, with HP4 and ST11's `disease.move()` redrawing the exit. The declared alternative is a resolution option on one transition (`resolve=10`, proposed: draw the rare branch k times per person at weight 1/k), which could live entirely in the agent engine.

**The network.** `hpv.SexualNetwork` runs unchanged with a declared disease on it. There is no ST9-style crash, because its `net_beta()` reads only `disease_beta`, and per-edge probabilities are identical to the coded HPV's. It is 2.3 times faster under v4 (2.26 s against 5.12 s). A best-effort port to two `ss.MFNet` layers loses age mixing (an age gap of 0.7 ± 19 years against 3.9 ± 6.5), which halves young women's prevalence. That is N15 again, with HPVsim numbers. HPVsim is the third model, after EpiModel and STIsim, to need the same short list:
- pairing by an age-mixing matrix;
- degree targets by sex;
- participation as an eligibility evaluated each step;
- layers sharing one concurrency budget;
- acts scaled by the couple's ages.

As for STIsim, this is not urgent, because the coded network is first-class and fast. Every `people.grow()` path (births at age 0, immigrants at any age, clones) has to be handled by every module, network and disease alike (ST1, HP9). That argues for one engine-level "new agents" hook with a conformance test that adds agents at arbitrary ages.

**HIV.** A mixed sim of STIsim's coded HIV and a declared HPV reproduces HPVsim's effect: HPV prevalence in HIV-positive against HIV-negative women is 1.71, against 1.64 on v3 and 1.70 for coded HPV on v4. It needs a subclass that skips `hpv.HIV`'s check for a coded HPV, conditions on HIV written as lambdas (N19, duplicate), and branch probabilities as functions. HPVsim fixes immunity at the HIV status at clearance; v4 can only scale by current status (HP19).

**Demography.** `AgeMigration`, `hpv.Births` and `AnnualBirths` run unchanged. `ss.people('Nigeria')` matches HPVsim's population without its loader (95.8M against 97.1M in 1990, share under 15 0.454 against 0.453). One item is spec drift (HP33): `04-parameters.md` says birth, death and migration rates are bundled named data, but `ss.Births(birth_rate='nigeria')` raises `TypeError: Updating timepar 20 from <class 'starsim.time.peryear'> to <class 'str'> is not supported`.

### 6. Compartmental runs

The natural history (split form) runs as an ODE, in 0.43 s for 100 years against 0.88 s as agents. P(CIN) and P(cancer) are within 5% of agents, and every approximation is printed. HPVsim as a whole can't run as an ODE: the network has no compartmental form (D10), sex is `people.female` (`E804`; D15), and the per-person immunity is agents-only (HP19).

| ID | Finding | Evidence | Fix | Size |
|---|---|---|---|---|
| HP28 | **HPV's dwells, with a coefficient of variation above 1, become a single Erlang stage (an exponential) under compartments, and the long tails that make cancer are lost.** The note says so, which is good | `hpvsim:v4port/natural_history/solvers_compare.py`: mean time to cancer 33.4 years under the ODE, against 26.6 as agents (v3: 29.0). Note: "approximated by 1 Erlang stage (... std 29.5 instead of 54.2 years)" | For a CV above 1, fit a two-branch hyperexponential, which is a split on entry that v4 already has; the staging in HP18 is the same machinery | About 40 lines. Related to X20 |
| HP29 | **Cross-genotype conditions cross the diseases' compartments, so the size grows as 20^k.** Under homogeneous mixing, one genotype runs in 0.3 s. With the cross-immunity `where=` modifiers, 2 genotypes give 8,000 compartments per disease (1.8 s) and 3 give 160,000 (25.5 s); 4 would be 3.2 million. With the "as soon as" cancel for one cancer per person, the 2-genotype ODE didn't finish its first step in 200 s (ST26, duplicate). `auto` picks agents even for one genotype, without saying why | `hpvsim:v4port/genotypes/probe_ode.py`, `probe_ode.log` | Print the crossed size before building; D13's history dimension (2^k, not 20^k) for cross-immunity; an immediate move (ST26) | Medium. Related to D13, X9 |

### 7. Small things

| ID | Finding | Evidence | Cross-reference |
|---|---|---|---|
| HP30 | 88 modifiers with lazy `where=` conditions add about 0.2 ms per modifier per step: each leaf rebuilds a `{name: module}` dict of every module (`conditions.py`, `Evaluator.module`). 0.64 s with none, 2.28 s with 88 (80 steps, 10,000 agents). It is why the fully declared four-genotype model is 1.6 times slower than HPVsim on v3 | `hpvsim:v4port/genotypes/probe_modifier_cost.py` | Cache the lookup at `init()`, and evaluate each distinct condition once per step. About 10 lines. Related to N29 |
| HP31 | Every `where=` transition with `dur=` prints `RuntimeWarning: divide by zero encountered in divide` (`solvers.py:3616`) on every run. Found by three agents | `hpvsim:v4port/interventions/repro_where_dur_warning.py` | `np.errstate` or `np.divide(..., where=)`. 1 line |
| HP32 | `describe()` of a sim with no `unit=` and dates in years says `steps 2000 to 2010, every 0.25 steps` | `hpvsim:v4port/framework/repro_describe_units.py` | Say years when the dates are years. Related to L18 |
| HP33 | Named birth, death and fertility rates are in the spec but not the code | `hpvsim:v4port/population/named_data.py` | Spec drift |
| HP34 | A modifier named like a `People` state crashes with `AttributeError: Module nab_imm already added`, without saying what clashes | `natural_history/probe_small.py` 7 | Friction |
| HP35 | `p=[ss.par(0.3, bounds=...), None]` on a branch asks for `ss.par(..., name="beta")`, a wrong example name for a branch | `framework/hpv_sim_v4.py` | Name it `p_<transition>`, as `calibrate()` reports it. Related to L20, L23 |
| HP36 | A branch whose `move=dict(dim='a -> b')` doesn't apply to someone is dropped, and the other branches take its probability. This is undocumented, though `describe()` shows it as `set=0 where=~(prior == 'naive')` | `population/probe_move_arrow.py` | Document it, or keep the remainder in place. Related to ST7 |
| | `describe()` prints `where=~~people.female`, lambdas as `<lambda>`, a `p` function with no note that it is evaluated at exit, float32 noise in modifier amounts, per-act betas as internal rates, and "as 1 Erlang stages" under `abm` | Several | Related to L27, ST2 |
| | `calibrate()` with one run per evaluation landed off the truth every time: 0.003 against 0.002; 0.12 and 0.19 against 0.3. It said nothing about either. HPVsim turns `reseed` off for this reason ("Optuna then picks the luckiest seed") | `natural_history/calibrate_nh.py`, `framework/declared_calibrate.py` | Duplicate of N16, R4, R5 |
| | `ss.parallel(sim after init(), n_runs=3)` gives three identical runs; `solver='ode'` on `hpv.Sim` runs as agents, silently; `stratify=` on `hpv.Sim` gives nothing; `hpv.HPV(beta=ss.par(...))` raises `TypeError` | `framework/v4_verbs_v4.log` | Duplicates of N5, X2, D23, R13 |
| | The automatic `prevalence` isn't stratified; a sim-level result across genotypes (the genotype share of cancers, a calibration target) raises `KeyNotFoundError` | `framework/declared_results_v4.log` | Duplicates of ST20, R17 |

## Before removing the v3 vocabulary (decision 2)

HPVsim subclasses ten Starsim 3 classes in its interventions and products, and `ss.Calibration` for fitting. These are what v4 needs first, so that HPVsim loses nothing:

| v3 class HPVsim uses | Where | What v4 needs first |
|---|---|---|
| `ss.BaseVaccination`, `ss.RoutineDelivery`, `ss.CampaignDelivery`, `ss.Vx` | `routine_vx`, `campaign_vx`, `hpv.vx` | Products over several diseases (HP7), all-or-nothing protection (D40), `at_age=` for routine doses (N8, N9) |
| `ss.BaseTest`, `ss.BaseScreening`, `ss.BaseTriage`, `ss.Dx` | Screening and triage, `hpv.dx` from `products_dx.csv` | Categorical results with a hierarchy across diseases (HP24); a remembered test date (ST14) |
| `ss.BaseTreatment`, `ss.treat_num` | `treat_num`, `treat_delay`, `hpv.tx`, `radiation`, the therapeutic vaccines | Cure by state over several diseases (HP25), `delay=` (HP26), `queue=True` (D31), several follow-ups (ST16), efficacy on a named branch (HP6). Radiation is an efficacy on the cancer exit's rate, which v4 already has (untested) |
| `ss.Calibration` (`hpv.Calibration`, `make_calib_sims`) | Every HPVsim calibration | The STIsim list (workers, resume, `check_fn`, saved results, top-N ensembles), plus several runs per evaluation by default (N16). HPVsim's targets need D15, HP20, HP23 and R17: cancer incidence per 100,000 women by age, the genotype mix in cancers, and the age-standardized rate. Its long data format has a `genotype` column, which needs a mapping to the family's members. Its own comment says SQLite deadlocks past about 32 workers, so storage needs a journal |
| `ss.Infection`, `ss.SexualNetwork`, `ss.Demographics`, `ss.Births`, `ss.Connector`, `ss.Analyzer`, `ss.FloatArr`, `ss.BoolState`, `People.scale_flows()` | The coded-module API | Keep it first-class, with the seam in the STIsim review's section 2 and HP4, HP12 and HP17. Don't delete `scale_flows()` |

## Speed and inefficiencies

All on the same interpreter. The machine was shared and heavily loaded throughout (load 15 to 47 on 20 cores), so these are best runs, with v3 and v4 interleaved or concurrent, and differences under about 30% are noise.

| What | Measured | Verdict |
|---|---|---|
| Unchanged HPVsim | One genotype, 10,000 agents: 4.0 s against 9.6 s (2.4 times). Demo with 4 genotypes and multiscale: best 29.9 s against 38.2 s, and 20.5 s against 46.3 s. 4 genotypes at 10,000 agents: 5.6 s against 8.7 s | Good. As for STIsim, the gain is in v4's core: v3's per-call `Dist.jump()` (17.5 s) and the copies in `process_pars` (7.2 s) are gone |
| `hpv.SexualNetwork.step()` | 2.26 s against 5.12 s | Good, with no code change |
| Interventions' own step (CPU time) | Algorithm 1: 0.17 s against 3.67 s; algorithm 3: 0.51 s against 6.61 s | v4's vectorized `Delivery.test()` against HPVsim's pandas `.loc` per state and genotype |
| Declared natural history as agents | Cohort of 20,000 women: 0.72 to 1.52 s against 1.55 to 3.47 s (about 2 times) | Good; no N29 penalty |
| Declared four genotypes as agents | Per-person immunity: 8.9 s against 8.7 s; history states with 88 modifiers: 13.5 s | The modifier count, not the engine (HP30) |
| Natural history as an ODE | 0.43 s for 100 years; binomial 0.42 s | Fast; its timing is off (HP28) |
| Four genotypes as an ODE | 3 genotypes 25.5 s at 160,000 compartments; the cancel hangs it | Fix (HP29, ST26) |

## Keep

HPVsim confirms the review's keep list:
- **One declared disease per genotype, composed in one sim.** The four genotypes are one loop, and `what_modifies()` lists every cross-immunity term.
- **`ss.modify` with a kept base.** It removes HPVsim's reset-then-multiply protocol on `rel_sus` and its dependence on module order.
- **The product/delivery split.** The WHO algorithms are 17 lines, follow-ups are structural, and every delivery has results, whereas HPVsim's triage reports nothing.
- **Stratified results**, which replace `by_age`.
- **`calibrate()` with no `build_fn`**, including on a parameter read only inside a branch function, and with JAX gradients on the ODE (7.8 s).
- **The printed approximations under compartments**, and clear refusals that name the function (`E804 ... the function "p_cin_exact" cannot run with solver="ode"`).
- **Common random numbers:** P(CIN) is identical across ages within a seed, as in v3.
- **The coded network and demography running beside declared diseases.**

One more, as for STIsim: **with HP12's 10 lines, a mixed `hpv.Sim` gets `new()`, sweeps, `diff()` and `calibrate()` for the whole sim.** That is the incremental path HPVsim would take.

## Suggested order

1. **A day:** HP12 and HP13 (the run tools on `hpv.Sim`); HP4 (weighted results, and keep `scale_flows()`); HP8 and HP31; HP5 in the migration guide.
2. **About a week:** HP1 to HP3 and HP6 to HP7, with ST1 to ST4, each with a regression test. Add to P10's contract tests:
   - "results weighted by `people.scale` equal the weighted sum";
   - "two spellings of one branch give the same shares";
   - "a pair written to the spec gives the same answer whichever sex a network puts in `p1`";
   - "a product delivered for a second disease either protects against it or raises".
3. **The natural-history shape:** a spec section on outcomes given the dwell, then `ss.given_dwell` (HP18) with its staged compartmental form (absorbing HP28), and per-person values on events (HP19), shared with STIsim's CD4.
4. **A disease family:** cross-immunity with a combination rule (HP22), joint seeding (HP15), union results and standardized rates (HP23), and products over a family (HP7's second half).
5. **Screening, in the cascade design pass:** categorical results (HP24), cure by state (HP25), `delay=` (HP26), `queue=` (D31), several follow-ups (ST16).
6. **Networks:** HP9 to HP11 in `MFNet`, then the shared `SexualNet` list (N15).
7. **Multiscale:** HP4 first, then decide between a coded bridge (ST11 with a redrawn exit) and a declared resolution option.

## Index of new findings

| ID | Finding | Kind |
|---|---|---|
| HP1 | Two spellings of a branch decide at different times | Silent wrong answer |
| HP2 | An immediate junction holds people a step as agents | Silent wrong answer |
| HP3 | A `beta` pair is p1 → p2, not male to female | Silent wrong answer |
| HP4 | Results ignore `people.scale` | Silent wrong answer |
| HP5 | Unchanged v3 seeds transmit one step fewer | Silent change |
| HP6 | An efficacy on a branching exit scales the exit's rate | Silent wrong answer |
| HP7 | One vaccine for two diseases protects against the first only | Silent wrong answer |
| HP8 | `MFNet` degree results multiplied by `pop_scale` | Silent wrong answer |
| HP9 | `MFNet` newcomers at any age never join | Silent wrong answer |
| HP10 | `MFNet` participation by age read at age 0 | Silent wrong answer |
| HP11 | `MFNet` degree target misses in an open population | Silent wrong answer |
| HP12 | `hpv.Sim` loses every run tool once a module is declared | Dead end |
| HP13 | `diff()` refuses coded-only sims | Dead end |
| HP14 | Strain cross-immunity can't run over networks | Dead end |
| HP15 | Initial conditions can't be functions, or joint across diseases | Crash |
| HP16 | A branch with no dwell crashes as agents | Crash |
| HP17 | A coded array as a modifier amount gives a misleading crash | Crash |
| HP18 | No outcome that depends on the drawn dwell; the spec's form puts cancer 22 years early | Missing (design) |
| HP19 | No per-person value updated on events or frozen at entry | Missing (design) |
| HP20 | A branching flow has no destination axis | Spec drift |
| HP21 | A remainder branch beside another timed exit raises E205 | Friction |
| HP22 | Cross-immunity needs 88 modifiers and can't combine as HPVsim does | Missing |
| HP23 | No union across diseases; no standardized rates | Missing |
| HP24 | Tests have two results, and act on one disease | Missing |
| HP25 | No treatment efficacy by state | Missing |
| HP26 | No `delay=` on `ss.treat` | Missing |
| HP27 | `ss.change` on a `beta` pair crashes | Crash |
| HP28 | Dwells with CV above 1 become one exponential stage | Approximation |
| HP29 | Cross-disease conditions make compartments grow as 20^k | Performance |
| HP30 | Lazy conditions cost 0.2 ms per modifier per step | Performance |
| HP31 | A divide-by-zero warning on every `where=` dwell | Friction |
| HP32 | `describe()` says "steps" for a sim in years | Friction |
| HP33 | Named birth and death rates are missing | Spec drift |
| HP34 | A modifier named like a `People` state crashes unhelpfully | Friction |
| HP35 | A branch `ss.par` with no name gets a wrong example | Friction |
| HP36 | A `move=` branch that doesn't apply is silently dropped | Minor |
