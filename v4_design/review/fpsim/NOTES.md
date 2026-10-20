# FPsim on Starsim v4

Source: `fpsim-v4` worktree (= fpsim main, starsim 3.6), Kenya, `SimpleChoice`. Starsim `v4-rs`, `002c4ec3` (HEAD moved to `b4f6086e` during the review; the final run used it).
Files: `port.py` (all variants: 44 s, 3.9 GB peak from the ODE), `repro_where_branch.py`, `repro_dur_vs_hazard.py`, `repro_misc.py` (9 checks).

Two formulations, because one model text did not survive both solvers:

| | `make_sim()`: faithful | `make_compartmental_sim()` |
|---|---|---|
| Structure | `fp` (9 states) + `contra` (16 states) + 55 `ss.modify` | one module; contraception is a 16-level dimension |
| Declared objects | 27 + 126 transitions, 1 transfer | 148 transitions, 100 transfers |
| Method durations | v3's lognormal/Weibull/gamma, age factors via `adjust=` | exponential, matched to 12-month continuation |
| Runs as | agents (ODE runs out of memory at init) | agents and ODE |
| `step()` code | none | none |

Results (Kenya, 2000–2020, 5k agents; 4-seed means for v3 and the faithful port):

| | mCPR 2005 / 2010 / 2019 | TFR 2005 / 2010 / 2019 | implants / injectables at end |
|---|---|---|---|
| FPsim v3 on Starsim v4 | .426 / .505 / .563 | 3.44 / 2.72 / 2.37 | .252 / .102 |
| v4 faithful, agents | .411 / .481 / .564 | 3.30 / 2.44 / 2.24 | .247 / .096 |
| v4 compartmental form, agents | .389 / .445 / .484 | 2.43 / 1.95 / 2.57 | .120 / .127 |
| v4 compartmental form, ODE (coarse) | .407 / .456 / .498 | 2.61 / 2.32 / 2.23 | .116 / .122 |

The faithful port reproduces v3's mCPR, its method mix (within 0.01 for every method), and its event rates. TFR is about 5% low. Likely causes: personal fecundity is replaced by its mean; postpartum activity is averaged over bands; and a pregnancy loss resets time since birth. The compartmental form is a different model: exponential durations shift the mix toward short-acting methods.

Lines: v3 `fpmod.py` + `methods.py` are about 2,000 non-blank lines, much of it method-list management. The faithful v4 model is about 235 lines, plus 25 lines for data and 80 for results.

FPsim v3 itself runs unchanged on v4. It takes 2.3 s against 4 s for the faithful port. 57 of 61 tests pass. The failures are:
- `test_baseline`: RNG drift, expected.
- Two tests with stochastic thresholds.
- `change_initiation`: `TypeError` at `interventions.py:778`. It tests an exact float year, `years[0] == year`, at `:757`, so the start is never matched and `expected_women_oncontra` stays `None`.

## Feature by feature

**1. Reproductive states and timing.** v3: `FPmod(ss.Pregnancy)`, `fpmod.py:552-602`, `606-757`, `1002-1046`. Clocks are `ti_*` arrays, and outcomes are classified after the fact by gestational age.
- v4: `NP, INF, P1, P2, PA..PE`.
  - `tri1 = P1 >> P2 @ constant(13 wk)`.
  - Delivery `P2 >> PA` uses `dur=ss.choice(weeks 19..29)`.
  - Competing `miscarriage` (`P1 >> NP`), `stillbirth` (`P2 >> NP`) and `maternal` (`P2 >> null`) hazards, with `ss.by(age=)`.
- **Clean.** This is the part v4 was built for.
- Losses at 13–20 weeks are stillbirths here; v3 counts them as miscarriages. That needs another state, or a gestational-age threshold on a state.
- **Bug (new):** as agents, a `dur=` competing with a hazard gives the hazard about one step less exposure for everyone who enters mid-run. Monthly steps: 0.076 against 0.107 exact; weekly 0.095, daily 0.105 (`repro_dur_vs_hazard.py`). `rate=` is exact. The ODE is exact. Miscarriages are 30% low as agents in the compartmental form. This hits every port with a `dur=` stage plus a death hazard.

**2. Fecundity by age and parity.** v3: `make_p_conceive` and `_adj_*`, `fpmod.py:832-920`. The rate is age fecundity × personal fecundity × nulliparity × exposure by age and by parity. It is then capped at 1 per year and converted to a monthly probability.
- v4: one `ss.by(table, dims=['age', 'parity'])` per conception route.
- **Clean-ish.**
  - Typed rates do not fit: `ss.permonth(ss.by(...))` and a 2-D `ss.by` of `ss.permonth` both raise (repro_misc 2), so tables are bare numbers per year.
  - Personal fecundity (uniform 0.7–1.1) needs a per-agent array plus a function, so it is agent-only. It is replaced by its mean.

**3. Contraceptive efficacy on conception.** v3: `rel_sus *= 1 - eff` (`fpmod.py:776-801`), applied *before* the cap and the monthly conversion.
- With Kenya's calibrated exposure by age (2.5 at age 30), a woman of 30 on no method conceives with probability 1 per active month. The pill then cuts the hazard to 0.4% of that, not 5.5%.
- So the method's effect on the hazard depends on age, parity and route. v4: 9 methods × 6 routes = 54 `ss.modify(fp.<route>, multiply=ss.by(ratio, dims=['age', 'parity']), where=contra.<m>)`, built in a loop.
- **Ugly.** A rate that depends jointly on two modules' states has to be expanded into one modifier per (state, transition). Each modifier gets an empty result group.
- Suggestion: a modifier target that is a list of transitions, and `multiply=` keyed by the other module's state, e.g. `ss.modify(conceptions, multiply=ss.by(contra=eff_table))`.

**4. Sexual activity and debut.** v3: `fpmod.py:383-497`. Activity is a Bernoulli redrawn every month from age-specific, postpartum-month-specific and spacing-preference data. Debut is a fated age per woman, drawn once.
- v4: activity is folded into the expected conception hazard: it is exact for conceptions, but aCPR and `n_sexually_active` are lost.
- Debut is `contra.pre >> dec0` at the hazard implied by the fated-debut distribution, plus `ss.modify(fp.conceive, multiply=0, where=contra.pre)`.
- **Clean-ish.** The fated age becomes a hazard, which is exact for a cohort but not per woman.

**5. Time since last birth.** v3: `ti - ti_delivery` and `ti - ti_live_birth`, read in four places:
- activity by month postpartum (0–17);
- spacing preference in 3-month bins to 54 months;
- LAM in months 0–5;
- contraception decisions at 1 and 6 months.

v4: a chain of postpartum states `PA` (0–1 m), `PB` (1–6), `PC` (6–18), `PD` (18–39), `PE` (39–51), then `NP`, with fixed dwells. Each state has its own conception transition and band-averaged multipliers.
- **Awkward.** It works under both solvers, but:
  - monthly data become band averages;
  - six conception transitions instead of one;
  - a loss during a later pregnancy drops time since the last birth (back to `NP`, which v3 does not do);
  - "postpartum" stops existing as a single condition.

This is FPsim's main need from the time-since-event proposal.

**6. Parity.** v3: a counter, `parity += 1` (+2 for twins), `fpmod.py:635`; `n_births` and friends; `birth_ages` 2-D arrays.
- v4: a dimension, `fp.stratify(parity=['0', '1', '2', '3', '4+'])`. Delivery is one transition per (level, singleton/twin), each with `move=dict(parity='i -> i+k')`: 10 transitions.
- **Awkward, but works in both solvers.** Parity as a stratification is the right answer. Per-woman counters and histories are impossible without `step()`: births, abortions, ages at each birth, birth intervals. v3 calibrates to birth spacing and age at first birth.
- **Spec mismatch:** `p=0.984` / `p=0.016` per level raises `E205 ... sum to 5`. The check sums branches across levels, though `move=` keeps each branch to one level (repro_misc 3). Workaround: a one-hot `p=ss.by(parity=...)`.
- Suggestion: `move=dict(parity=+1)` for ordered dimensions, with a cap at the last level.

**7. Abortion.** v3: conception, then 20.1% abort, which leaves the woman not pregnant and is counted (`fpmod.py:922-982`).
- v4: conception rates × 0.799, then `abortions = ss.outcome(pab/(1-pab)*conceptions)`, where `conceptions = ss.outcome([names])`.
- **Clean-ish.** Counted, not tracked per woman.
- Gotcha: `ss.outcome(w*(t1 + t2))` works for transitions used as given. It raises `E702 ... (None)` if the transitions were scaled (`t*0.8`) before being passed to the module (repro_misc 8). So the transitions have to be named by string.

**8. Births, twins, infant and maternal death.** v3: embryos are agents from conception, linked by `parent`. Twins are two agents. Infant death ends the mother's breastfeeding and LAM.
- v4: `births = ss.null >> (0.95*NP + 0.05*INF) @ k*P2.n`. Daughters surviving infancy are added at the expected delivery rate. No newborn is linked to a mother.
- **Ugly.** An inflow cannot be driven by a flow, so births are not deliveries. Infant death does not end LAM.
- Breastfeeding duration (lognormal, mean 20 months) only matters through LAM (≤ 5 months), so it is not modeled.
- **Bug:** `INF` (primary infertility), reached only from `ss.null`, is guessed to be a count. The run then crashes in `step_counts` with `KeyNotFoundError: "S" not found` (repro_misc 4). `ss.State(people=True)` fixes it; the guess should look at inflows too.
- Suggestion: `ss.birth(per=fp.deliver, n=embryos, into=...)`, an inflow whose count is a flow and whose agents get a parent.

**9. Method choice and switching.** v3 `SimpleChoice`, `methods.py:983-1074`, `1340-1443`. At the end of each method's dwell, the woman decides:
- use, with p = logistic(age, ever-used);
- then which method, from the switching row (from method × age group), weighted and renormalized;
- then a new dwell, from the method's fitted distribution with age factors on its scale.

v4:
- From each of 9 sources, 10 `ss.Transition(src, dst, p=ss.by(table, dims=['age', 'ever_used']), dur=dist)`: decide on entry, then wait.
- Age factors via `contra.stratify(age=E, adjust={transition: 1/multiplier})`, which divides the duration.
- `ever_used` is a dimension, set by `move=dict(ever_used=True)`. The branch is drawn *after* the move, which is right.
- Self-loops (`none -> none`) work.
- **ss.by is enough** for age × ever-used branch probabilities, including a sim dimension crossed with a disease dimension. The problems:
  - **Decide on entry vs on exit.** v3 decides at the end of the dwell, with the woman's age then. For an IUD, implant or BTL (years) the age at decision differs. `decide='exit'` (REVIEW §5.3) is needed.
  - **A `Dist` with a function parameter** (v3's age callbacks) as `dur=` raises `TypeError: float() argument must be a string or a real number, not 'function'`. `dwell_stats` computes the mean and std eagerly even for agents (`transitions.py:1041`; repro_misc 1).
  - A `dur=fn(sim, uids)` returning durations is read as a mean, with a shape that is not the user's. `adjust=` works, but only for factors on a scale.
  - v3's custom `Fisk` distribution has no stats rule; a Weibull is used as a stand-in.
  - 126 transitions in a loop: programmable but unreadable in `describe()`.
  - Self-loops are counted as flows (`none_to_none`).
- **Awkward.**

**10. Pregnancy pauses contraception; postpartum decisions.** v3: `method = 0` at conception (`fpmod.py:984-1000`). Decisions are scheduled through `ti_contra`: delivery + 1 month (pp1 coefficients and mix), +6 months for non-users (pp6), and the next step after a loss.
- v4: no `on=` for transitions, so these are conditions on `fp` states at a big rate (`BIG = permonth(30)`):
  - `pause`: any method to `paused`, `where=fp.P1 | fp.P2`;
  - `go_pp1`: `paused` to `dec1`, `where=fp.PA`;
  - `go_pp6`: `wait6` to `dec6`, `where=fp.PC`;
  - `resume`: `paused` to `dec0`, `where=fp.NP | fp.INF`.
- **Ugly**:
  - 3 junction states, each costing a step;
  - `rate=BIG` on immediate `p=` branches, since a `p=` branch with no rate crashes (REVIEW §6);
  - Euler with `substeps=1` for the ODE, so that BIG empties a compartment in one step.
- **Bug (new):** a `p=` branch whose rate is gated by `where=` on another module's state never fires. Everyone stays in the state (`repro_where_branch.py`: 1000/1000 still in X after the condition holds; plain hazards work).
  - This was the first version of the postpartum decisions. 212 women were stuck in `paused` against 51 pregnant. Nothing errored.
  - Likely cause: the dwell is drawn at rate 0, then rescaled.

**11. Initial conditions.** v3: women past their fated debut draw p_use (as never-users); users draw the initial mix by age (`methods.py:937-972`, `1318-1338`). Everyone starts nulliparous.
- v4: `init=dict(pill=ss.by(age=...), ...)`.
- **Bug:** `init=ss.by(..., dims=['age', 'ever_used'])` is silently zero: nobody starts on a method (repro_misc 5). The mCPR at the start was 0, against v3's 0.248. Workaround: init by age, then `ss.transfer(ever_used=True, rate=BIG, states=METHODS)`. Users at the start then draw their first decision as never-users.
- **Footgun:** a population split `ss.by(..., dims=['age', 'method'])` with rows summing to 1 is read as a joint distribution. It silently replaces the sim's age split with equal age groups (repro_misc 6: 480/520 for a 0.9/0.1 split). The women 15–49 jumped 8% in 2015 as the flat cohort aged in. 05-dimensions.md says the rows are normalized in one place and a joint split in another; it should be one rule, or an error.

**12. Programs.** v3: `update_methods`, `method_switching`, `change_initiation(_prob)`, `add_method` (`interventions.py`, 1,008 lines).
- *Method mix* (twice the injectable share): `ss.modify(<every *_inj branch>, multiply=2, start=2010)`.
  - **Clean syntax, wrong semantics.** v4 takes the extra probability from every other branch, *non-use included*: mCPR .564 → .614.
  - v3's `method_weights` reweight among methods only: p_use is separate, and its rows are renormalized after weighting.
  - Expressing that needs two-stage decisions (junction states) or hand-computed products.
  - Suggestion: `ss.modify(..., renormalize=[branches])`.
- *Uptake* (v3 intercept shift): `ss.modify(<every *_none branch>, multiply=0.7)`. **Clean**: mCPR .564 → .643, TFR 2.18 → 1.23.
  - A shift on the logit scale is not expressible.
  - Suggestion: `ss.modify(p, logit_add=)`.
- *Capacity* (100 implants a year to non-users aged 20–35):
  - Agents: `ss.treat(ss.Treatment(transition='program'), capacity=ss.peryear(100), eligible=contra.none & age, disease='contra')`, on an undriven edge `none >> impl @ 0`. It delivered 98 a year.
  - Compartmental: `ss.transfer(method='impl', capacity=..., eligible=(fpc.method == 'none') & age)`. It delivered 97 a year in the ODE and 100 as agents.
  - **Clean.** But `disease=` is required (`E506`) with two modules, and a lazy condition cannot name a dimension through `ss.People(1)` (`ppl.method` raises `AttributeError`); `fpc.method` works.
- **Bug:** the arrow form `ss.transfer(method='none -> impl', coverage=...)` raises `KeyError: '"n" is not a level'`: it indexes the string (`interventions.py:766`; repro_misc 7).
- *add_method* (a new method mid-run, e.g. DMPA-SC): **impossible.**
  - States must be declared up front, which is fine.
  - But `set=` and `add=` on a `p=` branch silently do nothing, and `multiply=` cannot lift a zero branch (repro_misc 9). The workaround is `p=1e-6` and `multiply=2e5`.

**13. Results.** Automatic and good:
- counts by state × age × parity;
- deliveries by age, from which TFR and ASFR are 3 lines;
- method mix by age;
- parity distribution.

mCPR (modern methods / women 15–49) is computed in Python from `.by('age')`. Missing without `step()`:
- age at first birth;
- birth intervals and the share of short intervals;
- method failures in the faithful form;
- aCPR.

`sim.solvers()` says "ode: approx (Erlang stages)" for all 120 rows of the faithful model, which then runs out of memory.

## The five questions

1. **A declared module, not `step()`.** Nothing in either formulation needed custom `step()`.
   - Parity is a **stratification** (a dimension moved by `move=` on delivery): exact, and it works under both solvers. As a counter it is impossible.
   - The costs are a chain of states for time since birth, per-level delivery transitions, BIG-rate junctions for events, and no per-woman history.
   - Everything history-shaped (spacing, ages at events, counts by type) needs either `step()` or the time-since-event primitive.
2. **`ss.by` is enough** for the matrices: `p=ss.by(table, dims=['age', 'ever_used'])`. Probabilities depending on parity or postpartum status work the same way. What is not enough:
   - decide at entry, where v3 decides at exit;
   - no age-dependent `Dist` (only scale factors, through `adjust=`);
   - branch modifiers that renormalize over non-use;
   - no `set=` on branches.
3. **Compartmental: plausible only as a rewrite.**
   - The faithful two-module form runs out of memory at init. At 7 ages × 1 parity (an earlier version, before the junction states) it was killed at 6.8 GB after 28 s in `np.kron` in `dim_map` (`solvers.py:714`). Crossing two modules makes each module's compartments a dimension of the other, *including* the generated sub-compartments per branch and Erlang stages. The dense n_groups² maps per flow then explode.
   - The rewrite (one module; method, parity and age as dimensions; no `p=` branches; exponential dwells) runs at:
     - 7 ages × 3 parity levels × 16 method levels × 9 states: about 0.5 s per simulated year (Euler);
     - 10 × 4 × 16: 3 s per year and 2.3 GB;
     - 11 × 5 × 16 (7.9k compartments): 31 s per year and 4.5 GB with RK4.
   - It agrees with its own agent version (mCPR .50 vs .48).
   - Matching the mean duration fails for heavy tails. The implant Weibull (c = 0.29) has a median of 18 months and a mean of 55 years. v4's automatic Erlang approximation matches the mean (k = 1 for CV > 1), so nobody would stop. The port matches 12-month continuation instead.
   - Suggestion: a 2-phase hyperexponential approximation when CV > 1.
4. **`ss.Pregnancy`: replaced, not built on.** v4's `ss.Pregnancy` is v3's class unchanged (`ti_*` timers, callbacks, embryos as agents), so FPmod as a subclass still runs.
   - But it is invisible to the declared world: no states, flows or modifiers, and its newborns enter no declared state (REVIEW §1.1).
   - The declared port drops it, and loses the mother–child link, twins as agents, and newborns from deliveries.
   - Suggestion: a declared `ss.Pregnancy` (non-pregnant → trimesters → postpartum states, a `deliver` transition with a birth side effect) whose rates and states FPsim can override or add to.
5. **Programs:** see 12 above.
   - `ss.modify` on branches covers uptake and mix syntactically; mix has the wrong semantics.
   - `ss.treat`/`ss.transfer` with `capacity=` covers count-based delivery under both solvers.
   - Introducing a method is not possible.

## What the three proposals must cover for FPsim

- **Time since event and within-host state:**
  - Time since last live birth must survive later losses. It feeds activity by month, spacing in 3-month bins to 54 months, and LAM in months 0–5.
  - Time on the current method.
  - Under an ODE, a "clock" dimension: bins of time since an event, generated aging, reset by `move=` on the event. That would replace the PA..PE chain and work under both solvers.
  - Per-woman counters and event ages: number of births, abortions, ages at births, intervals.
  - Persistent random traits: personal fecundity, fated debut age.
- **Program delivery and treatment:**
  - Contraceptive methods as held products with efficacy against named transitions. The efficacy is non-multiplicative (the cap) and depends on route.
  - Discontinuation as the product's waning or dwell, then a re-decision.
  - Branch modifiers renormalized within a subset; logit-scale shifts; `set=` on branches.
  - Introducing a product mid-run.
  - Postpartum delivery `on=fp.<transition>`.
  - "Ever used" as per-person delivery memory.
  - Coverage time series.
- **Transition timing and pregnancy vocabulary:**
  - `decide='exit'` with covariates at exit.
  - Immediate `p=` branches; fix the where-gated `p=` bug and the bias of a `dur=` stage competing with a hazard.
  - `on=`/`delay=` on transitions, so one module's event triggers another module's transition (pause at conception, decide 1 and 6 months postpartum).
  - Births driven by deliveries, with twins and a parent link.
  - Outcome classification by gestational-age threshold.
  - A loss returns the woman to her pre-pregnancy postpartum band.
  - An event with no state change (abortion as an outcome, not a self-loop).
  - Infant death acting on the mother.
  - A declared `ss.Pregnancy` that FPsim extends.

## v3 FPsim bugs found (independent of v4)

1. **Five methods never get their fitted durations.** `SimpleChoice.process_durations` passes `'lnorm'`, but `Method.set_dur_use` tests `'lognorm'` (`methods.py:1290` vs `136`). So none, pill, injectables, condoms and other modern keep RandomChoice's `lognorm_ex(1 y, 1 y)`. Their age factors are attached but unused.
2. **`lognorm_mean_callback` double-indexes:** `age_factors[age_bins]`, then `[age_bins]` again (`methods.py:98-99`). Each agent gets the factor of the agent whose position equals its age bin. Latent, behind bug 1.
3. **Weibull shape is probably used raw.** `c=par1` (`methods.py:156`), while gamma and log-logistic exponentiate their log-space shapes (`:144`, `:150`).
4. **Calibrated fecundity bounds never reach the distribution.** Kenya's `fecundity_low/high` are 0.744/1.650, but `fecundity` stays `uniform(0.7, 1.1)`: `FPPars.process_parameters` only runs in `__init__` (`parameters.py:116-127`). This applies to every location.
5. **A user's `contraception_module=` silently drops the location's calibrated contraception parameters** (`prob_use_intercept`, `prob_use_trend_par`, `method_weights`).
6. **Every woman starts nulliparous.** `parity` defaults to 0 and `ageparity.csv` is only used for calibration. So the nulliparous fecundity factor and debut gating apply to all women at the start, which needs a long burn-in.
7. **`change_initiation.step` compares years exactly** (`years[0] == year`, `interventions.py:757`). It fails on v4 with a `TypeError` at `:778`.

## Ranked

Top problems for this model:
1. **No time since event.** The postpartum chain, its band averages, and losing it after a loss.
2. **Silent engine bugs on the paths FPsim uses:**
   - a where-gated `p=` branch never fires;
   - a `dur=` stage competing with a hazard is biased by about one step;
   - `init` by (age, dimension) is zero;
   - a joint split replaces the age split;
   - `set=`/`add=` on branches do nothing.
3. **Events in one module can't trigger transitions in another.** Hence BIG rates, junction states, and steps lost.
4. **A rate that depends nonlinearly on two modules' states** became 54 modifiers.
5. **Branch semantics:** decided at entry not exit; modifiers renormalize over non-use; no `Dist` with an age-dependent parameter.
6. **Compartmental scale.** Crossing plus branch sub-compartments runs out of memory; dense `dim_map`s; heavy-tailed dwells have no stated approximation; `solvers()` gives no warning.
7. **Births are not deliveries.**

Surprisingly nice:
- The whole life course and the contraception model are declared, with no `step()`.
- 235 lines against about 2,000, and the result matches v3's mCPR and method mix closely.
- Parity, ever-used and method as dimensions, with `move=`.
- Results by age × parity × method come for free.
- `capacity=` on a transfer works identically under both solvers.
- FPsim v3 itself runs on v4 unchanged.
