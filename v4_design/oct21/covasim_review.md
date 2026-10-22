# Covasim: what v4 should change to make the port a joy

Written 2026-10-21, at commit `ee5d1fb0` of `ssv4` (branch `v4-simplify`), run from a clean detached worktree because the `ssv4` working tree was being edited. Covasim 4.0.1 (`75f8f6fe`, Covasim on Starsim 3.7.2) was used as a real model to test v4. No earlier review covered Covasim (the "New use cases" item in `admin/todo.md` is ticked, but no Covasim review was written in `oct17/`), so this is the first one. It also checks whether findings from the October 18 reviews still hold after the simplification plan.

Covasim was run unchanged on both Starsim 3.7.2 and v4, on the same interpreter (the `v4` env). A small shim restored the removed names so the run could get further. Three agents each took one part:
- the disease, its immunity and the parity gates;
- the interventions;
- the framework (sim, runs, calibration, networks, the unchanged test suite, speed).

Each part also prototyped the "joyful" version in v4's own vocabulary. Scripts, logs and the per-part notes are in the Covasim repo under `v4port/` (`disease/`, `interventions/`, `framework/` and their `NOTES.md` files; the shared brief is `v4port/BRIEF.md`). I reran every headline finding myself. A first headline, that v4 gave 3.3 times fewer infections, came from a bug in my own shim and is retracted; the mechanism behind it is CV4.

New findings get IDs CV1–CV18. Everything else is cross-referenced to the October 18 reviews: HPVsim (HP), STIsim (ST), FPsim (FP), [new-findings.md](../oct17/new-findings.md) (N), and the merged findings (D, L), and to this folder's [plan.md](plan.md) (S items, decisions).

## In brief

**What works:**
- **Default Covasim reproduces Starsim 3.7.2 on v4.** 20,000 agents, 120 days, 10 seeds: 21,131 ± 67 infections against 21,108 ± 50. Symptomatic, severe and deaths all agree within 1 SE.
- **Every disease feature matches.** Fifteen natural-history and immunity scenarios run and match within noise: variants, NAbs and waning, `historical_wave`, prior immunity, bed limits, `rel_*_prob`, imports, and dynamic rescaling. Nothing in v4's changes to `ss.Infection` changes Covasim's transmission or prognoses.
- **Interventions match too.** 16 of 17 scenarios give the same relative effect as on 3.7.2; contact tracing is the exception (under "Still open" below). Networks match layer by layer, in edges per agent and in degree variance.
- **It is faster.** Runs take 0.69 times as long as on 3.7.2, and building the sim 0.87 times (interleaved A/B on a loaded machine; every v4 run beat every 3.7.2 run).
- **The declared versions are much smaller and agree with Covasim:**

  | Part | v4 vocabulary | Covasim | Agreement |
  |---|---|---|---|
  | COVID natural history: nine states, age-dependent prognoses, overdispersion, viral load | about 60 lines | about 1,000 | Cohort outcomes exact. The epidemic matches within noise once the latent period is shifted one day (CV5). |
  | Test-trace-quarantine and vaccine rollout | about 90 lines | about 730 | `vaccinate_num('pfizer', 200, sequence='age')` reproduced exactly in every seed; `test_prob` within noise |
  | `cv.Calibration` | `sim.calibrate(gof=cv.mismatch)` plus a 10-line adapter | about 320 | The default `nae` finds the same beta (0.0158) |
  | `MultiSim`, `Scenarios`, `reduce()` | run-axis results, `diff()`, `pars=` sweeps | about 500 of `run.py`'s 1,095 | Not run end to end (CV9 blocks it) |

- **The earlier cascade findings are met for this cascade.** ST14, ST16, HP26 and D31 all held: the cascade ran first time with outcomes as events, remembered results, `delay=`, `trace=`, capacity in people, and doses as levels.

**What stops the port being a joy, in four groups:**

1. **Four new silent wrong answers** (CV1–CV4), and two timing traps that also give silent wrong answers (CV5, CV6):
   - **CV1:** v4's own products do nothing on a coded disease that overrides `infect()`.
   - **CV2:** results compared with `self.ti` are off by a step: `cum_quarantined` is always 0, and `n_infectious` is about 10% high.
   - **CV3:** a delayed test result doesn't stop a retest in the same step, so 16% of positives are repeats.
   - **CV4:** any wrapper around `update_pars()` silently drops every named argument.
   - **CV5:** porting durations unchanged into a declared model puts the epidemic peak 11 days late.
   - **CV6:** `run(until=)` takes one step fewer than before, and the migration skill doesn't say so.
2. **Removals that contradict earlier decisions or records** (CV7). `ss.Calibration` was kept for 4.0 by `oct17/decisions_oct18.md` item 3, but went in S2.1. `sim.current_scale` was promised as an alias (`admin/implementation.md:204`) and appears in no migration table. On its own it causes 55 of the 118 test failures.
3. **Starsim 3 settings and run tools that a package built on Starsim 3 relies on** (CV9–CV10):
   - Covasim's default `verbose=0.1` is refused, so `cv.Sim().run()` fails out of the box.
   - Every Covasim multi-run tool fails.
   - `sim.calibrate()` crashes on Covasim's data (FP8, still open) and can fit only module parameters.
4. **Vocabulary gaps for COVID** (CV11–CV18). The biggest is strains (S7.4): a strain dimension fits Covasim's variants, but it can't take cross-immunity over networks (HP14, still open), antibody-weighted waning protection, or protection against severity. The rest:
   - a "time since an event" clock for waning antibodies;
   - weighted sampling under a capacity (`test_num`);
   - per-layer tracing and quarantine factors;
   - tier-then-score priority;
   - a 3-times slowdown from `ss.by` age lookups.

**The unchanged test suite:**
- On Starsim 3.7.2: 290 passed, 2 failed (both warnings only).
- On v4 with the name shim: 174 passed, 118 failed. By first error: 55 `current_scale`, 24 `verbose`, 31 MultiSim, 2 calibration, and 6 others.
- With the first errors patched as well: 249 passed, 43 failed. These are mostly CV2, CV6 and CV9.

## A sketch of the joyful version

Covasim on v4 should be:
- a declared COVID with a strain dimension for variants;
- Covasim's NAbs as a per-person attribute that wanes with time since infection or vaccination;
- deliveries for testing, tracing and vaccination, each carrying a small Covasim product (diagnosis and isolation, quarantine, NAb vaccine);
- `sim.calibrate()` and v4's run-axis results in place of `cv.Calibration`, `MultiSim` and `Scenarios`;
- `ss.RandomNet(method='uniform')` and `ssl.HybridNet` for the networks (already enough).

The v3 compatibility layer (`compat.py`, `sim.summary`, `sim['beta']`) stays, as the Covasim port's goals require.

The parts that already work, from `v4port/interventions/joy_ttq.py`:

```python
pcr   = ss.Diagnostic(sensitivity=1.0, detects=lambda sim: sim.diseases.covid.infectious, delay=ss.days(1))
symp  = ss.screen(pcr, who=~covid.diagnosed & covid.symptomatic,  coverage=0.2,   interval=ss.days(1))
asymp = ss.screen(pcr, who=~covid.diagnosed & ~covid.symptomatic, coverage=0.002, interval=ss.days(1))
diag  = ss.Delivery(Diagnose(), who=symp.positive | asymp.positive)
trace = [ss.Delivery(Quarantine(), who=diag, trace=lk, coverage=p, delay=ss.days(d) if d else None)
         for lk, (p, d) in dict(h=(1.0, 0), s=(0.5, 1), w=(0.5, 1), c=(0.1, 2)).items()]
due   = vx.since >= 21
dose  = ss.vaccinate(vx, who=~vx.ever | due, capacity=200, priority=1000*due + ss.ppl.age, interval=ss.days(1))
```

Every `/sim.current_scale` in Covasim goes, since capacities are in people. Several things here should become shorter, and the findings below say how:
- the loop over layers (CV16);
- `1000*due + ss.ppl.age` (CV15);
- `Diagnose`, `Quarantine` and the vaccine as code rather than `ss.Vaccine` (CV1, CV16, CV17);
- variants and NAbs as code (CV12, CV13).

## What should change in v4

### Silent wrong answers

**CV1. A product or `ss.modify()` on a coded disease that overrides `infect()` does nothing.**
- **What happens:** `ss.vaccinate(ss.Vaccine(efficacy=1.0), coverage=0.9, at=[0], disease='covid')` reaches 9,000 of 10,000 agents and lowers the effective `rel_sus` to a mean of 0.09. Yet infections are identical, to the last digit, to the sim with no vaccine (8,208, 9,234 and 8,148 for three seeds). The same protection applied Covasim's way gives 63 to 67 infections.
- **Cause:** only `ss.Infection.infect()` multiplies by `self.modified()` (`starsim/diseases.py:770–790`). Its own docstring invites a multi-strain disease to override `infect()` and call `infect_route()` (`diseases.py:758–760`), which is what Covasim does. `Delivery.check_coded()` checks only that `rel_sus` exists.
- **Fix:** apply the modifiers in `infect_route()` (or `Network.transmit()`), which every override goes through. A few lines.
- **Other packages:** check any coded disease with its own `infect()` in STIsim and HPVsim.
- **Cross-reference:** new. S5.5 made products act on coded diseases, but not on overrides.
- **Repro:** `v4port/interventions/probe_vaccine_modifier.py`.

**CV2. `self.ti` in `update_results()` is the end of the step, which silently breaks code shared with `step()`.** Two Covasim results go wrong:
- **`cum_quarantined` is always 0.** 3.7.2 gives 7,630 and 3,581 for two seeds; on v4, 7,462 agents are quarantined and the count is still 0. The cause is `date_quarantined == ti` (`covid.py:1013`).
- **`n_infectious` is 8–10% too high.** The recorded value on day 60 is 5,231, but rebuilding it from the dates gives 4,797; on 3.7.2 the two agree exactly. The cause is a property, `infectious = infected & (ti_infectious <= self.ti)`, used both in `step()` and in `update_results()`.
- **Effect on the gates:** this is the only failure in the natural-history and vaccination parity gates (z = 7.9 and 10.2).
- **What's missing:** the migration skill documents the convention and says "a run warns about a result ... that still looks written the old way", but no warning fired for either result.
- **Fix:** give modules an explicit name for the step just taken (e.g. `self.t.ti_step`, which is k−1), and have `interop/migrate.py` flag date comparisons with `self.ti` inside `update_results()` and inside properties.
- **Cross-reference:** the convention is the migration skill's "`self.ti` in code" row; the shared-property trap and the missing warning are new.
- **Repro:** `v4port/disease/count_check.py`, `quar.py`.

**CV3. A delayed test result released at the start of a step doesn't stop the same step from offering the test again.**
- **What happens:** with `who=... & ~pcr.positive.ever`, 736 of 4,713 positives (16%) are a second positive, exactly one step after the first, and 12% of `delivered` are such retests.
- **Cause:** in `step_deliver()` (`interventions.py:945–960`), `step_waning()` fires `positive`, and then `receive()` evaluates `who=` before the outcome memory is flushed, which happens only at the end of `Delivery.step()`. An `ss.Event` fired by code is remembered at once, so the two kinds of event behave differently.
- **Fix:** flush the memory of outcomes released at the start of the step before choosing who is offered.
- **Cross-reference:** ST14 asked for remembered results; this ordering case is new.
- **Repro:** `v4port/interventions/probe_retest.py`.

**CV4. `update_pars()` finds a module's named arguments by call depth, so any wrapper, decorator or overriding `update_pars()` silently drops all of them.**
- **What happens:** with a one-line wrapper, every Covasim network fell back to the defaults: 10 contacts, dynamic edges, everyone in schools and workplaces. That gave 3.3 times fewer infections, with no error.
- **Cause:** `starsim/modules.py:546–552` uses `getframe(1)` and accepts it only if that frame is an `__init__`. Starsim 3 did the same (`modules.py:518`), so this is not a regression, but a downstream subclass trips it as easily as a shim does.
- **Fix:** find the frame by identity (the nearest `__init__` frame whose `self` is this module), or pass the named arguments explicitly. S6.3 already gathered frame inspection into one helper, `ss.utils.caller_names()`.
- **Cross-reference:** related to N17.
- **Repro:** `v4port/interventions/probe_update_pars.py`.

**CV5. A declared infection is dated at the end of its step, so durations ported unchanged from code that dated it at the start lengthen the first dwell by one step.**
- **What happens:** the declared COVID with Covasim's durations has the same final size, but its peak comes 11 days later (68.9 against 57.9, z = 2.8). Shortening the latent period by one day matches on every summary (|z| < 1).
- **What's wrong with the record:** the migration skill calls the change "negligible at small steps". It isn't for a daily model with a 4.5-day latent period.
- **Fix:** say so in the migration skill and the `import-model` idioms, and print "dwell counted from the end of the infection step" in `describe()`. An option to date at the step's start is another possibility.
- **Cross-reference:** new; related to HP5.
- **Repro:** `v4port/disease/compare_transmission.py`.

**CV6. `sim.run(until=t)` takes one step fewer than in Starsim 3, which the migration skill doesn't mention.**
- **What happens:** v4 stops at the end of the last step (`loop.py:518`), while Starsim 3 stopped after the step starting at `t`. After `run(until=10)`, `sim.ti` is 9 on v4 and 10 on 3.7.2, so code that schedules from `sim.ti` is off by one. It accounts for two of Covasim's intervention test failures.
- **Related:** a whole run also has one step fewer (no step at the first tick), so an intervention dated on the last day never runs. That one is documented (`stop=T`).
- **Fix:** a row in the migration skill, and `Sim.run()`'s docstring saying "the end of the last step taken".
- **Cross-reference:** new for `until=`; the whole-run case is HP5.
- **Repro:** `v4port/interventions/probe_until.py`, `probe_until_ss.py`.

### Removals that contradict the records

**CV7. `ss.Calibration` and `sim.current_scale` went without what the records promised.**
- **`ss.Calibration`:** `oct17/decisions_oct18.md` item 3 kept it for 4.0, because Covasim, HPVsim and STIsim subclass it ("at absolute minimum need to support all the interventions current packages use!"). S2.1 removed it, so `import covasim` now fails.
- **`sim.current_scale`:** it is `_current_scale` since `7c0c8c96`, although `admin/implementation.md:204` says it is "kept as an alias for Starsim 3 modules" and the user guide (`advanced_rescaling.qmd:85`) still documents it. v4 itself reads `_current_scale` (`pregnancy.py:392`), as do the October 18 review ports. It is not in `removed_names.md`, the migration skill or the migration script. It is the first error in 55 of Covasim's 118 failing tests: everything with testing, tracing, vaccination, variants, imports or the analyzers.
- **Fix:** restore `current_scale` as a public property. Three lines; the longer-term answer is "numbers are people" (proposal 02). For `ss.Calibration`, either keep the decision or record its reversal with the replacement (CV10).

**CV8. The smaller removed or renamed names, and the messages for them.**
- **Removed from the root:** `ss.HybridNet`, `ClusterNet` (now `ssl.`), `InfectionLog` (`ss.diseases.`), `multi_run`, `single_run`, `standardize_netkey` (`ss.utils.`), and `Filter`. All but `Filter` are in the migration skill.
- **`ss.people`:** it is now a function, so `ss.people.Filter` is unreachable; use `importlib` or `from starsim.people import Filter`. A small trap.
- **The unknown-keyword error:**
  - It says "2 unrecognized arguments for a: uniform_targets" for one argument, because it counts `len(pars)` rather than `len(remaining)` (`modules.py:570`).
  - It doesn't point to the migration skill, which S2.4 says it does.
- **`uniform_targets`:** the migration skill labels `RandomNet(uniform_targets=True)` "early Starsim 4", but it is in released Starsim 3.7.2.
- **Lazy `auids`:** `ss.People` now creates `auids`, `uid`, `slot` and `parent` lazily in `__getattr__`, so a subclass with its own `__getattr__`, such as Covasim's v3 layer, hides them. Covasim's fix is to fall back to `super().__getattr__`, but the lazy attributes could also be properties, which nothing can hide.
- **The migration script:** `--check` finds CV2's `cum_quarantined`, but not `current_scale`, `run(until=)`, or an `infect()` override that skips `modified()` (CV1). It also raises 8 false alarms on Covasim's own `diagnosed` state, from a check for Starsim 4 prerelease names that were never released.

### Settings and run tools from Starsim 3

**CV9. `verbose`, `MultiSim` and `run(n_runs=)`.**
- **`verbose`:**
  - Covasim's default is `ss.options.verbose = 0.1`, which v4 refuses at `sim.init()` (`parameters.py:651`), so `cv.Sim().run()` fails out of the box. The message is good; the problem is that it is enforced on a setting Covasim's users share with Starsim through `ss.options`.
  - Fix: either read a fraction as progress every `1/verbose` of the run and a negative value as 0, or enforce the check on `ss.Sim(verbose=)` only.
  - 24 test failures.
- **`MultiSim`:**
  - `MultiSim(pars=)` (the values for a sweep) collides with a sim's own `pars`. v4 stores `pars=None` in `run_args`, and Covasim's mirrored `multi_run()` sets `sim['pars'] = None`.
  - The constructor's keyword list is closed, so a subclass can't accept `noise=` or `keep_people=`.
  - There is no hook to change each run's sim before it runs. That hook is why Covasim copies `single_run`/`multi_run`, about 180 lines.
  - `reduce()`, `mean()` and `median()` are gone with no message (`MultiSim._removed` went in S2.4).
  - `combine()` now returns a new sim and leaves the MultiSim unchanged, which breaks a subclass that used its side effect.
  - Fix: rename `pars=` (e.g. `sweep=`); add a `make_run(sim, values, seed)` method as the override point; let a subclass declare extra run options; restore one-line messages for `reduce`, `mean` and `median`.
  - 31 test failures; all of `cv.Scenarios` stands on `median()`. Cross-reference: FP12, and FP7, ST10 and HP12 for subclasses.
- **`run(n_runs=)`:** `ss.Sim.run(n_runs=)` returns a MultiSim, so a subclass's `run()` that returns `self` silently drops every run. One way to do it (S2.8) would have `run()` always return the sim and leave `n_runs` to `ss.parallel()`.

**CV10. `sim.calibrate()` is close to replacing `cv.Calibration`, with three gaps.**
- **What works:** on a `cv.Sim`, `sim.calibrate(gof=<Covasim's mismatch>)` finds beta 0.0158 in 10 trials, and the default `nae` finds the same value.
- **Gap 1, a crash on Covasim's data:** a data column that matches no result crashes at `fit.py:63`, even when `gof` is a function that doesn't read the columns. Covasim's data keep a `date` column. This is FP8, still open.
- **Gap 2, which parameters can be fitted:** only module `pars` can be (`fit.py:1243`), so a coded intervention's settings (`test_prob.symp_prob`) have no spelling. A `custom_fn(sim, values)` hook, or fitting any attribute by path, would cover it.
- **Gap 3, workers:** there are no multi-process Optuna workers or resume (ST27).
- **Fix:** with gaps 1 and 2 closed, `cv.Calibration` becomes about 10 lines.

### Vocabulary for COVID

**CV11. `ss.by()` age lookups make a declared model about 3 times slower.**
- **What happens:** with Covasim's four age-dependent prognosis tables as `ss.by(dict(age=...))`, a run takes 4.7 s, against 1.6 s with constants and 1.0 s for Covasim's coded module. In a profile, `by.lookup()` (`parameters.py:1067`) takes 3.1 of 5.0 s: it is called 904 times and rebuilds the table through pandas `iterrows()` and `keys()` each time.
- **Fix:** cache the value grid and bin edges per sim, and run one `np.digitize` on the agents being scheduled.
- **Cross-reference:** new.
- **Repro:** `v4port/disease/prof_decl.py`, `time_byage.py`.

**CV12. Strains (S7.4): what Covasim's variants need from a strain dimension.**
- **What fits:** Covasim's variants are one infection at a time with per-variant `rel_beta` and `rel_*_prob`. That is a dimension (`kind='strain'`), not a family, and `adjust={'infection': [1, 1.67]}` runs.
- **What a Covasim-like model needs:**
  1. Cross-immunity over networks: `immunity=` still raises `CapabilityError` there (HP14, still open).
  2. Protection weighted by a per-person, waning attribute on a dimension, as `ss.immunity(protected=<attr>)` already allows for a family.
  3. Protection against branches (symptoms, severity), not only `rel_sus`.
  4. "Last strain had", not only "strains ever had".
- **Without these:** variants stay as code, and they are most of `immunity.py` and `connectors.py`.
- **Repro:** `v4port/disease/try_strains.py`.

**CV13. A clock of time since an event.**
- **What's missing:** Covasim's NAbs peak at infection or vaccination (scaled by severity, boosted on reinfection) and wane on a kinetic curve from the last event. Clocks are time in a state or set, so decay after the person is back in `S` can't be written, and a product's `waning=` can't share one level with the disease's antibodies.
- **Fix:** `event.time` (time since this person last had the event), plus `np.interp` among the allowed functions, since kinetics are usually tables.

**CV14. The time until a scheduled exit.**
- **What's missing:** Covasim's high viral load lasts `min(0.3 × time to recovery or death, 4)` days, which depends on an exit scheduled in a later state.
- **Fix:** `(set).progress` for a set whose exits are all scheduled on entry. The prototype used a constant 2.7 days, which is good enough for the epidemic.

**CV15. Priority under a capacity.**
- **Weighted sampling:** `test_num` samples its capacity weighted by odds ratios (symptomatic 100×). `priority=` only orders, which is a different model, with 17% fewer infections here. Suggestion: `priority=ss.weights(expr)`; STIsim's and HPVsim's `*_num` interventions would use it too.
- **Tier then score:** second doses first, then oldest, needed `1000*due + ss.ppl.age`. Suggestion: `priority=[due, ss.ppl.age]`.
- **Shared capacity:** `ss.Resource` went in S7.2, so two deliveries (e.g. a heterologous booster) can't share a daily capacity. One delivery with `doses=` was enough for Covasim.

**CV16. Tracing and quarantine by layer.**
- **Per-layer coverage and delay:** `trace=` takes one `coverage=` and one `delay=`, so Covasim's per-layer `trace_probs` and `trace_time` take four deliveries. Suggestion: `trace=dict(h=1.0, s=0.5, ...)`, with `delay=` taking the same keys.
- **Per-route factors:** quarantine multiplies both susceptibility and transmissibility by a factor per layer, but on a coded disease `ss.modify(..., route=)` can only scale the susceptibility of the person who would be infected (`connectors.py:385–390`). Suggestion: allow `route=` on `rel_trans` too.
- **Quarantine as a held product:** it needs a fixed `dur=`, renewal when notified again, and ending on diagnosis. Until then quarantine stays Covasim's state, and the deliveries only notify.
- **Tracing capacity:** `contact_tracing(capacity=)` counts index cases. That is expressible as an intermediate delivery (untested).

**CV17. A product whose effect is code still needs `efficacy=`.**
- **What happens:** Covasim's NAb vaccine had to pass `efficacy=1e-9`. Also, `ss.Vaccine.doses` (its dose levels) collides with Covasim's per-agent `doses` array.
- **Fix:** allow `efficacy=None` when `administer()` is overridden.

**CV18. Populations from a table, and the mean of a distribution.**
- **Populations:** Covasim refuses `pop_type='synthpops'`, and v4 has no network built from a table of household, school and workplace IDs per agent. A library network from such a table would let it accept them again.
- **Mean of a distribution:** Covasim's `get_contacts()` reads `ss.poisson.pars.lam` to report the mean number of contacts. An exact `ss.Dist.mean()` would remove that.

### Still open

- **Contact tracing is a little weaker on v4.** With layer-specific tracing, infections fall to 0.245 ± 0.006 of the baseline on v4, against 0.210 ± 0.010 on 3.7.2 (16 paired seeds, about 3σ). Quarantine coverage, isolation and contacts traced per diagnosis all match. The epidemic is near threshold under TTQ, where small differences in R are amplified. No cause was found.
- **`daily_stats` source counts:** they are 0 against 20 in the infection log on one day (`test_daily_stats`). This is probably CV2's offset, but not verified.

## What to keep

- **One transmission path for coded and declared diseases (S5.5):** Covasim's transmission, variants, NAbs and severity reproduce 3.7.2 exactly through it.
- **`ssl.HybridNet`, `ssl.ClusterNet` and `ss.RandomNet(method='uniform')`:** they reproduce Covasim's networks to the decimal. Also keep the `__getitem__` that still serves `network['p1']`, which Covasim's v3 layer uses.
- **The delivery vocabulary:** `trace=`, `delay=`, `capacity=` in people, `priority=`, `doses=`, `.ever`/`.since`, and outcomes as events. Covasim's cascade ran first time and reproduced `vaccinate_num` exactly.
- **The declared natural history with `ss.by()` by age:** the outcome tree is exact in about 60 lines; only its speed needs work (CV11).
- **`sim.calibrate()` with `gof=` a function:** Covasim keeps its own goodness of fit exactly.
- **The verbose error message:** it names the v4 spelling; it is applied too widely, not badly worded (CV9).

## Suggested order

1. **Silent wrong answers first** (plan.md's rule for SWAs):
   - CV1: apply modifiers in `infect_route()`, a few lines.
   - CV3: flush the outcomes released at step start before `who=`.
   - CV4: find the `__init__` frame by identity.
   - CV2: `ti_step`, plus the migration warning that is promised but missing.
2. **Records and messages** (S8.2, S8.3): restore `current_scale` (CV7) and decide `ss.Calibration` (keep or record the reversal), then CV8's message fixes and migration-skill rows (CV5, CV6, `uniform_targets`, `current_scale`), and CV8's script patterns.
3. **The verbose check** (CV9): about 5 lines. Together with `current_scale`, it clears 79 of the 118 first errors.
4. **Calibration and runs:**
   - CV10 gaps 1 and 2.
   - `MultiSim`: the `pars=` rename, a `make_run()` hook, and messages for `reduce`/`mean`/`median` (CV9).
5. **Vocabulary:** CV11 (speed), then CV12–CV13 as part of the S7.4 strains decision, then CV15–CV17.

## Index

| ID | Finding | Severity | Earlier |
|---|---|---|---|
| CV1 | Products and `ss.modify` ignored by a coded `infect()` override | Silent wrong answer | New (S5.5 incomplete) |
| CV2 | `self.ti` in `update_results()`: `cum_quarantined` 0, `n_infectious` +10%, no warning | Silent wrong answer | Migration row; trap new |
| CV3 | Delayed result doesn't stop a same-step retest | Silent wrong answer | New (ST14 related) |
| CV4 | `update_pars()` frame by depth drops all named arguments | Silent wrong answer | N17 related |
| CV5 | Declared infection dated at step end: peak 11 days late | Silent wrong answer in a port | New (HP5 related) |
| CV6 | `run(until=)` one step fewer, undocumented | Off by one | New (HP5 for runs) |
| CV7 | `ss.Calibration` and `current_scale` removed against the records | Error | decisions_oct18 #3; implementation.md:204 |
| CV8 | Root names, error count, `uniform_targets` label, lazy `auids`, migrate script gaps | Friction | S2.4 |
| CV9 | `verbose=0.1`, `MultiSim(pars=)`, no per-run hook, `reduce` gone, `run(n_runs=)` | Error | FP12, FP7, ST10, HP12 |
| CV10 | `calibrate()`: data column crash, module parameters only, no workers | Error, missing feature | FP8, ST27, N21 |
| CV11 | `ss.by()` lookup through pandas: 3× slower | Performance | New |
| CV12 | Strain dimension: networks, attribute-weighted, branch protection, last strain | Missing feature | HP14, S7.4 |
| CV13 | Time since an event; `np.interp` | Missing feature | New |
| CV14 | Time to a scheduled exit (`set.progress`) | Missing feature | New |
| CV15 | Weighted sampling, tier-then-score priority, shared capacity | Missing feature | S7.2 |
| CV16 | Per-layer tracing, per-route `rel_trans`, quarantine as a product | Missing feature | New |
| CV17 | `efficacy=` required for code effects; `doses` name | Friction | New |
| CV18 | Populations from an ID table; `Dist.mean()` | Missing feature | New |
