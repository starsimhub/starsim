# Starsim 3: would its users switch?

Everything below was run at `a20c4f7e` with `/software/conda/envs/v4/bin/python`. For the Starsim 3 side I unpacked `main` (3.7.2) with `git archive` into `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/starsim3/v3src` and ran the same scripts against it, so every v3 and v4 number comes from the same interpreter and the same machine. Scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/starsim3/scripts/`. The machine was heavily loaded the whole time (load average 8 to 19 on 20 cores), so treat timings as ±20%.

## Who this user is

They write Python and have a Starsim 3 model behind one or more papers: a custom `ss.Infection` subclass with its own `define_states`, `set_prognoses`, and `step_state` (per-agent CD4 counts, immunity levels, time since infection); `ss.MFNet`, `ss.RandomNet`, maybe a custom network; `ss.Births`/`ss.Deaths`; custom interventions beside `routine_vx`, `campaign_vx`, `treat_num`; a connector that writes another disease's `rel_sus`; `ss.Calibration` with `CalibComponent`s; `ss.MultiSim`; analyzers. Many use a package built on Starsim (STIsim, HPVsim, FPsim, tbsim) or came from Covasim or EMOD for agents, networks, multiple diseases, and the "it's just Python" escape hatch. Collaborators read their code. For them v4 is an upgrade, not a switch, and the two questions are: does my model still run and give the same answer, and what do I get if I move it to the new language?

## The first day

There are no vignettes for this framework, so the first day is the one this user actually has: `pip install -U starsim`, rerun the paper's model, then read what is new.

**It runs unchanged, and faster.** My stand-in for a realistic v3 model (`v3model.py`: library HIV with its CD4 code, `ss.SIS`, a tutorial-style `SIRS(ss.SIR)` subclass, `MFNet` + `RandomNet`, births and deaths, `ssl.ART`, `routine_vx`, `campaign_vx`, a custom `ss.Intervention`, a custom `ss.Connector`, and `ssl.CD4_analyzer`; 3,000 agents, monthly steps, 1995 to 2030) ran under v4 with no edits. Over five seeds:

| | v3.7.2 | v4 | |
|---|---|---|---|
| HIV infections | 189.2 | 193.4 | |
| SIS infections | 12,204 | 12,160 | |
| SIRS infections | 10,686 | 10,643 | |
| Alive at end | 3,304 | 3,295 | |
| Run time, 3,000 agents | 4.9 s | 2.2 s | 2.2x faster |
| Run time, 30,000 agents | 6.0 to 6.4 s | 3.7 s | 1.7x faster |

The statistical answer is the same and v4 is about twice as fast on unchanged v3 code: the most persuasive fact for this user, which the changelog's "no slower" undersells.

**But the numbers are not the same numbers.** The *Regression* bullet at `CHANGELOG.md:28` changes the time-point convention for code modules too: `n_<state>[0]` is now the initial state, a flow at index 0 is 0, and a sim with `stop=T` takes one step fewer. For `ss.Sim(n_agents=3000, diseases=ss.SIR(beta=0.1), networks='random')` (`series.py`):

```
v3  n_infected[:4] = 53, 99, 186, 356     new_infections[0] = 26
v4  n_infected[:4] = 27, 48, 98, 188      new_infections[0] = 0
```

The whole curve moves one step later and the draws diverge after that (cumulative 2,964 vs 2,960). For a model with annual steps (common in HIV, TB, and FP work), the published epidemic peaks a year later in v4. The changelog's opening line (`CHANGELOG.md:7`, "Models written as code ... work as before") is true of whether they run, not of what they print. A user who reruns a paper's script to regenerate a figure will see different numbers with no warning at run time. The new convention is the right one (it is the one every other framework uses, and all four reviews keep it), so this is not a reason to change v4; it is a reason the migration page must say it in its first paragraph, and to keep a 3.7.x branch for reproducing published work.

**Then they read the docs, and the docs are v3.** `docs/` is unchanged from 3.7 except the publish script (`git diff main..v4 -- docs`); fable 12 §7 found 34 of 35 pages run under v4 and none uses a v4 term. `docs/migration/` has `v2_v3/` (paired before/after scripts plus `validate_migration_scripts.py`, a good template) and nothing for v4. On day one that is fine, because their code works; in week two, when they want something new, it is the whole problem.

The best migration document in the repo is not in `docs/` at all: `skills/import-model/references/starsim3.md`, written for an LLM, has a good v3-to-v4 mapping table and the tbsim worked example. It also has two claims that my runs contradict (see below).

## The second week

### Task 1: use one v4 feature on the unchanged model

The things v4 advertises, tried on `v3model.py` (`v4features.py`, `v4features2.py`, `par3.py`):

| Tried | Result |
|---|---|
| `sim.describe()` | Runs; prints "Sim: 15 modules, 3,000 people, steps 1995 to 2030" and "schedule every 0.0833333 steps" (they are years); per disease, "written as code (states: ...)". Little a v3 user didn't know. |
| `sim.solvers()` | A clear table: every module "no (works on agents)". Useful and honest. |
| `hiv.equations()` | `ValueError: "hiv" has no declared transitions` |
| `ss.modify(hiv.rel_trans, multiply=0.5, start=2010)` | `E504 UnmodifiableTarget ... not <FloatArr "rel_trans">` |
| `ss.modify(sis.pars.beta, ...)` on `ss.SIS` | `E504 ... not 0.05 (peryear). A plain number has no name` |
| `ss.modify('sis.rel_sus', ...)` | `E504 ... Quantities that can be modified: randomnet.pars.n_contacts` |
| `ss.vaccinate(ss.Vaccine(0.8), disease='sis')` on `ss.SIS` | `E506 ... "sis", which is not a declared disease in the sim; these are .` (empty list) |
| `sim.calibrate()` | `ValueError: sim.new() is only available for sims with declared states and transitions` |
| `ss.SIS(...)` with `solver='ode'` | Runs as agents, silently (and with no network, nobody is infected); `approx=False` does not object |
| `ss.parallel(sim, n_runs=4)` | Runs (2.7 s); `msim.results` is `None`, so the v4 `msim.results.hiv.prevalence.quantiles()` is unavailable; v3's `msim.mean()` still works |
| `ss.parallel(sim, pars=dict(beta=[...]))`, `msim.diff('baseline')` | `ValueError: ... only available for sims with declared states and transitions` |
| `sim.to_code()` | `NotImplementedError ... cannot yet read a module of type Births` |
| `ss.SIS(beta=ss.par(ss.peryear(0.6), bounds=[0.1, 2]))` | `TypeError: Updating timepar 0.05 ... to par is not supported` |

Eleven of thirteen do nothing for a v3 model, and the errors blame the target ("a plain number has no name") instead of saying "this needs a declared disease". This is the reviews' "two systems" finding (fable §1, opus §1, fable2 theme 1); the rows they did not list are `msim.results` being `None`, sweeps, `diff()`, and `ss.par` being refused by the library classes.

### Task 2: port the model to declared diseases

`v4port2.py` is the same model written in v4: HIV as a class with four states (S, CD4 200 to 500, CD4 below 200, on ART with `rel_trans=0.04`), SIS and SIRS as declared diseases, the connector as two `ss.modify(sis.rel_sus, ..., where=hiv.H)`, the vaccines as `ss.vaccinate(ss.Vaccine(...))`, on the same networks and demographics. It is 43 lines of model against 65 lines of v3 user code plus the 221-line library HIV. Getting a first run took about 20 minutes; getting a run I believed took about two hours, because of three traps, none of which v4 flagged:

1. **A bare number in a `beta` dict changes meaning.** In v3, `beta=0.05` is per year (the default is `ss.peryear`), but `beta={'randomnet': 0.05}` is a per-step probability, used as is (`diseases.py:619`: `beta.to_prob(dt) if isinstance(beta, ss.Rate) else beta`). In a declared v4 disease both are per year. With monthly steps (`probcheck3.py`, SIS on a random network, 5,000 agents, 10 years):

   ```
   v3 ss.SIS   beta=0.05                    11 infections
   v3 ss.SIS   beta=dict(randomnet=0.05)    21,557 infections
   v4 Disease  beta=dict(randomnet=0.05)    11 infections
   ```

   The v3 docs teach the dict form for every network-specific model (`t4_diseases.qmd`: `beta = {'mf': [0.0008, 0.0004]}`), so a verbatim port silently divides transmission by about 12 at monthly steps and 52 at weekly ones. My first port had 33 HIV infections instead of about 190. `describe()` printed the dict but no unit. The fix is `ss.probpermonth(0.008)`; nothing told me. v4's rule is the consistent one; the trap is in v3. The skill's mapping row (`skills/import-model/references/starsim3.md:17`) calls this translation "Exact". No review has this.
2. **"Routine" means something else.** v3 `routine_vx(prob=0.05, start_year=2005)` gives each eligible person a 5% chance per year, so 72.7% are vaccinated by 2030. v4 `ss.vaccinate(coverage=0.05, start=2005)` is "routine" in the v4 sense, offered once at first eligibility: 274 doses, 5.5% (`vx_sem.py`). The v3 behavior is `ss.vaccinate(..., coverage=0.05, interval=ss.years(1))` (3,636 people, 72.7%). The skill's row (`starsim3.md:28`) says "Exact for keyword delivery".
3. **A fixed duration becomes exponential.** v3 `dur_inf=ss.years(1)` is a constant; v4 `ss.progression('I -> S', dur=ss.years(1))` is an exponential with that mean. `describe()` does mark this `[guessed: scalar duration]`, which is the design working as intended, but a v3 user does not expect a guess there.

Smaller: `ss.Transition(I, [R, ss.null], p=[0.99, 0.01])` for the SIR death branch raises `E205`, whose message gives the exact two-transition fix (good). `describe()` labels `ss.prob(0.05, ss.years(1/12))` as "0.05 probability per year" and `ss.prob(0.05, ss.days(7))` as "per day" (the value used is right; the label drops the period).

**What survives.** The structure, the networks (per-network, per-direction `beta`, as agents), `ss.Births`/`ss.Deaths`, the connector (better: see below), and the vaccines. A simple custom `step()` model survives exactly: the tutorial `SIRS(ss.SIR)` subclass against its three-line declared port, five seeds, 5,000 agents: 14,979 infections in v3, 14,973 for the v3 code under v4, 14,890 for the declared port (`sirs_cmp.py`).

**What does not.** Per-agent continuous quantities have no declared form. CD4 (a value that decays each step and scales the death probability) became two states; the SIS immunity level (cumulative, partial, decaying 5% a year) became an immune state with a 20-year mean; ART "offered once, one year after infection" became a rate; `CD4_analyzer` has nothing to record. Each is the standard compartmental move, and each changes the model: the full port gives HIV prevalence 3.9% against 1.8%, 131 on ART against 19, and 3,403 SIS infections against 12,204 (five seeds each). The port would have to be recalibrated and revalidated, which for a published model is a project, not an upgrade.

One of those differences is a v3 bug the port fixed. My v3 connector, written the way the tutorial's is, sets `sis.rel_sus[:] = 1.0` and then raises it for low CD4. That silently erases `ss.SIS`'s own immunity every step: mean `rel_sus` at the end is 1.05 with the connector and 0.10 without, and SIS infections are 18% higher (`noconn.py`). `ss.modify` cannot do this, because modifiers multiply a base that is never overwritten. That is a real, explainable reason for a v3 user to move connectors to `ss.modify` even if nothing else moves.

**What it buys.** As agents, nothing in speed: the declared port is 1.4x slower than the v3 code under v4 at 3,000 agents (2.0 vs 1.4 s) and 1.6x at 30,000 (5.3 vs 3.4 s), interleaved in one process (`speed_ab.py`). Its gains are elsewhere:

- `describe()` on the port is genuinely good: it showed the source of infection, the network each beta goes to, and that v3's `lognorm_ex(mean=10)` has `std=1.0`, a far narrower distribution than most users assume.
- The homogeneous-mixing version of the port (no MFNet, no Births) runs as an ODE with a million people: 11 to 17 ms bare, 0.6 to 0.7 s with either the cross-disease modifiers (which track HIV and SIS jointly, 208 compartments) or the yearly vaccine rounds, 1.4 s with both (`ode_check.py`). Far faster than agents, but a 100x step from adding one delivery.
- `sim.calibrate()` on that ODE, fitting HIV `beta` and the CD4-decline duration to biennial prevalence: L-BFGS-B with JAX gradients, 3.5 s, `beta` 0.430 against a true 0.42, but the duration 0.33 years against 1.2, with a mismatch of 0.0016 and no note that the duration is barely identified by prevalence alone. Adding the low-CD4 count as a second data column recovered both exactly (0.42, 1.2) in 5.6 s (`calib_port.py`, `calib_port2.py`). Against `ss.Calibration` with a `build_fn`, components, and conformers, this is a different world.

### Task 3: migrate one disease at a time

The realistic path for a published model is to keep the hard disease as code and declare the easy one. `mixed.py` and `demo_today.py` keep library HIV (CD4 and `ssl.ART` intact) and declare SIS:

- The mixed sim runs as agents, and `ss.vaccinate(..., disease=sis)` works in it.
- **One declared disease unlocks the run tools for the whole sim**: `ss.parallel(baseline=..., vx=..., n_runs=3)` gives `MultiResults`, `msim.diff('baseline')` works, and a `pars={'sis.beta': [...]}` sweep works. This is the best news in the review for this user, and nothing documents it.
- `sim.calibrate()` on the mixed sim with `beta=ss.par(ss.probpermonth(0.01), bounds=[0.003, 0.1])` on SIS: Optuna, 100 trials in 67 s, `beta` 0.0301 against a true 0.03. Three snags on the way: a data column `prevalence` (both diseases have one) raised `E705 ... matches no result. The closest results are prevalence, prevalence` (it matches two; the fix, `sis.prevalence`, is not in the message); an `ss.par` inside a `beta` dict is not found (`E1301 NothingToFit`); and `calibrate(n_trials=20)`, the Optuna name a v3 user knows, crashes inside Optuna (`got multiple values for keyword argument 'n_trials'`), since v4 spells it `trials=` (`calibration.py:2255`) and v3 spelled it `total_trials=`.
- **The bridge fails at the obvious place**: `ss.modify(sis.rel_sus, multiply=5, where=hiv.cd4 < 200)`, `where=hiv.on_art`, and `where=hiv.infected` all raise `AttributeError: Disease object has no attribute "cd4"` (`cond_bug.py`). A coded module's arrays have no module name when the condition is built, so the lookup at `conditions.py:295-315` searches only declared modules and then reads the attribute from the modifier's own module (SIS). `where=lambda sim: sim.diseases.hiv.cd4 < 200` works. No review has this; it is probably a small fix (record the owner in `define_states`, or resolve by array identity at init).

### Task 4: what tbsim's port says it costs

tbsim's `v1.0.0` branch (`git -C /home/cliffk/idm/tbsim log b92d7e8..v1.0.0`: four commits by Cliff on 2026-10-14 and 15) is the one real port.

- **Size.** 74 files, +3,406 / −12,228. The package went from 8,212 to 3,589 lines; without the unchanged analyzers and plots (1,680 lines), from 6,534 to 1,908. `tb.py` went from 718 lines (an `IntEnum` state, per-agent `rr_*` arrays, a hand-written competing-hazards `transition()`) to 294 lines of declared states and transitions; TPT (546 lines), diagnostics (313), treatments (397), and migration (645) became factory functions returning `ss.Vaccine`/`ss.Test`/`ss.Treatment` and deliveries (`prevention.py`, 83 lines; `care.py`, 210) and two parameters.
- **Gains.** The declared TB matches the LSHTM reference ODE to 1e-4 (`CHANGELOG.md` of tbsim); drug resistance, HIV, BCG, and TPT run under every solver; the ODE default `tbsim.Sim(start='1940', stop='2010')` runs in 0.08 s against 36.8 s for v0.10's default agent sim on Starsim 3 (5,000 growing to 10,128 agents, weekly). The first ODE run in a fresh environment took 34 s, which is Numba compiling the kernels (fable §5 measured 29 s).
- **Costs.** It is a breaking release ("most of the API has changed"). The default model changed from agents on a random network with demographics to a homogeneous ODE of 100,000 people, so `tbsim.Sim(n_agents=5000, ...)` now scales every result by 20. Per-agent heterogeneity became averages: v0.10's BCG drew responders (`p_take`) and per-agent modifiers; v1's is `efficacy = take × (1 − multiplier)` for everyone, the population mean, which loses the frailty effect of responders against non-responders; time-since-infection decline (`exp(-k·τ)`) became a recent/remote latent pair; per-agent reinfection risk became three states. The test suite went from 179 test functions (4,097 lines) to about 51 (1,107). The committed changelog lists features not ported (switching regimen during a course, clearing strains by TPT, resistance acquired under TPT); `v4_design/admin/implementation.md:55` says they were later done; I did not check that (the tbsim working tree has uncommitted changes to those files, and I read only the committed branch).
- **Who did it.** The author of v4, who changed v4 to fit (strains carried together, `move=`, `ss.immigration`, `ss.Crossing`, kernel deliveries; fable2 notes principle 9 was set aside for this). The skill says so plainly: "where a feature doesn't fit, change v4" (`starsim3.md`). A user at a university cannot do that. tbsim is evidence that a well-structured, transition-shaped v3 model ports to a third of its size and gains every solver; it is not evidence of what a port costs someone who cannot edit Starsim, or whose model is built on continuous per-agent state (HPVsim's immunity kinetics, FPsim's individual histories, Covasim's viral load).

## What would win them over

- **"Your v3 model runs unchanged and twice as fast."** 4.9 to 2.2 s on my model, same statistics. Lead the release with it.
- **One declared disease unlocks scenarios, sweeps, `diff()`, and run-dimension results for the whole sim**, alongside unchanged v3 modules (Task 3). This makes migration incremental, and it is the real answer to "is the upgrade worth it": you can take the gain one disease at a time.
- **`ss.modify` instead of connectors that overwrite arrays.** My tutorial-style connector erased SIS immunity silently (18% more infections); `ss.modify` cannot.
- **`sim.calibrate()`**: one call, bounds on the parameters, data by result name, 3.5 s on an ODE with gradients, 67 s with Optuna on agents, versus `ss.Calibration` with a `build_fn`, components, and conformers.
- **The ODE escape**: once a model has no agent-only parts, it runs in milliseconds and can be fitted by NUTS; tbsim's default went from 37 s to 0.08 s.
- **`describe()` on a declared model**, which surfaced two things about my own v3 parameters I had not known (the narrow lognormal default, and which states were infectious).

## What would stop them

| Gap | Kind | Fixable? (size) | Best point on the tradeoff? | In a review? |
|---|---|---|---|---|
| v4 verbs (`ss.modify`, `ss.vaccinate`, `calibrate`, `describe`, `equations`, compartmental solvers, sweeps, `diff`, `msim.results`) do nothing for v3 modules, with errors that blame the target | Missing feature | Partly: `ss.modify`/`ss.vaccinate` on coded `rel_sus`/`rel_trans`, `sim.new()` by copying an unrun sim, `calibrate(pars=...)` on any module's pars: about 1 to 2 weeks. Solvers and `equations()` never | No. The best point is "code and declarations are two lengths of one language": verbs that act on arrays work on both, verbs that need structure say "needs a declared disease" | fable §1, opus §1, fable2 theme 1 (not the run-tool rows) |
| Results shift one step and lose the last step for unchanged v3 code; published numbers do not reproduce | Inherent (convention change) plus docs | Docs and a 3.7.x LTS branch; a compatibility flag is possible but not worth its cost | Yes for the convention; no for how it is announced (`CHANGELOG.md:7` says "work as before") | No (all four keep the convention; none weighs reproducibility) |
| Bare numbers in a v3 `beta` dict are per-step probabilities; in v4 per unit time, a silent 12x to 52x drop on a verbatim port | Docs, plus a missing warning | Small: a note when a v3-style dict of bare numbers is given to a declared transmission in a sim with `dt < 1` unit, and a migration-page row | v4's rule is right; v3's was the inconsistency | No |
| "Routine" delivery means once at eligibility in v4, a yearly probability in v3 | Docs (naming) | A migration row; consider `rate=` on deliveries for "p per year" | Either meaning is defensible; one word should not mean both across versions | No |
| `where=hiv.cd4 < 200` on a coded module raises `AttributeError` naming the wrong module | Bug | Small | n/a | No |
| Continuous per-agent state (CD4, immunity level, time since event) has no declared form; ports must discretize and recalibrate | Inherent tradeoff (declared structure vs arbitrary state) | Partly: a per-agent `ss.FloatArr` with a declared update rule, agents only, would keep CD4 declared | Mostly yes: keeping `ss.Infection` and `step()` first-class is the right escape hatch, as long as the bridge rows above exist | No |
| Declared diseases as agents are 1.4 to 1.6x slower than the same model as v3 code | Performance | Probably (the agent engine for declared models is new) | No: a declared model should be no slower as agents | No |
| Docs are all v3; no migration page; the only mapping is an LLM skill with two wrong rows (`starsim3.md:17`, `:28`) and one wrong claim ("Both versions index results by step in the same way", `:79`) | Docs | Writing, about 2 to 3 days | n/a | fable 12 §7, fable2 09 (not the skill errors) |
| Three trial-count spellings (`total_trials`, Optuna's `n_trials` crashing inside Optuna, `trials`); `E705` saying "matches no result" for an ambiguous column; no note for a weakly identified parameter | Bugs and naming | Small each | n/a | No |
| Ecosystem: STIsim, HPVsim, FPsim are built on v3 classes; tbsim's port needed its author to change v4 | Ecosystem | By keeping the coded-module API first-class and stable | See the migration section | fable2 decisions 1 and 2 |

## The migration path: two languages, or a clean break?

The reviews converge on deleting the v3 vocabulary or moving it to `starsim.legacy` (fable §1, fable2 change 2, opus §1). From this seat that is right for half of what "v3" means and wrong for the other half.

There are two different things called v3. One is the **duplicated vocabulary**: `routine_vx`/`campaign_vx`/`treat_num`/`simple_vx` and the old `Vx`/`Dx`/`Tx`, `ss.Calibration` with its components, `ss.seasonality`, `ss.Samples`, and the coded `ss.SIR`/`SIS`/`SEIR`. These have v4 replacements, and two spellings of the same idea in one namespace is what makes the package feel like two languages. The other is the **coded-module API**: `ss.Module`, `ss.Infection`, `ss.Intervention`, `ss.Connector`, `ss.Analyzer`, `ss.Network`, `define_states`, `ss.FloatArr`, `step()`. That is not legacy. It is Starsim's escape hatch, what STIsim, HPVsim, and FPsim are made of, and the only home for CD4 counts and immunity kinetics. Moving it to a legacy namespace would split the community: the downstream packages would be "legacy" on day one and their users would conclude v4 is not for them.

So I would make the break along that line, and build the bridge that keeps it one language:

1. **Legacy namespace for the duplicates only.** `starsim.legacy` re-exports today's v3 delivery classes, products, `Calibration`, `seasonality`, `Samples`, and the coded SIR/SIS/SEIR (as `legacy.SIR`, so a published script changes one import line). `ss.SIR` becomes the declared class with the same keywords and result names (fable2 change 3; tbsim shows it is feasible). Keep `legacy` for the whole 4.x series, with a `DeprecationWarning` naming the replacement.
2. **The coded-module API stays first-class and is documented as such**, with the v4 verbs that act on arrays reaching it: `ss.modify` and `ss.vaccinate` on a coded module's `rel_sus`/`rel_trans` (through `.effective`, which declared agents already have), conditions on coded arrays (the `conditions.py` fix), `calibrate(pars=dict(beta=[lo, hi]))` on any module's `pars`, and `sim.new()` for any unrun sim. Verbs that need structure (`equations()`, compartmental solvers, `to_code()`) say "needs a declared disease" instead of `E504`.
3. **Make incremental migration the documented path**: run unchanged, then declare one disease at a time, validating each against several seeds of the unchanged model. The mixed sim already works and already unlocks the run tools.
4. **Keep a 3.7.x branch** with bug fixes for a year, for reproducing published results bit for bit.

The tradeoff is codebase simplicity: the coded path and the declared path both stay, so the codebase keeps two engines. But they already exist, and the user-side simplicity (one namespace of verbs, one set of docs, no orphaned ecosystem) is worth more.

### What the v3 to v4 migration page should say

In this order, in the `v2_v3/` format (paired `__v3.py`/`__v4.py` scripts plus a validation script):

1. **Nothing is required.** Your model runs unchanged and about twice as fast. Pin `starsim==3.7.2` only to reproduce published numbers exactly.
2. **Your results move by one step.** `n_<state>[0]` is now the starting state, flows at index 0 are 0, and `stop=T` takes one step fewer; statistically the same (show the five-seed table), not bit for bit.
3. **What needs a declared disease, and what does not.** A two-column table: works on any module (`ss.parallel`, `msim.mean()`, analyzers, `ss.Births`, networks, and, after the bridge, `ss.modify`/`ss.vaccinate`/`calibrate`) versus needs a declared disease (`describe()` in full, `equations()`, ODE and stochastic compartmental solvers, NUTS, `to_code()`, `hash()` meaning anything).
4. **Port one disease at a time**, with a mixed-sim example, and how to validate (seeds, means, the one-step shift).
5. **The mapping table**, from `skills/import-model/references/starsim3.md` with its rows corrected, plus the traps: a bare number in a `beta` dict was a per-step probability (use `ss.probpermonth`, etc.); v3 `routine_vx(prob=p)` is v4 `ss.vaccinate(coverage=p, interval=ss.years(1))`, not v4 "routine"; `dur=ss.years(1)` was fixed and is now exponential unless you give a distribution; `n_infected` is `n_I`; a death branch is two transitions with `p=`; `total_trials` is `trials`; data columns are `disease.result` when two diseases share a result name.
6. **Connectors to modifiers**, with the clobbering example as the reason.
7. **What has no declared form** (continuous per-agent state) and that keeping it as code is fine: say it is a supported choice, not a failure to migrate.
8. **The tbsim port** as the worked example, with its real numbers: a third of the code, every solver, 0.08 s instead of 37 s, and the model changes it made (averaged BCG response, recent/remote latency, states instead of per-agent risk) stated as such.

## Verdict

- **Today:** they upgrade (it is free and twice as fast) but do not switch: every v4 feature they try on their model fails, the docs do not mention v4, and a verbatim port of their betas is silently 12x too small.
- **After the four reviews' fixes:** most would start porting their simplest disease, if the fixes include the bridge for coded modules and the migration page; deleting the v3 vocabulary without the bridge would push the STIsim/HPVsim/FPsim community to stay on 3.7.
- **A clear win:** v4 verbs that work on coded modules, incremental migration as the documented path, and a declared model as fast as code when it runs as agents.

## The demo that would sell it

Take the paper's v3 sim, change nothing about the hard disease, declare the easy one, and get v4's tools:

```python
import starsim as ss, starsim.library as ssl

hiv = ssl.HIV(beta={'mf': [0.008, 0.004]}, init_prev=0.05)        # the paper's v3 module, CD4 code and all
sis = ss.SIS(beta=ss.par(ss.peryear(0.6), bounds=[0.1, 2]))        # the library SIS, now declared
low_cd4 = hiv.infected & (hiv.cd4 < 200)                           # a condition on v3 arrays

sim = ss.Sim([hiv, sis], networks=ss.MFNet(), n_agents=20_000, start=1995, stop=2030,
             connectors=ss.modify(sis.rel_sus, multiply=5, where=low_cd4),
             interventions=ssl.ART(year=[2000, 2020], coverage=[0, 0.8]),
             data='sis_prevalence.csv')
sim.describe()                                                     # HIV as code, SIS in full, the modifier, the schedule
fit = sim.calibrate()                                              # fits SIS beta by Optuna, as agents

vx = ss.vaccinate(ss.Vaccine(efficacy=0.8), disease=sis, coverage=0.05, interval=ss.years(1), start=2025)
halve = ss.modify(hiv.rel_trans, multiply=0.5, start=2025)         # a v4 verb on the v3 disease
msim = ss.parallel(baseline=fit.sim.new(), vaccine=fit.sim.new(interventions=vx), prep=fit.sim.new(interventions=halve), n_runs=20)
msim.diff('baseline').plot()                                       # infections averted, with intervals
```

Today, with the SIS written as an `ss.Disease` and the condition as a lambda, each of the calibration, the vaccine scenario through `ss.parallel`, and `diff()` ran in Task 3 on a 2,000-agent version (not as this one script). What fails: `ss.SIS(beta=ss.par(...))` (`TypeError`), `where=hiv.cd4 < 200` (`AttributeError`), and `ss.modify(hiv.rel_trans)` on the v3 disease (`E504`). All three are the bridge in item 2 of the migration path; none needs a new design.
