# The user's experience, end to end

I reviewed Starsim v4 as its user, not as a maintainer: I typed in the overview's "whole language in one model" example (`v4_design/design/00-overview.md`, 141 lines) and the two-line short form exactly as written and ran them step by step; wrote an SEIR with a latent period, case fatality, vaccination of over-65s, and data from scratch as a Starsim 3 modeler who had read only the overview; ran the code cells of every tutorial and user-guide page (`docs/tutorials/*.qmd`, `docs/user_guide/*.qmd`, 35 pages, 290 cells) under v4; and read `18-conformance.md`, `03-states-and-transitions.md`, `08-interventions.md`, the naming and `describe()` sections of `01-objects.md`, `17-vocabulary.md`, `19-open-questions.md`, the departures and known issues in `v4_design/admin/implementation.md`, the v4 section of `CHANGELOG.md`, and three vignettes (`epydemix/sir_kenya`, `flepimop/vaccine_scenarios`, `summer2/seir-age-clinical`) with their authors' findings. Scripts and their output are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/ux/` (`s1_overview.py` through `s11_coexist.py`, `docs_run/*.log`). One note on the material: `v4_design/examples/` contains only its README; the nine build-set models it describes are not in this repo, so the only worked v4 models a user can copy are the vignettes in the other repo, the changelog, and the spec pages.

## Summary

1. **The overview example does not run as written.** Of its nine steps, two fail: `ss.parallel(baseline=fit.sim, ...)` raises `AlreadyRunError: Simulation is already complete / Did you mean to copy the sim before running it?` (copying does not help either), and `msim.plot()` on `ss.parallel(ss.Sim(sir, solver='stochastic', **kw), n_runs=100)` raises `ValueError: No legend handles found`. The fix that exists, `fit.sim.new()`, is not mentioned by the error or the overview. Let `ss.parallel` and `MultiSim` accept a run sim (take `sim.new()` of it), make the error say `sim.new()`, and fix the plot. Not breaking. (Findings 1, bugs 1 and 2.)
2. **The two-line SIR runs flat.** With no `init_prev`, nobody starts infected, and `ss.Sim(sir, stop=100).run().plot()` draws six horizontal lines with no warning; `describe()` says `init: S = 1.0 [guessed: nothing given]`. A disease with a contact-mediated transition and no initial condition should seed something (one agent, or 1%) and say so as `[guessed]`, or at least warn at run time. A default changes; not breaking. (Finding 2.)
3. **Names are picked up from the caller's variables at two different moments** (transitions when `ss.Disease()` is called, the disease when `ss.Sim()` is called), so `ss.Sim(make_model(), ...)` produces `sim.results.disease` instead of `sim.results.seir`, transitions passed inline are called `S_to_E` and `I_to_R_D`, and `stratify(adjust={'infection': ...})` then fails with `E301` because the transition is not called `infection`. Name every object at its own construction, once, and never at the `ss.Sim` site; fall back to the class name loudly. Breaking only for code that relies on the Sim-site lookup. (Finding 3.)
4. **A `vaccinated=[False, True]` stratification starts with half the population vaccinated** (`[guessed: equal across levels]`): the spec's own "vaccination as a stratification plus a transfer" example (`08-interventions.md`) puts 499,500 of 999,000 susceptibles at `vaccinated=True` on day 0. A dimension that nothing in the model feeds except a transfer should start everyone at its first level. A default changes. (Finding 4.)
5. **`solver='ode'` is silently ignored for a Starsim 3 code module.** `ss.Sim(diseases=ss.SEIR(), solver='ode')` runs as agents: `sim.pars.solver == 'ode'`, `sim.solver == 'abm'`, nothing printed, `approx=False` raises nothing, and `describe()` prints no solver line at all. The spec promises `E804 CapabilityError`. Bug. (Bug 3.)
6. **The first wrong guesses a v3 user makes get Python's errors, not Starsim's**: `0.1 * ss.null` → `TypeError: unsupported operand type(s) for *: 'float' and 'Null'`; `ss.death('I ->', p=0.1)` → `TypeError: death() got an unexpected keyword argument 'p'`; `ss.Vx(pars=dict(efficacy=0.8))` → `TypeError: Timeline.__init__() got an unexpected keyword argument 'pars'`; `ss.routine_vx` on a declared disease → `KeyError: 'No module named "None" with declared transitions was found'`; `sim.init(force=True)` → `AttributeError: 'SimPars' object has no attribute 'people'`. Each deserves a coded error that names the v4 spelling. Not breaking. (Finding 6.)
7. **The documentation is Starsim 3's, and it still runs** (34 of 35 pages execute unchanged under v4; the one failure is a notebook-only `display()` call), **but it does not contain a single v4 term**: no page mentions `ss.Disease(`, `>>`, `solver=`, `describe()`, `ss.modify`, `ss.vaccinate`, `ss.transmission`, or `data=`. The only v4 documentation is a 2,000-word changelog entry. A user who finds the docs writes v3 code and never meets v4; a user who finds the spec meets an example that does not run. One "v4 in ten minutes" page and a v3-to-v4 migration page are worth more than any API change in this report. (Finding 7, with the table.)

Two things to protect: `describe()` is excellent and should be the center of the docs; and the three forms of writing a transition really do all work, first time, including branching with weighted destinations.

## Simpler for the user

### 1. The overview example fails at steps 8 and 9

**What happens.** `s1_overview.py` types the example in verbatim, with a synthetic `weekly_cases.csv` (`date`, `cases`). Steps 1 to 7 (ingredients, disease, sim, `describe()`, `run()`, `plot()`, `calibrate()`) run. Step 8:

```text
######## scenarios
/software/conda/envs/v4/lib/python3.13/site-packages/multiprocess/popen_fork.py:67: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  File "/home/cliffk/idm/ssv4/starsim/run.py", line 836, in single_run
    sim.run(**run_args)
  File "/home/cliffk/idm/ssv4/starsim/sim.py", line 1833, in run
    raise AlreadyRunError(errormsg)
starsim.sim.AlreadyRunError: Simulation is already complete
Did you mean to copy the sim before running it?
```

`fit.sim` is a run sim (`fit.sim.complete == True`). The message suggests copying; `sim.copy()` and `sc.dcp(sim)` give the same error (`s3_checks.py` section D). `sim.init(force=True)` and `sim.init(reset=True)` raise `AttributeError: 'SimPars' object has no attribute 'people'`; `sim.reset()` and `sim.run(reset=True)` do not exist. What works is `fit.sim.new()` (`s11_coexist.py` c5), which nothing the user sees mentions. Step 9:

```text
######## msim (n_runs=3)
solver: tau (exact Gillespie would need about 1e+08 events)
  File "/home/cliffk/idm/ssv4/starsim/run.py", line 659, in plot
    leg = sc.movelegend(fig.axes[0], ax, **kw.legend)
ValueError: No legend handles found in Axes(0.125,0.653529;0.227941x0.226471)
```

With a model whose figure has no spare subplot the same call succeeds with a `UserWarning: No artists with labels found to put in legend` and draws one line per result with an empty legend box (`s4_msim.png`): no median, no interval, nothing that says there were three runs. The spec says `msim.plot()` draws "the median and an interval".

**Why it matters.** This is the page the spec tells everyone to read first, and the first thing a user will paste. Also, the JAX fork warning after a calibration is real: `implementation.md` lists the deadlock as a known issue, so the overview's own sequence (calibrate, then `ss.parallel`) is the one that can hang.

**Proposal.** (a) `ss.parallel`, `ss.MultiSim`, and `sim.run()` on a complete sim should do `sim.new()` for a declared model (and say so once), rather than raise; the `AlreadyRunError` text should name `sim.new()`. (b) Fix the MultiSim plot for results with a run dimension: median line and quantile band, which `MultiResult.quantiles()` already computes (`s10_misc.py`). (c) After a gradient fit, `ss.parallel` should default to serial or spawn, since the library knows JAX has been imported. **Cost.** None breaking. **Confidence.** High.

### 2. The two-line short form runs, and nothing happens

**What happens.** `s2_short.py` is the short form verbatim:

```text
solver: ode (every part of the model has a compartmental form)
--- describe after run
Sim: 1 module, 10,000 people [guessed], steps 0 to 100
...
             init: S = 1.0                                   [guessed: nothing given]
```

and `sim.results.sir.n_I.values.max() == 0.0`; the plot (`s2_plot.png`) is six flat lines. The spec's own `describe()` transcript in `01-objects.md` shows exactly this `init: S = 1.0 [guessed: nothing given]`, so it is as specified, but the overview sells this as "the whole disease in two lines" and it is a model of nothing.

**Proposal.** A disease with a contact-mediated transition and no `init`/`init_prev` should seed one case (the ODE equivalent of one agent, or `init_prev=0.01`) and print `init: I = 1 [guessed: no initial condition given; nothing would happen otherwise]`. If that is too magical, warn at `run()` when every transition's flow is zero for the whole run (`W2xx NothingHappened`), which is cheap and catches many other mistakes (a `beta` in the wrong unit, a `where=` that selects nobody). **Cost.** A default changes for the two-line form only; the spec transcript changes. **Confidence.** High that something should change; medium on which.

### 3. Names come from the caller's variables, at two moments, and the user cannot predict the result

**What happens.** Transitions and states are named when `ss.Disease(...)` is called, from the variables in the calling frame (`starsim/diseases.py:74` → `transitions.py:1075 find_name`); the disease itself is named when `ss.Sim(...)` is called, from *that* frame (`starsim/sim.py:73` and `:109-121 _name_modules`). Modifiers, deliveries, products, and `PartnerNet` are also named at the Sim site. Observed (`s3_checks.py`, `s4_checks.py`, `s8_full.py`, `s9_split.py`):

| Code | Disease name | Transition names |
|---|---|---|
| `sir = ss.Disease(...)` then `ss.Sim(sir)` at module level, or inside one function | `sir` | from the variables |
| `seir = make(); ss.Sim(seir)` (variable in the Sim's frame) | `seir` | from `make()`'s variables |
| `ss.Sim(make(), ...)` | `disease` | from `make()`'s variables |
| `d = dict(sir=ss.Disease(...)); ss.Sim(d['sir'])` | `disease` | |
| `x = ss.Disease(...); y = x; ss.Sim(y)` | `disease` (two names, so none) | |
| `ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(5), I >> (0.9*R + 0.1*D) @ ss.days(10))` | | `S_to_E`, `E_to_I`, `I_to_R_D` |

The last row then makes `seir.stratify(vaccinated=[...], adjust={'infection': [1, 0.2]})` fail with `E301 UnknownTransition: adjust= names a transition "infection" that "disease" does not have. Its transitions are S_to_E, E_to_I, I_to_R_D`. In the first table's third row, `sim.results.seir` raises `AttributeError: 'Results' object has no attribute 'seir'` with no hint that the module is called `disease`. The `print(sir)` right after construction says `disease(pars=[dur, beta]; states=None)` in every case, because the name is not known yet, so the user cannot even check.

**Why it matters.** Result keys, `ss.modify` targets, data columns (`sir.cases`), and the model hash all depend on names, and the rule for a module is "whatever variable happens to hold it in the function that calls `ss.Sim`". Every real model is built in a function (`vignettes/flepimop/vaccine_scenarios/model.py` names its disease explicitly, and `summer2/seir-age-clinical/compare.py` passes `name='seir'`, for this reason). `19-open-questions.md` B12 already flags the mechanism as "slightly magical".

**Proposal.** One rule, one moment: an object is named where it is constructed. For a module, that means the frame of `ss.Disease(...)` (where `sir = ss.Disease(...)` is written), never the `ss.Sim` frame; if no single variable holds it there, use `name=` or the class name and print `[guessed: class name]` in `describe()`. Keep the keyword form (`ss.Disease(infection=...)`) as the documented spelling for anything built in a helper, and make `E301`'s message say "transitions passed without a variable or keyword are named from their endpoints". Rename `ss.Disease`'s fallback from `disease` to something that cannot be mistaken for a user's name, or refuse (`E2xx UnnamedModule`) the way `E210` refuses for states. **Cost.** Breaking for code that relies on the Sim-site lookup (I found none in the vignettes; the tests may). **Confidence.** High.

### 4. Vaccination as a stratification starts half the population vaccinated

**What happens.** The spec's recommended form (`08-interventions.md`, "Vaccination as a move between strata"):

```python
seir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.2]})
rollout = ss.transfer(vaccinated=True, coverage=0.8, years=[2021], eligible=ppl.age > 65)
```

gives, in `describe()`, `dimension    vaccinated (plain): False, True   [guessed: equal across levels]`, and at `t=0` (`s10_misc.py`) `n_S: unvaccinated 499500.0, vaccinated 499500.0`. Half the country was vaccinated before the rollout. The product form (`ss.vaccinate(ss.Vaccine(efficacy=0.8), ...)`) gets this right: its `has_vaccine` dimension starts everyone at `False`.

**Proposal.** A dimension whose levels are booleans, or that is the target of a transfer and of nothing else, starts everyone at its first level; print `[guessed: everyone starts at False]`. More generally, "equal across levels" is a safe guess for `sex` and a bad one for almost anything else; a stratification that is not `age`/`sex`/`patch`-like with a known split should probably require `split=`. **Cost.** A default changes. **Confidence.** High.

### 5. The product form silently makes the model agent-based, and the three vaccination forms are not ranked

**What happens.** Three ways to say "80% of over-65s in 2021" all run (`s7_seir_v4.py` V1 to V3). V1, `eligible=lambda sim: sim.people.age > 65` with no age stratification, prints nothing at run time beyond `solver: abm (...)`, and `describe()` says:

```text
solver       abm   [guessed: "vx" does not work on groups (people.age was asked for, but it is a per-agent attribute and not a stratification, so it does not exist in a compartmental run.)]
             10,000 agents, each standing for 100 people     [guessed]
```

That is a good sentence, but the user asked for a million-person model, has now got 10,000 agents, and the only sign at `run()` was one word. V2 (`ppl = ss.People(1); eligible=ppl.age > 65` plus `stratify=dict(age=[0, 65])`) stays an ODE, but requires the odd incantation `ss.People(1)` (a known issue in `implementation.md`) and an age stratification whose only purpose is a bin edge. V3 is finding 4. The spec says the choice between `ss.transfer` and `ss.vaccinate` is open (`19-open-questions.md` B6); the user has to pick between three spellings with different defaults and different solvers.

**Proposal.** (a) When `'auto'` falls back from ODE to agents, print the reason at `run()`, not only in `describe()`, and print the agent count: `solver: abm (vx asked for people.age; 10,000 agents stand for 1,000,000 people)`. (b) A lazy condition should not need `ss.People(1)`: expose `ss.people.age` (or `ss.cond.age`) as the spec's own "no global condition names" rule permits a module-scoped object; or let `eligible='age > 65'`. (c) Document one form as *the* way to vaccinate (the product form, since it starts from zero and reports `delivered`/`coverage`), and keep the transfer for things that are not products. **Cost.** (b) adds one name; (c) is documentation. **Confidence.** Medium-high.

### 6. The wrong guesses get Python errors

Each of these is a reasonable first attempt by someone who has read the overview or who knows Starsim 3 (`s5_seir_v3style.py`, `s7_seir_v4.py`, `s4_checks.py`, `s11_coexist.py`):

| The user writes | Gets | Should get |
|---|---|---|
| `I >> (0.9*R + 0.1*ss.null) @ ss.days(10)` | `TypeError: unsupported operand type(s) for *: 'float' and 'Null'` | "a weighted destination cannot include ss.null; write a separate `ss.death('I ->', ...)` or a state `D`" |
| `ss.death('I ->', p=0.1)` | `TypeError: death() got an unexpected keyword argument 'p'` | "a death with a probability is a branch: `ss.progression('I -> R \| D', dur=..., p=[0.9, 0.1])`" |
| `ss.Vx(pars=dict(efficacy=0.8))` (Starsim 3's idiom) | `TypeError: Timeline.__init__() got an unexpected keyword argument 'pars'` | "`ss.Vaccine(efficacy=0.8)`; products no longer take `pars=`" |
| `ss.routine_vx(product=ss.Vx(efficacy=0.8), ...)` with a declared disease | `KeyError: 'No module named "None" with declared transitions was found; these are sir'` | "`ss.routine_vx` is for diseases written as code; use `ss.vaccinate(...)`" |
| `ss.vaccinate(...)` with `ss.SIR()` | `E506 UnknownDisease: "vx" needs to know which disease it is for, and the sim has no declared diseases. Use disease=.` | fine, but `disease=` is not in the delivery keyword table of `08-interventions.md` |
| `sim.init(force=True)` on a run sim | `AttributeError: 'SimPars' object has no attribute 'people'` | `sim.new()` |
| `ss.Sim(seir, ...)` built in a helper, then `sim.results.seir` | `AttributeError: 'Results' object has no attribute 'seir'` | "...the modules are: disease" |

By contrast, `E210 UnnamedState`, `E301`, `E705 UnknownDataColumn` (which lists the closest names and both fixes), and `E506` are exactly what the spec asks for. **Proposal.** Wrap the operator and constructor entry points so that a `TypeError` from an arithmetic or keyword mistake becomes a coded error naming the fix; `Null.__rmul__` can raise `E2xx` itself. **Cost.** None. **Confidence.** High.

### 7. The docs: nothing is broken and nothing is there

I extracted every `{python}` cell of the 35 pages and ran each page as a script under v4 (`run_docs.sh`, `docs_run/*.log`). Only `t5_networks` fails, on `display(uid0_edges)`, which exists only in a notebook. Everything else runs, including calibration, `Samples`, profiling, and `MultiSim`. So the docs are not *wrong* in the sense of erroring; they are wrong in the sense of describing a different language. The table says what each page would have to change for v4 (I did not rewrite anything):

| Page | Runs under v4 | What is wrong or missing for v4 |
|---|---|---|
| `t1_intro` | yes | The "hello world" is `dict(type='sir', ...)` + `dict(type='random')`. The v4 hello world is the two-line `ss.Disease` (once finding 2 is fixed). No mention of `solver`, `describe()`. |
| `t2_model` | yes | Entirely `ss.SIR(pars=...)`/`ss.RandomNet`; results accessed via `sim.results.sir.n_infected` (v3 names) while a declared SIR gives `n_S`/`n_I`/`infection`. Needs the "which form when" guidance. |
| `t3_demographics` | yes | `ss.Births`/`ss.Deaths` only; v4's `ss.birth('-> S', rate=)`/`ss.death()` absent, and how they relate is undocumented. |
| `t4_diseases` | yes | Subclassing `ss.Infection` with `step_state()` is the only way shown to add a state; v4's `D` state plus `p=` branch is three lines. |
| `t5_networks` | no (`display()`) | `ss.RandomNet` etc. still valid; nothing on `ss.contacts()`, `ss.MixingPool` as the ODE's mixing, or that a network forces `abm`. |
| `t6_interventions` | yes | `ss.routine_vx`, `ss.Vx(pars=...)` in prose; `ss.vaccinate`/`ss.Vaccine` absent; the two do not mix (finding 6). |
| `solutions` | yes | Follows the tutorials. |
| `intro_starsim`, `intro_models` | no code | Say "Starsim is an agent-based framework"; v4 is not only that. |
| `basics_sim`, `basics_model`, `basics_parameters` | yes | `pars=dict(...)` everywhere; `ss.par`, `bounds=`, `unit=`, "time in timesteps when nothing has a unit" absent. `basics_sim` shows `sim.pars` with `dt=1 year` defaults that silently apply to code modules even with dates (`s5_seir_v3style.py` a2: a 2-year daily SEIR ran as 3 yearly steps). |
| `basics_people`, `basics_results` | yes | `n_infected`/`new_infections` naming; nothing on `result.select()`, `.by()`, `data=`, outcomes, or the run dimension. |
| `modules_adding`, `modules_diseases` | yes | Class-with-`step()` is the only way shown; v4's class form (`S = ss.State()` in a class body) absent. |
| `modules_networks`, `modules_demographics`, `modules_connectors`, `modules_analyzers` | yes | Still accurate for code modules; nothing on `ss.modify` as the replacement for most connectors. |
| `modules_interventions`, `modules_products` | yes | As `t6`. `ss.Vx(pars=...)` in prose now fails (finding 6). |
| `modules_library` | yes | Fine as is (library models are code). |
| `advanced_*` (7 pages) | yes | Accurate for agent-based runs; `advanced_time` does not say that a declared model with no unit runs in steps; `advanced_nonstandard` is where "compartmental" lived in v3 and now contradicts v4. |
| `workflows_run` | yes | `ss.parallel(sims)` positional; the keyword form and `n_runs=` with a run dimension absent. |
| `workflows_calibration`, `workflows_sir_calibration` | yes | Optuna `ss.Calibration` with `calib_pars`; `sim.calibrate()`/`bounds=`/`data=` absent. |
| `workflows_samples`, `workflows_deploy` | yes | Unaffected. |

Also: `docs/migration/` has `v2_v3` and `v3.0_v3.1.md` and nothing for v4; `CHANGELOG.md`'s 4.0.0 entry is a single 2,000-word block of bullets, which is currently the whole of the v4 user documentation. `18-conformance.md` §14 claims every documentation example is run in CI with its output compared; nothing of the sort exists for the spec's own examples (which is how the overview got to fail).

**Proposal.** Before any further API work: one page with the overview example *as it actually runs* (with `sim.new()`), one v3-to-v4 migration page (the table in finding 9), and a "which form when" paragraph at the top of `t2_model`. Then run the spec's examples in CI as §14 says. **Cost.** Writing. **Confidence.** High.

### 8. What `describe()`, `run()`, and `plot()` print: good signal, uneven noise

- **`describe()`** is the best thing in v4. For the SEIR it is 20 lines, every guess is marked, the expanded rate `0.3 × I.n / N [homogeneous mixing]` answers the question every modeler asks, and the `[guessed: "vx" does not work on groups ...]` sentence is a model error message. Three blemishes: it prints the init notes twice when called before `run()` (the `note:` lines at the top from `init()`, then again as `note` rows); for a code module it degenerates to two lines (`Sim: 1 module, 10,000 people [guessed], steps 2020.01.01 to 2022.01.01` / `schedule every step: seir`) with no solver, unit, or dt line, which is exactly when the user most needs to see `dt = 1 year`; and the schedule lists the product as a module (`every day: vx, vaccine, seir`), which is an implementation detail.
- **`run()`** on an ODE prints two lines (`solver: ode (...)`, `note: unit='day' (...)`), and on the same model with `solver='abm'` prints 73 progress lines for a two-year daily run (`s8_out.txt`), plus `Initializing sim with 2000 agents` and a `rate × dt` note. Starsim 3's `verbose=0.1` default is still in force for agents and not for compartments, so the two solvers feel like two libraries. If `ss.parallel` has to re-init, it prints the solver line again for every sim (`s1_out.txt`).
- **`print(sim)`** gives `Sim(n=3; 2020.01.01—2021.01.01; diseases=disease)` for a 50-million-person ODE; `n=3` is the number of age groups. **`print(fit)`** is a 20-line `sc.prettyobj` dump; `fit.summarize()` is the four lines the user wants. **`print(msim.results)`** gives `[sc.prepr(): not shown, recursion exceeded (6>5)]` for every result (`s3_out.txt`).
- **`plot()`** does the right thing with data (gray points over the line), but a weekly `every=ss.weeks(1)` outcome is drawn as a daily series that is zero six days in seven, so the cases panel is a comb of vertical spikes (`s8_cases_ode.png`); the data points sit on the spike tips. Plot it as a step or at reporting days. Panel titles are `seir: S` and `seir: Prevalence` (capitalized differently because one is a state and one a legacy result).

**Proposal.** One verbosity rule for all solvers (a progress bar only above some wall-clock time, say 2 s); `describe()` always prints solver, unit, and `dt`; `Sim.__repr__` says people, not groups; `Fit.__repr__` = `summarize()`; fix the MultiSim results repr; step plot for `every=` outcomes. **Cost.** None breaking. **Confidence.** High.

### 9. Two generations of the API coexist and the user has to pick without guidance

The `ss` namespace has about 330 public names. For the same four jobs there are now two spellings, and whether they mix is not stated anywhere a user would look (`s11_coexist.py`, `s5_seir_v3style.py`):

| Job | Starsim 3 | v4 | Together? |
|---|---|---|---|
| A disease | `ss.SIR(pars=dict(beta=...))`, `ss.SIR(beta=...)`, `dict(type='sir')` | `ss.Disease(...)`, `S >> I @ ...`, class with `ss.State()` | Yes, in one sim; but `solver=` is ignored for the first (bug 3) and results are named differently (`n_infected` vs `n_I`) |
| Mixing | `ss.RandomNet()`, `ss.MFNet()`, `dict(type='random')` | `ss.contacts()`, `ss.MixingPool`, none (homogeneous) | Yes; a v3 network with a declared disease quietly makes the sim `abm` |
| Vaccination | `ss.routine_vx(product=ss.Vx(...), prob=, eligibility=lambda sim: ...)` | `ss.vaccinate(ss.Vaccine(...), coverage=, eligible=cond)` | No: `routine_vx` on a declared disease → `KeyError ... "None"`; `vaccinate` on `ss.SIR` → `E506` |
| A parameter change | `ss.Intervention` subclass, or `sim.pars`/`disease.pars` edits in `step()` | `ss.change`, `ss.modify` | `ss.modify(sir.pars.beta, ...)` works on a declared disease; on a code disease untested here |
| Scenarios | `ss.parallel(s1, s2)` positional, `ss.MultiSim([...])` | `ss.parallel(baseline=..., masks=...)`, `n_runs=` | Yes |
| Calibration | `ss.Calibration(calib_pars=..., eval_fn=...)` + Optuna | `ss.par(bounds=)`, `data=`, `sim.calibrate()` | Both exist; the first still documented, the second not |
| A product | `ss.Vx(pars=dict(efficacy=))`, `ss.Dx`, `ss.Tx` | `ss.Vaccine`, `ss.Test`, `ss.Treatment` | `ss.Vx(efficacy=0.8)` now *returns* a `Vaccine` (`products.py:173`), `ss.Vx(pars=...)` fails |

The changelog says "Models written as code work as before", which is true, and the implicit message is "code modules are the old way". The spec (`01-objects.md`, "Library models") says `ss.SIR(beta=0.1)` is "kept as the way to ask for a model from the library", which the docs read as the primary way. **Proposal.** Say it in one place: declared models are the language; code modules are for behavior that is not a transition; the two cannot be mixed in one *delivery* (or make `ss.vaccinate` accept a code disease via `disease=`, which `E506` already hints at, and drop `routine_vx`/`campaign_vx`/`simple_vx` from the public namespace or make them thin aliases of `ss.vaccinate`). Rewrite the v3 library diseases (`ss.SIR`, `ss.SIS`, `ss.SEIR`) as declared models so that `solver=` and `describe()` work on them; that single change removes most of the double vocabulary from the tutorials. **Cost.** Rewriting the library diseases changes their result names (`n_infected` → `n_I`) unless aliased: breaking, but the spec already promises v3 names are kept for standard results. Removing `routine_vx` is breaking. **Confidence.** Medium-high.

### 10. The time I spent on the SEIR task, as a Starsim 3 modeler

| Step | Attempts | Time to a working line | What I had to read |
|---|---|---|---|
| SEIR with a 5-day latent period | 1 (both operator and string forms worked first time) | 5 min | overview only |
| 10% case fatality | 3 (`0.1*ss.null` and `ss.death(p=)` failed with Python errors; `I >> (0.9*R + 0.1*D)` worked) | 15 min | overview, then the branching section of `03` |
| Vaccinate 80% of over-65s in 2021 | 3 (all three forms "worked"; one silently became agents, one started half vaccinated) | 25 min, mostly deciding which was right | `08-interventions.md`, `07-effects.md` conditions, `implementation.md` known issues for `ss.People(1)` |
| Run as ODE and as agents | 1 (`sim.new(solver='abm', n_agents=2000)` or a rebuilt sim) | 5 min | the `sim.new` docstring |
| Data overlaid | 1 (CSV with `date`, `cases` matching the outcome's name) | 5 min | `09-results.md` |
| Compare the two | 1 (`ss.diff_sims` works unchanged) | 2 min | |

So the language itself is quick; what cost time was (a) errors that did not name the fix, (b) three ways to vaccinate with different defaults, and (c) nothing between the overview and the 5,000-line spec to read. A v3 modeler who started from the tutorials would instead have written `ss.SEIR(pars=...)` + subclass with `step_state()` for the fatality (which works), and never found out that `solver='ode'` was being ignored.

## Simpler codebase

### 1. Frame inspection in six places

Name lookup by walking the caller's frame is done at `starsim/diseases.py:74` (transitions, states, pars at `ss.Disease()`), `starsim/diseases.py:143` (`stratify`), `starsim/sim.py:73` and `:109-121` (modules, deliveries, modifiers, products, at `ss.Sim()`), `starsim/interventions.py:574-581` (walks `f_back` repeatedly), `starsim/modules.py:520`, and `starsim/arrays.py:947-949` (walks `f_back`). Each has its own notion of how far back to walk, which is why the same object gets a name in one path and `disease` in another (finding 3). One `ss.named(obj, frame)` call at construction, with no `ss.Sim`-site pass, would delete the second half of this and make the rule statable in one sentence. **Confidence.** High that the duplication exists; medium on the size of the saving.

### 2. The product is a module in the schedule

`describe()` lists `every day: vx, vaccine, seir` and `sim.results` has both `vx` and `vaccine` (the latter with only `timevec`): a product is being registered as a full module with its own results and loop slot (`products.py:12-24`, `init_pre` skip logic). If a product is "what" and a delivery is "how" (`08-interventions.md`), the product is data the delivery holds, not a module. **Cost.** Internal. **Confidence.** Medium (I did not trace why it is a module; it may be for the `has_<product>` dimension).

### 3. Two ways to re-run a sim and neither is `run()`

`sim.new()` (`sim.py:420`, declared models only), `sim.restart()` (`sim.py:1923`, kernel-only sims), `sim.copy()` (does not reset), and `AlreadyRunError` (`sim.py:1379, 1766, 1833, 1995`, four sites) are four answers to one question. One `sim.reset()` that works for every sim, with `restart()` as its fast path, and `ss.parallel` calling it, would let the overview example stand. **Confidence.** High on the user side; medium on feasibility for code modules.

## Spec and implementation

- **The overview's step 8 and 9 contradict the implementation** (finding 1). Either the spec says `fit.sim.new()` or the library accepts run sims. The spec's `msim.plot()` "median and interval" is not what is drawn.
- **`sim.run()` prints what `describe()`'s first lines print, and the spec says both.** `10-execution.md` step 10 and `00-overview.md` ("prints the unit ... and solver") are implemented for compartmental runs; for agent runs the Starsim 3 progress printing is layered on top, which the spec does not mention. One rule is needed.
- **`E804` for an explicit solver on a code module is specified (`12-backends.md`, `18-conformance.md` §4) and not implemented** (bug 3). `implementation.md`'s departure ("a module written as code ... makes the model agent-based") covers `'auto'`, not an explicit `solver='ode'`.
- **`18-conformance.md` §14 ("every code example in this folder and in the user documentation is run in continuous integration")** is not true of this folder: the overview fails. The docs are run (`docs/check_notebooks.py`) but are Starsim 3's.
- **`v4_design/examples/README.md` describes nine build-set models that are not in the repo.** The held-out vignettes are in `/home/cliffk/idm/lingua_franca_design`. A user (or a conformance test) cannot find "the SIR in all spellings" anywhere in `ssv4`.
- **`08-interventions.md` says nothing about `disease=` on a delivery**, but `E506` requires it when the sim has no declared disease, and `_name_modules`/`get_declared` (`sim.py:796`) pick "the one declared disease" otherwise. Two declared diseases plus `ss.vaccinate` without `disease=` presumably errors; the keyword should be in the delivery table.
- **Open question B12 (names from variables) is now effectively answered by the implementation: badly.** The lookup works at module level and in one function, and fails for helpers, dicts, aliases, and inline transitions (finding 3). The answer the implementation gives to "how robust is it" is the table above.
- **Open question B6 (`ss.transfer` vs `ss.vaccinate`) is answered by the defaults**: the product form starts from zero and the transfer form starts from half, so only the product form is safe today (finding 4).
- **Open question B8 "Is `ss.outcome` needed at all?"**: in the SEIR, `cases = ss.outcome(0.5 * latent, every=ss.weeks(1))` was the only way to get a result that matches weekly data, and it worked first time; the plotting of `every=` is what needs work, not the concept.
- **`01-objects.md`'s `describe()` transcript matches the implementation line for line** (good), which also means the "nothing happens" default of finding 2 is specified, not accidental.

## Bugs

1. **`msim.plot()` raises on a MultiSim with a run dimension when the figure has a spare subplot.** Repro: the overview's step 9 with `n_runs=3` (`s1_overview.py`), or any model with 7 results. Output: `ValueError: No legend handles found in Axes(0.125,0.653529;0.227941x0.226471)` from `starsim/run.py:659`. With 6 results it warns `No artists with labels found to put in legend` and draws one line per result with an empty legend box.
2. **`sim.init(force=True)` on a run sim** raises `AttributeError: 'SimPars' object has no attribute 'people'` (`s4_checks.py`). Either `force` should re-init or it should raise `AlreadyRunError` with the `sim.new()` hint.
3. **`solver='ode'` is silently ignored for a code module.** Repro (`s6_silent.py`): `ss.Sim(diseases=ss.SEIR(), start='2020-01-01', stop='2022-01-01', solver='ode').init()` → `pars.solver: ode | sim.solver: abm`, nothing printed; `approx=False` also raises nothing; `sim.solvers()` correctly says `ode: no (works on agents)`. Spec: `E804`.
4. **`print(msim.results)`** prints `[sc.prepr(): not shown, recursion exceeded (6>5)]` for every result of a MultiSim with a run dimension (`s3_out.txt`).
5. **`describe()` before `run()` prints every init note twice** (`s1_out.txt`: the `note: contact matrix ...` and `note: unit='day'` lines appear at the top from `init()` and again as rows of the table).
6. **A code-module sim with dates keeps `dt = 1 year`** (`s5_seir_v3style.py` a2, a3: `Running 2020.01.01 ( 0/3)`), so `dur_exp=ss.days(5)` is a fraction of one step. Starsim 3 behavior, but with `start`/`stop` as dates the unit rule of `02-types.md` ("the sim takes its unit from ... `start`/`stop`") says the unit should be days here, as it is for a declared model in the same sim. `implementation.md` lists this departure for "sims with no declared states", which makes the same `start=`/`stop=` mean different things depending on which kind of disease is in the sim.

## Keep

- **`describe()`**, as it is for declared models: the `[guessed: ...]` column, the expanded rate with its denominator, the one-sentence solver reasons, and the schedule line. It is the documentation. Add to it; do not trim it.
- **All three transition spellings work, first time, including `I >> (0.9*R + 0.1*D) @ ss.days(10)`** and `ss.progression('I -> R | D', dur=10, p=[0.9, 0.1])`, giving identical `describe()` output. The operator precedence story in `03` holds up in practice.
- **The ODE-to-agents switch is one keyword** (`sim.new(solver='abm', n_agents=2000)`) and `ss.diff_sims` compares the two unchanged from Starsim 3. The results agreed to within the known `O(dt)` difference (peak `I` 196,614 vs 199,000; deaths 94,055 vs 103,500 at 2,000 agents).
- **`data='file.csv'` with a column named like a result**, and the gray-points-over-line plot: zero configuration, and `E705` catches a typo with the closest names and both fixes.
- **`sim.calibrate()` with nothing but `bounds=` and `data=`**: `Searching on a log scale: beta / Fitting by l-bfgs-b (gradients by automatic differentiation; override with method=)` and a four-line `fit.summarize()` is the right amount of output.
- **The coded errors that exist** (`E210`, `E301`, `E506`, `E705`): each names the object, the alternatives, and the fix. The problem is only the paths that bypass them.
- **`ss.parallel(baseline=..., half=...)` with keyword labels and `msim.plot()`** for scenarios (`s11_scen.png`): a clean two-color figure with a legend, as the spec promises.
- **`sim.new()`** itself, which is the right primitive; it just needs to be what the errors and the overview point to.

## Outside my area

- The vignette authors' open findings are user-facing and worth folding in: no per-run draw from `prior=` for forward runs (`flepimop` finding 1), and a weight by level in an outcome fails with a NumPy broadcast error (`summer2` finding 11, `solvers.py derived_outcomes()`).
- `ss.contacts('Kenya')` worked here (the data package fetched a 16-group matrix and aggregated it with `W404`), whereas `epydemix/sir_kenya/README.md` says it raised `E605`; the vignette may now be out of date, and the fetch should probably say where the matrix came from at `run()` and not only in `describe()`.
- The `os.fork()`/JAX deadlock warning after any gradient fit is going to hit every user who follows the overview's calibrate-then-scenarios order; `run.py` could default to `parallel=False` or spawn once JAX has been imported.
