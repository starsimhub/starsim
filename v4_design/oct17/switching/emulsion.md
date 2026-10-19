# EMULSION: would its users switch?

## Who this user is

A veterinary or plant epidemiologist at INRAE (BIOEPAR, Nantes) or a partner lab who models livestock diseases (bovine viral diarrhea, paratuberculosis, Q fever, PRRS, avian influenza) in herds linked by animal trade. They are often not programmers: they write and read one YAML file of state machines (health, age group, gestation), parameters with `desc` and `source`, prototypes, calendars, and outputs, and they hand the diagrams (`emulsion diagrams`) to veterinarians and the farmers' organizations who fund them. A Python add-on file appears only when a process (data-driven trade, culling) does not fit the YAML. They chose EMULSION (Picault et al. 2019, PLoS Comput Biol 15(9):e1007342) for three things: the model is a document a non-programmer can audit; the same file runs as compartments, individuals, or hybrid groups with one keyword (`aggregation_type`); and it has the multi-level structure their field needs (animals in herds in a metapopulation). What they live with: it is slow (pure Python), version 1.1.1 dates from October 2020 and needs `bokeh<3` and `numpy<2` to start at all (`_envs/emulsion/README.md`), there is no calibration in the tool, and the project site could not be reached on 13 October 2026.

## The first day

The two vignettes are models this user recognizes.

| Vignette | Lines (EMULSION, Starsim) | One run (EMULSION, Starsim) | Agreement |
|---|---|---|---|
| `sir` (one SIR, three paradigms) | 81 to 92 per YAML file, against 8 for one model run two ways | 0.81 s as a command (0.06 s of simulation), against 0.017 s | Compartments agree (peak 37 on day 18 both; 85 recovered both). Agents are lower: 77 recovered against 81, as 22% of runs die out against 19% (the half-step dwell issue) |
| `quickstart` (age, gestation, health, births, quarantine) | 280 (about 80 of them `desc`, `name`, `fillcolor`), against 75 | 1.46 s as a command (0.52 s of simulation), against 0.25 s | Within 1.4 animals at every step once the quarantine window is shifted to the steps EMULSION applies it on; up to 6.5 animals in `Q` as written |

The SIR vignette is the pitch they would want: `solver='binomial'` or `solver='abm'` on one 8-line model is EMULSION's `aggregation_type` without three files. The quickstart is the honest one. It agrees, but it reaches that through six workarounds marked in `vignettes/emulsion/quickstart/model.py`: a `stay()` closure for EMULSION's minimum stay (`duration` as a guard), a rate of 10⁹ a day for "as soon as" (`J` to `A`), `where=lambda sim: sim.diseases.growth.A` because `where=growth.A` raised `TypeError` when it was written (it works now: the vignette's own repro runs at HEAD), a 30-line `Life(ss.Disease)` subclass that makes one newborn per farrowing with `people.grow()` and sets four modules' states by hand, `ti_NG = -1` so the initial animals' stay counts from before the first step, and an analyzer to count states because newborns appear in two modules' results a step late. This user would read the 75 lines and say the transitions are clearer than their YAML and the custom class is less clear than `produce_offspring: [infected_newborn, protected_newborn]`. A farmer's veterinarian could not audit `Life.step_state()`.

The `docs/` folder in ssv4 is all Starsim 3: `ss.SIR(beta=..., dur_inf=...)` agent classes, networks, no `solver=`, no `ss.load(..., format='emulsion')`, no `ss.Mobility(data=...)`. This user would find the importer and the movement register only through `v4_design/design/16-interop.md`, `06-routes.md`, and `skills/import-model/references/emulsion.md`. Opus already says the tutorials must be ported before anyone outside meets v4 (`opus_review/README.md:38`); from this seat it matters more than usual, because the importer is the whole on-ramp and nothing public mentions it.

## The second week

All scripts are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/emulsion/`.

### 1. Bring my YAML: `ss.load('model.yaml', format='emulsion')`

What this user would try first (`t1_import.py`):

```python
sim = ss.load('quickstart.yaml', format='emulsion')
sim.run()
```

- **The three SIR files** import in 0.01 s and run, each with the right solver (`compart_SIR` as `binomial`, `IBM_SIR` and `hybrid_SIR` as `abm`), printing how each rate was read (`S -> I force_of_infection : exposure to I (divided by the total: frequency-dependent) => transmission_I * I`). `sim.to_code()` writes a 37-line Python file that is close to hand-written (Fable 10 says the same). This is a good experience.
- **The quickstart** imports and runs, and warns `W1601 ImportIncomplete: 16 part(s) of the model were not translated`, each named precisely (the minimum stays of `NG`, `M`, and `E`; `cond: is_A`; `escape: is_I`; `when: quarantine_period`; both `on_cross` actions; `statevars`; the calendar). The list is honest and useful. But the sim still runs, and gives 15 animals alive at the end against EMULSION's 66: `J` to `A` was dropped (so `age_group` has 0 transitions), which made `A` "treated as a count, not people (no transition connects A to J)", which is Codex D4's concern about inferring people from connectivity, here in a real model. A run with 16 untranslated parts should not look like a result; `ss.load()` could return the sim with `sim.imported.incomplete` set and refuse `run()` until the user passes `allow_incomplete=True`.
- **The vignette READMEs lag the code.** `where=growth.A` now works (the README's finding 2 says it raises), so the quickstart's lambda can go; the READMEs should be rerun before anyone outside reads them.
- **A bug in the import: `proba` is read per step, but EMULSION reads it per time unit.** `emulsion.py:225` writes `-log(1 - p) / dt`; EMULSION aggregates a probability to the step as `1 - (1 - p)**delta_t` (`emulsion/tools/misc.py:199`, used at `agent/core/emulsion_agent.py:361` and `:430`). Verified with an SIR whose recovery is `proba: 0.1` and `delta_t: 7` (`proba/proba_SIR.yaml`): after one step EMULSION has 47.7 of 100 still infectious (0.9⁷ = 0.478), the import 89.6. In the quickstart this makes the imported abortion rate 7 times too low. The fix is one line (`-log(1 - p)` per time unit); `skills/import-model/references/emulsion.md` has the same wrong row and should be corrected with it.

### 2. EMULSION's own metapopulation example

`models/features/compart_SIR_metapop.yaml` (150 lines, 42 of them descriptions): 10 populations of 100, one starting with 10% infected, each with an SIR, juveniles maturing to adults at 1/20 a day, births from adults at `2b`, and density-dependent deaths at `b × N/150` within each population. EMULSION: 100 runs in 124 s (`metapop/run_compart.log`).

- **The import is silently wrong.** It lists one untranslated part (`statevars: initial_prevalence`) and drops two things without listing them: the `metapop` level (10 populations become one of 10 animals) and `productions` (the births). The two state machines become two diseases whose counts drift apart (health 10, age 6 at the end). Every key the importer does not read should be listed in `W1601`; the importer has a `skipped_keys` list (`emulsion.py:23`) for transitions but nothing for levels or `productions`.
- **Written by hand in v4, the natural form silently gives no births.** With age stage as a dimension (`sir.stratify(stage=['J', 'A'])`, `ss.transfer(stage='J -> A', ...)`), births from adults only cannot be declared: `ss.birth('-> S', rate=ss.by(stage=[0, 0.1]))` and the explicit `ss.Transition(ss.null, S, rate=ss.by(stage=[0, 0.1]) * N.n)` both run with 0 births (`metapop/v4_metapop2.py`). `ss.birth` takes no `where=` (its signature is `arrow, rate, name, label`), and setting `where = sir.stage == 'A'` on the birth after stratifying is ignored (26,704 births and 3,000 animals instead of 1,499). No error, no note.
- **Crossing the two machines by hand works, and it is fast.** Six states (`SJ, IJ, RJ, SA, IA, RA`), seven transitions, births `ss.null >> SJ @ 2*b * (SA + IA + RA).n`, deaths `ss.death(rate=b * N.n / K)`, and `sir.stratify(pop=[...], kind='space')`: 17 lines (`metapop/v4_metapop3.py`). Counts in rates are local to each population (checked in `metapop/count_check.py`). 100 runs take 0.59 s, 210 times faster than EMULSION, and the means agree at day 100: total 1,508 against 1,499, juveniles 771 against 764, adults 737 against 735, infectious 36.4 against 36.8, recovered 66.7 against 66.6. The same file switches solver with one word: `abm` 0.42 s a run (total 1,509), `gillespie` 0.38 s (1,473), `ode` 0.04 s (1,485). So v4's agents run this model three times faster than EMULSION's compartments, which removes most of the reason the hybrid mode exists.

### 3. Herds linked by a trade register

What their field does every week (`t2_herds.py`): an SIR (`beta` 0.3 a day, 14 days infectious) in herds of 50 to 300 animals, linked only by a year of weekly sales (`ss.Mobility(data=register)` with columns `date, from, to, n`, as SimInf's register, `06-routes.md:291`), five herds seeded with two infected each. The model is 9 lines plus the register.

| | 200 herds, 35,950 animals, 2,641 sales | 2,000 herds, 354,201 animals, 26,486 sales |
|---|---|---|
| `binomial`, a run | 0.21 s | 1.15 s |
| `abm`, a run | 2.9 s | 14.7 s |
| Herds with an outbreak (10% or more infected), binomial against abm | 16.9 ± 3.1 against 14.2 ± 1.9 (20 runs each) | 14.7 ± 3.8 against 12.0 ± 4.2 (3 runs each) |

The two agree within their noise. Adding `states='S,R'` to the register (only healthy animals sold, the selective movement their field models, `06-routes.md:294`) confines the outbreaks to the five seeded herds (4.8 ± 0.1 as compartments, 4.7 ± 0.1 as agents, 10 runs each), as it should. EMULSION does this with a hand-written `MetapopProcessManager` subclass (`models/features/hybrid_SIR_metapop_data.py`), which is why the import of `hybrid_SIR_metapop_data.yaml` fails, with an uncoded `ValueError: The model spec uses "total_herd", which is not a state, parameter, or transition of the module`.

### 4. Scale and the one-word switch on a real model

- **Scale.** The quickstart with 10,000 animals (`scale/quickstart10k.yaml`, initial infected and quarantine size scaled by 100): EMULSION did not finish one run in 600 s. The v4 vignette at 100, 10,000, and 100,000 animals takes about 4 s as a whole process each time, of which 2.1 s is `import starsim` and most of the rest is building; the run itself is about half a second. This is the single most persuasive number for this user, whose national herd models are the slow part of their work.
- **The switch.** The quickstart vignette with `solver='binomial'` fails with `AttributeError: Life object has no attribute "ti_NG"`, uncoded. `sim.solvers()` predicts it ("function: func ... no (takes uids)", five times), but every one of the `stay()` closures is called `func`, so the table cannot say which part blocks which solver. Once a model has the workarounds, it is agent-only; the paradigm switch survives only in models without minimum stays, offspring, or cross-machine conditions, which is to say not in the models this user publishes.
- **`describe()`** of the quickstart says `rate = func` five times, `init: J = 1.0 [guessed: nothing given]`, `100 people [guessed]` although `people=` was given, and `farrowing ... ss.constant(v=240), as 50 Erlang stages` under `abm`, where the constant is sampled exactly. For a user whose reason for choosing a tool is a readable model, this page is the replacement for their YAML, and it is not yet as good as the YAML.

## What would win them over

- **Speed at their scale, with the paradigm switch kept.** 210 times faster on EMULSION's own metapopulation model with the same answer (0.59 s against 124 s for 100 runs), one run of 100,000 animals in about half a second where EMULSION cannot finish 10,000 in ten minutes, and 2,000 herds with a trade register in 1.15 s as compartments. The one-word switch works on the hand-written metapopulation (`abm`, `gillespie`, `ode`, `binomial`, all within 2.5%).
- **A trade register as data**, with selective sales by state (`states=`) and shifts between states (`to_states=`), where EMULSION needs a Python add-on. Their movement databases (BDNI in France) are exactly this table.
- **An importer that already reads their files** and tells them, line by line, how each rate was read and what was not translated, with `to_code()` giving them a Python file to continue from. For simple models the import is complete and agrees with 200 EMULSION runs (`implementation.md:255`).
- **Calibration they do not have today.** EMULSION has no fitting; v4 has `ss.par(..., bounds=)`, `data=`, and `sim.calibrate()`, with a posterior by particle MCMC for stochastic compartmental models with a likelihood. For a herd model fitted to surveillance prevalence, that is new capability, not a convenience.
- **A maintained, installable stack.** EMULSION 1.1.1 needs `bokeh<3` and `numpy<2` and Python 3.9 to start; v4 runs on Python 3.13 with current NumPy.
- **Bundled geography** (`ss.geo('France', level=1)`) is useful for regional herd models, though their herd lists come from their own data.

## What would stop them

| Gap | Kind | Fixable? (rough size) | Is v4 at the best point of the tradeoff, and if not, where | Already in a review? |
|---|---|---|---|---|
| The model is Python, not a YAML document a veterinarian can audit | Inherent tradeoff (a language against a data format) | Partly | Not yet. The principle (`best-practices/principles.md:142`: "a data format is the right thing to export to and the wrong thing to write in") is sound, but then the readable outputs must replace the document: `describe()` that names functions and does not print `rate = func`, `diagram()` with the parameter `desc` and `source` on hover, and a `to_code()` that is the canonical saved form. The better point: v4 authoring, plus an export of the resolved model to a read-only YAML in EMULSION's layout for review | Fable README:123 and Fable 10 (make `to_code()` the product); not from the auditing angle |
| Minimum stay (`duration` as a guard, with `escape`) | Missing feature | Yes, small to medium: `after=`/`before=` on `ss.Transition` using `ti_<state>` as agents, a fixed-length sub-state as compartments (vignette finding 1) | No: today it is a closure that makes the model agent-only | No |
| Births on crossing a transition (`produce_offspring`, one newborn per farrowing, prototype by the mother's state) | Missing feature | Yes, medium: `offspring=` on a transition, or a public `add_people()` for custom steps (vignette finding 3) | No: today it is a 30-line subclass that sets four modules' states by hand | No |
| Births from one stratum into another (`productions: {from: A, to: J}`) as compartments; `ss.by(stage=...)` on an inflow gives 0 births silently, and a birth has no `where=` | Bug (silent), then missing feature | Yes, small: raise on `ss.by` over a non-space dimension on an inflow until supported; then support an inflow whose rate is a count of one stratum, entering another | No: the user has to cross the machines by hand into six states | No |
| Several state machines on the same animals as compartments | Missing feature | Medium: the crossing exists for diseases with cross-conditions (`07-effects.md:199`); it needs to cover births and deaths shared by two declared modules | The six-state hand crossing is exact, so the gap is length (17 lines here, more with three machines), not correctness | No |
| Import: `proba` divided by `delta_t` | Bug | Yes, one line, plus the skill's table | | No |
| Import: `metapop` level and `productions` dropped without a `W1601` entry; a sim with 16 untranslated parts runs as if complete | Bug | Yes, small | Better point: refuse `run()` on an incomplete import unless asked | No |
| No hybrid (groups that hold individuals), no herd-level processes (cull a herd on detection) | Missing feature, partly an inherent tradeoff | Hybrid: not needed at these speeds. Herd-level actions: medium, a delivery whose unit is a place | Hybrid, yes: v4's agents beat EMULSION's compartments. Herd-level: an intervention `on=` a detection that acts on the detection's place would be the right point | Paradigm-conversion lists levels as deferred (`best-practices/paradigm-conversion.md:265`) |
| Agents die out too often from one case (22% against 19%) | Known issue (half-step dwell) | Yes, medium | | In `implementation.md`; all reviews |
| `docs/` is Starsim 3, and nothing public mentions the importer or the register | Docs | Yes, large | | Opus README:38 |
| Ecosystem: EMULSION's papers, INRAE's community, French veterinary networks, and its validation against field studies | Ecosystem | Slowly | This user's collaborators read YAML; a v4 model they cannot read is a cost to their partnerships, not just to them | No |

## Verdict

- **Today:** no for the published models (minimum stays, offspring, and cross-machine conditions are agent-only workarounds, and the import of their richer files runs silently wrong); yes, enthusiastically, for new metapopulation and trade-register work, where v4 is 100 to 200 times faster with the same answer.
- **After the four reviews' fixes:** still no for their core models, since none of the reviews covers minimum stays, offspring, births between strata, or the importer's silent drops; the fixes would make the docs, `describe()`, and the people inference better, which helps.
- **A clear win needs:** `after=`/`before=` on transitions, `offspring=` on a transition, inflows from one stratum into another, the importer fixes above, and a readable export of the resolved model; with those, the quickstart imports complete and switches paradigm, and this user gets EMULSION's design at a hundred times the speed plus calibration.

## The demo that would sell it

Their own file, imported whole, run two ways, at national scale, and fitted:

```python
import starsim as ss
sim = ss.load('quickstart.yaml', format='emulsion')     # Nothing listed as not translated
sim.describe()                                          # Reads like the YAML: desc, source, prototypes
msim = sim.new(n_agents=100_000).run(n_runs=200)        # As individuals, 100,000 animals
cmp  = sim.new(solver='binomial').run(n_runs=200)       # Same file, as compartments
ss.parallel(individuals=msim, compartments=cmp).plot()

herds = ss.geo('herds.csv')                             # Their herd list
sim.stratify(herd=herds, kind='space', mobility=ss.Mobility(data='bdni_moves.csv', states='S,R,M'))
fit = sim.new(data='surveillance.csv').calibrate()      # Posterior on beta and detection
fit.plot()
```

Today: the first line runs with 16 parts not translated and gives 15 animals where EMULSION gives 66, so the rest of the demo is meaningless on the quickstart; it needs minimum stays, offspring, cross-machine conditions, and calendars in the importer. The second half (a trade register on a herd list, both solvers, calibration) runs today on a model written by hand, as task 3 shows.
