# The sim, people, modules, and demographics

I reviewed `01-objects.md` (the `ss.Sim` and module sections, the lifecycle, `describe()`) and `10-execution.md`, against `starsim/sim.py` (2,692 lines), `starsim/people.py` (952), `starsim/modules.py` (943), `starsim/parameters.py` (661), `starsim/demographics.py` (1,175, mainly `Births` and `Deaths`), `starsim/timeline.py` (604), the `Disease` base class in `starsim/diseases.py` (lines 18-390), the `ss.birth`/`ss.death`/`ss.immigration`/`ss.emigration` constructors in `starsim/transitions.py` (844-916), and `Loop.collect_funcs()` in `starsim/loop.py`. I ran eight scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/sim/` (`s1_three_sirs.py` through `s8_declared_sir.py`): the three SIRs (`ss.Sim(ss.SIR())`, the declared SIR, `ss.demo()`), every way of constructing a sim I could find, `total_pop`/`n_agents`/`people` combinations, `ss.People` under the ODE solver, declared `ss.birth`/`ss.death` against `ss.Births`/`ss.Deaths`, the loop labels for declared and coded diseases, and the class form with a `step()` override. All output quoted below is from those runs. The structure reviewer has already measured `sim.py`'s growth and argued for keeping `Births`/`Deaths`/`Pregnancy` as library modules; I build on both and do not repeat the line counts.

## Summary

1. **Make `ss.SIR`, `ss.SIS`, and `ss.SEIR` declared diseases** (class form), so the library model and the one-page model are the same model. Today `ss.SIR().equations()` raises, `ss.SIR()` cannot run under `solver='ode'`, and `ss.Sim(ss.SIR())` with no network silently simulates nothing, while the declared SIR does the opposite on each count. Breaking: result names (`n_susceptible` becomes `n_S` unless the states are named `susceptible`, `infected`, `recovered`) and the `rel_sus`/`infected` state API change for subclasses. The spec promises this already (`01-objects.md` §Library models).
2. **Give `pars=` one meaning.** `ss.Sim(sir, pars=dict(beta=0.1))` means model values and `ss.Sim(pars=dict(n_agents=500))` means sim settings; the code decides by sniffing (`sim.py:77`) and a mixed dict fails with an unrelated `KeyNotFoundError`. Make `pars=` the model values, as the spec says, and let sim settings arrive as keywords or as the (already supported) dict in the first position. Breaking for 21 test files and ~70 doc cells that write `ss.Sim(pars=pars)`; a one-release shim can accept the old form with a note.
3. **Do not force a declared disease with a `step()` override to run as agents.** The spec's canonical class-form example (`def step(self): super().step(); self.do_something_bespoke()`) is exactly what `sim.py:982-984` uses as a reason to pick `abm`, and the documented escape hatch `compartmental = True` is ignored for declared diseases. Treat an overridden `step()` like a function in a rate: try it on groups, as `12-backends.md` §Functions says. Not breaking.
4. **Replace the one-row `ss.People` stand-in for a compartmental run.** The ODE run builds `ss.People(1)`, deletes its `age` and `female` attributes, sets `pars.n_agents = n_groups`, and so `repr(sim)` reads `Sim(n=1; ...)` and `print(sim.people)` raises `E804`. Either give `People` an honest `n_agents=0`/groups mode with a working repr, or let the compartmental engine own its group table and have `sim.people` be `None` with a clear message. Lazy conditions should not need a throwaway `ss.People(1)`: class-level lazy descriptors (`ss.People.age < 15`) would give the spec's "real attributes" form with no instance.
5. **Unify population-size vocabulary around `total_pop`, and make `n_alive`/`N` a result under every solver.** The defaults are right (verified table below) but a compartmental run has no `n_alive`, so a user who switches solvers loses the denominator; and `pop_scale=` survives as a third spelling of the same thing.
6. **Collapse the demographics shortcuts to two.** There are four ways to add births: `ss.birth()` in a disease, `ss.Births()` as a module, `birth_rate=20` on the sim, and `demographics=True`. Keep the declared transition (every solver) and the module (agents, data-driven), and drop `birth_rate=`/`death_rate=`/`demographics=True` from `SimPars`, or make them build `ss.birth`/`ss.death` so a compartmental model is not silently turned into a 10,000-agent one. Breaking for the `birth_rate=` idiom in three tutorials.
7. **Move the stratification geometry and solver plumbing off `Sim`'s public surface.** `ss.Sim` has 76 public methods; about 20 are `init_*`, and `fold`, `fold_matrix`, `folded`, `group_vals`, `grid_index`, `group_raw`, `level_raw`, `dim_index`, `groups_changed`, `init_ragged`, `init_crossings`, `get_declared`, `agent_reasons`, `try_functions`, `kernel_only`, `kernel_reason` are engine internals that happen to live on the sim. A `sim.groups` (or `sim.strata`) object would hold the first group and `sim.engine` the second. Not breaking for users; mechanical for the code.

## Simpler for the user

### 1. `ss.SIR()` and the declared SIR are different models with different behavior

**Issue.** The spec (`01-objects.md` §Library models) says `ss.SIR(beta=0.1)` "is kept as the way to ask for a model from the library. A library model is an ordinary module: `describe()` prints it in full and `equations()` works on it." What the user gets today (`s5_library_models.py`):

```text
SIR 'sir': written as code (states: susceptible, infected, recovered, ti_infected, ti_recovered, ti_dead, rel_sus, rel_trans)
equations -> ERROR ValueError "sir" has no declared transitions, so it has no equations to print
diagram   -> ERROR ValueError "sir" has no declared transitions, so it has no diagram
R0        -> ERROR ValueError "sir" has no declared transitions, so R0 cannot be worked out from them
solver=ode on ss.SIR -> ERROR E804 CapabilityError: cannot run with solver="ode": the module "sir" (SIR) only works on agents.
```

The two SIRs also differ in what "no network" means. For the declared disease, no network is homogeneous mixing (spec, and verified: `mixing homogeneous, 1 contact per step [guessed: no network given]`). For `ss.SIR()`, no network is no transmission, and nothing says so:

```text
B: ss.Sim(ss.SIR()) with no network: silent?
warnings: []
n_infected: [2. 2. 0. 0.] new_infections total 0.0
```

And the results have different names: `n_susceptible, n_infected, n_recovered, prevalence, new_infections, cum_infections` for `ss.SIR()`; `n_S, n_I, n_R, infection, cum_infection, recovery, cum_recovery, new_infections, cum_infections, prevalence` for the declared one (`s1_three_sirs.py`). A user who reads the one-page model in `00-overview.md` and then types `ss.demo()` sees a 2000-2050 sim in years with `beta=peryear(0.1)` and `dur_inf=lognorm_ex(mean=6 years)`, not the model on the page.

**Why it matters.** The declared language and the library are supposed to be one language. Right now the first thing a Starsim 3 user reaches for (`ss.SIR()`) cannot do any of the four things v4 adds (equations, diagram, solver choice, R0), and the first thing the spec shows (`ss.Disease(...)`) has different result names from every example in the existing docs.

**Proposal.** Rewrite `ss.SIR`, `ss.SIS`, `ss.SEIR` (and `ss.NCD` if it fits) in the class form. I checked this is expressible (`s8_declared_sir.py`): a four-line class with `S, I, R, D`, `infection = S >> I @ beta * I`, and `recovery = ss.progression('I -> R | D', dur=ss.lognorm_ex(mean=ss.years(6), std=ss.years(1)), p=[0.99, 0.01])` runs as agents on a `RandomNet` (peak `n_I` 827 against `ss.SIR()`'s 845 with the same seed and network) and as an ODE with a note about 36 Erlang stages. To keep result names, declare `ss.states('susceptible infected recovered dead')`; to keep `sir.infected` working as a boolean state, nothing is needed, since in an agent-based run a declared state is a `BoolState` (`type(d.S)` is `starsim.arrays.BoolState`). Then make `ss.demo()` run that model, and make "no network" mean homogeneous mixing for every disease (or at least warn once when a coded `Infection` has no routes: `validate_beta()` at `diseases.py:462` already has the network keys in hand and could say so).

**Cost.** Breaking for subclasses of `ss.SIR` that override `set_prognoses`, `set_progression`, or read `ti_recovered`; the library diseases (`ssl.HIV`, cholera, Ebola, measles) are built on `ss.Infection` and should stay coded, so `ss.Infection` remains. Result names change only if the states are renamed. The alternative, amending the spec to say library models are coded and only `describe()` works, is cheaper but gives up the headline property that "the model is data" for the models every tutorial starts with. Confidence: high that the two should be one model; medium on the exact migration path.

### 2. `pars=` means two things

**Issue.** `ss.Sim.__init__` (`sim.py:77-78`):

```python
if isinstance(pars, dict) and len(pars) and not any(key in self.pars for key in pars):
    self.par_values, pars = dict(pars), None
```

So `pars=dict(beta=0.1)` is model values and `pars=dict(n_agents=500, stop=10)` is sim settings, decided by whether any key happens to be a `SimPars` key. A dict that mixes the two (`s2_constructors.py` case 8):

```text
8 pars= mixed model + sim
ERROR KeyNotFoundError Key(s) ['beta'] not found; available keys are ['label', 'verbose', 'n_agents', ...
```

The spec itself carries the ambiguity: `01-objects.md` lists `pars = None  # parameter values by name (04-parameters.md)` and two paragraphs later "A dict in that position is still read as Starsim reads it", and `19-open-questions.md` B1 records "First positional argument of `ss.Sim`: the model(s) / alternative: Starsim's `pars` dict". The implementation answered: both, by sniffing.

**Why it matters.** A model parameter called `label`, `verbose`, `start`, `unit`, or `dt` would flip the whole dict's meaning; `E201 ReservedName` protects some of these but not `verbose` or `label`. More simply, a reader of `ss.Sim(sir, pars=fit.pars)` and `ss.Sim(pars=pars)` in the same tutorial has to learn that the word means something different in each.

**Proposal.** `pars=` is model values, full stop, as `04-parameters.md` says. Sim settings are keywords, or the dict in the first position (which already works: `ss.Sim(dict(stop=10), diseases=sir)` ran). For one release, a `pars=` dict whose keys are all `SimPars` keys is accepted with a `W` note pointing at the keyword form. A clearer name for the model values would avoid the collision entirely: `values=`, or reuse the Starsim 3 idiom of setting them on the disease (`sir.set_pars(beta=0.1)` already exists at `diseases.py:105`) so `ss.Sim` needs no second parameter channel at all. I prefer the second: one fewer argument on `Sim`, and `fit.pars` becomes `sir.set_pars(fit.pars)` or `fit.sim`.

**Cost.** `ss.Sim(pars=pars)` appears in 21 of the test files (113 matches for the pars-dict idioms) and about 70 cells of the tutorials and user guide. Breaking, but mechanical. Confidence: high.

### 3. The class form's `step()` override forces an agent-based run

**Issue.** The spec's example of the class form (`00-overview.md` lines 105-113 and `01-objects.md` lines 129-144) is a `Disease` subclass with declared states and

```python
def step(self):
    super().step()
    self.do_something_bespoke()
```

and says "Subclassing adds behavior. It does not add vocabulary." `12-backends.md` §Functions says custom code is tried on groups. The implementation (`sim.py:982-984`):

```python
for method in ['step', 'step_state']:
    if getattr(type(mod), method) is not getattr(ss.Disease, method):
        reasons.append(f'"{mod.name}" defines its own {method}()')
```

Verified (`s4_demog_lifecycle.py` C): a declared SIR whose `step()` only calls `super().step()` gives `solver: abm ("sirx" defines its own step())`, and adding the class attribute `compartmental = True` (the documented way for coded modules to say they work on groups, `implementation.md` departures) changes nothing: `with compartmental=True -> abm "siry" defines its own step()`. A plain function passed as a member (`extra=f`) does stay ODE: `plain function member -> ode`.

**Why it matters.** The spec gives one reason to use the class form (behavior that is not a transition), and that reason silently costs the user the ODE solver, the equations, and R0, with a 10,000-agent run in place of a 1-group one. The function-member form and the class form are supposed to be equivalent (`01-objects.md` §The equivalence rule), and here they are not.

**Proposal.** Treat an overridden `step()`/`step_state()` as one more function to try (`try_functions()` at `sim.py:996` already builds a probe sim and calls things on it), and honor `compartmental = True` on declared diseases as the explicit override. If the probe cannot run the method, say so in the reason as today.

**Cost.** None for users. Some risk that a `step()` which happens not to raise on groups does the wrong thing on them; that is the same risk the spec accepts for functions, and `describe()` can list the method under "restricts the solver". Confidence: high.

### 4. `ss.People` is a stand-in in a compartmental run

**Issue.** Under `solver='ode'`, `init_people()` builds `ss.People(n_agents=1)` because `init_solver()` has set `pars.n_agents = self.n_groups` (`sim.py:385`); `init()` then pops `age` and `female` out of `people.__dict__` and records them in `people._no_groups` (`sim.py:277-282`) so `People.__getattr__` can raise `E804`. What the user sees (`s3_people_pop.py`, `s8_declared_sir.py`):

```text
Sim(n=1; 0—30; diseases=sir)
type <class 'starsim.people.People'> len 1 n_agents 1 states ['alive', 'age', 'female', 'ti_dead', 'ti_removed', 'scale']
pars.n_agents 1 total_pop 10000 pop_scale 1.0
>>> repr(s2.people)
ERROR CodedError E804 CapabilityError: people.age was asked for, but it is a per-agent attribute and not a stratification, so it does not exist in a compartmental run.
```

`n=1` in the repr of a 10,000-person ODE sim is wrong twice over (it is neither agents nor people), and the `people.age` error fires from `People.brief()` (`people.py:189`) so the object cannot even be printed. With an age stratification the one row per group gets its age set to the group midpoint (`sim.py:264-267`), which is how a function-of-the-sim condition "works" without complaint (the known issue in `implementation.md`), while a lazy condition needs a throwaway `ss.People(1)` to exist at all (`where=ppl.age < 15`), and a lazy condition whose cut is not a bin edge dies with a raw NumPy error rather than `E804` (see Bugs).

**Why it matters.** The spec's answer to "what is `sim.people` in a compartmental run" is `19-open-questions.md` B1: "the age group's midpoint / alternative: the bin edges; an error for anything but comparisons". The implementation chose the first, but through an object whose repr crashes and whose length is the number of groups. A user reading `sim.people` or `sim.pars.n_agents` after an ODE run gets numbers that mean something else.

**Proposal.** Two options, in order of preference:

- Let `sim.people` be `None` in a compartmental run, with `Sim.__getattr__`-style guidance (`E804`, the same text as now) when it is read, and have the compartmental engine own its group table (it already does: `CountState` objects and `n_groups`). `pars.n_agents` stays `None` and `repr(sim)` prints `Sim(10,000 people; ode; ...)`. Conditions on properties of people take a class-level lazy reference: give `ss.People` class attributes `age`, `female`, ... that are `Cond` descriptors, so `ss.People.age < 15` is the spec's "lazy expression on uninitialized arrays" without an instance; `ppl.age < 15` on a real instance keeps working as agents.
- Or keep `People` but give it an honest zero-agent/group mode: `People(n_agents=0, groups=n)` with a repr that says so, `age` and `female` present as per-group values (midpoints, `nan` for sex), and `n_agents` not overloaded.

**Cost.** Option one touches every place the engines read `sim.people` for group values (`group_vals`, `init_strat_vals`, the age midpoints); it is the same code, moved. `len(sim.people)` and `sim.pars.n_agents` change meaning for ODE runs (they are wrong now). Confidence: medium-high that the current state should not ship; medium on which option.

### 5. `total_pop`, `n_agents`, `pop_scale`, `people`

**Issue.** The defaults in the spec (`01-objects.md` §`total_pop` and `n_agents`) are implemented and I confirmed every row (`s3_people_pop.py` C):

```text
{}                                                      -> solver=ode  n_agents=       1 total_pop=      10,000 pop_scale=1 n_S0=9,900
{'total_pop': 50000000.0}                               -> solver=ode  n_agents=       1 total_pop=  50,000,000 pop_scale=1 n_S0=49,500,000
{'n_agents': 500}                                       -> solver=abm  n_agents=     500 total_pop=         500 pop_scale=1 n_S0=495
{'n_agents': 500, 'total_pop': 50000000.0}              -> solver=abm  n_agents=     500 total_pop=  50,000,000 pop_scale=100000 n_S0=49,500,000
{'total_pop': 50000000.0, 'solver': 'abm'}              -> solver=abm  n_agents=   10000 total_pop=  50,000,000 pop_scale=5000 n_S0=49,500,000
{'total_pop': 500, 'solver': 'abm'}                     -> solver=abm  n_agents=     500 total_pop=         500 pop_scale=1 n_S0=495
{'people': People(n=300)}                               -> solver=abm  n_agents=     300 total_pop=         300 pop_scale=1 n_S0=297
{'pop_scale': 10}                                       -> solver=ode  n_agents=       1 total_pop=     100,000 pop_scale=1 n_S0=99,000
{'pop_scale': 10, 'solver': 'abm'}                      -> solver=abm  n_agents=   10000 total_pop=     100,000 pop_scale=10 n_S0=99,000
```

Three things remain for the user to learn. (a) `pop_scale=` is a third spelling (`parameters.py:308`, `validate_total_pop()` at 403) that the spec does not mention; `total_pop` alone covers it. (b) In an ODE run there is no population result at all: `people results (ode) ['timevec', 'sir']` against `['timevec', 'sir', 'n_alive', 'n_female', 'new_deaths', 'new_emigrants', 'cum_deaths']` as agents (`s1_three_sirs.py` D). So `sim.results.n_alive` exists under one solver and not the other, and the spec's open question "the denominator: `N = S + I + R`, written by the user" (B1) is answered by making the user write it only sometimes. (c) `pars.n_agents` is `1` in an ODE run (finding 4).

**Proposal.** Drop `pop_scale=` as an input (keep it as a derived attribute). Record `n_alive` (and `new_deaths` where any outflow is a death) at the sim level under every solver, computed from the engines' people counts in a compartmental run; it is one result and it is the denominator every ratio wants. Then the B1 question closes: `N` is built in, and `S + I + R` is still available for those who want the module's own total.

**Cost.** `pop_scale=` is used in a handful of tests; `n_alive` under ODE is additive. Confidence: high for (a) and (b).

### 6. Four ways to add births and deaths

**Issue.** A user who wants vital dynamics can write any of:

```python
ss.Sim(sir, birth_rate=20, death_rate=10)                       # SimPars shortcut: parameters.py:505-522 build Births/Deaths
ss.Sim(sir, demographics=True)                                   # the same with defaults: parameters.py:501
ss.Sim(sir, demographics=[ss.Births(birth_rate=ss.peryear(20)), ss.Deaths(death_rate=ss.peryear(10))])
ss.Disease(..., births=ss.birth('-> S', rate=ss.peryear(0.02)), deaths=ss.death(rate=ss.peryear(0.01)))
```

They are not the same model. The first three force an agent-based run even for a purely compartmental disease and say so only at `verbose>0` (`s4_demog_lifecycle.py` A: `auto with Births -> abm Births only works on agents`; `s2_constructors.py` cases 12-13: `Sim(n=10000; ...; demographics=births, deaths)` for a disease that was an ODE one line earlier). The rates mean different things: `ss.Births` is per 1,000 per year with `rate_units=1e-3` (`demographics.py:49`), `ss.birth` is per person. The results land in different places with different names: `births.new`, `births.cumulative`, `births.cbr`, `deaths.new`, `deaths.cmr` for the modules; `sir.births`, `sir.cum_births`, `sir.deaths`, `sir.cum_deaths` for the declared transitions, next to the people-level `new_deaths`/`cum_deaths`. And a bare `birth_rate=20` in a sim with no unit warns that it is read in years "but the sim counts time in timesteps" (`s2_constructors.py`, `parameters.py:58`).

The spec says (`03-states-and-transitions.md` line 360): "Demographic transitions may be declared inside a disease. For anything beyond the simplest case a `ss.Demographics` module (`ss.Births`, `ss.Deaths`, `ss.Pregnancy`) keeps them in one place and lets them run at their own timestep." That is a reasonable division (declared for the simple per-capita case under every solver; module for data-driven, age-specific, agent-only), and the structure reviewer's recommendation to keep the modules in the library is right. What has no place in it is the `SimPars` shortcut layer.

**Proposal.** Remove `birth_rate=`, `death_rate=`, and `demographics=True` from `SimPars` (`parameters.py:331-333, 497-527`). If a shortcut on the sim is wanted, make it build `ss.birth`/`ss.death` transitions on the first disease (so a compartmental model stays compartmental), not `ss.Births`/`ss.Deaths`. Harmonize the rate convention: `ss.Births(birth_rate=ss.peryear(20))` meaning 20 per 1,000 while `ss.birth(rate=ss.peryear(0.02))` means 0.02 per person is a trap; `rate_units` defaulting to 1 with the UN loader supplying `1e-3` would make both per person. Name the module results like the declared ones (`births.new` is fine; `births.cbr`/`deaths.cmr` are the demographers' names and can stay).

**Cost.** Breaking for `birth_rate=` in three tutorials and `rate_units` for anyone loading UN data. The `use_aging` default (on when demographics are present, `parameters.py:526`; also turned on by an age stratification or a declared inflow, `sim.py:363, 649`) has three triggers and would be simpler as "on whenever the sim has a unit and agents". Confidence: high on removing the shortcuts; medium on the rate convention.

### 7. Variable-name lookup covers some modules and not others

**Issue.** `01-objects.md` §Names: "A module's name is its `name=`, else the variable it is bound to where it is passed to `ss.Sim`, else the lowercased class name." The code (`sim.py:113`) looks up the variable only for `type(mod) is ss.Disease`, `ss.modify`, `ss.Delivery`, and `ss.PartnerNet`. Verified (`s2_constructors.py` 15, `s7_misc.py` F): `measles = ss.Disease(...)` gives `sim.diseases.measles`, but `flu = MySIR()` (a `Disease` subclass) gives `mysir` and `school = ss.RandomNet()` gives `randomnet`.

**Why it matters.** The rule is either "the variable names the module" or it is not; a user who learns it from `measles` will expect `sim.results.flu`.

**Proposal.** Apply the lookup to every module whose `_default_name` is still true, or drop it for modules and keep it only for the pieces inside a module (states, transitions, parameters), where it is doing real work. I lean to the first: it is one `isinstance` check (`sim.py:113`) and the spec already says so.

**Cost.** A library module bound to a variable (`hiv = ssl.HIV()`) would be renamed `hiv` (same) or `my_hiv` (different, and then `sim.results.hiv` moves). That is the spec's rule, and `name=` overrides it. Confidence: medium.

### 8. Time in a sim with no unit leaks the implementation's unit

**Issue.** `describe()` says `unit none: time is in timesteps`, but (`s7_misc.py` A):

```text
t: Timeline(days(0)-days(10); dt=days(1); now=days(10); ti=10/10)
unit None unit_base day dt day
```

because `init_unit()` sets `self.unit_base = unit or 'day'` (`sim.py:1063`) and builds the timeline in days. Any bare `birth_rate=20` then warns about years versus timesteps (finding 6), and `s.t.start` prints `days(0)`.

**Proposal.** A unitless sim should have a unitless `Timeline` (steps). If the time machinery cannot do that cheaply, `Timeline.__repr__` and `describe()` should at least agree. Confidence: medium (the time reviewer owns the mechanism).

### 9. Smaller things a user meets

- `sim.new()`, `sim.check_dt()`, `sim.R0()`, `sim.gradient()`, and `sim.restart()` raise for a coded sim ("only available for sims with declared states and transitions", `sim.py:428, 464`) because `self.inputs` is only kept when a module has a `graph` (`sim.py:85`). `new()` and `check_dt()` are general ideas; keep the inputs for every sim (they are already deep-copied once by `sc.mergedicts(..., _copy=copy_inputs)`; `sim.py:86` copies them a second time).
- `results=` from the spec's `ss.Sim` signature (`01-objects.md` line 191) is not accepted: `KeyNotFoundError Key(s) ['results'] not found` (`s7_misc.py` C). Either implement or strike from the spec.
- `ss.Sim(sir, n_agents=None)` fails with `Must supply n_agents, a people object, or a popdict` (`parameters.py:399`); `popdict` has not existed for years, and `sim.new()` documents `None` as "back to the default".
- `sir.S.n` works after an ODE run (`CountState(S, n=9861.22)`) and raises `AttributeError: 'numpy.ndarray' object has no attribute 'n'` after an agent-based run (`s7_misc.py` B), although the spec says "`I.n` is the plain count, the same number as the result `n_I`". A `.n` property on `BoolState` (`self.sum()`) would make it one spelling.
- Running a finished sim raises `AlreadyRunError` (`sim.py:1832`), not `E1003 RunAfterFinish` as `15-errors.md` lists; the code is registered in `transitions.py:37` but never raised.

## Simpler codebase

### 1. `Sim` carries two engines' worth of helpers

**Issue.** 76 public methods and 8 properties on `ss.Sim` (`s2_constructors.py` 17). Beyond the structure reviewer's line counts, the point for a maintainer is which of them belong to the sim at all:

- Stratification geometry, agent-only or compartment-only, with no sim state of their own beyond `self.strats`: `fold` (694), `fold_matrix` (700), `group_vals` (708), `grid_index` (715), `init_ragged` (653), `folded` (899), `level_raw` (903), `group_raw` (822-897, a 75-line cache), `groups_changed` (807), `dim_index` (799), `init_strat_vals` (908). These read `self.strats`, `self.group_shape`, `self.group_coords`, `self.group_fold`, `self.group_inside`, `self.group_levels`, `self.group_mults`, `self.n_groups`, `self.group_split`, `self.birth_split`, `self.joint`, `self._group_cache`, `self._level_cache`, `self._group_dirty`: fourteen attributes that are one object (`Groups`) spread over the sim.
- Solver choice and kernels: `init_solver` (292-400), `agent_reasons` (963), `try_functions` (996), `init_crossings` (733), `get_declared` (791), `kernel_only` (1886), `kernel_reason` (1866), `restart` (1923), `gradient` (1895), plus `self.declared`, `self.mixing`, `self.mixed`, `self.crossings`, `self.cross_strats`, `self.modifiers`, `self.overlaps`, `self.clipped`, `self.jax`, `self.integrator`, `self.solver_reason`, `self.unit_base`, `self.unit_source`, `self.units_disagree`.
- Starsim 3 init phases exposed as public methods: `init_time`, `init_people`, `init_module_attrs`, `init_modules_pre`, `init_dists`, `init_people_vals`, `init_modules_post`, `init_results`, `init_data`, `init_rescale`, `reset_time_pars`, `lock_pars`, `set_par_values`. None is meant to be called by a user.

**Proposal.** `sim.groups = ss.Groups(sim)` owning the first list; `sim.engine`-level or `ss.solvers`-level functions for the second (`agent_reasons(sim, modules)` and `try_functions(sim)` are already pure functions of the sim); a leading underscore on the init phases. `Sim` would keep roughly: `init`, `run`, `describe`, `solvers`, `requires`, `stratify`, `what_modifies`, `check_dt`, `hash`, `to_spec`, `to_code`, `new`, `calibrate`, `plot`, `summarize`, `to_df`, `save`, `shrink`, `add_module`, `get_module(s)`, `R0`, `gradient`, `restart`, and the step hooks the loop calls. That is the spec's method table (`01-objects.md` §Methods of a sim) plus Starsim 3's. Confidence: high.

### 2. The loop runs people hooks in a compartmental sim

**Issue.** When the ODE sim cannot take the kernel fast path (an inserted function, `verbose>=1`, diagnostics), `Loop.collect_funcs()` schedules `people.step_die`, `people.finish_step`, `sim.people.remove_dead` and friends for a `People(1)` that never dies (`s4_demog_lifecycle.py` B: `declared ode loop (with insert) labels: ['<lambda>', 'people.finish_step', 'people.step_die', ...]`). Harmless, but it is one more way the one-row `People` is pretending to be a population. Fixing finding 4 (user side) removes this.

### 3. Two name-resolution passes

**Issue.** `Sim._name_modules()` (`sim.py:109-121`) walks the caller's frame for modules; `Graph.resolve_states()`/`find_name()` (`transitions.py:1075`) does the same for states, transitions, and parameters at `ss.Disease(...)` time; `SimPars.convert_modules()` (`parameters.py:559-565`) names modules from dict keys (`networks=dict(main=..., casual=...)`); and `SimPars.validate_name()` (`parameters.py:476`) renames duplicates. Four places decide a module's name. One `name_modules(modules, frame)` called once from `Sim.__init__` would hold the whole rule (keyword, `name=`, variable, class name, uniqueness) in one function and fix finding 7 at the same time.

### 4. `Disease.__init__` sniffs its arguments to decide which kind of disease it is

**Issue.** `diseases.py:69-76`: if the first positional is `None` or a dict it is `pars`; if any arg or member or keyword is one of six declared types, or a reserved key is present, build a graph; otherwise behave like Starsim 3's `Module.__init__`. This is what lets one class be both the declared constructor and the base of `Infection`. It works, but it means `ss.Disease` has two constructors with disjoint keyword sets chosen by type sniffing, and `ss.SIR(beta=0.1)` and `ss.Disease(beta=0.1)` go down different paths. If finding 1 (user side) is done, `Infection` can keep the Starsim 3 constructor and `Disease` can keep only the declared one, and the sniff goes away.

### 5. Dead or near-dead code in `people.py`

- `Filter` (`people.py:774-941`, about 170 lines, with `People.filter()` at 441): used in one test (`tests/test_people.py:64`) and nowhere in `starsim/` or `starsim/library/`. Candidate for removal or for `debugtools.py`.
- `People.__iter__` yields a `Person` per agent (`people.py:134`), which is an invitation to the slowest possible loop; `person()` on request is enough.
- `People.scale_flows()` (`people.py:457`): no callers anywhere in `starsim/` or `tests/` (verified by grep).

### 6. `Births.__init__` keeps three distributions for one draw

**Issue.** `demographics.py:64-68`: a `bernoulli` for the CRN path, a `poisson` and a `randint` for the non-CRN path, selected on `ss.options.crn` at step time (`demographics.py:116-126`). The non-CRN count path is an optimization that duplicates the CRN path's logic; `ss.bernoulli.filter()` could do the same switch internally for every module that scans agents with a small probability, rather than each module carrying two code paths.

## Spec and implementation

Beyond the departures `implementation.md` already lists:

- **Class form and the solver** (user finding 3): `01-objects.md` and `00-overview.md` present a `step()` override as adding behavior without changing the model; `sim.py:982-984` treats it as a reason to run as agents and ignores `compartmental = True` for declared diseases. One of the two has to move; I argue the code.
- **Library models** (user finding 1): `01-objects.md` §Library models promises `describe()` in full and `equations()`; neither holds for `ss.SIR()`.
- **`pars=`** (user finding 2): the spec's own signature and prose disagree; the code resolves it by sniffing. B1's "first positional argument" question is answered: the model, with the Starsim 3 dict also accepted.
- **Module names from variables** (user finding 7): spec says every module; code does four classes.
- **`results=`** is in the spec's `ss.Sim` signature and not in the code.
- **`E1003 RunAfterFinish`** is in `15-errors.md` and registered in `transitions.py:37` but the sim raises `AlreadyRunError`.
- **Build order**: `10-execution.md` §Building the sim step 7 says `init()` writes the model spec and computes `sim.hash()`; the code computes both lazily (`to_spec()` and `hash()` call `init()` if needed, `sim.py:1281, 1298`). The lazy form is better; the spec should say so.
- **`sim.run(solver=)`**: spec says the second wins; code raises `RuntimeError` if the sim is already initialized (`sim.py:1818`). Reasonable; the spec should say "before `init()`".
- **People in a compartmental run** (B1 "`sim.people.age`: the age group's midpoint"): implemented as the midpoint on a one-row `People` whose repr crashes. The guess is answered but the vehicle is not finished (user finding 4).
- **The denominator** (B1): still written by the user, and only as agents does `n_alive` exist (user finding 5).
- **Lifecycle table** (`01-objects.md` §Lifecycle) lists `init_pre`, `init_post`, `step`, `update_results`, `finalize`; the loop also calls `start_step`, `step_state` (diseases), `step_die` (via people), `finish_step`, and `finalize_results`. The declared and coded diseases have the same lifecycle (verified: identical label sets `sir.start_step, sir.step_state, sir.step, sir.update_results, sir.finish_step` in `s4_demog_lifecycle.py` B), which is good; the spec table should list the real hooks, since `10-execution.md`'s row table already does.
- **Spec is over-specified** on `requires`/`sim.requires()` (a 50-line method, `sim.py:1209-1256`, for metadata the spec itself calls optional and never enforced) and on `track_writes`/`what_modifies` (B1 already asks whether to drop it). Neither is wrong; both are surface that could wait for a vignette that needs it.

## Bugs

1. **`print(sim.people)` or `repr(sim.people)` raises in a compartmental run.** `People.brief()` reads `self.age.mean()` (`people.py:189`), and `age` has been deleted from the instance (`sim.py:281`). Repro (`s8_declared_sir.py` H): `ss.Sim(sir, stop=10).run().people` at a REPL gives `E804 CapabilityError: people.age was asked for, but it is a per-agent attribute...`. Fix: `brief()` should check `_no_groups`, or (finding 4) not have a `People` there at all.

2. **A lazy age condition whose threshold is not a bin edge crashes with a NumPy error, not `E804`.** `implementation.md` says the lazy form "is checked". Repro (`s6_age_bug.py`): `ss.Sim(sir, stratify=dict(age=[0,15,65]), connectors=ss.modify(sir.pars.beta, multiply=0.5, where=ppl.age < 10), stop=10).run()` gives

   ```text
   File "starsim/solvers.py", line 4079, in infect_groups
       rate = np.broadcast_to(np.asarray(self.eval(tr.rate), dtype=float), ng)
   ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (10000,)  and requested shape (3,)
   ```

   (the condition was evaluated against the 10,000-agent `ss.People(1)`-derived array rather than the 3 groups). With the threshold at 15 the same sim runs and `describe()` shows `where=(people.age < 15)`.

3. **`ss.Sim(sir, n_agents=None)` fails with a stale message** (`parameters.py:399`: "Must supply n_agents, a people object, or a popdict"), while `sim.new(n_agents=None)` is documented to mean "back to the default".

4. **`sir.S.n` depends on the solver** (user finding 9): `AttributeError` after an agent-based run, a number after an ODE run.

## Keep

- **One module lifecycle for declared and coded diseases.** `Disease.step_state()`/`step()` dispatch to `self.engine` when there is a graph and fall through to the subclass otherwise (`diseases.py:281-351`), and the loop labels are identical for the two kinds. A declared disease is scheduled exactly where the spec's row table says (spontaneous transitions at `step_state`, contact-mediated at `step`), with no new loop concepts. This is the property that lets v4 sit on Starsim 3 without a second scheduler; protect it.
- **The ordered, printable loop** (`loop.py`, `sim.loop.to_df()`, `insert()`), unchanged from Starsim 3 and sufficient for the spec's within-step chains.
- **`describe()` output.** The per-line `[guessed: ...]` marks (`sim.py:1089-1093`) are exactly what the spec asked for and read well; the schedule line groups by timestep as `10-execution.md` requires.
- **The `total_pop`/`n_agents` defaults** (table above) are correct in every combination I tried, including a `people=` object with `total_pop`.
- **Deferred removals via `request_death`/`step_die`** (`people.py:471-529`), and the declared `ss.death` using the same path (verified: `deaths(sir) 18.0; new_deaths(people) 18.0` agree).
- **`ss.Births`/`ss.Deaths`/`ss.Pregnancy` as library modules** for data-driven and agent-only demography, per the structure reviewer; the declared `ss.birth`/`ss.death` for the compartmental case. The division is right; only the `SimPars` shortcut layer on top of it is not.
- **`solver='auto'` printing its reason** (`solver: abm ("sirx" defines its own step())`). Even where the reason is wrong (finding 3) the user can see it, which is the spec's risk 2 mitigation working.
- **The `@ss.required()` method-call check** (`modules.py:130-166`), which catches the forgotten `super()` call that would otherwise be a silent no-op.

## Outside my area

- `W103`: with `dur=ss.days(10)` and `beta=ss.peryear(...)` and no `unit=`, the smallest unit wins with a note. The same choice gives a sim with `stop=ss.weeks(12)` weeks as its unit (known issue). An error asking for `unit=` when the typed values disagree would be simpler to explain than the ranking rule (`sim.py:1053-1060`); the time reviewer should weigh it.
- `ss.lognorm_ex(mean=6, std=1.0)` with a typed `mean=ss.years(6)` and a bare `std=1.0` prints `std 1 instead of 1 years` in the Erlang note (`s8_declared_sir.py`): the note's wording is confusing when the two are the same number.
- The declared disease in an agent-based run registers 14 `People` states for a three-state SIR (`S, ti_S, ti_exit_S, exit_S, I, ti_I, ...`, `s7_misc.py` B), including exit times for states with no dwell. The solvers reviewer may want to allocate `ti_exit_*`/`exit_*` only for states with a dwell time.
