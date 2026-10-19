# Core modeling language review: states, transitions, parameters, types and time

Reviewer scope: `starsim/transitions.py`, the declared-model parts of `starsim/diseases.py`, `starsim/parameters.py`, `starsim/time.py` (bare numbers and units), `starsim/modules.py` (naming and construction), against design docs 01–04 and best-practices `model-structure.md`, `time-and-units.md`, `parameters-and-distributions.md`. All repros were run with `/software/conda/envs/v4/bin/python` on branch `v4`; scripts are in the session scratchpad.

## Summary

The core language is in good shape conceptually: one `ss.State`, one `ss.Transition`, operators that read like the equations, a symbolic expression layer that lets one model run as an ODE or as agents, and `describe()` as the place where guesses are shown. The problems are concentrated in three places. First, **meaning that depends on context the user did not think of as part of the model**: the unit of a bare number is set by `dt` or by how `start`/`stop` are written (changing only `dt` from a day to a week makes an epidemic vanish), a keyword number on a disease declares a new parameter rather than setting the existing one (`SIR(beta=0.3)` is silently ignored, and typos pass), and `dur=` and `rate=` (declared equivalent) differ as agents, in branching, and in `stages=`. Second, **silent outcomes where the principles promise a guess or an error**: the flagship two-line SIR seeds nobody and plots a flat line, five specified checks (E101, E102, E604, E606, W207) are never raised, and `describe()` misreports several of the guesses it exists to show. Third, **structure that is larger than the concept**: `ss.Disease` is two unrelated classes behind one name (so the library's own SIR is not written in the language and cannot be described), the Graph mutates and claims the user's objects (so the same transitions cannot build two diseases, and `@`/`>>` alias), and `transitions.py` holds geography, stratification, and parameter forms alongside the language. The largest simplification available is to make `ss.Disease` always declared and rewrite `ss.SIR`/`ss.SIS`/`ss.SEIR` in the language; the cheapest high-value change is to print R0 and "nobody is infected" in the line that already announces the solver.

### What is good and should be kept

- The operator form (`S >> I @ beta * I`, `I >> R @ ss.days(10)`) and the rule that the kind of a transition is derived from its rate, not chosen. It is the best idea in the design and it works as described.
- The bare state as "exposure through the model's routes" versus `.n` as a count, with the linearity check (E208) and the not-people check (E211). It is what lets one model text run with homogeneous mixing, a contact matrix, or a network.
- States that are not people, inferred from connectivity, with no separate class; the cholera and malaria examples from `03` run as written.
- The coded-error table and the habit of naming the fix in the message (E205, E209, E210, E211 are good examples).
- `ss.null` as a named object rather than `None`, for the precedence reason given in the spec.
- `sim.check_dt()`, which holds the model fixed and varies only the step; it is the right tool and should be what changing `dt` means everywhere (finding 1).

## Findings

### 1. The meaning of a bare number depends on how the sim's time is written, including `dt`

- **Type:** design (high impact)
- **Evidence:** one model, `S >> I @ 0.3 * I` (bare) and `I >> R @ ss.days(10)` (typed), run for the same 12 weeks written five ways (`t8.py`):

| Sim arguments | Unit chosen | Peak `n_I` | Final `n_R` |
|---|---|---|---|
| `stop=84` | day (from the typed `ss.days`) | 3036 | 9342 |
| `start='2020-01-01', stop='2020-03-25'` | day | 3036 | 9342 |
| `stop=ss.weeks(12)` | week | 100 | 172 |
| `stop=84, dt=ss.weeks(1)` | week | 100 | 173 |
| `start=2020, stop=2021` | year, and `dt` = 1 year (2 time points) | 100 | 101 |

The unit rule (`02-types.md` §The sim's unit; `Sim.init_unit`, `sim.py:1026–1066`) ranks `dt`, then the form of `start`/`stop`, above the typed values in the model, so changing the timestep from a day to a week (a numerical choice) silently turns `beta` from 0.3/day into 0.3/week and the epidemic disappears. The only signal is `note: unit='week' (from dt)`, which, unlike `W103`, does not name the bare numbers it has just given a unit to. `sim.check_dt()` holds the unit fixed, so it reports a discretization error of ~25%, while the user's own change of `dt` changed the model by 98%. The known issue "with `stop=ss.weeks(12)` and no `unit=`, the sim runs in weeks even if every rate is per day" is a symptom of the same rule. With `start=2020, stop=2021` a 10-day disease also gets a 1-year step by default (W102 fires as agents; the ODE just returns two time points).
- **Proposal:** make the unit of bare numbers a property of the model, not of the run. (1) `dt` never sets the unit; it only sets the step. (2) The model's own typed values (and a `unit=` on the module) come before the sim's dates. (3) If bare numbers coexist with units from two different sources that disagree (typed days in the model, year-formatted `start`), raise an error asking for `unit=` rather than picking one; the spec already lists this as an open question and this repro is the case for answering "error". (4) The default `dt` should come from the unit chosen, not from the format of `start`/`stop`. Before/after:

```python
# before: the epidemic vanishes when only dt changes
ss.Sim(sir, stop=84, dt=ss.weeks(1))     # beta now per week
# after: beta stays per day (from ss.days in the model); dt is just the step
ss.Sim(sir, stop=84, dt=ss.weeks(1))     # note: unit='day' (from ss.days(10) in 'recovery'); dt = 1 week
```

- **Cost:** a reordering of the rule table plus one error; breaking only for sims that currently (and probably unknowingly) depend on `dt` or year-format dates to define their bare numbers. Simpler to explain: "bare numbers are in the model's unit".

### 2. A number passed as a keyword declares a new, unused parameter instead of setting the existing one, and typos pass silently

- **Type:** bug / design
- **Evidence:** `Graph.collect` (`transitions.py:1193`) turns any keyword whose value is a number into a new `ss.par`; `resolve_pars` then sees the clash and renames the model's own inline value to `beta_<transition>` (`:1426`). Repro (`t12.py`):

```python
class SIR(ss.Disease):
    S, I, R   = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery  = I >> R @ ss.days(10)

SIR(init_prev=0.01).R0()                     # 0.5
SIR(init_prev=0.01, beta=0.3).R0()           # 0.5   <- beta=0.3 ignored
SIR(init_prev=0.01, beta=0.3).pars           # beta: 0.3, beta_infection: 0.05, dur: 10 days
SIR(init_prev=0.01, pars=dict(beta=0.3)).R0()   # 3.0  <- only pars= works
SIR(init_prev=0.01, bta=0.3).pars            # bta: 0.3, beta: 0.05   <- typo accepted, no E602
```

The same happens in the constructor form (`ss.Disease(S >> I @ 0.05*I, ..., beta=0.3)` gives `beta=0.3` unused and `beta_S_to_I=0.05` used). This breaks the most natural thing a Starsim 3 user will write (`SIR(beta=0.3)`), contradicts `04-parameters.md` ("a misspelled parameter that is silently ignored is one of the hardest bugs to find"), and contradicts the equivalence rule (keyword and `pars=` mean different things). `ss.Sim(pars=dict(bta=...))` does raise E602, so only the module-keyword path is broken.
- **Proposal:** one rule: a keyword whose name is a parameter of the module (declared, class-level, or auto-named from a slot) sets its value; any other plain-number keyword raises `E602` with the closest names. Declaring a *new* named parameter is done with `ss.par` (or a variable name), never by a bare keyword. Optionally also flag any registered parameter that no transition, outcome, or init uses (`W6xx UnusedParameter`), which catches the remaining typo cases.

```python
# after
SIR(beta=0.3)          # sets beta
SIR(bta=0.3)           # E602 UnknownParameter: 'sir' has no parameter 'bta'. Did you mean 'beta'?
```

- **Cost:** small (move the keyword-number branch from "declare" to "override, else error"); breaking only for code that declared a parameter by a keyword number and referenced it by name elsewhere, which is rare and can use `ss.par`.

### 3. The flagship two-line SIR does nothing: no seed by default

- **Type:** design / doc-code mismatch
- **Evidence:** the example in `00-overview.md`, `01-objects.md`, `02-types.md` and `03-states-and-transitions.md` (`ss.Sim(sir, stop=100).run().plot()`) runs with `init: S = 1.0 [guessed: nothing given]`, so `n_I` is 0 at every step and the plot is flat. No warning is printed.

```python
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
sim = ss.Sim(sir, stop=100).run()
sim.results.sir.n_I.max()   # 0.0
```

- **Proposal:** a disease that has a contact-mediated transition and no `init`/`init_prev` should default to a small seed (Starsim 3's `init_prev=0.01`), printed as a guess (`init: I = 0.01 [guessed: a transmission with nothing infected never starts]`). Principle 3 ("guess the default") fits exactly: nobody writes `ss.transmission` and wants zero cases. If a guess is felt to be too magical, the alternative is a `W2xx NothingInfected` note at `init()`; silence is the one wrong answer. Separately, the doc example's numbers (`beta=0.05`, `dur=10`) give R0 = 0.5, so even seeded it dies out; the showcase should use something like `beta=0.3`.
- **Cost:** small (a few lines in `Graph`/init); changes the result of any model that relied on starting with nobody infected, which is unlikely to be intended.

### 4. `dur=` and `rate=` are "the same model" in the spec but not as agents, and people who start in a state stay one step too long

- **Type:** bug / doc-code mismatch
- **Evidence:** `03-states-and-transitions.md`: "`dur=ss.days(10)` and `rate=ss.perday(0.1)` say the same thing and produce the same model spec." As agents they are realized differently: a `dur=` is a sampled dwell time rounded to a step (the departure "every agent's dwell time is sampled, including exponential ones"), while a `rate=` is a per-step hazard `1 − exp(−r·dt)`. Measured for people infected during the run, with a mean of 2 days (`t25.py`, 50,000 agents):

| dt | Driver | Share leaving in the first step | Exact (continuous) |
|---|---|---|---|
| 1 day | `@ ss.days(2)` | 0.528 | 0.393 |
| 1 day | `@ ss.perday(0.5)` | 0.394 | 0.393 |
| 0.25 day | `@ ss.days(2)` | 0.168 | 0.118 |
| 0.25 day | `@ ss.perday(0.5)` | 0.117 | 0.118 |

And for people who *start* in a state (`init=`), a sampled dwell is one step longer than for anyone else (`t23.py`, `t24.py`): with `I >> R @ ss.constant(ss.days(2))` and `dt = 1 day`, people seeded at `t = 0` leave at `t = 3`, after three days (`n_I = [1000, 1000, 1000, 0, ...]`), while people infected at `t = 1` leave at `t = 3`, after two days, as they should. With an exponential `dur=` the seeds' first-step exit is 0.214 instead of 0.393. So the initial cohort, which drives the early epidemic and R0-type comparisons against the ODE, is infectious for an extra step.

The same split leaks into structure: `resolve_exits` (`transitions.py:1461–1466`) files a transition with `dur=` as "timed" and one with `rate=` as a "hazard", and only a timed one can silently take the remainder of a branch split. So `severe = ss.progression('I -> H', p=0.1, dur=5)` plus `mild = ss.progression('I -> R', dur=10)` makes `mild` the 0.9 branch (and `describe()` does not show that guess: it prints no `p` for `mild`), while the same with `mild = ss.progression('I -> R', rate=0.1)` raises `E205 ... sum to 0.1, not 1` (`t26.py`).

And `stages=` is silently ignored when the dwell time is given as a rate (`t31.py`): `ss.Transition(I, R, rate=0.1, stages=3)` gives exactly the exponential curve (`n_I[10]/n_I[0] = 0.3679`), while `ss.Transition(I, R, dur=10, stages=3)` gives the Erlang one (0.4232); `describe()` shows `rate = 0.1` with no mention of stages.

- **Proposal:** (1) Fix the seed off-by-one (initial occupants should be scheduled as if they entered at `t = 0`). (2) Normalize at build time: a constant `rate=` on a spontaneous transition is a dwell time (`dur = 1/rate`), so it is filed, staged, and branched exactly as `dur=` is. (3) Realize an exponential dwell (scalar `dur=` or a rate) as a per-step hazard in agents, exactly as a `rate=` is, and sample only non-exponential distributions. That makes the spec's equivalence true, matches the ODE's discretization, and fixes the known issue "an `adjust` on a duration does not follow a person who later moves to another level" for the commonest case, because a hazard is re-read every step. If sampling is kept for exponentials (e.g. for speed), round with the same convention that the hazard implies, and test `dur=d` against `rate=1/d` as agents.
- **Cost:** moderate (engine change in the agent solver plus a test); changes agent results slightly, toward the ODE.

### 5. `ss.Disease` is two unrelated classes behind one name

- **Type:** design / code simplicity / doc-code mismatch
- **Evidence:** `Disease.__init__` (`diseases.py:65`) decides by inspecting its arguments whether to build a graph or behave as a Starsim 3 module, and nearly every method then branches on `self.graph is None` (`describe`, `equations`, `diagram`, `R0`, `to_json`, `init_pre`, `init_post`, `start_step`, `step_state`, `step`, `update_results`, `finalize_results`, `stratify`). The two halves share almost nothing: the graph path routes everything through `self.engine`; the code path is Starsim 3's `Infection`. Consequences: `ss.Disease(beta=0.1)` alone is a code-form module with no transitions; `ss.Disease(dict(...), infection=...)` treats the dict as `pars`. And `01-objects.md` §Library models says "a library model is an ordinary module: `describe()` prints it in full and `equations()` works on it", but (`t11.py`):

```python
ss.SIR().equations()   # ValueError: "sir" has no declared transitions, so it has no equations to print
ss.SIR().describe()    # SIR 'sir': written as code (states: susceptible, infected, recovered, ti_infected, ...)
ss.Sim(diseases='sir', networks='random').describe()   # only "schedule  every step: randomnet, sir"
```

So the library's own SIR, SIS, and SEIR are the old code models, not models written in the language, and the user meets two SIRs: `ss.SIR` (states `susceptible`/`infected`, results `n_infected`, parameter `dur_inf`, can be described only as code) and the declared `ss.Disease` SIR (states `S`/`I`, results `n_I`, parameter `dur`).
- **Proposal:** rewrite `ss.SIR`, `ss.SIS`, `ss.SEIR` as declared models (class form, as in the docstring of `ss.Disease`), keeping their Starsim 3 parameter names (`beta`, `dur_inf`, `init_prev`, `p_death`) and state names as aliases if compatibility matters. Then move the code path into `ss.Infection` only (which already exists for exactly that), so `ss.Disease` is always declared and the `graph is None` branches go away. A subclass of `ss.Disease` that wants imperative code still has `step()`.
- **Cost:** large but mostly deletion: about 200 lines of branching in `Disease`, plus rewriting three library models (~300 lines of Starsim 3 code become ~30 declared lines). Breaking for code that reads `sir.infected` or `n_infected` from `ss.SIR`, unless aliases are kept. This is the single biggest simplification available in my area: one disease class, one SIR, and "the library models are written in the language" becomes true.

### 6. A state, once used, belongs to one module forever, so the same transitions cannot build two diseases

- **Type:** design / API
- **Evidence:** `Graph.resolve_states` sets `state.module = self.module` on the user's objects and raises `E206` if another module uses them (`transitions.py:1273`). Building two variants from the same ingredients fails (`t2.py`):

```python
S, I, R = ss.states(3)
infection = S >> I @ beta * I
recovery  = I >> R @ dur
sir  = ss.Disease(infection, recovery, init_prev=0.01)
ss.Sim(sir, stop=100).run()
sir2 = ss.Disease(infection, recovery, init_prev=0.02)
# E206 DuplicateDeclaration: state "S" is used by transitions in two modules, "sir" and "disease"
```

The message calls the new module "disease" although the user wrote `sir2` (a module's variable name is found only when it is passed to `ss.Sim`), and suggests nothing to do. The Graph also rewrites the user's transitions in place (`tr.rate = beta * source`, `tr.beta = None`, `tr.src` strings replaced by states, `per_capita` births multiplied by `N`), so the objects the user holds are no longer what they wrote, and `infection is sir.infection` is True. Meanwhile, passing the *same* `sir` to two sims works (each sim copies it), so the rule is "share modules, never share their parts", which is the reverse of what a user would expect from "the model is data".
- **Proposal:** treat states, transitions, and parameters as immutable values; the Graph deep-copies what it is given (one `sc.dcp` of the args with a shared memo, so identity between states is preserved) and binds the copies. Then `E206` across modules disappears except for the real case it was meant for (cross-module *references* in a rate, e.g. one disease's `I` in another's rate, which should resolve by name to the other module). `sir.infection` is then the bound copy, and the user's `infection` is unchanged and reusable.
- **Cost:** moderate: references to another module's states (which rely on identity today) need to resolve to the bound copy; probably 30–60 lines in `Graph` plus tests. Removes a class of confusing errors and the in-place mutation.

### 7. `@` and `>>` mutate and return the same transition, so partial transitions alias

- **Type:** bug
- **Evidence:** `Transition.__matmul__` (`transitions.py:730`) sets `self.rate` and returns `self`; `_set_src` (`:695`) does the same for `>>`. Repro (`t3.py`):

```python
S, I, R = ss.states('S I R')
base = S >> I
fast = base @ 0.2
slow = base @ 0.1
fast, slow, fast is slow     # (S -> I @ 0.1, S -> I @ 0.1, True)

into = I @ 0.1
a = S >> into; b = R >> into
a, b, a is b                 # (R -> I @ 0.1, R -> I @ 0.1, True)
```

Arithmetic on a transition (`_arith`) already copies, so the operators are inconsistent with each other. Anyone factoring a common driver (`into_I = I @ beta * I` then several sources) gets one transition with the last source.
- **Proposal:** make `>>` and `@` return a copy whenever the operand already has the field being set (or always, since transitions are small). Keep the in-place path only for the internal chain `S >> (I @ x)` where the right-hand side is fresh.
- **Cost:** a few lines; no API change.

### 8. Five of the core language's checks are specified and listed but never raised: E101, E102, E604, E606, W207

- **Type:** bug / doc-code mismatch
- **Evidence:** a scan of `starsim/` for each code in the `codes` table (`transitions.py:31–41`) finds no raise site for `E101 IncompatibleTypes`, `E102 WrongDimension`, `E604 UnvaluedParameter`, `E606 OutOfBounds`, or `W207 UnreachableState` (nor, outside my area, `E1003`, `W309`, `W1304`). `ss.perday(0.1) + ss.probperday(0.1)` does raise, but as a plain `TypeError` with the message "you added 0.1 + 0.1", which names neither type. `rate=ss.peryear(0.1) * ss.days(3)` (the exact example in `02-types.md` for E102) is accepted and becomes a bare rate of 0.00082 per day. Repro for the parameter checks (`t14.py`):

```python
beta = ss.par(5, bounds=[0.01, 1])            # outside its bounds
ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01), stop=5).run()
# runs, n_I = 6756; no E606

beta = ss.par()                               # no value, no bounds
# solver='ode': runs, every result is NaN, no message
# n_agents=1000: TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'
```

Also, `ss.par(bounds=[0.001, 1.0])` starts at the arithmetic midpoint 0.5005 (`transitions.py:208`), while `04-parameters.md` says "the middle of its bounds (on the search scale)", which for a rate is log scale (0.0316); and the midpoint is computed eagerly at construction, so it is not reported by `describe()` as a guess.
- **Proposal:** implement them or delete them from the spec and the table; a code table that promises checks the library does not make is worse than none, because the spec tells users (and AIs) that these mistakes are caught. E604/E606 are two short loops over `graph.pars` at `init()`. E102 needs the slot check the spec describes (a typed expression whose dimension is not 1/time in a rate slot). E101 is a matter of raising `ss.error('E101', ...)` with the two type names in the existing `TypeError` branch. Compute the starting value of a bounds-only parameter lazily at init, on the search scale of its slot, and mark it `[guessed: middle of bounds]` in `describe()`.
- **Cost:** small; no API change.

### 9. Naming from the caller's variables breaks on the first refactor into a helper function

- **Type:** design / API
- **Evidence:** states from `ss.states(3)` get their names when the module is built, by searching the frame that calls `ss.Disease` (`find_name`, `transitions.py:1075`; `resolve_states`, `:1253–1260`). Any model whose states are made in one function and whose disease is built in another fails, which is how models get organized as soon as they are reused (`t18.py`):

```python
def make_transitions():
    S, I, R = ss.states(3)
    return S >> I @ 0.3 * I, I >> R @ ss.days(10)

def build():
    infection, recovery = make_transitions()
    return ss.Disease(infection, recovery)

build()   # E210 UnnamedState: state number 1 of the module has no name. It is used in ['infection'] ...
```

Binding one state to two names (`Inf = I`) also raises E210, and transitions written inline get fallback names (`S_to_I`) that then become result names. The same lookup names transitions and `ss.par`s, and a module is named by a second, different lookup at `ss.Sim` time, so the user's mental model has to include "which frame is searched, and when". `03-states-and-transitions.md` already flags this as "slightly magical; under review".
- **Proposal:** name at creation, not at build. For `S, I, R = ss.states(3)` and `I = ss.State()`, the assignment targets are in the caller's bytecode at the moment of the call (the instructions after the call are `UNPACK_SEQUENCE 3` and then the stores; on the project's Python 3.13 a 15-line prototype reading them gets `S, I, R` inside a function, where 3.13 fuses two stores into `STORE_FAST_STORE_FAST`, and at module level), which is what `varname`/`executing` do; that is robust to helpers, aliases, and returning states from functions, and the names exist before any transition is written, so reprs show `S -> I` rather than `<unnamed state> -> <unnamed state>`. Where the targets cannot be read (a comprehension, a dict), keep E210 with the same advice. If that is too much magic, the alternative that removes all of it is to document only `ss.states('S I R')` (sympy's idiom), at the cost of typing the names twice. Either is simpler than the current two-stage lookup.
- **Cost:** ~30 lines plus tests for the bytecode approach (CPython-specific, so keep the frame lookup as a fallback for other interpreters); none for the documentation approach.

### 10. Names of inline parameters change when the model grows, and `sir.infection.beta` is not addressable

- **Type:** design / doc-code mismatch
- **Evidence:** `resolve_pars` names an inline value after its slot, qualified by the transition only when two would clash (`transitions.py:1421–1427`). So the name of an existing parameter depends on what else is in the model (`t17.py`):

```python
ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10)).pars.keys()
# ['beta', 'dur']
# add one transition with a duration:
... waning=ss.progression('R -> S', dur=100)
# ['beta', 'dur_recovery', 'dur_waning']    <- 'dur' is gone
```

Anything that referred to `sir.dur` (a `pars=` file, `ss.change(sir.pars.dur, ...)`, a saved calibration, a spec hash) breaks when an unrelated transition is added. And `04-parameters.md` §The default says inline values are "addressable through their transition, as `sir.infection.beta` and `sir.recovery.dur`"; `sir.recovery.dur` works, but `sir.infection.beta` is `None` because the Graph folds `beta` into `rate` and clears it (`transitions.py:1328–1329`).
- **Proposal:** give every inline value one stable, structural name, `<transition>.<slot>` (`infection.beta`, `recovery.dur`), and keep the short `beta`/`dur` only as an alias that resolves when it is unique (and raises a clear "ambiguous: infection.beta or reinfection.beta" otherwise). Keep `Transition.beta` pointing at the coefficient after resolution. Names the user gave (`ss.par`, a variable, a keyword) are untouched.
- **Cost:** moderate; changes the printed names of inline parameters (`beta_infection` → `infection.beta`) in specs and hashes, so it is breaking for saved specs, but the current names are not stable anyway.

### 11. Initial values are a fraction or a count depending on their size, and `1` differs from `1.0`

- **Type:** design / API
- **Evidence:** `03-states-and-transitions.md` §Initial conditions rule 1: "a value below 1 is a fraction of the population; a value above 1 is a count. A Python `int` of exactly 1 is one individual and a `float` of exactly 1.0 is everyone." So (`t5.py`, total_pop 1000) `init=dict(I=1)` gives 1 person, `I=1.0` gives 1000, `I=0.5` gives 500, `I=1.5` gives 1.5 people. The rule is principled as written, but it guesses meaning from magnitude (which `02-types.md` forbids for units: "the unit is never guessed from the size of the numbers"), it makes a count below 1 inexpressible (listed in `deferred.md`), and the `1`/`1.0` trap is exactly the kind of difference an AI or a reader will not notice.
- **Proposal:** decide by type, not size: an `int` is a count and a `float` is a fraction (and a float above 1 raises "use an int for a count, or a fraction between 0 and 1"). `init=dict(Sv=50_000)` and `init=dict(I=0.01)` read the same as today; the only changes are that `I=1.0` means everyone without a special case and `I=2.0` is an error rather than two people. NumPy integers count as ints.
- **Cost:** small; breaking only for float counts above 1 (e.g. `init=dict(Sv=5e4)`, which would need `int(5e4)` or `50_000`), and that case would get a clear error.

### 12. Two silent defaults that should at least be visible: `ss.transmission` without `beta`, and a dwell time shorter than a step

- **Type:** design
- **Evidence:** `ss.transmission('S -> I')` with no `beta=` gets `beta = 1` (`sc.ifelse(beta, 1.0)`, `transitions.py:822`), and `describe()` prints `beta = 1 per step` with no `[guessed]` mark (`t7.py`). A forgotten `beta` is an explosive epidemic, not an obvious one. Separately, `I >> R @ 10` (the spec's own example of a likely mistake, meant as "10 days") is a hazard of 10 per day, a mean dwell of 2.4 hours; in an ODE run nothing is printed (W102 is only for solvers that take whole steps), and the epidemic simply never happens (`n_I`: 100, 0.006, 4e-7, ...; `t7.py`).
- **Proposal:** (1) Make `beta=` required on `ss.transmission`, the same way `ss.progression` requires `dur=` or `rate=` (E209 already exists for this). (2) Under every solver, note any spontaneous transition whose mean dwell time is shorter than one step (`note: recovery has a mean dwell of 0.1 days, shorter than the 1-day step; a duration is written ss.days(10)`). This is not guessing from size: it is a property of the model and the step, and it catches the `@ 10` case without changing what `@` means.
- **Cost:** a few lines each; (1) is breaking only for code relying on `beta=1`.

### 13. `describe()` is the safety net for every guess, and it misreports several of them

- **Type:** doc-code mismatch / bug
- **Evidence:** principle 3 rests on "a guess the user can see is not hidden behavior", so `describe()` has to be right. Found (`t7.py`, `t11.py`, `t26.py`, `t29.py`, `t33.py`):
    - In the operator form, where the source is written in the rate (`S >> I @ 0.3 * I`), it prints `I infectious [guessed: source of 'S_to_I']`; nothing was guessed (`transitions.py:1520–1523` marks any source of a transition as a guess).
    - The remainder branch is a guess that is not shown: with `severe` at `p=0.1` and `mild` with no `p`, `mild` has `p = 0.9` but no `p` line is printed.
    - The default `beta = 1` of `ss.transmission` is printed without `[guessed]` (finding 12).
    - The user's own module describes itself in the wrong unit: `ss.Sim` copies the module, so after `sim.init()` the user's `sir.describe()` still prints `beta = 0.3 per step` and an inflow of `ss.perday(5_000)` as `rate = 5000 ... in total per step`, while `sim.diseases.sir.describe()` says `per day`. A module described before it is in a sim always says "per step", even when its own typed values fix the unit.
    - `describe(verbose=True)`, which `02-types.md` and `01-objects.md` promise for the per-step conversions, does not exist (`TypeError: Sim.describe() got an unexpected keyword argument 'verbose'`).
    - For library models, `describe()` prints only a list of internal arrays (`susceptible, infected, ..., ti_dead, rel_sus, rel_trans`), and `sim.describe()` of a sim with `ss.SIR` and `ss.RandomNet` prints one line, the schedule (finding 5).
- **Proposal:** fix the five bugs above (each is a line or two), and add a conformance test that, for each guess in the table of `03-states-and-transitions.md` §What is guessed, builds a model that triggers it and checks that `describe()` marks it, and one that does not and checks that it is not marked. That turns "every guess is printed" from a claim into a test. If the unit becomes a model property (finding 1), a module can describe itself correctly outside a sim.
- **Cost:** small.

### 14. One printed number would catch most of the silent failures above: R0 at `init()`

- **Type:** design
- **Evidence:** the failures in findings 1, 3, 12, and the documented change of meaning of `beta` when a network is added (`t35.py`: the same `S >> I @ 0.03 * I`, `I >> R @ ss.days(10)` has R0 0.3 with no network and 3.0 with `networks='random'` or `ss.MixingPool(n_contacts=10)`) all produce a model that runs without complaint and does something other than what was meant. Each one is obvious from a single number. `sim.R0()` already exists and works for declared models (next-generation matrix).
- **Proposal:** print it in the line that already announces the solver, when the model is declared and the computation is cheap (no custom functions in rates, small enough state space): `solver: ode (every part has a compartmental form); R0 = 0.3 at the start; nobody is infected`. Two facts (R0 and whether anyone is infected) make the two-line SIR's flat line, a `beta` read in the wrong unit, a forgotten `beta`, and a network that multiplied the contact rate visible before anyone looks at a plot.
- **Cost:** small (call the existing `next_generation()` on the engine already built; skip silently when it cannot be computed). Adds a little to `init()` time for large stratified models, so gate it on size.

### 15. Three notations for "several states": `S | I`, `S + I`, and the condition `S | I`

- **Type:** API / vocabulary
- **Evidence:** `S | I` builds a `StateSet` (`transitions.py:518`, 50 lines of forwarding dunders) used for many-to-one sources; `S + I` builds a sum (`BinOp`) that is a source of infection, a denominator, a result, and, via `BinOp.__rshift__`, also a set of sources; and `S | I` / `S & I` / `~S` are also conditions (`ss.Cond`). In practice (`t15.py`, `t16.py`) `S + I + R >> D @ 0.1` already builds the same many-to-one transition as `(S | I | R) >> D @ 0.1`, and it needs no parentheses, because `+` binds more tightly than `>>` while `|` binds more loosely (which is why `03-states-and-transitions.md` needs a precedence paragraph and a dedicated error for the missing parentheses).
- **Proposal:** document `S + I + R >> D @ mu` as the many-to-one spelling and drop `|` as a source-set operator; keep `|`, `&`, `~` only for conditions, where they mean what they mean in NumPy and pandas. `StateSet` collapses into the sum (`BinOp`) already used everywhere else. The spec's precedence caveat and the special error go away.

```python
# before
(S | I | R) >> D @ mu
# after
S + I + R >> D @ mu
```

- **Cost:** small; `|` could stay as a deprecated alias for a release. One fewer concept ("a set of states" vs "a sum of states") in the core vocabulary.

### 16. `ss.freq` means nothing in a declared model, and the time vocabulary could shrink

- **Type:** design / vocabulary
- **Evidence:** in every slot of a declared model, `ss.freqperday(0.5)`, `ss.perday(0.5)`, and `1/ss.days(2)` (which returns an `ss.freq`) give identical results under both solvers (`t21.py`); the engine treats any rate in a rate slot as a hazard. The distinction between `per` and `freq` survives only in `to_prob()` for Starsim 3 code. Meanwhile the arithmetic of the time types has corners a user cannot predict (`t19.py`, `t20.py`): `1/ss.days(10)` is a `freq` while `02-types.md` says "number ÷ dur → a rate"; `1/ss.perday(0.1)` raises a bare `NotImplementedError()` with no message (the spec says it gives a `dur`); `ss.probperday(0.1) + ss.probperday(0.1)` is 0.2 but `2 * ss.probperday(0.1)` is 0.19, so `p + p != 2 * p`.
- **Proposal:** in the declared language, have two rate types, not three: `ss.per` (a hazard) and `ss.prob` (a chance per period). Keep `ss.freq` as a Starsim 3 class for code-form modules but drop its four constructors from the v4 vocabulary (22 → 18 names) and from the docs for declared models. Make `number / dur` return `ss.per`, implement `number / per → dur`, and make `prob` arithmetic consistently hazard-based (or refuse `prob + prob`), so that the table in `02-types.md` is the complete truth.
- **Cost:** small in code; `1/ss.days(10)` changing type is technically breaking for Starsim 3 code that multiplies it by a duration (`freq * dur` and `per * dur` both give a number, so most such code is unaffected).

### 17. Arithmetic on a finished transition means two different things

- **Type:** design / code simplicity
- **Evidence:** `0.5 * infection` is, at the same time, a new transition at half the rate *and* half of `infection`'s flow (`Transition._arith`, `transitions.py:739–761`, sets both `rate` and `flow_of`). Passed to `ss.Disease` it is a transition with R0 halved (`t13.py`: R0 1.5); passed to `ss.outcome` it is half the reported flow; `infection / S` is a third thing, a per-capita flow ratio (a `BinOp` of `Flow`s); `infection + recovery` is a `FlowSum`; `infection - x` scales the rate; `infection + 0.1` raises. The spec documents the rate meaning ("`infection * 0.5` is the same transition at half the rate") and the outcome meaning (`ss.outcome(0.5 * infection, ...)`), and they collide in one operator.
- **Proposal:** arithmetic on a *finished* transition means its flow, always (for outcomes and results: `0.5 * infection`, `infection / S`, `mild + severe`). Arithmetic on an *unfinished* one (`I @ beta`, no source yet) keeps forwarding to the rate, which is the only place the precedence trick needs it. Halving a transition's rate is written where the rate is (`S >> I @ 0.5 * beta * I`) or as a modifier. This removes the dual `flow_of`/`rate` bookkeeping from `_arith` and lets `Flow` absorb `FlowSum`.
- **Cost:** small; breaking only for code that used `0.5 * infection` as a transition, which is unidiomatic.

### 18. `ss.par` "behaves like its value in arithmetic", except where a user will first try it

- **Type:** API / doc-code mismatch
- **Evidence:** `04-parameters.md`: "An `ss.par` behaves like its value in arithmetic, so it can be used anywhere a number can." In practice (`t28.py`), with `beta = ss.par(0.3, bounds=[0.1, 1])`: `float(beta)` raises `TypeError`; `f'{beta:.2f}'` raises; `if beta > 0.2:` raises "it has no value until the model runs" although every leaf has a value now; `beta * 2` is the expression `0.3 * 2` rather than 0.6; `beta * dur` (R0 by hand) is `0.3 * days(10)`.
- **Proposal:** give `par` `__float__` and `__format__`, returning the current value, and make `Expr.__bool__` evaluate when the expression contains no states, counts, or flows (so `if beta > 0.2` works and `if I.n > 20` still raises with the current message). Keep arithmetic symbolic, since that is what lets a fitted `beta` flow into rates, and say so in the doc rather than "behaves like its value".
- **Cost:** ~15 lines.

### 19. `transitions.py` is three modules in one, and the Graph does both building and presenting

- **Type:** code simplicity
- **Evidence:** of `transitions.py`'s 2,972 lines, about 1,000 are the language (expressions, `par`, `State`, `Transition`, the constructors, `outcome`); about 830 are `Graph`, of which ~330 are presentation (`describe`, `layout`, `diagram`, `equations`, `flow_terms`, `to_json`), and `flow_terms` builds a mock sim and a `ss.Compartmental` engine just to print equations (`:1705–1709`); and the last ~930 lines are not about transitions at all: the parameter forms (`by`, `timeseries`, `seasonal`), geography and movement (`geo`, `gravity`, `Mobility`), and stratification (`Strat`, `Transfer`, `transfer`, `stratify`). The error-code table for the whole library also lives here. The spec says nothing is computed when a model is written (`00-overview.md` §What happens at build time); in the code the Graph names, infers, resolves sources and parameters, and checks at `ss.Disease(...)` (which is why the user's objects are mutated, finding 6).
- **Proposal:** split by concept: `expressions.py` (Expr, par, Count, BinOp, Func, linear, evaluate), `transitions.py` (State, Transition, constructors, outcome), `graph.py` (Graph: build and check), `describe.py` or methods on the engine for describe/equations/diagram/spec, `parforms.py` (by, timeseries, seasonal), and move `geo`/`gravity`/`Mobility`/`Strat`/`Transfer` to the dimensions and networks code, and `codes` to an `errors.py` with `CodedError`. Update the spec to say that names and structure are resolved at construction (it is the better choice: errors arrive at the line that caused them) and values and units at `init()`.
- **Cost:** mechanical, no behavior change; worth doing before more features land in this file.

### 20. Two rules for bare numbers in one sim

- **Type:** design
- **Evidence:** in a declared model a bare number takes the sim's unit; in a module written as code it takes the unit of the parameter's default, as in Starsim 3 (`Module.check_bare_units`, `modules.py:715`; known issue in `implementation.md`). So in `ss.Sim([sir, ss.MFNet(duration=5), ss.Births(birth_rate=20)], unit='day')`, `sir`'s bare `beta` is per day while `duration=5` is five years (with a warning). The principle "the slot gives the dimension and the sim gives the unit" (`02-types.md`) is true for half the modules.
- **Proposal:** the honest single rule is "a bare number takes the unit of whatever it replaces; a value with nothing to replace takes the model's unit". That is what both halves already do, so the fix is mostly to state it as one rule in `02-types.md` and in the warning text. It also becomes moot for the library diseases if they are rewritten as declared models (finding 5).
- **Cost:** documentation, plus the warning wording.

### 21. Smaller items

- **Type:** bug / API / doc, each verified in the scratch scripts
- **Reserved names:** `Graph.add` checks names against `dir(ss.Module)` (`transitions.py:1208`), not the module's own class, so a transition, state, or result named `describe`, `diagram`, `equations`, `R0`, `stratify`, or `to_json` silently replaces the `ss.Disease` method: `ss.Disease(..., describe=I >> R @ 0.1).describe` is the transition (`t34.py`). `01-objects.md` says `describe` and `stratify` raise `E201`. Check against `dir(type(self.module))`.
- **`ss.states(1)` returns one state, not a list**, so `ss.states(' '.join(names))` breaks when `names` has one element (`TypeError: 'State' object is not iterable`, `t30.py`). `ss.State()` already exists for one; `ss.states` should always return a list (`X, = ss.states(1)` reads fine).
- **Error-message quality**, given principle "every error names the rule, shows the offending expression, and states the fix": E602 from `ss.Sim(pars=dict(bta=0.3))` suggests "did you mean d4.beta, d4.dur?" (every parameter, because the candidates are prefixed with the module name before matching); E206 can read "used by transitions in two modules, 'disease' and 'disease'" with no fix offered; `ss.perday(0.1) + ss.probperday(0.1)` says "you added 0.1 + 0.1" instead of naming `per` and `prob`; W102 reads "so 1 of people leave in a step".
- **Reprs before naming:** `0.5 * (S >> I @ 0.3 * I)` prints `Transition(<unnamed state> -> <unnamed state> @ 0.5 * 0.3 * None)`: the endpoint is `<unnamed state>` but the same state inside the rate is `None` (`format_expr` uses `e.name`). Moot if states are named at creation (finding 9).
- **`ss.Disease(dict(...), ...)`** treats a leading dict as `pars` (`diseases.py:70`), an undocumented positional convention that a reader cannot see; `pars=` is already a keyword.
- **Vocabulary count:** `ss.immigration` and `ss.emigration` are in the code and in `03-states-and-transitions.md` but not in `17-vocabulary.md`'s 94 names. They differ from `ss.null >> S @ r` and `ss.death()` only in what agents do (arrival age, being recorded as emigrants); a keyword on `ss.birth`/`ss.death` (`ss.death(rate=r, emigrate=True)`) would say the same with two fewer names.
