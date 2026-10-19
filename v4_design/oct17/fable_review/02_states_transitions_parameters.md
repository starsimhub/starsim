# States, transitions, and parameters: the declarative core

I reviewed `v4_design/design/00-overview.md`, `03-states-and-transitions.md`, and `04-parameters.md` (with the relevant parts of `01-objects.md`, `02-types.md`, and `19-open-questions.md`), and read `starsim/transitions.py` (2972 lines, in full) and `starsim/diseases.py` (1353 lines, in full). I wrote and ran nine scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/transitions/` (`s1_sir_spellings.py` through `s8_outcomes_bugs.py`): SIR in all four spellings, SEIR with an Erlang dwell time in each spelling, five branching models, the cholera reservoir and variants of it, the init semantics, every kind of object I could think to put after `@`, the name lookup under a dozen conditions, parameter naming and `set_pars`, and `ss.Disease` next to `ss.Infection`. Quoted output below is from those runs (ODE runs of 10,000 people or agent runs of at most 2,000 agents).

## Summary

1. **Resolve arrow-string names against the module's declared states, never against the caller's variables.** Today `ss.Transition('I', 'R')` finds whatever *variable* is called `I` in the caller's frame, so it silently grabs another module's state, or fails to find a state that is named `I` but bound to `I3`. The spec's own "fully explicit" example only works when variable names equal state names. Not breaking for correct code; fixes a trap. (Finding 1.)
2. **Make the people-versus-count guess anchor on transmission, not on creation order.** `W, S, I, R = ss.states('W S I R')` turns `S`, `I`, `R` into counts and `W` into the people, silently except for a `describe()` line. Anchor on the states of contact-mediated transitions (then the largest connected component), and warn when the guess has to fall back to "first state". Not breaking for every example in the spec. (Finding 2.)
3. **Tighten `@`: a transition that already has a driver should refuse a second one, and arithmetic on a duration-driven transition should apply to the duration or raise.** `I >> R @ ss.days(10) * 2` silently drops the `* 2`; `I >> R @ ss.days(10) @ 0.1` sets both `dur` and `rate`; `I >> R @ ss.by(age=[ss.days(5), ss.days(20)])` is read as a *rate* of 5 and 20 per day (and runs, giving a different epidemic). These are bugs, but they come from the rule "the type of what follows `@` decides the slot", which cannot see inside `ss.by`, functions, or `ss.par`. Keep the rule, but make `Transition.__matmul__` and `_arith` strict. (Findings 3, 4, and Bugs.)
4. **Drop the string-form source guess, or make it an error for anything but SIR/SIS shapes.** Every SEIR-shaped vignette writes `source='I'`; the one that forgets gets a wrong model that `describe()` alone reveals (peak 1657 instead of 2012 in my run). A transmission whose destination is not the first state reached from `S` by infection should require `source=`. Mildly breaking (the two-line SIR still works). (Finding 5.)
5. **Move the ~930 lines of stratification, geography, and mobility (`by`, `Strat`, `geo`, `gravity`, `Mobility`, `Transfer`, `transfer`, `stratify`) and the 53 lines of error codes out of `transitions.py`.** The file is called `transitions.py` but a third of it is 05-dimensions and 06-routes material. Pure code move, not breaking. (Codebase finding 1.)
6. **Decide what `ss.Disease` is: today it is both the declared-model class and the base class for `ss.Infection`/`ss.SIR`/`ss.NCD`, with 15 `graph is None` branches, and a declared `sir.S` that is an `ss.State` before `init()` and a `BoolState` after.** The cleanest cut is a `ss.Model`-style class for declared diseases and `ss.Disease` left as Starsim 3's base; the cheapest is to keep one class but stop rebinding `sir.S`. Breaking either way for `sir.S` users. (Finding 8, Codebase finding 2.)
7. **Keep the four spellings, but say which is canonical.** They do build identical models (verified: identical `to_json()` transitions and bit-identical results across strings, operators, explicit, and class). The explicit form earns its place in vignettes (19 uses, mainly for `p=` lists and `where=`). The operator form's `b @ x` (a transition with no source yet) and arithmetic on a finished transition are the parts that could go. (Finding 6, Keep.)

## Simpler for the user

### 1. A string in `ss.Transition('I', 'R')` names a variable, not a state

**Issue.** The spec (03, "Fully explicit") shows `ss.Transition(S, I, rate=beta * I)` next to `ss.Transition('I', 'R', dur=ss.days(10))` in one module. The code resolves `'I'` by looking for a *variable* called `I` in the caller's frame (`transitions.py:1246-1250`), not for a declared state *named* `I`. Two consequences, both verified in `s1b_string_lookup.py`:

```python
S, I, R = ss.states(3)
sirA = ss.Disease(S >> I @ 0.3*I, I >> R @ 10, name='sirA')      # claims the variables S, I, R
S3, I3, R3 = ss.states('S I R')                                   # a second set, with the same names
sirB = ss.Disease(ss.Transition(S3, I3, rate=0.3*I3, name='infection'),
                  ss.Transition('I', 'R', dur=10, name='recovery'), name='sirB')
# E206 DuplicateDeclaration: state "I" is used by transitions in two modules, "sirA" and "sirB"
```

and, with no competing module, `ss.Transition('I', 'R')` after `S3, I3, R3 = ss.states('S I R')` raises `E204 UnknownState: no state named "R" was declared in "sirC". Declared states are S, I` because `R3` is named `R` but no variable is called `R`. The test that covers the explicit form (`tests/test_transitions.py:62-67`) only passes because its variables happen to be called `S, I, R`.

**Why it matters.** This is the frame lookup doing something the spec never asks of it: 01-objects says names are looked up *for objects that have none*; here a string that already is a name gets second-guessed by the frame. It makes "mixing states and their names" work by coincidence.

**Proposal.** Resolve a string only against the states the module has (declared objects by `.name`, or inferred). Delete `transitions.py:1246-1250`. If no declared state has that name, `E204` as now. The spec text needs no change; it already says this.

**Cost.** None for correct code. Confidence: high.

### 2. "The first state" is the first one *created*, and the people guess hangs on it

**Issue.** The spec (03, "States that are not people") says a state is people if transitions connect it to the disease's first state, and "the first state" is "the first one returned by the first `ss.states(...)` call" (ordered by `State._counter`, `transitions.py:451,461,1251`). Declaring the reservoir first inverts the model, with nothing but a `describe()` line to say so (`s5_reservoir.py`):

```text
W, S, I, R = ss.states('W S I R')
states       W
             S a count, not people                           [guessed: no transition connects S to W]
             I a count, not people                           [guessed: no transition connects I to W]
             R a count, not people                           [guessed: no transition connects R to W]
             init: I = 0.001; the rest in W
```

The sim then runs with `total_pop` in `W`. The same rule also governs where the leftover `init` goes and (via `uninfected`) which states a strain applies to, so one ordering accident changes three things.

**Why it matters.** The guess is right for every example in the spec because the examples all happen to write `S` first. It is the kind of rule a user cannot be expected to know ("creation order of `ss.State` objects"), and in the class form or with `ss.State(...)` written one per line it is invisible.

**Proposal.** Anchor the people component on the states that a contact-mediated transition moves people *between* (its `src` and `dst`), since those are people by definition (the code already requires the source of exposure to be people, `E211`). Fall back to the largest connected component, then to the first state, and make `W212 StateNotPeople` (today a note at `sim.init()`, `sim.py:399`, which says `state 'S' treated as a count, not people` and which is easy to scroll past) a warning whenever the fallback was needed or more than one component has more than one state. Spec 03 would say "the component containing the transmission" instead of "the first state"; the cholera and malaria examples are unchanged (cholera has no contact-mediated transition, so it falls to "largest component", which is `S, I, R`).

**Cost.** A model with two equally large unconnected people components (rare) gets a warning it did not get before. Confidence: high.

### 3. The `@` rule is guessable for the common cases, but the edges are silent

**Issue.** The rule "a duration or a distribution after `@` is a dwell time, anything else a rate" (`transitions.py:730-737`) is clean and I could predict it for `ss.days`, `ss.perday`, `ss.probperday`, `ss.par(ss.days(10))`, functions, `ss.timeseries` of durations (the code peeks inside, `TimeValue.typed`). It breaks at three edges (`s2_after_at.py`, `s8_outcomes_bugs.py`):

| Written | Got | Should be |
|---|---|---|
| `I >> R @ ss.days(10) * 2` | `dur = days(10)`; the `* 2` is dropped (`_arith`, `transitions.py:750-756`: `rate is None`, `beta is None`, `op == '*'` passes through) | `dur = days(20)`, or an error |
| `I >> R @ ss.days(10) @ 0.1` | `dur = days(10)` **and** `rate = 0.1`, no error | an error: a transition takes one driver |
| `I >> R @ ss.by(age=[ss.days(5), ss.days(20)])` | `rate = by(...)`; it runs, and gives `n_R[-1] = 102` where `ss.perday`-equivalent gives `4636` | `dur`; `by.items` are durations |
| `I >> R @ ss.bernoulli(0.5)` | `dur = bernoulli(0.5)` (any `ss.Dist` is a dwell time) | an error |
| `I >> R @ 10` | a hazard of 10 per step; the epidemic never starts (peak `n_I = 100` of 10,000 over 30 days) | by the spec, correct; see below |

The `@ 10` case is spec-conformant and the spec's argument (0.1 and 10 cannot both be guessed from size) is right. But `I >> R @ 10` is the single most natural thing to type after reading `recovery = ss.progression('I -> R', dur=10)` two lines earlier, and it runs without a word. `describe()` prints `rate = 10` with no unit (bare rates get `per step` only for `beta`, `transitions.py:1540` vs `1567`).

**Proposal.** (a) `Transition.__matmul__` raises if `rate` or `dur` is already set. (b) `_arith` on a dur-driven transition either scales the duration (`*`, `/`) or raises; never passes through. (c) Extend the "look inside" that `TimeValue.typed` already gets to `ss.by` (`by.items`). (d) Only duration-valued distributions count as dwell times; `ss.bernoulli`, `ss.choice`, `ss.poisson` after `@` raise. (e) For a bare number above, say, 1 after `@` on a spontaneous transition, print a one-line note (`rate = 10 per step; did you mean ss.days(10) or dur=10?`), since the spec already prints every guessed default. None of these change the language.

**Cost.** (a)–(d) are bug fixes. (e) is a heuristic note, which the spec otherwise avoids; it is a judgment call. Confidence: high for (a)–(d), medium for (e).

### 4. `0.5 * infection` means two things

**Issue.** The spec says "`infection * 0.5` is the same transition at half the rate" (03, "The operators") and also "`0.5 * infection` is half of the flow" (as the source of an `ss.outcome`). The code does both at once: `_arith` halves `rate` *and* sets `flow_of = (infection, 0.5)` and clears the name (`transitions.py:748-761`). Which one it means is decided by the slot it lands in. Verified: `ss.Disease(infection=inf*0.5, ...)` builds `rate = 0.3 × I.n / N × 0.5`; `ss.outcome(0.5 * inf)` reports half the flow.

**Why it matters.** It works, but a reader cannot tell from the expression what it means, and the implementation carries a shadow attribute (`flow_of`) on every transition to support the outcome case. `(I @ 0.3) * 2` (no source yet) means "double the rate" and `(S >> I @ 0.3*I) * 2` means "twice the flow, as an outcome source"; the difference is whether `>>` has been applied.

**Proposal.** Keep arithmetic-on-rate only for the *unfinished* transition (`I @ beta`, which is what makes `S >> I @ beta * I` parse). For a finished transition, `k * infection` is a flow share and nothing else, so the outcome use is the only use and `ss.Disease(infection=inf*0.5)` raises ("a share of a flow is an outcome; to scale the rate write `S >> I @ 0.5*beta*I`"). Spec 03 line 190 goes.

**Cost.** Removes a line of the language nobody has used (0 uses in tests or vignettes of a scaled finished transition as a transition). Confidence: medium.

### 5. The string-form source guess is wrong for SEIR and cannot be checked

**Issue.** `ss.transmission('S -> E', beta=0.3)` with no `source=` makes `E` the source of infection ("source is the destination", `transitions.py:1322-1324`). `s3_seir.py`: peak `n_I` 1657 (guessed) versus 2012 (`source='I'`). The spec admits "this guess cannot be checked". All six SEIR-shaped vignettes write `source='I'` (grep over `/home/cliffk/idm/lingua_franca_design/vignettes`), which says experienced users already treat it as required.

**Proposal.** Make `source=` required whenever the destination of a `ss.transmission` has a spontaneous transition *out* of it into another state that nobody has declared infectious (that is, the destination looks like a latent state). SIR, SIS, SIRS keep working with no `source=`; SEIR without `source=` raises with the one-line fix. Spec 03 "Source inference" changes from "printed by `describe()`" to "an error when the destination has an onward progression".

**Cost.** Mildly breaking for SEIR models that relied on the wrong guess, which is the point. Confidence: high.

### 6. Four spellings: which earn their place

Verified (`s1_sir_spellings.py`): strings, operators, explicit, and class build identical `to_json()` transitions and bit-identical results.

- **Arrow strings** earn their place: the two-line SIR, and `p=` lists over `'I -> H | R'`. Keep.
- **Operators** earn their place for everything that mentions a state in a rate (65 uses in vignettes versus 13 of `ss.transmission`). Keep `>>` and `@`. Two sub-features could go: `b @ x` on its own as "a transition into `b` with no source yet" is only needed as an implementation trick for precedence and is never written by a user; and `Expr.__matmul__` (a weighted destination `(p*A + (1-p)*B) @ rate`, `transitions.py:138-140`) exists so that precedence works for branches. Neither should be *documented* as a form (03's operator table lists `b @ x` as a thing a user can write).
- **Explicit `ss.Transition`** earns its place: 19 vignette uses, all for `p=[...]` with several destinations, `where=`, `stages=`, or a `rate=` that is a function. Keep.
- **Class form** earns its place for `step()`. Keep.

One thing I would remove: **`ss.states(1)` returns a state, not a list** (`transitions.py:586`). `S, = ss.states(1)` versus `S = ss.states(1)` is the kind of special case that the spec had to write a sentence about.

Confidence: medium (this is taste, backed by usage counts).

### 7. `init` and `init_prev`

Verified (`s5_reservoir.py`): `init=dict(I=1)` is one person, `init=dict(I=1.0)` is everyone, `init=dict(I=50)` is 50 people, `init_prev=0.01` and `init=dict(I=0.01)` give identical runs, both together add (`init_prev=0.01, init=dict(I=0.05)` gives `n_I[0] = 60` of 1000), and `init_prev` on a disease with no transmission raises a clear error.

The `int 1` versus `float 1.0` rule is in the spec and the code honors it, but it is the only place in the library where the Python type of a number changes its meaning, and it will bite anyone who reads counts from a dataframe (floats). **Proposal:** `init` below 1 is a fraction, at or above 1 is a count, and "everyone" is `init=1.0` only when the user says `init=dict(I='all')` or simply gives no one else an `init`. Alternatively keep the rule and have `describe()` print `init: I = 1 (one person)` versus `(everyone)`; today it prints `init: I = 1`. Confidence: medium.

`init_prev` is also the name of a Starsim 3 `Infection` parameter (`ss.SIR().pars.init_prev` is an `ss.bernoulli`; a declared disease's `init_prev` is a float not in `pars`, `s7_disease_infection.py`). The engine accepts both (`solvers.py:1172`), so the overlap is harmless, but the two are not the same object and `sir.pars.init_prev` does not exist for a declared disease unless it was an `ss.par`. Worth one sentence in 04.

### 8. One `ss.Disease`, two kinds of disease

**Issue.** `ss.Disease` is simultaneously the constructor for declared models and the Starsim 3 base class of `ss.Infection`, `ss.SIR`, `ss.SEIR`, `ss.SIS`, `ss.NCD`. `ss.Disease()` with no arguments is a module with no states and no graph. A declared disease inherits `set_prognoses`, `make_naive`, `step_die`, `infection_log`, `define_states`, `state_dict`; `ss.Infection` adds `infect`, `validate_beta`, `set_congenital`, `step_congenital`, and so on, none of which a declared model uses. `diseases.py` has 15 `graph is None` / `graph is not None` branches (`s7_disease_infection.py`).

The sharpest user-facing symptom: `sir.S` is an `ss.State` object before `sim.init()` and a `BoolState` (a Starsim 3 array) after an agent-based `init()` (`s7`: `type(sir.S)` goes from `State` to `BoolState`; `state_dict` goes from `[]` to 14 entries including `ti_S`, `exit_S`, `ti_exit_S`). The `ss.State` the user declared and the `BoolState` the engine runs are two different concepts with the same word.

**Why it matters.** A user who writes `sir.S.init = 0.1` after init, or `sir.I.n` in custom code, is talking to a different object depending on when and under which solver. The spec says "a state is one concept"; the implementation has two.

**Proposal.** Keep `ss.Disease` as the one entry point (the spec's overview depends on it), but: (a) never rebind `sir.S`; give the agent engine its arrays under `sir.engine.arrays.S` or make `ss.State` itself hold the `BoolState` once built, so `sir.S` is always the declared state and `sir.S.uids`/`sir.S.n` work under every solver; (b) move `ss.Infection`'s machinery down one level so `ss.Disease` has only the lifecycle and the graph (the `Infection`-only attribute list above is already the dividing line); (c) rename Starsim 3's `ss.BoolState` to `ss.BoolArr` or similar so "state" means one thing in the docs. (c) is breaking for Starsim 3 code and is a naming question for the dimensions/arrays reviewer.

**Cost.** (a) touches the agent engine; (c) is a Starsim 3 rename. Confidence: medium for (a), low for (c).

### 9. Parameters: three places a value can live, and `sir.infection.beta` is not one of them

Verified (`s6_names_pars.py`):

- Anonymous slot values are named `beta`, `dur`, `rate`, qualified by transition when two clash (`dur_recovery`, `dur_relapse`). Good, and `describe()` shows them.
- An `ss.par` written inline in the slot (`S >> I @ ss.par(0.3, bounds=[0.1, 1]) * I`) is named `beta` and keeps its bounds. Good.
- **Spec 04 says `sir.infection.beta` and `sir.recovery.dur` are addressable.** `sir.recovery.dur` works; `sir.infection.beta` is `None` after the build because `resolve_transitions` folds `beta` into the rate and clears it (`transitions.py:1328-1329`). Either the spec drops `sir.infection.beta` or `Transition` gets a `beta` property that returns the coefficient of the source (the graph already finds it in `resolve_pars`, `transitions.py:1407-1415`).
- `ss.Disease(beta=0.3, infection=ss.transmission('S -> I', beta=0.3), ...)` yields *two* parameters, `beta` and `beta_infection`, with no link between them. Spec 04 says a keyword parameter is "collected into the module's pars", which it is, but nothing uses it. A keyword number that no slot refers to should probably be an error (`E602`-style: "`beta` is not used by any transition"), or the keyword should be looked up by the slot of the same name. 
- `sir.pars.gamma = 0.3` replaces the `ss.par` with a float; `sir.set_pars(gamma=0.3)` keeps the `ss.par` and sets its value. Both work (`par.raw` reads through `mod.pars`, `transitions.py:217-224`), but `pars` then holds a mix of `par` objects and floats. Prefer one: make `pars.__setitem__` route through `set_pars` for declared modules.
- A dimensionless factor in a transmission rate (`R0` in `R0 * gamma * I`) is exported in the model spec as `type: per` (`transitions.py:1798`: every non-literal par in a contact rate gets `'rate'`), while `describe()` correctly prints `R0 = 3` with no unit. Minor spec-export bug.

Confidence: high for the facts; medium for the keyword-parameter proposal.

### 10. A declared disease gets each flow twice

`s6`: results of a declared SIR are `n_S, n_I, n_R, infection, cum_infection, recovery, cum_recovery, new_infections, cum_infections, prevalence`. Spec 09 says `new_infections` is kept "until the vignettes have been written both ways". The vignettes are written; the question in 19-open-questions can be closed. My vote: keep the transition's name (it is what the user typed, and it is what `ss.outcome(0.5 * infection)` refers to), keep `prevalence`, and drop `new_infections`/`cum_infections` from declared diseases (Starsim 3 classes keep theirs). Breaking for data files that attach to `new_infections`; the fix is a one-line alias in `data=` matching. Confidence: medium. (Mostly the results reviewer's call; recorded here because the duplication is visible from the transitions side.)

### 11. Functions in slots, and `ss.func`

Spec 03 says a plain function passed as a keyword is "wrapped automatically" and `ss.func` exists "only to attach options". The code stores it in `graph.funcs` and calls it after the engine's `step()` (`diseases.py:349-350`); fine. **`ss.func` does not exist** (`hasattr(ss, 'func')` is `False` in the v4 environment; `grep` finds no definition). Either implement it or delete the sentence from 03; since `ss.Disease(seeding=seed_cluster)` already works and `ss.par(fn)` covers the parameter case, deleting is simpler.

## Simpler codebase

### 1. A third of `transitions.py` is not about transitions

Sections by `#%%` marker: Errors and notes 53 lines (29-81), Expressions 344 (82-425), States 182 (426-607), Transitions 402 (608-1009), Dwell times 63 (1010-1072), The graph 852 (1073-1924), Looking at the model 118 (1925-2042), **Stratification 930 (2043-2972)**. The last section is `by`, `TimeValue`/`timeseries`/`seasonal`, `geo`, `gravity`, `Mobility` (including a movement-register loader with its own CSV validation, `transitions.py:2619-2680`), `Strat`, `Transfer`, `transfer`, `stratify`. These implement spec sections 04 (ladder), 05 (dimensions), and 06 (routes), and are the subject of other reviews. Move them to `dimensions.py` (or `strata.py`) and `timevalues.py`; move the `codes` dict and `CodedError`/`error`/`note` (spec 15) to `errors.py`. `transitions.py` becomes about 2,000 lines, and the `__all__` at `transitions.py:24-26` stops exporting `geo`, `gravity`, `Mobility`, `Strat`. Pure move. Confidence: high.

### 2. `Transition` carries eight mode flags for the shorthand constructors

`Transition.__init__` sets `beta`, `source`, `where`, `move`, `per_capita`, `every_state`, `migration`, `flow_of`, `guessed` (`transitions.py:664-681`). `beta`/`source` exist only so that `ss.transmission` can defer "times the source" to build time (resolved and then cleared at `1316-1329`); `per_capita` only for `ss.birth` (`1332-1333`); `every_state` only for `ss.death()` with no arrow (`1376-1377`); `migration` for immigration/emigration; `flow_of` for outcomes (Finding 4). `grep` shows `per_capita` and `every_state` are read nowhere outside `transitions.py`. Each is a small thing, but together they mean a `Transition` is not "a movement between states with a driver"; it is that plus five pending instructions for the graph.

**Proposal.** Resolve as much as possible at construction. `ss.birth('-> S', rate=r)` can build `rate = r * Count([], kind='N')` immediately (the `N` placeholder is already how it works after `resolve_transitions`). `ss.death()` with no arrow can build `src = []` with a marker on `src` rather than a flag. `ss.transmission` genuinely has to wait for the source, so `beta`/`source` stay, but they could live in a small `Pending` object rather than as attributes on every transition. Confidence: medium; this is tidiness, and the gain is that `Transition.__repr__`/`driver_str`/`to_json` stop special-casing.

### 3. The graph does name lookup in four places

`find_name(obj, frame)` is called for transitions (`transitions.py:1170`), states (`1256`), parameters (`1421`, `1441`), results (`1186`), transfers (`1172`, `1849`), and from `Disease.init_graph` with `sys._getframe(1)` (`diseases.py:74`, `143`). Each caller has its own fallback policy. The spec's rule (keyword, then `name=`, then variable, then fallback) could be one function `resolve_name(obj, keyword, frame, fallback)` called everywhere. Low value on its own; worth doing if Finding 1 is done, since that removes one of the lookups.

### 4. `Graph.describe()` and `Graph.to_json()` each re-derive the kind of every value

`describe()` (74 lines, `1501-1574`) and `to_json()` (53 lines, `1764-1816`) each walk the transitions working out whether a `par` is a rate, a duration, or a plain factor (`leaf.slot`, `tr.contact`, `is_one`, `isinstance(raw, ss.Dist)`), and `flow_terms()` does it a third time for `equations()`. `resolve_pars` already assigns `par.slot` (`1398-1418`); if it also assigned `'factor'` for a dimensionless coefficient and `'count'` for an inflow total, the three renderers would read one field. That is also the fix for the `R0` export bug above.

### 5. Two expression systems meet at `Expr`

`Expr` (`transitions.py:96-172`) gives states and parameters arithmetic, comparisons, `__array_ufunc__`, and a `.n`/`.frac`. `StateSet` (`518-566`) duplicates the eight arithmetic dunders to forward to `self.sum()`, and `by` (`2045-2215`) has its own nine arithmetic dunders and a `map`. `ss.TimePar` checks `_is_expr` to defer (`transitions.py:101`). Three arithmetic tables for three kinds of value is what it takes to make `beta * I`, `(Ia | Is).n`, and `1/by` all work; I do not see a cheaper structure, but `StateSet` could simply *be* an `Expr` (a `BinOp` sum with an `is_set` marker), which removes 30 lines and the `as_expr` special case at `88-89`.

### 6. Dead or near-dead

- `W212 StateNotPeople` is emitted only as a `note` from `sim.py:399` by re-parsing the keys of `graph.guessed` (`key.endswith('.people')`, `key.split('.')[1]`); the graph should record the guess in a structured way rather than as a dotted string that the sim parses back.
- `Transition.__set_name__`, `State.__set_name__`, `par.__set_name__`, `outcome.__set_name__` (four copies of the same three lines) support the class form; a mixin or a shared base would do.
- `Graph.reserved = ['transitions', 'results', 'outcomes']` (`1118`) and the hard-coded list in `Graph.add` (`1208`) overlap with `dir(ss.Module)`; one list.

## Spec and implementation

- **Spec says `sir.infection.beta` is addressable (04 line 18); code has `beta = None` after the build** (`transitions.py:1329`). One of them should change (Finding 9).
- **Spec says "`infection * 0.5` is the same transition at half the rate" (03 line 190) and "a number times a flow is that fraction of it" (09, 19-A).** Code does both (Finding 4). The spec is internally inconsistent here; pick one.
- **Spec says a string in `ss.Transition` "names a declared state"; code resolves it through the caller's variables** (Finding 1).
- **Spec says a transition bound to two variables "takes the fallback name too, with a note" (01, Names).** Code gives the fallback (`S_to_I`) with no note (`s6`). Minor.
- **Spec: "a state that is only filled by an inflow and emptied by an outflow is not people" (03).** Code agrees, and so `ss.immigration('-> Q', rate=...)` into a holding state `Q` with only an outflow makes `Q` a count (`s5`: `Q a count, not people [guessed: no transition connects Q to S]`), although `ss.immigration` is defined as *people arriving*. The spec should say that an inflow made with `ss.birth`/`ss.immigration` is a connection for the purpose of the guess, or Finding 2's rule should treat it so.
- **Spec 03 "What may follow `@`": "a function: called each step to get the rate".** Code agrees, and also accepts a function returning durations only through `dur=`. Fine, but the table should say a function after `@` is always a rate.
- **Spec 03 "Dwell time" table says `ss.gamma` with integer shape is "exact" as `k` stages.** Verified: `ss.erlang(k=3, mean=ss.days(5))` and `stages=3` give identical peaks (2012.07). But `to_json()` writes the first as `{'dist': 'gamma', 'a': 3, 'scale': ...}` with `stages: None` and the second as `{'dist': 'expon', 'mean': [...]}` with `stages: 3` (`s3`). The spec (14) says the two "say the same thing"; the model spec should then normalize them to one form, or the hash of equal models differs.
- **Spec 03 says `describe()` prints `rate = 0.3 × I.n / N [homogeneous mixing]`.** It does, and also prints `beta = 0.3 per step` *before* the sim exists even when the sim will be in days; after `init()` it says `per day`. Fine, but a `describe()` on a disease outside a sim should probably say `per unit` or omit the unit rather than assert `step`.
- **19-open-questions, now effectively answered by the implementation:** (i) *"Where `init_prev` puts people"*: through the transmission, so `E` in an SEIR; verified (`n_E[0] = 100, n_I[0] = 0`). (ii) *"`init` as a fraction or a count"*: implemented as specified, including `int 1` versus `float 1.0`. (iii) *"How robust is picking up names from variables"*: robust in a function, through a helper, and under `exec` with globals (`s6`); fails for states in a list or bound to two names, with the `E210` message the spec promised; the one failure not foreseen is Finding 1. (iv) *"A transition whose rate uses only counts is classed as spontaneous"*: yes, and the cholera `infection` is printed `spontaneous` with no `infectious` state; `R0()` has a special sentence for it (`diseases.py:216-217`). (v) *"Flow result names"*: both exist (Finding 10); the vignettes are written, so the question can close. (vi) *"`I` and `I.n`"*: unchanged; the three `E208`/`E211` messages are good and I would not add `I.contact`.
- **Over-specified:** 03's operator table lists `b @ x` ("into b, at x") as a form a user can write. It is an implementation detail of precedence; listing it invites `I @ 0.1` as a standalone thing, which has no meaning until a source arrives.
- **Under-specified:** what `k * transition` means when the transition is finished (Finding 4); what a `ss.by` of durations after `@` means (Finding 3); whether `ss.Disease(beta=0.3, ...)` with a bare number is an error, a parameter, or the value of the slot called `beta` (Finding 9).

## Bugs

All reproduced in `s2_after_at.py` and `s8_outcomes_bugs.py`.

1. **`I >> R @ ss.days(10) * 2` silently drops the `* 2`.** `_arith` (`transitions.py:748-761`): `rate is None`, `beta is None`, and `op == '*'` is exempt from the `elif op != '*'` error, so the copy is returned unchanged. Output: `dur: par(days(10)) rate: None`.
2. **`I >> R @ ss.days(10) @ 0.1` sets both.** `__matmul__` (`730-737`) never checks for an existing driver. Output: `dur: par(days(10)) rate: par(0.1)`. `S >> I @ 0.3 * I @ 2` likewise replaces the rate with 2.
3. **`ss.by` of durations after `@` becomes a rate.** `__matmul__` tests `isinstance(value, (ss.dur, ss.Dist))` and peeks into `TimeValue.typed` but not `by.items`. Output: `recovery.dur = None`, `recovery.rate = par(by(age=[days(5), days(20)]))`; the model runs (`n_R[-1] = 102` versus `4636` for the equivalent `ss.perday`).
4. **A `FlowSum` passed positionally cannot be named.** `collect` converts it to an `outcome` *before* `find_name` (`transitions.py:1164-1170`), so the lookup searches for the new object and fails: `exits = recovery + death; ss.Disease(..., exits)` raises "Could not find a name for an outcome" even though `exits` is a variable. Workaround: pass by keyword.
5. **Dimensionless factors in a transmission rate are exported as rates.** `to_json` (`1798`) marks every non-literal `par` in a contact-mediated rate `'rate'`; `R0 * gamma * I` exports `R0` as `{'type': 'per', 'value': 3.0}`. `describe()` gets it right (`R0 = 3`, `gamma = 0.1 per step`).
6. **String state names resolve through the caller's frame** (Finding 1): `E206` against an unrelated module, or `E204` for a state that exists under another variable name.

## Keep

- **The equivalence of the spellings is real.** Identical `to_json()` and bit-identical results for strings, operators, explicit, and class (`s1`). This is the property everything else rests on; do not let the shorthand constructors grow options the operator form lacks (or the reverse).
- **The bare-state rule and its error messages.** `E211 ExposureToCount` and `E208 NonlinearExposure` say exactly what to do, and `describe()` expanding `beta * I` to `0.3 × I.n / N [homogeneous mixing]` answers the question a newcomer will ask. The cholera and malaria models read like the equations and run as an ODE or as agents with the reservoir staying a count (`s5`: ODE peak W 3.41e6, agents peak W 3.54e6). Keep `I` versus `I.n`; do not add `I.contact`.
- **"A duration after `@` is a dwell time, a bare number is a rate."** The rule is right; only its edges leak (Finding 3).
- **Branching semantics.** "Transitions with `p=` split on entry; transitions with only rates compete" covered all five branching shapes I tried, including a per-branch dwell time with a competing death hazard, with `describe()` making the split visible (`p = 0.1` / `p = 0.9` lines) and `E205` catching a 0.9 total. The "one timed transition takes the remainder" rule is a nice touch.
- **`ss.erlang` / `stages=` / `ss.gamma(a=k)` agree** to the last digit, and the lognormal note (`approximated by 11 Erlang stages (mean 5 days kept; std 1.51 instead of 1.5 days)`) is a model of how to report an approximation.
- **`init` only where nonzero, remainder to the first state, and `init_prev` through the transmission.** Simple, and `describe()` shows the result.
- **Anonymous slot parameters named `beta`, `dur`, `rate`, qualified only on a clash.** `sir.pars.dur_recovery` appearing only when there are two durations is the right amount of naming.
- **`E602 UnknownParameter` with suggestions**, on `set_pars` and on `ss.Sim(pars=)`.
- **`describe()` marking every guess.** Every surprise in this review was visible in `describe()` output. The problem is only that some of them should not be guesses.

## Outside my area

- `ss.Sim(verbose=1)` prints a boxed "Running days(N)" banner for every step of an ODE run (`s3`); at `verbose=1` this is 60 banners for 60 days. Probably a Starsim 3 default that does not fit a compartmental run.
- `sim.pars.solver` stays `'auto'` after the run even though the solver chosen is known and printed (`s3`, `s4`); a `sim.solver` attribute holding the resolved choice would help scripts that compare solvers.
- `ss.SIR` (Starsim 3) and a declared `ss.Disease` can run in one sim with a network (`s7`), which is good; but their result names differ (`n_susceptible` versus `n_S`, `new_infections` only on one side), so a plot of both diseases is two vocabularies.
- A `describe()` call on a disease that is not yet in a sim prints `beta = 0.3 per step`; the units reviewer may want `describe()` to say `[unit not yet known]`.
