# The language core: merged findings

This merges every finding about the language core from the four reviews (F = `fable_review/`, F2 = `fable2_review/`, O = `opus_review/`, C = `codex_review/`): states and the people-versus-count inference, transitions and `>>`/`@`, arrow strings and names taken from caller frames, parameters and keywords, time types and the unit of bare numbers, initial conditions, and the core vocabulary. About 110 source findings collapse to 38 here. The reviews agree closely on the facts: every factual claim I re-ran reproduced, with three small corrections noted under each finding and in the disagreements. They differ mainly on remedies (how much name capture to keep, how the unit is chosen, the initial-value rule). The verdict: the language itself (one transition class, `S >> I @ beta * I`, bare state versus `.n`, inferred states, `describe()`) is sound and every review says to keep it. What undermines it is a set of guesses that reinterpret what the user wrote (the unit of a bare number, keyword numbers, which state is "first", unit-typing of dimensionless factors) and that do it silently. Most fixes are small and none changes the language. My checks are scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/lang/` (`v1_units.py` to `v7_lazy.py`), run with `/software/conda/envs/v4/bin/python` at `a20c4f7e`.

## Findings

### L1. The unit of every bare number is set by how the run is written (`dt`, the form of `start`/`stop`, a typed `stop`), not by the model

**Raised by:** O core 1, F 01§5 and §8 and bug 1, F2 03 other ("default `dt` is one unit"), F2 01 f9, F2 06 f3, F 12 bug 6, C D3 and F09; ranked first-tier in the O and F2 syntheses.

**Kind:** silent wrong answer.

**Status:** reproduced by O, F, F2, C; I re-ran all of it (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/lang/v1_units.py`, `v2_pars.py`). One model, `S >> I @ 0.3*I` (bare) and `I >> R @ ss.days(10)` (typed), 10,000 people:

| Sim arguments | Unit | Peak `n_I` | Final `n_R` |
|---|---|---|---|
| `stop=84` | day | 3036 | 9342 |
| dates `2020-01-01` to `2020-03-25` | day | 3036 | 9342 |
| `stop=84, dt=ss.weeks(1)` | week | 100 | 173 |
| `stop=ss.weeks(12)` | week | 100 | 172 |
| `start=2020, stop=2021` | year, 2 time points | 100 | 101 |

Also: `stop=1899` gives 1,900 days but `stop=1950` gives the years 1900 to 1950 and `stop=2000` the years 1950 to 2000 (the year heuristic, `time.py:2250`, applies to `stop` alone, and the start comes from Starsim 3's 50-year default). `start=2000, stop=2005` with a `ss.days(10)` recovery as agents gives a one-year step and `cum_infection = 0`. A code module's typed defaults are ignored by rule 4 (`sim.py:1053`): `ss.Sim([ss.SIS(), sir], stop=100)` with a bare `sir` reports `unit=None`, runs on `days(1)` steps, converts `SIS`'s `peryear(0.05)` to 0.000137 per step, and records no note or warning. One correction to O: its row `stop=84, dt=ss.weeks(1)` runs 84 weeks (85 points), not "the same 12 weeks"; the conclusion (beta becomes per week) stands.

**Merged proposal:** a bare number is in the model's unit, and the model's unit is a property of the model, never of the run. In order: `unit=`; else the typed values in every module (declared and code), with an error asking for `unit=` when they disagree and bare numbers are present (F2 01 f9, O core 1, C D3); else, if the sim has calendar dates and nothing is typed, days with the existing note; else steps. `dt` only sets the step and its default is one unit. A numeric `start` or `stop` is always elapsed time in the unit; a calendar year is written as a date (`'2020'` or `ss.date(2020)`), so `stop=2000` stops meaning 1950 to 2000 (F, C). A typed `stop=ss.weeks(12)` sets the length, not the unit.

```python
# before: the epidemic vanishes when only the step changes
ss.Sim(sir, stop=84, dt=ss.weeks(1))     # unit='week' (from dt): beta is now 0.3 per week
# after
ss.Sim(sir, stop=84, dt=ss.weeks(1))     # note: unit='day' (from ss.days(10) in 'recovery'); dt = 1 week
ss.Sim(sir, start='2020', stop='2030')   # years are dates; stop=2030 alone is 2,030 units
```

**Cost:** a reordering of `Sim.init_unit` (`sim.py:1026-1074`) plus one error; deletes the `assume_cal_year` path for numeric stops. **Breaks:** `ss.Sim(stop=2020)` meaning "until the year 2020" (Starsim 3 idiom; needs `stop='2020'`), and any sim that relied on `dt` or year-like numbers to type its bare numbers, which in every case found was silently wrong.

### L2. Each bare factor in a rate is typed as a rate per unit, so `R0 * gamma * I` changes with the unit

**Raised by:** F 10 bug 1 (checked again in the F synthesis §2), with the export side in F 02§9 and bug 5 (`R0` exported as `type: per`) and the naming side in F2 04 f7 (the leftmost factor is named `beta`).

**Kind:** silent wrong answer.

**Status:** one review only; re-ran (`v2_pars.py`): `S >> E @ R0*gamma*I` with `R0 = ss.par(2.5)`, `gamma = ss.par(0.2)` peaks at 1277.9 with `unit='day'` and 82.7 with `unit='week'`, matching F's numbers exactly. Both factors are converted, so the product scales by 49 instead of 7. Every importer writes dimensionless parameters this way (F 10), so an import is right only at the unit it was imported in.

**Merged proposal:** only the rate as a whole carries a unit. In a rate slot, the constant coefficient (the product of all constant factors) is converted once; a bare or untyped `ss.par` inside a product is dimensionless. `E102 WrongDimension` (L19) then catches the typed cases (`ss.perday(2.5) * ss.perday(0.2)` already raises). Record a factor's kind in `resolve_pars` (`'factor'` versus `'rate'`, F 02 codebase §4) so `describe()`, `to_spec()`, and the engines read one field.

**Cost:** small, in `resolve_pars` and the unit conversion. **Breaks:** nothing correct; models that were calibrated at one unit keep their numbers at that unit.

### L3. A number passed as a keyword to a disease declares a new parameter: typos pass, and `SIR(beta=0.3)` is ignored

**Raised by:** O core 2, O surf 2, F2 03 f3, F 02§9 (two unlinked `beta`s); in the O and F2 synthesis tables of wrong guesses.

**Kind:** silent wrong answer.

**Status:** reproduced by O and F2; re-ran (`v2_pars.py`). A class with `infection = S >> I @ 0.05 * I`: `SIR(init_prev=0.01).R0()` is 0.5; `SIR(init_prev=0.01, beta=0.3)` is still 0.5 with pars `beta, beta_infection, dur` (the model's own value was renamed); `pars=dict(beta=0.3)` gives 3.0; `SIR(bta=0.3)` is accepted. `ss.Disease(..., init_prv=0.01)` builds with pars `init_prv, beta, dur` and runs with `max n_I = 0.0`. The cause is `Graph.collect` (`transitions.py:1151` onward) turning any number keyword into a new `ss.par`.

**Merged proposal:** one rule. A keyword that names an existing parameter of the module (declared, class-level, or slot-named) sets its value; any other bare-number keyword raises `E602` with the closest names. A new free parameter is declared with `ss.par(...)` (or a variable), never by a bare keyword. F2's extra check (a registered parameter that nothing uses is a `W` note) catches the remaining cases cheaply.

```python
SIR(beta=0.3)      # after: sets beta (R0 3.0)
SIR(bta=0.3)       # after: E602 UnknownParameter: 'sir' has no parameter 'bta'. Did you mean 'beta'?
```

**Cost:** small. **Breaks:** code that declared a parameter by a bare keyword and used it only in `step()`; it writes `ss.par(...)`.

### L4. "The first state" is the first one created, and it decides who is people and where the rest of the population starts

**Raised by:** F 02§2 (F synthesis §2); C D4 makes the general case (disconnected or reordered populations should not be inferred).

**Kind:** silent wrong answer.

**Status:** one review; re-ran with a correction (`v4_anchor.py`, `v3_ops.py`). States are sorted by creation order (`transitions.py:1251`, `key=lambda s: s.order`). In a cholera-shaped model (infection driven by `W.n`, shedding `ss.null >> W @ 10 * I.n`), `W, S, I, R = ss.states('W S I R')` puts all 1,000 people in `W` and gives final `n_R = 9.5`; `ss.states('S I R W')` gives 923.9. No note is stored. In a connected variant where people move from `I` to `W`, the remainder still goes to `W` (final `n_R` 9 versus 410). The correction to F: when the transmission uses a bare state (`S >> I @ 0.3*I`) the inverted model is not silent, it raises `E211 ExposureToCount`; the silent case is the reservoir-driven model, which is exactly the case the inference exists for.

**Merged proposal:** anchor the people component on the states that a contact-mediated transition moves people between (F), else the component containing the states with `init`/`init_prev`, else the largest connected component; send the remainder of `init` to the source state of the first transmission, not the first-created state. When the anchor needed a fallback, or more than one component has more than one state, warn (`W212` as a warning, not a note), or require `people=` as C D4 proposes for disconnected populations.

**Cost:** small, in `resolve_people`. **Breaks:** nothing in the spec or the vignettes, which all write `S` first.

### L5. The two-line SIR, the spec's front door, seeds nobody and has R0 = 0.5

**Raised by:** F 12§2, F2 03 f1, F2 06 f10, O core 3, O surf 1; first item in the F2 and O syntheses and §3 of F's. C's teaching example writes `init={'I': 0.01}` explicitly.

**Kind:** silent wrong answer.

**Status:** reproduced by all three; re-ran (`v2_pars.py`, the `init_prv` case gives the same flat run): `n_I` is 0.0 throughout, `describe()` says `init: S = 1.0 [guessed: nothing given]`, `sim.notes == []`. Seeded, `beta=0.05, dur=10` gives R0 = 0.5, so it still dies out.

**Merged proposal:** a disease with a contact-mediated transition and no `init`/`init_prev`/inflow downstream of it guesses `init_prev=0.01` and marks it (`init: I = 0.01 [guessed: nothing seeds the infection]`); this fills in a missing value, which is what principle 3 allows. Change the spec's numbers to a supercritical `beta=0.3`. Add O core 14's one line at `init()`: R0 and the number infected at the start, next to the solver line (`solver: ode (...); R0 = 3.0; 100 infected at the start`). That one line also exposes L1, L2, L3, and L12, and a `beta` read under a network that multiplies contacts (O core 14: R0 0.3 without a network, 3.0 with one). Gate it on a cheap next-generation computation and skip it silently otherwise.

**Cost:** a few lines in `Graph`/`init`, plus the existing `sim.R0()`. **Breaks:** a model that meant to start with no one infected writes `init_prev=0`.

### L6. `@` accepts a second driver, drops arithmetic, and cannot see inside `ss.by`

**Raised by:** F 02§3 and bugs 1 to 3 (checked in the F synthesis §2).

**Kind:** silent wrong answer.

**Status:** one review; re-ran all four (`v3_ops.py`): `I >> R @ ss.days(10) * 2` gives `dur = days(10)` (the `* 2` is dropped by `_arith`, `transitions.py:739-761`); `I >> R @ ss.days(10) @ 0.1` sets both `dur` and `rate`; `I >> R @ ss.by(age=[ss.days(5), ss.days(20)])` becomes `rate = by(...)`, a rate of 5 and 20 per step (F measured `n_R[-1]` 102 versus 4636); `I >> R @ ss.bernoulli(0.5)` becomes a dwell time.

**Merged proposal:** `Transition.__matmul__` (`transitions.py:730`) raises if a driver is already set; arithmetic on a duration-driven transition scales the duration or raises; the "look inside" that `TimeValue.typed` gets is extended to `ss.by` items; only distributions with a time dimension count as dwell times. All four are bug fixes; the `@` rule itself stays.

**Cost:** a few lines each. **Breaks:** nothing correct.

### L7. `>>` and `@` mutate the transition in place, so two transitions built from one partial are one object

**Raised by:** O core 7 (in the O synthesis bug table).

**Kind:** silent wrong answer.

**Status:** one review; re-ran (`v3_ops.py`): `base = S >> I; fast = base @ 0.2; slow = base @ 0.1` gives two names for one `S -> I @ 0.1`; `into = I @ 0.1; a = S >> into; b = R >> into` gives `a is b` with source `R`. `_arith` already copies, so the operators disagree with each other.

**Merged proposal:** `>>` and `@` return a copy when the field they set is already set (or always; transitions are small). Keep the in-place path only for the fresh right-hand side inside `S >> (I @ x)`.

**Cost:** a few lines. **Breaks:** nothing.

### L8. The string form guesses the source of infection, and for an SEIR it guesses the latent state

**Raised by:** F 02§5, F2 03 f6c and other ("three ways to say which state infects"); in the F and F2 synthesis tables.

**Kind:** silent wrong answer.

**Status:** reproduced by F (peak 1657 versus 2012) and F2 (180 versus 3061, a different model); re-ran (`v5_dwell.py`): `ss.transmission('S -> E', beta=0.3)` with no `source=` peaks at 1219 versus 1986 with `source='I'`, no note stored. All six SEIR-shaped vignettes already write `source='I'` (F).

**Merged proposal:** `source=` is required when the destination of a `ss.transmission` has a spontaneous onward transition (the latent shape); SIR, SIS, and SIRS keep the guess. See the disagreement on error versus run-time note.

**Cost:** a few lines in `resolve_transitions`. **Breaks:** SEIR string models that relied on the wrong guess.

### L9. A bare rate after `@` (or `rate=10`) is ten per step, and under the ODE nothing says so

**Raised by:** F 02§3(e), F2 03 other, F2 06 other, O core 12, O surf 15.

**Kind:** silent wrong answer.

**Status:** reproduced by all; re-ran (`v5_dwell.py`): `rate=10` on recovery gives peak `n_I` 100 (no epidemic) and `sim.notes == []` under `ode`; as agents the note fires with the wording bug "so 1 of people leave in a step". This is the spec's own rule, which every reviewer accepts.

**Merged proposal:** keep the rule; under every solver, note any spontaneous transition whose mean dwell is shorter than one step (O core 12's framing, which is a property of the model and the step rather than a guess from size, and catches the same cases as the "rate of at least 1 per step" note F, F2, and O surf 15 propose): `note: recovery has a mean dwell of 0.1 days, shorter than the 1-day step; a duration is written ss.days(10)`. Fix the `W102` wording.

**Cost:** a few lines. **Breaks:** nothing.

### L10. Values are not validated: bounds are not enforced, a parameter with no value runs as NaN, and negative initial populations are accepted

**Raised by:** C F07, O core 8, O surf 9, F2 01 f3 (`E604`); in the O synthesis (twelve codes never raised).

**Kind:** silent wrong answer.

**Status:** reproduced by C and O; re-ran (`v2_pars.py`): `ss.par(5, bounds=[0.01, 1])` runs (max `n_I` 899); `ss.par()` gives `n_I = [10, nan, nan]` under `ode` and a raw `TypeError` as agents; `init={'I': -0.1}` gives compartments `[110, -10, 0]`; `ss.par(bounds=[0.001, 1.0])` starts at the arithmetic midpoint 0.5005, where `04-parameters.md` says the middle on the search scale (0.0316 for a log-scaled rate) (O). `E604` and `E606` exist only in the code table (`transitions.py:31-41`).

**Merged proposal:** at `init()` (and on every `pars=`/`set_pars` change), check each `ss.par` is valued (`E604`) and inside its bounds (`E606`), and each initial value is finite and non-negative (C). Compute a bounds-only starting value lazily on the slot's search scale and mark it `[guessed: middle of bounds]`. Decide once whether bounds are validity constraints or only the search range (C); the spec says both.

**Cost:** two short loops. **Breaks:** models with out-of-bounds values, which were wrong.

### L11. `ss.by` arithmetic silently drops levels

**Raised by:** C F08.

**Kind:** silent wrong answer.

**Status:** one review; re-ran (`v3_ops.py`): `ss.by(age=[1, 2, 3]) + ss.by(age=[10, 20])` gives `[11, 22]` (`zip` at `transitions.py:2121-2127`).

**Merged proposal:** check level counts (ideally align by level labels) before arithmetic; a scalar broadcasts, a mismatch raises naming the dimension.

**Cost:** a few lines. **Breaks:** nothing correct.

### L12. `ss.transmission` with no `beta` gets `beta = 1` and does not mark it as a guess

**Raised by:** O core 12 and 13.

**Kind:** silent wrong answer.

**Status:** one review; re-ran (`v3_ops.py`): pars `beta = par(1)`, and `describe()` prints `beta = 1 per step` with no `[guessed]` (`transitions.py:822`, `sc.ifelse(beta, 1.0)`).

**Merged proposal:** require `beta=` (as `ss.progression` requires `dur=` or `rate=`, with `E209`). A forgotten beta is an explosive epidemic, not an obvious one.

**Cost:** one line. **Breaks:** code relying on `beta=1`.

### L13. `ss.Sim(a, b)` makes the second disease the sim's label and drops it

**Raised by:** O surf 3.

**Kind:** silent wrong answer.

**Status:** one review; re-ran (`v3_ops.py`): `type(sim.label)` is `Disease` and `sim.results` keys are `timevec, a`; `b` is gone with no message (`sim.py:65`, `def __init__(self, model=None, label=None, ...)`).

**Merged proposal:** `ss.Sim(*models, label=None, ...)` with everything after the models keyword-only. A list still works.

**Cost:** one signature. **Breaks:** `ss.Sim(pars, 'label')` positionally, which is rare.

### L14. As agents, a seeded cohort stays one step too long, `stages=` is ignored with `rate=`, and `dur=`/`rate=` (declared equivalent) branch differently

**Raised by:** O core 4 (in the O synthesis bug table).

**Kind:** silent wrong answer.

**Status:** one review; re-ran three of its four claims (`v5_dwell.py`): with `I >> R @ ss.constant(ss.days(2))` and everyone seeded at `t = 0`, agents give `n_I = [1000, 1000, 1000, 0, ...]` (three days, not two); `ss.Transition(I, R, rate=0.1, stages=3)` gives `n_I[10]/n_I[0] = 0.3679` (exponential) versus 0.4232 for `dur=10, stages=3`, and `describe()` does not mention stages; `severe = ss.progression('I -> H', p=0.1, dur=5)` with `mild` given `dur=10` silently takes the 0.9 remainder (and `describe()` prints no `p` for it), while the same with `rate=0.1` raises `E205 ... sum to 0.1`. I did not re-run O's first-step exit shares (0.528 sampled versus 0.394 hazard at `dt = 1 day`).

**Merged proposal:** schedule initial occupants as if they entered at `t = 0`; normalize a constant `rate=` on a spontaneous transition to a dwell time at build (filed, staged, and branched like `dur=`), or raise when `stages=` is given with `rate=`; print the remainder `p` as a guess. O's larger proposal, to realize exponential dwells as a per-step hazard under agents so `dur=d` and `rate=1/d` agree, is an engine change for the solvers area.

**Cost:** small for the first three; moderate for the engine change. **Breaks:** agent results move slightly toward the ODE.

### L15. A typo in an arrow string makes a new reservoir state

**Raised by:** O surf 9, F2 synthesis table ("misspelled destination ... a new state, nothing printed").

**Kind:** silent wrong answer.

**Status:** reproduced by O; re-ran (`v6_vocab.py`): `ss.progression('l -> R', dur=10)` after `'S -> I'` gives final `n_I = 10000` and, at `verbose=0`, `sim.notes == []` (O saw the "treated as a count" notes at the default verbosity). `W207 UnreachableState` is never raised.

**Merged proposal:** in inferred mode, a state that is the source of a transition, has no inflow and no `init`, and is not the anchor (L4) raises with a close-name suggestion (`'l' is never entered; did you mean 'I'?`), as `E204` already does for declared states. Raise `W207` or delete it.

**Cost:** a few lines. **Breaks:** a deliberately empty source state, which writes `ss.State('l', init=0)`.

### L16. `pars=` on `ss.Sim` means sim settings or model values, decided by sniffing the keys

**Raised by:** F 11§2, F2 03 f4, F2 06 f8, F synthesis §4.

**Kind:** silent wrong answer (a mixed dict crashes).

**Status:** reproduced by F, F2; re-ran (`v2_pars.py`): on an SEIR with `dur_latency`, `dur_recovery`, `ss.Sim(b, stop=100, pars=dict(dur=5))` is accepted, sets `sim.pars.dur = 5`, and changes nothing (the run still has 101 points because `stop` wins; F2 described it as setting the run length). The sniff is `sim.py:77-78`. F: a mixed dict raises an unrelated `KeyNotFoundError`; `ss.Sim(sir, pars=fit.pars)` fails when a fitted parameter is called `dur`.

**Merged proposal:** `pars=` is model values only, with `E602` for anything else; sim settings are keywords or the dict in the first position (already supported). Drop the Starsim 3 sim `dur` (a duplicate of `stop`). F 11 prefers going further (no `pars=` on `Sim`; `sir.set_pars(fit.pars)`); see Decisions.

**Cost:** small. **Breaks:** `ss.Sim(pars=sim_pars)` in 21 test files and about 70 doc cells (F 11), mechanical; one release can accept the old form with a note.

### L17. Initial values: `1` is one person and `1.0` is everyone, and the three seed spellings do not hash alike

**Raised by:** F 02§7, F2 01 f8, F2 03 f7, O core 11, C D4; in the F2 and O synthesis tables.

**Kind:** silent wrong answer.

**Status:** reproduced by all; re-ran (`v5_dwell.py`) with 1,000 people: `I=1` gives 1, `I=1.0` gives 1000, `I=0.5` gives 500, `I=1.5` gives 1.5. On hashing, a partial correction to F2: `init=dict(I=0.01)` and `ss.State('I', init=0.01)` give the same model hash (`17001c`) and `init_prev=0.01` a different one (`f46d0e`), with identical results; F2 reported three different hashes.

**Merged proposal:** see the disagreement below; I recommend "below 1 a fraction, 1 and above a count", no type distinction, with "everyone in one state" given by the remainder rule (L4). Resolve `init_prev` and `init` to one spec field before hashing, and put the initial condition in the `pars` hash rather than `model` (F2 03 f7).

**Cost:** small. **Breaks:** `init=dict(I=1.0)` meaning everyone, which no example uses.

### L18. A sim with no unit runs in days underneath

**Raised by:** F 01§8, F2 06 f3, F 11§8.

**Kind:** spec drift.

**Status:** reproduced by all three; consistent with L1's re-run (`unit=None`, `dt=days(1)`). `sim.py:1063` sets `unit_base = unit or 'day'`, so `t.yearvec` and `t.datevec` hold meaningless values (F2 06), a code module's year-typed rates are converted per day (L1), and with `dt=0.1` `describe()` says `beta = 2 per step` and `every 0.1 steps` (F 01).

**Merged proposal:** a real step unit (a `dur` with no calendar) so `yearvec`/`datevec` do not exist rather than being wrong; `describe()` says "per unit" whenever `dt != 1`. With L1, a sim that mixes a typed code module and a bare declared one gets the module's unit or an error, never days by default.

**Cost:** moderate, in `Timeline`. **Breaks:** nothing a user relies on.

### L19. Typed arithmetic does not match the spec, and the dimension check the spec uses as its worked example is not there

**Raised by:** F 01§6 and bug 4, F2 06 f5(a, b), F2 01 f3, O core 8 and 16, O surf 9; F synthesis §7 table.

**Kind:** spec drift (the missing E102 lets one silent wrong answer through).

**Status:** reproduced by all; re-ran (`v6_vocab.py`): `ss.perday(0.1) * ss.days(3)` is 0.259 (a probability; `02-types.md:62` says the expected count 0.3); `ss.freqperday(0.1) * ss.days(3)` is 0.3; `1 / ss.days(10)` is `freq(1/10)`; `2 / ss.perday(0.1)` raises an empty `NotImplementedError`; `perday + probperday` and `perday - peryear` raise plain `TypeError`s, not `E101`; `ss.probperday(0.1) + ss.probperday(0.1)` is 0.2 but `2 * ss.probperday(0.1)` is 0.19 (O); `ss.progression('I -> R', rate=ss.peryear(0.1) * ss.days(3))` builds as a rate of 0.00082 per step (the spec's `E102` example); `dur=ss.probperday(0.1)` builds as a dwell time.

**Merged proposal:** keep the code's `per × dur` (a probability; 46 call sites rely on it, F) and correct `02-types.md:62`. Make `number / dur` a `per`, implement `number / per` as a `dur`, make `-` work wherever `+` does, make `prob` arithmetic consistent or refuse `prob + prob`, raise `E101` as a `CodedError` naming both types, and implement `E102` in rate and `dur=` slots. Write the table once and test every row.

**Cost:** small. **Breaks:** `1 / ss.days(10)` changes from `freq` to `per`; under L28's cut of `freq` this is moot.

### L20. Names are taken from caller frames at two moments, and the rule fails on the first refactor into a helper

**Raised by:** F 01§5, F 12§3, F 11§7, F2 03 other (E206), F2 04 f7, F2 06 f5(d), O core 6 and 9, O surf 14, C D5; in all four syntheses.

**Kind:** first-week friction (a crash for states, silent renaming for modules and transitions).

**Status:** reproduced by all; re-ran (`v6_vocab.py`): states made in one function and the disease built in another raise `E210 UnnamedState: state number 1 of the module has no name`; `ss.Sim(make())` names the module `disease`, while `x = make(); ss.Sim(x)` names it `x`; inline operator transitions are `S_to_I`, `I_to_R` (so `adjust={'infection': ...}` then fails with `E301`, F 12); building a second disease from the same states or from the same transition objects raises `E206`, even though the first was never put in a sim. The module lookup applies only when `type(mod) is ss.Disease` (`sim.py:113`), so a `Disease` subclass bound to `flu` is named `mysir` (F 11, F2 06). F2 04 found the lookup robust in about sixty other cases; the failures are the helper, the alias, and the subclass `__init__`.

**Merged proposal:** see the disagreement below for the positions. My recommendation: (a) teach `ss.states('S I R')` as the one spelling and drop the integer form, so states never need a frame (F 01, C D5); `S = ss.State()` in a class body keeps `__set_name__`; (b) name a module by `name=`, else structurally from its states (`sir`, `seir`, `_1` on a clash; O surf 14), applied to every module, with no lookup at the `ss.Sim` site; (c) keep the transition lookup at `ss.Disease(...)` (it has a safe fallback), and make `E301` say "transitions passed without a variable or keyword are named from their endpoints"; (d) the graph copies the states and transitions it is given, so they can build any number of diseases and `E206` is raised only by the sim, where it belongs (O core 6, F2 03).

```python
# before
def make(): S, I, R = ss.states(3); return S >> I @ 0.3*I, I >> R @ ss.days(10)
ss.Disease(*make())                      # E210
ss.Sim(build_seir())                     # sim.results.disease
# after
def make(): S, I, R = ss.states('S I R'); return S >> I @ 0.3*I, I >> R @ ss.days(10)
ss.Disease(*make())                      # works anywhere
ss.Sim(build_seir())                     # sim.results.seir [guessed: from its states]
```

**Cost:** deletes the `ss.Sim`-site pass (`sim.py:109-121`) and the state half of `find_name`; the copy is 30 to 60 lines in `Graph` (O). **Breaks:** `ss.states(3)` in the spec and tests; `measles = ss.Disease(...)` scripts that expect `sim.results.measles` need `name='measles'`.

### L21. A string state name resolves to whatever caller variable has that name

**Raised by:** F 02§1 and bug 6 (F synthesis §2).

**Kind:** crash.

**Status:** one review; re-ran (`v3_ops.py`): inside a function with `S3, I3, R3 = ss.states('S I R')` and a module-level `I`, `ss.Transition('I', 'R')` raises `E206 ... two members of "c" are both named "I"`, because `'I'` was resolved to the global variable (`transitions.py:1246-1250`).

**Merged proposal:** resolve a string only against the module's own states (declared by `.name`, or inferred); delete the five frame-lookup lines. Unnecessary if L20(a) removes name capture for states.

**Cost:** none. **Breaks:** nothing correct.

### L22. `sir.pars.beta = x` replaces the `ss.par` with a float, losing bounds and unit and breaking `ss.modify`

**Raised by:** F 01§2 and bugs 2 and 3, F2 06 f9, F 02§9; checked in the F synthesis.

**Kind:** crash (at a later `ss.modify`), after silently losing bounds and unit.

**Status:** reproduced by F and F2; re-ran (`v2_pars.py`): after `sir.pars.beta = 0.6` the entry is a `float` and `ss.modify(sir.pars.beta, ...)` raises `E504 ... must be a named quantity, such as sir.pars.beta`, which is what the user wrote. The run itself uses the new value (F). On the warning, F2 06 is not reproduced: attribute assignment prints nothing; it is `sir.pars.update(beta=0.6)` that prints `No known mechanism for handling par → float` (as F says).

**Merged proposal:** `Pars.__setitem__` (`parameters.py:73`) sets `.value` when the existing entry is an `ss.par` and the new value is not; then `Disease.set_pars` and the `par.raw` read-back (`transitions.py:218-224`) go.

**Cost:** removes code. **Breaks:** nothing.

### L23. What counts as a parameter, and what it is called, depends on the rest of the model

**Raised by:** F2 03 f5, O core 10, F 01§4, F 02§9, F2 04 f5 and f7, F2 01 other; F2 synthesis decision 12. F 02 Keep takes the opposite view on naming (below).

**Kind:** first-week friction.

**Status:** reproduced by F, F2, O; re-ran (`v6_vocab.py`): a two-transition SIR has `beta, dur`; adding `waning=ss.progression('R -> S', dur=100)` renames `dur` to `dur_recovery`, so any `pars=`, `ss.modify`, saved fit, or data reference to `dur` breaks. `sir.infection.beta` is `None` (cleared at `transitions.py:1328-1329`) though `04-parameters.md` says it is addressable; `sir.recovery.dur` works. A bare number bound to a variable (`gamma = 0.1; I >> R @ gamma`) is named after its slot (`rate`), not the variable (F 01); in `S >> I @ R0 * 0.1 * I` the literal 0.1 becomes the `rate` parameter and `R0` gets no slot (F2 04); `beta * contacts` registers neither (F 01). `sir.infection.rate = 0.1` is accepted and then fails at run time, while `sir.recovery.dur = 5` works through a setter (F2 04).

**Merged proposal:** give every inline slot value one stable structural name, `infection.beta`, `recovery.dur`, with the short name (`beta`, `dur`) as an alias that resolves while unique and raises "ambiguous: recovery.dur or waning.dur" otherwise (O core 10); keep `tr.beta` pointing at its `par` after build. Constructor keywords (`beta=`, `dur=`, `rate=`) register a parameter because the keyword names it; a literal inside arithmetic is anonymous and printed as its value (F 01 option i); a name the user gave (`ss.par(name=)`, a keyword) is untouched. Drop the `dur` setter magic so all values change through `pars` (F2 04).

**Cost:** moderate; changes printed names in specs and hashes. **Breaks:** saved specs with `beta_infection`-style names, which are not stable today either.

### L24. Rates and conditions are two expression trees that crash where they meet, and lazy arrays are only partly lazy

**Raised by:** F2 04 f1, C F12, C D6; F2 synthesis "one expression tree".

**Kind:** crash (and one silent case: `isin` gives empty eligibility).

**Status:** F2 and C each reproduced their own; re-ran both (`v3_ops.py`, `v7_lazy.py`): `ss.Transition(I, Q, rate=0.1, where=Q.n < 20)` raises `TypeError: bad operand type for unary ~: 'Func'`; `S >> I @ 0.3*I*(ppl.age > 5)` builds and then dies at run time with `float() argument must be ... not 'Cond'`; on an uninitialized `ss.People(10)`, `age > 5` is a lazy `Cond`, `age + 1` raises `IndexError`, and `age.isin([1, 2])` silently returns an empty `BoolArr`, freezing eligibility before the population exists.

**Merged proposal:** one tree: `Cond`'s leaves become `Expr` leaves, `&`/`|`/`~` become `Func`s, and `Selector` is one more evaluation context (F2 04). The minimum is a guard at the seam that names the other form. For lazy arrays, either make arithmetic and `isin` lazy too or reject them at once with the function form as the alternative (C). The `ss.People(1)` placeholder that lazy conditions need is an interventions-area item.

**Cost:** about 190 lines of `conditions.py` go for the full merge. **Breaks:** nothing a user writes (`ss.Cond` appears in one test).

### L25. `0.5 * infection` is both a transition at half the rate and half the flow

**Raised by:** F 02§4, F2 04 f2, O core 17; F2 synthesis item 9.

**Kind:** surface.

**Status:** reproduced by all three (F2: `ss.Disease(half=0.5 * infection)` registers the 0.5 as a parameter `beta_half`, and `ss.Disease(2 * infection, recovery)` loses the name). Not re-run; the three agree.

**Merged proposal:** arithmetic on a finished transition is always a flow weight (for `ss.outcome`, results, `infection / S`); arithmetic on an unfinished one (`I @ beta`) keeps forwarding to the rate, which is what makes `S >> I @ beta * I` parse. Halving a rate is written in the rate or with `ss.modify`. Delete "`infection * 0.5` is the same transition at half the rate" from 03 and the rate rewrite and `flow_of` from `_arith`.

```python
cases  = ss.outcome(0.5 * infection)     # half the flow (unchanged)
slower = S >> I @ 0.5 * beta * I          # half the rate is written as a rate
ss.Disease(half=0.5 * infection)          # after: error naming both spellings
```

**Cost:** small. **Breaks:** no test or vignette uses a scaled finished transition as a transition.

### L26. The first wrong guesses get Python's errors, not Starsim's

**Raised by:** F 12§6, F2 03 other, F2 04 f10(g), F 01 bugs 4, 7, 8, O core 21.

**Kind:** first-week friction.

**Status:** reproduced by F, F2, O; the time-arithmetic ones re-ran (L19). The list: `0.1 * ss.null` (`TypeError: unsupported operand`), `ss.death('I ->', p=0.1)` (unexpected keyword), `S >> 0.5*A + 0.5*B @ r` (message points at "one of its states", not at the missing parentheses that 03 promises), `E206` naming the same module twice (`"disease" and "disease"`), `E602` suggesting every parameter because candidates are prefixed by module name, `E210` showing `<unnamed state> -> <unnamed state>`, empty `NotImplementedError`s in rate arithmetic, and reprs that print a state as `<unnamed state>` at the arrow and `None` in the rate.

**Merged proposal:** wrap the operator and constructor entry points so arithmetic and keyword mistakes become coded errors that name the v4 spelling (`Null.__rmul__` raises its own); fix the four message bugs. Most of the `E210`/repr issues disappear with L20.

**Cost:** small. **Breaks:** nothing.

### L27. `describe()` misreports some of the guesses it exists to show

**Raised by:** F 01§9, F 02 spec section, O core 13, F 12§8, F2 03 ("what describe looks like").

**Kind:** first-week friction.

**Status:** reproduced by F and O; re-ran the `beta = 1` case (L12). The list: `describe(verbose=True)`, promised by 01, 02, and 10 for the per-step conversions, raises `TypeError`; a disease described before it is in a sim says `per step` even when its own typed values fix the unit, and the user's own object keeps saying so after `sim.init()` because the sim copied it (O); with `dt=0.1` it says `per step` for per-unit values (L18); `I infectious [guessed: source of 'S_to_I']` in the operator form, where nothing was guessed (O); the remainder branch `p` is not printed (L14); a bare rate on a spontaneous transition has no unit while beta does (F 01); `the rest in sus` when the rest is zero (F 01); init notes printed twice (F 12, F2 03).

**Merged proposal:** fix each (a line or two), add `verbose=True` with the per-step line under each transition, and add the conformance test O proposes: for each row of 03 §What is guessed, one model that triggers the guess and checks it is marked, and one that does not and checks it is not. With L1, a module can describe itself correctly outside a sim.

**Cost:** small. **Breaks:** nothing.

### L28. The time vocabulary: 27 exported time names for what declared models use as three types

**Raised by:** F 01§7, F2 06 f6, F2 03 f9 and other, O core 16, F2 01 other; F synthesis §6 table.

**Kind:** surface.

**Status:** reproduced by F, F2, O; re-ran parts (`v6_vocab.py`): `ss.day(1)` raises `'days' object is not callable` (the singleton instances `ss.day`, `ss.week`, `ss.month`, `ss.year` are one letter from the classes); `ss.dur(10)` silently becomes `years(10)` (with a warning when warnings are on). O verified that `freqperday(0.5)`, `perday(0.5)`, and `1/ss.days(2)` give identical results in every declared slot under both solvers; `freq*` and `probper*` are used only in v3 modules (F, F2 06).

**Merged proposal:** the v4 vocabulary is `dur`, `per`, `prob` with `days`..`years`, `perday`..`peryear`, and `ss.prob(0.1, 'day')` for the rest. `ss.freq` stays as a class for code modules' count-scaled parameters but leaves the top level (F2 06, O; F keeps it, see the disagreement). Remove `probper*`, `freqper*`, the four singletons, and the deprecated `ss.rate`/`ss.time_prob`/`ss.rate_prob` (no uses outside the v2-to-v3 migration guide, F). `ss.dur(10)` and `ss.Rate(2)` require a unit, since v4 has "no default unit".

**Cost:** small. **Breaks:** about 20 uses of `freqper*`/`probper*` in tests and docs and two in `networks.py` (F).

### L29. The `ss.` namespace exports 253 names; the vocabulary counts 94

**Raised by:** F 01§1, F2 03 f9, F2 01 f2, C F20, O surf 5; in all four syntheses.

**Kind:** surface.

**Status:** the counts disagree only in method; re-ran (`v6_vocab.py`): `dir(ss)` has 285 public names, 32 of them submodules, so 253 non-module names (F's 253 and O's 285 are the same count; F2 and C's 258 count the names imported in `__init__.py`). `ss.func` (counted) does not exist; `ss.Expr` (listed as removed) is exported.

**Merged proposal:** top level is the spec's vocabulary plus what the class form needs (`Module`, `Intervention`, `Connector`, `Analyzer`, `uids`, the array classes, `options`); internals move to their submodules (`ss.solvers.Engine`, `ss.transitions.Graph`, `ss.calibration.Fit`); Starsim 3 duplicates go to a legacy namespace or are deleted (the one-system area owns that list). Generate `17-vocabulary.md` from the exports and test that they agree.

**Cost:** one file (`__init__.py`). **Breaks:** Starsim 3 scripts using the moved names, softened by a legacy re-export.

### L30. Several small spellings in the core that can go or be stated

**Raised by:** `|` versus `+` for state sets: F2 04 f6, O core 15. `ss.states(1)` returning a `State`: F 02§6, O core 21. Positional values after an arrow: F2 03 other, F2 01 other, O surf 15. `ss.immigration`/`ss.emigration` as two extra names: F2 04 other, O core 21. `ss.erlang` next to `stages=`: F2 06 other, F2 01 other (keep: F 01, F 02). `ss.Transition` accepting the private `beta=`/`source=` keywords: F2 04 f3, F2 01 f6. `ss.par` not behaving like its value: O core 18. `ss.Disease(dict, ...)` reading a leading dict as `pars`: O core 21.

**Kind:** surface.

**Status:** re-ran (`v6_vocab.py`): `ss.states(1)` returns a `State` (so `ss.states(' '.join(names))` breaks for one name); `ss.transmission('S -> I', 0.3)` sets `beta` and `ss.progression('I -> R', 10)` sets `dur`, though `01-objects.md` and `15-errors.md` say positional values are rejected; `float(ss.par(0.3))` raises and `ss.par(0.3) * 2` is the expression `0.3 * 2`. F2 04 and O verified that `|` and `+` build the same model hash.

**Merged proposal:** `S + I + R >> D @ mu` is the many-to-one spelling (it needs no parentheses), `StateSet` goes, and `|`/`&`/`~` stay for conditions; `ss.states` always returns a list; accept positional values as the constructor's main keyword and fix the spec; fold immigration/emigration into keywords on `ss.birth`/`ss.death` or count them in 17; make `beta=`/`source=` private on `ss.Transition`; give `par` `__float__` and `__format__` and let `Expr.__bool__` evaluate when the expression has no states (O core 18); remove the leading-dict convention. The disagreements on `|`, positional values, and `ss.erlang` are below.

**Cost:** small each. **Breaks:** `(S | I) >> D` as a source set (keep as a deprecated alias for a release), `S = ss.states(1)`.

### L31. `ss.Disease` is two classes behind one name, and `sir.S` changes type at `init()`

**Raised by:** F 02§8, O core 5, F2 06 f2, F 11§1, F2 04 other, F 11 codebase §4; in every synthesis as "one system".

**Kind:** surface.

**Status:** reproduced by F, O, F2 (15 `graph is None` branches; `type(sir.S)` goes from `ss.State` to `BoolState` after an agent-based `init()`; `ss.SIR().equations()` raises; `Disease.__init__` decides which kind it is by scanning its keyword types, `diseases.py:73`, so a Starsim 3 subclass given an `ss.par` keyword flips into graph mode). Not re-run.

**Merged proposal (language side only):** never rebind `sir.S`; give `ss.State` a handle to the engine's arrays so `sir.S.n` and `sir.S.uids` work under every solver (F 02, F 11 bug: `sir.S.n` fails after an agent run); decide "declared or code" by class, not by argument sniffing. Rewriting `ss.SIR`/`SIS`/`SEIR` as declared models and moving the code path to `ss.Infection` belong to the one-system area.

**Cost:** touches the agent engine. **Breaks:** code reading `sir.S` as an array after `init()`.

### L32. Spec promises in this area that the code does not keep (beyond those above)

**Raised by:** F 01§10, F 02§11 and spec section, F2 04 f10, F2 01 other, F2 06 other and f5(c), O core 21.

**Kind:** spec drift.

**Status:** each verified by at least one review with `hasattr` or a run; `ss.func` re-ran (`v6_vocab.py`).

**Merged proposal:** delete or implement, one line each: `ss.func` (03, 17; delete, the plain function already works); `sir.solvers()` (01, 17; only `sim.solvers()` exists); `I.prev` (17) versus `I.frac` (02, code; keep `frac`); `ss.Sim(results=[...])` (01; not implemented); the typed-`stop` unit source and the `unit=` override of year-like numbers (code, not in 02; moot under L1); rule 4 reading only declared modules (L1); "the timestep is one step" (02:139; L18); a declared model with no `stop` silently gets 50 units (F2 01); "a transition bound to two variables takes the fallback name with a note" (01; no note is given, F 02); `ss.immigration` making a holding state a count although it is people arriving (F 02); `ss.erlang(k=3)` and `stages=3` writing different specs for the same model (F 02).

**Cost:** writing. **Breaks:** nothing.

### L33. `transitions.py` is four spec chapters in one 2,972-line file

**Raised by:** F 02 codebase §1, F2 04 f9, O core 19; F synthesis §8.

**Kind:** codebase.

**Status:** the three line counts agree (about 930 lines of `by`/`timeseries`/`seasonal`/`geo`/`gravity`/`Mobility`/`Strat`/`Transfer`/`stratify`, 315 to 330 of `Graph` presentation, 53 of error codes).

**Merged proposal:** split along the spec: expressions and `par` (with `by`/`timeseries`/`seasonal` in `parameters.py`), states and transitions, `Graph` build and check, presentation (`describe`, `equations`, `diagram`, `to_json`, `format_expr`) in its own module, stratification to `dimensions.py`, geography and mobility to the routes code, and `codes`/`CodedError` to `errors.py`. Leaves a core of about 1,550 lines. Pure move.

**Cost:** mechanical. **Breaks:** internal imports only.

### L34. `Transition` and `Expr` carry pending flags, sentinels, and three printers

**Raised by:** F 02 codebase §2, §4, §5, §6; F2 04 f3, f8, other; O core 21.

**Kind:** codebase.

**Status:** by inspection (consistent across F and F2).

**Merged proposal:** keep one private `pending` for the transmission's deferred source and drop `per_capita`, `every_state`, `migration`, `flow_of` (with L25) as attributes; replace `Count(kind='N')` with an `everyone` leaf and `literal=True` pars with a `Const` leaf (14 tests of `.literal`/`is_one` go); one printer (`to_str` calls `format_expr`); `resolve_pars` records each value's kind (`rate`, `dur`, `factor`, `count`) so `describe`, `to_json`, and `flow_terms` stop re-deriving it (also fixes L2's export); `StateSet` goes with L30; one shared `__set_name__` mixin; record people guesses as structure rather than dotted strings the sim parses back (`sim.py:399`); one reserved-name list, checked against the module's own class (O core 21: a transition named `describe` currently replaces the method, re-ran in `v3_ops.py`).

**Cost:** tidiness; medium effort. **Breaks:** nothing a user writes.

### L35. Two parameter paths, kept in step by hand

**Raised by:** F 01 codebase §2, F2 06 f3 and f9, O core 20.

**Kind:** codebase.

**Status:** by inspection, consistent.

**Merged proposal:** after L22, `Graph.pars` and `Module.pars` hold the same `par` objects with no read-back; `set_pars` and the `dict.__setitem__` dodge in `diseases.py:98-100` go; the `_bare`/`note_bare`/`forget_bare`/`check_bare_units` bookkeeping (about 50 lines, `parameters.py:38-60`, `modules.py:715-725`) goes once a bare number has one rule (L1, and the library rewrite).

**Cost:** removes code. **Breaks:** nothing.

### L36. Name lookup is done in six places with six fallback policies

**Raised by:** F 02 codebase §3, F 12 codebase §1, F 11 codebase §3.

**Kind:** codebase.

**Status:** by inspection (`diseases.py:74, 143`, `sim.py:73, 109-121`, `interventions.py:574-581`, `modules.py:520`, `arrays.py:947-949`).

**Merged proposal:** after L20, one `resolve_name(obj, keyword, frame, fallback)` for the pieces inside a module, and none at the `ss.Sim` site.

**Cost:** small after L20. **Breaks:** nothing beyond L20.

### L37. `time.py` is 2,612 lines for four types

**Raised by:** F 01 codebase §1, F2 06 other.

**Kind:** codebase.

**Status:** by inspection, consistent.

**Merged proposal:** move calendar code (`date`, `DateArray`, `datedur`) and the Matplotlib converters (about 200 lines) to their own modules; merge `prob.to_prob` into `Rate.to_prob` (near copies); drop `Rate.__mul__`'s Starsim 2 message and the deprecated trio; replace `_defer_ops` (which rewrites every subclass's operators so `time.py` defers to `transitions.py`) by `Expr` defining the reflected operators and `NotImplemented`; with L28, the class-map plumbing becomes a four-entry dict. Whether `Timeline` can drop `datedur` for `pd.DateOffset` is worth a try (F, open question B13).

**Cost:** mechanical. **Breaks:** nothing.

### L38. A user who learns the spec's examples gets three surprises the spec could have warned about

**Raised by:** F2 03 f6(a, b) and other, F 12§5.

**Kind:** first-week friction.

**Status:** reproduced by F2; not re-run, since the causes are in other areas. The spec's own `seed_cluster` per-step function lets `auto` pick `ode` and then dies at step 31 with `'CountState' object has no attribute 'uids'` rather than `E804`; an unstratified population's ages are uniform on 0 to 60, so `eligible=ppl.age > 65` selects nobody, silently; `ss.Sim(sir, solver='ode', n_agents=1000)` ignores `n_agents` without a note.

**Merged proposal:** recorded here because each starts from a core example; the fixes belong to the solvers and interventions areas (re-try per-step functions on the first real call; a note when a delivery finds nobody eligible on every step; one default age distribution).

## Where the reviews disagree

**How the unit of a bare number is chosen (L1).** F2 01 f9: rules 1 to 3 (`unit=`, typed `dt`, dates) are right; only rule 4 (typed values) should become an error on disagreement. F 01§8: count code modules under rule 4; apply the year heuristic to `start` only. F2 03: the smallest typed duration should set `dt`; dates set the calendar only. O core 1: `dt` never sets the unit; the model's typed values come before the dates; disagreement is an error. C D3: one explicit time basis, steps when absent; numeric `stop` always elapsed time; years only as dates. Recommendation: O plus C, with F2 01's error. My re-run shows `dt` and year-like numbers are the two inputs that destroy an epidemic without a word, and neither is something a user thinks of as part of the model; dates with nothing typed can keep giving days, as they do correctly today. This is one rule a user can say in a sentence ("bare numbers are in the model's unit"), and it deletes the year heuristic.

**Default seed for a transmission model (L5).** F 12: seed one case, or a `NothingHappened` warning. F2 03 and F2 06: `init_prev=0.01` marked `[guessed]`, or a note. O core 3 and surf 1: guess `init_prev=0.01`. O core 14: print R0 and the number infected at `init()`. C §1 ("a precise refusal is often easier to resolve than a plausible answer"): be explicit. Recommendation: guess `init_prev=0.01` and mark it, and add O's R0 line. The guess fills in something missing (principle 3's own test), nobody writes `ss.transmission` wanting zero cases, and the R0 line catches the unit and keyword mistakes that a seed cannot.

**The initial-value rule (L17).** F 02§7 and F2 01 f8: decide by size (below 1 a fraction, 1 and above a count), drop the type distinction. O core 11: decide by type (`int` a count, `float` a fraction, a float above 1 an error). C D4: separate count and fraction initializers, or one convention plus an explicit alternative. Recommendation: size, as F and F2. It reads the same whether a number came from a literal or a dataframe column (floats), needs no new keyword, and keeps `init=dict(Sv=5e4)` working, which O's rule would turn into an error. The only thing lost is `I=1.0` meaning everyone, which the remainder rule already covers.

**How much name capture to keep (L20).** F2 04 f7: keep it; sixty probes, two local sharp edges; walk up to the first frame outside `starsim/` so subclasses work. F 01§5: drop `ss.states(3)`, keep the transition lookup. F 12§3: name every object once, where it is constructed, never at `ss.Sim`. O core 9: read the assignment targets from the caller's bytecode at `ss.states(3)` (CPython-specific, frame lookup as fallback), or else document only `ss.states('S I R')`. O surf 14: structural module names. C D5: do not make frame inspection a semantic dependency; teach `ss.states('S I R')`. Recommendation: as in L20 (`ss.states('S I R')` only; structural module names; transition lookup kept with its fallback; copy on construction). Bytecode reading is more magic, not less. F 12's "name the module where it is constructed" cannot see `sir = ss.Disease(...)` at call time without the same bytecode trick, so it reduces to O's structural name or to `name=`.

**Parameter names (L23).** F 02 Keep: short slot names qualified only on a clash are "the right amount of naming". F2 03 f5: always qualify (`dur_recovery` even alone), or drop slot-named values from `sir.pars` and address them as `sir.recovery.dur`. O core 10: canonical `infection.beta`, short alias while unique. F 01§4: only `ss.par` is a parameter. Recommendation: O's, which keeps F's short names in the common case and turns F2's silent rename into a loud ambiguity error.

**Keyword numbers on a disease (L3).** F2 03 f3 and O surf 2: an unused bare-number keyword is an error. O core 2: a keyword that names an existing parameter sets it; anything else is `E602`. Recommendation: O core 2, which also makes `SIR(beta=0.3)` work; F2's unused-parameter check as an extra note.

**`per × dur` (L19).** F 01§6: keep the code's probability, fix the spec row (46 call sites rely on it). F2 06 f5: the spec is the one that is wrong. O core 16: make the table the complete truth (direction unstated). Nobody proposes changing the code. Recommendation: fix `02-types.md:62`.

**`ss.freq` (L28).** F 01§7: keep the four types and cut only `probper*`/`freqper*`. F2 06 f6 and O core 16: drop `freq` from the user-facing vocabulary (identical results in every declared slot, verified by O). F2 01: asks whether it earns five names. Recommendation: drop it from the top level; its one real job, count scaling for a Poisson parameter, is a property of the distribution's parameter (F2 06).

**String-form source guess (L8).** F 02§5: an error when the destination has an onward progression. F2 03 f6c: a run-time note in that case. Recommendation: the error; the vignette authors already treat `source=` as required, and a note at run time is the kind that gets scrolled past.

**`|` on states (L30).** F2 04 f6: keep `|` as a synonym that returns the sum, delete `StateSet`. O core 15: `+` for sources, `|` only for conditions. Recommendation: O's, with `|` as a deprecated alias for one release; one spelling, no precedence caveat.

**Positional values after an arrow (L30).** F2 03 other: reject, as the spec says. F2 01 other and O surf 15: accept and fix the spec. Recommendation: accept (the positional is the constructor's main keyword); it is shorter and the two cases (`beta`, `dur`) are unambiguous. L9's note covers the `@ 10` confusion that F2 worries about.

**`ss.erlang` (L30).** F 01 and F 02 Keep: the one added distribution name, agrees with `stages=` to the last digit. F2 06 and F2 01: cut it, `stages=` says the same thing. Recommendation: cut it, keeping `ss.gamma(mean=, std=)`; one fewer name, and L32 shows the two spellings write different specs today.

**Factual disagreements, checked.** (1) Warning on `sir.pars.beta = 0.6`: F2 06 says it prints one; F says only `pars.update` does. Re-run: F is right. (2) Namespace size 253, 258, or 285: all correct for what they count (253 non-module names plus 32 submodules is 285; 258 is the `__init__.py` import list). (3) Seed-spelling hashes: F2 says three different hashes; re-run gives two (`init=` and `ss.State(init=)` agree). (4) The people anchor: F says the inverted model is silent; re-run shows it raises `E211` when the transmission uses a bare state and is silent (population in the reservoir) when infection is driven by the reservoir. (5) `pars=dict(dur=5)`: F2 says it sets the run length; re-run shows `stop=` still wins, so the key is silently accepted and has no effect. (6) O's "same 12 weeks" table: the `dt=ss.weeks(1)` row is 84 weeks.

## Keep

- The operator form and one `ss.Transition` whose kind follows from its rate; the four spellings build identical specs and bit-identical results (F 02, F2 03, F 12, O).
- The bare state as exposure through the routes and `.n` as a count, with `E208`/`E211` (all four).
- States that are not people, inferred, with no extra class (F, F2, O); fix only the anchor (L4).
- `ss.null` as a named source and sink (F 01, O, F2 04).
- "A duration or distribution after `@` is a dwell time, a bare number is a rate" (F 01, F 02, F2 06); add the note in L9.
- Branching semantics: `p=` splits on entry, rates compete, one timed exit takes the remainder (F 02, F2 04).
- `describe()` with its `[guessed: ...]` marks and the expanded rate `0.3 × I.n / N [homogeneous mixing]`, unshortened (all four).
- Coded errors that name the fix (`E204`, `E205`, `E209`, `E210`'s advice, `E211`, `E602`, `E705`) and `ss.note` (all four).
- `__set_name__` for the class form and the per-instance deep copy (F 01, F2 04, O).
- The `Expr` tree with one `evaluate()` and pluggable contexts, which JAX reuses (F2 04, F2 synthesis).
- `ss.par(bounds=)` as what `calibrate()` fits (F 12, F2 03).
- Erlang stages and the lognormal approximation note (`approximated by 11 Erlang stages ... std 1.51 instead of 1.5 days`) (F 02).
- `sim.check_dt()` holding the model fixed while varying the step (O).
- The `1 - ss.peryear(x)` refusal: dimensionally meaningless, and the message names the fix (F2 06).
- The printed source of the unit (`unit='day' (from start and stop dates)`) (F 01).

## Decisions for Cliff

1. **The unit rule:** bare numbers in the model's unit (`unit=`, else typed values in every module, error on disagreement, else days for dated sims, else steps), with `dt` and year-like numbers never setting it, and calendar years written as dates? (O, C, and F partly: yes; F2 01 would keep `dt` and dates as unit sources.)
2. **Default seed:** guess `init_prev=0.01` with a `[guessed]` mark, plus an R0 line at `init()`? (F, F2, O: guess; O adds the line; C leans to explicit.)
3. **Initial values:** decide by size with no `int`/`float` distinction, and hash `init`/`init_prev` as one field in `pars`? (F, F2: size; O: by type; C: separate initializers.)
4. **Names:** `ss.states('S I R')` as the only state spelling (dropping `ss.states(3)`), structural default module names with no lookup at `ss.Sim`, and copying states and transitions on construction? (F 01, C, O surf: yes to the first; O, F2: yes to copying; F2 04 would keep the current lookup with small fixes.)
5. **Inline parameter names:** canonical `<transition>.<slot>` with a unique short alias, and literals inside arithmetic anonymous? (O, F 01; F 02 would keep today's clash-only qualification.)
6. **Keyword numbers on `ss.Disease`:** set an existing parameter, else `E602`? (All three that raised it.)
7. **`pars=` on `ss.Sim`:** model values only, or remove it in favor of `sir.set_pars(...)`? (F, F2, O: model values only; F 11 prefers removal.)
8. **`ss.Sim(*models, ...)`** with everything else keyword-only? (O.)
9. **Time vocabulary:** `dur`/`per`/`prob` only at the top level; drop `freq` constructors, `probper*`, the `ss.day` singletons, and the deprecated aliases; require a unit in `ss.dur`? (F2 06, O; F keeps `freq`.) And fix `02-types.md` to the code's `per × dur`? (F, F2 06.)
10. **`k * transition`** is always a flow weight, never a rate change? (F, F2, O.)
11. **`source=` required** for SEIR-shaped string models? (F; F2 would note at run time.)
12. **Strict `@`** (second driver, arithmetic on a duration, `ss.by` of durations, non-time distributions all raise)? (F; no one opposes.)
13. **People anchor** on the transmission, with the `init` remainder going to its source state, and a warning on fallback, or `people=` required for disconnected populations? (F; C prefers required.)
14. **Merge `Cond` into `Expr`**, or only guard the seam? (F2 04 prefers the merge; C wants one expression system or callables only.)
15. **Small spellings:** `|` only for conditions; `ss.states` always a list; positional values accepted; `ss.erlang` cut; immigration/emigration folded into keywords? (Recommendations in the disagreements above.)
16. **Bounds:** validity constraints (an error outside them) or only the search range? (C asks; O and C want them enforced.)
17. **Agents' `dur=` versus `rate=`:** realize exponential dwells as a hazard so the declared equivalence holds, or keep sampling and state the difference? (O; the solvers area should weigh in.)
