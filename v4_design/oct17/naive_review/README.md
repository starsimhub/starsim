# The naive user (review 4)

Review agent RN, 2026-10-19, on branch `v4`. This is review 4 of [next-reviews.md](../next-reviews.md), run at a small scale: four simulated newcomers, 16 tasks, and 36 attempts at a task.

## Summary

**Tasks.** 16 modeling tasks, each stated in epidemiological terms with a hidden reference answer computed without Starsim (SciPy and NumPy only; [reference/](reference/)):
- Set 1, T1 to T10 ([benchmark/tasks.md](benchmark/tasks.md)): SIR final size, SEIR peak day, SIRS equilibrium, births and deaths, two age groups with a contact matrix, an all-or-nothing vaccine, a school closure, estimating R0 from weekly cases, the probability of a major outbreak, and weekly rates at a daily step.
- Set 2, H1 to H6 ([benchmark/tasks2.md](benchmark/tasks2.md)): an age-targeted campaign on a date, a case fatality ratio, the calendar date of peak incidence, a 90% interval from stochastic runs, and two traps from set 1 again.

An expert solution for each task ([expert/](expert/)) confirmed that v4 can reach every reference answer.

**Personas.** These were subagents that saw only what a newcomer has, never the source or the spec:
- **The careful novice:** all of `docs/`.
- **The assistant:** `llms.txt` only.
- **The Starsim 3 user:** the two migration pages and their v3 habits.
- **The habitual user:** `ten_minutes.qmd` once, then guesses.

All of them could also use docstrings and the messages Starsim printed.

**Success rates.** The final answers were right in every case:

| Persona | Tasks | Right | Wrong | Gave up | Silent wrong first attempts |
|---|---|---|---|---|---|
| Careful novice | T1–T10 | 10 | 0 | 0 | 0 |
| Assistant | T1–T10 | 10 | 0 | 0 | 0 (one stochastic run where a deterministic one was asked, which happened to be within tolerance) |
| Starsim 3 user | T1–T10 | 10 | 0 | 0 | 2 (T2 day 72 against 79; T7 785,847 against 714,767, +10%) |
| Habitual user | H1–H6 | 6 | 0 | 0 | 2 (H6 370,945 against 390,956, −5%; H4 `[0.8425, 0.8425]` against `[0.78, 0.81]`) |
| **All** | 36 | **36 (100%)** | 0 | 0 | **4 of 36 (11%)** |

Scoring is automatic: `python benchmark/score.py transcripts/*/answers.json`. Starsim raised almost no exceptions. Every wrong number came from code that ran cleanly. The four silent wrong first attempts were caught only because:
- the persona doubted the task's wording (H6);
- the number was obviously implausible (H4);
- the persona tried a second formulation (T2, T7).

A user who trusts a clean run would have reported all four. The tasks were easier than the brief's examples (no 10-million-agent task, no fit by age), so the 100% is an upper bound. The next run should add harder tasks (see [What to run next](#what-to-run-next)).

**Problems, ranked by severity.** These use the tiers in [README.md](../README.md#every-finding-ranked). "New" means not in [findings/](../findings/), [new-findings.md](../new-findings.md), or [status_oct18/](../status_oct18/).

| # | Tier | Problem | Met by | Status | Fix proposed |
|---|---|---|---|---|---|
| NR1 | 1 | `ss.SIR`, `ss.SIS`, and `ss.SEIR` with a plain duration (`dur_inf=ss.days(7)`) keep the Starsim 3 default `lognorm_ex(std=1)`, so the infectious period is nearly fixed rather than exponential. This contradicts migration item 5 and the user guide, which call these "declared diseases". T7 is 10% off, and T2 peaks 7 days early. The library diseases also still default to `p_death=0.01`. | v3 user, assistant | New (related to X11) | Code: a duration alone on a library disease is the mean of an exponential, as in `ss.Disease`, and `p_death` defaults to 0. Until then, say in migration item 5 that the library diseases are the exception, and have the note say "lognormal with std 1 day (a nearly fixed duration), not exponential". |
| NR2 | 2 | `msim.extinction()` reports "died out in 100% of N runs" for any SIR epidemic, because every SIR epidemic ends. The ODE's own note sends users to it ("msim.extinction() shows how often") to answer the question it cannot answer: was there a major outbreak? It returns `None`. | novice, assistant, v3 user (all three noticed) | New (DC30 covers `summarize()` only) | For a model with no reinfection, report "died out before X% were ever infected" (a minor outbreak) with a threshold argument, plus the share of runs that were major, and return the numbers. |
| NR3 | 2 | After `sim.run(n_runs=...)` on a compartmental sim, `msim.sims` holds one sim. Looping over it, the Starsim 3 habit, silently gives one run's value N times. | habitual | New (R2 fixed `len`, not `sims`) | Make `msim.sims` lazily give one sim per run, or raise on iterating it with a pointer to `msim.results.<disease>.<result>.values` (runs first). Put the run-axis spelling in `ten_minutes.qmd` next to `n_runs=`. |
| NR4 | 2 | `ss.Vaccine(efficacy=0.9)` is leaky by default, and nothing says so where newcomers learn vaccines. `ten_minutes.qmd` uses it without a word, and neither the migration table nor the `campaign_vx` removal message mentions `leaky=`. An all-or-nothing task comes out 5% low. | habitual, novice, v3 user | New (D40 added `leaky=False`) | One sentence in `ten_minutes.qmd` and the migration row. `describe()` marks `leaky=True` as a guess. The `campaign_vx`/`routine_vx` removal messages give the `leaky=False` spelling for Starsim 3's all-or-nothing default. |
| NR5 | 2 | `init_prev=` seeds the first state after `S`, so in an SEIR it seeds `E`, silently at `verbose=0`. With `verbose=1` the only sign is "with 0 infectious at the start". The expert's first SEIR peaked on day 83 against 79. `init=dict(I=...)` is in the migration table, the `coming_from/` pages, and the docstring, but not in `ten_minutes.qmd` or the tutorials. | expert (the personas found `init=` first) | New (L8 is about the source of infection, not seeding) | Show `init=dict(I=10)` in `ten_minutes.qmd`. When `init_prev` lands on a state that is not infectious, raise the note to a warning that names the state and the `init=` spelling. |
| NR6 | 3 | `n_agents=` silently switches `auto` to agents, printed only as "solver: abm (n_agents was given)". With `solver='ode'`, the note "n_agents=100000 is not used" is wrong, since the population still comes out at 100,000. `total_pop=` is on neither migration page. | assistant, v3 user | Known (X9, N22), plus a new doc gap | Add `total_pop=` to the migration table's population row. Have the note say "is used as total_pop". |
| NR7 | 5 | The structure features that tasks need are only on the `coming_from/` pages: named groups, `split=`, `ss.contacts(age=matrix)`, and `use_aging`. The age shares task (H1) sent the habitual user through `age_dist=` to `ss.People(age_data=)`, whose format is not documented, and that silently switched the run to agents. | novice, habitual | New (docs) | Add `split=` and a named-group `ss.contacts` to the age section of `ten_minutes.qmd`. Document the `age_data` columns. Have the `age_dist=` "not found" error suggest `stratify(age=..., split=...)`. |
| NR8 | 5 | `llms.txt` lacks what an assistant needs most. The `ss.Result` entry lists no methods (`select`, `by`, `to_series`). Nothing says `n_agents=` makes `auto` choose agents, or that `total_pop=` is the population of a compartmental run. Nothing says how to write an SIRS (there is no `ss.SIRS`) or how a `MultiSim` holds its runs. Of the assistant's six wrong turns, `llms.txt` could have prevented four (the first three of these, plus NR2). It helped catch the other two, because it shows the `lognorm_ex` and `p_death` defaults. | assistant | New (docs) | `make_api.py`: list each class's public methods with one-line summaries. Add notes to the header for `total_pop` against `n_agents`, the run axis, and "any model not in the library is `ss.Disease(...)`". |
| NR9 | 6 | The step-count warning: "start=0, stop=200, and dt=1.0, which results in 73001 timesteps", with years and days unnamed. It points to `ss.options.warn_convert`, which does not sound related, and the warning was harmless (a 2 s ODE). | novice, assistant, v3 user | New | Print `stop=years(200)` and `dt=days(1)`, and don't warn for a compartmental run that is fast. Give the option a name about steps. |
| NR10 | 6 | `UnknownParameter` for `init_infected=100` advises turning it into a transition parameter or `define_pars()`, and never mentions `init=` or `init_prev=`. | habitual | New instance of N21 | For an `init*` or `seed*` keyword, suggest `init=dict(I=100)` and `init_prev=`. |
| NR11 | 6 | "Not converged: the 3 searches ... ended at different values (0.01312, 0.9863, 0.9863)". The numbers are objective values, unlabeled, and the best is plainly the global optimum: two searches stuck where `beta` gives no epidemic. All three personas who fitted saw it and overrode it. | novice, assistant, v3 user | Related to A2/R5 (known), but the wording is new | Say "objective values" and give the `beta` each search ended at. When the best is far below the rest and on the scan's best region, say "two searches stalled; the best is likely global". |
| NR12 | 6 | "std 1 instead of 1 days" (the Erlang-stage note) cannot be parsed, and it gives the std without a unit. | assistant, v3 user | Known wording (language.md F 02), still unclear | "approximated with std 1.0 days (asked: 1.0, no unit, read as days)". |
| NR13 | 6 | Small message and labeling items: <ul><li>"Module name "vaccine" is already used, so renaming this module "vaccine_1"" appears when the delivery is assigned to a variable called `vaccine` and its unnamed product is also "vaccine" (one vaccine in the sim).</li><li>Times in notes are unrounded and have no unit ("at 124.99999984499999").</li><li>"vaccinate was tried on groups" is unclear.</li><li>A progress bar appears for a 0.14 s run.</li><li>`print(ss.data(...))` shows only an address.</li><li>With unit `'week'` (T10), `timevec` runs to 365, in days.</li><li>Daily flows are labeled at the end of their day (H3: 2025-02-23 for the day starting 2025-02-22), which is not stated.</li><li>A non-reciprocal contact matrix is accepted without a note.</li><li>A disease made inside a function is named "disease" (L20, known).</li></ul> | various | New, except L20 and the matrix (D14 validated shape, not reciprocity) | Name the product from the delivery's name (`vaccine.product`). Round times and give units in notes. Say how flows are dated in `basics_results.qmd`. Add a reciprocity note when population shares are known. |

**Known problems met again** (not counted as new):
- X9 and N22: `auto` chooses agents because `n_agents=` or `people=` was given.
- X4: binomial at `dt=1` is biased. H4 gave `[0.822, 0.844]`, and the note led to `'gillespie'`.
- L1 and L18: a bare `beta=0.05` on `ss.SIR` takes its unit from a typed duration, so it is per day rather than Starsim 3's per year. It printed R0 = 0.35, which the v3 user noticed.
- L20: a disease is named after the variable it was assigned to.
- A2 and R5: fit convergence.

What went well, by the personas' own account:
- The `birth_rate=` and `campaign_vx` removal messages led straight to the fix.
- `ss.modify(..., start=, stop=)`, `ss.outcome(0.3 * infection)` with `sim.calibrate()`, `ss.birth`/`ss.death`, and the typed units `perweek`/`weeks`/`years` all worked on the first try.
- The printed "R0 = ..., with N infectious at the start" was each persona's main check that its model was right. It is the most useful line Starsim prints.
- The "is not a level of age; the levels are 0-4, 5-14, 15+" error fixed itself in one step.

## Measures

**First spellings.** These are evidence for the naming decisions. Each cell gives the persona's first try, and whether it worked.

| Concept | Novice | Assistant | Starsim 3 user | Habitual |
|---|---|---|---|---|
| R0 | converted to `beta = R0/D` by hand | the same | the same | the same |
| Population | `total_pop=` (works) | `n_agents=` (agents) | `n_agents=` (agents) | `total_pop=` (works) |
| Seeds | `init=dict(I=10)` (from the migration table) | `init_prev=10`, then `init=dict(I=)` (from the docstring) | `init_prev=`, then `init=` | `init_infected=100` (error, wrong advice) |
| Infectious period | `ss.days(7)` in a declared disease (exponential) | `dur_inf=ss.days(7)` on `ss.SIR` (lognormal) | `dur_inf=ss.days(7)` on `ss.SIR` (lognormal) | `ss.days(7)` (exponential) |
| All-or-nothing vaccine | `leaky=False` (found in `modules_products.qmd`) | `leaky=False` (docstring) | `ss.campaign_vx` (removed; message helped) | `efficacy=0.9` alone (leaky, 5% off) |
| Age shares | `stratify(..., split=)` (from `coming_from/`) | `stratify(group=..., split=)` | `stratify` | `age_dist=` (error), then `People(age_data=)` |
| Results by group | `select(age=)` | found `select` via `dir()` | `select` | `select(age='0-5')` (error, fixed by its message) |
| Many runs | `sim.run(n_runs=)` | `ss.parallel(sim, n_runs=)` | `sim.run(n_runs=)` | `sim.run(n_runs=)`, then a loop over `msim.sims` (silently one run) |
| Probability of an outbreak | `msim.extinction()` (100%) | `msim.extinction()` (100%) | `msim.extinction()` (100%) | not asked |

The pattern: users reach for R0, the infected count, and `n_agents` first. Nobody found a way to give R0 directly, and all four computed "ever infected" as `1 - n_S[-1]/N`, since a declared disease has no cumulative result.

**Errors met.** Five errors in all:
- `IncompatibleTypes` (expert only).
- `UnknownParameter` for `init_infected` (its advice misled).
- `KeyNotFoundError` for `age_dist` (helped indirectly).
- The level error for `select` (helped).
- The two removal errors (helped).

Every other wrong turn was silent or came only as a note.

**Time and tool calls** (subagent totals, for all tasks):

| Persona | Tool calls | Minutes |
|---|---|---|
| Novice | 25 | 3.8 |
| Assistant | 31 | 5.3 |
| Starsim 3 user | 34 | 6.1 |
| Habitual | 24 | 3.7 |

## Method

1. Wrote the tasks, then the hidden references with SciPy and NumPy:
    - [reference/reference.py](reference/reference.py) and [reference/reference2.py](reference/reference2.py), with outputs in `answers.txt` and `answers2.txt`;
    - the data for T8, generated by an SIR ODE with Poisson noise (`t8_weekly_cases.csv`).
2. Wrote expert v4 solutions to confirm that each reference is reachable, and found NR2 and NR5 on the way ([expert/](expert/)).
3. Generated `llms.txt` into a scratch folder (`docs/` was not touched).
4. Ran the four personas as subagents, at most two at a time. Each was told what it could read, and told to log each first spelling, error, and note.
5. Scored the answers ([benchmark/score.py](benchmark/score.py)). Reproduced each problem the personas reported in a minimal script ([repros/](repros/), with `.out` files), and checked it against the registers.
6. Wrote one strict-`xfail` test per confirmed code problem in [tests/test_naive.py](tests/test_naive.py): NR1, NR2, NR3, NR5, NR9, and NR10. Each fails today for the reason stated (checked with `--runxfail`).

These tests live here rather than in `tests/`, since this review was read-only on `tests/`. They should move to `tests/` when the fixes land.

Limits:
- Subagents are not people: they read carefully, and they checked their answers against formulas that a public-health analyst might not know. The silent wrong first attempts are the more telling measure.
- One run per persona, so the rates have wide error bars.
- The personas could not read the rendered docs, only the `.qmd` sources.

## What to run next

Rerun the benchmark after each docs phase. To make it discriminate, add tasks that today's set lacks:
- a fit to hospitalizations by age;
- a scenario with 90% intervals over a fitted posterior;
- 10 million people (as agents, with a time budget);
- weekly data given as new infections;
- a Starsim 3 `beta` dict with networks;
- a seasonal `beta` with a fitted multiplier.

Also run the habitual and assistant personas on set 1, and the novice on set 2. Each of those cells was run once or not at all.

## Files

- [benchmark/](benchmark/): the task sheets, the T8 data, and `score.py`.
- [reference/](reference/): the hidden answers (SciPy and NumPy only).
- [expert/](expert/): expert v4 solutions.
- [transcripts/](transcripts/): each persona's `log.md` (its account of every wrong turn), its scripts, and `answers.json`. Also `first_attempts.json` for the silent wrong first attempts, and `h*_first.py` for the habitual user's first tries.
- [repros/](repros/): minimal repros with their output:
    - `library_sir_durations.py`: NR1;
    - `seir_library.py`: NR1 and NR5;
    - `messages.py`: NR2, NR11, NR13;
    - `messages2.py`: NR9 and NR13;
    - `assistant_first.py`: NR6 and NR12.
- [tests/test_naive.py](tests/test_naive.py): the strict xfails.
