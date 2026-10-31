# Review 3: the textbook canon

Review agent RC, 2026-10-19, against the `ssv4` working tree (branch `v4`, HEAD `445b0e6d` plus uncommitted changes), with `/software/conda/envs/v4/bin/python`. This is review 3 of [next-reviews.md](../next-reviews.md). Nothing in `starsim/`, `tests/`, `docs/`, or the spec was changed.

## Summary

**What the canon covers.** 42 models: every program of Keeling and Rohani (2008), chapters 2 to 8, that the method in next-reviews.md names (simple models, risk and age structure, multiple strains and hosts, vectors, seasonal forcing and chaos, stochastic models and critical community size, metapopulations, lattices and networks, and control), plus the model files of Vynnycky and White (2010), chapters 2 to 8 (SEIR, herd immunity, measles dynamics, maternal immunity, Reed-Frost and Greenwood, WAIFW, and an STI core group). It includes the exact 2^k three-strain history model that D13 lacks, and a two-host and a host-vector model.

**Caveat on the sources.** Neither book's program pages could be reached from this machine (no network route to the Warwick or book sites), so the inventory and parameter values are from the reviewers' knowledge of the books, and the reference for each model is the book's equations integrated in SciPy (DOP853, `rtol=1e-12`), or an exact numpy simulation for the stochastic ones. The lines of the original programs were therefore not counted. The v4 models are 1 to 13 lines each (median about 6), against about the same number of lines for the SciPy right-hand side alone.

**What passes.** v4 expresses 40 of the 42, and 38 agree with the reference:

| Result | Count | Models |
|---|---|---|
| Pass at the default settings (1e-6 for an ODE, a distributional test for a stochastic model) | 17 | KR 2.2, 2.3, 2.6, 2.7, 3.3, 5.1 (SIR and chaotic SEIR), 5.3, 6.3 (`gillespie`), 6.5, 6.6, 8.1, 8.2, 8.3, 8.5; VW 3, 8.1 |
| Pass only with more ODE substeps (CR1) | 19 | KR 2.1, 2.4, 2.5, 3.1, 3.2, 4.1, 4.x (2^k), 4.3, 4.4, 5.2, 7.1, 7.2, 7.4, 8.4; VW 2.1, 4.1, 5.1, 5.2, 7.1 |
| Pass with a workaround | 2 | KR 7.3 (only the fast-commuting limit), 7.6 (agents need a quarter-day step, X4) |
| Fail | 2 | VW 6.1 Reed-Frost, VW 6.2 Greenwood (CR4, L14) |
| Not expressible | 2 | KR 6.1 and 6.2 (an ODE with noise), 6.4 (explicit tau-leaping), both gone by decision (X17) |

Several of the passing models also needed a workaround in a second spelling (KR 3.1 as four states needs `people=True`, CR3; KR 3.3 stratified needs `R` given explicitly, CR2; KR 4.1 with `kind='strain'` runs as agents and needs a small step). Every model ran under the other solvers; apart from the known whole-step bias (X4) and expected fade-outs below the critical community size, `binomial`, `gillespie`, and `abm` at 1e4 agents agree with the ODE (`out/solvers.txt`, and the chapter outputs). `auto` chose `ode` for every model with a compartmental form, `abm` for the network (7.6), contact tracing (8.5), and strain-immunity (4.1, stratified) models, which is what the spec says.

`describe()` reads like the book for most models: it prints `rate = 1.4247 × I.n / N [homogeneous mixing]`, inflows as totals, and the people-or-count guesses. Where it does not, the cases are CR5, CR6, and the tier 6 items below.

**No new tier 1 problem.** The most serious new problems are CR1 (the default ODE is 2.5% off on the book's first program) and CR2 (a given initial value of the first state is discarded), both tier 2.

## Failures, ranked

Tiers are those of [README.md](../README.md#every-finding-ranked). "Known" means the problem is already in `findings/`, `new-findings.md`, `status_oct18/`, or the fuzz and stats reviews; those are listed for the new evidence the canon gives, and are not counted as new.

| Rank | ID | Tier | New or known | Finding | Repro |
|---|---|---|---|---|---|
| 1 | CR7 | 1 | known: X4, with a new detail | At the book's daily step, agents and chain binomial miss the ODE in opposite directions on a closed SIR in 10,000 people (KR 6.3): agents peak 25% low and 9 days late, binomial 23% high, so they also disagree with each other, against O's "agree at every `dt`". On a 4-regular network (KR 7.6) agents peak 7.5% low and 4 days late; the two-strain model (KR 4.1) is off by about 2 times in `n_S` at day 30. The W102 note prints but gives no size. | `repros/kr6_abm_dt_bias.py`, `repros/kr7_network_daily_step.py` |
| 2 | CR1 | 2 | new | The default ODE substep rule (`kernels.py` about line 2106: "no compartment turns over by more than a quarter in one") looks only at per-capita outflow rates, so a compartment that grows fast through its inflow gets one RK4 step a day: KR 2.1 and 2.5 are 2.5% and 2.3% off the exact solution, and 19 of the 42 models miss 1e-6 (1e-5 to 1e-2) until `substeps=16` or more. Nothing says so. The stats review's A1 is the same kind of cap in the JAX density, not this one. | `repros/kr2_rk4_default_substeps.py`; `tests/test_canon_xfail.py::test_ode_default_substeps_kr21` |
| 3 | CR2 | 2 | new | `init=dict(S=0.1, I=1e-4)`, the textbook start of KR 2.2 and 3.3 (and of any model started near equilibrium), puts the rest of the population in `S` anyway, so `S` starts at 9,999 of 10,000 instead of 1,000, silently. `describe()` prints "init: S = 0.1, I = 0.0001; the rest in S". Giving `R` explicitly works around it. | `repros/kr3_init_rest_in_given_state.py`; `test_init_of_first_state_is_kept` |
| 4 | CR4 | 2 | known: L14 (needs Cliff), now under every solver | A seeded case with a fixed stay of one step (`I >> R @ ss.constant(ss.days(1))`) stays two steps under `ode`, `binomial`, and `abm` (L14 had it for agents and binomial). In the Reed-Frost and Greenwood models the index case is then infectious for two generations: the chance of no further case is 0.013 against the exact 0.131, and the mean final size 76.5 against 63.2 (chi-square p of about 1e-13). | `models/vw_stoch.py`, `out/vw_stoch.txt`; `test_seeded_fixed_stay_lasts_one_step` |
| 5 | CR3 | 2 | known: L4 (and C's D4) | KR 3.1 written as the book's four equations (`SH, IH, SL, IL`, with no transition between the risk groups) makes the low-risk states a count and puts the rest of the population in `SH`: 9,000 high-risk infected on day 30 against 1,003, with only a note. `people=True` fixes it. | `repros/kr3_disconnected_groups.py`; `test_disconnected_risk_groups_are_people` |
| 6 | | 2 | known: D13 | The canon's two-strain model with partial cross-immunity as `kind='strain'` under `ode` ("keeps the last strain") ends with 16 susceptible against the exact 77 (closed, 10,000 people). `auto` avoids it by running agents, which agree with the exact 2^k model at `dt=0.1`. The exact history model can be written by hand as states (KR 4.x, 8 histories and 12 infected states, generated in a loop), and passes. | `models/kr_ch4.py` |
| 7 | CR8 | 3 | known: by decision (X17, stats A3) | KR 6.1 and 6.2 (an ODE with additive or scaled noise) and 6.4 (explicit tau-leaping) cannot be written: there is no SDE solver or noise keyword, and `solver='tau'` raises. Demographic noise is available through `gillespie`; additive noise only through a coded module. A teacher of chapter 6 meets this. | `repros/kr6_no_sde.py`, `repros/kr6_tau_bias.py` |
| 8 | CR9 | 6 | new | Gillespie's automatic tau-leaping (at 1e7 people) is biased by +0.74% in peak (about 4 sd) and +0.21% in final size (about 6 sd) against `approx=False`, and its note does not say so. | `repros/kr6_tau_bias.py` |
| 9 | CR10 | 6 | new (spec drift) | The `ss.Sim` docstring (`sim.py:41`, and `status_oct18/stats.md`) says Gillespie switches above about a million events; the code switches when the estimated exact run time exceeds 1 s, so 1.6 million events in 1e6 people stay exact. | `repros/kr6_tau_bias.py` |
| 10 | CR5 | 6 | new | `describe()` prints `ss.birth(rate=ss.peryear(0.02))` in a daily sim as "inflow, in total per day; rate = 0.02 × (S + I + R).n", a per-year number under a per-day label. The run is right (0.548 births a day in 10,000). | `repros/kr3_describe_birth_unit.py`; `test_describe_birth_rate_in_sim_unit` |
| 11 | CR6 | 6 | new | With a contact matrix, `ss.birth()` gets the D17 note "uses (S + I + R).n literally, so contacts does not apply", about a rate the user never wrote. | `repros/kr3_describe_birth_unit.py`; `test_no_literal_note_on_builtin_births` |
| 12 | CR11 | 6 | new | Contact tracing under `solver='gillespie'` is refused (correctly), with a message that names `solver="ode"`; under `binomial` it names the right solver. | `repros/kr8_capability_message.py`; `tests/test_canon_kr78.py` |
| 13 | CR12 | 6 | new | `describe()` prints `ss.seasonal` as `seasonal(mean=1.3076923076923077, ...)` with no unit and every digit, a term-time function as `<lambda>`, a `ss.Mobility` as `Mobility(matrix, kind='force')` without the matrix, and a pulse delivery without its coverage or interval. | `repros/kr5_describe_seasonal.py` |
| 14 | CR13 | 6 | new (friction) | The book's 60-year and 100-year runs at a daily step (KR 2.2, 2.3, 2.6, 2.7) each print a `RuntimeWarning` that 21,901 steps is "above the recommended maximum of 20000", which is noise for an ODE that runs them in a fraction of a second. KR 2.1 prints "R0 = 9.97, with 0 infectious at the start" when 1e-6 are infectious. | `models/kr_ch2.py`, `out/kr_ch2.txt` |
| 15 | CR14 | 6 | new (gap) | The book's commuter model (KR 7.3), with finite rates of leaving and returning home, has no direct form; v4 has only the fast-commuting limit (within 0.4% of the explicit model at a leaving rate of 0.5 a day), or a by-hand home-by-current-place stratification. | `models/kr_ch7.py` |

Also seen, and not counted: the `ss.by` share in `init` is a share of each group for `risk=` but behaves differently for a strain dimension (`I=ss.by(strain=[0.01, 0.01])` seeds 0.5% in each), which needs a check against the spec; and the network models needed `ss.StaticNet` over a networkx regular graph, since `ss.RandomNet` gives Poisson-like degrees rather than the book's fixed four.

## Per model

"Default" and "fine" are the largest error relative to the reference's peak, at the default substeps and at `substeps=64`; a blank fine means the default already passed. Lines are the model's own lines in v4.

| Model | Shows | Lines | `auto` | Default | Fine | Other solvers | Result |
|---|---|---|---|---|---|---|---|
| KR 2.1 SIR | epidemic curve | 3 | ode | 2.5e-2 | 3.6e-9 | seeded 1e-6 of 1e4 rounds to 0 (noted) | pass with substeps (CR1) |
| KR 2.2 SIR with demography | damped oscillations | 6 | ode | 2.1e-10 | | | pass |
| KR 2.3 SIR, disease mortality, frequency-dependent | | 7 | ode | 6.2e-10 | | | pass |
| KR 2.4 SIR, disease mortality, density-dependent (`Y.n`) | | 7 | ode | 2.5e-5 | 2.0e-12 | | pass with substeps; note says density-dependent, correctly |
| KR 2.5 SIS | | 1 | ode | 2.3e-2 | 3.3e-9 | | pass with substeps (CR1) |
| KR 2.6 SEIR with demography | | 7 | ode | 4.6e-7 | | binomial, gillespie, abm run (3 years) | pass |
| KR 2.7 SIR with carriers | | 8 | ode | 1.7e-10 | | | pass |
| KR 3.1 SIS, high and low risk | core group | 3 | ode | 3.4e-5 | 6.2e-10 | gillespie -2%; binomial +84% (X4) | pass with substeps; as states needs `people=True` (CR3) |
| KR 3.2 SIS, five risk groups | proportionate mixing | 3 | ode | 1.3e-6 | 1.9e-8 | | pass with substeps |
| KR 3.3 SIR, two ages, aging | WAIFW with aging | 10 (states), 4 (stratify) | ode | 2.0e-8, 1.5e-8 | | | pass; stratified needs `R` given (CR2) |
| KR 4.1 two strains, partial cross-immunity | interference | 11 | ode | 2.7e-5 | 2.6e-12 | as `kind='strain'`: auto abm, agrees at `dt=0.1`; ode off (D13) | pass with substeps |
| KR 4.x three strains, exact 2^k | history model | 14 (a loop) | ode | 1.3e-4 | 1.1e-11 | | pass with substeps |
| KR 4.3 two host species | spillover | 7 (states), 4 (stratify) | ode | 1.2e-4 | 9e-12 | all within 10% of peak | pass with substeps |
| KR 4.4 host and vector | Ross-Macdonald | 8 | ode | 1.3e-5 | 4.8e-12 | all within 7% | pass with substeps |
| KR 5.1 SIR, sinusoidal forcing | forced oscillations | about 10 | ode | 7e-9 | | fade out from a trough, as expected | pass |
| KR 5.1 SEIR, strong forcing | chaos, multiannual | about 10 | ode | 2.5e-7 (10 years) | | | pass; diverges after 20 years, as chaos does |
| KR 5.2 SEIR, term-time forcing | biennial | about 10 | ode | 4.6e-6 | 3e-13 | | pass with substeps; as a function or a periodic `ss.timeseries` |
| KR 5.3 SIR, seasonal births | | about 10 | ode | 1.4e-9 | | | pass |
| KR 6.1 and 6.2 SIR with noise | | | | | | | not expressible (CR8) |
| KR 6.3 event-driven SIR | stochastic epidemic | | gillespie (asked) | KS p 0.83, 0.08, 0.97 | | binomial and abm fail by X4 (CR7) | pass under `gillespie` |
| KR 6.4 tau-leap SIR | | | | | | | not expressible explicitly (CR8); the automatic one is biased (CR9) |
| KR 6.5 persistence, critical community size | fade-out against size | | gillespie, binomial, abm | CCS between 1e5 and 5e5 in every solver | | | pass |
| KR 6.6 SIR with imports | reintroduction | | ode, gillespie | exact; 1.50 and 1.80 reintroductions against 1.73 | | | pass |
| KR 7.1 two patches, coupled force of infection | delayed epidemic | 3 | ode | 6.2e-3 | 8.8e-10 | all close | pass with substeps |
| KR 7.2 three patches, migration | | 3 | ode | 6.4e-3 | 9.9e-10 | all close | pass with substeps |
| KR 7.3 commuters | | 3 | ode | 4.4e-3 | 5.1e-10 | all close | pass, fast-commuting limit only (CR14) |
| KR 7.4 lattice, 5 by 5 | spatial wave | 6 | ode | 1.2e-2 | 5.3e-9 | all close | pass with substeps |
| KR 7.6 random 4-regular network, pair approximation | network SIR | 5 | abm | 7e-2 at a quarter-day step | | compartmental solvers ignore the graph, with a note | pass at a quarter day; lags at a day (CR7) |
| KR 8.1 vaccination of newborns | | 1 (+13 shared) | ode | 4.2e-10 | | binomial overshoots (X4) | pass |
| KR 8.2 constant vaccination | | 2 (+13) | ode | 2.1e-10 | | | pass |
| KR 8.3 pulse vaccination | pulses | 3 (+13) | ode | 8.6e-7 | | | pass |
| KR 8.4 isolation (SIQR) | | 9 | ode | 2.7e-3 | 2.6e-10 | gillespie matches; binomial +12%, abm +9% (X4) | pass with substeps |
| KR 8.5 contact tracing, ring vaccination | | 8 | abm | final size 1,664 against 9,956 untraced | | compartmental refuse (message CR11) | pass |
| VW 2.1 SEIR, closed | influenza-like epidemic | 3 | ode | 1.5e-6 | 1.8e-12 | binomial +57% at `dt=1` with 2-day stays (X4) | pass with substeps |
| VW 3 SEIR with prior immunity | herd immunity | 3 | ode | 5.2e-7 | | | pass |
| VW 4.1 measles SEIR, births and deaths | interepidemic period | 8 | ode | 1.3e-6 | 2.4e-11 | | pass with substeps |
| VW 5.1 MSEIR | maternal immunity | 10 | ode | 1.3e-6 | 1.6e-11 | | pass with substeps |
| VW 5.2 MSEIR, infant vaccination | | 10 | ode | 1.3e-6 | 8.2e-12 | | pass with substeps; a good note on a trough at 1 infected |
| VW 6.1 Reed-Frost | chain binomial | 4 | binomial (asked) | P(no further case) 0.013 against 0.131 | | abm the same | fail (CR4) |
| VW 6.2 Greenwood | | 4 | binomial (asked) | 0.013 against 0.132 | | | fail (CR4) |
| VW 7.1 WAIFW, two ages | who infects whom | 3 | ode | 2.8e-3 | 2.9e-10 | all within 4% | pass with substeps |
| VW 8.1 STI core group, assortativity | | 3 | ode | 3.4e-10 | | all within 3% | pass |

## What this means for the plan

- **CR1** is a one-line kind of fix with a large reach: include the growth of each compartment through its inflows (or the largest eigenvalue's estimate, `max(|dy/dt| / y)`) in the substep rule, or default to an error-controlled step. Then 36 of the 42 pass at the default settings.
- **CR2** and **CR3** are the same question as L4 (where the rest of the population goes, and who is people), and both are met in the first chapter a student writes. The fix proposed for L4 (anchor on the transmission's source, warn on a disconnected population) covers CR3; CR2 needs "the rest" to go to a state with no given value, and an error when every state is given and they do not sum to 1.
- **CR4** settles L14 for the Reed-Frost model, which every course teaches: a seeded cohort must leave on the same schedule as one that enters by a transition.
- The canon is good docs material: 38 of the 42 are short and read like the book. The gallery in `docs/` and the move of the models into `tests/corpus.py` that next-reviews.md asks for are left for whoever may edit those folders; the model files here are written to be moved (each is a builder plus a SciPy reference).

## Files

- `models/common.py`: the harness (a tight SciPy reference, scoring at the default and at 64 substeps, and the other solvers).
- `models/kr_ch2.py` to `models/kr_ch8.py`: Keeling and Rohani, by chapter. Run `python kr_ch2.py` (or with model names); `kr_ch6.py` takes `closed|noise|tau|ccs|imports`.
- `models/vw.py`, `models/vw_stoch.py`: Vynnycky and White (deterministic, and Reed-Frost and Greenwood).
- `models/solvers.py`: one model, many solvers, for chapters 2 to 4 and Vynnycky and White.
- `out/`: the outputs of those runs.
- `repros/`: one script per failure, named by chapter.
- `tests/test_canon_xfail.py` (CR1 to CR6), `tests/test_canon_kr56.py` (CR7 for agents, CR9, CR12, and two guards), `tests/test_canon_kr78.py` (CR7 on a network, CR11): strict xfail tests, to move into `tests/` once fixed. Run with `pytest -p no:xdist`.
