# A branch probability that depends on the time in the state (HP18)

## Summary

**Revision, 2026-10-19.** Decision 2 is (b), per Cliff: `P.dwell` may also appear in the rate of any exit out of `P`. Decision 3 has no note and is taken as recommended (the whole-step conveyor). All four decisions are settled, and the code is under way.

**The problem.** In HPVsim, how long an infection lasts decides where it goes: the chance of CIN rises with the time spent infected (`P`), and the chance of cancer with the time spent in CIN (`C`). In Starsim 3, HPVsim draws the duration and then the outcome from it (`hpv.py:533` to `580`: `p_cin = compute_severity(dur_precin, rel_sev, cin_fn)`, then `p_cancer = compute_severity(dur_cin, ...)`). v4 has no declared way to say this. The spec's form (branches decided on entry, one dwell for all) gets the total right but the timing badly wrong: in a cohort of 10,000 women infected at 25, P(cancer) is 0.0274 against 0.0293, but the median time to cancer is 3.0 years against 24.0, and the mean age at cancer 32.1 against 53.7 (`status_oct18/hpv.md` HP18). Splitting by outcome with conditional dwells worked out outside the model gets closer (19.5 years, age 50.3) but cannot be fitted. The exact form today reads engine internals (`v4port/natural_history/hpv_declared.py`, `years_in()`: `(d.ti - d.ti_P[uids])*dt`) and runs as agents only.

**The proposal.** A state's time in it is an expression, `P.dwell`, as its count is `P.n`. In the `p=` of a transition with several destinations out of `P` (decided when the person leaves, now documented in 03) it is the time the person spent in `P`. No new `ss.` name; `ss.given_dwell` is not added.

```python
import hpvsim as hpv    # hpv.logf2() and hpv.intlogf2(): HPVsim's functions, rewritten with np.maximum in place of indexing, so they take an expression as they take an array
S, P, C, X, R = ss.states('S P C X R')
k_cin, k_ca, tp = ss.par(0.3, bounds=[0.1, 0.5]), ss.par(0.3), ss.par(2e-3)

hpv16 = ss.Disease(S, P, C, X, R,
    infection = (S | R) >> P @ beta*(P + C),
    precin    = ss.progression('P -> C | R', dur=ss.lognorm_ex(ss.years(3), ss.years(9)),
                               p=[hpv.logf2(P.dwell, k=k_cin, ttc=50), None]),                          # P(CIN) given the years spent in P
    cin       = ss.progression('C -> X | R', dur=ss.lognorm_ex(ss.years(5), ss.years(20)),
                               p=[1 - (1 - tp)**(hpv.intlogf2(C.dwell, k=k_ca, ttc=50)**2), None]),     # P(cancer) given the years in CIN
)
```

`k_cin` is then a parameter that `sim.calibrate()` can fit, and the model runs under every solver (agents exactly; compartments staged by time in state, exact to the step).

**Decisions for Cliff.**

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | The spelling | (a) `P.dwell`, an expression of the time in the state, usable in `p=` (and any expression); (b) `ss.given_dwell(fn)`, a wrapper on a Python function of the dwell; (c) a `kind='clock'` dimension set to 0 on entering `P`, with `p=ss.by(since_P=...)`; (d) stages declared by the user (`P1 -> P2 -> ...`, a `p` per stage) | (a). It is one attribute next to `.n` and `.frac`, it is an expression (so `to_spec()` writes it as text, `k_cin` can be fitted, and gradients flow), it composes with people's properties (`ss.People.rel_sev*P.dwell`) and `ss.by`, and the compartmental form is worked out from it. (b) adds a name and an opaque function (agents only in practice, unfittable); (c) needs a `stratify()` and a `move=` on every way into `P` (infection, reactivation), and a table; (d) is approximate (the split form above is 4.5 years off) and verbose [CK: ok] |
| 2 | Where `P.dwell` may appear | (a) only in `p=` of a transition with several destinations out of `P`; (b) also in the rate of any exit out of `P` (a hazard that depends on time in state); (c) anywhere | **Decided: (b).** In `p=` of a transition with several destinations out of `P`, and in the rate of any exit out of `P` (a hazard that depends on the time in `P`: under compartments, a hazard per slot of the same conveyor). Elsewhere it raises: separate transitions with `p=` are decided on entry, when no time has passed, so `p=P.dwell` there raises `ss.CapabilityError` saying to write one transition with several destinations; in a transmission rate or a result it raises, since it is a value per person in `P`, not a count [CK: i think we can do B] |
| 3 | Compartmental form | (a) a conveyor of whole steps in the state (the one `after=` uses), exit hazards from the dwell distribution, and `p` worked out per slot; (b) phase-type stages (Erlang or hyperexponential, HP28) with a share per stage; (c) agents only (`CapabilityError`) | (a). It is exact to the step for any dwell distribution and any `p`, and it reuses an existing conveyor; (b) is approximate because a stage does not know the time in state. Cost: one compartment per step up to a cap (below) (taken as the recommendation; no `[CK:]` note) |
| 4 | Agents: the time actually spent, or the dwell drawn | (a) the time actually spent, in whole steps (`ti - ti_P`, plus half a step); (b) the dwell drawn, unrounded | (a). It is what the compartmental form gives too, it follows a dwell rescaled by a modifier (C17), and it differs from HPVsim's drawn value by under half a step (0.125 years at HPVsim's quarterly step) [CK: ok] |

## Use cases

**HP18 (HPVsim): CIN and cancer given the duration.** Today in Starsim 3: `set_prognoses()`, about 110 lines (`hpv.py:474` to `590`), draws `dur_precin` (shortened by `sev_imm`), `p_cin` from it (scaled by `rel_sev`), the CIN branch, `dur_cin` (lengthened by `age_risk`), `p_cancer` from it, and schedules every time by hand with stochastic rounding. In the v4 port today: three forms, of which only the agents-only `exact` reads the engine's `ti_P` (`hpv_declared.py`), and the `split` form needs about 20 lines of moments by Monte Carlo (`_moments()`). Under the proposal: `pc = hpv.logf2(ss.People.rel_sev*P.dwell, k=k_cin, ttc=50)` and `p=[pc, (1 - pc)*control, None]` (HPVsim's latency share), with `sev_imm` and `age_risk` as `ss.modify()` on the dwell (they stay as they are: the dwell they lengthen or shorten is the time `P.dwell` reads, as in HPVsim, where `p_cin` is computed from the shortened duration). The port deletes `years_in()`, `p_cin_exact()`, `p_cancer_exact()`, `_moments()`, and the `naive` and `split` forms. Simpler than Starsim 3: yes, by about 90 lines, and declared.

With `rel_sev` as a continuous property of people, the model runs as agents (as `behavior_vocabulary.md` decision 5 says for any value by a property that is not a dimension); without it, or with `rel_sev` binned as a dimension, it runs under every solver.

**Elsewhere.** The same shape is a dwell-dependent case fatality (longer in hospital, more likely to die), TB's progression by time since infection (with decision 2 (b)), and STIsim's syphilis stages by duration; none is blocking now.

## Semantics

**Definition.** For a person leaving `P` by a transition with several destinations, `P.dwell` is the time since they entered `P` (the `ti_P` the engine already keeps), in the unit of the expression's other durations (years, by the units rule of 02); a function of it is evaluated then, per person. Re-entering `P` starts it again.

| Solver | What happens |
|---|---|
| agents | In `finish_moves()`, `branch_probs_for()` evaluates `p` for the leavers with `P.dwell = (ti - ti_P + 0.5)*dt` (the dwell to the middle of the step of the exit; the exit is at the step's start, so this matches the conveyor's slot). Exact. Cost: one subtraction per leaver |
| ode, binomial | `P` becomes a conveyor of slots, slot `j` holding those who have been in `P` `j` whole steps (the machinery of `stays()`), up to a cap `K`, then one open slot. From slot `j` the exit hazard is the dwell distribution's discrete hazard `(F(j+1) - F(j))/(1 - F(j))`, and the leavers split by `p` at dwell `(j + 0.5) dt`. The open slot exits at the rate matching the mean beyond `K`, with `p` at that mean. Exact to the step below the cap |
| gillespie | The same conveyor (the engine shares it; slots move at step boundaries, as for `after=`) |
| crossing, JAX | As the ODE; `p` is an expression, so its parameters have gradients |

**The cap.** `K` is the smaller of the run's length and the dwell's 99.9th percentile, in steps. For HPV16 at a quarterly step over 100 years, that is 400 slots for `P` (the 99.9th percentile of lognormal(3, 9) is about 103 years) and 400 for `C`, so about 800 compartments per age group and genotype instead of 2; with 16 age groups, about 13,000 per genotype. A note (`ss.LargeStep`'s kind of note, with the count) says so when it exceeds 10,000, and the mass beyond the cap. Coarser slots beyond some time (with `p` at each slot's mean) would cut this and are a later option, not proposed now.

**Modifiers on the dwell.** As agents, a modifier on the transition or its duration rescales what is left of the dwell (C17), and `P.dwell` is the time spent, so `sev_imm` and `age_risk` act as in HPVsim. Under compartments, a modifier constant in time within a group gives that group the dwell distribution scaled by it, which is exact; one that changes during a person's stay is applied to the hazard of the slot they are in, an approximation that a note states (as for time-varying dwells today).

**Interplay.** HP6's modifier on one destination (`ss.modify('hpv.cin.X', multiply=0.5)`) multiplies `p` after `P.dwell` is put in, and the other destinations share the rest. The remainder (`None`) is worked out per person or per slot. The destination results (`destination_results.md`) count by destination as for any `p`. HP28 (dwells with a coefficient of variation above 1 become one exponential under compartments) is fixed for every state that uses `P.dwell`, since the conveyor carries the exact distribution; for other states HP28 keeps its own (hyperexponential) fix.

**`describe()`**: `precin  P -> [C, R]  dur lognormal(3, 9) years; p decided at exit, from the time in P: [<the expression in P.dwell, written out>, rest]`, and under compartments `P: 401 compartments by time in state (quarterly to 100 years)`. **`to_spec()`**: `P.dwell` is written as `"P.dwell"` in expression text, as `W.n` is `"W.n"` (14, "Rate expressions"), so the hash depends only on what was written. **`to_code()`**: writes `P.dwell` back.

## Fit with what exists

- **Reused:** the exit-time decision of one transition with several destinations (03, as documented this week), state attributes in expressions (`.n`, `.frac`, `.total`), NumPy functions in expressions (`np.exp`, `np.log`, `np.minimum`, `**`), `ss.par` for fitting, the conveyor of `after=`, `ti_<state>`, C17's rescaling.
- **New:** one attribute of a state, `.dwell` (an expression leaf: the time in that state). It is a value for people in the state, not a count, so `P.dwell` in a rate of transmission or a result raises, saying where it can be used. The clock of `behavior_vocabulary.md` is time since an event that persists after leaving; `P.dwell` exists only while in `P` and needs no declaration, which is why it is not a clock (though under compartments both are conveyors, and the code is shared).
- **Not added:** `ss.given_dwell` (the review's proposal and the name in decisions_oct19 item 10); a Python function of the dwell. A function of `(sim, uids)` in `p=` still works as agents, and reads the time in state as `hpv.P.dwell[uids]` instead of the engine's `ti_P`.
- **Removed:** nothing public. The workaround of reading `ti_<state>` in a branch function stops being needed (it keeps working).

**Spec changes.** 03-states-and-transitions.md, the branching section: "a branch probability given the time in the state", with the rule of decision 2 and the error. 07 or 02, the units of `.dwell`. 12-backends.md: the conveyor by time in state, the cap, the note. 14-model-spec.md: `.dwell` among the attributes of a state in expressions. 17-vocabulary.md: `.dwell` in the slot table beside `.n` and `.frac` (no new counted `ss.` name, so the count stays). 18-conformance.md: the contract below.

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 1 | `State.dwell`: an expression leaf, its text and parsing, the check of where it is used (decision 2 (b): `p=` of a transition with several destinations, and the rate of any exit out of the state), the error for separate transitions | S | `transitions.py`, `expressions.py`, `spec.py`, `graph.py` |
| 2 | Agents: evaluate it in `branch_probs_for()` and in the rates of exits out of the state, from `ti_<state>`; `hpv.P.dwell[uids]` for functions | S | `solvers/agents.py`, `solvers/engine.py` |
| 3 | Compartments: the conveyor by time in state for a state whose exit reads it; hazards per slot from the dwell distribution (by group, with constant modifiers), and an exit rate that reads `.dwell` evaluated per slot; `p` per slot in `update_targets()`; the cap and note | M | `solvers/compartmental.py`, `kernels.py`, `solvers/crossing.py` |
| 4 | `describe()`, `to_code()`, docs (a section in the diseases guide with the HPV example), CHANGELOG | S | `presentation.py`, `docs/user_guide/modules_diseases.qmd` |

About four days. **Tests:** the HP18 cohort (10,000 women infected at 25, `P -> [C, R]` and `C -> [X, R]` with HPV16's dwells and functions, no `rel_sev`): median time to cancer within 1 year of the Starsim 3 code's 24.0 and P(cancer) within 3 standard errors of 0.0293 as agents; a contract across agents (20 runs), binomial (20 runs), and ODE at a quarterly step on P(CIN), P(cancer) and the mean time to cancer, within 2% (ODE) and 3 standard errors (stochastic); a case with a known answer (exponential dwell with rate 1, `p = 1 - exp(-P.dwell)`, so the share to C is 1/2), within 1% under ODE; `p=P.dwell` on separate transitions raises; a fit of `k_ca` from cancer counts recovers it (gradients through the ODE).

**What the ports delete.** HPVsim: the functions and forms named in the use case (about 60 lines of `hpv_declared.py`) and, at the end, most of `set_prognoses()`'s scheduling in favor of the declared disease.

## Out of scope and dependencies

- **Per-person values written on events** (HP19: `rel_sev`, immunity as a running maximum) belong to the cascade and behavior designs; this design only reads people's properties, as `ss.People.<property>` (or `ss.ppl`, per Cliff's note in `behavior_vocabulary.md`).
- **HP28 for states that do not read `.dwell`**: its own fix.
- **HP20** (`destination_results.md`): independent, but HPV needs both, since this is why HPV's cancer transition is one transition.
- **Family of diseases**: members run side by side under compartments (`disease_family.md` decision 3), each with its own conveyors.
