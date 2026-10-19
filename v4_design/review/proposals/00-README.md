# v4 proposals: index and reconciliation

| # | File | Covers |
|---|---|---|
| 01 | `01-time-and-within-host-state.md` | clocks (`X.time`, `(A\|B).time`, `.progress`), `ss.attr`, events and hooks, scheduling exits at entry, `ss.spread` |
| 02 | `02-programs-and-treatment.md` | undriven edges, `ss.Treatment(moves=, on_fail=)`, products across several diseases, multi-outcome tests, `.since`/`.n`/`.held`, `repeat=`, `dose=`, `target=`/`remove=`, `priority=`, `queue=`, `ss.Capacity`, `then=`, numbers are people |
| 03 | `03-transition-timing-and-structure.md` | split-at-exit rule, the rest stay, `ss.on`, undriven edges, the conveyor, `after=`/`before=`, Dists with `ss.by`, `ss.mother`, `preg.*` events, `ss.people`, same-step cascade |
| 04 | `04-pregnancy-and-births.md` | declared `ss.Pregnancy` (`NP`, `P` as a gestation conveyor), conception and its modifiers, losses by GA, abortion, twins, `ss.birth(per=, n=)`, fetus as agent or fetal level, `ss.child`, `since_birth`, parity, `ss.Lactation`/LAM, maternal and infant death, `ss.LegacyPregnancy` |

## Shared mechanism

All three proposals need one engine primitive: a **conveyor**, a staged dimension advanced one level per step.
- 01 uses it for clocks.
- 02 uses it for `since_`, `waiting_` and `n_`.
- 03 uses it for fixed delays, `stages='step'` and `after=`.

Products' `waning=` levels and the test `pending_` levels are early versions of it. Build it once, in the engine, and use it for all of these.

## Conflicts resolved here (edit the proposals to match)

| Conflict | Resolution |
|---|---|
| 01 uses `decide='exit'`; 03 rejects it | **03.** A single transition with several destinations splits when it is taken, and its `p=` reads `X.time`. 01 §7.5 should drop `decide='exit'`. |
| 01 §7.1 uses `when=` on a transition; 03 rejects `when=` | **03.** Write `to_late = EL >> LL @ ss.on(syph.infected.time >= ss.years(2))`. |
| Error code `E212` means ClockOutsideDomain in 01 and TriggerLoop in 03 | `E212 ClockOutsideDomain`, `E214 TriggerLoop`. |
| Warning for an unused undriven edge: `W207` (02) or `W209` (03) | `W209 UndrivenTransition`. `W207` stays UnreachableState. |
| 03 refers to `ss.Treatment(to=)`; 02 defines `moves=` | **02:** `moves=`. |
| Lazy handle on people: `ppl` (01, 02) or `ss.people` (03) | **`ss.people`** (03). No dummy `ss.People(1)`. |
| Union across diseases: `ss.any(...)` (02) or `functools.reduce` plus `ss.state('hpv18.X')` (03) | Both. `ss.any(iterable)` is a reduction with `\|`. `ss.state('mod.X')` is a lazy path reference, for factories that build one module before its siblings exist. |
| Owner of "who went through X this step" | 01: transitions, `tr.to(B)`, `X.enter` and `X.exit` are events, usable as conditions, `.uids` and `on=`. 02 and 03 use these. |
| Exits scheduled at entry, and moves in the same step | One change, specified in 01 §4 together with 03 §3 (the cascade). Implement them together. |

## Decisions for Robyn and Cliff

The proposals list ~35 open questions in total. These are the ones that change the language. Everything else can be settled during implementation.

1. **Implicit "the rest stay"** (03 Q1). If branch probabilities sum to less than 1, should the remainder be guessed and printed, or must the model use `ss.stay`?
2. **Routine delivery default** (02 §2.4). Change the default from once per life to once per eligibility spell? This is a behaviour change for existing v4 code.
3. **Numbers are people** (02 rule 2). Capacity, targets and results read during a run would all be in people.
4. **Stock-target timing in an ODE** (02 Q1). Should a target be an impulse at each time point (which agrees with agents) or a tracking flow (smooth, differentiable) by default?
5. **Clock realization** (01 Q1–2). A conveyor (exact on the step grid, but stepped) or Erlang aging (smooth, but blurs elapsed time)? And should a staged dwell reuse its stages as the clock?
6. **`ss.attr` naming and scope** (01 Q5, Q9). Is one constructor for drawn, recorded, trait and derived values right, or should there be two?
7. **Names**:
   - `ss.on`, `ss.stay`, `ss.rest`, `moves=`, `repeat=`, `then=`
   - `ss.share`, `ss.desc`, `.progress`
   - `stages='step'`
8. **Compartmental scope** (01 §7.5, 02 Q9, 03 §4.5). Crossing HPV genotypes is 8⁴ joint states per group. Should we accept that HPVsim's compartmental version is single-genotype, or genotypes without cross-immunity? Or is a cheaper approximation for crossed diseases worth building?

## FPsim gaps not covered by 01–03

These come from `../fpsim/NOTES.md`. Items the proposals already cover are left out: clocks for time since birth, the split-at-exit rule for method choice, and `ss.on` for postpartum decisions.

| Gap | Proposed owner |
|---|---|
| A declared `ss.Pregnancy` (states, a `deliver` transition with a birth side effect) that FPsim and STIsim extend; v4's is the v3 class and is invisible to the declared world | 03 §2.9, which already flags it (Q10); needs its own proposal |
| Births driven by a flow, with twins and a parent link: `ss.birth(per=preg.deliver, n=..., into=...)` | as above |
| Per-woman counters and histories (number of births, age at first birth, birth intervals) | 01: `ss.attr(..., combine='sum'/'first')`; intervals are a clock on `deliver` |
| Branch modifiers that renormalize within a subset (method mix without changing uptake), logit-scale shifts, `set=` on branches | 03 §2.1 modifiers; new |
| Introducing a method mid-run | 02; new |
| Compartmental memory: dense `n_groups²` `dim_map`s (`np.kron`, `solvers.py:714`) run out of memory when modules are crossed | **implementation bug**, use sparse maps |
| Heavy-tailed dwells (CV > 1): Erlang with k=1 matches the mean but not the median | 03 §2.6; add a 2-phase hyperexponential |
| An event with no state change (abortion as an outcome, not a self-loop) | 03; `ss.outcome` on a branch of a transition |

## Suggested implementation order (after the v4-rs bug fixes)

Each step unblocks ports.

1. Exits scheduled at entry and the same-step cascade, together with the split-at-exit spec change and immediate branches (03 §1–3, 01 §4).
2. Undriven edges, `ss.Treatment(moves=, on_fail=)`, and the cured/failed/overtreated results (02 §2.1).
3. Events made public, `X.time` and `(A|B).time` for agents (01 §2, §4).
4. The conveyor primitive, then clocks, `since`, `queue` and fixed delays in compartmental runs.
5. `ss.on`, `ss.stay`, `ss.people`, and Dists with `ss.by` (03 §2.2, 2.3, 2.8).
6. `priority=`, `queue=`, `target=`/`remove=`, and `ss.Capacity` (02 §2.5–2.7).
7. `ss.attr` (01 §3), then `ss.spread` (01 §5).
8. Products across several diseases and multi-outcome tests (02 §2.2–2.3).
9. `ss.mother` and `preg.*`, which need a declared pregnancy for compartmental runs (03 §2.9).
