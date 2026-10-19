# Handoff, October 18 (second session)

Follows [handoff_oct18.md](handoff_oct18.md), whose "Waiting on Cliff" answers (`[CK:]`) set this round. Nothing since commit `f895f48f` is committed.

## This round

State of the tree after the merges: full suite (`test_*.py -n 8`, with `STARSIM_LFD` set) 392 passed, 11 skipped; `pytest switching -n 8` 40 passed, 1 skipped, 10 xfailed, 1 xpassed, with `test_siminf_scale` failing under load and passing alone. All worktrees are removed; no background jobs are left.

Three agents in worktrees under this session's scratchpad (`/tmp/claude-1000/-home-cliffk-idm-ssv4/b6c7bebd-4e33-4ab5-b7f7-a4da8b0ebaa5/scratchpad/wt/`), with rules in `wt/PREAMBLE15.md`; each keeps `wt/<name>/WORKLOG.md`:

- **PC**: item 18 as option B (exit chance `r·dt/(1+r·dt)` for exponential `dur=` and `rate=`, if not substantially slower; else revert and document), then Phase C (D-A: decide at exit).
- **PE**: Phase E (D-C: transitions out of the starting state are acquisition by default, with a documented override).
- **PM**: memory per agent below 114 bytes (handoff item 3, all approved).

Status:

- **PC: merged.** Done: C17 (a dwell already drawn follows a later change to its duration or rate: 825, 4,323, 4,755 against the ODE's 498 become 503, 517, 526), FP16, FP17, FP13, HP6 (one destination as a target, `ss.modify('hpv.precin.C', ...)`, efficacy `{'precin.C': e}`; the spelling is the agent's, for Cliff to confirm), D-A documented, `test_dwell_forms`. Speed unchanged. Targeted tests pass on main (85). CHANGELOG and plan updated. Worklog: `wt/WORKLOG_pc.md`.
  - **For Cliff, item 18:** option B (exit chance `r·dt/(1+r·dt)`) is not slower and puts a contact-mediated SIR within 0.1% of the ODE (from +4.7% for `dur=`, −6.1% for `rate=`, at a daily step), but makes rates written on counts worse: an SIR with infection `beta*I.frac` from −2.7% to +7.4% at the peak, the cholera reservoir's peak from +1% to +15%. That is beyond what was approved, so PC reverted it and documented the original bias (spec 12, user guide). Option B′, B plus count-driven rates taken from the counts after the start-of-step exits, should bring both within about 1% (M, changes every agent model with rates on counts). Patch for B: `wt/scratch-pc/item18_optionB.diff`.
  - **For Cliff, ST4:** a dwell entered by transmission shows one more step in results, but people act for exactly the dwell; the proposed fix would make them act a step early (fixed-dwell SIR peak from −0.4% to −11.6%). The same half-step issue as item 18; B′ could settle both.
  - Deferred: HP20 (results with a destination axis: design), HP18 (`ss.given_dwell`: design), MA7's agents part (with PE).
- **PE: merged** (two conflicts with PC, in `solvers/agents.py` and spec 08, resolved by hand: PC's `draw_competing()` kept, PE's `step_counts()` moved before people's transitions). Done: acquisition by D-C (the spec's bednet idiom and a vaccine of efficacy 1 stop every Ross-Macdonald infection, 7,387 to 0; `ss.treat` reaches 2,857; R0 12.42 against the analytic 12.42, was 0), the override `acquisition=True/False` on a transition (documented in spec 03 and 14 and `modules_diseases.qmd`), typed values and functions evaluated once (ODE 2.5–3.2× faster on MA27's repro), counts as agents with substeps and modifiers (mosquitoes settle at 50,000 at a monthly step instead of dying out). malariasim 284/284 before and after. Worklog: `wt/WORKLOG_pe.md`.
  - **For Cliff, the acquisition carve-out:** taken literally, "all transitions out of the starting state" makes an SIRV's `vax = S >> V` an acquisition (V counted as infected, peak 0.401 to 0.510; a vaccine slows vaccination, 3,136 to 2,347). PE applied: contact-mediated transitions are acquisitions, and so is every other non-outflow transition out of the first state unless a contact-mediated one also leaves it. No model with a transmission changes. The literal rule is one line in `Graph.resolve_acquisition()`.
  - Deferred (PE): MA7 (the conveyor's last slot is exempt from other exits; fixing it moves every ODE with a fixed delay and a competing exit, so it is Cliff's), MA28, and `prevalence`/`new_infections` for vector models (added only when a disease has a transmission).

- **PM: merged** (clean). Memory per agent at the peak (the `test_laser_memory` method, 50k against 200k agents) 116.8 to 80.6 bytes: index arrays made on first use, and `slot`, `parent`, `scale`, `rel_sus`, `rel_trans` take no memory until set (`ss.Arr(shared=True)`: a read-only view of the default until the first write; a direct write into `.raw` before that raises NumPy's read-only error); distributions keep no slots of their last draw; `ss.histogram.ppf()` by `np.interp` (SciPy's made about 88 bytes per agent of temporaries at init); 16-bit `ti_dead`/`ti_removed`; 32-bit `people.auids` (`ss.dtypes.uid`). `sim.inputs` no longer holds about 20 bytes per agent of `People` copies (not in the test's count). Results identical on seven models; speed unchanged or faster (SIR 1e6 6.5 to 5.9 s). STIsim and HPVsim (last round's migrated copies) fail the same tests before and after (13 and 12). Worklog: `wt/WORKLOG_pm.md`.
  - **For Cliff, 80 to 50 bytes:** group arrays (`patch` and `sim.groups`'s arrays as int16, one shared array when there is one stratification), about 19 to 5 bytes; dropping `people.uid` (always `arange`), 4.8; entry times such as `ti_S` only when something reads them, 2.4 each.

## Done by the coordinating session

- **`ss.MaternalNet` kept** (handoff item 2, read as "leave it in"): restored as Starsim 3 had it, a `PrenatalNet` named `maternalnet`, with a docstring saying it may become one network for prenatal and postnatal transmission. The migration script no longer renames it or the network's name (`network_rename` removed), and the skill's row and the CHANGELOG line are updated. `test_other.py` and `test_networks.py` pass. If "leave in" meant "leave it removed", revert these four files (`networks.py`, `__init__.py`, `migrate.py`, `test_other.py`, plus the skill and CHANGELOG lines).

- Merged the three worktrees, resolved PC/PE's two conflicts, added `MaternalNet` to `code_names` in `test_spec.py`, added the CHANGELOG lines, ticked the plan, and synced the six changed spec files (`design/03`, `07`, `08`, `12`, `14`, `18`) to `lingua_franca_design` (which also has older uncommitted syncs in `admin/` and `best-practices/`).

## Still open

- Phase H: four design documents are being written (D-F: design before code), in `v4_design/review_oct17/designs/`: `care_cascade.md` (h1), `disease_family.md` (h2), `per_act_transmission.md` (h3), `behavior_vocabulary.md` (h4, with dimensions over people's properties and units of bare numbers); brief at `wt/DESIGN_BRIEF.md`. All four are written; [designs/README.md](designs/README.md) lists every decision (with `[CK:]` slots), where the documents disagree, and what can start without waiting. Code waits on Cliff's answers. Agent PD7 (merged; worklog `wt/WORKLOG_pd7.md`) fixed two silent wrong answers the designs found: an expression `multiply=` frozen at its start value under the compartmental solvers, and HP7 (one product with two diseases protected nothing against the second: now raises `ss.UnknownDisease`), plus a reference to a state of a disease declared later (it silently read an empty state of its own; now refused, `ss.DuplicateDeclaration`). With the multiplier fixed, the probe's ODE `a.n_I` goes from 1,605 to 271 (agents 414; the rest is the solvers' existing difference on disease `b`). Suite after the merge: 392 passed, 11 skipped. The designs README's family decision 7 is done.
- Smaller: no fuzz tests in the repo unless small and fast (Cliff). malariasim is at `/home/cliffk/idm/ngm/starsim-ngm/malariasim`.

## How to resume

Keep the machine's load under half (10 cores) across all agents and test runs (Cliff, 2026-10-19): about `-n 3` per agent with three running, `-n 8` for the coordinator's suite when no agent is running.

If an agent's worktree is unmerged: copy its `WORKLOG.md` out, run `python wt/merge.py wt/<name>` from the scratchpad (deletions are not applied; do them by hand), clear `starsim/__pycache__/*.nb*` after kernel changes, run the suite and switching, add the worklog's CHANGELOG lines, tick the plan, then `git worktree remove --force`.
