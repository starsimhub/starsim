---
name: v4.review
description: Review one Starsim-family model (covasim, fpsim, hpvsim, malariasim, stisim, or tbsim) on Starsim v4 against its Starsim 3 version, for the targets in v4_design/targets.md (fewer lines, faster as ABM, no hacks, no single-model code in v4, and the model's compartmental goal), and write one report to v4_design/<date>/reviews/<model>.md. Use when Cliff runs /v4.review <model>; one model per session.
---

# Starsim-family model review

Arguments: the model name, one of `covasim`, `fpsim`, `hpvsim`, `malariasim`, `stisim`, `tbsim`. If none is given or it isn't one of these, ask which. Review only that model; the others are run in their own sessions.

Checks the "Starsim-family models" targets in `v4_design/targets.md` for this model. Read [../README.md](../README.md) first: its rules (snapshot, line counts, timing, subagents, report shape) apply here. Like the earlier port reviews, this is a test of **v4**: what matters is what v4 should change so that each target is met, not the model's own bugs (note those briefly, at the end).

## The models

| Model | Repo | Starsim 3 version | v4 version | Earlier reviews (in `v4_design/`) | Compartmental goal |
|---|---|---|---|---|---|
| covasim | `/home/cliffk/idm/covasim` | `main` (Covasim 4.0.1, on Starsim 3.7.2) | `v4` branch, and the prototypes in `v4port/` | `oct21/covasim_review.md` | Works, except with networks, including with strains |
| fpsim | `/home/cliffk/idm/fpsim` | `main` | `v4` branch, and `v4port/` | `oct17/fpsim_review.md`, `oct21/fpsim_recheck.md` | Not required; report whether it works or how close it is |
| hpvsim | `/home/cliffk/idm/hpvsim` | `main` | `v4` branch, and `v4port/` | `oct17/hpvsim_review.md`, `oct21/hpvsim_recheck.md` | Works, except with networks, including with strains (genotypes) |
| malariasim | `/home/cliffk/idm/ngm/starsim-ngm` | `main` | `v4` branch, and `v4port/` | `oct17/malaria_review.md`, `oct21/malaria_recheck.md` | Unified compartmental and ABM representations, for both mosquitoes and humans |
| stisim | `/home/cliffk/idm/stisim` | `main` | `v4` branch, and `v4port/` | `oct17/stisim_review.md`, `oct21/stisim_port.md` | Compartmental works for each disease alone; networks and some combinations of diseases may not be expressible |
| tbsim | `/home/cliffk/idm/tbsim` | `main` | `v1.0.0` branch (the first real port to v4) | `oct21/tbsim_recheck.md` | Unified compartmental and ABM representation, except for households and the like |

Check these before relying on them: `git -C <repo> branch --show-current`, `git -C <repo> log --oneline -3 <branch>`, and `git -C <repo> status --short`. The checkout is live (Cliff may be working in it), so read the branches with `git -C <repo> show` or a detached worktree in the scratchpad, never by switching branches, and never change the checkout's files. If the table is out of date (a branch renamed, a port merged), say so in the report and suggest the fix to this file.

## Steps

1. **Snapshot v4** as in the README. **Pin the model:** a detached worktree of the Starsim 3 version and one of the v4 version, both in the scratchpad (`git -C <repo> worktree add --detach <scratchpad>/<model>-v3 main`). Get Starsim 3 with `git -C /home/cliffk/idm/starsim archive <tag> starsim | tar -x -C <scratchpad>/ss3` at the version the model's `main` pins (its `pyproject.toml`), and run the v3 side with `PYTHONPATH=<scratchpad>/ss3:<scratchpad>/<model>-v3`.
2. **Find the v4 version.** Use the most complete one that exists: the v4 branch if it has commits past `main`, then the prototypes in `<repo>/v4port/` (the latest dated folder first), then the earlier reviews' declared versions. Say which parts of the model each comes from. Where a part has no v4 version, and a target can't be measured without one, write the smallest one that measures it (in the model's own vocabulary on v4, the "joyful" version the earlier reviews sketch), in `<repo>/v4port/<date>/`, and mark it as written by this review.
3. **Measure each target:**
    - **Fewer lines.** The model package on each side, by module (diseases, networks, interventions, analyzers, and the rest), by the README's rule. Where only part of the model is ported, compare those parts like for like and give the total as "ported parts only".
    - **Faster as ABM.** The model's default sim (or the benchmark its repo uses, e.g. a `tests/` timing script) at the same size and duration on each side: build and run time separately, interleaved, as in the README. Check that the two give the same results within noise; a faster run that gives different results is not met.
    - **No ugly hacks in the model.** List every workaround the v4 version needs: monkeypatches, private attributes of Starsim objects, copies of Starsim internals, compatibility shims beyond the model's own documented v3 interface (for covasim, its v3 compatibility shims are allowed by design), and code that exists only because v4 lacks something. Each is a finding against v4.
    - **No single-model code in v4.** Search the snapshot's `starsim/` for code that only this model uses: its name (`grep -rni <model> starsim/`), and each v4 feature its port relies on that no test, doc, or other model uses (check `git -C /home/cliffk/idm/ssv4 log --oneline -i --grep=<model>` for features added for it). Each is a finding: either generalize it or move it into the model.
    - **The compartmental goal** in the table. Build the compartmental version of the model's disease (or diseases) as the goal states, run it under `solver='ode'` and as agents, and compare (prevalence and incidence over time, at the same parameters). Report what can't be expressed, and whether that is inside the goal's stated exceptions.
4. **Fan out if useful,** to at most four subagents by part of the model (as the earlier reviews did: e.g. diseases, interventions, framework and speed, compartmental), each writing its findings to `<repo>/v4port/<date>/<part>/NOTES.md`. Brief each with the README's rules, the pinned paths, and this skill's definitions of the targets.
5. **Check** every headline number and every "not met" yourself against the pinned versions. Cross-reference the earlier reviews' findings (by their IDs, e.g. CV, FP, HP, ST) and say which still hold, which are fixed, and which are new; and `admin/deferred.md` and the latest plan.
6. **Write** `v4_design/<date>/reviews/<model>.md` in the README's shape. The scorecard has one row per general target and one for the compartmental goal. The details give lines by module, timings, the hacks list, the single-model code list, and the compartmental comparison. End with the model's own bugs found along the way, briefly.
7. **Clean up** the worktrees (`git -C <repo> worktree remove --force ...`) and any background shells.
