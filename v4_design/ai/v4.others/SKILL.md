---
name: v4.others
description: Review Starsim v4 against models from other frameworks (Ghana measles, LASER, and the 20 examples from the approaches survey) for lines of code and speed, and write one report to v4_design/<date>/reviews/others.md. Use when Cliff runs /v4.others or asks to rerun the other-models review.
---

# Other models review

Checks the "Other models" targets in `v4_design/targets.md`. Read [../README.md](../README.md) first: its rules (snapshot, line counts, timing, subagents, report shape) apply here.

## Where the models are

Everything is in `/home/cliffk/idm/lingua_franca_design` (LFD):

- `examples/<framework>/<model>/`: the original, with `meta.yaml` (version, source, and `lines`), `README.md` (what it is, and its reference run time), `reference/` (its results), and `run.py`.
- `vignettes/<framework>/<model>/`: the Starsim version (`model.py`), `compare.py` (which checks it against `reference/`), and `README.md`, whose "Comparison" table (lines, one run, results) and "Findings" are what this review updates.
- `_envs/<framework>/`: what each original needs to run (`python _envs/envs.py check` says what is installed).

The models: Atomica (`sir`, `hiv_dyn`), camdl (`sir`, `stratified`), EMULSION (`quickstart`, `sir`), EpiModel (`sis-network`, `hiv`), epydemix (`sir_kenya`, `covid19`), flepiMoP (`seir_2pop`, `vaccine_scenarios`), MEmilio (`secir`, `secir_ageres`), odin (`stochastic_sir_arrays`, `seir_array`), SimInf (`sir_herds`, `sise3`), summer2 (`sir-age-location`, `seir-age-clinical`), LASER (`england_wales`, `hub_spoke_seir`), and Ghana measles (`seir_districts`, whose original is `ghana_measles/model.py` in `/home/cliffk/idm/ghana-measles/gm-port`).

The vignettes are the maintained Starsim versions: when the API has changed, update `model.py` and `compare.py` in place in LFD (git keeps the history), and update the vignette's `README.md` with the new numbers and the date. Don't edit `examples/`. Don't commit.

## Targets, and how to measure each

- **Survey models: fewer lines, except for very specialized things.** Lines of `model.py` against the original's `lines` in `meta.yaml`, counted the same way (see the README's rule). A model that is longer is met only if the vignette's README says what specialized thing makes it so, and the reason holds.
- **Survey models: faster, or within 2x for already very fast models.** "Very fast" means the original's run takes under about 0.1 s. Time the run, not the build or the import, on both sides where the original's env is installed; otherwise use the original's time in its README and say so. For an original that compiles (odin, camdl), compare the run and report the compile time separately, as the vignettes already do.
- **Ghana measles: works as-is as a custom module.** Wrap the original model's code (`gm-port/ghana_measles/model.py`) as one `ss.Module` whose `step()` calls the original's update, with as few changes as possible, and check its results against `examples/ghana_measles/seir_districts/reference/`. Keep the wrapper in the vignette folder as `custom_module.py`. Met if it runs and agrees with changes only at the boundary (state in, state out); list each change that went inside the original code.
- **Ghana measles: written in v4, faster and fewer lines.** The vignette's `model.py` against the original, by the two rules above.
- **LASER England and Wales.** First the vignette against the original, by the two rules above. If v4 is slower, the target becomes: the LASER model's own components, used as a custom module in a v4 sim (kept as `laser_module.py` in the vignette folder), run within about 1.5 times LASER's time with about as many lines as the original. Report both.
- **LASER hub and spoke:** as a survey model.

For every model, also run `compare.py`: a model that is short and fast but disagrees with the reference is not met.

## Steps

1. **Snapshot** v4 as in the README.
2. **Fan out** to at most five subagents, each with a group of frameworks (e.g. Atomica, camdl, EMULSION, EpiModel; epydemix, flepiMoP, MEmilio; odin, SimInf, summer2; LASER; Ghana measles). Each runs every model in its group by the rules above, updates the vignettes in LFD as needed, and writes a findings file to the scratchpad: per model, lines on each side, time on each side, agreement, met or not on each target, and what v4 should change for each target not met (with a minimal repro). The LASER and Ghana measles agents also build the custom modules.
3. **Check** every "not met" yourself, and every number that moved by more than 20% since the vignette's README.
4. **Write** `v4_design/<date>/reviews/others.md` in the README's shape. The details section is one table with a row per model: framework and model, lines (original, v4), one run (original, v4), agreement, the two targets, and the change since the last report. For the first run, "the last report" is the numbers in each vignette's README as they were before this run (read them before updating them).
5. **Clean up** the snapshot worktree and any background shells.
