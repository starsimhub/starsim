---
name: validate-import
description: Check a Starsim port of a model from another framework against the original, by running both and comparing their results exactly (deterministic models) or statistically (stochastic ones), and explain any difference. Use after the import-model skill, or whenever someone asks whether a translated model matches its original.
---

# Validate an imported model

Run the original in its own framework and the port in Starsim, write both as the same long CSV, and compare them with `scripts/compare.py`. Then explain every difference that is left: a difference is either a bug in the port, a known difference of convention (stated, with the reason), or a finding about Starsim.

## Workflow

1. **Run the original** in the framework's own environment, never in Starsim's (don't install a framework without asking). The framework's guide in `../import-model/references/<framework>.md` says how, under "Running the original". For a stochastic model, save every run (e.g. 200), not only a summary, so the comparison can test distributions. In the `lingua_franca_design` repo, `examples/<framework>/<model>/run.py` already does this and writes `reference/results.csv`.
2. **Write the original's results as a long CSV**: columns `time`, `series`, `value`, and `run` for replicates (or `low` and `high` for a summary, e.g. the 5% and 95% quantiles). One row per run, time, and series. Use the source's own series names.
3. **Run the port** with the same population, duration, step, and number of runs:
   ```bash
   python scripts/compare.py run model.py port.csv --runs 200       # model.py defines sim or make_sim()
   ```
   This writes every result as `<disease>.<result>` (e.g. `sir.I`, `sir.infection`), at the sim's times (in its unit from the start, or in years with dates; `--time index` gives the step instead).
4. **Compare**, mapping the original's names to the port's:
   ```bash
   python scripts/compare.py compare original.csv port.csv --map I=sir.I R=sir.R --plot compare.png
   ```
   The mode is chosen from what the two files hold:
   - **exact**: one run on each side (both deterministic). Passes if the largest difference is within `--rtol` (default 1e-3) of the largest value. Use a tighter tolerance when both use the same integrator and step; ODE solvers that differ (RK4 against LSODA, a fixed step against an adaptive one) agree only to their error.
   - **statistical**: runs on both sides. At each time, the difference of the means in standard errors (`frac_z3` is the share of times where it exceeds 3; passes at 5% or less), and two-sample Kolmogorov-Smirnov tests of each run's peak, time of the peak, and final value (pass at `--alpha`, default 0.01, as in the conformance spec).
   - **band**: the original is a summary (mean, low, high). The port's mean and the same quantiles of its runs are compared with them, within `--rtol` (default 0.1) of the largest value. This is weaker than the statistical mode: prefer raw runs.
5. **Explain what differs.** Go through the usual causes below before concluding anything, and say which applied.
6. **Report** a table of the series compared, the mode, the numbers, and the verdict, with the reason for every tolerance and every difference. Inside `lingua_franca_design`, also write the result into the vignette's `README.md` (agreement, lines of model code, and time of one run, as `examples/README.md` says to count them).

## Usual causes of a difference

- **Time labels.** Starsim's `X[ti]` is the number at time point `ti` (`X[0]` is the start), and a flow at `ti` is the number during the step that ends there (0 at `ti = 0`). Most frameworks do the same, but some label a step by its start, or report a flow at the start of its step. If the port looks one step early or late, check with `--shift 1` (or `-1`), and say which convention each side uses rather than leaving the shift unexplained.
- **New against cumulative, and per step against per unit of time.** A transition's result (`sir.infection`) is the number in the step; `sir.infection.cum` the running total. An incidence per day from a weekly step is the step's number divided by 7.
- **Rates and probabilities.** A per-step probability imported as a rate (or the reverse), or a probability per unit of time confused with a rate, shifts everything a little at small steps and a lot at large ones.
- **The step.** Agents and a compartmental run of the same model differ by an amount of order `dt` (5 to 10% at a weekly step in some models); a discrete-time stochastic model differs from the ODE by the same. Compare like with like (the same kind of run), or at a finer step, before calling it a disagreement.
- **Order within a step.** Frameworks that apply processes one after another in a step (EMULSION's `processes:`, Atomica's order of transitions) differ from Starsim's simultaneous competing hazards, by order `dt`.
- **Initial conditions.** Counts against fractions, the remainder placed in the first state, initial numbers drawn rather than fixed, and a source that starts people "before" the first step.
- **Extinction.** In small populations many runs die out; means are pulled down and peaks are bimodal. Compare the share of runs that die out, and the distributions of the others.
- **Random numbers.** A stochastic port never matches run for run; only distributions can agree. Two deterministic runs that should match exactly but do not point to a real difference in the model.
- **What was not translated.** Everything in `sim.imported.incomplete` (or the import's `ImportIncomplete` warning) that was left out changes the results; say which differences it explains.

A difference that remains after all of these is reported with the numbers, as a finding: either the port is wrong (fix it), or Starsim cannot yet say what the original says (a gap to report).
