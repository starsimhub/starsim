---
name: migrate-starsim3
description: Upgrade a Starsim 3 model, script, or package (a model built on ss.SIR/SIS/SEIR, ss.Infection, routine_vx, ss.Calibration, and so on) to Starsim 4. Use when someone's Starsim 3 code fails or changes its numbers under Starsim 4, asks what a removed name became, or wants to use Starsim 4's tools (ss.modify, conditions, sim.fit, declared diseases) on an existing model. For porting a disease to declared states and transitions, continue with the convert-model skill's references/starsim3.md.
---

# Upgrade a Starsim 3 model to Starsim 4

Starsim 4 keeps Starsim 3's module API: a disease written as code (`ss.Disease`, or `ss.Infection`, now `ss.AgentDisease`, with `define_states()`, `set_prognoses()`, `step_state()`), networks, demographics, interventions, connectors, and analyzers all still work, and are a supported way to write a model. What changed is that where Starsim 4 has its own way to say something, Starsim 3's duplicate spelling was removed. So an upgrade is in three steps: make it run, check the numbers, then, if wanted, use the new tools or declare one disease at a time.

## Where the answers are

- [Migration guide](https://starsim.org/v4/migration/v3_v4.html) (`docs/migration/v3_v4.md`): "The migration script"; "What changes the numbers" (each change that runs without an error but gives different results); "Jobs and spellings" (each Starsim 3 job and its Starsim 4 spelling); "Migrating one disease at a time" and "What needs a declared disease" (the Starsim 4 tools that work on an unchanged model).
- [Full changelog](https://starsim.org/v4/migration/v4_changelog.html) (`docs/migration/v4_changelog.md`): every rename and removal ("Vocabulary and renames", the *Migration* items), and every change that may affect results ("Changes that may affect results").
- The errors: every removed name fails loudly, and its error says what replaced it (from `starsim/internals/v3_deprecations.py`). Nothing removed fails silently.

## Workflow

1. **Run the migration script** on the package, its tests, and its docs: `python -m starsim.converters.migrate mypackage/ tests/ docs/` (`--check` lists the changes without writing). It reads `.py` files and the code cells of `.ipynb`, `.qmd`, and `.md` files. It makes the mechanical changes in place (the renames below, `verbose=-1`, `annualize()`, the bases `ss.SIR`, `ss.SIS`, and `ss.SEIR` of diseases written as code (to `ssl.AgentSIR` and the others), a `beta` pair on `ss.MFNet` (to `ss.peract(m2f=..., f2m=...)`; `ss.MFNet` and `ss.SexualNetwork` are now `ss.SexualNet`) or on a mother-to-child network (`[b, 0]` to `b`), `ss.uids.cat()`, `preg.breastfeeding`, `ss.RandomNet(uniform_targets=True)`, and the counts of the states of Starsim's own modules after results (`results.n_alive` to `results.alive`, `res['n_infected']` to `res['infected']`, unless the code makes a result of that name itself)) and lists, by file and line, what needs a person: work through that list with the tables below, and read the diff. It doesn't rewrite the time index: Starsim 3's `self.ti` means the same in Starsim 4 (below). It lists each `update_results()`, which Starsim 4 no longer calls (a class that defines it raises a `TypeError`): split it by hand into `record_states()`, at the start of the step, and `record_events()`, at its end (below).
2. **Run it and fix each remaining error**, using the error's message and the changelog.
3. **Check each change in "What changes the numbers"** and the changelog's "Changes that may affect results" that applies to the model: these run without an error but print different numbers.
4. **Validate against Starsim 3**: run the model under Starsim 3 (pin `starsim==3.7.2` in a separate environment) and under Starsim 4 with 10 or more seeds each, and compare the means and spreads of the results you care about (see the convert-model skill's `references/validate.md`). Expect them to agree statistically, not bit for bit: the random streams differ, and stocks are recorded one index later.
5. **Optionally, use the Starsim 4 tools on the unchanged model**, or port diseases to declared form one at a time (convert-model skill, `references/starsim3.md`), validating after each.

## Reporting

Say which names were changed, which behavior changes applied and how the validation came out (means and spreads, Starsim 3 against Starsim 4), and anything that could not be upgraded.
