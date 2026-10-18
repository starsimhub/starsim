# Names removed from `ss.` in Starsim 4

Every name Starsim 3 exported at `ss.` that Starsim 4 does not, plus the internals that early Starsim 4 exported and that now live in submodules. Each raises an `AttributeError` that says what to write instead (`starsim/__init__.py`, `_removed`; checked by `tests/test_spec.py::test_namespace`), and each is mapped in `skills/migrate-starsim3/SKILL.md`; `python -m starsim.migrate` (`starsim/migrate.py`) rewrites the mechanical ones. Removed methods say what to write too: `Result.annualize()`, and `MultiSim.reduce()`, `mean()`, `median()`, and `reset()`.

The rule: a family of names stays or goes together. The time names are one family, so all of them are kept (`ss.perday` to `ss.peryear`, `ss.probperday` to `ss.probperyear`, `ss.freq` and `ss.freqperday` to `ss.freqperyear`, and `ss.day` to `ss.year`), and only the Starsim 2 names that Starsim 3 already deprecated, with a warning, are removed.

"Uses" counts lines (and files) that use the name in seven downstream repos (stisim, hpvsim, tbsim, mighti, covasim, fpsim, rsvsim), on their `main` branch unless a `v4` branch is named, in `.py`, `.ipynb`, `.qmd`, and `.md` files. For comparison, the time names that are kept: `ss.probper*` 16 lines in 11 files, `ss.freqper*` 19 lines in 4 files, `ss.freq` 5 in 1, `ss.day` to `ss.year` 7 in 2 (`.py` and `.ipynb` only).

## Starsim 3 names

| Removed | Write instead | Why it was removed | Uses | Recommendation |
|---|---|---|---|---|
| `ss.rate`, `ss.time_prob`, `ss.rate_prob` | `ss.peryear(x)`; `ss.probperyear(x)`; `ss.peryear(x)` | Starsim 2 names, deprecated in Starsim 3 with a warning | `ss.rate`: tbsim 2 lines (1 file) | Remove (already deprecated) [CK: ok] |
| `ss.MaternalNet` | `ss.PrenatalNet` | An alias of `ss.PrenatalNet` | stisim 36 (22 files, on `main` and `v4`), rsvsim 23 (22), mighti 8 (7), hpvsim 1, fpsim 1 | Removed (Cliff, 2026-10-18: no aliases; the port should make the code simpler). `python -m starsim.migrate` renames it, and the network's name with it (`maternalnet` to `prenatalnet`, a `beta` key `'maternal'` to `'prenatal'`)  [CK: i guess, although i wonder if 'maternalnet' is useful for keep for people who don't want to distinguish between prenatal and postnatal .... but i guess ultimately i agree with the decision to remove it] |
| `ss.simple_vx`, `ss.routine_vx`, `ss.campaign_vx` | `ss.Vaccine(efficacy=e)`; `ss.vaccinate(..., coverage=p, start=y, interval=ss.years(1))`; `ss.vaccinate(..., at=[...])` | Routine versus campaign became a keyword (`interval=` or `at=`), not a class; one delivery class per action | `routine_vx`: hpvsim 2 (1 file, on `main` and `v4`) | A family with the four below. Either remove all seven (few uses, and the error says what to write), or keep all seven as thin classes composed from the base classes that remain (`class routine_vx(ss.BaseVaccination, ss.RoutineDelivery)`), with their Starsim 3 keywords (`prob=`, `start_year=`, `years=`)  [CK: ok to remove] |
| `ss.routine_screening`, `ss.campaign_screening`, `ss.routine_triage`, `ss.campaign_triage` | `ss.screen(ss.Test(...), ...)`; triage as `on_positive=ss.screen(...)` | As above | mighti: `routine_screening` 3, `campaign_screening` 2, `routine_triage` 1, `campaign_triage` 1 (1 file each) | As above  [CK: ok to remove] |
| `ss.seasonality` | `ss.modify(flu.rel_trans, multiply=ss.seasonal(...))` in `connectors=` | It overwrote `rel_trans` every step, erasing any other effect on it; a modifier multiplies a base that is never overwritten | none | Remove (the old behavior was a quiet error) [CK: ok] |
| `ss.multi_run` | `ss.parallel(sim, n_runs=n)` or `sim.run(n_runs=n)`; `.sims` is the list it returned | One way to run many sims | covasim 1 | Remove [CK: ok] |
| `ss.RandomSafeNet`, `ss.ClusterNet`, `ss.HybridNet`, `ss.MSMNet` | `ssl.RandomSafeNet` and so on (`import starsim.library as ssl`) | Moved to the library unchanged; rarely used | covasim: `HybridNet` 3, `ClusterNet` 1 (1 file) | Remove from `ss.` (they are in `ssl.`) [CK: ok to remove; MSMNet should probably be moved to STIsim if it's not already present] |
| `ss.RandomExactNet` | `ss.RandomNet(method='exact')` | One random network, with a method | none | Remove [CK: ok] |
| `ss.Route` | `ss.networks.Route` | The base class of networks and mixing pools; most code subclasses `ss.Network` | stisim 1 (on `main` and `v4`) | Remove from `ss.` [CK: ok] |
| `ss.InfectionLog` | `ss.diseases.InfectionLog` | A type you get back, not one you make | covasim 4 (3 files) | Remove from `ss.` [CK: ok] |

## Early Starsim 4 names (never in a Starsim 3 release)

| Removed | Write instead | Why | Uses |
|---|---|---|---|
| `ss.stratify(model, ...)` | `model.stratify(...)` | Two spellings of one thing | none [CK: ok] |
| `ss.Expr`, `ss.Graph`, `ss.Strat`, `ss.Transfer` | `ss.transitions.Expr` and so on | Internals (the namespace tiers, L29) | none [CK: ok] |
| `ss.Cond` | `ss.conditions.Cond` | Internal | none [CK: ok] |
| `ss.MultiResults` | `ss.run.MultiResults` | Internal | none [CK: ok] |
| `ss.Engine`, `ss.Compartmental`, `ss.Agents`, `ss.Crossing`, `ss.Mixing`, `ss.CountState`, `ss.make_engine`, `ss.solver_names` | `ss.solvers.Compartmental` and so on | Internals | `Crossing`: tbsim 2 (`v1.0.0`, the Starsim 4 port) [CK: ok] |

## Keywords removed or renamed

Each raises an error that names the new keyword.

| Removed | Write instead |
|---|---|
| A delivery's `years=`, `dates=` | `at=` (as `ss.change()` and `ss.modify()` take) [CK: hm, 'when' seems like a better keyword than 'at'] |
| A delivery's `eligible=` | `where=`  [CK: hm, 'who' seems like a better keyword than 'where', which implies location to me] |
| `ss.Sim(birth_rate=, death_rate=)` | `demographics=[ss.birth(rate=...), ss.death(rate=...)]`, or `ss.Births`/`ss.Deaths` modules [CK: ok] |
| `ss.Sim(track_writes=)` | `sim.what_modifies()` lists the declared modifiers [CK: ok] |
| `ss.MFNet(duration=)` | `dur=` [CK: ok] |
| `ss.RandomNet(uniform_targets=True)` | `method='uniform'` [CK: ok] |
| `ss.contacts(matrix, age=edges)` | `bands=edges` (`age=` is a matrix over the model's own age groups) [CK: ok, except 'bands' is not clear t ome] |
| `integrator='jax'`, and the SciPy integrators (`'LSODA'`, `'RK45'`, ...) | `integrator=` is `'rk4'` or `'euler'`; JAX is used without being asked by `sim.gradient()`, `sim.log_density()`, gradient fits, and `ss.parallel()` with many sets of values [CK: ok] |
| `solver='tau'`, `solver='sde'` | `'gillespie'`, which takes tau leaps when a run would have too many events (`approx=` sets this), or `'binomial'` [CK: not sure about this; 'tau' and 'sde' both mean something to epidemiologists, so i think we should map these onto meaningful results] |
