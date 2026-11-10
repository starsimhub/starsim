# Starsim

The Starsim package. The files at this level are what a modeler might read; the machinery is in the folders, though the list below includes the few files in `internals/` that a modeler may also want. Everything public is at the top level (`import starsim as ss`), so none of these needs to be imported on its own.

## Files

- The sim and its run: `sim.py` (`ss.Sim`), `internals/loop.py` (the order of the steps), `run.py` (`ss.MultiSim`, `ss.parallel()`), `internals/fitting.py` (`sim.fit()`), `results.py`, and `analyzers.py`
- The model: `diseases.py`, `transitions.py` (states and transitions, `S >> I @ rate`), `networks.py`, `demographics.py` (births, deaths, and pregnancy), `interventions.py`, `products.py`, `connectors.py`, `strains.py`, and `internals/modules.py` (what they all build on)
- The people and their attributes: `people.py`, `internals/arrays.py`, `internals/conditions.py` (for `where=`), and `internals/dimensions.py` (stratifications)
- Values: `internals/parameters.py` (`ss.Pars`, `ss.by`, `ss.timeseries`), `distributions.py`, `internals/time.py` (durations and rates), `internals/dates.py`, and `internals/timeline.py`
- Places and data: `geo.py` and `data.py`
- Everything else: `settings.py` (`ss.options` and plotting style), `utils.py`, and `version.py`

## Folders

- `solvers/`: the engines that run declared models, as compartments or as agents, and their kernels (`solvers/backends/`, and a copy in Rust in `solvers/rust/`)
- `converters/`: importing models from other frameworks (`ss.import_from()`), exporting to them (`ss.export_to()`), and rewriting Starsim 3 code
- `internals/`: how Starsim works under the hood (the graph of a declared model, expressions, gradients, errors, debugging tools, the command line, and the fonts)
- `library/`: example and reference modules that build on Starsim but are not part of it (`import starsim.library as ssl`)
