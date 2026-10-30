# Starsim

The Starsim package. The files at this level are what a modeler might read; the machinery is in the folders. Everything public is at the top level (`import starsim as ss`), so none of these needs to be imported on its own.

## Files

- The sim and its run: `sim.py` (`ss.Sim`), `loop.py` (the order of the steps), `run.py` (`ss.MultiSim`, `ss.parallel()`), `fit.py` (`sim.fit()`), `results.py`, and `analyzers.py`
- The model: `diseases.py`, `transitions.py` (states and transitions, `S >> I @ rate`), `networks.py`, `demographics.py` (births, deaths, and pregnancy), `interventions.py`, `products.py`, `connectors.py`, `strains.py`, and `modules.py` (what they all build on)
- The people and their attributes: `people.py`, `arrays.py`, `conditions.py` (for `where=`), and `dimensions.py` (stratifications)
- Values: `parameters.py` (`ss.Pars`, `ss.by`, `ss.timeseries`), `distributions.py`, `time.py` (durations and rates), `dates.py`, and `timeline.py`
- Places and data: `geo.py` and `data.py`
- Everything else: `settings.py` (`ss.options` and plotting style), `utils.py`, and `version.py`

## Folders

- `solvers/`: the engines that run declared models, as compartments or as agents, and their kernels (`solvers/backends/`, and a copy in Rust in `solvers/rust/`)
- `converters/`: importing models from other frameworks (`ss.import_from()`), exporting to them (`ss.export_to()`), and rewriting Starsim 3 code
- `internals/`: how Starsim works under the hood (the graph of a declared model, expressions, gradients, errors, debugging tools, the command line, and the fonts)
- `library/`: example and reference modules that build on Starsim but are not part of it (`import starsim.library as ssl`)
