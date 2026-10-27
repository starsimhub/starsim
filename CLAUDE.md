# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Starsim is an agent-based modeling framework for simulating disease spread among agents via dynamic transmission networks. It supports co-transmission of multiple diseases and detailed modeling of intervention strategies.

For a compact index of the whole public API (signatures, summaries, and default parameters), see https://docs.starsim.org/llms.txt (or `llms-full.txt`, which adds an example for each entry). These are generated when the docs are built and are not committed; to generate them locally for the current branch (as `docs/llms.txt` etc.), run `cd docs && python make_api.py`. The equivalent for Sciris is at https://docs.sciris.org/llms.txt.

The design record of v4 (the constitution, the spec, the current plan, and what was deferred or decided against) is in `v4_design/`; start at `v4_design/README.md`.

## Core architecture

The framework follows a modular design with these key components:

- **Sim**: Main simulation class that orchestrates all modules and runs simulations (`starsim/sim.py`)
- **People**: Manages individual agents and their state updates (`starsim/people.py`)
- **Networks**: Defines how agents interact with each other (`starsim/networks.py`)
- **Diseases**: Models disease transmission and progression (`starsim/diseases.py`)
- **Interventions**: Implements prevention/treatment strategies (`starsim/interventions.py`)
- **Demographics**: Handles population dynamics like births/deaths (`starsim/demographics.py`)
- **Pregnancy**: Pregnancy as declared states and transitions (Demographics+Disease hybrid) that provides events/states for other modules (`starsim/demographics.py`)
- **Connectors**: Modules that link other modules, e.g. disease interactions (`starsim/connectors.py`)
- **Analyzers**: Custom result collection and analysis (`starsim/analyzers.py`)
- **Results**: Stores simulation outputs (`starsim/results.py`)

The main simulation loop is orchestrated by `starsim/loop.py`, while parameter management is handled by `starsim/parameters.py`. Random number generation and statistical distributions are centralized in `starsim/distributions.py`.

## Development commands

### Running tests
```bash
# Run the core tests (tests/test_*.py) with parallel execution
cd tests && ./run_tests

# Run the developer tests (slower and more exhaustive; not run by default)
cd tests && pytest devtests -n auto

# Run specific test file
cd tests && pytest test_sim.py

# Run tests with coverage
cd tests && ./check_coverage
```

### Documentation
```bash
# Build documentation (requires quarto)
cd docs && ./render

# Build and publish to GitHub Pages
cd docs && ./publish

# Generate the API index locally (api.json, llms.txt, llms-full.txt); these are gitignored, and are built automatically with the docs
cd docs && python make_api.py
```

### Installation
```bash
# Development installation
pip install -e .

# With dev dependencies
pip install -e .[dev]

# Using uv for faster installs
uv add starsim
uv add starsim[dev]  # With dev dependencies
```

## Project structure

### Core modules (`starsim/`)
- `analyzers.py`: Analyzers for custom results (e.g. `ss.infection_log`, `ss.dynamics_by_age`)
- `arrays.py`: State management arrays for people/networks
- `conditions.py`: Conditions on states and people, for `where=` (`ss.conditions.Cond`)
- `connectors.py`: Connectors between modules (e.g. `ss.modify`)
- `data.py`: Data by place (`ss.data.population('Kenya')`, `ss.data.contacts(...)`), read by `ss.people('Kenya')`, `ss.contacts('Kenya')`, and `ss.geo.geography('Kenya')`
- `dates.py`: Dates and calendar durations (`ss.date`, `ss.DateArray`, `ss.datedur`)
- `demographics.py`: Births, deaths, and migration (`ss.Births`, `ss.Deaths`), and pregnancy as declared states and transitions (`ss.Pregnancy`, which provides events such as `conception` and `delivery`, and states such as `P`, for other modules)
- `dimensions.py`: Stratifications (`ss.dimensions.Strat`, `ss.transfer`) and the groups they give (`sim.groups`, an `ss.dimensions.Groups`)
- `diseases.py`: Diseases, declared (states and transitions, in the class form or with `define_states()` and `define_transitions()` in `__init__()`), written as code, or both (`ss.Disease`, `ss.Infection`, `ss.SIR`, ...)
- `distributions.py`: Statistical distributions and random number generation
- `fit.py`: Fitting to data, as `ss.fit` (imported when first used; `sim.fit()` is the entry point)
- `geo.py`: Places and the movement between them (`ss.geo.geography`, `ss.geo.Mobility`, `ss.geo.gravity`)
- `loop.py`: Main simulation integration loop
- `modules.py`: Base module classes and update logic
- `networks.py`: Networks and mixing pools
- `parameters.py`: Parameter containers (`ss.Pars`, `ss.SimPars`) and values that vary by level or in time (`ss.by`, `ss.timeseries`, `ss.seasonal`)
- `people.py`: The people (`ss.People`, `ss.people()`, `ss.ppl`)
- `products.py`: Vaccine and treatment deployment
- `run.py`: Running multiple simulations (`ss.MultiSim`, `ss.parallel`)
- `settings.py`: Global options (`ss.options`) and plotting style
- `sim.py`: The sim (`ss.Sim`)
- `solvers/`: The engines that run declared models: `engine.py` (what they share), `mixing.py` (who meets whom in a compartmental run), `compartmental.py`, `agents.py`, `crossing.py` (diseases tracked jointly), and `choose.py` (which solver, and why: `ss.solvers.choose_solver()`); every name is also `ss.solvers.X`
- `strains.py`: Strains (`ss.Strains`) and immunity between them (`ss.immunity`)
- `time.py`: Durations and rates (`ss.dur`, `ss.days`, `ss.peryear`, `ss.probperyear`, etc.), and plotting dates and durations with Matplotlib
- `timeline.py`: Time coordination between modules (`ss.Timeline`)
- `transitions.py`: States and transitions (`ss.State`, `ss.states()`, `ss.Transition` and its `S >> I @ rate` spelling, ...); the names of the modules above that it builds on are also `ss.transitions.X`
- `utils.py`: Helper functions
- `internals/`: How Starsim works under the hood, each also at the top level (`ss.errors`, `ss.spec`, ...): `graph.py` (the graph of a declared module: its states, transitions, and parameters, completed with defaults and checked; `ss.transitions.Graph`), `presentation.py` (how it is shown: `describe()`, `diagram()`, `equations()`), `spec.py` (the model spec, and models written as code: `sim.to_code()`, `ss.diff()`), `expressions.py` (expressions in rates and results, `ss.par`, and how they are evaluated and printed), `gradients.py` (gradients and log-densities in JAX: `sim.gradient()`, `sim.log_density()`), `errors.py` (errors, warnings, and notes, as classes, e.g. `ss.errors.UnknownState`; the base classes and `ss.CapabilityError` are also at the root), `debugtools.py` (profiling, debugging, and mock objects for testing), and `cli.py` (the `starsim` command)
- `perflib/`: Machinery for speed: `kernels.py` (the Numba kernels that run compartmental models; also `ss.kernels`), `draws.py` (exact draws in Numba), `passes.py` (compiled passes over the agents), `chain.py` (the chain binomial in JAX), `jit.py` (`ss.jit`; `ss.perflib.jit` is the function), `cache.py`, and `memory.py`
- `interop/`: Importers and exporters for other modeling frameworks

### Library (`starsim/library/`)
Example and reference modules that build on core Starsim but are not part of it, available via `import starsim.library as ssl` (e.g. `ssl.Cholera` or `ssl.diseases.Cholera`):
- `diseases/`: HIV, cholera, Ebola, and measles
- `mnch/`: Maternal, newborn, and child health (fetal health, maternal infections, neonatal sepsis)
- `networks/`: Household, spatial, and theoretical (e.g. Erdős–Rényi) networks

Core disease base classes (Disease, Infection, NCD, SIR, SIS, SEIR) live in `starsim/diseases.py`.

### Testing (`tests/`)
- `test_*.py`: Core test files (about 300 tests, under a minute in parallel); keep new tests here few and quick
- `devtests/`: Developer tests (contracts, inference, the spec, errors, interop, and slower feature tests), in files named like the core ones
- `run_tests`: Main test runner script
- `pytest.ini`: Pytest configuration

### Documentation (`docs/`)
- `tutorials/`: Quarto tutorial notebooks (`.qmd`)
- `user_guide/`: Quarto user guide notebooks (`.qmd`)
- `examples/`: Worked examples
- `migration/`: Migration guides between versions
- `api/`: API documentation source files (`.qmd`)
- `make_api.py`: Generates the machine-readable API index (`api.json`, `llms.txt`, `llms-full.txt`)
- Quarto-based documentation system

## Key conventions

- Uses `sciris` library extensively for utilities and plotting
- Follows Google Python style guide with project-specific exceptions
- Tests use pytest with parallel execution via `pytest-xdist`
- Documentation built with Quarto and executed via Jupyter
- Random number generation is centralized in `distributions.py`, with Common Random Numbers (CRN) for reproducibility

## Module architecture

The framework uses a modular architecture where all components inherit from base classes:

- **Base class hierarchy**: All modules inherit from `ss.Base` → `ss.Module` (defined in `modules.py`)
- **Module types**: Networks, Demographics, Diseases, Interventions, Analyzers, Connectors
- **Module registration**: Modules are automatically discovered and registered via the `find_modules()` function
- **Integration loop**: The `Loop` class (in `loop.py`) orchestrates module execution during simulation
- **State management**: People and network states are managed through specialized array classes (`arrays.py`)

## Key dependencies

- **Sciris**: Core utility library for data structures, plotting, and utilities
- **Numba**: Used for performance-critical code sections
- **NetworkX**: Network analysis and manipulation
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn**: Plotting and visualization

## Important notes

- Follow the style guide here: https://github.com/starsimhub/styleguide/blob/main/README.md
- Use built-in Starsim plotting commands if possible, only falling back to Matplotlib if necessary
- Use Sciris where possible to shorten commands
- When creating a multiline dictionary, put a space around the equals for arguments (as if it were a class)
- The framework requires Python 3.11+
- All modules inherit from base classes in `modules.py`
- Parameter handling is centralized through `parameters.py` and `SimPars` class
- By default, use "Sentence case", not "Title Case", for headings (e.g. Markdown headings, table column headings, etc)
