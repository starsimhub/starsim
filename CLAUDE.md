# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Starsim is an agent-based modeling framework for simulating disease spread among agents via dynamic transmission networks. It supports co-transmission of multiple diseases and detailed modeling of intervention strategies.

For a compact index of the whole public API (signatures, summaries, and default parameters), see https://docs.starsim.org/llms.txt (or `llms-full.txt`, which adds an example for each entry). These are generated when the docs are built and are not committed; to generate them locally for the current branch (as `docs/llms.txt` etc.), run `cd docs && python make_api.py`. The equivalent for Sciris is at https://docs.sciris.org/llms.txt.

## Core architecture

The framework follows a modular design with these key components:

- **Sim**: Main simulation class that orchestrates all modules and runs simulations (`starsim/sim.py`)
- **People**: Manages individual agents and their state updates (`starsim/people.py`)
- **Networks**: Defines how agents interact with each other (`starsim/networks.py`)
- **Diseases**: Models disease transmission and progression (`starsim/diseases.py`)
- **Interventions**: Implements prevention/treatment strategies (`starsim/interventions.py`)
- **Demographics**: Handles population dynamics like births/deaths (`starsim/demographics.py`)
- **Pregnancy**: Pregnancy as declared states and transitions (Demographics+Disease hybrid) that provides events/states for other modules (`starsim/pregnancy.py`)
- **Connectors**: Modules that link other modules, e.g. disease interactions (`starsim/connectors.py`)
- **Analyzers**: Custom result collection and analysis (`starsim/analyzers.py`)
- **Results**: Stores simulation outputs (`starsim/results.py`)

The main simulation loop is orchestrated by `starsim/loop.py`, while parameter management is handled by `starsim/parameters.py`. Random number generation and statistical distributions are centralized in `starsim/distributions.py`.

## Development commands

### Running tests
```bash
# Run all tests with parallel execution
cd tests && ./run_tests

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
- `dates.py`: Dates and calendar durations (`ss.date`, `ss.DateArray`, `ss.datedur`)
- `debugtools.py`: Profiling, debugging, and mock objects for testing
- `dimensions.py`: Stratifications (`ss.dimensions.Strat`, `ss.transfer`) and the groups they give (`sim.groups`, an `ss.dimensions.Groups`)
- `distributions.py`: Statistical distributions and random number generation
- `errors.py`: Errors, warnings, and notes, as classes (e.g. `ss.errors.UnknownState`; the base classes and `ss.CapabilityError` are also at the root)
- `expressions.py`: Expressions in rates and results (`ss.par`, `ss.transitions.Expr`), and how they are evaluated and printed
- `fit.py`: Fitting to data, as `ss.fit` (imported when first used; `sim.calibrate()` is the entry point)
- `geo.py`: Places and the movement between them (`ss.geo.location`, `ss.geo.Mobility`, `ss.geo.gravity`)
- `gradients.py`: Gradients and log-densities in JAX (`sim.gradient()`, `sim.log_density()`)
- `graph.py`: The graph of a declared module: its states, transitions, and parameters, completed with defaults and checked (`ss.transitions.Graph`)
- `kernels.py`: The Numba kernels that run compartmental models
- `loop.py`: Main simulation integration loop
- `modules.py`: Base module classes and update logic
- `named.py`: Places and contact matrices by name (`ss.people('Kenya')`, `ss.contacts('Kenya')`)
- `networks.py`: Networks and mixing pools
- `parameters.py`: Parameter containers (`ss.Pars`, `ss.SimPars`) and values that vary by level or in time (`ss.by`, `ss.timeseries`, `ss.seasonal`)
- `presentation.py`: How a declared model is shown: `describe()`, `diagram()`, `equations()`, and its model spec
- `products.py`: Vaccine and treatment deployment
- `pregnancy.py`: Pregnancy as declared states and transitions; provides events (`conception`, `delivery`) and states (`P`) for other modules
- `run.py`: Running multiple simulations (`ss.MultiSim`, `ss.parallel`)
- `samples.py`: Storage for large-scale simulation results
- `settings.py`: Global options (`ss.options`) and plotting style
- `solvers/`: The engines that run declared models: `engine.py` (what they share), `mixing.py` (who meets whom in a compartmental run), `compartmental.py`, `agents.py`, `crossing.py` (diseases tracked jointly), and `choose.py` (which solver, and why: `ss.solvers.choose_solver()`); every name is also `ss.solvers.X`
- `spec.py`: The model spec, and models written as code (`ss.to_code()`, `ss.diff()`)
- `time.py`: Durations and rates (`ss.dur`, `ss.days`, `ss.peryear`, `ss.probperyear`, etc.)
- `timeline.py`: Time coordination between modules (`ss.Timeline`)
- `timeplot.py`: Plotting dates and durations with Matplotlib
- `transitions.py`: States and transitions (`ss.State`, `ss.transmission()`, `ss.progression()`, ...); the names of the modules above that it builds on are also `ss.transitions.X`
- `utils.py`: Helper functions
- `interop/`: Importers and exporters for other modeling frameworks

### Library (`starsim/library/`)
Example and reference modules that build on core Starsim but are not part of it, available via `import starsim.library as ssl` (e.g. `ssl.Cholera` or `ssl.diseases.Cholera`):
- `diseases/`: HIV, cholera, Ebola, and measles
- `mnch/`: Maternal, newborn, and child health (fetal health, maternal infections, neonatal sepsis)
- `networks/`: Household, spatial, and theoretical (e.g. Erdős–Rényi) networks

Core disease base classes (Disease, Infection, NCD, SIR, SIS, SEIR) live in `starsim/diseases.py`.

### Testing (`tests/`)
- `test_*.py`: Main test files
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
