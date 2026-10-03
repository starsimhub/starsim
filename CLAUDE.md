# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Starsim is an agent-based modeling framework for simulating disease spread among agents via dynamic transmission networks. It supports co-transmission of multiple diseases and detailed modeling of intervention strategies.

For a compact index of the whole public API (signatures, summaries, and default parameters), see `docs/llms.txt` (or `docs/llms-full.txt`, which adds an example for each entry), also published at https://docs.starsim.org/llms.txt. The equivalent for Sciris is at https://docs.sciris.org/llms.txt.

## Core architecture

The framework follows a modular design with these key components:

- **Sim**: Main simulation class that orchestrates all modules and runs simulations (`starsim/sim.py`)
- **People**: Manages individual agents and their state updates (`starsim/people.py`)
- **Networks**: Defines how agents interact with each other (`starsim/networks.py`)
- **Diseases**: Models disease transmission and progression (`starsim/diseases.py`)
- **Interventions**: Implements prevention/treatment strategies (`starsim/interventions.py`)
- **Demographics**: Handles population dynamics like births/deaths (`starsim/demographics.py`)
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

# Regenerate the API index (api.json, llms.txt, llms-full.txt) after changing the public API; tests/test_api.py fails if it is out of date
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
- `calibration.py`: Model calibration to data
- `connectors.py`: Connectors between modules (e.g. `ss.seasonality`)
- `debugtools.py`: Profiling, debugging, and mock objects for testing
- `distributions.py`: Statistical distributions and random number generation
- `loop.py`: Main simulation integration loop
- `modules.py`: Base module classes and update logic
- `products.py`: Vaccine and treatment deployment
- `run.py`: Running multiple simulations (`ss.MultiSim`, `ss.parallel`)
- `samples.py`: Storage for large-scale simulation results
- `settings.py`: Global options (`ss.options`) and plotting style
- `time.py`: Dates, durations, and rates (`ss.date`, `ss.dur`, `ss.peryear`, etc.)
- `timeline.py`: Time coordination between modules (`ss.Timeline`)
- `utils.py`: Helper functions

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
