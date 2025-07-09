# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Starsim is an agent-based modeling framework for simulating disease spread among agents via dynamic transmission networks. It supports co-transmission of multiple diseases and detailed modeling of intervention strategies.

## Core Architecture

The framework follows a modular design with these key components:

- **Sim**: Main simulation class that orchestrates all modules and runs simulations (`starsim/sim.py`)
- **People**: Manages individual agents and their state updates (`starsim/people.py`)
- **Networks**: Defines how agents interact with each other (`starsim/networks.py`)
- **Diseases**: Models disease transmission and progression (`starsim/disease.py` + `starsim/diseases/`)
- **Interventions**: Implements prevention/treatment strategies (`starsim/interventions.py`)
- **Demographics**: Handles population dynamics like births/deaths (`starsim/demographics.py`)
- **Results**: Analyzes and stores simulation outputs (`starsim/results.py`)

The main simulation loop is orchestrated by `starsim/loop.py`, while parameter management is handled by `starsim/parameters.py`. Random number generation and statistical distributions are centralized in `starsim/distributions.py`.

## Development Commands

### Running Tests
```bash
# Run all tests with parallel execution
cd tests && ./run_tests

# Run specific test file
cd tests && pytest test_sim.py

# Run tests with coverage
cd tests && ./check_coverage
```

### Code Quality
```bash
# Check code style with pylint
cd tests && ./check_style
```

### Documentation
```bash
# Build documentation (requires quarto)
cd docs && ./build_docs

# Execute tutorials only
./run_tutorials

# Build with different execution modes:
./build_docs jupyter  # Default: parallel execution with jcache
./build_docs quarto   # Serial execution with quarto
./build_docs quarto --cache-refresh  # Refresh cache
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

## Project Structure

### Core Modules (`starsim/`)
- `arrays.py`: State management arrays for people/networks
- `calibration.py` & `calib_components.py`: Model calibration to data
- `distributions.py`: Statistical distributions and random number generation
- `loop.py`: Main simulation integration loop
- `modules.py`: Base module classes and update logic
- `products.py`: Vaccine and treatment deployment
- `run.py`: Parallel simulation execution (MultiSim, Scenarios)
- `samples.py`: Storage for large-scale simulation results
- `time.py`: Time coordination between modules
- `utils.py`: Helper functions

### Disease Models (`starsim/diseases/`)
Pre-implemented disease models including HIV, syphilis, gonorrhea, cholera, Ebola, measles, and SIR/SIS models.

### Testing (`tests/`)
- `test_*.py`: Main test files
- `devtests/`: Development testing scripts
- `run_tests`: Main test runner script
- `pytest.ini`: Pytest configuration

### Documentation (`docs/`)
- `tutorials/`: Jupyter notebook tutorials
- `user_guide/`: User guide notebooks  
- `api/`: API documentation source files
- Quarto-based documentation system

## Key Conventions

- Uses `sciris` library extensively for utilities and plotting
- Follows Google Python style guide with project-specific exceptions
- Tests use pytest with parallel execution via `pytest-xdist`
- Documentation built with Quarto and executed via Jupyter
- Random number generation is centralized and uses Numba for performance

## Module Architecture

The framework uses a modular architecture where all components inherit from base classes:

- **Base class hierarchy**: All modules inherit from `ss.Base` → `ss.Module` (defined in `modules.py`)
- **Module types**: Networks, Demographics, Diseases, Interventions, Analyzers, Connectors
- **Module registration**: Modules are automatically discovered and registered via the `find_modules()` function
- **Integration loop**: The `Loop` class (in `loop.py`) orchestrates module execution during simulation
- **State management**: People and network states are managed through specialized array classes (`arrays.py`)

## Key Dependencies

- **Sciris**: Core utility library for data structures, plotting, and utilities
- **Numba**: Used for performance-critical code sections and random number generation
- **NetworkX**: Network analysis and manipulation
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn**: Plotting and visualization

## Important Notes

- Follow the style guide here: https://github.com/starsimhub/styleguide/blob/main/README.rst
- Use built-in Starsim plotting commands if possible, only falling back to Matplotlib if necessary
- Use Sciris where possible to shorten commands
- When creating a multiline dictionary, put a space around the equals for arguments (as if it were a class)
- Set `SCIRIS_BACKEND='agg'` environment variable when running tests to prevent figure display
- The framework requires Python 3.9-3.13
- Uses Numba for performance-critical code sections
- All modules inherit from base classes in `modules.py`
- Parameter handling is centralized through `parameters.py` and `SimPars` class
- By default, use "Sentence case", not "Title Case", for headings (e.g. Markdown headings, table column headings, etc)