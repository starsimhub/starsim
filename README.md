# Starsim

[![PyPI version](https://badgen.net/pypi/v/starsim/?color=blue)](https://pypi.org/project/starsim)
[![Downloads](https://static.pepy.tech/personalized-badge/starsim?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/starsim)
[![Tests](https://github.com/starsimhub/starsim/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/starsimhub/starsim/actions/workflows/tests.yml)

[Starsim](https://starsim.org) is an agent-based modeling framework for simulating disease spread through dynamic transmission networks. It supports modeling multiple co-transmitting diseases, including infectious and non-infectious diseases plus other health states (such as pregnancy and birth), and captures their biological and behavioral interactions. It lets users compare intervention strategies, such as vaccines or treatments, across delivery methods like mass campaigns or targeted outreach. It also includes built-in tools for scenario analyses and calibrating models to data.

Starsim builds on our previous suite of disease-specific models, which included [Covasim](https://covasim.org), [HPVsim](https://hpvsim.org), and [FPsim](https://fpsim.org). It can model different levels of detail, including agent-based, metapopulation, and compartmental modeling. It is designed to be simple and fast enough to enable rapid or exploratory analyses (from project start to calibrated model and scenario analyses in days to weeks), while also being flexible enough to support complex, long-term research projects (such as intrahost immunity, data-driven spatial network transmission, and health economics optimization).

Examples of health areas that have already been [modeled using Starsim](https://docs.starsim.org/user_guide/intro_models.html) include [sexually transmitted infections](https://stisim.org) (HIV, HPV, and syphilis, including co-transmission), respiratory infections ([tuberculosis](https://starsim.org/tbsim) and RSV), other infectious diseases (Ebola and cholera), and underlying determinants of health (such as [family planning](https://fpsim.org) and malnutrition). 

For more information, see the [documentation](https://docs.starsim.org). For Starsim for R, see [r.starsim.org](https://r.starsim.org). For Starsim's AI tools, see [ai.starsim.org](https://ai.starsim.org).


## Requirements

Python 3.10-3.14 or R.

We recommend, but do not require, installing Starsim in a virtual environment, such as [uv](https://github.com/astral-sh/uv) or [Miniconda](https://docs.anaconda.com/miniconda/).


## Installation

### Python

Starsim is most easily installed via [PyPI](https://pypi.org):
```sh
pip install starsim
```

Or with [uv](https://github.com/astral-sh/uv):
```sh
uv init example
cd example
uv add starsim
```

Starsim can also be installed locally (including optional dependencies for testing and documentation). To do this, clone first this repository, then run:
```sh
pip install -e .[dev]
```


### R

R-Starsim can be installed with:

```R
# install.packages("devtools")
devtools::install_github("starsimhub/rstarsim")
library(starsim)
init_starsim()
````

See [r.starsim.org](https://r.starsim.org) for more information.

## Usage and documentation

Full documentation, including tutorials and an API reference, is available at [docs.starsim.org](https://docs.starsim.org).

You can run a simple demo via:

```py
import starsim as ss
ss.demo()
```

Here is a slightly more realistic example of an SIR model with random connections between agents:

```py
import starsim as ss

# Define the parameters
pars = dict(
    n_agents = 5_000,     # Number of agents to simulate
    networks = dict(      # Networks define how agents interact w/ each other
        type = 'random',  # Here, we use a 'random' network
        n_contacts = 10   # Each person has 10 contacts with other people  
    ),
    diseases = dict(      # *Diseases* add detail on what diseases to model
        type = 'sir',     # Here, we're creating an SIR disease
        init_prev = 0.01, # Proportion of the population initially infected
        beta = 0.05,      # Probability of transmission between contacts
    )
)

# Make the sim, run and plot
sim = ss.Sim(pars)
sim.run()
sim.plot() # Plot all the sim results
sim.diseases.sir.plot() # Plot the standard SIR curves
```

Or from R:

```R
library(starsim)
load_starsim()

# Define the parameters
pars <- list(
    n_agents = 5000,
    networks = list(type = 'random'),
    diseases = list(type = 'sir', beta = 0.1)
)

# Create, run, and plot the simulation
sim <- ss$Sim(pars)
sim$run()
sim$diseases$sir$plot()
```

More usage examples are available in the tutorials, as well as the `tests` folder.


## AI integration

Starsim-AI includes AI agents, skills, plugins, and MCP servers that ensure your favorite AI-enabled editor/tool is Starsim-aware. These include skills trained on different components of Starsim, as well as general software engineering and debugging skills. For details, see the [Starsim-AI](https://github.com/starsimhub/starsim_ai) project.


## Starsim structure

All core model code is located in the `starsim` subfolder; standard usage is `import starsim as ss`.

The model consists of core classes including `Sim`, `People`, `Disease`, `Network`, `Intervention`, and more. These classes contain methods for running, building simple or dynamic networks, generating random numbers, calculating results, plotting, etc.

The submodules of the Starsim folder are as follows:

- `analyzers.py`: The Analyzers class (for performing analyses on the sim while it's running), and other classes and functions for analyzing simulations.
- `arrays.py`: Classes to handle, store, and update states for people in networks in the simulation including living, mother, child, susceptible, infected, inoculated, recovered, etc.
- `calibration.py`: Class to handle automated calibration of the model to data.
- `connectors.py`: Classes for modulating interactions between modules (e.g. between two diseases).
- `debugtools.py`: Helper functions and classes to aid with debugging model results and performance.
- `demographics.py`: Classes to transform initial condition input parameters for use in building and utilizing networks.
- `diseases.py`: Classes to manage infection rate of spread, prevalence, waning effects, and other parameters for specific diseases.
- `distributions.py`: Classes that handle statistical distributions used throughout Starsim to produce random numbers.
- `interventions.py`: The Intervention class, for adding interventions and dynamically modifying parameters, and classes for each of the specific interventions derived from it. 
- `loop.py`: The logic for the main simulation integration loop.
- `modules.py`: Class to handle "module" logic, such as updates (diseases, networks, etc). 
- `networks.py`: Classes for creating simple and dynamic networks of people based on input parameters.
- `parameters.py`: Classes for creating the simulation parameters.
- `people.py`: The People class, for handling updates of state for each person.
- `products.py`: Classes to manage the deployment of vaccines and treatments.
- `results.py`: Classes to analyze and save results from simulations.
- `run.py`: Classes for running simulations (e.g. parallel runs and the Scenarios and MultiSim classes).
- `samples.py`: Class to store data from a large number of simulations.
- `settings.py`: User-customizable options for Starsim (e.g. default font size).
- `sim.py`: The Sim class, which performs most of the heavy lifting: initializing the model, running, and plotting.
- `time.py`: Time classes, such as dates, durations, probabilities, and frequencies.
- `timeline.py`: The Timeline class, which coordinates time between the Sim and different modules.
- `utils.py`: Helper functions.
- `version.py`: Version, date, and license information.

Starsim also includes a `starsim_examples` folder, which contains definitions of different examples of diseases, including STIs, Ebola, and cholera. **Note**: these are illustrative examples only for demonstrating Starsim usage and functionality; for actual scientific research, please see other Starsim models, e.g. [STIsim](https://stisim.org).

## Contributing

Questions or comments can be directed to [info@starsim.org](mailto:info@starsim.org) , or on this project's [GitHub](https://github.com/starsimhub/starsim) page. Full information about Starsim is provided in the [documentation](https://docs.starsim.org).

## Disclaimer

The code in this repository was developed by [IDM](https://idmod.org), the [Burnet Institute](https://burnet.edu.au), and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
