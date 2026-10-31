# Starsim

[![PyPI version](https://badgen.net/pypi/v/starsim/?color=blue)](https://pypi.org/project/starsim)
[![Downloads](https://static.pepy.tech/personalized-badge/starsim?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/starsim)
[![Tests](https://github.com/starsimhub/starsim/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/starsimhub/starsim/actions/workflows/tests.yml)

[Starsim](https://starsim.org) is fast, flexible disease modeling framework. It supports modeling multiple co-transmitting diseases, including infectious and non-infectious diseases plus other health states (such as pregnancy and birth), and captures their biological and behavioral interactions. It lets users compare intervention strategies, such as vaccines or treatments, across delivery methods like mass campaigns or targeted outreach. It also includes built-in tools for scenario analyses and calibrating models to real-world data.

Starsim builds on our previous suite of disease-specific models, including [Covasim](https://covasim.org) and [HPVsim](https://hpvsim.org), as well as many other disease modeling frameworks (see below). It can model different levels of detail, including agent-based, metapopulation, and deterministic or stochastic compartmental modeling. It is designed to be simple and fast enough to enable rapid or exploratory analyses (from project start to calibrated model and scenario analyses in hours to days), while also being flexible enough to support complex, long-term research projects (such as intrahost immunity, data-driven spatial network transmission, and health economics optimization).

For more information, see the [documentation](https://docs.starsim.org) or the [code](https://github.com/starsimhub/starsim). For Starsim for R, see [r.starsim.org](https://r.starsim.org). For Starsim's AI tools, see [ai.starsim.org](https://ai.starsim.org).


## Starsim features

Starsim includes an extensive library of features aimed to support a broad diversity of disease modeling needs. The full set is listed in the [documentation](https://docs.starsim.org/user_guide/features). Because the features are modular and written in Python, you can mix-and-match Starsim features with custom-written code, and you only use what you need. Highlights include:

- **Modeling modalities:** Switch between compartmental, stochastic compartmental, metapopulation, and agent-based modeling using a flag (`solver='abm'`) without needing to rewrite any code;
- **Performance:** Numeric kernels are heavily optimized, including just-in-time compilation and GPU support (via JAX);
- **Diseases:** A large library of already-implemented diseases including HIV, TB, malaria, COVID, flu, measles, HPV, syphilis, gonorrhea, chlamydia, rotavirus, typhoid, cholera, Ebola and many others; diseases support coinfection between different diseases, as well as multiple strains of a single disease;
- **Networks:** Contact networks including age-stratified contact matrices, sexual (including male-female, male-male, and commercial), respiratory (household/workplace/school); also supports other forms of disease transmission including environmental and vector-borne;
- **Interventions:** Flexible options for representing interventions, including screening, diagnostics, treatments, vaccination, and care cascades; health system considerations such as prioritization, capacity constraints, and clinic visits;
- **Costing:** Support for intervention- and disease-related costs, health outcomes like disability-adjusted life years (DALYs), and budget optimization;
- **Time:** Unit-aware time and date handling, including natural dates for data and plotting, module-specific timesteps, and timestep-aware conversion of rates and probabilities;
- **Data:** Extensive built-in data, including population and age structures for every country, contact matrices (Prem et al., 2021), and geospatial boundaries;
- **Fitting:** Automatic gradients for compartmental models, plus Bayesian fitting and calibrating algorithms including no-U-turn sampling (NUTS) and the tree-structured Parzen estimator (Optuna);
- **Scientific rigor:** Automatic dimension checking to prevent common programming errors; common random numbers that allow exact comparison between even stochastic model runs;
- **Intercompatibility:** Automated import and export from more than a dozen other disease modeling frameworks, and interoperable with other disease modeling tools like [GeoPops](TBC) and [HubVerse](TBC);
- **AI support:** Extensive set of skills and plugins that help implement Starsim models, including no-code chat interfaces for policymakers ([EpiChat](TBC)).

Starsim "stands on the shoulders of giants", and aims to build on the best aspects of other disease modeling frameworks. It is especially indebted to [EpiModel](TBC), [MEmilio](TBC), [SimInf](TBC), [summer2](TBC), [camdl](TBC), [epidemics](TBC), and [epydemix](TBC) (see the [full list](TBC)). If there's a feature you think Starsim should support, please [get in touch](mailto:info@starsim.org).


## Starsim models

Starsim is the basis of several other established disease modeling frameworks. These include:

- [Covasim](https://covasim.org), for modeling COVID-19;
- [HPVsim](https://hpvsim.org), for modeling human papillomavirus and cervical cancer;
- [STIsim](https://stisim.org), for modeling HIV, syphilis, gonorrhea, chlamydia, trichomoniasis, and bacterial vaginosis (including co-transmission);
- [TBsim](https://starsim.org/tbsim), for modeling tuberculosis, including drug resistance;
- [FPsim](https://fpsim.org), for modeling family planning and women's life courses.

A full list of diseases modeled using Starsim is available in the [documentation](https://docs.starsim.org/user_guide/intro_models).


## Installation

If you're new to Python, check out our [getting started guide](https://starsim.org/getting-started/). You can also use our [installer](https://starsim.org/install.sh), which will install Python along with all required dependencies.

### Requirements

Python 3.12-3.14 or R.

We recommend, but do not require, installing Starsim in a virtual environment, such as [uv](https://github.com/astral-sh/uv) or [conda](https://conda-forge.org/download/).

### Python

Starsim is most easily installed via [PyPI](https://pypi.org):
```sh
pip install "starsim[jax]"
starsim init
```

JAX is recommended, for gradients, fitting with them, and long runs. It is optional since it has no builds for Windows on Arm: there, use `pip install starsim`, and everything else works.

`starsim init` compiles Starsim (once, which takes a few minutes and makes the first runs faster) and installs its skill for AI coding agents such as Claude Code and Codex (type `/starsim` to use it). To do just one of these, run `starsim compile` or `starsim ai`.

Or with [uv](https://github.com/astral-sh/uv):
```sh
uv init example
cd example
uv add "starsim[jax]"
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
```

See [r.starsim.org](https://r.starsim.org) for more information.


## Usage and documentation

Full documentation, including tutorials and an API reference, is available at [docs.starsim.org](https://docs.starsim.org).

You can run a simple demo via:

```py
import starsim as ss
ss.demo()
```

Here's how you'd write a minimal compartmental SIR model with an infection rate of 0.3 per step and a mean recovery duration of 10 steps:

```py
import starsim as ss

S, I, R = ss.states(3)
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3*I),
    recovery = ss.Transition(I, R, dur=10)
)

sim = ss.Sim(sir).run()
sim.sir.plot()
```

If you prefer something that looks more like equations rather than code, there's an even more compact notation available:

```py
import starsim as ss

S, I, R = ss.states(3) # Create model states
inf = S >> I @ 0.3*I   # Define infection at rate beta, mediated by I (*S/N implied)
rec = I >> R @ 0.1     # Define spontaneous recovery
ss.Sim(inf, rec).run().plot() # Create, run, and plot the simulation
```

The code above runs as a compartmental model by default, but you can also run it as an agent-based model (ABM) if you prefer. ABMs come with additional flexibility, such as defining explicit contact networks:

```py
import starsim as ss

# Define the parameters
pars = dict(
    n_agents = 5_000,     # Number of agents to simulate
    networks = dict(      # Networks define how agents interact with each other
        type = 'random',  # Use the built-in 'random' network
        n_contacts = 10   # Each person has 10 contacts with other people  
    ),
    diseases = dict(      # Define the disease dynamics
        type = 'sir',     # Use the built-in SIR disease (including built-in parameters)
        init_prev = 0.01, # Proportion of the population initially infected
        beta = 0.05,      # Probability of transmission between contacts
    )
)

# Make the sim, run and plot
sim = ss.Sim(pars)
sim.run() # Runs as an agent-based model since uses an explicit network
sim.plot() # Plot all the sim results
sim.sir.plot() # Plot the standard SIR curves
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

More usage examples are available in the tutorials and user guide.


## AI integration

Starsim-AI includes AI agents, skills, plugins, and MCP servers that ensure your favorite AI-enabled editor/tool is Starsim-aware. These include skills trained on different components of Starsim, as well as general software engineering and debugging skills. For details, see the [Starsim-AI](https://ai.starsim.org) project.


## Contributing

We welcome contributions of all kinds! Questions or comments can be directed to [info@starsim.org](mailto:info@starsim.org) , or on this project's [GitHub](https://github.com/starsimhub/starsim) page. Full information about Starsim is provided in the [documentation](https://docs.starsim.org).


## AI disclaimer

We believe in being transparent about AI use. Each page comes with a little note estimating the human vs. AI contribution; hover to get more info. For this page:

[*Authorship*: 🧠90%, 🤖10%](# "Humans wrote everything, except some of the code examples were AI-generated")
