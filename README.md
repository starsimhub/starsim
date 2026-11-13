# Starsim

[![PyPI version](https://badgen.net/pypi/v/starsim/?color=blue)](https://pypi.org/project/starsim)
[![Downloads](https://static.pepy.tech/personalized-badge/starsim?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/starsim)
[![Tests](https://github.com/starsimhub/starsim/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/starsimhub/starsim/actions/workflows/tests.yml)

> **Note: Starsim v4 is now available.** It's quite different from v3, with a new and simpler syntax for creating diseases, and support for compartmental and geospatial models. Models written for v3 will need to be [migrated](https://docs.starsim.org/v3v4); see the [changelog](https://starsim.org/v4/whatsnew.html) for details, or [contact us](mailto:info@starsim.org) for help. To continue using [Starsim v3](https://starsim.org/v3), use `pip install starsim<4`; the [v3 docs](https://starsim.org/v3) are also still available. But really, please upgrade to v4! You'll be glad you did!

[Starsim](https://starsim.org) is a fast, flexible disease modeling framework, from compartments to agents. It supports modeling multiple co-transmitting diseases, including infectious and non-infectious diseases plus other health states (such as pregnancy and birth), and captures their biological and behavioral interactions. It lets users compare intervention strategies, such as vaccines or treatments, across delivery methods like mass campaigns or targeted outreach. It also includes built-in tools for scenario analyses and calibrating models to real-world data.

Starsim builds on our previous suite of disease-specific models, including [Covasim](https://covasim.org) and [HPVsim](https://hpvsim.org), as well as many other disease modeling frameworks (see below). It can model different levels of detail, including agent-based, metapopulation, and deterministic or stochastic compartmental modeling. It is designed to be simple and fast enough to enable rapid or exploratory analyses (from project start to calibrated model and scenario analyses in hours to days), while also being flexible enough to support complex, long-term research projects (such as intrahost immunity, data-driven spatial network transmission, and health economics optimization).

For more information, see the [documentation](https://docs.starsim.org) or the [code](https://github.com/starsimhub/starsim). For Starsim for R, see [r.starsim.org](https://r.starsim.org). For Starsim's AI tools, see [ai.starsim.org](https://ai.starsim.org).


## Starsim features

Starsim includes an extensive library of features aimed to support a broad diversity of disease modeling needs. The full set is covered in the [documentation](https://docs.starsim.org/user_guide.html). Because the features are modular and written in Python, you can mix-and-match Starsim features with custom-written code, and you only use what you need. Highlights include:

- **Modeling modalities:** Switch between compartmental, stochastic compartmental, metapopulation, and agent-based modeling using a flag (`solver='abm'`) without needing to rewrite any code ([solvers](https://starsim.org/v4/user_guide/running_solvers.html));
- **Performance:** Numeric kernels are heavily optimized, with Numba and Rust kernels that give the same numbers, plus GPU support (via JAX) ([performance](https://starsim.org/v4/user_guide/running_performance.html));
- **Diseases:** A large library of already-implemented diseases including COVID, flu, measles, malaria, syphilis, gonorrhea, cholera, Ebola and many others (plus HIV, TB, COVID, HPV, chlamydia, typhoid, and others in the models below); diseases support coinfection between different diseases, as well as multiple strains of a single disease ([library diseases](https://starsim.org/v4/user_guide/library_diseases.html), [diseases](https://starsim.org/v4/user_guide/building_diseases.html));
- **Networks:** Contact networks including age-stratified contact matrices, sexual (including male-female, male-male, and commercial), respiratory (household/workplace/school); also supports other forms of disease transmission including environmental and vector-borne ([networks](https://starsim.org/v4/user_guide/populations_mixing.html), [library networks](https://starsim.org/v4/user_guide/library_networks.html));
- **Interventions:** Flexible options for representing interventions, including screening, diagnostics, treatments, vaccination, and care cascades; health system considerations such as prioritization, capacity constraints, and clinic visits ([interventions](https://starsim.org/v4/user_guide/interventions_interventions.html), [products](https://starsim.org/v4/user_guide/interventions_products.html));
- **Costing:** Support for intervention- and disease-related costs, health outcomes like disability-adjusted life years (DALYs), and budget optimization ([costs](https://starsim.org/v4/user_guide/library_econ.html), [optimization](https://starsim.org/v4/user_guide/calibration_optimization.html));
- **Time:** Unit-aware time and date handling, including natural dates for data and plotting, module-specific timesteps, and timestep-aware conversion of rates and probabilities ([time](https://starsim.org/v4/user_guide/building_time.html));
- **Data and places:** Extensive built-in data, including population and age structures for 187 countries and territories, contact matrices (Prem et al., 2021), and geospatial boundaries, with maps and animations of results by place ([population data](https://starsim.org/v4/user_guide/populations_places.html), [geography](https://starsim.org/v4/user_guide/populations_places.html));
- **Fitting:** Automatic gradients for compartmental models, plus Bayesian fitting and calibrating algorithms including no-U-turn sampling (NUTS) and the tree-structured Parzen estimator (Optuna) ([fitting](https://starsim.org/v4/user_guide/calibration_fitting.html), [methods](https://starsim.org/v4/user_guide/calibration_methods.html));
- **Scientific rigor:** Automatic checking of units to prevent common programming errors; common random numbers that allow exact comparison between even stochastic model runs ([time](https://starsim.org/v4/user_guide/building_time.html), [random numbers](https://starsim.org/v4/user_guide/mechanics_random.html));
- **Intercompatibility:** Automated import from more than a dozen other disease modeling frameworks, and export to SBML and odin ([import and export](https://starsim.org/v4/user_guide/building_import.html)); interoperable with other disease modeling tools like [GeoPops](https://github.com/starsimhub/geopops) and [HubVerse](https://hubverse.io/);
- **AI support:** Extensive set of skills and plugins that help implement Starsim models ([AI integration](#ai-integration)), including no-code chat interfaces for policymakers ([EpiChat](https://github.com/starsimhub/epichat)).
<!-- V4-PREVIEW: restore the starsim.org/v4 links above to docs.starsim.org for normal publishing -->

Starsim "stands on the shoulders of giants", and aims to build on the best aspects of other disease modeling frameworks. It is especially indebted to [EpiModel](https://www.epimodel.org/), [MEmilio](https://github.com/SciCompMod/memilio), [SimInf](https://github.com/stewid/SimInf), [summer2](https://github.com/monash-emu/summer2), [camdl](https://github.com/vsbuffalo/camdl), [epidemics](https://epiverse-trace.github.io/epidemics/), and [epydemix](https://github.com/epistorm/epydemix) (see the [full list](https://starsim.org/v4/user_guide/intro_starsim.html)). If there's a feature you think Starsim should support, please [get in touch](mailto:info@starsim.org).


## Starsim models

Starsim is the basis of several other established disease modeling frameworks. These include:

- [Covasim](https://covasim.org), for modeling COVID-19;
- [HPVsim](https://hpvsim.org), for modeling human papillomavirus and cervical cancer;
- [STIsim](https://stisim.org), for modeling HIV, syphilis, gonorrhea, chlamydia, trichomoniasis, and bacterial vaginosis (including co-transmission);
- [TBsim](https://starsim.org/tbsim), for modeling tuberculosis, including drug resistance;
- [FPsim](https://fpsim.org), for modeling family planning and women's life courses.
- [Typhoidsim](https://starsim.org/typhoidsim/), for modeling typhoid, including environmental transmission.

A full list of diseases modeled using Starsim is available in the [documentation](https://starsim.org/v4/user_guide/intro_starsim.html#models-built-with-starsim).


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

JAX is recommended if you'll be calibrating compartmental models -- otherwise you can skip it, i.e. `pip install starsim`.

`starsim init` compiles numeric kernels if needed (usually it's not). It also asks if you want to install Starsim's AI skills (so you can type `/starsim` in your agent to use them). To do either the compile or the AI install, run `starsim init compile` or `starsim init ai`.

You can also of course install with [uv](https://github.com/astral-sh/uv):
```sh
uv init example
cd example
uv add "starsim[jax]"
```

To work on Starsim itself, clone this repository, then run:
```sh
pip install -e .
starsim init dev
```

This installs the optional dependencies for testing and documentation, and JAX; installs Rust if it is missing; and builds Starsim's kernels. See the [developer guide](https://docs.starsim.org/contributing/developer.html) for details.

### R

R-Starsim can be installed with:

```R
# install.packages("devtools")
devtools::install_github("starsimhub/rstarsim")
library(starsim)
init_starsim()
```

See [r.starsim.org](https://r.starsim.org) for more information.


## Documentation

Full documentation, including tutorials and an API reference, is available at [docs.starsim.org](https://docs.starsim.org).

## Usage

You can run a simple demo via:

```py
import starsim as ss
ss.demo()
```

Here's a minimal SIR model, with an infection rate of β = 0.3 per day and an infectious period of 10 days, written much as you'd write it on paper:

```py
import starsim as ss

S, I, R = ss.states('S I R') # Create the SIR states
infection = S >> I @ 0.3*I # From S to I, at a rate of β times the fraction infectious
recovery  = I >> R @ ss.days(10) # From I to R, after 10 days on average
sir = ss.Disease(infection, recovery) # Create the disease from the transitions

sim = ss.Sim(sir)
sim.run()
sim.plot()
```

The same model, written out in full:

```py
import starsim as ss

S = ss.State(label='Susceptible')
I = ss.State(label='Infectious')
R = ss.State(name='Recovered')
beta = ss.perday(0.3)
dur_inf = ss.expon(mean=ss.days(10))
sir = ss.Disease(
    name = 'SIR',
    transitions = [
        ss.Transition(source=S, dest=I, rate=beta*I, name='infection'),
        ss.Transition(source=I, dest=R, dur=dur_inf, name='recovery'),
    ],
)

sim = ss.Sim(diseases=sir, solver='ode')
sim.run()
sim.plot()
```

These run as a compartmental model (an ODE). To run the same model as agents, give the number of agents, e.g. `ss.Sim(sir, n_agents=5_000)`.

You can also run the same example from R:

```R
library(starsim)

states("S I R")
sir <- Disease(
  infection = Transition(S, I, rate = 0.3 * I), # From S to I, at a rate of β times the fraction infectious
  recovery  = Transition(I, R, dur = days(10))  # From I to R, after 10 days on average
)

sim <- Sim(sir)
sim$run()
sim$plot()
```

More usage examples are available in the tutorials and user guide.


## AI integration

Starsim comes with skills for AI coding agents (Claude Code, Codex, and others), which always match the version installed: how to write, check, and run a model, and how to import a model from another framework or upgrade one from Starsim v3. `starsim init ai` links them into your project, and `starsim ai install --global` sets them up for all your projects instead. You can also install them with other standard tools, such as `uvx library-skills`. `starsim ai "how would I model a leaky vaccine?"` asks your agent a question about Starsim (`-i` for an interactive session), and `starsim help RandomNet` looks a name or term up in the docs, without an agent. For anything else, [llms.txt](https://docs.starsim.org/llms.txt) is a map of the docs and the API for agents.


## Contributing

We welcome contributions of all kinds! Questions or comments can be directed to [info@starsim.org](mailto:info@starsim.org) , or on this project's [GitHub](https://github.com/starsimhub/starsim) page. Full information about Starsim is provided in the [documentation](https://docs.starsim.org).


## AI disclaimer

We believe in being transparent about AI use. Each page comes with a little note estimating the human vs. AI contribution; hover to get more info. For this page:

[*Authorship*: 🧠90%, 🤖10%](# "Humans wrote everything, except some of the code examples were AI-generated, which were then human-reviewed")
