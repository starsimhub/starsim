Starsim
=======

**Warning! Starsim is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not ready to be used for real research or policy questions.**

Starsim is an agent-based disease modeling framework in which users can design and configure simulations of pathogens that progress over time within each agent and pass from one agent to the next along dynamic transmission networks. The framework explicitly supports co-transmission of multiple pathogens, allowing users to concurrently simulate several diseases while capturing behavioral and biological interactions. Non-communicable diseases can easily be included as well, either as a co-factor for transmissible pathogens or as an independent exploration. Detailed modeling of mother-child relationships can be simulated from the timepoint of conception, enabling study of congenital diseases and associated birth outcomes. Finally, Starsim facilitates the comparison of one or more intervention scenarios to a baseline scenario in evaluating the impact of various products like vaccines, therapeutics, and novel diagnostics delivered via flexible routes including mass campaigns, screen and treat, and targeted outreach.

The framework is appropriate for simulating one or more sexually transmitted infections (including syphilis, gonorrhea, chlamydia, HPV, and HIV), respiratory infections (like RSV and tuberculosis), and other diseases and underlying determinants (such as Ebola, diabetes, and malnutrition).


Background
------------

Starsim leveraged advances made in Covasim, HPVsim, FPsim, and SynthPops to enable the modeling of additional diseases and conditions. 

The goal of Starsim is to enable modeling of multiple diseases, co-infections, and networks, as well as the evolution of population conditions.


Requirements
------------

Python 3.9-3.11 (64-bit). (Note: Python 2.7 and Python 3.12 are not supported, the latter being due to Numba not supporting Python 3.12 at the time of writing.) 

We also recommend, but do not require, installing Starsim in a virtual environment. For more information, see documentation, e.g. Anaconda, Anaconda. Cloud, Google Colab, and GitHub Codespaces.


Installation
------------

Starsim is most easily installed via PyPI: ``pip install starsim``.

Starsim can also be installed locally. To do this, clone first this repository, then run ``pip install -e .`` (don't forget the dot at the end!).


Quick start guide
-----------------

If everything is working, the following Python commands will run a simulation with the simplest version of a Starsim model. We’ll make a version of a classic SIR model::

  import starsim as ss
  
  #Define the parameters
  
  pars = dict(
  	n_agents = 5_000, # Number of agents to simulate
  	networks = dict(  # *Networks* add detail on how agents interact w/ each other
  	type = 'random',  # Here, we use a 'random' network
  	n_contacts = 10   # Each person has an average of 10 contacts w/ other people  
  	),
  
  diseases = dict(    # *Diseases* add detail on what diseases to model
  	type = 'sir',     # Here, we're creating an SIR disease
  	init_prev = 0.1,  # Proportion of the population initially infected
  	beta = 0.5,       # Probability of transmission between contacts
  	)
  )
  
  # Make the sim, run and plot
  sim = ss.Sim(pars)
  sim.run()
  sim.plot()


Usage and documentation
-----------------------

Documentation is available at https://docs.starsim.org. 

Usage examples are available in the ``tests`` folder.

Model structure
---------------

All core model code is located in the ``starsim`` subfolder; standard usage is ``import starsim as ss``. The data subfolder is described below.

The model consists of core classes including Sim, Run, People, State, Network, Connectors, Analyzers, Interventions, Results, and more. These classes contain methods for running, building simple or dynamic networks, generating random numbers, calculating results, plotting, etc.

The structure of the starsim folder is as follows, roughly in the order in which the modules are imported, building from most fundamental to most complex:

•	``connectors.py``: Functions that handle connections between disease modules.
•	``demographics.py``: Functions to transform initial condition input parameters for use in building and utilizing networks.
•	``disease.py``: Functions to manage infection rate of spread, prevalence, waning effects, and other parameters for specific diseases.
•	``distributions.py``: Functions that handle SciPy library statistical distributions used throughout Starsim.
•	``interventions.py``: The Intervention class, for adding interventions and dynamically modifying parameters, and classes for each of the specific interventions derived from it. The Analyzers class (for performing analyses on the sim while it’s running), and other classes and functions for analyzing simulations.
•	``modules.py``: Functions to handle disease-specific parameters.
•	``network.py``: Functions for creating simple and dynamic networks of people based on input parameters.
•	``parameters.py``: Functions for creating the parameters dictionary and loading the input data.
•	``people.py``: The People class, for handling updates of state for each person.
•	``products.py``: Functions manage the deployment of vaccines and treatments.
•	``results.py``: Functions to analyze and save results from simulations.
•	``run.py``: Functions for running simulations (e.g. parallel runs and the Scenarios and MultiSim classes).
•	``plotting.py``: Plotting scripts, including Plotly graphs for the webapp (used in other Covasim classes, and hence defined first).
•	``samples.py``: Functions to store data from a large number of simulations.
•	``settings.py``: User-customizable options for Starsim (e.g. default font size).
•	``sim.py``: The Sim class, which performs most of the heavy lifting: initializing the model, running, and plotting.
•	``states.py``: Functions to handle store and update states for people in networks in the simulation including living, mother, child, susceptible, infected, inoculated, recovered, etc.
•	``utils.py``: Functions for choosing random numbers, many based on Numba, plus other helper functions.
•	``version.py``: Version, date, and license information.

The ``diseases`` folder within the Starsim package contains loading scripts for the epidemiological data specific to each respective disease.

Other folders
------------

Please see the README in each subfolder for more information.


API Reference
------------

A list of Starsim’s full API, including all functions and classes is available at https://docs.starsim.org.


Tutorials
------------

This IDM Starsim Tutorials website contains demonstrations of simple Starsim usage structured as follows: 

•	T1 - Getting started
•	T2 - How to build your model
•	T3 - Demographics
•	T4 - Networks
•	T5 - Diseases
•	T6 - Interventions


Contributing
------------

If you wish to contribute, please see the code of conduct and contributing documents.


Disclaimer
----------

The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.


