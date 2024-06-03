Starsim
=======

**Warning! Starsim is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not ready to be used for real research or policy questions.**

Starsim is an agent-based modeling framework in which users can design and configure simulations of diseases (or other health states) that progress over time within each agent and pass from one agent to the next along dynamic transmission networks. The framework explicitly supports co-transmission of multiple pathogens, allowing users to concurrently simulate several diseases while capturing behavioral and biological interactions. Non-communicable diseases can be included as well, either as a co-factor for transmissible pathogens or as an independent exploration. Detailed modeling of mother-child relationships can be simulated from the timepoint of conception, enabling study of congenital diseases and associated birth outcomes. Finally, Starsim facilitates the comparison of one or more intervention scenarios to a baseline scenario in evaluating the impact of various products like vaccines, therapeutics, and novel diagnostics delivered via flexible routes including mass campaigns, screen and treat, and targeted outreach.

The framework is appropriate for simulating sexually transmitted infections (including syphilis, gonorrhea, chlamydia, HPV, and HIV, including co-transmission), respiratory infections (like RSV and tuberculosis), and other diseases and underlying determinants (such as Ebola, diabetes, and malnutrition).

Starsim is a general-purpose modeling framework that is part of the same suite of tools as `Covasim <https://covasim.org>`_, `HPVsim <https://hpvsim.org>`_, and `FPsim <https://fpsim.org>`_.


Requirements
------------

Python 3.9-3.12.

We recommend, but do not require, installing Starsim in a virtual environment, such as `Anaconda <https://www.anaconda.com/products>`__.


Installation
------------

Starsim is most easily installed via PyPI: ``pip install starsim``.

Starsim can also be installed locally. To do this, clone first this repository, then run ``pip install -e .`` (don't forget the dot at the end!).


Usage and documentation
-----------------------

Documentation, including tutorials and an API reference, is available at https://docs.starsim.org. 

If everything is working, the following Python commands will run a simulation with the simplest version of a Starsim model. We'll make a version of a classic SIR model::

  import starsim as ss
  
  # Define the parameters
  pars = dict(
    n_agents = 5_000,   # Number of agents to simulate
    networks = dict(    # *Networks* add detail on how agents interact w/ each other
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

More usage examples are available in the ``tests`` folder.


Model structure
---------------

All core model code is located in the ``starsim`` subfolder; standard usage is ``import starsim as ss``.

The model consists of core classes including Sim, Run, People, State, Network, Connectors, Analyzers, Interventions, Results, and more. These classes contain methods for running, building simple or dynamic networks, generating random numbers, calculating results, plotting, etc.

The structure of the starsim folder is as follows, roughly in the order in which the modules are imported, building from most fundamental to most complex:

•	``demographics.py``: Classes to transform initial condition input parameters for use in building and utilizing networks.
•	``disease.py``: Classes to manage infection rate of spread, prevalence, waning effects, and other parameters for specific diseases.
•	``distributions.py``: Classes that handle statistical distributions used throughout Starsim.
•	``interventions.py``: The Intervention class, for adding interventions and dynamically modifying parameters, and classes for each of the specific interventions derived from it. The Analyzers class (for performing analyses on the sim while it's running), and other classes and functions for analyzing simulations.
•	``modules.py``: Class to handle "module" logic, such as updates (diseases, networks, etc).
•	``network.py``: Classes for creating simple and dynamic networks of people based on input parameters.
•	``parameters.py``: Classes for creating the simulation parameters.
•	``people.py``: The People class, for handling updates of state for each person.
•	``products.py``: Classes to manage the deployment of vaccines and treatments.
•	``results.py``: Classes to analyze and save results from simulations.
•	``run.py``: Classes for running simulations (e.g. parallel runs and the Scenarios and MultiSim classes).
•	``samples.py``: Class to store data from a large number of simulations.
•	``settings.py``: User-customizable options for Starsim (e.g. default font size).
•	``sim.py``: The Sim class, which performs most of the heavy lifting: initializing the model, running, and plotting.
•	``states.py``: Classes to handle store and update states for people in networks in the simulation including living, mother, child, susceptible, infected, inoculated, recovered, etc.
•	``utils.py``: Helper functions.
•	``version.py``: Version, date, and license information.

The ``diseases`` folder within the Starsim package contains loading scripts for the epidemiological data specific to each respective disease.


Contributing
------------

Questions or comments can be directed to `info@starsim.org <mailto:info@starsim.org>`__ , or on this project’s `GitHub <https://github.com/starsimhub/starsim>`__ page. Full information about Starsim is provided in the `documentation <https://docs.starsim.org>`__.


Disclaimer
----------

The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.


