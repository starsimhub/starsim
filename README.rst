Starsim
=======

Starsim is an agent-based modeling framework for simulating the spread of diseases among agents via dynamic transmission networks. Starsim supports the co-transmission of multiple diseases at once, capturing how they interact biologically and behaviorally. Users can also include non-infectious diseases either on their own or as factors affecting infectious diseases. Starsim allows detailed modeling of mother-child relationships from conception to study birth-related diseases. Additionally, Starsim lets users compare different intervention strategies, like vaccines or treatments, to see their impact through various delivery methods such as mass campaigns or targeted outreach.

Examples of diseases that have already been implemented in Starsim include sexually transmitted infections (HIV, HPV, and syphilis, including co-transmission), respiratory infections (tuberculosis and RSV), plus other diseases (Ebola and cholera) and underlying determinants of health (such as malnutrition).

Note: Starsim is a general-purpose, multi-disease framework that builds on our previous suite of disease-specific models, which included `Covasim <https://covasim.org>`_, `HPVsim <https://hpvsim.org>`_, and `FPsim <https://fpsim.org>`_. In cases where a distinction needs to be made, Starsim is also known as "the Starsim framework" or "Starsim Core", while this collection of other models is known as the "Starsim suite".

For more information about Starsim, please see the `documentation <https://docs.starsim.org>`__. Information about Starsim for R is available at `r.starsim.org <https://r.starsim.org>`__.


Requirements
------------

Python 3.9-3.12 or R.

We recommend, but do not require, installing Starsim in a virtual environment, such as `Miniconda <https://docs.anaconda.com/miniconda/>`__.


Installation
------------

Python
~~~~~~

Starsim is most easily installed via PyPI: ``pip install starsim``.

Starsim can also be installed locally. To do this, clone first this repository, then run ``pip install -e .`` (don't forget the dot at the end!).

R
~
R-Starsim is still under development. You can install it with::

  # install.packages("devtools")
  devtools::install_github("starsimhub/rstarsim")
  library(starsim)
  init_starsim()


Usage and documentation
-----------------------

Full documentation, including tutorials and an API reference, is available at https://docs.starsim.org. 

You can run a simple demo via::

  import starsim as ss
  ss.demo()

Here is a slightly more realistic example of an SIR model with random connections between agents::

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

More usage examples are available in the tutorials, as well as the ``tests`` folder.


Starsim structure
-----------------

All core model code is located in the ``starsim`` subfolder; standard usage is ``import starsim as ss``.

The model consists of core classes including ``Sim``, ``People``, ``Disease``, ``Network``, ``Intervention``, and more. These classes contain methods for running, building simple or dynamic networks, generating random numbers, calculating results, plotting, etc.

The submodules of the Starsim folder are as follows:

• ``arrays.py``: Classes to handle store and update states for people in networks in the simulation including living, mother, child, susceptible, infected, inoculated, recovered, etc.
• ``calibration.py``: Class to handle automated calibration of the model to data.
• ``calib_components.py``: Classes to handle additional operations related to calibration (e.g., the likelihood function).
•	``demographics.py``: Classes to transform initial condition input parameters for use in building and utilizing networks.
•	``disease.py``: Classes to manage infection rate of spread, prevalence, waning effects, and other parameters for specific diseases.
•	``distributions.py``: Classes that handle statistical distributions used throughout Starsim to produce random numbers.
•	``interventions.py``: The Intervention class, for adding interventions and dynamically modifying parameters, and classes for each of the specific interventions derived from it. The Analyzers class (for performing analyses on the sim while it's running), and other classes and functions for analyzing simulations.
• ``loop.py``: The logic for the main simulation integration loop.
•	``modules.py``: Class to handle "module" logic, such as updates (diseases, networks, etc).
•	``networks.py``: Classes for creating simple and dynamic networks of people based on input parameters.
•	``parameters.py``: Classes for creating the simulation parameters.
•	``people.py``: The People class, for handling updates of state for each person.
•	``products.py``: Classes to manage the deployment of vaccines and treatments.
•	``results.py``: Classes to analyze and save results from simulations.
•	``run.py``: Classes for running simulations (e.g. parallel runs and the Scenarios and MultiSim classes).
•	``samples.py``: Class to store data from a large number of simulations.
•	``settings.py``: User-customizable options for Starsim (e.g. default font size).
•	``sim.py``: The Sim class, which performs most of the heavy lifting: initializing the model, running, and plotting.
• ``time.py``: The Time class, which coordinates time between the Sim and different modules.
•	``utils.py``: Helper functions.
•	``version.py``: Version, date, and license information.

The ``diseases`` folder within the Starsim package contains definitions of different types of diseases, including STIs, Ebola, and cholera.


Contributing
------------

Questions or comments can be directed to `info@starsim.org <mailto:info@starsim.org>`__ , or on this project’s `GitHub <https://github.com/starsimhub/starsim>`__ page. Full information about Starsim is provided in the `documentation <https://docs.starsim.org>`__.


Disclaimer
----------

The code in this repository was developed by `IDM <https://idmod.org>`_, the `Burnet Institute <https://burnet.edu.au>`_, and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
