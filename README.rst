Starsim
=======

**Warning! Starsim is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not ready to be used for real research or policy questions.**

Starsim is an agent-based disease modeling framework in which users can design and configure simulations of pathogens that progress over time within each agent and pass from one agent to the next along dynamic transmission networks. The framework explicitly supports co-transmission of multiple pathogens, allowing users to concurrently simulate several diseases while capturing behavioral and biological interactions. Non-communicable diseases can easily be included as well, either as a co-factor for transmissible pathogens or as an independent exploration. Detailed modeling of mother-child relationships can be simulated from the timepoint of conception, enabling study of congenital diseases and associated birth outcomes. Finally, Starsim facilitates the comparison of one or more intervention scenarios to a baseline scenario in evaluating the impact of various products like vaccines, therapeutics, and novel diagnostics delivered via flexible routes including mass campaigns, screen and treat, and targeted outreach.

The framework is appropriate for simulating one or more sexually transmitted infections (including syphilis, gonorrhea, chlamydia, HPV, and HIV), respiratory infections (like RSV and tuberculosis), and other diseases and underlying determinants (such as Ebola, diabetes, and malnutrition).


Installation
------------

To install, clone this repository, then run ``pip install -e .`` (don't forget the dot!) in this folder to install ``starsim`` and its dependencies. This will make ``starsim`` available on the Python path.


Usage and documentation
-----------------------

Documentation is available at https://docs.starsim.org. 

Usage examples are available in the ``tests`` folder.


Contributing
------------

If you wish to contribute, please see the code of conduct and contributing documents.


Disclaimer
----------

The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.


