==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1

Version 0.1.3 (2024-01-22)
--------------------------
- Read in age distributions for people initializations 
- *GitHub info*: PR `205 <https://github.com/amath-idm/stisim/pull/205>`_


Version 0.1.2 (2024-01-19)
--------------------------
- Functionality for converting birth & fertility data to a callable parameter within SciPy distributions
- *GitHub info*: PR `203 <https://github.com/amath-idm/stisim/pull/203>`_


Version 0.1.1 (2024-01-12)
--------------------------
- Improving performance of MultiRNG
- Now factoring the timestep, ``dt``, into transmission calculations
- *GitHub info*: PRs `204 <https://github.com/amath-idm/stisim/pull/204>`_


Version 0.1.0 (2023-12-10)
--------------------------
- Allows SciPy distributions to be used as parameters
- Optionally use multiple random number streams and other tricks to maintain coherence between simulations
- Adding functionality to convert death rate data to a callable parameter within a SciPy distribution
- *GitHub info*: PRs `170 <https://github.com/amath-idm/stisim/pull/170>`_ and `202 <https://github.com/amath-idm/stisim/pull/202>`_


Version 0.0.8 (2023-10-04)
--------------------------
- Enable removing people from simulations following death
- *GitHub info*: PR `121 <https://github.com/amath-idm/stisim/pull/121>`_


Version 0.0.7 (2023-09-08)
--------------------------
- Refactor distributions to use new Distribution class
- *GitHub info*: PR `112 <https://github.com/amath-idm/stisim/pull/112>`_


Version 0.0.6 (2023-08-30)
--------------------------
- Changes agent IDs from index-based to UID-based
- Allows states to store their own data and live within modules
- *GitHub info*: PR `88 <https://github.com/amath-idm/stisim/pull/88>`_


Version 0.0.5 (2023-08-29)
--------------------------
- Refactor file structure 
- *GitHub info*: PRs `77 <https://github.com/amath-idm/stisim/pull/77>`_ and `86 <https://github.com/amath-idm/stisim/pull/86>`_


Version 0.0.2 (2023-06-29)
--------------------------
- Adds in basic *sim functionality
- *GitHub info*: PR `17 <https://github.com/amath-idm/stisim/pull/17>`__


Version 0.0.1 (2023-06-22)
--------------------------
- Initial version.
