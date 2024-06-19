==========
What's new
==========

.. currentmodule:: starsim

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".


Version 0.5.x (2024-xx-xx)
--------------------------
- If a `People` object is provided to the ``Arr`` constructor, the arrays will be pre-initialized to index the current UIDs in the ``People`` object. This enables construction of temporary ``Arr`` instances that can be used to perform intermediate calculations (e.g., inside ``Intervention.apply()`` or within a module update step)
- Deprecated `Arr(raw=...)` argument to simplify initialization, as in practice the ``raw`` variable is not directly set, and this update also introduces a new pathway for initializating the `raw` attribute


Version 0.5.4 (2024-06-18)
--------------------------
- Adjusted ``RandomNet`` to avoid connections to unborn agents and use random rounding for half edges
- Adds ``get_analyzers`` and ``get_analyzer``
- Refactor how data is pre-processed for births/pregnancy/death rates, giving about a 10% decrease in run time for the STIsim HIV model
- ``BoolArr.uids`` is automatically called when doing set operations on ``uids`` with a ``BoolArr``
- *GitHub info*: PR `555 <https://github.com/starsimhub/starsim/pull/555>`_


Version 0.5.3 (2024-06-10)
--------------------------
- ``ss.uids`` class implements set operators to facilitate combining or otherwise operating on collections of UIDs
- ``FloatArr.isnan`` and ``FloatArr.notnan`` return ``BoolArr`` instances rather than UIDs (so as to facilitate logical operations with other ``BoolArr`` instances, and to align more closely with `np.isnan`)
- ``Arr.true()`` and ``Arr.false()`` are supported for all ``Arr`` subclasses
- ``BoolArr.isnan`` and ``Boolarr.notnan`` are also implemented (although since ``BoolArr`` cannot store NaN values, these always return ``False`` and ``True``, respectively)
- *GitHub info*: PR `544 <https://github.com/starsimhub/starsim/pull/544>`_


Version 0.5.2 (2024-06-04)
--------------------------
- Renames ``network.contacts`` to ``network.edges``.
- For modules (including diseases, networks, etc.), renames ``initialize()`` to ``init_pre()`` and ``init_vals()`` to ``init_post()``.
- Renames ``ss.delta()`` to ``ss.constant()``.
- Allows ``Arr`` objects to be indexed by integer (which are assumed to be UIDs).
- Fixes bug when using callable parameters with ``ss.lognorm_ex()`` and ``ss.lognorm_im()``.
- Fixes bug when initializing ``ss.StaticNet()``.
- Updates default birth rate from 0 to 30 (so ``demographics=True`` is meaningful).
- Adds ``min_age`` and ``max_age`` parameters to the ``Pregnancy`` module (with defaults 15 and 50 years).
- Adds an option for the ``sir_vaccine`` to be all-or-nothing instead of leaky.
- Updates baseline test from HIV to SIR + SIS.
- Fixes issue with infection log not being populated.
- *GitHub info*: PR `527 <https://github.com/starsimhub/starsim/pull/527>`_


Version 0.5.1 (2024-05-15)
--------------------------
- Separates maternal transmission into prenatal and postnatal modules.
- *GitHub info*: PR `509 <https://github.com/starsimhub/starsim/pull/509>`_


Version 0.5.0 (2024-05-14)
--------------------------

Summary
~~~~~~~
All inputs to the sim and modules now use a ``ss.Pars()`` class, which handles updating and validation. It is now not necessary to ever use ``pars=`` (although you still can if you want), so what was previously:

``sim = ss.Sim(pars=dict(diseases='sir', networks='random'))``

is now just:

``sim = ss.Sim(diseases='sir', networks='random')``

Updates happen recursively, so distributions etc. can be flexibly updated.

This has significantly changed how modules are initialized; what was previously:

.. code-block:: python

    def __init__(self, pars=None, **kwargs):

        pars = ss.omergeleft(pars,
            dur_inf = 6,
            init_prev = 0.01,
            p_death = 0.01,
            beta = 0.5,
        )

        par_dists = ss.omergeleft(par_dists,
            dur_inf = ss.lognorm_ex,
            init_prev = ss.bernoulli,
            p_death = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

is now:

.. code-block:: python
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            beta = 0.5,
            init_prev = ss.bernoulli(0.01),
            dur_inf = ss.lognorm_ex(6),
            p_death = ss.bernoulli(0.01),
        )
        self.update_pars(pars, **kwargs)


Parameter changes
~~~~~~~~~~~~~~~~~
- Added a ``ss.Pars`` class (and a ``ss.SimPars`` subclass) that handles parameter creation, updates, and validation.
- Initialization has been moved from ``sim.py`` to ``parameters.py``; ``ss.Sim.convert_plugins()`` has been replaced by ``ss.SimPars.convert_modules()``.
- The key method is ``ss.Pars.update()``, which performs all necessary validation on the parameters being updated.

Initialization changes
~~~~~~~~~~~~~~~~~~~~~~
- Previously, the people were initialized first, then the states were initialized and the values populated, then the modules were initialized, and finally the distributions are initialized. This led to circular logic with the states being initialized based on uninitialized distributions. Now, states and modules are *linked* to the ``People`` and ``Sim`` objects, but further initialization is not done at this step. This ensures all distributions are created but not yet used. Next, distributions are initialized. Finally, the initial values are populated, and everything is initialized.
- New methods supporting these changes include ``ss.link_dists()``, ``dist.link_sim()``, ``dist.link_module()``, ``sim.init_vals()``, ``people.init_vals()``, ``module.init_vals()``, 

Module changes
~~~~~~~~~~~~~~
- Whereas modules previously initialized a dict of parameters and then called ``super().__init__(pars, **kwargs)``, they now call ``super().__init__()`` first, then ``self.default_pars(par1=x, par2=y)``, then finally ``self.update_pars(pars, **kwargs)``.
- What was previously e.g. ``ss.Module(pars=dict(par=x))`` is now ``ss.Module(par=x)``.
- ``par_dists`` has been removed; instead, distributions are specified in the default parameters, and are updated via the ``Pars`` object.
- Modules now contain a link back to the ``Sim`` object. This means that all methods that used to have ``sim`` as an argument now do not, e.g. ``self.update()`` instead of ``self.update(sim)``.
- ``ss.module_map()`` maps different module types to their location in the sim.
- ``ss.find_modules()`` finds all available modules (including subclasses) in Starsim.
- Removed ``ss.dictmerge()`` and ``ss.dictmergeleft`` (now handled by ``ss.Pars.update()``).
- Removed ``ss.get_subclasses()`` and ``ss.all_subclasses()`` (now handled by ``ss.find_modules()``).
- Modules can no longer be initialized with a ``name`` key; it must be ``type`` (e.g. ``dict(type='sir')`` rather than ``dict(name='sir')``.
- Added ``to_json()`` and ``plot()`` methods to ``Module``.
- Removed ``connectors.py``; connectors still exist but as an empty subclass of ``Module``.

People and network changes
~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``BasePeople`` has been removed and merged with ``People``.
- Time parameters (``ti``, ``dt``, etc.) have been removed from ``People``. Use ``sim.ti``, ``sim.dt`` etc. instead. One consequence of this is that ``people.request_death()`` now requires a ``sim`` argument. Another is that network methods (e.g. ``add_pairs()``) now take ``sim`` arguments instead of ``people`` arguments.
- ``SexualNetwork`` is now a subclass of ``DynamicNetwork``.
- Removed ``ss.Networks`` (now just an ``ss.ndict``).
- Network connectors have been removed.
- ``Person`` has been implemented as a slice of ``sim.people[i]``.
- There is a new parameter ``use_aging``; this defaults to ``True`` if demographic modules are supplied, and ``False`` otherwise.

Other changes
~~~~~~~~~~~~~
- Boolean arrays have new methods ``true()``, ``false()``, and ``split()``, which return the UIDs for the ``True`` values (alias to ``arr.uids``), ``False`` values, and both sets of values, respectively. ``ss.bernoulli.split()`` has been added as an alias of ``ss.bernoulli.filter(both=True)``.
- All inputs to a sim are now copied by default. To disable, use ``ss.Sim(..., copy_inputs=False)``.
- There is a new ``Plugin`` class, which contains shared logic for Interventions and Analyzers. It has a ``from_func()``, which will generate an intervention/analyzer from a function.
- Diseases no longer have a default value of ``beta=1`` assigned; beta must be defined explicitly if being used.
- Individual diseases can now be plotted via either e.g. ``sim.plot('hiv')`` or ``sim.diseases.hiv.plot()``.
- Distributions can be created from dicts via ``ss.make_dist()``.
- A new function ``ss.check_sims_match()`` will check if the results of two or more simulations match.
- ``ndict`` values can be accessed through a call; e.g. ``sim.diseases()`` is equivalent to ``sim.diseases.values()``.
- Merged ``test_dcp.py`` and ``test_base.py`` into ``test_other.py``.
- Renamed ``test_simple.py`` to ``test_sim.py``.
- Renamed ``test_dists.py`` to ``test_randomness.py``.
- *GitHub info*: PR `488 <https://github.com/starsimhub/starsim/pull/488>`_


Version 0.4.0 (2024-04-24)
--------------------------
- Replace ``UIDArray``, ``ArrayView``, and ``State`` with ``Arr``, which has different subclasses for different data types (e.g. ``FloatArr``, ``BoolArr``, and ``IndexArr``). States are usually represented by ``BoolArr`` (e.g. ``sir.infected``), while other agent properties are represented by ``FloatArr`` (e.g. ``sir.rel_trans``).
- Arrays that had previously been represented using an integer data type (e.g. ``sir.ti_infected``) are now also ``FloatArr``, to allow the use of ``np.nan``. Integer arrays are supported via ``IndexArr``, but these are only intended for use for slots and UIDs.
- ``Arr`` objects automatically skip over dead (or otherwise removed) agents; the "active" UIDs are stored in ``sim.people.auids``, which is updated when agents are born or die. This array is linked to each ``Arr``, so that e.g. ``sim.people.age.mean()`` will only calculate the mean over alive agents. To access the underlying Numpy array, use ``sim.people.age.raw``.
- ``FloatArr`` has ``isnan``, ``notnan``, and ``notnanvals`` properties. ``BoolArr`` has logical operations defined. For example, ``~people.female`` works, but ``~people.ti_dead`` does not; ``people.ti_dead.notnan`` works, but ``people.female.notnan`` does not.
- UIDs used to be NumPy integer arrays; they are now ``ss.uids`` objects (which is a class, but is lowercase for consistency with ``np.array()``, which it is functionally similar to). Indexing a state by an integer array rather than ``ss.uids()`` now raises an exception, due to the ambiguity involved. To index the underlying array with an integer array, use ``Arr.raw[int_arr]``; to index only the active/alive agents, use ``Arr.values[int_arr]``.
- Dead agents are no longer removed, so ``uid`` always corresponds to the position in the array. This means that no remapping is necessary, which has a significant performance benefit (roughly 2x faster for large numbers of agents).
- Renamed ``omerge`` to ``dictmerge`` and ``omergeleft`` to ``dictmergeleft``.
- *GitHub info*: PR `456 <https://github.com/starsimhub/starsim/pull/456>`_


Version 0.3.4 (2024-04-18)
--------------------------
- Default duration of edges in ``ss.RandomNet`` changed from 1 to 0; this does not matter if ``dt=1``, but does matter with smaller ``dt`` values.
- Removed ``ss.HPVNet``.
- ``new_deaths`` now counted for cholera.
- Crude birth and death rates now take ``dt`` into account.
- The ability to use a centralized random number generator has been restored via ``ss.options(_centralized=True)``; this option not advised, but can be used for testing.
- *GitHub info*: PR `473 <https://github.com/starsimhub/starsim/pull/473>`_


Version 0.3.3 (2024-04-16)
--------------------------
- Changed Ebola model transmission logic.
- Fixed bug with module names not being preserved with multiple initialization.
- *GitHub info*: PR `463 <https://github.com/starsimhub/starsim/pull/463>`_


Version 0.3.2 (2024-04-08)
--------------------------
- Change to syphilis model to permit latent transmission.
- *GitHub info*: PR `450 <https://github.com/starsimhub/starsim/pull/450>`_


Version 0.3.1 (2024-03-31)
--------------------------
- Added SIS model.
- Fixes distribution initialization.
- Allows interventions and analyzers to be functions.
- Tidies up tests.
- Performance improvements in ``UIDArray`` (~3x faster for large numbers of agents).
- *GitHub info*: PR `428 <https://github.com/amath-idm/stisim/pull/428>`_


Version 0.3.0 (2024-03-30)
--------------------------

New RNGs & distributions
~~~~~~~~~~~~~~~~~~~~~~~~
- Replaces ``ss.SingleRNG()``, ``ss.MultiRNG()``, ``ss.ScipyDistribution()``, and ``ss.ScipyHistogram()`` with a single ``ss.Dist()`` class. The ``starsim.random`` and ``starsim.distributions`` submodules have been removed, and ``starsim.dists`` has been added.
- The ``ss.Dist`` class uses ``np.random.default_rng()`` rather than ``scipy.stats`` by default, although a ``scipy.stats`` distribution can be supplied as an alternative. This is up to 4x faster (including, critically, for Bernoulli distributions).
- Also removes ``ss.options.multirng`` (the new version is equivalent to it being always on).
- Removes duplicate logic for transmission (``make_new_cases()``)
- Adds new custom distributions such as ``ss.choice()`` and ``ss.delta()``.
- These distributions can be called directly, e.g. ``dist = ss.weibull(c=2); dist(5)`` will return 5 random variates from a Weibull distribution.
- Instead of being manually initialized based on the name, the ``Sim`` object is parsed and all distributions will be initialized with a unique identifier based on their place in the object (e.g. ``sim.diseases.sir.pars.dur_inf``), which is used to set their unique seed.

Other changes
~~~~~~~~~~~~~
- This PR also fixes bugs with lognormal parameters, and makes it clear whether the parameters are for the *implicit* normal distribution (``ss.lognorm_im()``, the NumPy/SciPy default, equivalent to ``ss.lognorm_mean()`` previously) or the "explicit" lognormal distribution (``ss.lognorm_ex()``, equivalent to ``ss.lognorm()`` previously).
- Renames ``ss.dx``, ``ss.tx``, ``ss.vx`` to``ss.Dx``, ``ss.Tx``, ``ss.Vx``.
- Removed ``set_numba_seed()`` as a duplicate of ``set_seed()``.
- *GitHub info*: PR `392 <https://github.com/amath-idm/stisim/pull/392>`_

Version 0.2.10 (2024-03-18)
---------------------------
- SIR duration of infection now accounts for dt
- Reworked sir_vaccine to modify rel_sus instead of moving agents from susceptible to recovered.
- n_years no longer necessarily an integer
- *GitHub info*: PR `389 <https://github.com/amath-idm/stisim/pull/389>`_


Version 0.2.9 (2024-03-18)
--------------------------
- Renames and extends the multirng option in settings, now called 'rng', which set how random numbers are handled in Starsim with three options:

    - "centralized" uses the centralized numpy random number generator for all distributions.
    - "single" uses a separate (SingleRNG) random number generator for each distribution.
    - "multi" uses a separate (MultiRNG) random number generator for each distribution.
- *GitHub info*: PR `349 <https://github.com/amath-idm/stisim/pull/349>`_


Version 0.2.8 (2024-03-13)
--------------------------
- Add ``ss.demo()`` to quickly create a default simulation.
- *GitHub info*: PR `380 <https://github.com/amath-idm/stisim/pull/380>`_


Version 0.2.7 (2024-03-09)
--------------------------
- Update ``StaticNet`` with defaults and correct argument passing
- *GitHub info*: PR `339 <https://github.com/amath-idm/stisim/pull/339>`_


Version 0.2.6 (2024-02-29)
--------------------------
- Make random number streams independent for SIR
- *GitHub info*: PR `307 <https://github.com/amath-idm/stisim/pull/307>`_


Version 0.2.5 (2024-02-29)
--------------------------
- Improve logic for making new cases with multi-RNG
- *GitHub info*: PR `337 <https://github.com/amath-idm/stisim/pull/337>`_


Version 0.2.4 (2024-02-27)
--------------------------
- Improve ``sim.summarize()``
- Improve ``sim.plot()``
- Improve SIR model defaults
- *GitHub info*: PR `320 <https://github.com/amath-idm/stisim/pull/320>`_


Version 0.2.3 (2024-02-26)
--------------------------
- Removes ``STI`` class
- Changes default death rate from units of per person to per thousand people
- Allows ``ss.Sim(demographics=True)`` to enable births and deaths
- Fix pickling of ``State`` objects
- Rename ``networks.py`` to ``network.py``, and fix HIV mortality
- *GitHub info*: PRs `305 <https://github.com/amath-idm/stisim/pull/305>`_, `308 <https://github.com/amath-idm/stisim/pull/308>`_, `317 <https://github.com/amath-idm/stisim/pull/317>`_


Version 0.2.2 (2024-02-26)
--------------------------
- Add the ``Samples`` class
- *GitHub info*: PR `311 <https://github.com/amath-idm/stisim/pull/311>`_


Version 0.2.1 (2024-02-22)
--------------------------
- Only remove dead agents on certain timesteps
- *GitHub info*: PR `294 <https://github.com/amath-idm/stisim/pull/294>`_


Version 0.2.0 (2024-02-15)
--------------------------
- Code reorganization, including making ``networks.py`` and ``disease.py`` to the top level
- Networks moved from ``People`` to ``Sim``
- Various classes renamed (e.g. ``FusedArray`` to ``UIDArray``, ``STI`` to ``Infection``)
- Better type checking
- Added ``MultiSim``
- Added cholera, measles, and Ebola
- Added vaccination
- More flexible inputs
- *GitHub info*: PR `235 <https://github.com/amath-idm/stisim/pull/235>`_


Version 0.1.8 (2024-01-30)
--------------------------
- Transmission based on number of contacts
- *GitHub info*: PR `220 <https://github.com/amath-idm/stisim/pull/220>`_


Version 0.1.7 (2024-01-27)
--------------------------
- Performance enhancement for disease transmission, leading to a 10% decrease in runtime.
- *GitHub info*: PR `217 <https://github.com/amath-idm/stisim/pull/217>`_


Version 0.1.6 (2024-01-23)
--------------------------
- Adds template interventions and products for diagnostics and treatment
- Adds syphilis screening & treatment interventions
- *GitHub info*: PR `210 <https://github.com/amath-idm/stisim/pull/210>`_


Version 0.1.5 (2024-01-23)
--------------------------
- Renamed ``stisim`` to ``starsim``.
- *GitHub info*: PR `200 <https://github.com/amath-idm/stisim/pull/200>`_


Version 0.1.4 (2024-01-23)
--------------------------
- Adds a syphilis module
- *GitHub info*: PR `206 <https://github.com/amath-idm/stisim/pull/206>`_


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
- Adds in basic Starsim functionality
- *GitHub info*: PR `17 <https://github.com/amath-idm/stisim/pull/17>`__


Version 0.0.1 (2023-06-22)
--------------------------
- Initial version.
