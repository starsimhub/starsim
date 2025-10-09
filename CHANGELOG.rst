==========
What's new
==========

.. currentmodule:: starsim

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".


Version 2.3.2 (2025-07-16)
---------------------------
- Fix argument passing in ``Infection.infect``. This will be the final Starsim v2.x release.
- *GitHub info*: PR `1008 <https://github.com/starsimhub/starsim/pull/1008>`_


Version 2.3.1 (2025-02-25)
---------------------------
- Updated ``ss.Sim.shrink()`` to remove additional objects, resulting in a smaller sim size.
- ``ss.Calibration.save_csv()`` has been replaced by ``ss.Calibration.to_df()`` (to save to a CSV, use ``ss.Calibration.to_df().to_csv()``.
- ``ss.Result.shape`` has been renamed ``ss.Result._shape``, so ``ss.Result.shape`` now correctly returns the actual size of the array.
- Results by default convert all result keys to lowercase; use ``keep_case=True`` to turn off this behavior.
- Fixed a bug with an ``ss.date`` object converting to a ``pd.Timestamp`` upon copy.
- *GitHub info*: PR `865 <https://github.com/starsimhub/starsim/pull/865>`_


Version 2.3.0 (2025-02-14)
---------------------------
- The calibration class has been completely redesigned. Calibration now relies on "components", which capture mismatch with a particular data type (e.g., new infections). The new approach also adds additional statistical rigor for calculating mismatches.
- ``ss.MixingPool`` has been updated to be more modular, and behave more like ``ss.Network``; in particular, ``compute_transmission()`` rather than ``step()`` is called to determine new infections.
- ``ss.Result`` now has a ``summarize_by`` argument, which determines how a result should be summarized as a scalar (e.g., mean for a prevalence, sum for a count, last entry for a cumulative count).
- Fixed a bug with time parameters incorrectly pulling the parent unit from the Sim, rather than the parent module. 
- *GitHub info*: PR `831 <https://github.com/starsimhub/starsim/pull/831>`_


Version 2.2.0 (2024-11-18)
---------------------------
- Starsim is now available for R! See https://r.starsim.org for details.
- The ``Calibration`` class has been completely rewritten. See the calibration tutorial for more information.
- A negative binomial distribution is now available as ``ss.nbinom()``.
- ``ss.Births()`` now uses a binomial draw of births per timestep, rather than the expected value.
- Added ``ss.load()`` and ``ss.save()`` functions, and removed ``ss.Sim.load()``.
- *GitHub info*: PR `778 <https://github.com/starsimhub/starsim/pull/778>`_


Version 2.1.1 (2024-11-08)
---------------------------
- Adds improved Jupyter support for plotting (to prevent plots from appearing twice); you can disable this by setting ``ss.options.set(jupyter=False)``.
- Adds ``auto_plot`` to ``Result`` objects, to indicate if it should appear in ``sim.plot()`` by default.
- Adds ``copy()`` to the Sim and modules.
- Networks now store their length on each timestep as a result.
- Improves ``sim.shrink()``, with typical size reductions of >99%.
- Adds additional plotting options ``show_module`` (include the module name in the plot title), ``show_label`` (use the simulation label as the figure title), and ``show_skipped`` (shows results even if ``auto_plot=False``).
- *GitHub info*: PR `745 <https://github.com/starsimhub/starsim/pull/745>`_


Version 2.1.0 (2024-11-07)
---------------------------

Summary
~~~~~~~
- Time in simulations is now handled by an ``ss.Time()`` class, which unifies how time is represented between the ``Sim`` and each module.
- In addition to networks, there is now a new way of implementing disease transmission via mixing pools.

Time
~~~~
- Time handling now performed by the ``ss.Time()`` class. This has inputs similar to before (``start``, ``stop``, ``unit``, ``dt``, with ``dur`` still available as a sim input). However, instead of the previous ``timevec`` and ``abs_tvec`` arrays, there are now multiple ways of representing time (including ``datevec`` and ``yearvec``), regardless of what the inputs were.
- Dates are now represented in a native format, ``ss.date``, that is based on ``pd.Timestamp``.

Mixing pools
~~~~~~~~~~~~
- Adds a new approach to disease transmission called mixing pools. A mixing pool is a "mean field" coupling wherein susceptible agents are exposed to the average infectious agent. The user can create a single mixing pool using the ``ss.MixingPool`` class, or create many pools using ``MixingPools``. Such mixing pools could be used to simulate contact matrices, for example as published by Prem et al.
- There is a new ``ss.Route`` class, which is the base class for ``ss.Network`` and ``ss.MixingPool``.

Other changes
~~~~~~~~~~~~~
- Demographic modules have been updated to fix various bugs around different time units.
- The method for hashing distribution trace strings into seeds has changed, meaning that results will be stochastically different compared to Starsim v2.0.
- Fixed a bug with how timepars were updated in parameters.
- There is a new ``ss.Base`` class, which both ``ss.Sim`` and ``ss.Module`` inherit from.
- Results now print as a single line rather than the full array. The latter is available as ``result.disp()``.
- ``sim.to_df()`` now works even if different modules have different numbers of timepoints.
- The ``timepars`` module has been renamed to ``time``.
- In demographics modules, ``units`` has been renamed ``rate_units``.
- There are two new options, ``ss.options.date_sep`` and  ``ss.options.license``. The former sets the date separator (default ``.``, e.g. ``2024.04.0.4``), and the latter sets if the license prints when Starsim is imported.
- *GitHub info*: PR `724 <https://github.com/starsimhub/starsim/pull/724>`_


Version 2.0.0 (2024-10-01)
--------------------------

Summary
~~~~~~~
Version 2.0 contains several major changes. These include: module-specific timesteps and time-aware parameters (including a day/year ``unit`` flag for modules, and  ``ss.dur()`` and ``ss.rate()`` classes for parameters), and changes to module types and integration (e.g. renaming ``update()`` and ``apply()`` methods to ``step()``;).

Time-aware parameters and modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Added ``ss.dur()``, ``ss.rate()``, and ``ss.time_prob()`` classes, for automatic handling of time units in simulations. There are also convenience classes ``ss.days()``, ``ss.years()``, ``ss.perday()``, ``ss.peryear()``, and ``ss.beta()`` for special cases of these.
- ``ss.dur()`` and ``ss.rate()``, along with modules and the sim itself, have a ``unit`` parameter which can be ``'day'``, ``'week'``, ``'month'``, or ``'year'`` (default). Modules now also have their own timestep ``dt``. Different units and timesteps can be mixed and matched. Time parameters have a ``to()`` method, e.g. ``ss.dur(1, 'year').to('day')`` will return ``ss.dur(365, unit='day')``.
- The ``ss.Sim`` parameter ``n_years`` has been renamed ``dur``; ``sim.yearvec`` is now ``sim.timevec``, which can have units of days (usually starting at 0), dates (e.g. ``'2020-01-01'``), or years (e.g. ``2020``). ``sim.abs_tvec`` is the translation of ``sim.timevec`` as a numeric array starting at 0, using the sim's units (usually ``'day'`` or ``'year'``). For example, if ``sim.timevec`` is a list of daily dates from ``'2022-01-01'`` to ``'2022-12-31'``, ``sim.abs_tvec`` will be ``np.arange(365)``.
- Each module also has its own ``mod.timevec``; this can be different from the sim if it defines its own time unit and/or timestep. ``mod.abs_tvec`` always starts at 0 and always uses the sim's unit.
- There is a new ``Loop`` class which handles the integration loop. You can view the integration plan via ``sim.loop.to_df()`` or ``sim.loop.plot()``. You can see how long each part of the sim took with ``sim.loop.plot_cpu()``.
- There are more advanced debugging tools. You can run a single sim timestep with ``sim.run_one_step()`` (which in turn calls multiple functions), and you can run a single function from the integration loop with ``sim.loop.run_one_step()``.

Module changes
~~~~~~~~~~~~~~
- Functionality has been moved from ``ss.Plugin`` to ``ss.Module``, and the former has been removed.
- ``ss.Connector`` functionality has been moved to ``ss.Module``. ``ss.Module`` objects can be placed anywhere in the list of modules (e.g., in demographics, networks, diseases, interventions), depending on when you want them to execute. However, ``ss.Connector`` objects are applied after ``Disease.step_state()`` and before ``Network.step()``.
- Many of the module methods have been renamed; in particular, all modules now have a ``step()`` method, which replaces ``update()`` (for demographics and networks), ``apply()`` (for interventions and analyzers), and ``make_new_cases()`` (for diseases). For both the sim and modules, ``initialize()`` has been renamed ``init()``.
- All modules are treated the same in the integration loop, except for diseases, which have ``step_state()`` and ``step_die()`` methods.
- The Starsim module ``states.py`` has been moved to ``arrays.py``, and ``network.py`` has been moved to ``networks.py``.

State and array changes
~~~~~~~~~~~~~~~~~~~~~~~
- ``ss.Arr``, ``ss.TimePar``, and ``ss.Result`` all inherit from the new class ``ss.BaseArr``, which provides functionality similar to a NumPy array, except all values are stored in ``arr.values`` (like a ``pd.Series``).
- Whereas before, computations on an ``ss.Arr`` usually returned a NumPy array, calculations now usually return the same type. To access the NumPy array, use ``arr.values``. 
- There is a new ``ss.State`` class, which is a subtype of ``ss.BoolArr``. Typically, ``ss.State`` is used for boolean disease states, such as ``infected``, ``susceptible``, etc., where you want to automatically generate results (e.g. ``n_infected``). You can continue using ``ss.BoolArr`` for other agent attributes that you don't necessarily want to automatically generate results for, e.g. ``ever_vaccinated``.

Results changes
~~~~~~~~~~~~~~~
- Results are now defined differently. They should be defined in ``ss.Module.init_results()``, not ``ss.Module.init_pre()``. They now take the module name, number of points, and time vector from the parent module. As a result, they are usually initialized via ``ss.Module.define_results(res1, res2)`` (as opposed to ``mod.results += [res1, res2]`` previously). ``define_results()`` automatically adds these properties from the parent module; they can still be defined explicitly if needed however.
- Because results now store their own time information, they can be plotted in a self-contained way. Both ``ss.Result`` and ``ss.Results`` objects now have ``plot()`` and ``to_df()`` methods.

Demographics changes
~~~~~~~~~~~~~~~~~~~~
- Fixed a bug in how results were defined for ``ss.Births`` and ``ss.Deaths``.
- The ``ss.Pregnancy`` module has been significantly rewritten, including: (1) Agents now have a ``parent`` which indicates the UID of the parent; (2) Women now track ``child_uid``; (3) On neonatal death, the pregnancy state of the mother is corrected; (4) Pregnancy rates now adjusted for infecund rather than pregnant; (4) Pregnancy now has a burn-in, which defaults to ``True``; (5) Pregnancy has a ``p_neonatal_death`` parameter to capture fetal and neonatal death if the mother dies.
- Slots now has a minimum, default of 100, to account for small initial population sizes that grow dramatically over time.

Computational changes
~~~~~~~~~~~~~~~~~~~~~
- There have been several performance improvements. The default float type is now ``np.float32``. Transmission is now handled by a specialized ``Infection.compute_transmission()`` method. Several additional functions now use Numba, including ``fastmath=True``, which leverages Intel's short vector math library.
- A new ``ss.multi_random()`` distribution class has been added, that allows random numbers to be generated by two (or more) agents. It largely replaces ``ss.combine_rands()`` and is 5-10x faster.
- A new ``ss.gamma()`` distribution has also been added.
- Distributions have a new ``jump_dt`` method that jumps by much more than a single state update.
- ``ss.parallel()`` and ``ss.MultiSim.run()`` now modify simulations in place by default. Instead of ``sims = ss.parallel(sim1, sim2).sims; sims[0].plot()``, you can now simply do ``ss.parallel(sim1, sim2); sim1.plot()``.

Other changes
~~~~~~~~~~~~~
- Data can now be supplied to a simulation; it will be automatically plotted by ``sim.plot()``.
- ``ss.Calibration`` has been significantly reworked, and now includes more flexible parameter setting, plus plotting (``calib.plot_sims()`` and ``calib.plot_trend()``). It also has a ``debug`` argument (which runs in serial rather than paralell), which can be helpful for troubleshooting issues.
- ``MultiSim`` now has display methods ``brief()`` (minimal), ``show()`` (moderate), and ``disp`` (verbose).
- ``sim.export_df()`` has been renamed ``sim.to_df()``.
- Most classes now have ``to_json()`` methods (which can also export to a dict).
- Fixed a bug in how the ``InfectionLog`` is added to disease modules.
- ``Sim.gitinfo`` has been replaced with ``Sim.metadata`` (which includes git info).
- ``Infection.validate_beta()`` is now applied on every timestep, so changes to beta during the simulation are now honored.
- ``sim.get_intervention()`` and ``sim.get_analyzer()`` have been removed; use built-in ``ndict`` operations (e.g., the label) to find the object you're after.
- ``requires`` has been removed from modules, but ``ss.check_requires()`` is still available if needed. Call it manually from ``init_pre()`` if desired, e.g. a PMTCT intervention might call ``ss.check_requires(self.sim, ['hiv', 'maternalnet'])``.
- For networks, ``contacts`` has been renamed ``edges`` except in cases where it refers to an *agent's* contacts. For example, ``network.contacts`` has been renamed ``network.edges``, but ``ss.find_contacts()`` remains the same.
- Networks now have a ``to_graph()`` method that exports to NetworkX.
- ``ss.diff_sims()`` can now handle ``MultiSim`` objects.
- ``Sim._orig_pars`` has been removed.
- ``ss.unique()`` has been removed.

Regression information
~~~~~~~~~~~~~~~~~~~~~~
- Note: the list here covers major changes only; in general, Starsim v1.0 scripts will not be compatible with Starsim v2.0.
- Results from Starsim v2.0 will be stochastically (but not statistically) different from Starsim v1.0.
- All duration and rate parameters should now be wrapped with ``ss.dur()`` and ``ss.rate()``. Events that represent probabilities over time (i.e. hazard rates) can also be wrapped with ``ss.time_prob()``, although this is similar to ``ss.rate()`` unless the value is relatively large.
- ``ss.Plugin`` has been removed. Use ``ss.Module`` instead.
- ``init_results()`` is now called by ``init_pre()``, and does not need to be called explicitly.
- ``default_pars()`` has been renamed ``define_pars()``.
- ``add_states()`` has been renamed ``define_states()``
- ``initialize()`` has been renamed ``init()``.
- ``Demographics.update()`` has been renamed ``Demographics.step()``.
- ``Network.update()`` has been renamed ``Network.step()``.
- ``Disease.update_pre()`` has been renamed ``Disease.step_state()``.
- ``Disease.make_new_cases()`` has been renamed ``Disease.step()``.
- ``Disease.update_death()`` has been renamed ``Disease.step_die()`` (which is now called by ``People.step_die()``).
- ``Infection._set_cases()`` has been renamed ``Infection.set_outcomes()``.
- ``Intervention.apply(sim)`` has been renamed ``Intervention.step()``; ditto for ``Analyzer``.
- ``Module.step()`` no longer takes ``sim`` as an argument (e.g., replace ``intervention.apply(sim)`` with ``intervention.step()``).
- All modules now have methods for ``start_step()``, ``finish_step()``, ``init_results()``, and ``update_results()``.
- ``Network.contacts`` has been renamed ``Network.edges``.
- ``sim.get_intervention()`` and ``sim.get_analyzer()`` have been removed; simply call directly instead (e.g. replace ``sim.get_intervention('vaccine')`` with ``sim.interventions['vaccine']``).
- ``requires`` is no longer an attribute of modules; call the ``ss.check_requires()`` function directly if needed.
- ``People.resolve_deaths()`` has been renamed ``People.check_deaths()``
- ``ss.unique()`` has been removed.
- *GitHub info*: PR `626 <https://github.com/starsimhub/starsim/pull/626>`_


Version 1.0.3 (2024-09-26)
---------------------------
- Fixes a bug in which some intervention parameters (e.g. eligibility) do not get set properly.
- *GitHub info*: PR `639 <https://github.com/starsimhub/starsim/pull/639>`_


Version 1.0.2 (2024-09-25)
---------------------------
- Fixes a bug in which random numbers drawn from auto-jumped distributions would overlap with random numbers drawn from subsequent timesteps.
- *GitHub info*: PR `639 <https://github.com/starsimhub/starsim/pull/639>`_


Version 1.0.1 (2024-07-22)
---------------------------
- Adds a new distribution, ``ss.rand_raw()``, that samples raw integers from the random number bit generator, for use with calculating transmission. This version is roughly 20-30% faster than the previous implementation.
- Adds interpolation to age-standardized fertility rate (ASFR) data.
- Adds flexibility to ART initiation.
- *GitHub info*: PR `593 <https://github.com/starsimhub/starsim/pull/593>`_


Version 1.0.0 (2024-07-10)
---------------------------
- Official release of Starsim!
- Adds a ``Calibration`` class, based on `Optuna <https://optuna.org>`_, to facilitate the calibration of Starsim models.
- Adds ``mean()``, ``median()``, and ``plot()`` methods to ``MultiSim``.
- Adds ``low`` and ``high`` attributes to ``Result`` objects.
- Adds a ``flatten()`` method to ``Results``, allowing nested ``Results`` objects to be turned into flat dictionaries.
- Removes duplicate UIDs among new infections, and adds a ``unique()`` method to ``ss.uids``.
- Fixes a bug that prevented ``ss.lognorm_im()`` from using callable parameters.
- Updates the default ``Sim`` string representation to be a single line; the more verbose version is available via ``sim.disp()``.
- *GitHub info*: PR `581 <https://github.com/starsimhub/starsim/pull/581>`_


Version 0.5.10 (2024-07-03)
---------------------------
- Adds two new common-random-number-safe networks. The first is an Erdős-Rényi network that is similar to ``RandomNet`` but parameterized differently. The second is a 2D spatial network with connectivity between agents within a given radius; these agents can also optionally move.
- *GitHub info*: PR `575 <https://github.com/starsimhub/starsim/pull/575>`_


Version 0.5.9 (2024-06-30)
--------------------------
- Added a ``ss.histogram()`` distribution, which allows generating new random values from an empirical histogram.
- When binned age data is provided to specify the initial ages for new agents, the ages are now distributed throughout the year/bin rather than new agents being assigned integer ages
- Initial age data is now accepted as a ``pd.Series`` rather than a ``pd.DataFrame`` where the index corresponds to the age values, thereby avoiding the need for specific dataframe column names to be used to specify the age and value
- *GitHub info*: PR `572 <https://github.com/starsimhub/starsim/pull/572>`_


Version 0.5.8 (2024-06-30)
--------------------------
- Revert to making infection logging disabled by default. However, the infection log will now always be created so disease subclasses can override logging behaviour where required (e.g., to capture additional metadata)
- **Backwards-compatibility notes:** Logging has been moved from an argument to ``Disease`` to ``pars``. Existing code such as ``Disease(log=True)`` should be changed to ``Disease(pars={'log':True})``. The 'log' option can be added to the pars passed to any subclass e.g., ``ss.HIV(pars={...,log=True})``.
- *GitHub info*: PR `573 <https://github.com/starsimhub/starsim/pull/573>`_

Version 0.5.7 (2024-06-27)
--------------------------
- Implemented a new ``ss.combine_rands()`` function based on a bitwise-XOR, since the previous modulo-based approach could introduce correlations between pairs of agents.
- *GitHub info*: PR `546 <https://github.com/starsimhub/starsim/pull/546>`_


Version 0.5.6 (2024-06-22)
--------------------------
- ``ss.Infection.make_new_cases()`` now returns the index of the network associated with each transmission event
- If a ``People`` object is provided to the ``Arr`` constructor, the arrays will be pre-initialized to index the current UIDs in the ``People`` object. This enables construction of temporary ``Arr`` instances that can be used to perform intermediate calculations (e.g., inside ``Intervention.apply()`` or within a module update step)
- Deprecated ``Arr(raw=...)`` argument to simplify initialization, as in practice the ``raw`` variable is not directly set, and this update also introduces a new pathway for initializating the `raw` attribute
- ``ss.uids.to_numpy()`` now returns a view rather than a copy
- ``ss.bernoulli.filter()`` now supports ``ss.BoolArr`` as an input, where the filtering will operate on the ``uids`` returned by ``ss.BoolArr.uids``
- ``ss.uids()`` supports construction from ``set`` objects (via ``np.fromiter()``)
- *GitHub info*: PR `565 <https://github.com/starsimhub/starsim/pull/555>`_


Version 0.5.5 (2024-06-19)
--------------------------
- Added labels to ``Result`` and state (``Arr``) objects.
- Added Numba decorator to ``find_contacts`` to significantly increase performance.
- Fixed bug when comparing ``uids`` and ``BoolArr`` objects.
- *GitHub info*: PR `562 <https://github.com/starsimhub/starsim/pull/555>`_


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
