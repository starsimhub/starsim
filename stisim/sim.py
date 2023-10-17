"""
Define core Sim classes
"""

# Imports
import numpy as np
import sciris as sc
import stisim as ss


__all__ = ['Sim', 'AlreadyRunError']


class Sim(sc.prettyobj):

    def __init__(self, pars=None, label=None, people=None, demographics=None, diseases=None, connectors=None, **kwargs):

        # Set attributes
        self.label = label  # The label/name of the simulation
        self.created = None  # The datetime the sim was created
        self.people = people  # People object
        self.demographics  = ss.ndict(demographics, type=ss.DemographicModule)
        self.diseases      = ss.ndict(diseases, type=ss.Disease)
        self.connectors    = ss.ndict(connectors, type=ss.Connector)
        self.results       = ss.ndict(type=ss.Result)  # For storing results
        self.summary       = None  # For storing a summary of the results
        self.initialized   = False  # Whether initialization is complete
        self.complete      = False  # Whether a simulation has completed running # TODO: replace with finalized?
        self.results_ready = False  # Whether results are ready
        self.filename      = None

        # Time indexing
        self.ti      = None  # The time index, e.g. 0, 1, 2 # TODO: do we need all of these?
        self.yearvec = None
        self.tivec   = None
        self.npts    = None

        # Make default parameters (using values from parameters.py)
        self.pars = ss.make_pars()  # Start with default pars
        self.pars.update_pars(sc.mergedicts(pars, kwargs))  # Update the parameters

        # Initialize other quantities
        self.interventions = ss.ndict(type=ss.Intervention)
        self.analyzers = ss.ndict(type=ss.Analyzer)

        return

    @property
    def dt(self):
        return self.pars['dt']

    @property
    def year(self):
        return self.yearvec[self.ti]

    def initialize(self, popdict=None, reset=False, **kwargs):
        """
        Perform all initializations on the sim.
        """
        # Validation and initialization
        self.ti = 0  # The current time index
        self.validate_pars()  # Ensure parameters have valid values
        self.validate_dt()
        self.init_time_vecs()  # Initialize time vecs
        ss.set_seed(self.pars['rand_seed'])  # Reset the random seed before the population is created

        # Initialize the core sim components
        self.init_people(popdict=popdict, reset=reset, **kwargs)  # Create all the people (the heaviest step)
        self.init_networks()
        self.init_demographics()
        self.init_diseases()
        self.init_connectors()
        self.init_interventions()
        self.init_analyzers()

        # Perform post-initialization validation
        self.validate_post_init()

        # Reset the random seed to the default run seed, so that if the simulation is run with
        # reset_seed=False right after initialization, it will still produce the same output
        ss.set_seed(self.pars['rand_seed'] + 1)

        # Final steps
        self.initialized = True
        self.complete = False
        self.results_ready = False

        return self

    def validate_dt(self):
        """
        Check that 1/dt is an integer value, otherwise results and time vectors will have mismatching shapes.
        init_results explicitly makes this assumption by casting resfrequency = int(1/dt).
        """
        dt = self.dt
        reciprocal = 1.0 / dt  # Compute the reciprocal of dt
        if not reciprocal.is_integer():  # Check if reciprocal is not a whole (integer) number
            # Round the reciprocal
            reciprocal = int(reciprocal)
            rounded_dt = 1.0 / reciprocal
            self.pars['dt'] = rounded_dt
            if self.pars['verbose']:
                warnmsg = f"Warning: Provided time step dt: {dt} resulted in a non-integer number of steps per year. Rounded to {rounded_dt}."
                print(warnmsg)

    def validate_pars(self):
        """
        Some parameters can take multiple types; this makes them consistent.
        """
        # Handle n_agents
        if self.pars['n_agents'] is not None:
            self.pars['n_agents'] = int(self.pars['n_agents'])
        else:
            if self.people is not None:
                self.pars['n_agents'] = len(self.people)
            else:
                if self.popdict is not None:
                    self.pars['n_agents'] = len(self.popdict)
                else:
                    errormsg = 'Must supply n_agents, a people object, or a popdict'
                    raise ValueError(errormsg)

        # Handle end and n_years
        if self.pars['end']:
            self.pars['n_years'] = int(self.pars['end'] - self.pars['start'])
            if self.pars['n_years'] <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self.pars['start'])} and " \
                           f"end={str(self.pars['end'])}, which gives n_years={self.pars['n_years']}"
                raise ValueError(errormsg)
        else:
            if self.pars['n_years']:
                self.pars['end'] = self.pars['start'] + self.pars['n_years']
            else:
                errormsg = 'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Handle verbose
        if self.pars['verbose'] == 'brief':
            self.pars['verbose'] = -1
        if not sc.isnumber(self.pars['verbose']):  # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self.pars["verbose"])} "{self.pars["verbose"]}"'
            raise ValueError(errormsg)

        return

    def init_time_vecs(self):
        """
        Construct vectors things that keep track of time
        """
        self.yearvec = sc.inclusiverange(start=self.pars['start'], stop=self.pars['end'] + 1 - self.pars['dt'],
                                         step=self.pars['dt'])  # Includes all the timepoints in the last year
        self.npts = len(self.yearvec)
        self.tivec = np.arange(self.npts)

    def init_people(self, popdict=None, reset=False, verbose=None, **kwargs):
        """
        Initialize people within the sim
        Sometimes the people are provided, in which case this just adds a few sim properties to them.
        Other time people are not provided and this method makes them.
        Args:
            popdict         (any):  pre-generated people of various formats.
            reset           (bool): whether to regenerate the people even if they already exist
            verbose         (int):  detail to print
            kwargs          (dict): passed to ss.make_people()
        """

        # Handle inputs
        if verbose is None:
            verbose = self.pars['verbose']
        if verbose > 0:
            resetstr = ''
            if self.people:
                resetstr = ' (resetting people)' if reset else ' (warning: not resetting sim.people)'
            print(f'Initializing sim{resetstr} with {self.pars["n_agents"]:0n} agents')

        # If people have not been supplied, make them
        if self.people is None:
            self.people = ss.People(n=self.pars['n_agents'], **kwargs)  # This just assigns UIDs and length

        # If a popdict has not been supplied, we can make one from location data
        if popdict is None:
            if self.pars['location'] is not None:
                # Check where to get total_pop from
                if self.pars[
                    'total_pop'] is not None:  # If no pop_scale has been provided, try to get it from the location
                    errormsg = 'You can either define total_pop explicitly or via the location, but not both'
                    raise ValueError(errormsg)
                total_pop, popdict = ss.make_popdict(n=self.pars['n_agents'], location=self.pars['location'], verbose=self.pars['verbose'])

            else:
                if self.pars[
                    'total_pop'] is not None:  # If no pop_scale has been provided, try to get it from the location
                    total_pop = self.pars['total_pop']
                else:
                    if self.pars['pop_scale'] is not None:
                        total_pop = self.pars['pop_scale'] * self.pars['n_agents']
                    else:
                        total_pop = self.pars['n_agents']

        self.pars['total_pop'] = total_pop
        if self.pars['pop_scale'] is None:
            self.pars['pop_scale'] = total_pop / self.pars['n_agents']

        # Any other initialization
        if not self.people.initialized:
            self.people.initialize()

        # Set time attributes
        self.people.ti = self.ti
        self.people.dt = self.dt
        self.people.year = self.year
        self.people.init_results(self)
        return self

    def init_demographics(self):
        for module in self.demographics.values():
            module.initialize(self)
            self.results[module.name] = module.results

    def init_diseases(self):
        """ Initialize modules and connectors to be simulated """
        for disease in self.diseases.values():
            disease.initialize(self)

            # Add the disease's parameters and results into the Sim's dicts
            self.pars[disease.name] = disease.pars
            self.results[disease.name] = disease.results

            # Add disease states to the People's dicts
            self.people.add_module(disease)

        return

    def init_connectors(self):
        for connector in self.connectors.values():
            connector.initialize(self)

    def init_networks(self):
        """ Initialize networks if these have been provided separately from the people """

        # One possible workflow is that users will provide a location and a set of networks but not people.
        # This means networks will be stored in self.pars['networks'] and we'll need to copy them to the people.
        if self.people.networks is None or len(self.people.networks) == 0:
            if self.pars['networks'] is not None:
                self.people.networks = ss.Networks(self.pars['networks'])

        if not isinstance(self.people.networks, ss.Networks):
            self.people.networks = ss.Networks(networks=self.people.networks)

        self.people.networks.initialize(self)

        # for key, network in self.people.networks.networks.items():  # TODO rename
            # if network.label is not None:
            #     layer_name = network.label
            # else:
            #     layer_name = key
            #     network.label = layer_name
            # network.initialize(self)

            # Add network states to the People's dicts
            # self.people.add_module(network)
            # self.people.networks[network.name] = network

        return

    def init_interventions(self):
        """ Initialize and validate the interventions """

        # Translate the intervention specs into actual interventions
        for i, intervention in enumerate(self.pars['interventions']):
            if isinstance(intervention, type) and issubclass(intervention, ss.Intervention):
                intervention = intervention()  # Convert from a class to an instance of a class
            if isinstance(intervention, ss.Intervention):
                intervention.initialize(self)
                self.interventions += intervention
            elif callable(intervention):
                self.interventions += intervention
            else:
                errormsg = f'Intervention {intervention} does not seem to be a valid intervention: must be a function or Intervention subclass'
                raise TypeError(errormsg)

        return

    def init_analyzers(self):
        """ Initialize the analyzers """

        # Interpret analyzers
        for ai, analyzer in enumerate(self.pars['analyzers']):
            if isinstance(analyzer, type) and issubclass(analyzer, ss.Analyzer):
                analyzer = analyzer()  # Convert from a class to an instance of a class
            if not (isinstance(analyzer, ss.Analyzer) or callable(analyzer)):
                errormsg = f'Analyzer {analyzer} does not seem to be a valid analyzer: must be a function or Analyzer subclass'
                raise TypeError(errormsg)
            self.analyzers += analyzer  # Add it in

        for analyzer in self.analyzers.values():
            if isinstance(analyzer, ss.Analyzer):
                analyzer.initialize(self)

        return

    def validate_post_init(self):
        """
        Validate inputs again once everything has been initialized.
        TBC whether we keep this or incorporate the checks into the init methods
        """
        # Make sure that there's a contact network if any diseases are present
        if self.diseases and not self.people.networks:
            warnmsg = f'Warning: simulation has {len(self.diseases)} diseases but no contact network(s).'
            ss.warn(warnmsg, die=False)
        return

    def step(self):
        """ Step through time and update values """

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Clean up dead agents, if removing agents is enabled
        if self.pars.remove_dead:
            self.people.remove_dead(self)

        # Update demographic modules (create new agents from births/immigration, schedule non-disease deaths and emigration)
        for module in self.demographics.values():
            module.update(self)

        # Carry out autonomous state changes in the disease modules. This allows autonomous state changes/initializations
        # to be applied to newly created agents
        for disease in self.diseases.values():
            disease.update_pre(self)

        # Update connectors -- TBC where this appears in the ordering
        for connector in self.connectors.values():
            connector.update(self)

        # Update networks - this takes place here in case autonomous state changes at this timestep
        # affect eligibility for contacts
        self.people.update_networks()

        # Apply interventions - new changes to contacts will be visible and so the final networks can be customized by
        # interventions, by running them at this point
        for intervention in self.interventions.values():
            intervention.apply(self)

        # Carry out transmission/new cases
        for disease in self.diseases.values():
            disease.make_new_cases(self)

        # Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed
        # before analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep
        self.people.resolve_deaths()

        # Update results
        self.people.update_results(self)

        for disease in self.diseases.values():
            disease.update_results(self)

        for analyzer in self.analyzers.values():
            analyzer.update_results(self)

        # Tidy up
        self.ti += 1
        self.people.ti = self.ti
        self.people.update_post(self)

        if self.ti == self.npts:
            self.complete = True

        return

    def run(self, until=None, reset_seed=True, verbose=None):
        """ Run the model once """

        # Initialization steps
        T = sc.timer()
        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(self.pars)  # Create a copy of the parameters to restore after the run

        if verbose is None:
            verbose = self.pars['verbose']

        if reset_seed:
            ss.set_seed(self.pars['rand_seed'] + 1)

        # Check for AlreadyRun errors
        errormsg = None
        if until is None: until = self.npts
        if until > self.npts:
            errormsg = f'Requested to run until t={until} but the simulation end is ti={self.npts}'
        if self.ti >= until:  # NB. At the start, self.t is None so this check must occur after initialization
            errormsg = f'Simulation is currently at t={self.ti}, requested to run until ti={until} which has already been reached'
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.initialize() to re-run)'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop
        while self.ti < until:

            # Check if we were asked to stop
            elapsed = T.toc(output=True)

            # Print progress
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.yearvec[self.ti]:0.1f} ({self.ti:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose > 0:
                    if not (self.ti % int(1.0 / verbose)):
                        sc.progressbar(self.ti + 1, self.npts, label=string, length=20, newline=True)

            # Actually run the model
            self.step()

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize(verbose=verbose)
            sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)

        return self

    def finalize(self, verbose=None):
        """ Compute final results """

        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Final settings
        self.results_ready = True  # Set this first so self.summary() knows to print the results
        self.ti -= 1  # During the run, this keeps track of the next step; restore this be the final day of the sim
        return

    def shrink(self, skip_attrs=None, in_place=True):
        """
        "Shrinks" the simulation by removing the people and other memory-intensive
        attributes (e.g., some interventions and analyzers), and returns a copy of
        the "shrunken" simulation. Used to reduce the memory required for RAM or
        for saved files.

        Args:
            skip_attrs (list): a list of attributes to skip (remove) in order to perform the shrinking; default "people"
            in_place (bool): whether to perform the shrinking in place (default), or return a shrunken copy instead

        Returns:
            shrunken (Sim): a Sim object with the listed attributes removed
        """
        # By default, skip people (~90% of memory), popdict, and _orig_pars (which is just a backup)
        if skip_attrs is None:
            skip_attrs = ['people']

        # Create the new object, and copy original dict, skipping the skipped attributes
        if in_place:
            shrunken = self
            for attr in skip_attrs:
                setattr(self, attr, None)
        else:
            shrunken = object.__new__(self.__class__)
            shrunken.__dict__ = {k: (v if k not in skip_attrs else None) for k, v in self.__dict__.items()}

        # Don't return if in place
        if in_place:
            return
        else:
            return shrunken

    def save(self, filename=None, keep_people=None, skip_attrs=None, **kwargs):
        """
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            keep_people (bool or None): whether to keep the people
            skip_attrs (list): attributes to skip saving
            kwargs: passed to sc.makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        **Example**::

            sim.save() # Saves to a .sim file
        """

        # Set keep_people based on whether we're in the middle of a run
        if keep_people is None:
            if self.initialized and not self.results_ready:
                keep_people = True
            else:
                keep_people = False

        # Handle the filename
        if filename is None:
            filename = self.simfile
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename  # Store the actual saved filename

        # Handle the shrinkage and save
        if skip_attrs or not keep_people:
            obj = self.shrink(skip_attrs=skip_attrs, in_place=False)
        else:
            obj = self
        sc.save(filename=filename, obj=obj)

        return filename

    @staticmethod
    def load(filename, *args, **kwargs):
        """
        Load from disk from a gzipped pickle.
        """
        sim = sc.load(filename, *args, **kwargs)
        if not isinstance(sim, Sim):  # pragma: no cover
            errormsg = f'Cannot load object of {type(sim)} as a Sim object'
            raise TypeError(errormsg)
        return sim


class AlreadyRunError(RuntimeError):
    """
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    :py:func:`Sim.run` and not taking any timesteps, would be an inadvertent error.
    """
    pass
