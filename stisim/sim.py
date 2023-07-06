"""
Define core Sim classes
"""

# Imports
import numpy as np
import sciris as sc
from . import base as ssb
from . import misc as ssm
from . import settings as sss
from . import utils as ssu
from . import people as ssppl
from . import parameters as sspar
from . import interventions as ssi
from . import analyzers as ssa
from .results import Result


# Define the model
class Sim(ssb.BaseSim):

    def __init__(self, pars=None, label=None, people=None, popdict=None, modules=None,
                 networks=None, version=None, **kwargs):

        # Set attributes
        self.label = label  # The label/name of the simulation
        self.created = None  # The datetime the sim was created
        self.people = people  # People object
        self.popdict = popdict  # popdict used to create people
        self.networks = networks  # List of provided networks
        self.modules = ssu.named_dict(modules)  # List of modules to simulate
        self.results = sc.objdict()  # For storing results
        self.summary = None  # For storing a summary of the results
        self.initialized = False  # Whether initialization is complete
        self.complete = False  # Whether a simulation has completed running
        self.results_ready = False  # Whether results are ready
        self._default_ver = version  # Default version of parameters used
        self._orig_pars = None  # Store original parameters to optionally restore at the end of the simulation

        # Time indexing
        self.ti = None  # The time index, e.g. 0, 1, 2
        self.yearvec = None
        self.tivec = None
        self.npts = None

        # Make default parameters (using values from parameters.py)
        default_pars = sspar.make_pars(version=version)  # Start with default pars
        super().__init__(default_pars)  # Initialize and set the parameters as attributes

        # Update parameters
        self.update_pars(pars, **kwargs)  # Update the parameters

        # Initialize other quantities
        self.interventions = None
        self.analyzers = None

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
        ssu.set_seed(self['rand_seed'])  # Reset the random seed before the population is created

        # Initialize the core sim components
        self.init_people(popdict=popdict, reset=reset, **kwargs)  # Create all the people (the heaviest step)
        self.init_networks()
        self.init_results()
        self.init_modules()
        self.init_interventions()
        self.init_analyzers()
        self.validate_layer_pars()

        # Reset the random seed to the default run seed, so that if the simulation is run with
        # reset_seed=False right after initialization, it will still produce the same output
        ssu.set_seed(self['rand_seed'] + 1)

        # Final steps
        self.initialized = True
        self.complete = False
        self.results_ready = False

        return self

    def layer_keys(self):
        """
        Attempt to retrieve the current network names
        """
        try:
            keys = list(self.people['networks'].keys())
        except:  # pragma: no cover
            keys = []
        return keys

    def validate_layer_pars(self):
        """
        Check if there is a contact network
        """

        if self.people is not None:
            modules = len(self.modules) > 0
            pop_keys = set(self.people.networks.keys())
            if modules and not len(pop_keys):
                warnmsg = f'Warning: your simulation has {len(self.modules)} modules but no contact layers.'
                ssm.warn(warnmsg, die=False)

        return

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
            self['dt'] = rounded_dt
            if self['verbose']:
                warnmsg = f"Warning: Provided time step dt: {dt} resulted in a non-integer number of steps per year. Rounded to {rounded_dt}."
                print(warnmsg)

    def validate_pars(self):
        """
        Some parameters can take multiple types; this makes them consistent.
        """
        # Handle n_agents
        if self['n_agents'] is not None:
            self['n_agents'] = int(self['n_agents'])
        else:
            if self.people is not None:
                self['n_agents'] = len(self.people)
            else:
                if self.popdict is not None:
                    self['n_agents'] = len(self.popdict)
                else:
                    errormsg = 'Must supply n_agents, a people object, or a popdict'
                    raise ValueError(errormsg)

        # Handle end and n_years
        if self['end']:
            self['n_years'] = int(self['end'] - self['start'])
            if self['n_years'] <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self['start'])} and " \
                           f"end={str(self['end'])}, which gives n_years={self['n_years']}"
                raise ValueError(errormsg)
        else:
            if self['n_years']:
                self['end'] = self['start'] + self['n_years']
            else:
                errormsg = 'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Handle verbose
        if self['verbose'] == 'brief':
            self['verbose'] = -1
        if not sc.isnumber(self['verbose']):  # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self["verbose"])} "{self["verbose"]}"'
            raise ValueError(errormsg)

        return

    def init_time_vecs(self):
        """
        Construct vectors things that keep track of time
        """
        self.yearvec = sc.inclusiverange(start=self['start'], stop=self['end'] + 1 - self['dt'],
                                         step=self['dt'])  # Includes all the timepoints in the last year
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
            verbose = self['verbose']
        if verbose > 0:
            resetstr = ''
            if self.people:
                resetstr = ' (resetting people)' if reset else ' (warning: not resetting sim.people)'
            print(f'Initializing sim{resetstr} with {self["n_agents"]:0n} agents')

        # If people have not been supplied, make them
        if self.people is None:
            self.people = ssppl.People(self['n_agents'], kwargs)  # This just assigns UIDs and length

        # If a popdict has not been supplied, we can make one from location data
        if popdict is None:
            if self['location'] is not None:
                # Check where to get total_pop from
                if self['total_pop'] is not None:  # If no pop_scale has been provided, try to get it from the location
                    errormsg = 'You can either define total_pop explicitly or via the location, but not both'
                    raise ValueError(errormsg)
                total_pop, popdict = ssppl.make_popdict(n=self['n_agents'], location=self['location'], verbose=self['verbose'], dt=self.dt)

            else:
                if self['total_pop'] is not None:  # If no pop_scale has been provided, try to get it from the location
                    total_pop = self['total_pop']
                else:
                    if self['pop_scale'] is not None:
                        total_pop = self['pop_scale'] * self['n_agents']
                    else:
                        total_pop = self['n_agents']

        self['total_pop'] = total_pop
        if self['pop_scale'] is None:
            self['pop_scale'] = total_pop / self['n_agents']

        # Finish initialization
        if not self.people.initialized:
            self.people.initialize(popdict=popdict)  # Fully initialize the people

        # Add time attributes
        self.people.ti = self.ti
        self.people.dt = self.dt

        return self

    def init_modules(self):
        """ Initialize modules to be simulated """
        for module in self.modules.values():
            module.initialize(self)
        return

    def init_networks(self):
        """ Initialize networks if these have been provided separately from the people """

        # One possible workflow is that users will provide a location and a set of networks but not people.
        # This means networks will be stored in self['networks'] and we'll need to copy them to the people.
        if self.people.networks is None or len(self.people.networks) == 0:
            if self['networks'] is not None:
                self.people.networks = ssu.named_dict(self['networks'])

        for key, network in self.people.networks.items():
            if network.label is not None:
                layer_name = network.label
            else:
                layer_name = key
                network.label = layer_name
            network.initialize(self.people)
            self.people.networks[layer_name] = network

        return

    def init_results(self):
        """
        Create the main results structure.
        """
        # Make results
        results = ssu.named_dict(
            Result('births', None, self.npts, sss.default_float),
            Result('deaths', None, self.npts, sss.default_float),
            Result('n_alive', None, self.npts, sss.default_int),
        )

        # Final items
        self.results = results
        self.results_ready = False

        return

    def init_interventions(self):
        """ Initialize and validate the interventions """

        # Translate the intervention specs into actual interventions
        for i, intervention in enumerate(self['interventions']):
            if isinstance(intervention, type) and issubclass(intervention, ssi.Intervention):
                intervention = intervention()  # Convert from a class to an instance of a class
            if isinstance(intervention, ssi.Intervention):
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

        self.analyzers = sc.autolist()

        # Interpret analyzers
        for ai, analyzer in enumerate(self['analyzers']):
            if isinstance(analyzer, type) and issubclass(analyzer, ssa.Analyzer):
                analyzer = analyzer()  # Convert from a class to an instance of a class
            if not (isinstance(analyzer, ssa.Analyzer) or callable(analyzer)):
                errormsg = f'Analyzer {analyzer} does not seem to be a valid analyzer: must be a function or Analyzer subclass'
                raise TypeError(errormsg)
            self.analyzers += analyzer  # Add it in

        for analyzer in self.analyzers:
            if isinstance(analyzer, ssa.Analyzer):
                analyzer.initialize(self)

        return

    def step(self):
        """ Step through time and update values """

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Update states, modules, partnerships
        self.update_demographics()
        self.update_networks()
        self.update_modules()
        # self.update_connectors()  # TODO: add this when ready

        # Tidy up
        self.ti += 1
        if self.ti == self.npts:
            self.complete = True

        return

    def update_demographics(self):
        """
        TODO: decide whether this method is needed
        """
        self.people.update_demographics(dt=self.dt, ti=self.ti)

    def update_networks(self):
        """
        Update networks
        TODO: resolve where the networks live - sim.networks (akin to sim.modules), sim.people.networks, both?
        """
        for layer in self.people.networks.values():
            layer.update(self.people)

    def update_modules(self):
        """
        Update modules
        """
        for module in self.modules.values():
            module.update(self)

    def update_connectors(self):
        """ Update connectors """
        if len(self.modules) > 1:
            connectors = self['connectors']
            if len(connectors) > 0:
                for connector in connectors:
                    if callable(connector):
                        connector(self)
                    else:
                        warnmsg = f'Connector must be a callable function'
                        ssm.warn(warnmsg, die=True)
            elif self.ti == 0:  # only raise warning on first timestep
                warnmsg = f'No connectors in sim'
                ssm.warn(warnmsg, die=False)
            else:
                return
        return

    def run(self, until=None, reset_seed=True, verbose=None):
        """ Run the model once """

        # Initialization steps
        T = sc.timer()
        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(self.pars)  # Create a copy of the parameters to restore after the run

        if verbose is None:
            verbose = self['verbose']

        if reset_seed:
            # Reset the RNG. The primary use case (and why it defaults to True) is to ensure that
            #
            # >>> sim0.initialize()
            # >>> sim0.run()
            # >>> sim1.initialize()
            # >>> sim1.run()
            #
            # produces the same output as
            #
            # >>> sim0.initialize()
            # >>> sim1.initialize()
            # >>> sim0.run()
            # >>> sim1.run()
            #
            # The seed is offset by 1 to avoid drawing the same random numbers as those used for population generation,
            # otherwise the first set of random numbers in the model (e.g., deaths) will be correlated with the first
            # set of random numbers drawn in population generation (e.g., sex)
            ssu.set_seed(self['rand_seed'] + 1)

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
            if self['timelimit'] and elapsed > self['timelimit']:
                sc.printv(
                    f"Time limit ({self['timelimit']} s) exceeded; call sim.finalize() to compute results if desired",
                    1, verbose)
                return
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv(
                    "Stopping function terminated the simulation; call sim.finalize() to compute results if desired", 1,
                    verbose)
                return

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

        # Perform calculations on results
        # self.compute_results(verbose=verbose) # Calculate the rest of the results
        self.results = sc.objdict(
            self.results)  # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses

        return


class AlreadyRunError(RuntimeError):
    """
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    :py:func:`Sim.run` and not taking any timesteps, would be an inadvertent error.
    """
    pass
