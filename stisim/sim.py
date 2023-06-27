"""
Define core Sim classes
"""

# Imports
import numpy as np
import sciris as sc
from . import base as ssb
from . import misc as ssm
from . import utils as ssu
from . import population as sspop
from . import parameters as sspar
from . import interventions as ssi
from . import analyzers as ssa


# Define the model
class Sim(ssb.BaseSim):

    def __init__(self, pars=None, label=None, people=None, modules=None,
                 version=None, **kwargs):

        # Set attributes
        self.label = label    # The label/name of the simulation
        self.created = None     # The datetime the sim was created
        self.people = people   # People object
        self.modules = ssu.named_dict(modules)  # List of modules to simulate
        self.results = sc.objdict()       # For storing results
        self.summary = None     # For storing a summary of the results
        self.initialized = False    # Whether initialization is complete
        self.complete = False    # Whether a simulation has completed running
        self.results_ready = False    # Whether results are ready
        self._default_ver = version  # Default version of parameters used
        self._orig_pars = None     # Store original parameters to optionally restore at the end of the simulation

        # Time indexing
        self.ti = None  # The time index, e.g. 0, 1, 2
        self.t = None   # The year, e.g. 2015.2

        # Make default parameters (using values from parameters.py)
        default_pars = sspar.make_pars(version=version)  # Start with default pars
        super().__init__(default_pars)  # Initialize and set the parameters as attributes

        # Update parameters
        self.update_pars(pars, **kwargs)   # Update the parameters

        return

    @property
    def dt(self):
        return self.pars['dt']

    def initialize(self, reset=False, **kwargs):
        """
        Perform all initializations on the sim.
        """
        self.ti = 0  # The current time index
        self.validate_pars() # Ensure parameters have valid values
        ssu.set_seed(self['rand_seed']) # Reset the random seed before the population is created
        self.init_interventions()
        self.init_people(reset=reset, **kwargs) # Create all the people (the heaviest step)
        self.init_network()
        self.init_results()
        for module in self.modules.values():
            module.initialize(self)
        self.init_analyzers()
        self.validate_layer_pars()
        ssu.set_seed(self['rand_seed']+1)  # Reset the random seed to the default run seed, so that if the simulation is run with reset_seed=False right after initialization, it will still produce the same output
        self.initialized   = True
        self.complete      = False
        self.results_ready = False

        return self

    def layer_keys(self):
        '''
        Attempt to retrieve the current layer keys.
        '''
        try:
            keys = list(self.people['contacts'].keys()) # Get keys from acts
        except: # pragma: no cover
            keys = []
        return keys

    def validate_layer_pars(self):
        '''
        Check if there is a contact network
        '''

        if self.people is not None:
            modules = len(self.modules)>0
            pop_keys = set(self.people.contacts.keys())
            if modules and not len(pop_keys):
                warnmsg = f'Warning: your simulation has {len(self.modules)} modules but no contact layers.'
                ssm.warn(warnmsg, die=False)

        return


    def validate_pars(self):
        '''
        Some parameters can take multiple types; this makes them consistent.

        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle types
        for key in ['n_agents']:
            try:
                self[key] = int(self[key])
            except Exception as E:
                errormsg = f'Could not convert {key}={self[key]} of {type(self[key])} to integer'
                raise ValueError(errormsg) from E

        # Handle start
        if self['start'] in [None, 0]: # Use default start
            self['start'] = 2015

        # Handle end and n_years
        if self['end']:
            self['n_years'] = int(self['end'] - self['start'])
            if self['n_years'] <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self['start'])} and end={str(self['end'])}, which gives n_years={self['n_years']}"
                raise ValueError(errormsg)
        else:
            if self['n_years']:
                self['end'] = self['start'] + self['n_years']
            else:
                errormsg = 'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Construct other things that keep track of time
        self.years      = sc.inclusiverange(self['start'],self['end'])
        self.yearvec    = sc.inclusiverange(start=self['start'], stop=self['end']+1-self['dt'], step=self['dt']) # Includes all the timepoints in the last year
        self.npts       = len(self.yearvec)
        self.tvec       = np.arange(self.npts)

        # Handle verbose
        if self['verbose'] == 'brief':
            self['verbose'] = -1
        if not sc.isnumber(self['verbose']): # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self["verbose"])} "{self["verbose"]}"'
            raise ValueError(errormsg)

        return
    
    
    def init_interventions(self):
        ''' Initialize and validate the interventions '''

        # Initialization
        self.interventions = sc.autolist()

        # Translate the intervention specs into actual interventions
        for i,intervention in enumerate(self['interventions']):
            if isinstance(intervention, type) and issubclass(intervention, ssi.Intervention):
                intervention = intervention() # Convert from a class to an instance of a class
            if isinstance(intervention, ssi.Intervention):
                intervention.initialize(self)
                self.interventions += intervention
            elif callable(intervention):
                self.interventions += intervention
            else:
                errormsg = f'Intervention {intervention} does not seem to be a valid intervention: must be a function or hpv.Intervention subclass'
                raise TypeError(errormsg)

        return


    def init_people(self, popdict=None, reset=False, verbose=None, **kwargs):
        '''
        Create the people.
        Args:
            popdict         (any):  pre-generated people of various formats.
            reset           (bool): whether to regenerate the people even if they already exist
            verbose         (int):  detail to print
            kwargs          (dict): passed to ss.make_people()
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if popdict is not None:
            self.popdict = popdict
        if verbose > 0:
            resetstr= ''
            if self.people:
                resetstr = ' (resetting people)' if reset else ' (warning: not resetting sim.people)'
            print(f'Initializing sim{resetstr} with {self["n_agents"]:0n} agents')
        if self.popfile and self.popdict is None: # If there's a popdict, we initialize it
            self.load_population(init_people=False)

        # Make the people
        self.people, total_pop = sspop.make_people(self, reset=reset, verbose=verbose, **kwargs)

        # Figure out the scale factors
        if self['total_pop'] is not None and total_pop is not None: # If no pop_scale has been provided, try to get it from the location
            errormsg = 'You can either define total_pop explicitly or via the location, but not both'
            raise ValueError(errormsg)
        elif total_pop is None and self['total_pop'] is not None:
            total_pop = self['total_pop']
            
        if self['pop_scale'] is None:
            if total_pop is None:
                self['pop_scale'] = 1.0
            else:
                self['pop_scale'] = total_pop/self['n_agents']

        # Finish initialization
        self.people.initialize(sim_pars=self.pars) # Fully initialize the people

        return self


    def init_results(self, frequency='annual', add_data=True):
        '''
        Create the main results structure.
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/etc) on any particular timestep

        Arguments:
            sim         (hpv.Sim)       : a sim
            frequency   (str or float)  : the frequency with which to save results: accepts 'annual', 'dt', or a float which is interpreted as a fraction of a year, e.g. 0.2 will save results every 0.2 years
            add_data    (bool)          : whether or not to add data to the result structures
        '''

        # Handle frequency
        if type(frequency) == str:
            if frequency == 'annual':
                resfreq = int(1 / self['dt'])
            elif frequency == 'dt':
                resfreq = 1
            else:
                errormsg = f'Result frequency not understood: must be "annual", "dt" or a float, but you provided {frequency}.'
                raise ValueError(errormsg)
        elif type(frequency) == float:
            if frequency < self['dt']:
                errormsg = f'You requested results with frequency {frequency}, but this is smaller than the simulation timestep {self["dt"]}.'
                raise ValueError(errormsg)
            else:
                resfreq = int(frequency / self['dt'])
        self.resfreq = resfreq
        if not self.resfreq > 0:
            errormsg = f'The results frequence should be a positive integer, not {self.resfreq}: dt may be too large'
            raise ValueError(errormsg)

        # Construct the tvec that will be used with the results
        points_to_use = np.arange(0, self.npts, self.resfreq)
        self.res_yearvec = self.yearvec[points_to_use]
        self.res_npts = len(self.res_yearvec)
        self.res_tvec = np.arange(self.res_npts)

        # Function to create results
        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = ssb.Result(*args, **kwargs, npts=self.res_npts)
            return output

        # Initialize storage
        results = sc.objdict()
        na = len(self['age_bin_edges']) - 1  # Number of age bins

        # Demographics
        dem_keys = ['births', 'other_deaths', 'migration']
        dem_names = ['births', 'other deaths', 'migration']
        dem_colors = ['#fcba03', '#000000', '#000000']

        # Results by sex
        by_sex_keys = ['infections_by_sex', 'other_deaths_by_sex']
        by_sex_names = ['infections by sex', 'deaths from other causes by sex']
        by_sex_colors = ['#000000', '#000000']

        # Create demographic flows
        for var, name, color in zip(dem_keys, dem_names,dem_colors):
            results[var] = init_res(name, color=color)

        # Create results by sex
        for var, name, color in zip(by_sex_keys, by_sex_colors, by_sex_colors):
            results[var] = init_res(name, color=color, n_rows=2)

        # Other results
        results['n_alive'] = init_res('Number alive')
        results['n_alive_by_sex'] = init_res('Number alive by sex', n_rows=2)
        results['n_alive_by_age'] = init_res('Number alive by age', n_rows=na)
        results['cdr'] = init_res('Crude death rate', scale=False)
        results['cbr'] = init_res('Crude birth rate', scale=False, color='#fcba03')

        # Time vector
        results['year'] = self.res_yearvec
        results['t'] = self.res_tvec

        # Final items
        self.results = results
        self.results_ready = False

        return



    def step(self):
        ''' Step through time and update values '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Shorten key variables
        t = self.t
        year = self.yearvec[t]

        # Update states, modules, partnerships
        self.people.update_states(t=t, sim=self) # This runs modules
        self.update_connectors()

        for module in self.modules.values():
            module.make_new_cases(self)
            module.update_results(self)

        # Do demographic updates
        # Occurs after running modules in case modeling pregnancies to get migration right
        self.people.update_demography(t=t, year=year)  # This ages people and does births, deaths, migrations

        # Index for results
        resfreq = int(1 / self['dt'])
        idx = int(t / resfreq)

        # Update counts for this time step: flows
        for key,count in self.people.demographic_flows.items():
            self.results[key][idx] += count

        # Make stock updates every nth step, where n is the frequency of result output
        if t % resfreq == resfreq-1:

            # Save number alive
            alive_inds = ssu.true(self.people.alive)
            alive_female_inds = ssu.true(self.people.alive*self.people.is_female)
            self.results['n_alive'][idx] = self.people.scale_flows(alive_inds)
            self.results['n_alive_by_sex'][0,idx] = self.people.scale_flows((self.people.alive*self.people.is_female).nonzero()[0])
            self.results['n_alive_by_sex'][1,idx] = self.people.scale_flows((self.people.alive*self.people.is_male).nonzero()[0])
            self.results['n_alive_by_age'][:,idx] = np.histogram(self.people.age[alive_inds], bins=self.people.age_bin_edges, weights=self.people.scale[alive_inds])[0]

        # Tidy up
        self.t += 1
        if self.t == self.npts:
            self.complete = True

        return


    def run(self, until=None, restore_pars=True, reset_seed=True, verbose=None):
        ''' Run the model once '''
        # Initialization steps -- start the timer, initialize the sim and the seed, and check that the sim hasn't been run
        T = sc.timer()

        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(self.pars) # Create a copy of the parameters, to restore after the run, in case they are dynamically modified

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
            # The seed is offset by 1 to avoid drawing the same random numbers as those used for population generation, otherwise
            # the first set of random numbers in the model (e.g., deaths) will be correlated with the first set of random numbers
            # drawn in population generation (e.g., sex)
            ssu.set_seed(self['rand_seed']+1)

        # Check for AlreadyRun errors
        errormsg = None
        if until is None: until = self.npts
        if until > self.npts:
            errormsg = f'Requested to run until t={until} but the simulation end is t={self.npts}'
        if self.t >= until: # NB. At the start, self.t is None so this check must occur after initialization
            errormsg = f'Simulation is currently at t={self.t}, requested to run until t={until} which has already been reached'
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.initialize() to re-run)'
        if self.people.t not in [self.t, self.t-1]: # Depending on how the sim stopped, either of these states are possible
            errormsg = f'The simulation has been run independently from the people (t={self.t}, people.t={self.people.t}): if this is intentional, manually set sim.people.t = sim.t. Remember to save the people object before running the sim.'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop
        while self.t < until:

            # Check if we were asked to stop
            elapsed = T.toc(output=True)
            if self['timelimit'] and elapsed > self['timelimit']:
                sc.printv(f"Time limit ({self['timelimit']} s) exceeded; call sim.finalize() to compute results if desired", 1, verbose)
                return
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv("Stopping function terminated the simulation; call sim.finalize() to compute results if desired", 1, verbose)
                return

            # Print progress
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.yearvec[self.t]:0.1f} ({self.t:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose>0:
                    if not (self.t % int(1.0/verbose)):
                        sc.progressbar(self.t+1, self.npts, label=string, length=20, newline=True)

            # Actually run the model
            self.step()

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize(verbose=verbose)
            sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)

        return self


    def finalize(self, verbose=None):
        ''' Compute final results '''

        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Final settings
        self.results_ready = True # Set this first so self.summary() knows to print the results
        self.t -= 1 # During the run, this keeps track of the next step; restore this be the final day of the sim

        # Perform calculations on results
        # self.compute_results(verbose=verbose) # Calculate the rest of the results
        self.results = sc.objdict(self.results) # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses

        return

    def init_analyzers(self):
        ''' Initialize the analyzers '''

        self.analyzers = sc.autolist()

        # Interpret analyzers
        for ai, analyzer in enumerate(self['analyzers']):
            if isinstance(analyzer, type) and issubclass(analyzer, ssa.Analyzer):
                analyzer = analyzer()  # Convert from a class to an instance of a class
            if not (isinstance(analyzer, ssa.Analyzer) or callable(analyzer)):
                errormsg = f'Analyzer {analyzer} does not seem to be a valid analyzer: must be a function or hpv.Analyzer subclass'
                raise TypeError(errormsg)
            self.analyzers += analyzer  # Add it in

        for analyzer in self.analyzers:
            if isinstance(analyzer, ssa.Analyzer):
                analyzer.initialize(self)

        return

    def init_network(self):
        for i, network in enumerate(self['networks']):
            if network.label is not None:
                layer_name = network.label
            else:
                layer_name = f'layer{i}'
                network.label = layer_name
            network.initialize(self.people)
            self.people.contacts[layer_name] = network

        return

    def update_connectors(self):
        if len(self.modules) > 1:
            connectors = self['connectors']
            if len(connectors) > 0:
                for connector in connectors:
                    if callable(connector):
                        connector(self)
                    else:
                        warnmsg = f'Connector must be a callable function'
                        ssm.warn(warnmsg, die=True)
            elif self.t == 0:  # only raise warning on first timestep
                warnmsg = f'No connectors in sim'
                ssm.warn(warnmsg, die=False)
            else:
                return
        return


class AlreadyRunError(RuntimeError):
    '''
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    :py:func:`Sim.run` and not taking any timesteps, would be an inadvertent error.
    '''
    pass

