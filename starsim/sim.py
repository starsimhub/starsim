"""
Define core Sim classes
"""

# Imports
import numpy as np
import sciris as sc
import starsim as ss
import itertools
import numba as nb

__all__ = ['Sim', 'AlreadyRunError', 'diff_sims']

@nb.njit
def set_numba_seed(value):
    # Needed to ensure reproducibility when using random calls in numba, e.g. RandomNetwork
    # Note, these random numbers are not currently common-random-number safe
    np.random.seed(value)

class Sim(sc.prettyobj):
    """
    The Sim class in the provided code appears to represent a simulation. It has attributes for various aspects of a simulation, such as:
    """
    
    def __init__(self, pars=None, label=None, people=None, demographics=None, diseases=None, connectors=None, **kwargs):
        """
        ## Sim
        Initialize a new instance of the Sim class.
        
        ### Args
        ::
        
            pars (dict, optional):      Parameters for the simulation.
            label (str, optional):      The label/name of the simulation.
            people (People object, optional):   People object.
            demographics (ndict, optional):     Demographics of the simulation.
            diseases (ndict, optional):         Diseases in the simulation.
            connectors (ndict, optional):       Connectors in the simulation.
            **kwargs:     Additional keyword arguments.
            
        The Sim class in the provided code appears to represent a simulation. It has attributes for various aspects of a simulation, such as:
        
        ## Class Attributes
        ------------------
        ::

            label (str):          The label/name of the simulation.
            created (datetime):   The datetime the sim was created.
            people (People object): People object.
            demographics (ndict):   Demographics of the simulation.
            diseases (ndict):       Diseases in the simulation.
            connectors (ndict):     Connectors in the simulation.
            results (ndict):        For storing results.
            summary (Summary object): For storing a summary of the results.
            initialized (bool):     Whether initialization is complete.
            complete (bool):        Whether a simulation has completed running.
            results_ready (bool):   Whether results are ready.
            filename (str):         The filename of the simulation.
            ti (int):               The time index, e.g. 0, 1, 2.
            yearvec (list):         Year vector.
            tivec (list):           Time vector.
            npts (int):             Number of points.
            pars (Parameters object): Parameters of the simulation.
            interventions (ndict):  Interventions in the simulation.
            analyzers (ndict):      Analyzers in the simulation.
            rng_container (RNGContainer object): The random number generator container.
        
        The `Sim` class represents a simulation in the provided code. It encapsulates all the details and behaviors of a simulation, including its parameters, state, and methods for running and analyzing the simulation. Here's a breakdown of its functionality:

        1. `Attributes` It has attributes for various aspects of a simulation, such as the label/name of the simulation, the datetime the simulation was created, the people involved in the simulation, the demographics and diseases in the simulation, the connectors in the simulation, the results and summary of the simulation, the initialization and completion status, the filename of the simulation, the time index, the year and time vectors, the number of points, the parameters of the simulation, the interventions and analyzers in the simulation, and the random number generator container.

        2. `Initialization` The `__init__()` method is used to initialize a new instance of the `Sim` class. It takes arguments for the parameters, label, people, demographics, diseases, and connectors of the simulation, as well as any additional keyword arguments.

        3. `Methods` The `Sim` class would typically have methods for running the simulation, analyzing the results, saving and loading the simulation state, and other simulation-related tasks. These methods are not shown in the provided code excerpt.

        The `Sim` class provides a structured way to organize all the details and behaviors of a simulation, making it easier to create, run, and analyze simulations.
        """
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

        # Initialize the random number generator container
        self.rng_container = ss.RNGContainer()

        return

    @property
    def dt(self):
        """
        Get the time step size from the simulation parameters.
        
        Returns:
            float: The time step size.
        """
        return self.pars['dt']

    @property
    def year(self):
        """
        Get the current year of the simulation.
        
        Returns:
            int: The current year.
        """        
        return self.yearvec[self.ti]

    @property
    def modules(self):
        """
        Get an iterator over all Module instances stored in standard places in the Sim.
        
        Returns:
            iterator: An iterator over all Module instances.
        """        
        # Return iterator over all Module instances (stored in standard places) in the Sim
        products = [intv.product for intv in self.interventions.values() if hasattr(intv, 'product') and isinstance(intv.product, ss.Product)]
        return itertools.chain(
            self.demographics.values(),
            self.people.networks.values(),
            self.diseases.values(),
            self.connectors.values(),
            self.interventions.values(),
            products,
            self.analyzers.values(),
        )

    def initialize(self, popdict=None, reset=False, **kwargs):
        """
        Perform all initializations on the sim.
        
        ### Args:
        ::
        
            popdict (dict, optional):   A dictionary representing the population.
            reset (bool, optional):     Whether to reset the simulation.
            **kwargs:                   Additional keyword arguments.
            
        The `initialize()` function is typically used in a simulation to set up the initial state of the simulation 
        before it starts running. Here's a breakdown of its functionality:

        1. `Setting initial parameters:` It sets up the initial parameters of the simulation. This could include
        things like the initial population size, the initial number of infected individuals, the transmission rate, etc.
        2. `Creating initial objects:` It creates the initial objects that make up the simulation. In a disease 
        simulation, this could include creating individual people, setting up the social network that connects them, etc.
        3. `Setting initial state:` It sets the initial state of the simulation. This could involve setting the 
        initial health status of each individual, setting the initial time step, etc.
        4. `Preparing for the simulation run:` It prepares the simulation for running. This could involve setting 
        up data structures to store the results, initializing random number generators, etc.
        The exact details of what the `initialize()` function does can vary depending on the specifics of the 
        simulation.            
            
        """
        # Validation and initialization
        self.ti = 0  # The current time index
        self.validate_pars()  # Ensure parameters have valid values
        self.validate_dt()
        self.init_time_vecs()  # Initialize time vecs
        ss.set_seed(self.pars['rand_seed'])  # Reset the random seed before the population is created
        set_numba_seed(self.pars['rand_seed'])

        # Initialize the core sim components
        self.rng_container.initialize(self.pars['rand_seed'] + 2) # +2 ensures that seeds from the above population initialization and the +1-offset below are not reused within the rng_container
        self.init_people(reset=reset, **kwargs)  # Create all the people (the heaviest step)
        self.init_demographics()
        self.init_networks()
        self.init_diseases()
        self.init_connectors()
        self.init_interventions()
        self.init_analyzers()

        # Perform post-initialization validation
        self.validate_post_init()

        # Reset the random seed to the default run seed, so that if the simulation is run with
        # reset_seed=False right after initialization, it will still produce the same output
        ss.set_seed(self.pars['rand_seed'] + 1) # Hopefully not used now that we can use multiple random number generators

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
        return

    def validate_pars(self):
        """
        Some parameters can take multiple types; this makes them consistent.
        This method validates the parameters of the simulation, ensuring they are of the correct type and value.
        Raises ValueError if the parameters are not valid.
        """
        # Handle n_agents
        if self.people is not None:
            self.pars['n_agents'] = len(self.people)
        #elif self.popdict is not None: # Starsim does not currenlty support self.popdict
            #self.pars['n_agents'] = len(self.popdict)
        elif self.pars['n_agents'] is not None:
            self.pars['n_agents'] = int(self.pars['n_agents'])
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
        Construct vectors that keep track of time.
        This method initializes the time vectors of the simulation, including the year vector and the time index vector.
        """
        self.yearvec = sc.inclusiverange(start=self.pars['start'], stop=self.pars['end'] + 1 - self.pars['dt'],
                                         step=self.pars['dt'])  # Includes all the timepoints in the last year
        self.npts = len(self.yearvec)
        self.tivec = np.arange(self.npts)
        return

    def init_people(self, reset=False, verbose=None, **kwargs):
        """
        ### Args:
        ::
        
            reset (bool):       Whether to regenerate the people even if they already exist.
            verbose (int):      Detail to print.
            **kwargs (dict):    Passed to ss.make_people().
            
        The init_people function is used to initialize the people within the simulation.
        
        - Sometimes the people are provided, in which case this just adds a few sim properties to them.
        - If people have not been provided or if the reset parameter is set to True, it creates a new People object with the number of agents specified in the simulation parameters (self.pars['n_agents']). This new People object is then assigned to self.people.
        - If a location is specified in the simulation parameters (self.pars['location']), it checks where to get the total population from. If self.pars['total_pop'] is not None, it raises a ValueError because you can either define total_pop explicitly or via the location, but not both.
        - If self.pars['total_pop'] is None, it tries to get the total population from the location. If self.pars['pop_scale'] is not None, it calculates the total population as self.pars['pop_scale'] * self.pars['n_agents']. Otherwise, it sets the total population to self.pars['n_agents'].
        - It then sets self.pars['total_pop'] to the total population and if self.pars['pop_scale'] is None, it calculates the population scale as total_pop / self.pars['n_agents'].
        - Finally, if self.people is not initialized, it calls self.people.initialize(self) to initialize it. It also sets the time attributes of self.people and calls self.people.init_results(self) to initialize the results.
        
        """

        # Handle inputs
        if verbose is None:
            verbose = self.pars['verbose']
        if verbose > 0:
            resetstr = ''
            if self.people and reset:
                resetstr = ' (resetting people)'
            print(f'Initializing sim{resetstr} with {self.pars["n_agents"]:0n} agents')

        # If people have not been supplied, make them
        if self.people is None or reset:
            self.people = ss.People(n=self.pars['n_agents'], **kwargs)  # This just assigns UIDs and length

        # If a popdict has not been supplied, we can make one from location data
        if self.pars['location'] is not None:
            # Check where to get total_pop from
            if self.pars['total_pop'] is not None:  # If no pop_scale has been provided, try to get it from the location
                errormsg = 'You can either define total_pop explicitly or via the location, but not both'
                raise ValueError(errormsg)

        else:
            if self.pars['total_pop'] is not None:  # If no pop_scale has been provided, try to get it from the location
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
            self.people.initialize(self)

        # Set time attributes
        self.people.ti = self.ti
        self.people.dt = self.dt
        self.people.year = self.year
        self.people.init_results(self)
        return self

    def init_demographics(self):
        """
        The ``init_demographics`` function is used to initialize the demographics modules for the simulation.

        For each module in the demographics dictionary, it calls the initialize method of the module, passing 
        the current simulation instance (self) as an argument. This is typically used to set up any initial 
        state or parameters that the module needs to operate.

        After initializing the module, it adds the module's results to the simulation's results dictionary. 
        The key for each set of results is the name of the module. This allows the simulation to keep track 
        of the results produced by each demographics module.
        """        
        for module in self.demographics.values():
            module.initialize(self)
            self.results[module.name] = module.results

    def init_diseases(self):
        """ 
        The ``init_diseases`` function is used to initialize the disease modules for the simulation.

        For each disease in the diseases dictionary, it calls the initialize method of the disease, 
        passing the current simulation instance (self) as an argument. This is typically used to set up 
        any initial state or parameters that the disease module needs to operate.

        After initializing the disease, it adds the disease's parameters and results to the simulation's 
        pars and results dictionaries respectively. The key for each set of parameters and results is the 
        name of the disease. This allows the simulation to keep track of the parameters and results produced 
        by each disease module.

        Finally, it adds the disease states to the People object's dictionaries by calling the add_module 
        method of self.people with the disease as an argument. This allows the People object to keep track 
        of the state of each disease for each person in the simulation.
        """
        for disease in self.diseases.values():
            disease.initialize(self)

            # Add the disease's parameters and results into the Sim's dicts
            self.pars[disease.name] = disease.pars
            self.results[disease.name] = disease.results

            # Add disease states to the People's dicts
            self.people.add_module(disease)

        return

    def init_connectors(self):
        """
        The ``init_connectors`` function is used to initialize the connector modules 
        for the simulation.

        - For each connector in the connectors dictionary, it calls the initialize 
        method of the connector, passing the current simulation instance (self) as an argument. 
        This is typically used to set up any initial state or parameters that the connector 
        module needs to operate.

        - Connectors in a simulation can represent various types of interactions 
        or connections between the agents (people) in the simulation, such as social 
        connections, physical connections, etc. These connectors can influence how diseases 
        spread or how interventions are applied in the simulation.        
        """        
        for connector in self.connectors.values():
            connector.initialize(self)

    def init_networks(self):
        
        """
        The ``init_networks`` function is used to initialize the networks for the simulation.

        - If the networks have been provided separately from the people (i.e., they 
        are stored in self.pars['networks']), it copies them to the People object (self.people.networks).

        - It then checks if self.people.networks is an instance of ss.Networks. 
        If not, it converts self.people.networks to an instance of ss.Networks.

        - Finally, it calls the initialize method of self.people.networks, passing 
        the current simulation instance (self) as an argument. This is typically used 
        to set up any initial state or parameters that the networks need to operate.

        - Networks in a simulation can represent various types of interactions or connections 
        between the agents (people) in the simulation, such as social connections, physical 
        connections, etc. These networks can influence how diseases spread or how interventions 
        are applied in the simulation.
        """
        
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
        """
        The ``init_interventions`` function is used to initialize and validate the 
        interventions for the simulation.

        - It iterates over each intervention in self.pars['interventions']. 
        If an intervention is a class that is a subclass of ss.Intervention, 
        it converts it to an instance of the class. If an intervention is an 
        instance of ss.Intervention, it calls the initialize method of the intervention, 
        passing the current simulation instance (self) as an argument. This is typically 
        used to set up any initial state or parameters that the intervention needs to operate.

        - The initialized intervention is then added to self.interventions.

        - If an intervention is a callable (i.e., a function), it is directly added to self.interventions.

        - If an intervention is not a class or instance of ss.Intervention and is not callable, 
        a TypeError is raised.

        - Interventions in a simulation can represent various types of actions 
        or strategies that are applied to the agents (people) in the simulation to 
        influence the course of the simulation, such as vaccination, quarantine, etc.
        """

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

            # Add the intervention parameters and results into the Sim's dicts
            self.pars[intervention.name] = intervention.pars
            self.results[intervention.name] = intervention.results

            # Add intervention states to the People's dicts
            self.people.add_module(intervention)

            # Intervention and product RNGs
            for rng in intervention.rngs:
                rng.initialize(self.rng_container, self.people.slot)

            # If there's a product module present, initialize and add it
            if hasattr(intervention, 'product') and isinstance(intervention.product, ss.Product):
                intervention.product.initialize(self)
                self.people.add_module(intervention.product)
                for rng in intervention.product.rngs:
                    rng.initialize(self.rng_container, self.people.slot)

        return

    def init_analyzers(self):
        """
        The ``init_analyzers`` function is used to initialize the analyzers for the simulation.

        - It iterates over each analyzer in self.pars['analyzers']. If an analyzer is a class that 
        is a subclass of ss.Analyzer, it converts it to an instance of the class. If an analyzer is 
        not an instance of ss.Analyzer and is not callable (i.e., a function), a TypeError is raised.

        - The initialized analyzer is then added to self.analyzers.

        - Finally, for each analyzer in self.analyzers.values(), if it is an instance of ss.Analyzer, 
        it calls the initialize method of the analyzer, passing the current simulation instance (self) 
        as an argument. This is typically used to set up any initial state or parameters that the analyzer needs to operate.

        - Analyzers in a simulation can represent various types of data analysis or data collection 
        strategies that are applied to the agents (people) in the simulation to gather and analyze 
        data during the course of the simulation.

        Raises:
            TypeError: _description_
        """
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
        """
        The ``step`` function is the main function that advances the simulation by one time step. Here's a breakdown of what it does:

        - Check if the simulation is complete: If the simulation has already been run to completion, it raises an AlreadyRunError.
        - Advance random number generators: This is done in preparation for any random number calls that may be necessary on this step.
        - Remove dead agents: If the simulation parameter remove_dead is set to True, it removes any agents that have died.
        - Update demographic modules: This could involve creating new agents due to births or immigration, and scheduling non-disease deaths and emigration.
        - Update disease modules: This allows for autonomous state changes/initializations to be applied to newly created agents.
        - Update connectors: The exact function of connectors can vary, but they generally represent some form of interaction or connection between agents.
        - Update networks: This updates the state of the networks, which can affect eligibility for contacts.
        - Apply interventions: Any interventions that are scheduled for this time step are applied.
        - Carry out transmission/new cases: The disease modules calculate and carry out any new cases of the disease.
        - Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed before 
        analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep        
        - update people, disease and analyzer results
        """        
        
        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Advance random number generators forward to prepare for any random number calls that may be necessary on this step
        self.rng_container.step(self.ti+1) # +1 offset because ti=0 is used on initialization

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
        uids = self.people.resolve_deaths()
        for disease in self.diseases.values():
            disease.update_death(self, uids)

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
        """
        The ``run`` function initializes the simulation if it has not been initialized, checks for errors, and then enters 
        the main simulation loop. 
        In each iteration of the loop, it checks if the simulation has been asked to stop, prints progress if verbosity is 
        enabled, and then advances the simulation by one step.
        If the simulation reaches the end, it finalizes the results.


        ### Args:
        ::
        
            until (int, optional):      The time step at which to stop the simulation. If None, the simulation runs until the end.
            verbose (int, optional):    Level of verbosity during the simulation run. If None, the verbosity level is taken from the simulation parameters.
            reset_seed (bool, optional): If True, the random seed is reset before the simulation run.

        ### Raises:
            `AlreadyRunError`: If the simulation has already been run to completion, or if the requested end time has 
            already been reached or is beyond the end of the simulation.

        ### Returns:
            self: The simulation instance, for chaining commands.

        The ``run`` function is the main driver of the simulation. It's responsible for initializing the simulation, running it 
        for a specified number of steps, and finalizing the results. 
        Here's a breakdown of its functionality:

        - Initialization: If the simulation hasn't been initialized yet, it calls the initialize() 
        method to set up the simulation. It also stores a copy of the original parameters.
        - Setting verbosity and random seed: The verbosity level and random seed are set based on 
        the function arguments and simulation parameters.
        - Checking for AlreadyRun errors: It checks if the simulation has already been run to completion, 
        or if the requested end time has already been reached or is beyond the end of the simulation. 
        If any of these conditions are met, it raises an AlreadyRunError.
        - Main simulation loop: It enters a loop that runs until the current time step (self.ti) is less 
        than the specified end time (until). In each iteration of the loop, it checks if the simulation 
        has been asked to stop, prints progress if verbosity is enabled, and then advances the simulation by one step by calling the step() method.
        - Finalizing results: If the simulation reaches the end (i.e., self.complete is True), it calls the finalize() method to finalize the results.
        The function returns the simulation instance itself, allowing for method chaining.
        
        """
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
        """
        ### Summary:
            The ``finalize`` function is called at the end of the simulation run to compute and
            prepare the final results for analysis. Here's a breakdown of its functionality:

            - Check if results are ready: If the results have already been finalized (i.e., 
            self.results_ready is True), it raises an AlreadyRunError to prevent the results 
            from being finalized again.

            - Scale the results: For each result in self.results, if the result is an instance 
            of ss.Result and its scale attribute is True, it scales the result by the population 
            scale factor (self.pars.pop_scale).

            - Finalize modules: It calls the finalize method of each module in self.modules, 
            passing the simulation instance (self) as an argument. This allows each module to 
            perform any necessary final calculations or clean up.

            - Summarize results: It calls the summarize method to compute a summary of the results.

            - Update simulation state: It sets self.results_ready to True to indicate that the results 
            are ready for analysis, and it decrements self.ti by 1 to restore it to the final day of the simulation.

            The function does not return anything. Its purpose is to modify the state of the simulation instance.

        ### Raises:
            AlreadyRunError: _description_
        """
        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Scale the results
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result) and res.scale:
                self.results[reskey] = self.results[reskey]*self.pars.pop_scale

        for module in self.modules:
            module.finalize(self)

        self.summarize()
        self.results_ready = True  # Set this first so self.summary() knows to print the results
        self.ti -= 1  # During the run, this keeps track of the next step; restore this be the final day of the sim
        return
    
    def summarize(self):
        summary = sc.objdict()
        flat = sc.flattendict(self.results, sep='_')
        for k,v in flat.items():
            summary[k] = v.mean()
        self.summary = summary
        return summary
        

    def shrink(self, skip_attrs=None, in_place=True):
        """
        ### Summary:
        The ``shrink`` function is used to reduce the memory required for RAM or for saved files.
        
        ### Args:
        ::
        
            skip_attrs (list):  A list of attributes to skip (remove) in order to perform the shrinking; default "people"
            in_place (bool):    Whether to perform the shrinking in place (default), or return a shrunken copy instead

        ###  Returns:
            shrunken (Sim): a Sim object with the listed attributes removed

        The `shrink()` function is used to reduce the memory footprint of the simulation by removing 
        memory-intensive attributes. This can be useful when you want to save the simulation results 
        but don't need all the details of the simulation state, or when you're running many simulations and need to conserve RAM.
        Here's a breakdown of its functionality:

        - `Specify attributes to remove` If no attributes are provided, by default removes the `people` 
        attribute, which typically takes up the majority of the memory. You can specify additional or 
        different attributes to remove by passing a list to the `skip_attrs` argument  (e.g., some interventions and analyzers).
        - `Remove attributes` If `in_place` is True, it removes the specified attributes from the 
        simulation instance itself. If `in_place` is False, it creates a new simulation instance and copies over
        all attributes except the ones specified to be removed.
        - `Return shrunken simulation` If `in_place` is False, it returns the new, shrunken simulation instance. 
        - If `in_place` is True, it doesn't return anything, as the original simulation instance has been modified in place.

        This function is a good example of how you can manage memory usage in large-scale simulations.        
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

    def _get_ia(self, which, label=None, partial=False, as_list=False, as_inds=False, die=True, first=False):
        """ Helper method for get_interventions() and get_analyzers(); see get_interventions() docstring """

        # Handle inputs
        if which not in ['interventions', 'analyzers']: # pragma: no cover
            errormsg = f'This method is only defined for interventions and analyzers, not "{which}"'
            raise ValueError(errormsg)

        ia_ndict = self.analyzers if which == 'analyzers' else self.interventions # List of interventions or analyzers
        n_ia = len(ia_ndict)  # Number of interventions/analyzers

        position = 0 if first else -1 # Choose either the first or last element
        if label is None:  # Get all interventions if no label is supplied, e.g. sim.get_interventions()
            label = np.arange(n_ia)
        if isinstance(label, np.ndarray):  # Allow arrays to be provided
            label = label.tolist()
        labels = sc.promotetolist(label)

        # Calculate the matches
        matches = []
        match_inds = []

        for label in labels:
            if sc.isnumber(label):
                matches.append(ia_ndict[label])
                label = n_ia + label if label < 0 else label  # Convert to a positive number
                match_inds.append(label)
            elif sc.isstring(label) or isinstance(label, type):
                for ind, ia_key, ia_obj in ia_ndict.enumitems():
                    if sc.isstring(label) and ia_obj.label == label or (partial and (label in str(ia_obj.label))):
                        matches.append(ia_obj)
                        match_inds.append(ind)
                    elif isinstance(label, type) and isinstance(ia_obj, label):
                        matches.append(ia_obj)
                        match_inds.append(ind)
            else: # pragma: no cover
                errormsg = f'Could not interpret label type "{type(label)}": should be str, int, list, or {which} class'
                raise TypeError(errormsg)

        # Parse the output options
        if as_inds:
            output = match_inds
        elif as_list: # Used by get_interventions()
            output = matches
        else:
            if len(matches) == 0: # pragma: no cover
                if die:
                    errormsg = f'No {which} matching "{label}" were found'
                    raise ValueError(errormsg)
                else:
                    output = None
            else:
                output = matches[position] # Return either the first or last match (usually), used by get_intervention()

        return output

    def get_interventions(self, label=None, partial=False, as_inds=False):
        """
        `get_interventions` function finds the matching intervention(s) by label, index, or type. 
        
        - If None, returns all interventions. 
        - If the label provided is "summary", then print a summary
        of the interventions (index, label, type).

        ### Args:
        ::
        
            label (str, int, Intervention, list): 
                the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool):             
                if true, return partial matches (e.g. 'beta' will match all beta interventions)
            as_inds (bool): 
                if true, return matching indices instead of the actual interventions
        
        ### Returns:
        list: A list of retrieved interventions or their indices.

        
        The get_interventions() function is a method of the Sim class that retrieves interventions 
        applied in a simulation based on certain criteria.

        Here's a breakdown of its functionality:

        - Input: It takes as input a label, which can be a string, integer, Intervention object, 
        or a list of these. It also takes two boolean flags: partial and as_inds. If partial is true, 
        it returns partial matches (e.g., 'beta' will match all beta interventions). If as_inds is true, 
        it returns matching indices instead of the actual interventions.

        - Retrieval: It calls the _get_ia() method of the Sim class, passing 'interventions' 
        as the first argument and the input arguments as keyword arguments. The _get_ia() method 
        likely retrieves interventions that match the given criteria.

        - Output: It returns the retrieved interventions. If as_inds is true, this will be a list of 
        indices. Otherwise, it will be a list of Intervention objects.

        If no label is provided, it returns all interventions. If the label provided is "summary", 
        then it prints a summary of the interventions (index, label, type).   
            
        """
        return self._get_ia('interventions', label=label, partial=partial, as_inds=as_inds, as_list=True)


    def get_intervention(self, label=None, partial=False, first=False, die=True):
        """
        Find the matching intervention(s) by label, index, or type.
        If more than one intervention matches, return the last by default.
        If no label is provided, return the last intervention in the list.

        ### Args:
        ::
        
            label (str, int, Intervention, list, optional): 
                The label, index, or type of intervention to get. If a list, iterate over one of those types. Defaults to None.
            partial (bool, optional): 
                If true, return partial matches. Defaults to False.
            first   (bool, optional): 
                If true, return first matching intervention (otherwise, return last). Defaults to False.
            die     (bool, optional): 
                Whether to raise an exception if no intervention is found. Defaults to True.
            
        ### Returns:
            Intervention: The matching intervention.
        
        """
        return self._get_ia('interventions', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)

    def save(self, filename=None, keep_people=None, skip_attrs=None, **kwargs):
        """
        Save to disk as a gzipped pickle.

        ### Args:
            ``filename`` (str or None):     The name or path of the file to save to; if None, uses stored
            ``keep_people`` (bool or None): Whether to keep the people
            ``skip_attrs`` (list):          Attributes to skip saving
            ``**kwargs``:                   Passed to sc.makefilepath()

        ### Returns:
            filename (str): the validated absolute path to the saved file
            
        The `save()` function is used to save the current state of the simulation to disk as a gzipped pickle file. This can be useful for checkpointing the simulation, for saving the final state of the simulation for later analysis, or for sharing the simulation with others. 
        
        Here's a breakdown of its functionality:
        
        - `Set keep_people:` If keep_people is not specified, it determines whether to keep the people attribute based on the current state of the simulation. If the simulation is in the middle of a run, it keeps people; otherwise, it doesn't.

        - `Handle the filename:` If filename is not specified, it uses the stored filename (self.simfile). It then validates and normalizes the filename using sc.makefilepath() and stores the actual saved filename in self.filename.

        - `Handle the shrinkage and save:` If skip_attrs is specified or keep_people is False, it creates a shrunken copy of the simulation using self.shrink() and saves that. Otherwise, it saves the simulation itself. The saving is done using sc.save().

        - `Return the filename:` It returns the validated absolute path to the saved file.

        The function takes several arguments that control its behavior, including the filename to save to, whether to keep the people attribute, and a list of additional attributes to skip when saving.

        ### Example
        ::
        
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
        """ Load from disk from a gzipped pickle. 
        
        ### Args:
            ``filename`` (str): The name or path of the file from which to load the simulation.
            ``*args``: Additional arguments passed to sc.load().
            ``**kwargs``: Additional keyword arguments passed to sc.load().

        ### Raises:
            TypeError: If the loaded object is not a Sim object.

        ### Returns:
            Sim: The loaded simulation object.
         
        The ``load``  function is used to load a previously saved simulation from a file. 
        This allows you to save the state of a simulation, then load it later to continue 
        the simulation or analyze the results.

        Here's a breakdown of its functionality:

        - Input: It takes as input the filename of the file from which to load the simulation. 
        This file would typically have been created by a previous call to Sim.save().
        - Loading: It reads the file and reconstructs the simulation object. 
        This typically involves deserializing the file contents into a dictionary, then using 
        this dictionary to set the attributes of a new Sim object.
        - Output: It returns the loaded simulation object."""

        sim = sc.load(filename, *args, **kwargs)
        if not isinstance(sim, Sim):  # pragma: no cover
            errormsg = f'Cannot load object of {type(sim)} as a Sim object'
            raise TypeError(errormsg)
        return sim

    def export_pars(self, filename=None, indent=2, *args, **kwargs):
        '''
        ## export_pars
        Return parameters for JSON export -- see also to_json().

        This method is required so that interventions can specify
        their JSON-friendly representation.

        ### Args:
            ``filename`` (str): filename to save to; if None, do not save
            ``indent`` (int): indent (int): if writing to file, how many indents to use per nested level
            ``*args`` (list): passed to savejson()
            ``**kwargs`` (dict): passed to savejson()

        ### Returns:
            pardict (dict): a dictionary containing all the parameter values
        '''
        pardict = {}
        for key in self.pars.keys():
            if key == 'interventions':
                pardict[key] = [intervention.to_json() for intervention in self.pars[key]]
            elif key == 'start_day':
                pardict[key] = str(self.pars[key])
            else:
                pardict[key] = self.pars[key]
        if filename is not None:
            sc.savejson(filename=filename, obj=pardict, indent=indent, *args, **kwargs)
        return pardict

    def to_json(self, filename=None, keys=None, tostring=False, indent=2, verbose=False, *args, **kwargs):
        '''
        ## to_json
        Export results and parameters as JSON.

        ### Args
            ``filename`` (str): if None, return string; else, write to file
            ``keys`` (str or list): attributes to write to json (default: results, parameters, and summary)
            ``tostring`` (bool): if not writing to file, whether to write to string (alternative is sanitized dictionary)
            ``indent`` (int): if writing to file, how many indents to use per nested level
            ``verbose`` (bool): detail to print
            ``*args`` (list): passed to savejson()
            ``**kwargs`` (dict): passed to savejson()

        ### Returns
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        The `to_json()` function is used to convert the simulation object into a JSON format. JSON (JavaScript Object Notation) is a popular data interchange format that is easy to read and write for humans and easy to parse and generate for machines.

        Here's a breakdown of its functionality:

        1. `Serialization:` It serializes the simulation object, converting it into a format that can be written to a file. This typically involves converting the object's attributes into a dictionary.

        2. `Conversion to JSON:` It converts the serialized object into a JSON string. This is done using a JSON library, such as json in Python or JSON in JavaScript.

        3. `Writing to file:` If a filename is provided, it writes the JSON string to a file. Otherwise, it returns the JSON string.

        This function is useful when you want to save the state of the simulation in a format that can be easily shared or read by other programs. It's also useful for debugging, as you can inspect the JSON to see the state of the simulation at a particular point in time.
        
        ### Examples
        ::
        
            json = sim.to_json()
            sim.to_json('results.json')
            sim.to_json('summary.json', keys='summary')
        
        '''

        # Handle keys
        if keys is None:
            keys = ['results', 'pars', 'summary', 'short_summary']
        keys = sc.promotetolist(keys)

        # Convert to JSON-compatible format
        d = {}
        for key in keys:
            if key == 'results':
                if self.results_ready:
                    resdict = self.export_results(for_json=True)
                    d['results'] = resdict
                else:
                    d['results'] = 'Results not available (Sim has not yet been run)'
            elif key in ['pars', 'parameters']:
                pardict = self.export_pars()
                d['parameters'] = pardict
            elif key == 'summary':
                if self.results_ready:
                    d['summary'] = dict(sc.dcp(self.summary))
                else:
                    d['summary'] = 'Summary not available (Sim has not yet been run)'
            elif key == 'short_summary':
                if self.results_ready:
                    d['short_summary'] = dict(sc.dcp(self.short_summary))
                else:
                    d['short_summary'] = 'Full summary not available (Sim has not yet been run)'
            else: # pragma: no cover
                try:
                    d[key] = sc.sanitizejson(getattr(self, key))
                except Exception as E:
                    errormsg = f'Could not convert "{key}" to JSON: {str(E)}; continuing...'
                    print(errormsg)

        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, *args, **kwargs)

        return output
    
    def plot(self):
        flat = sc.flattendict(self.results, sep=': ')
        fig, axs = sc.getrowscols(len(flat), make=True)
        for ax,(k,v) in zip(axs.flatten(), flat.items()):
            ax.plot(v)
            ax.set_title(k)
        return fig
            

class AlreadyRunError(RuntimeError):
    """
    ## AlreadyRunError
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    :py:func:`Sim.run` and not taking any timesteps, would be an inadvertent error.
    """
    pass


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    ## diff_sims
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    ### Parameters
    ``sim1`` : (sim/dict) 
        Either a simulation object or the simulation summary dictionary.
        
    ``sim2`` : (sim/dict)    
        A simulation object or a simulation summary dictionary to compare against.
        
    ``skip_key_diffs`` : bool, optional
        If True, keys that don't match between simulations are skipped. Defaults to False.
        
    ``skip`` : list, optional
        A list of keys to skip during comparison. Defaults to None.
        
    ``full`` : bool, optional      
        If True, uses the full summary for comparison. Otherwise, uses a brief summary. Defaults to False.
        
    ``output`` : bool, optional
        If True, returns the output as a string. Otherwise, prints the output. Defaults to False.
        
    ``die`` : bool, optional       
        If True, raises an exception if the simulations don't match. Defaults to False.
    
    ### Raises:
        TypeError: If the provided simulations are not Sim objects or simulation summary dictionaries.

    ### Returns
        None or str: If output is True, returns a string detailing the differences between the simulations. Otherwise, prints the differences and returns None.

    ### Example
    ::
    
        s1 = hpv.Sim(rand_seed=1).run()
        s2 = hpv.Sim(rand_seed=2).run()
        hpv.diff_sims(s1, s2)
        
    '''

    if isinstance(sim1, Sim):
        sim1 = sim1.summarize()
    if isinstance(sim2, Sim):
        sim2 = sim2.summarize()
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict): # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs: # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra   = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\ns'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    skip = sc.tolist(skip)
    for key in sim2.keys(): # To ensure order
        if key in sim1_keys and key not in skip: # If a key is missing, don't count it as a mismatch
            sim1_val = sim1[key] if key in sim1 else 'not present'
            sim2_val = sim2[key] if key in sim2 else 'not present'
            if not np.isclose(sim1_val, sim2_val, equal_nan=True):
                mismatches[key] = {'sim1': sim1_val, 'sim2': sim2_val}

    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n'
        df = sc.dataframe.from_dict(mismatches).transpose()
        diff   = []
        ratio  = []
        change = []
        small_change = 1e-3 # Define a small change, e.g. a rounding error
        for mdict in mismatches.values():
            old = mdict['sim1']
            new = mdict['sim2']
            numeric = sc.isnumber(sim1_val) and sc.isnumber(sim2_val)
            if numeric and old>0:
                this_diff  = new - old
                this_ratio = new/old
                abs_ratio  = max(this_ratio, 1.0/this_ratio)

                # Set the character to use
                if abs_ratio<small_change:
                    change_char = '≈'
                elif new > old:
                    change_char = '↑'
                elif new < old:
                    change_char = '↓'
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio >= 1.1:
                    repeats = 2
                if abs_ratio >= 2:
                    repeats = 3
                if abs_ratio >= 10:
                    repeats = 4

                this_change = change_char*repeats
            else: # pragma: no cover
                this_diff   = np.nan
                this_ratio  = np.nan
                this_change = 'N/A'

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)

        df['diff'] = diff
        df['ratio'] = ratio
        for col in ['sim1', 'sim2', 'diff', 'ratio']:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg: # pragma: no cover
        if die:
            raise ValueError(mismatchmsg)
        elif output:
            return mismatchmsg
        else:
            print(mismatchmsg)
    else:
        if not output:
            print('Sims match')
    return
