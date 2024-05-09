"""
Set parameters
"""

from numbers import Number
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['Pars', 'SimPars', 'make_pars']

# Define classes to not descend into further -- based on sciris.sc_nested
atomic_classes = (str, Number, list, np.ndarray, pd.Series, pd.DataFrame, type(None))


class Pars(sc.objdict):
    """
    Dict-like container of parameters
    
    Acts like an ``sc.objdict()``, except that adding new keys are disallowed by
    default, and auto-updates known types.
    """
    def __init__(self, pars=None, **kwargs):
        if pars is not None:
            if isinstance(pars, dict):
                kwargs = pars | kwargs
            else:
                errormsg = f'Cannot supply parameters as type {type(pars)}: must be a dict'
                raise ValueError(errormsg)
        super().__init__(**kwargs)
        return
    
    def update(self, pars=None, create=False, **kwargs):
        """
        Update internal dict with new pars.
        
        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
            kwargs (dict): merged with pars
        """
        # Get parameters, and return if none
        pars = {} if pars is None else dict(pars) # Make it a simple dict, since just iterating over
        pars = pars | kwargs # Merge dictionaries
        if not len(pars): 
            return self
        
        # Check if there are any mismatches
        if not create:
            self.check_key_mismatch(pars)
        
        # Perform the update
        for key,new in pars.items():
            
            # Should only be the case if create=True
            if key not in self.keys(): 
                self[key] = new
            
            # Main use case: update from previous values
            else: 
                old = self[key] # Get the object we're about to update
                
                # It's a number, string, etc: update directly
                if isinstance(old, atomic_classes):
                    self[key] = new
                
                # It's a Pars object: update recursively
                elif isinstance(old, Pars): 
                    old.update(new, create=create)
                
                # Update module containers
                elif isinstance(old, ss.ndict):
                    if not len(old): # It's empty, just overwrite
                        self[key] = new
                    else: # Don't overwrite an existing ndict
                        if isinstance(new, dict):
                            for newkey,newvals in new.items():
                                old[newkey].pars.update(newvals) # e.g. pars = {'diseases": {'sir': {'dur_inf': 6}}}
                        else:
                            errormsg = f'Cannot update an ndict with {type(new)}: must be a dict to set new parameters'
                            raise TypeError(errormsg)
                
                # Update modules
                elif isinstance(old, ss.Module): 
                    if isinstance(new, dict):
                        old[key].pars.update(newvals) # e.g. pars = {'dur_inf': 6}
                    else:
                        errormsg = f'Cannot update a module with {type(new)}: must be a dict to set new parameters'
                        raise TypeError(errormsg)
                
                # Update a distribution
                elif isinstance(old, ss.Dist):
                    
                    # It's a Dist, e.g. dur_inf = ss.normal(6,2); use directly
                    if isinstance(new, ss.Dist): 
                        if isinstance(old, ss.bernoulli) and not isinstance(new, ss.bernoulli):
                            errormsg = f"Bernoulli distributions can't be changed to another type: {type(new)} is invalid"
                            raise TypeError(errormsg)
                        else:
                            self[key] = new
                    
                    # It's a single number, e.g. dur_inf = 6; set parameters
                    elif isinstance(new, Number):
                        old.set(new)
                    
                    # It's a list number, e.g. dur_inf = [6, 2]; set parameters
                    elif isinstance(new, list):
                        old.set(*new)
                    
                    # It's a dict, figure out what to do
                    elif isinstance(new, dict):
                        if 'type' not in new.keys(): # Same type of dist, set parameters
                            old.set(**new)
                        else: # We need to create a new distribution
                            newtype = new['type']
                            if isinstance(old, ss.bernoulli) and newtype != 'bernoulli':
                                errormsg = f"Bernoulli distributions can't be changed to another type: {newtype} is invalid"
                                raise TypeError(errormsg)
                            else:
                                dist = ss.make_dist(new)
                                self[key] = dist
                
                # Everything else
                else:
                    errormsg = 'No known mechanism for handling {type(old)} → {type(new)}; using default'
                    raise TypeError(errormsg)
                
        return self
        

    def check_key_mismatch(self, pars):
        """ Check whether additional keys are being added to the dictionary """
        available_keys = list(self.keys())
        new_keys = pars.keys()
        mismatches = [key for key in new_keys if key not in available_keys]
        if len(mismatches):
            errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return
    
    def dict_update(self, *args, **kwargs):
        """ Redefine default dict.update(), since overwritten in this class; should not usually be used """
        super().update(*args, **kwargs)
        return
    
    def to_json(self, **kwargs):
        """ Convert to JSON representation """
        return sc.jsonify(self, **kwargs)


class SimPars(Pars):
    """
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = ss.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        kwargs (dict): any additional kwargs are interpreted as parameter names
    """

    def __init__(self, **kwargs):
        
        # General parameters
        self.label   = '' # The label of the simulation
        self.verbose = ss.options.verbose # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

        # Population parameters
        self.n_agents  = 10e3  # Number of agents
        self.total_pop = None  # If defined, used for calculating the scale factor
        self.pop_scale = None  # How much to scale the population

        # Simulation parameters
        self.unit       = 'year' # The time unit to use (NOT YET IMPLEMENTED)
        self.start      = 2000   # Start of the simulation
        self.end        = None   # End of the simulation
        self.n_years    = 50     # Number of years to run, if end isn't specified. Note that this includes burn-in
        self.dt         = 1.0    # Timestep
        self.rand_seed  = 1      # Random seed, if None, don't reset
        self.slot_scale = 5      # Random slots will be assigned to newborn agents between min=n_agents and max=slot_scale*n_agents

        # Demographic parameters
        self.location   = None  #  NOT CURRENTLY FUNCTIONAL - what demographics to use
        self.birth_rate = None
        self.death_rate = None

        # Modules: demographics, diseases, connectors, networks, analyzers, and interventions
        self.people = None
        self.networks      = ss.ndict()
        self.demographics  = ss.ndict()
        self.diseases      = ss.ndict()
        self.connectors    = ss.ndict()
        self.interventions = ss.ndict()
        self.analyzers     = ss.ndict()

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)

        return

    def initialize(self, sim, reset=False, **kwargs):
        
        # Validation
        self.validate_dt()
        self.validate_pars()
        
        # Time indexing; derived values live in the sim rather than in the pars
        sim.dt = self.dt
        sim.yearvec = np.arange(start=self.start, stop=self.end + self.dt, step=self.dt)
        sim.results.yearvec = sim.yearvec # Copy this here
        sim.npts = len(sim.yearvec)
        sim.tivec = np.arange(sim.npts)
        sim.ti = 0  # The time index, e.g. 0, 1, 2
        
        # Initialize the people
        self.init_people(sim=sim, reset=reset, **kwargs)  # Create all the people (the heaviest step)
        
        # Allow shortcut for default demographics # TODO: think about whether we want to enable this, when we have birth_rate and death_rate
        if self.demographics == True:
            self.demographics = [ss.Births(), ss.Deaths()]
        
        # Get all modules into a consistent list format
        modmap = ss.module_map()
        for modkey in modmap.keys():
            if not isinstance(self[modkey], ss.ndict):
                self[modkey] = sc.tolist(self[modkey])
        
        # Initialize and convert modules
        self.convert_modules() # General initialization
        self.init_demographics() # Demographic-specific initialization
        
        # Convert from lists to ndicts
        for modkey,modclass in modmap.items():
            self[modkey] = ss.ndict(self[modkey], type=modclass)
        
        return self

    def validate_dt(self):
        """
        Check that 1/dt is an integer value, otherwise results and time vectors will have mismatching shapes.
        init_results explicitly makes this assumption by casting resfrequency = int(1/dt).
        """
        if self.unit == 'year': # TODO: implement properly
            dt = self.dt
            reciprocal = 1.0 / dt  # Compute the reciprocal of dt
            if not reciprocal.is_integer():  # Check if reciprocal is not a whole (integer) number
                # Round the reciprocal
                reciprocal = int(reciprocal)
                rounded_dt = 1.0 / reciprocal
                self.dt = rounded_dt
                if self.verbose:
                    warnmsg = f"Warning: Provided time step dt: {dt} resulted in a non-integer number of steps per year. Rounded to {rounded_dt}."
                    print(warnmsg)
        return
    
    def validate_pars(self):
        """
        Some parameters can take multiple types; this makes them consistent.
        """
        # Handle n_agents
        if self.people is not None:
            self.n_agents = len(self.people)
        elif self.n_agents is not None:
            self.n_agents = int(self.n_agents)
        else:
            errormsg = 'Must supply n_agents, a people object, or a popdict'
            raise ValueError(errormsg)

        # Handle end and n_years
        if self.end:
            self.n_years = self.end - self.start
            if self.n_years <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self.start)} and " \
                           f"end={str(self.end)}, which gives n_years={self.n_years}"
                raise ValueError(errormsg)
        else:
            if self.n_years:
                self.end = self.start + self.n_years
            else:
                errormsg = 'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Handle verbose
        if self.verbose == 'brief':
            self.verbose = -1
        if not sc.isnumber(self.verbose):  # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self.par.verbose)} "{self.par.verbose}"'
            raise ValueError(errormsg)
            
        return self
    
    def init_people(self, sim, reset=False, verbose=None, **kwargs):
        """
        Initialize people within the sim
        Sometimes the people are provided, in which case this just adds a few sim properties to them.
        Other time people are not provided and this method makes them.
        
        Args:
            reset   (bool): whether to regenerate the people even if they already exist
            verbose (int):  detail to print
            kwargs  (dict): passed to ss.make_people()
        """

        # Handle inputs
        if verbose is None:
            verbose = self.verbose
        if verbose > 0:
            resetstr = ''
            if self.people and reset:
                resetstr = ' (resetting people)'
            print(f'Initializing sim{resetstr} with {self.n_agents:0n} agents')

        # If people have not been supplied, make them
        if self.people is None or reset:
            self.people = ss.People(n_agents=self.n_agents, **kwargs)  # This just assigns UIDs and length

        # If a popdict has not been supplied, we can make one from location data
        if self.location is not None:
            # Check where to get total_pop from
            if self.total_pop is not None:  # If no pop_scale has been provided, try to get it from the location
                errormsg = 'You can either define total_pop explicitly or via the location, but not both'
                raise ValueError(errormsg)

        else:
            if self.total_pop is not None:  # If no pop_scale has been provided, try to get it from the location
                total_pop = self.total_pop
            else:
                if self.pop_scale is not None:
                    total_pop = self.pop_scale * self.n_agents
                else:
                    total_pop = self.n_agents

        self.total_pop = total_pop
        if self.pop_scale is None:
            self.pop_scale = total_pop / self.n_agents

        # Any other initialization
        sim.people = self.pop('people')
        if not sim.people.initialized:
            sim.people.initialize(sim)

        # Set time attributes
        sim.people.init_results(sim)
        return sim.people
    
    def convert_modules(self):
        """
        Common logic for converting plug-ins to a standard format; they are still
        a list at this point.  Used for networks, demographics, diseases, analyzers, 
        interventions, and connectors.
        """
        
        modmap = ss.module_map() # List of modules and parent module classes, e.g. ss.Disease
        modules = ss.find_modules() # Each individual module class option, e.g. ss.SIR
        
        for modkey,ssmoddict in modules.items():
            expected_cls = modmap[modkey]
            modlist = self[modkey]
            if isinstance(modlist, list): # Skip over ones that are already ndict format, assume they're already initialized
                for i,mod in enumerate(modlist):
                    
                    # Convert first from a string to a dict
                    if isinstance(mod, str):
                        mod = dict(type=mod)
                        
                    # Convert from class to class instance (used for interventions and analyzers only)
                    if isinstance(mod, type) and modkey in ['interventions', 'analyzers']:
                        mod = mod() # Call it to create a class instance
                    
                    # Now convert from a dict to a module
                    if isinstance(mod, dict):
                        
                        # Get the module type as a string
                        if 'type' in mod:
                            modtype = mod.pop('type')
                        else:
                            errormsg = f'When specifying a {modkey} module with a dict, one of the keys must be "type"; you supplied {mod}'
                            raise ValueError(errormsg)
                        
                        # Get the module type as a class
                        if isinstance(modtype, str): # Usual case, a string, e.g. dict(type='sir', dur_inf=6)
                            modtype = modtype.lower() # Because our map is in lowercase
                            moddictkeys = ssmoddict.keys()
                            moddictvals = ssmoddict.values()
                            if modtype in moddictkeys:
                                modcls = ssmoddict[modtype]
                            else:
                                errormsg = f'Invalid module name "{modtype}" for "{modkey}"; must be one of {moddictkeys}'
                                raise TypeError(errormsg)
                        else: # Allow supplying directly as a class, e.g. dict(type=ss.SIR, dur_inf=6)
                            if modtype in moddictvals:
                                modcls = modtype
                            else:
                                errormsg = f'Invalid module class "{modtype}" for "{modkey}"; must be one of {moddictvals}'
                                raise TypeError(errormsg)
                        
                        # Create the module and store it in the list
                        mod = modcls(**mod)
                    
                    # Special handling for interventions and analyzers: convert class and function to class instance
                    if modkey in ['interventions', 'analyzers']:
                        if isinstance(mod, type) and issubclass(mod, expected_cls):
                            mod = mod()  # Convert from a class to an instance of a class
                        elif not isinstance(mod, ss.Module) and callable(mod):
                            mod = expected_cls.from_func(mod)
                    
                    # Do final check
                    if not isinstance(mod, expected_cls):
                        errormsg = f'Was expecting {modkey} entry {i} to be class {expected_cls}, but was {type(mod)} instead'
                        raise TypeError(errormsg)
                    modlist[i] = mod
                
        return

    def init_demographics(self):
        """ Initialize demographics """

        # Allow users to add vital dynamics by entering birth_rate and death_rate parameters directly to the sim
        if self.birth_rate is not None:
            births = ss.Births(birth_rate=self.birth_rate)
            self.demographics += births
        if self.death_rate is not None:
            background_deaths = ss.Deaths(death_rate=self.death_rate)
            self.demographics += background_deaths
        return


def make_pars(**kwargs):
    return SimPars(**kwargs)


