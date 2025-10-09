"""
Set parameters
"""
from numbers import Number
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import datetime as dt

__all__ = ['Pars', 'SimPars', 'make_pars']

# Define classes to not descend into further -- based on sciris.sc_nested
atomic_classes = (str, Number, list, np.ndarray, pd.Series, pd.DataFrame, type(None), dt.date)


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
        if not len(pars): # Check that new parameters are supplied, and if not return unchanged
            return self
        if not create: # Check if there are any mismatches, which raises an exception if there are
            self.check_key_mismatch(pars)

        # Perform the update
        for key,new in pars.items():
            if key not in self.keys(): # It's a new parameter and create=True: update directly
                self[key] = new
            else:
                old = self[key] # Get the existing object we're about to update
                if isinstance(old, atomic_classes): # It's a number, string, etc: update directly
                    self[key] = new

                elif isinstance(old, Pars): # It's a Pars object: update recursively
                    old.update(new, create=create)
                elif isinstance(old, ss.ndict): # Update module containers
                    self._update_ndict(key, old, new)
                elif isinstance(old, ss.Module):  # Update modules
                    self._update_module(key, old, new)
                elif isinstance(old, ss.TimePar):
                    self._update_timepar(key, old, new)
                elif isinstance(old, ss.Dist): # Update a distribution
                    self._update_dist(key, old, new)
                elif callable(old): # It's a function: update directly
                    self[key] = new
                elif isinstance(old, dict):
                    self[key] = new # Take dictionaries directly, without warning the user
                else: # Everything else; not used currently but could be
                    warnmsg = f'No known mechanism for handling {type(old)} → {type(new)}; using default'
                    ss.warn(warnmsg)
                    self[key] = new
        return self

    def _update_ndict(self, key, old, new):
        """ Update an ndict object in the parameters, e.g. sim.pars.diseases """
        if not len(old): # It's empty, just overwrite
            self[key] = new
        else: # Don't overwrite an existing ndict
            if isinstance(new, dict):
                for newkey,newvals in new.items():
                    old[newkey].pars.update(newvals) # e.g. pars = {'diseases": {'sir': {'dur_inf': 6}}}
            else:
                errormsg = f'Cannot update an ndict with {type(new)}: must be a dict to set new parameters'
                raise TypeError(errormsg)
        return

    def _update_module(self, key, old, new):
        """ Update a Module object in the parameters, e.g. sim.pars.diseases.sir """
        if isinstance(new, dict):
            old[key].pars.update(new) # e.g. pars = {'dur_inf': 6}
        else:
            errormsg = f'Cannot update a module with {type(new)}: must be a dict to set new parameters'
            raise TypeError(errormsg)
        return

    def _update_timepar(self, key, old, new):
        """ Update a time parameter (duration or rate) """

        # It's a TimePar, e.g. dur_inf = ss.dur(6); use directly
        if isinstance(new, ss.TimePar):
            self[key] = new

        # It's a dataframe, allow the update -- used for demographics
        elif isinstance(new, (pd.Series, pd.DataFrame)):
            self[key] = new # TODO: add validation or convert to timepar

        # It's a single number, e.g. dur_inf = 6; set parameters
        elif isinstance(new, Number):
            old.set(new)

        # It's a list of numbers, e.g. dur_inf = [6, 2]; set parameters
        elif isinstance(new, list):
            old.set(*new)

        # It's a dict, figure out what to do
        elif isinstance(new, dict):
            if isinstance(old, ss.beta):
                self[key] = new # TODO: use an actual set here
            else:
                old.set(**new)

        # Give up
        else:
            errormsg = f'Updating timepar from {type(old)} to {type(new)} is not supported'
            raise TypeError(errormsg)

        return

    def _update_dist(self, key, old, new):
        """ Update a Dist object in the parameters, e.g. sim.pars.diseases.sir.dur_inf """

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

        # It's a list of numbers, e.g. dur_inf = [6, 2]; set parameters
        elif isinstance(new, list):
            old.set(*new)

        # It's a timepar, set it like a number, e.g. dur_inf = ss.dur(6)
        elif isinstance(new, ss.TimePar):
            oldpar = old.pars[0]
            if isinstance(oldpar, ss.TimePar): # If changing from one timepar to another, validate
                dur_mismatch = isinstance(oldpar, ss.dur) != isinstance(new, ss.dur)
                if dur_mismatch:
                    errormsg = f'Cannot change a duration to a non-duration vice versa: old={type(oldpar)}, new={type(new)}'
                    raise TypeError(errormsg)
            old.set(new)

        # It's a dict, figure out what to do
        elif isinstance(new, dict):
            newtype = new.get('type')
            if newtype is None: # Same type of dist, set parameters
                old.set(**new)
            else: # We need to create a new distribution
                if isinstance(old, ss.bernoulli) and newtype != 'bernoulli':
                    errormsg = f"Bernoulli distributions can't be changed to another type: {newtype} is invalid"
                    raise TypeError(errormsg)
                else:
                    dist = ss.make_dist(new)
                    self[key] = dist

        # It's a function, treat it like a number
        elif sc.isfunc(new):
            old.set(new)

        # Give up
        else:
            errormsg = f'Updating dist from {type(old)} to {type(new)} is not supported'
            raise TypeError(errormsg)

        return

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
        return sc.jsonify(dict(self), **kwargs)


class SimPars(Pars):
    """
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = ss.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        label (str): The name of the simulation
        n_agents (int/float): The number of agents to run (default 10,000)
        total_pop (int/float): If provided, scale the agents to this effective population size
        pop_scale (float): If provided, use this agent-to-population scale factor (total_pop = n_agents*pop_scale)
        unit (str): The time unit for the simulation (default 'year'; other choices are 'day', 'week', 'month')
        start (float/str/date): The starting date for the simulation (default 2000); can be a year or date
        stop (float/str/date): If provided, the ending date for the simulation (if not provided, calculate from "dur")
        dur (int): How many timesteps to simulate, if "stop" is not provided (default 50)
        dt (float): The timestep, in units of "unit" (default 1.0)
        rand_seed (int): The overall random seed for the simulation (used to set module-specific random seeds)
        birth_rate (float): If provided, include births with this rate (per 1000 people per year)
        death_rate (float): If provided, include deaths with this rate (per 1000 people per year)
        use_aging (bool): Specify whether agents age (by default, agents age if and only if births and/or deaths are included)
        people (People): If provided, use a pre-existing People object rather than creating one (in which case n_agents will be ignored)
        networks (str/list/Module): The network module(s); can be a string, single module (i.e. Network), or list
        demographics (str/list/Module): As above
        diseases (str/list/Module): As above
        connectors (str/list/Module): As above
        interventions (str/list/Module): As above
        analyzers (str/list/Module): As above
        verbose (float): How much detail to print (1 = every timestep, 0.1 = every 10 timesteps, etc.)
    """
    def __init__(self, **kwargs):

        # General parameters
        self.label   = '' # The label of the simulation
        self.verbose = ss.options.verbose # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

        # Population parameters
        self.n_agents  = 10e3 # Number of agents
        self.total_pop = None # If defined, used for calculating the scale factor
        self.pop_scale = None # How much to scale the population

        # Simulation parameters
        self.unit      = ''    # The time unit to use; options are 'year' (default), 'day', 'week', 'month', or 'none'
        self.start     = None  # Start of the simulation (default 2020)
        self.stop      = None  # End of the simulation
        self.dur       = None  # Duration of time to run, if stop isn't specified (default 50 steps of self.unit)
        self.dt        = 1.0   # Timestep (in units of self.unit)
        self.rand_seed = 1     # Random seed; if None, don't reset

        # Demographic parameters
        self.birth_rate = None
        self.death_rate = None
        self.use_aging  = None # True if demographics, false otherwise

        # Modules: demographics, diseases, networks, analyzers, and interventions
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

    def is_default(self, key):
        """ Check if the provided value matches the default """
        default_pars = SimPars() # Create a default SimPars object
        default_val = default_pars[key]
        current_val = self[key]
        match = (current_val == default_val) # Check if the value matches
        return match

    def validate(self):
        """ Call parameter validation methods """
        self.validate_sim_pars()
        self.validate_modules()
        return

    def validate_sim_pars(self):
        """ Validate each of the parameter values """
        self.validate_verbose()
        self.validate_agents()
        self.validate_total_pop()
        self.validate_time()
        return

    def validate_verbose(self):
        """ Validate verbosity """
        if self.verbose == 'brief':
            self.verbose = -1
        if not sc.isnumber(self.verbose):  # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self.par.verbose)} "{self.par.verbose}"'
            raise ValueError(errormsg)
        return

    def validate_agents(self):
        """ Check that n_agents is supplied and convert to an integer """
        if self.people is not None:
            len_people = len(self.people)
            if (len_people != self.n_agents) and not self.is_default('n_agents'):
                errormsg = f'Cannot set a custom value of n_agents ({self.n_agents}) that does not match a pre-created People object ({len_people}); use n_agents to create the People object instead'
                raise ValueError(errormsg)
            self.n_agents = len_people
        elif self.n_agents is not None:
            self.n_agents = int(self.n_agents)
        else:
            errormsg = 'Must supply n_agents, a people object, or a popdict'
            raise ValueError(errormsg)
        return

    def validate_total_pop(self):
        """ Ensure one but not both of total_pop and pop_scale are defined """
        if self.total_pop is not None:  # In future, if no pop_scale has been provided, try to get it from the location
            if self.pop_scale is not None:
                errormsg = f'You can define total_pop ({self.total_pop}) or pop_scale ({self.pop_scale}), but not both, since one is calculated from the other'
                raise ValueError(errormsg)
            total_pop = self.total_pop
        else:
            if self.pop_scale is not None:
                total_pop = self.pop_scale * self.n_agents
            else:
                total_pop = self.n_agents
        self.total_pop = total_pop
        if self.pop_scale is None:
            self.pop_scale = total_pop / self.n_agents
        return

    def validate_time(self):
        """ Ensure at least one of dur and stop is defined, but not both """

        # Handle the unit
        if self.unit == '':
            self.unit = ss.time.default_unit
        self.unit = ss.time.validate_unit(self.unit)

        # Handle start
        if self.start is None:
            self.start = ss.time.default_start[self.unit]

        # Handle stop and dur
        if self.stop is not None:
            if self.dur is None:
                self.dur = ss.date_diff(self.start, self.stop, self.unit)
            else:
                errormsg = f'You can supply either stop ({self.stop}) or dur ({self.dur}) but not both, since one is calculated from the other'
                raise ValueError(errormsg)
            if self.dur <= 0:
                errormsg = f"Duration must be >0, but you supplied start={str(self.start)} and stop={str(self.stop)}, which gives dur={self.dur}"
                raise ValueError(errormsg)
        else:
            if self.dur is None:
                self.dur = ss.time.default_dur
            self.stop = ss.date_add(self.start, self.dur, self.unit)
        return

    def validate_modules(self):
        """ Validate modules passed in pars"""

        # Do special validation on demographics (must be before modules are created)
        self.validate_demographics()

        # Get all modules into a consistent list format
        modmap = ss.module_map()
        for modkey in modmap.keys():
            if not isinstance(self[modkey], ss.ndict):
                self[modkey] = sc.tolist(self[modkey])

        # Convert any modules that are not already Module objects
        self.convert_modules()

        # Convert from lists to ndicts
        for modkey,modclass in modmap.items():
            self[modkey] = ss.ndict(self[modkey], type=modclass)

        # Do special validation on networks (must be after modules are created)
        self.validate_networks()
        return

    def validate_demographics(self):
        """ Validate demographics-related input parameters"""
        # Allow shortcut for default demographics
        if self.demographics == True:
            self.demographics = sc.autolist()
            if self.birth_rate is None:
                self.demographics += ss.Births()
            if self.death_rate is None:
                self.demographics += ss.Deaths()

        # Allow users to add vital dynamics by entering birth_rate and death_rate parameters directly to the sim
        valid = isinstance(self.demographics, ss.ndict) and not len(self.demographics)
        if self.birth_rate is not None:
            if not valid:
                errormsg = 'You can only specify birth_rate together with (optionally) death_rate, not other demographics modules; add ss.Births() manually'
                raise ValueError(errormsg)
            births = ss.Births(birth_rate=self.birth_rate)
            self.demographics += births
        if self.death_rate is not None:
            if not valid:
                errormsg = 'You can only specify death_rate together with (optionally) birth_rate, not other demographics modules; add ss.Deaths() manually'
                raise ValueError(errormsg)
            background_deaths = ss.Deaths(death_rate=self.death_rate)
            self.demographics += background_deaths

        # Decide whether to use aging based on if demographics modules are present
        if self.use_aging is None:
            self.use_aging = True if self.demographics else False
        return

    def validate_networks(self):
        """ Validate networks """
        # Don't allow more than one prenatal or postnatal network
        prenatal_nets = []
        postnatal_nets = []
        for k,nw in self.networks.items():
            if isinstance(nw, ss.Network):
                if nw.prenatal: prenatal_nets.append(k)
                if nw.postnatal: postnatal_nets.append(k)
        if len(prenatal_nets) > 1:
            errormsg = f'Starsim currently only supports one prenatal network; prenatal networks are: {prenatal_nets.keys()}'
            raise ValueError(errormsg)
        if len(postnatal_nets) > 1:
            errormsg = f'Starsim currently only supports one postnatal network; postnatal networks are: {postnatal_nets.keys()}'
            raise ValueError(errormsg)
        if len(postnatal_nets) and not len(prenatal_nets):
            errormsg = 'Starsim currently only supports adding a postnatal network if a prenatal network is present'
            raise ValueError(errormsg)
        return

    def convert_modules(self):
        """
        Convert different types of representations for modules into a
        standardized object representation that can be parsed and used by
        a Sim object.
        Used for starsim classes:
        - networks,
        - demographics,
        - diseases,
        - analyzers,
        - interventions, and
        - connectors.
        """
        modmap = ss.module_map() # List of modules and parent module classes, e.g. ss.Disease
        modules = ss.find_modules() # Each individual module class option, e.g. ss.SIR

        for modkey,ssmoddict in modules.items():
            moddictkeys = ssmoddict.keys()
            moddictvals = ssmoddict.values()
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
                    if not isinstance(mod, (expected_cls, ss.Module)): # TEMP: check if this check still works?
                        errormsg = f'Was expecting {modkey} entry {i} to be class {expected_cls} or Plugin, but was {type(mod)} instead'
                        raise TypeError(errormsg)
                    modlist[i] = mod
        return


def make_pars(**kwargs):
    """ Shortcut for making a new instance of SimPars """
    return SimPars(**kwargs)
