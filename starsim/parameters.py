"""
Set parameters
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Parameters', 'make_pars']


class Parameters(sc.objdict):
    """
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = ss.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        kwargs (dict): any additional kwargs are interpreted as parameter names
    """

    def __init__(self, **kwargs):
        
        # Overall parameters
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
        self.seed       = 1      # Random seed, if None, don't reset
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

    def update_pars(self, pars=None, create=False, **kwargs):
        """
        Update internal dict with new pars.
        
        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
            kwargs (dict): merged with pars
        """
        if pars or kwargs:
            pars = sc.mergedicts(pars, kwargs)
            if not create:
                available_keys = list(self.keys())
                mismatches = [key for key in pars.keys() if key not in available_keys]
                if len(mismatches):
                    errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
                    raise sc.KeyNotFoundError(errormsg)
            self.update(pars)
        return
    
    def initialize(self, sim, reset=False, **kwargs):
        
        # Validation
        self.validate_dt()
        self.validate_pars()
        
        # Time indexing
        sim.dt = self.dt
        sim.yearvec = np.arange(start=self.start, stop=self.end, step=self.dt)
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
                        modlist[i] = dict(type=mod)
                        
                    # Convert from class to class instance (used for interventions and analyzers only)
                    if isinstance(mod, type) and modkey in ['interventions', 'analyzers']:
                        modlist[i] = mod() # Call it to create a class instance
                    
                    # Now convert from a dict to a module
                    if isinstance(mod, dict):
                        
                        # Get the module type as a string
                        try:
                            modtype = mod.pop('type')
                        except KeyError as E:
                            errormsg = f'When defining a module as a dict, you must supply the type ({mod})'
                            raise ValueError(errormsg) from E
                        
                        # Get the module type as a class
                        try:
                            if isinstance(modtype, str): # Usual case, a string, e.g. dict(type='sir', dur_inf=6)
                                modtype = modtype.lower() # Because our map is in lowercase
                                modcls = ssmoddict[modtype]
                            else: # Allow supplying directly as a class, e.g. dict(type=ss.SIR, dur_inf=6)
                                assert modcls in ssmoddict.values(), 'Not a valid module'
                                modcls = modtype
                        except Exception as E:
                            errormsg = f'Invalid module type "{modtype}" for "{modkey}"; must be one of {sc.strjoin(ss.moddict.keys())}'
                            raise ValueError(errormsg) from E
                        
                        # Create the module and store it in the list
                        try:
                            newmod = modcls(**mod)
                            modlist[i] = newmod # Replace
                        except Exception as E:
                            errormsg = f'Failed to create module {modtype} with arguments {mod}; see above for full error'
                            raise ValueError(errormsg) from E
                    
                    # Special handling for interventions and analyzers: convert class and function to class instance
                    if modkey in ['interventions', 'analyzers']:
                        if isinstance(mod, type) and issubclass(mod, expected_cls):
                            modlist[i] = mod()  # Convert from a class to an instance of a class
                        elif isinstance(mod, ss.Module) and callable(mod):
                            modlist[i] = expected_cls.from_func(mod)
                    
                    # Do final check
                    thismod = modlist[i]
                    if not isinstance(thismod, expected_cls):
                        errormsg = f'Was expecting {modkey} entry {i} to be class {expected_cls}, but was {type(thismod)} instead'
                        raise TypeError(errormsg)
                
        return

    # def convert_plugins(self, plugin_class, plugin_name=None):
    #     """
    #     Common logic for converting plug-ins to a standard format
    #     Used for networks, demographics, diseases, connectors, analyzers, and interventions
        
    #     Args:
    #         plugin: class
    #     """

    #     if plugin_name is None: plugin_name = plugin_class.__name__.lower()

    #     # Get lower-case names of all subclasses
    #     known_plugins = {n.__name__.lower():n for n in ss.all_subclasses(plugin_class)}
    #     if plugin_name == 'networks': # Allow "msm" or "msmnet"
    #         known_plugins.update({k.removesuffix('net'):v for k,v in known_plugins.items()})

    #     # Figure out if it's in the sim pars or provided directly
    #     attr_plugins = getattr(self, plugin_name)  # Get any plugins that have been provided directly

    #     # See if they've been provided in the pars dict
    #     if self.get(plugin_name):

    #         par_plug = self[plugin_name]

    #         # String: convert to ndict
    #         if isinstance(par_plug, str):
    #             plugins = ss.ndict(dict(name=par_plug))

    #         # List or dict: convert to ndict
    #         elif sc.isiterable(par_plug) and len(par_plug):
    #             if isinstance(par_plug, dict) and 'type' in par_plug and 'name' not in par_plug:
    #                 par_plug['name'] = par_plug['type'] # TODO: simplify/remove this
    #             plugins = ss.ndict(par_plug)

    #     else:  # Not provided directly or in pars
    #         plugins = {}

    #     # Check that we don't have two copies
    #     for attr_key in attr_plugins.keys():
    #         if plugins.get(attr_key):
    #             errormsg = f'Sim was created with {attr_key} module, cannot create another through the pars dict.'
    #             raise ValueError(errormsg)

    #     plugins = sc.mergedicts(plugins, attr_plugins)

    #     # Process
    #     processed_plugins = sc.autolist()
    #     for plugin in plugins.values():

    #         if not isinstance(plugin, plugin_class):

    #             if isinstance(plugin, dict):
    #                 ptype = (plugin.get('type') or plugin.get('name') or '').lower()
    #                 name = plugin.get('name') or ptype
    #                 if ptype in known_plugins:
    #                     # Make an instance of the requested plugin
    #                     plugin_pars = {k: v for k, v in plugin.items() if k not in ['type', 'name']}
    #                     pclass = known_plugins[ptype]
    #                     plugin = pclass(name=name, pars=plugin_pars) # TODO: does this handle par_dists, etc?
    #                 else:
    #                     errormsg = (f'Could not convert {plugin} to an instance of class {plugin_name}.'
    #                                 f'Try specifying it directly rather than as a dictionary.')
    #                     raise ValueError(errormsg)
    #             elif plugin_name in ['analyzers', 'interventions'] and callable(plugin):
    #                 pass # This is ok, it's a function instead of an Intervention object
    #             else:
    #                 errormsg = (
    #                     f'{plugin_name.capitalize()} must be provided as either class instances or dictionaries with a '
    #                     f'"name" key corresponding to one of these known subclasses: {known_plugins}.')
    #                 raise ValueError(errormsg)

    #         processed_plugins += plugin

    #     return processed_plugins

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

    # def init_interventions(self):
    #     """ Initialize and validate the interventions """

    #     # interventions = self.convert_plugins(ss.Intervention, plugin_name='interventions')

    #     # Translate the intervention specs into actual interventions
    #     for i, intervention in enumerate(self.interventions):
    #         if isinstance(intervention, type) and issubclass(intervention, ss.Intervention):
    #             intervention = intervention()  # Convert from a class to an instance of a class
    #         elif not isinstance(intervention, ss.Intervention) and callable(intervention):
    #             intv_func = intervention
    #             intervention = ss.Intervention(name=f'intervention_func_{i}')
    #             intervention.apply = intv_func # Monkey-patch together an intervention from a function
    #         else:
    #             errormsg = f'Intervention {intervention} does not seem to be a valid intervention: must be a function or Intervention subclass'
    #             raise TypeError(errormsg)
            
    #         if intervention.name not in self.interventions:
    #             self.interventions += intervention

    #         # Add intervention states to the People's dicts
    #         self.people.add_module(intervention)

    #     # TODO: combine this with the code above
    #     for k,intervention in self.interventions.items():
    #         if not isinstance(intervention, ss.Intervention):
    #             intv_func = intervention
    #             intervention = ss.Intervention(name=f'intervention_func_{k}')
    #             intervention.apply = intv_func # Monkey-patch together an intervention from a function
    #             self.interventions[k] = intervention
        
    #     self.interventions = ss.ndict(self.interventions, type=ss.Intervention)
    #     return

    # def init_analyzers(self):
    #     """ Initialize the analyzers """
        
    #     analyzers = self.analyzers
    #     if not np.iterable(analyzers):
    #         analyzers = sc.tolist(analyzers)

    #     # Interpret analyzers
    #     for ai, analyzer in enumerate(analyzers):
    #         if isinstance(analyzer, type) and issubclass(analyzer, ss.Analyzer):
    #             analyzer = analyzer()  # Convert from a class to an instance of a class
    #         if not (isinstance(analyzer, ss.Analyzer) or callable(analyzer)):
    #             errormsg = f'Analyzer {analyzer} does not seem to be a valid analyzer: must be a function or Analyzer subclass'
    #             raise TypeError(errormsg)
    #         self.analyzers += analyzer  # Add it in

    #     # TODO: should tidy/remove this code
    #     for k,analyzer in self.analyzers.items():
    #         if not isinstance(analyzer, ss.Analyzer) and callable(analyzer):
    #             ana_func = analyzer
    #             analyzer = ss.Analyzer(name=f'analyzer_func_{k}')
    #             analyzer.apply = ana_func # Monkey-patch together an intervention from a function
    #             self.analyzers[k] = analyzer
        
    #     self.analyzers = ss.ndict(self.analyzers, type=ss.Analyzer)

    #     return


def make_pars(**kwargs):
    return Parameters(**kwargs)


