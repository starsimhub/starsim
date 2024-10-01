"""
General module class -- base class for diseases, interventions, etc. Also
defines Analyzers and Connectors.
"""
import sciris as sc
import starsim as ss
from functools import partial

__all__ = ['module_map', 'find_modules', 'Module', 'Analyzer', 'Connector']


def module_map(key=None):
    """ Define the mapping between module names and types """
    module_map = sc.objdict(
        networks      = ss.Network,
        demographics  = ss.Demographics,
        diseases      = ss.Disease,
        interventions = ss.Intervention,
        analyzers     = ss.Analyzer,
        connectors    = ss.Connector,
    )
    return module_map if key is None else module_map[key]


def find_modules(key=None, flat=False):
    """ Find all subclasses of Module present in Starsim, divided by type """
    modules = sc.objdict()
    modmap = module_map()
    attrs = dir(ss) # Find all attributes in Starsim (note: does not parse user code)
    for modkey, modtype in modmap.items(): # Loop over each module type
        modules[modkey] = sc.objdict()
        for attr in attrs: # Loop over each attribute (inefficient, but doesn't need to be optimized)
            item = getattr(ss, attr)
            try:
                assert issubclass(item, modtype) # Check that it's a class, and instance of this module
                low_attr = attr.lower()
                modules[modkey][low_attr] = item # It passes, so assign it to the dict
                if modkey == 'networks' and low_attr.endswith('net'): # Also allow networks without 'net' suffix
                    modules[modkey][low_attr.removesuffix('net')] = item
            except:
                pass
    if flat:
        modules = sc.objdict({k:v for vv in modules.values() for k,v in vv.items()}) # Unpack the nested dict into a flat one
    return modules if key is None else modules[key]


class Module(sc.quickobj):
    """
    The main base class for all Starsim modules: diseases, networks, interventions, etc.

    Args:
        name (str): a short, key-like name for the module (e.g. "randomnet")
        label (str): the full, human-readable name for the module (e.g. "Random network")
        unit (str): the time unit (e.g. 'day', 'year'); inherits from sim if not supplied
        dt (float): the timestep (e.g. 1.0, 0.1); inherits from sim if not supplied
    """

    def __init__(self, name=None, label=None, unit=None, dt=None):
        # Handle parameters
        self.pars = ss.Pars() # Usually populated via self.define_pars()
        self.set_metadata(name, label) # Usually reset as part of self.update_pars()
        self.set_time_pars(unit, dt)

        # Properties to be added by init_pre()
        self.sim = None
        self.dists = None # Turned into a Dists object by sim.init_dists() if this module has dists
        self.results = ss.Results(self.name)

        # Time properties, added by init_time_pars()
        self.unit = None
        self.dt = None
        self.timevec = None
        self.npts = None
        self.ti = None

        # Finish initialization
        self.pre_initialized = False
        self.initialized = False
        self.finalized = False
        return

    def __bool__(self):
        """ Ensure that zero-length modules (e.g. networks) are still truthy """
        return True

    def __call__(self, *args, **kwargs):
        """ Allow modules to be called like functions """
        return self.step(*args, **kwargs)

    def disp(self, output=False):
        """ Display the full object """
        out = sc.prepr(self)
        if not output:
            print(out)
        else:
            return out

    def set_metadata(self, name=None, label=None):
        """ Set metadata for the module """
        # Validation
        for key,val in dict(name=name, label=label).items():
            if val is not None:
                if not isinstance(val, str):
                    errormsg = f'Invalid value for {key}: must be str, not {type(val)}: {val}'
                    raise TypeError(errormsg)

        # Set values
        self.name  = sc.ifelse(name,  getattr(self, 'name',  self.pars.get('name', self.__class__.__name__.lower()))) # Default name is the class name
        self.label = sc.ifelse(label, getattr(self, 'label', self.pars.get('label', self.name)))
        return

    def set_time_pars(self, unit=None, dt=None):
        """ Set time units for the module """
        self.unit  = sc.ifelse(unit,  getattr(self, 'unit', self.pars.get('unit')))
        self.dt    = sc.ifelse(dt,    getattr(self, 'dt',   self.pars.get('dt')))
        return

    def define_pars(self, inherit=True, **kwargs): # TODO: think if inherit should default to true or false
        """ Create or merge Pars objects """
        if inherit: # Merge with existing
            self.pars.update(**kwargs, create=True)
        else: # Or overwrite
            self.pars = ss.Pars(**kwargs)
        return self.pars

    def update_pars(self, pars, **kwargs):
        """ Pull out recognized parameters, returning the rest """
        pars = sc.mergedicts(pars, kwargs)

        # Update matching module parameters
        matches = {}
        for key in list(pars.keys()): # Need to cast to list to avoid "dict changed during iteration"
            if key in self.pars:
                matches[key] = pars.pop(key)
        self.pars.update(matches)

        # Update module attributes
        metadata = {key:pars.pop(key, None) for key in ['name', 'label']}
        timepars = {key:pars.pop(key, None) for key in ['unit', 'dt']}
        self.set_metadata(**metadata)
        self.set_time_pars(**timepars)

        # Should be no remaining pars
        if len(pars):
            errormsg = f'{len(pars)} unrecognized arguments for {self.name}: {sc.strjoin(pars.keys())}'
            raise ValueError(errormsg)
        return

    def init_pre(self, sim, force=False):
        """
        Perform initialization steps

        This method is called once, as part of initializing a Sim. Note: after
        initialization, initialized=False until init_vals() is called (which is after
        distributions are initialized).
        """
        if force or not self.pre_initialized:
            self.sim = sim # Link back to the sim object
            ss.link_dists(self, sim, skip=ss.Sim) # Link the distributions to sim and module
            sim.pars[self.name] = self.pars
            sim.results[self.name] = self.results
            sim.people.add_module(self) # Connect the states to the people
            self.init_time_pars() # Initialize the modules' time parameters and link them to the sim
            self.init_results()
            self.pre_initialized = True
        return

    def init_results(self):
        """ Initialize any results required; part of init_pre() """
        self.results.timevec = self.timevec # Store the timevec in the results for plotting
        return

    def init_post(self):
        """ Initialize the values of the states; the last step of initialization """
        for state in self.states:
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def init_time_pars(self, force=False):
        """ Initialize all time parameters by ensuring all parameters are initialized; part of init_post() """
        pars = self.sim.pars

        # Find all modules and set the timestep
        if force or self.unit is None:
            self.unit = pars.unit
        if force or self.dt is None:
            self.dt = pars.dt

        # Find all time parameters in the module
        timepars = sc.search(self.pars, type=ss.TimePar) # Should it be self or self.pars?

        # Initialize them with the parent module
        for timepar in timepars.values():
            if force or not timepar.initialized:
                timepar.init(parent=self, die=False) # In some cases, the values can't be initialized; that's OK here

        # Create the module-specific time vector
        self.timevec = ss.make_timevec(pars.start, pars.stop, self.dt, self.unit)
        self.npts = len(self.timevec)
        self.ti = 0 # Track the current timestep, which may or may not match the sim's
        return

    @property
    def now(self):
        """ Return the current time, i.e. the time vector at the current timestep """
        try:
            return self.timevec[self.ti]
        except Exception as E:
            ss.warn(f'Encountered exception when getting the current time in {self.name}: {E}')
            return None

    def start_step(self):
        """ Tasks to perform at the beginning of the step """
        if self.dists is not None: # Will be None if no distributions are defined
            self.dists.jump_dt() # Advance random number generators forward for calls on this step
        return

    def step(self):
        """ Define how the module updates over time -- the key part of Starsim!! """
        pass

    def finish_step(self):
        """ Define what should happen at the end of the step; at minimum, increment ti """
        self.ti += 1
        return

    def update_results(self):
        """ Perform any results updates on each timestep """
        pass

    def finalize(self):
        """ Perform any final operations, such as removing unneeded data """
        self.finalize_results()
        self.finalized = True
        return

    def finalize_results(self): # TODO: this is confusing, needs to be not redefined by the user, or called *after* a custom finalize_results()
        """ Finalize results """
        # Scale results
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result) and res.scale:
                self.results[reskey] = self.results[reskey]*self.sim.pars.pop_scale
        return

    def define_states(self, *args, check=True):
        """
        Define states of the module with the same attribute name as the state

        Args:
            args (states): list of states to add
            check (bool): whether to check that the object being added is a state
        """
        for arg in args:
            if isinstance(arg, (list, tuple)):
                state = ss.State(*arg)
            elif isinstance(arg, dict):
                state = ss.State(**arg)
            else:
                state = arg

            if check:
                assert isinstance(state, ss.Arr), f'Could not add {state}: not an Arr object'

            # Add the state to the module
            setattr(self, state.name, state)
        return

    def define_results(self, *args, check=True):
        """ Add results to the module """
        for arg in args:
            if isinstance(arg, (list, tuple)):
                result = ss.Result(*arg)
            elif isinstance(arg, dict):
                result = ss.Result(**arg)
            else:
                result = arg

            # Update with module information
            result.update(module=self.name, shape=self.npts, timevec=self.timevec)

            # Add the result to the dict of results; does automatic checking
            self.results += result
        return

    @property
    def states(self):
        """
        Return a flat list of all states

        The base class returns all states that are contained in top-level attributes
        of the Module. If a Module stores states in a non-standard location (e.g.,
        within a list of states, or otherwise in some other nested structure - perhaps
        due to supporting features like multiple genotypes) then the Module should
        overload this attribute to ensure that all states appear in here.
        """
        return [x for x in self.__dict__.values() if isinstance(x, ss.Arr)] # TODO: use ndict

    @property
    def statesdict(self): # TODO: remove
        """
        Return a flat dictionary (objdict) of all states

        Note that name collisions may affect the output of this function
        """
        return sc.objdict({s.name:s for s in self.states})

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create a module instance by name

        Args:
            name (str): A string with the name of the module class in lower case, e.g. 'sir'
        """
        for subcls in ss.all_subclasses(cls):
            if subcls.__name__.lower() == name:
                return subcls(*args, **kwargs)
        else:
            raise KeyError(f'Module "{name}" did not match any known Starsim modules')

    @classmethod
    def from_func(cls, func):
        """ Create an module from a function """
        def step(mod): # TODO: see if this can be done more simply
            return mod.func(mod.sim)
        name = func.__name__
        mod = cls(name=name)
        mod.func = func
        mod.step = partial(step, mod)
        mod.step.__name__ = name # Manually add these in as for a regular class method
        mod.step.__self__ = mod
        return mod

    def to_json(self):
        """ Export to a JSON-compatible format """
        out = sc.objdict()
        out.type = self.__class__.__name__
        out.name = self.name
        out.label = self.label
        out.pars = self.pars.to_json()
        return out

    def plot(self):
        """ Plot all results in the module """
        with sc.options.with_style('fancy'):
            flat = sc.flattendict(self.results, sep=': ')
            timevec = self.timevec
            fig, axs = sc.getrowscols(len(flat), make=True)
            for ax, (k, v) in zip(axs.flatten(), flat.items()):
                ax.plot(timevec, v)
                ax.set_title(k)
                ax.set_xlabel('Year')
        return fig


class Analyzer(Module):
    """
    Base class for Analyzers. Analyzers are used to provide more detailed information
    about a simulation than is available by default -- for example, pulling states
    out of sim.people on a particular timestep before they get updated on the next step.

    The key method of the analyzer is ``step()``, which is called with the sim
    on each timestep.

    To retrieve a particular analyzer from a sim, use sim.get_analyzer().
    """
    pass


class Connector(Module):
    """
    Base class for Connectors, which mediate interactions between disease (or other) modules

    Because connectors can do anything, they have no specified structure: it is
    up to the user to define how they behave.
    """
    pass
