"""
General module class -- base class for diseases, interventions, etc. Also
defines Analyzers and Connectors.
"""
import inspect
import functools as ft
import sciris as sc
import starsim as ss

__all__ = ['module_map', 'find_modules', 'required', 'Base', 'Module']

module_args = ['name', 'label'] # Define allowable module arguments


def module_map(key=None, include_modules=True):
    """ Define the mapping between module names and types; not for the user """
    module_map = sc.objdict(
        modules       = None, # Handled separately as a fallback
        demographics  = ss.Demographics,
        connectors    = ss.Connector,
        networks      = ss.Network,
        interventions = ss.Intervention,
        diseases      = ss.Disease,
        analyzers     = ss.Analyzer,
    )
    if not include_modules:
        module_map.pop('modules')
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
            low_attr = attr.lower()
            try:
                assert issubclass(item, modtype) # Check that it's a class, and instance of this module
                modules[modkey][low_attr] = item # It passes, so assign it to the dict
                if modkey == 'networks' and low_attr.endswith('net'): # Also allow networks without 'net' suffix
                    modules[modkey][low_attr.removesuffix('net')] = item
            except:
                if isinstance(item, type) and issubclass(item, ss.Module): # For any other modules, add them to the "modules" list
                    modules['modules'][low_attr] = item
    if flat:
        modules = sc.objdict({k:v for vv in modules.values() for k,v in vv.items()}) # Unpack the nested dict into a flat one
    return modules if key is None else modules[key]


def required(val=True):
    """
    Decorator to mark module methods as required.

    A common gotcha in Starsim is to forget to call super(), or to mistype a method
    name so it's never called. This decorator lets you mark methods (of Modules only)
    to be sure that they are called either on sim initialization or on sim run.

    Args:
        val (True/'disable'): by default, mark method as required; if set to 'disable', then disable method checking for parent classes as well (i.e. remove previous "required" calls)

    **Example**:

        class CustomSIS(ss.SIS):

            def step(self):
                super().step()
                self.custom_step() # Will raise an exception if this line is not here
                return

            @ss.required() # Mark this method as required on run
            def custom_step(self):
                pass
    """
    # Wrap the function
    def decorator(func):
        func._call_required = val  # Mark for collection

        @ft.wraps(func)
        def wrapper(self, *args, **kwargs):
            key = func.__qualname__
            self._call_required[key] += 1
            return func(self, *args, **kwargs)

        return wrapper
    return decorator


class Base(sc.quickobj):
    """
    The parent class for Sim and Module objects
    """
    def __bool__(self):
        """ Ensure that zero-length modules (e.g. networks) are still truthy """
        return True

    def __len__(self):
        """ The length of a module is the number of timepoints; see also len(sim) """
        try:    return self.t.npts
        except: return 0

    def disp(self, output=False, **kwargs):
        """ Display the full object """
        out = sc.prepr(self, **kwargs)
        if not output:
            print(out)
        else:
            return out

    @property
    def ti(self):
        """ Get the current module timestep """
        try:    return self.t.ti
        except: return None

    @property
    def now(self):
        """ Shortcut to self.t.now() """
        try:    return self.t.now()
        except: return None

    @property
    def timevec(self):
        """ Shortcut to self.t.timevec """
        try:    return self.t.timevec
        except: return None

    def copy(self, die=True):
        """
        Perform a deep copy of the module/sim

        Args:
            die (bool): whether to raise an exception if copy fails (else, try a shallow copy)
        """
        out = sc.dcp(self, die=die)
        return out


class Module(Base):
    """
    The main base class for all Starsim modules: diseases, networks, interventions, etc.

    Args:
        name (str): a short, key-like name for the module (e.g. "randomnet")
        label (str): the full, human-readable name for the module (e.g. "Random network")
        kwargs (dict): passed to ss.Time() (e.g. start, stop, unit, dt)
    """
    _immutable_attrs = ['pars', 't', 'sim', 'dists']

    def __init__(self, name=None, label=None, **kwargs):
        # Housekeeping
        self._collect_required() # First, collect methods marked as required on creation

        # Handle parameters
        self.pars = ss.Pars() # Usually populated via self.define_pars()
        self.set_metadata(name, label) # Usually reset as part of self.update_pars()
        self.t = ss.Time(**kwargs, name=self.name)

        # Properties to be added by init_pre()
        self.sim = None
        self.dists = None # Turned into a Dists object by sim.init_dists() if this module has dists
        self.results = ss.Results(self.name)

        # Finish initialization
        self.pre_initialized = False
        self.initialized = False
        self.finalized = False
        self._lock_attrs = True
        return

    def __call__(self, *args, **kwargs):
        """ Allow modules to be called like functions """
        return self.step(*args, **kwargs)

    def __getitem__(self, key):
        """ Allow modules to act like dictionaries """
        return getattr(self, key)

    def __setattr__(self, name, value):
        """ Don't allow locked attributes to be overwritten """
        if getattr(self, '_lock_attrs', False) and name in self._immutable_attrs:
            errormsg = f'Cannot modify attribute "{name}"; reserved attributes are {sc.strjoin(self._immutable_attrs)}.\n'
            errormsg += 'If you really mean to do this, use module.setattribute()'
            raise AttributeError(errormsg)
        else:
            super().__setattr__(name, value)
        return

    def setattribute(self, name, value):
        """ Method for setting an attribute that does not perform checking against immutable attributes """
        return super().__setattr__(name, value)

    def _reconcile(self, key, value=None, default=None):
        """ Reconcile module attributes, parameters, and input arguments """
        parval = self.pars.get(key)
        attrval = getattr(self, key, parval)
        val = sc.ifelse(value, attrval, default)
        return val

    def _collect_required(self):
        """ Collect all methods marked as required """
        reqs = {}
        disabled = []
        for cls in inspect.getmro(type(self)):
            for attr,method in cls.__dict__.items():
                req = getattr(method, '_call_required', False)
                if req:
                    valid = [True, 'disable']
                    if req not in valid:
                        errormsg = f'ss.require() must be True or "disable", not "{req}"'
                        raise ValueError(errormsg)
                    elif req == 'disable':
                        disabled.append(attr)
                    key = f'{cls.__qualname__}.{attr}'
                    reqs[key] = 0 # Meaning it's been called 0 times

        # Collate and manually increment any that are set to be disabled
        self._call_required = reqs
        if disabled:
            for k in self._call_required.keys():
                if k.split('.')[-1] in disabled: # Look for matches for disabled, omitting the class name
                    self._call_required[k] += 1 # Manually increment the call to pass checking
        return required

    def check_method_calls(self):
        """
        Check if any required methods were not called.

        Typically called automatically by `sim.run()`.
        """
        missing = [key for key, called in self._call_required.items() if not called]
        return missing

    @required()
    def set_metadata(self, name=None, label=None):
        """ Set metadata for the module """
        # Validation
        for key,val in dict(name=name, label=label).items():
            if val is not None:
                if not isinstance(val, str):
                    errormsg = f'Invalid value for {key}: must be str, not {type(val)}: {val}'
                    raise TypeError(errormsg)

        # Set values
        self.name = self._reconcile('name', name, self.__class__.__name__.lower())
        self.label = self._reconcile('label', label, self.name)
        return

    def define_pars(self, inherit=True, **kwargs): # TODO: think if inherit should default to true or false
        """ Create or merge Pars objects """
        if inherit: # Merge with existing
            self.pars.update(**kwargs, create=True)
        else: # Or overwrite
            self.pars = ss.Pars(**kwargs)
        return self.pars

    def update_pars(self, **pars):
        """ Pull out recognized parameters, returning the rest """

        # Update matching module parameters
        matches = {}
        for key in list(pars.keys()): # Need to cast to list to avoid "dict changed during iteration"
            if key in self.pars:
                matches[key] = pars.pop(key)
        self.pars.update(matches)

        # Update module attributes
        metadata = {key:pars.get(key, self.pars.get(key)) for key in module_args}
        timepars = {key:pars.get(key, self.pars.get(key)) for key in ss.Time.time_args}
        self.set_metadata(**metadata)
        self.t.update(**timepars)

        # Should be no remaining pars
        remaining = set(pars.keys()) - set(module_args) - set(ss.Time.time_args)
        if len(remaining):
            errormsg = f'{len(pars)} unrecognized arguments for {self.name}: {sc.strjoin(remaining)}'
            raise ValueError(errormsg)
        return

    @required()
    def init_pre(self, sim, force=False):
        """
        Perform initialization steps

        This method is called once, as part of initializing a Sim. Note: after
        initialization, initialized=False until init_vals() is called (which is after
        distributions are initialized).
        """
        if force or not self.pre_initialized:
            self.setattribute('sim', sim) # Link back to the sim object
            ss.link_dists(self, sim, skip=ss.Sim) # Link the distributions to sim and module
            self.t.init(sim=self.sim) # Initialize time vector
            sim.pars[self.name] = self.pars
            sim.results[self.name] = self.results
            sim.people.add_module(self) # Connect the states to the people
            self.init_results()
            self.pre_initialized = True
        return

    @required()
    def init_results(self):
        """ Initialize any results required; part of init_pre() """
        self.results.timevec = self.t.timevec # Store the timevec in the results for plotting
        return

    @required()
    def init_post(self):
        """ Initialize the values of the states; the last step of initialization """
        for state in self.states:
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def match_time_inds(self, inds=None):
         """ Find the nearest matching sim time indices for the current module """
         self_tvec = self.t.yearvec
         sim_tvec = self.sim.t.yearvec
         if len(self_tvec) == len(sim_tvec): # Shortcut to avoid doing matching
             return Ellipsis if inds is None else inds
         else:
             out = sc.findnearest(sim_tvec, self_tvec)
             return out

    @required()
    def start_step(self):
        """ Tasks to perform at the beginning of the step """
        if self.dists is not None: # Will be None if no distributions are defined
            self.dists.jump_dt() # Advance random number generators forward for calls on this step
        return

    def step(self):
        """ Define how the module updates over time -- the key part of Starsim!! """
        errormsg = f'Module "{self.name}" does not define a "step" method: use "def step(self): pass" if this is intentional'
        raise NotImplementedError(errormsg)

    @required()
    def finish_step(self):
        """ Define what should happen at the end of the step; at minimum, increment ti """
        self.t.ti += 1
        return

    def update_results(self):
        """ Perform any results updates on each timestep """
        pass

    @required()
    def finalize(self):
        """ Perform any final operations, such as removing unneeded data """
        self.finalize_results()
        self.finalized = True
        return

    @required()
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
            result.update(module=self.name, shape=self.t.npts, timevec=self.t.timevec)

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
        mod.step = ft.partial(step, mod)
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

    def shrink(self):
        """ Shrink the size of the module for saving to disk """
        shrunk = ss.utils.shrink()
        self.sim = shrunk
        self.dists = shrunk
        for state in self.states:
            with sc.tryexcept():
                state.people = shrunk
                state.raw = shrunk
        return

    def plot(self):
        """ Plot all results in the module """
        with sc.options.with_style('fancy'):
            flat = sc.flattendict(self.results, sep=': ')
            timevec = self.t.timevec
            fig, axs = sc.getrowscols(len(flat), make=True)
            for ax, (k, v) in zip(axs.flatten(), flat.items()):
                ax.plot(timevec, v)
                ax.set_title(k)
                ax.set_xlabel('Year')
        return ss.return_fig(fig)