"""
General module class -- base class for diseases, interventions, etc. Also
defines Analyzers and Connectors.
"""
import sys
import inspect
import functools as ft
import sciris as sc
import starsim as ss

__all__ = ['module_map', 'module_types', 'register_modules', 'find_modules', 'required', 'Base', 'Module']

module_args = ['name', 'label'] # Define allowable module arguments

custom_modules = [] # Allow the user to register custom modules


def module_map(key=None):
    """ Define the mapping between module names and types; this is the source of truth about module types and ordering """
    module_map = sc.objdict(
        modules       = None, # Handled separately since otherwise isinstance(module, modtype) will match all
        demographics  = ss.Demographics,
        connectors    = ss.Connector,
        networks      = ss.Route, # NB, not ss.Network!
        interventions = ss.Intervention,
        diseases      = ss.Disease,
        analyzers     = ss.Analyzer,
    )
    return module_map if key is None else module_map[key]


def module_types():
    """ Return a list of known module types; based on `module_map()` """
    return list(module_map().keys())


def is_module(obj):
    """ Check whether something is a Starsim module """
    return isinstance(obj, type) and issubclass(obj, ss.Module)


def register_modules(*args):
    """
    Register custom modules with Starsim so they can be referred to by string.

    Note: "modules" here refers to Starsim modules. But this function registers
    Starsim "modules" (e.g. `ss.SIR`) that are found within Python "modules"
    (e.g. `starsim`).

    Args:
        args (list): the additional modules to register; can be either a module or a list of objects

    **Examples**:

        # Standard use case, register modules automatically
        import my_custom_disease_model as mcdm
        ss.register_modules(mcdm)
        ss.Sim(diseases='mydisease', networks='random').run() # This will work if mcdm.MyDisease() is defined

        # Manual usage
        my_modules = [mcdm.MyDisease, mcdm.MyNetwork]
        ss.register_modules(my_modules)
        ss.Sim(diseases='mydisease', networks='mynetwork').run()
    """
    for arg in args:
        custom_modules.append(arg)
    return


def find_modules(key=None, flat=False, verbose=False):
    """ Find all subclasses of Module present in Starsim, divided by type """
    modules = sc.objdict()
    modmap = module_map()
    attr_lists = [[ss, dir(ss)]] # Find all attributes in Starsim (note: does not parse user code; use register_modules() for that)
    for custom in custom_modules:
        if verbose: print(f'Loading {custom}...')
        if sc.ismodule(custom):
            attr_lists.append([custom, dir(custom)])
        else:
            attr_lists.append([None, sc.tolist(custom)])

    # Initialize dict
    for modkey in modmap.keys():
        modules[modkey] = sc.objdict()

    # Loop over all attributes
    for pymodule,attrs in attr_lists:
        if verbose: sc.heading(f'Processing {pymodule} ...')
        for attr in attrs: # Loop over each attribute (inefficient, but doesn't need to be optimized)
            if verbose: print(f'  Checking {attr}')

            # Handle modules or a list of objects being provided
            if pymodule is not None: # Main use case: pymodule=ss, attr='HIV'
                item = getattr(pymodule, attr)
                low_attr = attr.lower()
            else: # Alternate use case: pymodule=None, attr=HIV
                if is_module(attr): # It's a module, proceed
                    item = attr
                    low_attr = item.__name__.lower()
                else: # It's not a module, skip
                    item = None

            # Check whether it's a module
            ismodule = is_module(item)
            if ismodule:
                assigned = False
                for modkey, modtype in modmap.items(): # Loop over each module type
                    if modtype is not None and issubclass(item, modtype):
                        modules[modkey][low_attr] = item # It passes, so assign it to the dict
                        if modkey == 'networks' and low_attr.endswith('net'): # Also allow networks without 'net' suffix
                            altname = low_attr.removesuffix('net')
                            modules[modkey][altname] = item
                        assigned = True
                        if verbose: print(f'     Module {modkey}.{low_attr} added')
                        break # Exit the innermost loop
                if not assigned:
                    modules['modules'][low_attr] = item
                    if verbose: print(f'     Module modules.{low_attr} added')

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


class Base:
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

    def __repr__(self):
        """ Equivalent to sc.prettyobj() """
        return sc.prepr(self)

    def disp(self, output=False, **kwargs):
        """ Display the full object """
        out = sc.prepr(self, **kwargs)
        if not output:
            print(out)
        else:
            return out

    @property
    def dt(self):
        """ Get the current module timestep """
        try:    return self.t.dt
        except: return None

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

    By convention, keyword arguments to modules are assigned to "_", which is an
    alias for None. These are then populated with defaults by `mod.define_pars()`,
    which are then updated with `mod.update_pars()`, which inspects the `__init__`
    function to get these arguments.

    Note that there is no *functional* difference between specifying arguments
    this way rather than simply via `**kwargs`, but having the arguments shown in the
    function signature can make it easier to read, and "_" is a convetion to indicate
    that the default is specified below.

    It is also of course OK to specify the actual values rather than "_"; however
    module arguments are often complex objects (e.g. `ss.bernoulli`) that can't
    be easily specified in the function signature.

    Finally, note that you can (and should) call `super().__init__()` with no arguments:
    the name, label, and time arguments get correctly updated via `self.update_pars()`.

    Args:
        name (str): a short, key-like name for the module (e.g. "randomnet")
        label (str): the full, human-readable name for the module (e.g. "Random network")
        kwargs (dict): passed to `ss.Timeline()` (e.g. start, stop, unit, dt)

    **Example**:

        class SIR(ss.Module):
            def __init__(self, pars=_, beta=_, init_prev=_, p_death=_, **kwargs):
                super().__init__() # Call this first with no arguments
                self.define_pars( # Then define the parameters, including their default value
                    beta = ss.peryear(0.1),
                    init_prev = ss.bernoulli(p=0.01),
                    p_death = ss.bernoulli(p=0.3),
                )
                self.update_pars(pars, **kwargs) # Update with any user-supplied parameters, and raise an exception if trying to set a parameter that wasn't defined in define_pars()
                return
    """
    def __init__(self, name=None, label=None, **kwargs):
        # Housekeeping
        self._locked_attrs = ['pars', 't', 'sim', 'dists'] # Define key attributes that shouldn't be overwritten; note, 'results' would be included but self.results += result calls __setattr__
        self._collect_required() # First, collect methods marked as required on creation

        # Handle parameters
        self.pars = ss.Pars() # Usually populated via self.define_pars()
        self.set_metadata(name, label) # Usually reset as part of self.update_pars()
        self.t = ss.Timeline(**kwargs, name=self.name)

        # Properties to be added by init_pre()
        self.sim = None
        self.dists = None # Turned into a Dists object by sim.init_dists() if this module has dists
        self.results = ss.Results(self.label)

        # Finish initialization
        self.pre_initialized = False
        self.initialized = False
        self.finalized = False
        self._lock_attrs = True # Prevent key attributes from being overwritten directly
        self._auto_states = [] # Store automatic states (State objects) added via define_states()
        return

    def __call__(self, *args, **kwargs):
        """ Allow modules to be called like functions """
        return self.step(*args, **kwargs)

    def __getitem__(self, key):
        """ Allow modules to act like dictionaries """
        return getattr(self, key)

    def __setattr__(self, attr, value):
        """ Don't allow locked attributes to be overwritten """
        if getattr(self, '_lock_attrs', False) and attr in self._locked_attrs:
            errormsg = f'Cannot modify attribute "{attr}"; locked attributes are {sc.strjoin(self._locked_attrs)}.\n'
            errormsg += 'If you really mean to do this, use module.setattribute() or set module._lock_attrs = False'
            raise AttributeError(errormsg)
        else:
            super().__setattr__(attr, value)
        return

    def setattribute(self, attr, value):
        """ Method for setting an attribute that does not perform checking against immutable attributes """
        return super().__setattr__(attr, value)

    def brief(self, output=False):
        """ Show a brief representation of the module; used by __repr__ """
        name = self.name # e.g. 'sir'
        label = self.label # e.g. 'My SIR'
        class_name = self.__class__.__name__ # e.g. 'SIR'
        class_str = f':{class_name}' if name != class_name.lower() else '' # e.g. 'SIR'
        label_str = f'"{label}"; ' if (label.lower() != name and label != class_name) else ''
        pars_str = '[' + sc.strjoin(self.pars.keys()) + ']' if len(self.pars) else 'None'
        states_str = '[' + sc.strjoin(self.state_dict.keys()) + ']' if len(self.state_dict) else 'None'
        out = f'{name}{class_str}({label_str}pars={pars_str}; states={states_str})'
        return out if output else print(out)

    def __repr__(self):
        """ Show the object including parameters and states """
        try: # Default representation
            return self.brief(output=True)
        except: # If for any reason that fails, return the full repr
            return sc.prepr(self)

    @property
    def _debug_name(self):
        """ The module name and class shown as a string, used in debugging/error messages """
        out = f'"{self.name}" ({type(self)}'
        return out

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

        # Set values for name and label
        cls_name = self.__class__.__name__
        cls_lower = cls_name.lower()
        self.name = self._reconcile('name', name, cls_lower)
        default_label = self.name if self.name != cls_lower else cls_name
        self.label = self._reconcile('label', label, default_label)
        return

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

    def define_pars(self, inherit=True, **kwargs): # TODO: think if inherit should default to true or false
        """
        Create or merge Pars objects

        Note: this method also automatically pulls in keyword arguments from the
        calling function (which is almost always a module's `__init__()` method)
        """
        if inherit: # Merge with existing
            self.pars.update(**kwargs, create=True)
        else: # Or overwrite
            self.pars = ss.Pars(**kwargs)
        return self.pars

    # Warning: do not try to use a decorator with this function, that will break argument passing!
    def update_pars(self, pars=None, **kwargs):
        """
        Pull out recognized parameters, returning the rest
        """
        # Merge pars and kwargs
        pars = sc.mergedicts(pars, kwargs)

        # Inspect the parent frame and pull out any arguments
        frame = sys._getframe(1)  # Go back 1 frame (to __init__, most likely)
        if frame.f_code.co_name == '__init__': # If it's not being called from init, don't do this
            _, _, _, kw = inspect.getargvalues(frame) # Get the values provided
            for k,v in kw.items():
                if k in self.pars and v is not None:
                    pars[k] = v

        # Update matching module parameters
        matches = {}
        for key in list(pars.keys()): # Need to cast to list to avoid "dict changed during iteration"
            if key in self.pars:
                matches[key] = pars.pop(key)
        self.pars.update(matches)

        # Update module attributes
        metadata = {key:pars.get(key, self.pars.get(key)) for key in module_args}
        timepars = {key:pars.get(key, self.pars.get(key)) for key in ss.Timeline.time_args}
        self.set_metadata(**metadata)
        self.t.update(**timepars)

        # Should be no remaining pars
        remaining = set(pars.keys()) - set(module_args) - set(ss.Timeline.time_args)
        if len(remaining):
            errormsg = f'{len(pars)} unrecognized arguments for {self.name}: {sc.strjoin(remaining)}'
            raise ValueError(errormsg)
        return

    def define_states(self, *args, check=True, reset=False, lock=True):
        """
        Define states of the module with the same attribute name as the state

        In addition to registering the state with the module by attribute, it adds
        it to `mod._all_states`, which is used by `mod.state_list` and `mod.state_dict`.

        Args:
            args (states): list of states to add
            check (bool): whether to check that the object being added is a state, and that it's not already present
            reset (bool): whether to reset the list of module states and use only the ones provided
            lock (bool): if True, prevent states from being
        """
        # Optionally reset the states (note: does not remove them from the people object or others if already added); see example in ss.SIR()
        if reset:
            for state in self.state_list:
                attr = state.name
                delattr(self, attr)
                if attr in self._locked_attrs:
                    self._locked_attrs.remove(attr)
            self._auto_states = []

        # If we're not checking, don't lock the attrs
        if not check:
            orig_lock_attrs = self._lock_attrs
            self._lock_attrs = False

        # Add the new states
        for arg in args:
            if isinstance(arg, (list, tuple)):
                state = ss.BoolState(*arg)
            elif isinstance(arg, dict):
                state = ss.BoolState(**arg)
            else:
                state = arg

            if check:
                assert isinstance(state, ss.Arr), f'Could not add {state}: not an Arr object'

            # Add the state to the module
            attr = state.name
            if check and hasattr(self, attr):
                present = [s.name for s in self.state_list]
                new = [s.name for s in args]
                errormsg = f'Cannot add "{attr}" to {self._debug_name} since already present in module.\n'
                errormsg += 'Did you mean to use define_states(reset=True) (skip inherited states) or define_states(check=False) (skip this check)?\n'
                errormsg += f'States already in module:\n{present}\n'
                errormsg += f'New states being added:\n{new}\n'
                errormsg += f'Conflicting states:\n{set(present) & set(new)}\n'
                raise AttributeError(errormsg)
            setattr(self, attr, state)
            if lock:
                self._locked_attrs.append(attr)

            # Add it to the list of auto states, if needed
            if isinstance(state, ss.BoolState):
                self._auto_states.append(state)

        # Reset the lock state
        if not check:
            self._lock_attrs = orig_lock_attrs
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
            result.update(module=self.label, shape=self.t.npts, timevec=self.t.timevec)

            # Add the result to the dict of results; does automatic checking
            self.results += result
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
            ss.link_dists(self, sim, skip=[ss.Sim, ss.Module]) # Link the distributions to sim and module, skipping any nested sim or module instances
            self.t.init(sim=self.sim) # Initialize time vector
            self.link_rates() # Add module dt to the timepars
            sim.pars[self.name] = self.pars
            sim.results[self.name] = self.results
            sim.people.add_module(self) # Connect the states to the people
            self.init_results()
            self.pre_initialized = True
        return

    @required()
    def link_rates(self, force=False):
        """ Find all time parameters in the module and link them to the module's dt """
        rates = sc.search(self, type=ss.Rate, skip=dict(keys=['sim', 'module'])) # Should it be self or self.pars?

        # Initialize them with the parent module
        for rate in rates.values():
            if force or rate.default_dur is None:
                rate.set_default_dur(self.t.dt)
        return

    @required()
    def init_results(self):
        """
        Initialize results output; called during `init_pre()`

        By default, modules all report on counts for any explicitly defined "States", e.g. if
        a disease contains an `ss.BoolState` called 'susceptible' it will automatically contain a
        Result for 'n_susceptible'. For identical behavior that does not automatically
        generate results, use `ss.BoolArr` instead of `ss.BoolState`.
        """
        self.results.timevec = self.t.timevec # Store the timevec in the results for plotting; not a Result so don't use ss.ndict.append()
        results = sc.autolist()
        for state in self.auto_state_list:
            results += ss.Result(f'n_{state.name}', dtype=int, scale=True, label=state.label)
        self.define_results(*results)
        return

    @required()
    def init_post(self):
        """ Initialize the values of the states; the last step of initialization """
        for state in self.state_list:
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def init_mock(self, n_agents=100, dur=10):
        """ Initialize with a mock simulation -- for debugging purposes only """
        sim = ss.mock_sim(n_agents=n_agents, dur=dur)
        self.init_pre(sim)
        for state in self.state_list: # Manually link the people
            state.people = self.sim.people
        obj = [self.pars, self.state_list]
        dists = sc.search(obj, type=ss.Dist)
        for i,dist in dists.enumvals():
            dist.trace = f'dist_{i}'
        ss.link_dists(obj=obj, sim=sim, module=self, skip=[ss.Sim, ss.Module], init=True)
        self.init_post()
        return self

    @property
    def state_list(self):
        """
        Return a flat list of all states (`ss.Arr` objects)

        The base class returns all states that are contained in top-level attributes
        of the Module. If a Module stores states in a non-standard location (e.g.,
        within a list of states, or otherwise in some other nested structure - perhaps
        due to supporting features like multiple genotypes) then the Module should
        overload this attribute to ensure that all states appear in here.
        """
        out = [val for val in self.__dict__.values() if isinstance(val, ss.Arr)]
        return out

    @property
    def state_dict(self):
        """
        Return a flat dictionary (objdict) of all states

        Will raise an exception if multiple states have the same name.
        """
        return ss.utils.nlist_to_dict(self.state_list)

    @property
    def auto_state_list(self):
        """
        List of "automatic" states with boolean type (`ss.BoolState`) that were added
        via `define_states()`

        For diseases, these states typically represent attributes like 'susceptible',
        'infectious', 'diagnosed' etc. These variables automatically generate results
        like n_susceptible, n_infectious, etc. For a list of all states, see
        `state_list`.
        """
        return self._auto_states[:]

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
        if self.finalized:
            errormsg = f'The module {self._debug_name} has already been run. Did you mean to copy it before running it?'
            raise RuntimeError(errormsg)
        if self.dists is not None: # Will be None if no distributions are defined
            self.dists.jump_dt() # Advance random number generators forward for calls on this step
        return

    def step(self):
        """ Define how the module updates over time -- the key part of Starsim!! """
        errormsg = f'The module {self._debug_name} does not define a "step" method. This is usually a bug, but use "def step(self): pass" if this is intentional.'
        ss.warn(errormsg)
        return

    @required()
    def finish_step(self):
        """ Define what should happen at the end of the step; at minimum, increment ti """
        self.t.ti += 1
        return

    def update_results(self):
        """
        Update results; by default, compute counts of each state at each point in time

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
        for state in self.auto_state_list:
            self.results[f'n_{state.name}'][self.ti] = state.sum()
        return

    @required()
    def finalize(self):
        """ Perform any final operations, such as removing unneeded data """
        # Update the time index (since otherwise pointing to a timepoint that doesn't exist)
        self.t.ti -= 1

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
        self.setattribute('sim', shrunk) # Use setattribute since locked otherwise
        self.setattribute('dists', shrunk)
        for state in self.state_list:
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