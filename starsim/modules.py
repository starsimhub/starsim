"""
General module class -- base class for diseases, interventions, etc. Also
defines Analyzers and Connectors.
"""
import inspect
import functools as ft
import numpy as np
import sciris as sc
import starsim as ss


module_args = ['name', 'label']  # Define allowable module arguments

custom_modules = []  # Allow the user to register custom modules


def module_map(key=None):
    """
    Map modules to standard types

    This is the source of truth about module types. Modules with types that do not
    map to anything in the module_map are treated as custom modules.
    """
    module_map = sc.objdict(
        demographics  = ss.Demographics,
        networks      = ss.networks.Route, # NB, not ss.Network!
        diseases      = ss.Disease,
        connectors    = ss.Connector,
        interventions = ss.Intervention,
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

    Examples:
        ```python
        # Standard use case, register modules automatically
        import my_custom_disease_model as mcdm
        ss.register_modules(mcdm)
        ss.Sim(diseases='mydisease', networks='random').run() # This will work if mcdm.MyDisease() is defined

        # Manual usage
        my_modules = [mcdm.MyDisease, mcdm.MyNetwork]
        ss.register_modules(my_modules)
        ss.Sim(diseases='mydisease', networks='mynetwork').run()
        ```
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
    modules['modules'] = sc.objdict() # Catch-all for modules that do not match a standard type (e.g. user-created ss.Module classes)

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


def required():
    """
    Decorator to mark module methods as required.

    A common gotcha in Starsim is to forget to call super(), or to mistype a method
    name so it's never called. This decorator lets you mark methods (of Modules only)
    to be sure that they are called either on sim initialization or on sim run.

    Examples:
        ```python
        import starsim.library as ssl

        class CustomSIS(ssl.AgentSIS):

            def step(self):
                super().step()
                self.custom_step() # Will raise an exception if this line is not here
                return

            @ss.required() # Mark this method as required on run
            def custom_step(self):
                pass
        ```
    """
    # Wrap the function
    def decorator(func):
        func._call_required = True  # Mark for collection

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
    function signature can make it easier to read, and "_" is a convention to indicate
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

    Examples:
        ```python
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
        ```
    """
    _state_aliases = {} # Default state aliases; defined on the class so it always exists, e.g. while unpickling. See define_aliases()

    def __init__(self, name=None, label=None, **kwargs):
        # Housekeeping
        self._locked_attrs = ['pars', 't', 'sim', 'dists', 'results'] # Define key attributes that shouldn't be overwritten
        self._collect_required() # First, collect methods marked as required on creation

        # Handle parameters
        self.pars = ss.Pars() # Usually populated via self.define_pars()
        self._default_name = True # Whether the name is the class default, in which case the sim can rename it to make it unique
        self._implicit = False # Whether it was made or named by the sim (from the transitions it was given, or its states), so its results are also at the top of the sim's, e.g. sim.results.I
        named = type(self) is ss.Disease or isinstance(self, (ss.modify, ss.Delivery, ss.Product)) # The modules that take the name of their variable (see ss.Sim._name_modules())
        self._assigned = ss.utils.assigned_name(self) if named else None # The variable it was first assigned to, e.g. campaign = ss.vaccinate(...)
        self.set_metadata(name, label) # Usually reset as part of self.update_pars()
        unknown = [key for key in kwargs if key not in ss.Timeline.keywords]
        if unknown:
            errormsg = f'{type(self).__name__}() got {"an unexpected keyword" if len(unknown) == 1 else "unexpected keywords"} {sc.strjoin(f"{key}=" for key in unknown)} (see the migrate-starsim3 skill for keywords that have changed)'
            raise TypeError(errormsg)
        self.t = ss.Timeline(**kwargs, name=self.name, init=False) # Initialized in init_pre(), so the sim's dt is used if not supplied

        # Properties to be added by init_pre()
        self.sim = None
        self.dists = None # Turned into a Dists object by sim._init_dists() if this module has dists
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

    def __getattr__(self, attr):
        """ Called only if the attribute isn't found, in which case fall back to the aliases; see define_aliases() """
        if attr not in self._state_aliases:
            prop = getattr(type(self), attr, None)
            if isinstance(prop, property): # The property raised an AttributeError of its own (e.g. in a user's function), so let it through rather than reporting the property as missing
                return prop.fget(self)
            raise AttributeError(f'{type(self).__name__} object has no attribute "{attr}"') # Note: don't use self or self.name here, since both can recurse into __getattr__
        alias = self._state_aliases[attr]
        if not callable(alias):
            return getattr(self, alias)
        out = alias(self)
        if isinstance(out, ss.Arr) and out.name is None: # An anonymous Arr is a temporary, so writing to it would be silently discarded; a named one is another state, so leave it alone
            out.raw.flags.writeable = False
        return out

    def __setattr__(self, attr, value):
        """ Don't allow locked attributes to be overwritten """
        if getattr(self, '_lock_attrs', False) and attr in self._locked_attrs:
            if value is getattr(self, attr, None): # Allow assigning the same object back - this happens automatically with some in-place operators (e.g., +=)
                return
            if attr == 't': # Common in Covasim v3 analyzers, e.g. self.t = []
                errormsg = f'"t" is reserved for the module\'s Timeline; store your own time data under another name, e.g. self.tvec'
                raise AttributeError(errormsg)
            errormsg = f'Cannot modify attribute "{attr}"; locked attributes are {sc.strjoin(self._locked_attrs)}.\n'
            errormsg += 'If you really mean to do this, use module.setattribute() or set module._lock_attrs = False'
            try:
                if isinstance(getattr(self, attr),ss.Arr) and len(getattr(self, attr)) == len(value):
                    errormsg += f'\nOften this error can be resolved by replacing `self.{attr} = x` with `self.{attr}[:] = x`'
            except:
                pass
            raise AttributeError(errormsg)
        else:
            super().__setattr__(attr, value)
        return

    def setattribute(self, attr, value):
        """ Method for setting an attribute that does not perform checking against immutable attributes """
        return super().__setattr__(attr, value)

    def check_not_given(self, action):
        """ Raise if this module was given to a sim, which runs its own copy, so that `action` on this one would never be seen """
        if getattr(self, '_given', False):
            kind = next((key for key,cls in module_map().items() if isinstance(self, cls)), 'custom')
            errormsg = (f'{action} on "{self.name}", which was given to a sim: the sim runs its own copy, so it would never be used. '
                        f'Do it before making the sim, or on the sim\'s copy, sim.{kind}.{self.name}, after sim.init()')
            raise RuntimeError(errormsg)
        return

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
        for cls in inspect.getmro(type(self)):
            for attr,method in cls.__dict__.items():
                if getattr(method, '_call_required', False):
                    key = f'{cls.__qualname__}.{attr}'
                    reqs[key] = 0 # Meaning it's been called 0 times
        self._call_required = reqs
        return reqs

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
        old_name = getattr(self, 'name', None)
        self.name = self._reconcile('name', name, cls_lower)
        if name is not None or self.name != cls_lower: # The user chose the name, so never rename it
            self._default_name = False
        default_label = self.name if self.name != cls_lower else cls_name
        if label is None and getattr(self, 'label', None) in [old_name, cls_name]: # The existing label is a default, so update it to match the new name
            label = default_label
        self.label = self._reconcile('label', label, default_label)
        return

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create a module instance by name

        Args:
            name (str): A string with the name of the module class in lower case, e.g. 'sir'
        """
        subcls = ss.modules.find_modules(flat=True).get(name)
        if subcls is not None and issubclass(subcls, cls):
            return subcls(*args, **kwargs)
        else:
            raise KeyError(f'Module "{name}" did not match any known Starsim modules')

    @classmethod
    def from_func(cls, func):
        """
        Create a module from a function

        Wraps a plain function into a module of the calling class, so that simple,
        one-off logic can be used anywhere a module is expected without writing a
        full class. The function must take the sim as its only argument; it becomes
        the module's `step()` method and is therefore called once on every timestep
        (so it must do its own time gating, e.g. `if sim.now == 2015:`). The module's
        `name` is taken from the function's name.

        This is the mechanism behind "function-based" interventions and analyzers:
        passing a function to `interventions=`, `analyzers=`, or `custom=`
        automatically calls `from_func()` on the appropriate class
        (`ss.Intervention`, `ss.Analyzer`, or `ss.Module` respectively), so usually
        you don't need to call this method directly. Call it explicitly when you want
        to construct the module yourself, e.g. to add it to a sim later.

        Keyword arguments of the function with defaults are the module's parameters,
        passed to it at each call; one given as `ss.par` with bounds can be fitted
        (`sim.fit()`), sampled (`sim.sensitivity()`), and set by name (`ss.Sim(pars=...)`,
        `ss.parallel(pars=...)`) as any other. This is how to fit a value that is not a
        parameter of any module, e.g. one set in code (see the second example).

        Function-based modules are otherwise minimal: they have no states or results
        of their own. As soon as you need `define_states()` or `define_results()`,
        write a proper subclass instead.

        Args:
            func (callable): a function taking the sim, and optionally keyword arguments with defaults (its parameters)

        Returns:
            mod (Module): an instance of `cls` whose `step()` calls `func(sim)`

        **Example**::

            import starsim as ss

            def knockdown(sim): # Halve transmission for one year starting in 2015
                if sim.now == 2015:
                    sim.diseases.sis.rel_trans[:] *= 0.5

            # Explicit construction ...
            intv = ss.Intervention.from_func(knockdown)
            sim = ss.Sim(diseases='sis', networks='random', interventions=intv)

            # ... or equivalently, let the sim convert it automatically:
            sim = ss.Sim(diseases='sis', networks='random', interventions=knockdown)

            # A value set in code, fitted: the share of transmission left in 2015
            def knockdown(sim, share=ss.par(0.5, bounds=[0.1, 1])):
                if sim.now == 2015:
                    sim.diseases.sis.rel_trans[:] *= share
        """
        def step(mod): # TODO: see if this can be done more simply
            return mod.func(mod.sim, **mod.pars)
        name = func.__name__
        mod = cls(name=name)
        defaults = {k:p.default for k,p in inspect.signature(func).parameters.items() if p.default is not inspect.Parameter.empty}
        if defaults:
            mod.define_pars(**sc.dcp(defaults)) # Copied, since a default is shared by every call (and sim)
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
            declared = '_declared' in vars(self) or getattr(self, 'graph', None) is not None # Declared, or being declared (see ss.Disease.define_transitions()), without building it
            given = {k:v for k,v in kwargs.items() if isinstance(v, ss.par)} if not declared else {} # In a module written as code, a parameter given as ss.par (e.g. with bounds, to be fitted) is kept as one, and its code uses the value, as for update_pars()
            self.pars.update(**{k:(v.value if k in given else v) for k,v in kwargs.items()}, create=True)
            if given:
                self.pars.update(**given)
            self.pars.forget_bare(kwargs) # A module's own defaults are in the unit it means
        else: # Or overwrite
            self.pars = ss.Pars(**kwargs)
        return self.pars

    def value(self, name):
        """
        The current value of a parameter, as code reads it

        Once the module is in a sim: a typed value as a number in the sim's unit (a duration in its units, a rate per unit), a value
        that varies in time (`ss.timeseries`, `ss.seasonal`) at the current time, a value by level of one dimension (`ss.by(patch=[...])`)
        as one value for each level (of several, one for each group of the sim), and the modifiers active on a declared parameter applied
        (if they differ between groups or agents, one value for each). Anything else (a distribution, a function, a dict) is as it was given.
        Before the module is in a sim, the value it was given. This is the same for a declared module and one written as code; once the sim
        is initialized, arithmetic on a declared parameter (e.g. `1 - disease.pars.eff_condom`) uses this value too.

        Args:
            name (str): the name of the parameter (or the `ss.par` itself)

        Examples:
            ```python
            eff = disease.value('eff_condom')
            bites = self.sim.diseases.malaria.value('bites_per_female_per_day') # Per day, in a sim in days
            ```
        """
        p = self.pars[name] if isinstance(name, str) else name
        raw = p.raw if isinstance(p, ss.par) else p
        if self.sim is None or getattr(self.sim, 'unit_base', None) is None: # Not in a sim yet, or its unit is not yet known
            return raw
        engine = getattr(self, 'engine', None)
        if isinstance(p, ss.par) and engine is not None: # A declared parameter: as the engine reads it, with its modifiers, unless a value by level with none
            if not isinstance(raw, (ss.by, ss.parameters.by_func)) or self.sim.modifiers.get((self.name, 'par', p.name)):
                return engine.value(p)
        return ss.parameters.current_value(raw, self.sim, cache=self.__dict__.setdefault('_converted', {}))

    # Warning: do not try to use a decorator with this function, that will break argument passing!
    def update_pars(self, pars=None, **kwargs):
        """
        Pull out recognized parameters and update them; raises on unrecognized keys
        """
        # Merge pars and kwargs
        if isinstance(pars, ss.Pars): # E.g. a package's full set of defaults for the module: those given as ss.par keep their bounds
            pars = pars.with_bounded()
        pars = sc.mergedicts(pars, kwargs)

        # Pull out the named arguments of this module's nearest __init__ (found by identity, so a wrapper or an overriding update_pars() doesn't hide it)
        frame = ss.utils.getframe(1)
        while frame is not None and not (frame.f_code.co_name == '__init__' and frame.f_locals.get('self') is self):
            frame = frame.f_back
        if frame is not None: # If it's not being called from init, don't do this
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
            errormsg = f'{len(remaining)} unrecognized argument{"s" if len(remaining) > 1 else ""} for {self.name}: {sc.strjoin(sorted(remaining))}'
            raise ValueError(errormsg)
        return

    def define_states(self, *args, reset=False, lock=True, overwrite=False, **kwargs):
        """
        Define states of the module with the same attribute name as the state

        In addition to registering the state with the module by attribute, it adds
        it to `mod._auto_states`, which is used by `mod.state_list` and `mod.state_dict`.

        Any keyword arguments other than the ones listed below are passed to
        `define_aliases()`, so states and their aliases can be defined together.

        Args:
            args (states): list of states to add
            reset (bool/str/list): if True, remove all existing states and aliases; if a name or list of names, remove only those
            lock (bool): if True, prevent the state attributes from being overwritten after definition
            overwrite (bool): if True, replace any existing state or alias of the same name, rather than raising
            kwargs (dict): passed to `define_aliases()`

        **Examples**:

            self.define_states(ss.BoolState('exposed'), reset='ti_infected') # Add a state and remove an inherited one
            self.define_states(ss.BoolState('infectious'), overwrite=True) # Replace an inherited state or alias
        """
        # Optionally remove existing states and aliases: all of them (reset=True), or just the named
        # ones (note: does not remove them from the people object or others if already added); see example in ss.SIR()
        if reset:
            names = ([state.name for state in self.state_list] + list(self._state_aliases)) if reset is True else sc.tolist(reset)
            for name in names:
                self._remove_state(name)

        # Add the new states
        for arg in args:
            if isinstance(arg, (list, tuple)):
                state = ss.BoolState(*arg)
            elif isinstance(arg, dict):
                state = ss.BoolState(**arg)
            else:
                state = arg
            assert isinstance(state, ss.Arr), f'Could not add {state}: not an Arr object'

            # Add the state to the module, replacing any existing state or alias of the same
            # name if overwrite=True (removing it first, so it isn't listed or counted twice)
            attr = state.name
            if overwrite:
                self._remove_state(attr, die=False)
            elif hasattr(self, attr):
                present = [s.name for s in self.state_list]
                new = [s.name for s in args]
                errormsg = f'Cannot add "{attr}" to {self._debug_name} since already present in module (as a state or an alias).\n'
                errormsg += 'Did you mean to use define_states(overwrite=True) (replace it) or define_states(reset=...) (remove it first, by name or all of them)?\n'
                errormsg += f'States already in module:\n{present}\n'
                errormsg += f'Aliases already in module:\n{list(self._state_aliases)}\n'
                errormsg += f'New states being added:\n{new}\n'
                raise AttributeError(errormsg)
            setattr(self, attr, state)
            state.mod = self # So a condition on it, written before the sim exists, refers to this module
            if lock:
                self._locked_attrs.append(attr)

            # Add it to the list of auto states, if needed
            if isinstance(state, ss.BoolState):
                self._auto_states.append(state)

        # Handle any aliases
        if kwargs:
            self.define_aliases(overwrite=overwrite, **kwargs)
        return

    def define_aliases(self, overwrite=False, **kwargs):
        """
        Define aliases for states, e.g. `self.define_aliases(infectious='infected')`

        An alias is only used if the attribute isn't otherwise defined, so it acts as a
        default that a subclass overrides simply by defining a state of the same name.
        This is how `ss.Infection` treats everyone infected as infectious, while letting
        a subclass define `infectious` as a state (or another alias) in its own right.

        Each alias is either the name of another state, or a callable taking the module as
        its only argument. A callable behaves like a property: it is recomputed on each
        access, so it can be read but not written to.

        Note: a callable alias defined as a lambda can't be saved with plain `pickle`
        (`sc.save()` and `ss.MultiSim` are both fine, since they use `dill`). Use a
        module-level function, or a property, if plain pickling is required.

        A callable alias also generates an automatic result, in the same way a `BoolState`
        does: an alias named `infected` produces the result `infected`. A string alias does not,
        since the state it points to is already counted under its own name.

        Args:
            overwrite (bool): if True, replace any existing state or alias of the same name, rather than raising
            kwargs (dict): mapping of alias name to either the name of a state, or a callable

        **Examples**:

            self.define_aliases(infectious='infected') # Everyone infected is infectious
            self.define_aliases(infected=lambda self: self.exposed | self.infectious) # Both E and I are infected
            self.define_states(ss.BoolState('exposed'), infectious='infected') # Aliases can also be defined alongside states
        """
        for key in kwargs:
            if overwrite:
                self._remove_state(key, die=False)
            elif key not in self._state_aliases and hasattr(self, key): # An alias would never be consulted, since the state takes precedence
                errormsg = f'Cannot define alias "{key}" in {self._debug_name} since a state of that name already exists; '
                errormsg += f'use overwrite=True to replace it, or define_states(reset="{key}") to remove it first.'
                raise AttributeError(errormsg)
        self._state_aliases = self._state_aliases | kwargs # Copy rather than update in place, since the default is defined on the class
        return

    def _remove_state(self, name, die=True):
        """ Remove a single state or alias by name; see `define_states()` """
        if name in self._state_aliases:
            self._state_aliases = {k:v for k,v in self._state_aliases.items() if k != name} # Copy rather than modify in place, since the default is defined on the class
        elif isinstance(self.__dict__.get(name), ss.Arr):
            delattr(self, name)
            self._auto_states = [state for state in self._auto_states if state.name != name]
            if name in self._locked_attrs:
                self._locked_attrs.remove(name)
        elif die:
            errormsg = f'Cannot remove "{name}" from {self._debug_name} since it is not a state or alias; '
            errormsg += f'states are {sc.strjoin(self.state_dict.keys())} and aliases are {sc.strjoin(self._state_aliases.keys())}.'
            raise AttributeError(errormsg)
        return

    def define_results(self, *args):
        """ Add results to the module """
        auto_names = {state.name for state in self.auto_state_list} | set(self.derived_state_names) # Names that already generate a result of the same name, their count
        for arg in args:
            if isinstance(arg, (list, tuple)):
                result = ss.Result(*arg)
            elif isinstance(arg, dict):
                result = ss.Result(**arg)
            else:
                result = arg

            # Update with module information
            result.update(module=self.label, shape=self.t.npts, timevec=self.t.timevec)
            if getattr(result, 'groups', False): # Its values by group too, as a declared module's are: see ss.Result.set_groups()
                groups = self.sim.groups
                result.init_groups(groups.strats, [strat.name for strat in groups.strats], (self.t.npts, groups.n_groups))

            # Add the result to the dict of results; does automatic checking
            if result.name in self.results and result.name in auto_names:
                msg = f'"{self.name}": Another result named "{result.name}" already exists because a result was automatically created for the state "{result.name}" (the number in it).'
                raise ValueError(msg)
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
            ss.distributions.link_dists(self, sim, skip=[ss.Sim, ss.Module]) # Link the distributions to sim and module, skipping any nested sim or module instances
            self.t.init(sim=self.sim) # Initialize time vector
            self.link_timepars() # Add module dt to the timepars
            self.check_bare_units(sim)
            if isinstance(self.pars, ss.Pars) and getattr(self, 'graph', None) is None: # A declared module checks its own, against its graph (see ss.Disease.init_pre())
                self.pars.check_added(self.name) # A parameter set but never defined, e.g. renamed (MG7)
            if self.name in sim.pars:
                sim.pars[self.name] = self.pars
            else: # Add a new key, which is allowed even if the sim's parameters are locked
                sim.pars.update({self.name: self.pars}, create=True)
            sim.results[self.name] = self.results
            sim.people.add_module(self) # Connect the states to the people
            self.init_results()
            self.pre_initialized = True
        return

    def check_bare_units(self, sim):
        """ Warn about parameters given as bare numbers in a unit other than the sim's (see `ss.Pars.bare_units()`) """
        if getattr(sim, 'declared', None): # In a declared model, bare numbers are in the sim's unit (None for timesteps)
            unit = sim.unit and ss.time.normalize_unit(sim.unit)
        else: # Otherwise in its timestep's, if it has one (not in a mock sim)
            unit = getattr(sim.t.dt, 'base', None)
            if unit is None:
                return
        if isinstance(self.pars, ss.Pars):
            self.pars.bare_units(unit, label=self.name)
        return

    @required()
    def link_timepars(self, force=False):
        """
        Find all time parameters in the module and link them to the module's `dt`

        This is what lets a timepar resolve itself against the timestep with no
        argument, e.g. `self.pars.death_rate.to_prob()` or `self.pars.dur_inf.to_dt()`.
        """
        timepars = sc.search(self, type=ss.TimePar, skip=dict(keys=ss.utils.skipkeys(['sim', 'module', 't'] + list(getattr(self, 'timepar_skip', []))), subclasses=ss.distributions.no_dists())) # Skip the timeline, else the module's own dt would be linked to itself

        # Initialize them with the parent module
        dt = self.t.dt
        for timepar in timepars.values():
            if timepar is dt: # Avoid a self-reference, which would recurse when serializing
                continue
            if force or timepar.default_dur is None:
                timepar.set_default_dur(dt)
        return

    @required()
    def init_results(self):
        """
        Initialize results output; called during `init_pre()`

        By default, modules all report on counts for any explicitly defined "States", e.g. if
        a disease contains an `ss.BoolState` called 'susceptible' it will automatically contain a
        Result for 'susceptible' (the number susceptible). For identical behavior that does not automatically
        generate results, use `ss.BoolArr` instead of `ss.BoolState`.
        """
        self.results.timevec = self.t.timevec # Store the timevec in the results for plotting; not a Result so don't use ss.ndict.append()
        results = sc.autolist()
        for state in self.auto_state_list:
            results += ss.Result(state.name, dtype=int, scale=True, label=state.label, summarize_by='last')
        for name in self.derived_state_names: # A derived state has no state object, but is counted the same way
            results += ss.Result(name, dtype=int, scale=True, label=name.capitalize(), summarize_by='last')
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
        sim = ss.debugtools.mock_sim(n_agents=n_agents, dur=dur)
        self.init_pre(sim)
        for state in self.state_list: # Manually link the people
            state.people = self.sim.people
        obj = [self.pars, self.state_list]
        dists = sc.search(obj, type=ss.Dist)
        for i,dist in dists.enumvals():
            dist.trace = f'dist_{i}'
        ss.distributions.link_dists(obj=obj, sim=sim, module=self, skip=[ss.Sim, ss.Module], init=True)
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
        of the same names (susceptible, infectious, etc.), the number in each. For a list of all states, see
        `state_list`.
        """
        return self._auto_states[:]

    @property
    def derived_state_names(self):
        """
        Names of the aliases that are defined by a callable (see `define_aliases()`)

        Unlike a string alias, which points at a state that is already counted under its
        own name, a derived state has no state object of its own, so it generates an
        automatic result (e.g. `infected`) in the same way a `BoolState` does.
        """
        return [key for key,alias in self._state_aliases.items() if callable(alias)]

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
            raise ss.AlreadyRunError(errormsg)
        if self.dists is not None and ss.options.crn: # Will be None if no distributions are defined; jumping is a no-op under crn=False (see Dist.jump), so skip the per-dist loop entirely
            self.dists.jump_dt() # Advance random number generators forward for calls on this step
        return

    def step(self):
        """ Define how the module updates over time -- the key part of Starsim!! """
        errormsg = f'The module {self._debug_name} does not define a "step" method. This is usually a bug, but use "def step(self): pass" if this is intentional.'
        ss.warn(errormsg)
        return

    @required()
    def finish_step(self):
        """ Define what should happen at the end of the step, after the results are recorded """
        return

    def advance(self):
        """ For a module with a timeline of its own: move its time to the step about to be taken (the sim moves the time of the modules on its own timeline: see `ss.Sim.advance()`) """
        self.t.ti += 1
        return

    def update_results(self, names=None):
        """
        Update results; by default, compute counts of each state at each point in time

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.

        Args:
            names (list): the states to count (default: every `ss.BoolState` and derived state)
        """
        weights = ss.results.agent_weights(self.sim) # Agents of different weights (people.scale) are each counted by their weight, as in the result alive
        if names is None:
            names = [state.name for state in self.auto_state_list] + self.derived_state_names
        for name in names:
            res = self.results[name]
            if weights is None:
                res[self.ti] = self[name].sum()
            else:
                ss.results.as_float(res)[self.ti] = weights[self[name].uids].sum()
        return

    @required()
    def finalize(self):
        """ Perform any final operations, such as removing unneeded data """
        self.finalize_results()
        self.finalized = True
        return

    @required()
    def finalize_results(self): # TODO: this is confusing, needs to be not redefined by the user, or called *after* a custom finalize_results()
        """ Finalize results """
        # Scale results, including any nested ones
        scale = self.sim.result_scale(self)
        for res in self.results.flatten().values():
            if res.scale:
                res.apply_scale(scale)
        return

    def to_json(self):
        """ The module as a reference by its type and name, for `sc.jsonify()`: e.g. a module among the parameters in `sim.to_json()`, or in a parameter of another module """
        return dict(type=type(self).__name__, name=self.name)

    def shrink(self):
        """ Shrink the size of the module for saving to disk """
        shrunk = ss.utils.shrink()
        self.setattribute('sim', shrunk) # Use setattribute since locked otherwise
        self.setattribute('dists', shrunk)
        for state in self.state_list:
            with sc.tryexcept():
                to_shrink = ['people', 'raw']
                ss.utils.shrink(state, to_shrink)
        return

    def plot(self, *keys, **kwargs):
        """
        Plot the module's results

        Args:
            keys (str/list): what to plot: `'states'` (the number in each state, together in one axes; the default,
                if the module has states), `'flows'` (each transition's flow, in axes of its own), `'all'` (every result,
                each in axes of its own; the default otherwise), or the names of results: a list of them in one axes
                (`['S', 'I']`), and each argument in axes of its own (`'S', 'I'`)
            kwargs (dict): passed to `ss.Results.plot()`, e.g. `fig`, `figsize`, `annualize`

        Examples:
            ```python
            sir.plot()           # The states
            sir.plot('flows')    # Each flow
            sir.plot(['S', 'I']) # S and I, together
            sir.plot('S', 'I')   # S and I, apart
            ```
        """
        if not len(self.results): # E.g. a module given to a sim, which runs a copy of it
            errormsg = f'"{self.name}" has no results: a sim runs a copy of each module it is given, so plot the one in the sim, e.g. sim.{self.name}.plot()'
            raise ValueError(errormsg)
        results = {k:res for k,res in self.results.items() if isinstance(res, ss.Result)}
        graph = getattr(self, 'graph', None)
        states = [k for k in (graph.states if graph is not None else [s.name for s in self.auto_state_list] + self.derived_state_names) if k in results]
        flows = [k for k,res in results.items() if res.summarize_by == 'sum' and '.' not in k] # Not the flows to each destination of a transition with several
        groups = []
        for key in keys or (['states'] if states else ['all']):
            if key == 'states':
                groups.append(states)
            elif key == 'flows':
                groups += [[k] for k in flows]
            elif key == 'all':
                groups += [[k] for k,res in results.items() if res.auto_plot]
            else:
                groups.append(sc.tolist(key))
        return self.results.plot([group for group in groups if group], **kwargs)
