"""
The base classes of the parts of a sim: `ss.modules.Base` and `ss.Module`, which diseases, networks, interventions, and the rest inherit from

They exist so that every part of a model has the same parameters, timeline, results, and place in the integration loop.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import itertools
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
    module_map = sc.obj(
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
    Register custom modules (e.g. `ss.SIR`), or the Python modules that contain them, so they can be referred to by string

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
    """
    Find all subclasses of Module present in Starsim, divided by type

    Args:
        key (str): the type of module to return, e.g. 'diseases'; if None, return all
        flat (bool): whether to return one flat dict, not one for each type
        verbose (bool): whether to print each custom module as it is loaded
    """
    modmap = module_map()
    modules = sc.obj({modkey:sc.obj() for modkey in modmap})
    modules['modules'] = sc.obj() # Catch-all for modules that do not match a standard type (e.g. user-created ss.Module classes)
    candidates = [(attr.lower(), getattr(ss, attr)) for attr in dir(ss)] # Note: does not parse user code; use register_modules() for that
    for custom in custom_modules:
        if verbose: print(f'Loading {custom}...')
        if sc.ismodule(custom):
            candidates += [(attr.lower(), getattr(custom, attr)) for attr in dir(custom)]
        else:
            candidates += [(item.__name__.lower(), item) for item in sc.tolist(custom) if is_module(item)]

    for name,item in candidates:
        if not is_module(item):
            continue
        modkey = next((key for key,modtype in modmap.items() if issubclass(item, modtype)), 'modules')
        modules[modkey][name] = item
        if modkey == 'networks' and name.endswith('net'): # Also allow networks without 'net' suffix
            modules[modkey][name.removesuffix('net')] = item
        if verbose: print(f'  Module {modkey}.{name} added')

    if flat:
        modules = sc.obj({k:v for vv in modules.values() for k,v in vv.items()}) # Unpack the nested dict into a flat one
    return modules if key is None else modules[key]


def required():
    """
    Decorator to mark a module's method as required: the sim raises if it is never called (e.g. a missing `super()` call, or a mistyped name)

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


def class_parts(cls):
    """
    The parameters (`ss.par`), and the arrays (`ss.FloatArr`, `ss.BoolArr`, ...) and distributions (`ss.bernoulli`, ...), declared in the body of a module's class and its bases, by name, in the order defined

    A name rebound to something else in a subclass is dropped. Each instance gets its own copy of each (see `ss.Module.define_class_parts()`).

    Args:
        cls (type): the class of the module
    """
    pars = {}
    others = {}
    for base in reversed(cls.__mro__):
        for key,val in vars(base).items():
            pars.pop(key, None)
            others.pop(key, None)
            if isinstance(val, ss.par):
                pars[key] = val
            elif isinstance(val, (ss.Arr, ss.Dist)):
                others[key] = val
    return pars, others


class Base:
    """
    The parent class for Sim and Module objects

    It holds what both have in common, such as how they are shown.
    """
    def __bool__(self):
        """ Ensure that zero-length modules (e.g. networks) are still truthy """
        return True

    def __len__(self):
        """ The length of a module is the number of timepoints; see also len(sim) """
        try:    return self.t.npts
        except Exception: return 0

    def __repr__(self):
        """ Equivalent to sc.prettyobj() """
        return sc.prepr(self)

    def disp(self, output=False, **kwargs):
        """
        Display the full object

        Args:
            output (bool): whether to return the text rather than print it
            **kwargs (dict): passed to `sc.prepr()`
        """
        out = sc.prepr(self, **kwargs)
        if not output:
            print(out)
        else:
            return out

    @property
    def dt(self):
        """ Get the current module timestep """
        try:    return self.t.dt
        except Exception: return None

    @property
    def ti(self):
        """ Get the current module time index """
        try:    return self.t.ti
        except Exception: return None

    @property
    def now(self):
        """ Shortcut to self.t.now() """
        try:    return self.t.now()
        except Exception: return None

    @property
    def timevec(self):
        """ Shortcut to self.t.timevec """
        try:    return self.t.timevec
        except Exception: return None

    def copy(self, die=True):
        """
        Perform a deep copy of the module/sim, as it is (for a sim, with its results); to run a sim again, use `sim.run()`, and for a new, uninitialized sim, `sim.new()`

        Args:
            die (bool): whether to raise an exception if copy fails (else, try a shallow copy)
        """
        out = sc.dcp(self, die=die)
        return out


class Preference(str):
    """
    A reason in `agent_reasons()` for `solver='auto'` to run a module as agents that does not stop a compartmental solver that was asked for

    It exists so that a reason says this in one word.
    """


class Module(Base):
    """
    The main base class for all Starsim modules: diseases, networks, interventions, etc.

    A module declares its parameters (`ss.par`) and the arrays it keeps for each agent (`ss.FloatArr`,
    `ss.BoolArr`, ...) in its class body, and its behavior in methods such as `step()`. Each parameter
    can then be given by keyword (or as `pars=dict(...)`), and the module's code reads its value from
    `self.pars`; each array is an attribute of the same name. Each instance has its own copies.

    A module whose parameters depend on its arguments can instead define them in its `__init__()`, with
    `define_pars()` and then `update_pars()`, which applies the arguments given (and raises for any it
    does not have); its arrays are defined with `define_states()`.

    Args:
        name (str): a short, key-like name for the module (e.g. "randomnet"); if not given, the variable it is assigned to (`net = ss.RandomNet()` is "net"), else its `default_name` (see `ss.Sim._name_modules()`)
        label (str): the full, human-readable name for the module (e.g. "Random network")
        pars (dict): values of the parameters declared in its class body, as an alternative to giving them by keyword
        kwargs (dict): values of those parameters, by name; or passed to `ss.Timeline()` (e.g. start, stop, dt)

    Examples:
        ```python
        class Decay(ss.Intervention):
            half_life = ss.par(ss.years(2)) # A parameter, given as Decay(half_life=ss.years(5))
            level = ss.FloatArr(default=1.0) # A value for each agent

            def step(self):
                self.level[:] *= 0.5**(self.t.dt/self.pars.half_life)

        # The same, with the parameters defined in __init__()
        class Decay(ss.Intervention):
            def __init__(self, **kwargs):
                super().__init__()
                self.define_pars(half_life=ss.years(2))
                self.update_pars(**kwargs)
                self.define_states(ss.FloatArr('level', default=1.0))
        ```
    """
    _state_aliases = {} # Default state aliases; defined on the class so it always exists, e.g. while unpickling. See define_aliases()
    has_product = False # Whether it delivers a product: see ss.Intervention
    allow_modifiers = () # For a module written as code: which of its parameters (and for a disease, of rel_sus and rel_trans) its own code reads with current_value(), so ss.modify() of them is applied, including for some people only
    _serials = itertools.count() # Which module each is, kept by its copies: see __init__()

    def __init_subclass__(cls, **kwargs):
        """ A module that defines `update_results()` is one written for Starsim 3 or an early v4: say how to split it, rather than never call it """
        super().__init_subclass__(**kwargs)
        if 'update_results' in cls.__dict__:
            errormsg = (f'{cls.__name__} defines update_results(), which Starsim no longer calls: split it into record_states(), for the results that are the state at '
                        f'the start of the step (e.g. a count of a state), and record_events(), for those about what happened in the step (e.g. new infections); see ss.loop.Loop, and https://docs.starsim.org/v3v4 for help migrating from Starsim 3')
            raise TypeError(errormsg)
        renamed = {k: v for k, v in dict(record_stocks='record_states', record_flows='record_events').items() if k in cls.__dict__}
        if renamed:
            raise TypeError(f'{cls.__name__} defines {sc.strjoin(f"{k}()" for k in renamed)}, which Starsim no longer calls: rename to {sc.strjoin(f"{v}()" for v in renamed.values())}')
        return

    def __init__(self, name=None, label=None, pars=None, **kwargs):
        # Housekeeping
        self._locked_attrs = ['pars', 't', 'sim', 'dists', 'results'] # Define key attributes that shouldn't be overwritten
        self.setattribute('_serial', next(Module._serials)) # Which module this is, kept by its copies (e.g. the sim's), so what was made of the module given is found for the sim's (see ss.expressions.Clock.sets_of())
        self._collect_required() # First, collect methods marked as required on creation

        # Handle parameters
        self.pars = ss.Pars() # Usually populated via self.define_pars()
        self._name_source = 'class' # Where its name came from: 'class' (the default, see default_name), 'variable' (the variable it was assigned to; see ss.Sim._name_modules()), or 'given' (name=, or a key)
        self._name_where = None # For a name from a variable, the function whose variable it is (see name_origin())
        self._implicit = False # Whether it was made or named by the sim (from the transitions it was given, or its states), so its results are at the top of the sim's (e.g. sim.results.I) even beside other diseases
        self._assigned, self._assigned_in = ss.utils.assigned_name(self) # The variable it was first assigned to, e.g. campaign = ss.vaccinate(...), and the function it is in
        self.set_metadata(name, label) # Usually reset as part of self.update_pars()
        class_pars, class_arrs = class_parts(type(self))
        if isinstance(self, ss.Disease) and not isinstance(self, ss.AgentDisease): # A disease's parameters are part of what it declares, if it declares states and transitions: see ss.Disease.build_graph()
            class_pars = {}
        kwargs = sc.mergedicts(pars, kwargs)
        given = {key:kwargs.pop(key) for key in list(kwargs) if key in class_pars}
        unknown = [key for key in kwargs if key not in ss.Timeline.keywords]
        if unknown:
            valid = list(class_pars) + module_args + ss.Timeline.time_args
            how = '' if class_pars else '; parameters defined in __init__() are given to update_pars(), after define_pars()' # The usual slip without parameters in the class body: super().__init__(**kwargs)
            errormsg = (f'{type(self).__name__}() got {"an unexpected keyword" if len(unknown) == 1 else "unexpected keywords"} {sc.strjoin(f"{key}=" for key in unknown)}{ss.expressions.did_you_mean(unknown[0], list(class_pars))}\n'
                        f'     Valid keywords are {sc.strjoin(valid)}{how} (see the migrate-starsim3 skill for keywords that have changed)')
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
        self._early_results = [] # Results defined in __init__(): see define_results()
        self.define_class_parts(class_pars, class_arrs, given)
        return

    def define_class_parts(self, pars, arrs, given=None):
        """
        Define the parameters, arrays, and distributions declared in the class body (see `class_parts()`), each instance with its own copy, and apply the values given

        Each parameter is also an attribute (`mod.half_life`), the instance's own `ss.par`, which holds its value, as for a
        declared disease; setting it (`mod.half_life = 3`) sets the parameter. The module's code reads the value from `self.pars`.

        Args:
            pars (dict): the parameters (`ss.par`), by name
            arrs (dict): the arrays (`ss.Arr`) and distributions (`ss.Dist`), by name
            given (dict): the values given for the parameters, by name
        """
        if pars:
            self.define_pars(**{key:sc.dcp(p) for key,p in pars.items()}) # Each kept in pars.bounded(), which mod.<key> reads (see ss.par.__get__())
            self.setattribute('_class_pars', set(pars)) # Their defaults stand, even if a base class's __init__() defines them after this (see define_pars())
            if given:
                self.pars.update(given)
        states = []
        for key,arr in arrs.items():
            if isinstance(arr, ss.Dist): # Its own copy, found by the sim with the module's other distributions
                self.setattribute(key, sc.dcp(arr))
                continue
            if arr.name not in [None, key]:
                errormsg = f'The array "{key}" of {type(self).__name__} is named "{arr.name}"; an array in a class body takes the name it is assigned to, so leave out the name'
                raise ValueError(errormsg)
            arr = sc.dcp(arr)
            arr.name = key
            states.append(arr)
        if states:
            self.define_states(*states, overwrite=True) # Replacing only the class's own, which every instance copies
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
            if attr == 't': # An easy slip in an analyzer, e.g. self.t = []
                errormsg = f'"t" is reserved for the module\'s Timeline; store your own time data under another name, e.g. self.tvec'
                raise AttributeError(errormsg)
            errormsg = f'Cannot modify attribute "{attr}"; locked attributes are {sc.strjoin(self._locked_attrs)}.\n'
            errormsg += 'If you really mean to do this, use module.setattribute() or set module._lock_attrs = False'
            try:
                if isinstance(getattr(self, attr),ss.Arr) and len(getattr(self, attr)) == len(value):
                    errormsg += f'\nOften this error can be resolved by replacing `self.{attr} = x` with `self.{attr}[:] = x`'
            except Exception:
                pass
            raise AttributeError(errormsg)
        elif isinstance(self.__dict__.get(attr), ss.par) and attr in self.pars: # A declared parameter (e.g. sir.beta = 0.1), the same object as sir.pars.beta
            self.pars[attr] = value
        elif isinstance(getattr(type(self), attr, None), ss.par) and attr in self.pars.bounded(): # Declared in the class body of a module written as code (see define_class_parts())
            self.pars.update({attr:value})
        else:
            super().__setattr__(attr, value)
        return

    def setattribute(self, attr, value):
        """ Method for setting an attribute that does not perform checking against immutable attributes """
        return super().__setattr__(attr, value)

    def check_not_given(self, action):
        """ Raise if this module was given to a sim, which runs its own copy, so that `action` on this one would never be seen """
        if getattr(self, '_given', False):
            kind = ss.simpars.module_kind(self)
            errormsg = (f'{action} on "{self.name}", which was given to a sim: the sim runs its own copy, so it would never be used. '
                        f'Do it before making the sim, or on the sim\'s copy, sim.{kind}.{self.name}, after sim.init()')
            raise RuntimeError(errormsg)
        return

    def brief(self, output=False, extra=''):
        """
        Show a brief representation of the module, with any extra text after its name (e.g. a network's number of edges); used by __repr__

        Args:
            output (bool): whether to return the text rather than print it
            extra (str): text to show after the name
        """
        name = self.name # e.g. 'sir'
        label = self.label # e.g. 'My SIR'
        class_name = self.__class__.__name__ # e.g. 'SIR'
        graph = getattr(self, 'graph', None)
        states = graph.states if isinstance(graph, ss.graph.Graph) else self.state_dict # Its declared states (S, I, R), or the arrays of one written as code
        parts = [
            name if name != class_name.lower() else '',
            f'"{label}"' if (label.lower() != name and label != class_name) else '',
            extra,
            f'pars={sc.strjoin(self.pars.keys())}' if len(self.pars) else '',
            f'states={sc.strjoin(states.keys())}' if len(states) else '',
        ]
        out = f'{class_name}({"; ".join(part for part in parts if part)})'
        return out if output else print(out)

    def __repr__(self):
        """ Show the object including parameters and states """
        try: # Default representation
            return self.brief(output=True)
        except Exception: # If for any reason that fails, return the full repr
            return sc.prepr(self)

    @property
    def _debug_name(self):
        """ The module name and class shown as a string, used in debugging/error messages """
        out = f'"{self.name}" ({type(self)})'
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
                if inspect.isfunction(method) and getattr(method, '_call_required', False):
                    key = f'{cls.__qualname__}.{attr}'
                    reqs[key] = 0 # Meaning it's been called 0 times
        self._call_required = reqs
        return reqs

    def check_method_calls(self):
        """
        The required methods that were not called (see `ss.required()`); called by `sim.run()`
        """
        missing = [key for key, called in self._call_required.items() if not called]
        return missing

    @required()
    def set_metadata(self, name=None, label=None):
        """
        Set metadata for the module

        Args:
            name (str): the name of the module; if None, the one it has, else the lowercase name of its class
            label (str): the label of the module; if None, its name
        """
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
        if name is not None or (self.name != cls_lower and getattr(self, '_name_source', 'class') == 'class'): # The user (or the class itself, as ss.Pregnancy does) chose the name, so never rename it
            self._name_source = 'given'
        default_label = self.name if self.name != cls_lower else cls_name
        if label is None and getattr(self, 'label', None) in [old_name, cls_name]: # The existing label is a default, so update it to match the new name
            label = default_label
        self.label = self._reconcile('label', label, default_label)
        return

    @property
    def default_name(self):
        """
        The name the module has if it is not given one and is not assigned to a variable: the lower-case name of its class (`'sir'`
        for `ss.SIR`, `'randomnet'` for `ss.RandomNet`), or for a plain `ss.Disease`, the initials of its states (`'sir'`)

        A module can always be found by this name too, if no other module of its kind has it: `sim.diseases.sir` and `sim.results.sir`
        find the only `ss.SIR` even if it is called `measles` (see `ss.ndict`).
        """
        if type(self) is ss.Disease and getattr(self, 'graph', None) is not None:
            return ''.join(s.name[0].lower() for s in self.graph.people_states) or 'disease'
        return type(self).__name__.lower()

    def name_origin(self):
        """ Where the module's name came from, for messages: e.g. 'named from the variable `m` in `make_sim()`', or 'the default for ss.SIR' """
        source = getattr(self, '_name_source', 'class')
        if source == 'variable':
            where = self._name_where
            return f'named from the variable `{self.name}`' + (f' in `{where}()`' if where and not where.startswith('<') else '')
        elif source == 'given':
            return 'given'
        cls = type(self)
        if cls is ss.Disease:
            return 'the default, from its states'
        prefix = 'ss.' if cls.__module__.startswith('starsim.') and not cls.__module__.startswith('starsim.library') else ''
        return f'the default for {prefix}{cls.__name__}'

    def _set_name(self, name, source='given', where=None):
        """ Rename the module, keeping its label if one was given (the user gives `name=` instead); `source` and `where` say where the name came from (see `name_origin()`) """
        self.set_metadata(name=name)
        self._name_source = source
        self._name_where = where
        if isinstance(getattr(self, 'results', None), ss.Results):
            self.results.setattribute('_module', self.label)
        return

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create a module instance by name

        Args:
            name (str): A string with the name of the module class in lower case, e.g. 'sir'
        """
        subcls = find_modules(flat=True).get(name)
        if subcls is not None and issubclass(subcls, cls):
            return subcls(*args, **kwargs)
        else:
            raise KeyError(f'Module "{name}" did not match any known Starsim modules')

    @classmethod
    def from_func(cls, func):
        """
        Create a module from a function

        The function takes the sim as its only argument, and becomes the module's `step()`, so it is called on every
        timestep and does its own time gating (e.g. `if sim.now == 2015:`); the module is named after the function. A function
        given to `interventions=`, `analyzers=`, or to the sim is converted this way automatically.

        Keyword arguments of the function with defaults are the module's parameters, passed to it at each call; one given as
        `ss.par` with bounds can be fitted (`sim.fit()`), sampled (`sim.sensitivity()`), and set by name (`ss.Sim(pars=...)`)
        as any other, e.g. to fit a value set in code (see the last example). Such a module has no states or results of its
        own: for those, write a subclass.

        Args:
            func (callable): a function taking the sim, and optionally keyword arguments with defaults (its parameters)

        Returns:
            mod (Module): an instance of `cls` whose `step()` calls `func(sim)`

        Example:
            ```python
            import starsim as ss

            def knockdown(sim): # Halve transmission for one year starting in 2015
                if sim.now == 2015:
                    sim.diseases.sis.rel_trans[:] *= 0.5

            # Explicit construction ...
            intv = ss.Intervention.from_func(knockdown)
            sim = ss.Sim(diseases='sis', networks='random', interventions=intv)

            # ... or equivalently, let the sim convert it automatically:
            sim = ss.Sim(diseases='sis', networks='random', interventions=knockdown)

            # A value set in code, fitted: the fraction of transmission left in 2015
            def knockdown(sim, frac=ss.par(0.5, bounds=[0.1, 1])):
                if sim.now == 2015:
                    sim.diseases.sis.rel_trans[:] *= frac
            ```
        """
        def step(mod):
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

        A parameter declared in the class body (`ss.par`) is left as it is there, so that a subclass can change the
        default of a parameter its base defines here (e.g. `class MyNet(ss.RandomNet): n_contacts = ss.par(4)`).

        Args:
            inherit (bool): whether to merge with the parameters already defined, rather than replace them
            **kwargs (dict): the parameters
        """
        own = self.__dict__.get('_class_pars')
        keep = {} # A default in the class body (e.g. of a subclass, n_contacts = ss.par(4)), or the value given for it, stands over one defined by a base class's __init__()
        if own and inherit:
            keep = {k:sc.dcp(v) for k,v in self.pars.with_bounded().items() if k in own and k in kwargs} # Copies (the base's default would set the value of the ss.par kept), applied over the base's default, as update_pars() would apply it, so that a number sets a distribution's value or takes the default's unit
        if inherit: # Merge with existing
            declared = '_declared' in vars(self) or getattr(self, 'graph', None) is not None # Declared, or being declared (see ss.Disease.define_transitions()), without building it
            given = {k:v for k,v in kwargs.items() if isinstance(v, ss.par)} if not declared else {} # In a module written as code, a parameter given as ss.par (e.g. with bounds, to be fitted) is kept as one, and its code uses the value, as for update_pars()
            self.pars.update(**{k:(v.value if k in given else v) for k,v in kwargs.items()}, create=True)
            if given:
                self.pars.update(**given)
            self.pars.forget_bare(kwargs) # A module's own defaults are in the unit it means
            if keep:
                self.pars.update(**keep)
        else: # Or overwrite
            self.pars = ss.Pars(**kwargs)
        return self.pars

    def _value(self, p):
        """
        The current value of a parameter, as arithmetic on it reads it once the sim is initialized (see `ss.expressions.run_value()`)

        In the sim's unit: a typed value as a number (a duration in its units, a rate per unit), a value that varies in time
        at the current time, a value by level of one dimension as one value for each level (of several, one for each group of
        the sim), and the modifiers active on a declared parameter applied (if they differ between groups or agents, one value
        for each). Before the module is in a sim, the value it was given.

        Args:
            p (par): the parameter
        """
        raw = p.raw
        if self.sim is None or getattr(self.sim, 'unit_base', None) is None: # Not in a sim yet, or its unit is not yet known
            return raw
        engine = getattr(self, 'engine', None)
        if engine is not None: # A declared parameter: as the engine reads it, with its modifiers, unless a value by level with none
            if not isinstance(raw, (ss.by, ss.parameters.by_func)) or self.sim.modifiers.get((self.name, 'par', p.name)):
                return engine.value(p)
        return ss.parameters.current_value(raw, self.sim, cache=self.__dict__.setdefault('_converted', {}))

    def current_value(self, key, uids=None):
        """
        A parameter (`key` is its name) with the modifiers active on it (`ss.modify('tx.pars.p_treat', ...)`), in its own type: one value, or one for each agent if they differ between agents

        A module written as code reads a parameter from `self.pars` with the modifiers applied, which is one value. To apply a modifier that
        differs between people (e.g. with `who=`), its code reads the parameter with `current_value()`, and says so in its class, `allow_modifiers = ('p_treat',)`.

        Args:
            key (str): the name of the parameter
            uids (uids): the agents to give the values of (default: one value, or an array of every agent, by UID, if they differ between agents)

        Example:
            ```python
            class Treat(ss.Intervention):
                allow_modifiers = ('p_treat',) # ss.modify('treat.pars.p_treat', multiply=0.5, who=...) applies to those people only
                def step(self):
                    uids = self.sim.diseases.sis.infected.uids
                    treated = uids[np.random.rand(len(uids)) < self.current_value('p_treat', uids)] # ...
            ```
        """
        engine = getattr(self, 'engine', None)
        if engine is None: # Written as code, with no engine of its own: one that reads the modifiers, kept so that a modifier that holds (hold=, until=) is remembered between steps
            engine = self.__dict__.get('_effects')
            if engine is None or engine.sim is not self.sim:
                engine = ss.solvers.Effects(self, self.sim)
                self.setattribute('_effects', engine)
        return engine.current_par(key, uids)

    # Warning: do not try to use a decorator with this function, that will break argument passing!
    def update_pars(self, pars=None, **kwargs):
        """
        Pull out recognized parameters and update them; raises on unrecognized keys

        Args:
            pars (dict): the parameters to update
            **kwargs (dict): more parameters, merged into `pars`
        """
        # Merge pars and kwargs
        if isinstance(pars, ss.Pars): # E.g. a package's full set of defaults for the module: those given as ss.par keep their bounds
            pars = pars.with_bounded()
        pars = sc.mergedicts(pars, kwargs)
        ss.v3_deprecations.check(type(self), '__init__', [key for key in pars if key not in self.pars]) # A keyword of Starsim 3's that it no longer takes, saying what replaces it

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
            remaining = sorted(remaining)
            valid = list(self.pars.keys()) + module_args + ss.Timeline.time_args
            errormsg = f'{len(remaining)} unrecognized argument{"s" if len(remaining) > 1 else ""} for {self.name}: {sc.strjoin(remaining)}{ss.expressions.did_you_mean(remaining[0], list(self.pars.keys()))}\n     Valid arguments are: {sc.strjoin(valid)}'
            if type(self) is ss.Network:
                errormsg += '; its edges are given as p1=, p2=, weight=, or edges=dict(...) for any columns'
            errormsg += ss.v3_deprecations.hint(keys=remaining, call=f"ss.{type(self).__name__}")
            raise ValueError(errormsg)
        return

    def define_states(self, *args, reset=False, lock=True, overwrite=False, **kwargs):
        """
        Define states of the module, each an attribute of the same name; other keyword arguments are aliases (see `define_aliases()`)

        Args:
            args (states): list of states to add
            reset (bool/str/list): if True, remove all existing states and aliases; if a name or list of names, remove only those
            lock (bool): if True, prevent the state attributes from being overwritten after definition
            overwrite (bool): if True, replace any existing state or alias of the same name, rather than raising
            kwargs (dict): passed to `define_aliases()`

        Examples:
            ```python
            self.define_states(ss.BoolState('exposed'), reset='ti_infected') # Add a state and remove an inherited one
            self.define_states(ss.BoolState('infectious'), overwrite=True) # Replace an inherited state or alias
            ```
        """
        # Optionally remove existing states and aliases: all of them (reset=True), or just the named
        # ones (note: does not remove them from the people object or others if already added)
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
            if self.pre_initialized: # The module's arrays were added to the people (with their results) in init_pre(), so a later one would never have a value per agent
                errormsg = f'"{state.name}" is defined after "{self.name}" was added to the sim (e.g. in init_post()), so the people would not have it: define arrays in __init__() (or in the class body) instead'
                raise RuntimeError(errormsg)

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

        An alias is used only if the attribute isn't otherwise defined, so a subclass overrides it by defining a state of
        the same name (e.g. `ss.AgentDisease` treats everyone infected as infectious this way).

        Each alias is the name of another state, or a callable taking the module, which is recomputed on each access (read
        only) and also generates a result of its name, as an `ss.BoolState` does (a string alias doesn't, since its state is
        counted already). A lambda can't be saved with plain `pickle` (`sc.save()` and `ss.MultiSim` use `dill`, so are fine).

        Args:
            overwrite (bool): if True, replace any existing state or alias of the same name, rather than raising
            kwargs (dict): mapping of alias name to either the name of a state, or a callable

        Examples:
            ```python
            self.define_aliases(infectious='infected') # Everyone infected is infectious
            self.define_aliases(infected=lambda self: self.exposed | self.infectious) # Both E and I are infected
            self.define_states(ss.BoolState('exposed'), infectious='infected') # Aliases can also be defined alongside states
            ```
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
        """
        Add results to the module; in `__init__()`, they are kept, and added once the sim's timeline is known

        Args:
            *args (`ss.Result`/list/dict): the results; a list or dict is passed to `ss.Result()`
        """
        if not self.t.initialized:
            self._early_results += args
            return
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
                result._init_groups(groups.strats, [strat.name for strat in groups.strats], (self.t.npts, groups.n_groups))

            # Add the result to the dict of results; does automatic checking
            if result.name in self.results and result.name in auto_names:
                msg = f'"{self.name}": Another result named "{result.name}" already exists because a result was automatically created for the state "{result.name}" (the number in it).'
                raise ValueError(msg)
            self.results += result
        return

    def agent_reasons(self, sim):
        """
        What needs agents in this module, from what it is rather than by running it, for choosing the solver (see `ss.solvers.rules`):
        each a sentence, which stops a compartmental solver that was asked for, or a `ss.modules.Preference`, which does not

        Here, for a module written as code: `compartmental = True` or `False` on it (or its class) says whether it works on
        groups; without it, keeping a value for each person (`define_states()`) needs agents. A subclass can say more.

        Args:
            sim (`ss.Sim`): the sim, with its groups and unit worked out, but no solver yet
        """
        own = [f'"{self.name}" has a solver of its own, solver={self.solver!r}, which is for agents only'] if isinstance(self, ss.Disease) and getattr(self, 'solver', None) is not None else [] # e.g. ss.Disease(..., solver=ss.solvers.abm(step='wholestep'))
        flag = getattr(self, 'compartmental', None)
        if flag or getattr(self, 'graph', None) is not None:
            return own
        elif flag is False:
            return own + [f'"{self.name}" ({type(self).__name__}) has compartmental = False, so runs only as agents']
        elif self.state_list:
            return own + [f'"{self.name}" ({type(self).__name__}) keeps a value for each person ({sc.strjoin(s.name for s in self.state_list[:3])}), which only agents have; if its code works on groups, set compartmental = True on it']
        return own

    @required()
    def init_pre(self, sim, force=False):
        """
        Perform initialization steps, once, as part of initializing a sim; `initialized` stays False until `init_post()`, after the distributions are initialized

        Args:
            sim (`ss.Sim`): the sim
            force (bool): whether to run again if already done
        """
        if force or not self.pre_initialized:
            self.setattribute('sim', sim) # Link back to the sim object
            ss.dists.link_dists(self, sim, skip=[ss.Sim, ss.Module]) # Link the distributions to sim and module, skipping any nested sim or module instances
            self.t.init(sim=self.sim) # Initialize time vector
            self.link_timepars() # Add module dt to the timepars
            self.check_bare_units(sim)
            if isinstance(self.pars, ss.Pars) and getattr(self, 'graph', None) is None: # A declared module checks its own, against its graph (see ss.Disease.init_pre())
                self.pars.check_added(self.name) # A parameter set but never defined, e.g. renamed
            if isinstance(self.pars, ss.Pars):
                self.pars.bind(self) # Arithmetic on a value that varies is on its value now, once the sim is initialized
            if self.name not in sim.pars: # Add a new key, which is allowed even if the sim's parameters are locked
                sim.pars.update({self.name: self.pars}, create=True)
            elif isinstance(sim.pars[self.name], ss.Pars): # Not a setting of the sim with the same name, e.g. a module called "branching", which is left as it is
                sim.pars[self.name] = self.pars
            sim.results[self.name] = self.results
            self.results.setattribute('_default', self.default_name) # So that sim.results.sir finds them by the default name too
            self.results.setattribute('_origin', self.name_origin())
            sim.people.add_module(self) # Connect the states to the people
            self.init_results()
            self.define_results(*self._early_results)
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
        Link the module's time parameters to its `dt`, so that e.g. `self.pars.death_rate.to_prob()` needs no argument

        Args:
            force (bool): whether to relink parameters that are already linked
        """
        timepars = sc.search(self, type=ss.TimePar, skip=dict(keys=ss.utils.skipkeys(['sim', 'module', 't', 'owner'] + list(getattr(self, 'timepar_skip', []))), subclasses=ss.dists.no_dists())) # Skip the timeline, else the module's own dt would be linked to itself

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

        By default, the number in each `ss.BoolState` (e.g. `susceptible`) and derived state is a result; an `ss.BoolArr` has none.
        """
        self.results.timevec = self.t.timevec # Store the timevec in the results for plotting; not a Result so don't use ss.ndict.append()
        results = sc.autolist()
        for state in self.auto_state_list:
            results += ss.Result(state.name, dtype=int, scale=True, label=state.label, kind='stock')
        for name in self.derived_state_names: # A derived state has no state object, but is counted the same way
            results += ss.Result(name, dtype=int, scale=True, label=name.capitalize(), kind='stock')
        self.define_results(*results)
        return

    @required()
    def init_post(self):
        """ Initialize the values of the states, and link its schedules (`ss.schedule`) to its timeline; the last step of initialization """
        for state in self.state_list:
            if not state.initialized:
                state.init_vals()
        for obj in [*self.__dict__.values(), *self.pars.values()]:
            if isinstance(obj, ss.schedule):
                obj.init(self)
        self.initialized = True
        return

    def init_mock(self, n_agents=100, dur=10):
        """
        Initialize with a mock simulation -- for debugging purposes only

        Args:
            n_agents (int): the number of agents in the mock sim
            dur (float): the duration of the mock sim
        """
        sim = ss.debugtools.mock_sim(n_agents=n_agents, dur=dur)
        self.init_pre(sim)
        for state in self.state_list: # Manually link the people
            state.people = self.sim.people
        obj = [self.pars, self.state_list]
        dists = sc.search(obj, type=ss.Dist)
        for i,dist in dists.enumvals():
            dist.trace = f'dist_{i}'
        ss.dists.link_dists(obj=obj, sim=sim, module=self, skip=[ss.Sim, ss.Module], init=True)
        self.init_post()
        return self

    @property
    def state_list(self):
        """
        A flat list of the module's states (`ss.Arr` objects) in its attributes; a module that keeps states elsewhere (e.g. in a list) overrides this
        """
        out = [val for val in self.__dict__.values() if isinstance(val, ss.Arr)]
        return out

    @property
    def state_dict(self):
        """
        A flat dict of the module's states by name; raises if two have the same name
        """
        return ss.utils.nlist_to_dict(self.state_list)

    @property
    def auto_state_list(self):
        """
        The `ss.BoolState`s added with `define_states()`, each of which has a result of the same name, the number in it (e.g. `susceptible`)
        """
        return self._auto_states[:]

    @property
    def derived_state_names(self):
        """
        Names of the aliases defined by a callable (see `define_aliases()`), each of which has a result of its own, as an `ss.BoolState` does
        """
        return [key for key,alias in self._state_aliases.items() if callable(alias)]

    def match_time_inds(self, inds=None):
        """
        Find the nearest matching sim time indices for the current module

        Args:
            inds (array): indices to return if the module and sim have the same number of timesteps; if None, all of them
        """
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

    def on_birth(self, parents, children):
        """
        What to do when people are born (e.g. to give the children traits from their parents); by default, nothing

        Called for every module that defines it, wherever children are born (by `ss.Births`, at a delivery of
        `ss.Pregnancy`, or by a declared birth transition), once they are people.

        Args:
            parents (`ss.uids`): the parent of each child (so a parent of several children appears once for each), or -1 for a child with no parent (e.g. born by `ss.birth()` without `mothers=True`)
            children (`ss.uids`): the children
        """
        return

    def on_death(self, uids):
        """
        What to do when agents die or leave; by default, nothing (see `ss.Disease.on_death()` for diseases)

        Called by `people.step_die()` for every module that defines it, once the deaths of the step are resolved, each
        to one cause (`people.cause_of_death`; `people.died_of(module)` gives those of a module), and before the results
        are recorded. It isn't called in a step with no deaths.

        Args:
            uids (`ss.uids`): the agents who died or left in this step
        """
        return

    @required()
    def finish_step(self):
        """ Define what should happen at the end of the step, after the results are recorded """
        return

    def advance(self):
        """ For a module with a timeline of its own: move its time to the step about to be taken (the sim moves the time of the modules on its own timeline: see `ss.Sim.advance()`) """
        self.t.ti += 1
        return

    def record_states(self, names=None):
        """
        Record the results that are the state at the start of the step (stocks), at its index; by default, the count of each state

        Called at the start of every step, before any module's step (see `ss.loop.Loop`), so the results at a time point
        are the state there: e.g. `self.results.n_on_art[self.ti] = self.on_art.sum()`. A result about what happens
        in the step (a flow, or one that needs a flow) goes in `record_events()` instead.

        Args:
            names (list): the states to count (default: every `ss.BoolState`, and derived state)
        """
        weights = ss.results.agent_weights(self.sim) # Agents of different weights (people.scale) are each counted by their weight, as in the result alive
        if names is None:
            names = [state.name for state in self.auto_state_list] + self.derived_state_names
        if weights is None and len(names) and self._count_states(names): # Counted together, in one pass over the agents
            return
        for name in names:
            res = self.results[name]
            if weights is None:
                res[self.ti] = self[name].sum()
            else:
                ss.results.as_float(res)[self.ti] = weights[self[name].uids].sum()
        return

    def record_events(self):
        """
        Record the results about what happened in the step (flows), at its index: e.g. `self.results.doses[self.ti] = self.n_doses`

        Called at the end of every step, after every module's step and the deaths (see `ss.loop.Loop`). The state then is
        that of the next time point, so a result that also needs a stock reads it from its result, recorded at the
        start of the step (e.g. `self.sim.results.alive[self.ti]`), or is worked out over the whole run in
        `finalize_results()`.
        """
        return

    def _count_states(self, names):
        """ Record the number of active agents in each of some boolean states, as `record_states()` does one by one, in one pass by UID (as `people.record_states()` does); False if they can't be counted so """
        ppl = self.sim.people
        if not isinstance(getattr(self.sim, 'solver', None), ss.solvers.abm) or not isinstance(getattr(ppl, 'alive', None), ss.BoolArr): # Not with compartments, or in a mock sim
            return False
        states = [self.__dict__.get(name) for name in names]
        if not all(isinstance(state, ss.BoolArr) and state.raw.ndim == 1 and state.raw.flags.c_contiguous for state in states):
            return False # E.g. a derived state, or one with no memory of its own yet
        alive = ppl.alive.raw[:ppl.uid.len_used]
        counts = ss.solvers.numba.passes.count_alive(tuple(state.raw for state in states), alive) # Those alive in each, then the number alive...
        if counts[-1] != len(ppl.auids): # ...which should be the active agents, as at the start of a step
            return False
        ti = self.ti
        for name,count in zip(names, counts):
            self.results[name][ti] = count
        return True

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
                res._apply_scale(scale)
        return

    def to_json(self):
        """ The module as a reference by its type and name, for `sc.jsonify()`: e.g. a module among the parameters in `sim.to_json()`, or in a parameter of another module """
        return dict(type=type(self).__name__, name=self.name)

    def shrink(self):
        """ Shrink the size of the module for saving to disk """
        shrunk = ss.utils.shrink()
        self.setattribute('sim', shrunk) # Use setattribute since locked otherwise
        self.setattribute('dists', shrunk)
        self.__dict__.pop('_effects', None) # It holds the sim
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
        flows = [k for k,res in results.items() if res.kind == 'flow' and '.' not in k] # Not the flows to each destination of a transition with several
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
