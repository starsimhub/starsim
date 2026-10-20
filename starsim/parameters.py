"""
Set parameters
"""
from numbers import Number
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import datetime as dt
import itertools
from .expressions import as_expr, evaluate, Expr, par


# Define classes to not descend into further -- based on sciris.sc_nested
atomic_classes = (str, Number, list, np.ndarray, pd.Series, pd.DataFrame, type(None), dt.date)


class Pars(sc.objdict):
    """
    Dict-like container of parameters

    Acts like an `sc.objdict()`, except that adding new keys are disallowed by
    default, and auto-updates known types.

    Args:
        pars (dict): initial parameters to populate the object with
        kwargs: additional key-value pairs merged with pars
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

    _locked = False # Whether new keys can be added; set by lock()
    _bare = None # The parameters set to a bare number, with the unit it is read in: see bare_units()
    _bounded = None # For a module written as code: the parameters given as ss.par (e.g. with bounds, to be fitted), by name; see bounded()

    def bounded(self):
        """ The parameters of a module written as code that were given as `ss.par` (e.g. `beta=ss.par(ss.peryear(0.1), bounds=[0.05, 1])`), by name; their values are used as any other's """
        return dict(self._bounded or {})

    def note_bare(self, key, old, new):
        """ Note that a parameter with a unit (or a distribution of values with one) was set to a bare number, which keeps that unit """
        typed = [old] if isinstance(old, ss.TimePar) else [v for v in old.pars.values() if isinstance(v, ss.TimePar)]
        if typed and typed[0].base is not None and np.any(np.asarray(sc.tolist(new), dtype=float) != 0): # Zero is the same in any unit
            object.__setattr__(self, '_bare', (self._bare or {}) | {key:typed[0].base})
        return

    def forget_bare(self, keys):
        """ Forget the notes of `note_bare()` for these keys: given again, or a module's own defaults """
        if self._bare:
            object.__setattr__(self, '_bare', {k:v for k,v in self._bare.items() if k not in keys})
        return

    def bare_units(self, unit, label):
        """ Warn about each parameter set to a bare number whose unit (that of its default) is not the sim's, e.g. ss.MFNet(dur=5) in a sim in days """
        for key,base in (self._bare or {}).items():
            if base != unit:
                runs = f'runs in {unit}' if unit else 'counts time in timesteps'
                ss.warn(f'{label}: {key} was given as a number with no unit, so it is read in {base} (the unit of its default), but the sim {runs}. '
                        f'Give it a unit to say which is meant, e.g. ss.{base}(...)' + (f' or ss.{unit}(...).' if unit else '.'))
        return

    def lock(self, locked=True):
        """
        Lock the parameters, so that setting a key that doesn't exist raises an error

        Parameters are locked when the sim is initialized, so that e.g. a typo in
        `sim.pars['rand_sed'] = 2` raises an error rather than silently doing nothing.
        Use `pars.lock(False)` to unlock, or `pars.update(create=True)` to add a new key.
        """
        object.__setattr__(self, '_locked', locked) # Set as an attribute, not a key
        return self

    def __setitem__(self, key, value):
        if self._locked and key not in self:
            errormsg = f'Key "{key}" not found; available keys are {list(self.keys())}. To add a new parameter, use pars.update({key}=..., create=True).'
            raise sc.KeyNotFoundError(errormsg)
        current = self.get(key)
        if isinstance(current, ss.par): # A new value for a declared parameter (e.g. sir.pars.beta = 0.1) is taken by it, keeping its name and bounds, since the module's transitions use this same object
            if isinstance(value, ss.par):
                current.merge(value)
            else:
                current.value = value
            return
        return super().__setitem__(key, value)

    def __deepcopy__(self, memo):
        """ Copy the keys directly, since by default a copy sets the attributes first, so locked parameters couldn't be copied """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key,value in self.items():
            dict.__setitem__(new, key, sc.dcp(value, memo))
        new.__dict__.update(sc.dcp(self.__dict__, memo))
        return new

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
        self.forget_bare(pars) # Noted again below if still a bare number
        for key,new in pars.items():
            if isinstance(new, ss.par) and key in self.keys() and not isinstance(self[key], ss.par): # A parameter of a module written as code, given with bounds (say): its code uses the value, and the parameter is kept so it can be fitted
                new.name = new.name or key
                object.__setattr__(self, '_bounded', (self._bounded or {}) | {key:new})
                new = new.value
            if key not in self.keys(): # It's a new parameter and create=True: update directly
                super().__setitem__(key, new) # Skip the check for locked parameters
            else:
                old = self[key] # Get the existing object we're about to update
                if isinstance(old, atomic_classes + (ss.par,)): # It's a number, string, etc, or a declared parameter: update directly
                    self[key] = new

                elif isinstance(old, Pars): # It's a Pars object: update recursively
                    old.update(new, create=create)
                elif isinstance(old, ss.ndict): # Update module containers -- main use case
                    self._update_ndict(key, old, new)
                elif isinstance(old, ss.Module): # Update modules -- rare
                    self._update_module(old, new)
                elif isinstance(old, ss.TimePar):
                    self._update_timepar(key, old, new)
                elif isinstance(old, ss.Dist): # Update a distribution
                    self._update_dist(key, old, new)
                elif callable(old): # It's a function: update directly
                    self[key] = new
                elif isinstance(old, dict):
                    self[key] = new # Take dictionaries directly, without warning the user
                elif isinstance(old, (ss.dur, ss.Rate)):
                    self[key] = new
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

    def _update_module(self, old, new):
        """ Update a Module object in the parameters, e.g. sim.pars.diseases.sir """
        if isinstance(new, dict):
            old.pars.update(new) # e.g. pars = {'dur_inf': 6}
        else:
            errormsg = f'Cannot update a module with {type(new)}: must be a dict to set new parameters'
            raise TypeError(errormsg)
        return

    def _update_timepar(self, key, old, new):
        """ Update a time parameter """

        # It's a TimePar, e.g. dur_inf = ss.years(6); use directly
        if isinstance(new, ss.TimePar):
            self[key] = new

        # It's a dataframe, allow the update -- used for demographics
        elif isinstance(new, (pd.Series, pd.DataFrame)):
            self[key] = new # TODO: add validation or convert to timepar

        # It's a single number, e.g. dur_inf = 6; set parameters
        elif sc.isnumber(new):
            old.value = new
            self.note_bare(key, old, new)

        # It's array-like; set the value, but convert first to handle lists etc
        elif sc.checktype(new, 'arraylike'):
            old.value = sc.toarray(new)
            self.note_bare(key, old, new)

        # It's a dict, e.g. beta for different networks, just set it directly and trust the user
        elif isinstance(new, dict):
            # for key,val in new.items(): # TODO: should we include validation like this?
            #     if sc.isnumber(val):
            #         new[key] = old.__class__(val)
            self[key] = new

        # It's a distribution, e.g. a duration that varies; replace it
        elif isinstance(new, ss.Dist):
            self[key] = new

        # Give up
        else:
            errormsg = f'Updating timepar {old} from {type(old)} to {type(new)} is not supported'
            raise TypeError(errormsg)

        return

    def _update_dist(self, key, old, new):
        """ Update a Dist object in the parameters, e.g. sim.pars.diseases.sir.dur_inf """

        # It's a Dist, e.g. dur_inf = ss.normal(6,2); use directly
        if isinstance(new, ss.Dist):
            if isinstance(old, ss.bernoulli) and not isinstance(new, (ss.bernoulli, ss.choose_n)): # Both have filter()
                errormsg = f"Bernoulli distributions can only be changed to another Bernoulli distribution or ss.choose_n(): {type(new)} is invalid"
                raise TypeError(errormsg)
            else:
                self[key] = new

        # It's a single number, e.g. dur_inf = 6; set parameters, or a time parameter
        elif isinstance(new, (Number, ss.dur, ss.Rate)):
            old.set(new)
            if isinstance(new, Number):
                self.note_bare(key, old, new)

        # It's a list of numbers, e.g. dur_inf = [6, 2]; set parameters
        elif isinstance(new, list):
            old.set(*new)
            if all(isinstance(v, Number) for v in new):
                self.note_bare(key, old, new)

        # It's a dict, figure out what to do
        elif isinstance(new, dict):
            newtype = new.get('type')
            if newtype is None: # Same type of dist, set parameters
                old.set(**new)
            else: # We need to create a new distribution
                if isinstance(old, ss.bernoulli) and newtype not in ['bernoulli', 'choose_n']:
                    errormsg = f"Bernoulli distributions can only be changed to another Bernoulli distribution or ss.choose_n(): {newtype} is invalid"
                    raise TypeError(errormsg)
                else:
                    dist = ss.distributions.make_dist(new)
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
        rescale (bool): If True, start with a scale factor of 1 and increase it to pop_scale as the agents stop being naive, making some of them naive again each time (dynamic rescaling, as in Covasim; default False)
        rescale_threshold (float): If rescaling, the fraction of agents who are not naive that triggers rescaling (default 0.05)
        rescale_factor (float): If rescaling, the minimum factor by which to increase the scale each time (default 1.2)
        unit (str): The time unit for the simulation (default 'year'; other choices are 'day', 'week', 'month')
        start (float/str/date): The starting date for the simulation (default 2000); can be a year or date
        stop (float/str/date): If provided, the ending date for the simulation (if not provided, calculate from "dur")
        dur (int): How many timesteps to simulate, if "stop" is not provided (default 50)
        dt (float): The timestep, in units of "unit" (default 1.0)
        rand_seed (int): The overall random seed for the simulation (used to set module-specific random seeds)
        use_aging (bool): Specify whether agents age (by default, agents age if and only if births and/or deaths are included)
        people_results (bool): Whether to collect the automatic People-level results (n_alive, n_female, new_deaths, cum_deaths, etc.) every timestep (default True); set to False to skip this per-timestep cost in lightweight/ODE-style sims that do not need them (note: some modules, e.g. demographics, rely on results.n_alive)
        people (People): If provided, use a pre-existing People object rather than creating one (in which case n_agents will be ignored)
        networks (str/list/Module): The network module(s); can be a string, single module (i.e. Network), or list
        demographics (str/list/Module/bool): births, deaths, and migration of everyone: declared (`ss.birth()`, `ss.death()`, which run under every solver), or modules of agents (e.g. `ss.Births()`, `ss.Pregnancy()`); True for births and deaths at 20 and 10 per 1,000 people per year
        diseases (str/list/Module): As above
        connectors (str/list/Module): As above
        interventions (str/list/Module): As above
        analyzers (str/list/Module): As above
        verbose (int): How much to print: 0 nothing, 1 the solver, notes, and progress (default), 2 everything (e.g. each step)
    """
    def __init__(self, pars=None, create=True, **kwargs):

        # General parameters
        self.label   = '' # The label of the simulation
        self.verbose = ss.options.verbose # How much to print: 0 (nothing), 1 (the solver, notes, and progress; default), or 2 (everything)

        # Population parameters
        self.n_agents  = 10e3 # Number of agents
        self.total_pop = None # If defined, used for calculating the scale factor
        self.pop_scale = None # How much to scale the population
        self.rescale   = False # Whether to rescale the population dynamically (see docstring)
        self.rescale_threshold = 0.05 # Fraction of non-naive agents that triggers rescaling
        self.rescale_factor    = 1.2 # Minimum factor to rescale by each time
        self.people_results = True # Whether to collect automatic People-level results every timestep (see docstring)

        # Simulation parameters
        self.start     = None  # Start of the simulation (default 2000)
        self.stop      = None  # End of the simulation
        self.dur       = None  # Duration of time to run, if stop isn't specified (default 50 steps of self.unit)
        self.dt        = None  # Timestep
        self.rand_seed = 1     # Random seed; if None, don't reset

        # How the model is run; these only apply to modules with declared states and transitions (see ss.Disease)
        self.unit       = None   # The unit for bare numbers, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        self.solver     = 'auto' # 'auto', 'ode', 'binomial', 'gillespie' (which switches to tau-leaping for a run of too many events), or 'abm'
        self.approx     = None   # None prints a note for each approximation; True keeps the note in sim.notes without printing it; False makes it an error
        self.integrator = None   # The ODE integration scheme: 'rk4' (the default) or 'euler'
        self.substeps   = None   # The number of substeps per timestep: for the 'rk4' and 'euler' integrators (default: enough that no compartment turns over, or grows through its inflows, by more than a quarter in one), and the 'binomial' solver (default: 1, so each step is taken whole; the chain-binomial solver moves people through as many stages of a dwell time as they pass in a step)
        self.stratify   = None   # People dimensions for every module, e.g. dict(age=[0, 15, 65])
        self.results    = None   # The results to keep by group, by name, e.g. ['n_I', 'cases'] or ['sir.n_I']; the others keep only their totals (default: all, every few steps if every step would take more than about 100 MB: see ss.Sim.group_rows())

        # Demographic parameters
        self.use_aging  = None # True if demographics, false otherwise

        # Modules
        self.people = None
        self._reset_modules()

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(pars, create=create, **kwargs)
        return

    _names = None # The names of the sim parameters: see names()

    removed = dict( # Sim parameters that have been removed (or that are often guessed), and what replaces them
        track_writes = 'track_writes= has been removed: sim.what_modifies() lists the modifiers on a quantity, which are declared; code that assigns rel_sus or rel_trans directly is not tracked',
        birth_rate = 'birth_rate= (births per 1,000 people per year) has been removed: declare births for the population, per person, e.g. ss.Sim(..., demographics=[ss.birth(rate=ss.peryear(0.02))]) for 20 per 1,000, which runs under every solver; or use ss.Births(birth_rate=...) for data by year',
        death_rate = 'death_rate= (deaths per 1,000 people per year) has been removed: declare deaths for the population, per person, e.g. ss.Sim(..., demographics=[ss.death(rate=ss.peryear(0.01))]) for 10 per 1,000, which runs under every solver; or use ss.Deaths(death_rate=...) for data by year, age, and sex',
        age_dist = 'age_dist= is not a sim parameter: for age groups with given shares, stratify the disease, e.g. sir.stratify(age=[0, 15, 65], split=[0.3, 0.6, 0.1]), which runs under every solver; for the ages of agents, use ss.Sim(..., people=ss.People(n_agents, age_data=df)), with columns age and value',
    )

    def check_key_mismatch(self, pars):
        """ As for any parameters, with the removed sim parameters named """
        for key in pars:
            if key in self.removed:
                raise ValueError(self.removed[key])
        return super().check_key_mismatch(pars)

    @classmethod
    def names(cls):
        """ The names of the sim parameters (as opposed to keys added later, e.g. one per module) """
        if cls._names is None:
            cls._names = set(cls().keys())
        return cls._names

    def _reset_modules(self):
        """ Initialize ndicts for module storage """
        self.custom        = ss.ndict()
        self.demographics  = ss.ndict()
        self.connectors    = ss.ndict()
        self.networks      = ss.ndict()
        self.interventions = ss.ndict()
        self.diseases      = ss.ndict()
        self.analyzers     = ss.ndict()
        self.modules       = ss.ndict()
        return

    def is_default(self, key):
        """ Check if the provided value matches the default """
        default_pars = SimPars() # Create a default SimPars object
        default_val = default_pars[key]
        current_val = self[key]
        match = (current_val == default_val) # Check if the value matches # TODO: consider sc.equal() instead
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
        return

    def validate_verbose(self):
        """ Validate verbosity """
        self.verbose = check_verbose(self.verbose)
        return

    def validate_agents(self):
        """ Check that n_agents is supplied and convert to an integer """
        named = getattr(self.people, 'named', None)
        if named is not None and named.n_agents is None: # A named population is made at the sim's number of agents: see sim._init_named_people()
            self.n_agents = int(self.n_agents)
        elif self.people is not None:
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

    def validate_modules(self):
        """ Validate modules passed in pars"""

        # Do special validation on demographics (must be before modules are created)
        self.validate_demographics()

        # Get all modules into a consistent list format
        modmap = ss.modules.module_map()
        for modkey in modmap.keys():
            if not isinstance(self[modkey], ss.ndict):
                self[modkey] = sc.tolist(self[modkey])

        # Convert any modules that are not already Module objects
        modules = self.convert_modules()
        modules = self.expand_modules(modules)
        if not any(getattr(mod, 'graph', None) is not None for source,mod in modules): # With no declared disease, births and deaths declared for the population are run as modules of agents
            modules = [(source, m) for source,mod in modules for m in (mod.as_modules() if isinstance(mod, ss.demographics.Flows) else [mod])]
        self._reset_modules()
        products = []
        names = {} # Module names must be unique across all types, since they share namespaces (e.g. sim.people.<name>, sim.results.<name>)
        for source, mod in modules:
            if source == 'modules':
                # Modules from 'modules=' get sorted by type
                for modkey, modclass in modmap.items():
                    if isinstance(mod, modclass) or (modkey == 'diseases' and isinstance(mod, ss.Strains)): # A family of diseases is with its members
                        break
                else:
                    modkey = 'custom'
            else:
                # Modules from specific containers stay where the user put them
                modkey = source
            self.validate_name(mod, modkey, names)
            if getattr(mod, 'has_product', False) and not any(mod.product is p for p in products): # Products share the same namespaces; the same product can be used by more than one intervention
                self.validate_name(mod.product, 'products', names)
                products.append(mod.product)
            self[modkey].append(mod)

        # Do special validation on networks (must be after modules are created)
        self.validate_networks()
        return

    @staticmethod
    def expand_modules(modules):
        """
        Replace any module that defines an `expand()` method with the modules it returns, e.g. `ssl.HybridNet()` → four networks

        Args:
            modules (list): (source, module) tuples, as returned by `convert_modules()`
        """
        out = []
        for source, mod in modules:
            if hasattr(mod, 'expand'):
                out += [(source, m) for m in mod.expand()]
            else:
                out.append((source, mod))
        return out

    @staticmethod
    def validate_name(mod, modkey, names):
        """
        Ensure the module name is unique, renaming it if it is the default name (e.g. "sir" → "sir_1")

        Args:
            mod (`ss.Module`): the module to validate
            modkey (str): the type of module, e.g. "diseases"
            names (dict): the names used so far, mapped to their module type; updated in place
        """
        name = mod.name
        if name in names:
            if mod._default_name and isinstance(mod, ss.Product) and names[name] == 'interventions': # e.g. vaccine = ss.vaccinate(ss.Vaccine(...)), where the delivery takes the name of its variable: the product is vaccine_vaccine
                mod.name = mod.label = sc.uniquename(f'{name}_{type(mod).__name__.lower()}', list(names), style='_%d')
            elif mod._default_name:
                mod.name = sc.uniquename(name, list(names), style='_%d')
                if isinstance(mod, ss.Product): # Another product of the same default name
                    ss.warn(f'The product "{name}" is renamed "{mod.name}", since another module (in {names[name]}) is also called "{name}"; give the product a name to avoid this warning, e.g. ss.Vaccine(..., name="{name}_product")')
                elif not isinstance(mod, ss.modify): # A modifier's default name says what it does (beta_multiply), so a second one is simply numbered
                    ss.warn(f'Module name "{name}" is already used, so renaming this module "{mod.name}"; set the name explicitly to avoid this warning')
            else:
                errormsg = f'Cannot add module "{name}" to {modkey}: a module with that name already exists in {names[name]}; please give one of them a different name'
                raise ValueError(errormsg)
        names[mod.name] = modkey
        return

    def validate_demographics(self):
        """ Validate demographics-related input parameters"""
        # demographics=True is births and deaths at the default rates (20 and 10 per 1,000 people per year), declared for the population
        demog = self.demographics
        if demog in [True, 1]:
            demog = [ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.01))]
        elif demog in [False, 0]:
            demog = ss.ndict()

        # Births, deaths, and migration declared for the population are held together, until the sim adds them to a disease: see ss.demographics.Flows
        if not isinstance(demog, ss.ndict):
            demog = sc.tolist(demog)
            flows = [mod for mod in demog if isinstance(mod, ss.Transition)]
            if flows:
                demog = [mod for mod in demog if not isinstance(mod, ss.Transition)] + [ss.demographics.Flows(flows)]
        self.demographics = demog

        # Decide whether to use aging based on if demographics modules are present
        if self.use_aging is None:
            self.use_aging = True if self.demographics else False
        return

    def validate_networks(self):
        """ Validate networks """
        return

    def convert_modules(self):
        """
        Convert different types of representations for modules into a
        standardized object representation that can be parsed and used by
        a Sim object

        Returns a flat list of (source_key, ss.Module) tuples
        """

        out = []
        modmap = ss.modules.module_map() # List of modules and parent module classes, e.g. ss.Disease
        modmap['modules'] = ss.Module # Enable passing in modules of any type via `Pars(modules=[...])`
        modmap['custom'] = ss.Module # Enable passing in modules of any type via `Pars(custom=[...])`
        supported_modules = ss.modules.find_modules() # Each individual module class option, e.g. ss.SIR
        supported_modules['modules'] = sc.objdict({k:v for vv in supported_modules.values() for k,v in vv.items()})
        supported_modules['custom'] = supported_modules['modules']

        for modkey, ssmoddict in supported_modules.items():

            moddictkeys = ssmoddict.keys()
            moddictvals = ssmoddict.values()
            expected_cls = modmap[modkey]

            given = self[modkey]
            if isinstance(given, list) and len(given) == 1 and isinstance(given[0], dict): # A dict, wrapped in a list on the way in
                given = given[0]
            if isinstance(given, dict) and not isinstance(given, ss.ndict) and len(given) and all(isinstance(v, ss.Module) for v in given.values()): # Modules by name, e.g. networks=dict(main=..., casual=...)
                modlist = []
                for name,mod in given.items():
                    if mod._default_name:
                        mod.name = mod.label = name
                        mod._default_name = False
                    modlist.append(mod)
            elif isinstance(self[modkey], list):
                modlist = self[modkey]
            elif isinstance(self[modkey], ss.ndict):
                # If it is an ndict, then assume the keys specify names the values represent modules
                # They may already be module instances, in which case we will simply verify that the type
                # matches the expected type
                modlist = list(self[modkey].values())
            else:
                # If it is a dict with a single entry, then we wrap this in a list assuming that it is a dict
                # representation of a single module (as opposed to a dict containing modules)
                modlist = sc.promotetolist(self[modkey])

            for i,mod in enumerate(modlist):

                # Convert first from a string to a dict
                if isinstance(mod, str):
                    mod = dict(type=mod)

                # Convert from class to class instance (for any module type)
                if isinstance(mod, type) and issubclass(mod, ss.Module):
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

                # Convert class and function to class instance (for any module type). expected_cls is the
                # parent class for this key (e.g. ss.Disease), or ss.Module for 'modules'/'custom'.
                if isinstance(mod, type) and issubclass(mod, expected_cls):
                    mod = mod()  # Convert from a class to an instance of a class
                elif not isinstance(mod, ss.Module) and callable(mod):
                    mod = expected_cls.from_func(mod)

                # Do final check
                if isinstance(expected_cls, type) and not isinstance(mod, (expected_cls, ss.Module)): # TEMP: check if this check still works?
                    errormsg = f'Was expecting {modkey} entry {i} to be class {expected_cls} or Module, but was {type(mod)} instead'
                    raise TypeError(errormsg)

                out.append((modkey, mod))

        return out


def check_verbose(verbose):
    """ Check that `verbose` is 0 (print nothing), 1 (the solver, notes, and progress), or 2 or more (everything), and return it as an integer """
    if sc.isnumber(verbose) and verbose >= 0 and verbose == int(verbose): # Includes True and False
        return int(verbose)
    errormsg = f'verbose must be 0 (print nothing), 1 (the solver, notes, and progress), or 2 or more (everything), not {verbose!r}'
    if sc.isnumber(verbose) and 0 < verbose < 1:
        errormsg += '. A fraction of the steps (e.g. verbose=0.1 in Starsim 3) is now verbose=1, which shows the progress about ten times a run'
    elif verbose in ['brief', -1]:
        errormsg += '. verbose="brief" (or -1) in Starsim 3 is now verbose=0'
    raise ValueError(errormsg)


class Requires(sc.prettyobj):
    """
    One item of the data a module needs, for `sim.requires()`

    This is metadata only: it says what to supply, and is never enforced.

    Args:
        name (str): the name of the item; a parameter or attribute of the module with this name supplies it
        shape (str): its shape, e.g. 'age x age'
        source (str): where the data usually come from
        doc (str): a description
        default (any): the value used if none is supplied

    Examples:
        ```python
        class MyNetwork(ss.Network):
            requires = [
                ss.Requires('population',     shape='patch',     source='UN WPP 2024'),
                ss.Requires('contact_matrix', shape='age x age', source='Prem et al. 2021'),
            ]
        ```
    """
    def __init__(self, name, shape=None, source=None, doc=None, default=None):
        self.name = name
        self.shape = shape
        self.source = source
        self.doc = doc
        self.default = default
        return


def load_pars(filename):
    """
    Read parameter values by name from a file, for `ss.Sim(pars=...)`

    A CSV file has either a column of names and a column of values (`name,value`), or
    one row of values with a column per parameter; a TOML or JSON file is a table of
    values by name.

    Args:
        filename (str/path): the file

    Examples:
        ```python
        sim = ss.Sim(sir, pars='params/kenya_2024.csv', unit='day')
        ```
    """
    path = sc.path(filename)
    suffix = path.suffix.lower()
    if suffix == '.json':
        return dict(sc.loadjson(path))
    elif suffix == '.toml':
        import tomllib
        with open(path, 'rb') as f:
            return tomllib.load(f)
    elif suffix == '.csv':
        df = pd.read_csv(path)
        cols = [str(col).lower() for col in df.columns]
        if len(cols) == 2 and cols[1] == 'value':
            return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))
        elif len(df) == 1:
            return df.iloc[0].to_dict()
        errormsg = f'The parameter file {path} should have a column of names and a column named "value", or one row of values with a column per parameter; it has columns {list(df.columns)} and {len(df)} rows'
        raise ValueError(errormsg)
    errormsg = f'The parameter file {path} should be a CSV, TOML, or JSON file, not "{suffix}"'
    raise ValueError(errormsg)


#%% Values that vary by level or in time

def isnull(val):
    """ Whether a value in a table is missing (None or NaN), e.g. the row for people with no value of a property """
    return val is None or (isinstance(val, float) and np.isnan(val))


class by:
    """
    A value for each level of one or more stratifications, or of people's own properties

    With several dimensions as keywords, the value of a group is the product of
    its value in each. A table gives every combination directly: it has a column
    for each stratification (holding the labels of its levels) and a column of values.

    A property of people that is not a dimension of the sim (e.g. sex, read from
    `people.female`, or a property such as `urban`) is read from each person, as agents:
    give its values by name (`sex=dict(female=..., male=...)`), or as a table. A column of
    numbers in a table holds the lower edges of bins (the last one open-ended), so a table by
    single year of age works whatever the model's age groups (a compartmental run averages
    it over each group, with a note); a row whose key is None gives the value for people with
    no value of the property, who otherwise get the mean over the levels.

    Args:
        table (df/array): a dataframe (or a dict of columns) with a column per stratification or property and a column named `value`; or, with `dims`, an array with one axis per dimension
        dims (list): the dimensions of the axes of an array given as `table`
        kwargs (dict): each dimension, and one value per level (a list, by position), or a dict of values by label

    A value may itself vary in time (`ss.timeseries` or `ss.seasonal`), in the keyword and table forms.

    In `p=` of a transition with several destinations, a dimension called `to` is the destination: its labels are
    the names of destination states, and the entry of `p=` it is in is normalized over them (see `ss.Transition`).

    Examples:
        ```python
        beta = ss.by(age=[0.08, 0.05, 0.03])
        beta = ss.by(age=[0.08, 0.05, 0.03], sex=[1.0, 0.8]) # One per age group and sex: the product
        init = ss.by(pd.DataFrame(age=['child', 'adult'], patch=['north', 'north'], value=[10, 5])) # Every other combination is 0
        pop = ss.by([[3000, 1000], [5000, 2000]], dims=['age', 'patch']) # Ages by row, places by column
        beta = ss.by(age=[ss.timeseries({2020: 0.08, 2021: 0.04}), 0.05, 0.03]) # Children's value changes in 2021
        mu = ss.by(sex=dict(female=0.008, male=0.011)) # From people.female, with or without a dimension by sex
        p = ss.by(dict(age=range(15, 50), value=vals)) # By single year of age (bins with these lower edges)
        mix = ss.by(dict(age=[15, 15, 25, 25], to=['pill', 'iud', 'pill', 'iud'], value=[0.7, 0.3, 0.4, 0.6])) # A switching row by age, over the destinations
        ```
    """
    def __init__(self, table=None, dims=None, **kwargs):
        if (table is None) == (not kwargs):
            errormsg = 'ss.by() takes a table, or one value per level for each dimension, e.g. ss.by(age=[0.1, 0.2, 0.3])'
            raise ValueError(errormsg)
        self.array = None
        self.wrap = None # The type of a typed array given as the table, e.g. ss.permonth(np.ones((2, 2)))
        if dims is not None: # An array with one axis for each dimension
            if isinstance(table, ss.TimePar):
                self.wrap = type(table)
                table = table.value
            self.array = np.array(table, dtype=float)
            if self.array.ndim != len(dims):
                errormsg = f'ss.by() was given {len(dims)} dimensions ({sc.strjoin(dims)}), but an array with {self.array.ndim} axes'
                raise ValueError(errormsg)
            self.table = None
            self.dims = list(dims)
        elif table is not None:
            table = sc.dataframe(table)
            if 'value' not in table.columns:
                errormsg = f'A table given to ss.by() needs a column named "value"; its columns are {list(table.columns)}'
                raise KeyError(errormsg)
            self.table = table
            self.dims = [col for col in table.columns if col != 'value']
        else:
            self.table = None
            self.dims = list(kwargs.keys())
        self.given = kwargs
        return


    def to_json(self):
        """ The value as it is written in the model spec """
        if self.array is not None:
            return dict(by=self.dims, array=self.array.tolist())
        elif self.table is not None:
            return dict(by=self.dims, table={col:ss.presentation.quantity_json(list(vals), None, None) for col,vals in self.table.items()})
        values = {dim:dict(keyed={str(k):ss.presentation.quantity_json(v, None, None) for k,v in vals.items()}) if isinstance(vals, dict) else ss.presentation.quantity_json(list(vals), None, None) for dim,vals in self.given.items()}
        return dict(by=self.dims, values=values[self.dims[0]]) if len(self.dims) == 1 else dict(by=self.dims, product=values)
    def map(self, func):
        """ A new ss.by with a function applied to each value, e.g. `1/by` (with several dimensions as keywords, only scaling, since the value of a group is their product) """
        if self.array is not None:
            return by(func(self.array if self.wrap is None else self.wrap(self.array)), dims=self.dims)
        elif self.table is not None:
            table = self.table.copy()
            table['value'] = [func(v) for v in table['value']]
            return by(table)
        elif len(self.dims) == 1:
            vals = self.values
            return by(**{self.dim:{k:func(v) for k,v in vals.items()} if isinstance(vals, dict) else [func(v) for v in sc.tolist(vals)]})
        errormsg = f'Arithmetic on an ss.by() of several dimensions given as keywords ({sc.strjoin(self.dims)}) is not defined, since each group\'s value is a product; give it as a table or an array with dims= instead'
        raise TypeError(errormsg)

    def _op(self, other, func, op=None):
        """ Arithmetic with a number, or level by level with another ss.by over the same one dimension, or a product of ss.by over different dimensions (each group's value the product); anything else (an expression) is left to the other side """
        if sc.isnumber(other):
            return self.map(lambda v: func(v, other))
        elif isinstance(other, by) and self.given and other.given and not set(self.dims) & set(other.dims) and op == '*': # A product over different dimensions, given as keywords, e.g. by(age=...) * by(sex=...)
            return by(**self.given, **other.given)
        elif isinstance(other, by) and op == '*' and not set(self.dims) & set(other.dims) and self.as_table() is not None and other.as_table() is not None: # Tables (or values by label) over different dimensions: every combination of their rows
            rows = self.as_table().merge(other.as_table(), how='cross', suffixes=('', '_other'))
            rows['value'] = [func(a, b) for a,b in zip(rows['value'], rows['value_other'])]
            return by(rows.drop(columns='value_other'))
        elif isinstance(other, by) and self.dim is not None and other.dim == self.dim and isinstance(self.values, dict) and isinstance(other.values, dict):
            if self.values.keys() != other.values.keys():
                errormsg = f'Cannot combine an ss.by() of {sc.strjoin(self.values.keys())} with one of {sc.strjoin(other.values.keys())}'
                raise ValueError(errormsg)
            return by(**{self.dim:{k:func(v, other.values[k]) for k,v in self.values.items()}})
        elif isinstance(other, by) and self.dim is not None and other.dim == self.dim:
            a, b = sc.tolist(self.values), sc.tolist(other.values)
            if len(a) != len(b):
                errormsg = f'Cannot combine an ss.by() of {len(a)} levels of "{self.dim}" with one of {len(b)}'
                raise ValueError(errormsg)
            return by(**{self.dim:[func(x, y) for x,y in zip(a, b)]})
        return NotImplemented

    def __mul__(self, other):      return self._op(other, lambda a,b: a*b, '*')
    def __rmul__(self, other):     return self._op(other, lambda a,b: b*a, '*')
    def __truediv__(self, other):  return self._op(other, lambda a,b: a/b)
    def __rtruediv__(self, other): return self._op(other, lambda a,b: b/a)
    def __add__(self, other):      return self._op(other, lambda a,b: a + b)
    def __radd__(self, other):     return self._op(other, lambda a,b: b + a)
    def __sub__(self, other):      return self._op(other, lambda a,b: a - b)
    def __rsub__(self, other):     return self._op(other, lambda a,b: b - a)
    def __pow__(self, other):      return self._op(other, lambda a,b: a**b)
    def __neg__(self):             return self.map(lambda v: -v)

    @property
    def items(self):
        """ Every value it holds """
        if self.array is not None:
            return list(self.array.flat)
        if self.table is not None:
            return list(self.table['value'])
        return [v for vals in self.given.values() for v in (list(vals.values()) if isinstance(vals, dict) else sc.tolist(vals))]

    @property
    def varies(self):
        """ Whether any of its values varies in time """
        return any(isinstance(v, TimeValue) for v in self.items)

    @property
    def dim(self):
        """ The dimension, if there is one """
        return self.dims[0] if len(self.dims) == 1 else None

    @property
    def values(self):
        """ The values, if there is one dimension """
        if len(self.dims) != 1:
            return None
        elif self.array is not None: # ss.by([...], dims=['age'])
            return self.array.tolist()
        return self.given[self.dim] if self.given else None

    def __repr__(self):
        if self.array is not None:
            return f'by({self.array.tolist()}, dims={self.dims})'
        if self.table is not None:
            return f'by(table of {len(self.table)} rows by {sc.strjoin(self.dims)})'
        return f'by({", ".join(f"{k}={v}" for k,v in self.given.items())})'

    def as_table(self):
        """ The values as a table, if they are one or are given by label over one dimension (`ss.by(sex=dict(...))`); else None """
        if self.table is not None:
            return pd.DataFrame(self.table)
        vals = self.given.get(self.dim) if self.given and self.dim is not None else None
        if isinstance(vals, dict):
            return pd.DataFrame({self.dim:list(vals.keys()), 'value':list(vals.values())})
        return None

    def to_labels(self):
        """ The destinations of a value with a `to` axis, if it names them (a table's column, or a dict by name); None if it gives them by position """
        if self.table is not None:
            return list(dict.fromkeys(str(v) for v in self.table['to']))
        vals = self.given.get('to') if self.given else None
        return [str(k) for k in vals] if isinstance(vals, dict) else None

    def at(self, to, labels):
        """
        The value for one destination of a value with a `to` axis: a value, or an `ss.by` over its other dimensions

        Args:
            to (str): the destination
            labels (list): the destinations of the entry it is in, in order, for values given by position
        """
        others = [dim for dim in self.dims if dim != 'to']
        def position(n):
            if n != len(labels):
                raise ss.ShapeError(f'{self} gives {n} values over "to" by position, but the entry of p= it is in covers {len(labels)} destinations ({sc.strjoin(labels)})')
            return labels.index(to)
        if self.array is not None:
            axis = self.dims.index('to')
            arr = np.take(self.array, position(self.array.shape[axis]), axis=axis)
            return by(arr, dims=others) if others else float(arr)
        if self.table is not None:
            rows = self.table[[str(v) == to for v in self.table['to']]].drop(columns='to').reset_index(drop=True)
            if not len(rows): # A destination it does not name has none
                return 0.0
            return by(rows) if others else rows['value'].iloc[0]
        vals = self.given['to']
        val = {str(k):v for k,v in vals.items()}.get(to, 0.0) if isinstance(vals, dict) else sc.tolist(vals)[position(len(sc.tolist(vals)))]
        if others: # A product with values by other dimensions
            return as_expr(by(**{dim:self.given[dim] for dim in others}))*as_expr(val)
        return val

    def keys(self, dim):
        """ How the values along one dimension are given: ('edges', array) for a table's column of numbers (lower bin edges), ('keys', labels) for a table's column of labels or a dict of values by label, or None for values by position """
        if self.table is not None:
            vals = [v for v in self.table[dim] if not isnull(v)]
            if vals and all(sc.isnumber(v) and not isinstance(v, (bool, np.bool_)) for v in vals):
                return 'edges', np.unique(np.array(vals, dtype=float))
            return 'keys', list(dict.fromkeys(str(v) for v in vals))
        vals = self.given.get(dim)
        if isinstance(vals, dict):
            return 'keys', [str(k) for k in vals]
        return None

    def check(self, sim, note=False):
        """ Check the values against the sim's dimensions (`ss.ShapeError`), and that a property of people that is not one is given by value (with `note=True`, when the sim is built, also note a table averaged over groups); returns the positions of the sim's dimensions it is by, and those dimensions """
        names = [strat.name for strat in sim.groups.strats]
        if 'to' in self.dims: # Over the destinations of a transition, which is taken a destination at a time (see at())
            return [], []
        for dim in self.dims:
            if dim not in names and self.keys(dim) is None: # By position, which needs the dimension's levels
                if dim == 'sex':
                    fix = 'give them by name, ss.by(sex=dict(female=..., male=...)), which is read from people.female, or add sim.stratify(sex=["female", "male"])'
                elif dim == 'age':
                    fix = 'give a table by age with the lower edge of each bin, e.g. ss.by(dict(age=[0, 15, 50], value=[...])), or add sim.stratify(age=[...])'
                else:
                    fix = f'give them by value, ss.by({dim}=dict(...)), which is read from people.{dim}, or add sim.stratify({dim}=[...])'
                raise ss.MissingDimension(f'ss.by({dim}=...) gives a value for each level of "{dim}" by position, but the model is not stratified by "{dim}", so the positions have no order; {fix}.')
        ds = [names.index(dim) for dim in self.dims if dim in names]
        strats = [sim.groups.strats[d] for d in ds]
        if self.array is not None:
            shape = tuple(strat.n for strat in strats)
            if self.array.shape != shape:
                raise ss.ShapeError(f'ss.by() has an array of shape {self.array.shape}, but {sc.strjoin(self.dims)} have {shape} levels.')
        elif self.table is None:
            for strat in strats:
                vals = self.given[strat.name]
                labels = strat.coarse_labels if strat.coarse is not None else strat.labels # A clock's levels as given, when they are steps
                if isinstance(vals, dict):
                    missing = [str(label) for label in labels if str(label) not in [str(k) for k in vals]]
                    if missing:
                        raise ss.ShapeError(f'ss.by({strat.name}=dict(...)) has no value for {sc.strjoin(missing)} (the levels of "{strat.name}" are {sc.strjoin(labels)}).')
                    continue
                n = len(vals) if isinstance(vals, (list, tuple)) else len(sc.toarray(vals)) if np.ndim(vals) else len(labels) # One value is for every level, e.g. ss.by(subpop=2.2)
                if n != len(labels):
                    raise ss.ShapeError(f'ss.by({strat.name}=...) has {len(sc.toarray(vals))} values, but "{strat.name}" has {len(labels)} levels ({sc.strjoin(labels)}).')
        if note and sim.solver != 'abm':
            for strat in strats:
                kind = self.keys(strat.name)
                if kind is not None and kind[0] == 'edges' and not (strat.edges is not None and np.array_equal(strat.edges, kind[1])):
                    ss.note(f'{self} gives values by bins of {strat.name} other than the model\'s ({sc.strjoin(strat.labels)}), so a compartmental run averages it over each group, weighting each bin by how much of the group it covers; agents read each person\'s own {strat.name}', sim=sim)
        return ds, strats

    def converter(self, sim, convert=None, year=None):
        """ A function that gives a value as a number: a value that varies in time is taken at the sim's current time, or at `year` if given """
        def out(v):
            if not isinstance(v, TimeValue):
                return v if convert is None else convert(v)
            when = sim.t.now('year') if year is None else year
            if convert is None:
                return v.at(when, sim)
            return v.series([when], sim, convert)[0] # Its values converted, then interpolated as numbers
        return out

    def grid(self, sim, convert=None, year=None):
        """
        The value for every combination of the levels of its dimensions

        Returns the positions of the dimensions in the sim, and an array with one axis for each.
        A value that varies in time is taken at the sim's current time, or at `year` if given.
        """
        convert = self.converter(sim, convert, year)
        ds, strats = self.check(sim)
        if len(strats) < len(self.dims):
            raise ss.CapabilityError(f'{self} is by {sc.strjoin(dim for dim in self.dims if dim not in [strat.name for strat in strats])}, a property of each person that is not a dimension, so it has no value for a group.\n'
                                     f'     Stratify by it (sim.stratify(...)) to run as compartments, or run with solver="abm".')
        if self.array is not None:
            wrap = self.wrap or (lambda v: v)
            return ds, np.vectorize(lambda v: float(convert(wrap(v))))(self.array) if self.array.size else self.array.copy()
        if self.table is None:
            out = np.ones([strat.n for strat in strats])
            for i,(strat,(dim,vals)) in enumerate(zip(strats, self.given.items())):
                if isinstance(vals, dict): # By label, in the order of the levels
                    vals = {str(k):v for k,v in vals.items()}
                    vals = [vals[str(label)] for label in (strat.coarse_labels if strat.coarse is not None else strat.labels)]
                if strat.coarse is not None and isinstance(vals, (list, tuple)) and len(vals) == len(strat.coarse_labels): # A clock whose levels are steps: each step takes the value of the level as given that it is in
                    vals = [vals[c] for c in strat.coarse]
                if isinstance(vals, (list, tuple)): # The usual case, which NumPy need not inspect (slowly, for typed values)
                    single, items = False, vals
                else:
                    single, items = np.ndim(vals) == 0, sc.toarray(vals) # One value for every level, e.g. ss.by(subpop=2.2)
                vals = np.array([convert(v) for v in items], dtype=float)
                if single:
                    vals = np.full(strat.n, vals[0])
                shape = [1]*len(strats)
                shape[i] = strat.n
                out = out*vals.reshape(shape)
            return ds, out
        if any(self.keys(strat.name)[0] == 'edges' for strat in strats): # Bins of their own: averaged over each group
            return ds, self.binned_grid(strats, convert)
        out = np.zeros([strat.n for strat in strats])
        for _,row in self.table.iterrows():
            index = []
            for strat in strats:
                labels = [str(label) for label in strat.labels]
                label = str(row[strat.name])
                if label not in labels:
                    raise ss.ShapeError(f'the table given to ss.by() has "{label}" for "{strat.name}", which is not one of its levels ({sc.strjoin(labels)}).')
                index.append(labels.index(label))
            out[tuple(index)] += convert(row['value'])
        return ds, out

    def binned_grid(self, strats, convert):
        """ For a table with a column of bin edges (e.g. single years of age): the value for each group, the mean over the table's bins weighted by how much of the group each covers """
        num = np.zeros([strat.n for strat in strats])
        den = np.zeros_like(num)
        for _,row in self.table.iterrows():
            parts = [] # For each dimension: the groups the row is in, and its weight in each
            for strat in strats:
                kind, keys = self.keys(strat.name)
                val = row[strat.name]
                if kind == 'keys':
                    labels = [str(label) for label in strat.labels]
                    if str(val) not in labels:
                        raise ss.ShapeError(f'the table given to ss.by() has "{val}" for "{strat.name}", which is not one of its levels ({sc.strjoin(labels)}).')
                    parts.append([(labels.index(str(val)), 1.0)])
                elif isnull(val) and strat.kind == 'clock': # People who have not had the event: a clock's level none
                    parts.append([(strat.n - 1, 1.0)])
                elif isnull(val) or strat.edges is None: # People with no value, who are in no group
                    parts.append([])
                else:
                    edges = strat.edges[:-1] if strat.kind == 'clock' else strat.edges # A clock's last level is none, with no time
                    cap = max(keys[-1], edges[-1]) + 1 # The last bin of each is open-ended
                    lo = float(val)
                    hi = keys[keys > lo][0] if np.any(keys > lo) else cap
                    ends = np.append(edges[1:], cap)
                    overlap = np.maximum(np.minimum(hi, ends) - np.maximum(lo, edges), 0)
                    parts.append([(i, w) for i,w in enumerate(overlap) if w > 0])
            value = float(convert(row['value']))
            for combo in itertools.product(*parts):
                index = tuple(i for i,w in combo)
                weight = np.prod([w for i,w in combo])
                num[index] += weight*value
                den[index] += weight
        out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        for i,strat in enumerate(strats): # A clock with no row for none: the mean over its other levels, as agents take it (weighted by the people at each)
            if strat.kind == 'clock' and not np.take(den, [strat.n - 1], axis=i).any():
                inner = np.take(out, range(strat.n - 1), axis=i)
                np.put_along_axis(out, np.full([1 if j == i else s.n for j,s in enumerate(strats)], strat.n - 1), inner.mean(axis=i, keepdims=True), axis=i)
        return out

    def on_people(self, sim):
        """ Whether, as agents, it is read from each person's own values rather than by group: over a property that is not a dimension of the sim, or a table whose numbers are bin edges other than the dimension's (e.g. single years of age, against age groups) """
        known = getattr(self, '_on_people', None)
        if known is not None and known[0] == id(sim):
            return known[1]
        names = [strat.name for strat in sim.groups.strats]
        out = False
        for dim in self.dims:
            kind = self.keys(dim)
            strat = sim.groups.strats[names.index(dim)] if dim in names else None
            if strat is None or (kind is not None and kind[0] == 'edges' and not (strat.edges is not None and np.array_equal(strat.edges, kind[1]))):
                out = True
        self._on_people = (id(sim), out)
        return out

    def positions(self, sim, dim):
        """ As agents: the number of levels along one dimension, and each agent's position along it (indexed by UID), with that number for people with no value """
        people = sim.people
        names = [strat.name for strat in sim.groups.strats]
        kind = self.keys(dim)
        strat = sim.groups.strats[names.index(dim)] if dim in names else None
        if strat is not None and kind is None: # By level of a dimension of the sim
            level = sim.groups.level_raw(names.index(dim)).astype(int)
            return strat.n, np.where(level >= 0, level, strat.n)
        kind, keys = kind
        if strat is not None and strat.edges is None: # A dimension's labels, by name
            level = sim.groups.level_raw(names.index(dim)).astype(int)
            order = np.array([keys.index(str(label)) for label in strat.labels] + [len(keys)])
            return len(keys), order[np.where(level >= 0, level, strat.n)]
        if dim == 'sex' and strat is None: # people.female: a key is for women if it says so (as for a dimension called sex), and otherwise for men
            raw = people.female.raw
            keys = [str(key.lower() in ['female', 'females', 'f', 'women', 'woman']) for key in keys]
        else:
            if strat is not None:
                dim = strat.attr
            try:
                raw = people[dim].raw
            except AttributeError:
                raise ss.UnknownState(f'ss.by() is by "{dim}", which is neither a dimension of the sim nor a property of people{ss.expressions.did_you_mean(dim, list(people.states.keys()))}') from None
        n = len(keys)
        if kind == 'edges':
            pos = np.digitize(raw, keys) - 1
            return n, np.where((pos < 0) | np.isnan(raw.astype(float)), n, pos)
        out = np.full(len(raw), n)
        if raw.dtype == bool:
            vals = [key.lower() in ['true', '1'] for key in keys]
        elif raw.dtype.kind in 'iuf':
            vals = [float(key) if sc.isnumber(key) or key.replace('.', '', 1).lstrip('-').isdigit() else np.nan for key in keys]
        else:
            vals = keys
            raw = raw.astype(str)
        for i,val in enumerate(vals):
            out[raw == val] = i
        return n, out

    def lookup(self, sim, convert=None, year=None):
        """ As agents: the value for every agent (indexed by UID), from their own value of each property, or their level of each dimension """
        convert = self.converter(sim, convert, year)
        sizes, index = zip(*[self.positions(sim, dim) for dim in self.dims])
        auids = sim.people.auids
        weights = [np.bincount(ind[auids], minlength=n + 1)[:n] for n,ind in zip(sizes, index)] # People at each level, for the value of those with none
        weights = [w if w.sum() > 0 else np.ones(len(w)) for w in weights]
        if self.table is None:
            out = np.ones([n + 1 for n in sizes])
            for i,(dim,n) in enumerate(zip(self.dims, sizes)):
                vals = self.given[dim]
                vals = list(vals.values()) if isinstance(vals, dict) else [vals]*n if np.ndim(vals) == 0 else list(vals)
                vals = np.array([convert(v) for v in vals], dtype=float)
                vals = np.append(vals, np.average(vals, weights=weights[i]))
                shape = [1]*len(sizes)
                shape[i] = n + 1
                out = out*vals.reshape(shape)
        else:
            out = np.zeros([n + 1 for n in sizes])
            given = [False]*len(sizes) # Whether a row gives the value for people with no value
            for _,row in self.table.iterrows():
                idx = []
                for i,(dim,n) in enumerate(zip(self.dims, sizes)):
                    val = row[dim]
                    kind, keys = self.keys(dim)
                    if isnull(val):
                        idx.append(n)
                        given[i] = True
                    else:
                        idx.append(int(np.flatnonzero(keys == float(val))[0]) if kind == 'edges' else keys.index(str(val)))
                out[tuple(idx)] += convert(row['value'])
            for i,n in enumerate(sizes):
                if not given[i]:
                    inner = np.take(out, range(n), axis=i)
                    np.put_along_axis(out, np.full([1 if j == i else s + 1 for j,s in enumerate(sizes)], n), np.expand_dims(np.average(inner, axis=i, weights=weights[i]), i), axis=i)
        return out[index]

    def per_group(self, sim, convert=None, year=None):
        """ The value for each group of the sim """
        ds, grid = self.grid(sim, convert, year)
        grid, index = sim.groups.grid_index(ds, grid)
        return grid[index]

    def is_count(self):
        """ Whether the values are whole numbers (so that a 1 is one person rather than everyone) """
        if self.array is not None:
            return bool(np.all(self.array == np.round(self.array)))
        return np.asarray(self.items).dtype.kind in 'iu'


class Static:
    """ What an expression of parameters alone is evaluated with, outside the engine: see `const_value()` """
    @staticmethod
    def value(p):
        return p.raw


def const_value(val):
    """ The value of a number or of an expression of parameters (e.g. (1 + smin)/2), as it is now """
    return evaluate(val, Static) if isinstance(val, Expr) else val


class TimeValue:
    """ Base class for values that vary in time: called with the sim, they return the value at the current time """
    fields = [] # The attributes that can hold parameters
    _is_expr = True # So a typed value defers to it, and ss.perday(0.1) * ss.timeseries(...) is an expression as the other order is

    @property
    def pars(self):
        """ The parameters in its values, e.g. smin in ss.seasonal(mean=(1 + smin)/2, ...) """
        return [leaf for name in self.fields for v in sc.tolist(getattr(self, name)) if isinstance(v, Expr) for leaf in v.leaves if isinstance(leaf, par)]

    def __call__(self, sim):
        return self.at(sim.t.now('year'), sim)

    def series(self, years, sim, convert):
        """ The value at each of an array of years, converted to plain numbers by `convert` """
        return np.array([convert(self.at(year, sim)) for year in years], dtype=float)

    @property
    def __name__(self): # So it can be named in messages like any other function of the sim
        return type(self).__name__

    @property
    def typed(self):
        """ The typed quantities it holds, used to infer the sim's unit """
        return []

    # Arithmetic gives an expression, for a rate, e.g. 2 * ss.seasonal(...) or beta * ss.timeseries(...)
    def __mul__(self, other):      return as_expr(self) * other
    def __rmul__(self, other):     return other * as_expr(self)
    def __truediv__(self, other):  return as_expr(self) / other
    def __rtruediv__(self, other): return other / as_expr(self)
    def __add__(self, other):      return as_expr(self) + other
    def __radd__(self, other):     return other + as_expr(self)
    def __sub__(self, other):      return as_expr(self) - other
    def __rsub__(self, other):     return other - as_expr(self)
    def __neg__(self):             return -1 * as_expr(self)


class timeseries(TimeValue):
    """
    A value that varies in time, given at some times and interpolated between them

    It can be used wherever a value or a function of the sim can: as a rate or a
    duration, as the amount of `ss.modify()`, or as the coverage or capacity of a delivery.
    Before the first time the first value is used, and after the last time the last.

    Args:
        data (dict/str/dataframe/list): the value at each time, as a dict, a dataframe (the first column is
            the time, the second the value), or the name of a file holding one; or the times, with the values
            as the second argument (`ss.timeseries(times, values)`). A time is a date, or a number, which is
            a time in the sim's unit (in a sim with dates, that long after the start)
        values (list): the values, if `data` is the times
        mode (str): how to interpolate: 'constant' (each value holds until the next), 'linear', 'cosine' (a smooth S-shaped change between each value and the next, as MEmilio's dampings), or 'spline'
        period (dur): if given, the series repeats with this period, measured from its first time (e.g. `ss.years(1)` for a yearly pattern given over one year)

    Examples:
        ```python
        beta = ss.timeseries({'2020-01-01': 0.3, '2020-03-15': 0.1, '2020-06-01': 0.2})
        decay = ss.timeseries({0: 0.1, 91: 0.2, 183: 0.15, 274: 0.1}, period=ss.days(365)) # The same each year
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        sim = ss.Sim(sir, start='2020-01-01', stop='2020-09-01').run()
        ```
    """
    def __init__(self, data, values=None, mode='constant', period=None):
        if isinstance(values, str): # ss.timeseries(data, 'linear'): the mode
            values, mode = None, values
        if values is not None: # ss.timeseries(times, values), as np.interp() takes them (TL65: the values were read as mode=)
            if len(data) != len(values):
                errormsg = f'ss.timeseries() was given {len(data)} times but {len(values)} values; give one value for each time'
                raise ValueError(errormsg)
            data = dict(zip(data, values))
        if isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if not isinstance(data, dict):
            data = dict(zip(data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()))
        if mode not in ['constant', 'linear', 'cosine', 'spline']:
            errormsg = f'The mode of ss.timeseries() is "constant", "linear", "cosine", or "spline", not "{mode}"'
            raise ValueError(errormsg)
        self.times = list(data.keys())
        self.values = list(data.values())
        self.mode = mode
        self.period = period
        self._years = None # The times as years, worked out for the sim it is used in
        return

    def __repr__(self):
        return f'timeseries({dict(zip(self.times, self.values))}, mode={self.mode!r})'

    @property
    def typed(self):
        return self.values

    def years(self, sim):
        """ The times as years on the sim's timeline (for a sim with no dates, the time since the start, in years) """
        if self._years is None or self._years[0] is not sim:
            out = []
            for time in self.times:
                time = ss.connectors.to_time(sim, time)
                out.append(time.years if isinstance(time, ss.date) else time.to('year').value)
            if np.any(np.diff(out) <= 0):
                errormsg = f'The times of ss.timeseries() must be in order, not {self.times}'
                raise ValueError(errormsg)
            start, stop = sim.t.yearvec[0], sim.t.yearvec[-1]
            if len(out) > 1 and (out[-1] < start or out[0] > stop) and any(sc.isnumber(time) for time in self.times): # e.g. days of the year (0 to 365) beside calendar years: the value is flat over the run
                ss.note(f'every time of an ss.timeseries ({self.times[0]} to {self.times[-1]}) is outside the run, so its value is the same throughout; bare times are in the sim\'s unit ({sim.unit_base}s, or calendar years beside a year-like start); for another unit, give typed times, e.g. ss.days([...])', sim=sim)
            self._years = (sim, np.array(out, dtype=float))
        return self._years[1]

    fields = ['values']

    def weights(self, year, sim):
        """ For a year (or an array of them), the index of the value before it, and the weight of the value after it """
        years = self.years(sim)
        if self.period is not None: # Fold the time into one period from the first time
            period = self.period.years if isinstance(self.period, ss.dur) else float(self.period)
            year = years[0] + (year - years[0]) % period
        i = np.clip(np.searchsorted(years, year + 1e-9, side='right') - 1, 0, len(years) - 1)
        j = np.minimum(i + 1, len(years) - 1)
        w = np.where(j > i, np.clip((year - years[i])/np.where(j > i, years[j] - years[i], 1), 0, 1), 0) # 0 before the first time and after the last
        if self.mode == 'constant':
            w = np.zeros_like(w)
        elif self.mode == 'cosine':
            w = (1 - np.cos(np.pi*w))/2
        return year, i, j, w

    def at(self, year, sim):
        vals = [const_value(v) for v in self.values]
        year, i, j, w = self.weights(year, sim)
        i, j, w = int(i), int(j), float(w)
        if w == 0: # At or outside the times given, or with mode 'constant'
            return vals[i]
        elif self.mode == 'spline':
            import scipy.interpolate as spi
            typed = isinstance(vals[0], ss.TimePar)
            if typed: # Interpolate the values in the units of the first, and give a value of that type
                vals = [v.to(type(vals[0])) for v in vals]
            value = float(spi.CubicSpline(self.years(sim), np.array([v.value if typed else v for v in vals], dtype=float))(year))
            if typed:
                out = sc.dcp(vals[0])
                out.value = value
                return out
            return value
        return vals[i]*(1 - w) + vals[j]*w

    def series(self, years, sim, convert):
        vals = [const_value(v) for v in self.values]
        if self.mode == 'spline' or any(isinstance(v, ss.prob) for v in vals): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        vals = np.array([convert(v) for v in vals], dtype=float) # Converted once, rather than at each time
        year, i, j, w = self.weights(np.asarray(years, dtype=float), sim)
        return vals[i]*(1 - w) + vals[j]*w

    def __deepcopy__(self, memo): # The cached times belong to one sim
        new = object.__new__(type(self))
        new.__dict__.update(times=list(self.times), values=sc.dcp(self.values, memo=memo), mode=self.mode, period=self.period, _years=None)
        return new

    def to_json(self):
        out = dict(timeseries={str(time):ss.presentation.quantity_json(val, None, None) for time,val in zip(self.times, self.values)}, mode=self.mode)
        if self.period is not None:
            out['period'] = ss.presentation.quantity_json(self.period, 'dur', None)
        return out


class seasonal(TimeValue):
    """
    A value with a yearly cycle: `mean × (1 + amplitude × cos(2π(t − peak)))`, with `t` in years

    It can be used wherever a value or a function of the sim can. In a sim with no
    dates, time starts at the beginning of a year.

    Args:
        mean (any): the mean over the year: a number or a typed quantity
        amplitude (float): the size of the cycle relative to the mean (0.3 means from 70% to 130% of it)
        peak (str/float/dur): when the value is highest: a month (its middle is used), a fraction of the year (0 to 1), or a time of year (e.g. `ss.days(142)`)

    Examples:
        ```python
        beta = ss.seasonal(mean=0.3, amplitude=0.4, peak='January')
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> S', dur=10), init_prev=0.01)
        sim = ss.Sim(sis, start='2020-01-01', stop='2023-01-01').run()
        ```
    """
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    def __init__(self, mean=1.0, amplitude=0.0, peak=0.0):
        self.mean = mean
        self.amplitude = amplitude
        self.peak = peak
        if isinstance(peak, str):
            matches = [i for i,month in enumerate(self.months) if month.startswith(peak.lower())]
            if len(matches) != 1 or len(peak) < 3:
                errormsg = f'The peak of ss.seasonal() is a month (e.g. "January") or a fraction of the year, not "{peak}"'
                raise ValueError(errormsg)
            self.peak_frac = (matches[0] + 0.5)/12
        else:
            if isinstance(peak, ss.dur): # A time of year, e.g. ss.days(142)
                self.peak_frac = float(peak.years) % 1
            else:
                self.peak_frac = float(peak)
                if sc.isnumber(peak) and not 0 <= self.peak_frac <= 1:
                    errormsg = f'The peak of ss.seasonal() is a month (e.g. "May") or a fraction of the year from 0 to 1, not {peak}; for a day of the year, give ss.days({peak}) or {peak}/365'
                    raise ValueError(errormsg)
        return

    def __repr__(self):
        return f'seasonal(mean={self.mean}, amplitude={self.amplitude}, peak={self.peak!r})'

    @property
    def typed(self):
        return [self.mean]

    fields = ['mean', 'amplitude']

    def factor(self, year):
        """ The value relative to the mean, at a year or an array of them """
        return np.maximum(0.0, 1 + const_value(self.amplitude)*np.cos(2*np.pi*(year - self.peak_frac)))

    def at(self, year, sim=None):
        return const_value(self.mean)*float(self.factor(year))

    def series(self, years, sim, convert):
        mean = const_value(self.mean)
        if isinstance(mean, ss.prob): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        return convert(mean)*self.factor(np.asarray(years, dtype=float)) # Converted once, rather than at each time

    def to_json(self):
        return dict(seasonal=dict(mean=ss.presentation.quantity_json(self.mean, None, None), amplitude=self.amplitude, peak=self.peak))
