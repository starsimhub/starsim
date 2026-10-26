"""
Define core Sim classes
"""
import os
import inspect
import itertools
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt



def same_value(a, b):
    """ Whether two settings have the same value (e.g. 10000 and 10000.0); False if they cannot be compared to one True or False """
    try:
        return bool(a == b)
    except Exception:
        return False



def given_modules(*inputs):
    """ The modules among a sim's inputs, alone or in lists or dicts (e.g. `diseases=[sir, sis]`) """
    mods = []
    for inp in inputs:
        for val in (inp or {}).values():
            vals = val.values() if isinstance(val, dict) else sc.tolist(val) if isinstance(val, (list, tuple, ss.Module)) else []
            mods += [v for v in vals if isinstance(v, ss.Module)]
    return mods


def declared(obj):
    """ Whether an object is part of a declared model (a transition or an outcome) rather than a module """
    return isinstance(obj, (ss.Transition, ss.outcome, ss.transitions.FlowSum))


def make_disease(arg, names):
    """ Transitions given to the sim (alone or in a list, among modules) as one disease, named from its states (e.g. "sir"), whose results are also at the top of the sim's (`sim.results.I`) """
    items = sc.tolist(arg) if isinstance(arg, (list, tuple)) or declared(arg) else None
    if items is None:
        return arg
    flat = [obj for item in items for obj in (item if isinstance(item, (list, tuple)) else [item])]
    transitions = [obj for obj in flat if declared(obj)]
    if not transitions:
        return arg
    disease = ss.Disease(*transitions, _names=names)
    disease.setattribute('_implicit', True)
    return [obj for obj in flat if not declared(obj)] + [disease]

def num_str(x):
    """ A time as text: a number (or a duration, by its value) without a needless decimal point (100, not 100.0), or a date """
    x = x.value if isinstance(x, ss.TimePar) else x
    return f'{x:n}' if sc.isnumber(x) else str(x)


def time_str(x, unit_name, plural=False):
    """ A time point as text, for what a run prints: a date as a date (without a time of day, e.g. 2020-01-08), a duration with its own unit (e.g. year 2000.02, as on a timeline in calendar years), or a number in the sim's unit (e.g. day 27) """
    if isinstance(x, (ss.date, pd.Timestamp)):
        return str(ss.date(x)).partition(' ')[0]
    if isinstance(x, ss.dur) and not isinstance(x, ss.datedur):
        unit_name = type(x).base.removesuffix('s')
    return f'{unit_name}{"s"*plural} {num_str(x)}'


class Sim(ss.Base):
    """
    The Sim object

    All Starsim simulations run via the Sim class. It is responsible for initializing
    and running all modules and generating results. The sim orchestrates `ss.People`,
    `ss.Network`, `ss.Disease`, `ss.Intervention`, `ss.Demographics`, and other
    modules via the `ss.Loop` integration loop.

    Args:
        model (Module/list/dict): the model: a module or a list of modules, which are sorted by type as for `modules`; transitions, alone or in a list (or in `diseases=`), are one disease, named from its states (e.g. `sim.sir`).
            Alternatively, the sim parameters as an `ss.SimPars` object or a nested dictionary, which can include all other arguments
        pars (dict): the values of model parameters by name, e.g. `pars=dict(beta=0.1)` or `pars=fit.pars` (see `ss.par`), and of sim
            settings by their own names (e.g. `stop`) where no parameter of the model has the name; a name that is neither raises `ss.errors.UnknownParameter`
        label (str): the human-readable name of the simulation
        people (People): if provided, use this `ss.People` object
        demographics (str/Demographics/list/bool): births, deaths, and migration of everyone: declared (`ss.birth()`, `ss.death()`, for every solver), or modules (e.g. `ss.Births()`, `ss.Pregnancy()`), or a list; True for births and deaths at default rates
        connectors (str/Connector/list): as above, for connectors
        networks (str/Network/list): as above, for networks
        interventions (str/Intervention/list): as above, for interventions
        diseases (str/Disease/list): as above, for diseases
        analyzers (str/Analyzer/list): as above, for analyzers
        custom (Module/list): as above, for any custom user-created modules not matching one of the above types
        modules (Module/list): alternatively, supply all modules together and divide among demographics, diseases, etc. based on type
        solver (str): how to run modules with declared states and transitions (see `ss.Disease`): 'auto' (the default: an ODE if
            every part of the model has a compartmental form, otherwise agents), 'ode', 'binomial' (chain-binomial), 'gillespie' (exact,
            one event at a time), 'tau' (tau-leaping: close to exact, and much faster for a run of many events), or 'abm'
        integrator (str): the scheme of the ODE: 'rk4' (the default) or 'euler'
        unit (str): the unit of bare numbers in declared transitions, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        approx (bool): by default, each approximation a solver makes is noted in one line (kept in `sim.notes`, and printed unless `verbose=0`); True does not print them, and False makes them errors
        stratify (dict): stratify every module by a property of people, e.g. `dict(age=[0, 15, 65])`
        data (df): a dataframe (or dict) of data, with a column "time" plus data of the form "module.result", e.g. "hiv.new_infections": plotted with the results, and what `sim.fit()`, `sim.log_density()`, and `sim.gradient()` compare them to
        kwargs (dict): merged with pars; see ss.SimPars for all parameter values

    Note:
        Modules can be supplied either via individual keyword arguments (`diseases`, `networks`, etc.)
        or all together via the `modules` argument. When using `modules`, Starsim automatically sorts
        them by type. The individual kwargs and `modules` can be mixed; they are merged together.

    Examples:
        ```python
        sim = ss.Sim(diseases='sir', networks='random') # Simplest Starsim sim; equivalent to ss.demo()
        sim = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()) # Equivalent using objects instead of strings
        sim = ss.Sim(diseases=['sir', ss.SIS()], networks=['random', 'sexual']) # Example using list inputs; can mix and match types
        sim = ss.Sim(modules=[ss.SIR(), ss.RandomNet()]) # Can supply multiple types of module with the 'modules' argument
        ```
    """
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if cls is not Sim: # A subclass (e.g. FPsim's), which may build its modules after Sim.__init__(): the call is kept, so that sim.new() can make it again
            try:
                obj._call = (sc.dcp(args), sc.dcp(kwargs))
            except Exception:
                obj._call = None
        return obj

    def __init__(self, *models, label=None, people=None, demographics=None, connectors=None,
                 networks=None, diseases=None, interventions=None, analyzers=None, custom=None,
                 modules=None, data=None, pars=None, **kwargs):
        self.pars = ss.SimPars() # Make default parameters (using values from parameters.py)
        models = [arg for arg in models if arg is not None]
        given = dict(label=label, people=people, demographics=demographics, connectors=connectors, networks=networks, diseases=diseases,
                     interventions=interventions, analyzers=analyzers, custom=custom, modules=modules, data=data)
        for arg in models: # The model: modules or transitions, or lists of them; or the sim parameters, as a dict
            if isinstance(arg, dict):
                kwargs = sc.mergedicts({k:v for k,v in arg.items() if given.get(k) is None}, kwargs) # A keyword given outright wins over the same key in the dict
            elif not all(isinstance(obj, ss.Module) or declared(obj) for obj in (arg if isinstance(arg, (list, tuple)) else [arg])):
                errormsg = f'ss.Sim() takes the model (modules or transitions, or lists of them) or a dict of parameters as positional arguments, not {type(arg).__name__}; give everything else by keyword, e.g. label="..."'
                raise TypeError(errormsg)
        mods = dict(demographics=demographics, connectors=connectors, networks=networks, diseases=diseases, interventions=interventions, analyzers=analyzers, custom=custom, modules=modules)
        if any(val is None and key in kwargs for key,val in mods.items()): # Modules given in a dict of parameters (as sim.new() does) are named and ordered as if given by keyword
            mods = {key:(kwargs.pop(key) if val is None and key in kwargs else val) for key,val in mods.items()}
            demographics, connectors, networks, diseases, interventions, analyzers, custom, modules = mods.values()
        names = ss.utils.caller_names(1)
        models = make_disease(models, names)
        diseases = make_disease(diseases, names)
        positional = [arg for arg in models if not isinstance(arg, dict)]
        if positional:
            modules = sc.mergelists(*positional, modules)
        self._name_modules(names, demographics, connectors, networks, interventions, diseases, analyzers, custom, modules)
        interventions = self._order_steps(interventions)
        self.par_values = None # Values of model parameters by name, e.g. pars=dict(beta=0.1) or pars=fit.pars
        self.jax = None # Whether a whole run is taken in JAX: of the ODE (as ss.parallel() does for many runs with different values; see ss.gradients), or of the chain binomial and tau-leaping (see ss.perflib._chain); None to choose (see ss.solvers.choose_engine())
        if isinstance(pars, dict) and len(pars): # Values of the model's parameters by name; a sim setting only by its own name, if no parameter that could be fitted has it (so pars=fit.pars works with a parameter called dur, and pars=dict(dur=10) with a network that has one)
            given = [mod for arg in [modules, demographics, connectors, networks, interventions, diseases, analyzers, custom] for mod in sc.tolist(arg) if isinstance(mod, ss.Module)]
            names = {name for mod in given for name in (mod.pars.keys() if getattr(mod, 'graph', None) is not None else mod.pars.bounded())}
            settings = [key for key in pars if key in self.pars and key not in names]
            values = {key:val for key,val in pars.items() if key not in settings} # A name that is neither raises ss.errors.UnknownParameter when the sim is initialized
            self.par_values, pars = values or None, {key:pars[key] for key in settings} or None
            if pars and pars.get('interventions') is not None: # Modules given in pars (e.g. by a subclass of ss.Sim): in order too, before they are copied
                pars['interventions'] = self._order_steps(pars['interventions'])
        args = dict(label=label, people=people, demographics=demographics, connectors=connectors,
                    networks=networks, interventions=interventions, diseases=diseases, 
                    analyzers=analyzers, custom=custom, modules=modules)
        args = {key:val for key,val in args.items() if val is not None} # Remove None inputs
        for mod in given_modules(pars, args, kwargs): # Marked, so that what is added to them later, which this sim would never see, raises (see ss.Module.check_not_given())
            mod.setattribute('_given', True)
        input_pars = sc.mergedicts(pars, args, kwargs, _copy=True) # Copied together, so the modules given are never run themselves and can be given to other sims, and refer to each other as given
        for mod in given_modules(input_pars): # The sim's own copies are its to change
            mod.setattribute('_given', False)
        for mod in given_modules(pars, args, kwargs): # The modules given hold no results (the sim runs its copies), so reading one says where they are
            if isinstance(getattr(mod, 'results', None), ss.Results) and not len(mod.results):
                mod.results.setattribute('_given_to', mod.name)
        self.pars.update(input_pars)  # Update the parameters
        self.inputs = sc.dcp(input_pars) # Kept so the sim can be rebuilt (see sim.new()), e.g. with other parameter values for calibration
        self.given = [key for key,val in input_pars.items() if val is not None] # Which parameters were supplied, as opposed to defaults
        self.notes = [] # Notes on what was assumed or approximated when the sim was built
        self.solver = None # Set in init()
        self.groups = ss.dimensions.Groups(self) # The groups of people by the stratifications, worked out in init()

        # Set attributes; see also sim.init() for more
        self.created = sc.now()  # The datetime the sim was created
        self.version = ss.__version__ # The Starsim version
        self.metadata = ss.debugtools.metadata()
        self.dists = ss.distributions.Dists(obj=self) # Initialize the random number generator container
        self.loop = ss.Loop(self) # Initialize the integration loop
        self.results = ss.Results(module='Sim')  # For storing results
        self.data = data # For storing input data
        self.initialized = False  # Whether initialization is complete
        self.complete = False  # Whether a simulation has completed running
        self.results_ready = False  # Whether results are ready
        self.elapsed = None # The time required to run
        self.progress = None # How far a run has got, shown as it goes (see ss.presentation.Progress)
        self.summary = None  # For storing a summary of the results
        self.filename = None # Store the filename, if saved
        self.diagnostics = None # Store diagnostics results
        self._made = self._settings() # The settings as made, to find the ones set afterwards (see set_after())
        self._set_after = {}
        self._made_pars = self._module_pars() # The same for the modules' parameters (see pars_after())
        self._pars_after = {}
        return

    def _settings(self):
        """ The sim settings, not the modules or people (which would be found again by the search for distributions) """
        skip = ss.modules.module_types() + ['custom', 'modules', 'people']
        return {key:val for key,val in self.pars.items() if key not in skip}

    def set_after(self):
        """ The sim settings set after the sim was made and before it was initialized, or after it ran (e.g. `sim.pars.stop = 30`), which `sim.new()` keeps """
        made = self.__dict__.get('_made')
        if made is not None:
            for key,val in self._settings().items():
                before = made.get(key)
                plain = isinstance(val, (int, float, str, bool)) and type(val) is type(before) # Compared by value, since a sim run in another process comes back with copies of them
                if val is not before and not (plain and val == before):
                    self._set_after[key] = sc.dcp(val)
        return self.__dict__.get('_set_after', {})

    def _module_pars(self):
        """ The parameters of the modules, by name, as they are now """
        if self.initialized: # The sim's own modules
            mods = list(self.modules)
        else:
            vals = [self.pars.get(key) for key in ss.modules.module_types() + ['custom', 'modules']]
            mods = [mod for val in vals for mod in (val.values() if isinstance(val, dict) else sc.tolist(val))]
        return {mod.name:(mod, {k:(v, repr(v)) for k,v in mod.pars.items()}) for mod in mods if isinstance(mod, ss.Module)} # A value set in place (e.g. an ss.par given a new value) keeps its identity, so its repr is kept too

    def pars_after(self):
        """ The parameters of modules set after the sim was made and before it was initialized, or after it ran (e.g. `sim.pars.diseases.pars.init_prev = 0.5`), which `sim.new()` keeps """
        made = self.__dict__.get('_made_pars')
        if made is not None:
            for key,(mod,before) in made.items():
                changed = {k:sc.dcp(v) for k,v in mod.pars.items() if k not in before or v is not before[k][0] or repr(v) != before[k][1]}
                if changed:
                    self._pars_after[key] = changed
        return self.__dict__.get('_pars_after', {})

    @staticmethod
    def _name_modules(names, *args):
        """ A module built with the plain ss.Disease() constructor takes its name from the variable it is assigned to """
        taken = set(ss.SimPars()) | set(dir(ss.Sim)) # Names the sim already uses, e.g. interventions in def run(interventions): ss.Sim(interventions=interventions)
        for arg in args:
            mods = list(sc.tolist(arg))
            for mod in mods:
                named = [mod] if type(mod) is ss.Disease or isinstance(mod, (ss.modify, ss.Delivery)) else []
                named += [getattr(mod, 'product', None)]*isinstance(mod, ss.Delivery)
                for obj in named:
                    if obj is not None and obj._default_name and not getattr(obj, '_from_keywords', False): # A product its delivery made is named for the delivery: see ss.SimPars.validate_name()
                        name = getattr(obj, '_assigned', None) or names.get(id(obj)) # The variable it was first assigned to, else the one that holds it here
                        if name in taken:
                            name = None
                        if name is None and getattr(obj, 'graph', None) is not None: # Not bound to one variable, e.g. built in a helper: named from its states instead, e.g. 'sir'
                            name = ''.join(s.name[0].lower() for s in obj.graph.people_states) or None
                            obj._implicit = name is not None
                        if name is not None:
                            obj.name = obj.label = name
                            obj._default_name = False
        return

    @staticmethod
    def _order_steps(interventions):
        """ The deliveries run in the order of their triggers (see `ss.interventions.order_steps()`) """
        if not any(isinstance(mod, ss.Delivery) for mod in sc.tolist(interventions)):
            return interventions
        return ss.interventions.order_steps(sc.tolist(interventions))

    def _named_modules(self, attr):
        """ The modules called `attr`: the sim's own once it is initialized, else those it was given """
        if self.__dict__.get('initialized'):
            mods = self.modules
        else:
            mods = given_modules(self.__dict__.get('pars'))
        return [mod for mod in mods if mod.name == attr]

    def __getattr__(self, attr):
        """ A module by its name, e.g. `sim.sir` for `sim.diseases.sir` """
        if not attr.startswith('_') and 'pars' in self.__dict__ and attr not in ss.modules.module_types() + ['custom']: # Not the module types, which init() sets
            mods = self._named_modules(attr)
            if len(mods) == 1:
                return mods[0]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, key):
        """ Allow dict-like access, e.g. sim['diseases'] """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """ Allow dict-like access, e.g. sim['created'] = sc.now() """
        return setattr(self, key, value)

    def __repr__(self):
        """ Show a quick version of the sim """
        # Try a more custom repr first
        try:
            labelstr = f'{self.label}; ' if self.label else ''
            pars = self.pars
            n = int(sc.ifelse(pars.total_pop, pars.n_agents)) # The number of people modelled, not the default number of agents (or the number of groups of a compartmental run)
            agentstr = ''
            if self.initialized and getattr(self, 'solver', None) == 'abm':
                agents = int(pars.n_agents)
                agentstr = f' ({agents:n} agents)' if agents != n else ''
            if self.initialized:
                timestr = f'{self.t.start}—{self.t.stop}'
            else:
                timestr = f'{self.pars.start}—{self.pars.stop}'

            moddict = {}
            for modkey in ss.modules.module_map().keys():
                if hasattr(self, modkey):
                    thismodtype = self[modkey]
                elif modkey in self.pars:
                    thismodtype = self.pars[modkey]
                else:
                    thismodtype = {}

                if isinstance(thismodtype, dict) and len(thismodtype):
                    moddict[modkey] = sc.strjoin(thismodtype.keys())
                elif isinstance(thismodtype, str):
                    moddict[modkey] = thismodtype

            if len(moddict):
                modulestr = ''
                for k,mstr in moddict.items():
                    modulestr += f'; {k}={mstr}'
            else:
                modulestr = ''
            if not self.initialized:
                modulestr += '; not initialized'

            string = f'Sim({labelstr}n={n:n}{agentstr}; {timestr}{modulestr})'

        # Or just use default
        except Exception as E:
            ss.warn(f'Error displaying custom sim repr, falling back to default: {E}')
            string = sc.prepr(self, vals=False)

        return string

    def products(self):
        """ List all products across interventions; not an ndict like the other module types """
        products = []
        for intv in self.interventions():
            if intv.has_product and intv.product not in products:
                products.append(intv.product)
        return products

    def all_networks(self):
        """ The networks, by name, including those that a compartmental run replaces with contact rates (which are not run themselves) """
        return {net.name:net for net in list(self.networks.values()) + getattr(self, 'mixed', [])}

    @property
    def modules(self):
        """
        Return an iterator over all Module instances (stored in standard places) in the Sim

        This is the order they are initialized in: custom modules come after the others they
        may read (TL54: a custom module came first, so it read a demography's properties before
        they were set), and before the analyzers. Their step() still runs first: see `ss.Loop`.
        """
        for module in itertools.chain(
            self.demographics(),
            self.connectors(),
            self.networks(),
            self.interventions(),
            self.products(),
            self.diseases(),
            self.custom(),
            self.analyzers(),
            ):
            yield module

    def init(self, force=False, timer=False, **kwargs):
        """
        Perform all initializations for the sim

        Args:
            force (bool): whether to overwrite sim attributes even if they already exist
            timer (bool): if True, count the time required for initialization (otherwise just count run time)
            kwargs (dict): passed to ss.People()
        """
        if self.initialized and not force: # E.g. after sim.describe(), which initializes it
            return self

        # Handle the timer
        self.timer = sc.timer() # Store a timer for keeping track of how long the run takes
        if timer:
            self.timer.start()
        self._held = [] if self.pars.verbose else None # Notes are said after the line that starts the run

        # Validation and initialization -- this is "pre"
        made = self.__dict__.get('_made') or {}
        for key,val in self.set_after().items(): # Before the settings are converted
            if val is not None and key not in self.given and not same_value(val, made.get(key)): # A setting set before init() was given too, e.g. sim.pars.use_aging = True (but not one made a float, e.g. n_agents)
                self.given.append(key)
        self.pars_after()
        self._made = self._made_pars = None # And only then
        np.random.seed(self.pars.rand_seed) # Reset the seed before the population is created -- shouldn't matter if only using Dist objects
        self.pars.lock(False) # Allow parameters to be added during initialization
        self.pars.validate() # Validate parameters
        ss.solvers.choose_solver(self) # Build the declared model, and choose the unit and the solver
        self._init_time() # Initialize time
        if 'use_aging' not in self.given and not self.pars.use_aging and any(strat.kind == 'age' and strat.edges is not None for strat in self.groups.strats):
            long = self.t.dt_year*(self.t.npts - 1) >= 1 # Only then does it matter much
            ss.note('people do not age between the age groups, since nobody is born or dies; pass use_aging=True to age them', sim=self, verbose=None if long else 0)
        self.init_people(**kwargs) # Initialize the people
        self.init_module_attrs(force=force) # Load specified modules from parameters, initialize them, and move them to the Sim object
        self.init_data(check=False) # Load the data before the modules, which read it (e.g. data by group keeps their results by group at every step)
        self._init_modules_pre()
        ss.interventions.check_undriven(self)
        ss.connectors.check_applied(self) # Nothing accepted, then ignored

        # Final initializations -- this is "post"
        if self.pars.rescale:
            self._init_rescale()
        self._init_dists() # Initialize distributions
        self._init_people_vals() # Initialize the values in all the states and networks
        ss.solvers.init_group_values(self) # The people's levels of each dimension, or each group's age and sex
        self._init_modules_post() # Initialize the module values
        ss.solvers.settle_start(self) # The declared modules' people, once every module has its own
        self.init_results() # Initialize the results
        self.init_data() # Initialize the data
        self.loop.init() # Initialize the integration loop
        self.lock_pars() # Don't allow new parameters to be added by mistake

        self.verbose = self.pars.verbose # Store a run-specific value of verbose
        ss.solvers.group_people(self) # In a compartmental run, the people are groups

        # It's initialized
        self.initialized = True
        held, self._held = self._held, None
        if self.verbose and self.__dict__.pop('_quiet', False): # Initialized by sim.describe(): said when it is run
            self._unsaid = held
        elif self.verbose:
            print('\n'.join(self.start_lines(held)))
        if timer:
            self.elapsed = self.timer.toc(label='init', output=True, verbose=self.verbose)
        if self.diagnostics and self.diagnostics.states is not None: # Need not None since dict is empty at this point
            self.diagnostics.store_states(key='init')
        return self

    def start_lines(self, notes=()):
        """ What a run says as it starts: what is run and how, R0 (as a check that the model and its seed are what was meant), and the notes """
        t = self.t
        mods = list(self.diseases.keys()) or [mod.name for mod in self.modules]
        when = f'{time_str(t.tvec[0], None)} to {time_str(t.tvec[-1], None)}' if t.is_absolute else f'{time_str(t.tvec[0], self.unit_name, plural=True)} to {num_str(t.tvec[-1])}' # On the timeline's own axis, though stop may be given in another unit (e.g. ss.weeks(8))
        stop = t.stop.to_year() if isinstance(t.stop, ss.date) else t.stop.years if isinstance(t.stop, ss.dur) else None
        if stop is not None and stop - t.yearvec[-1] > 1e-6: # The stop is not a whole number of steps from the start, so the run ends at the last step before it
            given = time_str(t.stop, None) if t.is_absolute else num_str(type(t.tvec[-1])(t.stop) if isinstance(t.tvec[-1], ss.dur) else t.stop)
            when += f' (the last whole step before {given})'
        if self.solver == 'abm':
            rounded = float(f'{self.pars.pop_scale:.3g}') # To 3 significant figures, without an exponent (e.g. 25,300)
            scale = '' if rounded == 1 else f', each {rounded:,.0f} people' if rounded >= 10 else f', each {rounded:g} people'
            how = f'as {len(self.people):,} agents{scale}'
        else:
            kind = dict(ode='an ODE', binomial='a stochastic compartmental model (chain binomial)', gillespie='a stochastic compartmental model (Gillespie)', tau='a stochastic compartmental model (tau-leaping)')
            how = f'{self.pars.total_pop:,.0f} people, as {kind.get(self.solver, self.solver)}'
        label = f' "{self.label}"' if self.label else ''
        what = f'{sc.strjoin(mods)}, ' if mods else ''
        lines = [ss.presentation.style(f'Sim{label}: {what}{how}, {when}', 'bold', color=ss.presentation.in_color())]
        if self.solver != 'abm':
            for dis in self.declared:
                sources = dis.graph.infectious or dis.graph.density_sources
                if dis.graph.transmissions or sources:
                    try:
                        ngm, ngm_start = dis.engine.next_generation(), dis.engine.next_generation(start=True) # In a population with no immunity, and as it starts
                    except Exception: # Not every model has one; the line is only a check
                        ngm = None
                    if ngm is not None:
                        R0, R = [float(np.max(np.abs(np.linalg.eigvals(m)))) for m in [ngm, ngm_start]]
                        count = float(dis.engine.count(sources))
                        shown = f'{count:,.0f}' if count >= 100 or count == round(count) else f'{count:.3g}' # A share, with total_pop=1, is not 0
                        at_start = ''
                        if abs(R - R0) > 0.01*R0: # Some start immune
                            share = float(dis.engine.count([dis.graph.first]))/float(dis.engine.count(dis.graph.people_states))
                            at_start = f' (R = {R:.3g} at the start, with {share:.0%} susceptible)'
                        lines.append(f'  {dis.name}: R0 = {R0:.3g}{at_start}, with {shown} infectious at the start')
        lines += [f'  Note: {note}' for note in notes]
        return lines

    def end_line(self):
        """ What a run says when it ends: how long it took, and for each disease with a transmission, its infections and their peak """
        parts = []
        for dis in getattr(self, 'declared', []):
            trs = dis.graph.transmissions or (dis.graph.acquisitions if dis.graph.density_sources else [])
            if trs and all(tr.name in dis.results for tr in trs):
                total = sum(dis.results[tr.name].values.sum() for tr in trs)
                infectious = sum(dis.results[state.name].values for state in dis.graph.infectious or dis.graph.density_sources if state.name in dis.results)
                if np.ndim(infectious):
                    peak = int(np.argmax(infectious))
                    tvec = dis.t.tvec if len(dis.t.tvec) == len(infectious) else self.t.tvec # On the disease's own timeline, if it has one
                    when = time_str(tvec[peak], self.unit_name) # A date, or e.g. day 27, as in the first line
                    parts.append(f'{dis.name}: {total:,.0f} infections; at most {infectious[peak]:,.0f} infectious, on {when}')
        took = f'{self.elapsed:.2g}' if self.elapsed < 1 else f'{self.elapsed:.1f}'
        return f'  Done in {took} s' + (f'; {"; ".join(parts)}' if parts else '')

    def new(self, pars=None, inplace=False, **kwargs):
        """
        A new, uninitialized sim built from the same inputs as this one, optionally with other values

        A sim that has run is run again this way, so it is seldom needed: `sim.run()` runs a
        finished sim again from the start, and `sim.run(solver='abm')` or `sim.run(n_runs=10)`
        build what they need themselves.

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            inplace (bool): if True, this sim becomes the new one, rather than a new one being returned
            kwargs (dict): sim arguments to change, e.g. `rand_seed=2` or `solver='abm'`; `None` goes back to the default, e.g. `n_agents=None` for a compartmental run of an agent-based sim; also `data=`. A new `solver=` puts `substeps=` back to that solver's default unless it is given too, since substeps mean a different thing to each (e.g. an importer's `substeps=1`, whole steps for the chain binomial, would be one Runge-Kutta step of the whole timestep for the ODE)
        """
        if not hasattr(self, 'inputs'):
            errormsg = 'sim.new() needs the inputs the sim was built from, which this sim (e.g. one saved by an earlier version) does not have'
            raise ValueError(errormsg)
        if kwargs.get('solver') is not None and kwargs['solver'] != self.pars.solver and 'substeps' not in kwargs:
            kwargs['substeps'] = None
        data = sc.dcp(kwargs.pop('data', self.data)) # Data is not a sim parameter, so it is passed on separately
        call = self.__dict__.get('_call')
        if type(self) is Sim:
            inputs = sc.mergedicts(sc.dcp(self.inputs), kwargs)
            inputs = {key:val for key,val in inputs.items() if not (key in kwargs and val is None)} # As if it had not been given
            args, kw = [inputs], dict(data=data)
        else: # A subclass is made again by the call that made it
            if call is None:
                errormsg = f'sim.new() cannot make this {type(self).__name__} again, since the arguments it was made with were not kept (its arguments cannot be copied). Make it again by calling {type(self).__name__}(...)'
                raise ValueError(errormsg)
            args, kw = sc.dcp(call)
            kw = sc.mergedicts(kw, kwargs)
            kw = {key:val for key,val in kw.items() if not (key in kwargs and val is None)}
        set_after, pars_after = self.set_after(), self.pars_after()
        par_values, jax, imported = sc.mergedicts(self.par_values, pars), self.jax, self.__dict__.get('imported')
        if inplace: # This sim becomes the new one
            sim = self
            vars(sim).clear()
            if call is not None:
                sim._call = call
            type(sim).__init__(sim, *args, **kw)
        else:
            sim = type(self)(*args, **kw)
        if type(self) is not Sim:
            sim.data = data
        for key,val in set_after.items(): # Settings changed after the sim was made, e.g. sim.pars.stop = 30
            if key not in kwargs:
                sim.pars[key] = sc.dcp(val)
        for name,changed in pars_after.items(): # Parameters of modules changed after the sim was made
            target = sim._module_pars().get(name)
            if target is not None:
                target[0].pars.update(sc.dcp(changed))
        sim.par_values = par_values
        sim.jax = jax
        if imported is not None: # The same import, with what it did not translate
            sim.imported = imported
        return sim

    def fit(self, pars=None, data=None, weights=None, likelihood=None, gof=None, method=None, n_runs=None, rand_seed=None, verbose=None, **kwargs):
        """
        Fit every parameter that has bounds to every result that has data

        Fits every parameter that has `bounds` (see `ss.par`) to every result that has
        data (see `ss.Sim(data=...)`). The original sim is not changed. The fitting itself
        is in `ss.fit`, which is imported when it is first used.

        Args:
            pars (list): the names of the parameters to fit (default: all that have bounds)
            data (df): the data to fit to (default: the sim's data)
            weights (dict): the relative weight of each data series (default: equal)
            likelihood (dict): the likelihood of a data series, by its name (e.g. `dict(prevalence=ss.normal(scale=0.02))`), for series whose outcome does not declare one. If every series has a likelihood, the fit maximizes their sum (plus the log of the priors of the parameters that have one, for the methods that find a best fit: the most probable values)
            gof (str/func): the goodness of fit: 'nae' (normalized absolute error, the default), 'rmse', or a function of the sim; any other name is an error
            method (str/list): by default, a best fit (a point estimate), by the method that suits the sim, which is printed with the reason: 'l-bfgs-b' for a deterministic sim with gradients (from JAX; see `ss.gradients`), 'nelder-mead' for one without them, or with one or two fitted parameters (where it is faster than gradients), and 'optuna' for a stochastic sim, including an agent-based one. A posterior is asked for by name: 'nuts' (the No-U-Turn sampler, for a deterministic sim with gradients whose data series all have a likelihood, with each chain started from one of the best points of a scan over the bounds that are apart), 'pmcmc' (particle MCMC, for a stochastic compartmental sim whose data series all have a likelihood: a random-walk Metropolis sampler on the likelihood estimated by a particle filter; see `ss.fit.ParticleFilter`), or 'abc-smc' (approximate Bayesian computation by sequential Monte Carlo, for any sim, with no likelihood needed); the priors are the parameters' `prior=`, or uniform over their bounds. Also: 'if2' (iterated filtering, for the sims that 'pmcmc' fits: the maximum-likelihood values, from passes of a particle filter in which each particle's values are perturbed by a shrinking random walk; see `ss.fit.Swarm`), 'asd' (adaptive stochastic descent, `sc.asd()`), any other `scipy.optimize.minimize` method, or a list of methods, each started from the best fit of the one before (a stage that does not converge stops the list; see `ss.fit.fit_stages()`)
            scan (int): for the SciPy methods, NUTS, and particle MCMC, the number of points spread over the bounds to try first (default: 10 per parameter, at most 100); the search starts from each of the best `restarts=3` of them that are apart (for SciPy; 1 for a single parameter of a deterministic sim), each NUTS chain from one of the best that are apart, and particle MCMC from the best
            n_runs (int): runs per evaluation, for a stochastic sim (default 4 for a stochastic compartmental sim, and 1 for agents and for a deterministic one); every point is scored by the mean over the same seeds (common random numbers), except under Optuna, where each point takes fresh seeds so that its search averages over chance; `fit.mismatch` is the score at the best fit over the first seeds
            rand_seed (int): the random seed for the fitting method (default: the sim's, so a fit gives the same answer each time)
            verbose (bool): whether to print progress
            kwargs (dict): passed to the fitting method, e.g. `trials=200` for Optuna (and `n_workers=4` to run trials in 4 processes at once, and `storage='fit.log'` to keep the trials in a file, so that calling it again resumes the fit; see `ss.fit.fit_optuna()`), `maxiter=500` for SciPy, `draws=500`, `warmup=500`, and `chains=4` for NUTS (the defaults; 1,000 draws and 1,000 warm-up draws for particle MCMC, whose draws are more correlated), `particles=` for particle MCMC (default: chosen by each chain in its warm-up; see `ss.fit.pmmh()`), or `parallel=False` to run its chains one after another; for IF2, `particles=1000`, `iterations=50` (passes of the filter), `cooling=0.5` (the share of the first size of the random walk's steps that is left after 50 passes, as pomp's `cooling.fraction.50`), `sd=0.05` (that first size, on the scale searched on: the logit of each value's place between its bounds, of its log for wide bounds), and `chains=4` (each from its own start: the best of the initial values and a spread of points over the bounds, by the filter)

        Returns:
            An `ss.fit.Fit`, with the calibrated sim as `fit.sim` and the best-fit values as `fit.pars`

        Examples:
            ```python
            beta = ss.par(0.1, bounds=[0.01, 0.5])
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(10), init_prev=0.01)
            sim = ss.Sim(sir, data='cases.csv', start='2020-01-01', stop='2020-06-01')
            fit = sim.fit() # A best fit
            post = sim.fit(method='nuts') # A posterior, given a likelihood for each data series
            fit.sim.plot()
            ```
        """
        from . import fit
        if isinstance(method, (list, tuple)): # Staged fitting: each method starts from the best fit of the one before
            return fit.fit_stages(self, method, pars=pars, data=data, weights=weights, likelihood=likelihood, gof=gof, n_runs=n_runs, rand_seed=rand_seed, verbose=verbose, **kwargs)
        if method is not None:
            method = method.lower()
            if method not in fit.methods:
                errormsg = f'Unknown fitting method method="{method}"; the choices are {sc.strjoin(fit.methods.keys())}'
                raise ValueError(errormsg)
        problem = fit.Problem(self, pars=pars, data=data, weights=weights, likelihood=likelihood, gof=gof, n_runs=n_runs, rand_seed=rand_seed, verbose=verbose)
        method = problem.choose(method)
        if method != 'optuna' and ('n_workers' in kwargs or 'storage' in kwargs):
            errormsg = f'n_workers= and storage= are for method="optuna", not method="{method}"'
            raise ValueError(errormsg)
        problem.start()
        out = fit.methods[method](problem, **kwargs)
        return problem.result(method, out)

    def sensitivity(self, **kwargs):
        """ Which parameters matter: partial rank correlations of each parameter with bounds and each result, over samples within the bounds; see `ss.fit.sensitivity()` """
        from . import fit
        return fit.sensitivity(self, **kwargs)

    def check_dt(self, factor=2, output=False):
        """
        Rerun the sim with a smaller timestep, and report how much each result changes

        In a run that takes whole steps this covers everything that depends on the size
        of the step: turning rates into probabilities, the delay of a step between modules,
        and the order of the modules. In a stochastic run the change also includes the
        difference between two runs.

        Args:
            factor (int): how many times smaller to make the timestep
            output (bool): return the table (as a dataframe) instead of printing it

        Examples:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=4), init_prev=0.01)
            ss.Sim(sir, solver='binomial', unit='day', stop=100).check_dt()
            ```
        """
        return ss.presentation.check_dt(self, factor=factor, output=output)

    def what_modifies(self, target, output=False):
        """
        Print everything that modifies a quantity: each `ss.modify()` and `ss.change()`, and the efficacy of each product

        Args:
            target (par/str): the quantity, as for `ss.modify()`: e.g. `sir.pars.beta`, `sir.rel_sus`, or `'sir.pars.beta'`
            output (bool): return the text instead of printing it
        """
        return ss.presentation.what_modifies(self, target, output=output)

    def stratify(self, **kwargs):
        """
        Stratify every module in the sim by a property of people, e.g. `sim.stratify(age=[0, 15, 65])`

        Takes the same arguments as `ss.Disease.stratify()`, and returns the sim.
        """
        if self.initialized:
            errormsg = 'Cannot stratify a sim that has already been initialized; stratify it first'
            raise RuntimeError(errormsg)
        if repr(kwargs) in [repr(call) for call in sc.tolist(self.pars.stratify)]: # The same line run twice
            ss.note(f'the sim is already stratified by {sc.strjoin(kwargs)} in the same way, so this stratify() does nothing', sim=self)
            return self
        self.pars.stratify = sc.mergelists(self.pars.stratify, kwargs)
        if hasattr(self, 'inputs'): # So that sim.new() is stratified too
            self.inputs['stratify'] = sc.dcp(self.pars.stratify)
        return self

    @property
    def unit_name(self):
        """ The unit of time, for what the sim says: its unit, or, with none, "step" (or "unit", if a step is not one unit) """
        if self.unit:
            return self.unit
        dt = self.pars.dt
        dt = dt.value if isinstance(dt, ss.dur) else dt
        return 'step' if dt is None or dt == 1 else 'unit'

    def has_demography(self):
        """ Whether people are born or die other than of a disease: demographics modules, a declared birth, arrival, or death from every state """
        return bool(self.pars.demographics) or any(ss.solvers.flows_of(mod) for mod in self.declared)

    def pyramid(self, strat=None):
        """
        The people by 5-year age group (their lower edges, and the number in each) that the sim's ages come from:
        a named population's, or, as a guess, the world's; None if there is no named population and the split of
        the age groups was given

        Args:
            strat (Strat): the age stratification, if any
        """
        named = getattr(self, 'named_people', None)
        if named is not None:
            return named.ages, named.counts
        elif getattr(self, 'given_pyramid', None) is not None:
            return self.given_pyramid
        elif strat is None or strat.guessed.get('split') == ss.data.world.guess:
            return ss.data.world.ages, ss.data.world.counts
        return None

    def get_declared(self, name):
        """ A module with declared states and transitions, by name """
        mods = {mod.name:mod for mod in self.declared}
        if name not in mods:
            errormsg = f'No module named "{name}" with declared transitions was found; these are {sc.strjoin(mods.keys())}'
            raise KeyError(errormsg)
        return mods[name]

    def describe(self, output=False, verbose=False):
        """
        Print the resolved model: every state, transition, and default, with each guess marked

        In a terminal it is in color, and in a notebook it is shown as tables.

        Args:
            output (bool): if True, return the text rather than printing it
            verbose (bool): also show what each rate and dwell time is per step: the probability of leaving in a step, as the solvers that take whole steps use it, and the stages of a dwell time
        """
        return ss.presentation.describe_sim(self, output=output, verbose=verbose)

    def hash(self):
        """
        The model's fingerprint: short hashes of its structure, its parameter values, the run settings, the scenario, and the data, plus the seed

        Two sims with the same `model` hash have the same states, transitions, and
        modifiers (including the source of any function in them); the same `pars` hash,
        the same values. `config` covers the run settings and the numerical options
        (e.g. `ss.options.crn`), `scenario` the interventions, and `data` the data
        attached for fitting. This tells apart the many runs that come from one script
        (a sweep, a calibration), and tests whether two models written differently are
        the same model. Each is SHA-256 of the canonical model spec (`sim.to_spec()`), so it is
        the same in every process and on every machine. A sim that is not initialized
        is hashed without initializing it. Running and fitting a sim do not need it.

        A function is known by its source text and the plain values it uses (see
        `ss.presentation.func_json()`), or by where it can be imported from if its source
        is not available. User code can give its own identity, which is used in place
        of the source: a `version` attribute on a function (`fn.version = '2'`) or on
        the class of a module written as code. A module written as code is seen through
        its class and its parameters, not its code (see `ss.Module.to_spec()`). A function
        with none of these (e.g. a `functools.partial`) cannot be told apart from another,
        so the hash raises an error rather than treat two such sims as the same model.
        """
        return ss.spec.sim_hash(self)

    def to_spec(self, filename=None, **kwargs):
        """
        The model spec: the model (modules, networks, stratifications, interventions, and run settings) as plain, canonical data

        The spec is the normalized form of the model that `sim.hash()` fingerprints and
        `ss.diff()` compares, and that `sim.to_code()` writes as code for the declared parts.
        It holds no results and no state of a run, and is not read back into a sim. For
        the results as JSON, see `sim.to_json()`; to save a sim with its state and results
        (including partway through a run, to continue it), see `sim.save()`.

        A sim that is not initialized is described without initializing it, as for `sim.hash()`.

        Args:
            filename (str): if provided, save to this file
            kwargs (dict): passed to `sc.savejson()`
        """
        return ss.spec.sim_spec(self, filename=filename, **kwargs)

    def to_code(self, filename=None):
        """
        The sim as a Python file that builds it again, from its model spec (see `ss.to_code()`)

        Only the declared parts are written; what is not (e.g. a module written as code)
        is listed in the file and in an `ss.errors.LossyExport` warning. A sim imported from another framework (`ss.interop.load(..., format=...)`) also says
        where it came from, how each rate was read, and what was not translated.

        Args:
            filename (str): if provided, write the code to this file
        """
        return ss.spec.sim_code(self, filename=filename)

    def add_module(self, module, key=None):
        """
        Add a module to a sim that has already been initialized, and possibly run part-way

        Usually modules are supplied when the sim is created; use this method to add
        one later, e.g. an intervention after running the sim to a certain date. It
        should be called between timesteps (not from within a module's `step()`). The
        module's results cover the whole sim, but are only filled in from the current
        timestep onwards.

        Args:
            module (`ss.Module`): the module to add
            key (str): the type of module, e.g. "interventions" (if None, determine from the module's class)

        Examples:
            ```python
            sim = ss.Sim(diseases='sis', networks='random', start=2000)
            sim.run(until='2010-01-01')
            sim.add_module(ss.change(sim.diseases.sis.pars.beta, at='2012-01-01', by=0.5)) # Not a delivery of a new product, which changes the model's structure
            sim.run()
            ```
        """
        if not self.initialized:
            errormsg = 'The sim must be initialized before calling sim.add_module(); to add a module before initialization, include it when creating the sim'
            raise RuntimeError(errormsg)
        if self.complete:
            errormsg = f'Cannot add module "{module.name}" to a sim that has finished running'
            raise ss.AlreadyRunError(errormsg)

        # Figure out the type of module, and make sure the name is unique
        if key is None:
            key = 'custom'
            for modkey, modclass in ss.modules.module_map().items():
                if isinstance(module, modclass):
                    key = modkey
                    break
        names = {mod.name:None for mod in self.modules}
        ss.SimPars.validate_name(module, key, names)
        modules = [module]
        if getattr(module, 'has_product', False):
            same = [p for p in self.products() if p is module.product or p.name == module.product.name]
            if same: # The sim's copy of a product it already has
                module.product = same[0]
            else:
                errormsg = (f'Cannot add "{module.name}" after the sim is initialized: holding its product "{module.product.name}" is a dimension of the model ("{module.product.dim}"), '
                            f'which is laid out at sim.init(). Give it when creating the sim, with a start date (e.g. at=[...] or start=), or compare with sim.new(interventions=...).')
                raise ss.CapabilityError(errormsg)
        getattr(self, key).append(module) # Not self[key], since subclasses may override __getitem__
        self.pars[key][module.name] = module.pars

        # Initialize the module(s), as in sim.init()
        for mod in modules:
            mod.init_pre(self)
        self.dists.init(obj=self) # Only initializes the new distributions
        for mod in modules:
            self.dists.copy_to_module(mod)
            mod.init_post()
            mod.t.ti = np.searchsorted(mod.t.yearvec, self.t.yearvec[self.ti]) # Start the module at the current time point

        # Add the module to the loop, and resume from the same place
        loop = self.loop
        current = loop.plan[loop.index]
        loop.init()
        loop.index = sc.findfirst([(entry.func == current.func and entry.ti == current.ti) for entry in loop.plan])
        self.lock_pars()
        return module

    def get_module(self, query, die=True, match_case=False):
        """
        Retrieve a single module

        This method is guaranteed to return a single module if it succeeds, so can be relied on in places
        where this is expected to retrieve a specific module from the `Sim` (and retrieving no modules or
        multiple modules would be unexpected). Use `Sim.get_modules()` to retrieve all matching modules.

        Args:
            query (type/str): Query input supported by `get_modules` (either a `Type` e.g., `ss.SIR`) or a string that will match module names.
            die (bool): if True (default), raises an exception unless exactly 1 module is found; else return None if no modules found
            match_case (bool): if `query` is a string, whether to match case (default False)

        Returns:
            An ss.Module instance
        """
        matches = self.get_modules(query, match_case=match_case)
        if len(matches) > 1:
            errormsg = f'Multiple matching modules found for {query}; to retrieve all of them, use `Sim.get_modules()` instead'
            raise ValueError(errormsg)
        elif not matches:
            if die:
                errormsg = f'No matching module found for {query}; set die=False to return None instead'
                raise KeyError(errormsg)
            else:
                return None
        else:
            return matches[0]

    def get_modules(self, query=None, match_case=False, as_dict=False):
        """
        Retrieve modules from the Sim

        Retrieve a module by name or by type. The query term can be the type of module to match
        e.g., `ss.SIR`, `ss.Disease`, `ss.Intervention`, or a string which is tested against
        module names. The string can contain a `'*'` at the start or end.

        If the `Sim` is not initialized, this will search over any `ss.Module` instances contained in
        `Sim.pars`. Note that sim initialization might result in the creation of modules, therefore some modules
        may be retrieved after initialization but not before (e.g. `diseases='sir'` will not be registered
        as the module `ss.SIR()` until after initialization).

        Args:
            query (type/str): The module type (e.g. `ss.SIR`) or a case-insensitive string (e.g. `'sir'`); if None, return all modules
            match_case (bool): if `query` is a string, whether to match case (default False)
            as_dict (bool): only used if query=None, in which case return a dict rather than list of modules

        Returns:
            A list of `ss.Module` instances. The list will be empty if no modules were found
        """
        matches = []

        if self.initialized:
            modules = self.modules
        else:
            modules = sc.search(self.pars, type=ss.Module).values()

        # Return all modules if no query
        if query is None:
            if as_dict:
                return ss.utils.nlist_as_dict(modules)
            else:
                return list(modules)

        # Loop over all modules, looking for matches
        if isinstance(query, str) and not match_case:
            query = query.lower()
        for mod in modules:
            if isinstance(query, type) and isinstance(mod, query):
                matches.append(mod)
            elif isinstance(query, str):
                name = mod.name if match_case else mod.name.lower()
                if query.startswith('*') and query.endswith('*') and query[1:-1] in name:
                    matches.append(mod)
                elif query.startswith('*') and name.endswith(query[1:]):
                    matches.append(mod)
                elif query.endswith('*') and name.startswith(query[:-1]):
                    matches.append(mod)
                elif name == query:
                    matches.append(mod)
        return matches

    def _init_time(self):
        """ Time indexing; derived values live in the sim rather than in the pars """
        self.t = ss.Timeline(name='sim')
        self.t.init(self)
        if self.t.npts > 3001 and 'dt' not in self.given: # The step was not asked for
            why = f"the model's unit, from {self.unit_source}" if self.unit_source else f"unit='{self.pars.unit}'" if self.pars.unit is not None else "the model's unit" # The unit given (source None) is said as given
            ss.warn(f'the sim has {self.t.npts:n} steps of one {self.unit} ({why}); give dt= for a longer step, e.g. dt=ss.weeks(1)', ss.errors.UnitInferred)
        self.results.timevec = self.timevec # Store the timevec in the results for plotting
        return

    def init_people(self, **kwargs):
        """
        Initialize people within the sim
        Sometimes the people are provided, in which case this just adds a few sim properties to them.
        Other time people are not provided and this method makes them.

        Args:
            kwargs  (dict): passed to ss.People()
        """
        # Handle inputs
        people = self.pars.pop('people')
        if (self.named_people is not None or self.given_pyramid is not None) and self.solver != 'abm': # A compartmental run takes the size and age split of a population, not its agents
            people = None
        n_agents = self.pars.n_agents

        # If people have not been supplied, make them -- typical use case
        if people is None:
            if self.solver == 'abm' and self.groups.strats: # Properties that people are stratified by, other than age
                extra = [strat.people_arr() for strat in self.groups.strats if strat.kind != 'age']
                kwargs['extra_states'] = sc.mergelists(kwargs.get('extra_states'), extra)
                for strat in self.groups.strats: # Ages follow the split of the age groups, so every solver starts from the same population
                    if strat.kind == 'age' and strat.edges is not None and 'age_data' not in kwargs and (self.pyramid(strat) is None or strat.edges[0] > 0): # (Nobody is younger than the first group)
                        top = max(100, strat.edges[-1] + 10)
                        kwargs['age_data'] = np.column_stack([np.append(strat.edges, top), np.append(strat.split, 0)])
                    elif strat.female is not None and 'p_female' not in kwargs: # A sex dimension is people.female, so its split is the share of women
                        kwargs['p_female'] = strat.split[strat.female]
            if 'age_data' not in kwargs: # Otherwise the ages are a named population's, or the world's, as a guess (as is the split of age groups that are not given one)
                kwargs['age_data'] = ss.data.to_age_data(*self.pyramid())
            people = ss.People(n_agents=n_agents, **kwargs)  # This just assigns UIDs and length
        elif self.solver == 'abm':
            self.groups.kept = {strat.name for strat in self.groups.strats if strat.kind != 'age' and strat.female is None and strat.attr in people.states} # People who come with a property keep it: it is binned into the levels, not drawn from the split
            for strat in self.groups.strats:
                if strat.name in self.groups.kept and strat.split_given:
                    raise ss.errors.BadSplit(f'"{strat.name}" is a property the people already have, so their own values are kept and binned into its levels; split= would overwrite them. Leave out split=, or set people.{strat.attr} instead.')
                if strat.name in self.groups.kept:
                    strat.guessed.pop('split', None) # Not "everyone at the first level" (TL3): the people's own values are used
                if strat.attr not in people.states:
                    state = strat.people_arr()
                    people.states.append(state)
                    setattr(people, strat.attr, state)
                    state.link_people(people)

        # Finish up (NB: the People object is not yet initialized)
        self.people = people
        self.people.link_sim(self)
        return self.people

    @property
    def label(self):
        """
        Get the sim label from the parameters, if available.

        Note: for `ss.Sim` objects, the label is stored as `sim.pars.label`,
        and `sim.label` is an alias to it. Sims do not have names separate from
        their labels. For `ss.Module` objects, the name and label are both attributes
        (`mod.name` and `mod.label`), with the difference being that the names
        are machine-readable (e.g. `'my_sir'`) while the labels are human-readable
        (e.g. `'My SIR module'`).
        """
        try:    return self.pars.label
        except: return None

    @label.setter
    def label(self, label):
        """
        Set the sim label (actually sets `sim.pars.label`)
        """
        self.pars['label'] = label
        return

    def init_module_attrs(self, force=False):
        """ Move initialized modules to the sim """
        module_types = ss.modules.module_types()
        module_types.append('custom') # Support extra catch-all module category

        for attr in module_types:
            orig = getattr(self, attr, None)
            if not force and orig is not None:
                warnmsg = f'Skipping key "{attr}" in parameters since already present in sim and force=False'
                ss.warn(warnmsg)
            else:
                modules = self.pars.pop(attr) # Remove module type (e.g. 'diseases') from the parameters
                setattr(self, attr, modules) # Add the modules ndict to the sim
                self.pars[attr] = sc.objdict() # Recreate with just the module parameters
                for key,module in modules.items():
                    self.pars[attr][key] = module.pars

        for mod in self.modules: # Each module is also sim.<name> (see __getattr__()), unless the sim has an attribute of that name (module names are unique: see ss.SimPars.validate_name())
            if mod.name in dir(self):
                ss.warn(f'The sim has an attribute "{mod.name}", so sim.{mod.name} is not the module of that name; use e.g. sim.diseases.{mod.name}')
        return

    def lock_pars(self, locked=True):
        """ Lock the sim and module parameters, so setting a parameter that doesn't exist raises an error """
        self.pars.lock(locked)
        for mod in self.modules:
            mod.pars.lock(locked)
        return

    def _init_modules_pre(self):
        """ Initialize all the modules with the sim """
        for mod in self.modules:
            mod.init_pre(self)
        return

    def _init_dists(self):
        """ Initialize the distributions """
        # Initialize all distributions now that everything else is in place
        self.dists.init(obj=self, base_seed=self.pars.rand_seed, force=True)

        # Copy relevant dists to each module
        for mod in self.modules:
            self.dists.copy_to_module(mod)
        return

    def _init_people_vals(self):
        """ Initialize the People states with actual values """
        self.people.init_vals()
        return

    def _init_modules_post(self):
        """ Initialize values in other modules, including networks and time parameters, and do any other post-processing """
        for mod in self.modules:
            mod.init_post()
        return

    def init_results(self):
        """ Create initial results that are present in all simulations

        This method initializes results by calling `People.init_results()` to let
        People handle its own result initialization (including automatic BoolState results).
        """
        # Let People handle its own result initialization
        self.people.init_results()

        # With dynamic rescaling, store the scale factor for each timestep
        if self.pars.rescale:
            self.results += ss.Result('pop_scale', label='Population scale factor', scale=False, auto_plot=False, timevec=self.timevec, values=np.ones(self.t.npts))

        # A product with no results of its own (most are only given by a delivery, which keeps the results) is left out
        for prod in self.products():
            if not [key for key,res in prod.results.items() if isinstance(res, ss.Result) and key != 'timevec']:
                self.results.pop(prod.name, None)

        # The results of one disease made or named by the sim are also at the top, e.g. sim.results.I
        implicit = [mod.name for mod in self.diseases() if mod._implicit]
        if len(implicit) == 1:
            self.results.setattribute('_implicit', implicit[0])

        # The results named to be kept by group must exist
        if self.pars.results is not None:
            known = [f'{mod.name}.{key}' for mod in self.modules for key,res in mod.results.items() if isinstance(res, ss.Result)]
            for name in sc.tolist(self.pars.results):
                if name not in known and not any(key.split('.', 1)[1] == name for key in known):
                    errormsg = f'results= names "{name}", which is not a result of this sim; the closest are {ss.transitions.closest(name, known)}'
                    raise sc.KeyNotFoundError(errormsg)
        return

    @property
    def current_scale(self):
        """ The number of people each agent represents on the current timestep; varies over time if `pars.rescale=True` """
        if not self.pars.rescale:
            return self.pars.pop_scale
        elif 'pop_scale' not in self.results: # The results aren't initialized yet, e.g. during a module's init_post()
            return 1.0 # With rescaling, the scale starts at 1
        return self.results.pop_scale[self.ti]

    def result_scale(self, module=None):
        """
        The factor to scale results by: a number, or an array with one value per timestep if `pars.rescale=True`

        Args:
            module (`ss.Module`): if provided, map the scale onto the module's timevec (if different from the sim's)
        """
        if not self.pars.rescale:
            return self.pars.pop_scale
        scale = self.results.pop_scale.values
        if module is not None and not np.array_equal(module.t.yearvec, self.t.yearvec):
            inds = np.searchsorted(self.t.yearvec, module.t.yearvec + 1e-9, side='right') - 1 # Use the most recent sim timestep, since the scale only changes on sim timesteps; 1e-9 avoids floating-point error
            scale = scale[np.maximum(inds, 0)]
        return scale

    def _init_rescale(self):
        """ Check that the diseases support dynamic rescaling, and create the distribution for it """
        for disease in self.diseases():
            if disease.graph is None and (type(disease).not_naive is ss.Disease.not_naive or (isinstance(disease, ss.Infection) and not hasattr(disease, 'ti_infected'))):
                errormsg = f'Dynamic rescaling (rescale=True) requires each disease written as code to define not_naive() and make_naive(), or to be an ss.Infection with a ti_infected state, but "{disease.name}" does not'
                raise ValueError(errormsg)
        self.rng_rescale = ss.choose_n(name='rng_rescale') # Choose who to make naive when rescaling
        return

    def rescale(self):
        """
        Dynamically rescale the population; only used if `pars.rescale=True`

        The scale factor starts at 1. Once the fraction of agents who are not naive (i.e. have
        ever been infected with any disease) exceeds `pars.rescale_threshold`, the scale factor
        is increased by at least `pars.rescale_factor` (up to `pars.pop_scale`), and the same
        fraction of non-naive agents are made naive again (via `disease.make_naive()`), so that
        each agent now represents more people. Based on Covasim's dynamic rescaling.
        """
        pars = self.pars
        ti = self.ti
        self.rng_rescale.jump_dt(ti=ti) # Like other distributions, advance the random numbers for this timestep
        scale = self.results.pop_scale
        if scale[ti] < pars.pop_scale: # We have room to rescale
            not_naive = ss.uids.concatenate([disease.not_naive() for disease in self.diseases()]).unique()
            ratio = len(not_naive)/len(self.people) # Current proportion not naive
            if ratio > pars.rescale_threshold: # Check if we've reached the point to rescale
                max_ratio = pars.pop_scale/scale[ti] # Don't exceed the total population size
                scaling_ratio = min(max(ratio/pars.rescale_threshold, pars.rescale_factor), max_ratio)
                scale[ti:] *= scaling_ratio # Update the scale factor from here on
                n = int(round(len(not_naive)*(1.0 - 1.0/scaling_ratio))) # For example, rescaling by 2 makes half of the non-naive agents naive
                self.rng_rescale.set(n=n)
                new_naive = self.rng_rescale.filter(not_naive)
                for disease in self.diseases():
                    disease.make_naive(new_naive)
        return

    def init_data(self, data=None, check=True):
        """ Initialize or add data to the sim; with check=False, only load and validate it, since the results it is checked against do not exist yet """
        data = data if data is not None else self.data
        if isinstance(data, (str, os.PathLike)):
            data = ss.simdata(data)
        if isinstance(data, ss.simdata):
            self._simdata = data # To drop the columns that match no result, if asked, once the results exist
            data = data.load()
        self.data = ss.results.validate_sim_data(data)
        if not check:
            return
        source = self.__dict__.pop('_simdata', None)
        if self.data is not None and source is not None and source.ignore_extra:
            self.data = source.drop_extra(self, self.data)
        if self.data is not None and ss.results.by_year(self.data):
            ss.note('the data has whole years (an integer year column), so each row is compared with the year it names: a flow summed over the year, a stock or ratio averaged over it', sim=self)
        if self.data is not None:
            self.data = ss.results.widen_data(self, self.data)
        if self.data is not None and getattr(self, 'declared', None):
            ss.results.check_data(self, self.data)
        return

    def set_diagnostics(self, rvs=True, states=True, detailed=False):
        """ Store detailed diagnostics information in the sim

        Stores every random number produced during a sim run (by `ss.Dist` objects)
        and/or every state of every agent, at every timestep. This allows e.g.
        understanding exactly where two runs that should have been identical diverged.
        Results can be exported (in JSON format) and analyzed externally.

        Although the states export could be used for analysis, users are encouraged
        to write an `ss.Analyzer` instead, as enabling diagnostics significantly degrades
        sim run time and memory usage.

        Args:
            rvs (bool): whether to enable diagnostics for random numbers/variates (default True)
            states (bool): as above, for people states (default True)
            detailed (bool): if true, store literally every random number and agent state (otherwise, use summary stats) (default False)

        Examples:
            ```python
            # General settings -- use a small sim and few timesteps if possible
            kw = dict(n_agents=100, start=0, stop=10, networks='random', diseases='sis')

            # Simplest usage: store info and print states
            sim = ss.Sim(**kw)
            sim.set_diagnostics()
            sim.run()
            print(sim.diagnostics.states)

            # Configure separately, with custom file export and saving all detail
            sim = ss.Sim(**kw)
            sim.set_diagnostics(rvs='diagnostics_rvs.json', states='diagnostics_states.json', detailed=True)
            sim.run()
            sim.diagnostics.export()
            ```
        """
        if rvs or states:
            print('Enabling sim diagnostics ...') # Always print regardless of verbosity
            self.diagnostics = ss.debugtools.Diagnostics(sim=self, rvs=rvs, states=states, detailed=detailed)
        else:
            errormsg = 'sim.set_diagnostics() was invoked, but nothing is being tracked'
            raise ValueError(errormsg)
        return

    def start_step(self):
        """ Start the step -- only print progress; all actual changes happen in the modules """

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            errormsg = 'Simulation already complete'
            raise ss.AlreadyRunError(errormsg)

        # Print out progress if needed
        if self.verbose or self.progress is not None:
            self.elapsed = self.timer.toc(output=True)
            t = self.t
            if self.verbose >= 2:
                sc.heading(f'  Running {t.now("str")} ({t.ti:2.0f}/{t.npts-1}) ({self.elapsed:0.2f} s) ')
            elif self.progress is not None:
                self.progress.update()
        return

    def advance(self):
        """ Move the time to the step about to be taken, its time point being the end of it, where whatever it does is dated and recorded: for the sim, and the modules on its timeline (see `ss.Loop`) """
        self.t.ti += 1
        for t in self.loop.shared_timelines:
            t.ti += 1
        return

    def finish_step(self):
        """ Finish the simulation timestep """
        if self.diagnostics and self.diagnostics.states:
            self.diagnostics.store_states()
        return

    def run_one_step(self, verbose=None):
        """
        Run a single sim step; only used for debugging purposes.

        Note: sim.run_one_step() runs a single simulation timestep, which involves
        multiple function calls. In contrast, loop.run_one_step() runs a single
        function call.

        Note: the verbose here is only for the Loop object, not the sim.
        """
        if not self.initialized: self.init() # Automatically initialize if not initialized
        t = self.t
        self.loop.run(t.tvec[min(t.ti, t.npts - 2)], verbose) # Up to the next step, which starts at time point ti
        return self

    def run(self, until=None, verbose=None, shrink=False, check_method_calls=True, profile=False, solver=None, n_runs=None):
        """
        Run the model -- the main method for running a simulation.

        Args:
            until (date/str/float): run the steps up to and including the one that starts at this time
            verbose (int): how much to print: 0 nothing, 1 the solver, notes, and progress (default), 2 everything (e.g. each step)
            shrink (bool): whether to explicitly shrink the sim after running
            check_method_calls (bool): whether to check that all required methods were called
            profile (bool): if True, record per-entry CPU timing in the loop (see `sim.loop.to_df()`/`plot_cpu()`)
            solver (str): the solver to use, if different from the sim's (a sim already built for another is built again)
            n_runs (int): if provided, run the sim this many times with different seeds, and return an `ss.MultiSim`; a shortcut for `ss.parallel(sim, n_runs=n_runs)`

        A sim that has finished is run again from the start, with any values changed since.
        """
        imp = getattr(self, 'imported', None)
        unwarned = imp is not None and imp.get('incomplete') and not imp.get('warned') # Once per import: the runs and copies share it
        if unwarned and not self.initialized:
            imp.warned = True
            ss.warn(f'running an import with {len(imp.incomplete)} part(s) of the model not translated (sim.imported.incomplete); '
                    f'its results leave them out. {imp.prompt}', ss.errors.ImportIncomplete)
        new_solver = solver is not None and solver != (self.solver if self.initialized else self.pars.solver)
        if n_runs is not None:
            return ss.parallel(self.new(solver=solver) if new_solver else self, n_runs=n_runs)
        if new_solver and self.initialized: # Built for another solver, so built again
            self.new(solver=solver, inplace=True)
        elif solver is not None:
            self.pars.solver = solver
        if self.complete and until is None: # Run again from the start, with any values changed since
            self.new(inplace=True)

        # Initialization steps
        if verbose is not None and not self.initialized: # So that initializing says as much
            self.pars.verbose = ss.parameters.check_verbose(verbose)
        if not self.initialized: self.init() # Automatically initialize if not initialized
        if verbose is not None:
            self._orig_verbose = self.verbose
            self.verbose = ss.parameters.check_verbose(verbose)
        unsaid = self.__dict__.pop('_unsaid', None) # The start line, if sim.describe() initialized the sim
        if unsaid is not None and self.verbose:
            print('\n'.join(self.start_lines(unsaid)))
        self.timer.start()

        # Check for AlreadyRun errors
        if self.complete:
            errormsg = 'Simulation is already complete'
            raise ss.AlreadyRunError(errormsg)

        if self.verbose: # Otherwise none, or how far it has got for a MultiSim's progress (see ss.run.single_run())
            self.progress = ss.presentation.Progress(self)
        try:
            # Main simulation loop -- just one line!!!
            reason = self.kernel_reason() if until is None and not profile else None
            kernels = until is None and not profile and reason is None
            if reason is not None and self.solver != 'abm' and getattr(self, 'declared', None): # Say once why a compartmental run is taken a step at a time: only about speed, so only with verbose=2 (describe() says it too)
                sc.printv(f'Run step by step, since {reason}', 2, self.verbose)
            if kernels: # JAX or the kernels: see ss.solvers.choose_engine()
                use_jax, why = ss.solvers.choose_engine(self)
                sc.printv(f'engine: {why}', 2, self.verbose)
            if kernels: # Nothing needs to happen between the steps, so each module takes all of them at once
                for mod in self.declared:
                    mod.engine.run_all(use_jax=use_jax)
                self.complete = True # Without making the loop's plan, which is not needed
            else:
                self.loop.run(until, profile=profile)
                self.complete = self.loop.index == len(self.loop.plan)

            # If simulation reached the end, finalize the results
            self.elapsed = self.timer.toc(output=True)
            if self.complete:
                self.finalize()
                if check_method_calls and not kernels:
                    self.check_method_calls()
                self._made, self._made_pars = self._settings(), self._module_pars() # So that values set after the run are kept when it is run again (see new())
                if shrink:
                    self.shrink(full=False)
                if self.progress is not None:
                    self.progress.finish(self.end_line())
        finally:
            self.dists.clear_run_caches()
            self.progress = None
        
        return self # Allows e.g. ss.Sim().run().plot()

    def kernel_reason(self):
        """ Why the sim is not run in one call to the kernels, in a few words, or None if it is (see `sim.kernel_only()`) """
        if self.solver == 'abm' or not getattr(self, 'declared', None):
            return 'it runs as agents' if self.solver == 'abm' else 'it has no declared model'
        elif self.loop.index or self.loop.insertions:
            return 'it has been stepped already, or has functions inserted into its loop'
        elif self.diagnostics:
            return 'it records diagnostics'
        elif self.verbose >= 2:
            return 'it prints each step (verbose=2)'
        names = {func['module'] for func in self.loop.funcs} - {'sim', 'people'} - {mod.name for mod in self.modules if getattr(mod, 'tabulated', False)} # A modifier that depends only on time and on states is tabulated by the engines: see ss.solvers.Compartmental.tables()
        others = [mod.name for mod in self.modules if mod.name in names and not isinstance(getattr(mod, 'engine', None), ss.solvers.Compartmental)]
        if others:
            return f'{sc.strjoin(others)} {"has" if len(others) == 1 else "have"} a step of {"its" if len(others) == 1 else "their"} own (a modifier with where=, when=, or a function as its amount, an intervention, or an analyzer)'
        for mod in self.declared:
            reason = mod.engine.kernel_reason()
            if reason is not None:
                return reason
        return None

    def kernel_only(self):
        """ Whether the sim is nothing but compartments that the kernels can run from start to finish: see `sim.kernel_reason()` """
        return self.kernel_reason() is None

    def gradient(self, results=None, pars=None):
        """
        The derivative of results with respect to parameters, at their current values

        The ODE run is differentiated by JAX (see `ss.gradients`), so this needs JAX, a sim
        with `solver='ode'`, and one that the kernels run from start to finish (see
        `sim.kernel_only()`). The sim itself is not changed.

        Args:
            results (str/list): the results, by name (`'I'`, or `'sir.I'` where needed); default: every result of the declared modules
            pars (str/list): the parameters, by name; default: every parameter that has bounds

        Returns:
            An `sc.objdict` by parameter, then module, then result, of the derivative at each step

        Example:
            ```python
            beta = ss.par(0.3, bounds=[0.1, 1])
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
            sim = ss.Sim(sir, unit='day', stop=100)
            grad = sim.gradient('I')
            grad.beta.sir.I # How the number infectious at each step changes with beta
            ```
        """
        sim = self.new(verbose=0).init()
        return ss.gradients.of_results(sim, results=results, pars=pars)

    def log_density(self, pars=None, likelihood=None):
        """
        The log of the posterior density of the fitted parameters, as a function in JAX of their values, with its gradient by reverse mode; see `ss.gradients.LogDensity`

        Needs JAX, a sim with `solver='ode'` that the kernels run whole, data, and a likelihood for every data series.

        Args:
            pars (str/list): the parameters, by name (default: every one with bounds)
            likelihood (dict): the likelihood of each data series, if not declared on its outcome

        Example:
            ```python
            ld = sim.log_density()
            ld(ld.x0), ld.grad(ld.x0) # The value and its gradient at the current values
            ```
        """
        return ss.gradients.LogDensity(self.new(verbose=0).init(), pars=pars, likelihood=likelihood)

    def restart(self, pars=None, rand_seed=None):
        """
        Take a sim back to its start so that it can be run again, without building it again

        For a sim that is nothing but compartments (see `sim.kernel_only()`), this makes
        repeated runs many times faster; any other sim (or a change the compartments were
        not laid out for, such as a new distribution) is built again, as `sim.new()` would.
        `sim.run(n_runs=...)` and `sim.fit()` use it where they can.

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            rand_seed (int): the seed for the next run

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ 0.3*I, ss.Transition(I, R, dur=8), init=dict(I=0.01))
            sim = ss.Sim(sir, unit='day', stop=100).init()
            peaks = [sim.restart(pars=dict(beta=beta)).run().results.sir.I.max() for beta in [0.2, 0.3, 0.4]]
            ```
        """
        if not self.initialized:
            self.init()
        self.loop.index = 0
        rebuild = lambda: self.new(pars=pars, inplace=True, **({} if rand_seed is None else dict(rand_seed=rand_seed)))
        if not self.kernel_only() or any(isinstance(val, ss.Dist) for val in (pars or {}).values()): # Not run in one call to the kernels, or a distribution the compartments were not laid out for
            return rebuild()
        if rand_seed is not None:
            self.pars.rand_seed = rand_seed
        if pars:
            moving = [p for mod in self.declared for strat in mod.graph.strats if strat.mobility is not None for p in strat.mobility.fitted().values()]
            before = [float(p) for p in moving]
            self.par_values = sc.mergedicts(self.par_values, pars)
            ss.solvers.set_par_values(self)
            varies = lambda raw: isinstance(raw, ss.transitions.TimeValue) or (isinstance(raw, ss.by) and raw.varies)
            for mod in self.declared: # The kernels' program knows which values vary in time from when it was built
                built = getattr(mod.engine, 'time_pars', None)
                if built is not None and any(varies(p.raw) != any(p is q for q in built) for p in mod.graph.pars.values()):
                    return rebuild()
            if any(float(p) != val for p,val in zip(moving, before)): # The movement between places, worked out again with the new values
                self.mixing.init_movement()
                for mod in self.declared:
                    if not mod.engine.restart_movement():
                        return rebuild()
        self.complete = False
        self.results_ready = False
        self.t.ti = 0
        for crossing in self.crossings: # Nothing left over from the run that has ended
            crossing.reset()
        for mod in self.declared:
            mod.t.ti = 0
            mod.finalized = False
            mod.engine.restart()
        for crossing in self.crossings: # As in init()
            crossing.init()
        return self

    def finalize(self):
        """ Compute final results """
        # Finalize the results
        self.finalize_results()

        # Finalize each module, including the results
        for mod in self.modules:
            mod.finalize()
        self.check_finite()
        ss.solvers.people_results(self)

        # Resets verbose if needed
        if hasattr(self, '_orig_verbose'):
            self.verbose = self._orig_verbose
            delattr(self, '_orig_verbose')

        # Generate the summary and finish up
        self.summarize() # Create summary
        return

    def check_finite(self):
        """ Raise an error if the number of people in a declared state stopped being a number during the run, rather than return results that are not """
        for mod in self.declared:
            for name in mod.graph.states:
                res = mod.results.get(name)
                if res is not None and not np.all(np.isfinite(res.values)):
                    ti = int(np.flatnonzero(~np.isfinite(res.values))[0])
                    errormsg = (f'The number of people in "{name}" of "{mod.name}" is not a number from time point {ti} on. A rate too fast for the step can do this, '
                                f'e.g. a count of people where a share was meant (I.n rather than I.frac); see sim.notes, or use a smaller dt')
                    raise ValueError(errormsg)
        return

    def finalize_results(self):
        """ Scale the results and remove any "unused" results """
        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise ss.AlreadyRunError('Simulation has already been finalized')

        scale = self.result_scale()
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result):
                if res.scale: # Scale results; NB: disease-specific results are scaled in module.finalize() below
                    res.apply_scale(scale)
                if np.all(res == res[0]): # Results were not modified during the sim
                    self.results[reskey].auto_plot = False
        if self.pars.rescale and self.pars.people_results: # With a scale that varies over time, cumulative results must be summed after scaling
            self.results.cum_deaths[:] = np.cumsum(self.results.new_deaths)
        self.results_ready = True # Results are ready to use
        return

    def summarize(self, how=None):
        """
        Provide a quick summary of the sim: each result over the whole run, by its kind

        By default, a flow (e.g. new infections) is totaled, a stock (e.g. the number
        infected) takes its final value, and a ratio (e.g. prevalence) is averaged; see
        `ss.Result.summarize()`.

        Args:
            how (str/dict): 'mean', 'median', 'last', 'sum', or a function, to summarize every result that way; or a dict of these by result key (e.g. `dict(sir_infected='mean')`), with the rest by kind
        """
        funcs = dict(mean=np.mean, median=np.median, sum=np.sum, last=lambda res: res[-1])
        if not isinstance(how, dict):
            how = {} if how is None else {'':how} # '' matches every key

        summary = sc.objdict()
        flat = self.results.flatten(sep='_', only_results=False, columns=True) # Summarize each column of 2D results separately
        for key, res in flat.items():
            if 'timevec' in key: # Skip module-specific time vectors
                continue
            try:
                func = how.get(key, how.get(''))
                if func is None:
                    entry = res.summarize() if isinstance(res, ss.Result) else np.mean(res)
                else:
                    entry = func(res) if callable(func) else funcs[func](res)
            except Exception as E:
                entry = f'N/A {E}'
            summary[key] = entry
        self.summary = summary
        return summary

    def shrink(self, inplace=True, full=True, size_limit=1.0, base_size=30, die=True):
        """
        "Shrinks" the simulation by removing the people and other memory-intensive
        attributes (e.g., some interventions and analyzers), and returns a copy of
        the "shrunken" simulation. Used to reduce the memory required for RAM or
        for saved files.

        After shrinking, each module is expected to be smaller than
        `size_limit*(base_size + n_timesteps)` KB. The `base_size` term covers the
        fixed structural overhead of a module (parameters, results metadata, etc.) that
        does not depend on the simulation length, expressed as a number of timesteps'
        worth of allowance; the `n_timesteps` term covers the per-timestep results that
        legitimately grow with the simulation; and `size_limit` scales both. A module
        exceeding this is most likely still holding a per-agent array that failed to shrink.

        Args:
            inplace (bool): whether to perform the shrinking in place (default), or return a shrunken copy instead
            full (bool): whether to perform a full shrink, including the People object (otherwise just remove circular references)
            size_limit (float): the per-timestep size allowance in KB, which scales the whole limit (set to None to disable the check)
            base_size (float): the fixed module size to allow independent of simulation length, in timesteps' worth of allowance
            die (bool): whether to raise an exception if the shrink failed

        Returns:
            shrunken (Sim): a Sim object with the listed attributes removed
        """
        # Create the new object, and copy original dict, skipping the skipped attributes
        if inplace:
            sim = self
        else:
            sim = self.copy() # We need to do a deep copy to avoid modifying other objects

        # Shrink the people
        if full:
            ss.utils.shrink(sim, 'people')
        
        # If the sim is not initialized, we're done (ignoring the corner case where initialized modules are passed to an uninitialized sim)
        if sim.initialized:

            # Shrink the loop -- important; frees un-collectable sim references
            sim.loop.shrink()

            # Shrink the distributions
            ss.utils.shrink(sim.dists, attrs=['sim', 'obj'])
            for dist in sim.dists.dists.values():
                with sc.tryexcept(die=die):
                    dist.shrink() # Important; frees un-collectable sim references
            
            # The remaining shrinkages are only applied in the full case
            if full:

                # Shrink the modules
                for mod in sim.modules:
                    with sc.tryexcept(die=die):
                        mod.shrink()
                for crossing in sim.crossings:
                    crossing.shrink()

                # Check that the module successfully shrunk
                if size_limit:
                    npts = len(sim)
                    max_size = size_limit*(base_size + npts) # Maximum size in KB: size_limit scales both the fixed base and the per-timestep allowance
                    for mod in sim.modules:
                        size = sc.checkmem(mod, descend=0).bytesize[0]/1e3 # Size in KB
                        size -= sc.checkmem(mod.results, descend=0).bytesize[0]/1e3 # Not counting its results, which grow with its stratifications as well as with time
                        if size > max_size:
                            errormsg = f'Module {mod.name} did not successfully shrink: {size:n} KB, not counting its results, > {max_size:n} KB (= {size_limit:n} KB/timestep × ({base_size:n} base + {npts} timesteps)); use die=False to turn this message into a warning, or increase base_size/size_limit'
                            if die:
                                raise RuntimeError(errormsg)
                            else:
                                ss.warn(errormsg)

        # Finally, set a flag that the sim has been shrunken, but only if fully shrunken
        sim.shrunken = full
        return sim

    def check_results_ready(self, errormsg=None):
        """ Check that results are ready """
        if errormsg is None:
            errormsg = 'Please run the sim first'
        if not self.results_ready:  # pragma: no cover
            raise RuntimeError(errormsg)
        return

    def check_method_calls(self, die=None, warn=None, verbose=False):
        """
        Check if any required methods were not called.

        Typically called automatically by `sim.run()`; default behavior is to warn
        (see `options.check_method_calls`).

        Args:
            die (bool): whether to raise an exception if missing methods were found (default False)
            warn (bool): whether to raise a warning if missing methods were found (default False)
            verbose (bool): whether to print the number of times each method was called (default False)

        Returns:
            A list of missing method calls by module
        """
        # Handle arguments, or fall back to options
        valid = ['die', 'warn', False]
        if die is not None or warn is not None:
            check = 'die' if die else 'warn' if warn else False
        else:
            check = ss.options.check_method_calls
        if check not in valid:
            errormsg = f'Could not understand {check=}, must be one of {valid}'
            raise ValueError(errormsg)

        missing = []
        if check:
            if verbose:
                sc.pp({mod.name:mod._call_required for mod in self.modules})

            for mod in self.modules:
                modmissing = mod.check_method_calls()
                if modmissing:
                    missing.append([type(mod), modmissing])
            if missing:
                errormsg = 'The following methods are required, but were not called.\n'
                errormsg += 'Did you mistype a method name, forget a super() call,\n'
                errormsg += 'or did part of the sim not run (e.g. zero infections)?\n'
                for modtype,modmissing in missing:
                    errormsg += f'{modtype}: {sc.strjoin(modmissing)}\n'
                if check == 'die':
                    raise RuntimeError(errormsg)
                elif check == 'warn':
                    ss.warn(errormsg)
        return missing

    def to_df(self, **kwargs):
        """ The results as a dataframe, with a column for each (e.g. `sir_prevalence`); see `ss.Results.to_df()` for the arguments """
        self.check_results_ready('Please run the sim before exporting the results')
        return self.results.to_df(**kwargs)

    def to_data(self, results, resample=None, noise='poisson', seed=None):
        """
        Data drawn from the results of the run, as if observed: e.g. to check that a fit finds values that are known

        Args:
            results (str/list): the results, by name, e.g. 'cases' (or 'sir.cases' if more than one module has one of that name)
            resample (str): the period each value is over, e.g. 'week' for the total of a flow over each week
            noise (str): 'poisson' (the default), for counts drawn with each value as their mean; or None, for the values themselves
            seed (int): the random seed for the noise (default: the sim's)

        Returns:
            A dataframe with a column of times (`date`, or `time` in a sim without dates) and a column for each result, as `ss.Sim(data=...)` takes it

        Example:
            ```python
            truth = ss.Sim(sir, pars=dict(beta=0.025), rand_seed=2, **pars).run()
            data = truth.to_data('cases', resample='week') # Weekly cases, with Poisson noise
            sim = ss.Sim(sir, data=data, **pars)
            ```
        """
        self.check_results_ready('Please run the sim before drawing data from it')
        if noise not in [None, 'poisson']:
            errormsg = f'noise="{noise}" is not a kind of noise; the choices are "poisson" and None'
            raise ValueError(errormsg)
        rng = np.random.default_rng(sc.ifelse(seed, self.pars.rand_seed))
        columns = {}
        for name in sc.tolist(results):
            res = ss.results.match_column(self, name)
            if res is None:
                errormsg = f'The sim has no result "{name}" (or more than one); results are named as in sim.results, e.g. "cases", or "sir.cases"'
                raise KeyError(errormsg)
            if noise is not None and res.kind == 'ratio':
                errormsg = f'"{name}" is a ratio, which has no Poisson noise; give noise=None for its values'
                raise ValueError(errormsg)
            series = res.to_series(resample=resample)
            values = series.to_numpy(dtype=float)
            columns[name] = rng.poisson(np.maximum(values, 0)) if noise == 'poisson' else values
        timecol = 'date' if isinstance(series.index[0], (ss.date, pd.Timestamp)) else 'time'
        return pd.DataFrame({timecol:list(series.index), **columns})

    def profile(self, follow=None, do_run=True, plot=True, **kwargs):
        """
        Profile the performance of the simulation using a line profiler (`sc.profile()`)

        Args:
            follow (func/list): a list of functions/methods to follow in detail
            do_run (bool): whether to immediately run the sim
            plot (bool): whether to plot time spent per module step
            **kwargs (dict): passed to `sc.profile()`

        Examples:
            ```python
            import starsim as ss

            net = ss.RandomNet()
            sis = ss.SIS()
            sim = ss.Sim(networks=net, diseases=sis)
            prof = sim.profile(follow=[net.add_pairs, sis.step])
            ```
        """
        prof = ss.debugtools.Profile(self, follow=follow, do_run=do_run, plot=plot, **kwargs)
        return prof

    def cprofile(self, sort='cumtime', mintime=1e-3, **kwargs):
        """
        Profile the performance of the simulation using a function profiler (`sc.cprofile()`)

        Args:
            sort (str): default sort column; common choices are 'cumtime' (total time spent in a function, includin subfunctions) and 'selftime' (excluding subfunctions)
            mintime (float): exclude function calls less than this time in seconds
            **kwargs (dict): passed to `sc.cprofile()`

        Examples:
            ```python
            import starsim as ss

            net = ss.RandomNet()
            sis = ss.SIS()
            sim = ss.Sim(networks=net, diseases=sis)
            prof = sim.cprofile()
            ```
        """
        cprof = sc.cprofile(sort=sort, mintime=mintime, **kwargs)
        with cprof:
            self.run()
        return cprof

    def save(self, filename=None, shrink=None, **kwargs):
        """
        Save to disk as a gzipped pickle: the sim itself, with its results, read back by `ss.load()`

        This is the checkpoint of a sim: saved before it has finished, it keeps its full
        state (it is not shrunk), so the loaded sim can continue the run. A pickle needs the
        same code (Starsim and any modules written as code) to be read. For the model
        as plain data, see `sim.to_spec()`; for the results as JSON, `sim.to_json()`; for the
        model as Python, `sim.to_code()`.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            shrink (bool or None): whether to shrink the sim prior to saving (reduces size by ~99%)
            kwargs: passed to sc.makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        Examples:
            ```python
            sim.save() # Saves to a .sim file
            ```
        """
        # Set shrink based on whether we're in the middle of a run
        if shrink is None:
            shrink = False if self.initialized and not self.results_ready else True

        # Handle the filename
        if filename is None:
            filename = self.simfile
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename  # Store the actual saved filename

        # Handle the shrinkage and save
        sim = self.shrink(inplace=False) if shrink else self
        sc.save(filename=filename, obj=sim)
        return filename

    def to_json(self, filename=None, keys=None, indent=2, verbose=False, **kwargs):
        """
        Export the results and the sim's parameters as JSON

        This is a record of a run's output, for reading elsewhere; it is not the model
        (see `sim.to_spec()`), and cannot be loaded back into a sim (see `sim.save()`).

        Args:
            filename (str): if None, return string; else, write to file
            keys (str/list): attributes to write to json (choices: 'pars', 'summary', and/or 'results')
            verbose (bool): detail to print
            **kwargs (dict): passed to `sc.jsonify()`

        Returns:
            A dictionary representation of the parameters and/or summary results
            (or write that dictionary to a file)

        Examples:
            ```python
            json = sim.to_json() # Convert to a dict
            sim.to_json('sim.json') # Write everything
            sim.to_json('summary.json', keys='summary') # Just write the summary
            ```
        """
        # Handle keys
        if keys is None:
            keys = ['pars', 'summary', 'results']
        keys = sc.promotetolist(keys)

        # Convert to JSON-compatible format
        d = sc.objdict()
        for key in keys:
            if key in ['pars', 'parameters']:
                pardict = self.pars.to_json()
                d.pars = pardict
            elif key == 'summary':
                if self.results_ready:
                    d.summary = dict(sc.dcp(self.summary))
                else:
                    d.summary = 'Summary not available (Sim has not yet been run)'
            elif key == 'results':
                df = self.to_df()
                if isinstance(df, dict):  # For sims with heterogeneous timelines, to_df() returns a dict of dataframes rather than a single dataframe
                    d.results = {k:v.to_dict() for k,v in df.items()}
                else:
                    d.results = df.to_dict()
            else:  # pragma: no cover
                warnmsg = f'Could not convert "{key}" to JSON; continuing...'
                ss.warn(warnmsg)

        # Final conversion
        d = sc.jsonify(d, **kwargs)
        if filename is not None:
            sc.savejson(filename=filename, obj=d, indent=indent)
        return d

    def to_yaml(self, filename=None, sort_keys=False, **kwargs):
        """
        Export results and parameters as YAML.

        Args:
            filename (str): the name of the file to write to (default `{sim.label}.yaml`)
            kwargs (dict): passed to `sim.to_json()`

        Examples:
            ```python
            sim = ss.Sim(diseases='sis', networks='random').run()
            sim.to_yaml('results.yaml', keys='results')
            ```
        """
        if filename is None:
            if self.label:
                filename = f'{self.label}.yaml'
            else:
                filename = 'sim.yaml'
        json = self.to_json(filename=None, **kwargs)
        return sc.saveyaml(filename=filename, obj=json, sort_keys=sort_keys)

    def plot(self, key=None, fig=None, show_data=True, show_skipped=False, show_module=None,
             show_label=False, n_ticks=None, annualize=False, **kwargs):
        """
        Plot all results in the Sim object

        Args:
            key (str/list): the results key to plot (by default, all); if a list, plot exactly those keys
            fig (Figure): if provided, plot results into an existing figure
            style (str): the plotting style to use
            show_data (bool): plot the data, if available
            show_skipped (bool): show even results that are skipped by default
            show_module (int): whether to show the module as well as the result name; if an int, show if the label is less than that length (default, 26); if -1, use a newline
            show_label (str): if 'fig', reset the fignum; if 'title', set the figure suptitle
            n_ticks (tuple of ints): if provided, specify how many x-axis ticks to use (default: `(2,5)`, i.e. minimum of 2 and maximum of 5)
            annualize (bool): if True, resample results to annual values before plotting (each by its kind; see `ss.Result.resample()`)
            fig_kw (dict): passed to `sc.getrowscols()`, then `plt.subplots()` and `plt.figure()`
            plot_kw (dict): passed to `plt.plot()`
            data_kw (dict): passed to `plt.scatter()`, for plotting the data
            style_kw (dict): passed to `ss.style()`, for controlling the detailed plotting style (default "starsim"; other options are "simple", None, or any Matplotlib style)
            **kwargs (dict): known arguments (e.g. figsize, font) split between the above dicts; see `ss.plot_args()` for all valid options

        Examples:
            ```python
            sim = ss.Sim(diseases='sis', networks='random').run()

            # Basic usage
            sim.plot()

            # Plot a single result
            sim.plot('sis.prevalence')

            # Plot with a custom figure size, font, and style
            sim.plot(figsize=(12,16), font='Raleway', style='fancy')
            ```
        """
        self.check_results_ready('Please run the sim before plotting')
        data = ss.results.plot_data(self) if show_data else {} # The data, by the result that each column matches; a flow with data at a longer period (e.g. weekly) is shown as its totals over the data's periods, as the fit compares them
        if fig is None and show_label in ['fig', 'fignum'] and self.label:
            plotlabel = self.label
            figlist = sc.autolist()
            while plt.fignum_exists(plotlabel):
                figlist += plotlabel
                plotlabel = sc.uniquename(self.label, figlist, human=True)
            kwargs['num'] = plotlabel
        title = self.label if show_label in ['title', 'suptitle'] else None
        if key is None and not show_skipped: # The sim's own results that never change (e.g. the number alive and the deaths of a closed population) are left out
            own = [k for k,res in self.results.items() if isinstance(res, ss.Result)]
            flat = ss.results.match_result_keys(self.results, None)
            key = [k for k,res in flat.items() if not (k in own and np.allclose(res.values, res.values[0]))] or None
        return self.results.plot(key=key, fig=fig, data=data, label=self.label, title=title, show_skipped=show_skipped, show_module=show_module, n_ticks=n_ticks, annualize=annualize, **kwargs)



def demo(run=True, plot=True, summary=True, show=True, **kwargs):
    """
    Create a simple demo simulation for Starsim

    Defaults to using the SIR model with a random network, but these can be configured.

    Args:
        run (bool): whether to run the sim
        summary (bool): whether to print a summary of the results
        plot (bool): whether to plot the results
        kwargs (dict): passed to `ss.Sim()`

    Examples:
        ```python
        ss.demo() # Run, plot, and show results
        ss.demo(diseases='hiv', networks='sexual') # Run with different defaults
        ```
    """
    pars = sc.mergedicts(dict(diseases='sir', networks='random'), kwargs)
    sim = Sim(pars)
    if run:
        sc.heading('Running demo:')
        sim.run()
        if summary:
            sc.heading('Results:')
            print(sim.summary)
        if plot:
            sim.plot()
            if show:
                plt.show()
    return sim

