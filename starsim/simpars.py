"""
The settings of a sim (`ss.SimPars`, as `sim.pars`), and how the modules given to a sim are checked and sorted by kind

These are kept apart from `ss.Sim` so that the sim's own file is about running it. Each setting is described in `ss.Sim`.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss


def module_kind(mod, default='custom'):
    """ The kind of a module, e.g. 'diseases' (see `ss.modules.module_map()`), or `default` for one of no standard kind """
    return next((key for key,cls in ss.modules.module_map().items() if isinstance(mod, cls)), default)


def module_kinds():
    """ The kinds of module a sim keeps, e.g. 'diseases' (see `ss.modules.module_types()`), and 'custom', for those of no standard kind (`sim.custom`) """
    return ss.modules.module_types() + ['custom']


def flatten(items):
    """ Items given alone or in a list, with one level of lists inside it flattened, e.g. [sir, [vx1, vx2]] → [sir, vx1, vx2] """
    return [item for entry in sc.tolist(items) for item in (entry if isinstance(entry, list) else [entry])]


#%% Sorting what a sim is given

def sort_inputs(models, kwargs, pars, given, names):
    """
    Sort what `ss.Sim()` is given into the modules by kind, the settings of the sim, and the values of the model's parameters; not for the user

    The model (modules, transitions, or dicts of settings, given as arguments), `modules=`, `pars=`, and the other keywords can each
    hold any of the three. Each step below takes one of them out:

    1. `take_modules()`: the modules given with `modules=`, or in `pars=`, are part of the model
    2. `settings_in_models()`: a dict in the model holds settings of the sim (the keywords given win)
    3. the modules given in a dict of settings (e.g. `diseases=`) are as if given by keyword
    4. `model_pars()`: a keyword that is not a setting of the sim is a value of a parameter of the model, as in `pars=`
    5. `sort_by_kind()`: the model's modules (transitions made a disease) are sorted into their kinds
    6. `settings_in_pars()`: a name in `pars=` that is a setting of the sim, and not a parameter of the model, is a setting

    Args:
        models (list): the arguments, the model
        kwargs (dict): the keywords that `ss.Sim()` does not name, e.g. `n_agents=`, or `beta=` for a parameter of the model
        pars (dict/SimPars): `pars=`
        given (dict): the arguments that `ss.Sim()` names (e.g. `label=`, `diseases=`), with None for those not given
        names (dict): the names of the variables of the code that made the sim (see `ss.utils.caller_names()`)

    Returns:
        An `sc.obj` with `mods` (the modules by kind; `modules` for those sorted when the sim is initialized), `settings` (the sim's settings by name), `par_values` (the values of the model's parameters by name, or None), and `given_order` (the model's modules in the order given)
    """
    kwargs = dict(kwargs)
    models, pars = take_modules(models, kwargs, pars)
    kwargs = settings_in_models(models, kwargs, given)
    kinds = ['demographics', 'connectors', 'networks', 'diseases', 'interventions', 'analyzers', 'custom'] # In the order of ss.Sim()'s arguments, the order they are named and copied in
    mods = {key:(kwargs.pop(key) if given[key] is None and key in kwargs else given[key]) for key in kinds} # Modules given in a dict of settings are named and ordered as if given by keyword
    mods['modules'] = None # Those of a whole ss.SimPars, or of a kind given by name (see sort_by_kind()), to sort when the sim is initialized
    pars = model_pars(kwargs, ss.SimPars.names(), pars)
    models = make_disease(models, names)
    mods['diseases'] = make_disease(mods['diseases'], names)
    given_order = sort_by_kind(models, mods)
    settings, par_values = settings_in_pars(pars, kwargs, given, given_modules(dict(enumerate(models)), mods))
    return sc.obj(mods=mods, settings=settings, par_values=par_values, given_order=given_order)


def take_modules(models, kwargs, pars):
    """
    The model, with the modules given with `modules=` (by keyword, or in a dict of settings other than a whole `ss.SimPars`) as part of it, sorted by kind as
    arguments are (those of no standard kind are kept in `sim.custom`); and a whole `ss.SimPars` given as `pars=`, or the modules by kind in `pars=`, as if
    given in place of the model. Takes `modules=` out of `kwargs`, and returns the model and what is left of `pars`.
    """
    models = list(models)
    mods = kwargs.pop('modules', None)
    if mods is not None:
        models.append(mods)
    for i,arg in enumerate(models):
        if isinstance(arg, dict) and not isinstance(arg, ss.SimPars) and 'modules' in arg:
            arg = dict(arg)
            mods = arg.pop('modules')
            models[i] = arg
            if mods is not None:
                models.append(mods)
    if isinstance(pars, dict) and not isinstance(pars, ss.SimPars) and 'modules' in pars:
        pars = dict(pars)
        mods = pars.pop('modules')
        if mods is not None:
            models.append(mods)
    models = [arg for arg in models if arg is not None]
    kinds = module_kinds()
    if isinstance(pars, ss.SimPars): # The sim's settings as a whole, e.g. ss.Sim(pars=sim.pars), as if given in place of the model
        models, pars = models + [pars], None
    elif isinstance(pars, dict) and any(key in kinds for key in pars): # Modules in pars=, as if given in a dict in place of the model
        models, pars = models + [{key:val for key,val in pars.items() if key in kinds}], {key:val for key,val in pars.items() if key not in kinds}
    return models, pars


def settings_in_models(models, kwargs, given):
    """ The keywords, with the settings in the dicts of the model (e.g. `ss.Sim(dict(n_agents=1000))`) added, those given outright winning; what is not a dict must be the model """
    for arg in models:
        if isinstance(arg, dict):
            kwargs = sc.mergedicts({k:v for k,v in arg.items() if given.get(k) is None}, kwargs) # A keyword given outright wins over the same key in the dict
        elif not all(isinstance(obj, ss.Module) or ss.transitions.is_declaration(obj) or callable(obj) for obj in (arg if isinstance(arg, (list, tuple)) else [arg])): # A module's class, or a function of the sim (which becomes a module's step()), is made a module as for any kind
            errormsg = f'ss.Sim() takes the model (modules, their classes, functions of the sim, or transitions, or lists of them) or a dict of parameters as positional arguments, not {type(arg).__name__}; give everything else by keyword, e.g. label="..."'
            raise TypeError(errormsg)
    return kwargs


def model_pars(kwargs, settings, pars=None):
    """
    The values of the model's parameters, `pars` with those given by keyword (e.g. `ss.Sim(sir, beta=0.1)`, as in `pars=`) added, which are taken out of `kwargs`; a name that is a setting of the sim is a setting (the setting wins)

    Args:
        kwargs (dict): the keywords, which keep the settings
        settings (set): the names of the settings of the sim
        pars (dict): the values given as `pars=`
    """
    model = [key for key in kwargs if key not in settings and key not in ss.v3_deprecations.keys('Sim')]
    if model:
        pars = sc.mergedicts(pars, {key:kwargs.pop(key) for key in model})
    return pars


def sort_by_kind(models, mods):
    """
    Sort the model's modules into their kinds in `mods` (in place), after those given by keyword, as `init()` would, so that e.g. `sim.pars.diseases` has them before
    `init()` as after; and number each by its place in the model, for `step_order='given'` (see `SimPars.validate_modules()`). Returns the modules in the order given.
    """
    given_order = sc.mergelists(*[arg for arg in models if not isinstance(arg, dict)])
    for i,mod in enumerate(given_order):
        if isinstance(mod, ss.Module):
            mod._model_index = i
    for mod in given_order:
        key = module_kind(mod)
        if key == 'custom' or isinstance(mods[key], (dict, bool)): # Modules of no standard kind (sim.custom), or by name, e.g. networks=dict(main=...), or demographics=True: sorted at init() (see validate_modules())
            mods['modules'] = sc.mergelists(mods['modules'], mod)
        else:
            mods[key] = sc.mergelists(mods[key], mod)
    return given_order


def settings_in_pars(pars, kwargs, given, mods):
    """
    The settings of the sim, and the values of the model's parameters by name, from `pars=` and the other keywords

    `pars=` can hold settings of the sim too (as in Starsim 3), e.g. `pars=dict(n_agents=1000, beta=0.1)`: a name is a setting if no parameter of the model has it
    (so `pars=fit.pars` works with a parameter called `dur`, and `pars=dict(dur=10)` with a network that has one); a keyword given outright wins over it.
    A setting given as `ss.par` (e.g. `n_agents=ss.par(...)`, when it is fitted) takes its value from `pars=` too (`pars=fit.pars`).

    Args:
        pars (dict): `pars=`, with the values given by keyword (see `model_pars()`)
        kwargs (dict): the other keywords, the settings
        given (dict): the arguments that `ss.Sim()` names, with None for those not given
        mods (list): the modules of the model, whose parameters `pars=` can set

    Returns:
        The settings by name, and the values of the model's parameters by name (or None); a name that is neither raises a `KeyError` when the sim is initialized
    """
    if not (isinstance(pars, dict) and len(pars)):
        return kwargs, None
    pars = dict(pars)
    kwargs = kwargs | {key:pars.pop(key) for key in list(pars) if isinstance(kwargs.get(key), ss.par)}
    names = {name for mod in mods for name in (mod.pars.keys() if getattr(mod, 'graph', None) is not None else mod.pars.bounded())} # Those pars= can set (see ss.solvers.set_values())
    settings = {key:pars.pop(key) for key in list(pars) if (key in ss.SimPars.names() or key in ss.v3_deprecations.keys('Sim')) and key not in names}
    kwargs = sc.mergedicts({k:v for k,v in settings.items() if given.get(k) is None}, kwargs) # A keyword given outright wins
    return kwargs, pars or None


def given_modules(*inputs):
    """ The modules among a sim's inputs, alone or in lists or dicts (e.g. `diseases=[sir, sis]`) """
    mods = []
    for inp in inputs:
        for val in (inp or {}).values():
            vals = val.values() if isinstance(val, dict) else sc.tolist(val) if isinstance(val, (list, tuple, ss.Module)) else []
            mods += [v for v in vals if isinstance(v, ss.Module)]
    return mods


def make_disease(arg, names):
    """ Transitions given to the sim (alone or in a list, among modules) as one disease, named from its states (e.g. "sir"), whose results are also at the top of the sim's (`sim.results.I`) """
    items = sc.tolist(arg) if isinstance(arg, (list, tuple)) or ss.transitions.is_declaration(arg) else None
    if items is None:
        return arg
    flat = [obj for item in items for obj in (item if isinstance(item, (list, tuple)) else [item])]
    transitions = [obj for obj in flat if ss.transitions.is_declaration(obj)]
    if not transitions:
        return arg
    disease = ss.Disease(*transitions, _names=names)
    disease.setattribute('_implicit', True)
    return [obj for obj in flat if not ss.transitions.is_declaration(obj)] + [disease]


def copy_given(*inputs, copy_inputs=True):
    """ The sim's inputs merged and copied together, so the modules given are never run themselves and can be given to other sims, and refer to each other as given; with `copy_inputs=False`, the rest is copied, but not the modules """
    given = given_modules(*inputs)
    if not copy_inputs: # The sim's own, as given
        for mod in given:
            if mod.__dict__.get('sim') is not None or getattr(mod, '_given', False):
                errormsg = f'"{mod.name}" was already given to a sim, so it can\'t be given to another with copy_inputs=False, which runs the modules themselves: make it again, or give a copy (e.g. mod.copy())'
                raise ValueError(errormsg)
        memo = {id(mod):mod for mod in given} # Shared, so what refers to a module refers to the module itself
        return sc.mergedicts(*[sc.dcp(inp, memo=memo) for inp in inputs])
    for mod in given: # Marked, so that what is added to them later, which this sim would never see, raises (see ss.Module.check_not_given())
        mod.setattribute('_given', True)
    out = sc.mergedicts(*inputs, _copy=True)
    for mod in given_modules(out): # The sim's own copies are its to change, and hold their own results (also if the module given was given to a sim before)
        mod.setattribute('_given', False)
        if isinstance(getattr(mod, 'results', None), ss.Results):
            mod.results.setattribute('_given_to', None)
    for mod in given: # The modules given hold no results (the sim runs its copies), so reading one says where they are
        if isinstance(getattr(mod, 'results', None), ss.Results) and not len(mod.results):
            mod.results.setattribute('_given_to', mod.name)
    return out


class SimPars(ss.Pars):
    """
    The settings of a sim, as `sim.pars`; each is described in `ss.Sim`

    Typically made by the sim rather than by the user: `ss.Sim(stop=2030)`, then `sim.pars.stop`.

    Args:
        pars (dict): the settings, by name
        create (bool): whether a key that doesn't exist yet can be added
        kwargs (dict): the settings, by name
    """
    def __init__(self, pars=None, create=True, **kwargs):

        # General parameters
        self.label   = '' # The name of the sim, e.g. for its plots and in a MultiSim
        self.verbose = ss.options.verbose # How much to print: 0 (nothing), 1 (the solver, notes, and progress; default), or 2 (everything)

        # Population parameters
        self.n_agents  = 10e3 # The number of agents, if the model runs as agents: more give less noise, but run slower
        self.total_pop = None # The number of people (default: one per agent); each agent stands for total_pop/n_agents of them (sim.pop_scale)

        # Simulation parameters
        self.start     = None  # Start of the simulation: a date or a year gives it a calendar (default 2000 for a model in years, else 0)
        self.stop      = None  # End of the simulation, read as the start is: a date, a year (or a time in the unit from 0, with no calendar), or a duration from 0 (default: a run of 100 days in days, else 50 of the sim's unit)
        self.dur       = None  # The length of the run, e.g. ss.years(10) or a bare number (in years with a calendar of years, else in the sim's unit): the stop is the start plus it, or with a stop and no start, the start is the stop minus it
        self.dt        = None  # The timestep, in the sim's unit (default 1): smaller is more accurate, but slower
        self.rand_seed = 1     # The random seed, from which every module's random numbers are drawn, so the same seed gives the same results; if None, the seed is not reset

        # How the model is run; these only apply to modules with declared states and transitions (see ss.Disease)
        self.unit       = None   # The unit for bare numbers, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        self.solver     = 'auto' # 'auto', 'ode', 'binomial', 'gillespie' (exact), 'tau' (tau-leaping), or 'abm' (agents); or with options, ss.solvers.ode(), ss.solvers.binomial(), or ss.solvers.abm() (e.g. ss.solvers.abm(rescale=True); see ss.Sim)
        self.branching  = False  # Whether to run as a branching process: infection takes no one from the uninfected, nor gives anyone back (see ss.Sim)
        self.stratify   = None   # Properties of people to stratify every module by, e.g. dict(age=[0, 15, 65])
        self.group_results = None # The results to keep by group, by name, e.g. ['I', 'cases'] or ['sir.I']; the others keep only their totals (default: all, every few steps if every step would take more than about 100 MB: see ss.solvers.group_rows())

        # Demographic parameters
        self.aging = None # Whether agents age (by default, if there are births or deaths)

        # Modules
        self.step_order = 'type' # The order in which the modules step: 'type' (each kind at its place in the loop), or 'given' (the order they were given in)
        self.people = None
        self._reset_modules()

        self.update(pars, create=create, **kwargs)
        return

    _names = None # The names of the sim parameters: see names()
    _frozen = False # Whether the sim is initialized: see freeze()

    def freeze(self, frozen=True):
        """
        Once the sim is initialized, its settings belong to that run (`sim.new()` copies the sim as it was before), so setting one raises; but not its label, which only names it

        Args:
            frozen (bool): whether setting a value raises an error
        """
        object.__setattr__(self, '_frozen', frozen)
        return self

    def __setitem__(self, key, value):
        if self._frozen and key in self and key != 'label': # A key that does not exist raises as for any locked parameters
            val = repr(value) if len(repr(value)) < 40 else '...'
            errormsg = f'Cannot set sim.pars.{key} once the sim is initialized, since a run (and a rerun) uses the settings it had then; make a new sim with it instead, e.g. sim.new({key}={val})'
            raise RuntimeError(errormsg)
        return super().__setitem__(key, value)

    def __missing__(self, key):
        """ A module's parameters (added by name once it is initialized) by its default name, e.g. `sim.pars.sir` for an `ss.SIR` called `measles` (see `ss.utils.find_module()`) """
        if not isinstance(key, str) or key.startswith('_'):
            raise KeyError(key)
        mods = given_modules(self) + [val._module for val in self.values() if isinstance(val, ss.Pars) and isinstance(val._module, ss.Module)] # Before init(), the modules given; after, those whose parameters are here
        mod = ss.utils.find_module(mods, key, own=False)
        if mod is not None and dict.__contains__(self, mod.name):
            return dict.__getitem__(self, mod.name)
        raise KeyError(key)

    def __getattr__(self, attr):
        """ As `__missing__()`, for `sim.pars.sir` """
        try:
            return self.__missing__(attr)
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'") from None

    def check_key_mismatch(self, pars):
        """ As for any parameters, but a setting of Starsim 3's says what replaces it """
        ss.v3_deprecations.check('Sim', '__init__', pars)
        return super().check_key_mismatch(pars)

    @classmethod
    def names(cls):
        """ The names of the sim parameters (as opposed to keys added later, e.g. one per module) """
        if cls._names is None:
            cls._names = set(cls().keys())
        return cls._names

    def _reset_modules(self):
        """ Empty the list of modules of each kind (`custom` for those of no standard kind, and `modules` for those given to the sim as arguments, to be sorted by kind) """
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
        """
        Whether a setting has its default value

        Args:
            key (str): the name of the setting
        """
        return self[key] == SimPars()[key]

    def validate(self):
        """ Check the settings and the modules, and put them in the forms the sim uses """
        self.validate_sim_pars()
        self.validate_modules()
        return

    def validate_sim_pars(self):
        """ Check each of the settings """
        self.verbose = ss.parameters.check_verbose(self.verbose)
        self.validate_whole()
        self.validate_agents()
        if self.step_order not in ['type', 'given']:
            raise ValueError(f"step_order={self.step_order!r} is not 'type' or 'given'")
        self.solver = ss.solvers.as_solver(self.solver) # A solver by name is the solver itself, e.g. 'ode' is ss.solvers.ode(); the one that runs, if 'auto', is chosen when the sim is initialized (see ss.solvers.choose_solver())
        if self.total_pop is None: # One person per agent
            self.total_pop = self.n_agents
        return

    def validate_whole(self):
        """ Take a whole number given as a float as an integer where one is needed, e.g. rand_seed=2.0 from R """
        val = self.rand_seed
        if sc.isnumber(val) and not isinstance(val, (bool, np.bool_)) and val == int(val):
            dict.__setitem__(self, 'rand_seed', int(val))
        return

    def validate_agents(self):
        """ Make n_agents a whole number (the nearest, e.g. for a value a fit chose), and check it against the people given, if any """
        named = getattr(self.people, 'named', None)
        if named is not None and named.n_agents is None: # A named population is made at the sim's number of agents: see ss.solvers.build.init_named_people()
            self.n_agents = round(self.n_agents)
        elif self.people is not None:
            len_people = len(self.people)
            if (len_people != round(self.n_agents)) and not self.is_default('n_agents'):
                errormsg = f'Cannot set a custom value of n_agents ({self.n_agents}) that does not match a pre-created People object ({len_people}); use n_agents to create the People object instead'
                raise ValueError(errormsg)
            self.n_agents = len_people
        elif self.n_agents is None: # As if left out
            self.n_agents = int(SimPars().n_agents)
        elif self.n_agents == 0:
            errormsg = 'n_agents=0 gives no agents to run; for a compartmental run, leave out n_agents (or give solver="ode")'
            raise ValueError(errormsg)
        else:
            self.n_agents = round(self.n_agents)
        return

    @staticmethod
    def population():
        """ Everyone alive, as one state, to hold the births and deaths of a sim with no declared disease (see `ss.demographics.Flows`) """
        pop = ss.Disease(N=ss.State(label='Alive'), default_state='N', name='population')
        pop.sets_unit = False # Its typed defaults say nothing of the units of the model's bare numbers
        return pop

    def validate_modules(self):
        """ Check the modules given, make modules of what are not yet (e.g. 'sir'), and sort them by kind """
        self.validate_demographics() # Before the modules are made
        modmap = ss.modules.module_map()
        for modkey in modmap.keys():
            if not isinstance(self[modkey], ss.ndict):
                self[modkey] = flatten(self[modkey]) # E.g. a helper that returns several interventions
        modules = self.convert_modules()
        modules = self.expand_modules(modules)
        flows = any(isinstance(mod, ss.demographics.Flows) for source,mod in modules)
        if flows and not any(getattr(mod, 'graph', None) is not None for source,mod in modules): # With no declared disease, births and deaths declared for the population are held by everyone alive, a declared module like any other
            modules.append(('demographics', self.population()))
        self._reset_modules()
        products = []
        names = {} # Module names must be unique across all types, since they share namespaces (e.g. sim.people.<name>, sim.results.<name>)
        for source, mod in modules:
            modkey = module_kind(mod) if source == 'modules' else source # Sorted by kind (of no standard kind: kept in sim.custom), or where they were given
            self.validate_name(mod, modkey, names)
            if mod.has_product and not any(mod.product is p for p,d in products): # Products share the same namespaces; the same product can be used by more than one intervention
                products.append((mod.product, mod))
            self[modkey].append(mod)
        order = [mod for source,mod in modules if source == 'modules'] + [mod for source,mod in modules if source != 'modules']
        order.sort(key=lambda mod: getattr(mod, '_model_index', np.inf)) # The model's modules (given to the sim as arguments) first, in the order given
        for i,mod in enumerate(order):
            mod._given_index = i # Its place in the order given, for step_order='given': see ss.Loop.collect_given()
        for product, delivery in products: # Named after all the other modules, so that a product named for its kind ("treatment") never takes a name the user gave a module
            self.validate_name(product, 'products', names, delivery=delivery)
        return

    @staticmethod
    def expand_modules(modules):
        """
        Replace any module that defines an `expand()` method with the modules it returns, e.g. `ssl.HybridNet()` → four networks

        Args:
            modules (list): (source, module) tuples, as returned by `convert_modules()`
        """
        return [(source, m) for source, mod in modules for m in (mod.expand() if hasattr(mod, 'expand') else [mod])]

    @staticmethod
    def validate_name(mod, modkey, names, delivery=None):
        """
        Ensure the module name is unique, renaming it if it is the default name (e.g. "sir" → "sir_1")

        A product keeps its name if it is free; otherwise, if the name is the default (e.g. "vaccine", for a product that a delivery
        made from its keywords, `campaign = ss.vaccinate(efficacy=0.9, ...)`) or that of a delivery (`ss.screen(tbsim.DST(tb), name='dst')`),
        it is named for its delivery ("campaign_product", "dst_product"), with no warning

        Args:
            mod (`ss.Module`): the module to validate
            modkey (str): the type of module, e.g. "diseases"
            names (dict): the names used so far, mapped to their module type; updated in place
            delivery (`ss.Delivery`): for a product, the delivery it was found in
        """
        name = mod.name
        taken = [attr for attr in dir(ss.Results) if not attr.startswith('_')] # The module's results are sim.results.<name>, so a method's name is taken
        used = list(names) + taken
        if name in names or name in taken:
            if delivery is not None and (mod._name_source == 'class' or names.get(name) == 'interventions'): # A product named for its delivery
                mod.name = mod.label = sc.uniquename(f'{delivery.name}_product', used, style='_%d')
            elif mod._name_source != 'given': # A name the user did not give (the default for its class, or its variable's) is simply numbered (e.g. a second clip_edges is clip_edges_1)
                mod.name = sc.uniquename(name, used, style='_%d')
            elif name in taken:
                errormsg = f'A module cannot be called "{name}", since sim.results.{name} is a method of the results; please give it another name (the names taken are {sc.strjoin(taken)})'
                raise ValueError(errormsg)
            else:
                errormsg = f'Cannot add module "{name}" to {modkey}: a module with that name already exists in {names[name]}; please give one of them a different name'
                raise ValueError(errormsg)
        names[mod.name] = modkey
        return

    def validate_demographics(self):
        """ Put the demographics given in the form the sim uses, and decide whether agents age """
        # demographics=True is births and deaths at the default rates (20 and 10 per 1,000 people per year), declared for the population
        demog = self.demographics
        if demog in [True, 1]:
            demog = [ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.01))]
        elif demog in [False, 0]:
            demog = ss.ndict()

        # Births, deaths, and migration declared for the population are held together, until the sim adds them to a disease: see ss.demographics.Flows
        if not isinstance(demog, ss.ndict):
            demog = flatten(demog) # ss.migration() gives a list of transitions
            flows = [mod for mod in demog if isinstance(mod, ss.Transition)]
            if flows:
                demog = [mod for mod in demog if not isinstance(mod, ss.Transition)] + [ss.demographics.Flows(flows)]
        self.demographics = demog

        if self.aging is None: # Agents age if anyone is born or dies
            self.aging = bool(self.demographics)
        return

    def convert_modules(self):
        """
        Make a module of each one given in another form: by name ('sir'), as a dict (`dict(type='sir', dur_inf=6)`), as a class, or as a function (which becomes its `step()`)

        Returns a flat list of (source_key, ss.Module) tuples, where the source is the kind it was given as (e.g. 'diseases', or 'modules')
        """
        out = []
        classes = ss.modules.find_modules() # Each module class to choose from by name, by kind, e.g. ss.SIR in diseases
        classes['modules'] = classes['custom'] = sc.obj({k:v for vv in classes.values() for k,v in vv.items()}) # Modules of any kind, to sort by kind (those given to the sim as arguments), or of no standard kind, kept where they are given (custom=)
        for modkey in classes.keys():
            for i,mod in enumerate(self.given_list(modkey)):
                out.append((modkey, as_module(mod, modkey, classes[modkey], i)))
        return out

    def given_list(self, modkey):
        """ The modules given of one kind, as a list: from a list, a dict of them by name (e.g. `networks=dict(main=..., casual=...)`), an `ss.ndict`, or one alone """
        given = self[modkey]
        if isinstance(given, list) and len(given) == 1 and isinstance(given[0], dict): # A dict, wrapped in a list on the way in
            given = given[0]
        if isinstance(given, dict) and not isinstance(given, ss.ndict) and len(given) and all(isinstance(v, ss.Module) for v in given.values()): # Modules by name
            modlist = []
            for name,mod in given.items():
                if mod._name_source != 'given': # The key names it, as name= would
                    mod._set_name(name)
                modlist.append(mod)
        elif isinstance(self[modkey], list):
            modlist = self[modkey]
        elif isinstance(self[modkey], ss.ndict): # Modules by name, checked below for their kind
            modlist = list(self[modkey].values())
        else: # One module, e.g. a dict of its parameters
            modlist = sc.tolist(self[modkey])
        return [m for item in modlist for m in (item.values() if isinstance(item, ss.ndict) else [item])] # A set of modules given as one (e.g. the serotypes of ssl.Dengue), among others


def as_module(mod, modkey, classes, i=0):
    """
    A module, from one given to the sim in any form: by name ('sir'), as a dict of its type and parameters (`dict(type='sir', dur_inf=6)`), as a class, or as a function (which becomes its `step()`)

    Args:
        mod (str/dict/type/func/Module): the module, as given
        modkey (str): the kind it was given as, e.g. 'diseases' (or 'modules' or 'custom', for any kind)
        classes (dict): the classes it can be by name, e.g. {'sir': ss.SIR}
        i (int): its place among those of its kind, for the error if it is not one
    """
    expected_cls = ss.modules.module_map().get(modkey, ss.Module) # The base class of the kind, e.g. ss.Disease

    if isinstance(mod, str): # By name, e.g. 'sir', as a dict with only its type
        mod = dict(type=mod)

    if isinstance(mod, type) and issubclass(mod, ss.Module): # A class, e.g. ss.SIR
        mod = mod()

    if isinstance(mod, dict): # A dict of its type and parameters
        if 'type' not in mod:
            errormsg = f'When specifying a {modkey} module with a dict, one of the keys must be "type"; you supplied {mod}'
            raise ValueError(errormsg)
        modtype = mod.pop('type')
        if isinstance(modtype, str): # E.g. dict(type='sir', dur_inf=6), in lowercase as in the map
            modtype = modtype.lower()
            if modtype not in classes.keys():
                errormsg = f'Invalid module name "{modtype}" for "{modkey}"; must be one of {classes.keys()}'
                raise TypeError(errormsg)
            modcls = classes[modtype]
        elif modtype in classes.values(): # A class, e.g. dict(type=ss.SIR, dur_inf=6)
            modcls = modtype
        else:
            errormsg = f'Invalid module class "{modtype}" for "{modkey}"; must be one of {classes.values()}'
            raise TypeError(errormsg)
        mod = modcls(**mod)

    if not isinstance(mod, ss.Module) and callable(mod): # A function, as the step() of a module of this kind
        mod = expected_cls.from_func(mod)

    structural = modkey in ['networks', 'diseases'] # The sim relies on what these are; a module in any other slot only needs to step, e.g. ss.modify() in interventions=
    if not isinstance(mod, expected_cls if structural else (expected_cls, ss.Module)):
        errormsg = f'Was expecting {modkey} entry {i} to be {expected_cls.__name__}, but was {type(mod).__name__} instead'
        raise TypeError(errormsg)
    return mod
