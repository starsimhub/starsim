"""
The settings of a sim (`ss.SimPars`, as `sim.pars`), and how the modules given to a sim are checked and sorted by kind

These are kept apart from `ss.Sim` so that the sim's own file is about running it. Each setting is described in `ss.Sim`.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss

approximations = ['note', 'allow', 'error'] # What a sim does with each approximation a solver makes: see ss.Sim(approximations=)
rescale_options = dict(threshold=0.05, factor=1.2) # The options of dynamic rescaling, and their defaults: see ss.Sim(rescale=)


def flatten(items):
    """ Items given alone or in a list, with one level of lists inside it flattened, e.g. [sir, [vx1, vx2]] → [sir, vx1, vx2] """
    return [item for entry in sc.tolist(items) for item in (entry if isinstance(entry, list) else [entry])]


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
        self.rescale   = False # Whether to rescale the agents dynamically: True, or a dict of its options, e.g. dict(threshold=0.05, factor=1.2) (see ss.Sim)
        self.record_people = True # Whether to record the results about people (alive, died, etc.) every step; False saves a little time and memory

        # Simulation parameters
        self.start     = None  # Start of the simulation: a date or a year gives it a calendar (default 2000 for a model in years, else 0)
        self.stop      = None  # End of the simulation, read as the start is: a date, a year (or a time in the unit from 0, with no calendar), or a duration from 0 (default: a run of 100 days in days, else 50 of the sim's unit)
        self.dur       = None  # The length of the run, e.g. ss.years(10) or a bare number (in years with a calendar of years, else in the sim's unit): the stop is the start plus it, or with a stop and no start, the start is the stop minus it
        self.dt        = None  # The timestep, in the sim's unit (default 1): smaller is more accurate, but slower
        self.rand_seed = 1     # The random seed, from which every module's random numbers are drawn, so the same seed gives the same results; if None, the seed is not reset
        self.even_draws = False # Whether to space each call's random numbers evenly for agents, for less variable runs (see ss.Sim)
        self.keep_times = None  # Whether a declared disease run as agents keeps the time of entering every state, and the history of every transition; off by default to save memory (None: ss.options.keep_times)

        # How the model is run; these only apply to modules with declared states and transitions (see ss.Disease)
        self.unit       = None   # The unit for bare numbers, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        self.solver     = 'auto' # 'auto', 'ode', 'binomial', 'gillespie' (exact), 'tau' (tau-leaping), or 'abm' (agents); or with options, ss.solvers.ode(), ss.solvers.binomial(), or ss.solvers.abm() (see ss.Sim)
        self.branching  = False  # Whether to run as a branching process: infection takes no one from the uninfected, nor gives anyone back (see ss.Sim)
        self.approximations = 'note' # What to do when a solver approximates part of the model: 'note' prints a note; 'allow' keeps the note in sim.notes without printing it; 'error' makes it an error
        self.require_kernels = False # Whether to raise an error if a compartmental sim can't be run all at once in the compiled kernels, and would instead run step by step in plain Python (which can be 100 times slower); useful for keeping a fit or a large batch of runs fast
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

    def check_key_mismatch(self, pars):
        """ As for any parameters, but a setting of Starsim 3's says what replaces it; `custom` holds the modules of no standard type, which are given with `modules=` """
        ss.v3_deprecations.check('Sim', '__init__', pars)
        if 'custom' in pars and not isinstance(pars['custom'], ss.ndict): # Only as part of a whole SimPars, e.g. ss.Sim(pars=ss.SimPars(...))
            available = [key for key in self.keys() if key != 'custom']
            raise sc.KeyNotFoundError(f"Key(s) ['custom'] not found; available keys are {available}{ss.v3_deprecations.hint(keys='custom')}")
        return super().check_key_mismatch(pars)

    @classmethod
    def names(cls):
        """ The names of the sim parameters (as opposed to keys added later, e.g. one per module) """
        if cls._names is None:
            cls._names = set(cls().keys())
        return cls._names

    def _reset_modules(self):
        """ Empty the list of modules of each kind (`custom` for those of no standard kind, and `modules` for those given with `modules=`) """
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
        if self.approximations not in approximations:
            raise ValueError(f'approximations={self.approximations!r} is not one of {sc.strjoin(approximations)}')
        if self.even_draws and not ss.options.crn:
            raise ValueError('even_draws=True needs common random numbers, but ss.options.crn is False')
        if self.keep_times is None:
            self.keep_times = ss.options.keep_times
        self.solver = ss.solvers.as_solver(self.solver) # A solver by name is the solver itself, e.g. 'ode' is ss.solvers.ode(); the one that runs, if 'auto', is chosen when the sim is initialized (see ss.solvers.choose_solver())
        self.validate_rescale()
        if self.total_pop is None: # One person per agent
            self.total_pop = self.n_agents
        return

    def validate_rescale(self):
        """ Dynamic rescaling is True, False, or a dict of its options (see `rescale_options`) """
        val = self.rescale
        if isinstance(val, (bool, np.bool_)):
            return
        if not isinstance(val, dict):
            raise TypeError(f'rescale= is True, False, or a dict of its options, e.g. dict(threshold=0.05, factor=1.2), not {val!r}')
        unknown = [key for key in val if key not in rescale_options]
        if unknown:
            raise TypeError(f'rescale= has no option {sc.strjoin(unknown)}; its options are {sc.strjoin(rescale_options)}')
        threshold, factor = val.get('threshold', rescale_options['threshold']), val.get('factor', rescale_options['factor'])
        if not (0 < threshold < 1):
            raise ValueError(f'rescale=dict(threshold={threshold}) is a fraction of the agents, between 0 and 1')
        if not factor > 1:
            raise ValueError(f'rescale=dict(factor={factor}) is the least factor the scale grows by, more than 1')
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
        if named is not None and named.n_agents is None: # A named population is made at the sim's number of agents: see ss.solvers.choose.init_named_people()
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
            if source == 'modules': # Sorted by kind
                modkey = next((key for key, modclass in modmap.items() if isinstance(mod, modclass)), 'custom')
            else: # Where they were given
                modkey = source
            self.validate_name(mod, modkey, names)
            if mod.has_product and not any(mod.product is p for p,d in products): # Products share the same namespaces; the same product can be used by more than one intervention
                products.append((mod.product, mod))
            self[modkey].append(mod)
        for i,mod in enumerate([mod for source,mod in modules if source == 'modules'] + [mod for source,mod in modules if source != 'modules']):
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
            if delivery is not None and (mod._default_name or names.get(name) == 'interventions'): # A product named for its delivery
                mod.name = mod.label = sc.uniquename(f'{delivery.name}_product', used, style='_%d')
            elif mod._default_name: # A name the user did not give is simply numbered (e.g. a second clip_edges is clip_edges_1)
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
        modmap = ss.modules.module_map() # The base class of each kind, e.g. ss.Disease
        modmap['modules'] = ss.Module # Enable passing in modules of any type via `Pars(modules=[...])`
        supported_modules = ss.modules.find_modules() # Each individual module class option, e.g. ss.SIR
        supported_modules['modules'] = sc.obj({k:v for vv in supported_modules.values() for k,v in vv.items()})

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
            elif isinstance(self[modkey], ss.ndict): # Modules by name, checked below for their kind
                modlist = list(self[modkey].values())
            else: # One module, e.g. a dict of its parameters
                modlist = sc.tolist(self[modkey])

            modlist = [m for item in modlist for m in (item.values() if isinstance(item, ss.ndict) else [item])] # A set of modules given as one (e.g. the serotypes of ssl.Dengue), among others
            for i,mod in enumerate(modlist):

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
                        if modtype not in moddictkeys:
                            errormsg = f'Invalid module name "{modtype}" for "{modkey}"; must be one of {moddictkeys}'
                            raise TypeError(errormsg)
                        modcls = ssmoddict[modtype]
                    elif modtype in moddictvals: # A class, e.g. dict(type=ss.SIR, dur_inf=6)
                        modcls = modtype
                    else:
                        errormsg = f'Invalid module class "{modtype}" for "{modkey}"; must be one of {moddictvals}'
                        raise TypeError(errormsg)
                    mod = modcls(**mod)

                if not isinstance(mod, ss.Module) and callable(mod): # A function, as the step() of a module of this kind
                    mod = expected_cls.from_func(mod)

                structural = modkey in ['networks', 'diseases'] # The sim relies on what these are; a module in any other slot only needs to step, e.g. ss.modify() in interventions=
                if isinstance(expected_cls, type) and not isinstance(mod, expected_cls if structural else (expected_cls, ss.Module)):
                    errormsg = f'Was expecting {modkey} entry {i} to be {expected_cls.__name__}, but was {type(mod).__name__} instead'
                    raise TypeError(errormsg)

                out.append((modkey, mod))

        return out
