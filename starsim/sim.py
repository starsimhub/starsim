"""
Define core Sim classes
"""
import os
import inspect
import itertools
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt



class Sim(ss.Base):
    """
    The Sim object

    All Starsim simulations run via the Sim class. It is responsible for initializing
    and running all modules and generating results. The sim orchestrates `ss.People`,
    `ss.Network`, `ss.Disease`, `ss.Intervention`, `ss.Demographics`, and other
    modules via the `ss.Loop` integration loop.

    Args:
        model (Module/list/dict): the model: a module or a list of modules, which are sorted by type as for `modules`.
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
        data (df): a dataframe (or dict) of data, with a column "time" plus data of the form "module.result", e.g. "hiv.new_infections" (used for plotting only)
        kwargs (dict): merged with pars; see ss.SimPars for all parameter values

    Note:
        Modules can be supplied either via individual keyword arguments (`diseases`, `networks`, etc.)
        or all together via the `modules` argument. When using `modules`, Starsim automatically sorts
        them by type. The individual kwargs and `modules` can be mixed; they are merged together.

    Examples:
        ```python
        sim = ss.Sim(diseases='sir', networks='random') # Simplest Starsim sim; equivalent to ss.demo()
        sim = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()) # Equivalent using objects instead of strings
        sim = ss.Sim(diseases=['sir', ss.SIS()], networks=['random', 'mf']) # Example using list inputs; can mix and match types
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
        for arg in models: # The model: modules, or lists of them; or the sim parameters, as a dict
            if isinstance(arg, dict):
                kwargs = sc.mergedicts({k:v for k,v in arg.items() if given.get(k) is None}, kwargs) # A keyword given outright wins over the same key in the dict
            elif not (isinstance(arg, ss.Module) or (isinstance(arg, (list, tuple)) and all(isinstance(mod, ss.Module) for mod in arg))):
                errormsg = f'ss.Sim() takes the model (modules, or lists of them) or a dict of parameters as positional arguments, not {type(arg).__name__}; give everything else by keyword, e.g. label="..."'
                raise TypeError(errormsg)
        positional = [arg for arg in models if not isinstance(arg, dict)]
        if positional:
            modules = sc.mergelists(*positional, modules)
        self._name_modules(ss.utils.caller_names(1), demographics, connectors, networks, interventions, diseases, analyzers, custom, modules)
        interventions = self._order_steps(interventions)
        self.par_values = None # Values of model parameters by name, e.g. pars=dict(beta=0.1) or pars=fit.pars
        self.jax = False # Whether a whole run of the ODE is taken in JAX (as ss.parallel() does for many runs with different values): see ss.gradients
        if isinstance(pars, dict) and len(pars): # Values of the model's parameters by name; a sim setting only by its own name, if no parameter of the model has it (so pars=fit.pars works with a parameter called dur)
            given = [mod for arg in [modules, demographics, connectors, networks, interventions, diseases, analyzers, custom] for mod in sc.tolist(arg) if isinstance(mod, ss.Module)]
            names = {name for mod in given for name in mod.pars.keys()}
            settings = [key for key in pars if key in self.pars and key not in names]
            values = {key:val for key,val in pars.items() if key not in settings} # A name that is neither raises ss.errors.UnknownParameter when the sim is initialized
            self.par_values, pars = values or None, {key:pars[key] for key in settings} or None
            if pars and pars.get('interventions') is not None: # Modules given in pars (e.g. by a subclass of ss.Sim): in order too, before they are copied
                pars['interventions'] = self._order_steps(pars['interventions'])
        args = dict(label=label, people=people, demographics=demographics, connectors=connectors,
                    networks=networks, interventions=interventions, diseases=diseases, 
                    analyzers=analyzers, custom=custom, modules=modules)
        args = {key:val for key,val in args.items() if val is not None} # Remove None inputs
        input_pars = sc.mergedicts(pars, args, kwargs, _copy=True) # Copied together, so the modules given are never run themselves and can be given to other sims, and refer to each other as given
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
        self.summary = None  # For storing a summary of the results
        self.filename = None # Store the filename, if saved
        self.diagnostics = None # Store diagnostics results
        self._made = self._settings() # The settings as made, to find the ones set afterwards (see set_after())
        self._set_after = {}
        return

    def _settings(self):
        """ The sim settings, not the modules or people (which would be found again by the search for distributions) """
        skip = ss.modules.module_types() + ['custom', 'modules', 'people']
        return {key:val for key,val in self.pars.items() if key not in skip}

    def set_after(self):
        """ The sim settings set after the sim was made and before it was initialized (e.g. `sim.pars.stop = 30`), which `sim.new()` keeps """
        made = self.__dict__.get('_made')
        if made is not None:
            for key,val in self._settings().items():
                if val is not made.get(key):
                    self._set_after[key] = sc.dcp(val)
        return self.__dict__.get('_set_after', {})

    @staticmethod
    def _name_modules(names, *args):
        """ A module built with the plain ss.Disease() constructor takes its name from the variable it is assigned to """
        for arg in args:
            mods = list(sc.tolist(arg))
            for mod in mods:
                named = [mod] if type(mod) is ss.Disease or isinstance(mod, (ss.modify, ss.Delivery, ss.PartnerNet)) else []
                named += [getattr(mod, 'product', None)]*isinstance(mod, ss.Delivery)
                for obj in named:
                    if obj is not None and obj._default_name:
                        name = names.get(id(obj))
                        if name is None and getattr(obj, 'graph', None) is not None: # Not bound to one variable, e.g. built in a helper: named from its states instead, e.g. 'sir'
                            name = ''.join(s.name[0].lower() for s in obj.graph.people_states) or None
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

    @property
    def label_to_name(self):
        """ Return a mapping from module labels to module names, e.g. {'HIV': 'hiv', 'Pregnancy': 'pregnancy'} """
        return {mod.label: mod.name for mod in self.modules}

    def init(self, force=False, timer=False, **kwargs):
        """
        Perform all initializations for the sim

        Args:
            force (bool): whether to overwrite sim attributes even if they already exist
            timer (bool): if True, count the time required for initialization (otherwise just count run time)
            kwargs (dict): passed to ss.People()
        """
        # Handle the timer
        self.timer = sc.timer() # Store a timer for keeping track of how long the run takes
        if timer:
            self.timer.start()

        # Validation and initialization -- this is "pre"
        self.set_after() # Before the settings are converted
        self._made = None # And only then
        np.random.seed(self.pars.rand_seed) # Reset the seed before the population is created -- shouldn't matter if only using Dist objects
        self.pars.lock(False) # Allow parameters to be added during initialization
        self.pars.validate() # Validate parameters
        self._init_solver() # Choose the unit and the solver
        self._init_time() # Initialize time
        self.init_people(**kwargs) # Initialize the people
        self.init_module_attrs(force=force) # Load specified modules from parameters, initialize them, and move them to the Sim object
        self._init_modules_pre()
        ss.interventions.check_undriven(self)

        # Final initializations -- this is "post"
        if self.pars.rescale:
            self._init_rescale()
        self._init_dists() # Initialize distributions
        self._init_people_vals() # Initialize the values in all the states and networks
        if self.solver == 'abm' and self.groups.strats:
            self.groups.init_vals()
        elif self.solver != 'abm' and self.groups.strats: # One row per group: age is the middle of the group
            for d,strat in enumerate(self.groups.strats):
                if strat.kind == 'age':
                    self.people.age[self.people.auids] = self.groups.group_vals(d, strat.mids, outside=strat.mids @ strat.split)
                    self.people.age.__class__ = ss.arrays.GroupAges # So that a function comparing ages with an age inside a group raises
                    self.people.age.edges = strat.edges
                elif strat.female is not None: # A sex dimension is people.female
                    self.people.female[self.people.auids] = self.groups.group_vals(d, np.arange(strat.n) == strat.female, outside=False)
        self._init_modules_post() # Initialize the module values
        if self.solver == 'abm':
            for mod in self.declared: # Those placed in a state nobody waits in move on at once, once every module has its people
                mod.engine.settle_start()
        for crossing in self.crossings: # Once every disease in it has its initial conditions
            crossing.init()
        self.init_results() # Initialize the results
        self.init_data() # Initialize the data
        self.loop.init() # Initialize the integration loop
        self.lock_pars() # Don't allow new parameters to be added by mistake

        self.verbose = self.pars.verbose # Store a run-specific value of verbose
        if self.solver != 'abm': # One row per group, so properties that are not stratifications have no values to give
            dims = [strat.name for strat in self.groups.strats]
            missing = [key for key in ['age', 'female'] if key not in dims and not (key == 'female' and 'sex' in dims)]
            for key in missing:
                self.people.__dict__.pop(key, None)
            self.people._no_groups = missing

        # It's initialized
        self.initialized = True
        if self.verbose and self.solver != 'abm': # As a check that the model and its seed are what was meant
            for dis in self.declared:
                if dis.graph.transmissions:
                    try:
                        ngm = dis.engine.next_generation()
                    except Exception: # Not every model has one; the line is only a check
                        ngm = None
                    if ngm is not None:
                        R0 = float(np.max(np.abs(np.linalg.eigvals(ngm))))
                        count = float(dis.engine.count(dis.graph.infectious))
                        shown = f'{count:,.0f}' if count >= 100 or count == round(count) else f'{count:.3g}' # A share, with total_pop=1, is not 0
                        print(f'{dis.name}: R0 = {R0:.3g}, with {shown} infectious at the start')
        if timer:
            self.elapsed = self.timer.toc(label='init', output=True, verbose=self.verbose)
        if self.diagnostics and self.diagnostics.states is not None: # Need not None since dict is empty at this point
            self.diagnostics.store_states(key='init')
        return self

    def set_par_values(self):
        """ Apply parameter values given by name (pars=dict(beta=0.1), or 'sir.beta' if the name is used by more than one module; for a parameter with a value per level, 'beta[a]' for one level) """
        for mod in self.declared: # A parameter shared with a module outside this sim (e.g. one model built twice from the same ss.par) belongs to the first module here that uses it
            for p in mod.graph.pars.values():
                if p.module is None or not any(p.module is m for m in self.declared):
                    p.module = mod
        coded = [mod for key in ss.modules.module_types() + ['custom'] for mod in self.pars[key].values() if isinstance(mod, ss.Module) and getattr(mod, 'graph', None) is None]
        values = {}
        levels = sc.ddict(dict)
        for key,val in (self.par_values or {}).items():
            base, bracket, label = key.partition('[')
            if bracket and key.endswith(']'):
                levels[base][label[:-1]] = val
            else:
                values[key] = val
        for base,given in levels.items():
            values[base] = self.by_levels(base, given, values.get(base))
        for key,val in values.items():
            modname, _, name = key.rpartition('.')
            mods = [mod for mod in self.declared if name in mod.pars and modname in ['', mod.name]]
            if len(mods) > 1 and all(mod.graph.pars.get(name) is mods[0].graph.pars.get(name) is not None for mod in mods): # One parameter shared by several modules
                mods = mods[:1]
            if not mods: # A parameter of a module written as code: one given with bounds (e.g. when fitting it), else any
                mods = [mod for mod in coded if name in mod.pars.bounded() and modname in ['', mod.name]]
                mods = mods or [mod for mod in coded if name in mod.pars and modname in ['', mod.name]]
            if len(mods) != 1:
                options = [f'{mod.name}.{k}' for mod in self.declared for k in mod.graph.pars] + [f'{mod.name}.{k}' for mod in coded for k in mod.pars.bounded()]
                raise ss.errors.UnknownParameter(f'pars= has a value for "{key}", which {"matches no parameter" if not mods else "is a parameter of more than one module"}. '
                                          f'Parameters are {sc.strjoin(options)}{ss.transitions.did_you_mean(key, options)}')
            mods[0].pars.update({name:val})
        return

    def by_levels(self, key, given, whole=None):
        """ A parameter with a value per level (`ss.par(ss.by(patch=[...]))`), with the values of some levels replaced, e.g. from `pars={'beta[a]': 0.3}` """
        modname, _, name = key.rpartition('.')
        mods = [mod for mod in self.declared if name in mod.graph.pars and modname in ['', mod.name]]
        raw = whole if whole is not None else (mods[0].pars.get(name) if mods else None)
        raw = raw.raw if isinstance(raw, ss.par) else raw # E.g. given as pars=dict(beta=ss.par(ss.by(patch=[...]), bounds=...))
        if not isinstance(raw, ss.by) or raw.dim is None:
            raise ss.errors.UnknownParameter(f'pars= has values for levels of "{key}" ({sc.strjoin(given.keys())}), which is not a parameter with a value for each level of one stratification, e.g. ss.par(ss.by(patch=[0.2, 0.3]))')
        strats = [strat for mod in self.declared for strat in mod.graph.strats] + list(self.groups.sim_strats)
        strat = next((strat for strat in strats if strat.name == raw.dim), None)
        labels = [str(label) for label in strat.labels] if strat is not None else [str(i) for i in range(len(raw.values))]
        vals = list(sc.tolist(raw.values))
        for label,val in given.items():
            if label not in labels:
                raise ss.errors.UnknownParameter(f'pars= has a value for "{key}[{label}]", but "{raw.dim}" has no level "{label}"; its levels are {sc.strjoin(labels)}')
            vals[labels.index(label)] = val
        return ss.by(**{raw.dim:vals})

    def _init_solver(self):
        """ Work out the sim's unit and solver: see `ss.solvers.choose_solver()` """
        return ss.solvers.choose_solver(self)

    def _resolve_path_triggers(self):
        """
        A trigger on a transition of another module by its path (`on=ss.state('pregnancy.delivery')`, for a module declared
        before that one exists) is that transition's event, as `ss.on(preg.delivery)` is, not the condition of having gone through
        it: so it happens once, in the step, and to whoever the event also happens to (e.g. the fetuses at a delivery)
        """
        mods = {mod.name:mod for mod in self.declared}
        for mod in self.declared:
            for tr in mod.graph.transitions.values():
                x = getattr(tr.trigger, 'x', None)
                if tr.trigger is None or tr.trigger.event or not isinstance(x, ss.conditions.Cond) or x.op != 'state' or not isinstance(x.args[0], str):
                    continue
                other, name = x.args[0], x.args[1]
                if other != mod.name and other in mods and name in mods[other].graph.transitions:
                    tr.trigger = ss.on(ss.conditions.Event('tr', ss.conditions.Named(other, name)))
        return

    def _init_lifetables(self):
        """ In a compartmental run, a life table (`ss.death(lifetable=)`) is a death rate for each age group; as agents, each person's age at death is drawn (see `ss.solvers.Agents.step_lifetables()`) """
        for mod in self.declared:
            for name,tr in mod.graph.transitions.items():
                table = getattr(tr, 'lifetable', None)
                if table is None or self.solver == 'abm':
                    continue
                age = next((strat for strat in self.groups.strats if strat.kind == 'age' and strat.edges is not None), None)
                if age is None:
                    raise ss.CapabilityError(f'"{name}" of "{mod.name}" draws ages at death from a life table, which needs each person\'s age; it cannot run with solver="{self.solver}" unless the model is stratified by age.\n'
                                             f'     Stratify by age (e.g. sir.stratify(age=[0, 5, 15, 50, 65])), or run with solver="abm".')
                p = tr.rate.leaves[0]
                p.value = ss.by(**{age.name:[ss.peryear(h) for h in table.hazards(age.edges)]})
        return

    def check_par_values(self):
        """ Check that every parameter of the model has a value (`ss.errors.UnvaluedParameter`), inside its bounds (`ss.errors.OutOfBounds`) """
        for mod in self.declared:
            for name,p in mod.graph.pars.items():
                raw = p.raw
                if raw is None:
                    raise ss.errors.UnvaluedParameter(f'the parameter "{name}" of "{mod.name}" has no value. Give it one (ss.par(0.1)), or bounds to start from (ss.par(bounds=[0.01, 0.5])), or a value by name (ss.Sim(pars=dict({name}=...))).')
                if p.bounds is None:
                    continue
                val = raw.value if isinstance(raw, ss.TimePar) else raw.array if isinstance(raw, ss.by) else raw
                if not (sc.isnumber(val) or isinstance(val, np.ndarray)): # E.g. a distribution or a function
                    continue
                low, high = [(type(raw)(b) if isinstance(raw, ss.TimePar) else b).value if isinstance(b, ss.TimePar) else b for b in p.bounds] # A typed bound in the unit of the value
                tol = 1e-9*abs(high - low) # For rounding, e.g. on a log scale
                val = np.asarray(val, dtype=float)
                if np.any(val < low - tol) or np.any(val > high + tol):
                    raise ss.errors.OutOfBounds(f'the parameter "{name}" of "{mod.name}" is {ss.transitions.value_str(raw)}, outside its bounds [{low:n}, {high:n}].')
        return

    def new(self, pars=None, **kwargs):
        """
        A new, uninitialized sim built from the same inputs as this one, optionally with other values

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            kwargs (dict): sim arguments to change, e.g. `rand_seed=2` or `solver='abm'`; `None` goes back to the default, e.g. `n_agents=None` for a compartmental run of an agent-based sim; also `data=`
        """
        if not hasattr(self, 'inputs'):
            errormsg = 'sim.new() needs the inputs the sim was built from, which this sim (e.g. one saved by an earlier version) does not have'
            raise ValueError(errormsg)
        data = kwargs.pop('data', self.data) # Data is not a sim parameter, so it is passed on separately
        if type(self) is Sim:
            inputs = sc.mergedicts(sc.dcp(self.inputs), kwargs)
            inputs = {key:val for key,val in inputs.items() if not (key in kwargs and val is None)} # As if it had not been given
            sim = Sim(inputs, data=sc.dcp(data))
        else: # A subclass is made again by the call that made it
            call = self.__dict__.get('_call')
            if call is None:
                errormsg = f'sim.new() cannot make this {type(self).__name__} again, since the arguments it was made with were not kept (its arguments cannot be copied). Make it again by calling {type(self).__name__}(...)'
                raise ValueError(errormsg)
            args, kw = sc.dcp(call)
            kw = sc.mergedicts(kw, kwargs)
            kw = {key:val for key,val in kw.items() if not (key in kwargs and val is None)}
            sim = type(self)(*args, **kw)
            sim.data = sc.dcp(data)
        for key,val in self.set_after().items(): # Settings changed after the sim was made, e.g. sim.pars.stop = 30
            if key not in kwargs:
                sim.pars[key] = sc.dcp(val)
        sim.par_values = sc.mergedicts(self.par_values, pars)
        if hasattr(self, 'imported'): # The same import, with what it did not translate
            sim.imported = self.imported
        return sim

    def calibrate(self, pars=None, data=None, weights=None, likelihood=None, gof=None, method=None, n_runs=None, rand_seed=None, verbose=None, **kwargs):
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
            scan (int): for the SciPy methods, NUTS, and particle MCMC, the number of points spread over the bounds to try first (default: 10 per parameter); the search starts from each of the best `restarts=3` of them that are apart (for SciPy), each NUTS chain from one of the best that are apart, and particle MCMC from the best
            n_runs (int): runs per evaluation, for a stochastic sim (default 4 for a stochastic compartmental sim, and 1 for agents and for a deterministic one); every point is scored by the mean over the same seeds (common random numbers), except under Optuna, where each point takes fresh seeds so that its search averages over chance; `fit.mismatch` is the score at the best fit over the first seeds
            rand_seed (int): the random seed for the fitting method (default: the sim's, so a fit gives the same answer each time)
            verbose (bool): whether to print progress
            kwargs (dict): passed to the fitting method, e.g. `trials=200` for Optuna, `maxiter=500` for SciPy, `draws=500`, `warmup=500`, and `chains=4` for NUTS (the defaults; 1,000 draws and 1,000 warm-up draws for particle MCMC, whose draws are more correlated), `particles=` for particle MCMC (default: chosen by each chain in its warm-up; see `ss.fit.pmmh()`), or `parallel=False` to run its chains one after another; for IF2, `particles=1000`, `iterations=50` (passes of the filter), `cooling=0.5` (the share of the first size of the random walk's steps that is left after 50 passes, as pomp's `cooling.fraction.50`), `sd=0.05` (that first size, on the scale searched on: the logit of each value's place between its bounds, of its log for wide bounds), and `chains=4` (each from its own start: the best of the initial values and a spread of points over the bounds, by the filter)

        Returns:
            An `ss.fit.Fit`, with the calibrated sim as `fit.sim` and the best-fit values as `fit.pars`

        Examples:
            ```python
            beta = ss.par(0.1, bounds=[0.01, 0.5])
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(10), init_prev=0.01)
            sim = ss.Sim(sir, data='cases.csv', start='2020-01-01', stop='2020-06-01')
            fit = sim.calibrate() # A best fit
            post = sim.calibrate(method='nuts') # A posterior, given a likelihood for each data series
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
        if not hasattr(self, 'inputs'):
            errormsg = 'sim.check_dt() is only available for sims with declared states and transitions'
            raise ValueError(errormsg)
        coarse = self if self.results_ready else self.new(verbose=0).run()
        fine = self.new(dt=coarse.t.dt/factor, verbose=0)
        for val in fine.pars.values(): # A module with its own timestep takes a smaller one too
            for mod in sc.tolist(val):
                if isinstance(mod, ss.Module) and mod.t.dt is not None:
                    mod.t.dt = mod.t.dt/factor
        fine.run()
        rows = []
        for mod in coarse.declared:
            n = mod.t.npts - 1 # The module's own timeline, which may differ from the sim's
            for key,res in mod.results.items():
                if not isinstance(res, ss.Result) or key.startswith('cum_') or np.ndim(res.values) != 1:
                    continue
                other = fine.results[mod.name][key].values
                if res.kind == 'flow': # A flow: total it over each of the larger steps (a flow at a time point is over the step that ends there)
                    a, b = res.values[1:n+1], other[1:n*factor+1].reshape(n, factor).sum(axis=1)
                else: # A stock: at each of the larger steps' time points
                    b = other[::factor]
                    a, b = res.values[:len(b)], b[:len(res.values)]
                scale = np.abs(a).max()
                change = np.abs(b - a).max()/scale if scale else 0.0
                rows.append(dict(module=mod.name, result=key, peak=a.max(), peak_fine=b.max(), change=change))
        df = sc.dataframe(rows).sort_values('change', ascending=False).reset_index(drop=True)
        if output:
            return df
        dt, unit = coarse.t.dt.to(coarse.unit_base).value, coarse.unit_name
        print(f'Change in each result when the timestep goes from {dt:n} to {dt/factor:n} {unit}s (largest difference, as a share of the largest value):')
        for row in df.itertuples():
            print(f'  {row.module}.{row.result:<20s} {row.change:6.1%}   (peak {row.peak:n} → {row.peak_fine:n})')
        return

    def what_modifies(self, target, output=False):
        """
        Print everything that modifies a quantity: each `ss.modify()` and `ss.change()`, and the efficacy of each product

        Args:
            target (par/str): the quantity, as for `ss.modify()`: e.g. `sir.pars.beta`, `sir.rel_sus`, or `'sir.pars.beta'`
            output (bool): return the text instead of printing it
        """
        if not self.initialized:
            self.init()
        mod, kind, name = ss.connectors.parse_target(target)
        modname = mod if isinstance(mod, str) else mod.name
        if modname in self.all_networks(): # The contacts of a network
            kind = 'par'
        else:
            graph = self.get_declared(modname).graph
            if kind is None:
                kind = 'transition' if name in graph.transitions else 'par'
            if kind == 'transition':
                kind = graph.modifier_kind(name)
        path = f'{modname}.pars.{name}' if kind == 'par' else f'{modname}.{name or kind}'
        mods = self.modifiers.get((modname, kind, name), [])
        lines = [path, '  modifiers' if mods else '  modifiers: none']
        lines += [f'    {mod.describe()}' for mod in mods]
        if path in self.clipped:
            lines.append('  clipped: the modifiers made it negative during the run, so it was set to 0 where it was')
        text = '\n'.join(lines)
        if output:
            return text
        print(text)
        return

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
        return bool(self.pars.demographics) or any(self.flows_of(mod) for mod in self.declared)

    @staticmethod
    def flows_of(mod):
        """ The transitions of a declared disease that are births, arrivals, or deaths of everyone, so are the population's """
        placed = lambda tr: getattr(tr, 'per', None) is not None and ss.conditions.modname(tr.per.module) not in [None, mod.name] # Another module's births (ss.birth(per=)), which only say where its children start
        return [tr for tr in mod.graph.transitions.values() if (tr.inflow and tr.people and not placed(tr)) or getattr(tr, 'every_state', False)]

    def _init_flows(self):
        """
        Births, deaths, and migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...), ss.death(...)])`): see `ss.demographics.Flows`

        People are born and die once, so these become the transitions of one disease, the first declared (or, for births into a
        state that is given, the one it is a state of), whose people are everyone's: as agents, the diseases share the people, and
        in a compartmental run with several declared diseases, their states are tracked jointly (see `ss.solvers.init_crossings()`). For the
        same reason, births or deaths of everyone declared in more than one disease are an error.
        """
        for flows in [mod for mod in self.pars.demographics.values() if isinstance(mod, ss.demographics.Flows)]:
            for tr in flows.transitions:
                named = tr.dst[0] if tr.inflow and tr.state_given else None # A state of one of the diseases
                holder = next((mod for mod in self.declared if named is None or any(st is named for st in mod.graph.states.values())), self.declared[0])
                tr = sc.dcp(tr, memo={} if named is None else {id(named):named}) # The sim's own, so the module is unchanged if it is built again
                holder.graph.add_flow(tr)
        holders = [mod.name for mod in self.declared if self.flows_of(mod)]
        if len(holders) > 1:
            raise ss.errors.DuplicateDeclaration(f'births or deaths of everyone are declared in {sc.strjoin(holders)}; people are born and die once, whatever their diseases, so this would count them twice.\n'
                                          f'     Declare them once, for the population: ss.Sim(..., demographics=[ss.birth(rate=...), ss.death(rate=...)]).')
        return

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
        elif strat is None or strat.guessed.get('split') == ss.named.world.guess:
            return ss.named.world.ages, ss.named.world.counts
        return None

    def get_declared(self, name):
        """ A module with declared states and transitions, by name """
        mods = {mod.name:mod for mod in self.declared}
        if name not in mods:
            errormsg = f'No module named "{name}" with declared transitions was found; these are {sc.strjoin(mods.keys())}'
            raise KeyError(errormsg)
        return mods[name]

    def _init_named_people(self):
        """
        A named population (`ss.people('Kenya')`): looked up for the year the sim starts, at the sim's number of agents, and the population size taken from it

        With no people given, a country's contact matrix (`ss.contacts('Kenya')`) brings that country's people, for their ages and sexes
        (but not their number, since the contacts are only how people mix).
        """
        pars = self.pars
        named = getattr(pars.people, 'named', None)
        self.given_pyramid = None # People given to a compartmental solver: the ages their age groups are split by
        if isinstance(pars.people, ss.People) and named is None and self.declared and pars.solver not in ['abm', 'auto']: # Their ages and number, not their agents
            dist = sc.dcp(pars.people.age.default)
            dist.init(force=True)
            edges = ss.named.world.ages
            counts = np.histogram(np.asarray(dist.rvs(100_000)), bins=np.append(edges, np.inf))[0]
            self.given_pyramid = (edges, counts)
            if 'total_pop' not in self.given:
                pars.total_pop = pars.people.n_agents*pars.pop_scale
                self.pop_from = 'people'
        self.named_people = None
        self.named_unsized = False # A named population with no number of agents of its own does not make the model agent-based
        self.pop_from = None # Where the population size came from, if it was not given: named people, or the populations of places
        self.people_from = None # For people that come with a contact matrix, the matrix
        if named is None and pars.people is None and self.declared:
            places = {net.ref['place']:name for name,net in pars.networks.items() if getattr(net, 'ref', None)} # Each country's contact matrices
            if len(places) == 1:
                place, net = list(places.items())[0]
                named = sc.objdict(place=place, year=None, n_agents=None, revision=None)
                self.people_from = net
                ss.note(f'the ages of the people are those of {place}, since the contacts "{net}" are; give people= for others', sim=self)
        if named is None:
            return
        year = named.year
        if year is None and pars.start is not None and not (sc.isnumber(pars.start) and pars.start < 1800): # The year the sim starts, if it has dates
            try:
                year = ss.date(pars.start).year
            except Exception:
                year = None
        n_agents = sc.ifelse(named.n_agents, int(pars.n_agents))
        ppl = ss.people(named.place, year=year, n_agents=n_agents, revision=named.revision)
        ppl.named.update(year=named.year, n_agents=named.n_agents) # As given, so the sim can be rebuilt
        pars.people = ppl
        self.named_people = ppl.named
        self.named_unsized = named.n_agents is None
        if 'total_pop' not in self.given and self.people_from is None:
            pars.total_pop = ppl.named.total
            pars.pop_scale = pars.total_pop/pars.n_agents
            self.pop_from = 'people'
        return

    def _init_unit(self):
        """
        Work out the sim's unit, which is the unit of the model's bare numbers: from unit=, then the typed
        values in the model, then calendar dates (days), then a typed dt, start, or stop; else time is in timesteps

        The unit belongs to the model, so how the run is written (its step, or a stop given as weeks) does
        not change what the bare numbers mean. A year-like number for start or stop is a calendar year.
        """
        pars = self.pars
        time = ss.time
        unit = source = None
        self.units_disagree = None # The units of the typed values, if they differ and nothing else gave the unit
        def base(obj):
            return type(obj).base.removesuffix('s')
        def is_dur(obj):
            return isinstance(obj, ss.dur) and not isinstance(obj, ss.datedur)
        dates = [v for v in [pars.start, pars.stop] if isinstance(v, (str, ss.date))]
        unit_given = pars.unit is not None and pars.unit.removesuffix('s') != 'year' # A bare number is then in that unit, however large (stop=3650 days), not a calendar year
        years = [v for v in [pars.start, pars.stop] if sc.isnumber(v) and time.assume_cal_year(v)] if not unit_given else []
        if unit_given and sc.isnumber(pars.start) and time.assume_cal_year(pars.start): # Read in the unit given, as it says, but it looks like a year
            ss.note(f"start={pars.start} is {pars.start} {pars.unit.removesuffix('s')}s, since unit='{pars.unit}'; for a calendar year, give a date, e.g. start='{int(pars.start)}-01-01'", sim=self)
        durs = [v for v in [pars.start, pars.stop, pars.dur] if is_dur(v)]
        setters = [mod for mod in self.declared if getattr(mod, 'sets_unit', True)] # A module with typed defaults of its own (ss.Pregnancy's weeks and years) says nothing of the user's bare numbers
        units = [u.removesuffix('s') for mod in setters for u in mod.graph.unit_values()]
        units += [base(mod.t.dt) for mod in setters if is_dur(mod.t.dt)]
        if pars.unit is not None:
            unit = pars.unit.removesuffix('s')
        elif units:
            order = ['day', 'week', 'month', 'year']
            unit = sorted(set(units), key=order.index)[0]
            source = 'the typed values in the model'
            if len(set(units)) > 1:
                self.units_disagree = sorted(set(units), key=order.index)
                bare = [f'{mod.name}.{p.name}' if not p.literal else f'{p.raw} in {mod.name}' for mod in setters for p in mod.graph.all_pars() if p.slot in ['rate', 'dur'] and sc.isnumber(p.raw) and p.raw != 0] # 0 is 0 in any unit
                if bare: # Which unit they are in cannot be guessed
                    raise ss.errors.IncompatibleTypes(f'the typed values in the model have different units ({sc.strjoin(self.units_disagree)}), so the unit of the bare numbers {sc.strjoin(bare)} is not known. Set it with unit=, e.g. ss.Sim(..., unit=\'{unit}\').')
        elif dates and self.declared: # A model written as code keeps its step in years, as in Starsim 3
            unit, source = 'day', 'start and stop dates'
        elif is_dur(pars.dt) or isinstance(pars.dt, str):
            unit, source = (base(pars.dt) if is_dur(pars.dt) else pars.dt.removesuffix('s')), 'dt'
        elif durs:
            unit, source = base(durs[0]), 'start and stop'
        elif years:
            unit, source = 'year', 'start and stop years'
        elif not self.declared: # A model written as code, with nothing to say otherwise: years, as in Starsim 3
            unit, source = 'year', 'the default'
        if years and pars.unit is None and unit not in [None, 'year'] and sc.isnumber(pars.dt) and not isinstance(pars.dt, bool): # A bare step beside calendar years is in years, as start and stop are
            ss.warn(f'dt={pars.dt:n} is read in years, since start and stop are years, though the model\'s unit is {unit}s (from {source}); give dt=ss.{unit}s(...) or unit= to say otherwise', ss.errors.UnitInferred)
            pars.dt = ss.years(pars.dt)
        self.unit = unit
        self.unit_source = source
        self.unit_base = unit or 'day' # With no unit, the arithmetic is done in days, but the times have no dates (see ss.Timeline.datevec) and are said to be steps
        self.unitless = unit is None
        if self.unitless: # A yearly cycle has no meaning when time is counted in steps
            mods = list(pars.connectors.values()) + list(pars.interventions.values())
            yearly = [f'{mod.name}.{p.name}' for mod in self.declared for p in mod.graph.all_pars() if isinstance(p.raw, ss.seasonal)]
            yearly += [mod.name for mod in mods if isinstance(getattr(mod, 'amount', None), ss.seasonal)]
            if yearly:
                errormsg = f'{sc.strjoin(yearly)} is an ss.seasonal(), a yearly cycle, but nothing in the model has a unit of time, so its times are steps; give the sim a unit (e.g. unit="day") or dates'
                raise ValueError(errormsg)

        # Give the timeline the unit
        dur_class = time.get_dur_class(self.unit_base)
        if pars.dt is None or sc.isnumber(pars.dt):
            pars.dt = dur_class(sc.ifelse(pars.dt, 1.0))
        coded_years = not self.declared and unit == 'year' and pars.start is None # A model written as code in years starts in the year 2000 by default, as in Starsim 3 (see ss.Timeline)
        if not dates and not years and not coded_years:
            if pars.start is None or sc.isnumber(pars.start):
                pars.start = dur_class(sc.ifelse(pars.start, 0))
            if sc.isnumber(pars.stop):
                pars.stop = dur_class(pars.stop)
        return

    def describe(self, output=False, verbose=False):
        """
        Print the resolved model: every state, transition, and default, with each guess marked

        Args:
            output (bool): if True, return the text rather than printing it
            verbose (bool): also show what each rate and dwell time is per step: the probability of leaving in a step, as the solvers that take whole steps use it, and the stages of a dwell time
        """
        if not self.initialized:
            self.init()
        pars = self.pars
        unit = self.unit_name
        mods = list(self.modules)
        lines = []
        def add(key, text, guess=None):
            line = f'{key:12s} {text}'
            if guess:
                line = f'{line:60s} [{guess}]'
            lines.append(line.rstrip())
            return
        named = getattr(self, 'named_people', None)
        popguess = '' if ('total_pop' in self.given or 'n_agents' in self.given or getattr(self, 'pop_from', None)) else ' [guessed]'
        stopguess = '' if ('stop' in self.given or 'dur' in self.given) else ' [guessed]' # The default length of a run
        t = self.t
        lines.append(f'Sim: {len(mods)} module{"s" if len(mods) != 1 else ""}, {pars.total_pop:,.0f} people{popguess}, {unit}s {t.start} to {t.stop}{stopguess}' if self.unit is None or not t.is_absolute else
                     f'Sim: {len(mods)} module{"s" if len(mods) != 1 else ""}, {pars.total_pop:,.0f} people{popguess}, {t.start} to {t.stop}{stopguess}')
        if self.declared:
            guessed = pars.solver == 'auto'
            add('solver', self.solver, f'guessed: {self.solver_reason}' if guessed else 'requested')
            for cap in self.capabilities: # What matters to which solvers can run it: see ss.solvers.init_capabilities()
                add('', f'{cap.ingredient}: {cap.groups} as compartments ({cap.how})')
            if not guessed: # What auto would have done
                reasons = ss.solvers.auto_reasons(self)
                add('', f'auto would choose: ' + (f'abm ({reasons[0]})' if reasons else 'ode (every part of the model has a compartmental form)'))
            if self.solver == 'abm':
                scale = f', each standing for {pars.pop_scale:n} people' if pars.pop_scale != 1 else ''
                add('', f'{pars.n_agents:,} agents{scale}', None if 'n_agents' in self.given else 'guessed')
            if named is not None: # Where a named population came from
                ref = named.ref
                guess = None if named.year is not None else 'guessed: the year the sim starts' if self.t.is_absolute else 'guessed: the year of the revision'
                if self.people_from is not None:
                    guess = f'guessed: the ages of the country of "{self.people_from}"'
                add('people', f"{named.place}, {ref['year']}, from starsim_data {ref['version']} ({ref.get('study', '?')} {ref.get('revision', '')})".rstrip(), guess)
            if self.unit is None:
                text = 'none: time is in timesteps' if unit == 'step' else f'none: time is in units with no name, and a step is {pars.dt.value:n} of them'
                add('unit', text, 'nothing in the model has a unit')
            else:
                add('unit', self.unit, f'from {self.unit_source}' if self.unit_source else None)
            for strat in self.groups.strats: # Each dimension: its levels, the states it applies to, and what it generates
                text = f'{strat.name} ({strat.kind}): {sc.strjoin(strat.labels)}'
                text += f'; where {", ".join(f"{k}={sc.strjoin(sc.tolist(v))}" for k,v in strat.where.items())}' if getattr(strat, 'where', None) else ''
                text += f'; applies to {sc.strjoin(strat.states)}' if strat.states is not None else '; applies to every state' if strat.kind == 'space' else f'; applies to the people of {sc.strjoin(strat.holders)}' if strat.kind == 'disease' else ''
                if strat.kind == 'space':
                    text += f'; mobility: {strat.mobility if strat.mobility is not None else "none"}'
                    if strat.geo is not None and strat.geo.ref is not None: # Where a named geography came from
                        ref = strat.geo.ref
                        text += f"; places from starsim_data {ref['version']} ({ref.get('boundaries', '?')} boundaries; population: {ref.get('population', '?')})"
                elif strat.kind == 'age':
                    text += f'; split as the population of {named.place}' if getattr(strat, 'split_from', None) == 'people' else ''
                    text += '; people do not age (use_aging=False)' if not pars.use_aging else '; nobody ages, since time is in timesteps' if self.unit is None else '; people age between the groups'
                text += self.groups.moved_by(strat)
                if strat.name in getattr(self.groups, 'kept', set()):
                    text += f'; people.{strat.attr}, from the people given'
                guesses = list(strat.guessed.values())
                if strat.kind == 'age' and 'use_aging' not in self.given:
                    guesses.append('aging, since people are born or die' if pars.use_aging else 'no aging, since nobody is born or dies')
                add('dimension', text, f'guessed: {"; ".join(guesses)}' if guesses else None)
            for crossing in self.crossings: # Diseases tracked jointly: why, how big, and what is assumed (see ss.solvers.Crossing)
                sizes = [self.groups.strats[self.groups.dim_index(name)].n for name in crossing.names]
                add('jointly', f'{" × ".join(crossing.names)}: {" × ".join(str(n) for n in sizes)} = {int(np.prod(sizes))} joint states of people', '; '.join(crossing.reasons))
                for text in crossing.describe():
                    add('', text)
        for mod in mods:
            if hasattr(mod, 'describe') and (getattr(mod, 'graph', None) is not None or isinstance(mod, ss.Strains)): # A family after its members
                lines += ['', mod.describe(output=True)]
        if self.declared:
            lines.append('')
            place = f' within each {self.groups.strats[self.mixing.space].name}' if self.mixing.space is not None else ''
            nets = list(self.networks.values()) + getattr(self, 'mixed', [])
            if nets:
                for net in nets:
                    term = [term for term in self.mixing.terms if term.name == net.name] if self.solver != 'abm' else []
                    how = '' if not term else f': {term[0].contacts:n} contacts per {unit}' if term[0].kind == 'homogeneous' else f': the partnership approximation (approx), partnerships formed at {term[0].partnership.formed:.2g} a {unit}' if 'partnership' in term[0] else f': contact matrix by {self.groups.strats[term[0].dim].name}'
                    add('mixing', f'{net.name} ({type(net).__name__}){how}')
            elif any(mod.graph.transmissions for mod in self.declared): # How a state in a rate is read is shown with each rate
                add('mixing', f'homogeneous{place}, 1 contact per {unit}', 'guessed: no network given')
            for mod in list(self.connectors()) + list(self.interventions()):
                if isinstance(mod, ss.modify):
                    add('modifier', f'{mod.target}: {" ".join(mod.describe().split())}')
                elif isinstance(mod, ss.Delivery):
                    add('delivery', mod.describe())
            for mod in self.declared:
                names = [f'n_{name}' for name in mod.graph.states]
                extra = [key for key,res in mod.results.items() if isinstance(res, ss.Result) and key not in names and key not in mod.graph.transitions and not key.startswith('cum_')]
                text = f'{", ".join(names)}, and one flow per transition' + (f'; also {", ".join(extra)}' if extra else '')
                ti = getattr(mod.engine, 'group_ti', None)
                if ti is not None: # Kept by group every few steps
                    text += f'; by group every {ti[1]} {unit}s'
                add('results', text, 'guessed: to keep them by group within 100 MB; results=[...] keeps fewer, more often' if ti is not None else None)
            # What runs where: the declared modules in the kernels or as agents, and the rest in Python (see sim.kernel_reason())
            stepped = {func['module'] for func in self.loop.funcs} - {'sim', 'people'}
            declared = [mod.name for mod in self.declared]
            if self.solver == 'abm':
                add('plan', f'{sc.strjoin(declared)}: as agents, each step in Python')
            else:
                whole = self.kernel_reason()
                joint = ''.join(f'; {" × ".join(c.names)} as one program' for c in self.crossings)
                add('plan', f'{sc.strjoin(declared)}: in the kernels, ' + ('the whole run in one call' if whole is None else f'one step at a time, since {whole}') + joint)
            tabulated = [mod.name for mod in self.modules if mod.name in stepped and getattr(mod, 'tabulated', False) and self.solver != 'abm']
            python = [mod.name for mod in self.modules if mod.name in stepped and mod.name not in declared + tabulated]
            if tabulated:
                add('', f'{sc.strjoin(tabulated)}: in the kernels\' tables, as {"it depends" if len(tabulated) == 1 else "they depend"} only on time and states')
            if python:
                add('', f'{sc.strjoin(python)}: in Python, at {"its" if len(python) == 1 else "their"} place in each step')
            for note in self.notes:
                add('note', note)

        # The schedule: one line per timestep in use
        groups = sc.ddict(list)
        for mod in mods:
            if isinstance(mod, ss.Product) and mod.name not in self.results: # Given by a delivery, so not run on its own
                continue
            groups[mod.t.dt.to(self.unit_base).value if self.declared else mod.t.dt.years].append(mod.name)
        for i,(dt,names) in enumerate(sorted(groups.items())):
            every = f'every {unit}' if dt == 1 else f'every {dt:n} {unit}s'
            add('schedule' if i == 0 else '', f'{every}: {", ".join(names)}')
        if verbose and self.declared:
            lines += ['', *self.describe_steps()]
        text = '\n'.join(lines)
        return text if output else print(text)

    def describe_steps(self):
        """ The lines of `describe(verbose=True)`: each rate and dwell time per step of its module """
        unit = self.unit_name
        def plain(raw): # A single value in the sim's unit, or None if it varies (by group, by person, or in time)
            if sc.isnumber(raw) or (isinstance(raw, ss.TimePar) and np.ndim(raw.value) == 0):
                return ss.solvers.engine.convert(raw, self.unit_base)
            return None
        lines = []
        for mod in self.declared:
            dt = mod.t.dt.to(self.unit_base).value
            lines.append(f"Per step of {mod.name} ({dt:n} {unit}{'s' if dt != 1 else ''}):")
            for name,tr in mod.graph.transitions.items():
                if tr.contact: # The rate of each person depends on the others, so only its parameters can be converted
                    for leaf in tr.rate.leaves:
                        if isinstance(leaf, ss.transitions.par) and not leaf.literal and (val := plain(leaf.raw)) is not None:
                            lines.append(f'  {name:12s} {leaf.name} = {val:n} per {unit} → {val*dt:n} per step')
                    continue
                dw = mod.engine.dwells.get(name)
                if dw is None:
                    continue
                val = dw.mean if dw.dist is not None else plain(dw.par.raw)
                if val is None or not np.isfinite(val):
                    lines.append(f'  {name:12s} varies (by group, by person, or in time)')
                    continue
                mean = 1/val if dw.is_rate else val
                if dw.fixed:
                    lines.append(f'  {name:12s} a fixed delay of {mean:n} {unit}s = {mean/dt:n} steps')
                else:
                    rate = dw.k/mean # The rate of leaving each stage
                    stages = f' in each of {dw.k} stages' if dw.k > 1 else ''
                    lines.append(f'  {name:12s} rate {rate:.4g} per {unit}{stages} → p = 1 - exp(-{rate:.4g} × {dt:n}) = {1 - np.exp(-rate*dt):.4f} per step')
        return lines

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
        if not self.initialized: # A copy, so that this sim is not changed
            return sc.dcp(self).init().hash()
        spec = self.to_spec()
        digest = ss.utils.short_hash
        model = dict(modules=spec['modules'], networks=spec['networks'], stratify=spec['stratify'], modifiers=spec['modifiers'])
        for mod in model['modules']: # Where a value came from is not part of the model
            if isinstance(mod.get('pars'), list): # A declared module's parameters
                for p in mod['pars']:
                    p.pop('source', None)
            if isinstance(mod.get('transitions'), list): # The order of the transitions is not part of the model
                mod['transitions'] = sorted(mod['transitions'], key=lambda t: t['name'])
            dis = self.diseases.get(mod.get('name')) if mod.get('init_prev') else None
            if isinstance(dis, ss.Disease) and len(dis.graph.transmissions[0].dst) == 1: # init_prev=0.01 is init=dict(I=0.01), the same model
                state = next(s for s in mod['states'] if s['name'] == dis.graph.transmissions[0].dst[0].name)
                if state.get('init') is None:
                    state['init'], mod['init_prev'] = mod['init_prev'], None
        unknown = ss.presentation.unknown_funcs(dict(model, interventions=spec['interventions']))
        if unknown:
            errormsg = f'sim.hash() cannot tell these functions apart from others, since their source is not available and they have no name to import them by: {sc.strjoin(unknown)}. Define each with def, or give it a version (e.g. fn.version = "1")'
            raise ValueError(errormsg)
        structure, values = ss.utils.split_numbers(model, unit=self.unit, skip=['guessed', 'label', 'doc'])
        config = {k:v for k,v in spec['config'].items() if k != 'rand_seed'}
        data = None if self.data is None else digest(self.data.to_csv())
        return sc.objdict(model=digest(structure), pars=digest(values), config=digest(config), scenario=digest(spec['interventions']), data=data, seed=self.pars.rand_seed)

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
        if not self.initialized: # A copy, so that this sim is not changed
            return sc.dcp(self).init().to_spec(filename=filename, **kwargs)
        pars = self.pars
        def tstr(val):
            if isinstance(val, ss.dur) and type(val).base != self.unit_base + 's': # A duration in another unit than the sim's (e.g. start=2000 in years, with rates per day) is written with its unit
                return ss.transitions.quantity_json(val, 'dur', None)
            elif isinstance(val, ss.dur):
                return float(val.value)
            return val.strftime('%Y-%m-%d') if isinstance(val, ss.date) and val == val.normalize() else str(val) # Dates as ISO 8601
        out = sc.objdict()
        out.schema = 'lingua-franca/1.0'
        out.modules = [mod.to_spec() for key in ['custom', 'demographics', 'diseases'] for mod in self[key]() if not isinstance(mod, ss.demographics.Flows)] # Births and deaths declared for the population are, by now, transitions of a disease
        out.networks = [mod.to_spec() for mod in list(self.networks()) + getattr(self, 'mixed', [])]
        out.stratify = [sc.dcp(strat.spec) for strat in self.groups.sim_strats] # Those of a module are in its own spec, and those of an intervention are part of the scenario
        out.modifiers = [mod.to_spec() for mod in self.connectors()]
        out.interventions = [mod.to_spec() for mod in self.interventions()]
        out.config = sc.objdict(solver=pars.solver, integrator=pars.integrator, approx=pars.approx, unit=self.unit or pars.unit, # A sim of no declared modules has only the unit it was given
                                start=tstr(self.t.start), stop=tstr(self.t.stop), dt=float(self.t.dt.to(self.unit_base).value),
                                total_pop=float(pars.total_pop), n_agents=int(pars.n_agents) if self.solver == 'abm' else None, rand_seed=pars.rand_seed)
        out.config.crn = ss.options.crn # Numerical options that change the results
        out.config.precision = ss.options.precision
        out.config.substeps = pars.substeps
        out.config.use_aging = bool(pars.use_aging) if 'use_aging' in self.given else None # Only when given, e.g. an age stratification without aging
        if getattr(self, 'named_people', None) is not None and self.people_from is None: # Where a named population came from (people that come with a contact matrix come with it again)
            out.config.people = dict(self.named_people.ref, total=self.named_people.total)
        out.meta = sc.objdict(library_version=ss.__version__, solver_chosen=self.solver, solver_reason=self.solver_reason, notes=list(self.notes))
        out = sc.jsonify(out)
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

    def to_code(self, filename=None):
        """
        The sim as a Python file that builds it again, from its model spec (see `ss.to_code()`)

        Only the declared parts are written; what is not (e.g. a module written as code)
        is listed in the file and in an `ss.errors.LossyExport` warning. A sim imported from another framework (`ss.load(..., format=...)`) also says
        where it came from, how each rate was read, and what was not translated.

        Args:
            filename (str): if provided, write the code to this file
        """
        header, footer, code = None, None, None
        missing = [] if self.data is None else [f'the data ({len(self.data.columns)} series: {sc.strjoin(self.data.columns)}): pass it with ss.Sim(..., data=...)']
        imp = getattr(self, 'imported', None)
        if imp is not None:
            code = imp.get('code')
            source = sc.path(imp.file).name if imp.file else 'an object'
            header = f'Imported from {imp.format} ({source}) by ss.load()\n\nHow each rate was read:\n' + '\n'.join(imp.choices)
            if imp.incomplete:
                footer = ['Not translated (ImportIncomplete):'] + [f'  {name}: {why}' for name,why in imp.incomplete]
        return ss.to_code(self.to_spec(), filename=filename, header=header, footer=footer, code=code, missing=missing)

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
            raise AlreadyRunError(errormsg)

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
            mod.t.ti = np.searchsorted(mod.t.yearvec, self.t.now('year')) # Start the module at the current time

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
        self.results.timevec = self.timevec # Store the timevec in the results for plotting
        return

    def init_people(self, verbose=None, **kwargs):
        """
        Initialize people within the sim
        Sometimes the people are provided, in which case this just adds a few sim properties to them.
        Other time people are not provided and this method makes them.

        Args:
            verbose (int):  detail to print
            kwargs  (dict): passed to ss.People()
        """
        # Handle inputs
        people = self.pars.pop('people')
        if (self.named_people is not None or self.given_pyramid is not None) and self.solver != 'abm': # A compartmental run takes the size and age split of a population, not its agents
            people = None
        n_agents = self.pars.n_agents
        verbose = sc.ifelse(verbose, self.pars.verbose)
        if verbose > 0 and self.solver == 'abm':
            labelstr = f' "{self.label}"' if self.label else ''
            print(f'Initializing sim{labelstr} with {n_agents:0n} agents')

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
            if self.declared and 'age_data' not in kwargs: # Otherwise the ages of a declared model are the world's, as a guess (as is the split of age groups that are not given one)
                kwargs['age_data'] = ss.named.age_data(*self.pyramid())
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

    def reset_time_pars(self, force=True):
        """ Reset the time parameters in the modules; used for imposing the sim's timestep on the modules """
        for mod in self.modules:
            mod.init_time_pars(force=force)
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

        # The results named to be kept by group must exist
        if self.pars.results is not None:
            known = [f'{mod.name}.{key}' for mod in self.modules for key,res in mod.results.items() if isinstance(res, ss.Result)]
            for name in sc.tolist(self.pars.results):
                if name not in known and not any(key.split('.', 1)[1] == name for key in known):
                    errormsg = f'results= names "{name}", which is not a result of this sim; the closest are {ss.transitions.closest(name, known)}'
                    raise sc.KeyNotFoundError(errormsg)
        return

    def kept(self, mod, key):
        """ Whether a result of a module is kept by group: every one, unless the sim parameter `results` names the ones to keep (by name, or as "module.name") """
        names = self.pars.results
        return names is None or key in sc.tolist(names) or f'{mod.name}.{key}' in sc.tolist(names)

    def group_rows(self, npts, row_bytes, budget, name=None):
        """
        The time points at which results are kept by group: each one, or the end of every few steps if each one would take more than `budget` bytes

        The number of steps between them is the shortest of a week, two weeks, four weeks, a quarter, half a year, or
        a year (or of 2, 5, 10, 20, ... steps for a sim with no unit, or one longer than a day) that keeps them within
        the budget. A row of values by group then holds a flow added up over its steps, and anything else as it is
        at the end of them (so a ratio of states is the ratio then). See `ss.Result.select()`.

        Args:
            npts (int): the number of time points
            row_bytes (float): the memory the results take at each time point
            budget (float): the most memory they may take
            name (str): the module whose results they are, to say so in a note

        Returns the row of each time point and the time point each row ends at, or (None, None) for every time point.
        """
        columns = getattr(self.data, 'columns', self.data.keys() if isinstance(self.data, dict) else [])
        if npts*row_bytes <= budget or npts <= 2 or self.pars.rescale: # With rescaling, the scale differs at each step, and is applied to each one
            return None, None
        elif any('[' in str(col) for col in columns): # Data by group, which are compared with the results at each step
            return None, None
        most = max(int(budget//row_bytes), 2) # The most rows that fit
        need = int(np.ceil((npts - 1)/(most - 1))) # The fewest steps per row
        days = float(self.t.dt.days) if self.unit is not None else None
        if days is not None and days <= 1:
            ladder = [max(int(round(d/days)), 1) for d in [7, 14, 28, 91, 182, 364]]
        else:
            ladder = [m*10**e for e in range(8) for m in [1, 2, 5]]
        every = next((k for k in ladder if k >= need), need)
        rows = (np.arange(npts) + every - 1)//every # The start is a row of its own, then each `every` steps
        ends = np.minimum(np.arange(rows[-1] + 1)*every, npts - 1)
        unit = self.unit or 'step'
        span = f'{every} {unit}s' if days is None else f'{every*days:g} days'
        ss.note(f'the results of "{name}" are kept by group every {span} (a flow added up over them), since every {unit} would take {npts*row_bytes/1e6:,.0f} MB; '
                f'name the results to keep by group with results=[...] to keep them more often', sim=self)
        return rows, ends

    @property
    def _current_scale(self):
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
            if isinstance(disease, ss.Infection) and not hasattr(disease, 'ti_infected'):
                errormsg = f'Dynamic rescaling (rescale=True) requires each disease written as code to have a ti_infected state, but "{disease.name}" does not'
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
        self.rng_rescale.jump_dt(ti=ti+1) # Like other distributions, advance the random numbers for this timestep
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

    def init_data(self, data=None):
        """ Initialize or add data to the sim """
        data = data if data is not None else self.data
        if isinstance(data, ss.data):
            data = data.load(self)
        elif isinstance(data, (str, os.PathLike)):
            data = ss.data(data).load(self)
        self.data = ss.results.validate_sim_data(data)
        if self.data is not None and self.groups.strats:
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
            raise AlreadyRunError(errormsg)

        # Print out progress if needed
        if self.verbose: # Print progress
            self.elapsed = self.timer.toc(output=True)
            t = self.t
            simlabel = f'"{self.label}": ' if self.label else ''
            string = f'  Running {simlabel}{t.now("str")} ({t.ti:2.0f}/{t.npts}) ({self.elapsed:0.2f} s) '
            if self.verbose >= 2:
                sc.heading(string)
            elif not (t.ti % max(1, t.npts//10)): # About ten lines in all
                sc.progressbar(t.ti + 1, t.npts, label=string, length=20, newline=True)
        return

    def advance(self):
        """ Move the time to the end of the step, where its results are recorded: for the sim, and the modules on its timeline (see `ss.Loop`) """
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
        self.loop.run(t.tvec[min(t.ti + 1, t.npts - 1)], verbose) # Up to the end of the next step
        return self

    def run(self, until=None, verbose=None, shrink=False, check_method_calls=True, profile=False, solver=None, n_runs=None):
        """
        Run the model -- the main method for running a simulation.

        Args:
            until (date/str/float): the date to run the sim until
            verbose (int): how much to print: 0 nothing, 1 the solver, notes, and progress (default), 2 everything (e.g. each step)
            shrink (bool): whether to explicitly shrink the sim after running
            check_method_calls (bool): whether to check that all required methods were called
            profile (bool): if True, record per-entry CPU timing in the loop (see `sim.loop.to_df()`/`plot_cpu()`)
            solver (str): the solver to use, if different from the sim's (only before the sim is initialized)
            n_runs (int): if provided, run the sim this many times with different seeds, and return an `ss.MultiSim`; a shortcut for `ss.parallel(sim, n_runs=n_runs)`
        """
        imp = getattr(self, 'imported', None)
        unwarned = imp is not None and imp.get('incomplete') and not imp.get('warned') # Once per import: the runs and copies share it
        if unwarned and not self.initialized:
            imp.warned = True
            ss.warn(f'running an import with {len(imp.incomplete)} part(s) of the model not translated (sim.imported.incomplete); '
                    f'its results leave them out. {imp.prompt}', ss.errors.ImportIncomplete)
        if n_runs is not None:
            return ss.parallel(self, n_runs=n_runs)
        if solver is not None:
            if self.initialized:
                errormsg = 'The solver can only be chosen before the sim is initialized; use sim.new(solver=...) to get a fresh copy'
                raise RuntimeError(errormsg)
            self.pars.solver = solver

        # Initialization steps
        if not self.initialized: self.init() # Automatically initialize if not initialized
        if verbose is not None:
            self._orig_verbose = self.verbose
            self.verbose = ss.parameters.check_verbose(verbose)
        self.timer.start()

        # Check for AlreadyRun errors
        if self.complete:
            errormsg = 'Simulation is already complete'
            raise AlreadyRunError(errormsg)

        try:
            # Main simulation loop -- just one line!!!
            reason = self.kernel_reason() if until is None and not profile else None
            kernels = until is None and not profile and reason is None
            if reason is not None and self.solver != 'abm' and getattr(self, 'declared', None): # Say once why a compartmental run is taken a step at a time
                sc.printv(f'Run step by step, since {reason}', 1, self.verbose)
            use_jax = False
            if self.jax:
                why = ss.gradients.reason(self)
                use_jax = why is None
                if why:
                    ss.note(f'JAX cannot run this sim, since {why}, so it is run by the kernels', sim=self)
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
                if shrink:
                    self.shrink(full=False)
                sc.printv(f'Run finished after {self.elapsed:0.2f} s.\n', 2, self.verbose)
        finally:
            self.dists.clear_run_caches()
        
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
            results (str/list): the results, by name (`'n_I'`, or `'sir.n_I'` where needed); default: every result of the declared modules
            pars (str/list): the parameters, by name; default: every parameter that has bounds

        Returns:
            An `sc.objdict` by parameter, then module, then result, of the derivative at each step

        Example:
            ```python
            beta = ss.par(0.3, bounds=[0.1, 1])
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
            sim = ss.Sim(sir, unit='day', stop=100)
            grad = sim.gradient('n_I')
            grad.beta.sir.n_I # How the number infectious at each step changes with beta
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

        Only for a sim that is nothing but compartments (see `sim.kernel_only()`), where
        it makes repeated runs many times faster. `sim.run(n_runs=...)` and `sim.calibrate()`
        use it where they can.

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            rand_seed (int): the seed for the next run

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ 0.3*I, ss.Transition(I, R, dur=8), init=dict(I=0.01))
            sim = ss.Sim(sir, unit='day', stop=100).init()
            peaks = [sim.restart(pars=dict(beta=beta)).run().results.sir.n_I.max() for beta in [0.2, 0.3, 0.4]]
            ```
        """
        if not self.initialized:
            self.init()
        self.loop.index = 0
        if not self.kernel_only():
            errormsg = f'This sim cannot be restarted, since it is not run in one call to the kernels: {self.kernel_reason()}. Use sim.new() to get a fresh copy instead'
            raise ValueError(errormsg)
        if rand_seed is not None:
            self.pars.rand_seed = rand_seed
        if pars:
            if any(isinstance(val, ss.Dist) for val in pars.values()):
                errormsg = 'A sim cannot be restarted with a new distribution, since its compartments were laid out for the old one; use sim.new(pars=...) instead'
                raise ValueError(errormsg)
            moving = [p for mod in self.declared for strat in mod.graph.strats if strat.mobility is not None for p in strat.mobility.fitted().values()]
            before = [float(p) for p in moving]
            self.par_values = sc.mergedicts(self.par_values, pars)
            self.set_par_values()
            if any(float(p) != val for p,val in zip(moving, before)): # The movement between places, worked out again with the new values
                self.mixing.init_movement()
                for mod in self.declared:
                    mod.engine.restart_movement()
            varies = lambda raw: isinstance(raw, ss.transitions.TimeValue) or (isinstance(raw, ss.by) and raw.varies)
            for mod in self.declared: # The kernels' program knows which values vary in time from when it was built
                timed = getattr(mod.engine, 'time_pars', None)
                if timed is not None:
                    for name,p in mod.graph.pars.items():
                        if varies(p.raw) != any(p is q for q in timed):
                            errormsg = f'This sim cannot be restarted with "{name}" {"varying" if varies(p.raw) else "constant"} in time, since it was built with it {"constant" if varies(p.raw) else "varying"}; use sim.new(pars=...) instead'
                            raise ValueError(errormsg)
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
                res = mod.results.get(f'n_{name}')
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
            raise AlreadyRunError('Simulation has already been finalized')

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
            how (str/dict): 'mean', 'median', 'last', 'sum', or a function, to summarize every result that way; or a dict of these by result key (e.g. `dict(sir_n_infected='mean')`), with the rest by kind
        """
        funcs = dict(mean=np.mean, median=np.median, sum=np.sum, last=lambda res: res[-1])
        if not isinstance(how, dict):
            how = {} if how is None else {'':how} # '' matches every key

        summary = sc.objdict()
        flat = self.results.flatten(sep='_', only_results=False, keep_case=True, columns=True) # Summarize each column of 2D results separately
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
            ss.shrink(sim, 'people')
        
        # If the sim is not initialized, we're done (ignoring the corner case where initialized modules are passed to an uninitialized sim)
        if sim.initialized:

            # Shrink the loop -- important; frees un-collectable sim references
            sim.loop.shrink()

            # Shrink the distributions
            ss.shrink(sim.dists, attrs=['sim', 'obj'])
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
            prof = sim.profile(follow=[net.add_pairs, sis.infect])
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
        data = {} # The data, by the result that each column matches
        if show_data and self.data is not None:
            x = self.data.index.values
            x = ss.DateArray(x) if len(x) and isinstance(x[0], ss.date) else x # As the results' time vector is, so they share an axis
            for col in self.data.columns:
                res = ss.results.match_name(self, col)
                if res is not None:
                    data[(res.module, res.name)] = (x, self.data[col].values)
        if fig is None and show_label in ['fig', 'fignum'] and self.label:
            plotlabel = self.label
            figlist = sc.autolist()
            while plt.fignum_exists(plotlabel):
                figlist += plotlabel
                plotlabel = sc.uniquename(self.label, figlist, human=True)
            kwargs['num'] = plotlabel
        title = self.label if show_label in ['title', 'suptitle'] else None
        return self.results.plot(key=key, fig=fig, data=data, label=self.label, title=title, show_skipped=show_skipped, show_module=show_module, n_ticks=n_ticks, annualize=annualize, **kwargs)


class AlreadyRunError(RuntimeError):
    """ Raised if trying to re-run an already-run sim without re-initializing """
    def __init__(self, *args, **kwargs):
        default = 'To run it again, make a new sim with sim.new(), or copy the sim before running it'
        if args:
            args = (f'{args[0]}\n{default}',) + args[1:]
        else:
            args = (default,)
        super().__init__(*args, **kwargs)


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
        ss.demo(diseases='hiv', networks='mf') # Run with different defaults
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

