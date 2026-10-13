"""
Define core Sim classes
"""
import sys
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
        pars (dict): either the sim parameters (as for the first argument), or the values of model parameters by name,
            e.g. `pars=dict(beta=0.1)` or `pars=fit.pars`; these apply to modules with declared parameters (see `ss.par`)
        label (str): the human-readable name of the simulation
        people (People): if provided, use this `ss.People` object
        demographics (str/Demographics/list): a string naming the demographics module to use, the module itself, or a list
        connectors (str/Connector/list): as above, for connectors
        networks (str/Network/list): as above, for networks
        interventions (str/Intervention/list): as above, for interventions
        diseases (str/Disease/list): as above, for diseases
        analyzers (str/Analyzer/list): as above, for analyzers
        custom (Module/list): as above, for any custom user-created modules not matching one of the above types
        modules (Module/list): alternatively, supply all modules together and divide among demographics, diseases, etc. based on type
        solver (str): how to run modules with declared states and transitions (see `ss.Disease`): 'auto' (the default: an ODE if
            every part of the model has a compartmental form, otherwise agents), 'ode', 'gillespie', 'tau', 'binomial', 'sde', 'abm', or
            'stochastic' (a stochastic compartmental solver chosen for the population size)
        unit (str): the unit of bare numbers in declared transitions, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        approx (bool): by default, each approximation a solver makes is noted in one line; True hides the notes, False makes them errors
        stratify (dict): stratify every module by a property of people, e.g. `dict(age=[0, 15, 65])`
        copy_inputs (bool): if True (default), copy modules as they're inserted into the sim, allowing
            the same module objects to be reused across multiple sims. Set to False if you want external
            references to the module objects to reflect state changes after sim.run().
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
    def __init__(self, model=None, label=None, people=None, demographics=None, connectors=None,
                 networks=None, diseases=None, interventions=None, analyzers=None, custom=None,
                 modules=None, copy_inputs=True, data=None, pars=None, **kwargs):
        self.pars = ss.SimPars() # Make default parameters (using values from parameters.py)
        if isinstance(model, ss.Module) or (isinstance(model, (list, tuple)) and len(model)): # The first argument can be the model: a module or a list of modules
            modules = sc.mergelists(model, modules)
        elif model is not None: # Or the sim parameters, as a dict
            kwargs = sc.mergedicts(model, kwargs)
        self._name_modules(sys._getframe(1), demographics, connectors, networks, interventions, diseases, analyzers, custom, modules)
        interventions = self._add_followups(interventions)
        self.par_values = None # Values of model parameters by name, e.g. pars=dict(beta=0.1) or pars=fit.pars
        if isinstance(pars, dict) and len(pars) and not any(key in self.pars for key in pars):
            self.par_values, pars = dict(pars), None
        args = dict(label=label, people=people, demographics=demographics, connectors=connectors,
                    networks=networks, interventions=interventions, diseases=diseases, 
                    analyzers=analyzers, custom=custom, modules=modules)
        args = {key:val for key,val in args.items() if val is not None} # Remove None inputs
        input_pars = sc.mergedicts(pars, args, kwargs, _copy=copy_inputs)
        self.pars.update(input_pars)  # Update the parameters
        if any(getattr(mod, 'graph', None) is not None for val in input_pars.values() for mod in sc.tolist(val)):
            self.inputs = sc.dcp(input_pars) # Kept so the sim can be rebuilt, e.g. with other parameter values for calibration
        self.given = [key for key,val in input_pars.items() if val is not None] # Which parameters were supplied, as opposed to defaults
        self.notes = [] # Notes on what was assumed or approximated when the sim was built
        self.solver = None # Set in init()

        # Set attributes; see also sim.init() for more
        self.created = sc.now()  # The datetime the sim was created
        self.version = ss.__version__ # The Starsim version
        self.metadata = ss.metadata()
        self.dists = ss.Dists(obj=self) # Initialize the random number generator container
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
        return

    @staticmethod
    def _name_modules(frame, *args):
        """ A module built with the plain ss.Disease() constructor takes its name from the variable it is assigned to """
        for arg in args:
            for mod in sc.tolist(arg):
                named = [mod] if type(mod) is ss.Disease or isinstance(mod, (ss.modify, ss.Delivery)) else []
                named += [getattr(mod, 'product', None)]*isinstance(mod, ss.Delivery)
                for obj in named:
                    if obj is not None and obj._default_name:
                        name = ss.transitions.find_name(obj, frame)
                        if name is not None:
                            obj.name = obj.label = name
                            obj._default_name = False
        return

    @staticmethod
    def _add_followups(interventions):
        """ A delivery that is the follow-up of another (on_positive=, on_negative=) is a module of its own, so is added to the list """
        if not any(isinstance(mod, ss.Delivery) for mod in sc.tolist(interventions)):
            return interventions
        out = []
        def add(mod, parent=None, suffix=None):
            if any(mod is other for other in out):
                return
            if parent is not None:
                mod.triggered = True
                if mod._default_name:
                    mod.name = mod.label = f'{parent.name}_{suffix}'
                    mod._default_name = False
            out.append(mod)
            if isinstance(mod, ss.Delivery):
                for key in ['positive', 'negative']:
                    child = getattr(mod, f'on_{key}')
                    if isinstance(child, ss.Delivery):
                        add(child, mod, key)
            return
        for mod in sc.tolist(interventions):
            add(mod)
        return out

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
            n = int(self.pars.n_agents)
            if self.initialized:
                timestr = f'{self.t.start}—{self.t.stop}'
            else:
                timestr = f'{self.pars.start}—{self.pars.stop}'

            moddict = {}
            for modkey in ss.module_map().keys():
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

            string = f'Sim({labelstr}n={n:n}; {timestr}{modulestr})'

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

    @property
    def modules(self):
        """
        Return an iterator over all Module instances (stored in standard places) in the Sim
        """
        for module in itertools.chain(
            self.custom(),
            self.demographics(),
            self.connectors(),
            self.networks(),
            self.interventions(),
            self.products(),
            self.diseases(),
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
        np.random.seed(self.pars.rand_seed) # Reset the seed before the population is created -- shouldn't matter if only using Dist objects
        self.pars.lock(False) # Allow parameters to be added during initialization
        self.pars.validate() # Validate parameters
        self.init_solver() # Choose the unit and the solver
        self.init_time() # Initialize time
        self.init_people(**kwargs) # Initialize the people
        self.init_module_attrs(force=force) # Load specified modules from parameters, initialize them, and move them to the Sim object
        self.init_modules_pre()

        # Final initializations -- this is "post"
        if self.pars.rescale:
            self.init_rescale()
        self.init_dists() # Initialize distributions
        self.init_people_vals() # Initialize the values in all the states and networks
        if self.solver == 'abm' and self.strats:
            self.init_strat_vals()
        elif self.solver != 'abm' and self.strats: # One row per group: age is the middle of the group
            for d,strat in enumerate(self.strats):
                if strat.kind == 'age':
                    self.people.age[self.people.auids] = strat.mids[self.group_coords[:, d]]
        self.init_modules_post() # Initialize the module values
        self.init_results() # Initialize the results
        self.init_data() # Initialize the data
        self.loop.init() # Initialize the integration loop
        self.lock_pars() # Don't allow new parameters to be added by mistake

        self.verbose = self.pars.verbose # Store a run-specific value of verbose
        if self.solver != 'abm': # One row per group, so properties that are not stratifications have no values to give
            dims = [strat.name for strat in self.strats]
            missing = [key for key in ['age', 'female'] if key not in dims and not (key == 'female' and 'sex' in dims)]
            for key in missing:
                self.people.__dict__.pop(key, None)
            self.people._no_groups = missing

        # It's initialized
        self.initialized = True
        if timer:
            self.elapsed = self.timer.toc(label='init', output=True, verbose=self.verbose)
        if self.diagnostics and self.diagnostics.states is not None: # Need not None since dict is empty at this point
            self.diagnostics.store_states(key='init')
        return self

    def init_solver(self):
        """
        Work out the sim's unit and solver; see `ss.Disease` for declaring a model that can be run with different solvers

        A sim with no declared states and transitions is run as agents, as in earlier
        versions of Starsim, and none of this applies to it.
        """
        pars = self.pars
        solvers = ss.solvers
        modules = [mod for key in ss.modules.module_types() + ['custom'] for mod in pars[key].values()]
        self.declared = [mod for mod in modules if getattr(mod, 'graph', None) is not None]
        solver = solvers.aliases.get(pars.solver, pars.solver)
        if solver not in ss.solver_names:
            errormsg = f'Unknown solver "{pars.solver}"; the choices are {sc.strjoin(ss.solver_names)}'
            raise ValueError(errormsg)
        self.unit = None # e.g. 'day'; None means time is counted in timesteps
        self.unit_base = 'year' # The unit that bare numbers are in; with no unit, this is whatever the timestep is
        self.unit_source = None
        self.modifiers = sc.ddict(list) # The modifiers on each quantity: ss.modify(), ss.change(), and the efficacy of products
        self.overlaps = [] # Pairs of modifiers that set the same quantity at once, which is noted
        self.set_par_values()
        self.init_strats()
        self.mixing = solvers.Mixing(self)
        if not self.declared:
            self.solver = 'abm'
            self.solver_reason = None
            return

        # The unit
        self.init_unit()

        # The solver
        agent_reasons = self.agent_reasons(modules)
        if solver == 'auto' and not agent_reasons:
            agent_reasons = self.try_functions()
        if solver == 'auto':
            if agent_reasons:
                solver, reason = 'abm', agent_reasons[0]
            else:
                solver, reason = 'ode', 'every part of the model has a compartmental form'
        elif solver == 'stochastic':
            total_pop = sc.ifelse(pars.total_pop, pars.n_agents)
            n_events = total_pop*sum(len(mod.graph.transitions) for mod in self.declared)
            if n_events <= 1e6:
                solver, reason = 'gillespie', f'exact; about {sc.sigfig(n_events, 1, sep=",")} events at this population size'
            else:
                solver, reason = 'tau', f'exact Gillespie would need about {n_events:.0e} events'
        else:
            reason = 'requested'
        self.solver = solver
        self.solver_reason = reason

        # The population
        if solver == 'abm':
            if 'n_agents' not in self.given and 'people' not in self.given and 'total_pop' in self.given:
                pars.n_agents = int(min(pars.total_pop, pars.n_agents))
                pars.pop_scale = pars.total_pop/pars.n_agents
        else:
            for mod in modules:
                if getattr(mod, 'graph', None) is None and not isinstance(mod, ss.Route) and not getattr(mod, 'compartmental', False):
                    ss.error('E804', f'cannot run with solver="{solver}": the module "{mod.name}" ({type(mod).__name__}) only works on agents.\n'
                                     f'     Run with solver="abm", or leave the solver as "auto".')
            self.integrator = sc.ifelse(pars.integrator, 'rk4')
            pars.pop_scale = 1.0
            pars.people_results = False

            self.mixing.init(approx=pars.approx)
            self.mixed = list(pars.networks.values()) # The networks are replaced by contact rates between groups, so are not run themselves
            pars.networks = ss.ndict()
            pars.n_agents = self.n_groups

        # Say what was chosen
        if pars.verbose:
            print(f'solver: {solver} ({reason})')
            if self.unit_source:
                ss.note(f"unit='{self.unit}' (from {self.unit_source})", code='W101', sim=self)
            for mod in self.declared:
                for key,why in mod.graph.guessed.items():
                    if key.endswith('.people'):
                        ss.note(f"state '{key.split('.')[1]}' treated as a count, not people ({why})", code='W212', sim=self)
        return

    def set_par_values(self):
        """ Apply parameter values given by name (pars=dict(beta=0.1), or 'sir.beta' if the name is used by more than one module) """
        for key,val in (self.par_values or {}).items():
            modname, _, name = key.rpartition('.')
            mods = [mod for mod in self.declared if name in mod.pars and modname in ['', mod.name]]
            if len(mods) != 1:
                options = [f'{mod.name}.{k}' for mod in self.declared for k in mod.graph.pars]
                ss.error('E602', f'pars= has a value for "{key}", which {"matches no parameter" if not mods else "is a parameter of more than one module"}. '
                                 f'Parameters are {sc.strjoin(options)}; did you mean {ss.transitions.closest(key, options)}?')
            mods[0].set_pars({name:val})
        return

    def new(self, pars=None, **kwargs):
        """
        A new, uninitialized sim built from the same inputs as this one, optionally with other values

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            kwargs (dict): sim arguments to change, e.g. `rand_seed=2` or `solver='abm'`
        """
        if not hasattr(self, 'inputs'):
            errormsg = 'sim.new() is only available for sims with declared states and transitions'
            raise ValueError(errormsg)
        inputs = sc.mergedicts(sc.dcp(self.inputs), kwargs)
        sim = Sim(inputs, data=sc.dcp(self.data))
        sim.par_values = sc.mergedicts(self.par_values, pars)
        return sim

    def calibrate(self, **kwargs):
        """
        Fit every parameter that has bounds to every result that has data; see `ss.fit()` for the arguments

        Returns an `ss.Fit`, with the calibrated sim as `fit.sim` and the best-fit values as `fit.pars`.
        """
        return ss.fit(self, **kwargs)

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
            sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=4), init_prev=0.01)
            ss.Sim(sir, solver='binomial', unit='day', stop=100).check_dt()
            ```
        """
        if not hasattr(self, 'inputs'):
            errormsg = 'sim.check_dt() is only available for sims with declared states and transitions'
            raise ValueError(errormsg)
        coarse = self if self.results_ready else self.new(verbose=0).run()
        fine = self.new(dt=coarse.t.dt/factor, verbose=0).run()
        n = coarse.t.npts - 1
        rows = []
        for mod in coarse.declared:
            for key,res in mod.results.items():
                if not isinstance(res, ss.Result) or key.startswith('cum_') or np.ndim(res.values) != 1:
                    continue
                other = fine.results[mod.name][key].values
                if (res.summarize_by or res.summary_method()) == 'sum': # A flow: total it over each of the larger steps
                    a, b = res.values[:n], other[:n*factor].reshape(n, factor).sum(axis=1)
                else:
                    a, b = res.values, other[::factor]
                scale = np.abs(a).max()
                change = np.abs(b - a).max()/scale if scale else 0.0
                rows.append(dict(module=mod.name, result=key, peak=a.max(), peak_fine=b.max(), change=change))
        df = sc.dataframe(rows).sort_values('change', ascending=False).reset_index(drop=True)
        if output:
            return df
        dt, unit = coarse.t.dt.to(coarse.unit_base).value, coarse.unit or 'step'
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
        graph = self.get_declared(modname).graph
        if kind is None:
            kind = 'transition' if name in graph.transitions else 'par'
        path = f'{modname}.pars.{name}' if kind == 'par' else f'{modname}.{name or kind}'
        mods = self.modifiers.get((modname, kind, name), [])
        lines = [path, '  modifiers' if mods else '  modifiers: none']
        lines += [f'    {mod.describe()}' for mod in mods]
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
        self.pars.stratify = sc.mergelists(self.pars.stratify, kwargs)
        return self

    def init_strats(self):
        """ Collect the stratifications of the sim and its modules, and work out the groups they give """
        trans = ss.transitions
        strats = sc.objdict()
        self.sim_strats = [] # The ones given to the sim, which apply to every module
        for call in sc.tolist(self.pars.stratify):
            call = dict(call)
            opts = {key:call.pop(key, None) for key in ['split', 'adjust', 'kind', 'mobility']}
            for name,levels in call.items():
                strat = trans.Strat(name, levels, **opts)
                if name in strats:
                    ss.error('E305', f'the sim is already stratified by "{name}".')
                strats[name] = strat
                self.sim_strats.append(strat)
        for mod in self.declared:
            for strat in mod.graph.strats:
                if strat.name in strats:
                    other = strats[strat.name]
                    if not strat.same_levels(other):
                        ss.error('E308', f'"{mod.name}" is stratified by "{strat.name}" with levels {strat.labels}, but another module or the sim uses {other.labels}.\n'
                                         f'     A property of people has the same groups for every module; stratify the sim instead: sim.stratify({strat.name}=[...]).')
                else:
                    strats[strat.name] = strat
            for tf in mod.graph.transfers.values(): # A boolean coordinate that does not exist yet is created, with everyone starting False
                if tf.dim not in strats:
                    if isinstance(tf.dst, bool):
                        strats[tf.dim] = trans.Strat(tf.dim, [False, True], split=[1.0, 0.0])
                        mod.graph.strats.append(strats[tf.dim])
                    else:
                        ss.error('E1102', f'the transfer "{tf.name}" moves people along "{tf.dim}", but the model is not stratified by it. Add e.g. {mod.name}.stratify({tf.dim}=[...]).')
        self.dim_moves = [] # Movement between levels that belongs to no one module, e.g. the protection of a product waning
        for mod in self.pars.interventions.values(): # Properties of people that the interventions need, e.g. holding a product
            if hasattr(mod, 'dims'):
                for strat in mod.dims():
                    strat.owner = mod.name # Part of the scenario, not of the model
                    strats.setdefault(strat.name, strat)
                self.dim_moves += [move for move in mod.dim_moves() if move.dim not in [m.dim for m in self.dim_moves]]
                for dim in (mod.move or {}):
                    if dim not in strats:
                        ss.error('E1102', f'the transfer "{mod.name}" moves people along "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        self.strats = list(strats.values())
        self.group_shape = tuple(strat.n for strat in self.strats)
        self.group_mults = np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:] # How far apart adjacent levels of each dimension are, in groups
        self.n_groups = int(np.prod(self.group_shape)) if self.strats else 1
        self.group_coords = np.indices(self.group_shape).reshape(len(self.strats), -1).T if self.strats else np.zeros((1, 0), dtype=int) # The level of each group in each dimension
        split = np.ones(self.n_groups)
        births = np.ones(self.n_groups)
        for d,strat in enumerate(self.strats):
            split *= strat.split[self.group_coords[:, d]]
            first = np.zeros(strat.n)
            first[0] = 1
            births *= (first if strat.kind == 'age' else strat.split)[self.group_coords[:, d]] # People are born into the youngest group
        self.group_split = split
        self.birth_split = births
        for strat in self.strats:
            strat.sim_coords = self.group_coords # So a result can look up which group is which
        for strat in self.strats: # Each property that people do not already have gets a distribution to assign it
            if strat.kind != 'age':
                setattr(self, f'rng_strat_{strat.name}', ss.random(name=f'rng_strat_{strat.name}'))
        if self.strats and self.pars.use_aging is None and any(strat.kind == 'age' for strat in self.strats):
            self.pars.use_aging = True
        return

    def get_declared(self, name):
        """ A module with declared states and transitions, by name """
        mods = {mod.name:mod for mod in self.declared}
        if name not in mods:
            errormsg = f'No module named "{name}" with declared transitions was found; these are {sc.strjoin(mods.keys())}'
            raise KeyError(errormsg)
        return mods[name]

    def dim_index(self, name):
        """ The position of a stratification dimension """
        names = [strat.name for strat in self.strats]
        if name not in names:
            errormsg = f'The model is not stratified by "{name}"; its dimensions are {names}'
            raise KeyError(errormsg)
        return names.index(name)

    def group_raw(self):
        """ In an agent-based run: the group of every agent, as an array indexed by UID """
        cache = getattr(self, '_group_cache', None)
        if cache is not None and cache[0] == self.ti and len(cache[1]) == len(self.people.uid.raw):
            return cache[1]
        people = self.people
        out = np.zeros(len(people.uid.raw), dtype=int)
        for strat,mult in zip(self.strats, np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:]):
            if strat.edges is not None:
                levels = np.clip(np.digitize(people[strat.name].raw, strat.edges) - 1, 0, strat.n - 1)
            else:
                levels = np.nan_to_num(people[strat.name].raw).astype(int)
            out += levels*int(mult)
        self._group_cache = (self.ti, out)
        return out

    def init_strat_vals(self):
        """ In an agent-based run: assign each agent a level of every property they do not already have """
        people = self.people
        uids = people.auids
        for strat in self.strats:
            if strat.kind != 'age':
                rvs = getattr(self, f'rng_strat_{strat.name}').rvs(uids)
                people[strat.name][uids] = np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1)
        return

    def agent_reasons(self, modules):
        """ The reasons, if any, that the model has to run as agents """
        reasons = []
        if 'n_agents' in self.given:
            reasons.append('n_agents was given')
        if 'people' in self.given:
            reasons.append('people were given')
        for mod in modules:
            graph = getattr(mod, 'graph', None)
            if hasattr(mod, 'agent_reasons'):
                reasons += mod.agent_reasons()
            if graph is None:
                if isinstance(mod, ss.Network):
                    reasons.append(f'{type(mod).__name__} is an individual-level network')
                elif not isinstance(mod, (ss.MixingPool, ss.MixingPools)) and not getattr(mod, 'compartmental', False):
                    reasons.append(f'{type(mod).__name__} only works on agents')
            else:
                for method in ['step', 'step_state']:
                    if getattr(type(mod), method) is not getattr(ss.Disease, method):
                        reasons.append(f'"{mod.name}" defines its own {method}()')
                for p in graph.all_pars():
                    raw = p.raw
                    if callable(raw) and not isinstance(raw, ss.Dist) and len(inspect.signature(raw).parameters) >= 2:
                        reasons.append(f'the function "{raw.__name__}" takes uids')
                if self.pars.approx is False:
                    for name,tr in graph.transitions.items():
                        raw = None if tr.dur is None else tr.dur.raw
                        if isinstance(raw, ss.Dist) and tr.stages is None and type(raw).__name__ not in ['expon', 'gamma']:
                            reasons.append(f'"{name}" has a {type(raw).__name__} dwell time, and approx=False')
        return reasons

    def try_functions(self):
        """ Try each function in the model on groups, to see whether the model can run as compartments; returns the reasons it cannot """
        funcs = [p for mod in self.declared for p in mod.graph.all_pars() if callable(p.raw) and not isinstance(p.raw, ss.Dist)]
        funcs += [mod for mod in list(self.pars.interventions.values()) + list(self.pars.connectors.values()) if hasattr(mod, 'try_groups')]
        if not funcs or not hasattr(self, 'inputs'):
            return []
        reasons = []
        try:
            probe = self.new(solver='ode', verbose=0, approx=True).init()
            for mod in probe.declared:
                mod.engine.start_step()
                for p in mod.graph.all_pars():
                    if callable(p.raw) and not isinstance(p.raw, ss.Dist):
                        try:
                            mod.engine.value(p)
                        except Exception as E:
                            what = str(E).split(':', 1)[-1].split('.')[0].strip() if isinstance(E, ss.CodedError) else f'{type(E).__name__}: {E}'
                            reasons.append(f'the function "{p.raw.__name__}" does not work on groups ({what})')
            for mod in probe.interventions() + probe.connectors():
                if hasattr(mod, 'try_groups'):
                    try:
                        mod.try_groups()
                    except Exception as E:
                        what = str(E).split(':', 1)[-1].split('\n')[0].strip() if isinstance(E, ss.CodedError) else f'{type(E).__name__}: {E}'
                        reasons.append(f'"{mod.name}" does not work on groups ({what})')
        except ss.CodedError as E:
            reasons.append(f'the model does not run as compartments ({str(E).splitlines()[0]})')
        return reasons

    def init_unit(self):
        """ Work out the sim's unit: from unit=, then dt, then start and stop, then the typed values in the model; else time is in timesteps """
        pars = self.pars
        time = ss.time
        unit = source = None
        def base(obj):
            return type(obj).base.removesuffix('s')
        def is_dur(obj):
            return isinstance(obj, ss.dur) and not isinstance(obj, ss.datedur)
        dates = [v for v in [pars.start, pars.stop] if isinstance(v, (str, ss.date))]
        years = [v for v in [pars.start, pars.stop] if sc.isnumber(v) and time.assume_cal_year(v)]
        durs = [v for v in [pars.start, pars.stop, pars.dur] if is_dur(v)]
        if pars.unit is not None:
            unit = pars.unit.removesuffix('s')
        elif is_dur(pars.dt):
            unit, source = base(pars.dt), 'dt'
        elif isinstance(pars.dt, str):
            unit, source = pars.dt.removesuffix('s'), 'dt'
        elif dates:
            unit, source = 'day', 'start and stop dates'
        elif years:
            unit, source = 'year', 'start and stop years'
        elif durs:
            unit, source = base(durs[0]), 'start and stop'
        else:
            units = [u.removesuffix('s') for mod in self.declared for u in mod.graph.unit_values()]
            units += [base(mod.t.dt) for mod in self.declared if is_dur(mod.t.dt)]
            if units:
                order = ['day', 'week', 'month', 'year']
                unit = sorted(set(units), key=order.index)[0]
                source = 'the typed values in the model'
                if len(set(units)) > 1:
                    source += f', which have different units ({sc.strjoin(sorted(set(units), key=order.index))}); the smallest is used for bare numbers. Set unit= to choose'
        self.unit = unit
        self.unit_source = source
        self.unit_base = unit or 'day'

        # Give the timeline the unit
        dur_class = time.get_dur_class(self.unit_base)
        if pars.dt is None or sc.isnumber(pars.dt):
            pars.dt = dur_class(sc.ifelse(pars.dt, 1.0))
        if not dates and not years:
            if pars.start is None or sc.isnumber(pars.start):
                pars.start = dur_class(sc.ifelse(pars.start, 0))
            if sc.isnumber(pars.stop):
                pars.stop = dur_class(pars.stop)
        return

    def describe(self, output=False):
        """
        Print the resolved model: every state, transition, and default, with each guess marked

        Args:
            output (bool): if True, return the text rather than printing it
        """
        if not self.initialized:
            self.init()
        pars = self.pars
        unit = self.unit or 'step'
        mods = list(self.modules)
        lines = []
        def add(key, text, guess=None):
            line = f'{key:12s} {text}'
            if guess:
                line = f'{line:60s} [{guess}]'
            lines.append(line.rstrip())
            return
        popguess = '' if ('total_pop' in self.given or 'n_agents' in self.given) else ' [guessed]'
        t = self.t
        lines.append(f'Sim: {len(mods)} module{"s" if len(mods) != 1 else ""}, {pars.total_pop:,.0f} people{popguess}, {unit}s {t.start} to {t.stop}' if self.unit is None or not t.is_absolute else
                     f'Sim: {len(mods)} module{"s" if len(mods) != 1 else ""}, {pars.total_pop:,.0f} people{popguess}, {t.start} to {t.stop}')
        if self.declared:
            guessed = pars.solver in ['auto', 'stochastic']
            add('solver', self.solver, f'guessed: {self.solver_reason}' if guessed else 'requested')
            if self.solver == 'abm':
                scale = f', each standing for {pars.pop_scale:n} people' if pars.pop_scale != 1 else ''
                add('', f'{pars.n_agents:,} agents{scale}', None if 'n_agents' in self.given else 'guessed')
            if self.unit is None:
                add('unit', 'none: time is in timesteps', 'nothing in the model has a unit')
            else:
                add('unit', self.unit, f'from {self.unit_source}' if self.unit_source else None)
            for strat in self.strats: # Each dimension: its levels, the states it applies to, and what it generates
                text = f'{strat.name} ({strat.kind}): {sc.strjoin(strat.labels)}'
                text += f'; applies to {sc.strjoin(strat.states)}' if strat.states is not None else '; applies to every state' if strat.kind == 'space' else ''
                if strat.kind == 'space':
                    text += f'; mobility: {strat.mobility if strat.mobility is not None else "none"}'
                add('dimension', text, f'guessed: {"; ".join(strat.guessed.values())}' if strat.guessed else None)
        for mod in mods:
            if hasattr(mod, 'describe') and getattr(mod, 'graph', None) is not None:
                lines += ['', mod.describe(output=True)]
        if self.declared:
            lines.append('')
            place = f' within each {self.strats[self.mixing.space].name}' if self.mixing.space is not None else ''
            nets = list(self.networks.values()) + getattr(self, 'mixed', [])
            if nets:
                for net in nets:
                    term = [term for term in self.mixing.terms if term.name == net.name] if self.solver != 'abm' else []
                    how = '' if not term else f': {term[0].contacts:n} contacts per {unit}' if term[0].kind == 'homogeneous' else f': contact matrix by {self.strats[term[0].dim].name}'
                    add('mixing', f'{net.name} ({type(net).__name__}){how}')
            else:
                add('mixing', f'homogeneous{place}, 1 contact per {unit}, frequency-dependent', 'guessed: no network given')
            for mod in list(self.connectors()) + list(self.interventions()):
                if isinstance(mod, ss.modify):
                    add('modifier', f'{mod.target}: {" ".join(mod.describe().split())}')
                elif isinstance(mod, ss.Delivery):
                    add('delivery', mod.describe())
            for mod in self.declared:
                names = [f'n_{name}' for name in mod.graph.states]
                extra = [key for key,res in mod.results.items() if isinstance(res, ss.Result) and key not in names and key not in mod.graph.transitions and not key.startswith('cum_')]
                add('results', f'{", ".join(names)}, and one flow per transition' + (f'; also {", ".join(extra)}' if extra else ''))
            for note in self.notes:
                add('note', note)

        # The schedule: one line per timestep in use
        groups = sc.ddict(list)
        for mod in mods:
            groups[mod.t.dt.to(self.unit_base).value if self.declared else mod.t.dt.years].append(mod.name)
        for i,(dt,names) in enumerate(sorted(groups.items())):
            every = f'every {unit}' if dt == 1 else f'every {dt:n} {unit}s'
            add('schedule' if i == 0 else '', f'{every}: {", ".join(names)}')
        text = '\n'.join(lines)
        return text if output else print(text)

    def solvers(self, output=False):
        """
        Print which solvers can run the model: one row per ingredient that matters, one column per solver

        Args:
            output (bool): if True, return the text rather than printing it
        """
        if not self.initialized:
            self.init()
        rows = []
        for net in list(self.networks.values()) + getattr(self, 'mixed', []):
            if isinstance(net, ss.Network):
                rows.append([f'network: {type(net).__name__}', 'approx (contact rates)', 'approx (contact rates)', 'yes'])
        for mod in self.modules:
            graph = getattr(mod, 'graph', None)
            if graph is None:
                if not isinstance(mod, ss.Route) and not getattr(mod, 'compartmental', False):
                    rows.append([f'module: {mod.name}', 'no (works on agents)', 'no (works on agents)', 'yes'])
                continue
            for name,tr in graph.transitions.items():
                raw = None if tr.dur is None else tr.dur.raw
                if isinstance(raw, ss.Dist) and type(raw).__name__ != 'expon':
                    shape = raw.pars.get('a')
                    exact = tr.stages is not None or (type(raw).__name__ == 'gamma' and sc.isnumber(shape) and shape == int(shape))
                    how = 'yes (Erlang stages)' if exact else 'approx (Erlang stages)'
                    rows.append([f'dwell time: {name} ({type(raw).__name__})', how, how, 'yes'])
            for p in graph.all_pars():
                raw = p.raw
                if callable(raw) and not isinstance(raw, ss.Dist):
                    agent_only = len(inspect.signature(raw).parameters) >= 2
                    how = 'no (takes uids)' if agent_only else '? (tried when run)'
                    rows.append([f'function: {raw.__name__}', how, how, 'yes'])
        header = ['', 'ode', 'stochastic', 'abm']
        table = [header] + rows
        widths = [max(len(row[i]) for row in table) + 3 for i in range(4)]
        lines = [''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in table]
        if not rows:
            lines.append('(every part of the model runs under every solver)')
        reasons = self.agent_reasons(list(self.modules)) if self.declared else ['the model is written as code']
        choice = f'abm ({reasons[0]})' if reasons else 'ode (every part of the model has a compartmental form)'
        lines += ['', f'auto would choose: {choice}']
        text = '\n'.join(lines)
        return text if output else print(text)

    def hash(self):
        """
        The run's identity: short hashes of the model's structure, its parameter values, and the run settings, plus the seed

        Two sims with the same `model` hash have the same states and transitions; the
        same `pars` hash, the same values. This tells apart the many runs that come
        from one script (a sweep, a calibration), and tests whether two models written
        differently are the same model.
        """
        if not self.initialized:
            self.init()
        spec = self.to_spec()
        digest = ss.utils.short_hash
        model = dict(modules=spec['modules'], networks=spec['networks'], stratify=spec['stratify'])
        structure, values = ss.utils.split_numbers(model, unit=self.unit, skip=['guessed', 'label', 'doc', 'source'])
        config = {k:v for k,v in spec['config'].items() if k != 'rand_seed'}
        return sc.objdict(model=digest(structure), pars=digest(values), config=digest(config), scenario=digest(spec['interventions']), seed=self.pars.rand_seed)

    def to_spec(self, filename=None, **kwargs):
        """
        The model spec: everything needed to rebuild the model (modules, networks, interventions, and run settings), as plain data

        Args:
            filename (str): if provided, save to this file
            kwargs (dict): passed to `sc.savejson()`
        """
        if not self.initialized:
            self.init()
        pars = self.pars
        def modspec(mod):
            if getattr(mod, 'graph', None) is not None or isinstance(mod, (ss.modify, ss.Delivery)):
                return mod.to_json()
            return sc.jsonify(ss.Module.to_json(mod))
        def tstr(val):
            return float(val.value) if isinstance(val, ss.dur) else str(val)
        out = sc.objdict()
        out.schema = 'lingua-franca/1.0'
        out.modules = [modspec(mod) for key in ['custom', 'demographics', 'diseases'] for mod in self[key]()]
        out.networks = [modspec(mod) for mod in list(self.networks()) + getattr(self, 'mixed', [])]
        out.stratify = [sc.dcp(strat.spec) for strat in getattr(self, 'strats', []) if getattr(strat, 'owner', None) is None]
        out.modifiers = [modspec(mod) for mod in self.connectors()]
        out.interventions = [modspec(mod) for mod in self.interventions()]
        out.config = sc.objdict(solver=pars.solver, integrator=pars.integrator, approx=pars.approx, unit=self.unit,
                                start=tstr(self.t.start), stop=tstr(self.t.stop), dt=float(self.t.dt.to(self.unit_base).value),
                                total_pop=float(pars.total_pop), n_agents=int(pars.n_agents) if self.solver == 'abm' else None, rand_seed=pars.rand_seed)
        out.meta = sc.objdict(library_version=ss.__version__, solver_chosen=self.solver, solver_reason=self.solver_reason, notes=list(self.notes))
        out = sc.jsonify(out)
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

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
            sim = ss.Sim(diseases='sis', networks='random')
            sim.run(until='2010-01-01')
            sim.add_module(ss.routine_vx(product=ss.simple_vx(disease='sis'), prob=0.5))
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
            for modkey, modclass in ss.module_map().items():
                if isinstance(module, modclass):
                    key = modkey
                    break
        names = {mod.name:None for mod in self.modules}
        ss.SimPars.validate_name(module, key, names)
        modules = [module]
        if getattr(module, 'has_product', False) and not any(module.product is p for p in self.products()):
            ss.SimPars.validate_name(module.product, 'products', names)
            modules.append(module.product)
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

    def init_time(self):
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
        n_agents = self.pars.n_agents
        verbose = sc.ifelse(verbose, self.pars.verbose)
        if verbose > 0 and self.solver == 'abm':
            labelstr = f' "{self.label}"' if self.label else ''
            print(f'Initializing sim{labelstr} with {n_agents:0n} agents')

        # If people have not been supplied, make them -- typical use case
        if people is None:
            if self.solver == 'abm' and self.strats: # Properties that people are stratified by, other than age
                extra = [ss.FloatArr(strat.name, default=0, label=strat.name) for strat in self.strats if strat.kind != 'age']
                kwargs['extra_states'] = sc.mergelists(kwargs.get('extra_states'), extra)
                for strat in self.strats: # Ages follow the split of the age groups, so every solver starts from the same population
                    if strat.kind == 'age' and strat.edges is not None and 'age_data' not in kwargs:
                        top = max(100, strat.edges[-1] + 10)
                        kwargs['age_data'] = np.column_stack([np.append(strat.edges, top), np.append(strat.split, 0)])
            people = ss.People(n_agents=n_agents, **kwargs)  # This just assigns UIDs and length
        elif self.solver == 'abm':
            for strat in self.strats:
                if strat.name not in people.states:
                    state = ss.FloatArr(strat.name, default=0, label=strat.name)
                    people.states.append(state)
                    setattr(people, strat.name, state)
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

    def init_modules_pre(self):
        """ Initialize all the modules with the sim """
        for mod in self.modules:
            mod.init_pre(self)
        return

    def init_dists(self):
        """ Initialize the distributions """
        # Initialize all distributions now that everything else is in place
        self.dists.init(obj=self, base_seed=self.pars.rand_seed, force=True)

        # Copy relevant dists to each module
        for mod in self.modules:
            self.dists.copy_to_module(mod)
        return

    def init_people_vals(self):
        """ Initialize the People states with actual values """
        self.people.init_vals()
        return

    def init_modules_post(self):
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

    def init_rescale(self):
        """ Check that the diseases support dynamic rescaling, and create the distribution for it """
        for disease in self.diseases():
            if not hasattr(disease, 'ti_infected'):
                errormsg = f'Dynamic rescaling (rescale=True) requires each disease to have a ti_infected state, but "{disease.name}" does not'
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
            not_naive = ss.uids.concatenate([disease.ti_infected.notnan.uids for disease in self.diseases()]).unique()
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
        elif isinstance(data, str):
            data = ss.data(data).load(self)
        self.data = ss.validate_sim_data(data)
        if self.data is not None and getattr(self, 'declared', None):
            ss.data.check(self, self.data)
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
            self.diagnostics = ss.Diagnostics(sim=self, rvs=rvs, states=states, detailed=detailed)
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
        self.elapsed = self.timer.toc(output=True)
        if self.verbose: # Print progress
            t = self.t
            simlabel = f'"{self.label}": ' if self.label else ''
            string = f'  Running {simlabel}{t.now("str")} ({t.ti:2.0f}/{t.npts}) ({self.elapsed:0.2f} s) '
            if self.verbose >= 1:
                sc.heading(string)
            elif self.verbose > 0:
                if not (t.ti % int(1.0 / self.verbose)):
                    sc.progressbar(t.ti + 1, t.npts, label=string, length=20, newline=True)
        return

    def finish_step(self):
        """ Finish the simulation timestep """
        if self.diagnostics and self.diagnostics.states:
            self.diagnostics.store_states()
        self.t.ti += 1
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
        self.loop.run(self.t.now(), verbose)
        return self

    def run(self, until=None, verbose=None, shrink=False, check_method_calls=True, profile=False, solver=None, n_runs=None):
        """
        Run the model -- the main method for running a simulation.

        Args:
            until (date/str/float): the date to run the sim until
            verbose (float): the level of detail to print (default 0.1, i.e. output once every 10 steps)
            shrink (bool): whether to explicitly shrink the sim after running
            check_method_calls (bool): whether to check that all required methods were called
            profile (bool): if True, record per-entry CPU timing in the loop (see `sim.loop.to_df()`/`plot_cpu()`)
            solver (str): the solver to use, if different from the sim's (only before the sim is initialized)
            n_runs (int): if provided, run the sim this many times with different seeds, and return an `ss.MultiSim`; a shortcut for `ss.parallel(sim, n_runs=n_runs)`
        """
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
            self.verbose = verbose
        self.timer.start()

        # Check for AlreadyRun errors
        if self.complete:
            errormsg = 'Simulation is already complete'
            raise AlreadyRunError(errormsg)

        try:
            # Main simulation loop -- just one line!!!
            self.loop.run(until, profile=profile)

            # Check if the simulation is complete
            if self.loop.index == len(self.loop.plan):
                self.complete = True

            # If simulation reached the end, finalize the results
            if self.complete:
                self.finalize()
                if check_method_calls:
                    self.check_method_calls()
                if shrink:
                    self.shrink(full=False)
                sc.printv(f'Run finished after {self.elapsed:0.2f} s.\n', 1, self.verbose)
        finally:
            self.dists.clear_run_caches()
        
        return self # Allows e.g. ss.Sim().run().plot()

    def finalize(self):
        """ Compute final results """
        # Reset the time index (done in the modules as well in mod.finalize())
        self.t.ti -= 1  # During the run, this keeps track of the next step; restore this be the final day of the sim

        # Finalize the results
        self.finalize_results()

        # Finalize each module, including the results
        for mod in self.modules:
            mod.finalize()

        # Resets verbose if needed
        if hasattr(self, '_orig_verbose'):
            self.verbose = self._orig_verbose
            delattr(self, '_orig_verbose')

        # Generate the summary and finish up
        self.summarize() # Create summary
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

    def summarize(self, how='default'):
        """
        Provide a quick summary of the sim; returns the last entry for count and
        cumulative results, and the mean otherwise.

        Args:
            how (str): how to summarize: can be 'mean', 'median', 'last', or a mapping of result keys to those
        """
        def get_func(key, how, default='mean'):
            """
            Find the right function by matching the "how" key with the result key

            For example, hkey="cum_ " will match result key "cum_infections"
            """
            func = None
            for hkey,hfunc in how.items():
                if hkey in key:
                    func = hfunc
                    break
            if func is None:
                func = default
            return func

        def get_result(res, func):
            """ Convert a string to the actual function to use, e.g. "median" maps to np.median() """
            if   func == 'mean':   return res.mean()
            elif func == 'median': return np.median(res)
            elif func == 'last':   return res[-1]
            elif callable(func):   return func(res)
            else: raise Exception(f'"{func}" is not a valid function')

        # Convert "how" from a string to a dict
        if how == 'default':
            how = {'cumulative':'last', 'n_':'mean', 'new_':'mean', 'cum_':'last', 'timevec':'last', '':'mean'}
        elif isinstance(how, str):
            how = {'':how} # Match everything

        summary = sc.objdict()
        flat = self.results.flatten(sep='_', only_results=False, keep_case=True, columns=True) # Summarize each column of 2D results separately
        for key, res in flat.items():
            if 'timevec' not in key: # Skip module-specific time vectors
                try:
                    func = get_func(key, how)
                    entry = get_result(res, func)
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

                # Check that the module successfully shrunk
                if size_limit:
                    npts = len(sim)
                    max_size = size_limit*(base_size + npts) # Maximum size in KB: size_limit scales both the fixed base and the per-timestep allowance
                    for mod in sim.modules:
                        size = sc.checkmem(mod, descend=0).bytesize[0]/1e3 # Size in KB
                        if size > max_size:
                            errormsg = f'Module {mod.name} did not successfully shrink: {size:n} KB > {max_size:n} KB (= {size_limit:n} KB/timestep × ({base_size:n} base + {npts} timesteps)); use die=False to turn this message into a warning, or increase base_size/size_limit'
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

    def to_df(self, sep='_', **kwargs):
        """
        Export results as a Pandas dataframe
        Args:
            sep (str): separator for the keys
            kwargs: passed to results.to_df()
        """
        self.check_results_ready('Please run the sim before exporting the results')
        df = self.results.to_df(sep=sep, descend=True, **kwargs)
        return df

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
        prof = ss.Profile(self, follow=follow, do_run=do_run, plot=plot, **kwargs)
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
        Save to disk as a gzipped pickle.

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
        Export results and parameters as JSON.

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
            annualize (bool): if True, resample results to annual values before plotting (uses each Result's summarize_by to determine sum/mean/last)
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

        # Figure out the flat structure of results to plot
        flat = ss.utils.match_result_keys(self.results, key, show_skipped=(show_skipped or key)) # Always show skipped with a custom key

        # Annualize results if requested
        if annualize:
            flat = sc.objdict({k: res.annualize() if isinstance(res, ss.Result) else res for k, res in flat.items()})

        # Set figure defaults
        n_cols,_ = sc.getrowscols(len(flat)) # Number of columns of axes
        default_figsize = np.array([8, 6])
        figsize_factor = np.clip((n_cols-3)/6+1, 1, 1.5) # Scale the default figure size based on the number of rows and columns
        figsize = default_figsize*figsize_factor

        # Set plotting defaults
        kw = ss.plot_args(kwargs, figsize=figsize, alpha=0.8, data_alpha=0.3, data_color='k')

        # Do the plotting
        with ss.style(**kw.style):

            # Get the figure
            if fig is None:
                if show_label in ['fig', 'fignum'] and self.label:
                    plotlabel = self.label
                    figlist = sc.autolist()
                    while plt.fignum_exists(plotlabel):
                        figlist += plotlabel
                        plotlabel = sc.uniquename(self.label, figlist, human=True)
                    kw.fig['num'] = plotlabel
                fig, axs = sc.getrowscols(len(flat), make=True, **kw.fig)
                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()
            else:
                axs = fig.axes
            if not sc.isiterable(axs):
                axs = [axs]

            # Do the plotting
            df = self.data if show_data else None # For plotting the data
            for ax, (key, res) in zip(axs, flat.items()):

                # Plot data
                if df is not None:
                    mod = res.module
                    if mod in self.label_to_name: # res.module stores the label; resolve to the name
                        mod = self.label_to_name[mod]
                    name = res.name
                    found = False
                    for dfkey in [f'{mod}.{name}', f'{mod}_{name}', name]: # Allow dot, underscore, or just the name
                        if dfkey in df.cols:
                            found = True
                            break
                    if found:
                        ax.scatter(df.index.values, df[dfkey].values, **kw.data)

                # Plot results, with one line per column for 2D results
                if res.columns is None:
                    ax.plot(res.timevec, res.values, **kw.plot, label=self.label)
                else:
                    labels = [f'{col} ({self.label})' if self.label else col for col in res.columns]
                    ax.plot(res.timevec, res.values, **kw.plot, label=labels)
                    ax.legend()
                ss.utils.format_axes(ax, res, n_ticks, show_module)

        if show_label in ['title', 'suptitle'] and self.label:
            fig.suptitle(self.label, weight='bold')

        sc.figlayout(fig=fig)

        return ss.return_fig(fig, **kw.return_fig)


class AlreadyRunError(RuntimeError):
    """ Raised if trying to re-run an already-run sim without re-initializing """
    def __init__(self, *args, **kwargs):
        default = 'Did you mean to copy the sim before running it?'
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


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    Args:
        sim1 (Sim/MultiSim/dict): either a simulation/MultiSim object or the sim.summary dictionary
        sim2 (im/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        full (bool): whether to print out all values (not just those that differ)
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match

    Examples:
        ```python
        s1 = ss.Sim(rand_seed=1).run()
        s2 = ss.Sim(rand_seed=2).run()
        ss.diff_sims(s1, s2)
        ```
    '''

    # Convert to dict
    if isinstance(sim1, (Sim, ss.MultiSim)):
        sim1 = sim1.summarize()
    if isinstance(sim2, (Sim, ss.MultiSim)):
        sim2 = sim2.summarize()
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict):  # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)


    # Check if it's a multisim
    sim1 = sc.objdict(sim1)
    sim2 = sc.objdict(sim2)
    multi = isinstance(sim1[0], dict) # If it's a dict, then it's a multisim
    if multi:
        for sim in [sim1, sim2]:
            assert 'mean' in sim[0], f"Can only compare multisims with summarize(method='mean'), but you have keys {sim[0].keys()}"

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs:  # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\n'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    n_mismatch = 0
    skip = sc.tolist(skip)
    for key in sim2.keys():  # To ensure order
        if key in sim1_keys and key not in skip:  # If a key is missing, don't count it as a mismatch
            d = sc.objdict()
            if multi:
                d.sim1 = sim1[key]['mean']
                d.sim2 = sim2[key]['mean']
                d.sim1_sem = sim1[key]['sem']
                d.sim2_sem = sim2[key]['sem']
            else:
                d.sim1 = sim1[key]
                d.sim2 = sim2[key]
            mm = not np.isclose(d.sim1, d.sim2, equal_nan=True)
            n_mismatch += mm
            if mm or full:
                mismatches[key] = d

    df = sc.dataframe() # Preallocate in case there were no mismatches
    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n' if not full else ''
        df = sc.dataframe.from_dict(mismatches).transpose()
        diff = []
        ratio = []
        change = []
        zscore = []
        small_change = 1 + 1e-3  # Define a small change, e.g. a rounding error
        for d in mismatches.values():
            old = d.sim1
            new = d.sim2
            if multi:
                old_sem = d.sim1_sem
                new_sem = d.sim2_sem
                sem = old_sem + new_sem
                small_change = 1.96 # 95% CI, roughly speaking

            numeric = sc.isnumber(old) and sc.isnumber(new) # Should all be numeric, but just in case
            if numeric:
                this_diff = new - old
                this_ratio = new / old if old else np.nan
                if multi:
                    abs_ratio = abs(this_diff)/sem
                    this_zscore = abs_ratio
                else:
                    abs_ratio = max(this_ratio, 1.0/this_ratio) if this_ratio > 0 else np.inf # Ratio is undefined for zero or sign changes
                    this_zscore = np.nan

                # Set the character to use
                approx_eq = abs_ratio < small_change
                if approx_eq:    change_char = '≈'
                elif new > old:  change_char = '↑'
                elif new < old:  change_char = '↓'
                elif new == old: change_char = '='
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio == 0:   repeats = 0
                if abs_ratio >= 1.1: repeats = 2
                if abs_ratio >= 2:   repeats = 3
                if abs_ratio >= 10:  repeats = 4
                this_change = change_char * repeats
            else:  # pragma: no cover
                this_diff =  np.nan
                this_ratio  = np.nan
                this_change = 'N/A'
                this_zscore = np.nan

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)
            zscore.append(this_zscore)

        df['diff'] = diff
        df['ratio'] = ratio
        numeric_cols = ['sim1', 'sim2', 'diff', 'ratio']
        if multi:
            numeric_cols += ['sim1_sem', 'sim2_sem']
        for col in numeric_cols:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        if multi:
            df['zscore'] = zscore
            df['statsig'] = df.zscore > small_change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg:  # pragma: no cover
        if die and n_mismatch: # To catch full=True case
            raise ValueError(mismatchmsg)
        elif output:
            return df
        else:
            print(mismatchmsg)
            return True
    else:
        if not output:
            print('Sims match')
        return False


def check_sims_match(*args, full=False):
    """
    Shortcut to using ss.diff_sims() to check if multiple sims match

    Args:
        args (list): a list of 2 or more sims to compare
        full (bool): if True, return whether each sim matches the first

    Examples:
        ```python
        s1 = ss.Sim(diseases='sir', networks='random').run()
        s2 = ss.Sim(pars=dict(diseases='sir', networks='random')).run()
        s3 = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()).run()
        assert ss.check_sims_match(s1, s2, s3)
        ```
    """
    if len(args) < 2:
        errormsg = 'Must compare at least 2 sims'
        raise ValueError(errormsg)
    base = args[0]
    matches = []
    for other in args[1:]:
        diff = diff_sims(base, other, full=False, output=False, die=False)
        matches.append(not(diff)) # Return the opposite of the diff
    if full:
        return matches
    else:
        return all(matches)
