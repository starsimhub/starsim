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
            switching to tau-leaping, with a note, when an ODE run estimates more than a million events; `approx=False` keeps it exact), or 'abm'
        integrator (str): the scheme of the ODE: 'rk4' (the default) or 'euler'
        unit (str): the unit of bare numbers in declared transitions, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        approx (bool): by default, each approximation a solver makes is noted in one line (kept in `sim.notes`, and printed unless `verbose=0`); True does not print them, and False makes them errors
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
    def __init__(self, *models, label=None, people=None, demographics=None, connectors=None,
                 networks=None, diseases=None, interventions=None, analyzers=None, custom=None,
                 modules=None, copy_inputs=True, data=None, pars=None, **kwargs):
        self.pars = ss.SimPars() # Make default parameters (using values from parameters.py)
        models = [arg for arg in models if arg is not None]
        for arg in models: # The model: modules, or lists of them; or the sim parameters, as a dict
            if isinstance(arg, dict):
                kwargs = sc.mergedicts(arg, kwargs)
            elif not (isinstance(arg, ss.Module) or (isinstance(arg, (list, tuple)) and all(isinstance(mod, ss.Module) for mod in arg))):
                errormsg = f'ss.Sim() takes the model (modules, or lists of them) or a dict of parameters as positional arguments, not {type(arg).__name__}; give everything else by keyword, e.g. label="..."'
                raise TypeError(errormsg)
        positional = [arg for arg in models if not isinstance(arg, dict)]
        if positional:
            modules = sc.mergelists(*positional, modules)
        self._name_modules(ss.utils.getframe(1), demographics, connectors, networks, interventions, diseases, analyzers, custom, modules)
        interventions = self._add_followups(interventions)
        self.par_values = None # Values of model parameters by name, e.g. pars=dict(beta=0.1) or pars=fit.pars
        self.jax = False # Whether a whole run of the ODE is taken in JAX (as ss.parallel() does for many runs with different values): see ss.gradients
        if isinstance(pars, (str, os.PathLike)): # A file of values by name
            pars = ss.load_pars(pars)
        if isinstance(pars, dict) and len(pars) and not any(key in self.pars for key in pars):
            self.par_values, pars = dict(pars), None
        args = dict(label=label, people=people, demographics=demographics, connectors=connectors,
                    networks=networks, interventions=interventions, diseases=diseases, 
                    analyzers=analyzers, custom=custom, modules=modules)
        args = {key:val for key,val in args.items() if val is not None} # Remove None inputs
        input_pars = sc.mergedicts(pars, args, kwargs, _copy=copy_inputs)
        self.pars.update(input_pars)  # Update the parameters
        self.inputs = sc.dcp(input_pars) # Kept so the sim can be rebuilt (see sim.new()), e.g. with other parameter values for calibration
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
                named = [mod] if type(mod) is ss.Disease or isinstance(mod, (ss.modify, ss.Delivery, ss.PartnerNet)) else []
                named += [getattr(mod, 'product', None)]*isinstance(mod, ss.Delivery)
                for obj in named:
                    if obj is not None and obj._default_name:
                        name = ss.transitions.find_name(obj, frame)
                        if name is None and getattr(obj, 'graph', None) is not None: # Not bound to one variable, e.g. built in a helper: named from its states instead, e.g. 'sir'
                            name = ''.join(s.name[0].lower() for s in obj.graph.people_states) or None
                        if name is not None:
                            obj.name = obj.label = name
                            obj._default_name = False
                if isinstance(mod, ss.Delivery):
                    mod.name_product()
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
                    self.people.age[self.people.auids] = self.group_vals(d, strat.mids, outside=strat.mids @ strat.split)
                elif strat.female is not None: # A sex dimension is people.female
                    self.people.female[self.people.auids] = self.group_vals(d, np.arange(strat.n) == strat.female, outside=False)
        self.init_modules_post() # Initialize the module values
        for crossing in self.crossings: # Once every disease in it has its initial conditions
            crossing.init()
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
        if self.verbose and self.solver != 'abm': # As a check that the model and its seed are what was meant
            for dis in self.declared:
                if dis.graph.transmissions:
                    try:
                        ngm = dis.engine.next_generation()
                    except Exception: # Not every model has one; the line is only a check
                        ngm = None
                    if ngm is not None:
                        R0 = float(np.max(np.abs(np.linalg.eigvals(ngm))))
                        print(f'{dis.name}: R0 = {R0:.3g}, with {float(dis.engine.count(dis.graph.infectious)):,.0f} infectious at the start')
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
        self.init_flows()
        for mod in self.declared: # A transition for some people only: a modifier that sets its rate to 0 for everyone else
            for name,tr in mod.graph.transitions.items():
                where = getattr(tr, 'where', None)
                key = f'where_{mod.name}_{name}'
                if where is not None and key not in pars.connectors:
                    if callable(where) and not isinstance(where, ss.conditions.Cond): # A function of the sim
                        others = (lambda func: lambda sim: ~np.asarray(func(sim), dtype=bool))(where)
                    else: # A condition, a state, or an array
                        others = ~ss.conditions.as_cond(where)
                    pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=0, where=others, name=key)
        for m in list(pars.interventions.values()) + list(pars.connectors.values()): # A parameter used only by a modifier (its amount, e.g. ss.par(0.6, bounds=[0.1, 1])) belongs to the module it modifies, so it can be set by name and fitted
            if isinstance(m, ss.modify) and isinstance(m.amount, ss.par):
                p = m.amount
                target = next((mod for mod in self.declared if mod.name == ss.conditions.modname(m.target_module)), None)
                p.name = p.name or m.name
                if target is not None and p.name not in target.graph.pars:
                    target.graph.pars[p.name] = p
                    target.define_pars(**{p.name:p})
        solver = pars.solver
        if solver in solvers.removed:
            raise ValueError(solvers.removed[solver])
        elif solver not in ss.solvers.solver_names:
            errormsg = f'Unknown solver "{pars.solver}"; the choices are {sc.strjoin(ss.solvers.solver_names)}'
            raise ValueError(errormsg)
        if pars.integrator == 'jax':
            errormsg = 'integrator="jax" has been removed: integrator= is the scheme of the ODE ("rk4" or "euler"). JAX is used without being asked by sim.gradient(), sim.log_density(), gradient fits, and ss.parallel() with many sets of values.'
            raise ValueError(errormsg)
        elif pars.integrator is not None and pars.integrator not in solvers.integrators:
            errormsg = f'Unknown integrator "{pars.integrator}"; integrator= is the scheme of the ODE, {sc.strjoin(solvers.integrators)}'
            raise ValueError(errormsg)
        self.unit = None # e.g. 'day'; None means time is counted in timesteps
        self.unitless = False # Whether time is counted in timesteps, with no unit: set in init_unit()
        self.unit_base = 'year' # The unit that bare numbers are in; with no unit, this is whatever the timestep is
        self.unit_source = None
        self.modifiers = sc.ddict(list) # The modifiers on each quantity: ss.modify(), ss.change(), and the efficacy of products
        self.overlaps = [] # Pairs of modifiers that set the same quantity at once, which is noted
        self.clipped = set() # Quantities whose modifiers took them out of their range, which were clipped to it
        self.crossings = [] # Diseases whose states are tracked jointly in a compartmental run: see init_crossings()
        self.cross_strats = []
        for key in [key for key in pars.keys() if key not in ss.SimPars.names()]: # A model parameter assigned on the sim's parameters, e.g. sim.pars.beta = 0.1, is a value by name
            if any(key in mod.pars for mod in self.declared):
                self.par_values = sc.mergedicts(self.par_values, {key:pars.pop(key)})
        self.set_par_values()
        self.check_par_values() # Not on a restart, so a fit can probe values at its bounds
        self.init_named_people()
        self.init_strats()
        self.mixing = solvers.Mixing(self)
        if not self.declared:
            if solver not in ['auto', 'abm']:
                raise ss.CapabilityError(f'cannot run with solver="{solver}": the model is written as code (it has no declared states and transitions), so it runs as agents.\n'
                                         f'     Leave the solver as "auto", or declare the model\'s states and transitions to run it as compartments.')
            self.solver = 'abm'
            self.solver_reason = None
            self.capabilities = []
            return

        for mod in self.declared: # The initial values, by now all given
            mod.graph.guess_init()

        # States that nothing enters: a typo in an arrow string makes a new state
        if self.mixing.register is None: # A register can add people to any state
            for mod in self.declared:
                for state in mod.graph.unentered():
                    others = [name for name in mod.graph.states if name != state.name]
                    raise ss.UnknownState(f'nothing moves people into the state "{state.name}" of "{mod.name}", and it starts empty, so it may be a typo{ss.transitions.did_you_mean(state.name, others)}\n'
                                          f'     If it is meant to start empty, declare the states (S, I, R = ss.states("S I R")) or give it a value, e.g. init=dict({state.name}=0).')

        # The unit
        self.init_unit()

        # The solver
        self.init_capabilities(modules)
        if solver == 'auto' and not self.auto_reasons():
            self.try_capabilities()
        if solver == 'auto':
            reasons = self.auto_reasons()
            tried = [cap.ingredient.split(': ', 1)[-1] for cap in self.capabilities if cap.probe is not None and cap.groups == 'yes']
            if reasons:
                solver, reason = 'abm', reasons[0]
            else:
                solver, reason = 'ode', 'every part of the model has a compartmental form' + (f'; {sc.strjoin(tried)} {"was" if len(tried) == 1 else "were"} tried on groups' if tried else '')
        elif solver == 'gillespie' and pars.approx is not False: # One event at a time, unless that would take too long
            n_events = self.count_events()
            if n_events > solvers.max_events:
                solver, reason = 'tau', f'tau-leaping for solver="gillespie": an ODE run gives about {n_events:.1e} events'
                ss.note(f'solver="gillespie" takes tau leaps: an ODE run gives about {n_events:.1e} events, more than {solvers.max_events:.0e}; approx=False keeps it exact', sim=self, verbose=0 if pars.approx else None)
            else:
                reason = f'exact; an ODE run gives about {sc.sigfig(n_events, 2, sep=",")} events'
        else:
            reason = 'requested'
        self.solver = solver
        self.solver_reason = reason
        self.init_lifetables()

        # The population
        if solver == 'abm':
            if 'n_agents' not in self.given and ('people' not in self.given or self.named_unsized) and 'total_pop' in self.given:
                pars.n_agents = int(min(pars.total_pop, pars.n_agents))
                pars.pop_scale = pars.total_pop/pars.n_agents
            self.mixing.init_pools()
        else:
            for cap in self.capabilities:
                if cap.fatal:
                    raise ss.CapabilityError(f'cannot run with solver="{solver}": {cap.auto}.\n'
                                             f'     Run with solver="abm", or leave the solver as "auto".')
            for mod in self.declared: # Variation between people is lost without agents
                for p in mod.graph.spread_pars():
                    msg = f'"{p.name}" of "{mod.name}" is a distribution across people ({type(p.raw).__name__}), so a compartmental run uses its mean'
                    if pars.approx is False:
                        raise ss.CapabilityError(f'{msg}, which approx=False does not allow.\n     Run as agents, or bin it into a dimension, e.g. {mod.name}.stratify({p.name}=[...]).')
                    else:
                        ss.note(f'{msg}; to keep the variation, run as agents or bin it into a dimension', sim=self, verbose=0 if pars.approx else None)
            self.integrator = sc.ifelse(pars.integrator, 'rk4')
            pars.pop_scale = 1.0
            pars.people_results = False
            if any(mod.staged(self) for mod in pars.interventions.values() if isinstance(mod, ss.Delivery)): # Levels that the deliveries need now the solver is known: see ss.Delivery.dims()
                self.init_strats()
                self.mixing = solvers.Mixing(self)

            self.init_crossings()
            self.mixing.init(approx=pars.approx)
            self.mixed = list(pars.networks.values()) # The networks are replaced by contact rates between groups, so are not run themselves
            pars.networks = ss.ndict()
            pars.n_agents = self.n_groups

        # A beta for each route names every route and nothing else; a route has no beta of its own
        nets = list(self.mixed if solver != 'abm' else pars.networks.values())
        routes = [net.name for net in nets]
        for mod in self.declared:
            for tr in mod.graph.transmissions:
                for p in [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, dict)]:
                    given = [ss.standardize_netkey(k) for k in p.raw]
                    unknown = [k for k in p.raw if ss.standardize_netkey(k) not in [ss.standardize_netkey(r) for r in routes]]
                    missing = [r for r in routes if ss.standardize_netkey(r) not in given]
                    if unknown or missing:
                        raise ss.RouteMismatch(f'the beta of "{mod.name}.{tr.name}" is given for each route, as {sc.strjoin(p.raw.keys())}, but the routes are {sc.strjoin(routes) or "none (homogeneous mixing)"}: '
                                               f'{"no route is called " + sc.strjoin(unknown) if unknown else ""}{"; " if unknown and missing else ""}{sc.strjoin(missing) + " " + ("has" if len(missing) == 1 else "have") + " no value" if missing else ""}.')
        for net in nets: # beta times the number of contacts is a hazard, so only one of them may carry a time unit
            n = net.pars.get('n_contacts')
            rate = n.pars.get('v') if isinstance(n, ss.constant) else n
            if isinstance(rate, ss.Rate):
                for mod in self.declared:
                    for tr in mod.graph.transmissions:
                        typed = [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.Rate)]
                        if typed:
                            raise ss.WrongDimension(f'the beta of "{mod.name}.{tr.name}" ({typed[0].raw!r}) and the n_contacts of "{net.name}" ({rate!r}) both carry a time unit, so their product is not a hazard; give one of them as a bare number, e.g. a probability per contact for beta.')
        if all(getattr(dis, 'graph', None) is not None for dis in pars.diseases.values()):
            for net in nets:
                beta = net.pars.get('beta', 1.0) if isinstance(net, (ss.MixingPool, ss.MixingPools)) else 1.0
                if not (sc.isnumber(beta) and beta == 1):
                    raise ss.RouteMismatch(f'the route "{net.name}" has beta={beta}, but in a declared model the beta belongs to the transmission; give it there, e.g. ss.transmission("S -> I", beta=dict({net.name}=...)).')

        # A count of people in a rate is literal, so the routes do not apply to it
        for mod in self.declared if routes else []:
            for name,tr in mod.graph.transitions.items():
                counts = [leaf for leaf in (tr.rate.leaves if tr.rate is not None else []) if isinstance(leaf, ss.transitions.Count) and any(getattr(s, 'people', True) for s in leaf.states)]
                if counts:
                    ss.note(f'"{mod.name}.{name}" uses {counts[0]} literally, so {sc.strjoin(routes)} {"does" if len(routes) == 1 else "do"} not apply to it; write the bare state (e.g. {counts[0].states[0].label_name}) for exposure through the routes', sim=self)

        # Say what was chosen; the notes are kept in sim.notes whatever the verbosity
        for mod in self.declared: # A value for each level, checked when the sim is built rather than when it first runs
            for p in mod.graph.all_pars():
                if isinstance(p.raw, ss.by):
                    p.raw.check(self)
        if pars.verbose:
            print(f'solver: {solver} ({reason})')
        if self.unit_source:
            ss.note(f"unit='{self.unit}' (from {self.unit_source})", ss.UnitInferred, sim=self)
        for mod in self.declared:
            for state in mod.graph.count_states:
                if 'people' in state.guessed:
                    ss.note(f"state '{state.name}' treated as a count, not people ({state.guessed.people})", ss.StateNotPeople, sim=self)
        return

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
                raise ss.UnknownParameter(f'pars= has a value for "{key}", which {"matches no parameter" if not mods else "is a parameter of more than one module"}. '
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
            raise ss.UnknownParameter(f'pars= has values for levels of "{key}" ({sc.strjoin(given.keys())}), which is not a parameter with a value for each level of one stratification, e.g. ss.par(ss.by(patch=[0.2, 0.3]))')
        strats = [strat for mod in self.declared for strat in mod.graph.strats] + list(getattr(self, 'sim_strats', []))
        strat = next((strat for strat in strats if strat.name == raw.dim), None)
        labels = [str(label) for label in strat.labels] if strat is not None else [str(i) for i in range(len(raw.values))]
        vals = list(sc.tolist(raw.values))
        for label,val in given.items():
            if label not in labels:
                raise ss.UnknownParameter(f'pars= has a value for "{key}[{label}]", but "{raw.dim}" has no level "{label}"; its levels are {sc.strjoin(labels)}')
            vals[labels.index(label)] = val
        return ss.by(**{raw.dim:vals})

    def init_lifetables(self):
        """ In a compartmental run, a life table (`ss.death(lifetable=)`) is a death rate for each age group; as agents, each person's age at death is drawn (see `ss.solvers.Agents.step_lifetables()`) """
        for mod in self.declared:
            for name,tr in mod.graph.transitions.items():
                table = getattr(tr, 'lifetable', None)
                if table is None or self.solver == 'abm':
                    continue
                age = next((strat for strat in self.strats if strat.kind == 'age' and strat.edges is not None), None)
                if age is None:
                    raise ss.CapabilityError(f'"{name}" of "{mod.name}" draws ages at death from a life table, which needs each person\'s age; it cannot run with solver="{self.solver}" unless the model is stratified by age.\n'
                                             f'     Stratify by age (e.g. sir.stratify(age=[0, 5, 15, 50, 65])), or run with solver="abm".')
                p = tr.rate.leaves[0]
                p.value = ss.by(**{age.name:[ss.peryear(h) for h in table.hazards(age.edges)]})
        return

    def check_par_values(self):
        """ Check that every parameter of the model has a value (`ss.UnvaluedParameter`), inside its bounds (`ss.OutOfBounds`) """
        for mod in self.declared:
            for name,p in mod.graph.pars.items():
                raw = p.raw
                if raw is None:
                    raise ss.UnvaluedParameter(f'the parameter "{name}" of "{mod.name}" has no value. Give it one (ss.par(0.1)), or bounds to start from (ss.par(bounds=[0.01, 0.5])), or a value by name (ss.Sim(pars=dict({name}=...))).')
                if p.bounds is None:
                    continue
                val = raw.value if isinstance(raw, ss.TimePar) else raw.array if isinstance(raw, ss.by) else raw
                if not (sc.isnumber(val) or isinstance(val, np.ndarray)): # E.g. a distribution or a function
                    continue
                low, high = [(type(raw)(b) if isinstance(raw, ss.TimePar) else b).value if isinstance(b, ss.TimePar) else b for b in p.bounds] # A typed bound in the unit of the value
                tol = 1e-9*abs(high - low) # For rounding, e.g. on a log scale
                val = np.asarray(val, dtype=float)
                if np.any(val < low - tol) or np.any(val > high + tol):
                    raise ss.OutOfBounds(f'the parameter "{name}" of "{mod.name}" is {ss.transitions.value_str(raw)}, outside its bounds [{low:n}, {high:n}].')
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
        inputs = sc.mergedicts(sc.dcp(self.inputs), kwargs)
        inputs = {key:val for key,val in inputs.items() if not (key in kwargs and val is None)} # As if it had not been given
        sim = Sim(inputs, data=sc.dcp(data))
        sim.par_values = sc.mergedicts(self.par_values, pars)
        if hasattr(self, 'imported'): # The same import, with what it did not translate
            sim.imported = self.imported
        return sim

    def calibrate(self, **kwargs):
        """
        Fit every parameter that has bounds to every result that has data; see `ss.fit()` for the arguments

        Returns an `ss.Fit`, with the calibrated sim as `fit.sim` and the best-fit values as `fit.pars`.
        """
        return ss.fit(self, **kwargs)

    def sensitivity(self, **kwargs):
        """ Which parameters matter: partial rank correlations of each parameter with bounds and each result, over samples within the bounds; see `ss.calibration.sensitivity()` """
        return ss.calibration.sensitivity(self, **kwargs)

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

    def init_strats(self):
        """ Collect the stratifications of the sim and its modules, and work out the groups they give """
        trans = ss.transitions
        strats = sc.objdict()
        self.sim_strats = [] # The ones given to the sim, which apply to every module
        for call in sc.tolist(self.pars.stratify):
            for strat in trans.parse_stratify(call, graphs=[mod.graph for mod in self.declared], sim=True)[0]:
                if strat.name in strats:
                    raise ss.DuplicateStructuralDimension(f'the sim is already stratified by "{strat.name}".')
                strats[strat.name] = strat
                self.sim_strats.append(strat)
        for mod in self.declared:
            for strat in mod.graph.strats:
                if strat.name in strats:
                    other = strats[strat.name]
                    if not strat.same_levels(other):
                        raise ss.InconsistentDimension(f'"{mod.name}" is stratified by "{strat.name}" with levels {strat.labels}, but another module or the sim uses {other.labels}.\n'
                                                       f'     A property of people has the same groups for every module; stratify the sim instead: sim.stratify({strat.name}=[...]).')
                    elif 'split' not in strat.guessed and strat.joint is None and not np.allclose(strat.split, other.split): # An explicit split is used, whichever stratification it is on
                        if 'split' not in other.guessed:
                            raise ss.InconsistentDimension(f'"{mod.name}" is stratified by "{strat.name}" with split {strat.split.tolist()}, but another module or the sim gives it the split {other.split.tolist()}; give it once.')
                        other.split = strat.split
                        other.guessed.pop('split')
                else:
                    strats[strat.name] = strat
            for tf in mod.graph.transfers.values(): # A boolean coordinate that does not exist yet is created, with everyone starting False
                if tf.dim not in strats:
                    if isinstance(tf.dst, bool):
                        strats[tf.dim] = trans.Strat(tf.dim, [False, True], split=[1.0, 0.0])
                        mod.graph.strats.append(strats[tf.dim])
                    else:
                        raise ss.MissingDimension(f'the transfer "{tf.name}" moves people along "{tf.dim}", but the model is not stratified by it. Add e.g. {mod.name}.stratify({tf.dim}=[...]).')
        self.dim_moves = [] # Movement between levels that belongs to no one module, e.g. the protection of a product waning
        for mod in self.pars.interventions.values(): # Properties of people that the interventions need, e.g. holding a product
            if hasattr(mod, 'dims'):
                for strat in mod.dims(self):
                    strat.owner = mod.name # Part of the scenario, not of the model
                    strats.setdefault(strat.name, strat)
                self.dim_moves += [move for move in mod.dim_moves() if move.dim not in [m.dim for m in self.dim_moves]]
                for dim in (mod.move or {}):
                    if dim not in strats:
                        raise ss.MissingDimension(f'the transfer "{mod.name}" moves people along "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        moved = [tf.dim for mod in self.declared for tf in mod.graph.transfers.values()] # A dimension that people are moved along, with no split of its own, starts with everyone at its first level
        moved += [dim for mod in self.declared for tr in mod.graph.transitions.values() for dim in (getattr(tr, 'move', None) or {})]
        moved += [dim for mod in self.pars.interventions.values() for dim in (getattr(mod, 'move', None) or {})]
        for name in moved:
            strat = strats.get(name)
            if strat is not None and strat.guessed.get('split') == 'equal across levels':
                strat.split = strat.first()
                strat.guessed['split'] = f'everyone at {strat.labels[0]}, the first level, since people are moved between its levels'
        for strat in self.cross_strats:
            if strat.name in strats:
                raise ss.DuplicateStructuralDimension(f'the sim has a dimension called "{strat.name}", which is also the name of a disease whose states are tracked jointly with another\'s; rename one of them.')
            strats[strat.name] = strat
        self.strats = list(strats.values())
        for strat in self.strats: # The kind comes from the value, not the name; a name that suggests another kind is noted
            if strat.hint:
                ss.note(f'the dimension "{strat.name}" is {strat.guessed["kind"]}', sim=self)
        named = getattr(self, 'named_people', None)
        given = getattr(self, 'given_pyramid', None)
        for strat in self.strats: # Age groups with no split of their own take that of a named population, or of the people given to a compartmental run
            if (named is not None or given is not None) and strat.kind == 'age' and strat.edges is not None and strat.guessed.get('split') == ss.named.world.guess:
                ages, counts = (named.ages, named.counts) if named is not None else given
                strat.split = ss.named.age_split(strat.edges, ages, counts)
                strat.split_from = 'people'
                strat.guessed.pop('split')
        pops = [strat for strat in self.strats if strat.geo is not None and strat.geo.pop is not None]
        if len(pops) == 1 and 'total_pop' not in self.given and self.pop_from is None: # Places with populations, and no other population size: their total
            self.pars.total_pop = float(pops[0].geo.pop.sum())
            self.pars.pop_scale = self.pars.total_pop/self.pars.n_agents
            self.pop_from = f'the places of "{pops[0].name}"'
        self.group_shape = tuple(strat.n for strat in self.strats)
        self.group_mults = np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:] # How far apart adjacent levels of each dimension are, in groups
        self.n_groups = int(np.prod(self.group_shape)) if self.strats else 1
        self.group_coords = np.indices(self.group_shape).reshape(len(self.strats), -1).T if self.strats else np.zeros((1, 0), dtype=int) # The level of each group in each dimension
        split = np.ones(self.n_groups)
        births = np.ones(self.n_groups)
        for d,strat in enumerate(self.strats):
            strat.dim = d
        joints = [strat.joint for strat in self.strats if strat.joint is not None and strat.states is None] # A split over several dimensions of some states only is the share entering each level: see below
        if len(joints) > 1:
            errormsg = 'Only one split= can cover several dimensions at once'
            raise ValueError(errormsg)
        self.joint = None # For a split over several dimensions at once: their positions, and the share of people at each combination of their levels
        if joints:
            ds, grid = joints[0].grid(self)
            grid = grid/grid.sum()
            for i,d in enumerate(ds): # Each dimension's own split is the total over the others
                self.strats[d].split = grid.sum(axis=tuple(j for j in range(len(ds)) if j != i))
                self.strats[d].guessed.pop('split', None)
            self.joint = (ds, grid)
            split *= grid[tuple(self.group_coords[:, d] for d in ds)]
        for d,strat in enumerate(self.strats):
            if self.joint is None or d not in self.joint[0]:
                split *= strat.split[self.group_coords[:, d]]
            first = np.zeros(strat.n)
            first[0] = 1
            births *= (first if strat.kind == 'age' else strat.split)[self.group_coords[:, d]] # People are born into the youngest group
        for d,strat in enumerate(self.strats): # A split of a dimension of some states only, given the levels of others (e.g. clinical severity by age): normalized over this dimension
            if strat.joint is not None and strat.states is not None:
                ds, grid = strat.joint.grid(self)
                if d not in ds:
                    errormsg = f'The split of "{strat.name}" is given over {sc.strjoin(strat.joint.dims)}, which does not include "{strat.name}"'
                    raise ValueError(errormsg)
                axis = ds.index(d)
                totals = grid.sum(axis=axis, keepdims=True)
                grid = np.divide(grid, totals, out=np.full(grid.shape, 1/strat.n), where=totals>0)
                strat.cond = (ds, grid)
                weights = np.prod([self.strats[dd].split[self.group_coords[:, dd]] for dd in ds if dd != d], axis=0) # The split overall, for anything that has no group to go by
                split_all = np.bincount(self.group_coords[:, d], weights=weights*strat.shares(self.group_coords), minlength=strat.n)
                strat.split = split_all/split_all.sum()
                strat.guessed.pop('split', None)
        self.init_ragged()
        self.group_split = self.fold(split)
        self.birth_split = self.fold(births)
        for strat in self.strats:
            strat.sim_coords = self.group_levels # So a result can look up which group is which
        for strat in self.strats: # Each property that people do not already have gets a distribution to assign it
            if strat.kind != 'age':
                setattr(self, f'rng_strat_{strat.name}', ss.random(name=f'rng_strat_{strat.name}'))
        if 'use_aging' not in self.given: # People age if and only if they are born or die, unless told otherwise
            self.pars.use_aging = self.has_demography()
            if not self.pars.use_aging and any(strat.kind == 'age' and strat.edges is not None for strat in self.strats):
                ss.note('people do not age between the age groups, since nobody is born or dies; pass use_aging=True to age them', sim=self)
        return

    @property
    def unit_name(self):
        """ The unit of time, for what the sim says: its unit, or, with none, "step" (or "unit", if a step is not one unit) """
        if self.unit:
            return self.unit
        dt = self.pars.dt
        dt = dt.value if isinstance(dt, ss.dur) else dt
        return 'step' if dt is None or dt == 1 else 'unit'

    def has_demography(self):
        """ Whether people are born or die other than of a disease: demographics modules, or a declared birth, arrival, or death from every state """
        return bool(self.pars.demographics) or any(self.flows_of(mod) for mod in self.declared)

    @staticmethod
    def flows_of(mod):
        """ The transitions of a declared disease that are births, arrivals, or deaths of everyone, so are the population's """
        return [tr for tr in mod.graph.transitions.values() if (tr.inflow and tr.people) or getattr(tr, 'every_state', False)]

    def init_flows(self):
        """
        Births, deaths, and migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...), ss.death(...)])`): see `ss.demographics.Flows`

        People are born and die once, so these become the transitions of one disease, the first declared (or, for births into a
        state that is named, the first with that state), whose people are everyone's: as agents, the diseases share the people, and
        in a compartmental run with several declared diseases, their states are tracked jointly (see `init_crossings()`). For the
        same reason, births or deaths of everyone declared in more than one disease are an error.
        """
        for flows in [mod for mod in self.pars.demographics.values() if isinstance(mod, ss.demographics.Flows)]:
            for tr in flows.transitions:
                tr = sc.dcp(tr) # The sim's own, so the module is unchanged if it is built again
                named = tr.dst[0] if tr.inflow and tr.arrow_given else None
                holder = next((mod for mod in self.declared if named is None or named in mod.graph.states), self.declared[0])
                holder.graph.add_flow(tr)
        holders = [mod.name for mod in self.declared if self.flows_of(mod)]
        if len(holders) > 1:
            raise ss.DuplicateDeclaration(f'births or deaths of everyone are declared in {sc.strjoin(holders)}; people are born and die once, whatever their diseases, so this would count them twice.\n'
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

    def init_ragged(self):
        """
        For a dimension with `where=` (e.g. age groups for women only): which groups have a level of it

        The groups stay a grid, with a level of every dimension. People in a group outside a
        dimension's `where=` have no level of it, so they are all counted at its first level
        (the "fold" of their group), as people in a state that a dimension does not apply to are;
        the groups at its other levels stay empty.
        """
        coords = self.group_coords.copy()
        inside = np.ones(coords.shape, dtype=bool) # Whether each group (rows) has a level of each dimension (columns)
        pending = [d for d,strat in enumerate(self.strats) if getattr(strat, 'where', None)]
        names = [strat.name for strat in self.strats]
        for d in pending:
            for dim in self.strats[d].where:
                if dim not in names:
                    raise ss.MissingDimension(f'"{self.strats[d].name}" is stratified where {dim} is given, but the model is not stratified by "{dim}". Add e.g. sim.stratify({dim}=[...]).')
                if dim == self.strats[d].name:
                    errormsg = f'The where= of "{dim}" refers to "{dim}" itself'
                    raise ValueError(errormsg)
        while pending: # A dimension whose where= refers to another with a where= is worked out after it
            ready = [d for d in pending if not any(names.index(dim) in pending for dim in self.strats[d].where)]
            if not ready:
                errormsg = f'The where= of {sc.strjoin(names[d] for d in pending)} refer to each other'
                raise ValueError(errormsg)
            for d in ready:
                strat = self.strats[d]
                mask = np.ones(self.n_groups, dtype=bool)
                for dim,levels in strat.where.items():
                    dd = names.index(dim)
                    inds = [self.strats[dd].index(level) for level in sc.tolist(levels)]
                    mask &= np.isin(coords[:, dd], inds) & inside[:, dd]
                inside[:, d] = mask
                coords[~mask, d] = 0
                pending.remove(d)
        ragged = not inside.all()
        self.group_inside = inside
        self.group_fold = np.ravel_multi_index(coords.T, self.group_shape) if ragged else None # The group that each group's people are counted in
        self.group_levels = np.where(inside, self.group_coords, -1) if ragged else self.group_coords # The level of each group in each dimension, or -1 for none
        return

    def fold(self, vals):
        """ Move values for each group (e.g. a split) to the groups their people are counted in: see `init_ragged()` """
        if self.group_fold is None:
            return vals
        return np.bincount(self.group_fold, weights=np.broadcast_to(vals, (self.n_groups,)), minlength=self.n_groups)

    def fold_matrix(self, matrix):
        """ The same, for a matrix from each group (rows) to each group (columns) """
        if self.group_fold is None or matrix is None:
            return matrix
        out = np.zeros_like(matrix)
        np.add.at(out.T, self.group_fold, matrix.T)
        return out

    def group_vals(self, d, vals, outside=1.0):
        """ A value for each level of a dimension (by position), as a value for each group; a group with no level of it gets `outside` """
        out = vals[self.group_coords[:, d]]
        if self.group_fold is not None:
            out = np.where(self.group_inside[:, d], out, outside)
        return out

    def grid_index(self, ds, grid, count=False):
        """
        Index a grid of values over some dimensions (by position) by group: returns the grid and the index, so `grid[index]` is the value for each group

        A group with no level of a dimension (outside its `where=`) takes the mean over its
        levels, weighted by the split; or, for a number of people at each level, none.
        """
        index = []
        for i,d in enumerate(ds):
            levels = self.group_coords[:, d]
            if self.group_fold is not None and not self.group_inside[:, d].all():
                strat = self.strats[d]
                extra = np.zeros_like(np.take(grid, [0], axis=i)) if count else np.average(grid, axis=i, weights=strat.split, keepdims=True)
                grid = np.concatenate([grid, extra], axis=i)
                levels = np.where(self.group_inside[:, d], levels, strat.n)
            index.append(levels)
        return grid, tuple(index)

    def init_crossings(self):
        """
        In a compartmental run: track jointly the states of diseases when a condition in one refers to a state of another, or when people are born or die (see `init_flows()`)

        For example, `ss.modify(hiv.rel_trans, multiply=3, where=syph.I)` needs the number of people with
        HIV who also have syphilis. Each disease in a crossing becomes a dimension of the sim, whose levels
        are its compartments and which applies to the people of the others, so the model grows to the
        product of their compartments. See `ss.solvers.Crossing`.
        """
        names = [mod.name for mod in self.declared]
        links = {name:{name} for name in names}
        if any(self.flows_of(mod) for mod in self.declared): # People born or dying, who are born or die once in every disease: see init_flows()
            links = {name:set(names) for name in names}
        for mod in list(self.pars.connectors.values()) + list(self.pars.interventions.values()):
            home = getattr(mod, 'target_module', None) or getattr(mod, 'disease', None) # The disease whose engine evaluates the conditions
            home = ss.conditions.modname(home)
            if home not in links:
                continue
            for other,state in ss.conditions.state_refs(mod):
                if other in links and other != home and state in self.get_declared(other).graph.states:
                    group = links[home] | links[other]
                    for name in group:
                        links[name] = group
        groups = []
        for name in names:
            if len(links[name]) > 1 and links[name] not in groups:
                groups.append(links[name])
        if not groups:
            return

        # A dimension for each disease, whose levels are its compartments of people
        for group in groups:
            members = [name for name in names if name in group] # In the order of the sim
            for name in members:
                mod = self.get_declared(name)
                engine = ss.solvers.Compartmental(mod, self) # Only to lay out the compartments; the module's own engine is made later
                engine.init_dims()
                engine.init_dwells()
                engine.build_layout()
                labels = []
                for ind in engine.people_comps: # A state with several compartments (branches, or the stages of a dwell time) has a level for each
                    state = engine.comps[ind].state.name
                    own = engine.state_comps[state]
                    labels.append(state if len(own) == 1 else f'{state}.{own.index(ind) + 1}')
                first = engine.graph.first.name
                tracks = [track for track in engine.tracks[first] if not track.immediate]
                if len(tracks) != 1:
                    raise ss.CapabilityError(f'the states of "{name}" cannot be tracked jointly with those of {sc.strjoin(n for n in members if n != name)} with solver="{self.solver}" yet, since people entering its first state ("{first}") are split between branches.\n'
                                             f'     Run with solver="abm".')
                split = np.zeros(len(labels))
                split[engine.people_comps.index(tracks[0].first)] = 1.0 # Where people born to (or arriving in) the others start: in this disease's first state, as agents do
                strat = ss.transitions.Strat(name, labels, split=split, kind='disease')
                strat.holders = [n for n in members if n != name] # The diseases whose people have this dimension
                strat.state_levels = {state.name:[engine.people_comps.index(ind) for ind in engine.state_comps[state.name]] for state in engine.graph.people_states}
                self.cross_strats.append(strat)
            self.crossings.append(ss.solvers.Crossing(self, members))
        self.init_strats()
        self.mixing = ss.solvers.Mixing(self)
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

    def groups_changed(self, uids=None):
        """
        Say that the level of some agents in a dimension has changed (or of any agent, if uids is None), so their groups are worked out again

        Code that changes a property of people that a dimension is made from (e.g. `people.patch`)
        calls this; see `group_raw()`.
        """
        self._group_cache = None
        for mod in getattr(self, 'declared', []): # The agent engines keep counts by group: see ss.solvers.Agents.regroup()
            if isinstance(mod.engine, ss.solvers.Agents):
                mod.engine.regroup(uids)
        dirty = getattr(self, '_group_dirty', None)
        if uids is None or dirty is False:
            self._group_dirty = False # Everyone
        else:
            self._group_dirty = (dirty or []) + [np.asarray(uids, dtype=np.int64)]
        return

    def group_raw(self):
        """
        In an agent-based run: the group of every agent, as an array indexed by UID

        It is kept until `groups_changed()` says it has changed, and then only the groups of the
        agents named are worked out again. A dimension binned from a property that changes on its
        own, such as age, is checked again on every step.
        """
        cache = getattr(self, '_group_cache', None)
        n = len(self.people.uid.raw)
        binned = any(strat.edges is not None for strat in self.strats)
        if cache is not None and (cache[0] == self.ti or not binned) and len(cache[1]) == n:
            return cache[1]
        dirty = getattr(self, '_group_dirty', None)
        self._group_dirty = None
        kept = getattr(self, '_level_cache', None)
        if isinstance(dirty, list) and not binned and kept is not None and '' in kept and len(kept['']) == n: # Only some agents have changed, and we know which: work out their groups alone
            uids = np.unique(np.concatenate(dirty))
            levels = []
            out = kept[''] # Changed in place: the groups are only ever used straight away, so nothing holds the old ones
            for strat,mult in zip(self.strats, np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:]):
                raw = self.people[strat.attr].raw
                old_raw, old_level = kept[strat.name]
                if len(old_raw) != n: # The arrays have grown since: work everything out
                    self._group_dirty = False
                    return self.group_raw()
                if len(uids):
                    before = old_level[uids]
                    after = np.clip(np.digitize(raw[uids], strat.edges) - 1, 0, strat.n - 1) if strat.edges is not None else ss.kernels.levels(raw[uids])
                    old_level[uids] = after
                    old_raw[uids] = raw[uids]
                    out[uids] += (after - before)*int(mult)
                levels.append(kept[strat.name][1])
            kept[''] = out
            self._group_cache = (self.ti, self.folded(out), levels)
            return self._group_cache[1]
        people = self.people
        kept = getattr(self, '_level_cache', None) # The values each level was worked out from: most properties change rarely, and checking is much cheaper than working the levels out again
        if kept is None:
            kept = self._level_cache = dict()
        levels = []
        changed = False
        def level_of(strat, raw): # As 32-bit integers, to save memory
            return (np.clip(np.digitize(raw, strat.edges) - 1, 0, strat.n - 1) if strat.edges is not None else ss.kernels.levels(raw)).astype(np.int32)
        mults = np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:]
        n = len(people.uid.raw)
        rebuild = '' not in kept or len(kept['']) != n # The last, for a sim with no dimensions whose people have grown
        updates = [] # For each dimension whose values changed for some agents: who, their old level, and their new one
        for strat,mult in zip(self.strats, mults):
            raw = people[strat.attr].raw
            old = kept.get(strat.name)
            if old is None or len(old[0]) != len(raw):
                old = kept[strat.name] = (raw.copy(), level_of(strat, raw))
                rebuild = True
            else:
                idx = np.flatnonzero(old[0] != raw) # Usually few: those born, or who moved, since it was last worked out
                if len(idx):
                    before = old[1][idx]
                    levels_new = old[1] # In place, as above
                    levels_new[idx] = level_of(strat, raw[idx])
                    old[0][idx] = raw[idx]
                    old = kept[strat.name] = (old[0], levels_new)
                    updates.append((idx, before, levels_new[idx], int(mult)))
            levels.append(old[1])
        if rebuild:
            out = np.zeros(n, dtype=np.int32)
            for level,mult in zip(levels, mults):
                out += level*int(mult)
            kept[''] = out
        elif updates:
            out = kept['']
            for idx,before,after,mult in updates:
                out[idx] += (after - before)*mult
            kept[''] = out
        self._group_cache = (self.ti, self.folded(kept['']), levels)
        return self._group_cache[1]

    def folded(self, groups):
        """ The groups that agents are counted in: see `init_ragged()` """
        return groups if self.group_fold is None else self.group_fold[groups]

    def level_raw(self, d):
        """ In an agent-based run: the level of every agent in one dimension (by its position), as an array indexed by UID """
        self.group_raw()
        return self._group_cache[2][d]

    def init_strat_vals(self):
        """ In an agent-based run: assign each agent a level of every property they do not already have """
        people = self.people
        uids = people.auids
        joint = self.joint[0] if self.joint is not None else []
        for d,strat in enumerate(self.strats):
            if strat.female is not None and d not in joint and strat.cond is None: # A sex dimension is people.female, whose share is the split: see init_people()
                self.set_sex(uids)
            elif strat.kind != 'age' and d not in joint and strat.cond is None: # Exactly the split, rounded to whole agents, in a random order
                rvs = getattr(self, f'rng_strat_{strat.name}').rvs(uids)
                counts = ss.solvers.Compartmental.round_preserving(strat.split*len(uids)).astype(int)
                levels = np.repeat(np.arange(strat.n), counts)
                people[strat.attr][uids[np.argsort(rvs, kind='stable')]] = levels
        if joint: # The levels of a joint split, together, given the age group if age is one of them
            ds, grid = self.joint
            ages = [i for i,d in enumerate(ds) if self.strats[d].kind == 'age']
            others = [i for i in range(len(ds)) if i not in ages]
            grid = np.moveaxis(grid, ages + others, list(range(len(ds)))).reshape(len(ages) and self.strats[ds[ages[0]]].n or 1, -1) # Rows: the age group (one row if no age); columns: the other levels together
            row = np.clip(np.digitize(people.age[uids], self.strats[ds[ages[0]]].edges) - 1, 0, grid.shape[0] - 1) if ages else np.zeros(len(uids), dtype=int)
            cdf = np.cumsum(grid, axis=1)
            cdf = cdf/np.maximum(cdf[:, -1:], 1e-300)
            rvs = getattr(self, f'rng_strat_{self.strats[ds[others[0]]].name}').rvs(uids)
            cell = np.minimum((cdf[row] <= rvs[:, None]).sum(axis=1), grid.shape[1] - 1)
            for i,level in zip(others, np.unravel_index(cell, [self.strats[ds[i]].n for i in others])):
                people[self.strats[ds[i]].attr][uids] = level
        for strat in self.strats: # A split given the levels of other dimensions, once those are known
            if strat.cond is not None:
                self.groups_changed()
                people[strat.attr][uids] = strat.draw(self, uids, getattr(self, f'rng_strat_{strat.name}').rvs(uids))
        if joint or any(strat.cond is not None for strat in self.strats): # A sex drawn with other dimensions sets people.female
            self.set_sex(uids, from_level=True)
        self.groups_changed()
        return

    def set_sex(self, uids, from_level=False):
        """ In an agent-based run with a dimension called sex: its level for some agents from people.female (or, if from_level, the other way round) """
        strat = next((strat for strat in self.strats if strat.female is not None), None)
        if strat is None:
            return
        people = self.people
        if from_level:
            people.female[uids] = people.sex[uids] == strat.female
        else:
            people.sex[uids] = np.where(people.female[uids], strat.female, 1 - strat.female)
        return

    def init_named_people(self):
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

    def init_capabilities(self, modules):
        """
        One record for each ingredient that matters to which solvers can run the model, read by `solver='auto'`, `sim.solvers()`, `sim.describe()`, and the `CapabilityError` of a compartmental solver that was asked for

        Each record has the ingredient; what a compartmental solver does with it (`groups`: 'yes', 'approx', 'no', or '?' for code
        that is tried on groups when it matters: see `try_capabilities()`), and how; the reason it makes `auto` run the model as
        agents (`auto`), if it does; and whether it stops a compartmental solver that was asked for (`fatal`). Setting
        `compartmental = True` or `False` on a module (or its class) says whether it works on groups, in place of trying it.
        """
        caps = []
        def add(ingredient, groups, how, auto=None, fatal=False, probe=None):
            caps.append(sc.objdict(ingredient=ingredient, groups=groups, how=how, auto=auto, fatal=fatal, probe=probe))
            return

        approx = self.pars.approx
        timed = (ss.Dist, ss.transitions.TimeValue) # Values that always work on groups, so need no trying
        if 'n_agents' in self.given:
            add('n_agents', 'yes', 'not used', auto='n_agents was given')
        if 'people' in self.given and not self.named_unsized:
            add('people', 'yes', 'counted into groups', auto='people were given')
        for mod in modules:
            graph = getattr(mod, 'graph', None)
            flag = getattr(mod, 'compartmental', None) # Whether it works on groups, if the module says
            reasons = mod.agent_reasons() if hasattr(mod, 'agent_reasons') else [] # What needs agents, which needs no trying
            for reason in reasons:
                add(f'module: {mod.name}', 'no', reason, auto=reason)
            if approx is False and isinstance(mod, ss.Delivery) and mod.holds and mod.product.staged: # Kept as levels of time since delivery otherwise: see ss.Held.init_stages()
                add(f'module: {mod.name}', 'approx', 'levels of time since delivery', auto=f'the protection of "{mod.product.name}" fades by a function of time, and approx=False')
            if graph is None:
                if isinstance(mod, ss.Network):
                    contacts = self.mixing.net_contacts(mod)[0]
                    add(f'network: {type(mod).__name__}', 'approx' if contacts is not None else 'no', 'contact rates' if contacts is not None else 'no approximation', auto=f'{type(mod).__name__} is an individual-level network')
                elif isinstance(mod, ss.MixingPool) and (mod.pars.src is not None or mod.pars.dst is not None): # A pool of some people, which has no compartmental form yet
                    add(f'network: {mod.name}', 'no', 'no compartmental form', auto=f'the mixing pool "{mod.name}" is between some people (src= or dst=), which only agents have', fatal=True)
                elif isinstance(mod, ss.MixingPool) and isinstance(mod.pars.n_contacts, ss.Dist) and not isinstance(mod.pars.n_contacts, ss.constant): # A number of contacts that differs between people
                    add(f'network: {mod.name}', 'approx', 'the mean number of contacts', auto=f'the mixing pool "{mod.name}" has a number of contacts that differs between people ({type(mod.pars.n_contacts).__name__})')
                elif isinstance(mod, ss.networks.Route): # A mixing pool, which works on groups
                    continue
                elif flag:
                    if hasattr(mod, 'try_groups') and not getattr(mod, 'timed', False) and not reasons: # Its conditions and functions, tried on groups
                        add(f'module: {mod.name}', '?', 'tried on groups', probe=(mod.name, 'try_groups'))
                elif flag is None and callable(getattr(mod, 'func', None)) and not isinstance(mod, ss.Demographics): # A function given as a module (e.g. interventions=func)
                    add(f'function: {mod.name}', '?', 'tried on groups', probe=(mod.name, 'step'))
                else:
                    if isinstance(mod, (ss.Births, ss.Deaths)) and flag is None:
                        why = f'"{mod.name}" ({type(mod).__name__}) works on agents; ss.birth() and ss.death() in the disease are its compartmental form'
                    else:
                        why = f'"{mod.name}" ({type(mod).__name__}) works on agents' + ('' if flag is False else '; if its code works on groups, set compartmental = True on it')
                    add(f'module: {mod.name}', 'no', 'works on agents', auto=why, fatal=True)
                continue

            # A declared module: code of its own, functions, and what needs agents
            own = [name for name in ['step_state', 'step'] if getattr(type(mod), name) is not getattr(ss.Disease, name)]
            if flag is False:
                add(f'module: {mod.name}', 'no', 'compartmental = False', auto=f'"{mod.name}" has compartmental = False', fatal=True)
            elif own and flag is None:
                add(f'code: {mod.name}.{own[0]}()', '?', 'tried on groups', probe=(mod.name, own))
            for i,p in enumerate(graph.all_pars()):
                raw = p.raw
                if callable(raw) and not isinstance(raw, ss.Dist):
                    name = getattr(raw, '__name__', 'function')
                    if len(inspect.signature(raw).parameters) >= 2:
                        add(f'function: {name}', 'no', 'takes uids', auto=f'the function "{name}" takes uids')
                    elif not isinstance(raw, timed):
                        add(f'function: {name}', '?', 'tried on groups', probe=(mod.name, i))
            for name,func in graph.funcs.items(): # Custom per-step behavior
                add(f'function: {name}', '?', 'tried on groups', probe=(mod.name, func))
            for p in graph.spread_pars():
                add(f'distribution: {p.name}', 'approx', 'its mean', auto=f'"{p.name}" is a distribution across people')
            aged = any(strat.kind == 'age' and strat.edges is not None for strat in list(graph.strats) + list(self.strats))
            for name,tr in graph.transitions.items():
                raw = None if tr.dur is None else tr.dur.raw
                if isinstance(raw, ss.Dist) and type(raw).__name__ != 'expon':
                    shape = raw.pars.get('a')
                    exact = tr.stages is not None or (type(raw).__name__ == 'gamma' and sc.isnumber(shape) and shape == int(shape))
                    auto = f'"{name}" has a {type(raw).__name__} dwell time, and approx=False' if approx is False and tr.stages is None and type(raw).__name__ not in ['expon', 'gamma'] else None
                    add(f'dwell time: {name} ({type(raw).__name__})', 'yes' if exact else 'approx', 'Erlang stages', auto=auto)
                if getattr(tr, 'after', None) is not None or getattr(tr, 'before', None) is not None:
                    add(f'transition: {name}', 'no', 'needs the time each person entered its source', auto=f'"{name}" depends on how long people have been in its source state (after= or before=), which needs the time each person entered it')
                if getattr(tr, 'offspring', None) is not None:
                    add(f'transition: {name}', 'no', 'adds offspring for each person', auto=f'"{name}" has offspring (offspring=), which are added for each person who goes through it')
                if getattr(tr, 'lifetable', None) is not None and not aged:
                    add(f'transition: {name}', 'no', 'needs each person\'s age', auto=f'"{name}" draws ages at death from a life table, which needs each person\'s age (or a stratification by age)')
            for strat in list(graph.strats) + list(self.strats):
                immunity = getattr(strat, 'immunity', None)
                if immunity is not None and not np.all(immunity == 1): # A compartmental run keeps only the last strain each person has had: see ss.solvers.Engine.init_immunity()
                    add(f'immunity: {strat.name}', 'approx', 'keeps the last strain', auto=f'the immunity between the strains of "{strat.name}" is partial, and only agents keep every strain each person has had')
                    break
        self.capabilities = caps
        return caps

    def auto_reasons(self):
        """ The reasons, from the capability record, that `solver='auto'` runs the model as agents (none if it runs it as compartments) """
        return [cap.auto for cap in getattr(self, 'capabilities', []) if cap.auto]

    def try_capabilities(self):
        """ Try each function and piece of code in the model that is not known to work on groups, on groups, once, on a throwaway sim, and fill in its record (see `init_capabilities()`) """
        rows = [cap for cap in self.capabilities if cap.groups == '?']
        if not rows or not hasattr(self, 'inputs'):
            return
        try:
            probe = self.new(solver='ode', verbose=0, approx=True).init()
            for mod in probe.declared:
                mod.engine.start_step()
        except ss.CodedError as E:
            for cap in rows:
                cap.groups, cap.how = 'no', 'the model does not run as compartments'
            rows[0].auto = f'the model does not run as compartments ({str(E).splitlines()[0]})'
            return
        except Exception as E:
            raise self.tried_error('the model', E) from E
        mods = {mod.name:mod for mod in probe.modules}
        for cap in rows:
            modname, what = cap.probe
            mod = mods[modname]
            name = cap.ingredient.split(': ', 1)[-1]
            try:
                if isinstance(what, int): # A function in a value
                    mod.engine.value(mod.graph.all_pars()[what])
                elif callable(what): # Custom per-step behavior, called with the sim
                    what(probe)
                elif what == 'try_groups':
                    mod.try_groups()
                else: # A step of its own, or a function given as a module
                    for method in sc.tolist(what):
                        getattr(mod, method)()
            except (ss.CodedError, AttributeError) as E: # It asked for something that only agents have
                why = str(E).splitlines()[0].strip()
                cap.groups, cap.how = 'no', why
                cap.auto = f'"{name}" does not work on groups ({why})'
            except Exception as E: # A mistake in the code, which agents would not fix
                raise self.tried_error(f'"{name}"', E) from E
            else:
                cap.groups = 'yes'
        return

    def count_events(self):
        """ The number of events that solver="gillespie" would take, from an ODE run: people moved along each transition, once for each stage of its dwell time """
        if not hasattr(self, 'inputs'):
            return 0
        ode = self.new(solver='ode', verbose=0, approx=True).run()
        n = 0
        for mod in ode.declared:
            dwells = mod.engine.dwells
            for name,tr in mod.graph.transitions.items():
                if tr.people:
                    k = dwells[name].k if name in dwells else 1
                    n += k*np.sum(mod.results[name].values)
        return n

    @staticmethod
    def tried_error(what, E):
        """ The error for a function that failed when it was tried on groups for a reason other than needing agents """
        errormsg = (f'{what} raised {type(E).__name__} ({E}) when it was tried on groups, to see whether the model can run as compartments.\n'
                    f'     If it is a mistake, fix it; if it needs values that only agents have, run with solver="abm".')
        return ValueError(errormsg)

    def init_unit(self):
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
        units = [u.removesuffix('s') for mod in self.declared for u in mod.graph.unit_values()]
        units += [base(mod.t.dt) for mod in self.declared if is_dur(mod.t.dt)]
        if pars.unit is not None:
            unit = pars.unit.removesuffix('s')
        elif units:
            order = ['day', 'week', 'month', 'year']
            unit = sorted(set(units), key=order.index)[0]
            source = 'the typed values in the model'
            if len(set(units)) > 1:
                self.units_disagree = sorted(set(units), key=order.index)
                bare = [f'{mod.name}.{p.name}' if not p.literal else f'{p.raw} in {mod.name}' for mod in self.declared for p in mod.graph.all_pars() if p.slot in ['rate', 'dur'] and sc.isnumber(p.raw)]
                if bare: # Which unit they are in cannot be guessed
                    raise ss.IncompatibleTypes(f'the typed values in the model have different units ({sc.strjoin(self.units_disagree)}), so the unit of the bare numbers {sc.strjoin(bare)} is not known. Set it with unit=, e.g. ss.Sim(..., unit=\'{unit}\').')
        elif dates:
            unit, source = 'day', 'start and stop dates'
        elif is_dur(pars.dt) or isinstance(pars.dt, str):
            unit, source = (base(pars.dt) if is_dur(pars.dt) else pars.dt.removesuffix('s')), 'dt'
        elif durs:
            unit, source = base(durs[0]), 'start and stop'
        elif years:
            unit, source = 'year', 'start and stop years'
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
            guessed = pars.solver == 'auto' or self.solver != pars.solver # Chosen by auto, or tau-leaping for gillespie
            add('solver', self.solver, f'guessed: {self.solver_reason}' if guessed else 'requested')
            for cap in self.capabilities: # What matters to which solvers can run it: see sim.solvers()
                add('', f'{cap.ingredient}: {"not tried" if cap.groups == "?" else cap.groups} on groups{"" if cap.groups == "?" else f" ({cap.how})"}')
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
            for strat in self.strats: # Each dimension: its levels, the states it applies to, and what it generates
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
                guesses = list(strat.guessed.values())
                if strat.kind == 'age' and 'use_aging' not in self.given:
                    guesses.append('aging, since people are born or die' if pars.use_aging else 'no aging, since nobody is born or dies')
                add('dimension', text, f'guessed: {"; ".join(guesses)}' if guesses else None)
            for crossing in self.crossings:
                sizes = [self.strats[self.dim_index(name)].n for name in crossing.names]
                add('jointly', f'{" × ".join(crossing.names)}: {" × ".join(str(n) for n in sizes)} = {int(np.prod(sizes))} compartments of people in each group', 'a condition in one refers to a state of another')
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
        if self.declared:
            self.try_capabilities() # Anything not tried yet, when a solver was asked for
        rows = []
        for cap in self.capabilities: # See init_capabilities()
            how = cap.groups if cap.groups == '?' else f'{cap.groups} ({cap.how})'
            rows.append([cap.ingredient, how, how, 'yes'])
        header = ['', 'ode', 'binomial, gillespie', 'abm']
        table = [header] + rows
        widths = [max(len(row[i]) for row in table) + 3 for i in range(4)]
        lines = [''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in table]
        if not rows:
            lines.append('(every part of the model runs under every solver)')
        reasons = self.auto_reasons() if self.declared else ['the model is written as code']
        if self.pars.solver == 'auto' and self.declared: # What it did
            choice = f'{self.solver} ({self.solver_reason})'
        else:
            choice = f'abm ({reasons[0]})' if reasons else 'ode (every part of the model has a compartmental form)'
        lines += ['', f'auto would choose: {choice}']
        text = '\n'.join(lines)
        return text if output else print(text)

    def requires(self, output=False):
        """
        Print the data the modules in the sim require, and which items are still missing

        Each module's `requires` list (of `ss.Requires`) is shown; for a declared model, the
        list is its parameters. An item is given if the module has a parameter or attribute
        of that name that is not None, and otherwise uses its default or is missing.

        Args:
            output (bool): if True, return a dataframe rather than printing the table
        """
        if self.initialized:
            modules = list(self.modules)
        else: # Without initializing, since a missing item can stop the sim from initializing
            pars = sc.dcp(self.pars)
            pars.validate()
            modules = [mod for key in ss.modules.module_types() + ['custom'] for mod in pars[key].values()]
        rows = []
        for mod in modules:
            graph = getattr(mod, 'graph', None)
            if graph is not None: # A declared model requires its parameters
                for p in graph.all_pars():
                    if p.literal:
                        continue
                    raw = p.raw
                    shape = ' x '.join(raw.dims) if isinstance(raw, ss.by) else ''
                    status = 'missing' if raw is None else 'given'
                    rows.append(dict(module=mod.name, name=p.name, shape=shape, source=p.source or '', status=status, doc=p.doc or ''))
            for req in sc.tolist(mod.requires):
                if isinstance(req, ss.Requires):
                    pars = getattr(mod, 'pars', {})
                    val = pars[req.name] if req.name in pars else getattr(mod, req.name, None)
                    status = 'given' if val is not None else 'default' if req.default is not None else 'missing'
                    rows.append(dict(module=mod.name, name=req.name, shape=req.shape or '', source=req.source or '', status=status, doc=req.doc or ''))
                else: # Another module, by class or name, as in Starsim 3
                    name = req if isinstance(req, str) else req.__name__
                    found = any(m.name == req or type(m) is req for m in modules)
                    rows.append(dict(module=mod.name, name=name, shape='module', source='', status='given' if found else 'missing', doc=''))
        df = pd.DataFrame(rows, columns=['module', 'name', 'shape', 'source', 'status', 'doc'])
        if output:
            return df
        if not len(df):
            print('The modules in this sim list no data that they require')
            return
        n_missing = (df.status == 'missing').sum()
        print(df.to_string(index=False))
        print(f'\n{n_missing} of {len(df)} item{"s" if len(df) != 1 else ""} missing' if n_missing else f'\nNothing is missing ({len(df)} item{"s" if len(df) != 1 else ""})')
        return

    def R0(self):
        """
        The basic reproduction number of each declared disease, with the sim's mixing and stratification: see `ss.Disease.R0()`

        Returns a number if the sim has one declared disease with a transmission, and otherwise a dict by disease.
        """
        sim = self.new(solver='ode', verbose=0).init()
        out = sc.objdict()
        for dis in sim.diseases():
            ngm = dis.engine.next_generation() if dis.graph is not None else None
            if ngm is not None:
                out[dis.name] = float(np.max(np.abs(np.linalg.eigvals(ngm))))
        return out[0] if len(out) == 1 else out

    def hash(self):
        """
        The run's identity: short hashes of the model's structure, its parameter values, the run settings, the scenario, and the data, plus the seed

        Two sims with the same `model` hash have the same states, transitions, and
        modifiers (including the source of any function in them); the same `pars` hash,
        the same values. `config` covers the run settings and the numerical options
        (e.g. `ss.options.crn`), `scenario` the interventions, and `data` the data
        attached for fitting. This tells apart the many runs that come from one script
        (a sweep, a calibration), and tests whether two models written differently are
        the same model. A sim that is not initialized is hashed without initializing it.
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
            dis = self.diseases.get(mod.get('name')) if mod.get('init_prev') else None
            if dis is not None and dis.graph is not None and len(dis.graph.transmissions[0].dst) == 1: # init_prev=0.01 is init=dict(I=0.01), the same model
                state = next(s for s in mod['states'] if s['name'] == dis.graph.transmissions[0].dst[0].name)
                if state.get('init') is None:
                    state['init'], mod['init_prev'] = mod['init_prev'], None
        structure, values = ss.utils.split_numbers(model, unit=self.unit, skip=['guessed', 'label', 'doc'])
        config = {k:v for k,v in spec['config'].items() if k != 'rand_seed'}
        data = None if self.data is None else digest(self.data.to_csv())
        return sc.objdict(model=digest(structure), pars=digest(values), config=digest(config), scenario=digest(spec['interventions']), data=data, seed=self.pars.rand_seed)

    def to_spec(self, filename=None, **kwargs):
        """
        The model spec: everything needed to rebuild the model (modules, networks, interventions, and run settings), as plain data

        A sim that is not initialized is described without initializing it, as for `sim.hash()`.

        Args:
            filename (str): if provided, save to this file
            kwargs (dict): passed to `sc.savejson()`
        """
        if not self.initialized: # A copy, so that this sim is not changed
            return sc.dcp(self).init().to_spec(filename=filename, **kwargs)
        pars = self.pars
        def parspec(val):
            """ A parameter of a module that is not declared, as plain data: typed values and distributions as in the rest of the spec """
            if isinstance(val, ss.par):
                val = val.raw
            if isinstance(val, ss.Dist):
                return dict(dist=type(val).__name__) | {k:parspec(v) for k,v in val.pars.items()}
            elif isinstance(val, (ss.dur, ss.Rate)):
                return ss.transitions.quantity_json(val, None, None)
            elif isinstance(val, dict):
                return {k:parspec(v) for k,v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [parspec(v) for v in val]
            return sc.jsonify(val)
        def modspec(mod):
            if getattr(mod, 'graph', None) is not None:
                return mod.to_spec()
            if isinstance(mod, (ss.modify, ss.Delivery)):
                return mod.to_json()
            out = sc.objdict(type=type(mod).__name__, name=mod.name, label=mod.label)
            ref = getattr(mod, 'ref', None)
            if ref is not None: # Named data, e.g. ss.contacts('Kenya'): where it came from, and the parameters that are single values (the rest come from the data)
                out.pars = {k:parspec(v) for k,v in mod.pars.items() if v is None or isinstance(v, (str, bool, int, float, ss.dur, ss.Rate, ss.Dist))}
                out.data = sc.jsonify(ref)
            else:
                out.pars = {k:parspec(v) for k,v in mod.pars.items()}
            return out
        def tstr(val):
            if isinstance(val, ss.dur):
                return float(val.value)
            return val.strftime('%Y-%m-%d') if isinstance(val, ss.date) and val == val.normalize() else str(val) # Dates as ISO 8601
        out = sc.objdict()
        out.schema = 'lingua-franca/1.0'
        out.modules = [modspec(mod) for key in ['custom', 'demographics', 'diseases'] for mod in self[key]() if not isinstance(mod, ss.demographics.Flows)] # Births and deaths declared for the population are, by now, transitions of a disease
        out.networks = [modspec(mod) for mod in list(self.networks()) + getattr(self, 'mixed', [])]
        out.stratify = [sc.dcp(strat.spec) for strat in getattr(self, 'sim_strats', [])] # Those of a module are in its own spec, and those of an intervention are part of the scenario
        out.modifiers = [modspec(mod) for mod in self.connectors()]
        out.interventions = [modspec(mod) for mod in self.interventions()]
        out.config = sc.objdict(solver=pars.solver, integrator=pars.integrator, approx=pars.approx, unit=self.unit or pars.unit, # A sim of no declared modules has only the unit it was given
                                start=tstr(self.t.start), stop=tstr(self.t.stop), dt=float(self.t.dt.to(self.unit_base).value),
                                total_pop=float(pars.total_pop), n_agents=int(pars.n_agents) if self.solver == 'abm' else None, rand_seed=pars.rand_seed)
        out.config.crn = ss.options.crn # Numerical options that change the results
        out.config.precision = ss.options.precision
        if pars.substeps is not None: # Only when given, so that the hash of every other sim is as it was
            out.config.substeps = pars.substeps
        if 'use_aging' in self.given: # Likewise; e.g. an age stratification without aging
            out.config.use_aging = bool(pars.use_aging)
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

        A sim imported from another framework (`ss.load(..., format=...)`) also says
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
            for modkey, modclass in ss.module_map().items():
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
        if (self.named_people is not None or self.given_pyramid is not None) and self.solver != 'abm': # A compartmental run takes the size and age split of a population, not its agents
            people = None
        n_agents = self.pars.n_agents
        verbose = sc.ifelse(verbose, self.pars.verbose)
        if verbose > 0 and self.solver == 'abm':
            labelstr = f' "{self.label}"' if self.label else ''
            print(f'Initializing sim{labelstr} with {n_agents:0n} agents')

        # If people have not been supplied, make them -- typical use case
        if people is None:
            if self.solver == 'abm' and self.strats: # Properties that people are stratified by, other than age
                extra = [ss.FloatArr(strat.attr, default=0, label=strat.name) for strat in self.strats if strat.kind != 'age']
                kwargs['extra_states'] = sc.mergelists(kwargs.get('extra_states'), extra)
                for strat in self.strats: # Ages follow the split of the age groups, so every solver starts from the same population
                    if strat.kind == 'age' and strat.edges is not None and 'age_data' not in kwargs and (self.pyramid(strat) is None or strat.edges[0] > 0): # (Nobody is younger than the first group)
                        top = max(100, strat.edges[-1] + 10)
                        kwargs['age_data'] = np.column_stack([np.append(strat.edges, top), np.append(strat.split, 0)])
                    elif strat.female is not None and 'p_female' not in kwargs: # A sex dimension is people.female, so its split is the share of women
                        kwargs['p_female'] = strat.split[strat.female]
            if self.declared and 'age_data' not in kwargs: # Otherwise the ages of a declared model are the world's, as a guess (as is the split of age groups that are not given one)
                kwargs['age_data'] = ss.named.age_data(*self.pyramid())
            people = ss.People(n_agents=n_agents, **kwargs)  # This just assigns UIDs and length
        elif self.solver == 'abm':
            for strat in self.strats:
                if strat.attr not in people.states:
                    state = ss.FloatArr(strat.attr, default=0, label=strat.name)
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
            if disease.graph is None and not hasattr(disease, 'ti_infected'):
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
        self.data = ss.validate_sim_data(data)
        if self.data is not None and getattr(self, 'strats', None):
            self.data = ss.data.widen(self, self.data)
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
                    f'its results leave them out. {imp.prompt}', ss.ImportIncomplete)
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
        it makes repeated runs many times faster. `sim.run(n_runs=...)` and `ss.fit()`
        use it where they can.

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            rand_seed (int): the seed for the next run

        Example:
            ```python
            sir = ss.Disease(ss.transmission('S -> I', beta=0.3), ss.progression('I -> R', dur=8), init=dict(I=0.01))
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
            self.par_values = sc.mergedicts(self.par_values, pars)
            self.set_par_values()
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

        # Figure out the flat structure of results to plot
        flat = ss.utils.match_result_keys(self.results, key, show_skipped=(show_skipped or key)) # Always show skipped with a custom key

        # Annualize results if requested
        if annualize:
            flat = sc.objdict({k: res.resample('year') if isinstance(res, ss.Result) else res for k, res in flat.items()})

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
                        x = df.index.values
                        x = ss.DateArray(x) if len(x) and isinstance(x[0], ss.date) else x # As the results' time vector is, so they share an axis
                        ax.scatter(x, df[dfkey].values, **kw.data)

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
