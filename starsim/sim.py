"""
The sim (`ss.Sim`)

This is what a user runs. It exists to gather the people, modules, and settings of a model in one object; to choose how to run it (the solver); and to hold the results. It works the same way for every kind of model.

Used with: agents and compartments; declared and coded modules
"""
import os
import time
import fnmatch
import itertools
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


class Sim(ss.modules.Base):
    """
    The sim: a model, how to run it, and its results

    Every Starsim model, whatever its kind, is run with a sim. It gathers the people and the
    modules (`ss.Disease`, `ss.Network`, `ss.Intervention`, `ss.Demographics`, and others),
    runs them step by step (see `ss.loop.Loop`), or all at once in the compiled kernels when
    nothing needs to happen between the steps, and holds the results.

    Args:
        model (Module/list/dict): the model: a module or a list of modules, which are sorted by kind as for `modules`; transitions, alone or in a list (or in `diseases=`), are one disease, named from its states (e.g. `sim.sir`).
            Alternatively, the sim's settings as an `ss.SimPars` or a dict, which can include any of the other arguments
        pars (dict/SimPars): the values of the model's parameters by name, e.g. `pars=dict(beta=0.1)` or `pars=fit.pars` (see `ss.par`). As in Starsim 3, it can also hold settings of the sim (e.g. `pars=dict(start=2000, dur=10, n_agents=1000, beta=0.1)`): a name is a setting if no parameter of the model has it, and a keyword given directly wins over the same setting here. A whole `ss.SimPars` is taken as if given in place of the model. A name that is neither raises a `KeyError` when the sim is initialized
        label (str): a name for the sim, shown in plots and printouts
        people (People): the people to use (by default, the sim makes them)
        demographics (str/Demographics/list/bool): births, deaths, and migration: declared (`ss.birth()`, `ss.death()`, which run under every solver), or modules (e.g. `ss.Births()`, `ss.Pregnancy()`), or a list; True for births and deaths at default rates
        connectors (str/Connector/list): as above, for connectors
        networks (str/Network/list): as above, for networks
        interventions (str/Intervention/list): as above, for interventions
        diseases (str/Disease/list): as above, for diseases
        analyzers (str/Analyzer/list): as above, for analyzers
        modules (Module/list): modules of any kind, sorted among demographics, diseases, etc.; a module of your own that is none of these (an `ss.Module`, or a function of the sim, which becomes its `step()`) is kept in `sim.custom`
        solver (str/Solver): how a declared model is run: 'auto' (the default: chosen from the model; `sim.solver_reason` says why), 'ode' (deterministic and fastest), 'binomial' (stochastic compartments, a chain binomial), 'gillespie' (stochastic and exact, event by event), 'tau' (tau-leaping, a faster approximation to Gillespie), or 'abm' (agents, for detail about individuals). A solver can take options, e.g. `ss.solvers.ode(integrator=, substeps=, dwells=)` (`dwells='exact'` keeps each dwell time as it is, rather than as a chain of stages), `ss.solvers.binomial(substeps=)`, or `ss.solvers.abm(step=)` (`step='events'` for agents in continuous time). A declared module can have its own way of running as agents, `ss.Disease(..., solver=ss.solvers.abm(step='wholestep'))`, which the sim's does not override. `sim.solver` is the one that runs, e.g. `isinstance(sim.solver, ss.solvers.abm)`
        branching (bool): run the declared model as a branching process, under any solver: infection takes no one from the uninfected and gives no one back, so the susceptible side stays as it would with no infection. This is useful for the early growth or extinction of an outbreak, and for the size of chains of transmission; as agents, only the infected are made, so a large population costs little. See `sim.branching` for the size and outcome of each run, and `sim.run(until=)` to stop one
        unit (str): the unit of bare numbers in declared transitions, e.g. 'day'; if nothing in the model has a unit, time is in timesteps
        approximations (str): what to do when a solver has to approximate part of the model (e.g. a dwell time run as stages): 'note' (the default) says so in one line (kept in `sim.notes`, and printed unless `verbose=0`); 'allow' keeps the note without printing it; and 'error' raises an error, for when the results must be exact
        stratify (dict): stratify every module by properties of people, e.g. `dict(age=[0, 15, 65], risk={'low': 0.8, 'high': 0.2})`, as `sim.stratify()` takes them
        data (df): data to compare the results with: a dataframe (or dict, or the name of a file) with a column "time" and columns named "module.result", e.g. "hiv.infection"; it is plotted with the results, and is what `sim.fit()`, `sim.log_density()`, and `sim.gradient()` fit to
        copy_modules (bool): by default, the sim runs a copy of each module it is given, so the same modules can be given to several sims (read the ones that ran from the sim, e.g. `sim.sir`); if False, it runs the modules themselves, which saves a copy (and lets you read them directly after the run), but then the sim can't be run again or copied with `sim.new()`, and the modules can't be given to another sim
        n_agents (int/float): the number of agents, if the model runs as agents (default 10,000); more agents give less noisy results but run slower. Not for a compartmental run, which has no agents (give total_pop=)
        total_pop (int/float): the number of people (default: one per agent); as agents, each stands for `total_pop/n_agents` of them (`sim.pop_scale`), and the results that count people are scaled up by it, so a large population can be modeled with fewer agents
        rescale (bool/dict): dynamic rescaling of agents, as in Covasim (default False): each agent starts standing for one person, and the number grows to `sim.pop_scale` as the agents are infected, with some made naive again each time. This keeps the detail of a small outbreak in a large population without an agent per person. True, or a dict of its options, `dict(threshold=0.05, factor=1.2)` (the defaults): the share of agents who are not naive that triggers rescaling, and the least factor by which the scale grows each time
        start (float/str/date): when the sim starts: a date (e.g. '2020-01-01') or a year (a number from 1800 on, e.g. 2000), which give it a calendar; any other number (or a duration) is a time in the sim's unit from 0 (default: 2000 for a model in years, else 0). With a calendar, every bare number given as a time (stop=, when=, the times of ss.schedule and ss.timeseries) is a year; without one, a time in the sim's unit from 0. A duration given as a time (when=ss.days(10), until=, the start= of a module) is that long after the start with a calendar, else a time from 0 (day 10; the same when the start is 0)
        stop (float/str/date/dur): when the sim ends, read as the start is: a date, a bare number (a year, or with no calendar a time in the sim's unit from 0), or a duration from 0 (e.g. stop=ss.days(100), with no start; ss.years(2030) in a sim of calendar years is the year 2030); with no start, a year starts the sim in 2000, unless unit= is other than years (unit='day', stop=3650 is 3,650 days from 0; default: a run of 100 days in days, else 50 of the sim's unit)
        dur (float/dur): the length of the run from its start, e.g. `start=2000, dur=10` (a bare number is in years in a sim of calendar years, as a bare stop is, else in the sim's unit) or `start='2020-01-01', dur=ss.days(90)`; any two of the start, stop, and length define the third (`stop=2035, dur=10` starts in 2025), and all three must agree
        dt (float/dur): the timestep, in the sim's unit (default 1.0); a smaller step is more accurate but slower (`sim.check_dt()` shows how much the results depend on it)
        rand_seed (int): the random seed, from which every module's random numbers are drawn; the same seed gives the same results, and different seeds give different runs of a stochastic model
        even_draws (bool): for agents, space each call's random numbers evenly, so that close to the expected number of agents move each time (default False). This makes runs much less variable (about 3 to 10 times less in the final size of a simple SIR), which helps when comparing scenarios with few runs; but outbreaks die out less often than they should, and the mean of a result can shift (e.g. the peak of an SIR on a network, by about 10%). Agents only, and needs `ss.options.crn`
        keep_times (bool): for a declared disease run as agents, keep the time each agent entered every state (e.g. `ti_S`) and the history of every transition (e.g. `sir.infection.n_times`), for your own code to read after the run, e.g. in an analyzer. By default, only those the model needs are kept, which saves memory and time (default: `ss.options.keep_times`, False)
        aging (bool): whether agents age (by default, they age if there are births or deaths)
        record_people (bool): whether to record the results about people (alive, female, died, emigrated, etc.) every step (default True); False saves a little time in a sim that doesn't need them, but some modules, e.g. demographics, rely on `results.alive`
        step_order (str): the order in which the modules step: 'type' (default) at the place of each kind of module in the loop (see `ss.Loop`), or 'given' in the order they were given (the model and `modules=` first, then the modules given by kind, e.g. `diseases=`), before the deaths and results of the step; 'given' is for when one module must act on what another did earlier in the same step. Analyzers, which read the step's results, still step last
        require_kernels (bool): raise an error (`ss.CapabilityError`) if a compartmental sim can't be run all at once in the compiled kernels, and would instead run step by step in plain Python, which can be 100 times slower; useful for making sure a fit or a large batch of runs stays fast (`sim.describe()` says why a sim can't use them)
        group_results (list): the results to keep by group (e.g. by age), by name, e.g. `['I', 'cases']` or `['sir.I']`; the others keep only their totals, which saves memory in a sim with many groups (default: all, at every few steps if every step would take more than about 100 MB)
        verbose (int/float): how much to print: 0 nothing, 1 the solver, notes, and progress (default), 2 everything (e.g. each step); a fraction is as 1, with progress every that share of the run (e.g. 0.1 for every 10%)
        kwargs (dict): the settings above, by name (they are kept in `sim.pars`, an `ss.SimPars`)

    Modules can be given by kind (`diseases=`, `networks=`, etc.), all together with `modules=`, or both: they are sorted by kind and merged.

    Examples:
        ```python
        S, I, R = ss.states(3)
        sim = ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01)) # A declared model; here run as an ODE, or give solver= (e.g. solver='abm')
        sim = ss.Sim(diseases='sir', networks='random') # Agents on a network; equivalent to ss.demo()
        sim = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()) # The same, with objects instead of strings
        sim = ss.Sim(diseases=['sir', ss.SIS()], networks=['random', 'degree']) # Lists, mixing strings and objects
        sim = ss.Sim(modules=[ss.SIR(), ss.RandomNet()]) # Modules of several kinds, sorted by kind
        ```
    """
    def __init_subclass__(cls, **kwargs):
        """ A subclass whose init() builds modules of its own before Sim.init() (as STIsim's does): the sim is copied before it runs (see `outermost()`) """
        super().__init_subclass__(**kwargs)
        if 'init' in cls.__dict__:
            cls.init = ss.utils.outermost(cls.__dict__['init'])
        return

    def __init__(self, *models, label=None, people=None, demographics=None, connectors=None,
                 networks=None, diseases=None, interventions=None, analyzers=None,
                 modules=None, data=None, pars=None, copy_modules=True, **kwargs):
        self.pars = ss.SimPars() # Make default parameters
        models = [arg for arg in models if arg is not None]
        kinds = ss.modules.module_types() + ['modules']
        if isinstance(pars, ss.SimPars): # The sim's settings as a whole, e.g. ss.Sim(pars=sim.pars), as if given in place of the model
            models, pars = models + [pars], None
        elif isinstance(pars, dict) and any(key in kinds for key in pars): # Modules in pars=, as if given in a dict in place of the model
            models, pars = models + [{key:val for key,val in pars.items() if key in kinds}], {key:val for key,val in pars.items() if key not in kinds}
        given = dict(label=label, people=people, demographics=demographics, connectors=connectors, networks=networks, diseases=diseases,
                     interventions=interventions, analyzers=analyzers, modules=modules, data=data)
        for arg in models: # The model: modules or transitions, or lists of them; or the sim parameters, as a dict
            if isinstance(arg, dict):
                kwargs = sc.mergedicts({k:v for k,v in arg.items() if given.get(k) is None}, kwargs) # A keyword given outright wins over the same key in the dict
            elif not all(isinstance(obj, ss.Module) or ss.transitions.is_declaration(obj) for obj in (arg if isinstance(arg, (list, tuple)) else [arg])):
                errormsg = f'ss.Sim() takes the model (modules or transitions, or lists of them) or a dict of parameters as positional arguments, not {type(arg).__name__}; give everything else by keyword, e.g. label="..."'
                raise TypeError(errormsg)
        mods = dict(demographics=demographics, connectors=connectors, networks=networks, diseases=diseases, interventions=interventions, analyzers=analyzers, modules=modules)
        if any(val is None and key in kwargs for key,val in mods.items()): # Modules given in a dict of parameters are named and ordered as if given by keyword
            mods = {key:(kwargs.pop(key) if val is None and key in kwargs else val) for key,val in mods.items()}
            demographics, connectors, networks, diseases, interventions, analyzers, modules = mods.values()
        names = ss.utils.caller_names(1)
        models = ss.utils.make_disease(models, names)
        diseases = ss.utils.make_disease(diseases, names)
        kinds = dict(demographics=demographics, networks=networks, diseases=diseases, connectors=connectors, interventions=interventions, analyzers=analyzers)
        for mod in sc.mergelists(*[arg for arg in models if not isinstance(arg, dict)]): # The model's modules, sorted into their kinds now (after those given by keyword, as init() would), so that e.g. sim.pars.diseases has them before init() as after
            key = next((key for key,cls in ss.modules.module_map().items() if isinstance(mod, cls)), 'custom')
            if key not in kinds or isinstance(kinds[key], (dict, bool)): # Modules of no standard kind (sim.custom), or by name, e.g. networks=dict(main=...), or demographics=True: sorted at init() (see ss.SimPars.validate_modules())
                modules = sc.mergelists(mod, modules)
            else:
                kinds[key] = sc.mergelists(kinds[key], mod)
        demographics, networks, diseases, connectors, interventions, analyzers = kinds.values()
        self._name_modules(names, demographics, connectors, networks, interventions, diseases, analyzers, modules)
        interventions = self._order_steps(interventions)
        self.par_values = None # Values of model parameters by name, e.g. pars=dict(beta=0.1) or pars=fit.pars
        self.jax = None # Whether a whole run of the ODE is taken in JAX (see ss.gradients): True to ask for it; otherwise the kernels take a run (see ss.solvers.choose_engine()), and JAX is used only where it is needed (gradients, and ss.parallel() for many runs with different values, unless False)
        if isinstance(pars, dict) and len(pars): # Values of the model's parameters by name; a setting of the sim given as ss.par too (e.g. n_agents=ss.par(...), when it is fitted: pars=fit.pars)
            pars = dict(pars)
            kwargs |= {key:pars.pop(key) for key in list(pars) if isinstance(kwargs.get(key), ss.par)}
            mods = ss.utils.given_modules(dict(enumerate(models)), dict(demographics=demographics, connectors=connectors, networks=networks, diseases=diseases, interventions=interventions, analyzers=analyzers, modules=modules))
            names = {name for mod in mods for name in (mod.pars.keys() if getattr(mod, 'graph', None) is not None else mod.pars.bounded())} # Those pars= can set (see ss.solvers.set_values())
            settings = {key:pars.pop(key) for key in list(pars) if (key in self.pars or key in ss.v3_deprecations.keys('Sim')) and key not in names} # A setting of the sim by its own name, if no parameter of the model has it (so pars=fit.pars works with a parameter called dur, and pars=dict(dur=10) with a network that has one)
            kwargs = sc.mergedicts({k:v for k,v in settings.items() if given.get(k) is None}, kwargs) # A keyword given outright wins
            self.par_values = pars or None # A name that is neither raises a KeyError when the sim is initialized
        pars = None
        args = dict(label=label, people=people, demographics=demographics, connectors=connectors,
                    networks=networks, interventions=interventions, diseases=diseases, 
                    analyzers=analyzers, modules=modules)
        args = {key:val for key,val in args.items() if val is not None} # Remove None inputs
        input_pars = ss.utils.copy_given(pars, args, kwargs, copy_modules=copy_modules)
        if not copy_modules: # Nothing to start again from, since the modules given are the ones that run
            self._before_init = 'This sim was made with copy_modules=False, so it runs the modules it was given themselves, and can\'t be run again or copied with sim.new(): make it again instead (with new modules)'
        self.check_model_keywords(input_pars, kwargs)
        self.pars.update(input_pars)  # Update the parameters
        self.given = [key for key,val in input_pars.items() if val is not None] # Which parameters were supplied, as opposed to defaults
        self.notes = [] # Notes on what was assumed or approximated when the sim was built
        self.solver = None # Set in init()
        self.groups = ss.dimensions.Groups(self) # The groups of people by the stratifications, worked out in init()

        # Set attributes; see also sim.init() for more
        self.created = sc.now()  # The datetime the sim was created
        self.version = ss.__version__ # The Starsim version
        self.metadata = ss.debugtools.metadata()
        self.dists = ss.dists.Dists(obj=self) # Initialize the random number generator container
        self.loop = ss.loop.Loop(self) # Initialize the integration loop
        self.results = ss.Results(module='Sim')  # For storing results
        self.data = data # For storing input data
        self.initialized = False  # Whether initialization is complete
        self.complete = False  # Whether a simulation has completed running
        self.results_ready = False  # Whether results are ready
        self.elapsed = None # The time required to run
        self.progress = None # How far a run has got, shown as it goes (see ss.display.Progress)
        self.filename = None # Store the filename, if saved
        self.diagnostics = None # Store diagnostics results
        return

    def check_model_keywords(self, given, kwargs):
        """ A keyword that is not a setting of the sim but names a parameter of the model (e.g. ss.Sim(sir, beta=0.1)) is an error that says to give it in pars= """
        unknown = [key for key in kwargs if key not in self.pars and key not in ss.v3_deprecations.keys('Sim')]
        if unknown:
            mods = ss.utils.given_modules(given)
            names = {name for mod in mods for name in getattr(mod, 'pars', {})}
            model = [key for key in unknown if key in names]
            if model:
                vals = ', '.join(f'{key}={kwargs[key]!r}' if len(repr(kwargs[key])) < 30 else f'{key}=...' for key in model)
                errormsg = f'{sc.strjoin(model)} {"is a parameter" if len(model) == 1 else "are parameters"} of the model, not a setting of the sim: give {"it" if len(model) == 1 else "them"} by name in pars=, e.g. ss.Sim(..., pars=dict({vals}))'
                raise TypeError(errormsg)
        return

    def _settings(self):
        """ The sim settings, not the modules or people (which would be found again by the search for distributions) """
        skip = ss.modules.module_types() + ['custom', 'modules', 'people']
        return {key:val for key,val in self.pars.items() if key not in skip}

    @staticmethod
    def _name_modules(names, *args):
        """ A module built with the plain ss.Disease() constructor takes its name from the variable it is assigned to """
        taken = set(ss.SimPars()) | set(dir(ss.Sim)) | ss.People.taken_names() # Names the sim or the people already use, e.g. interventions in def run(interventions): ss.Sim(interventions=interventions), or version
        for arg in args:
            mods = list(sc.tolist(arg))
            for mod in mods:
                named = [mod] if type(mod) is ss.Disease or isinstance(mod, (ss.modify, ss.Event)) else []
                named += [getattr(mod, 'product', None)]*isinstance(mod, ss.Event)
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
        steps = [mod for arg in args for mod in sc.tolist(arg) if isinstance(mod, ss.Event)]
        used = [mod.name for mod in steps if not mod._default_name]
        for mod in steps: # Steps made in a function, with no variable to be named by: numbered, e.g. screen, screen_1, here rather than when the sim copies them, since a condition refers to a step by its name
            if mod._default_name:
                mod.name = mod.label = sc.uniquename(mod.name, used, style='_%d')
            used.append(mod.name)
        return

    @staticmethod
    def _order_steps(interventions):
        """ The deliveries run in the order of their triggers (see `ss.interventions.order_steps()`) """
        if not any(isinstance(mod, ss.Event) for mod in sc.tolist(interventions)):
            return interventions
        return ss.interventions.order_steps(sc.tolist(interventions))

    def _named_modules(self, attr):
        """ The modules called `attr`: the sim's own once it has them (from early in `init()`, so e.g. one module's `init_post()` can read `sim.<name>` of another), else those it was given """
        if 'custom' in self.__dict__: # Set by init_module_attrs(), the last of the module types
            mods = self.modules
        else:
            mods = ss.utils.given_modules(self.__dict__.get('pars'))
        return [mod for mod in mods if mod.name == attr]

    def __getattr__(self, attr):
        """ A module by its name, e.g. `sim.sir` for `sim.diseases.sir` """
        if not attr.startswith('_') and 'pars' in self.__dict__ and attr not in ss.modules.module_types() + ['custom']: # Not the module types, which init() sets
            mods = self._named_modules(attr)
            if len(mods) == 1:
                return mods[0]
        if not self.__dict__.get('initialized') and attr in ss.modules.module_types() + ['people']: # Made by init(), from what was given
            given = f'; until then, what was given is sim.pars.{attr}' if attr != 'people' else ''
            raise AttributeError(f'sim.{attr} is made when the sim is initialized, by sim.init() (or sim.run()){given}')
        ss.v3_deprecations.check_attr(type(self), attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, key):
        """ Allow dict-like access, e.g. sim['diseases'] """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """ Allow dict-like access, e.g. sim['created'] = sc.now() """
        return setattr(self, key, value)

    def __repr__(self):
        """ A one-line summary of the sim: its label, size, time span, and modules """
        try:
            pars = self.pars
            n = int(sc.ifelse(pars.total_pop, pars.n_agents)) # The number of people modeled, not the default number of agents (or the number of groups of a compartmental run)
            agents = int(pars.n_agents) if self.initialized and isinstance(self.solver, ss.solvers.abm) else n
            parts = [f'{self.label}; ' if self.label else '', f'n={n:n}', f' ({agents:n} agents)' if agents != n else '']
            t = self.t if self.initialized else pars
            parts.append(f'; {t.start}—{t.stop}')
            for kind in ss.modules.module_map():
                mods = self.__dict__.get(kind, pars.get(kind)) # The sim's modules once it is initialized, else those given
                if isinstance(mods, dict) and len(mods):
                    parts.append(f'; {kind}={sc.strjoin(mods.keys())}')
                elif isinstance(mods, str):
                    parts.append(f'; {kind}={mods}')
            parts.append('' if self.initialized else '; not initialized')
            return f'Sim({"".join(parts)})'
        except Exception as E:
            ss.warn(f'Error displaying custom sim repr, falling back to default: {E}')
            return sc.prepr(self, vals=False)

    def products(self):
        """ The products of the interventions, each once; a list, not an ndict like the other kinds of module """
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
        Every module of the sim, in the order they are initialized

        Custom modules come after the others they may read, and before the analyzers; their
        `step()` still runs first (see `ss.loop.Loop`).
        """
        yield from itertools.chain(self.demographics(), self.connectors(), self.networks(), self.interventions(),
                                   self.products(), self.diseases(), self.custom(), self.analyzers())

    @ss.utils.outermost
    def init(self, force=False, **kwargs):
        """
        Perform all initializations for the sim

        Args:
            force (bool): whether to overwrite sim attributes even if they already exist
            kwargs (dict): passed to ss.People()
        """
        if self.initialized and not force: # E.g. after sim.describe(), which initializes it
            return self

        self.timer = sc.timer() # Store a timer for keeping track of how long the run takes
        self._held = [] if self.pars.verbose else None # Notes are said after the line that starts the run

        # Validation and initialization -- this is "pre"
        defaults = ss.SimPars()
        for key,val in self._settings().items():
            if val is not None and key not in self.given and not ss.utils.same_value(val, defaults.get(key)): # A setting set before init() was given too, e.g. sim.pars.aging = True (but not one made a float, e.g. n_agents)
                self.given.append(key)
        self.pars.lock(False) # Allow parameters to be added during initialization
        self.pars.freeze(False) # And set (frozen again once it is initialized)
        self.__dict__.pop('_valued', None) # Arithmetic on a parameter builds the model until it is built (see below)
        self.pars.validate()
        np.random.seed(self.pars.rand_seed) # Reset the seed before the population is created -- shouldn't matter if only using Dist objects
        ss.solvers.choose_solver(self) # Build the declared model, and choose the unit and the solver
        self._init_time()
        if 'aging' not in self.given and not self.pars.aging and any(strat.kind == 'age' and strat.edges is not None for strat in self.groups.strats):
            long = self.t.dt_year*(self.t.npts - 1) >= 1 # Only then does it matter much
            ss.note('people do not age between the age groups, since nobody is born or dies; pass aging=True to age them', sim=self, verbose=None if long else 0)
        self.init_people(**kwargs)
        self._move_modules(force=force) # From sim.pars to the sim, e.g. sim.diseases
        self._init_data(check=False) # Load the data before the modules, which read it (e.g. data by group keeps their results by group at every step)
        self._init_modules_pre()
        self.groups.check_used() # A stratification that nothing uses
        ss.connectors.check_applied(self) # Nothing accepted, then ignored

        # Final initializations -- this is "post"
        if self.pars.rescale:
            self._init_rescale()
        self._init_dists()
        self._init_people_vals() # Initialize the values in all the states and networks
        ss.solvers.init_group_values(self) # The people's levels of each dimension, or each group's age and sex
        self._valued = True # From here, arithmetic on a parameter is on its value (see ss.expressions.has_values())
        self._init_modules_post() # Initialize the module values
        ss.solvers.settle_start(self) # The declared modules' people, once every module has its own
        self.init_results()
        self._init_data()
        self.loop.init()
        self.lock_pars() # Don't allow new parameters to be added by mistake

        self.verbose = self.pars.verbose # Store a run-specific value of verbose
        ss.solvers.group_people(self) # In a compartmental run, the people are groups

        # It's initialized
        self.initialized = True
        self.__dict__.pop('_valued', None)
        self.pars.freeze() # Its settings are this run's: a change is made with sim.new()
        held, self._held = self._held, None
        if self.verbose and self.__dict__.pop('_quiet', False): # Initialized by sim.describe(): said when it is run
            self._unsaid = held
        elif self.verbose:
            print('\n'.join(ss.display.start_lines(self, held)))
        if self.diagnostics and self.diagnostics.states is not None: # Need not None since dict is empty at this point
            self.diagnostics.store_states(key='init')
        return self

    def new(self, pars=None, inplace=False, **kwargs):
        """
        A new, uninitialized sim: a copy of this one as it was when it was initialized, optionally with other values

        A sim that has not been initialized is copied as it is, with any changes made to it since it
        was made. Once a sim is initialized, its settings are those of its run: setting one (e.g. `sim.pars.stop = 30`
        after `sim.run()`) raises, so give it here instead (`sim.new(stop=30)`). Other changes made to it then (e.g. to a
        module's parameters) belong to that run only, and are not kept.

        To run a finished sim again from the start, `sim.run()` is enough (and `sim.restart(pars=...).run()` with other
        values, which for a sim that is nothing but compartments is much faster than a new sim); `sim.new()` is for other
        settings, e.g. `sim.new(solver='abm').run()`. To copy a sim as it is, including its results, use `sim.copy()`.

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`, and of sim settings only if they were given as `ss.par` (e.g. `ss.Sim(n_agents=ss.par(5000, bounds=[1000, 20000]))`, so they can be fitted)
            inplace (bool): if True, this sim becomes the new one, rather than a new one being returned
            kwargs (dict): sim arguments to change, e.g. `rand_seed=2`, `solver='abm'`, or `interventions=...`; `None` goes back to the default, e.g. `n_agents=None` for a compartmental run of an agent-based sim; also `data=`. A subclass's own arguments (e.g. FPsim's `location=`) are not taken again: make the subclass again for those.
        """
        base = self.__dict__.get('_before_init', None if self.initialized else self)
        if not isinstance(base, Sim):
            errormsg = base or 'sim.new() copies the sim as it was before it was initialized, and this one has no such copy (e.g. it was saved by an earlier version); make it again instead'
            raise ValueError(errormsg)
        data, jax = kwargs.pop('data', self.data), self.jax # Not sim parameters: taken from this sim, as they may be set once it is initialized
        kinds = ss.modules.module_types() + ['modules']
        if any(key in kinds for key in kwargs): # Modules given here are named and ordered as ss.Sim() does
            self._name_modules(ss.utils.caller_names(1), *[kwargs.get(key) for key in kinds])
            if kwargs.get('interventions') is not None:
                kwargs['interventions'] = self._order_steps(kwargs['interventions'])
        pars = dict(pars or {})
        kwargs |= {key:pars.pop(key) for key in list(pars) if key in base.pars.bounded()} # A sim setting given as ss.par (e.g. n_agents=ss.par(5000, bounds=...)) is a value by name too, as when fitting it
        kwargs = ss.utils.copy_given(kwargs) # And copied
        sim = self if inplace else object.__new__(type(base))
        state = sc.dcp(vars(base), memo={id(base):sim}) # What refers to the sim (e.g. its loop) refers to the new one
        vars(sim).clear()
        vars(sim).update(state)
        sim.pars.check_key_mismatch(kwargs)
        for key, other in dict(stop='dur', dur='stop').items(): # The stop and the length are two ways of saying one thing, so a new one replaces the other
            if kwargs.get(key) is not None and other not in kwargs:
                kwargs[other] = None
        defaults = ss.SimPars()
        for key,val in kwargs.items():
            sim.pars[key] = sc.dcp(defaults.get(key)) if val is None else val
            if val is None and key in sim.given:
                sim.given.remove(key)
            elif val is not None and key not in sim.given:
                sim.given.append(key)
        sim.data = sc.dcp(data)
        sim.jax = jax
        sim.par_values = sc.mergedicts(sim.par_values, pars or None)
        return sim

    def fit(self, pars=None, data=None, weights=None, likelihood=None, loss=None, penalty=None, method=None, rand_seed=None, verbose=None, **kwargs):
        """
        Fit every parameter that has bounds to every result that has data

        Fits every parameter that has `bounds` (see `ss.par`) to every result that has
        data (see `ss.Sim(data=...)`). The original sim is not changed. The fitting itself
        is in `ss.fitting`, which is imported when it is first used.

        A parameter can be one of a declared model, of a module written as code (`ssl.HIV(beta=ss.par(...))`,
        also a value in a dict parameter, fitted as e.g. `cin_fn.k`, or a setting kept as an attribute),
        a modifier's amount (`ss.modify(..., multiply=ss.par(...))`), or a keyword argument of a function
        given as a module (`def set_testing(sim, symp_prob=ss.par(0.1, bounds=[0.01, 0.5]))`, for a value
        set in code); one named alike in several modules is fitted with its module's name, e.g.
        `hpv16.rel_beta`. A parameter whose value is an array, with bounds for each element
        (`ss.par(np.array([...]), bounds=[lows, highs])`), has each element fitted, as `knots[0]`, ...,
        except those whose bounds are equal. A data column names a result, as `sim.results` has it:
        e.g. `prevalence`, `sir.prevalence`, an analyzer's `by_age.cancers_0_15`, a result nested deeper
        (`hpv.genotype.cancers`), or a column of a result with columns (`sis.infection.alpha`).

        The fit uses either likelihoods or a loss: every data series has a likelihood (declared on its
        outcome, or given here), and the fit maximizes their sum; or none has, and it minimizes the loss.
        `loss=` is used in place of likelihoods declared on the outcomes (with a note), and `likelihood=False`
        switches them off. Some series with a likelihood and some without is an error, as is `loss=` with `likelihood=`.

        By default, the fit finds the best values (a point estimate), by the method that suits the sim, and
        prints which and why:

        - 'l-bfgs-b' for a deterministic sim, with gradients from JAX (see `ss.gradients`), or by finite differences for one that JAX cannot run
        - 'nelder-mead' for one whose likelihood or priors JAX does not cover (e.g. a likelihood given as a function), or with one or two fitted parameters (where it is faster than gradients)
        - 'optuna' for a stochastic sim, including an agent-based one

        Other methods for a best fit are 'powell' (no gradients), 'slsqp' (with gradients, like 'l-bfgs-b'),
        and 'if2' (iterated filtering, for the sims that 'pmcmc' fits: the maximum-likelihood values, from passes
        of a particle filter in which each particle's values are perturbed by a shrinking random walk; see
        `ss.fitting.Swarm`). For the uncertainty in the values (a posterior), ask for one by name:

        - 'nuts' (the No-U-Turn sampler), for a deterministic sim with gradients whose data series all have a likelihood; each chain starts from a draw near one of the best optima found from a scan over the bounds
        - 'pmcmc' (particle MCMC), for a sim whose data series all have a likelihood: a random-walk Metropolis sampler on the likelihood estimated by a particle filter for a stochastic compartmental sim (see `ss.fitting.ParticleFilter`), or on the exact likelihood for a deterministic sim (e.g. one JAX cannot run); each chain starts from a different one of the best points of a scan, and a run is kept with each draw, `fit.trajectories`, which `fit.project()` continues
        - 'abc-smc' (approximate Bayesian computation by sequential Monte Carlo), for any stochastic sim, with no likelihood needed; its accuracy depends on the tolerance it reaches, and it gives no posterior for a deterministic sim

        The priors are the parameters' `prior=`, or uniform over their bounds.

        Args:
            pars (list): the names of the parameters to fit (default: all that have bounds)
            data (df): the data to fit to (default: the sim's data)
            weights (dict): the relative weight of each data series (default: equal)
            likelihood (`ss.Dist`/dict/bool): the likelihood of every data series (e.g. `ss.poisson()`), or of each by its name (e.g. `dict(prevalence=ss.normal(std=0.02))`), in place of any declared on its outcome; or False, for none (the declared ones are not used, and the fit minimizes the loss). The fit then maximizes their sum (plus the log of the priors of the parameters that have one, for the methods that find a best fit: the most probable values)
            loss (str/func): the measure of the mismatch to minimize, with no likelihoods (or in place of those declared on the outcomes): 'nae' (normalized absolute error, the default), 'rmse', or a function of the sim; any other name is an error
            penalty (func): a function of the parameter values (a dict by name, an array whole) returning a number added to what a best fit minimizes, e.g. a smoothness penalty, `lambda pars: 10*np.sum(np.diff(pars['knots'], 2)**2)`
            method (str/`ss.fitting.method`/list): how to fit (see below); a method with settings is `ss.fitting.method()`, e.g. `ss.fitting.method('optuna', n_trials=200)` (the settings of each method, such as the number of runs per point for a stochastic sim, are listed there); a list of methods runs each from the best fit of the one before (a stage that does not converge stops the list, but not one whose convergence is not assessed, such as ABC-SMC's; see `ss.fitting.fit_stages()`)
            rand_seed (int): the random seed for the fitting method (default: the sim's, so a fit gives the same answer each time)
            verbose (bool): whether to print progress

        Returns:
            An `ss.fitting.Fit`, with the calibrated sim as `fit.sim` and the best-fit values as `fit.pars`

        Examples:
            ```python
            beta = ss.par(0.1, bounds=[0.01, 0.5])
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(10), init_prev=0.01)
            sim = ss.Sim(sir, data='cases.csv', start='2020-01-01', stop='2020-06-01')
            fit = sim.fit() # A best fit
            post = sim.fit(method='nuts', likelihood=ss.poisson()) # A posterior, with a likelihood for each data series
            more = sim.fit(method=ss.fitting.method('nuts', draws=1000, chains=2), likelihood=ss.poisson()) # With settings of the method
            fit.sim.plot()
            ```
        """
        from .internals import fitting as fit
        if kwargs: # The settings of a method, which were keywords of sim.fit()
            key = list(kwargs)[0]
            name = method if isinstance(method, str) else getattr(method, 'name', 'optuna')
            errormsg = f'sim.fit() takes no {key}=; the settings of a method are given with it, e.g. sim.fit(method=ss.fitting.method("{name}", {key}=...))'
            raise TypeError(errormsg)
        common = dict(pars=pars, data=data, weights=weights, likelihood=likelihood, loss=loss, penalty=penalty, rand_seed=rand_seed, verbose=verbose)
        if isinstance(method, (list, tuple)): # Staged fitting: each method starts from the best fit of the one before
            return fit.fit_stages(self, method, **common)
        method = fit.as_method(method)
        problem = fit.Problem(self, **common, **(method.scoring if method is not None else {}))
        name = problem.choose(method.name if method is not None else None)
        problem.start()
        out = fit.methods[name](problem, **(method.options if method is not None else {}))
        return problem.result(name, out)

    def optimize(self, goal=None, total=None, start=None, spend=None, bounds=None, target=None, maxiter=50, verbose=None, **kwargs):
        """
        Allocate spending between programs to reach a goal, keeping the total fixed; or find the least total that reaches a target

        The spending of each program is a parameter of the model: one per program of an imported program book
        (`sim.imported.programs`), or any parameters, given by name with `spend=`. The search (SciPy's SLSQP) is
        over the shares of the total, and each point is a run (restarted, if the sim is nothing but compartments).
        With `target=`, the total is doubled until its best allocation reaches the target, then found by bisection
        between a total that reaches it and one that does not (to within 1%), assuming that more money never does worse.
        The sim itself is not changed. The search is in `ss.fitting`, which is imported when it is first used.

        Args:
            goal (`ss.fitting.maximize`/`ss.fitting.minimize`): what to make as large or as small as possible: a result (e.g. `ss.fitting.maximize('all_vs', at=2030)`, or `ss.fitting.minimize('hiv.new_infections')`), or a function of the run sim
            total (float): the total to allocate (default: the current total); with `target=`, the total the search starts from
            start (float/date): when the new allocation takes effect (default: the start of the sim), read as any time is (see `ss.connectors.to_time()`: a bare number is a year if the sim has a calendar)
            spend (dict/list): the spending parameters: names, or {program: name} (default: the imported programs)
            bounds (tuple/dict): the least and most a program can have, as amounts (default: 0 and the total); a dict gives them by program, e.g. `dict(testing=(1e5, None))` (None for no limit), and a program not in it has the default
            target (float): the value of the result to reach (at least this with `ss.fitting.maximize()`, at most with `ss.fitting.minimize()`) with the least total
            maxiter (int): the most iterations of the search (of each total's, with `target=`)
            verbose (bool): whether to print progress

        Returns an `ss.fitting.Optimization`.

        Examples:
            ```python
            sim = ss.import_from('hiv_framework.xlsx', format='atomica', databook='hiv_databook.xlsx', progbook='hiv_progbook.xlsx', programs_start=2020)
            best = sim.optimize(ss.fitting.maximize('all_vs', at=2030), start=2020)
            best.summarize()
            least = sim.optimize(ss.fitting.maximize('all_vs', at=2030), start=2020, target=1.1*best.value) # The least money that reaches it
            ```
        """
        from .internals import fitting as fit
        for key in ['maximize', 'minimize', 'at']:
            if key in kwargs:
                errormsg = f'sim.optimize() takes the goal as goal=ss.fitting.{"maximize" if key == "at" else key}(...){", with at= in it" if key == "at" else ""}, e.g. goal=ss.fitting.{"maximize" if key == "at" else key}("all_vs", at=2030), not {key}='
                raise TypeError(errormsg)
        if kwargs:
            raise TypeError(f'sim.optimize() got an unexpected keyword argument "{list(kwargs)[0]}"')
        return fit.optimize(self, goal=goal, total=total, start=start, spend=spend, bounds=bounds, target=target, maxiter=maxiter, verbose=verbose)

    def sensitivity(self, **kwargs):
        """ Which parameters matter: partial rank correlations of each parameter with bounds and each result, over samples within the bounds; see `ss.fitting.sensitivity()`

        Args:
            kwargs (dict): passed to `ss.fitting.sensitivity()`
        """
        from .internals import fitting as fit
        return fit.sensitivity(self, **kwargs)

    def check_dt(self, factor=2):
        """
        Rerun the sim with a smaller timestep, and report how much each result changes

        In a run that takes whole steps this covers everything that depends on the size
        of the step: turning rates into probabilities, the delay of a step between modules,
        and the order of the modules. In a stochastic run the change also includes the
        difference between two runs.

        Args:
            factor (int): how many times smaller to make the timestep

        Returns:
            The report, as text that shows itself (`ss.display.Text`), with the table behind it (a dataframe) as `.table`

        Examples:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=4), init_prev=0.01)
            ss.Sim(sir, solver='binomial', unit='day', stop=100).check_dt()
            ```
        """
        return ss.display.check_dt(self, factor=factor)

    def what_modifies(self, target):
        """
        Everything that modifies a quantity: each `ss.modify()`, and the efficacy of each product, as text that shows itself (`ss.display.Text`; `print()` it in a script)

        Args:
            target (par/str): the quantity, as for `ss.modify()`: e.g. `sir.pars.beta`, `sir.rel_sus`, or `'sir.pars.beta'`
        """
        return ss.display.what_modifies(self, target)

    def stratify(self, **kwargs):
        """
        Stratify every module in the sim by properties of people, e.g. `sim.stratify(age=[0, 15, 65])`

        Takes the same arguments as `ss.Disease.stratify()`, and returns the sim.

        Args:
            kwargs (dict): each dimension and its levels (or `ss.levels()`), as for `ss.Disease.stratify()`
        """
        if self.initialized:
            errormsg = 'Cannot stratify a sim that has already been initialized; stratify it first'
            raise RuntimeError(errormsg)
        ss.dimensions.check_replaced(kwargs)
        current = ss.diseases.check_stratify(self.pars.stratify, 'ss.Sim') if self.pars.stratify is not None else {}
        if kwargs and all(key in current and repr(current[key]) == repr(val) for key,val in kwargs.items()): # The same line run twice
            ss.note(f'the sim is already stratified by {sc.strjoin(kwargs)} in the same way, so this stratify() does nothing', sim=self)
            return self
        again = [key for key in kwargs if key in current]
        if again:
            raise ValueError(f'the sim is already stratified by "{again[0]}".')
        self.pars.stratify = dict(current, **kwargs)
        return self

    @property
    def _unit_name(self):
        """ The unit of time, for what the sim says: its unit, or, with none, "step" (or "unit", if a step is not one unit) """
        if self.unit:
            return self.unit
        dt = self.pars.dt
        dt = dt.value if isinstance(dt, ss.Duration) else dt
        return 'step' if dt is None or dt == 1 else 'unit'

    def _has_demography(self):
        """ Whether people are born or die other than of a disease: demographics modules, a declared birth, arrival, or death from every state """
        return bool(self.pars.demographics) or any(ss.solvers.flows_of(mod) for mod in self.declared)

    def _declared_module(self, name):
        """ A module with declared states and transitions, by name

        Args:
            name (str): the name of the module
        """
        mods = {mod.name:mod for mod in self.declared}
        if name not in mods:
            errormsg = f'No module named "{name}" with declared transitions was found; these are {sc.strjoin(mods.keys())}'
            raise KeyError(errormsg)
        return mods[name]

    def describe(self):
        """
        The resolved model: every state, transition, and default, with each guess marked; what runs where (and if a compartmental run is taken a step at a time, why); and what each rate and dwell time is per step

        It is text that shows itself (`ss.display.Text`), as `equations()` is: in color at an interactive prompt in a terminal,
        as tables in a notebook, and as plain text with `print(sim.describe())` (e.g. in a script) or `str()`.
        """
        return ss.display.describe_sim(self)

    def diagram(self, kind='plot', **kwargs):
        """
        The state-transition diagram of every declared disease, with how they act on each other (e.g. a modifier of one disease with a condition on another's state, as a dotted arrow)

        Args:
            kind (str): 'plot' to draw it as a Matplotlib figure, or 'text' for the text of a Mermaid flowchart (see `ss.Disease.diagram()`)
            kwargs (dict): for a plot, passed to `ss.return_fig()` (e.g. `filename=`, `do_show=`)

        Example:
            ```python
            sim = ss.Sim(diseases=[ss.SIR(), ss.SIS()])
            sim.diagram()
            ```
        """
        self._init_quietly()
        if not self.declared:
            errormsg = 'The sim has no declared diseases to draw (a module written as code has no diagram)'
            raise ValueError(errormsg)
        graphs = [mod.graph for mod in self.declared]
        return ss.display.show_diagram(graphs, [mod.name for mod in self.declared], links=ss.display.diagram_links(self), sim=self, kind=kind, **kwargs)

    def equations(self):
        """
        The ODE systems of every declared disease, with symbolic parameter names, and how the diseases act on each other, as text that shows itself (`ss.display.Text`; `print()` it in a script); see `ss.Disease.equations()`
        """
        self._init_quietly()
        return ss.display.sim_equations(self)

    def _kernel_reason(self):
        """ Why the sim is not run in one call to the kernels, in a few words, or None if it is (`sim.describe()` says so) """
        return ss.solvers.why_not(self, 'kernels')

    def _init_quietly(self):
        """ Initialize the sim, if it isn't, without the line a run says as it starts """
        if not self.initialized:
            self._quiet = True
            self.init()
        return

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
            sim = ss.Sim(diseases='sis', networks='random', start='2000-01-01') # The default run of 100 days
            sim.run(until='2000-02-01')
            sim.add_module(ss.modify(sim.diseases.sis.pars.beta, multiply=0.5, when=ss.schedule(start='2000-03-01'))) # Not a delivery of a new product, which changes the model's structure
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
        if module.has_product:
            same = [p for p in self.products() if p is module.product or p.name == module.product.name]
            if same: # The sim's copy of a product it already has
                module.product = same[0]
            else:
                errormsg = (f'Cannot add "{module.name}" after the sim is initialized: holding its product "{module.product.name}" is a dimension of the model ("{module.product.dim}"), '
                            f'which is laid out at sim.init(). Give it when creating the sim, with a start date (e.g. start=) and its when=, or compare with sim.new(interventions=...).')
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
        for mod in self.declared: # What the engines keep for the run can depend on the other modules (e.g. a modifier of a beta)
            if hasattr(mod.engine, 'modules_changed'):
                mod.engine.modules_changed()

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
        if not matches and die:
            errormsg = f'No matching module found for {query}; set die=False to return None instead'
            raise KeyError(errormsg)
        return matches[0] if matches else None

    def get_modules(self, query=None, match_case=False, as_dict=False):
        """
        Retrieve modules from the Sim

        Retrieve a module by name or by type. The query term can be the type of module to match
        e.g., `ss.SIR`, `ss.Disease`, `ss.Intervention`, or a string which is tested against
        module names, in which `*` matches anything (e.g. `'hiv*'`).

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
        if self.initialized or 'demographics' in self.__dict__: # Initialized, or initializing: the modules have been moved to the sim (see init_module_attrs())
            modules = self.modules
        else:
            modules = sc.search(self.pars, type=ss.Module).values()

        if query is None:
            return ss.utils.nlist_to_dict(modules) if as_dict else list(modules)

        # Match by type, or by name, where * matches anything (e.g. 'hiv*')
        if isinstance(query, str):
            name = lambda mod: mod.name if match_case else mod.name.lower()
            return [mod for mod in modules if fnmatch.fnmatchcase(name(mod), query if match_case else query.lower())]
        return [mod for mod in modules if isinstance(query, type) and isinstance(mod, query)]

    def _init_time(self):
        """ Time indexing; derived values live in the sim rather than in the pars """
        self.t = ss.Timeline(name='sim')
        self.t.init(self)
        if self.t.npts > 3001 and 'dt' not in self.given: # The step was not asked for
            why = f"the model's unit, from {self.unit_source}" if self.unit_source else f"unit='{self.pars.unit}'" if self.pars.unit is not None else "the model's unit" # The unit given (source None) is said as given
            ss.warn(f'the sim has {self.t.npts:n} steps of one {self.unit} ({why}); give dt= for a longer step, e.g. dt=ss.weeks(1)')
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
        if (self.named_people is not None or self.given_pyramid is not None) and not isinstance(self.solver, ss.solvers.abm): # A compartmental run takes the size and age split of a population, not its agents
            people = None
        n_agents = self.pars.n_agents

        # If people have not been supplied, make them -- typical use case
        if people is None:
            if isinstance(self.solver, ss.solvers.abm) and self.groups.strats: # Properties that people are stratified by, other than age
                extra = [strat.people_arr() for strat in self.groups.strats if strat.kind != 'age']
                kwargs['extra_states'] = sc.mergelists(kwargs.get('extra_states'), extra)
                for strat in self.groups.strats: # Ages follow the split of the age groups, so every solver starts from the same population
                    if strat.kind == 'age' and strat.edges is not None and 'age_data' not in kwargs and (ss.data.age_pyramid(self, strat) is None or strat.edges[0] > 0): # (Nobody is younger than the first group)
                        top = max(100, strat.edges[-1] + 10)
                        kwargs['age_data'] = np.column_stack([np.append(strat.edges, top), np.append(strat.fractions, 0)])
                    elif strat.female is not None and 'p_female' not in kwargs: # A sex dimension is people.female, so its split is the share of women
                        kwargs['p_female'] = strat.fractions[strat.female]
            if 'age_data' not in kwargs: # Otherwise the ages are a named population's, or the world's, as a guess (as is the split of age groups that are not given one)
                kwargs['age_data'] = ss.data.to_age_data(*ss.data.age_pyramid(self))
            people = ss.People(n_agents=n_agents, **kwargs)  # This just assigns UIDs and length
        elif isinstance(self.solver, ss.solvers.abm):
            self.groups.kept = {strat.name for strat in self.groups.strats if strat.kind != 'age' and strat.female is None and strat.attr in people.states} # People who come with a property keep it: it is binned into the levels, not drawn from the split
            for strat in self.groups.strats:
                if strat.name in self.groups.kept and strat.fractions_given:
                    raise ValueError(f'"{strat.name}" is a property the people already have, so their own values are kept and binned into its levels; shares given for its levels would overwrite them. Give the levels as a list (e.g. {strat.name}={list(strat.labels)!r}), or set people.{strat.attr} instead.')
                if strat.name in self.groups.kept:
                    strat.guessed.pop('fractions', None) # Not "everyone at the first level": the people's own values are used
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
        The sim's label, which names it in plots and printouts; kept as `sim.pars.label`

        A sim has no name apart from its label. A module has both: `mod.name` is for code (e.g. `'my_sir'`), and `mod.label` is for people (e.g. `'My SIR module'`).
        """
        pars = self.__dict__.get('pars')
        return pars.get('label') if pars is not None else None

    @label.setter
    def label(self, label):
        self.pars['label'] = label
        return

    def _move_modules(self, force=False):
        """ Move the modules from `sim.pars` to the sim (e.g. `sim.pars.diseases` to `sim.diseases`), leaving their parameters in `sim.pars` """
        for attr in ss.modules.module_types() + ['custom']:
            orig = getattr(self, attr, None)
            if not force and orig is not None:
                warnmsg = f'Skipping key "{attr}" in parameters since already present in sim and force=False'
                ss.warn(warnmsg)
            else:
                modules = self.pars.pop(attr) # Remove module type (e.g. 'diseases') from the parameters
                setattr(self, attr, modules) # Add the modules ndict to the sim
                self.pars[attr] = sc.obj() # Recreate with just the module parameters
                for key,module in modules.items():
                    self.pars[attr][key] = module.pars

        for mod in self.modules: # Each module is also sim.<name> (see __getattr__()), unless the sim has an attribute of that name (module names are unique: see ss.SimPars.validate_name())
            if mod.name in dir(self):
                ss.warn(f'The sim has an attribute "{mod.name}", so sim.{mod.name} is not the module of that name; use e.g. sim.diseases.{mod.name}')
        return

    def lock_pars(self, locked=True):
        """ Lock the sim and module parameters, so setting a parameter that doesn't exist raises an error

        Args:
            locked (bool): whether to lock the parameters
        """
        self.pars.lock(locked)
        for mod in self.modules:
            mod.pars.lock(locked)
        return

    def _init_modules_pre(self):
        """ Initialize all the modules with the sim """
        for mod in self.modules:
            mod.init_pre(self)
        for mod in self.modules: # Once every modifier is known: one in a module made later (e.g. an intervention) modifies a module made before it (e.g. a connector)
            if isinstance(getattr(mod, 'engine', None), ss.solvers.Engine):
                mod.engine.init_dwell_mods()
        return

    def _init_dists(self):
        """ Initialize the distributions, now that everything else is in place, and give each module its own """
        self.dists.init(obj=self, base_seed=self.pars.rand_seed, force=True)
        for mod in self.modules:
            self.dists.copy_to_module(mod)
        return

    def _init_people_vals(self):
        """ Initialize the People states with actual values """
        self.people.init_vals()
        return

    def _init_modules_post(self):
        """ Initialize values in other modules, including networks and time parameters, and do any other post-processing: the diseases first, after the demographics (whose unborn people they leave out), so that any other module's `init_post()` sees who starts infected """
        order = lambda mod: 0 if mod.name in self.demographics else 1 if mod.name in self.diseases else 2
        for mod in sorted(self.modules, key=order):
            mod.init_post()
        return

    def init_results(self):
        """ Make the results of the sim itself: the people's (see `ss.People.init_results()`), and those of dynamic rescaling """
        self.people.init_results()
        ss.solvers.init_people(self) # In a compartmental run, the same results, from the declared modules

        # With dynamic rescaling, store the scale factor for each timestep, and the people no agent stands for yet (see rescale())
        if self.pars.rescale:
            self.results += ss.Result('pop_scale', label='Population scale factor', kind='ratio', scale=False, auto_plot=False, timevec=self.timevec, values=np.ones(self.t.npts))
            self.results += ss.Result('unrepresented', label='People no agent stands for yet', kind='stock', scale=False, auto_plot=False, timevec=self.timevec, values=np.full(self.t.npts, (self.pop_scale - 1)*self.people._n_agents_alive(), dtype=float))

        # A product with no results of its own (most are only given by a delivery, which keeps the results) is left out
        for prod in self.products():
            if not [key for key,res in prod.results.items() if isinstance(res, ss.Result) and key != 'timevec']:
                self.results.pop(prod.name, None)

        # The results of the sim's one disease (the one it made or named, else its only one) are also at the top, e.g. sim.results.I
        implicit = [mod.name for mod in self.diseases() if mod._implicit] or [mod.name for mod in self.diseases()]
        if len(implicit) == 1:
            self.results.setattribute('_implicit', implicit[0])

        # The results named to be kept by group must exist
        if self.pars.group_results is not None:
            known = [f'{mod.name}.{key}' for mod in self.modules for key,res in mod.results.items() if isinstance(res, ss.Result)]
            for name in sc.tolist(self.pars.group_results):
                if name not in known and not any(key.split('.', 1)[1] == name for key in known):
                    errormsg = f'group_results= names "{name}", which is not a result of this sim; the closest are {sc.strjoin(ss.expressions.closest(name, known))}'
                    raise sc.KeyNotFoundError(errormsg)
        return

    @property
    def pop_scale(self):
        """ The number of people each agent stands for (with dynamic rescaling, at the end of the run): `total_pop/n_agents` as agents, and 1 in a compartmental run, which has no agents; set by `total_pop=` and `n_agents=` """
        pars = self.pars
        if (self.solver is not None and not isinstance(self.solver, ss.solvers.abm)) or not pars.n_agents or pars.total_pop is None:
            return 1.0
        return pars.total_pop/pars.n_agents

    @property
    def current_scale(self):
        """ The number of people each agent represents on the current timestep; varies over time if `pars.rescale=True` """
        if not self.pars.rescale:
            return self.pop_scale
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
            return self.pop_scale
        scale = self.results.pop_scale.values
        if module is not None and not np.array_equal(module.t.yearvec, self.t.yearvec):
            inds = np.searchsorted(self.t.yearvec, module.t.yearvec + 1e-9, side='right') - 1 # Use the most recent sim timestep, since the scale only changes on sim timesteps; 1e-9 avoids floating-point error
            scale = scale[np.maximum(inds, 0)]
        return scale

    def _init_rescale(self):
        """ Check that the diseases support dynamic rescaling, and create the distribution for it """
        for disease in self.diseases():
            if disease.graph is None and (type(disease).not_naive is ss.Disease.not_naive or type(disease).make_naive is ss.Disease.make_naive):
                errormsg = f'Dynamic rescaling (rescale=True) requires each disease written as code to define not_naive() and make_naive() (see ss.AgentDisease), but "{disease.name}" does not'
                raise ValueError(errormsg)
        self.rng_rescale = ss.choose_n(name='rng_rescale') # Choose who to make naive when rescaling
        return

    def rescale(self):
        """
        Dynamically rescale the population; only used if `pars.rescale=True`

        The scale factor starts at 1. Once the fraction of agents who are not naive (i.e. have
        ever been infected with any disease) exceeds the threshold (`rescale=dict(threshold=)`), the scale factor
        is increased by at least `rescale=dict(factor=)` (up to `sim.pop_scale`), and the same
        fraction of non-naive agents are made naive again (via `disease.make_naive()`), so that
        each agent now represents more people. Based on Covasim's dynamic rescaling.

        The people no agent stands for yet are counted as one number, `results.unrepresented`:
        `pop_scale - 1` for each agent at the start, less the people the agents alive take on
        at each rescaling (from scale `s` to `s'`, `s' - s` for each); the deaths and births of
        agents do not change it. So `alive` is `scale` people for each agent alive, plus these.
        """
        opts = sc.objdict(ss.simpars.rescale_options | (self.pars.rescale if isinstance(self.pars.rescale, dict) else {})) # Its options, or the defaults for rescale=True
        full = self.pop_scale
        ti = self.ti
        self.rng_rescale.jump_dt(ti=ti) # Like other distributions, advance the random numbers for this timestep
        scale = self.results.pop_scale
        if scale[ti] < full: # We have room to rescale
            not_naive = ss.uids.concatenate([disease.not_naive() for disease in self.diseases()]).unique()
            ratio = len(not_naive)/len(self.people) # Current proportion not naive
            if ratio > opts.threshold: # Check if we've reached the point to rescale
                max_ratio = full/scale[ti] # Don't exceed the total population size
                scaling_ratio = min(max(ratio/opts.threshold, opts.factor), max_ratio)
                new_scale = full if scaling_ratio == max_ratio else scale[ti]*scaling_ratio # Exactly the full scale at the end, so no rounding is left over
                self.results.unrepresented[ti:] -= (new_scale - scale[ti])*self.people._n_agents_alive() # The agents alive now stand for that many more people
                scale[ti:] = new_scale # Update the scale factor from here on
                n = int(round(len(not_naive)*(1.0 - 1.0/scaling_ratio))) # For example, rescaling by 2 makes half of the non-naive agents naive
                self.rng_rescale.set(n=n)
                new_naive = self.rng_rescale.filter(not_naive)
                for disease in self.diseases():
                    disease.make_naive(new_naive)
        return

    def _init_data(self, data=None, check=True):
        """ Initialize or add data to the sim; with check=False, only load and validate it, since the results it is checked against do not exist yet

        Args:
            data (dataframe/str): the data, or the name of a file holding it (default: `sim.data`)
            check (bool): whether to check it against the results
        """
        data = data if data is not None else self.data
        if isinstance(data, (str, os.PathLike)):
            data = ss.SimData(data)
        if isinstance(data, ss.SimData):
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
        """ Start the step: only print progress, since everything else happens in the modules """
        if self.complete:
            errormsg = 'Simulation already complete'
            raise ss.AlreadyRunError(errormsg)
        if self.verbose or self.progress is not None:
            self.elapsed = self.timer.toc(output=True)
            t = self.t
            if self.verbose >= 2:
                sc.heading(f'  Running {t.now("str")} ({t.ti:2.0f}/{t.npts-1}) ({self.elapsed:0.2f} s) ')
            elif self.progress is not None:
                self.progress.update()
        return

    def advance(self):
        """ Move the time to the step about to be taken, from the time point it starts at, where whatever it does is dated and recorded: for the sim, and the modules on its timeline (see `ss.loop.Loop`) """
        self.t.ti += 1
        for t in self.loop.shared_timelines:
            t.ti += 1
        return

    def finish_step(self):
        """ Finish the simulation timestep """
        if self.diagnostics and self.diagnostics.states:
            self.diagnostics.store_states()
        return

    @property
    def started(self):
        """
        Whether the run has begun, i.e. at least part of its first step has been taken (False before the first step, and again after `sim.restart()`)

        `sim.ti` is the step being taken, or the last one taken, so it is 0 both before the first step and after it;
        the number of steps taken is `sim.ti + sim.started`.
        """
        return bool(self.initialized and self.loop.index)

    def run_one_step(self, verbose=None):
        """
        Run a single step of the sim (every module's part of it), e.g. to look at the sim between steps while debugging

        `sim.loop.run_one_step()` instead runs one call of the loop, e.g. one module's `step()`.

        Args:
            verbose (bool): how much the loop prints (not the sim)
        """
        if not self.initialized:
            self.init()
        t = self.t
        nxt = t.ti + self.started # The next step, which starts at its time point
        self.loop.run(t.tvec[min(nxt, t.npts - 1)], verbose)
        return self

    def run(self, until=None, verbose=None, **kwargs):
        """
        Run the model -- the main method for running a simulation.

        Args:
            until (date/str/float/dur/Expr/func): run the steps up to and including the one that starts at this time (a duration is that long after the start, in a sim with a calendar); or a condition, checked at the end of each step, that stops and finalizes the run once it holds: an expression of the model (`sir.I.n + sir.R.n >= 1000`) or a function of the sim; the results after the stop are NaN, and `sim.stopped` is the last time point with results (the condition held at the end of its step)
            verbose (int): how much to print: 0 nothing, 1 the solver, notes, and progress (default), 2 everything (e.g. each step)

        Returns:
            The sim, with its results (e.g. `ss.Sim(...).run().plot()`)

        A sim that has finished is run again from the start, as it was before it was initialized. For a variant of it,
        give the changes first: `sim.restart(pars=dict(beta=0.1), rand_seed=2).run()` runs it again from the start with
        other values or another seed (for a sim that is nothing but compartments, without building it again, which makes
        repeated runs many times faster), and `sim.new(solver='abm').run()` builds a new sim with other settings. To run it
        many times, use `ss.parallel(sim, n_runs=10)`; to time its parts, `sim.profile()`. To copy a sim as it is, including
        its results, use `sim.copy()`.

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ 0.3*I, ss.Transition(I, R, dur=8), init=dict(I=0.01))
            sim = ss.Sim(sir, unit='day', stop=100)
            peaks = [sim.restart(pars=dict(beta=beta)).run().results.sir.I.max() for beta in [0.2, 0.3, 0.4]]
            ```
        """
        if kwargs:
            ss.v3_deprecations.check(ss.Sim, 'run', kwargs)
            errormsg = f'sim.run() got an unexpected keyword argument "{next(iter(kwargs))}"; it takes until= and verbose='
            raise TypeError(errormsg)
        return self._run(until=until, verbose=verbose)

    def _run(self, until=None, verbose=None, profile=False):
        """ Run the model, as `sim.run()`; with `profile`, a step at a time, recording how long each call of the loop takes (see `sim.profile(kind='loop')`) """
        imp = getattr(self, 'imported', None)
        unwarned = imp is not None and imp.get('incomplete') and not imp.get('warned') # Once per import: the runs and copies share it
        if unwarned and not self.initialized:
            imp.warned = True
            ss.warn(f'running an import with {len(imp.incomplete)} part(s) of the model not translated (sim.imported.incomplete); '
                    f'its results leave them out. {ss.converters.core.finishing(imp)}', ss.errors.ImportIncomplete)
        if isinstance(until, (bool, np.bool_)):
            errormsg = f'until={until} is True or False, neither a time nor a condition. A comparison that already has a value (e.g. A.n <= 1 made after the sim was initialized, when A.n is a number) is not a condition of the model: make it before sim.init(), or give a function of the sim, e.g. until=lambda sim: ...'
            raise TypeError(errormsg)
        if self.complete and until is None: # Run again from the start
            self.restart()

        # Initialization steps
        if verbose is not None and not self.initialized: # So that initializing says as much
            self.pars.verbose = ss.parameters.check_verbose(verbose)
        if not self.initialized: self.init() # Automatically initialize if not initialized
        if verbose is not None:
            self._orig_verbose = self.verbose
            self.verbose = ss.parameters.check_verbose(verbose)
        unsaid = self.__dict__.pop('_unsaid', None) # The start line, if sim.describe() initialized the sim
        if unsaid is not None and self.verbose:
            print('\n'.join(ss.display.start_lines(self, unsaid)))
        self.timer.start()
        started = time.perf_counter()

        # Check for AlreadyRun errors
        if self.complete:
            errormsg = 'Simulation is already complete'
            raise ss.AlreadyRunError(errormsg)

        self.stopped = None # The last time point with results, if a condition given as until= stopped the run
        if self.verbose: # Otherwise none, or how far it has got for a MultiSim's progress (see ss.run.single_run())
            self.progress = ss.display.Progress(self)
        try:
            # Main simulation loop
            restarted = self.__dict__.pop('_restarted', None)
            reason = self._kernel_reason() if until is None and not profile and restarted is None else None
            kernels = until is None and not profile and reason is None
            if reason is not None and not isinstance(self.solver, ss.solvers.abm) and getattr(self, 'declared', None): # Say once why a compartmental run is taken a step at a time (describe() says it too)
                ss.note(f'run a step at a time, which is much slower than all at once, since {reason}', sim=self, verbose=self.verbose)
            if self.pars.require_kernels and not kernels:
                raise ss.CapabilityError(f'require_kernels=True, but this sim would be run a step at a time in Python, since {reason or "until= was given, or it is being profiled"}')
            if kernels: # JAX or the kernels: see ss.solvers.choose_engine()
                known = self.__dict__.get('_engine_known')
                if restarted == 'same' and known is not None and known[0] is self.jax and not self.jax: # A choice from the model and its values, which are as they were
                    use_jax, why = known[1:]
                else:
                    use_jax, why = ss.solvers.choose_engine(self)
                    self._engine_known = (self.jax, use_jax, why)
                sc.printv(f'engine: {why}', 2, self.verbose)
            if kernels: # Nothing needs to happen between the steps, so each module takes all of them at once
                for mod in self.declared:
                    mod.engine.run_all(use_jax=use_jax)
                self.complete = True # Without making the loop's plan, which is not needed
            else:
                if isinstance(until, ss.Duration): # As any duration given as a time is (see ss.Timeline.to_time()): that long after the start with a calendar (or a calendar year, e.g. sim.t.tvec[5]), else a time from 0
                    until = self.t.to_time(until, name='until=')
                self.loop.run(until, profile=profile)
                self.stopped = self.loop.stopped
                self.complete = self.loop.index == len(self.loop.plan) or self.stopped is not None

            # If simulation reached the end, finalize the results
            self.elapsed = time.perf_counter() - started # As a number: sc.timer also formats a message
            if self.complete:
                self.finalize()
                if getattr(self, 'stopped', None) is not None:
                    self._blank_after(self.stopped)
                if self.pars.branching:
                    self._summarize_branching()
                if not kernels:
                    self._check_method_calls()
                if self.progress is not None:
                    self.progress.finish(ss.display.end_line(self))
        finally:
            self.dists.clear_run_caches()
            self.progress = None
        
        return self # Allows e.g. ss.Sim().run().plot()

    def _condition_holds(self, cond):
        """ Whether a condition given as `sim.run(until=)` holds now: an expression of the declared modules' states (`sir.I.n == 0`), or a function of the sim """
        if isinstance(cond, ss.expressions.Expr):
            names = {state.module.name for leaf in cond.leaves if isinstance(leaf, ss.expressions.Count) for state in leaf.states if state.module is not None}
            if len(names) != 1:
                errormsg = f'until={cond!r} needs the states of one declared module, not {len(names)}; for more, give a function of the sim, e.g. until=lambda sim: ...'
                raise ValueError(errormsg)
            return bool(np.all(self._declared_module(names.pop()).engine.eval(cond)))
        return bool(cond(self))

    @property
    def branching(self):
        """
        For a run as a branching process (`ss.Sim(branching=True)`): what came of it, as an `sc.obj` (by module, if several
        are declared), or None for any other run

        - `size`: the people infected, the seeds included
        - `outcome`: `'extinct'` if no one is infected at the end, `'stopped'` if `until=` stopped the run, or else `'ongoing'`
        - and as agents: `length`, the most generations of the chain; for each agent made, by UID, `infector` (-1 for a seed,
          or an infection from no one), `generation`, `ti` (the time point of infection), `offspring` (the number they
          infected), and `complete`, whether they were infected long enough before the end that all their offspring are counted

        See `msim.branching` for the runs of a MultiSim together.
        """
        return self.__dict__.get('_branching')

    def _summarize_branching(self):
        """ Work out `sim.branching` once the run is finished (see there) """
        out = sc.obj()
        last = self.stopped if self.stopped is not None else self.t.npts - 1
        for mod in self.declared:
            graph = mod.graph
            if not graph.acquisitions:
                continue
            uninfected = set(graph.uninfected)
            infected = [s.name for s in graph.people_states if s.name not in uninfected]
            active = {s.name for s in graph.infectious} or set(infected) # The states that transmit, and those that lead to one: someone there can still infect others
            grew = True
            while grew:
                before = len(active)
                active |= {src.name for tr in graph.transitions.values() for src in tr.source or [] if src.name in infected and any(dst.name in active for dst in tr.dest or [])}
                grew = len(active) > before
            res = mod.results
            start = sum(float(res[name][0]) for name in infected)
            new = sum(float(np.nansum(res[tr.name].values)) for tr in graph.acquisitions)
            now = sum(float(res[name][last]) for name in active)
            b = sc.obj(size=start + new, outcome='stopped' if self.stopped is not None else 'extinct' if now < 0.5 else 'ongoing')
            engine = mod.engine
            if isinstance(self.solver, ss.solvers.abm):
                n = self.people.uid.len_used
                infector = engine.infector[:n].copy()
                b.length = int(engine.generation[:n].max()) if n else 0
                b.infector = infector
                b.generation = engine.generation[:n].copy()
                b.ti = engine.infected_ti[:n].copy()
                b.offspring = np.bincount(infector[infector >= 0], minlength=n)
                b.complete = b.ti <= last - engine.infectious_window()/engine.dt # The same cut in every run, whatever came of it, so the people kept are not chosen by their offspring
            out[mod.name] = b
        self._branching = out[0] if len(out) == 1 else out
        return

    def _blank_after(self, ti):
        """ Set every result after time point ti to NaN, for a run stopped there by `until=` """
        for res in self.results.flatten().values():
            if isinstance(res, ss.Result) and len(res) == self.t.npts:
                res.values = res.values.astype(float) if res.values.dtype.kind in 'iub' else res.values
                res.values[ti+1:] = np.nan
        return

    def gradient(self, results=None, pars=None):
        """
        The derivative of results, or of a function of the run sim, with respect to parameters, at their current values

        The ODE run is differentiated by JAX (see `ss.gradients`), so this needs JAX, a sim
        with `solver='ode'`, and one that the kernels run from start to finish (see
        `sim.describe()`). The sim itself is not changed. A parameter with several values
        (a value for each level, `ss.by(patch=[...])`; an array; or a time series) has a
        derivative for each, named as `sim.fit()` names them (`beta[a]`, `beta[b]`, ...).

        Args:
            results (str/list/func): the results, by name (`'I'`, or `'sir.I'` where needed); or a function of the run sim that returns a number or an array (e.g. a cost); default: every result of the declared modules
            pars (str/list): the parameters, by name (or one value of one, `'beta[a]'`); default: every parameter that has bounds

        Returns:
            An `sc.obj` by parameter, then module, then result, of the derivative at each step; for a function, by parameter, of the derivative of what it returns

        Example:
            ```python
            beta = ss.par(0.3, bounds=[0.1, 1])
            S, I, R = ss.states(3)
            sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
            sim = ss.Sim(sir, unit='day', stop=100)
            grad = sim.gradient('I')
            grad.beta.sir.I # How the number infectious at each step changes with beta
            sim.gradient(lambda sim: sim.results.sir.infection.sum()).beta # How the total infections change with it
            ```
        """
        sim = self.new(verbose=0).init()
        return ss.gradients.of_results(sim, results=results, pars=pars)

    def log_density(self, pars=None, likelihood=None):
        """
        The log of the posterior density of the fitted parameters, as a function in JAX of their values, with its gradient by reverse mode; see `ss.gradients.LogDensity`

        Needs JAX, a sim with `solver='ode'` that the kernels run whole, data, and a likelihood for every data series.

        Args:
            pars (str/list): the parameters, by name (default: every one with bounds); one with several values has a value for each, named as `sim.fit()` names them
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
        Take a sim back to its start, optionally with new values of its parameters or a new seed, without running it

        `sim.run()` does this itself for a finished sim; `restart()` is for running it again with other values or
        another seed (`sim.restart(pars=dict(beta=0.1), rand_seed=2).run()`), for stepping it yourself from the start
        (`sim.restart().run(until=2010)`), or for a function that sets the values to try (as for `ss.gradients.Density`).
        The values given are kept for later runs (as `sim.par_values`), until changed again.

        For a sim that is nothing but compartments (one that the kernels run from start to finish: see `sim.describe()`), this makes
        repeated runs many times faster; any other sim (or a change the compartments were
        not laid out for, such as a new distribution) is built again, as `sim.new()` would.
        `ss.parallel(sim, n_runs=...)` and `sim.fit()` use it where they can.

        Args:
            pars (dict): values of model parameters by name, e.g. `dict(beta=0.1)`
            rand_seed (int): the seed for the next run

        Returns:
            The sim (itself, or itself built again)
        """
        if not self.initialized:
            self.init()
        self.loop.index = 0
        rebuild = lambda: self.new(pars=sc.mergedicts(self.par_values, pars), inplace=True, rand_seed=sc.ifelse(rand_seed, self.pars.rand_seed)) # With the values of earlier restarts
        same = not pars
        numbers = all(isinstance(val, (int, float, np.number)) for val in (pars or {}).values()) # A number in place of a number does not change how the sim runs
        if not (numbers and self.__dict__.get('_kernels_known') or self._kernel_reason() is None) or any(isinstance(val, ss.Dist) or key in self.pars.bounded() for key,val in (pars or {}).items()): # Not run in one call to the kernels, a distribution the compartments were not laid out for, or a sim setting (e.g. n_agents)
            return rebuild()
        base = self.__dict__.get('_before_init') # The copy that sim.new() and a rerun start from, which takes the same values
        if rand_seed is not None:
            dict.__setitem__(self.pars, 'rand_seed', rand_seed) # The seed of this run, which is set while the sim is initialized (see ss.SimPars.freeze())
            if isinstance(base, Sim):
                base.pars.rand_seed = rand_seed
        if pars:
            moving = [p for mod in self.declared for strat in mod.graph.strats if strat.mobility is not None for p in strat.mobility.fitted().values()]
            before = [float(p) for p in moving]
            self.par_values = sc.mergedicts(self.par_values, pars)
            if isinstance(base, Sim):
                base.par_values = sc.mergedicts(base.par_values, pars)
            ss.solvers.set_par_values(self)
            varies = lambda raw: isinstance(raw, ss.parameters.TimeValue) or (isinstance(raw, ss.by) and raw.varies)
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
        for joint in self.joint_states: # Nothing left over from the run that has ended
            joint.reset()
        for mod in self.declared:
            mod.t.ti = 0
            mod.finalized = False
            mod.engine.restart()
        for joint in self.joint_states: # As in init()
            joint.init()
        self._kernels_known = True
        self._restarted = 'same' if same else 'new' # For the run that follows, which need not check the same things again
        return self

    def finalize(self):
        """ Compute final results """
        # Finalize the results
        self.finalize_results()

        # Finalize each module, including the results
        for mod in self.modules:
            mod.finalize()
        self._check_finite()
        ss.solvers.record_people(self)

        # Resets verbose if needed
        if hasattr(self, '_orig_verbose'):
            self.verbose = self._orig_verbose
            delattr(self, '_orig_verbose')
        return

    def _check_finite(self):
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

        scale = stock_scale = self.result_scale()
        if self.pars.rescale and 'alive' in self.results: # With dynamic rescaling, a count of people (e.g. alive) includes those no agent stands for yet, who are like the agents, shared out as the agents alive are
            n = np.asarray(self.results.alive.values, dtype=float)
            stock_scale = scale + np.divide(self.results.unrepresented.values, n, out=np.zeros(len(n)), where=n>0)
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result):
                if res.scale: # Scale results; NB: disease-specific results are scaled in module.finalize() below
                    res._apply_scale(stock_scale if res.kind == 'stock' else scale)
                if np.all(res.values == res.values[0]): # Results were not modified during the sim
                    self.results[reskey].auto_plot = False
        self.results_ready = True # Results are ready to use
        return

    def summarize(self, key=None, how=None):
        """
        A table of each result over the whole run, by its `summarize_by`, as `msim.summarize()` gives for a scenario run once

        By default, a flow (e.g. new infections) is totaled, a stock (e.g. the number
        infected) takes its final value, and a ratio (e.g. prevalence) is averaged; see
        `ss.Result.summarize()`. Results that are 0 throughout are kept, so the tables of
        different runs line up row for row.

        Args:
            key (str): if given, only this result (e.g. `'sir_infection'` or `'sir.infection'`): its summary, as a number (or the rows of its columns, for a 2D result)
            how (str/dict): 'mean', 'median', 'sum', 'last', 'max', 'min', or a function, to summarize every result that way; or a dict of these by result, named as for `key` (e.g. `dict(sir_I='max')` or `{'sir.I': 'max'}`), with the rest by their `summarize_by`

        Returns:
            A dataframe with a row for each result and one column, `value`; with `key`, that result's summary

        Example:
            ```python
            sim.summarize()                  # Every result
            sim.summarize('sir_infection')   # The total infections
            sim.summarize('sir_I', how='max') # The peak number infectious
            ```
        """
        how = ss.results.summary_funcs(how, self.results)
        one = key is not None
        flat = self.results.flatten(sep='_', only_results=False, columns=True) if not one else ss.results.split_columns(ss.results.match_result_keys(self.results, key, show_skipped=True)) # Only the result asked for, with each column of 2D results separate
        values = {}
        for k, res in flat.items():
            func = how.get(k, how.get(''))
            if 'timevec' in k or (func is None and getattr(res, 'summarize_by', '') is None): # Skip module-specific time vectors, and results not summarized over time
                continue
            if not isinstance(res, ss.Result) and not (isinstance(res, np.ndarray) and np.issubdtype(res.dtype, np.number)): # Other data a module keeps in its results, e.g. a table of partnerships
                continue
            value = res.summarize() if func is None and isinstance(res, ss.Result) else (func or np.mean)(np.asarray(res))
            if np.ndim(value) and not one: # A 2D result with no columns= has no one number (ask for it by key)
                continue
            values[k] = float(value) if not np.ndim(value) else value
        summary = ss.results.Summary(dict(value=list(values.values())), index=pd.Index(list(values), name='result'))
        if one:
            if not values:
                errormsg = f'"{key}" has no summary: it is not summarized over time (give how=)'
                raise ValueError(errormsg)
            return summary.value.iloc[0] if len(summary) == 1 else summary # A 2D result has a row for each column
        return summary

    def shrink(self, inplace=True, full=True, size_limit=1.0, base_size=30, die=True):
        """
        Make the sim much smaller, for keeping many sims in memory or saving them, by removing the people and other large objects (in place by default)

        The results are kept, so a shrunk sim can still be plotted and summarized.
        Each module is then checked: if it is still larger than `size_limit*(base_size + n_timesteps)`
        KB, not counting its results, it is most likely holding an array of its agents that it failed
        to let go of (e.g. a module of your own whose `shrink()` misses one).

        Args:
            inplace (bool): whether to shrink this sim (default), or return a shrunk copy
            full (bool): whether to remove the people and the large parts of each module (otherwise, only the references that keep the sim from being freed from memory)
            size_limit (float): the size each module may have, in KB per timestep (None to skip the check)
            base_size (float): the size each module may have whatever the length of the sim, in timesteps' worth of `size_limit` (for its parameters and the like)
            die (bool): whether a module that is too large raises an error (otherwise, a warning)

        Returns:
            The shrunk sim
        """
        sim = self if inplace else self.copy() # A deep copy, so that nothing it shares with this sim is shrunk
        if full:
            ss.utils.shrink(sim, 'people')

        # A sim that is not initialized has nothing more to shrink
        if sim.initialized:
            sim.loop.shrink() # The loop and the distributions refer to the sim, which keeps it from being freed
            ss.utils.shrink(sim.dists, attrs=['sim', 'obj'])
            for dist in sim.dists.dists.values():
                with sc.tryexcept(die=die):
                    dist.shrink()

            if full:
                for mod in sim.modules:
                    with sc.tryexcept(die=die):
                        mod.shrink()
                for joint in sim.joint_states:
                    joint.shrink()

                # Check that each module did shrink
                if size_limit:
                    npts = len(sim)
                    max_size = size_limit*(base_size + npts) # In KB
                    for mod in sim.modules:
                        size = ss.utils.own_size(mod)/1e3 # In KB, not counting its results, which grow with its stratifications as well as with time
                        if size > max_size:
                            errormsg = f'Module {mod.name} did not successfully shrink: {size:n} KB, not counting its results, > {max_size:n} KB (= {size_limit:n} KB/timestep × ({base_size:n} base + {npts} timesteps)); use die=False to turn this message into a warning, or increase base_size/size_limit'
                            if die:
                                raise RuntimeError(errormsg)
                            ss.warn(errormsg)

        sim.shrunken = full # Only a full shrink counts
        return sim

    def _check_results_ready(self, errormsg='Please run the sim first'):
        """ Raise an error if the sim has not been run """
        if not self.results_ready:
            raise RuntimeError(errormsg)
        return

    def _check_method_calls(self):
        """
        Warn about any method marked as required (`@ss.required()`) that was not called, e.g. a `super()` call left out of a module of your own; called by `sim.run()`

        Returns:
            A list of the methods not called, by module
        """
        missing = []
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
            ss.warn(errormsg)
        return missing

    def to_df(self, **kwargs):
        """ The results as a dataframe, with a column for each (e.g. `sir_prevalence`); see `ss.Results.to_df()` for the arguments

        Args:
            kwargs (dict): passed to `ss.Results.to_df()`
        """
        self._check_results_ready('Please run the sim before exporting the results')
        return self.results.to_df(**kwargs)

    def to_synthetic_data(self, results, resample=None, noise='poisson', rand_seed=None):
        """
        Data drawn from the results of the run, as if observed: e.g. to check that a fit finds values that are known

        Args:
            results (str/list): the results, by name, e.g. 'cases' (or 'sir.cases' if more than one module has one of that name)
            resample (str): the period each value is over, e.g. 'week' for the total of a flow over each week
            noise (str/`ss.Dist`): 'poisson' (the default), for counts drawn with each value as their mean; a likelihood as `sim.fit()` takes it, with each value as its mean, e.g. `ss.nbinom(n=10)` for counts with a variance of mean + mean²/10 (see `ss.fitting.logpdf()`); or None, for the values themselves
            rand_seed (int): the random seed for the noise (default: the sim's)

        Returns:
            A dataframe with a column of times (`date`, or `time` in a sim without dates) and a column for each result, as `ss.Sim(data=...)` takes it

        Example:
            ```python
            truth = ss.Sim(sir, pars=dict(beta=0.025), rand_seed=2, **pars).run()
            data = truth.to_synthetic_data('cases', resample='week') # Weekly cases, with Poisson noise
            data = truth.to_synthetic_data('cases', resample='week', noise=ss.nbinom(n=10)) # Overdispersed, as fitted with likelihood=ss.nbinom(n=10)
            sim = ss.Sim(sir, data=data, **pars)
            ```
        """
        self._check_results_ready('Please run the sim before drawing data from it')
        if noise == 'poisson':
            noise = ss.poisson()
        elif not (noise is None or isinstance(noise, ss.Dist)):
            errormsg = f'noise={noise!r} is not a kind of noise; the choices are "poisson", a likelihood such as ss.nbinom(n=10), and None'
            raise ValueError(errormsg)
        rng = np.random.default_rng(sc.ifelse(rand_seed, self.pars.rand_seed))
        columns = {}
        for name in sc.tolist(results):
            res = ss.results.match_column(self, name)
            if res is None:
                errormsg = f'The sim has no result "{name}" (or more than one); results are named as in sim.results, e.g. "cases", or "sir.cases"'
                raise KeyError(errormsg)
            if isinstance(noise, (ss.poisson, ss.nbinom)) and res.kind == 'ratio':
                errormsg = f'"{name}" is a ratio, which has no noise of counts; give noise=None for its values, or a likelihood for a ratio, e.g. ss.normal(std=0.01)'
                raise ValueError(errormsg)
            series = res.to_series(resample=resample)
            values = series.to_numpy(dtype=float)
            columns[name] = values if noise is None else ss.fitting.draw(noise, values, rng)
        timecol = 'date' if isinstance(series.index[0], (ss.date, pd.Timestamp)) else 'time'
        return sc.dataframe({timecol:list(series.index), **columns})

    def profile(self, kind='line', follow=None, plot=True, **kwargs):
        """
        Profile the performance of the simulation: how long each call of the loop, each line, or each function takes

        Args:
            kind (str): what to time: 'loop', how long each call of the loop takes (each module's step, and so on), by running the sim itself a step at a time (then `sim.loop.plot_cpu()` and `sim.loop.to_df()`); 'line' (default), how long each line of the functions followed takes, with a line profiler (`sc.profile()`), on a copy of the sim; or 'function', how long every function the run calls takes, with a function profiler (`sc.profile(kind='function')`)
            follow (func/list): for `kind='line'`, the functions/methods to follow in detail (by default, those of the loop)
            plot (bool): for `kind='loop'` and `'line'`, whether to plot time spent per module step
            **kwargs (dict): passed to `sc.profile()` (for 'line') or `sc.profile(kind='function')` (for 'function', e.g. `sort='selftime'` or `mintime=1e-3`)

        Returns:
            For 'loop', the loop (`sim.loop`, with `to_df()` and `plot_cpu()`); for 'line', an `ss.debugtools.Profile`; for 'function', an `sc.cprofile`

        Examples:
            ```python
            import starsim as ss

            net = ss.RandomNet()
            sis = ss.SIS()
            sim = ss.Sim(networks=net, diseases=sis)
            sim.profile(kind='loop') # Which modules take the time
            prof = sim.profile(follow=[net.add_pairs, sis.step]) # Which lines of these take it
            prof = sim.profile(kind='function') # Which functions take it
            ```
        """
        kinds = ['loop', 'line', 'function']
        if kind not in kinds:
            errormsg = f'sim.profile(kind="{kind}") is not a kind of profile; the kinds are {sc.strjoin(kinds)}'
            raise ValueError(errormsg)
        if kind != 'line' and follow is not None:
            errormsg = f'follow= is for kind="line" (the lines of the functions followed), not "{kind}"'
            raise ValueError(errormsg)
        if kind == 'loop':
            if kwargs:
                errormsg = f'sim.profile(kind="loop") takes plot=, not {sc.strjoin(kwargs)}'
                raise TypeError(errormsg)
            if self.complete:
                self.restart()
            self._run(profile=True)
            if plot:
                self.loop.plot_cpu()
            return self.loop
        if kind == 'line':
            return ss.debugtools.Profile(self, follow=follow, plot=plot, **kwargs)
        cprof = sc.profile(kind='function', **sc.mergedicts(dict(sort='cumtime', mintime=1e-3), kwargs))
        with cprof:
            self.run()
        return cprof

    def save(self, filename=None, shrink=None, **kwargs):
        """
        Save to disk as a gzipped pickle: the sim itself, with its results, read back by `ss.load()`

        This is the checkpoint of a sim: saved before it has finished, it keeps its full
        state (it is not shrunk), so the loaded sim can continue the run. A pickle needs the
        same code (Starsim and any modules written as code) to be read. For the results
        as JSON, see `sim.to_json()`.

        Args:
            filename (str): the name or path of the file to save to (default: the sim's label, or "starsim", with ".sim")
            shrink (bool): whether to shrink the sim before saving, which makes the file about 99% smaller (see `sim.shrink()`); by default, yes unless the sim is in the middle of a run
            kwargs: passed to sc.makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        Examples:
            ```python
            sim.save('my.sim')
            ```
        """
        if shrink is None: # A sim in the middle of a run keeps its full state, so that it can carry on
            shrink = not (self.initialized and not self.results_ready)
        filename = sc.makefilepath(filename=filename or f'{self.label or "starsim"}.sim', **kwargs)
        self.filename = filename
        sim = self.shrink(inplace=False) if shrink else self
        sc.save(filename=filename, obj=sim)
        return filename

    def to_json(self, filename=None, keys=None, indent=2, **kwargs):
        """
        Export the results and the sim's parameters as JSON

        This is a record of a run's output, for reading elsewhere; it cannot be loaded
        back into a sim (see `sim.save()`).

        Args:
            filename (str): the file to write to, if any
            keys (str/list): what to write: 'pars', 'summary', and/or 'results' (default all three)
            indent (int): the number of spaces to indent by, in the file
            **kwargs (dict): passed to `sc.jsonify()`

        Returns:
            The parameters, summary, and/or results as a dict (also written to the file, if one is given)

        Examples:
            ```python
            json = sim.to_json() # Convert to a dict
            sim.to_json('sim.json') # Write everything
            sim.to_json('summary.json', keys='summary') # Just write the summary
            ```
        """
        d = sc.obj()
        for key in sc.tolist(keys or ['pars', 'summary', 'results']):
            if key in ['pars', 'parameters']:
                d.pars = self.pars.to_json()
            elif key == 'summary':
                if self.results_ready:
                    d.summary = self.summarize().value.to_dict()
                else:
                    d.summary = 'Summary not available (Sim has not yet been run)'
            elif key == 'results':
                df = self.to_df()
                if isinstance(df, dict):  # For sims with heterogeneous timelines, to_df() returns a dict of dataframes rather than a single dataframe
                    d.results = {k:v.to_dict() for k,v in df.items()}
                else:
                    d.results = df.to_dict()
            else:
                errormsg = f'keys="{key}" is not something sim.to_json() writes; the choices are "pars", "summary", and "results"'
                raise ValueError(errormsg)

        d = sc.jsonify(d, **kwargs)
        if filename is not None:
            sc.savejson(filename=filename, obj=d, indent=indent)
        return d

    def plot(self, key=None, fig=None, show_data=True, show_skipped=False, annualize=False, show_interventions=True, **kwargs):
        """
        Plot the results

        Args:
            key (str/list): the result to plot, e.g. `'sis.prevalence'` (by default, all); if a list, exactly those
            fig (Figure): an existing figure to plot into
            style (str): the plotting style to use
            show_data (bool): plot the data, if the sim has any
            show_skipped (bool): also show the results that are left out by default (e.g. those that never change)
            annualize (bool): if True, resample results to annual values before plotting (each by its `summarize_by`; see `ss.Result.resample()`)
            show_interventions (bool): mark when interventions and modifiers act: a dashed line at each round of a delivery, and a shaded span from the start to the stop of routine delivery or of an `ss.modify()` (those on an event or condition are not marked)
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
        self._check_results_ready('Please run the sim before plotting')
        data = ss.results.plot_data(self) if show_data else {} # The data, by the result that each column matches; a flow with data at a longer period (e.g. weekly) is shown as its totals over the data's periods, as the fit compares them
        if key is None and not show_skipped: # The sim's own results that never change (e.g. the number alive and the deaths of a closed population) are left out
            own = [k for k,res in self.results.items() if isinstance(res, ss.Result)]
            flat = ss.results.match_result_keys(self.results, None)
            key = [k for k,res in flat.items() if not (k in own and np.allclose(res.values, res.values[0]))] or None
        marks = {None:ss.results.intervention_marks(self)} if show_interventions else None
        return self.results.plot(key=key, fig=fig, data=data, label=self.label, show_skipped=show_skipped, annualize=annualize, marks=marks, **kwargs)


def demo(run=True, plot=True, summary=True, show=None, **kwargs):
    """
    Create a simple demo simulation for Starsim

    Defaults to using the SIR model with a random network, but these can be configured. On the
    network, β is per contact, so the demo's SIR has β = 0.03 a day: with 10 contacts each,
    infectious for 10 days, an R0 of 3.

    Args:
        run (bool): whether to run the sim
        summary (bool): whether to print a summary of the results
        plot (bool): whether to plot the results
        show (bool): whether to show the plot (default: `ss.options.show`)
        kwargs (dict): passed to `ss.Sim()`

    Examples:
        ```python
        ss.demo() # Run, plot, and show results
        ss.demo(diseases='sis', networks='degree') # Run with different defaults
        ```
    """
    defaults = dict(diseases=ss.SIR(beta=ss.perday(0.03)), networks='random') if 'diseases' not in kwargs and 'networks' not in kwargs else dict(diseases='sir', networks='random') # beta per contact on the network, for an R0 of 3 (see ss.SIR)
    pars = sc.mergedicts(defaults, kwargs)
    sim = Sim(pars)
    if run:
        sc.heading('Running demo:')
        sim.run()
        if summary:
            sc.heading('Results:')
            print(sim.summarize())
        if plot:
            sim.plot(**({} if show is None else dict(do_show=show)))
    return sim

