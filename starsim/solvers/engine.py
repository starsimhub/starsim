"""
The base of the engines that run a declared model (`Engine`), and what they share
"""
import zlib
import inspect
import re
import numpy as np
import sciris as sc
import starsim as ss
from types import SimpleNamespace as Bunch # A plain object with attributes; much faster to read than an objdict, which matters inside the ODE's right-hand side

__all__ = ['Effects', 'Engine', 'CountState', 'StockCtx', 'convert', 'add_rows', 'at_ends', 'by_level', 'no_tables', 'kept_by_group', 'group_rows', 'people_results']


from .backends.kernels import no_tables # For a step whose program holds for the whole step: see ss.kernels.step()


def convert(val, unit, cache=None):
    """ Convert a typed quantity to a plain number in a unit: a duration to units, a rate to per unit; with a `cache` (a dict), each is converted once, unless its number is changed """
    if cache is not None and isinstance(val, ss.TimePar):
        known = cache.get(id(val))
        if known is None or known[0] is not val or changed(known[1], val.value):
            if len(cache) > 10_000: # Values made as the sim runs (e.g. by a time series) are kept with the others, so are let go of now and then
                cache.clear()
            number = np.copy(val.value) if isinstance(val.value, np.ndarray) else val.value
            known = cache[id(val)] = (val, number, convert(val, unit))
        return known[2]
    if isinstance(val, ss.TimePar):
        if isinstance(val, ss.prob): # A probability per period becomes the hazard that gives it
            period = 1.0 if val.unit is None else val.unit.to(unit).value # With no unit, the probability is per unit of the sim's time
            with np.errstate(divide='ignore'): # A probability of 1 is an infinite hazard: certain within the step (FP1)
                return -np.log1p(-np.asarray(val.value, dtype=float))[()]/period
        return val.to(unit).value
    return val


def unconvert(val, number, unit):
    """ The reverse of `convert()`: a number, or an array, in a unit as a typed quantity like `val` """
    if not isinstance(val, ss.TimePar):
        return number
    if isinstance(val, ss.prob):
        period = 1.0 if val.unit is None else val.unit.to(unit).value
        out = sc.dcp(val)
        out.value = -np.expm1(-number*period)
        return out
    out = val.to(unit)
    out.value = number
    return out


def ratio_parts(expr):
    """ A result's numerator and denominator, if it is a ratio (scaled by a number, e.g. infection/N*1e5, is a numerator scaled by it); else the whole """
    BinOp, number = ss.expressions.BinOp, lambda e: isinstance(e, ss.par) and e.literal and sc.isnumber(e.raw)
    if isinstance(expr, BinOp) and expr.op == '/':
        return [expr.a, expr.b]
    elif isinstance(expr, BinOp) and expr.op == '*' and number(expr.b) and isinstance(expr.a, BinOp) and expr.a.op == '/':
        return [BinOp('*', expr.a.a, expr.b), expr.a.b]
    elif isinstance(expr, BinOp) and expr.op == '*' and number(expr.a) and isinstance(expr.b, BinOp) and expr.b.op == '/':
        return [BinOp('*', expr.a, expr.b.a), expr.b.b]
    return [expr]


def changed(old, new):
    """ Whether the number inside a typed quantity differs from the one it was converted from """
    if isinstance(new, np.ndarray):
        return not np.array_equal(old, new)
    return old != new


def times(x, *factors):
    """ An array times each of the factors, leaving out those that are a number equal to 1 that would not change its dtype, so the values are the same, bit for bit, in fewer passes (but may be `x` itself, not a copy) """
    for f in factors:
        if not (np.ndim(f) == 0 and f == 1 and np.result_type(x, f) == x.dtype):
            x = x*f
    return x


def add_rows(out, rows, tis, vals):
    """ Add values at consecutive time points `tis` (along the first axis of `vals`) into the rows of `out` they belong to: see `ss.solvers.group_rows()` """
    r = rows[tis]
    firsts = np.flatnonzero(np.diff(r, prepend=-1)) # Where each row starts
    out[r[firsts]] += np.add.reduceat(vals, firsts, axis=0)
    return


def at_ends(out, rows, ends, tis, vals):
    """ Put the values at the time points among `tis` that end a row into that row of `out` """
    last = np.isin(tis, ends)
    out[rows[tis[last]]] = vals[last]
    return


def kept_by_group(sim, mod, key):
    """ Whether a result of a module is kept by group: every one, unless the sim parameter `results` names the ones to keep (by name, or as "module.name") """
    names = sim.pars.results
    return names is None or key in sc.tolist(names) or f'{mod.name}.{key}' in sc.tolist(names)


def group_rows(sim, npts, row_bytes, budget, name=None):
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
    columns = [] if sim.data is None else sim.data.columns
    if npts*row_bytes <= budget or npts <= 2 or sim.pars.rescale: # With rescaling, the scale differs at each step, and is applied to each one
        return None, None
    elif any('[' in str(col) for col in columns): # Data by group, which are compared with the results at each step
        return None, None
    most = max(int(budget//row_bytes), 2) # The most rows that fit
    need = int(np.ceil((npts - 1)/(most - 1))) # The fewest steps per row
    days = float(sim.t.dt.days) if sim.unit is not None else None
    if days is not None and days <= 1:
        ladder = [max(int(round(d/days)), 1) for d in [7, 14, 28, 91, 182, 364]]
    else:
        ladder = [m*10**e for e in range(8) for m in [1, 2, 5]]
    every = next((k for k in ladder if k >= need), need)
    rows = (np.arange(npts) + every - 1)//every # The start is a row of its own, then each `every` steps
    ends = np.minimum(np.arange(rows[-1] + 1)*every, npts - 1)
    unit = sim.unit or 'step'
    span = f'{every} {unit}s' if days is None else f'{every*days:g} days'
    ss.note(f'the results of "{name}" are kept by group every {span} (a flow added up over them), since every {unit} would take {npts*row_bytes/1e6:,.0f} MB; '
            f'name the results to keep by group with results=[...] to keep them more often', sim=sim)
    return rows, ends


def people_results(sim):
    """ In a compartmental run, the results that agents give the sim (alive, new_deaths, new_emigrants, cum_deaths), from the declared states and flows """
    mods = [mod for mod in sim.declared if mod.graph.people_states]
    if sim.solver in [None, 'abm'] or 'alive' in sim.results or not mods:
        return
    first = mods[0] # Each disease's states hold everyone
    kw = dict(module='People', timevec=sim.t.timevec, scale=False)
    alive = sum(np.asarray(first.results[state.name].values, dtype=float) for state in first.graph.people_states)
    gone = dict(deaths=0, emigrants=0)
    for mod in mods:
        for tr in mod.graph.transitions.values():
            if tr.deaths is not None and tr.deaths in mod.results:
                gone['emigrants' if getattr(tr, 'migration', False) else 'deaths'] += np.asarray(mod.results[tr.deaths].values, dtype=float)
    deaths, emigrants = [np.broadcast_to(gone[key], alive.shape).copy() for key in ['deaths', 'emigrants']]
    sim.results += [
        ss.Result('alive', label='Number alive', values=alive, kind='stock', **kw),
        ss.Result('new_deaths', label='Deaths', values=deaths, **kw),
        ss.Result('new_emigrants', label='Emigrants', values=emigrants, **kw),
        ss.Result('cum_deaths', label='Cumulative deaths', values=np.cumsum(deaths), **kw),
    ]
    return


def by_level(levels, n):
    """ The positions of the entries at each of n levels, as where each level's start and end, and the positions in order (`np.flatnonzero(levels == i)` for each level, by one sort) """
    order = np.argsort(levels, kind='stable')
    return np.searchsorted(levels[order], np.arange(n + 1)), order


def no_set_count(mod, name):
    """ Setting the number of people in a state (`sir.I.n = 0`), which would make or lose people: they are moved instead (`ss.move()`), so the number of people stays the same in every disease """
    raise ss.CapabilityError(f'{mod}.{name}.n is the number of people in {name}, which is read, not set: setting it would make or lose people.\n'
                             f'     Move them with ss.move(), e.g. ss.move({mod}.{name}, <the state they go to>, n=...); people come into the model and leave it by births and deaths, which are declared.')


class CountState:
    """
    A state as it is at run time when it holds a count rather than agents

    Args:
        engine (Engine): the engine that holds the values
        state (`ss.State`): the declared state
    """
    def __init__(self, engine, state):
        self.engine = engine
        self.state = state
        self.name = state.name
        return

    def __repr__(self):
        return f'CountState({self.name}, n={self.n:n})'

    @property
    def n(self):
        """ The number in this state """
        return self.engine.count([self.state])

    @n.setter
    def n(self, value):
        if self.state.people:
            no_set_count(self.engine.mod.name, self.name)
        self.engine.set_count(self.state, value)
        return

    @property
    def frac(self):
        """ The number in this state as a fraction of the module's people """
        return self.engine.frac([self.state])

    def count(self):
        """ The number in this state, as `.n` (the same spelling as agents have, where count() is the number of agents) """
        return self.n

    @property
    def values(self):
        """ The number in this state in each group """
        return self.engine.count([self.state], total=False)

    def no_agents(self, what):
        """ A read of agents, which a compartmental run does not have (TC3: a module that reads each person's state, run beside the compartments) """
        raise ss.CapabilityError(f'{self.engine.mod.name}.{self.name}{what} was asked for, but a compartmental run has the number in each state ({self.engine.mod.name}.{self.name}.n), not agents.\n'
                                 f'     Run with solver="abm"; a module written as code that reads agents can say so with compartmental = False, so that solver="auto" runs them.')

    @property
    def uids(self):
        """ Agents, which a compartmental run does not have """
        self.no_agents('.uids')

    @property
    def raw(self):
        """ Each agent's value, which a compartmental run does not have """
        self.no_agents('.raw')

    def __getitem__(self, key):
        self.no_agents('[...]')

    def true(self):
        self.no_agents('.true()')

    def false(self):
        self.no_agents('.false()')


class Evaluating:
    """
    What an engine is evaluating for, set for a block of code and restored after it, whatever happens in it (see `Engine.evaluating()`)

    The flags are the engine's attributes of the same names with a leading underscore: `by_group` (rates for each group rather than
    for each agent), `uids` and `for_uids` (the agents, and whether each value is for them), `strain`, `route` and `routes`, `rtr`,
    `src`, `leaving` and `drawing`; what each means is said where the engine reads it.
    """
    __slots__ = ['engine', 'flags', 'saved']

    def __init__(self, engine, flags):
        self.engine = engine
        self.flags = flags
        return

    def __enter__(self):
        engine = self.engine
        self.saved = [(key, getattr(engine, key, None)) for key in self.flags]
        for key,val in self.flags.items():
            setattr(engine, key, val)
        return engine

    def __exit__(self, *args):
        for key,val in self.saved:
            setattr(self.engine, key, val)
        return False


class Telling:
    """ The notes and warnings about a layout (of an engine, or of the mixing), which are held while it is made to choose the solver (see `ss.solvers.choose.lay_out()`), and said only once it is used """
    unsaid = None # The notes and warnings held, as calls; None to say them at once

    def tell(self, func, *args, **kwargs):
        """ Say a note (`ss.note`) or warning (`ss.warn`) about the layout, or hold it until it is used (see `say()`) """
        if self.unsaid is None:
            func(*args, **kwargs)
        else:
            self.unsaid.append((func, args, kwargs))
        return

    def say(self):
        """ Say the notes and warnings held from making the layout, now that it is used """
        unsaid, self.unsaid = self.unsaid or [], None
        for func,args,kwargs in unsaid:
            func(*args, **kwargs)
        return


class Effects:
    """
    What the engine of every disease gives the modules that act on it, whether the disease is declared or written as code: the modifiers
    and products active on its quantities (`mult_raw()`, `contact_mult()`), for the people their conditions select (`selector`), in the sim's unit

    Args:
        module (`ss.Module`): the module
        sim (`ss.Sim`): the sim
    """
    cross = None # In a compartmental run, the diseases whose states are tracked jointly with this one's, if any: see ss.solvers.Crossing

    def __init__(self, module, sim):
        self.mod = module
        self.sim = sim
        self.cache = dict() # Values for the current step
        self.converted = dict() # Typed values converted to the sim's unit: see convert()
        return

    @property
    def dt(self):
        """ The module's timestep, in the sim's unit (worked out again only if the timestep is replaced) """
        dt = self.mod.t.dt
        cached = self.__dict__.get('_dt')
        if cached is None or cached[0] is not dt:
            cached = self._dt = (dt, dt.to(self.sim.unit_base).value)
        return cached[1]

    def convert(self, val):
        """ Convert a typed quantity to a plain number in the sim's unit: a duration to units, a rate to per unit; each is converted once, unless its number is changed """
        return convert(val, self.sim.unit_base, self.converted)

    @property
    def selector(self):
        """ Evaluates conditions for this module: see `ss.conditions.Selector` """
        if getattr(self, '_selector', None) is None:
            self._selector = ss.conditions.Selector(self.sim, self.mod)
        return self._selector

    def contact_mult(self, name):
        """ The multiplier on the contacts of a network from the modifiers of its `n_contacts` (e.g. `ss.modify('school.pars.n_contacts', multiply=0.2)`): 1 if it has none """
        mods = self.sim.modifiers.get((name, 'par', 'n_contacts')) if name is not None else None
        if not mods:
            return 1.0
        key = ('contacts', name, None)
        if key not in self.cache:
            self.cache[key] = ss.connectors.contact_mult(self, name, mods)
        return self.cache[key]

    def clip(self, value, path):
        """ A modified rate, duration, or multiplier that has gone below 0 (e.g. by add=) is 0; the first time, a note says so, and sim.what_modifies() reports it """
        if isinstance(value, dict): # A value for each route, e.g. beta=dict(random=..., sexualnet=[...])
            return {k:[self.clip(x, path) for x in v] if isinstance(v, (list, tuple)) else self.clip(v, path) for k,v in value.items()}
        if np.any(np.asarray(value) < 0):
            if path not in self.sim.clipped:
                self.sim.clipped.add(path)
                ss.note(f'the modifiers of {path} made it negative, so it was set to 0 where it was', sim=self.sim)
            value = np.maximum(value, 0)
        return value

    def reduce(self, value, state=None, what=None):
        """ A modified value as the engine uses it; see the subclasses """
        return value

    def current_par(self, name, uids=None):
        """ A parameter with the modifiers active on it, in its own type: one value, or, if they differ between people, one for each of `uids` (see `ss.Disease.current_value()`) """
        raw = self.mod.pars[name]
        raw = raw.raw if isinstance(raw, ss.par) else raw
        mods = self.sim.modifiers.get((self.mod.name, 'par', name))
        value = self.convert(raw)
        if mods:
            value = self.clip(self.reduce(ss.connectors.combine(mods, value, self), what=name), f'{self.mod.name}.pars.{name}')
        if np.ndim(value) and uids is not None:
            value = np.asarray(value)[uids]
        return unconvert(raw, value, self.sim.unit_base)

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on a quantity that is not a parameter (`rel_sus`, `rel_trans`, or a transition's rate) from the modifiers active on it, at least 0 (1 if there are none) """
        mods = self.sim.modifiers.get((self.mod.name, kind, name))
        if not mods:
            return 1.0
        return self.clip(self.reduce(ss.connectors.combine(mods, 1.0, self), state=state, what=name or kind), f'{self.mod.name}.{name or kind}')


class Engine(Telling, Effects):
    """
    Base class for the engines that run a graph

    Args:
        module (`ss.Module`): the module whose graph is run
        sim (`ss.Sim`): the sim
        solver (str): the solver it is laid out for, if not the sim's (the ODE, when it is laid out to choose the solver: see `ss.solvers.choose.lay_out()`)
    """
    max_map_bytes = 2e9 # The most memory a map between groups may take in a compartmental run (see dim_map())
    group_bytes = 100e6 # The most memory that a module's results may take by group in one run; above it, they are kept every few steps (see ss.solvers.group_rows())

    def __init__(self, module, sim, solver=None):
        super().__init__(module, sim)
        self.solver = solver or sim.solver
        self.graph = module.graph
        self.fixed = dict() # Converted values of plain parameters, which hold for the whole run
        self.arities = dict() # The number of arguments of each user function: see n_args()
        self.held_series = dict() # Values that vary in time, at the start of each step of the sim: see held_value()
        self.rng = self.make_rng() # For draws that are not per agent
        self.dwells = sc.objdict() # Dwell-time information for each transition that has one
        self.flowcounts = sc.objdict({name:0.0 for name in list(self.graph.transitions) + self.graph.destinations()}) # Events in the current step, and those of each destination of a transition with several ('cin.X')
        self.stocks = sc.objdict() # The number in each state at the start of the step
        self._by_group = False # Whether rates are being worked out for groups rather than for agents (in an agent-based run, for the states that are counts)
        self.strain = None # The position of the module's strain dimension, if it has one
        self.move_levels = dict() # For each transition with move=: see moves()
        self._strain = None # The strain whose force of infection is being worked out
        self._routes = None # For a transmission with a beta for each route, the beta of each term of the mixing: see Compartmental.route_betas()
        self._route = None # As agents, the route (its name and itself) a transmission's beta is being worked out for: see Agents.route_beta()
        self.routed = set() # In a compartmental run, the ids of the betas applied to the exposure through each route instead: see Compartmental.route_betas()
        self.partnership_info = dict() # Under the partner-change model, the chance a partnership transmits, by transmission, source state, network, and side of the partner: see Compartmental.partnership_chance()
        self.register = None # The movement register, as arrays for the kernels
        self.year = None # The year in the step that values that vary smoothly in time are taken at, for its stages (None for its start, sim.now)
        self.immune_states = [] # Protection between strains: the states whose people can be infected again, and the protection by level in a state (see init_immunity())
        self.protect = {}
        self.kernel_rng = None # The random numbers of a run stepped by the kernels: see init_program()
        self.compiled = None # Whether the run is taken by the compiled kernels or as plain Python, once chosen: see Compartmental.kernels()
        return

    def step_funcs(self):
        """ Call the module's custom per-step functions, after its transitions """
        for func in self.graph.funcs.values():
            func(self.sim)
        return

    def shrink(self):
        """ Drop the sim and the arrays the run worked in, for saving; the results are kept on the module """
        arrays = [key for key,val in vars(self).items() if isinstance(val, np.ndarray) and val.nbytes > 1000]
        ss.utils.shrink(self, ['sim', '_selector', 'cache', 'series', 'deliveries', 'program', 'register', 'shifts', 'jax_run'] + arrays) # The selector holds the sim, the cache the values per agent, and the series the tables of a compartmental run
        return

    def make_rng(self, stream=0, seed=None):
        """ The random numbers of the engine: stream 0 for draws made in Python, 1 for the kernels, 2 for the events of a register at the start; from the sim's seed (or the one given) and the module's name, so no two modules share them """
        seed = self.sim.pars.rand_seed if seed is None else seed
        return np.random.default_rng([int(seed or 0), zlib.crc32(self.mod.name.encode()), stream])

    #%% Dimensions

    def init_dims(self):
        """
        Work out which dimensions each state has

        A state that a dimension does not apply to (a count, or a state left out of
        `states=`) is kept at the first level of that dimension.
        """
        sim = self.sim
        graph = self.graph
        coords = sim.groups.group_coords
        own = {strat.name:strat for strat in graph.strats}
        for tr in graph.inflows: # A birth rate for each level of a property of people would be births from the people at each level, which cannot be declared yet
            if tr.rate is None: # Driven by a flow (ss.birth(per=))
                continue
            dims = [dim for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, (ss.by, ss.parameters.by_func)) for dim in leaf.raw.dims]
            others = [dim for dim in dims if sim.groups.strats[sim.groups.dim_index(dim)].kind != 'space']
            if tr.per_capita and others:
                errormsg = (f'The birth rate of "{tr.name}" is given for each level of {sc.strjoin(others)}, which would be births from the people at each level into the first state; that cannot be declared yet. '
                            f'Give one birth rate, or a total inflow for each level that people arrive at (ss.null >> {tr.dst[0].name} @ ss.by({others[0]}=[...])).')
                raise ValueError(errormsg)
        self.applies = sc.objdict() # For each state: whether each dimension applies to it
        self.state_split = sc.objdict() # The share of a state's initial people (or count) in each group
        self.inflow_splits = sc.objdict() # Where an inflow into each state arrives
        self.inflow_factors = sc.objdict() # The same, as a factor for each dimension
        self.inflow_local = sc.objdict() # The same, for an inflow whose rate is worked out for each place, so is already the inflow there
        self.collapse = sc.objdict() # The group that each group's members are counted in
        self.maps = dict()
        for state in graph.states.values():
            flags = []
            split = np.ones(sim.groups.n_groups)
            inflow = np.ones(sim.groups.n_groups)
            local = np.ones(sim.groups.n_groups)
            collapse = np.arange(sim.groups.n_groups)
            factors = []
            for d,strat in enumerate(sim.groups.strats):
                names = own[strat.name].states if strat.name in own else ([] if strat.states is not None else None)
                if getattr(strat, 'stages_of', None) is not None: # The stages of a transfer: for its own module's states only
                    names = (strat.states or [s.name for s in graph.people_states]) if strat.stages_of[0] == self.mod.name else []
                flag = bool(state.people or strat.kind == 'space') if names is None else state.name in names # A place applies to everything; other properties of people, to people
                if strat.kind == 'disease': # The compartments of another disease, tracked jointly with this one: see ss.solvers.Crossing
                    flag = bool(state.people) and self.mod.name in strat.holders
                levels = coords[:, d]
                first = levels == 0
                if flag:
                    split = split*strat.shares(coords)
                    share = first if strat.kind == 'age' else strat.shares(coords) # People are born into the youngest group
                    inflow = inflow*share
                    factors.append(share)
                    local = local*(1.0 if strat.kind == 'space' else share)
                else:
                    split = split*first
                    inflow = inflow*first
                    factors.append(first.astype(float))
                    local = local*first
                    collapse = collapse - levels*int(sim.groups.group_mults[d])
                flags.append(flag)
            if sim.groups.joint is not None and all(flags[d] for d in sim.groups.joint[0]): # A split over several dimensions at once replaces the product of their own splits
                ds, grid = sim.groups.joint
                product = np.prod([sim.groups.strats[d].split[coords[:, d]] for d in ds], axis=0)
                split = split*np.divide(grid[tuple(coords[:, d] for d in ds)], product, out=np.zeros(sim.groups.n_groups), where=product>0)
            self.applies[state.name] = np.array(flags, dtype=bool)
            self.state_split[state.name] = sim.groups.fold(split) # People in a group with no level of a dimension (outside its where=) are counted at its first level
            self.inflow_splits[state.name] = sim.groups.fold(inflow)
            self.inflow_factors[state.name] = factors
            self.inflow_local[state.name] = sim.groups.fold(local)
            self.collapse[state.name] = sim.groups.folded(collapse)
        strains = [d for d,strat in enumerate(sim.groups.strats) if strat.kind == 'strain' and strat.name in own]
        if len(strains) > 1:
            errormsg = f'"{self.mod.name}" has more than one strain dimension ({sc.strjoin(sim.groups.strats[d].name for d in strains)}); a person carries one strain of a disease at a time, so combine them into one'
            raise ValueError(errormsg)
        self.strain = strains[0] if strains else None
        self.foreign = set() # In a crossing: the dimensions that only another disease's states have, whose level this one's people keep
        if self.cross is not None:
            people = [state.name for state in graph.people_states]
            self.foreign = {d for d,strat in enumerate(sim.groups.strats) if strat.kind != 'disease' and not any(self.applies[name][d] for name in people)}
        return

    def moves(self, tr):
        """ The change of level of a transition with `move=` or `increment=`, as ((dimension, level), ...), where the level of a counter is ('+', amount); or None """
        if tr.move is None and getattr(tr, 'increment', None) is None:
            return None
        return self.own_moves(tr) or None

    def own_moves(self, tr):
        """ The moves of `move=` and `increment=`: see `moves()` """
        if tr.name not in self.move_levels:
            out = []
            for dim,level in list((tr.move or {}).items()) + [(dim, ('+', k)) for dim,k in (tr.increment or {}).items()]:
                if dim not in [strat.name for strat in self.sim.groups.strats]:
                    example = f'stratify({dim}=list(range(8)))' if isinstance(level, tuple) else 'stratify(...)'
                    raise sc.KeyNotFoundError(f'the transition "{tr.name}" of "{self.mod.name}" moves people in "{dim}", which is not a dimension of the sim. Stratify the disease or the sim by it, e.g. sim.{example}.')
                d = self.sim.groups.dim_index(dim)
                for state in tr.dst:
                    if not self.applies[state.name][d] and not (isinstance(level, tuple) and level[0] == 'matrix'): # A change of strains for each strain carried, for those left with one (see ss.strains.prepare())
                        raise sc.KeyNotFoundError(f'the transition "{tr.name}" of "{self.mod.name}" moves people in "{dim}", but the state they go to ("{state.name}") has no level of it.')
                strat = self.sim.groups.strats[d]
                if isinstance(level, tuple) and level[0] == '+':
                    self.note_counter(strat, f'the transition "{tr.name}" of "{self.mod.name}"')
                out.append((d, level if isinstance(level, tuple) else strat.move_level(level)))
            self.move_levels[tr.name] = tuple(out)
        return self.move_levels[tr.name]

    def dim_map(self, src, dst, strain=None, record=False, move=None):
        """
        Where the people who leave one state land in another, when the two do not have the same dimensions

        Returns None if they stay in their own group, or else a matrix of groups by groups:
        a dimension the destination lacks is dropped, one that only the destination has is
        entered in proportion to its split, and a new infection takes the strain of its source.

        Args:
            src (str): the state they leave (None for an inflow)
            dst (str): the state they enter
            strain (int): for a new infection, the strain of the people it came from
            record (bool): if True, give the groups the flow is recorded in (the source's, with the strain acquired)
            move (tuple): for a transition with `move=`, the level of each dimension people move to: see `moves()`
        """
        key = (src, dst, strain, record, move)
        if key not in self.maps:
            a = self.applies[sc.ifelse(src, dst)]
            b = self.applies[dst]
            moved = dict(move or ())
            out = None
            if strain is not None or moved or not np.array_equal(a, b):
                ng = self.sim.groups.n_groups
                if self.solver != 'abm' and 8*ng**2 > self.max_map_bytes: # A dense matrix of groups by groups (FP84: 25.7 GB for 3,264 × 18 groups)
                    raise ss.CapabilityError(f'"{self.mod.name}" cannot run with solver="{self.solver}": people moving from {src or "outside"} to {dst} change group, and the map between its {ng:,} groups would take {8*ng**2/1e9:.3g} GB (a dense matrix of groups by groups).\n'
                                             f'     Run with solver="abm", or with fewer groups (fewer dimensions, or fewer diseases tracked jointly).')
                out = np.ones((1, 1))
                for d,strat in enumerate(self.sim.groups.strats):
                    n = strat.n
                    factor = np.zeros((n, n))
                    if d in moved and isinstance(moved[d], tuple) and moved[d][0] == 'matrix' and b[d]: # A share of each level to each (see ss.strains.prepare())
                        factor = np.array(moved[d][1])
                    elif d in moved and isinstance(moved[d], tuple) and moved[d][0] != 'matrix': # A counter (each level to the one k up), or a change of strains
                        factor[np.arange(n), strat.target(moved[d])] = 1
                    elif d in moved and not isinstance(moved[d], tuple):
                        factor[:, moved[d]] = 1
                    elif d == self.strain and strain is not None and (b[d] or record):
                        if a[d] and not record and strat.carries is not None and (strat.carried is None or src in strat.carried): # Someone infected with other strains carries this one as well
                            factor[np.arange(n), strat.adds[:, strain]] = 1
                        else:
                            factor[:, strain] = 1
                    elif d in self.foreign:
                        factor = np.eye(n)
                    elif not b[d]:
                        factor[:, 0] = 1
                    elif a[d]:
                        factor = np.eye(n)
                    else:
                        factor[:] = strat.split
                    out = np.kron(out, factor)
                coords = self.sim.groups.group_coords
                for d,strat in enumerate(self.sim.groups.strats): # Entering a dimension whose split depends on the levels of others: by the row for the group people come from
                    if strat.cond is not None and b[d] and not a[d]:
                        ng = len(coords)
                        pairs = np.repeat(coords[:, None, :], ng, axis=1) # From each group (rows) to each group (columns), with the level of this dimension from the column
                        pairs[:, :, d] = coords[None, :, d]
                        cond = strat.shares(pairs.reshape(ng*ng, -1)).reshape(ng, ng)
                        split = np.broadcast_to(strat.split[coords[:, d]], (ng, ng))
                        out = out*np.divide(cond, split, out=np.zeros((ng, ng)), where=split>0) # The kron above entered each level in proportion to the split overall
                out = self.sim.groups.fold_matrix(out)
            self.maps[key] = out
        return self.maps[key]

    def inflow_split(self, tr):
        """ Where an inflow arrives: people are born into the youngest group, and a count into the groups of its own dimensions """
        name = tr.dst[0].name
        if tr.rate is None: # Driven by a flow (ss.birth(per=)), whose children are placed by their parents' groups (see Compartmental.birth_map())
            return self.inflow_splits[name]
        if self.sim.mixing.space is not None and any(isinstance(leaf, ss.expressions.Count) for leaf in tr.rate.leaves): # A rate that uses the number in a place gives the inflow in that place
            return self.inflow_local[name]
        given = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, (ss.by, ss.parameters.by_func))]
        if given: # A rate given for each level of some dimensions is the inflow at that level, so is shared out over the other dimensions only
            dims = {self.sim.groups.dim_index(dim) for by in given for dim in by.dims}
            factors = self.inflow_factors[name]
            return self.sim.groups.fold(np.prod([np.ones(len(factor)) if d in dims and self.applies[name][d] else factor for d,factor in enumerate(factors)], axis=0)) if factors else self.inflow_splits[name]
        return self.inflow_splits[name]

    def init_immunity(self):
        """
        The protection between the strains of a disease with shared states (`ss.Strains(immunity=...)`, see `ss.immunity`)

        - Due to a state (`due_to=R`): its people are protected by the strains they carry in it, against each
          strain; a multiplier for each level of the strain dimension and strain, in `protect`. A state people
          are not infected in (R) can be infected again, moving on as from the first state (`immune_states`);
          one they are infected in (L) can carry another strain as well (see `ss.strains.add_superinfection()`).
        """
        self.immune_states = []
        self.protect = {}
        strat = None if self.strain is None else self.sim.groups.strats[self.strain]
        strains = getattr(strat, 'strains', None)
        if strains is None or not strains.immunity:
            return
        graph = self.graph
        reason = strains.needs_agents(graph)
        if reason and self.solver != 'abm':
            raise ss.CapabilityError(f'the strains of "{self.mod.name}" need agents: {reason}. Run with solver="abm" (or "auto").')
        sources = {id(src) for tr in graph.transmissions for src in tr.src}
        infected = {id(s) for s in graph.people_states if s.name in (strat.carried or [])} # Carrying strains as an infection (see ss.Strains._carriers())
        for imm in strains.immunity:
            for name in imm.names:
                state = graph.states[name]
                self.protect[name] = self.protect.get(name, 1.0)*(1 - strains.level_protection(imm)) # On every transmission out of the state (e.g. a reinfection R >> I of its own, HP46)
                if id(state) not in infected and id(state) not in sources and state not in self.immune_states:
                    self.immune_states.append(state)
        return

    def protection(self, state, levels, strain):
        """ The share of the force of infection of a strain that reaches people in a state at some levels of the strain dimension, by the strains they carry there (see `init_immunity()`) """
        table = self.protect.get(state.name)
        return 1.0 if table is None else table[levels, strain]

    def strain_effect(self, strain, state=None):
        """ The multiplier on the force of infection of one strain from the products that protect against acquiring it (`ss.by(strain=...)`): for each agent, or for each group of a state """
        out = 1.0
        for effect in self.sim.modifiers.get((self.mod.name, 'strain_sus', None)) or []:
            val = effect.for_strain(strain).apply(1.0, self)
            out = out*(val if state is None else self.reduce(val, state=state, what='strain_sus'))
        return out

    def enters(self, tr, src):
        """ Whether a transition with one destination takes people into a dimension that the state they leave does not have, so is recorded at the destination's level """
        if src is None or tr.dst is None or len(tr.dst) != 1:
            return False
        return bool(np.any(self.applies[tr.dst[0].name] & ~self.applies[src.name]))

    def strain_levels(self):
        """ The strains of the module's strain dimension, if it has one: each has a force of infection of its own """
        if self.strain is None:
            return [None]
        strat = self.sim.groups.strats[self.strain]
        return list(range(strat.n)) if strat.carries is None else [int(j) for j in strat.base]

    @property
    def carriage(self):
        """ The strain dimension, if its levels include people who carry several strains at once (see `ss.dimensions.Strat.init_carriage()`) """
        strat = None if self.strain is None else self.sim.groups.strats[self.strain]
        return strat if strat is not None and strat.carries is not None else None

    def init_strain_weights(self):
        """
        The share of the infectiousness of the people at each level of the strain dimension that goes to each strain (levels by strains)

        Someone who carries one strain passes it on; someone who carries several passes on
        each in proportion to its infectiousness (its `rel_trans`); their own infectiousness is that of their level.
        """
        self.strain_weights = None
        if self.strain is None:
            return
        strat = self.sim.groups.strats[self.strain]
        if strat.carries is None:
            self.strain_weights = np.eye(strat.n)
            return
        rel = np.ones(strat.n)
        for d,vals in self.rel_adjust['rel_trans']:
            if d == self.strain:
                rel = rel*vals
        w = strat.carries*rel[None, :]
        self.strain_weights = np.divide(w, w.sum(axis=1, keepdims=True), out=np.zeros_like(w), where=w.sum(axis=1, keepdims=True)>0)
        return

    def can_take(self, levels, strain):
        """ Whether people at some levels of a strain dimension whose levels carry several strains are changed by infection with a strain: if they do not carry it, and there is a level for carrying it as well """
        return self.carriage.adds[levels, strain] != levels

    @staticmethod
    def superinfects(tr, state):
        """ Whether a transmission leaves people in the state they are in (so only changes the strains they carry) """
        return len(tr.dst) == 1 and tr.dst[0] is state

    @property
    def ti(self):
        """
        The time point the engine takes people's state at: the start of the step being taken (`self.mod.ti - 1`,
        since the step's own time point, the loop's `ti`, is its end), and the start of the run before the first step.
        So people who move at the start of a step (as all but infections do) are timed from then, and infections from its end;
        either way their time of entering (`ti_X`) reads as the step's index (see `ss.solvers.Agents.index_of()`).
        """
        ti = self.__dict__.get('_ti') # Set while people are back-dated: see Agents.backfill()
        return max(self.mod.t.ti - 1, 0) if ti is None else ti

    @property
    def total_pop(self):
        return self.sim.pars.total_pop

    #%% Values

    def value(self, p):
        """ The value of a parameter, as a number in the sim's unit """
        if isinstance(p.raw, ss.expressions.Expr): # E.g. a dwell time given as an expression of attributes: for whoever it is worked out for, so not cached
            return self.eval(p.raw)
        route = self._route is not None and (isinstance(p.raw, (dict, ss.peract)) or p.slot == 'rate') # A beta, worked out for one route
        key = (id(p), self._route[0]) if route else id(p)
        if key in self.cache:
            if route and getattr(self._route[1], 'has_acts', False):
                self._per_act = True
            return self.cache[key]
        raw = p.raw
        if route:
            out = self.route_value(p, raw)
        elif isinstance(raw, ss.parameters.by_func): # A value for each level of a dimension, from a function of the sim (MS2)
            out = self.by_levels(raw)
            out = self.by_group(out) if isinstance(out, ss.by) else out
        elif isinstance(raw, ss.Dist): # Not a dwell time (those are handled separately), so use the mean
            out = ss.transitions.dwell_stats(raw, self.convert)[0]
        elif callable(raw):
            out = self.call(raw, p)
            if isinstance(out, ss.expressions.Expr): # A function that uses the model's own states (I.n) gives an expression, which is worked out here
                self.cache.setdefault('held', []).append(out) # Values are cached by id, so the expression's parts are kept until the cache is cleared, so that no new object can take their id
                out = self.eval(out)
            elif p.slot in ['rate', 'dur'] and not isinstance(out, ss.TimePar) and not isinstance(raw, ss.parameters.TimeValue) and self.inferred_unit: # An expression has the units of its own values
                self.bare_return(p, raw, out)
            return self.modified(p, self.convert(out)) # Not cached, since it can depend on the current state
        elif isinstance(raw, dict) and self._routes is None and not len(self.sim.networks): # A beta for each route, under homogeneous mixing: the value of the routes it does not name
            if 'default' not in raw:
                raise ValueError(f'beta for "{self.mod.name}" has a value for each of the routes {sc.strjoin(raw.keys())}, but there are none (homogeneous mixing); give default= for the routes it does not name, or give the sim its networks')
            out = self.plain_value(raw['default'])
        elif isinstance(raw, dict): # A beta for each route; in a compartmental run it is applied to the exposure through each instead (see Compartmental.route_betas())
            out = 1.0 if self._routes is not None else {k:self.convert_beta(v) for k,v in raw.items()}
        elif self._routes is not None and key in self.routed: # A beta applied to the exposure through each route (see Compartmental.route_betas())
            out = 1.0
        elif isinstance(raw, ss.by):
            out = self.by_group(raw)
        else:
            fixed = self.fixed.get(key) # A plain value is converted once, and reused for as long as the parameter holds the same object with the same number in it
            number = raw.value if isinstance(raw, ss.TimePar) else None # A typed quantity's number can be changed in place
            if fixed is None or fixed[0] is not raw or changed(fixed[1], number):
                fixed = self.fixed[key] = (raw, np.copy(number) if isinstance(number, np.ndarray) else number, self.convert(raw))
            out = fixed[2]
        out = self.modified(p, out)
        self.cache[key] = out
        return out

    def modified(self, p, value):
        """ Apply the modifiers that are active on a parameter: set, then multiply, then add """
        mods = self.sim.modifiers.get((self.mod.name, 'par', p.name)) if (p.name and not p.literal) else None
        if not mods or getattr(self, '_unmodified', False): # See note_steps()
            return value
        out = self.reduce(ss.connectors.combine(mods, value, self), what=p.name)
        return self.clip(out, f'{self.mod.name}.pars.{p.name}') if p.slot in ['rate', 'dur'] else out

    @property
    def net_mods(self):
        """ The modifiers of the contacts of networks """
        if getattr(self, '_net_mods', None) is None:
            nets = self.sim.all_networks()
            self._net_mods = [m for (modname,kind,name),mods in self.sim.modifiers.items() if modname in nets and name == 'n_contacts' for m in mods]
        return self._net_mods

    def branch_mult(self, tr, uids=None):
        """ The multiplier on a branch's probability from its modifiers (for some agents, if `uids` are given), or None if it has none """
        return self.branch_mult_of(tr.name, uids) if tr is not None else None

    def where_mults(self, branches, state, mults, uids=None):
        """
        The branches' multipliers with their where=: a branch is closed to the people (or groups) it does not hold for, so the
        others share the rest; but if every branch is closed to someone, they are split as before and wait until one opens
        """
        ws = [self.where_share(tr, state, uids) if tr is not None and getattr(tr, 'where', None) is not None else None for tr in branches]
        if all(w is None for w in ws):
            return mults
        n = len(uids) if uids is not None else self.n_groups
        open_any = np.logical_or.reduce([np.ones(n, dtype=bool) if w is None else np.asarray(w) > 0 for w in ws])
        out = list(mults)
        for i,w in enumerate(ws):
            if w is not None:
                w = np.where(open_any, w, 1.0)
                out[i] = w if out[i] is None else out[i]*w
        return out

    def where_share(self, tr, state, uids=None):
        """ For a branch with where=: 1 for the people it is open to and 0 for the others (agents: for `uids`; compartments: for each group of `state`, if the condition is on states and strata), or None """
        sel = self.selector
        if sel.abm:
            mask = np.asarray(sel.select(tr.where))
            return (mask[uids] if uids is not None else mask).astype(float) # For the agents asked for only
        if not ss.conditions.is_static(tr.where, self.sim):
            return None
        frac = np.broadcast_to(np.asarray(sel.select(tr.where), dtype=float), (len(sel.states), self.n_groups))
        return np.clip(frac[sel.states.index(getattr(state, 'name', state))], 0, 1)

    def branch_mult_of(self, name, uids=None):
        """ As `branch_mult()`, by name: a transition, or one destination of a transition with several (`'progression.C'`); with the adjust= of the stratifications on a branch """
        adjust = self.branch_adjust.get(name)
        if not self.sim.modifiers.get((self.mod.name, 'branch', name)):
            if adjust is None:
                return None
            out = 1.0
        else:
            out = self.mult_raw('branch', name)
            if uids is not None and np.ndim(out):
                out = out[uids]
        if adjust is not None: # By group, as an agent engine's modifiers are by agent
            out = out*(adjust if uids is None else adjust[self.sim.groups.group_raw()[uids]])
        return out

    def dest_shares(self, tr, ps, uids=None, stay=0.0):
        """ The probability of each destination of a transition with several, with the modifiers of each (`'progression.C'`), as `branch_shares()` does for branches; `stay` is the probability of `ss.stay`, if given """
        mults = [self.branch_mult_of(f'{tr.name}.{dst.name}', uids) for dst in tr.dst]
        return self.branch_shares([f'{tr.name}.{dst.name}' for dst in tr.dst], ps, mults, stay=stay)

    def branch_shares(self, trs, ps, mults, stay=0.0):
        """
        The probability of each branch out of a state, with their modifiers

        A modifier on a branch multiplies its probability, and the branches with none share
        what is left in proportion to their own; if every branch is modified (or the modified
        ones come to more than 1), they are scaled to sum to 1.

        Args:
            trs (list): the transitions (or their names)
            ps (list): the probability of each, or None for the one that takes the rest
            mults (list): the multiplier of each, or None for those with no modifiers
            stay (float): the probability of `ss.stay` among the destinations, if given, which the one that takes the rest does not get
        """
        given = sum(p if m is None else np.where(np.asarray(m) == 0, 0.0, p) for p,m in zip(ps, mults) if p is not None) + stay # A branch its modifiers rule out (e.g. one for another level of a move= arrow) does not count
        if np.any(np.asarray(given) > 1 + 1e-9) or any(np.any(np.asarray(p) < 0) for p in ps if p is not None): # TL13: was used as 1 (or as a negative remainder) with no note
            names = sc.strjoin([getattr(tr, 'name', tr) for tr,p in zip(trs, ps) if p is not None])
            raise ValueError(f'the branch probabilities of {self.mod.name}.{names} come to as much as {np.max(given):n}, more than 1 (or one is negative); each must be between 0 and 1, and together at most 1')
        rest = 1.0 - given
        ps = [rest if p is None else p for p in ps]
        if all(m is None for m in mults):
            return ps
        new = [None if m is None else np.maximum(p*m, 0) for p,m in zip(ps, mults)]
        taken = sum(p for p in new if p is not None)
        others = sum(p for p,m in zip(ps, mults) if m is None)
        if np.any(np.asarray(taken) > 1 + 1e-12):
            path = f'{self.mod.name}.{sc.strjoin([getattr(tr, "name", tr) for tr,m in zip(trs, mults) if m is not None])}'
            if path not in self.sim.clipped:
                self.sim.clipped.add(path)
                ss.note(f'the modifiers of {path} made the probabilities of the branches come to more than 1, so they were scaled to sum to 1', sim=self.sim)
        denom = np.where(np.asarray(others) > 0, np.maximum(taken, 1.0), taken) # What the modified ones are divided by
        left = 1.0 - np.minimum(taken, 1.0)
        div = lambda a, b: np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=np.asarray(b) > 0)
        out = []
        for p,q in zip(ps, new):
            if q is None:
                out.append(div(p*left, others))
            else:
                out.append(div(q, denom))
        return out

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on a quantity that is not a parameter (`rel_sus`, `rel_trans`, or a transition's rate) from its modifiers (once a step), and from the adjust= of the stratifications """
        out = 1.0
        if self.sim.modifiers.get((self.mod.name, kind, name)):
            key = (kind, name, state.name if state is not None else None)
            if key not in self.cache:
                with self.evaluating(for_uids=False): # For everyone, even when asked while a rate is evaluated for some agents (CV23)
                    self.cache[key] = super().mult_raw(kind, name, state)
            out = self.cache[key]
        if getattr(self, 'rel_adjust', None) and self.rel_adjust.get(kind):
            out = out*self.adjusted_rel(kind, state)
        if kind in self.__dict__.get('rel_scale', ()): # A strain with its own course: its value (see ss.Strains)
            out = out*self.rel_scale[kind]
        return out

    def adjusted_rel(self, kind, state=None):
        """ The multiplier on rel_trans or rel_sus for each group in a state, from the adjust= of the stratifications: see the subclasses """
        out = 1.0
        for d,vals in self.rel_adjust[kind]:
            if state is None or self.applies[state.name][d]:
                out = out*self.sim.groups.group_vals(d, vals)
        return out

    def by_levels(self, bf):
        """ The values of `ss.by(dim=fn)` (`ss.parameters.by_func`) this step, as a number or an `ss.by`, once a step """
        key = ('by_func', id(bf))
        if key not in self.cache:
            self.cache[key] = bf.levels(self.sim, self.convert)
        return self.cache[key]

    def by_group(self, by):
        """ Expand a value for each level of one or more dimensions to a value for each group, once a step """
        key = ('by', id(by))
        if key not in self.cache:
            if by.varies and (self.year is None or ss.solvers.Compartmental.held(by)): # At the start of the step: worked out for every step at once, as in call()
                self.cache[key] = np.array(self.held_value(by))
            else:
                self.cache[key] = by.per_group(self.sim, self.convert, year=None if ss.solvers.Compartmental.held(by) else self.year) # A step function at the start of the step, as in call()
        return self.cache[key]

    def check_counted_strains(self):
        """
        With strains, an acquisition whose rate counts the disease's own states (`beta*I.frac`, `beta*I.n`) would give each new
        case a strain from the strain set's split, not from who infected them, since a count carries no strain: refused, with
        the transmission that says the same (`beta*I` is the same force under homogeneous mixing, weighted by infectiousness)
        """
        if self.strain is None:
            return
        for tr in self.graph.acquisitions:
            if tr.contact or not isinstance(tr.rate, ss.expressions.Expr):
                continue
            for leaf in tr.rate.leaves:
                states = getattr(leaf, 'states', None) if isinstance(leaf, ss.expressions.Count) else None
                own = [s for s in (states or []) if isinstance(s, ss.transitions.State) and self.graph.states.get(s.name) is s]
                if own:
                    names = sc.strjoin(s.name for s in own)
                    errormsg = (f'the rate of "{tr.name}" counts {names} of "{self.mod.name}" ({leaf}), which has strains: a count carries no strain, so each new case would take a strain from the split rather than from who infected them.\n'
                                f'     Write it as a transmission, which has the same force under homogeneous mixing and passes on the strain: beta*I for beta*I.frac (and for beta*I.n, beta times the population size, times I).')
                    raise ValueError(errormsg)
        return

    def init_adjust(self):
        """ The multiplier on each transition in each group, from the adjust= of every stratification """
        sim = self.sim
        self.check_counted_strains()
        self.adjust = sc.objdict({name:None for name in self.graph.transitions})
        self.branch_adjust = dict() # For a branch chosen with p=: a multiplier on its probability, as for a modifier on it (TL16: the dwell time was divided instead)
        self.adjust_dims = sc.objdict({name:set() for name in self.graph.transitions}) # The dimensions each transition's multiplier is by
        self.strain_adjust = sc.objdict({name:None for name in self.graph.transitions}) # For a transmission: a multiplier for each strain, which is the strain of the person it comes from
        self.rel_adjust = dict(rel_trans=[], rel_sus=[]) # Multipliers on infectiousness and susceptibility, as (dimension, value for each level), for the states the dimension applies to
        for strat in self.graph.strats + [s for s in sim.groups.sim_strats if s not in self.graph.strats]:
            d = sim.groups.dim_index(strat.name)
            for key,vals in strat.adjust.items():
                modname, _, name = key.rpartition('.') # A sim's stratification can name a transition with its module, e.g. 'flu.infection'
                if modname not in ['', self.mod.name]:
                    continue
                if name in self.rel_adjust:
                    states = self.graph.infectious if name == 'rel_trans' else [src for tr in self.graph.acquisitions for src in tr.src]
                    if not any(self.applies[state.name][d] for state in states):
                        raise ValueError(f'the adjustment of {name} by "{strat.name}" cannot apply, since the people it is for ({sc.strjoin(s.name for s in states)}) have no level of "{strat.name}".')
                    self.rel_adjust[name].append((d, vals))
                elif name in self.adjust:
                    tr = self.graph.transitions[name]
                    if tr.trigger is not None:
                        raise sc.KeyNotFoundError(f'the adjustment of "{name}" by "{strat.name}" cannot apply, since "{name}" is triggered by {tr.trigger} and has no rate to scale; use where= on the transition, or a p= branch.')
                    if tr.src and not any(self.applies[src.name][d] for src in tr.src) and not (d == self.strain and tr.contact):
                        raise ValueError(f'the adjustment of "{name}" by "{strat.name}" cannot apply, since the people it moves ({sc.strjoin(src.name for src in tr.src)}) have no level of "{strat.name}". '
                                                   f'To make where they arrive depend on it, give "{strat.name}" a split instead.')
                    if not tr.src and tr.dst and not any(self.applies[dst.name][d] for dst in tr.dst): # An inflow into states with no level of it: all of it would arrive at the first level, at that level's multiplier
                        raise ValueError(f'the adjustment of "{name}" by "{strat.name}" cannot apply, since what it brings in ({sc.strjoin(dst.name for dst in tr.dst)}) has no level of "{strat.name}". '
                                                   f'Stratify {sc.strjoin(dst.name for dst in tr.dst)} by "{strat.name}" too (e.g. kind=\'space\' for a place), or leave the adjustment out.')
                    if d == self.strain and tr.contact:
                        self.strain_adjust[name] = vals
                    elif self.graph.modifier_kind(name) == 'branch':
                        self.branch_adjust[name] = self.branch_adjust.get(name, 1.0)*sim.groups.group_vals(d, vals)
                    else:
                        msg = self.graph.dwell_not_branch(name, f'the adjustment by "{strat.name}"', self.mod.name)
                        if msg:
                            ss.note(msg, sim=sim)
                        current = 1.0 if self.adjust[name] is None else self.adjust[name]
                        self.adjust[name] = current*sim.groups.group_vals(d, vals)
                        self.adjust_dims[name].add(d)
        self.rel_scale = {}
        for name,val in (getattr(self.graph, 'scale', None) or {}).items(): # A strain with its own course: its values, which multiply as adjust= does (see ss.Strains)
            if name in self.rel_adjust:
                self.rel_scale[name] = val
            elif self.graph.modifier_kind(name) == 'branch':
                self.branch_adjust[name] = self.branch_adjust.get(name, 1.0)*val
            else:
                self.adjust[name] = (np.ones(sim.groups.n_groups) if self.adjust[name] is None else self.adjust[name])*val
        for name,tr in self.graph.transitions.items(): # A value by level of a dimension that the states moved have no level of: there is no level to take (it took the first)
            ends = tr.src or tr.dst or []
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p):
                for leaf in (expr.leaves if isinstance(expr, ss.expressions.Expr) else []):
                    raw = getattr(leaf, 'raw', None)
                    if isinstance(raw, ss.by) and not raw.on_people(sim) and not set(raw.dims) & {strat.name for strat in self.graph.derived}: # Not read from each person, or by a derived attribute's levels, which have no groups
                        for d in raw.check(sim)[0]:
                            if ends and d != self.strain and not any(self.applies[s.name][d] for s in ends):
                                dim = sim.groups.strats[d].name
                                raise sc.KeyNotFoundError(f'"{name}" uses a value by level of "{dim}" ({leaf.raw}), but {sc.strjoin(s.name for s in ends)} {"has" if len(ends) == 1 else "have"} no level of "{dim}", so there is no level to take it from.\n'
                                                          f'     Stratify {sc.strjoin(s.name for s in ends)} by "{dim}" too, or give the value without ss.by(); for a count with each person weighted by their level (e.g. how much each infectious person is bitten), weight the states inside the count: (ss.by({dim}=...)*I).n.')
        self.init_strain_weights()
        self.note_function_reads()
        return

    def note_function_reads(self):
        """ Say when a parameter that a modifier changes (e.g. ss.change) is read inside a function, which sees its own value rather than the changed one """
        modified = [name for (modname, kind, name) in self.sim.modifiers if modname == self.mod.name and kind == 'par']
        if not modified:
            return
        funcs = {getattr(f, '__name__', 'a function'):f for f in self.graph.funcs.values()}
        funcs.update({getattr(p.raw, '__name__', 'a function'):p.raw for p in self.graph.all_pars() if callable(p.raw) and not isinstance(p.raw, (ss.Dist, ss.parameters.TimeValue))})
        for fname,func in funcs.items():
            source = ss.presentation.func_source(func) or ''
            read = [name for name in modified if re.search(rf'\bpars\.{name}\b|\[[\'"]{name}[\'"]\]', source)]
            if read:
                ss.note(f'{sc.strjoin(read)} of "{self.mod.name}" {"is" if len(read) == 1 else "are"} changed by a modifier and read inside {fname}: arithmetic on the parameter there (e.g. 1 - {self.mod.name}.pars.{read[0]}) gives the changed value, but float(par), par.value, and par.raw give the value without the change', sim=self.sim)
        return

    def strain_rate(self, tr, rate, strain):
        """ The part of a transmission's rate that comes from one strain """
        if strain is None:
            return rate
        if tr.const is not None and strain > 0: # A part of the rate with no source of infection has no strain, so is counted once, with the first
            rate = rate - self.eval(tr.const)
        adjust = self.strain_adjust[tr.name]
        return rate if adjust is None else rate*adjust[strain]

    def convert_beta(self, val):
        if isinstance(val, (list, tuple)):
            return [self.convert(v) for v in val]
        return self.convert(val)

    def route_value(self, p, raw):
        """
        The value of a transmission's beta on the route it is being worked out for (`self._route`, as agents): a beta
        for each route gives its value for this one; on a network with acts, a probability per act (`ss.peract`, or a
        bare number) gives the hazard per act, −ln(1 − p), in each of the network's directions (see `act_hazards()`)
        """
        nkey, route = self._route
        if isinstance(raw, dict):
            val = ss.utils.route_beta(raw, nkey)
            if val is None:
                raise ValueError(f'beta for "{self.mod.name}" has keys {list(raw.keys())}, but there is a network called "{nkey}" with no value.')
            raw = val
        if getattr(route, 'has_acts', False):
            self._per_act = True
            return self.act_hazards(raw, route)
        elif isinstance(raw, ss.peract): # Checked when the sim is built: see ss.solvers.choose.check_betas()
            raise ValueError(f'beta for "{self.mod.name}" on "{nkey}" is {raw}, a probability per act, but "{nkey}" has no acts')
        return self.plain_value(raw)

    def plain_value(self, val):
        """ A value written inside another (a direction of `ss.peract`, or a route of a beta dict) as a number, or a value for each group """
        if isinstance(val, ss.expressions.Expr):
            return self.eval(val)
        elif isinstance(val, ss.by):
            return self.by_group(val)
        elif callable(val) and not isinstance(val, ss.Dist):
            return self.convert(self.call(val))
        return self.convert(val)

    def act_hazards(self, val, route):
        """ The hazard per act, −ln(1 − p), of a probability per act on a network, in each of its directions (one if it has no sides): an array of directions, or of directions by groups """
        vals = val.resolve(route) if isinstance(val, ss.peract) else [val]*max(len(route.directions), 1)
        ps = np.array(np.broadcast_arrays(*[np.asarray(self.plain_value(v), dtype=float) for v in vals]))
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError(f'the probability per act of "{self.mod.name}" on "{route.name}" is {val}, which gives {ps.tolist()}, outside 0 to 1')
        return -np.log1p(-np.minimum(ps, 1 - 1e-15))

    def n_args(self, func):
        """ The number of arguments a user function takes, worked out once for each function """
        known = self.arities.get(id(func))
        if known is None or known[0] is not func:
            known = self.arities[id(func)] = (func, ss.utils.n_args(func))
        return known[1]

    def call(self, func, p=None):
        """ Call a user function of the sim: once a step, if it takes only the sim """
        if isinstance(func, ss.parameters.TimeValue): # Its values converted, then interpolated as numbers
            if self.year is None or ss.solvers.Compartmental.held(func): # A step function is held at its value at the start of the step, as in a whole run (see Compartmental.tables())
                return self.held_value(func)
            return func.series([self.year], self.sim, self.convert)[0]
        if self.n_args(func) >= 2:
            return self.call_uids(func, p)
        key = ('called', id(func))
        if key not in self.cache:
            self.cache[key] = func(self.sim)
        return self.cache[key]

    def held_value(self, func):
        """ A value that varies in time (or an `ss.by` of them, for each group), at the start of the current step: worked out for every step at once, the first time it is needed (as in `Compartmental.tables()`), rather than at each call """
        key = ('held', id(func))
        if key in self.cache: # Already looked up this step
            return self.cache[key]
        t = self.sim.t
        inner = tuple(p.raw for p in getattr(func, 'pars', [])) # The values of any parameters inside it, which can change between runs
        kept = self.held_series.get(id(func))
        if kept is None or kept[0] is not func or kept[1] != inner:
            kept = self.held_series[id(func)] = (func, inner, func.series(t.yearvec, self.sim, self.convert))
        ti = max(t.ti - 1, 0) # The step's start: the loop's ti is its end (the start of the run before the first step)
        if ti < len(kept[2]):
            out = kept[2][ti]
        else: # Before or after the run
            out = func.series([t.now('year')], self.sim, self.convert)[0]
        self.cache[key] = out
        return out

    @property
    def inferred_unit(self):
        """ Whether the sim's unit was worked out from the typed values in the model, rather than given with unit= (see `ss.solvers.choose.init_unit()`) """
        return self.sim.pars.unit is None and self.sim.unit_source == 'the typed values in the model'

    def bare_return(self, p, func, out):
        """ ST33: a function for a rate or a duration that returns a bare number, when the unit is inferred from the typed values, raises rather than guess its unit (0 is 0 in any unit) """
        if np.any(np.asarray(out, dtype=float) != 0):
            name, unit = getattr(func, '__name__', 'function'), self.sim.unit
            example = f'ss.per{unit}(...)' if p.slot == 'rate' else f'ss.{unit}s(...)'
            errormsg = (f'the function "{name}" gives "{p.name}" of "{self.mod.name}", a {"rate" if p.slot == "rate" else "duration"}, as a bare number ({np.ravel(out)[0]:.4g}), so its unit is not known: '
                        f'the sim\'s unit ({unit}s) is only inferred from the typed values in the model.\n'
                        f'     Return a typed value, e.g. {example}, or give the sim a unit, e.g. ss.Sim(..., unit=\'{unit}\')')
            raise TypeError(errormsg)
        return

    def frac(self, states):
        return self.count(states)/self.n_people()

    def local_count(self, states):
        """ The number in some states as it is used in a rate: in a model with places, the number in the same place """
        return self.count(states)

    def local_frac(self, states):
        return self.local_count(states)/self.local_count(self.graph.people_states)

    def total_count(self, states):
        """ The number in some states in the whole population, wherever they are (`I.total`) """
        return self.count(states)

    def split_states(self, states):
        """ Some states split into this disease's own and other diseases' (e.g. `syph.I` in a rate of HIV), by the name of each other disease """
        own, others = [], {}
        for state in states:
            name = getattr(getattr(state, 'module', None), 'name', None)
            if name is None or name == self.mod.name:
                own.append(state)
            else:
                others.setdefault(name, []).append(state)
        return own, others

    def count_expr(self, expr):
        """ The value of a count in a rate (`I.n`, `I.total`, `I.frac`), with another disease's states counted by that disease """
        own, others = self.split_states(expr.states)
        if expr.weighted: # Each person counted by a value of theirs (MS3, TY6)
            out = self.count_weighted(own, expr) if own else 0.0
            for name,states in others.items():
                out = out + self.sim.get_declared(name).engine.count_weighted(states, expr)
            return out
        method = dict(n='local_count', total='total_count', frac='local_frac')[expr.kind]
        out = getattr(self, method)(own) if own else 0.0
        for name,states in others.items():
            out = out + getattr(self.sim.get_declared(name).engine, method)(states)
        return out

    def count_weighted(self, states, expr):
        """ A weighted count (`(ss.by(biting=w)*I).n`, `A.n_trans`: see `ss.expressions.Count`), in the same place as whoever it is evaluated for; by the engines """
        raise NotImplementedError

    def product_of(self, expr):
        """ The sim's copy of the product of a coverage (`itn.coverage`, see `ss.expressions.Coverage`) """
        product = next((p for p in self.sim.products() if p.name == expr.product.name), None)
        if product is None or not product.holds:
            errormsg = f'{expr} is the coverage of "{expr.product.name}", which is not a product that people hold delivered in the sim; deliver it, e.g. ss.vaccinate({expr.product.name}, ...)'
            raise ValueError(errormsg)
        return product

    #%% Dwell times

    def init_dwells(self):
        """ Work out the dwell-time distribution of every transition that has one """
        trans = ss.transitions
        unit = self.sim.unit_base
        to_unit = lambda val: convert(val, unit) # As self.convert, but without holding the engine, since a function parameter of a dwell time keeps it
        for name,tr in self.graph.transitions.items():
            p = tr.dur if tr.dur is not None else tr.rate if (isinstance(tr.rate, trans.par) and not tr.contact) else None
            if p is None or tr.immediate: # dur=0 has no dwell: see Graph.resolve_exits()
                continue
            raw = trans.by_dist(p.raw) if tr.dur is not None else p.raw
            scaled = trans.scaled_dist(raw) if tr.dur is not None and isinstance(raw, trans.Expr) else None # A distribution times a factor: each draw is scaled, so it keeps its shape
            if scaled is not None:
                raw = trans.by_dist(scaled[0].raw)
            if tr.dur is not None and callable(raw) and not isinstance(raw, (ss.Dist, ss.TimePar, ss.by, trans.TimeValue)): # ST18: a function's value is the mean of an exponential, not each person's own time
                fname = getattr(raw, '__name__', 'the function')
                self.tell(ss.warn, f'"{self.mod.name}.{name}" has dur={fname}, whose value is the mean of an exponential dwell time, so people leave at a constant rate; for each person to leave at the time it gives, write dur=ss.constant({fname})')
            dw = Bunch(par=p, raw=raw, dist=None, k=tr.stages or 1, exact=True, is_rate=tr.dur is None, family='exponential', std=None, fixed=False, kg=None, scale=None if scaled is None else scaled[1])
            if isinstance(raw, ss.Dist) and tr.dur is not None and any(isinstance(v, ss.by) for v in trans.varying_pars(raw).values()): # Parameters by group
                self.varying_dwell(name, tr, raw, dw, to_unit)
            elif isinstance(raw, ss.Dist) and tr.dur is not None:
                mean, std = trans.dwell_stats(raw, to_unit)
                dw.family = type(raw).__name__
                dw.mean, dw.std = mean, std
                dw.dist = trans.numeric_dist(raw, to_unit)
                if np.isnan(mean): # A parameter is a function of the agents
                    if self.solver != 'abm':
                        raise ss.CapabilityError(f'"{name}" cannot run with solver="{self.solver}": its dwell time {raw} has a parameter that is a function of the agents. Run with solver="abm".')
                    typed_factor = scaled is not None and any(isinstance(getattr(leaf, 'raw', None), ss.TimePar) for leaf in trans.as_expr(scaled[1]).leaves) # e.g. dur=d*ss.days(1), whose numbers are days (TY5)
                    if getattr(raw, 'unit', None) is None and not typed_factor and not any(isinstance(v, ss.TimePar) for v in raw.pars.values()): # What the function returns has no unit to say otherwise
                        funcs = sc.strjoin(getattr(v, '__name__', 'a function') if callable(v) else str(v) for v in raw.pars.values() if callable(v) or isinstance(v, (ss.conditions.Cond, ss.expressions.Expr)))
                        self.tell(ss.note, f'the dwell time of "{name}" comes from {funcs}, whose numbers are read in {self.sim.unit_base}s, the sim\'s unit; if they are in another unit, give it on the distribution, e.g. unit="month"', sim=self.sim)
                elif dw.family == 'expon':
                    dw.family = 'exponential'
                elif dw.family == 'constant' and tr.stages is None: # A fixed delay: a conveyor in a compartmental run (see Compartmental.build_layout())
                    dw.fixed = True
                elif tr.stages is None:
                    dw.k = trans.exact_stages(raw)
                    if dw.k is None:
                        dw.k = trans.erlang_stages(mean, std)
                        dw.exact = False
                        self.note_stages(name, dw)
                dw.staged = True
            else:
                dw.staged = dw.k > 1
            tr.guessed.pop('dwell', None)
            if dw.family == 'exponential' and not isinstance(raw, ss.Dist) and tr.stages is None and tr.dur is not None:
                tr.guessed['dwell'] = 'scalar duration'
            self.dwells[name] = dw
        return

    def init_dwell_mods(self):
        """ The modifiers of each dwell time (see `dwell_mods()`), once every module has registered its modifiers (see `Sim._init_modules_pre()`): after the layout, which may be made before them, to choose the solver """
        for name,dw in self.dwells.items():
            self.dwell_mods(name, dw)
        return

    def dwell_mods(self, name, dw):
        """
        The modifiers of a dwell time's parameter when it is a distribution (e.g. `ss.modify('sir.pars.dur_inf', set=ss.constant(ss.years(5)))`), as agents:
        `dw.mods` are the modifiers, and `dw.sets` the distribution each set= draws from, by the modifier's index (not its id, which would be in the path its seed is made from) (a duration is the mean of an exponential, as for dur=), with its mean.
        People who enter the state while one is active draw from it; those already there have what is left of their dwell rescaled by the change in its mean (see `Agents.rescale_exits()`).
        """
        p, raw = dw.par, dw.raw
        mods = getattr(self.sim, 'modifiers', {}).get((self.mod.name, 'par', p.name)) if (p.name and not p.literal) else None
        if not mods:
            return
        unit = self.sim.unit_base
        to_unit = lambda val: convert(val, unit) # Without holding the engine, as in init_dwells()
        path = f'{self.mod.name}.pars.{p.name}'
        if dw.dist is None and not getattr(dw, 'varying', False): # A duration or rate, which value() modifies
            for m in mods:
                if m.op == 'set' and isinstance(m.amount, ss.Dist):
                    raise ValueError(f'the modifier "{m.name}" sets {path} to a distribution ({m.amount}), but {path} is a duration, the mean of an exponential dwell time for "{name}", so it can only be set to another duration.\n'
                                     f'     Give a duration (e.g. set=ss.years(5)), or make the parameter a distribution (e.g. ss.expon(ss.years(1))) so it can be set to another.')
            return
        if self.solver != 'abm':
            raise ss.CapabilityError(f'the modifier "{mods[0].name}" changes {path}, the dwell time of "{name}", which is a distribution ({raw}); with solver="{self.solver}" it is laid out in compartments once, at the start, so it cannot change during the run.\n'
                                     f'     Run with solver="abm", or modify the transition instead (e.g. ss.modify("{self.mod.name}.{name}", multiply=0.5)), which scales its rate.')
        funcs = [key for key,val in ss.transitions.varying_pars(raw).items() if not isinstance(val, ss.by)] if isinstance(raw, ss.Dist) else []
        if funcs or (dw.mean is not None and np.isnan(dw.mean)): # By group is fine, since each agent's mean is known (see Agents.agent_dwell_mean()); a function of the agents is not
            raise ss.CapabilityError(f'the modifier "{mods[0].name}" changes {path}, the dwell time of "{name}", whose distribution ({raw}) is a function of the agents, which a modifier cannot change yet; modify the transition instead (e.g. ss.modify("{self.mod.name}.{name}", multiply=0.5)).')
        dw.mods = mods
        dw.sets = {}
        for i,m in enumerate(mods):
            if m.op != 'set':
                continue
            if m.where is not None or m.hold is not None:
                raise ss.CapabilityError(f'the modifier "{m.name}" sets {path}, the dwell time of "{name}", with where= or hold=, which is not supported yet for a dwell time that is a distribution; set it for everyone, or use multiply= with where=.')
            new = m.amount if isinstance(m.amount, ss.Dist) else ss.expon(scale=m.amount) if isinstance(m.amount, ss.TimePar) or sc.isnumber(m.amount) else None
            if new is None or ss.transitions.varying_pars(new):
                raise ss.CapabilityError(f'the modifier "{m.name}" sets {path}, the dwell time of "{name}", to {m.amount}; a dwell time that is a distribution can be set to another distribution with fixed parameters, or to a duration (the mean of an exponential).')
            mean, std = ss.transitions.dwell_stats(new, to_unit)
            dw.sets[i] = Bunch(key=f'dwell_{name}_{i}', dist=ss.transitions.numeric_dist(new, to_unit), mean=mean)
        return

    def transfer_dist(self, tf, per_agent=None):
        """ The distribution of a transfer's dur= as numbers in the sim's unit, if it is not exponential (None if it is a rate, a scalar, or exponential): agents draw it for each person (see `Agents.step_transfers()`) """
        raw = ss.transitions.by_dist(getattr(tf.dur, 'raw', None))
        if not isinstance(raw, ss.Dist) or isinstance(raw, ss.expon):
            return None
        unit = self.sim.unit_base
        return ss.transitions.numeric_dist(raw, lambda val: convert(val, unit), per_agent=per_agent)

    def note_steps(self):
        """ In a run that takes whole steps, say which transitions are fast compared with the timestep: by the parameters' own values, without their modifiers, which may read what does not exist until the run (e.g. a property of people another module adds) """
        with self.evaluating(unmodified=True):
            self.note_fast_steps()
        self.cache.clear() # Values worked out without their modifiers
        return

    def note_fast_steps(self):
        """ See `note_steps()` """
        sim = self.sim
        for tr in self.graph.transitions.values(): # A rate per person that grows with a count of people, which is often a count written for a share
            if tr.people and not tr.inflow and tr.rate is not None and tr.flow_given is None and ss.expressions.count_power(tr.rate) > 0 and not self.plausible_density(tr): # Not for the whole flow written without a total (beta*S*I), which says so
                ss.note(f'{tr.name} has a rate per person of {ss.expressions.format_expr(tr.rate)}, which grows with the number of people (density-dependent); for a share of the population, write .frac (e.g. I.frac), or write the transmission with the bare state (e.g. beta * I)', sim=sim)
        if sim.solver in ['gillespie', 'tau'] and self.smooth: # Exact only for rates that are constant within each quarter of a step
            what = 'solver="gillespie"' if sim.solver == 'gillespie' else 'Tau-leaping'
            msg = f'{what} holds values that vary in time for a quarter of a step at a time, at their middle'
            if sim.pars.approx is False:
                raise ss.CapabilityError(f'{msg}, which approx=False does not allow.\n     Run with solver="ode", or allow the approximation; a shorter step follows the values more closely.')
            ss.note(f'{msg}; a shorter step follows them more closely', sim=sim, verbose=0 if sim.pars.approx else None)
        euler = sim.solver == 'ode' and sim.integrator == 'euler' and sim.pars.substeps is not None # Euler with steps of a given size; otherwise the ODE takes enough substeps
        unit = sim.unit_name
        inflows = [tr.name for tr in self.graph.inflows] # Their rate is a number of people per unit time, so nobody leaves
        if sim.solver not in ['abm', 'binomial'] and not euler: # The solver is right whatever the step, but a mean dwell shorter than a step is usually a rate meant as a duration (rate=10)
            for name,dw in self.dwells.items():
                if name not in inflows and dw.dist is None and not callable(dw.par.raw) and not isinstance(dw.par.raw, ss.by):
                    mean = self.dwell_mean(name)
                    if np.ndim(mean) == 0 and 0 < mean < self.dt:
                        ss.note(f'{name} has a mean dwell of {mean:.2g} {unit}{"s" if mean != 1 else ""}, shorter than the step; a duration is written dur=, or ss.days(10)', sim=sim)
            return
        if sim.solver == 'binomial' and sim.pars.substeps is None: # It takes enough substeps (see ss.solvers.Compartmental.substeps())
            return
        dt = self.dt/(sim.pars.substeps if euler or sim.solver == 'binomial' else 1)
        step = 'step' if dt == self.dt else 'substep'
        plural = lambda n: f'{n:.2g} {unit}{"s" if n != 1 else ""}'
        fast = [] # What is fast compared with the step, as (rate × step, its name, what it is)
        for tr in self.graph.transitions.values(): # Transmission: someone infected in a step infects others only from the next, so a fast one is slowed down (as measles is at daily steps)
            if tr.contact and tr.people and not tr.inflow: # Its β for each infectious state, per contact (with a network) or per share infectious
                E = ss.expressions
                if any(isinstance(leaf, (ss.transitions.State, E.Count, E.Clock, E.Flow, E.Stock)) for coef in tr.terms.values() for leaf in getattr(coef, 'leaves', [])):
                    continue # It reads states (e.g. another disease's prevalence), which have no values yet
                try:
                    rate = max(float(np.max(np.asarray(self.eval(coef), dtype=float))) for coef in tr.terms.values())
                except Exception: # A value not known until the run (e.g. a function of time)
                    continue
                per = 'per infectious contact' if len(sim.networks) else 'with everyone infectious'
                fast.append((rate*dt, tr.name, f'{tr.name} transmits at up to {rate:.2g} per {unit} ({per}), {-np.expm1(-rate*dt):.0%} in a {step}'))
        for name,dw in self.dwells.items():
            tr = self.graph.transitions[name]
            if name in inflows or self.never_taken(tr):
                continue
            if dw.dist is None and (callable(dw.par.raw) or isinstance(dw.par.raw, (ss.by, ss.expressions.Expr))): # Not known until the sim runs, or not one number
                continue
            mean = dw.mean if dw.dist is not None else self.dwell_mean(name)
            if sim.solver == 'abm' and dw.dist is not None and dw.family != 'exponential' and mean is not None and getattr(dw, 'std', None) is not None and np.all(np.asarray(mean, dtype=float) - 2*np.asarray(dw.std, dtype=float) >= dt): # As agents, a dwell drawn from a distribution that seldom reaches below a step (e.g. a pregnancy of 32 to 42 weeks, at a monthly step) loses nobody to whole steps (FP7)
                continue
            if mean is not None and np.ndim(mean) == 0 and mean > 0: # The chain-binomial solver moves people through the stages exactly (see ss.kernels.draw_chain()), and agents draw the whole dwell time, so it is the mean that matters
                x = dt/mean
                share = min(x, 1) if euler else -np.expm1(-x)
                leave = 'almost everyone leaves' if share > 0.99 else f'{share:.0%} leave'
                src = sc.strjoin([state.name for state in tr.src or []], sep=' or ') or name # Named by the state left, which its exits (e.g. recovery and death) share
                fast.append((x, src, f'people leave {src} after {plural(mean)} on average, so {leave} in a {step}'))
        fast = sorted([f for f in fast if f[0] > 0.1], key=lambda f: -f[0]) # The fastest first, since it is what the step is too long for
        fastest = {}
        for f in fast: # Each state once, at its fastest
            fastest.setdefault(f[1], f)
        fast = list(fastest.values())
        if fast:
            x, _, what = fast[0]
            others = f' (and {sc.strjoin(f[1] for f in fast[1:])} {"is" if len(fast) == 2 else "are"} fast too)' if len(fast) > 1 else ''
            fixes = 'a shorter step (dt=) or more substeps=' if euler or sim.solver == 'binomial' else f'a shorter step (e.g. dt=ss.{unit}s({dt/4:.2g}))'
            msg = f'consider {fixes}: each {step} is {plural(dt)}, and {what}{others}'
            if euler:
                msg += ' (Euler scales outflows that would take more than everyone down together)' if x > 1 else ''
            else: # Whole steps are off the ODE by an amount of order rate × dt, which can be large: about 15% in the prevalence of a seasonal SIRS at 0.12
                who = 'agents' if sim.solver == 'abm' else 'the chain binomial'
                msg += f'. Taking whole steps, {who} may differ from the ODE by an amount of that order'
            ss.note(f'{msg}; sim.check_dt() shows how much it matters', sim=sim)
        return

    def never_taken(self, tr):
        """ Whether a transition to one destination has a branch probability of 0 everywhere (e.g. death in `ss.SIR` with `p_death=0`) """
        if tr.p is None or len(tr.p) != 1 or tr.p[0] is None:
            return False
        try:
            return not np.any(np.asarray(self.eval(tr.p[0]), dtype=float))
        except Exception: # Not known until the run
            return False

    def plausible_density(self, tr):
        """
        Whether a transition is density-dependent transmission as meant (`S >> I @ beta*I.n`, see `ss.graph.Graph.density_sources`), rather than
        a count written for a share: whether one infectious person infects each susceptible at a rate under 5% a step. A beta meant for a share
        (0.3 per day) is far above one for density dependence, R0/(N × the time infectious), unless the population is a handful of people
        """
        if not tr.acquisition or not self.graph.density_sources:
            return False
        one = ss.expressions.substitute(tr.rate, {id(leaf):1.0 for leaf in tr.rate.leaves if isinstance(leaf, ss.expressions.Count)})
        try:
            return bool(np.all(np.asarray(self.eval(one))*self.dt < 0.05))
        except Exception: # A value not known until the run (e.g. a function of time)
            return False

    def varying_dwell(self, name, tr, raw, dw, to_unit):
        """ A dwell time whose distribution has parameters with a value for each group (`ss.by`): see each engine's own """
        raise NotImplementedError

    def note_stages(self, name, dw):
        """ Say how a dwell time that is not exponential or Erlang is approximated; agents draw it as it is (see `Compartmental.note_stages()`) """
        return

    def dwell_mean(self, name):
        """ The current mean dwell time of a transition, in the sim's unit """
        dw = self.dwells[name]
        if dw.dist is not None or getattr(dw, 'varying', False):
            return dw.mean if getattr(dw, 'scale', None) is None else dw.mean*self.value(dw.scale) # A distribution times a factor
        val = self.value(dw.par)
        return np.divide(1, val, out=np.full(np.shape(val), np.inf), where=np.asarray(val)!=0)[()] if dw.is_rate else val # A rate of 0 is a dwell that never ends

    #%% Initial conditions

    def init_fractions(self):
        """ The fraction of people who start in each state, and the count in each state that is not people """
        graph = self.graph
        fracs = sc.objdict({s.name:0.0 for s in graph.people_states})
        counts = sc.objdict({s.name:0.0 for s in graph.count_states})
        for state in graph.states.values():
            is_n = state.init_n is not None # init_n= is a number of people, and init= a share of them (or, for a state that is not people, its amount)
            init = state.init_n if is_n else state.init
            if init is None:
                continue
            if isinstance(init, ss.expressions.par):
                init = init.raw
            elif isinstance(init, ss.expressions.Expr) and ss.expressions.per_person_in(init): # An expression of people's properties and parameters, e.g. 1 - (A - age)/(k*age + A) (TY2)
                if state.people and not is_n and self.solver == 'abm': # Read for each person, as agents: see Agents.init_post()
                    continue
                init = np.asarray(self.eval(init), dtype=float)[()] # For each group
            elif isinstance(init, ss.expressions.Expr): # An expression of parameters, e.g. I0/2
                init = float(self.eval(init))
            if callable(init) and not isinstance(init, (ss.by, ss.Dist)): # A function of the sim, as for init_prev
                init = self.call(init)
            if isinstance(init, ss.by) and self.strain is not None and self.sim.groups.strats[self.strain].name in init.dims: # A share by strain would be of the people at a level of the strains, where nobody is before they are infected
                strat = self.sim.groups.strats[self.strain]
                errormsg = f'init= of "{state.name}" in "{self.mod.name}" is given by "{strat.name}", the strains people carry; give how many start infected in init= (e.g. init=dict({state.name}=0.1)), and the share of each strain, or set of strains carried together, in ss.Strains(split=...), e.g. split={{"{strat.labels[0]}": 0.5, "{strat.labels[-1]}": 0.5}}'
                raise ValueError(errormsg)
            if isinstance(init, ss.by) and state.people and not is_n and self.solver == 'abm' and init.on_people(self.sim): # Read from each person's own properties, as agents: see Agents.init_post()
                continue
            if isinstance(init, ss.by): # A value for each level of one or more dimensions: a number of people, or a fraction of the people at that level
                sim = self.sim
                ds, given = init.grid(sim)
                if state.people and not is_n:
                    self.check_share(state, given, init.is_count())
                grid, index = sim.groups.grid_index(ds, given, count=is_n if state.people else init.is_count() or np.any(given > 1)) # A group with no level of a dimension gets the mean fraction, or no one of a number
                cells = np.ravel_multi_index(index, grid.shape) # The combination of levels each group is in
                split = np.bincount(cells, weights=sim.groups.group_split, minlength=grid.size)[cells] # The share of people at that combination
                if state.people:
                    split = np.bincount(cells, weights=self.state_split[state.name], minlength=grid.size)[cells] # ... of those this state is split between, which for a dimension of some states only is not everyone
                    vals = grid[index] # For each group
                    size = self.total_pop*split
                    fracs[state.name] = np.divide(vals, size, out=np.zeros(sim.groups.n_groups), where=size>0) if is_n else vals
                    lost = (vals > 0) & (size <= 0) # Given where the split puts nobody
                    if np.any(lost):
                        strats = sim.groups.strats
                        cells = sorted({'(' + ', '.join(f'{strats[d].name}={strats[d].labels[ix[g]]}' for d,ix in zip(ds, index) if ix[g] < strats[d].n) + ')' for g in np.flatnonzero(lost)})
                        msg = f'init= of "{state.name}" in "{self.mod.name}" is given at {sc.strjoin(cells[:3])}{" and more" if len(cells) > 3 else ""}, where the split puts nobody'
                        if not np.any((vals > 0) & (size > 0)):
                            raise ValueError(f'{msg}, so nobody would start in {state.name}. Give a split= that puts people there, or init= at levels that have people.')
                        ss.note(f'{msg}; nobody starts in {state.name} there', sim=sim)
                else:
                    share = np.divide(self.state_split[state.name], split, out=np.zeros(sim.groups.n_groups), where=split>0) # Its split over the other dimensions
                    counts[state.name] = given.sum()
                    self.state_split[state.name] = share*grid[index]/max(given.sum(), 1e-300)
                continue
            if not state.people:
                counts[state.name] = float(init)
            elif is_n:
                fracs[state.name] = init/self.total_pop
            else:
                self.check_share(state, init, isinstance(init, (int, np.integer)))
                fracs[state.name] = np.asarray(init, dtype=float)[()] # A share, or one for each group (from a function)

        # Initial prevalence is seeded through the transmission, so people start wherever a new infection starts
        init_prev = graph.init_prev
        share = 1.0
        family = getattr(graph, 'seeded_by', None)
        if family is not None and self.solver != 'abm': # A strain with its own course, seeded by the disease (ss.Strains(shared=...)): the disease's share, times this member's split (independently of the other members, so unlike agents some are seeded with two)
            family = self.sim.diseases[family]
            init_prev, share = ss.parameters.starting(family.init_prev), family.split[family.member_names.index(self.mod.name)]
        if isinstance(init_prev, ss.expressions.par): # A parameter (e.g. sir.pars.init_prev), whose value can be a distribution, or a value for each level (worked out for each group below, not for each agent)
            init_prev = init_prev.raw if isinstance(init_prev.raw, (ss.Dist, ss.by)) else self.value(init_prev)
        if init_prev is not None and self.seed_dist() is not None: # Drawn for each agent instead: see Agents.init_post()
            init_prev = None
        if isinstance(init_prev, ss.by): # A share for each level of some dimensions (a property of each person that is not one raises here, unless run as agents: see seed_dist())
            init_prev = init_prev.per_group(self.sim, self.convert)
        elif callable(init_prev) and not isinstance(init_prev, ss.Dist): # A function of the sim, giving a share (or one for each group)
            init_prev = self.call(init_prev)
        if init_prev is not None and np.any(init_prev):
            if isinstance(init_prev, ss.bernoulli):
                init_prev = self.call(init_prev.pars.p) if callable(init_prev.pars.p) else init_prev.pars.p # A function of the agents raises here: see call_uids()
            elif isinstance(init_prev, ss.choose_n): # A number of people, as Starsim 3 gave it
                init_prev = init_prev.pars.n
            if np.ndim(init_prev) and np.any(np.asarray(init_prev) > 1):
                raise ValueError(f'init_prev of "{self.mod.name}" gives a share of more than 1 for some groups ({np.max(init_prev):n}); a value for each group is a share of the people in it')
            elif np.ndim(init_prev) == 0 and init_prev > 1: # More than 1 is a number of people, and up to 1 a share (1.0 is everyone)
                init_prev = init_prev/self.total_pop
            init_prev = init_prev*share
            trs = self.seed_transmissions()
            if not trs:
                errormsg = f'"{self.mod.name}" has no transmission or acquisition to seed init_prev through; use init= on a state instead'
                raise ValueError(errormsg)
            dst = [state for tr in trs for state in tr.dst]
            if graph.infectious and not any(state.infectious for state in dst): # e.g. E in an SEIR
                seeded = sc.strjoin(dict.fromkeys(state.name for state in dst))
                ss.note(f'init_prev seeds {seeded} of "{self.mod.name}", where a new infection starts, so nobody is infectious at the start; to seed infectious people, use init=dict({graph.infectious[0].name}=...) (a share) or init_n= (a number)', sim=self.sim)
            with self.evaluating(by_group=True): # The share of each destination in each group, before there are agents to evaluate it for
                reach = self.seed_reach(trs)
                for tr,w in zip(trs, reach):
                    for state,p in zip(tr.dst, self.branch_probs(tr)):
                        fracs[state.name] = fracs[state.name] + init_prev*w*np.asarray(p, dtype=float)

        for name,val in list(fracs.items()) + list(counts.items()):
            if not np.all(np.isfinite(val)) or np.any(np.asarray(val) < 0):
                raise ValueError(f'the initial value of "{name}" in "{self.mod.name}" is {val}; an initial value must be finite and not negative.')

        # Whatever is left over goes to the first state, or, if that is given a value, to the one state that is not (see Graph.resolve_rest())
        total = sum(fracs.values())
        if np.max(total) > 1 + 1e-9:
            errormsg = f'The initial fractions of "{self.mod.name}" sum to {np.max(total):n}, which is more than 1: {dict(fracs)}. If some of these states are not people (e.g. mosquitoes), declare them with ss.State(people=False), whose init= is a count'
            raise ValueError(errormsg)
        if graph.people_states:
            rest = 1 - total
            left = np.max(np.abs(rest)) > 1e-9
            if graph.rest is None and left:
                first = graph.first.name
                unset = [s.name for s in graph.people_states if not s.given_init]
                where = f'Say where the rest start, e.g. {unset[-1]}={np.max(rest):.4g}' if unset else 'Give initial values that sum to 1'
                errormsg = (f'The initial values of "{self.mod.name}" sum to {np.min(total):n}, not 1, and it is not clear where the rest start: {first} is given a value, which is kept, '
                            f'and {"no other state is left without one" if not unset else f"{sc.strjoin(unset)} are not given one"}.\n'
                            f'     {where}.')
                raise ValueError(errormsg)
            if graph.rest is not None:
                fracs[graph.rest.name] += rest
            if left and graph.rest is not graph.first:
                n_rest = self.total_pop*np.sum(np.broadcast_to(rest, (self.sim.groups.n_groups,))*self.sim.groups.group_split)
                ss.note(f'the initial value of {graph.first.name} in "{self.mod.name}" is kept, and the other {n_rest:,.0f} people start in {graph.rest.name}, the only state not given one', sim=self.sim)
        return fracs, counts

    def check_share(self, state, vals, whole):
        """ Check a share given with init=: more than 1 raises, and a whole number 1 is everyone, with a warning to write 1.0 (or init_n= for one person) """
        name, mod = state.name, self.mod.name
        if np.any(np.asarray(vals) > 1):
            errormsg = f'init= of "{name}" in "{mod}" is {np.max(vals):n}, but init= is a share of people (1.0 is everyone); for a number of people, use init_n=dict({name}={np.max(vals):n})'
            raise ValueError(errormsg)
        if whole and np.any(np.asarray(vals) == 1):
            ss.warn(f'init= of "{name}" in "{mod}" is the whole number 1, which is read as a share: everyone starts in {name}. Write 1.0 for everyone, or init_n=dict({name}=1) for one person')
        return

    def seed_dist(self):
        """ An init_prev that chooses who starts infected one by one, which only agents have (see `Agents.seed_dist()`) """
        return None

    def seed_transmissions(self):
        """ The transmissions that init_prev seeds through: those out of the first state (e.g. one for each sex), or the first transmission if none is, or if they all lead to the same states; with none, the first acquisition """
        trs = [tr for tr in self.graph.transmissions if not getattr(tr, 'generated', False)] or [tr for tr in self.graph.acquisitions if tr.dst] # With no transmission (a cohort), where an infection puts people
        own = [tr for tr in trs if any(src is self.graph.first for src in tr.src)]
        if len(own) > 1 and len({tuple(s.name for s in tr.dst) for tr in own}) > 1:
            return own
        return trs[:1]

    def seed_reach(self, trs, uids=None):
        """
        For seeding through several transmissions (see `seed_transmissions()`): the share of the seeds of each group (or of each agent, with `uids`)
        that each takes, equal among those that can infect them (a coefficient above 0 for them), e.g. a transmission to women only seeds women
        """
        if len(trs) == 1:
            return [1.0]
        n = self.sim.groups.n_groups if uids is None else len(uids)
        reach = [np.broadcast_to(self.can_infect(tr, uids), (n,)).astype(float) for tr in trs]
        total = np.sum(reach, axis=0)
        return [np.divide(r, total, out=np.full(n, 1.0 if i == 0 else 0.0), where=total>0) for i,r in enumerate(reach)] # Nobody it can infect: the first, as before

    def can_infect(self, tr, uids=None):
        """ Whether a transmission can infect the people of each group (with a coefficient above 0 for them) """
        return np.logical_or.reduce([np.asarray(self.eval(coef), dtype=float) > 0 for coef in tr.terms.values()] or [True])

    def branch_probs(self, tr, ps=None):
        """ The probability of each destination of a transition (1 if there is only one), from its p= or the expressions given for them """
        if not tr.split:
            return [1.0]*len(tr.dst)
        ps = [None if p is None else self.eval(p) for p in (tr.p if ps is None else ps)]
        return self.dest_shares(tr, ps, stay=self.ends_given(tr, self.eval)) # With ss.stay or ss.null, what the states leave over stays or dies

    @staticmethod
    def ends_given(tr, ev):
        """ The probabilities given to ss.stay and ss.null in a split, together: the destination that takes the rest does not get them """
        return sum((ev(p) for p in [tr.p_stay, tr.p_die] if p is not None), 0.0)

    def die_share(self, tr, shares, ev=None):
        """ The probability of ss.null in a split (0 if it is not one of the destinations): what the shares of the states leave over, less what ss.stay takes """
        if not tr.dies:
            return 0.0
        ev = ev or self.eval
        left = 1.0 - sum(shares, 0.0)
        if not tr.stays:
            return left
        elif tr.p_die is not None:
            return np.minimum(ev(tr.p_die), left)
        return np.maximum(left - ev(tr.p_stay), 0.0)

    def eval(self, expr):
        return ss.expressions.evaluate(expr, self)

    def evaluating(self, **flags):
        """ Evaluate a block with some of the engine's flags set (e.g. `with self.evaluating(by_group=True):`), restored after it: see `Evaluating` """
        return Evaluating(self, {f'_{key}':val for key,val in flags.items()})

    #%% Results

    def init_results(self):
        """ One count per state, one flow per transition (`res.infection.cum` is its cumulative total), and for the library's diseases, Starsim's standard disease results """
        mod = self.mod
        graph = self.graph
        abm = self.solver == 'abm'
        results = sc.autolist()
        for state in graph.states.values():
            key = state.name
            if key not in mod.results:
                results += ss.Result(key, dtype=float, scale=abm and state.people, label=state.label or state.name, kind='stock')
        for name,tr in graph.transitions.items():
            scale = abm and tr.people
            results += ss.Result(name, dtype=float, scale=scale, label=tr.label or name, kind='flow')
            for end in tr.ends: # A flow for each destination, e.g. res.cin.X, whose sum is the total
                dst = graph.states.get(end)
                results += ss.Result(f'{name}.{end}', dtype=float, scale=scale, label=f'{tr.label or name} to {"death" if dst is None else dst.label or dst.name}', kind='flow', auto_plot=False)
        for name,expr in graph.exprs.items(): # A ratio, a sum of flows, or otherwise a stock
            if len(ratio_parts(expr)) == 2: # Including one scaled by a number, e.g. I/N*1e5
                kind = 'ratio'
            elif any(isinstance(leaf, ss.expressions.Flow) for leaf in expr.leaves) and all(isinstance(leaf, (ss.expressions.Flow, ss.par)) for leaf in expr.leaves):
                kind = 'flow'
            else:
                kind = 'stock'
            results += ss.Result(name, dtype=float, scale=False, label=name, kind=kind)
        self.init_outcomes(results)
        for name,tf in graph.transfers.items():
            results += ss.Result(name, dtype=float, scale=abm, label=tf.label or name, kind='flow')
            self.flowcounts[name] = 0.0
            stages = next((strat for strat in self.sim.groups.strats if getattr(strat, 'stages_of', None) == (mod.name, name)), None)
            if stages is not None and not stages.dwell.exact: # Erlang stages: see ss.dimensions.transfer_stages()
                self.note_stages(name, stages.dwell)
        mod.define_results(*results)
        written = {r.name for r in results if isinstance(r, ss.Result)} | set(graph.states) | set(graph.transitions) # Those the engine writes, which are scaled when written (see update_results()); not those of the disease's own code, which are counted in agents and scaled at the end
        ss.results.in_people(*[r for key,r in mod.results.items() if isinstance(r, ss.Result) and (key in written or key.partition('.')[0] in graph.transitions)])

        # Stratified results are one result with a dimension for each stratification: see res.select() and res.by()
        ng = self.sim.groups.n_groups
        self.stock_groups = sc.objdict()
        self.flow_groups = sc.objdict()
        self.rows = self.group_ti = None # The row of each time point in the results by group, and the time point of each row, if they are kept every few steps: see ss.solvers.group_rows()
        self.row_flows = sc.objdict() # Then, the flows by group so far in the current row
        if ng > 1:
            names = [strat.name for strat in self.sim.groups.strats]
            dims = sc.objdict() # The dimensions of each result
            for state in graph.states.values():
                dims[state.name] = self.applies[state.name]
            for name,tr in graph.transitions.items():
                dims[name] = np.any([self.applies[state.name] for state in tr.src + tr.dst], axis=0)
                for end in tr.ends:
                    dims[f'{name}.{end}'] = dims[name]
            for name in graph.transfers:
                dims[name] = np.ones(len(names), dtype=bool)
            self.ratios = sc.objdict() # The results that are a ratio of two sums of states, as (numerator, denominator)
            self.denom_dims = sc.objdict() # For a ratio whose denominator has fewer dimensions than its numerator: the denominator's
            for name,expr in graph.exprs.items(): # A sum of states (or flows) that all have the same dimensions has them too; so does a ratio of two, which keeps both, so that selecting groups adds up each before dividing
                parts = ratio_parts(expr)
                plain = all(all(isinstance(leaf, (ss.expressions.Stock, ss.par, ss.expressions.Flow)) for leaf in part.leaves) and not any(op in ss.expressions.format_expr(part) for op in ['/', '**']) for part in parts)
                if not plain: # E.g. a count (I.n), which is the total over every group, so the result is not by group
                    continue
                flags = [[self.applies[leaf.name] if isinstance(leaf, ss.expressions.Stock) else dims[leaf.name] for leaf in part.leaves if not isinstance(leaf, ss.par)] for part in parts]
                same = all(len(f) for f in flags) and all(np.array_equal(x, flags[0][0]) for x in flags[0]) # Within the numerator (or the sum)
                if same and len(parts) == 1:
                    dims[name] = flags[0][0]
                elif same:
                    den = np.all(flags[1], axis=0) # The dimensions every part of the denominator has; it is added up over the others
                    if np.all(flags[0][0] >= den): # None that the numerator lacks
                        dims[name] = flags[0][0]
                        self.ratios[name] = parts
                        if not all(np.array_equal(x, flags[0][0]) for x in flags[1]): # E.g. infection/S with a strain, which susceptibles do not carry
                            self.denom_dims[name] = [n for n,flag in zip(names, den) if flag]
            for name,info in self.outcomes.items(): # An outcome has the dimensions of what it is counted from (in order, since outcomes can chain)
                dims[name] = np.any([dims[key] for key,weight in info.sources if key in dims], axis=0)
                if f'n_{name}' in mod.results:
                    dims[f'n_{name}'] = dims[name]
            grouped = [] # The results kept by group, and their dimensions
            for key,flags in dims.items():
                if np.any(flags) and kept_by_group(self.sim, mod, key):
                    grouped.append((mod.results[key], flags))
            n_arrays = sum(1 + (res.name in self.ratios) for res,flags in grouped) # The numerator by group, and the denominator alongside it
            self.rows, self.group_ti = group_rows(self.sim, mod.t.npts, n_arrays*ng*8, self.group_bytes, name=mod.name)
            n_rows = mod.t.npts if self.rows is None else len(self.group_ti)
            for res,flags in grouped:
                res.init_groups(self.sim.groups.strats, [name for name,flag in zip(names, flags) if flag], (n_rows, ng), self.group_ti, ratio=res.name in self.ratios)
                if res.name in self.denom_dims:
                    res.denom_dims = self.denom_dims[res.name]
        self.init_derived_results()
        return

    def init_derived_results(self):
        """
        Results by a stratification of a derived attribute (e.g. `hiv.stratify(cd4=[0, 200, 500])`), which adds no
        groups: the number in each of its states, and each flow out of them, by the level of the attribute (at the
        source, for a flow), as well as by the sim's own groups
        """
        mod = self.mod
        sim = self.sim
        self.derived_res = sc.objdict() # The stratification of each result kept by one
        self.stock_derived = sc.objdict()
        self.flow_derived = sc.objdict()
        self.derived_trs = sc.objdict() # The stratification each transition is recorded by
        ng = sim.groups.n_groups
        names = [s.name for s in sim.groups.strats]
        for strat in self.graph.derived:
            L = strat.n
            ext = [sc.cp(s) for s in sim.groups.strats] + [sc.cp(strat)]
            coords = np.column_stack([np.repeat(np.asarray(sim.groups.group_levels).reshape(ng, -1), L, axis=0), np.tile(np.arange(L), ng)])
            for d,s in enumerate(ext):
                s.sim_coords = coords
                s.dim = d
            keys = list(strat.states)
            for name,tr in self.graph.transitions.items():
                if tr.src and all(s.name in strat.states for s in tr.src) and name not in self.derived_trs:
                    keys.append(name)
                    self.derived_trs[name] = strat
            for key in keys:
                if key in self.derived_res:
                    continue
                self.derived_res[key] = strat
                mod.results[key].init_groups(ext, names + [strat.name], (mod.t.npts, ng*L))
        return

    def derived_groups(self, strat, groups, levels):
        """ The index of each (group, level) pair among the results by a derived stratification """
        return np.asarray(groups)*strat.n + np.asarray(levels)

    def init_outcomes(self, results):
        """ Work out the sources and delay of each outcome, and the order to compute them in """
        graph = self.graph
        trans = ss.transitions
        abm = self.solver == 'abm'
        dt = self.dt
        self.outcomes = sc.objdict()
        todo = dict(graph.outcomes)
        while todo: # Outcomes can chain, so compute each one after its sources
            progress = False
            for name,oc in list(todo.items()):
                sources = []
                for base,weight in oc.flows:
                    key = ss.transitions.flow_key(base)
                    if key not in graph.transitions and key not in graph.outcomes and graph.destination(key) is None:
                        options = list(graph.transitions.keys()) + list(graph.outcomes.keys()) + graph.destinations()
                        raise sc.KeyNotFoundError(f'the source of the outcome "{name}" is not a transition or outcome of "{self.mod.name}" ({key}){trans.did_you_mean(str(key), options)}')
                    sources.append((key, weight))
                if any(key in todo for key,weight in sources):
                    continue
                info = sc.objdict(sources=sources, now=0.0, n=0.0)

                # The delay, as the share of events that are reported after each number of steps
                delay = oc.delay
                if delay is None:
                    kernel = np.ones(1)
                elif isinstance(delay, ss.Dist):
                    rvs = self.sample(delay, 20_000)
                    steps = np.maximum(np.round(rvs/dt), 0).astype(int)
                    kernel = np.bincount(steps)/len(steps)
                else:
                    steps = self.convert(delay)/dt
                    lo = int(np.floor(steps))
                    kernel = np.zeros(lo + 2)
                    kernel[lo] = 1 - (steps - lo)
                    kernel[lo + 1] = steps - lo
                info.kernel = kernel
                info.pending = np.zeros((len(kernel) + 1, self.sim.groups.n_groups)) # The events still to be reported at each of the next steps (in turn, as a ring), by group, so an outcome can be selected by stratum
                info.p_leave = info.stay = None
                if isinstance(oc.duration, ss.Dist): # The share who stay each number of steps (at least one)
                    rvs = self.sample(oc.duration, 20_000)
                    info.stay = np.bincount(np.maximum(np.round(rvs/dt), 1).astype(int))/len(rvs)
                    info.leaving = np.zeros((len(info.stay) + 1, self.sim.groups.n_groups)) # Likewise, those leaving
                elif oc.duration is not None: # An exponential stay
                    info.p_leave = -np.expm1(-dt/self.convert(oc.duration))
                self.outcomes[name] = info
                results += ss.Result(name, dtype=float, scale=abm, label=oc.label or name, kind='flow')
                if oc.duration is not None:
                    results += ss.Result(f'n_{name}', dtype=float, scale=abm, label=f'Number in {oc.label or name}', kind='stock')
                del todo[name]
                progress = True
            if not progress:
                raise ValueError(f'the outcomes {sc.strjoin(todo.keys())} of "{self.mod.name}" depend on each other in a loop.')
        return

    def sample(self, dist, n):
        """ Draw samples from a distribution of durations, in the sim's unit (used to build delay kernels, not during the run) """
        new = ss.transitions.numeric_dist(dist, self.convert)
        new.strict = False
        new.init(force=True)
        return np.asarray(new.rvs(n), dtype=float)

    def weighted_events(self, key, weight, integer):
        """ An outcome's amount by group from the people who went through a transition, each with their own weight; None if they are not known (see `Agents.weighted_events()`) """
        return None

    def step_outcomes(self):
        """ Work out this step's value of each outcome, in each group, from the flows it is counted from """
        res = self.mod.results
        ti = self.mod.ti
        ng = self.sim.groups.n_groups
        integer = self.solver in ['abm', 'gillespie', 'tau', 'binomial']

        def flow_of(key):
            if key in self.outcomes:
                return self.outcomes[key].now
            elif ng == 1:
                return np.array([self.flowcounts[key]])
            return np.asarray(self.flow_groups.get(key, np.zeros(ng)), dtype=float) # A transition with no events this step has no entry

        def store(key, vals, flow):
            res[key][ti] = vals.sum()
            res[key].store_group(ti, vals, self.rows, flow=flow)
            return

        for name,info in self.outcomes.items():
            amount = np.zeros(ng)
            for key,weight in info.sources:
                flow = flow_of(key)
                if isinstance(weight, ss.expressions.Expr):
                    own = self.weighted_events(key, weight, integer)
                    if own is not None: # Each person's own weight
                        amount += own
                        continue
                    with self.evaluating(by_group=True): # One weight for each group, as the flows are (TL51: as agents, a weight by age crashed)
                        weight = self.eval(weight)
                amount += self.outcome_share(flow, weight, integer)
            info.now = self.report_outcome(info, info, amount, ti, integer)
            if info.stay is not None or info.p_leave is not None:
                store(f'n_{name}', info.n, False)
            store(name, info.now, True)
        return

    def outcome_share(self, flow, weight, draw):
        """ The events of an outcome from those of one of its sources, times its weight: a fraction of whole events is a draw if `draw` """
        if draw and np.all((0 <= weight) & (weight <= 1)):
            return self.rng.binomial(np.round(flow).astype(int), weight)
        return flow*weight

    def report_outcome(self, info, buf, amount, ti, draw):
        """
        One step of an outcome: its events in the step (`amount`, for each group, or each run and group) are reported after
        its delay, and stay in it for its duration, each drawn for each event if `draw`. `buf` keeps the events still to be
        reported (`pending`) and to leave (`leaving`), each a ring over the next steps, and the number in the outcome (`n`).
        A step at a time for `step_outcomes()`, and over the steps of a whole run for `Compartmental.derived_outcomes()`,
        so the two draw the same. Returns the events reported in the step.
        """
        kernel = info.kernel
        slots = (ti + np.arange(len(kernel))) % len(buf.pending) # This step and the next ones
        if amount.any():
            if draw and len(kernel) > 1:
                for g in np.flatnonzero(amount):
                    buf.pending[slots, g] += self.rng.multinomial(int(round(amount[g])), kernel/kernel.sum())
            else:
                buf.pending[slots] += np.outer(kernel, amount)
        now = buf.pending[slots[0]].copy()
        buf.pending[slots[0]] = 0
        if info.stay is not None: # Each arrival leaves after a stay drawn for it
            slots = (ti + np.arange(len(info.stay))) % len(buf.leaving)
            if draw:
                for g in np.flatnonzero(now):
                    buf.leaving[slots, g] += self.rng.multinomial(int(round(now[g])), info.stay)
            else:
                buf.leaving[slots] += np.outer(info.stay, now)
            buf.n = buf.n + now - buf.leaving[slots[0]]
            buf.leaving[slots[0]] = 0
        elif info.p_leave is not None:
            buf.n = buf.n + now - buf.n*info.p_leave
        return now

    def start_step(self):
        """ Start counting the flows of the step """
        self.cache.clear()
        for name in self.flowcounts:
            self.flowcounts[name] = 0.0
        if hasattr(self, 'flow_groups'):
            self.flow_groups.clear()
        if getattr(self, 'flow_derived', None):
            self.flow_derived.clear()
        return

    def record_stocks(self):
        """ Record the number in each state at the end of the step (so `R[ti]` has the people who recovered in the step that ends at time point `ti`, as `recovery[ti]` does) """
        for state in self.graph.states.values():
            self.stocks[state.name] = self.stock(state)
        return

    def with_unrepresented(self):
        """
        The number in each state (and by group), as the results count them: with dynamic rescaling (rescale=True), including the people no agent stands for yet

        Each agent stands for `scale` people; the people no agent stands for (`sim.results.unrepresented`, see `ss.Sim.rescale()`) are
        like the agents, and are shared out as the agents alive are, in agents (as the other counts are, before they are scaled). For a
        disease, which rescaling makes naive again, they have never been infected, so their share is in its first state; for another
        module (e.g. `ss.Pregnancy`), each state has its share. The engine's own stocks (used in the dynamics) are untouched.
        """
        sim = self.sim
        if not sim.pars.rescale or sim.solver != 'abm' or 'unrepresented' not in sim.results:
            return self.stocks, self.stock_groups
        n_alive = sim.people.n_alive
        extra = sim.results.unrepresented[sim.ti]/self.scale/n_alive if n_alive else 0.0 # For each agent alive, in agents
        if not extra:
            return self.stocks, self.stock_groups
        people = [state.name for state in self.graph.people_states]
        stocks = sc.objdict(self.stocks)
        stock_groups = sc.objdict(self.stock_groups)
        first = self.graph.first
        if any(self.mod is mod for mod in sim.diseases()): # In its first state
            if first.people:
                stocks[first.name] = stocks[first.name] + extra*sum(stocks.get(name, 0.0) for name in people)
                if first.name in stock_groups:
                    stock_groups[first.name] = stock_groups[first.name] + extra*sum(stock_groups[name] for name in people if name in stock_groups)
        else: # In each state, as the agents are
            for counts in [stocks, stock_groups]:
                for name in people:
                    if name in counts:
                        counts[name] = counts[name]*(1 + extra)
        return stocks, stock_groups

    def update_results(self):
        mod = self.mod
        graph = self.graph
        res = mod.results
        ti = mod.ti
        scale = ss.results.people_scale(self.sim) # As agents, counts of people are written in people
        if getattr(self, '_in_people', (None, None))[0] != len(res): # The results counted in people, found once (results are not added during the run)
            self._in_people = (len(res), [r for r in res.values() if getattr(r, 'in_people', False)])
        people = self._in_people[1]
        for r in people: # The values by group, as they are kept
            r.write_scale = scale
        if getattr(getattr(self.sim.people, 'scale', None), 'weighted', False): # Agents of different weights (people.scale) make fractions of people
            for key in list(self.stocks) + list(self.flowcounts):
                if res[key].values.dtype.kind in 'iu':
                    res[key].init_values(dtype=float, force=True)
        stocks, stock_groups = self.with_unrepresented()
        for name,val in stocks.items():
            res[name][ti] = val
        for name,val in self.flowcounts.items():
            res[name][ti] = val
        self.written_ti = ti # See Agents.infect_now()
        self.step_outcomes()
        derived = getattr(self, 'derived_res', {})
        for name,val in stock_groups.items():
            if name not in derived:
                res[name].store_group(ti, val, self.rows)
        for name,val in self.flow_groups.items():
            if name not in derived:
                res[name].store_group(ti, val, self.rows, flow=True)
        for key,strat in derived.items(): # By a derived stratification instead (whose results keep every time point)
            res[key].store_group(ti, self.stock_derived.get(key, 0.0) if key in self.graph.states else self.flow_derived.get(key, 0.0))
        if self.rows is not None: # The flows by group so far in this row, for the results worked out from them
            if ti == 0 or self.rows[ti] != self.rows[ti-1]:
                self.row_flows = sc.objdict()
            for name,val in self.flow_groups.items():
                self.row_flows[name] = self.row_flows.get(name, 0) + val
        if graph.exprs:
            ctx = StockCtx(self, stocks)
            for name,expr in graph.exprs.items():
                res[name][ti] = ss.expressions.evaluate(expr, ctx)
                if getattr(res[name], 'strata', None) is not None: # By group: in a row of several steps, from the states at its end and the flows over it
                    ctx_g = StockCtx(self, stock_groups, self.flow_groups if self.rows is None else self.row_flows)
                    if name in self.ratios:
                        num, den = [ss.expressions.evaluate(part, ctx_g) for part in self.ratios[name]]
                        row = ti if self.rows is None else self.rows[ti]
                        res[name].strata[row], res[name].denom[row] = num, den
                    else:
                        res[name].store_group(ti, ss.expressions.evaluate(expr, ctx_g), self.rows)
        if scale != 1:
            for r in people: # The values written this step
                r[ti] = r[ti]*scale
        coded = [state.name for state in mod.auto_state_list if state.name not in graph.states] + mod.derived_state_names
        if coded: # The module's own ss.BoolStates (and derived states), counted as for a module written as code
            ss.Module.update_results(mod, names=coded)
        return

    def register_sets(self):
        """
        For a movement register: the sets of states its events name, with the default (None) first,
        and for each, the states of this module that people are chosen from and that people are added to

        A state that the register names must belong to some module of the sim; one that is not in
        this module is left out, and a module with none of the states of a set uses its default.
        """
        sim = self.sim
        graph = self.graph
        d = sim.mixing.space
        placed = [s.name for s in graph.people_states if self.applies[s.name][d]] # States that people in a place can be in
        ev = sim.mixing.events()
        sets = [(None, None)]
        for key in ev.sets:
            if key[0] is not None and key not in sets:
                sets.append(key)
        known = {s.name for mod in sim.declared for s in mod.graph.people_states}
        out = []
        for states,to_states in sets:
            if states is not None:
                unknown = [name for name in states + (to_states or ()) if name not in known]
                if unknown:
                    errormsg = f'The movement register names the state{"s" if len(unknown) > 1 else ""} {sc.strjoin(unknown)}, which no disease has; the states are {sc.strjoin(sorted(known))}'
                    raise KeyError(errormsg)
            own = [name for name in (states or []) if name in placed]
            shift = {name:to for name,to in zip(states, to_states) if name in placed} if to_states is not None else {} # Where the people chosen from each state go
            out.append(sc.objdict(states=states, to_states=to_states, chosen=own or placed, added=own or placed[:1], shift=shift))
        if np.any((ev.src < 0) & np.array([key[0] is None for key in ev.sets], dtype=bool)[ev.set_of]):
            ss.note(f'people added by the movement register go into {placed[0]}, the first state of "{self.mod.name}", since the register does not name states for them', sim=sim)
        return out

    def check_short(self, short):
        """ Warn, once, if events of a movement register asked for more people than there were """
        if short > 0 and not getattr(self, '_warned_short', False):
            self._warned_short = True
            ss.warn(f'Events of the movement register asked "{self.mod.name}" for {short:n} more people than there were in the places and states they named; they moved (or removed) everyone there was instead')
        return

    def note_chance(self):
        """ For a run with real numbers (ode): say where chance would matter, as a stochastic run would show """
        sim = self.sim
        infected = [s for s in self.graph.infected if s.people]
        if sim.solver != 'ode' or not infected or not self.graph.transmissions:
            return
        res = self.mod.results
        n = sum(np.asarray(res[s.name].values, dtype=float) for s in infected)
        name = self.mod.name
        if 0 < self.n_start < 10:
            ss.note(f'"{name}" starts with {self.n_start:.3g} infected, so whether it takes off is a matter of chance, which solver="{sim.solver}" does not show; '
                    f'solver="gillespie" with ss.parallel(n_runs=...) does', sim=sim)
        low = np.flatnonzero((n > 0) & (n < 1))
        if len(low) and n[low[0]:].max() > 10:
            when = res[infected[0].name].timevec[low[0]]
            when = f'{sim.unit_name} {float(when):.4g}' if sc.isnumber(when) else when # e.g. day 125, not 124.99999984
            ss.note(f'the number infected in "{name}" falls to {n[low[0]]:.2g} (at {when}) and then rises to {n[low[0]:].max():,.0f}; a stochastic run would usually die out there. '
                    f'solver="gillespie" with ss.parallel(n_runs=...) shows how often, in the runs of msim.results.{name}.{infected[0].name} after that point', sim=sim)
        return

    def finalize_results(self):
        """ Cumulative results, computed after scaling """
        res = self.mod.results # Already scaled, by group too: see ss.Result.apply_scale()
        self.note_chance()
        return


class StockCtx:
    """
    Evaluates result expressions (sums and ratios of states, and of flows) from the counts after the step

    Args:
        engine (`ss.solvers.Engine`): the engine
        stocks (dict): the number in each state: one number for the current step (the default), or an array over the steps
        flows (dict): the same for the flow along each transition
    """
    def __init__(self, engine, stocks=None, flows=None):
        self.engine = engine
        self.stocks = engine.stocks if stocks is None else stocks
        self.flows = engine.flowcounts if flows is None else flows
        self.scale = engine.scale if isinstance(engine, ss.solvers.Agents) else 1.0 # As agents, the stocks are numbers of agents
        return

    def flow(self, flow):
        """ The number of people who made a transition in the step """
        return self.flows.get(flow.name, 0.0)*(self.scale if flow.tr.people else 1.0)

    def value(self, p):
        return self.engine.value(p)

    def prop(self, cond):
        errormsg = f'a result is a sum or ratio of states and flows, which {cond} (a property of each person) is not; for a result by a property, stratify by it (sim.stratify(...)) and use res.<result>.by(...), or count the people it holds for with an analyzer'
        raise ss.CapabilityError(errormsg)

    def stock(self, state):
        """ The number of people (or the count) in a state """
        return self.stocks[state.name]*(self.scale if state.people else 1.0)

    def count(self, states):
        return sum(self.stock(s) for s in states)

    def frac(self, states):
        n = self.count(self.engine.graph.people_states)
        return np.divide(self.count(states), n, out=np.zeros(np.shape(n)), where=n>0)[()]

    local_count = count # A result is a total, wherever the people are
    local_frac = frac
