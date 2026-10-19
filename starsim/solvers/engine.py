"""
The base of the engines that run a declared model (`Engine`), and what they share
"""
import zlib
import inspect
import numpy as np
import sciris as sc
import starsim as ss
from types import SimpleNamespace as Bunch # A plain object with attributes; much faster to read than an objdict, which matters inside the ODE's right-hand side

__all__ = ['Engine', 'CountState', 'StockCtx', 'convert', 'add_rows', 'at_ends', 'by_level', 'no_tables']


no_tables = (np.zeros((0, 0, 0)),)*4 # For a step whose program holds for the whole step: see ss.kernels.step()


def convert(val, unit):
    """ Convert a typed quantity to a plain number in a unit: a duration to units, a rate to per unit """
    if isinstance(val, ss.TimePar):
        if isinstance(val, ss.prob): # A probability per period becomes the hazard that gives it
            period = 1.0 if val.unit is None else val.unit.to(unit).value # With no unit, the probability is per unit of the sim's time
            return -np.log1p(-val.value)/period
        return val.to(unit).value
    return val


def changed(old, new):
    """ Whether the number inside a typed quantity differs from the one it was converted from """
    if isinstance(new, np.ndarray):
        return not np.array_equal(old, new)
    return old != new


def add_rows(out, rows, tis, vals):
    """ Add values at consecutive time points `tis` (along the first axis of `vals`) into the rows of `out` they belong to: see `ss.Sim.group_rows()` """
    r = rows[tis]
    firsts = np.flatnonzero(np.diff(r, prepend=-1)) # Where each row starts
    out[r[firsts]] += np.add.reduceat(vals, firsts, axis=0)
    return


def at_ends(out, rows, ends, tis, vals):
    """ Put the values at the time points among `tis` that end a row into that row of `out` """
    last = np.isin(tis, ends)
    out[rows[tis[last]]] = vals[last]
    return


def by_level(levels, n):
    """ The positions of the entries at each of n levels, as where each level's start and end, and the positions in order (`np.flatnonzero(levels == i)` for each level, by one sort) """
    order = np.argsort(levels, kind='stable')
    return np.searchsorted(levels[order], np.arange(n + 1)), order


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
        self.engine.set_count(self.state, value)
        return

    @property
    def frac(self):
        """ The number in this state as a fraction of the module's people """
        return self.engine.frac([self.state])

    @property
    def values(self):
        """ The number in this state in each group """
        return self.engine.count([self.state], total=False)

    @property
    def uids(self):
        """ Agents, which a compartmental run does not have """
        raise ss.CapabilityError(f'{self.engine.mod.name}.{self.name}.uids was asked for, but a compartmental run has the number in each state ({self.engine.mod.name}.{self.name}.n), not agents.\n'
                                 f'     Run with solver="abm".')


class Engine:
    """
    Base class for the engines that run a graph

    Args:
        module (`ss.Module`): the module whose graph is run
        sim (`ss.Sim`): the sim
    """
    group_bytes = 100e6 # The most memory that a module's results may take by group in one run; above it, they are kept every few steps (see ss.Sim.group_rows())

    def __init__(self, module, sim):
        self.mod = module
        self.sim = sim
        self.graph = module.graph
        self.cache = dict() # Parameter values for the current step
        self.fixed = dict() # Converted values of plain parameters, which hold for the whole run
        self.arities = dict() # The number of arguments of each user function: see n_args()
        self.converted = dict() # Typed values converted to the sim's unit: see convert()
        self.rng = self.make_rng() # For draws that are not per agent
        self.dwells = sc.objdict() # Dwell-time information for each transition that has one
        self.flowcounts = sc.objdict({name:0.0 for name in self.graph.transitions}) # Events in the current step
        self.stocks = sc.objdict() # The number in each state at the start of the step
        self._by_group = False # Whether rates are being worked out for groups rather than for agents (in an agent-based run, for the states that are counts)
        self.strain = None # The position of the module's strain dimension, if it has one
        self.move_levels = dict() # For each transition with move=: see moves()
        self._strain = None # The strain whose force of infection is being worked out
        self._routes = None # For a transmission with a beta for each route, the beta of each term of the mixing: see Compartmental.route_betas()
        self.register = None # The movement register, as arrays for the kernels
        self.year = None # The year in the step that values that vary in time are taken at, for its stages (None for its start)
        self.immunity = None # Protection between strains, and the states whose people it applies to: see init_immunity()
        self.immune_states = []
        self.cross = None # In a compartmental run, the diseases whose states are tracked jointly with this one's, if any: see ss.solvers.Crossing
        self.kernel_rng = None # The random numbers of a run stepped by the kernels: see init_program()
        ss.kernels.announce(verbose=sim.pars.get('verbose', 0)) # A mock sim, e.g. for equations(), runs nothing
        return

    def shrink(self):
        """ Drop the sim and the arrays the run worked in, for saving; the results are kept on the module """
        arrays = [key for key,val in vars(self).items() if isinstance(val, np.ndarray) and val.nbytes > 1000]
        ss.shrink(self, ['sim', '_selector', 'cache', 'series', 'deliveries', 'program', 'register', 'shifts', 'jax_run'] + arrays) # The selector holds the sim, the cache the values per agent, and the series the tables of a compartmental run
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
            dims = [dim for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by) for dim in leaf.raw.dims]
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
        """ The change of level of a transition with `move=`, as ((dimension, level), ...), or None """
        if tr.move is None:
            return None
        if tr.name not in self.move_levels:
            out = []
            for dim,level in tr.move.items():
                if dim not in [strat.name for strat in self.sim.groups.strats]:
                    raise ss.MissingDimension(f'the transition "{tr.name}" of "{self.mod.name}" moves people in "{dim}", which is not a dimension of the sim. Stratify the disease or the sim by it.')
                d = self.sim.groups.dim_index(dim)
                for state in tr.dst:
                    if not self.applies[state.name][d]:
                        raise ss.MissingDimension(f'the transition "{tr.name}" of "{self.mod.name}" moves people in "{dim}", but the state they go to ("{state.name}") has no level of it.')
                out.append((d, self.sim.groups.strats[d].index(level)))
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
                out = np.ones((1, 1))
                for d,strat in enumerate(self.sim.groups.strats):
                    n = strat.n
                    factor = np.zeros((n, n))
                    if d in moved:
                        factor[:, moved[d]] = 1
                    elif d == self.strain and strain is not None and (b[d] or record):
                        if a[d] and not record and strat.carries is not None: # Someone who carries other strains carries this one as well
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
        if self.sim.mixing.space is not None and any(isinstance(leaf, ss.transitions.Count) for leaf in tr.rate.leaves): # A rate that uses the number in a place gives the inflow in that place
            return self.inflow_local[name]
        given = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by)]
        if given: # A rate given for each level of some dimensions is the inflow at that level, so is shared out over the other dimensions only
            dims = {self.sim.groups.dim_index(dim) for by in given for dim in by.dims}
            factors = self.inflow_factors[name]
            return self.sim.groups.fold(np.prod([np.ones(len(factor)) if d in dims and self.applies[name][d] else factor for d,factor in enumerate(factors)], axis=0)) if factors else self.inflow_splits[name]
        return self.inflow_splits[name]

    def init_immunity(self):
        """
        With immunity= on a strain dimension: the states whose people can be infected again (those that
        have a strain, are not infected, and are not otherwise infected, such as R), and their protection
        """
        self.immunity = None
        self.immune_states = []
        if self.strain is None:
            return
        strat = self.sim.groups.strats[self.strain]
        own = [s for s in self.graph.strats if s.name == strat.name]
        immunity = own[0].immunity if own else strat.immunity
        if immunity is None:
            return
        sources = {id(src) for tr in self.graph.transmissions for src in tr.src}
        infected = {id(s) for s in self.graph.infected}
        self.immunity = immunity
        self.immune_states = [s for s in self.graph.people_states if self.applies[s.name][self.strain] and id(s) not in infected and id(s) not in sources]
        if not self.immune_states:
            ss.note(f'immunity= on "{strat.name}" has no effect, since nobody who has had a strain of "{self.mod.name}" can be infected again', sim=self.sim)
        elif self.sim.solver != 'abm':
            ss.note(f'a compartmental run keeps only the last strain of "{strat.name}" each person has had, so with partial immunity they can be infected again by one they had before; run as agents to keep every strain', sim=self.sim)
        return

    def protection(self, levels, strain):
        """ The share of the force of infection of a strain that reaches people whose last strain was each of some levels """
        return 1 - self.immunity[levels, strain]

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
        """ The strain dimension, if its levels include people who carry several strains at once (see `ss.transitions.Strat.init_carriage()`) """
        strat = None if self.strain is None else self.sim.groups.strats[self.strain]
        return strat if strat is not None and strat.carries is not None else None

    def init_strain_weights(self):
        """
        The share of the infectiousness of the people at each level of the strain dimension that goes to each strain (levels by strains)

        Someone who carries one strain passes it on; someone who carries several passes on
        each in proportion to its infectiousness (from `adjust=dict(rel_trans=...)`), and their
        own infectiousness is that of their level.
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
    def dt(self):
        """ The module's timestep, in the sim's unit (worked out again only if the timestep is replaced) """
        dt = self.mod.t.dt
        cached = self.__dict__.get('_dt')
        if cached is None or cached[0] is not dt:
            cached = self._dt = (dt, dt.to(self.sim.unit_base).value)
        return cached[1]

    @property
    def total_pop(self):
        return self.sim.pars.total_pop

    #%% Values

    def convert(self, val):
        """ Convert a typed quantity to a plain number in the sim's unit: a duration to units, a rate to per unit; each is converted once, unless its number is changed """
        if not isinstance(val, ss.TimePar):
            return val
        known = self.converted.get(id(val))
        if known is None or known[0] is not val or changed(known[1], val.value):
            if len(self.converted) > 10_000: # Values made as the sim runs (e.g. by a time series) are kept with the others, so are let go of now and then
                self.converted.clear()
            number = np.copy(val.value) if isinstance(val.value, np.ndarray) else val.value
            known = self.converted[id(val)] = (val, number, convert(val, self.sim.unit_base))
        return known[2]

    def value(self, p):
        """ The value of a parameter, as a number in the sim's unit """
        key = id(p)
        if key in self.cache:
            return self.cache[key]
        raw = p.raw
        if isinstance(raw, ss.Dist): # Not a dwell time (those are handled separately), so use the mean
            out = ss.transitions.dwell_stats(raw, self.convert)[0]
        elif callable(raw):
            out = self.call(raw, p)
            if isinstance(out, ss.transitions.Expr): # A function that uses the model's own states (I.n) gives an expression, which is worked out here
                self.cache.setdefault('held', []).append(out) # Values are cached by id, so the expression's parts are kept until the cache is cleared, so that no new object can take their id
                out = self.eval(out)
            return self.modified(p, self.convert(out)) # Not cached, since it can depend on the current state
        elif isinstance(raw, dict): # A beta for each route; in a compartmental run it is applied to the exposure through each instead (see Compartmental.route_betas())
            out = 1.0 if self._routes is not None else {k:self.convert_beta(v) for k,v in raw.items()}
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

    @property
    def selector(self):
        """ Evaluates conditions for this module: see `ss.conditions.Selector` """
        if getattr(self, '_selector', None) is None:
            self._selector = ss.conditions.Selector(self.sim, self.mod)
        return self._selector

    def modified(self, p, value):
        """ Apply the modifiers that are active on a parameter: set, then multiply, then add """
        mods = self.sim.modifiers.get((self.mod.name, 'par', p.name)) if (p.name and not p.literal) else None
        if not mods:
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

    def contact_mult(self, name):
        """ The multiplier on the contacts of a network from the modifiers of its `n_contacts` (e.g. `ss.modify('school.pars.n_contacts', multiply=0.2)`): 1 if it has none """
        mods = self.sim.modifiers.get((name, 'par', 'n_contacts')) if name is not None else None
        if not mods:
            return 1.0
        key = ('contacts', name, None)
        if key not in self.cache:
            val = self.sim.all_networks()[name].pars.n_contacts
            base = ss.solvers.Mixing.mean(val, self.sim.unit_base) if np.ndim(val) == 0 else 1.0 # A matrix of contacts can only be scaled
            base = base if base > 0 else 1.0
            value = self.reduce(ss.connectors.combine(mods, base, self), what=f'{name}.pars.n_contacts')
            self.cache[key] = self.clip(value, f'{name}.pars.n_contacts')/base
        return self.cache[key]

    def branch_mult(self, tr, uids=None):
        """ The multiplier on a branch's probability from its modifiers (for some agents, if `uids` are given), or None if it has none """
        return self.branch_mult_of(tr.name, uids) if tr is not None else None

    def branch_mult_of(self, name, uids=None):
        """ As `branch_mult()`, by name: a transition, or one destination of a transition with several (`'progression.C'`) """
        if not self.sim.modifiers.get((self.mod.name, 'branch', name)):
            return None
        out = self.mult_raw('branch', name)
        if uids is not None and np.ndim(out):
            out = out[uids]
        return out

    def dest_shares(self, tr, ps, uids=None):
        """ The probability of each destination of a transition with several, with the modifiers of each (`'progression.C'`), as `branch_shares()` does for branches """
        mults = [self.branch_mult_of(f'{tr.name}.{dst.name}', uids) for dst in tr.dst]
        if all(m is None for m in mults):
            return ps
        return self.branch_shares([f'{tr.name}.{dst.name}' for dst in tr.dst], ps, mults)

    def branch_shares(self, trs, ps, mults):
        """
        The probability of each branch out of a state, with their modifiers

        A modifier on a branch multiplies its probability, and the branches with none share
        what is left in proportion to their own; if every branch is modified (or the modified
        ones come to more than 1), they are scaled to sum to 1.

        Args:
            trs (list): the transitions (or their names)
            ps (list): the probability of each, or None for the one that takes the rest
            mults (list): the multiplier of each, or None for those with no modifiers
        """
        rest = 1.0 - sum(p for p in ps if p is not None)
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
                ss.note(f'the modifiers of {path} made the probabilities of the branches come to more than 1, so they were scaled to sum to 1', ss.ClippedToRange, sim=self.sim)
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

    def clip(self, value, path):
        """ A modified rate, duration, or multiplier that has gone below 0 (e.g. by add=) is 0; the first time, a note says so, and sim.what_modifies() reports it """
        if np.any(np.asarray(value) < 0):
            if path not in self.sim.clipped:
                self.sim.clipped.add(path)
                ss.note(f'the modifiers of {path} made it negative, so it was set to 0 where it was', ss.ClippedToRange, sim=self.sim)
            value = np.maximum(value, 0)
        return value

    def reduce(self, value, state=None, what=None):
        """ A modified value as the engine uses it; see the subclasses """
        return value

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on a quantity that is not a parameter (`rel_sus`, `rel_trans`, or a transition's rate) from its modifiers, and from the adjust= of the stratifications """
        mods = self.sim.modifiers.get((self.mod.name, kind, name))
        out = 1.0
        if mods:
            key = (kind, name, state.name if state is not None else None)
            if key not in self.cache:
                self.cache[key] = self.clip(self.reduce(ss.connectors.combine(mods, 1.0, self), state=state, what=name or kind), f'{self.mod.name}.{name or kind}')
            out = self.cache[key]
        if getattr(self, 'rel_adjust', None) and self.rel_adjust.get(kind):
            out = out*self.adjusted_rel(kind, state)
        return out

    def adjusted_rel(self, kind, state=None):
        """ The multiplier on rel_trans or rel_sus for each group in a state, from the adjust= of the stratifications: see the subclasses """
        out = 1.0
        for d,vals in self.rel_adjust[kind]:
            if state is None or self.applies[state.name][d]:
                out = out*self.sim.groups.group_vals(d, vals)
        return out

    def by_group(self, by):
        """ Expand a value for each level of one or more dimensions to a value for each group, once a step """
        key = ('by', id(by))
        if key not in self.cache:
            self.cache[key] = by.per_group(self.sim, self.convert, year=self.year)
        return self.cache[key]

    def init_adjust(self):
        """ The multiplier on each transition in each group, from the adjust= of every stratification """
        sim = self.sim
        self.adjust = sc.objdict({name:None for name in self.graph.transitions})
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
                        raise ss.InvalidAdjustment(f'the adjustment of {name} by "{strat.name}" cannot apply, since the people it is for ({sc.strjoin(s.name for s in states)}) have no level of "{strat.name}".')
                    self.rel_adjust[name].append((d, vals))
                elif name in self.adjust:
                    tr = self.graph.transitions[name]
                    if tr.src and not any(self.applies[src.name][d] for src in tr.src) and not (d == self.strain and tr.contact):
                        raise ss.InvalidAdjustment(f'the adjustment of "{name}" by "{strat.name}" cannot apply, since the people it moves ({sc.strjoin(src.name for src in tr.src)}) have no level of "{strat.name}". '
                                                   f'To make where they arrive depend on it, give "{strat.name}" a split instead.')
                    if not tr.src and tr.dst and not any(self.applies[dst.name][d] for dst in tr.dst): # An inflow into states with no level of it: all of it would arrive at the first level, at that level's multiplier
                        raise ss.InvalidAdjustment(f'the adjustment of "{name}" by "{strat.name}" cannot apply, since what it brings in ({sc.strjoin(dst.name for dst in tr.dst)}) has no level of "{strat.name}". '
                                                   f'Stratify {sc.strjoin(dst.name for dst in tr.dst)} by "{strat.name}" too (e.g. kind=\'space\' for a place), or leave the adjustment out.')
                    if d == self.strain and tr.contact:
                        self.strain_adjust[name] = vals
                    else:
                        current = 1.0 if self.adjust[name] is None else self.adjust[name]
                        self.adjust[name] = current*sim.groups.group_vals(d, vals)
        for name,tr in self.graph.transitions.items(): # A value by level of a dimension that the states moved have no level of: there is no level to take (it took the first)
            ends = tr.src or tr.dst or []
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p):
                for leaf in (expr.leaves if isinstance(expr, ss.transitions.Expr) else []):
                    if isinstance(getattr(leaf, 'raw', None), ss.by):
                        for d in leaf.raw.check(sim)[0]:
                            if ends and d != self.strain and not any(self.applies[s.name][d] for s in ends):
                                dim = sim.groups.strats[d].name
                                raise ss.MissingDimension(f'"{name}" uses a value by level of "{dim}" ({leaf.raw}), but {sc.strjoin(s.name for s in ends)} {"has" if len(ends) == 1 else "have"} no level of "{dim}", so there is no level to take it from.\n'
                                                          f'     Stratify {sc.strjoin(s.name for s in ends)} by "{dim}" too, or give the value without ss.by().')
        self.init_strain_weights()
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

    def n_args(self, func):
        """ The number of arguments a user function takes, worked out once for each function """
        known = self.arities.get(id(func))
        if known is None or known[0] is not func:
            known = self.arities[id(func)] = (func, len(inspect.signature(func).parameters))
        return known[1]

    def call(self, func, p=None):
        """ Call a user function of the sim: once a step, if it takes only the sim """
        if isinstance(func, ss.transitions.TimeValue): # Its values converted, then interpolated as numbers
            held = self.year is None or ss.solvers.Compartmental.held(func) # A step function is held at its value at the start of the step, as in a whole run (see Compartmental.tables())
            return func.series([self.sim.t.now('year') if held else self.year], self.sim, self.convert)[0]
        if self.n_args(func) >= 2:
            return self.call_uids(func, p)
        key = ('called', id(func))
        if key not in self.cache:
            self.cache[key] = func(self.sim)
        return self.cache[key]

    def call_uids(self, func, p=None):
        name = getattr(func, '__name__', 'function')
        raise ss.CapabilityError(f'the function "{name}" cannot run with solver="{self.sim.solver}".\n'
                                 f'     It takes uids, which exist only per agent. Run with solver="abm".')

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
        method = dict(n='local_count', total='total_count', frac='local_frac')[expr.kind]
        out = getattr(self, method)(own) if own else 0.0
        for name,states in others.items():
            out = out + getattr(self.sim.get_declared(name).engine, method)(states)
        return out

    #%% Dwell times

    def init_dwells(self):
        """ Work out the dwell-time distribution of every transition that has one """
        trans = ss.transitions
        unit = self.sim.unit_base
        to_unit = lambda val: convert(val, unit) # As self.convert, but without holding the engine, since a function parameter of a dwell time keeps it
        for name,tr in self.graph.transitions.items():
            p = tr.dur if tr.dur is not None else tr.rate if (isinstance(tr.rate, trans.par) and not tr.contact) else None
            if p is None:
                continue
            raw = p.raw
            dw = Bunch(par=p, dist=None, k=tr.stages or 1, exact=True, is_rate=tr.dur is None, family='exponential', std=None, fixed=False)
            if isinstance(raw, ss.Dist) and tr.dur is not None:
                mean, std = trans.dwell_stats(raw, to_unit)
                dw.family = type(raw).__name__
                dw.mean, dw.std = mean, std
                dw.dist = trans.numeric_dist(raw, to_unit)
                if np.isnan(mean): # A parameter is a function of the agents
                    if self.sim.solver != 'abm':
                        raise ss.CapabilityError(f'"{name}" cannot run with solver="{self.sim.solver}": its dwell time {raw} has a parameter that is a function of the agents. Run with solver="abm".')
                elif dw.family == 'expon':
                    dw.family = 'exponential'
                elif dw.family == 'constant' and tr.stages is None: # A fixed delay: a conveyor in a compartmental run (see Compartmental.build_layout())
                    dw.fixed = True
                elif tr.stages is None:
                    shape = raw.pars.get('a')
                    if dw.family == 'gamma' and sc.isnumber(shape) and shape == int(shape):
                        dw.k = int(shape)
                    else:
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

    def note_beta(self):
        """ Warn about a beta dict of bare numbers with a step shorter than the unit: in Starsim 3 these were probabilities per step, so a port copied verbatim is too low by a factor of 1/dt """
        if self.dt >= 1:
            return
        bare = lambda v: sc.isnumber(v) or (isinstance(v, (list, tuple)) and all(sc.isnumber(x) for x in v))
        for tr in self.graph.transmissions:
            for leaf in tr.rate.leaves:
                raw = getattr(leaf, 'raw', None)
                if isinstance(raw, dict) and raw and all(bare(v) for v in raw.values()):
                    unit = self.sim.unit_name
                    ss.warn(f'{self.mod.name}.{tr.name}: beta={raw} has numbers with no unit, so they are rates per {unit}, with dt = {self.dt:.3g} {unit}. '
                            f'In Starsim 3, a beta dict of bare numbers was a probability per step: if these came from Starsim 3, give each as one (e.g. ss.prob(..., "month") for monthly steps); '
                            f'to keep them as rates and silence this, give each a unit (e.g. ss.per{unit}(...)).')
        return

    def note_steps(self):
        """ In a run that takes whole steps, say which transitions are fast compared with the timestep """
        sim = self.sim
        self.note_beta()
        for tr in self.graph.transitions.values(): # A rate per person that grows with a count of people, which is often a count written for a share
            if tr.people and not tr.inflow and tr.rate is not None and ss.transitions.count_power(tr.rate) > 0:
                ss.note(f'{tr.name} has a rate per person of {ss.transitions.format_expr(tr.rate)}, which grows with the number of people (density-dependent); for a share of the population, write .frac (e.g. I.frac), or give the transmission as ss.transmission(..., beta=...)', sim=sim)
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
                        ss.note(f'{name} has a mean dwell of {mean:.2g} {unit}{"s" if mean != 1 else ""}, shorter than the step; a duration is written dur=, or ss.days(10)', ss.LargeStep, sim=sim)
            return
        dt = self.dt/(sim.pars.substeps if euler else 1)
        for name,dw in self.dwells.items():
            if name in inflows:
                continue
            if dw.dist is None and (callable(dw.par.raw) or isinstance(dw.par.raw, ss.by)): # Not known until the sim runs, or not one number
                continue
            mean = dw.mean if dw.dist is not None else self.dwell_mean(name)
            if np.ndim(mean) == 0 and mean > 0 and dt/mean > 0.1: # The chain-binomial solver moves people through the stages exactly (see ss.kernels.draw_chain()), and agents draw the whole dwell time, so it is the mean that matters
                x = dt/mean
                share = min(x, 1) if euler else -np.expm1(-x)
                msg = f'{name} has rate × dt = {x:.2g} ({1/mean:.3g} per {unit} × {dt:n} {unit}{"s" if dt != 1 else ""}), so {"almost everyone leaves" if share > 0.99 else f"{share:.2g} of people leave"} in a step'
                if euler:
                    msg += ' (Euler scales outflows that would take more than everyone down together)' if x > 1 else ''
                else: # Whole steps are off the ODE by an amount of order rate × dt, which can be large: about 15% in the prevalence of a seasonal SIRS at 0.12
                    fixes = 'a smaller dt, substeps=,' if sim.solver == 'binomial' else 'a smaller dt'
                    msg += f'. Taking whole steps, solver="{sim.solver}" can differ from the ODE by several times that in prevalence; use {fixes} or solver="gillespie"'
                ss.note(f'{msg}; sim.check_dt() shows how much a smaller step changes the results', ss.LargeStep, sim=sim)
        return

    def note_stages(self, name, dw):
        """ In a compartmental run, a dwell time that is not exponential or Erlang is approximated by Erlang stages, which is said in one line """
        sim = self.sim
        if sim.solver == 'abm':
            return
        approx = sim.pars.approx
        unit = f' {sim.unit_name}s'
        std = dw.mean/np.sqrt(dw.k)
        if f'{std:.3g}' == f'{dw.std:.3g}':
            kept = f'mean {dw.mean:n}{unit} and std {std:.3g}{unit}, as given'
        else:
            kept = f'mean {dw.mean:n}{unit} kept; std {std:.3g}{unit}, against {dw.std:.3g}{unit} given'
        what = f'{dw.k} Erlang stage{"s" if dw.k > 1 else ""} ({kept})'
        if approx is False:
            raise ss.CapabilityError(f'"{name}" cannot run with solver="{sim.solver}" and approx=False.\n'
                                     f'     It has a {dw.family} dwell time, which would be approximated by {what}.\n'
                                     f'     Run with solver="abm", give stages= on the transition, or allow the approximation.')
        else:
            fixed = ' (nearly fixed, not exponential)' if dw.std < dw.mean/2 else '' # e.g. ss.SIR(dur_inf=ss.days(7)), whose std is 1 day
            ss.note(f'{name} has a {dw.family} dwell time{fixed}, approximated by {what}', ss.StagesUnderAuto, sim=sim, verbose=0 if approx else None)
        return

    def dwell_mean(self, name):
        """ The current mean dwell time of a transition, in the sim's unit """
        dw = self.dwells[name]
        if dw.dist is not None:
            return dw.mean
        val = self.value(dw.par)
        return np.divide(1, val, out=np.full(np.shape(val), np.inf), where=np.asarray(val)!=0)[()] if dw.is_rate else val # A rate of 0 is a dwell that never ends

    #%% Initial conditions

    def init_fractions(self):
        """ The fraction of people who start in each state, and the count in each state that is not people """
        graph = self.graph
        fracs = sc.objdict({s.name:0.0 for s in graph.people_states})
        counts = sc.objdict({s.name:0.0 for s in graph.count_states})
        for state in graph.states.values():
            init = state.init
            if init is None:
                continue
            if isinstance(init, ss.transitions.par):
                init = init.raw
            elif isinstance(init, ss.transitions.Expr): # An expression of parameters, e.g. I0/2
                init = float(self.eval(init))
            if isinstance(init, ss.by): # A value for each level of one or more dimensions: a number of people, or a fraction of the people at that level
                sim = self.sim
                ds, given = init.grid(sim)
                grid, index = sim.groups.grid_index(ds, given, count=init.is_count() or np.any(given > 1)) # A group with no level of a dimension gets the mean fraction, or no one of a number
                cells = np.ravel_multi_index(index, grid.shape) # The combination of levels each group is in
                split = np.bincount(cells, weights=sim.groups.group_split, minlength=grid.size)[cells] # The share of people at that combination
                if state.people:
                    split = np.bincount(cells, weights=self.state_split[state.name], minlength=grid.size)[cells] # ... of those this state is split between, which for a dimension of some states only is not everyone
                    vals = grid[index] # For each group
                    size = self.total_pop*split
                    count = vals >= 1 # As for a single value: 1 or more is a number of people, and less a share
                    fracs[state.name] = np.where(count, np.divide(vals, size, out=np.zeros(sim.groups.n_groups), where=size>0), vals)
                else:
                    share = np.divide(self.state_split[state.name], split, out=np.zeros(sim.groups.n_groups), where=split>0) # Its split over the other dimensions
                    counts[state.name] = given.sum()
                    self.state_split[state.name] = share*grid[index]/max(given.sum(), 1e-300)
                continue
            if not state.people:
                counts[state.name] = float(init)
            elif init >= 1: # A number of people; below 1, a share of everyone (everyone in a state is the remainder, in the first state)
                fracs[state.name] = init/self.total_pop
            else:
                fracs[state.name] = float(init)

        # Initial prevalence is seeded through the transmission, so people start wherever a new infection starts
        init_prev = graph.init_prev
        if isinstance(init_prev, ss.transitions.par): # A parameter (e.g. sir.pars.init_prev), whose value can be a distribution
            init_prev = init_prev.raw if isinstance(init_prev.raw, ss.Dist) else self.value(init_prev)
        if init_prev is not None and self.seed_dist() is not None: # Drawn for each agent instead: see Agents.init_post()
            init_prev = None
        if init_prev:
            if isinstance(init_prev, ss.bernoulli):
                init_prev = init_prev.pars.p
            elif isinstance(init_prev, ss.choose_n): # A number of people, as Starsim 3 gave it
                init_prev = init_prev.pars.n
            if init_prev > 1: # More than 1 is a number of people, and up to 1 a share (1.0 is everyone)
                init_prev = init_prev/self.total_pop
            trs = graph.transmissions
            if not trs:
                errormsg = f'"{self.mod.name}" has no contact-mediated transition to seed init_prev through; use init= on a state instead'
                raise ValueError(errormsg)
            dst = trs[0].dst
            if graph.infectious and not any(state.infectious for state in dst): # e.g. E in an SEIR
                seeded = sc.strjoin(state.name for state in dst)
                ss.note(f'init_prev seeds {seeded} of "{self.mod.name}", where a new infection starts, so nobody is infectious at the start; to seed infectious people, use init=dict({graph.infectious[0].name}=...)', sim=self.sim)
            ps = self.branch_probs(trs[0])
            for state,p in zip(dst, ps):
                fracs[state.name] += init_prev*np.mean(p)

        for name,val in list(fracs.items()) + list(counts.items()):
            if not np.all(np.isfinite(val)) or np.any(np.asarray(val) < 0):
                raise ss.OutOfBounds(f'the initial value of "{name}" in "{self.mod.name}" is {val}; an initial value must be finite and not negative.')

        # Whatever is left over goes to the first state, or, if that is given a value, to the one state that is not (see Graph.resolve_rest())
        total = sum(fracs.values())
        if np.max(total) > 1 + 1e-9:
            errormsg = f'The initial fractions of "{self.mod.name}" sum to {np.max(total):n}, which is more than 1: {dict(fracs)}'
            raise ValueError(errormsg)
        if graph.people_states:
            rest = 1 - total
            left = np.max(np.abs(rest)) > 1e-9
            if graph.rest is None and left:
                first = graph.first.name
                unset = [s.name for s in graph.people_states if s.init is None]
                where = f'Say where the rest start, e.g. {unset[-1]}={np.max(rest):.4g}' if unset else 'Give initial values that sum to 1'
                errormsg = (f'The initial values of "{self.mod.name}" sum to {np.min(total):n}, not 1, and it is not clear where the rest start: {first} is given a value, which is kept, '
                            f'and {"no other state is left without one" if not unset else f"{sc.strjoin(unset)} are not given one"}.\n'
                            f'     {where}.')
                raise ValueError(errormsg)
            if graph.rest is not None:
                fracs[graph.rest.name] += rest
            if left and graph.rest is not graph.first:
                ss.note(f'the initial value of {graph.first.name} in "{self.mod.name}" is kept, and the rest ({np.max(rest):.3g}) start in {graph.rest.name}, the only state not given one', sim=self.sim)
        return fracs, counts

    def seed_dist(self):
        """ As agents, an init_prev given as `ss.choose_n()` (a number of agents), or as `ss.bernoulli()` with a probability for each agent, chooses who starts infected, as in Starsim 3 """
        init_prev = self.graph.init_prev
        if isinstance(init_prev, ss.transitions.par):
            init_prev = init_prev.raw
        trs = self.graph.transmissions
        if self.sim.solver != 'abm' or not len(trs) or len(trs[0].dst) != 1:
            return None
        elif isinstance(init_prev, ss.choose_n):
            return init_prev
        elif isinstance(init_prev, ss.bernoulli) and callable(init_prev.pars.p):
            return init_prev
        return None

    def branch_probs(self, tr):
        """ The probability of each destination of a transition (1 if there is only one) """
        if len(tr.dst) <= 1:
            return [1.0]*len(tr.dst)
        ps = [None if p is None else self.eval(p) for p in tr.p]
        rest = 1.0 - sum(p for p in ps if p is not None)
        return self.dest_shares(tr, [rest if p is None else p for p in ps])

    def eval(self, expr):
        return ss.transitions.evaluate(expr, self)

    #%% Results

    def init_results(self):
        """ One count per state, one flow per transition (`res.infection.cum` is its cumulative total), and for the library's diseases, Starsim's standard disease results """
        mod = self.mod
        graph = self.graph
        abm = self.sim.solver == 'abm'
        results = sc.autolist()
        for state in graph.states.values():
            key = f'n_{state.name}'
            if key not in mod.results:
                results += ss.Result(key, dtype=float, scale=abm and state.people, label=state.label or state.name, kind='stock')
        for name,tr in graph.transitions.items():
            scale = abm and tr.people
            results += ss.Result(name, dtype=float, scale=scale, label=tr.label or name, kind='flow')
        taken = list(graph.transitions.keys()) + list(graph.exprs.keys()) + list(graph.outcomes.keys())
        self.standard = bool(graph.transmissions) and mod.standard_results
        if self.standard:
            kinds = dict(new_infections='flow', cum_infections='stock', prevalence='ratio')
            for key,label in dict(new_infections='New infections', cum_infections='Cumulative infections', prevalence='Prevalence').items():
                if key not in taken:
                    results += ss.Result(key, dtype=float, scale=abm and key=='new_infections', label=label, auto_plot=key=='prevalence', kind=kinds[key])
        for name,expr in graph.exprs.items(): # A ratio, a sum of flows, or otherwise a stock
            if isinstance(expr, ss.transitions.BinOp) and expr.op == '/':
                kind = 'ratio'
            elif any(isinstance(leaf, ss.transitions.Flow) for leaf in expr.leaves) and all(isinstance(leaf, (ss.transitions.Flow, ss.par)) for leaf in expr.leaves):
                kind = 'flow'
            else:
                kind = 'stock'
            results += ss.Result(name, dtype=float, scale=False, label=name, kind=kind)
        self.init_outcomes(results)
        for name,tf in graph.transfers.items():
            results += ss.Result(name, dtype=float, scale=abm, label=tf.label or name, kind='flow')
            self.flowcounts[name] = 0.0
        mod.define_results(*results)

        # Stratified results are one result with a dimension for each stratification: see res.select() and res.by()
        ng = self.sim.groups.n_groups
        self.stock_groups = sc.objdict()
        self.flow_groups = sc.objdict()
        self.rows = self.group_ti = None # The row of each time point in the results by group, and the time point of each row, if they are kept every few steps: see ss.Sim.group_rows()
        self.row_flows = sc.objdict() # Then, the flows by group so far in the current row
        if ng > 1:
            names = [strat.name for strat in self.sim.groups.strats]
            dims = sc.objdict() # The dimensions of each result
            for state in graph.states.values():
                dims[f'n_{state.name}'] = self.applies[state.name]
            for name,tr in graph.transitions.items():
                dims[name] = np.any([self.applies[state.name] for state in tr.src + tr.dst], axis=0)
            for name in graph.transfers:
                dims[name] = np.ones(len(names), dtype=bool)
            if 'new_infections' in mod.results and 'new_infections' not in graph.transitions:
                dims['new_infections'] = np.ones(len(names), dtype=bool)
            self.ratios = sc.objdict() # The results that are a ratio of two sums of states, as (numerator, denominator)
            for name,expr in graph.exprs.items(): # A sum of states (or flows) that all have the same dimensions has them too; so does a ratio of two, which keeps both, so that selecting groups adds up each before dividing
                parts = [expr.a, expr.b] if isinstance(expr, ss.transitions.BinOp) and expr.op == '/' else [expr]
                plain = all(all(isinstance(leaf, (ss.State, ss.par, ss.transitions.Flow)) for leaf in part.leaves) and not any(op in ss.transitions.format_expr(part) for op in ['/', '**']) for part in parts)
                if not plain: # E.g. a count (I.n), which is the total over every group, so the result is not by group
                    continue
                flags = [self.applies[leaf.name] if isinstance(leaf, ss.State) else dims[leaf.name] for part in parts for leaf in part.leaves if not isinstance(leaf, ss.par)]
                if flags and all(np.array_equal(f, flags[0]) for f in flags):
                    dims[name] = flags[0]
                    if len(parts) == 2:
                        self.ratios[name] = parts
            for name,info in self.outcomes.items(): # An outcome has the dimensions of what it is counted from (in order, since outcomes can chain)
                dims[name] = np.any([dims[key] for key,weight in info.sources if key in dims], axis=0)
                if f'n_{name}' in mod.results:
                    dims[f'n_{name}'] = dims[name]
            grouped = [] # The results kept by group, and their dimensions
            for key,flags in dims.items():
                if np.any(flags) and self.sim.kept(mod, key):
                    grouped.append((mod.results[key], flags))
            n_arrays = sum(1 + (res.name in self.ratios) for res,flags in grouped) # The numerator by group, and the denominator alongside it
            self.rows, self.group_ti = self.sim.group_rows(mod.t.npts, n_arrays*ng*8, self.group_bytes, name=mod.name)
            n_rows = mod.t.npts if self.rows is None else len(self.group_ti)
            for res,flags in grouped:
                res.init_groups(self.sim.groups.strats, [name for name,flag in zip(names, flags) if flag], (n_rows, ng), self.group_ti, ratio=res.name in self.ratios)
        return

    def init_outcomes(self, results):
        """ Work out the sources and delay of each outcome, and the order to compute them in """
        graph = self.graph
        trans = ss.transitions
        abm = self.sim.solver == 'abm'
        dt = self.dt
        self.outcomes = sc.objdict()
        todo = dict(graph.outcomes)
        while todo: # Outcomes can chain, so compute each one after its sources
            progress = False
            for name,oc in list(todo.items()):
                sources = []
                for base,weight in oc.flows:
                    key = base if isinstance(base, str) else base.name
                    if key not in graph.transitions and key not in graph.outcomes:
                        options = list(graph.transitions.keys()) + list(graph.outcomes.keys())
                        raise ss.UnknownOutcomeSource(f'the source of the outcome "{name}" is not a transition or outcome of "{self.mod.name}" ({key}){trans.did_you_mean(str(key), options)}')
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
                raise ss.OutcomeCycle(f'the outcomes {sc.strjoin(todo.keys())} of "{self.mod.name}" depend on each other in a loop.')
        return

    def sample(self, dist, n):
        """ Draw samples from a distribution of durations, in the sim's unit (used to build delay kernels, not during the run) """
        new = ss.transitions.numeric_dist(dist, self.convert)
        new.strict = False
        new.init(force=True)
        return np.asarray(new.rvs(n), dtype=float)

    def step_outcomes(self):
        """ Work out this step's value of each outcome, in each group, from the flows it is counted from """
        res = self.mod.results
        ti = self.mod.ti
        ng = self.sim.groups.n_groups
        integer = self.sim.solver in ['abm', 'gillespie', 'tau', 'binomial']

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
                if isinstance(weight, ss.transitions.Expr):
                    weight = self.eval(weight)
                if integer and np.all((0 <= weight) & (weight <= 1)): # A fraction of whole events is a draw
                    amount += self.rng.binomial(np.round(flow).astype(int), weight)
                else:
                    amount += flow*weight
            slots = (ti + np.arange(len(info.kernel))) % len(info.pending) # This step and the next ones
            if amount.any():
                if integer and len(info.kernel) > 1:
                    for g in np.flatnonzero(amount):
                        info.pending[slots, g] += self.rng.multinomial(int(round(amount[g])), info.kernel/info.kernel.sum())
                else:
                    info.pending[slots] += np.outer(info.kernel, amount)
            info.now = info.pending[slots[0]].copy()
            info.pending[slots[0]] = 0
            if info.stay is not None: # Each arrival leaves after a stay drawn for it
                slots = (ti + np.arange(len(info.stay))) % len(info.leaving)
                if integer:
                    for g in np.flatnonzero(info.now):
                        info.leaving[slots, g] += self.rng.multinomial(int(round(info.now[g])), info.stay)
                else:
                    info.leaving[slots] += np.outer(info.stay, info.now)
                info.n = info.n + info.now - info.leaving[slots[0]]
                info.leaving[slots[0]] = 0
                store(f'n_{name}', info.n, False)
            elif info.p_leave is not None:
                info.n = info.n + info.now - info.n*info.p_leave
                store(f'n_{name}', info.n, False)
            store(name, info.now, True)
        return

    def start_step(self):
        """ Start counting the flows of the step """
        self.cache.clear()
        for name in self.flowcounts:
            self.flowcounts[name] = 0.0
        if hasattr(self, 'flow_groups'):
            self.flow_groups.clear()
        return

    def record_stocks(self):
        """ Record the number in each state at the end of the step (so `n_R[ti]` has the people who recovered in the step that ends at time point `ti`, as `recovery[ti]` does) """
        for state in self.graph.states.values():
            self.stocks[state.name] = self.stock(state)
        return

    def update_results(self):
        mod = self.mod
        graph = self.graph
        res = mod.results
        ti = mod.ti
        if getattr(getattr(self.sim.people, 'scale', None), 'weighted', False): # Agents of different weights (people.scale) make fractions of people
            for key in [f'n_{name}' for name in self.stocks] + list(self.flowcounts):
                if res[key].values.dtype.kind in 'iu':
                    res[key].init_values(dtype=float, force=True)
        for name,val in self.stocks.items():
            res[f'n_{name}'][ti] = val
        for name,val in self.flowcounts.items():
            res[name][ti] = val
        self.step_outcomes()
        for name,val in self.stock_groups.items():
            res[f'n_{name}'].store_group(ti, val, self.rows)
        for name,val in self.flow_groups.items():
            res[name].store_group(ti, val, self.rows, flow=True)
        if self.flow_groups and 'new_infections' in res and 'new_infections' not in graph.transitions:
            res.new_infections.store_group(ti, sum(self.flow_groups.get(tr.name, 0) for tr in graph.transmissions), self.rows, flow=True)
        if self.rows is not None: # The flows by group so far in this row, for the results worked out from them
            if ti == 0 or self.rows[ti] != self.rows[ti-1]:
                self.row_flows = sc.objdict()
            for name,val in self.flow_groups.items():
                self.row_flows[name] = self.row_flows.get(name, 0) + val
        if self.standard:
            if 'new_infections' in res and 'new_infections' not in graph.transitions:
                res.new_infections[ti] = sum(self.flowcounts[tr.name] for tr in graph.transmissions)
            if 'prevalence' in res and 'prevalence' not in graph.exprs:
                n = sum(self.stocks[s.name] for s in graph.people_states)
                res.prevalence[ti] = sum(self.stocks[s.name] for s in graph.infected)/n if n else 0
        if graph.exprs:
            ctx = StockCtx(self)
            for name,expr in graph.exprs.items():
                res[name][ti] = ss.transitions.evaluate(expr, ctx)
                if getattr(res[name], 'strata', None) is not None: # By group: in a row of several steps, from the states at its end and the flows over it
                    ctx_g = StockCtx(self, self.stock_groups, self.flow_groups if self.rows is None else self.row_flows)
                    if name in self.ratios:
                        num, den = [ss.transitions.evaluate(part, ctx_g) for part in self.ratios[name]]
                        row = ti if self.rows is None else self.rows[ti]
                        res[name].strata[row], res[name].denom[row] = num, den
                    else:
                        res[name].store_group(ti, ss.transitions.evaluate(expr, ctx_g), self.rows)
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
        """ For a run with real numbers (ode): say where chance would matter, as a stochastic run would show (ss.SmallCounts, ss.ExtinctionPossible) """
        sim = self.sim
        infected = [s for s in self.graph.infected if s.people]
        if sim.solver != 'ode' or not infected or not self.graph.transmissions:
            return
        res = self.mod.results
        n = sum(np.asarray(res[f'n_{s.name}'].values, dtype=float) for s in infected)
        name = self.mod.name
        if 0 < self.n_start < 10:
            ss.note(f'"{name}" starts with {self.n_start:.3g} infected, so whether it takes off is a matter of chance, which solver="{sim.solver}" does not show; '
                    f'solver="gillespie" with ss.parallel(n_runs=...) does', ss.SmallCounts, sim=sim)
        low = np.flatnonzero((n > 0) & (n < 1))
        if len(low) and n[low[0]:].max() > 10:
            when = res[f'n_{infected[0].name}'].timevec[low[0]]
            when = f'{sim.unit_name} {float(when):.4g}' if sc.isnumber(when) else when # e.g. day 125, not 124.99999984
            ss.note(f'the number infected in "{name}" falls to {n[low[0]]:.2g} (at {when}) and then rises to {n[low[0]:].max():,.0f}; a stochastic run would usually die out there. '
                    f'solver="gillespie" with ss.parallel(n_runs=...) shows how often, in the runs of msim.results.{name}.n_{infected[0].name} after that point', ss.ExtinctionPossible, sim=sim)
        return

    def finalize_results(self):
        """ Cumulative results, computed after scaling """
        res = self.mod.results # Already scaled, by group too: see ss.Result.apply_scale()
        self.note_chance()
        if self.standard and 'cum_infections' in res and 'new_infections' in res:
            res.cum_infections[:] = np.cumsum(res.new_infections[:])
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

    def exposure(self, state): # In a result, a bare state stands for the number of people in it
        return self.stock(state)
