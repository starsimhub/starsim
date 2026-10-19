"""
Dimensions: how a model is divided into groups by the properties of people (age, sex, place, ...)
"""
import re
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import as_expr, did_you_mean
from .parameters import by
from .networks import geo, gravity, Mobility
from .transitions import parse_arrow

__all__ = ['Strat', 'Transfer', 'transfer', 'Groups']


def level_dtype(n):
    """ The integers that levels or groups are kept as, n of them: 16-bit if they fit, to save memory """
    return np.int16 if n < 2**15 else np.int32


#%% Stratifications

hints = dict(patch='space', node='space', region='space', district='space', county='space', place='space', location='space', province='space',
             strain='strain', variant='strain', genotype='strain', serotype='strain', resistance='strain') # Names that suggest a kind, for a note only: the kind comes from the value (see ss.transitions.Strat)


class Strat:
    """
    One stratification: a property of people, and the levels it is split into

    Args:
        name (str): the dimension, e.g. 'age'
        levels (list): bin edges (for a continuous property such as age) or labels
        split (list): the share of the population in each level (if not given: from an `ss.geo`'s populations; for age groups, the world's ages; for a boolean dimension, everyone at the first level; otherwise equal)
        adjust (dict): multipliers on named transitions, one per level
        kind (str): 'age', 'space', 'strain', or 'plain'; if not given, 'space' for an `ss.geo` or with `mobility=`, 'strain' with `immunity=`, 'age' for a dimension called age, and otherwise 'plain'
        states (list): the names of the states it applies to (default: every state that is people)
        where (dict): the groups it applies to, as a dimension and a level (or list of levels), e.g. `dict(sex='female')`; people in other groups have no level of it (default: every group)
        mobility (gravity/Mobility/array): for a spatial stratification, the movement between its levels
        immunity (array): for a strain, the protection that past infection with each strain (rows) gives against each strain (columns); 1 is complete
    """
    def __init__(self, name, levels, split=None, adjust=None, kind=None, states=None, where=None, mobility=None, immunity=None):
        self.name = name
        self.geo = levels if isinstance(levels, geo) else None
        self.kind = kind or ('space' if (self.geo is not None or mobility is not None) else 'strain' if immunity is not None else 'age' if name == 'age' else 'plain')
        self.hint = hints.get(name) if kind is None and self.kind == 'plain' else None # A name that suggests another kind, which is noted
        self.guessed = sc.objdict()
        if self.hint == 'space':
            self.guessed['kind'] = 'plain, so its levels mix; for places, give them as ss.geo(names=[...]) or pass kind="space"'
        elif self.hint == 'strain':
            self.guessed['kind'] = 'plain, so it does not change who infects whom; for strains, give immunity= or pass kind="strain"'
        self.states = states
        if where is not None and not isinstance(where, dict):
            errormsg = f'where= takes a dict of a dimension and its level (or levels), e.g. dict(sex="female"), not {type(where).__name__}'
            raise TypeError(errormsg)
        if where and self.kind in ['space', 'strain']:
            errormsg = f'where= is for a property of people such as age or risk, but "{name}" is of kind "{self.kind}"; a {"place" if self.kind == "space" else "strain"} is one that everyone (or every infection) has'
            raise ValueError(errormsg)
        self.where = dict(where) if where else None
        if self.geo is not None:
            levels = self.geo.names
            if split is None and self.geo.pop is not None:
                split = self.geo.pop/self.geo.pop.sum()
        if mobility is not None and not isinstance(mobility, Mobility): # A movement model or a matrix on its own is commuting
            mobility = Mobility(mobility) if isinstance(mobility, gravity) else Mobility(matrix=mobility)
            self.guessed['mobility'] = 'commuting, since kind= was not given'
        if mobility is not None and mobility.guessed:
            self.guessed['mobility'] = sc.strjoin([self.guessed['mobility'], mobility.guessed]) if 'mobility' in self.guessed else mobility.guessed
        if mobility is not None and self.kind != 'space':
            errormsg = f'mobility= is for a spatial stratification, but "{name}" is of kind "{self.kind}"; pass kind="space" if it is a place'
            raise ValueError(errormsg)
        self.mobility = mobility
        levels = list(levels)
        numeric = all(sc.isnumber(v) and not isinstance(v, bool) for v in levels)
        self.edges = None
        if numeric: # Bin edges; the last bin is open-ended
            self.edges = np.array(levels, dtype=float)
            ends = list(levels[1:]) + [None]
            self.labels = [f'{lo:n}+' if hi is None else f'{lo:n}' if hi - lo <= 1 else f'{lo:n}-{hi-1:n}' for lo,hi in zip(levels, ends)] # e.g. 0, 1-4, 5+
        else:
            self.labels = levels
        self.n = len(self.labels)
        self.female = None # For a dimension called sex, which is people.female: the position of the level for women
        if name == 'sex' and kind is None: # kind= says it is something else, e.g. a place
            women = [i for i,label in enumerate(self.labels) if str(label).lower() in ['female', 'females', 'f', 'women', 'woman']]
            if self.n != 2 or len(women) != 1:
                errormsg = f'A dimension called "sex" is people.female, so it has two levels, one of them "female" (e.g. sex=["female", "male"]), not {sc.strjoin(self.labels)}; give another name for any other property'
                raise ValueError(errormsg)
            self.female = women[0]
        self.carries = None # For a strain with levels for people who carry several strains at once ('A+B'): see init_carriage()
        if self.kind == 'strain' and any('+' in str(label) for label in self.labels):
            self.init_carriage()
        if self.kind == 'age' and self.edges is None and kind is None: # Age groups given as labels have no ages, so nobody ages between them
            self.kind = 'plain'
            self.guessed['kind'] = 'plain, since the age groups are labels and not ages'
        self.adjust = sc.objdict()
        for key,vals in (adjust or {}).items():
            vals = np.array(vals, dtype=float)
            if len(vals) != self.n:
                raise ss.ShapeError(f'the adjustment of "{key}" has {len(vals)} values, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}).')
            if np.any(vals < 0):
                raise ss.InvalidAdjustment(f'the adjustment of "{key}" by "{name}" has a negative multiplier ({vals}).')
            self.adjust[key] = vals
        self.immunity = None
        if immunity is not None:
            immunity = np.array(immunity, dtype=float)
            if self.kind != 'strain':
                errormsg = f'immunity= is for a strain dimension, but "{name}" is of kind "{self.kind}"; pass kind="strain" if it is one'
                raise ValueError(errormsg)
            if immunity.shape != (self.n, self.n):
                raise ss.ShapeError(f'immunity= has shape {immunity.shape}, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}), so it needs to be {self.n}×{self.n}.')
            self.immunity = immunity
        self.joint = None # A split over several dimensions at once, as an ss.by; resolved by the sim
        self.cond = None # For a dimension of some states only, a split given the levels of other dimensions: their positions, and the share in each level of this one (summing to 1 over it)
        if isinstance(split, by):
            self.joint = split
            split = None
        if split is not None:
            split = np.array(split, dtype=float)
            if len(split) != self.n:
                raise ss.BadSplit(f'split= has {len(split)} entries, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}).')
            if not np.isclose(split.sum(), 1):
                self.guessed['split'] = f'normalized, since it summed to {split.sum():n}'
                split = split/split.sum()
        elif self.kind == 'age' and self.edges is not None: # As the people of a declared model, unless the sim has a named population
            world = ss.named.world
            split = ss.named.age_split(self.edges, world.ages, world.counts)
            self.guessed['split'] = world.guess
        elif all(isinstance(v, (bool, np.bool_)) for v in self.labels): # e.g. vaccinated=[False, True]
            split = self.first()
            self.guessed['split'] = f'everyone at {self.labels[0]}, the first level'
        else:
            split = np.ones(self.n)/self.n
            self.guessed['split'] = 'equal across levels'
        self.split = split
        return

    def first(self):
        """ A split with everyone at the first level """
        out = np.zeros(self.n)
        out[0] = 1
        return out

    def init_carriage(self):
        """
        For a strain with a level such as 'A+B', for people who carry strains A and B at once: which strains each level carries, and where carrying one more leads

        Each strain carried needs a level of its own ('A' and 'B'). Someone who carries several
        strains passes on one of them, in proportion to their infectiousness (`adjust=dict(rel_trans=...)`),
        and someone infected with a strain they do not carry moves to the level that adds it, if there is one
        (otherwise they keep the strains they have).
        """
        sets = [frozenset(part.strip() for part in str(label).split('+')) for label in self.labels]
        names = {next(iter(s)):i for i,s in enumerate(sets) if len(s) == 1}
        for label,s in zip(self.labels, sets):
            missing = s - set(names)
            if missing:
                raise ss.ShapeError(f'the level "{label}" of "{self.name}" carries {sc.strjoin(sorted(missing))}, which has no level of its own; each strain carried needs one.')
        if len(set(sets)) != len(sets):
            raise ss.ShapeError(f'"{self.name}" has two levels that carry the same strains ({sc.strjoin(self.labels)}).')
        lookup = {s:i for i,s in enumerate(sets)}
        self.base = np.array(sorted(names.values())) # The levels of a single strain
        self.carries = np.zeros((self.n, self.n), dtype=bool) # carries[l, j]: people at level l carry strain j
        self.adds = np.zeros((self.n, self.n), dtype=int) # adds[l, j]: the level of people at level l once infected with strain j as well (l itself if they carry it already, or there is no level for carrying it as well)
        for l,s in enumerate(sets):
            for name in s:
                self.carries[l, names[name]] = True
            for name,j in names.items():
                self.adds[l, j] = lookup.get(s | {name}, l)
        return

    def __repr__(self):
        return f'Strat({self.name}: {sc.strjoin(self.labels)})'

    def draw(self, sim, uids, rvs):
        """ A level for each of some agents, from uniform random numbers: in proportion to the split, or to its row for each agent's levels of the other dimensions """
        if getattr(self, 'cond', None) is None:
            return np.minimum(np.searchsorted(np.cumsum(self.split), rvs, side='right'), self.n - 1)
        coords = sim.groups.group_coords[sim.groups.group_raw()[uids]]
        cdf = np.cumsum(np.stack([self.shares(coords, np.full(len(uids), level)) for level in range(self.n)], axis=1), axis=1)
        return np.minimum((cdf <= rvs[:, None]).sum(axis=1), self.n - 1)

    def shares(self, coords, levels=None):
        """
        The share of each group in its level of this dimension (or in `levels`, if given): from the split, or, for a split given the levels of other dimensions, from the row of that split for the group

        Args:
            coords (array): the level of each group in every dimension of the sim, by row
            levels (array): the levels of this dimension to give the share of (by default, each group's own)
        """
        d = self.dim
        levels = coords[:, d] if levels is None else levels
        if getattr(self, 'cond', None) is None:
            return self.split[levels]
        ds, grid = self.cond
        return grid[tuple(levels if dd == d else coords[:, dd] for dd in ds)]

    @property
    def attr(self):
        """ The name of the property of people that holds each agent's level; a dimension with labels that is named like something people already have (age) is kept apart from it """
        return f'{self.name}_group' if (self.name == 'age' and self.edges is None) else self.name

    def people_arr(self):
        """ The property of people that holds each agent's level, when they do not have it already: for a dimension with labels, the level itself, as 16-bit integers (so with one dimension it is also the group: see `Groups.group_raw()`); for a binned one, floats """
        if self.edges is None:
            return ss.Arr(self.attr, dtype=level_dtype(self.n), default=0, nan=-1, label=self.name)
        return ss.FloatArr(self.attr, default=0, label=self.name)

    @property
    def spec(self):
        """ The stratification as it is written in the model spec """
        levels = self.edges.tolist() if self.edges is not None else list(self.labels)
        named = dict(geo=self.geo.ref) if self.geo is not None and self.geo.ref is not None else {} # Where a named geography came from; only when there is one, so that the hash of every other model is as it was
        if getattr(self, 'split_from', None): # A split taken from a named population (ss.people)
            named['split_from'] = self.split_from
        return dict(**named, name=self.name, kind=self.kind, levels=levels, split=self.split.tolist() if self.joint is None else repr(self.joint), adjust={k:v.tolist() for k,v in self.adjust.items()}, states=self.states, where=self.where,
                    mobility=None if self.mobility is None else dict(kind=self.mobility.kind, away=self.mobility.away if self.mobility.away is None else float(self.mobility.away), **({} if self.mobility.scale == 'each' else dict(scale='mean')), **({} if self.mobility.cap is None else dict(cap=float(self.mobility.cap))), **(dict(matrix=self.mobility.matrix(self).tolist()) if self.mobility.data is None else dict(register=dict(events=len(self.mobility.data), hash=sc.sha(self.mobility.data.astype(str).to_csv()).hexdigest())))))

    @property
    def mids(self):
        """ The middle of each bin, for a binned property """
        if self.edges is None:
            return np.arange(self.n, dtype=float)
        ends = np.append(self.edges[1:], max(100, self.edges[-1] + 10) if self.kind == 'age' else self.edges[-1] + 1)
        return (self.edges + ends)/2

    def index(self, level):
        """ The position of a level, given by position or by label """
        if isinstance(level, str) or isinstance(level, bool) or level in self.labels:
            if level not in self.labels:
                errormsg = f'"{level}" is not a level of "{self.name}"; the levels are {sc.strjoin(self.labels)}'
                raise KeyError(errormsg)
            return self.labels.index(level)
        return int(level)

    def same_levels(self, other):
        return list(self.labels) == list(other.labels)

    def same_as(self, other):
        """ Whether another stratification is this one given again: the same levels, kind, split, adjustments, and states """
        same = self.same_levels(other) and self.kind == other.kind and self.states == other.states and self.where == other.where
        same = same and np.allclose(self.split, other.split) and list(self.adjust.keys()) == list(other.adjust.keys())
        return same and all(np.allclose(self.adjust[key], other.adjust[key]) for key in self.adjust)


class Transfer:
    """
    Movement between the levels of a stratification, without a change of disease state

    Args:
        rate (any): the per-person hazard
        dur (any): the mean time before moving (instead of `rate`)
        states (list): the disease states it applies to (default: every state)
        name (str): the name; if not given, taken from the keyword it is passed under
        kwargs (dict): the dimension, and an arrow between two of its levels (e.g. `risk='low -> high'`),
            or the level people move to (e.g. `vaccinated=True`)

    Examples:
        ```python
        ss.transfer(risk='low -> high', rate=ss.peryear(0.1))
        ss.transfer(vaccinated=True, rate=ss.peryear(0.2))
        ```
    """
    def __init__(self, rate=None, dur=None, states=None, name=None, label=None, **kwargs):
        if len(kwargs) != 1:
            errormsg = f'ss.transfer() takes one dimension, e.g. ss.transfer(risk="low -> high", rate=0.1), not {list(kwargs.keys())}'
            raise ValueError(errormsg)
        if (rate is None) == (dur is None):
            errormsg = 'ss.transfer() takes exactly one of rate= or dur='
            raise ValueError(errormsg)
        self.dim, target = list(kwargs.items())[0]
        if isinstance(target, str) and re.search(r'->|>>', target):
            src, dst = parse_arrow(target)
            self.src, self.dst = src, dst[0]
        else:
            self.src, self.dst = None, target # From every other level
        self.rate = None if rate is None else as_expr(rate)
        self.dur = None if dur is None else as_expr(dur)
        self.states = None if states is None else [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
        self.name = name
        self.label = label
        return

    def __repr__(self):
        return f'transfer({self.name}: {self.dim} {self.src or "*"} -> {self.dst})'


def transfer(rate=None, dur=None, states=None, name=None, label=None, **kwargs):
    """
    Movement between the levels of a stratification, without a change of disease state

    People move all the time, at a rate or after a duration, and the transfer is part of the
    model: pass it to `ss.Disease()` or `stratify()`. A move delivered to some people, at some
    times, is a delivery: `ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021])`.

    Args:
        rate (any): the per-person hazard
        dur (any): the mean time before moving (instead of `rate`)
        states (list): the disease states it applies to (default: every state)
        name (str): the name; if not given, taken from the keyword or variable it is assigned to
        kwargs (dict): the dimension, and an arrow between two of its levels (e.g. `risk='low -> high'`),
            or the level people move to (e.g. `vaccinated=True`)

    Examples:
        ```python
        ss.transfer(risk='low -> high', rate=ss.peryear(0.1))
        ss.transfer(vaccinated=True, rate=ss.peryear(0.2))
        ```
    """
    opts = [key for key in kwargs if key in ss.interventions.delivery_keys + list(ss.interventions.renamed_keys) + ['disease']]
    if opts:
        dim = next((key for key in kwargs if key not in opts), 'vaccinated')
        errormsg = (f'ss.transfer() is movement that happens all the time, at a rate (rate= or dur=); {sc.strjoin(f"{key}=" for key in opts)} {"is" if len(opts) == 1 else "are"} for a delivery. '
                    f'A move delivered to some people is ss.Delivery(move=dict({dim}=...), ...), e.g. ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021])')
        raise TypeError(errormsg)
    return Transfer(rate=rate, dur=dur, states=states, name=name, label=label, **kwargs)


strat_options = ['split', 'adjust', 'kind', 'states', 'where', 'mobility', 'immunity']


def parse_stratify(kwargs, graphs, sim=False):
    """
    The one parser of the arguments of `stratify()`, for a module, a sim, or `ss.Sim(stratify=...)`: returns the new
    stratifications and (for a module) the transfers

    Every keyword that is not an option is a dimension, with its levels. A sim's stratification applies to the graphs of all
    its declared modules, which `states=` and `adjust=` are checked against: `adjust=` names a transition by its name, if only
    one module has it, or as e.g. 'flu.infection'.

    Args:
        kwargs (dict): the arguments
        graphs (list): the graphs of the modules it applies to (for a sim, all of them; it can be empty before they are known)
        sim (bool): whether it is the sim's stratification
    """
    kwargs = dict(kwargs)
    opts = {key:kwargs.pop(key, None) for key in strat_options}
    if opts['immunity'] is not None and np.ndim(opts['immunity']) == 1 and not any(isinstance(v, (list, tuple, np.ndarray, geo)) for v in kwargs.values()): # Levels, not the protection between strains: a dimension called immunity
        kwargs['immunity'] = opts.pop('immunity')
        opts['immunity'] = None
    misplaced = [key for key in kwargs if key in ['inplace', 'frame']]
    if misplaced:
        errormsg = f'{sc.strjoin(misplaced)} is not an option of {"sim.stratify() or ss.Sim(stratify=...)" if sim else "stratify() here"}, and cannot be the name of a dimension'
        raise ValueError(errormsg)
    dims = {k:v for k,v in kwargs.items() if not isinstance(v, Transfer)}
    transfers = {k:v for k,v in kwargs.items() if isinstance(v, Transfer)}
    given = [key for key,val in opts.items() if val is not None]
    if not dims and not transfers:
        errormsg = f'stratify() needs a dimension and its levels, e.g. stratify(age=[0, 15, 65]); it was given only {sc.strjoin(given) or "nothing"}' + (' (immunity= is the protection between the levels of a strain dimension given in the same call)' if 'immunity' in given else '')
        raise ValueError(errormsg)
    if transfers and sim:
        errormsg = f'A transfer ({sc.strjoin(transfers)}) belongs to a disease: pass it to that disease\'s stratify()'
        raise ValueError(errormsg)
    if len(dims) > 1 and given:
        errormsg = f'When using {sc.strjoin(f"{key}=" for key in given)}, give one dimension per call to stratify(), so it is clear which one they belong to'
        raise ValueError(errormsg)
    names = lambda key: {name for g in graphs for name in getattr(g, key)}
    states = opts['states']
    if states is not None:
        states = opts['states'] = [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
        for name in states:
            if graphs and name not in names('states'):
                where = f'"{graphs[0].module.name}" does not have. Its states are' if not sim else 'no module in the sim has. The states are'
                raise ss.UnknownState(f'states= names a state "{name}" that {where} {sc.strjoin(names("states"))}{did_you_mean(name, names("states"))}')
    adjust = opts['adjust'] or {}
    for key in adjust:
        if key in ['rel_trans', 'rel_sus'] or not graphs:
            continue
        modname, _, bare = key.rpartition('.')
        having = [g for g in graphs if bare in g.transitions and modname in ['', g.module.name]]
        if len(having) != 1:
            options = [f'{g.module.name}.{t}' if sim else t for g in graphs for t in g.transitions]
            what = 'names a transition' if not having else 'names a transition of more than one module,'
            whose = f'"{graphs[0].module.name}" does not have. Its transitions are' if not sim else 'no single module has. The transitions are'
            raise ss.UnknownTransition(f'adjust= {what} "{key}" that {whose} {sc.strjoin(options)} (and it can also adjust rel_trans and rel_sus).')
    strats = []
    for name,levels in dims.items():
        strat = Strat(name, levels, **opts)
        if strat.kind == 'strain' and states is None and graphs: # A strain belongs to an infection, so the states people can be in without ever being infected have none
            uninfected = sorted({s for g in graphs for s in g.uninfected})
            strat.states = [s.name for g in graphs for s in g.people_states if s.name not in g.uninfected]
            strat.guessed['states'] = f'every state reached only through infection (not {sc.strjoin(uninfected)})'
        strats.append(strat)
    return strats, transfers


#%% The groups of a sim

class Groups:
    """
    The groups of a sim: its stratifications (`ss.transitions.Strat`), and how their levels combine into groups

    In a compartmental run, each group is a row of every compartment; as agents, each agent is in one group,
    which is worked out from their properties when needed (see `group_raw()`). Made when the sim is initialized,
    as `sim.groups`.

    Args:
        sim (`ss.Sim`): the sim
    """
    def __init__(self, sim):
        self.sim = sim
        self.strats = [] # The stratifications, in the order of the dimensions of the groups: see init()
        self.sim_strats = [] # The ones given to the sim, which apply to every module
        self.cross_strats = [] # The dimensions for diseases whose states are tracked jointly: see ss.solvers.init_crossings()
        self.dim_moves = [] # Movement between levels that belongs to no one module, e.g. the protection of a product waning
        return

    def init(self):
        """ Collect the stratifications of the sim and its modules, and work out the groups they give """
        sim = self.sim
        strats = sc.objdict()
        self.sim_strats = [] # The ones given to the sim, which apply to every module
        for call in sc.tolist(sim.pars.stratify):
            for strat in parse_stratify(call, graphs=[mod.graph for mod in sim.declared], sim=True)[0]:
                if strat.name in strats:
                    raise ss.DuplicateStructuralDimension(f'the sim is already stratified by "{strat.name}".')
                strats[strat.name] = strat
                self.sim_strats.append(strat)
        for mod in sim.declared:
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
                        strats[tf.dim] = Strat(tf.dim, [False, True], split=[1.0, 0.0])
                        mod.graph.strats.append(strats[tf.dim])
                    else:
                        raise ss.MissingDimension(f'the transfer "{tf.name}" moves people along "{tf.dim}", but the model is not stratified by it. Add e.g. {mod.name}.stratify({tf.dim}=[...]).')
        self.dim_moves = [] # Movement between levels that belongs to no one module, e.g. the protection of a product waning
        for mod in sim.pars.interventions.values(): # Properties of people that the interventions need, e.g. holding a product
            if hasattr(mod, 'dims'):
                for strat in mod.dims(sim):
                    strat.owner = mod.name # Part of the scenario, not of the model
                    strats.setdefault(strat.name, strat)
                self.dim_moves += [move for move in mod.dim_moves() if move.dim not in [m.dim for m in self.dim_moves]]
                for dim in (mod.move or {}):
                    if dim not in strats:
                        raise ss.MissingDimension(f'the transfer "{mod.name}" moves people along "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        moved = [tf.dim for mod in sim.declared for tf in mod.graph.transfers.values()] # A dimension that people are moved along, with no split of its own, starts with everyone at its first level
        moved += [dim for mod in sim.declared for tr in mod.graph.transitions.values() for dim in (getattr(tr, 'move', None) or {})]
        moved += [dim for mod in sim.pars.interventions.values() for dim in (getattr(mod, 'move', None) or {})]
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
                ss.note(f'the dimension "{strat.name}" is {strat.guessed["kind"]}', sim=sim)
        named = getattr(sim, 'named_people', None)
        given = getattr(sim, 'given_pyramid', None)
        for strat in self.strats: # Age groups with no split of their own take that of a named population, or of the people given to a compartmental run
            if (named is not None or given is not None) and strat.kind == 'age' and strat.edges is not None and strat.guessed.get('split') == ss.named.world.guess:
                ages, counts = (named.ages, named.counts) if named is not None else given
                strat.split = ss.named.age_split(strat.edges, ages, counts)
                strat.split_from = 'people'
                strat.guessed.pop('split')
        pops = [strat for strat in self.strats if strat.geo is not None and strat.geo.pop is not None]
        if len(pops) == 1 and 'total_pop' not in sim.given and sim.pop_from is None: # Places with populations, and no other population size: their total
            sim.pars.total_pop = float(pops[0].geo.pop.sum())
            sim.pars.pop_scale = sim.pars.total_pop/sim.pars.n_agents
            sim.pop_from = f'the places of "{pops[0].name}"'
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
            ds, grid = joints[0].grid(sim)
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
                ds, grid = strat.joint.grid(sim)
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
                setattr(sim, f'rng_strat_{strat.name}', ss.random(name=f'rng_strat_{strat.name}'))
        if 'use_aging' not in sim.given: # People age if and only if they are born or die, unless told otherwise
            sim.pars.use_aging = sim.has_demography()
            if not sim.pars.use_aging and any(strat.kind == 'age' and strat.edges is not None for strat in self.strats):
                ss.note('people do not age between the age groups, since nobody is born or dies; pass use_aging=True to age them', sim=sim)
        return

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
        sim = self.sim
        self._group_cache = None
        for mod in getattr(sim, 'declared', []): # The agent engines keep counts by group: see ss.solvers.Agents.regroup()
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
        sim = self.sim
        cache = getattr(self, '_group_cache', None)
        n = sim.people.uid.len_tot
        binned = any(strat.edges is not None for strat in self.strats)
        if cache is not None and (cache[0] == sim.ti or not binned) and len(cache[1]) == n:
            return cache[1]
        dirty = getattr(self, '_group_dirty', None)
        self._group_dirty = None
        if len(self.strats) == 1 and not binned and self.group_fold is None and sim.people[self.strats[0].attr].raw.dtype.kind in 'iu': # One dimension, whose property is the level (see Strat.people_arr()): it is the group too, so nothing is kept
            raw = sim.people[self.strats[0].attr].raw
            self._group_cache = (sim.ti, raw, [raw])
            return raw
        shared = len(self.strats) == 1 # With one dimension, the groups are its levels: one array is kept for both
        kept = getattr(self, '_level_cache', None)
        if isinstance(dirty, list) and not binned and kept is not None and '' in kept and len(kept['']) == n: # Only some agents have changed, and we know which: work out their groups alone
            uids = np.unique(np.concatenate(dirty))
            levels = []
            out = kept[''] # Changed in place: the groups are only ever used straight away, so nothing holds the old ones
            for strat,mult in zip(self.strats, np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:]):
                raw = sim.people[strat.attr].raw
                old_raw, old_level = kept[strat.name]
                if len(old_raw) != n: # The arrays have grown since: work everything out
                    self._group_dirty = False
                    return self.group_raw()
                if len(uids):
                    before = old_level[uids]
                    after = np.clip(np.digitize(raw[uids], strat.edges) - 1, 0, strat.n - 1) if strat.edges is not None else ss.kernels.levels(raw[uids])
                    old_level[uids] = after
                    old_raw[uids] = raw[uids]
                    if not shared:
                        out[uids] += (after - before)*int(mult)
                levels.append(kept[strat.name][1])
            kept[''] = out
            self._group_cache = (sim.ti, self.folded(out), levels)
            return self._group_cache[1]
        people = sim.people
        kept = getattr(self, '_level_cache', None) # The values each level was worked out from: most properties change rarely, and checking is much cheaper than working the levels out again
        if kept is None:
            kept = self._level_cache = dict()
        levels = []
        changed = False
        def level_of(strat, raw): # As 16-bit integers if they fit, to save memory
            return (np.clip(np.digitize(raw, strat.edges) - 1, 0, strat.n - 1) if strat.edges is not None else ss.kernels.levels(raw)).astype(level_dtype(strat.n), copy=False)
        dtype = level_dtype(self.n_groups)
        mults = np.cumprod((self.group_shape + (1,))[::-1])[::-1][1:]
        n = people.uid.len_tot
        rebuild = '' not in kept or len(kept['']) > n
        n0 = n if rebuild else len(kept['']) # Fewer than n if the people have grown since: the new agents' levels are added to what is kept, rather than all worked out again
        updates = [] # For each dimension whose values changed for some agents: who, their old level, and their new one
        for strat,mult in zip(self.strats, mults):
            raw = people[strat.attr].raw
            old = kept.get(strat.name)
            if old is not None and n0 < n and len(old[0]) == n0:
                old = kept[strat.name] = (np.concatenate([old[0], raw[n0:]]), np.concatenate([old[1], level_of(strat, raw[n0:])]))
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
        if shared:
            kept[''] = levels[0] # Already up to date
        elif rebuild:
            out = np.zeros(n, dtype=dtype)
            for level,mult in zip(levels, mults):
                out += level*int(mult)
            kept[''] = out
        else:
            out = kept['']
            if n0 < n:
                tail = np.zeros(n - n0, dtype=dtype)
                for level,mult in zip(levels, mults):
                    tail += level[n0:]*int(mult)
                out = np.concatenate([out, tail])
            for idx,before,after,mult in updates:
                out[idx] += (after - before)*mult
            kept[''] = out
        self._group_cache = (sim.ti, self.folded(kept['']), levels)
        return self._group_cache[1]

    def folded(self, groups):
        """ The groups that agents are counted in: see `init_ragged()` """
        return groups if self.group_fold is None else self.group_fold[groups]

    def level_raw(self, d):
        """ In an agent-based run: the level of every agent in one dimension (by its position), as an array indexed by UID """
        self.group_raw()
        return self._group_cache[2][d]

    def init_vals(self):
        """ In an agent-based run: assign each agent a level of every property they do not already have """
        sim = self.sim
        people = sim.people
        uids = people.auids
        joint = self.joint[0] if self.joint is not None else []
        for d,strat in enumerate(self.strats):
            if strat.female is not None and d not in joint and strat.cond is None: # A sex dimension is people.female, whose share is the split: see init_people()
                self.set_sex(uids)
            elif strat.kind != 'age' and d not in joint and strat.cond is None: # Exactly the split, rounded to whole agents, in a random order
                order = np.argsort(getattr(sim, f'rng_strat_{strat.name}').rvs(uids), kind='stable')
                counts = ss.solvers.Compartmental.round_preserving(strat.split*len(uids)).astype(int)
                levels = np.repeat(np.arange(strat.n, dtype=level_dtype(strat.n)), counts) # As they are kept, to save memory
                people[strat.attr][uids[order]] = levels
        if joint: # The levels of a joint split, together, given the age group if age is one of them
            ds, grid = self.joint
            ages = [i for i,d in enumerate(ds) if self.strats[d].kind == 'age']
            others = [i for i in range(len(ds)) if i not in ages]
            grid = np.moveaxis(grid, ages + others, list(range(len(ds)))).reshape(len(ages) and self.strats[ds[ages[0]]].n or 1, -1) # Rows: the age group (one row if no age); columns: the other levels together
            row = np.clip(np.digitize(people.age[uids], self.strats[ds[ages[0]]].edges) - 1, 0, grid.shape[0] - 1) if ages else np.zeros(len(uids), dtype=int)
            cdf = np.cumsum(grid, axis=1)
            cdf = cdf/np.maximum(cdf[:, -1:], 1e-300)
            rvs = getattr(sim, f'rng_strat_{self.strats[ds[others[0]]].name}').rvs(uids)
            cell = np.minimum((cdf[row] <= rvs[:, None]).sum(axis=1), grid.shape[1] - 1)
            for i,level in zip(others, np.unravel_index(cell, [self.strats[ds[i]].n for i in others])):
                people[self.strats[ds[i]].attr][uids] = level
        for strat in self.strats: # A split given the levels of other dimensions, once those are known
            if strat.cond is not None:
                self.groups_changed()
                people[strat.attr][uids] = strat.draw(sim, uids, getattr(sim, f'rng_strat_{strat.name}').rvs(uids))
        if joint or any(strat.cond is not None for strat in self.strats): # A sex drawn with other dimensions sets people.female
            self.set_sex(uids, from_level=True)
        self.groups_changed()
        return

    def place_new(self, uids):
        """ In an agent-based run: the levels of agents added by a module written as code (e.g. `ss.Births`), drawn from each dimension's split, as the declared births draw them (see `ss.solvers.Agents.add_people()`) """
        sim = self.sim
        people = sim.people
        for strat in self.strats:
            if strat.kind in ['age', 'strain', 'disease'] or strat.female is not None: # Ages are their own, a sex dimension is people.female (see set_sex()), and newborns have no strain
                continue
            rvs = getattr(sim, f'rng_strat_{strat.name}').rvs(uids)
            if strat.cond is not None:
                people[strat.attr][uids] = strat.draw(sim, uids, rvs)
            else:
                people[strat.attr][uids] = np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1)
        self.groups_changed(uids)
        return

    def set_sex(self, uids, from_level=False):
        """ In an agent-based run with a dimension called sex: its level for some agents from people.female (or, if from_level, the other way round) """
        sim = self.sim
        strat = next((strat for strat in self.strats if strat.female is not None), None)
        if strat is None:
            return
        people = sim.people
        if from_level:
            people.female[uids] = people.sex[uids] == strat.female
        else:
            people.sex[uids] = np.where(people.female[uids], strat.female, 1 - strat.female)
        return
