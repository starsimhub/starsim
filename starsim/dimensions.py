"""
Dimensions: how a model is divided into groups by the properties of people (age, sex, place, ...)
"""
import numpy as np
import sciris as sc
import starsim as ss
from .internals.expressions import as_expr, did_you_mean
from .parameters import by
from .geo import geography, gravity, radiation, Mobility
from .transitions import level_pair, parse_increment

__all__ = ['Strat', 'Transfer', 'transfer', 'Groups']


def level_dtype(n):
    """ The integers that levels or groups are kept as, n of them: 16-bit if they fit, to save memory """
    return np.int16 if n < 2**15 else np.int32


#%% Stratifications

class Strat:
    """
    One stratification: a property of people, and the levels it is split into

    Args:
        name (str): the dimension, e.g. 'age'
        levels (list/dict): bin edges (for a continuous property such as age) or labels; or a dict of the share of the population in each, e.g. `{0: 0.3, 15: 0.6, 65: 0.1}`
        split (list): the share of the population in each level (if not given: from an `ss.geo.geography`'s populations; for age groups, the world's ages; for a boolean dimension, everyone at the first level; otherwise equal)
        adjust (dict): multipliers on named transitions, one per level: on its rate (a duration is divided), or on its probability for a branch chosen with p=
        kind (str): 'age', 'space', or 'plain'; if not given, 'space' for an `ss.geo.geography` or with `mobility=`, 'age' for a dimension called age, and otherwise 'plain'
            (a strain dimension, kind 'strain', is made by `ss.Strains`: see `strains`)
        states (list): the names of the states it applies to (default: every state that is people)
        where (dict): the groups it applies to, as a dimension and a level (or list of levels), e.g. `dict(sex='female')`; people in other groups have no level of it (default: every group)
        mobility (gravity/radiation/Mobility/array): for a spatial stratification, the movement between its levels
    """
    strains = None # For a strain dimension: the ss.Strains it is made from

    def __init__(self, name, levels, split=None, adjust=None, kind=None, states=None, where=None, mobility=None):
        self.name = name
        self.geo = levels if isinstance(levels, geography) else None
        self.kind = kind or ('space' if (self.geo is not None or mobility is not None) else 'age' if name == 'age' else 'plain')
        self.guessed = sc.objdict()
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
        if isinstance(levels, dict): # Each level with its share, e.g. age={0: 0.3, 15: 0.6, 65: 0.1}
            if split is not None:
                errormsg = f'"{name}" was given the share in each level twice, as a dict and as split=; give one'
                raise ValueError(errormsg)
            levels, split = list(levels.keys()), list(levels.values())
        if isinstance(mobility, (list, tuple)) and any(isinstance(m, (Mobility, gravity, radiation)) for m in mobility):
            errormsg = f'mobility= takes one movement for "{name}", not {len(mobility)}; movements of one kind can be combined by adding their matrices (ss.geo.Mobility(matrix=M1 + M2)), but several kinds at once (e.g. a register and commuting) are not supported'
            raise TypeError(errormsg)
        if mobility is not None and not isinstance(mobility, Mobility): # A movement model or a matrix on its own is commuting
            mobility = Mobility(mobility) if isinstance(mobility, (gravity, radiation)) else Mobility(matrix=mobility)
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
        self.carried = None # For a strain: the states people carry it in as an infection, so a new infection adds to it (see ss.Strains._carriers()); in its other states (e.g. R, for its protection), a new infection replaces it
        if self.kind == 'age' and self.edges is None and kind is None: # Age groups given as labels have no ages, so nobody ages between them
            self.kind = 'plain'
            self.guessed['kind'] = 'plain, since the age groups are labels and not ages'
        self.adjust = sc.objdict()
        for key,vals in (adjust or {}).items():
            vals = np.array(vals, dtype=float)
            if len(vals) != self.n:
                raise ValueError(f'the adjustment of "{key}" has {len(vals)} values, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}).')
            if np.any(vals < 0):
                raise ValueError(f'the adjustment of "{key}" by "{name}" has a negative multiplier ({vals}).')
            self.adjust[key] = vals
        self.joint = None # A split over several dimensions at once, as an ss.by; resolved by the sim
        self.cond = None # For a dimension of some states only, a split given the levels of other dimensions: their positions, and the share in each level of this one (summing to 1 over it)
        if isinstance(split, by):
            self.joint = split
            split = None
        self.split_given = split is not None or self.joint is not None
        if split is not None:
            split = np.array(split, dtype=float)
            if len(split) != self.n:
                raise ValueError(f'split= has {len(split)} entries, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}).')
            if not np.isclose(split.sum(), 1):
                self.guessed['split'] = f'normalized, since it summed to {split.sum():n}'
                split = split/split.sum()
        elif self.kind == 'age' and self.edges is not None: # As the people of a declared model, unless the sim has a named population
            world = ss.data.world
            split = ss.data.age_split(self.edges, world.ages, world.counts)
            self.guessed['split'] = world.guess
        elif all(isinstance(v, (bool, np.bool_)) for v in self.labels): # e.g. vaccinated=[False, True]
            split = self.first()
            self.guessed['split'] = f'everyone at {self.labels[0]}, the first level'
        else:
            split = np.ones(self.n)/self.n
            self.guessed['split'] = 'equal across levels'
        self.split = split
        return

    def worth(self):
        """ The value of the property for each level, as a compartmental run takes it in an expression: the lower edge for whole numbers (a counter), and otherwise the middle of the bin """
        if self.edges is not None and np.allclose(self.edges, np.round(self.edges)):
            return self.edges.copy()
        return self.mids

    def first(self):
        """ A split with everyone at the first level """
        out = np.zeros(self.n)
        out[0] = 1
        return out

    def init_carriage(self, strains):
        """
        For a strain dimension whose levels are sets of strains carried at once ('A+B', from `ss.Strains(max_strains=)`): which strains each level carries, and where carrying one more leads

        The first levels are the strains alone. Someone who carries several strains passes on one of them
        (see `ss.Strains(transmit=)`), and someone infected with a strain they do not carry moves to the
        level that adds it (once full, the level that replaces the oldest: see `ss.Strains._adds()`).
        """
        n = len(strains.names)
        self.base = np.arange(n) # The levels of a single strain
        self.carries = np.zeros((self.n, self.n), dtype=bool) # carries[l, j]: people at level l carry strain j
        for l,s in enumerate(strains.sets):
            self.carries[l, list(s)] = True
        self.adds = strains._adds() # adds[l, j]: the level of people at level l once infected with strain j as well (l itself if they carry it already)
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

    def value(self, index):
        """ The value people hold for a level, given its position: the level's lower edge for a numeric dimension (people hold values, which are binned into levels), the position itself otherwise """
        if self.edges is None:
            return index
        return self.edges[np.asarray(index, dtype=int)]

    def moved(self, vals, level):
        """ As agents: the values of people moved along this dimension: to a level (its position), up a counter (`('+', k)`), or from each level to another (`('map', levels)`, a change of strains) """
        if isinstance(level, tuple):
            op, x = level
            if op == 'map':
                return self.target(level)[np.asarray(vals, dtype=int)]
            return self.add(vals, x)
        return self.value(level)

    def target(self, level):
        """ Where people at each level go when moved along this dimension: as `moved()`, a level for each level """
        if isinstance(level, tuple):
            op, x = level
            if op == 'map': # Those at a level it does not apply to stay
                return np.where(np.asarray(x) >= 0, x, np.arange(self.n))
            return self.shifted(x)
        return np.full(self.n, level)

    def move_level(self, level):
        """ A level that people are moved to, as given in `move=`: its position """
        return self.index(level)

    def people_arr(self):
        """ The property of people that holds each agent's level, when they do not have it already: for a dimension with labels, the level itself, as 16-bit integers (so with one dimension it is also the group: see `Groups.group_raw()`); for a binned one, floats """
        if self.edges is None:
            return ss.Arr(self.attr, dtype=level_dtype(self.n), default=0, nan=-1, label=self.name)
        return ss.FloatArr(self.attr, default=0, label=self.name)

    @property
    def spec(self):
        """ The stratification as it is written in the model spec """
        levels = self.edges.tolist() if self.edges is not None else list(self.labels)
        mobility = None
        if self.mobility is not None:
            mob = self.mobility
            away = mob.away if mob.away is None else mob.away.name if isinstance(mob.away, ss.par) and mob.away.name else float(mob.away) # A fitted share by its parameter's name
            mobility = dict(kind=mob.kind, away=away)
            if mob.data is None:
                mobility['matrix'] = mob.matrix(self).tolist()
            else: # A register of movements, by its size and a hash of its contents
                mobility['register'] = dict(events=len(mob.data), hash=sc.sha(mob.data.astype(str).to_csv()).hexdigest())
        out = dict(geo=self.geo.ref if self.geo is not None else None, split_from=getattr(self, 'split_from', None) or None, # Where a named geography or split came from
                   name=self.name, kind=self.kind, levels=levels, split=self.split.tolist() if self.joint is None else repr(self.joint), adjust={k:v.tolist() for k,v in self.adjust.items()}, states=self.states, where=self.where, mobility=mobility)
        if self.strains is not None: # As ss.Strains takes it
            out['strains'] = self.strains.spec
        return out

    @property
    def mids(self):
        """ The middle of each bin, for a binned property """
        if self.edges is None:
            return np.arange(self.n, dtype=float)
        ends = np.append(self.edges[1:], max(100, self.edges[-1] + 10) if self.kind == 'age' else self.edges[-1] + 1)
        return (self.edges + ends)/2

    def shifted(self, k):
        """ For a counter (`increment=`): the level that people at each level reach when it goes up by k (or down, for k < 0), staying in the last (open-ended) level or the first """
        if self.edges is None:
            return np.clip(np.arange(self.n) + k, 0, self.n - 1)
        return np.clip(np.digitize(self.edges + k, self.edges) - 1, 0, self.n - 1)

    def add(self, vals, k):
        """ As agents: the values of a counter after it goes up by k: a value past the last edge stays in the last level, and none goes below the first """
        if self.edges is None: # The property is the level
            return np.clip(vals + k, 0, self.n - 1)
        return np.maximum(vals + k, self.edges[0])

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
    """ Movement between the levels of a stratification, without a change of disease state: see `ss.transfer()`, which makes it """
    def __init__(self, rate=None, dur=None, p=None, states=None, name=None, label=None, increment=None, **kwargs):
        increment = parse_increment(increment, 'ss.transfer()')
        if len(kwargs) + len(increment or {}) != 1:
            errormsg = f'ss.transfer() takes one dimension, e.g. ss.transfer(risk=("low", "high"), rate=0.1) or ss.transfer(increment="doses", rate=0.1), not {list(kwargs.keys()) + list(increment or {})}'
            raise ValueError(errormsg)
        if (rate is None) == (dur is None):
            errormsg = 'ss.transfer() takes exactly one of rate= or dur='
            raise ValueError(errormsg)
        self.increment = None # For a counter: the amount people move up it
        if increment:
            self.dim, self.increment = list(increment.items())[0]
            target = None
        else:
            self.dim, target = list(kwargs.items())[0]
        src, self.dst = level_pair(target, 'ss.transfer()') # From a level, or from every other level
        if src is not None and len(src) != 1:
            errormsg = f'ss.transfer() moves people from one level (or from every other level, if only the level they move to is given), not from {src}; give one transfer for each'
            raise ValueError(errormsg)
        self.src = None if src is None else src[0]
        self.dsts = self.dst if isinstance(self.dst, list) else [self.dst] # The levels it moves people to
        if len(self.dsts) > 1 and self.src is None:
            errormsg = f'A transfer to one of several levels names the level people move from, as a pair, e.g. ss.transfer({self.dim}=("a", ["b", "c"]), p=[0.6, 0.4], dur=...), not {self.dim}={target!r}'
            raise ValueError(errormsg)
        self.p = None # With several, the probability of each, as people move
        if len(self.dsts) > 1 or p is not None:
            p = sc.tolist(p)
            p = p + [None] if len(p) == len(self.dsts) - 1 else p # With two levels, one value is the first one's probability
            if len(self.dsts) == 1 or len(p) != len(self.dsts) or sum(v is None for v in p) > 1:
                what = f'{len(self.dsts)} levels to move to' if len(self.dsts) > 1 else 'one level to move to'
                errormsg = (f'p= of ss.transfer() gives the probability of each level it moves people to, one each (or None for the rest), as people move, e.g. ss.transfer(method=("none", ["pill", "iud"]), p=[0.6, 0.4], dur=...); '
                            f'it has {what} and p={p}. For only some people to move, give a rate (rate=), or where= on the stratification')
                raise ValueError(errormsg)
            self.p = [None if v is None else as_expr(v) for v in p]
        self.rate = None if rate is None else as_expr(rate)
        self.dur = None if dur is None else as_expr(dur)
        self.states = None if states is None else [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
        self.name = name
        self.label = label
        return

    def __repr__(self):
        if self.increment is not None:
            return f'transfer({self.name}: {self.dim} {self.increment:+d})'
        return f'transfer({self.name}: {self.dim} {self.src or "*"} -> {" | ".join(map(str, self.dsts))})'

    def __or__(self, other): return ss.conditions.as_cond(self) | other # As an event: the people who moved this step
    def __ror__(self, other): return other | ss.conditions.as_cond(self)
    def __and__(self, other): return ss.conditions.as_cond(self) & other
    def __rand__(self, other): return other & ss.conditions.as_cond(self)


def transfer(rate=None, dur=None, p=None, states=None, name=None, label=None, increment=None, **kwargs):
    """
    Movement between the levels of a stratification, without a change of disease state

    People move all the time, at a rate or after a duration, and the transfer is part of the
    model: pass it to `ss.Disease()` or `stratify()`. A move delivered to some people, at some
    times, is a delivery: `ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021])`.

    Args:
        rate (any): the per-person hazard
        dur (any): the time before moving (instead of `rate`): a scalar is the mean of an exponential, and a distribution is drawn for each person when they reach a level it moves them from
        p (list): with several levels to move to (`method=('none', ['pill', 'iud'])`), the probability of each, decided as people move; one may be None, for the rest
        states (list): the disease states it applies to (default: every state)
        name (str): the name; if not given, taken from the keyword or variable it is assigned to
        increment (str/dict): instead of a level, a counter that goes up by one (its name), or by an amount (`dict(schooling=1)`, `dict(immunity=-1)`)
        kwargs (dict): the dimension, and a pair of its levels, from and to (e.g. `risk=('low', 'high')`),
            or the level people move to, from every other level (e.g. `vaccinated=True`)

    Examples:
        ```python
        ss.transfer(risk=('low', 'high'), rate=ss.peryear(0.1))
        ss.transfer(vaccinated=True, rate=ss.peryear(0.2))
        ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5)) # A counter, down a level at a time
        ss.transfer(method=('none', ['pill', 'iud']), p=[0.6, 0.4], dur=ss.lognorm_ex(ss.years(2), ss.years(1))) # To one of two levels, after a time drawn for each person
        ```
    """
    opts = [key for key in kwargs if key in ss.interventions.delivery_keys + ['disease']]
    if opts:
        dim = next((key for key in kwargs if key not in opts), 'vaccinated')
        errormsg = (f'ss.transfer() is movement that happens all the time, at a rate (rate= or dur=); {sc.strjoin(f"{key}=" for key in opts)} {"is" if len(opts) == 1 else "are"} for a delivery. '
                    f'A move delivered to some people is ss.Delivery(move=dict({dim}=...), ...), e.g. ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021])')
        raise TypeError(errormsg)
    return Transfer(rate=rate, dur=dur, p=p, states=states, name=name, label=label, increment=increment, **kwargs)


def transfer_stages(sim, mod, tf, strat):
    """
    For a compartmental run: the Erlang stages of a transfer whose dur= is a distribution, as a dimension of the people at the levels it moves from (None if it needs none)

    People enter its first stage when they reach a level the transfer moves them from (or one of its states), go up a stage at k
    over its mean, and move from the last; the k stages match its mean and standard deviation, as for a dwell time (see `ss.solvers.Engine.init_dwells()`).
    """
    trans = ss.transitions
    raw = trans.by_dist(getattr(tf.dur, 'raw', None))
    if not isinstance(raw, ss.Dist) or isinstance(raw, ss.expon) or trans.varying_pars(raw): # Values by group stay at the rate of their mean, with a note
        return None
    mean, std = trans.dwell_stats(raw, lambda val: ss.solvers.engine.convert(val, sim.unit_base))
    k = trans.exact_stages(raw) or trans.erlang_stages(mean, std)
    if k == 1:
        return None
    if tf.increment is not None:
        levels = [i for i,j in enumerate(strat.shifted(tf.increment)) if i != j]
    else:
        levels = [strat.index(tf.src)] if tf.src is not None else [i for i in range(strat.n) if i != strat.index(tf.dst)]
    out = Strat(f'{tf.name}_stage', [f'stage {i+1}' for i in range(k)], split=[1.0] + [0.0]*(k - 1), states=tf.states, where={tf.dim:levels})
    out.owner = mod.name
    out.stages_of = (mod.name, tf.name)
    out.dwell = sc.objdict(mean=mean, std=std, k=k, family=type(raw).__name__, exact=trans.exact_stages(raw) is not None) # For the note on how close the stages are: see ss.solvers.Engine.note_stages()
    return out


strat_options = ['split', 'adjust', 'kind', 'states', 'where', 'mobility']
strat_kinds = ['age', 'space', 'plain'] # The kinds stratify() takes (a strain dimension is made by ss.Strains, and a disease one by the solvers)


def check_kind(kind):
    """ Check the kind= given to stratify() """
    if kind is not None and kind not in strat_kinds:
        errormsg = f'kind="{kind}" is not a kind of stratification; the kinds are {sc.strjoin([repr(k) for k in strat_kinds], sep=", ")} (or leave it out to have it guessed)'
        raise ValueError(errormsg)
    return


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
    if opts['kind'] == 'strain':
        errormsg = 'strains are part of the disease, not a stratification: ss.Disease(..., strains=ss.Strains(["A", "B"], ...)), which takes the split, the values of each strain, and the protection between them (immunity=)'
        raise ValueError(errormsg)
    check_kind(opts['kind'])
    misplaced = [key for key in kwargs if key in ['inplace', 'frame']]
    if misplaced:
        errormsg = f'{sc.strjoin(misplaced)} is not an option of {"sim.stratify() or ss.Sim(stratify=...)" if sim else "stratify() here"}, and cannot be the name of a dimension'
        raise ValueError(errormsg)
    dims = {k:v for k,v in kwargs.items() if not isinstance(v, Transfer)}
    transfers = {k:v for k,v in kwargs.items() if isinstance(v, Transfer)}
    given = [key for key,val in opts.items() if val is not None]
    if not dims and not transfers:
        errormsg = f'stratify() needs a dimension and its levels, e.g. stratify(age=[0, 15, 65]); it was given only {sc.strjoin(given) or "nothing"}'
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
                raise sc.KeyNotFoundError(f'states= names a state "{name}" that {where} {sc.strjoin(names("states"))}{did_you_mean(name, names("states"))}')
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
            raise sc.KeyNotFoundError(f'adjust= {what} "{key}" that {whose} {sc.strjoin(options)} (and it can also adjust rel_trans and rel_sus).')
    strats = []
    for name,levels in dims.items():
        strat = Strat(name, levels, **opts)
        sets = [str(label).split('+') for label in strat.labels]
        if any(len(parts) > 1 and all(part.strip() in strat.labels for part in parts) for parts in sets): # A level that is other levels carried together
            errormsg = f'"{name}" has a level that joins others with "+" ({sc.strjoin(strat.labels)}): sets of strains carried together are made by ss.Strains, e.g. ss.Disease(..., strains=ss.Strains({[label for label,parts in zip(strat.labels, sets) if len(parts) == 1]}, max_strains=2))'
            raise ValueError(errormsg)
        strats.append(strat)
    return strats, transfers


#%% The groups of a sim

class Groups:
    """
    The groups of a sim: its stratifications (`ss.dimensions.Strat`), and how their levels combine into groups

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
                    raise ValueError(f'the sim is already stratified by "{strat.name}".')
                strats[strat.name] = strat
                self.sim_strats.append(strat)
        for strat in strats.values(): # An age group nobody is in: the edges are lower edges, so [0, 15, 120] has a group 120+
            if strat.kind == 'age' and strat.edges is not None and strat.edges[-1] >= 100:
                ss.note(f'the age group {strat.labels[-1]} of "{strat.name}" is empty, since ages run to 100; the edges are the lower edges of each group, so the last is open-ended', sim=sim)
        if self.sim_strats and not sim.declared: # Only modules written as code, which keep their own results: used only if a module's values are by it
            mods = [mod for key in ss.modules.module_types() + ['custom'] for mod in sim.pars[key].values() if isinstance(mod, ss.Module)]
            byvals = [val for mod in mods for val in [*mod.pars.values(), *vars(mod).values()] if isinstance(val, ss.by)] # Its parameters, and options such as a delivery's coverage
            unused = [strat.name for strat in self.sim_strats if not any(strat.name in val.dims for val in byvals)]
            if unused:
                raise ValueError(f'the sim is stratified by {sc.strjoin(unused)}, but none of its modules is declared and none has values by it, so nothing would use it: modules written as code keep their own results, not by group. '
                                          f'Declare a disease (ss.Disease) to have results by group, or leave the stratification out.')
        for mod in sim.declared:
            for strat in mod.graph.strats:
                if strat.name in strats:
                    other = strats[strat.name]
                    if not strat.same_levels(other):
                        raise ValueError(f'"{mod.name}" is stratified by "{strat.name}" with levels {strat.labels}, but another module or the sim uses {other.labels}.\n'
                                                       f'     A property of people has the same groups for every module; stratify the sim instead: sim.stratify({strat.name}=[...]).')
                    elif 'split' not in strat.guessed and strat.joint is None and not np.allclose(strat.split, other.split): # An explicit split is used, whichever stratification it is on
                        if 'split' not in other.guessed:
                            raise ValueError(f'"{mod.name}" is stratified by "{strat.name}" with split {strat.split.tolist()}, but another module or the sim gives it the split {other.split.tolist()}; give it once.')
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
                        raise sc.KeyNotFoundError(f'the transfer "{tf.name}" moves people along "{tf.dim}", but the model is not stratified by it. Add e.g. {mod.name}.stratify({tf.dim}=[...]).')
        self.dim_moves = [] # Movement between levels that belongs to no one module, e.g. the protection of a product waning
        for mod in sim.pars.interventions.values(): # Properties of people that the interventions need, e.g. holding a product
            if hasattr(mod, 'dims'):
                for strat in mod.dims(sim):
                    strat.owner = mod.name # Part of the scenario, not of the model
                    strats.setdefault(strat.name, strat)
                self.dim_moves += [move for move in mod.dim_moves(sim) if move.dim not in [m.dim for m in self.dim_moves]]
                for dim in list(mod.move or {}) + list(getattr(mod, 'increment', None) or {}):
                    if dim not in strats:
                        raise sc.KeyNotFoundError(f'the transfer "{mod.name}" moves people along "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        if sim.solver not in [None, 'abm']: # In a compartmental run, a transfer whose dur= is a distribution passes people through Erlang stages: a dimension of the people at the levels it moves from (see ss.solvers.Compartmental.init_movers())
            for mod in sim.declared:
                for tf in mod.graph.transfers.values():
                    strat = transfer_stages(sim, mod, tf, strats[tf.dim])
                    if strat is not None:
                        strats[strat.name] = strat
        moved = [tf.dim for mod in sim.declared for tf in mod.graph.transfers.values()] # A dimension that people are moved along, with no split of its own, starts with everyone at its first level
        moved += [dim for mod in sim.declared for tr in mod.graph.transitions.values() for dim in list(getattr(tr, 'move', None) or {}) + list(getattr(tr, 'increment', None) or {})]
        moved += [dim for mod in sim.pars.interventions.values() for dim in list(getattr(mod, 'move', None) or {}) + list(getattr(mod, 'increment', None) or {})]
        for name in moved:
            strat = strats.get(name)
            if strat is not None and strat.guessed.get('split') == 'equal across levels':
                strat.split = strat.first()
                strat.guessed['split'] = f'everyone at {strat.labels[0]}, the first level, since people are moved between its levels'
        for strat in self.cross_strats:
            if strat.name in strats:
                raise ValueError(f'the sim has a dimension called "{strat.name}", which is also the name of a disease whose states are tracked jointly with another\'s; rename one of them.')
            strats[strat.name] = strat
        self.strats = list(strats.values())
        named = getattr(sim, 'named_people', None)
        given = getattr(sim, 'given_pyramid', None)
        for strat in self.strats: # Age groups with no split of their own take that of a named population, or of the people given to a compartmental run
            if (named is not None or given is not None) and strat.kind == 'age' and strat.edges is not None and strat.guessed.get('split') == ss.data.world.guess:
                ages, counts = (named.ages, named.counts) if named is not None else given
                strat.split = ss.data.age_split(strat.edges, ages, counts)
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
        joints = [strat for strat in self.strats if strat.joint is not None and strat.states is None] # A split over several dimensions of some states only is the share entering each level: see below
        if len(joints) > 1:
            errormsg = 'Only one split= can cover several dimensions at once'
            raise ValueError(errormsg)
        self.joint = None # For a split over several dimensions at once: their positions, and the share of people at each combination of their levels
        for strat in [strat for strat in self.strats if strat.joint is not None and strat.name not in strat.joint.dims]: # E.g. split=ss.by(sex=dict(female=[...], male=[...]))
            others = sc.strjoin(strat.joint.dims)
            errormsg = (f'The split of "{strat.name}" is given by {others} only; a split over several dimensions is over all of them, including "{strat.name}" itself: '
                        f'ss.by(table, dims=[{", ".join(repr(d) for d in strat.joint.dims)}, {strat.name!r}]), whose table has one row for each level of {others} and one column for each level of "{strat.name}", '
                        f'and is the share of people at each combination (e.g. the share in each level of "{strat.name}" within each row, times the share of the row)')
            raise ValueError(errormsg)
        if joints:
            ds, grid = joints[0].joint.grid(sim)
            ages = [i for i,d in enumerate(ds) if self.strats[d].kind == 'age']
            within = [] # A split over age and other dimensions whose rows each sum to 1 is the share at each level of the others within each age group, so people keep their ages (otherwise it is the share at each combination, and sets the ages)
            if ages:
                a = ages[0]
                totals = grid.sum(axis=tuple(j for j in range(len(ds)) if j != a), keepdims=True)
                if np.allclose(totals, 1):
                    within = ages
                    grid = grid*self.strats[ds[a]].split.reshape([-1 if j == a else 1 for j in range(len(ds))])
            grid = grid/grid.sum()
            for i,d in enumerate(ds): # Each dimension's own split is the total over the others
                if i not in within:
                    strat, total = self.strats[d], grid.sum(axis=tuple(j for j in range(len(ds)) if j != i))
                    if strat.joint is None and strat.split_given and 'split' not in strat.guessed and not np.allclose(strat.split, total, atol=1e-6): # TL4: the given split was silently replaced
                        others = [self.strats[dd].name for dd in ds if dd != d]
                        raise ValueError(f'"{strat.name}" is given the split {np.round(strat.split, 4).tolist()}, but the split over {sc.strjoin(self.strats[dd].name for dd in ds)} gives it {np.round(total, 4).tolist()}: '
                                                       f'a split over several dimensions is the share at each combination of their levels. For rows that are the shares of {sc.strjoin(others)} given "{strat.name}", '
                                                       f'multiply them by the share at each level of "{strat.name}" ({np.round(strat.split, 4).tolist()}) first, or leave out one of the two splits.')
                    strat.split = total
                    self.strats[d].guessed.pop('split', None)
                    if within:
                        self.strats[d].guessed['split'] = 'within each age group (the ages are kept), since the split over age has rows that each sum to 1'
            self.joint = (ds, grid)
            split *= grid[tuple(self.group_coords[:, d] for d in ds)]
            born = grid[tuple(0 if i in ages else slice(None) for i in range(len(ds)))] # Newborns are in the youngest age group, so take its share of the others (or everyone's, if it has nobody)
            born = born if born.sum() > 0 else grid.sum(axis=tuple(ages))
            births *= (born/born.sum())[tuple(self.group_coords[:, d] for i,d in enumerate(ds) if i not in ages)]
        for d,strat in enumerate(self.strats):
            share = strat.split[self.group_coords[:, d]]
            if self.joint is None or d not in self.joint[0]:
                split *= share
            if strat.kind == 'age':
                births *= self.group_coords[:, d] == 0 # People are born into the youngest group
            elif self.joint is None or d not in self.joint[0]:
                births *= share
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
        if 'use_aging' not in sim.given: # People age if and only if they are born or die, unless told otherwise (said in ss.Sim.init(), once the length of the run is known)
            sim.pars.use_aging = sim.has_demography()
        return

    def moved_by(self, strat):
        """ For describe(): what moves people along a dimension, e.g. '; counter: +1 by delivery; +2 by twins' """
        sim = self.sim
        events = [] # (what, amount or level)
        for mod in getattr(sim, 'declared', []):
            for name,tr in mod.graph.transitions.items():
                if strat.name in (getattr(tr, 'increment', None) or {}):
                    events.append((name, f'{tr.increment[strat.name]:+d}'))
                elif strat.name in (getattr(tr, 'move_given', None) or {}):
                    events.append((name, str(tr.move_given[strat.name])))
            for name,tf in mod.graph.transfers.items():
                if tf.dim == strat.name:
                    events.append((name, f'{tf.increment:+d}' if tf.increment is not None else ' | '.join(map(str, tf.dsts))))
        for mod in sim.pars.interventions.values():
            if strat.name in (getattr(mod, 'increment', None) or {}):
                events.append((mod.name, f'{mod.increment[strat.name]:+d}'))
            elif strat.name in (getattr(mod, 'move', None) or {}):
                events.append((mod.name, str(mod.move[strat.name])))
        if not events:
            return ''
        amounts = {}
        for name,amount in events:
            amounts.setdefault(amount, []).append(name)
        verb = lambda amount: amount if amount[0] in '+-' else f'to {amount}'
        what = 'counter' if any(amount[0] in '+-' for amount in amounts) else 'moved'
        return f'; {what}: ' + '; '.join(f'{verb(amount)} by {sc.strjoin(names)}' for amount,names in amounts.items())

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
                    raise sc.KeyNotFoundError(f'"{self.strats[d].name}" is stratified where {dim} is given, but the model is not stratified by "{dim}". Add e.g. sim.stratify({dim}=[...]).')
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
            example = "sex=['female', 'male']" if name == 'sex' else f'{name}=[...]'
            errormsg = f'The model is not stratified by "{name}"; its dimensions are {names}. Add e.g. sim.stratify({example}), which as agents adds no compartments'
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
        if cache is not None and cache[0] == sim.ti and len(cache[1]) == n: # The usual case, first, since this is called at every change of a state that is counted
            return cache[1]
        binned = any(strat.edges is not None for strat in self.strats)
        if cache is not None and not binned and len(cache[1]) == n:
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
                    after = np.clip(np.digitize(raw[uids], strat.edges) - 1, 0, strat.n - 1) if strat.edges is not None else ss.perflib.passes.levels(raw[uids])
                    old_level[uids] = after
                    old_raw[uids] = raw[uids]
                    if not shared:
                        out[uids] += (after.astype(out.dtype) - before)*int(mult) # In the groups' integers, which may be wider than the levels'
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
            return (np.clip(np.digitize(raw, strat.edges) - 1, 0, strat.n - 1) if strat.edges is not None else ss.perflib.passes.levels(raw)).astype(level_dtype(strat.n), copy=False)
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
                if strat.edges is not None: # A binned property (e.g. age) changes for nearly everyone at each step, but only those who crossed an edge (or have no value) change level
                    inner = strat.edges[1:strat.n]
                    lo, hi = np.append(-np.inf, inner), np.append(inner, np.inf) # The bounds of each level, as digitize() bins them
                    idx = ss.perflib.passes.crossed(raw, old[1], lo, hi)
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
        moved = None # The agents whose group changed, and their groups before and after
        if shared:
            kept[''] = levels[0] # Already up to date
            if updates and not rebuild:
                moved = updates[0][:3]
        elif rebuild:
            out = np.zeros(n, dtype=dtype)
            for level,mult in zip(levels, mults):
                out += level.astype(dtype)*int(mult) # Widened first: a level times its stride may not fit in the level's integers
            kept[''] = out
        else:
            out = kept['']
            if n0 < n:
                tail = np.zeros(n - n0, dtype=dtype)
                for level,mult in zip(levels, mults):
                    tail += level[n0:].astype(dtype)*int(mult)
                out = np.concatenate([out, tail])
            if updates:
                idx = updates[0][0] if len(updates) == 1 else np.unique(np.concatenate([u[0] for u in updates]))
                before = out[idx].copy()
            for idx_,before_,after,mult in updates:
                out[idx_] += (after.astype(dtype) - before_)*mult
            if updates:
                moved = (idx, before, out[idx])
            kept[''] = out
        self._group_cache = (sim.ti, self.folded(kept['']), levels)
        if moved is not None:
            self.regrouped(*moved)
        elif rebuild and cache is not None: # Worked out again from scratch: whatever keeps counts by group counts again
            for mod in getattr(sim, 'declared', []):
                if isinstance(mod.engine, ss.solvers.Agents):
                    mod.engine.regroup()
        return self._group_cache[1]

    def regrouped(self, uids, before, after):
        """ Some agents' groups have changed by themselves (e.g. they reached the next age group): the agent engines move them between the counts they keep by group (see `ss.solvers.Agents.moved_groups()`) """
        before, after = self.folded(before), self.folded(after)
        changed = before != after
        if not changed.any():
            return
        uids, before, after = uids[changed], before[changed], after[changed]
        for mod in getattr(self.sim, 'declared', []):
            if isinstance(mod.engine, ss.solvers.Agents):
                mod.engine.moved_groups(uids, before, after)
        return

    def folded(self, groups):
        """ The groups that agents are counted in: see `init_ragged()` """
        return groups if self.group_fold is None else self.group_fold[groups]

    def level_raw(self, d):
        """ In an agent-based run: the level of every agent in one dimension (by its position), as an array indexed by UID; people outside the dimension's `where=` have a level too, which counts nothing (see `level()`) """
        self.group_raw()
        return self._group_cache[2][d]

    def level(self, dim):
        """
        In an agent-based run: the level of every agent in one dimension, as an array indexed by UID, with -1 for people outside its `where=`

        Args:
            dim (str/int): the dimension, by name or position

        **Example**:

            sim.groups.level('risk') # 0, 1, 2 for each risk group; -1 for people the risk groups don't apply to
        """
        d = self.dim_index(dim) if isinstance(dim, str) else dim
        levels = self.level_raw(d)
        if self.group_fold is None or self.group_inside[:, d].all():
            return levels
        inside = self.group_inside[self.group_raw(), d]
        return np.where(inside, levels, -1)

    def init_vals(self):
        """ In an agent-based run: assign each agent a level of every property they do not already have """
        sim = self.sim
        people = sim.people
        uids = people.auids
        joint = self.joint[0] if self.joint is not None else []
        own = getattr(self, 'kept', set()) # Properties the people came with, which are kept and binned (see ss.Sim.init_people())
        for d,strat in enumerate(self.strats):
            if strat.name in own:
                continue
            if strat.female is not None and d not in joint and strat.cond is None: # A sex dimension is people.female, whose share is the split: see init_people()
                self.set_sex(uids)
            elif strat.kind != 'age' and d not in joint and strat.cond is None: # Exactly the split, rounded to whole agents, in a random order
                order = np.argsort(getattr(sim, f'rng_strat_{strat.name}').rvs(uids), kind='stable')
                counts = ss.solvers.Compartmental.round_preserving(strat.split*len(uids)).astype(int)
                levels = np.repeat(np.arange(strat.n, dtype=level_dtype(strat.n)), counts) # As they are kept, to save memory
                people[strat.attr][uids[order]] = strat.value(levels)
        if joint:
            self.draw_joint(uids)
        for strat in self.strats: # A split given the levels of other dimensions, once those are known
            if strat.cond is not None:
                self.groups_changed()
                people[strat.attr][uids] = strat.value(strat.draw(sim, uids, getattr(sim, f'rng_strat_{strat.name}').rvs(uids)))
        if joint or any(strat.cond is not None for strat in self.strats): # A sex drawn with other dimensions sets people.female
            self.set_sex(uids, from_level=True)
        self.groups_changed()
        self.vals_set = True # Agents added from now on are placed as they are made: see place_new()
        return

    def draw_joint(self, uids):
        """ In an agent-based run: the levels of a split over several dimensions at once, together, for some agents, given their age group if age is one of them """
        sim = self.sim
        people = sim.people
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
            people[self.strats[ds[i]].attr][uids] = self.strats[ds[i]].value(level)
        return

    def place_new(self, uids):
        """
        In an agent-based run: the levels of agents added by a module written as code (e.g. `ss.Births`), drawn from each dimension's split, as the declared births draw them (see `ss.solvers.Agents.add_people()`)

        A newborn with a parent (`people.parent`) is in the parent's place (a dimension of kind "space"), and a split over age and
        other dimensions is drawn from the newborn's age group.
        """
        sim = self.sim
        people = sim.people
        joint = self.joint[0] if self.joint is not None else []
        if len(joint):
            self.draw_joint(uids)
        parent = people.parent.raw[uids]
        has_parent = parent >= 0
        for d,strat in enumerate(self.strats):
            if strat.kind in ['age', 'strain', 'disease'] or strat.female is not None or d in joint: # Ages are their own, a sex dimension is people.female (see set_sex()), newborns have no strain, and a joint split is drawn above
                continue
            if strat.kind == 'space' and has_parent.all() and strat.cond is None: # Born where the parent is
                people[strat.attr][uids] = people[strat.attr].raw[parent.astype(int)]
                continue
            rvs = getattr(sim, f'rng_strat_{strat.name}').rvs(uids)
            if strat.cond is not None:
                people[strat.attr][uids] = strat.value(strat.draw(sim, uids, rvs))
            else:
                people[strat.attr][uids] = strat.value(np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1))
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
